import time
from collections import OrderedDict, deque
from collections.abc import MutableMapping
import contextlib

from fsspec.caching import BaseCache, register_cache


class ReadAheadChunked(BaseCache):
    """
    An optimized ReadAhead cache that fetches multiple chunks in a single
    HTTP request but manages them as separate bytes objects to avoid
    expensive memory slicing.

    While this approach primarily optimizes for CPU and memory allocation overhead,
    it strictly maintains the same semantics as the existing readahead cache.
    For example, if a user requests 5MB and the cache fetches 10MB, it serves the
    requested 5MB but retains that data in memory to handle potential backward seeks.
    This mirrors the standard readahead behavior, which does not eagerly discard served
    chunks until a new fetch is required.
    """

    name = "readahead_chunked"

    def __init__(self, blocksize: int, fetcher, size: int) -> None:
        super().__init__(blocksize, fetcher, size)
        self.chunks = deque()  # Entries: (start, end, data_bytes)

    @property
    def cache(self):
        """
        Compatibility property for tests/legacy code that expects 'cache'
        to be a single bytestring.

        WARNING: Accessing this property forces a memory copy of the
        entire current buffer, negating the Zero-Copy optimization
        of ReadAheadChunked. Use for debugging/testing only.
        """
        if not self.chunks:
            return b""
        return b"".join(chunk[2] for chunk in self.chunks)

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None or end > self.size:
            end = self.size
        if start >= self.size:
            return b""

        # Handle backward seeks that go beyond the start of our cache window
        if self.chunks and self.chunks[0][0] > start:
            self.chunks.clear()

        parts = []
        current_pos = start

        # Satisfy as much as possible from the existing cache (Zero-Copy)
        for c_start, c_end, c_data in self.chunks:
            if c_end <= start:
                continue  # Skip chunks completely before our window

            if c_start >= end:
                break  # If we've reached chunks completely past our window, stop

            if c_end > current_pos:
                slice_start = max(0, current_pos - c_start)
                slice_end = min(len(c_data), end - c_start)

                if slice_start == 0 and slice_end == len(c_data):
                    # Zero-copy: Direct reference to the full object
                    parts.append(c_data)
                else:
                    # Slicing creates a copy, but it's unavoidable for partials
                    parts.append(c_data[slice_start:slice_end])

                current_pos += slice_end - slice_start

        # Fetch missing data if necessary
        should_fetch_backend = current_pos < end
        if should_fetch_backend:
            # On a cache miss, we replace the entire window (standard readahead behavior)
            self.chunks.clear()

            missing_len = min(self.size - current_pos, end - current_pos)
            readahead_block = min(
                self.size - (current_pos + missing_len), self.blocksize
            )

            self.miss_count += 1
            chunk_lengths = [missing_len]
            if readahead_block > 0:
                chunk_lengths.append(readahead_block)

            # Vector read call
            new_chunks = self.fetcher(start=current_pos, chunk_lengths=chunk_lengths)

            # Process the requested data
            req_data = new_chunks[0]
            self.chunks.append((current_pos, current_pos + len(req_data), req_data))
            self.total_requested_bytes += len(req_data)
            parts.append(req_data)

            # Process the readahead data (if any)
            if len(new_chunks) > 1:
                ra_data = new_chunks[1]
                ra_start = current_pos + len(req_data)
                self.chunks.append((ra_start, ra_start + len(ra_data), ra_data))
                self.total_requested_bytes += len(ra_data)

        if not parts:
            return b""

        if not should_fetch_backend:
            self.hit_count += 1

        # Optimization: return the single object directly if possible
        if len(parts) == 1:
            return parts[0]

        return b"".join(parts)


register_cache(ReadAheadChunked, clobber=True)


import os
import json
import urllib.parse
import threading

class InfoCache(MutableMapping):
    """
    Caching of single-object metadata (e.g., from info() calls).
    Implemented with a cross-process disk cache (via /dev/shm) and an L1 memory cache.
    """

    def __init__(
        self,
        use_info_cache=True,
        info_expiry_time=None,
        max_paths=100000,
    ):
        self._cache = OrderedDict()
        self._times = {}
        self.use_info_cache = use_info_cache
        self.info_expiry_time = info_expiry_time
        self.max_paths = max_paths

        if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
            self.cache_dir = "/dev/shm/gcsfs_info_cache"
        else:
            self.cache_dir = "/tmp/gcsfs_info_cache"
            
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:
            self.cache_dir = None

    def _get_disk_path(self, key):
        if not self.cache_dir:
            return None
        safe_key = urllib.parse.quote_plus(str(key))
        if len(safe_key) > 200:
            import hashlib
            h = hashlib.sha256(safe_key.encode()).hexdigest()
            safe_key = safe_key[:150] + "_" + h
        return os.path.join(self.cache_dir, safe_key + ".json")

    def __getitem__(self, item):
        # L1 Check
        if item in self._cache:
            if self.info_expiry_time is not None:
                if self._times.get(item, 0) - time.time() < -self.info_expiry_time:
                    self._cache.pop(item, None)
                    self._times.pop(item, None)
                else:
                    val = self._cache[item]
                    self._cache.move_to_end(item)
                    return val

        # L2 Check
        disk_path = self._get_disk_path(item)
        if disk_path and os.path.exists(disk_path):
            if self.info_expiry_time is not None:
                mtime = os.path.getmtime(disk_path)
                if time.time() - mtime > self.info_expiry_time:
                    try:
                        os.remove(disk_path)
                    except OSError:
                        pass
                    raise KeyError(item)
            try:
                with open(disk_path, "r") as f:
                    val = json.load(f)
                
                # Decode datetime fields if they exist
                import datetime
                for date_key in ["mtime", "ctime", "timeCreated", "updated", "timeStorageClassUpdated", "timeFinalized"]:
                    if date_key in val and isinstance(val[date_key], str):
                        try:
                            # gcsfs expects datetime objects in UTC
                            if val[date_key].endswith("+00:00"):
                                dt_str = val[date_key]
                            elif val[date_key].endswith("Z"):
                                dt_str = val[date_key][:-1] + "+00:00"
                            else:
                                dt_str = val[date_key]
                            val[date_key] = datetime.datetime.fromisoformat(dt_str)
                        except ValueError:
                            pass
                            
                # Promote to L1
                self._cache[item] = val
                if self.info_expiry_time is not None:
                    self._times[item] = os.path.getmtime(disk_path)
                if self.max_paths and len(self._cache) > self.max_paths:
                    k, _ = self._cache.popitem(last=False)
                    self._times.pop(k, None)
                return val
            except Exception:
                pass

        raise KeyError(item)

    def __setitem__(self, key, value):
        print(f"InfoCache __setitem__ called! use_info_cache={self.use_info_cache}, key={key}")
        if not self.use_info_cache:
            return

        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value

        now = time.time()
        if self.info_expiry_time is not None:
            self._times[key] = now

        if self.max_paths and len(self._cache) > self.max_paths:
            k, _ = self._cache.popitem(last=False)
            self._times.pop(k, None)

        # L2 Write
        disk_path = self._get_disk_path(key)
        if disk_path:
            try:
                tmp_path = f"{disk_path}.tmp.{os.getpid()}_{threading.get_ident()}"
                with open(tmp_path, "w") as f:
                    json.dump(value, f, default=str)
                os.rename(tmp_path, disk_path)
            except Exception as e:
                import traceback
                traceback.print_exc()

    @contextlib.asynccontextmanager
    async def lock_key(self, key):
        if not self.use_info_cache or not self.cache_dir:
            yield
            return
            
        disk_path = self._get_disk_path(key)
        lock_file = disk_path + ".lock"
        import fcntl
        import asyncio
        
        f = None
        try:
            f = open(lock_file, "w")
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, IOError):
                    await asyncio.sleep(0.01)
            yield
        finally:
            if f:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()

    def __delitem__(self, key):
        self._cache.pop(key, None)
        self._times.pop(key, None)
        disk_path = self._get_disk_path(key)
        if disk_path:
            try:
                os.remove(disk_path)
            except OSError:
                pass

    def invalidate_prefix(self, prefix):
        # L1 Clear
        keys_to_delete = [
            k for k in self._cache
            if k == prefix or k.startswith(f"{prefix}#") or k.startswith(f"{prefix}/")
        ]
        for k in keys_to_delete:
            self._cache.pop(k, None)
            self._times.pop(k, None)
            
        # L2 Clear
        if self.cache_dir:
            try:
                safe_prefix = urllib.parse.quote_plus(prefix)
                for f_name in os.listdir(self.cache_dir):
                    if f_name.startswith(safe_prefix):
                        try:
                            os.remove(os.path.join(self.cache_dir, f_name))
                        except OSError:
                            pass
            except Exception:
                pass

    def __contains__(self, item):
        try:
            self[item]
            return True
        except KeyError:
            return False

    def clear(self):
        self._cache.clear()
        self._times.clear()
        # Only clears L1, we leave L2 for other processes unless explicitly invalidated

    def __len__(self):
        return len(self._cache)

    def __iter__(self):
        return iter(list(self._cache))

import random
import fcntl
import hashlib
class CrossProcessBlockCache(BaseCache):
    """
    A strictly block-aligned cross-process file cache built on /dev/shm.
    Aligns reads into chunk blocks (e.g., 5MB) to maximize deduplication across
    concurrent processes (like PyTorch dataloader workers).
    Includes probabilistic background LRU eviction to strictly limit RAM usage.
    """
    name = "crossprocess_block"

    def __init__(
        self,
        blocksize: int,
        fetcher,
        size: int,
        cache_dir: str = "/dev/shm/gcsfs_block_cache",
        max_size_mb: int = 1024,
        **kwargs
    ):
        super().__init__(blocksize, fetcher, size)
        
        if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = "/tmp/gcsfs_block_cache"
            
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:
            pass

        path = kwargs.get("path", "unknown_path")
        self.file_id = hashlib.sha256(path.encode()).hexdigest()

    def _get_block_path(self, block_number: int):
        return os.path.join(self.cache_dir, f"{self.file_id}_{block_number}")

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""
            
        start_block_number = start // self.blocksize
        end_block_number = (end - 1) // self.blocksize
        
        start_pos = start % self.blocksize
        end_pos = end % self.blocksize
        if end_pos == 0:
            end_pos = self.blocksize
            
        if start_block_number == end_block_number:
            block = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]
        else:
            out = [self._fetch_block_cached(start_block_number)[start_pos:]]
            for b in range(start_block_number + 1, end_block_number):
                out.append(self._fetch_block_cached(b))
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])
            return b"".join(out)

    def _fetch_block_cached(self, block_number: int) -> bytes:
        block_path = self._get_block_path(block_number)
        
        if os.path.exists(block_path):
            try:
                os.utime(block_path, None)
                with open(block_path, "rb") as f:
                    return f.read()
            except Exception:
                pass
                
        lock_file = block_path + ".lock"
        f_lock = None
        try:
            try:
                f_lock = open(lock_file, "w")
                while True:
                    try:
                        fcntl.flock(f_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except (BlockingIOError, IOError):
                        time.sleep(0.01)
            except Exception:
                start = block_number * self.blocksize
                end = min(start + self.blocksize, self.size)
                return self.fetcher(start, end)
                
            if os.path.exists(block_path):
                try:
                    os.utime(block_path, None)
                    with open(block_path, "rb") as f:
                        return f.read()
                except Exception:
                    pass
                    
            start = block_number * self.blocksize
            end = min(start + self.blocksize, self.size)
            data = self.fetcher(start, end)
            
            try:
                tmp_path = f"{block_path}.tmp.{os.getpid()}"
                with open(tmp_path, "wb") as f_tmp:
                    f_tmp.write(data)
                os.rename(tmp_path, block_path)
            except Exception:
                pass
            
            if random.random() < 0.05:
                self._evict()
                
            return data
        finally:
            if f_lock:
                try:
                    fcntl.flock(f_lock, fcntl.LOCK_UN)
                    f_lock.close()
                except Exception:
                    pass

    def _evict(self):
        try:
            if not os.path.exists(self.cache_dir):
                return
                
            files = []
            total_size = 0
            for f in os.listdir(self.cache_dir):
                if f.endswith('.lock') or '.tmp.' in f:
                    continue
                full_path = os.path.join(self.cache_dir, f)
                try:
                    st = os.stat(full_path)
                    files.append((st.st_mtime, full_path, st.st_size))
                    total_size += st.st_size
                except Exception:
                    pass
                    
            if total_size <= self.max_size_bytes:
                return
                
            files.sort(key=lambda x: x[0])
            for mtime, path, size in files:
                try:
                    os.remove(path)
                    total_size -= size
                    if total_size <= self.max_size_bytes:
                        break
                except Exception:
                    pass
        except Exception:
            pass

register_cache(CrossProcessBlockCache, clobber=True)

