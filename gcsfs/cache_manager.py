import asyncio
import errno

try:
    import fcntl

    HAS_FCNTL = True
except ImportError:
    fcntl = None
    HAS_FCNTL = False
import hashlib
import logging
import os
import shutil
import time
import uuid
import weakref
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("gcsfs.cache")


class AsyncProcessFileLock:
    """Cooperative cross-process file lock using fcntl.flock."""

    def __init__(
        self,
        lock_path: Union[str, Path],
        poll_interval: float = 0.05,
        timeout: float = 3600.0,
    ):
        self.lock_path = Path(lock_path)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._fd: Optional[int] = None

    async def __aenter__(self):
        if not HAS_FCNTL:
            # Never silently hand back an unlocked context manager: callers rely
            # on this for single-flight election. Callers must gate on HAS_FCNTL
            # (see GCSFileSystem._cache_enabled) before constructing the lock.
            raise RuntimeError(
                "AsyncProcessFileLock requires fcntl, which is unavailable on "
                "this platform. The gcsfs cross-process cache must be disabled."
            )
        start_time = time.monotonic()

        # 0o600 for four reasons:
        #   1. Floor for O_RDWR: both owner read and write bits are required.
        #   2. Local DoS protection: anyone able to open the lock file can take
        #      LOCK_EX and hold it, wedging single-flight election until
        #      GCSFS_CACHE_LOCK_TIMEOUT_SEC expires.
        #   3. Trust domain parity: matches the 0o700 cache directory (see
        #      GCSFileSystemCacheManager.__init__).
        #   4. Umask determinism: unlike 0o666 (which yields 0o644 under 0o022
        #      vs. 0o600 under 0o077), 0o600 produces identical permissions
        #      across ambient process umasks.
        self._fd = await asyncio.to_thread(
            os.open, self.lock_path, os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            while True:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except (BlockingIOError, OSError) as e:
                    if isinstance(e, OSError) and not isinstance(e, BlockingIOError):
                        if e.errno not in (
                            errno.EACCES,
                            errno.EAGAIN,
                            errno.EWOULDBLOCK,
                        ):
                            raise
                    if time.monotonic() - start_time > self.timeout:
                        raise TimeoutError(
                            f"Timed out waiting for lock: {self.lock_path}"
                        )
                    await asyncio.sleep(self.poll_interval)
        except BaseException:
            if self._fd is not None:
                fd = self._fd
                self._fd = None
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._fd is not None:
            fd = self._fd
            self._fd = None

            def _cleanup():
                try:
                    if HAS_FCNTL and fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                finally:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

            await asyncio.to_thread(_cleanup)


class GCSFileSystemCacheManager:
    """Manages single-flight downloads and zero-copy materialization for GCSFS."""

    # Process-wide latch so the cross-mount warning is emitted only once.
    _warned_exdev = False

    def __init__(self, cache_dir: Optional[str] = None):
        base_dir = cache_dir or os.environ.get("GCSFS_CACHE_DIR", "~/.cache/gcsfs")
        self.cache_dir = Path(os.path.expanduser(base_dir))
        self.data_dir = self.cache_dir / "data"
        self.lock_dir = self.cache_dir / "locks"
        # 0o700 restricts cache access to the current UID. Explicit chmod
        # bypasses umask and tightens pre-existing directories; ignore EPERM if
        # we don't own a shared parent directory.
        for directory in (self.cache_dir, self.data_dir, self.lock_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                directory.chmod(0o700)
            except OSError as e:
                logger.debug(
                    "gcsfs cache: could not tighten permissions on %s (%s)",
                    directory,
                    errno.errorcode.get(e.errno, e.errno),
                )
        self._intra_locks = weakref.WeakValueDictionary()

    def get_cache_key(
        self, rpath: str, generation: Optional[str] = None, size: Optional[int] = None
    ) -> str:
        if not generation:
            raise ValueError(
                f"Refusing to build a cache key for {rpath!r} without an object "
                "generation: the key would not distinguish object versions."
            )
        raw_identity = f"{rpath}:{generation}:{size or 0}".encode("utf-8")
        return hashlib.sha256(raw_identity).hexdigest()

    def get_intra_lock(self, cache_key: str) -> asyncio.Lock:
        """Return the per-key intra-process lock, creating it on first use.

        Entries are weakly held so the registry cannot grow without bound: a
        caller holding (or awaiting) the lock keeps it alive, and it is
        reclaimed once nothing references it. There must be no ``await``
        between the lookup and the insert, which makes this atomic with
        respect to the event loop.
        """
        lock = self._intra_locks.get(cache_key)
        if lock is None:
            lock = asyncio.Lock()
            self._intra_locks[cache_key] = lock
        return lock

    def materialize(
        self,
        cache_file: Path,
        lpath: Union[str, Path],
        rpath: str = "",
        writable: bool = False,
    ) -> None:
        """Projects the master cache file to lpath via zero-copy hardlink or copy fallback."""
        dest = Path(lpath)
        if dest.is_dir() or str(lpath).endswith(("/", "\\")):
            dest.mkdir(parents=True, exist_ok=True)
            target_name = Path(rpath).name or cache_file.name
            dest = dest / target_name
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)

        # 1. Short-Circuit Check
        if dest.exists():
            try:
                dest_st = dest.stat()
                cache_st = cache_file.stat()
                if (dest_st.st_dev, dest_st.st_ino) == (
                    cache_st.st_dev,
                    cache_st.st_ino,
                ):
                    if not writable:
                        return
                elif (
                    dest_st.st_size == cache_st.st_size
                    and dest_st.st_mtime_ns == cache_st.st_mtime_ns
                    and (not writable or bool(dest_st.st_mode & 0o200))
                ):
                    return
            except OSError:
                pass

        # 2. Priority 1: Zero-Copy Read-Only Hardlink (os.link)
        if not writable:
            try:
                try:
                    os.link(cache_file, dest)
                    return
                except FileExistsError:
                    try:
                        if (dest.stat().st_dev, dest.stat().st_ino) == (
                            cache_file.stat().st_dev,
                            cache_file.stat().st_ino,
                        ):
                            return
                    except OSError:
                        pass
                    tmp_target = (
                        dest.parent
                        / f".{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
                    )
                    try:
                        os.link(cache_file, tmp_target)
                        os.replace(tmp_target, dest)
                        return
                    finally:
                        if tmp_target.exists():
                            tmp_target.unlink(missing_ok=True)
            except OSError as e:
                if e.errno not in (
                    errno.EXDEV,
                    errno.EMLINK,
                    errno.EPERM,
                    errno.EACCES,
                ):
                    raise
                if (
                    e.errno == errno.EXDEV
                    and not GCSFileSystemCacheManager._warned_exdev
                ):
                    GCSFileSystemCacheManager._warned_exdev = True
                    logger.warning(
                        "gcsfs cache: %s and %s are on different mounts, so the "
                        "zero-copy hardlink is unavailable and every destination "
                        "costs a full copy. Set GCSFS_CACHE_DIR to a directory on "
                        "the same filesystem as your download destinations.",
                        cache_file,
                        dest,
                    )
                else:
                    logger.debug(
                        "gcsfs cache: hardlink %s -> %s failed (%s); falling back "
                        "to copy",
                        cache_file,
                        dest,
                        errno.errorcode.get(e.errno, e.errno),
                    )

        # 3. Priority 2: Isolated Chunked Copy (EXDEV, EMLINK, EPERM, or writable=True)
        logger.debug(
            "gcsfs cache: copying %s -> %s (writable=%s)", cache_file, dest, writable
        )
        tmp_target = (
            dest.parent / f".{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        try:
            with open(cache_file, "rb") as src, open(tmp_target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            shutil.copystat(cache_file, tmp_target)
            os.chmod(tmp_target, 0o644)
            os.replace(tmp_target, dest)
        finally:
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)

    def _entry_is_idle(self, cache_file: Path) -> bool:
        """True when no process holds the download lock for this entry.

        Probed with a non-blocking ``LOCK_EX`` that is released immediately.
        ``flock`` is owned by the open file description, so a lock held by this
        same process through another descriptor also registers as busy, which
        is what we want.
        """
        lock_file = self.lock_dir / f"{cache_file.stem}.lock"
        if not HAS_FCNTL or not lock_file.exists():
            return not lock_file.exists()
        fd = None
        try:
            fd = os.open(lock_file, os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        except OSError:
            return False
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def reclaim(self, needed_bytes: int, exclude=None) -> int:
        """Delete unreferenced cache entries, oldest first, to free space.

        An entry is reclaimable when nothing points at its blocks
        (``st_nlink == 1``, so no destination was materialized from it by
        hardlink) and no process holds its download lock. Returns bytes freed.

        Ordering is by ``st_atime``, which under the usual ``relatime`` mount
        option is only refreshed when it predates ``st_mtime`` or is over a day
        old -- so this approximates LRU rather than implementing it, and
        degenerates to FIFO under ``noatime``. That is adequate here: the goal
        is to keep the cache usable across checkpoint generations, not to
        maximise hit rate.

        Lock files are deliberately left behind. Unlinking one races a process
        about to open it, which would leave two processes flocking different
        inodes and break single-flight election. They are empty.
        """
        if needed_bytes <= 0:
            return 0
        exclude = exclude or set()

        candidates = []
        for entry in self.data_dir.glob("*.data"):
            if entry.name in exclude:
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            if st.st_nlink > 1:
                # A destination still shares these blocks; removing the cache
                # name would free nothing.
                continue
            candidates.append((st.st_atime, st.st_size, entry))
        candidates.sort(key=lambda c: c[0])

        freed = 0
        for _atime, size, entry in candidates:
            if freed >= needed_bytes:
                break
            if not self._entry_is_idle(entry):
                continue
            try:
                entry.unlink()
            except OSError:
                continue
            freed += size
            logger.debug("gcsfs cache: reclaimed %s (%d bytes)", entry.name, size)

        if freed:
            logger.info(
                "gcsfs cache: reclaimed %d bytes from %s to make room",
                freed,
                self.data_dir,
            )
        return freed

    def clear_cache(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
