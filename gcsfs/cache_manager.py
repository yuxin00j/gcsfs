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
            raise RuntimeError(
                "AsyncProcessFileLock requires fcntl, which is unavailable on "
                "this platform. The gcsfs cross-process cache must be disabled."
            )
        start_time = time.monotonic()
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
                fd, self._fd = self._fd, None
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._fd is not None:
            fd, self._fd = self._fd, None

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

    _warned_exdev = False

    def __init__(self, cache_dir: Optional[str] = None):
        base_dir = cache_dir or os.environ.get("GCSFS_CACHE_DIR", "~/.cache/gcsfs")
        self.cache_dir = Path(os.path.expanduser(base_dir))
        self.data_dir = self.cache_dir / "data"
        self.lock_dir = self.cache_dir / "locks"
        for directory in (self.cache_dir, self.data_dir, self.lock_dir):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                directory.chmod(0o700)
            except OSError:
                pass
        self._intra_locks = weakref.WeakValueDictionary()

    def get_cache_key(
        self, rpath: str, generation: Optional[str] = None, size: Optional[int] = None
    ) -> str:
        """Derive the content-addressed key for one normalized object version."""
        if not generation:
            raise ValueError(
                f"Refusing to build a cache key for {rpath!r} without an object "
                "generation: the key would not distinguish object versions."
            )
        raw_identity = f"{rpath}:{generation}:{size or 0}".encode("utf-8")
        return hashlib.sha256(raw_identity).hexdigest()

    def get_intra_lock(self, cache_key: str) -> asyncio.Lock:
        """Return the weakly-held per-key intra-process lock."""
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
        """Projects the master cache file to lpath via hardlink or copy fallback."""
        dest = Path(lpath)
        if dest.is_dir() or str(lpath).endswith(("/", "\\")):
            dest.mkdir(parents=True, exist_ok=True)
            target_name = Path(rpath).name or cache_file.name
            dest = dest / target_name
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)

        # 1. Priority 1: Zero-Copy Read-Only Hardlink (os.link)
        if not writable:
            try:
                if dest.exists() and os.path.samefile(cache_file, dest):
                    return
                try:
                    os.link(cache_file, dest)
                    return
                except FileExistsError:
                    if os.path.samefile(cache_file, dest):
                        return
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
                        "gcsfs cache: %s and %s are on different mounts; falling "
                        "back to copy. Set GCSFS_CACHE_DIR on the same filesystem "
                        "as download destinations for zero-copy hardlinks.",
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

        # 2. Priority 2: Isolated Chunked Copy (EXDEV, EMLINK, EPERM, or writable=True)
        if dest.exists():
            try:
                dest_st = dest.stat()
                cache_st = cache_file.stat()
                if (
                    dest_st.st_size == cache_st.st_size
                    and dest_st.st_mtime_ns == cache_st.st_mtime_ns
                    and (not writable or bool(dest_st.st_mode & 0o200))
                ):
                    return
            except OSError:
                pass

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

    def clear_cache(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
