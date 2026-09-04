import asyncio
import errno
import logging
import os
import time

try:
    import fcntl
except ImportError:
    fcntl = None  # Non-POSIX (e.g. Windows)

logger = logging.getLogger("gcsfs")

_INTRA_PROCESS_LOCKS: dict[str, asyncio.Lock] = {}
_INTRA_LOCK_GUARD = asyncio.Lock()


async def get_intra_process_lock(canonical_path: str) -> asyncio.Lock:
    """Returns an asyncio.Lock for coordinating coroutines within the same process."""
    async with _INTRA_LOCK_GUARD:
        if canonical_path not in _INTRA_PROCESS_LOCKS:
            _INTRA_PROCESS_LOCKS[canonical_path] = asyncio.Lock()
        return _INTRA_PROCESS_LOCKS[canonical_path]


class AsyncProcessFileLock:
    """Async-friendly POSIX cross-process file lock.

    Uses non-blocking fcntl.flock + asyncio.sleep to coordinate multiple OS processes
    without blocking the fsspecIO asyncio event loop.

    Automatic cleanup: if a process crashes, terminates, or is killed via SIGKILL (OOM),
    the OS kernel automatically closes the file descriptor and releases the lock.
    """

    def __init__(
        self,
        lock_path: str,
        timeout: float | None = 600.0,
        poll_interval: float = 0.05,
    ):
        self.lock_path = lock_path
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.fd: int | None = None

    async def __aenter__(self):
        if fcntl is None:
            return self

        os.makedirs(os.path.dirname(self.lock_path) or os.curdir, exist_ok=True)
        # Open lock file. NEVER delete or unlink this file to avoid inode-recycle races.
        self.fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o666)

        start_time = time.monotonic()
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except (BlockingIOError, OSError) as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if self.timeout is not None and (time.monotonic() - start_time) > self.timeout:
                    raise TimeoutError(
                        f"Timed out after {self.timeout}s waiting for lock on {self.lock_path}"
                    )
                await asyncio.sleep(self.poll_interval)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None
