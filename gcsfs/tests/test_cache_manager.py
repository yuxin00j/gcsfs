import os
import tempfile
from pathlib import Path

import pytest

from gcsfs.cache_manager import AsyncProcessFileLock, GCSFileSystemCacheManager


@pytest.mark.asyncio
async def test_async_process_file_lock():
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "test.lock"
        async with AsyncProcessFileLock(lock_path, poll_interval=0.01):
            assert lock_path.exists()
            # Try to acquire in a competing coroutine with short timeout -> should time out
            with pytest.raises(TimeoutError):
                async with AsyncProcessFileLock(
                    lock_path, poll_interval=0.01, timeout=0.05
                ):
                    pass


@pytest.mark.asyncio
async def test_async_process_file_lock_cleanup_on_timeout():
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "test.lock"
        async with AsyncProcessFileLock(lock_path, poll_interval=0.01):
            lock2 = AsyncProcessFileLock(lock_path, poll_interval=0.01, timeout=0.02)
            with pytest.raises(TimeoutError):
                async with lock2:
                    pass
            assert lock2._fd is None


@pytest.mark.asyncio
async def test_cache_manager_key_and_locks():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = GCSFileSystemCacheManager(cache_dir=tmpdir)
        key1 = mgr.get_cache_key("bucket/file.bin", generation="123", size=1000)
        key2 = mgr.get_cache_key("bucket/file.bin", generation="123", size=1000)
        key3 = mgr.get_cache_key("bucket/file.bin", generation="124", size=1000)
        assert key1 == key2
        assert key1 != key3

        lock1 = mgr.get_intra_lock(key1)
        lock2 = mgr.get_intra_lock(key1)
        assert lock1 is lock2


def test_cache_manager_materialize_hardlink():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = GCSFileSystemCacheManager(cache_dir=tmpdir)
        cache_file = mgr.data_dir / "test.data"
        cache_file.write_bytes(b"hello world")
        os.chmod(cache_file, 0o444)

        dest = Path(tmpdir) / "dest" / "output.bin"
        mgr.materialize(cache_file, dest)

        assert dest.exists()
        assert dest.read_bytes() == b"hello world"
        dest_st = dest.stat()
        cache_st = cache_file.stat()
        # Should be a hardlink sharing the exact same inode
        assert (dest_st.st_dev, dest_st.st_ino) == (cache_st.st_dev, cache_st.st_ino)

        # Re-materializing should hit short-circuit
        mgr.materialize(cache_file, dest)
        assert dest.exists()


def test_cache_manager_materialize_writable_copy():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = GCSFileSystemCacheManager(cache_dir=tmpdir)
        cache_file = mgr.data_dir / "test.data"
        cache_file.write_bytes(b"hello world")
        os.chmod(cache_file, 0o444)

        dest = Path(tmpdir) / "dest" / "output_writable.bin"
        mgr.materialize(cache_file, dest, writable=True)

        assert dest.exists()
        assert dest.read_bytes() == b"hello world"
        dest_st = dest.stat()
        cache_st = cache_file.stat()
        # Should be an independent copy with different inode
        assert dest_st.st_ino != cache_st.st_ino

        # Should be writable (0o644 or similar)
        dest.write_bytes(b"modified world")
        assert dest.read_bytes() == b"modified world"
        assert cache_file.read_bytes() == b"hello world"
