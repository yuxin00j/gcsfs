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


def test_cache_key_requires_a_generation():
    """A placeholder generation would collide same-sized versions of an object."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = GCSFileSystemCacheManager(cache_dir=tmpdir)
        with pytest.raises(ValueError, match="generation"):
            mgr.get_cache_key("bucket/file.bin", generation="", size=1000)
        with pytest.raises(ValueError, match="generation"):
            mgr.get_cache_key("bucket/file.bin", generation=None, size=1000)


def test_cache_directories_are_private():
    """Anyone who can read the cache reads every object in it without credentials."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = GCSFileSystemCacheManager(cache_dir=os.path.join(tmpdir, "cache"))
        for directory in (mgr.cache_dir, mgr.data_dir, mgr.lock_dir):
            assert directory.stat().st_mode & 0o777 == 0o700, directory


def test_reclaim_evicts_oldest_unreferenced_entries_only():
    """Reclamation frees entries nothing is using, oldest first."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = GCSFileSystemCacheManager(cache_dir=os.path.join(tmpdir, "cache"))
        dest_dir = Path(tmpdir) / "dest"
        dest_dir.mkdir()

        old = mgr.data_dir / "aaaa.data"
        new = mgr.data_dir / "bbbb.data"
        linked = mgr.data_dir / "cccc.data"
        for entry in (old, new, linked):
            entry.write_bytes(b"x" * 1024)

        # A materialized destination still shares `linked`'s blocks, so removing
        # the cache name would free nothing.
        os.link(linked, dest_dir / "model.ckpt")

        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        os.utime(linked, (500, 500))

        freed = mgr.reclaim(1024)

        assert freed == 1024
        assert not old.exists()
        assert new.exists()
        assert linked.exists()


def test_reclaim_skips_entries_with_a_held_lock():
    """An entry another process is downloading into must not be pulled out from under it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = GCSFileSystemCacheManager(cache_dir=os.path.join(tmpdir, "cache"))
        busy = mgr.data_dir / "dddd.data"
        busy.write_bytes(b"x" * 1024)
        lock_path = mgr.lock_dir / "dddd.lock"

        import fcntl

        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert mgr.reclaim(1024) == 0
            assert busy.exists()
        finally:
            os.close(fd)

        # Once released it becomes reclaimable.
        assert mgr.reclaim(1024) == 1024
        assert not busy.exists()


def test_reclaim_is_a_no_op_when_nothing_is_needed():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = GCSFileSystemCacheManager(cache_dir=os.path.join(tmpdir, "cache"))
        (mgr.data_dir / "eeee.data").write_bytes(b"x" * 1024)
        assert mgr.reclaim(0) == 0
        assert (mgr.data_dir / "eeee.data").exists()


def test_crc32c_backends_agree():
    """google-crc32c and crcmod must be interchangeable for cache verification.

    google-crc32c uses the CPU's CRC instructions and is roughly 18x faster
    than crcmod's table-driven C extension, so it is preferred -- but only if
    the digests are identical, since GCS compares against one published value.
    """
    import base64

    from gcsfs.checkers import crcmod, google_crc32c

    if google_crc32c is None or crcmod is None:
        pytest.skip("needs both crc32c backends installed")

    for payload in (b"", b"a", os.urandom(1000), os.urandom(1024 * 1024 + 7)):
        a = crcmod.Crc(0x11EDC6F41, initCrc=0, xorOut=0xFFFFFFFF)
        a.update(payload)
        b = google_crc32c.Checksum()
        b.update(payload)
        assert a.digest() == b.digest()
        assert base64.b64encode(a.digest()) == base64.b64encode(b.digest())
