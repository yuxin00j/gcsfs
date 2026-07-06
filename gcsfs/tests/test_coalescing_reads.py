import asyncio
import os
import shutil
import pytest
from unittest import mock
from gcsfs.core import coalesced_read

# Determine cache directory dynamically based on environment / system support
if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
    CACHE_DIR = "/dev/shm/gcsfs_shared_cache"
else:
    CACHE_DIR = "/tmp/gcsfs_shared_cache"

@pytest.mark.asyncio
async def test_in_process_single_flighting():
    """
    Tests that multiple concurrent reads in the same event loop for the
    same file path and range are coalesced into a single actual fetch.
    """
    # Clean up any existing cache for test isolation
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    fetch_count = 0
    async def mock_fetch():
        nonlocal fetch_count
        fetch_count += 1
        await asyncio.sleep(0.05)  # Simulate network latency
        return b"coalesced_bytes"

    # Issue 5 concurrent coalesced reads
    tasks = [
        coalesced_read("test_ns", "bucket/file", 0, 100, mock_fetch)
        for _ in range(5)
    ]

    results = await asyncio.gather(*tasks)

    # Check results
    assert all(res == b"coalesced_bytes" for res in results)
    # The actual network fetch should have been called EXACTLY once!
    assert fetch_count == 1

    # Clean up after test
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


@pytest.mark.asyncio
async def test_multi_process_caching_and_locking():
    """
    Tests that concurrent/sequential reads utilizing the multi-process
    shared cache write to and hit the cache correctly, blocking via flock.
    """
    # Force clean up before/after
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    fetch_count = 0
    async def mock_fetch():
        nonlocal fetch_count
        fetch_count += 1
        await asyncio.sleep(0.05)
        return b"shared_process_bytes"

    # We mock os.environ to make sure coalescing is enabled
    with mock.patch.dict(os.environ, {"GCSFS_COALESCE_READS": "true"}):
        # 1. First coalesced read (cache miss, performs fetch)
        res1 = await coalesced_read("test_mp", "bucket/file_mp", 0, 50, mock_fetch)
        assert res1 == b"shared_process_bytes"
        assert fetch_count == 1

        # Check that cache file exists
        assert os.path.exists(CACHE_DIR)
        files = os.listdir(CACHE_DIR)
        assert len(files) > 0

        # 2. Second coalesced read (cache hit, bypasses fetch)
        res2 = await coalesced_read("test_mp", "bucket/file_mp", 0, 50, mock_fetch)
        assert res2 == b"shared_process_bytes"
        # fetch_count should STILL be 1!
        assert fetch_count == 1

    # Cleanup
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
