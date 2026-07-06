import asyncio
import os
import shutil
from unittest import mock

import pytest

from gcsfs.caching import ReadAheadChunked
from gcsfs.core import shared_cached_read

# Determine cache directory dynamically based on environment / system support
if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
    CACHE_DIR = "/dev/shm/gcsfs_shared_cache"
else:
    CACHE_DIR = "/tmp/gcsfs_shared_cache"


class MockVectorFetcher:
    """Simulates a backend capable of vector reads (accepting chunk_lengths)."""

    def __init__(self, data: bytes):
        self.data = data
        self.call_log = []

    def __call__(self, start, chunk_lengths):
        self.call_log.append({"start": start, "chunk_lengths": chunk_lengths})
        results = []
        current = start
        for length in chunk_lengths:
            end = min(current + length, len(self.data))
            results.append(self.data[current:end])
            current += length
        return results


@pytest.fixture
def source_data():
    """Generates 100 bytes of sequential data."""
    return bytes(range(100))


@pytest.fixture
def cache_setup(source_data):
    """Returns a tuple of (cache_instance, fetcher_mock)."""
    fetcher = MockVectorFetcher(source_data)
    # Blocksize 10, File size 100
    cache = ReadAheadChunked(blocksize=10, fetcher=fetcher, size=100)
    return cache, fetcher


def test_initial_state(cache_setup):
    cache, _ = cache_setup
    assert cache.cache == b""
    assert len(cache.chunks) == 0
    assert cache.hit_count == 0
    assert cache.miss_count == 0


def test_fetch_with_readahead(cache_setup, source_data):
    """Test a basic fetch. Should retrieve requested data + blocksize readahead."""
    cache, fetcher = cache_setup

    # Request bytes 0-5
    result = cache._fetch(0, 5)

    # 1. Verify data correctness
    assert result == source_data[0:5]

    # 2. Verify Fetcher calls
    # Should fetch requested (5) + readahead (10)
    assert len(fetcher.call_log) == 1
    assert fetcher.call_log[0]["start"] == 0
    assert fetcher.call_log[0]["chunk_lengths"] == [5, 10]

    # 3. Verify Internal State (Deque)
    # We expect two chunks: the requested part (0-5) and readahead (5-15)
    assert len(cache.chunks) == 2
    assert cache.chunks[0] == (0, 5, source_data[0:5])
    assert cache.chunks[1] == (5, 15, source_data[5:15])

    # 4. Verify compatibility property
    assert cache.cache == source_data[0:15]


def test_cache_hit_fully_contained(cache_setup, source_data):
    """Test fetching data that is already inside the readahead buffer."""
    cache, fetcher = cache_setup

    # Prime the cache (fetch 0-5, readahead 5-15)
    cache._fetch(0, 5)

    # Reset call log to ensure next fetch doesn't hit backend
    fetcher.call_log = []

    # Request 5-10 (Should be inside the readahead chunk)
    result = cache._fetch(5, 10)

    assert result == source_data[5:10]
    assert len(fetcher.call_log) == 0  # No backend calls
    assert cache.hit_count == 1


def test_cache_hit_spanning_chunks(cache_setup, source_data):
    """Test fetching data that spans across the requested chunk and the readahead chunk."""
    cache, fetcher = cache_setup

    # Prime cache: Chunk 1 (0-5), Chunk 2 (5-15)
    cache._fetch(0, 5)

    # Request 2-8 (Spans Chunk 1 and Chunk 2)
    result = cache._fetch(2, 8)

    assert result == source_data[2:8]
    # Should join parts internally without fetching new data
    assert cache.hit_count == 1
    assert len(fetcher.call_log) == 1  # Only the initial prime call


def test_backward_seek_clears_cache(cache_setup, source_data):
    """Test that seeking backwards (before current window) clears cache and refetches."""
    cache, fetcher = cache_setup

    # Prime cache at 50-60 (Readahead 60-70)
    cache._fetch(50, 60)
    assert cache.chunks[0][0] == 50

    # Seek backwards to 20
    fetcher.call_log = []
    result = cache._fetch(20, 30)

    assert result == source_data[20:30]
    # Cache should have cleared and fetched new
    assert fetcher.call_log[0]["start"] == 20
    assert cache.chunks[0][0] == 20


def test_forward_seek_miss(cache_setup, source_data):
    """Test requesting data far ahead of the current window."""
    cache, fetcher = cache_setup

    # Prime 0-5
    cache._fetch(0, 5)

    # Jump to 50
    fetcher.call_log = []
    result = cache._fetch(50, 55)

    assert result == source_data[50:55]
    # Should clear old chunks and fetch new
    assert len(cache.chunks) == 2  # 50-55 and readahead
    assert cache.chunks[0][0] == 50


def test_zero_copy_optimization(cache_setup, source_data):
    """Verify that if we request a chunk exactly, it returns the original object without slicing (identity check)."""
    cache, _ = cache_setup

    # Prime cache: Chunks will be (0, 5, data) and (5, 15, data)
    cache._fetch(0, 5)

    # Fetch exactly the second chunk (readahead buffer)
    # The logic inside _fetch has a check: if slice_start==0 and slice_end==len...
    exact_chunk = cache._fetch(5, 15)

    # Verify values
    assert exact_chunk == source_data[5:15]

    # Verify Identity (Zero Copy)
    # Note: string/bytes literals might be interned, but since we slice from source_data,
    # identity checks on the deque contents vs result should pass if logic holds.
    stored_readahead = cache.chunks[1][2]
    assert exact_chunk is stored_readahead


def test_end_of_file_truncation(cache_setup, source_data):
    """Ensure readahead doesn't go past file size."""
    cache, fetcher = cache_setup
    # File size is 100.

    # Fetch 95-100.
    # missing_len = 5.
    # readahead would usually be 10, but file ends at 100.
    result = cache._fetch(95, 100)

    assert result == source_data[95:100]
    assert len(fetcher.call_log) == 1

    # Check lengths requested.
    # Request: 5 bytes. Remaining space: 0. Readahead should be 0.
    args = fetcher.call_log[0]
    assert args["start"] == 95
    # Should only request the 5 bytes needed, no readahead
    assert args["chunk_lengths"] == [5]

    # Ensure no empty readahead chunk was added
    assert len(cache.chunks) == 1


def test_none_arguments(cache_setup, source_data):
    """Test behavior when start/end are None."""
    cache, _ = cache_setup

    # Fetch all
    result = cache._fetch(None, None)
    assert len(result) == 100
    assert result == source_data


def test_out_of_bounds(cache_setup):
    """Test start >= size returns empty."""
    cache, _ = cache_setup
    assert cache._fetch(150, 200) == b""


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

    try:
        # 1. First coalesced read (cache miss, performs fetch)
        res1 = await shared_cached_read("bucket/file_mp", 0, 50, mock_fetch)
        assert res1 == b"shared_process_bytes"
        assert fetch_count == 1

        # Check that cache file exists
        assert os.path.exists(CACHE_DIR)
        files = os.listdir(CACHE_DIR)
        assert len(files) > 0

        # 2. Second coalesced read (cache hit, bypasses fetch)
        res2 = await shared_cached_read("bucket/file_mp", 0, 50, mock_fetch)
        assert res2 == b"shared_process_bytes"
        # fetch_count should STILL be 1!
        assert fetch_count == 1
    finally:
        # Cleanup guaranteed even if assertions fail
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


@pytest.mark.asyncio
async def test_filesystem_instances_share_cache():
    """
    Tests that distinct GCSFileSystem instances share the same file cache
    so that a file range read by one instance is served from cache for another.
    """
    from gcsfs.core import GCSFileSystem

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    fs1 = GCSFileSystem(token="anon")
    fs2 = GCSFileSystem(token="anon")

    call_count = 0

    async def mock_call(method, url, headers=None, **kwargs):
        nonlocal call_count
        call_count += 1
        return {}, b"file_instance_shared_data"

    try:
        with mock.patch.object(fs1, "_call", side_effect=mock_call):
            with mock.patch.object(fs2, "_call", side_effect=mock_call):
                # Instance 1 fetches data (cache miss)
                res1 = await fs1._cat_file_sequential(
                    "mybucket/shared_file.txt", start=0, end=26
                )
                assert res1 == b"file_instance_shared_data"
                assert call_count == 1

                # Instance 2 reads the same range (cache hit, _call not invoked)
                res2 = await fs2._cat_file_sequential(
                    "mybucket/shared_file.txt", start=0, end=26
                )
                assert res2 == b"file_instance_shared_data"
                assert call_count == 1
    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


def _process_multi_range_read_task(
    path, ranges, cache_dir, barrier, result_queue, fetch_counter
):
    """Worker process function for testing multi-range multi-process cache concurrency."""
    import asyncio

    from gcsfs.core import shared_cached_read

    async def _async_worker():
        results = []
        for start, end in ranges:

            async def mock_fetch(s=start, e=end):
                with fetch_counter.get_lock():
                    fetch_counter.value += 1
                await asyncio.sleep(0.05)
                return f"data_{s}_{e}".encode()

            # Wait at barrier for each range so both processes hit shared_cached_read simultaneously
            barrier.wait(timeout=5)
            res = await shared_cached_read(path, start, end, mock_fetch)
            results.append(((start, end), res))
        result_queue.put(results)

    asyncio.run(_async_worker())


@pytest.mark.asyncio
async def test_e2e_multiprocess_shared_cache_concurrency_and_invalidation():
    """
    End-to-End Test:
    1. Spawns two distinct OS processes that simultaneously attempt to read multiple ranges of the same file.
    2. Verifies that for each range, only 1 process queries the backend while the other reads from cache.
    3. Verifies that repeated reads of previously fetched ranges hit the cache.
    4. Verifies that invalidate_cache purges all shared range cache files for the path.
    """
    import hashlib
    import multiprocessing

    from gcsfs.core import GCSFileSystem

    path = "e2e_bucket/multi_range_test_file.bin"
    ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (0, 100)]
    unique_ranges = {(0, 100), (100, 200), (200, 300), (300, 400)}

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    try:
        # 1. Multi-Process Concurrency Test across 5 range reads (4 unique)
        barrier = multiprocessing.Barrier(2)
        result_queue = multiprocessing.Queue()
        fetch_counter = multiprocessing.Value("i", 0)

        p1 = multiprocessing.Process(
            target=_process_multi_range_read_task,
            args=(path, ranges, CACHE_DIR, barrier, result_queue, fetch_counter),
        )
        p2 = multiprocessing.Process(
            target=_process_multi_range_read_task,
            args=(path, ranges, CACHE_DIR, barrier, result_queue, fetch_counter),
        )

        p1.start()
        p2.start()

        p1.join(timeout=15)
        p2.join(timeout=15)

        assert p1.exitcode == 0, "Process 1 failed"
        assert p2.exitcode == 0, "Process 2 failed"

        r1 = result_queue.get(timeout=2)
        r2 = result_queue.get(timeout=2)

        # Verify correct data returned for all ranges in both processes
        for (start, end), data in r1:
            assert data == f"data_{start}_{end}".encode()
        for (start, end), data in r2:
            assert data == f"data_{start}_{end}".encode()

        # Exactly 4 fetches must be performed across both OS processes for 4 unique ranges
        assert fetch_counter.value == len(unique_ranges)

        # 2. Verify cache files exist for all 4 unique ranges
        path_hash = hashlib.sha256(path.encode()).hexdigest()
        assert os.path.exists(CACHE_DIR)
        cached_files = [
            f
            for f in os.listdir(CACHE_DIR)
            if f.startswith(path_hash) and not f.endswith(".lock")
        ]
        assert len(cached_files) == len(unique_ranges)

        # 3. Invalidation Test: Purge all ranges for this path
        fs = GCSFileSystem(token="anon")
        fs.invalidate_cache(path)

        # All range cache files for this path should be unlinked
        matching_files = [f for f in os.listdir(CACHE_DIR) if f.startswith(path_hash)]
        assert (
            len(matching_files) == 0
        ), f"Cache files remaining after invalidation: {matching_files}"

    finally:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
