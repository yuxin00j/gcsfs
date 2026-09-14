"""Integration tests for the cross-process `get_file` cache.

These exercise `GCSFileSystem._get_file` end to end with the network layer
(`_info` / `_get_file_direct`) stubbed out, so they run offline and assert on
the cache's own behaviour: single-flight election, inode sharing, eligibility
gates, and graceful degradation.
"""

import asyncio
import io
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

import gcsfs.cache_manager as cache_manager
from gcsfs.core import GCSFileSystem

TEST_DATA = b"Simulated 45GB checkpoint payload"
GENERATION = "1234567890"


class _Harness:
    """A GCSFileSystem with the network stubbed and a download counter."""

    def __init__(self, tmpdir, **fs_kwargs):
        self.tmpdir = Path(tmpdir)
        self.cache_dir = self.tmpdir / "cache"
        self.download_calls = 0
        self.direct_kwargs = []
        self.data = TEST_DATA
        self.generation = GENERATION
        self.fs = GCSFileSystem(
            token="anon",
            asynchronous=True,
            cache_dir=str(self.cache_dir),
            **fs_kwargs,
        )
        self.fs._info = self._info
        self.fs._get_file_direct = self._get_file_direct

    async def _info(self, rpath, **kwargs):
        return {
            "name": rpath,
            "size": len(self.data),
            "generation": self.generation,
        }

    async def _get_file_direct(self, rpath, lpath, callback=None, **kwargs):
        self.download_calls += 1
        self.direct_kwargs.append(kwargs)
        # Simulate network download latency so concurrent callers really overlap.
        await asyncio.sleep(0.01)
        dest = Path(lpath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.data)

    def cache_file(self, rpath):
        key = self.fs.cache_manager.get_cache_key(
            rpath, generation=self.generation, size=len(self.data)
        )
        return self.fs.cache_manager.data_dir / f"{key}.data"


def _same_inode(a, b):
    return (a.stat().st_dev, a.stat().st_ino) == (b.stat().st_dev, b.stat().st_ino)


@pytest.fixture
def harness():
    """Cache-enabled filesystem with the min-size gate disabled.

    The gate defaults to 8 MiB, which every payload here is far below, so it
    has to be lowered or none of these tests would touch the cache at all.
    `test_small_object_bypasses_cache` covers the gate itself.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch.dict(os.environ, {"GCSFS_CACHE_MIN_SIZE_BYTES": "0"}):
            yield _Harness(tmpdir, enable_cross_process_cache=True)


@pytest.mark.asyncio
async def test_download_then_hardlink(harness):
    """A cold call downloads once and materializes by hardlink, not by copy."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "rank_0" / "model.ckpt"

    await harness.fs._get_file(rpath, str(dest))

    assert dest.read_bytes() == TEST_DATA
    assert harness.download_calls == 1
    assert _same_inode(dest, harness.cache_file(rpath))


@pytest.mark.asyncio
async def test_second_caller_hits_cache(harness):
    """A warm call performs no network download and shares the same inode."""
    rpath = "my-bucket/checkpoint.ckpt"
    first = harness.tmpdir / "rank_0" / "model.ckpt"
    second = harness.tmpdir / "rank_1" / "model.ckpt"

    await harness.fs._get_file(rpath, str(first))
    await harness.fs._get_file(rpath, str(second))

    assert second.read_bytes() == TEST_DATA
    assert harness.download_calls == 1
    assert _same_inode(second, harness.cache_file(rpath))


@pytest.mark.asyncio
async def test_concurrent_ranks_single_flight(harness):
    """Eight simultaneous callers elect exactly one downloader."""
    rpath = "my-bucket/concurrent_model.ckpt"
    dests = [harness.tmpdir / f"worker_{i}" / "model.ckpt" for i in range(8)]

    await asyncio.gather(*(harness.fs._get_file(rpath, str(p)) for p in dests))

    assert harness.download_calls == 1
    cache_file = harness.cache_file(rpath)
    for p in dests:
        assert p.read_bytes() == TEST_DATA
        assert _same_inode(p, cache_file)


@pytest.mark.asyncio
async def test_writable_gets_isolated_copy(harness):
    """writable=True yields a private copy; mutating it cannot corrupt the cache."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "writable" / "model.ckpt"

    await harness.fs._get_file(rpath, str(dest), writable=True)

    cache_file = harness.cache_file(rpath)
    assert not _same_inode(dest, cache_file)
    dest.write_bytes(b"modified")
    assert cache_file.read_bytes() == TEST_DATA


@pytest.mark.asyncio
async def test_cache_disabled_by_constructor_flag(harness):
    """enable_cross_process_cache=False routes straight to the direct downloader."""
    harness.fs.enable_cross_process_cache = False
    dest = harness.tmpdir / "disabled" / "model.ckpt"

    await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(dest))

    assert harness.download_calls == 1
    assert dest.exists()


@pytest.mark.asyncio
async def test_cache_toggled_by_environment(harness):
    """With no explicit flag, GCSFS_CACHE_ENABLED decides."""
    rpath = "my-bucket/checkpoint.ckpt"
    harness.fs.enable_cross_process_cache = None

    off = harness.tmpdir / "env_off" / "model.ckpt"
    with mock.patch.dict(os.environ, {"GCSFS_CACHE_ENABLED": "false"}):
        await harness.fs._get_file(rpath, str(off))
    assert harness.download_calls == 1
    assert off.exists()

    on = harness.tmpdir / "env_on" / "model.ckpt"
    with mock.patch.dict(os.environ, {"GCSFS_CACHE_ENABLED": "true"}):
        await harness.fs._get_file(rpath, str(on))
    # The cache was still cold, so this call populates it.
    assert harness.download_calls == 2

    again = harness.tmpdir / "env_on_2" / "model.ckpt"
    with mock.patch.dict(os.environ, {"GCSFS_CACHE_ENABLED": "true"}):
        await harness.fs._get_file(rpath, str(again))
    assert harness.download_calls == 2
    assert _same_inode(again, harness.cache_file(rpath))


@pytest.mark.asyncio
async def test_cache_disabled_without_fcntl(harness):
    """Without fcntl there is no way to elect a downloader, so the cache is off.

    Silently proceeding would let every process elect itself and race
    `os.replace` onto the same master file.
    """
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "no_fcntl" / "model.ckpt"

    with mock.patch("gcsfs.core.HAS_FCNTL", False):
        assert harness.fs._cache_enabled() is False
        await harness.fs._get_file(rpath, str(dest))

    assert harness.download_calls == 1
    assert dest.read_bytes() == TEST_DATA
    # Nothing was staged into the cache.
    assert not harness.cache_file(rpath).exists()


@pytest.mark.asyncio
async def test_lock_refuses_to_run_unlocked_without_fcntl():
    """The lock must fail loudly rather than hand back an unlocked context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lock_path = Path(tmpdir) / "test.lock"
        with mock.patch.object(cache_manager, "HAS_FCNTL", False):
            with pytest.raises(RuntimeError, match="fcntl"):
                async with cache_manager.AsyncProcessFileLock(lock_path):
                    pass


@pytest.mark.asyncio
async def test_cache_only_kwargs_never_reach_the_network_layer(harness):
    """`writable` must not survive into **kwargs, where it becomes a query param.

    `_get_file_request` funnels unrecognised keywords through `_get_params`,
    so a leaked kwarg silently ends up in the request URL.
    """
    harness.fs.enable_cross_process_cache = False
    dest = harness.tmpdir / "kwargs" / "model.ckpt"

    await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(dest), writable=True)

    assert "writable" not in harness.direct_kwargs[-1]


@pytest.mark.asyncio
async def test_small_object_bypasses_cache(harness):
    """Objects under GCSFS_CACHE_MIN_SIZE_BYTES skip the cache machinery."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "small" / "model.ckpt"

    with mock.patch.dict(os.environ, {"GCSFS_CACHE_MIN_SIZE_BYTES": "1048576"}):
        await harness.fs._get_file(rpath, str(dest))

    assert harness.download_calls == 1
    assert dest.read_bytes() == TEST_DATA
    assert not harness.cache_file(rpath).exists()


@pytest.mark.asyncio
async def test_staging_download_forces_a_checksum(harness):
    """The cache must not inherit the consistency="none" default.

    A corrupt payload staged into the cache would be served to every later
    reader on the host, so the staging download always asks for a real
    checksum even though `GCSFileSystem` defaults to size-only.
    """
    assert harness.fs.consistency == "none"
    await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(harness.tmpdir / "c"))

    assert harness.direct_kwargs[-1]["consistency"] in ("crc32c", "md5")


@pytest.mark.asyncio
async def test_lock_timeout_degrades_to_direct_download(harness):
    """A contended lock must slow a caller down, never fail it."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "timeout" / "model.ckpt"

    async def always_times_out(self):
        raise TimeoutError(f"Timed out waiting for lock: {self.lock_path}")

    with mock.patch.object(
        cache_manager.AsyncProcessFileLock, "__aenter__", always_times_out
    ):
        await harness.fs._get_file(rpath, str(dest))

    assert harness.download_calls == 1
    assert dest.read_bytes() == TEST_DATA
    assert not harness.cache_file(rpath).exists()


@pytest.mark.asyncio
async def test_clear_cache_is_exposed_on_the_filesystem(harness):
    """Milestone 1 ships no eviction, so the public purge API is the only escape."""
    rpath = "my-bucket/checkpoint.ckpt"
    await harness.fs._get_file(rpath, str(harness.tmpdir / "a" / "model.ckpt"))
    assert harness.cache_file(rpath).exists()

    harness.fs.clear_cache()

    assert not harness.cache_file(rpath).exists()
    assert harness.fs.cache_manager.data_dir.is_dir()


@pytest.mark.asyncio
async def test_file_like_destination_is_streamed(harness):
    """A file-like `lpath` is written directly and bypasses the cache.

    There is no filesystem destination to hardlink, so there is nothing for a
    later reader to share.
    """
    harness.fs._cat_file = mock.AsyncMock(
        side_effect=lambda rpath, start=None, end=None, **kw: TEST_DATA[start:end]
    )
    buf = io.BytesIO()

    await harness.fs._get_file("my-bucket/checkpoint.ckpt", buf)

    assert buf.getvalue() == TEST_DATA
    assert harness.download_calls == 0


@pytest.mark.asyncio
async def test_directory_destination_is_a_no_op(harness):
    """An existing directory as `lpath` returns without downloading."""
    dest = harness.tmpdir / "a_directory"
    dest.mkdir()

    await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(dest))

    assert harness.download_calls == 0
