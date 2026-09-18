"""Integration tests for the cross-process `get_file` cache.

These exercise `GCSFileSystem._get_file` end to end with the network layer
(`_info` / `_get_file_direct`) stubbed out, so they run offline and assert on
the cache's own behaviour: single-flight election, inode sharing, eligibility
gates, and graceful degradation.
"""

import asyncio
import errno
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
        # Callers can point the cache somewhere unusable to exercise the
        # degradation paths, so this is an override rather than a fixed value.
        self.cache_dir = Path(
            fs_kwargs.pop("cross_process_cache_dir", self.tmpdir / "cache")
        )
        self.download_calls = 0
        self.info_calls = 0
        self.direct_kwargs = []
        self.data = TEST_DATA
        self.generation = GENERATION
        self.fs = GCSFileSystem(
            token="anon",
            asynchronous=True,
            cross_process_cache_dir=str(self.cache_dir),
            **fs_kwargs,
        )
        self.fs._info = self._info
        self.fs._get_file_direct = self._get_file_direct

    async def _info(self, rpath, **kwargs):
        self.info_calls += 1
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
            self.fs._strip_protocol(rpath),
            generation=self.generation,
            size=len(self.data),
        )
        return self.fs.cache_manager.data_dir / f"{key}.data"


def _same_inode(a, b):
    return (a.stat().st_dev, a.stat().st_ino) == (b.stat().st_dev, b.stat().st_ino)


@pytest.fixture
def harness():
    """Cache-enabled filesystem with the min-size gate disabled.

    The gate defaults to 1 MiB, which every payload here is far below, so it
    has to be lowered or none of these tests would touch the cache at all.
    `test_small_object_bypasses_cache` covers the gate itself.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch.dict(os.environ, {"GCSFS_CACHE_MIN_SIZE_BYTES": "0"}):
            yield _Harness(tmpdir, cross_process_cache=True)


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
async def test_cross_mount_falls_back_to_copy(harness):
    """When the cache and the destination sit on different mounts, copy instead."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "other_mount" / "model.ckpt"
    real_link = os.link

    def fake_link(src, dst, **kwargs):
        if Path(dst).parent == dest.parent:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return real_link(src, dst, **kwargs)

    with mock.patch("gcsfs.cache_manager.os.link", fake_link):
        await harness.fs._get_file(rpath, str(dest))

    cache_file = harness.cache_file(rpath)
    assert dest.read_bytes() == TEST_DATA
    assert not _same_inode(dest, cache_file)
    # The copy is independent, so writing to it cannot corrupt the cache.
    dest.write_bytes(b"modified")
    assert cache_file.read_bytes() == TEST_DATA


@pytest.mark.asyncio
async def test_cache_disabled_by_constructor_flag(harness):
    """cross_process_cache=False routes straight to the direct downloader."""
    harness.fs.cross_process_cache = False
    dest = harness.tmpdir / "disabled" / "model.ckpt"

    await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(dest))

    assert harness.download_calls == 1
    assert dest.exists()


@pytest.mark.asyncio
async def test_cache_toggled_by_environment(harness):
    """With no explicit flag, GCSFS_CACHE_ENABLED decides."""
    rpath = "my-bucket/checkpoint.ckpt"
    harness.fs.cross_process_cache = None

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
async def test_transfer_failure_propagates_without_a_second_download(harness):
    """A failed GCS read must surface, not be retried uncached.

    aiohttp's connection errors subclass OSError, so the cache's fail-safe
    guard would otherwise treat a dropped connection as a broken cache and
    transfer the whole object a second time.
    """
    dest = harness.tmpdir / "transfer_fail" / "model.ckpt"

    async def failing_download(rpath, lpath, callback=None, **kwargs):
        harness.download_calls += 1
        raise ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")

    harness.fs._get_file_direct = failing_download

    with pytest.raises(ConnectionResetError):
        await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(dest))

    assert harness.download_calls == 1


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
async def test_staging_stream_is_never_hashed_inline(harness):
    """The staging download must not ask the stream to checksum.

    `_get_file_concurrent` calls `checker.update()` on the event loop thread,
    so hashing a multi-GiB object inline stalls every other coroutine in the
    process. Staging validates payload size only -- which is safe precisely
    because a caller who configured a checksum never reaches the cache (see
    `test_configured_consistency_bypasses_the_cache`).
    """
    await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(harness.tmpdir / "c"))

    # Not forced to "none" -- simply never overridden, so the filesystem's own
    # default applies. Forcing it here is what would silently downgrade a
    # caller who asked for crc32c.
    assert "consistency" not in harness.direct_kwargs[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("consistency", ["crc32c", "md5", "size"])
async def test_configured_consistency_bypasses_the_cache(harness, consistency):
    """A caller who asked for a checksum gets an uncached, verified download.

    The cache validates size only, so serving this caller from it would
    silently weaken a guarantee they explicitly configured. Declining is the
    only honest option until verification moves off the event loop.
    """
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "verified" / "model.ckpt"
    harness.fs.consistency = consistency

    await harness.fs._get_file(rpath, str(dest))

    assert harness.download_calls == 1
    assert dest.read_bytes() == TEST_DATA
    assert not harness.cache_file(rpath).exists()
    # The caller's choice reaches the network layer untouched.
    assert harness.direct_kwargs[-1].get("consistency", consistency) == consistency


@pytest.mark.asyncio
async def test_per_call_consistency_bypasses_the_cache(harness):
    """A per-call `consistency=` is honoured the same way as the fs-level one."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "percall" / "model.ckpt"

    await harness.fs._get_file(rpath, str(dest), consistency="crc32c")

    assert harness.download_calls == 1
    assert not harness.cache_file(rpath).exists()
    assert harness.direct_kwargs[-1]["consistency"] == "crc32c"


@pytest.mark.asyncio
async def test_cache_key_ignores_the_protocol_prefix(harness):
    """`gs://bucket/obj` and `bucket/obj` must resolve to one cache entry.

    `fs.get()` hands `_get_file` a stripped path while a direct
    `fs.get_file("gs://...")` does not. Keying on the raw string would give the
    same object two keys, two locks and two downloads, with no symptom beyond
    the wall clock.
    """
    plain = "my-bucket/checkpoint.ckpt"
    prefixed = "gs://my-bucket/checkpoint.ckpt"
    first = harness.tmpdir / "plain" / "model.ckpt"
    second = harness.tmpdir / "prefixed" / "model.ckpt"

    await harness.fs._get_file(plain, str(first))
    await harness.fs._get_file(prefixed, str(second))

    assert harness.download_calls == 1
    assert second.read_bytes() == TEST_DATA
    assert _same_inode(first, second)
    assert _same_inode(second, harness.cache_file(prefixed))


@pytest.mark.asyncio
async def test_bypasses_reuse_the_metadata_already_fetched(harness):
    """A bypass must not re-fetch `_info` the cache already paid for.

    The size gate needs the object metadata, so the cache fetches it before it
    can decline. Handing that same `details` to the fallback keeps an enabled
    cache from doubling metadata round trips on every small object.
    """
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "reuse" / "model.ckpt"

    with mock.patch.dict(os.environ, {"GCSFS_CACHE_MIN_SIZE_BYTES": "1048576"}):
        await harness.fs._get_file(rpath, str(dest))

    assert harness.info_calls == 1
    assert harness.direct_kwargs[-1]["details"]["generation"] == GENERATION


@pytest.mark.asyncio
async def test_get_file_concurrent_uses_supplied_details(harness):
    """`_get_file_concurrent` skips its own `_info` when handed `details`."""
    fs = harness.fs
    fs._get_file_request = mock.AsyncMock()
    fs._get_threshold_for_disk_reads = mock.AsyncMock(return_value=0)
    details = {"name": "my-bucket/o", "size": 0, "generation": GENERATION}

    await fs._get_file_concurrent(
        "my-bucket/o",
        str(harness.tmpdir / "unused"),
        4,
        chunk_size=1024,
        max_prefetch_size=1024,
        details=details,
    )

    assert harness.info_calls == 0


@pytest.mark.asyncio
async def test_truncated_download_degrades_to_direct_download(harness):
    """A truncated stage (staged_size != expected_size) is rejected and degrades cleanly."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "truncated" / "model.ckpt"

    call_count = 0

    async def _truncate_first_then_succeed(rpath_, lpath, callback=None, **kwargs):
        nonlocal call_count
        call_count += 1
        harness.download_calls += 1
        harness.direct_kwargs.append(kwargs)
        Path(lpath).parent.mkdir(parents=True, exist_ok=True)
        if call_count == 1:
            # Truncated stage into the cache temp file
            Path(lpath).write_bytes(TEST_DATA[:-5])
        else:
            # Direct fallback download succeeds
            Path(lpath).write_bytes(TEST_DATA)

    harness.fs._get_file_direct = _truncate_first_then_succeed

    await harness.fs._get_file(rpath, str(dest))

    # Staging was rejected and unlinked; cache file was not published; caller got direct download.
    assert not harness.cache_file(rpath).exists()
    assert not any(harness.fs.cache_manager.data_dir.iterdir())
    assert dest.read_bytes() == TEST_DATA
    assert harness.download_calls == 2


@pytest.mark.asyncio
async def test_missing_generation_bypasses_cache(harness):
    """Without a generation the key cannot distinguish versions, so do not cache.

    Keying on a placeholder would collide two same-sized versions of an object
    and serve stale bytes with complete confidence.
    """
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "no_gen" / "model.ckpt"
    harness.generation = ""

    await harness.fs._get_file(rpath, str(dest))

    assert harness.download_calls == 1
    assert dest.read_bytes() == TEST_DATA
    assert not any(harness.fs.cache_manager.data_dir.iterdir())


@pytest.mark.asyncio
async def test_full_cache_degrades_to_direct_download(harness):
    """A full cache directory must slow a job down, never fail it."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "enospc" / "model.ckpt"

    # Report the filesystem as full so the pre-flight check cannot be satisfied.
    with mock.patch(
        "gcsfs.core.shutil.disk_usage",
        return_value=mock.Mock(total=100, used=100, free=0),
    ):
        await harness.fs._get_file(rpath, str(dest))

    assert dest.read_bytes() == TEST_DATA
    assert not harness.cache_file(rpath).exists()


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
async def test_purge_is_exposed_on_the_filesystem(harness):
    """The cache reclaims only on demand under space pressure, so an explicit purge is
    still the only way to drop a cache that nothing is currently competing for."""
    rpath = "my-bucket/checkpoint.ckpt"
    await harness.fs._get_file(rpath, str(harness.tmpdir / "a" / "model.ckpt"))
    assert harness.cache_file(rpath).exists()

    harness.fs.clear_cross_process_cache()

    assert not harness.cache_file(rpath).exists()
    assert harness.fs.cache_manager.data_dir.is_dir()


def test_purge_is_not_named_clear_cache():
    """`clear_cache` would read as a sibling of `invalidate_cache`.

    One drops an in-memory listing, the other unlinks tens of gigabytes. The
    names must not invite confusion between them.
    """
    assert not hasattr(GCSFileSystem, "clear_cache")
    assert hasattr(GCSFileSystem, "clear_cross_process_cache")


@pytest.mark.asyncio
async def test_directory_destination_is_a_no_op(harness):
    """An existing directory as `lpath` returns without downloading."""
    dest = harness.tmpdir / "a_directory"
    dest.mkdir()

    await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(dest))

    assert harness.download_calls == 0


@pytest.mark.asyncio
async def test_cache_logs_to_gcsfs_cache_logger(harness, caplog):
    """Cache events log to the dedicated `gcsfs.cache` logger (Section 7.2)."""
    import logging

    rpath = "my-bucket/checkpoint.ckpt"
    first = harness.tmpdir / "log_0" / "model.ckpt"
    second = harness.tmpdir / "log_1" / "model.ckpt"

    with caplog.at_level(logging.DEBUG, logger="gcsfs.cache"):
        await harness.fs._get_file(rpath, str(first))
        await harness.fs._get_file(rpath, str(second))

    messages = [rec.message for rec in caplog.records if rec.name == "gcsfs.cache"]
    assert any("miss for" in m for m in messages)
    assert any("elected downloader for" in m for m in messages)
    assert any("hit for" in m for m in messages)


@pytest.mark.asyncio
async def test_uncreatable_cache_dir_degrades_to_direct_download():
    """Setting the cache up must not fail a download that would have succeeded.

    `~/.cache/gcsfs` is uncreatable under a read-only root filesystem, which is
    routine in hardened containers. Constructing the cache manager (and so
    creating the directory) therefore has to sit inside the same guard as every
    other cache failure, not above it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        readonly_parent = tmpdir / "readonly"
        readonly_parent.mkdir()
        os.chmod(readonly_parent, 0o500)
        try:
            harness = _Harness(
                tmpdir,
                cross_process_cache=True,
                cross_process_cache_dir=str(readonly_parent / "cache"),
            )
            dest = tmpdir / "out" / "model.ckpt"

            with mock.patch.dict(os.environ, {"GCSFS_CACHE_MIN_SIZE_BYTES": "0"}):
                await harness.fs._get_file("my-bucket/checkpoint.ckpt", str(dest))

            assert dest.read_bytes() == TEST_DATA
            assert harness.download_calls == 1
        finally:
            # Restore write permission so the TemporaryDirectory can clean up.
            os.chmod(readonly_parent, 0o700)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variable", ["GCSFS_CACHE_MIN_SIZE_BYTES", "GCSFS_CACHE_LOCK_TIMEOUT_SEC"]
)
async def test_malformed_cache_env_var_degrades_to_direct_download(harness, variable):
    """A mistyped GCSFS_CACHE_* value must not fail the caller either."""
    rpath = "my-bucket/checkpoint.ckpt"
    dest = harness.tmpdir / "badenv" / "model.ckpt"

    with mock.patch.dict(os.environ, {variable: "not-a-number"}):
        await harness.fs._get_file(rpath, str(dest))

    assert dest.read_bytes() == TEST_DATA
    assert harness.download_calls == 1
    assert not harness.cache_file(rpath).exists()
