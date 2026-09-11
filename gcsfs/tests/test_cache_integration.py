import asyncio
import os
import tempfile
from pathlib import Path

from gcsfs.core import GCSFileSystem


async def run_integration_tests():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        fs = GCSFileSystem(
            token="anon",
            asynchronous=True,
            cache_dir=str(cache_dir),
            enable_cross_process_cache=True,
        )

        test_data = b"Simulated 45GB checkpoint payload"
        download_calls = 0

        async def mock_info(rpath, **kwargs):
            return {
                "name": rpath,
                "size": len(test_data),
                "generation": "1234567890",
            }

        async def mock_get_file_direct(rpath, lpath, callback=None, **kwargs):
            nonlocal download_calls
            download_calls += 1
            # Simulate network download delay
            await asyncio.sleep(0.01)
            dest_path = Path(lpath)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lpath, "wb") as f:
                f.write(test_data)

        fs._info = mock_info
        fs._get_file_direct = mock_get_file_direct

        # Test 1: Single get_file call downloads and materializes via hardlink
        dest1 = Path(tmpdir) / "rank_0" / "model.ckpt"
        await fs._get_file("my-bucket/checkpoint.ckpt", str(dest1))

        assert dest1.exists()
        assert dest1.read_bytes() == test_data
        assert download_calls == 1

        cache_key = fs.cache_manager.get_cache_key(
            "my-bucket/checkpoint.ckpt", generation="1234567890", size=len(test_data)
        )
        cache_file = fs.cache_manager.data_dir / f"{cache_key}.data"
        assert cache_file.exists()
        # Verify hardlink: same inode
        assert (dest1.stat().st_dev, dest1.stat().st_ino) == (
            cache_file.stat().st_dev,
            cache_file.stat().st_ino,
        )
        print("Test 1 (Single download & hardlink): PASSED", flush=True)

        # Test 2: Second call hits cache (0 additional downloads)
        dest2 = Path(tmpdir) / "rank_1" / "model.ckpt"
        await fs._get_file("my-bucket/checkpoint.ckpt", str(dest2))

        assert dest2.exists()
        assert dest2.read_bytes() == test_data
        assert download_calls == 1  # No additional network download!
        assert (dest2.stat().st_dev, dest2.stat().st_ino) == (
            cache_file.stat().st_dev,
            cache_file.stat().st_ino,
        )
        print("Test 2 (Cache hit 0-download): PASSED", flush=True)

        # Test 3: 8 concurrent ranks calling _get_file simultaneously on a new object
        download_calls = 0
        new_rpath = "my-bucket/concurrent_model.ckpt"

        async def mock_info2(rpath, **kwargs):
            return {
                "name": rpath,
                "size": len(test_data),
                "generation": "999999",
            }

        fs._info = mock_info2

        rank_dests = [Path(tmpdir) / f"worker_{i}" / "model.ckpt" for i in range(8)]
        tasks = [fs._get_file(new_rpath, str(p)) for p in rank_dests]
        await asyncio.gather(*tasks)

        assert download_calls == 1  # Exactly 1 download performed for all 8 ranks!
        new_key = fs.cache_manager.get_cache_key(
            new_rpath, generation="999999", size=len(test_data)
        )
        new_cache_file = fs.cache_manager.data_dir / f"{new_key}.data"

        for p in rank_dests:
            assert p.exists()
            assert p.read_bytes() == test_data
            # All 8 ranks share the exact same inode in RAM / disk!
            assert (p.stat().st_dev, p.stat().st_ino) == (
                new_cache_file.stat().st_dev,
                new_cache_file.stat().st_ino,
            )
        print(
            "Test 3 (8 Concurrent Ranks Single-Flight & Inode Sharing): PASSED",
            flush=True,
        )

        # Test 4: Writable copy
        dest_writable = Path(tmpdir) / "writable" / "model.ckpt"
        await fs._get_file(new_rpath, str(dest_writable), writable=True)
        assert dest_writable.exists()
        dest_stat = (dest_writable.stat().st_dev, dest_writable.stat().st_ino)
        cache_stat = (new_cache_file.stat().st_dev, new_cache_file.stat().st_ino)
        assert dest_stat != cache_stat
        dest_writable.write_bytes(b"modified")
        assert new_cache_file.read_bytes() == test_data  # master cache untouched!
        print("Test 4 (Writable isolated copy): PASSED", flush=True)

        # Test 5: Cache disabled via constructor flag
        fs.enable_cross_process_cache = False
        dest_disabled = Path(tmpdir) / "disabled" / "model.ckpt"
        download_calls = 0
        await fs._get_file(new_rpath, str(dest_disabled))
        assert download_calls == 1
        assert dest_disabled.exists()
        print("Test 5 (Cache disabled via constructor): PASSED", flush=True)

        # Test 6: Controlled by GCSFS_CACHE_ENABLED env variable
        fs_env = GCSFileSystem(cache_dir=str(cache_dir), asynchronous=True)
        dest_env_off = Path(tmpdir) / "env_off" / "model.ckpt"
        with mock.patch.dict(os.environ, {"GCSFS_CACHE_ENABLED": "false"}):
            download_calls = 0
            fs_env._info = mock_info2
            fs_env._get_file_direct = mock_get_file_direct
            await fs_env._get_file(new_rpath, str(dest_env_off))
            assert download_calls == 1  # bypassed cache
            assert dest_env_off.exists()
        print("Test 6 (GCSFS_CACHE_ENABLED=false bypass): PASSED", flush=True)

        dest_env_on = Path(tmpdir) / "env_on" / "model.ckpt"
        with mock.patch.dict(os.environ, {"GCSFS_CACHE_ENABLED": "true"}):
            download_calls = 0
            fs_env._info = mock_info2
            fs_env._get_file_direct = mock_get_file_direct
            await fs_env._get_file(new_rpath, str(dest_env_on))
            assert download_calls == 0  # cache hit from master cache!
            assert dest_env_on.exists()
        print("Test 7 (GCSFS_CACHE_ENABLED=true hit): PASSED", flush=True)


if __name__ == "__main__":
    from unittest import mock

    asyncio.run(run_integration_tests())
    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!", flush=True)
