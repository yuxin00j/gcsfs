import asyncio
import time
import fsspec

async def main():
    fs = fsspec.filesystem('gcs', project="gcs-aiml-clients-testing-101")
    t0 = time.time()
    await fs._get_client()
    t1 = time.time()
    # Find the grpc channel!
    client = fs._client
    from google.cloud.storage.retry import DEFAULT_RETRY
    from google.cloud.storage.constants import _DEFAULT_UNIVERSE_DOMAIN
    
    # We just know it's in the client somewhere! Or just time create_mrd!
    from google.cloud.storage.asyncio.async_multi_range_downloader import AsyncMultiRangeDownloader
    
    t2 = time.time()
    mrd = await AsyncMultiRangeDownloader.create_mrd(client, "test-bucket", "test-obj")
    t3 = time.time()
    print(f"create_mrd took: {(t3-t2)*1000:.2f} ms")

asyncio.run(main())
