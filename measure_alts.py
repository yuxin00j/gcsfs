import asyncio
import time
import fsspec

async def main():
    fs = fsspec.filesystem('gcs', project="gcs-aiml-clients-testing-101")
    # trigger client creation
    client = fs._client
    channel = client.grpc_client.transport.grpc_channel
    
    t0 = time.time()
    await channel.channel_ready()
    t1 = time.time()
    print(f"ALTS handshake took: {(t1-t0)*1000:.2f} ms")

asyncio.run(main())
