import asyncio
import time
from google.cloud.storage.asyncio.client import Client

async def main():
    t0 = time.time()
    client = Client(project="gcs-aiml-clients-testing-101")
    channel = client._transport.grpc_channel
    
    t1 = time.time()
    await channel.channel_ready()
    t2 = time.time()
    print(f"Client init took: {(t1-t0)*1000:.2f} ms")
    print(f"ALTS handshake took: {(t2-t1)*1000:.2f} ms")

asyncio.run(main())
