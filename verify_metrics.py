import logging
import sys
import os

# Ensure local gcsfs is imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gcsfs
from gcsfs import GCSFileSystem

def verify_metrics():
    # Configure logging to show metrics
    logger = logging.getLogger('gcsfs')
    logger.setLevel(logging.INFO)
    
    print("Starting verification...")
    
    fs = GCSFileSystem()
    
    print("\nTesting GCSFileSystem._ls (async via sync interface)")
    try:
        # ls calls _ls
        fs.ls('yuxinj-test-regional')
    except Exception as e:
        print(f"Method finished: {type(e).__name__}")

    print("\nTesting GCSFileSystem._info (async via sync interface)")
    try:
        fs.info('yuxinj-test-regional/test-object.txt')
    except Exception as e:
        print(f"Method finished: {type(e).__name__}")

    print("\nTesting GCSFileSystem._open (sync)")
    try:
        # Call _open directly to test its instrumentation
        f = fs._open('yuxinj-test-regional/test-object.txt', mode='rb')
        print(f"File opened: {f}")
        # Test read
        data = f.read(10)
        print(f"Read 10 bytes: {data}")
    except Exception as e:
         print(f"Method finished: {type(e).__name__}")

if __name__ == "__main__":
    verify_metrics()
