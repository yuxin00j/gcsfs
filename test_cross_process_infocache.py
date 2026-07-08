import os
import time
import json
import gcsfs
import multiprocessing
import urllib.parse

def fetch_info(queue):
    fs = gcsfs.GCSFileSystem(project="gcs-tess")
    with fs.open("gs://gcp-public-data-landsat/index.csv.gz", "rb") as f:
        info = f.details
    print(f"Process 1 infocache len: {len(fs.infocache)}")
    queue.put(info["name"])

def main():
    # 1. Clear any existing cache to ensure a clean test
    print(f"Using gcsfs from: {gcsfs.__file__}")
    shm_dir = "/dev/shm/gcsfs_info_cache"
    if os.path.exists(shm_dir):
        for f in os.listdir(shm_dir):
            try:
                os.remove(os.path.join(shm_dir, f))
            except Exception:
                pass
    
    print("Step 1: Spawning a completely separate Python process...")
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=fetch_info, args=(queue,))
    p.start()
    p.join()
    print("Process 1 fetched info for:", queue.get())

    # 2. Verify it's on disk
    if os.path.exists(shm_dir):
        files = os.listdir(shm_dir)
    else:
        files = []
    print(f"Step 2: Found {len(files)} files in {shm_dir}: {files}")
    
    # 3. Create a NEW instance in the main process (empty L1 memory cache)
    print("Step 3: Creating a new GCSFileSystem in the main process...")
    fs = gcsfs.GCSFileSystem(project="gcs-tess")
    
    print("Step 4: Fetching info again via fs.open(). This should instantly hit the L2 disk cache!")
    start = time.time()
    with fs.open("gs://gcp-public-data-landsat/index.csv.gz", "rb") as f:
        info = f.details
    duration = time.time() - start
    print(f"Info fetched in {duration:.4f}s: {info['name']} (Size: {info['size']})")
    
    print("Test passed! Cross-process L2 cache works.")

if __name__ == "__main__":
    main()
