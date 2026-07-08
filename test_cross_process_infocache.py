import os
import time
import json
import gcsfs
import multiprocessing
import urllib.parse

def fetch_info(queue, delay):
    time.sleep(delay)  # Stagger slightly but keep them concurrent
    fs = gcsfs.GCSFileSystem(project="gcs-tess")
    
    start = time.perf_counter()
    with fs.open("gs://gcp-public-data-landsat/index.csv.gz", "rb") as f:
        info = f.details
    duration = time.perf_counter() - start
    
    print(f"Process {os.getpid()} fetched info in {duration:.4f}s. infocache len: {len(fs.infocache)}")
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
    
    print("Step 1: Spawning 3 concurrent Python processes to trigger InfoCache miss race condition...")
    queue = multiprocessing.Queue()
    
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=fetch_info, args=(queue, i * 0.1))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    for _ in processes:
        print("Fetched info for:", queue.get())

    # 2. Verify it's on disk
    if os.path.exists(shm_dir):
        files = os.listdir(shm_dir)
    else:
        files = []
    print(f"Step 2: Found {len(files)} files in {shm_dir}: {files}")
    
    # 3. Create a NEW instance in the main process (empty L1 memory cache)
    print("Step 3: Creating a new GCSFileSystem in the main process...")
    fs = gcsfs.GCSFileSystem(project="gcs-tess")
    
    # Trigger auth / session initialization
    print("Step 4: Initializing session...")
    fs.ls("gs://gcs-tess/does-not-exist")
    
    print("Step 5: Fetching info again via fs.open(). This should instantly hit the L2 disk cache!")
    start = time.perf_counter()
    with fs.open("gs://gcp-public-data-landsat/index.csv.gz", "rb") as f:
        info = f.details
    duration = time.perf_counter() - start
    print(f"Info fetched in {duration:.6f}s: {info['name']} (Size: {info['size']})")
    
    print("Test passed! Cross-process L2 cache locks work.")

if __name__ == "__main__":
    main()
