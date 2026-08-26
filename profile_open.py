import cProfile
import pstats
import io
import asyncio
import statistics
import gcsfs
import time
from concurrent.futures import ThreadPoolExecutor

def run_open(num_ranks=8, num_workers=4, num_threads=10):
    fs = gcsfs.GCSFileSystem()
    clean_bucket = "yuxinj-us-central1-b-test"
    try:
        all_files = fs.ls(clean_bucket)
        file_paths = [f"gs://{p}" for p in all_files if not p.endswith("/")]
    except Exception:
        file_paths = []
    if not file_paths:
        file_paths = [f"gs://{clean_bucket}/dummy_files/dummy_{i}.bin" for i in range(4096)]

    def open_one(rank_id, worker_id, thread_id):
        # Workers within a rank all open different files
        # Different ranks open the same set of files
        file_idx = (worker_id * num_threads + thread_id) % len(file_paths)
        file_path = file_paths[file_idx]
        t0 = time.perf_counter()
        with fs.open(file_path, "rb") as f:
            pass
        return time.perf_counter() - t0

    tasks = [
        (r, w, t)
        for r in range(num_ranks)
        for w in range(num_workers)
        for t in range(num_threads)
    ]

    total_threads = num_ranks * num_workers * num_threads
    with ThreadPoolExecutor(max_workers=min(total_threads, 128)) as pool:
        futures = [pool.submit(open_one, r, w, t) for r, w, t in tasks]
        latencies = [f.result() for f in futures]
    return latencies

pr = cProfile.Profile()
pr.enable()

latencies = run_open(num_ranks=8, num_workers=4, num_threads=10)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(35)
print(f"Total Requests: {len(latencies)}, Median: {statistics.median(latencies)*1000:.1f}ms, Max: {max(latencies)*1000:.1f}ms")
print("\n=== cProfile Top Cumulative Time ===")
print(s.getvalue())


