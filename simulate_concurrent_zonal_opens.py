#!/usr/bin/env python3
"""
Standalone simulation script to benchmark concurrent file opens on GCS zonal buckets.

Simulates N processes, each spawning M threads to concurrently open files in
a specified GCS bucket/prefix (e.g. gs://yuxinj-us-central1-b-test/dummy_files).
"""

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class SimulationConfig:
    bucket: str = "yuxinj-us-central1-b-test"
    prefix: str = "dummy_files"
    num_processes: int = 32
    num_threads_per_process: int = 10
    num_opens_per_thread: int = 1
    read_bytes: int = 0
    file_strategy: str = "random"  # 'random', 'shared_subset', 'round_robin'
    shared_files_count: int = 50
    output_json: Optional[str] = None


def thread_worker(
    thread_id: int,
    process_id: int,
    config: SimulationConfig,
    file_list: List[str],
    thread_ready_barrier: threading.Barrier,
    process_start_event: threading.Event,
    results: List[Dict[str, Any]],
):
    import gcsfs

    fs = gcsfs.GCSFileSystem()
    latencies: List[float] = []
    errors: List[str] = []

    # Signal ready and wait for coordinated start
    thread_ready_barrier.wait()
    process_start_event.wait()

    t_start = time.perf_counter()

    for i in range(config.num_opens_per_thread):
        if config.file_strategy == "random":
            fpath = random.choice(file_list)
        elif config.file_strategy == "shared_subset":
            subset = file_list[: min(config.shared_files_count, len(file_list))]
            fpath = random.choice(subset)
        elif config.file_strategy == "round_robin":
            idx = (thread_id * config.num_opens_per_thread + i) % len(file_list)
            fpath = file_list[idx]
        else:
            fpath = random.choice(file_list)

        if fpath.startswith("gs://"):
            full_path = fpath
        elif fpath.startswith(f"{config.bucket}/"):
            full_path = f"gs://{fpath}"
        else:
            full_path = f"gs://{config.bucket}/{fpath}"

        t0 = time.perf_counter()
        try:
            with fs.open(full_path, "rb") as f:
                if config.read_bytes > 0:
                    f.read(config.read_bytes)
                elif config.read_bytes == -1:
                    f.read()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
        except Exception as e:
            t1 = time.perf_counter()
            errors.append(f"{type(e).__name__}: {e}")

    t_end = time.perf_counter()
    results.append(
        {
            "thread_id": thread_id,
            "latencies": latencies,
            "errors": errors,
            "start_time": t_start,
            "end_time": t_end,
            "duration": t_end - t_start,
        }
    )


def process_worker(
    process_id: int,
    config: SimulationConfig,
    file_list: List[str],
    proc_ready_event: Any,
    global_start_event: Any,
    result_queue: mp.Queue,
):
    thread_results: List[Dict[str, Any]] = []
    threads = []
    num_threads = config.num_threads_per_process

    # Internal thread synchronization
    thread_ready_barrier = threading.Barrier(num_threads + 1)
    local_start_event = threading.Event()

    for tid in range(num_threads):
        t = threading.Thread(
            target=thread_worker,
            args=(
                tid,
                process_id,
                config,
                file_list,
                thread_ready_barrier,
                local_start_event,
                thread_results,
            ),
        )
        threads.append(t)
        t.start()

    # Wait until all threads in this process are spawned and ready
    thread_ready_barrier.wait()

    # Signal to main process that this worker process is fully initialized
    proc_ready_event.set()

    # Wait for the synchronized go-signal across all processes
    global_start_event.wait()
    local_start_event.set()

    for t in threads:
        t.join()

    # Aggregate process-level results
    all_latencies = []
    all_errors = []
    earliest_start = float("inf")
    latest_end = 0.0

    for tr in thread_results:
        all_latencies.extend(tr["latencies"])
        all_errors.extend(tr["errors"])
        if tr["start_time"] < earliest_start:
            earliest_start = tr["start_time"]
        if tr["end_time"] > latest_end:
            latest_end = tr["end_time"]

    process_summary = {
        "process_id": process_id,
        "latencies": all_latencies,
        "errors": all_errors,
        "num_success": len(all_latencies),
        "num_errors": len(all_errors),
        "start_time": earliest_start,
        "end_time": latest_end,
        "duration": latest_end - earliest_start if latest_end > earliest_start else 0.0,
    }

    result_queue.put(process_summary)


def fetch_file_list(bucket: str, prefix: str) -> List[str]:
    import gcsfs

    print(f"Listing files in gs://{bucket}/{prefix}...")
    fs = gcsfs.GCSFileSystem()
    target_path = f"{bucket}/{prefix}".rstrip("/")
    items = fs.ls(target_path, detail=True)
    raw_files = [
        item["name"]
        for item in items
        if item.get("type") == "file" or item.get("storageClass") is not None
    ]
    if not raw_files:
        # Fallback if detail=False or structure differs
        raw_files = [f for f in fs.ls(target_path) if not f.endswith("/")]
    files = [
        f if f.startswith("gs://") else (f"gs://{f}" if f.startswith(f"{bucket}/") else f"gs://{bucket}/{f}")
        for f in raw_files
    ]
    print(f"Found {len(files)} files in gs://{bucket}/{prefix}.")
    if not files:
        raise RuntimeError(f"No files found in gs://{bucket}/{prefix}!")
    return files


def print_report(all_results: List[Dict[str, Any]], config: SimulationConfig, total_wall_time: float):
    all_latencies = []
    all_errors = []
    total_successful_opens = 0
    total_failed_opens = 0

    for res in all_results:
        all_latencies.extend(res["latencies"])
        all_errors.extend(res["errors"])
        total_successful_opens += res["num_success"]
        total_failed_opens += res["num_errors"]

    total_ops = total_successful_opens + total_failed_opens
    throughput = total_successful_opens / total_wall_time if total_wall_time > 0 else 0.0

    print("\n" + "=" * 70)
    print(" CONCURRENT ZONAL FILE OPEN BENCHMARK REPORT ")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Bucket/Prefix:              gs://{config.bucket}/{config.prefix}")
    print(f"  Processes:                  {config.num_processes}")
    print(f"  Threads per process:        {config.num_threads_per_process}")
    print(f"  Total concurrent threads:   {config.num_processes * config.num_threads_per_process}")
    print(f"  Opens per thread:           {config.num_opens_per_thread}")
    print(f"  File selection strategy:    {config.file_strategy}")
    print(f"  Read payload after open:    {config.read_bytes} bytes")
    print("-" * 70)
    print(f"Summary Results:")
    print(f"  Total Opens Attempted:      {total_ops}")
    print(f"  Successful Opens:           {total_successful_opens}")
    print(f"  Failed Opens:               {total_failed_opens} ({(total_failed_opens/total_ops*100.0 if total_ops else 0):.2f}%)")
    print(f"  Total Wall Clock Time:      {total_wall_time:.3f} s")
    print(f"  Aggregate Open Rate:        {throughput:.2f} opens/sec")
    print("-" * 70)

    if all_latencies:
        lat_arr = np.array(all_latencies)
        p50 = np.percentile(lat_arr, 50)
        p75 = np.percentile(lat_arr, 75)
        p90 = np.percentile(lat_arr, 90)
        p95 = np.percentile(lat_arr, 95)
        p99 = np.percentile(lat_arr, 99)
        p99_9 = np.percentile(lat_arr, 99.9)

        print(f"Open Latency Distribution (ms):")
        print(f"  Min:      {np.min(lat_arr):8.2f} ms")
        print(f"  Mean:     {np.mean(lat_arr):8.2f} ms")
        print(f"  Median:   {p50:8.2f} ms")
        print(f"  P75:      {p75:8.2f} ms")
        print(f"  P90:      {p90:8.2f} ms")
        print(f"  P95:      {p95:8.2f} ms")
        print(f"  P99:      {p99:8.2f} ms")
        print(f"  P99.9:    {p99_9:8.2f} ms")
        print(f"  Max:      {np.max(lat_arr):8.2f} ms")
        print(f"  StdDev:   {np.std(lat_arr):8.2f} ms")
        print("-" * 70)

        # Histogram buckets
        bins = [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, float("inf")]
        bin_labels = [
            "< 5ms", "5-10ms", "10-20ms", "20-50ms", "50-100ms",
            "100-200ms", "200-500ms", "500ms-1s", "1s-2s", "2s-5s", "> 5s"
        ]
        counts, _ = np.histogram(lat_arr, bins=bins)
        print("Latency Histogram:")
        for label, count in zip(bin_labels, counts):
            pct = (count / len(lat_arr)) * 100.0
            bar = "#" * int(pct / 2)
            print(f"  {label:10s} : {count:6d} ({pct:5.1f}%) | {bar}")

    if all_errors:
        print("-" * 70)
        print("Errors Encountered (Top 10):")
        from collections import Counter
        err_counts = Counter(all_errors).most_common(10)
        for err, cnt in err_counts:
            print(f"  [{cnt}x] {err}")

    print("=" * 70 + "\n")

    if config.output_json:
        report_data = {
            "config": dataclasses.asdict(config),
            "summary": {
                "total_ops": total_ops,
                "successful_opens": total_successful_opens,
                "failed_opens": total_failed_opens,
                "wall_time_sec": total_wall_time,
                "opens_per_sec": throughput,
                "latency_min_ms": float(np.min(lat_arr)) if all_latencies else None,
                "latency_mean_ms": float(np.mean(lat_arr)) if all_latencies else None,
                "latency_p50_ms": float(p50) if all_latencies else None,
                "latency_p90_ms": float(p90) if all_latencies else None,
                "latency_p95_ms": float(p95) if all_latencies else None,
                "latency_p99_ms": float(p99) if all_latencies else None,
                "latency_max_ms": float(np.max(lat_arr)) if all_latencies else None,
                "latency_std_ms": float(np.std(lat_arr)) if all_latencies else None,
            },
            "errors": all_errors,
        }
        with open(config.output_json, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"JSON metrics written to: {config.output_json}")


def run_benchmark(config: SimulationConfig):
    file_list = fetch_file_list(config.bucket, config.prefix)

    ctx = mp.get_context("spawn")
    ready_events = [ctx.Event() for _ in range(config.num_processes)]
    global_start_event = ctx.Event()
    result_queue = ctx.Queue()

    print(
        f"\nSpawning {config.num_processes} processes (each with {config.num_threads_per_process} threads)..."
    )
    processes = []
    for pid in range(config.num_processes):
        p = ctx.Process(
            target=process_worker,
            args=(
                pid,
                config,
                file_list,
                ready_events[pid],
                global_start_event,
                result_queue,
            ),
        )
        processes.append(p)
        p.start()

    print("Waiting for all processes and threads to be ready...")
    for pid, ev in enumerate(ready_events):
        ev.wait()
    print("All processes and threads initialized. Releasing global start barrier NOW!")

    t_wall_start = time.perf_counter()
    global_start_event.set()

    # Collect results from all processes
    all_results = []
    for _ in range(config.num_processes):
        res = result_queue.get()
        all_results.append(res)

    for p in processes:
        p.join()

    t_wall_end = time.perf_counter()
    total_wall_time = t_wall_end - t_wall_start

    print_report(all_results, config, total_wall_time)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate concurrent processes and threads opening zonal GCS files."
    )
    parser.add_argument(
        "--bucket",
        default="yuxinj-us-central1-b-test",
        help="GCS bucket name (default: yuxinj-us-central1-b-test)",
    )
    parser.add_argument(
        "--prefix",
        default="dummy_files",
        help="Prefix/directory inside bucket (default: dummy_files)",
    )
    parser.add_argument(
        "-p",
        "--num-processes",
        type=int,
        default=32,
        help="Number of concurrent processes (default: 32)",
    )
    parser.add_argument(
        "-t",
        "--num-threads-per-process",
        type=int,
        default=10,
        help="Number of threads per process (default: 10)",
    )
    parser.add_argument(
        "-k",
        "--num-opens-per-thread",
        type=int,
        default=1,
        help="Number of file opens per thread (default: 1)",
    )
    parser.add_argument(
        "-b",
        "--read-bytes",
        type=int,
        default=0,
        help="Bytes to read after opening file. 0 for open only, -1 for full file (default: 0)",
    )
    parser.add_argument(
        "--file-strategy",
        choices=["random", "shared_subset", "round_robin"],
        default="random",
        help="File selection strategy (default: random)",
    )
    parser.add_argument(
        "--shared-files-count",
        type=int,
        default=50,
        help="Subset size when using shared_subset strategy (default: 50)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to output JSON results file",
    )

    args = parser.parse_args()

    config = SimulationConfig(
        bucket=args.bucket,
        prefix=args.prefix,
        num_processes=args.num_processes,
        num_threads_per_process=args.num_threads_per_process,
        num_opens_per_thread=args.num_opens_per_thread,
        read_bytes=args.read_bytes,
        file_strategy=args.file_strategy,
        shared_files_count=args.shared_files_count,
        output_json=args.output_json,
    )

    run_benchmark(config)


if __name__ == "__main__":
    main()
