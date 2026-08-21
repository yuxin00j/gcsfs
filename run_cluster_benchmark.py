#!/usr/bin/env python3
"""
run_cluster_benchmark.py

Orchestrates a distributed GKE cluster benchmark across all nodes.
On each node, it starts N processes, each process spawns M threads,
and each thread opens a distinct file from the dataset via gcsfs.

Features:
- Multi-Wave Benchmark Support:
    * Wave 1: Cold start synchronized opens across all processes/threads
    * Sleep 5 seconds (using same processes & fs instances)
    * Wave 2: Warm steady-state synchronized opens across all processes/threads
- Global Distributed Barrier across all nodes:
    * All pods register readiness to GCS barrier prefix.
    * Once all pods are ready, Leader calculates synchronized target epoch timestamp.
    * All processes across all physical nodes precision spin-wait and fire simultaneously.
- Metrics & Analysis:
    * Separate percentile metrics for Wave 1, Wave 2, and Overall.
    * Per-node and full-cluster aggregate summaries.
- Supports --local-pkg: Packages local gcsfs source into ConfigMap directly for sub-second iteration without git push.
- Supports --scalar: Prints cluster Max latency in seconds as the final stdout line for metric scrapers.
"""

import os
import sys
import time
import json
import uuid
import base64
import io
import tarfile
import argparse
import subprocess
import statistics

WORKER_SCRIPT = r'''#!/usr/bin/env python3
import os
import sys
import time
import json
import resource
import statistics
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def thread_task(fs, file_path):
    t0 = time.perf_counter()
    try:
        with fs.open(file_path, "rb") as f:
            pass
        latency_ms = (time.perf_counter() - t0) * 1000
        return True, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return False, latency_ms

def process_worker(args):
    node_index, proc_id, num_procs, num_threads, file_paths, target_time, sleep_between_waves = args
    
    # Precision spin-wait to synchronize all processes across all nodes to exact millisecond
    if target_time > 0:
        sleep_dur = target_time - time.time()
        if sleep_dur > 0.05:
            time.sleep(sleep_dur - 0.02)
        while time.time() < target_time:
            pass
            
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    
    wave1_results = []
    wave2_results = []
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # --- WAVE 1: Cold start opens ---
        futures1 = []
        for t in range(num_threads):
            global_worker_id = (node_index * num_procs + proc_id) * num_threads + t
            file_path = file_paths[global_worker_id % len(file_paths)]
            futures1.append(executor.submit(thread_task, fs, file_path))
            
        for f in futures1:
            success, lat = f.result()
            wave1_results.append((success, lat))
            
        # Sleep between waves in the same process
        if sleep_between_waves > 0:
            time.sleep(sleep_between_waves)
            
        # --- WAVE 2: Warm channel opens ---
        futures2 = []
        total_cluster_workers = int(os.environ.get("TOTAL_NODES", "8")) * num_procs * num_threads
        for t in range(num_threads):
            # Select distinct files for wave 2
            global_worker_id = (node_index * num_procs + proc_id) * num_threads + t + total_cluster_workers
            file_path = file_paths[global_worker_id % len(file_paths)]
            futures2.append(executor.submit(thread_task, fs, file_path))
            
        for f in futures2:
            success, lat = f.result()
            wave2_results.append((success, lat))
            
    return {"wave1": wave1_results, "wave2": wave2_results}

def calc_stats(results_list):
    success_lats = [lat for s, lat in results_list if s]
    fail_lats = [lat for s, lat in results_list if not s]
    success_lats.sort()
    
    p50 = statistics.median(success_lats) if success_lats else 0
    p90 = success_lats[int(len(success_lats) * 0.90)] if success_lats else 0
    p95 = success_lats[int(len(success_lats) * 0.95)] if success_lats else 0
    p99 = success_lats[int(len(success_lats) * 0.99)] if success_lats else 0
    max_lat = max(success_lats) if success_lats else 0
    min_lat = min(success_lats) if success_lats else 0
    
    return {
        "total": len(results_list),
        "success": len(success_lats),
        "failure": len(fail_lats),
        "min_ms": min_lat,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max_lat,
        "latencies_ms": success_lats
    }

def main():
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
    except Exception:
        pass

    node_index = int(os.environ.get("JOB_COMPLETION_INDEX", os.environ.get("NODE_RANK", "0")))
    node_name = os.environ.get("NODE_NAME", "unknown-node")
    pod_name = os.environ.get("POD_NAME", "unknown-pod")
    num_nodes = int(os.environ.get("TOTAL_NODES", "8"))
    num_procs = int(os.environ.get("NUM_PROCESSES", "128"))
    num_threads = int(os.environ.get("NUM_THREADS", "1"))
    sleep_between_waves = float(os.environ.get("SLEEP_BETWEEN_WAVES", "5.0"))
    bucket = os.environ.get("BUCKET", "gs://hf-pile-deduplicated-us-central1-b-gcsfs")
    barrier_dir = os.environ.get("BARRIER_DIR", "gs://yuxinj-us-central1-b-test/benchmark_barriers/default")

    print(f"=== Starting Worker on Node {node_index} ({node_name}) ===", flush=True)
    print(f"Processes: {num_procs}, Threads per Process: {num_threads}, Total opens per wave: {num_procs * num_threads}", flush=True)
    print(f"Sleep between waves: {sleep_between_waves}s", flush=True)
    print(f"Target Bucket: {bucket}", flush=True)

    import gcsfs
    fs_init = gcsfs.GCSFileSystem()
    clean_bucket = bucket.replace("gs://", "")
    all_files = fs_init.ls(clean_bucket)
    file_paths = [f"gs://{p}" for p in all_files if not p.endswith("/")]
    if not file_paths:
        file_paths = [f"{bucket}/dummy_files/dummy_{i}.bin" for i in range(4096)]

    print(f"Discovered {len(file_paths)} files in dataset.", flush=True)

    # ----------------------------------------------------
    # Distributed Synchronization Barrier Across All Pods
    # ----------------------------------------------------
    ready_file = f"{barrier_dir}/ready_{node_index}.txt"
    start_file = f"{barrier_dir}/start_time.json"

    with fs_init.open(ready_file, "w") as f:
        f.write(f"ready {node_index} {time.time()}")
    print(f"Node {node_index} registered readiness. Waiting for all {num_nodes} nodes to reach barrier...", flush=True)

    target_time = 0
    t_barrier_wait = time.time()
    clean_barrier_dir = barrier_dir.replace("gs://", "")

    if node_index == 0:
        # Leader node: wait for all ready files then write start_time.json
        while time.time() - t_barrier_wait < 180:
            try:
                fs_init.invalidate_cache(clean_barrier_dir)
                ready_files = [p for p in fs_init.ls(clean_barrier_dir, refresh=True) if "ready_" in p]
                if len(ready_files) >= num_nodes:
                    target_time = time.time() + 5.0  # Fire 5.0 seconds in the future
                    with fs_init.open(start_file, "w") as f:
                        f.write(json.dumps({"target_time": target_time}))
                    print(f"All {num_nodes} nodes ready! Set synchronized start time to: {target_time:.3f}", flush=True)
                    break
            except Exception as e:
                pass
            time.sleep(0.5)
    else:
        # Follower nodes: wait for start_time.json from leader
        while time.time() - t_barrier_wait < 180:
            try:
                fs_init.invalidate_cache(clean_barrier_dir)
                if fs_init.exists(start_file):
                    content = fs_init.cat(start_file).decode("utf-8")
                    data = json.loads(content)
                    target_time = data["target_time"]
                    print(f"Received synchronized start time: {target_time:.3f}", flush=True)
                    break
            except Exception as e:
                pass
            time.sleep(0.5)

    if target_time == 0:
        print("Warning: Barrier timed out! Proceeding without synchronization.", flush=True)
    else:
        remaining = target_time - time.time()
        print(f"Synchronized barrier active. All {num_nodes} nodes will fire Wave 1 in {remaining:.2f} seconds.", flush=True)

    tasks = [
        (node_index, p, num_procs, num_threads, file_paths, target_time, sleep_between_waves)
        for p in range(num_procs)
    ]

    ctx = mp.get_context("forkserver")
    t_start = time.time()

    with ctx.Pool(num_procs) as pool:
        worker_outputs = pool.map(process_worker, tasks)

    t_end = time.time()
    wall_time = t_end - (target_time if target_time > 0 and target_time < t_end else t_start)

    all_w1 = []
    all_w2 = []
    for out in worker_outputs:
        all_w1.extend(out["wave1"])
        all_w2.extend(out["wave2"])

    stats_w1 = calc_stats(all_w1)
    stats_w2 = calc_stats(all_w2)
    stats_all = calc_stats(all_w1 + all_w2)

    print(f"\n=== Node {node_index} Completed in {wall_time:.2f}s ===", flush=True)
    print(f"Wave 1 (Cold) -> P50: {stats_w1['p50_ms']:.1f}ms | P95: {stats_w1['p95_ms']:.1f}ms | Max: {stats_w1['max_ms']:.1f}ms (Success: {stats_w1['success']}/{stats_w1['total']})", flush=True)
    print(f"Wave 2 (Warm) -> P50: {stats_w2['p50_ms']:.1f}ms | P95: {stats_w2['p95_ms']:.1f}ms | Max: {stats_w2['max_ms']:.1f}ms (Success: {stats_w2['success']}/{stats_w2['total']})", flush=True)

    report = {
        "node_index": node_index,
        "node_name": node_name,
        "pod_name": pod_name,
        "processes": num_procs,
        "threads": num_threads,
        "wall_time": wall_time,
        "wave1": stats_w1,
        "wave2": stats_w2,
        "overall": stats_all
    }

    print("\n__JSON_REPORT_BEGIN__", flush=True)
    print(json.dumps(report), flush=True)
    print("__JSON_REPORT_END__", flush=True)

if __name__ == "__main__":
    main()
'''

def run_cmd(cmd, check=True):
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed (code {res.returncode}): {cmd}\nStderr: {res.stderr}\nStdout: {res.stdout}")
    return res.stdout.strip(), res.stderr.strip(), res.returncode

def package_local_gcsfs():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    gcsfs_src_dir = os.path.join(repo_dir, "gcsfs")
    if not os.path.isdir(gcsfs_src_dir):
        raise RuntimeError(f"gcsfs package directory not found at {gcsfs_src_dir}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(gcsfs_src_dir):
            if "tests" in dirs:
                dirs.remove("tests")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
            for f in files:
                if f.endswith(".pyc") or f.endswith(".so"):
                    continue
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, repo_dir)
                tar.add(full_p, arcname=rel_p)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def print_table(title, lats, num_nodes, processes, threads, total_opens, success_count, fail_count, max_wall_time):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)
    print(f"Total Physical Nodes:   {num_nodes}")
    print(f"Processes per Node (n): {processes}")
    print(f"Threads per Proc (m):   {threads}")
    print(f"Total Requests:         {total_opens}")
    print(f"Successful Requests:    {success_count} ({success_count/max(1, total_opens)*100:.1f}%)")
    print(f"Failed Requests:        {fail_count}")
    if max_wall_time > 0:
        print(f"Max Node Wall Time:     {max_wall_time:.2f} seconds")
    print()

    if lats:
        lats.sort()
        print("--- Latency Percentiles ---")
        print(f"Min:  {min(lats):.2f} ms")
        print(f"P50:  {statistics.median(lats):.2f} ms")
        print(f"P90:  {lats[int(len(lats) * 0.90)]:.2f} ms")
        print(f"P95:  {lats[int(len(lats) * 0.95)]:.2f} ms")
        print(f"P99:  {lats[int(len(lats) * 0.99)]:.2f} ms")
        print(f"Max:  {max(lats):.2f} ms")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Wave GKE Cluster Open Latency Benchmark with Global Synchronization Barrier")
    parser.add_argument("--cluster", type=str, default="yuxinj-8node-n4-cluster", help="GKE cluster name")
    parser.add_argument("--zone", type=str, default="us-central1-b", help="GKE cluster zone")
    parser.add_argument("--project", type=str, default="gcs-aiml-clients-testing-101", help="GCP project ID")
    parser.add_argument("--processes", "-n", type=int, default=32, help="Number of processes per node (default: 32)")
    parser.add_argument("--threads", "-m", type=int, default=10, help="Number of threads per process (default: 10)")
    parser.add_argument("--sleep-waves", type=float, default=5.0, help="Sleep in seconds between Wave 1 and Wave 2 (default: 5.0)")
    parser.add_argument("--bucket", type=str, default="gs://hf-pile-deduplicated-us-central1-b-gcsfs", help="Target GCS bucket")
    parser.add_argument("--barrier-bucket", type=str, default="gs://yuxinj-us-central1-b-test", help="Bucket for distributed barrier coordination")
    parser.add_argument("--branch", type=str, default="main", help="GCSFS git branch/tag to benchmark (default: main)")
    parser.add_argument("--local-pkg", action="store_true", help="Package local gcsfs repository files directly into ConfigMap")
    parser.add_argument("--scalar", action="store_true", help="Print only cluster max latency in seconds on last line")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    args = parser.parse_args()

    if not args.scalar:
        print("=" * 75)
        print(" GKE MULTI-WAVE DISTRIBUTED CLUSTER BENCHMARK (SYNCHRONIZED GLOBAL BARRIER)")
        print(f" Cluster:                {args.cluster} ({args.zone})")
        print(f" Project:                {args.project}")
        print(f" Processes per Node (n): {args.processes}")
        print(f" Threads per Proc (m):   {args.threads}")
        print(f" Sleep Between Waves:    {args.sleep_waves}s")
        print(f" Target Bucket:          {args.bucket}")
        print(f" Package Mode:           {'Local working copy' if args.local_pkg else f'Remote git branch: {args.branch}'}")
        print("=" * 75)

    # 1. Connect to GKE Cluster
    if not args.scalar:
        print(f"\n[1/5] Getting credentials for cluster {args.cluster}...")
    run_cmd(f"gcloud container clusters get-credentials {args.cluster} --zone={args.zone} --project={args.project}")

    # 2. Get Node Count
    out, _, _ = run_cmd("kubectl get nodes -o jsonpath='{.items[*].metadata.name}'")
    nodes = out.split()
    num_nodes = len(nodes)
    if not args.scalar:
        print(f"Detected {num_nodes} active physical nodes in cluster: {', '.join(nodes)}")
    if num_nodes == 0:
        print("Error: No nodes found in cluster!")
        sys.exit(1)

    total_cluster_procs = num_nodes * args.processes
    total_opens_per_wave = total_cluster_procs * args.threads
    total_all_opens = total_opens_per_wave * 2
    if not args.scalar:
        print(f"Cluster Workload: {num_nodes} nodes × {args.processes} procs × {args.threads} threads = {total_opens_per_wave} opens per wave ({total_all_opens} total across 2 waves)")

    job_id = f"gcsfs-bench-{uuid.uuid4().hex[:6]}"
    configmap_name = f"{job_id}-cm"
    barrier_dir = f"{args.barrier_bucket}/benchmark_barriers/{job_id}"

    # 3. Create ConfigMap with Worker Script
    if not args.scalar:
        print(f"\n[2/5] Creating ConfigMap {configmap_name} with benchmark worker script...")
    cm_data = {"worker.py": WORKER_SCRIPT}
    if args.local_pkg:
        cm_data["gcsfs_pkg.b64"] = package_local_gcsfs()

    cm_manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": configmap_name},
        "data": cm_data
    }
    cm_json = json.dumps(cm_manifest)
    run_cmd(f"cat << 'EOF' | kubectl apply -f -\n{cm_json}\nEOF")

    # 4. Container launch commands
    if args.local_pkg:
        container_cmd = [
            "apt-get update -qq && apt-get install -y -qq procps tar > /dev/null 2>&1 && "
            "pip install --no-cache-dir google-cloud-storage google-cloud-storage-control fsspec aiohttp requests decorator google-auth google-auth-oauthlib grpcio protobuf > /dev/null 2>&1 && "
            "mkdir -p /tmp/src && "
            "base64 -d /scripts/gcsfs_pkg.b64 | tar -xzf - -C /tmp/src && "
            "export PYTHONPATH=/tmp/src:$PYTHONPATH && "
            "python3 -u /scripts/worker.py"
        ]
    else:
        container_cmd = [
            f"apt-get update -qq && apt-get install -y -qq git procps > /dev/null 2>&1 && "
            f"pip install --no-cache-dir git+https://github.com/fsspec/filesystem_spec.git git+https://github.com/yuxin00j/gcsfs.git@{args.branch} > /dev/null 2>&1 && "
            f"python3 -u /scripts/worker.py"
        ]

    # 5. Create Kubernetes Indexed Job to spread 1 Pod per Node
    if not args.scalar:
        print(f"\n[3/5] Launching Kubernetes Indexed Job {job_id} across {num_nodes} nodes...")
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": job_id},
        "spec": {
            "completions": num_nodes,
            "parallelism": num_nodes,
            "completionMode": "Indexed",
            "backoffLimit": 0,
            "template": {
                "metadata": {
                    "labels": {"app": job_id}
                },
                "spec": {
                    "restartPolicy": "Never",
                    "topologySpreadConstraints": [
                        {
                            "maxSkew": 1,
                            "topologyKey": "kubernetes.io/hostname",
                            "whenUnsatisfiable": "DoNotSchedule",
                            "labelSelector": {
                                "matchLabels": {"app": job_id}
                            }
                        }
                    ],
                    "containers": [
                        {
                            "name": "benchmark-worker",
                            "image": "python:3.11-slim",
                            "command": ["/bin/bash", "-c"],
                            "args": container_cmd,
                            "env": [
                                {"name": "TOTAL_NODES", "value": str(num_nodes)},
                                {"name": "NUM_PROCESSES", "value": str(args.processes)},
                                {"name": "NUM_THREADS", "value": str(args.threads)},
                                {"name": "SLEEP_BETWEEN_WAVES", "value": str(args.sleep_waves)},
                                {"name": "BUCKET", "value": args.bucket},
                                {"name": "BARRIER_DIR", "value": barrier_dir},
                                {"name": "GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "value": "true"},
                                {
                                    "name": "NODE_NAME",
                                    "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}
                                },
                                {
                                    "name": "POD_NAME",
                                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}
                                }
                            ],
                            "resources": {
                                "requests": {"cpu": "4", "memory": "8Gi"},
                                "limits": {"cpu": "64", "memory": "128Gi"}
                            },
                            "volumeMounts": [
                                {"name": "script-vol", "mountPath": "/scripts"}
                            ]
                        }
                    ],
                    "volumes": [
                        {
                            "name": "script-vol",
                            "configMap": {"name": configmap_name}
                        }
                    ]
                }
            }
        }
    }
    job_json = json.dumps(job_manifest)
    run_cmd(f"cat << 'EOF' | kubectl apply -f -\n{job_json}\nEOF")

    # 6. Monitor Job Progress
    if not args.scalar:
        print(f"\n[4/5] Waiting for benchmark job {job_id} to finish across all {num_nodes} nodes...")
    t_wait_start = time.time()
    completed = False
    while time.time() - t_wait_start < args.timeout:
        pods_status, _, _ = run_cmd(
            f"kubectl get pods -l app={job_id} -o custom-columns=NAME:.metadata.name,STATUS:.status.phase --no-headers",
            check=False
        )
        lines = [l for l in pods_status.splitlines() if l.strip()]
        completed_count = sum(1 for l in lines if "Succeeded" in l or "Completed" in l)
        failed_count = sum(1 for l in lines if "Failed" in l or "Error" in l)
        running_count = sum(1 for l in lines if "Running" in l or "ContainerCreating" in l or "Pending" in l)

        if not args.scalar:
            print(f"Status ({int(time.time() - t_wait_start)}s): {completed_count}/{num_nodes} completed, {failed_count} failed. Active Pods: {running_count}")

        if completed_count + failed_count >= num_nodes and len(lines) > 0:
            completed = True
            break
            
        time.sleep(5)

    if not completed and not args.scalar:
        print(f"Warning: Timed out waiting for job completion after {args.timeout}s.")

    # 7. Fetch Logs & Collect Results
    if not args.scalar:
        print(f"\n[5/5] Collecting and aggregating multi-wave results from all pods...")
    pods_out, _, _ = run_cmd(f"kubectl get pods -l app={job_id} -o jsonpath='{{.items[*].metadata.name}}'")
    pod_names = pods_out.split()

    node_reports = []
    all_w1_lats = []
    all_w2_lats = []
    all_comb_lats = []

    total_w1_opens = 0
    total_w1_success = 0
    total_w1_fail = 0

    total_w2_opens = 0
    total_w2_success = 0
    total_w2_fail = 0

    for pod in sorted(pod_names):
        pod_log, _, _ = run_cmd(f"kubectl logs {pod} --tail=100", check=False)
        if "__JSON_REPORT_BEGIN__" in pod_log and "__JSON_REPORT_END__" in pod_log:
            try:
                json_str = pod_log.split("__JSON_REPORT_BEGIN__")[1].split("__JSON_REPORT_END__")[0].strip()
                rep = json.loads(json_str)
                node_reports.append(rep)
                
                w1 = rep.get("wave1", {})
                w2 = rep.get("wave2", {})
                
                all_w1_lats.extend(w1.get("latencies_ms", []))
                total_w1_opens += w1.get("total", 0)
                total_w1_success += w1.get("success", 0)
                total_w1_fail += w1.get("failure", 0)
                
                all_w2_lats.extend(w2.get("latencies_ms", []))
                total_w2_opens += w2.get("total", 0)
                total_w2_success += w2.get("success", 0)
                total_w2_fail += w2.get("failure", 0)
                
                all_comb_lats.extend(w1.get("latencies_ms", []))
                all_comb_lats.extend(w2.get("latencies_ms", []))
            except Exception as e:
                if not args.scalar:
                    print(f"Error parsing JSON from {pod}: {e}")
        else:
            if not args.scalar:
                print(f"Pod {pod} output log snippet:\n{pod_log[-500:]}\n")

    # Clean up Kubernetes resources & barrier files
    if not args.scalar:
        print("\nCleaning up benchmark job & configmap...")
    run_cmd(f"kubectl delete job {job_id} --ignore-not-found=true")
    run_cmd(f"kubectl delete configmap {configmap_name} --ignore-not-found=true")
    run_cmd(f"python3 -c \"import gcsfs; fs=gcsfs.GCSFileSystem(); fs.rm('{barrier_dir.replace('gs://', '')}', recursive=True)\" 2>/dev/null || true", check=False)

    if not args.scalar:
        # Display Per-Node Table
        print("\n" + "=" * 90)
        print(" PER-NODE MULTI-WAVE RESULTS")
        print("=" * 90)
        print(f"{'Node':<6} {'Node Name':<28} {'W1 P50':<10} {'W1 Max':<10} {'W2 P50':<10} {'W2 Max':<10} {'Wall Time':<10}")
        print("-" * 90)
        for rep in sorted(node_reports, key=lambda x: x["node_index"]):
            w1 = rep.get("wave1", {})
            w2 = rep.get("wave2", {})
            print(f"{rep['node_index']:<6} {rep['node_name'][:26]:<28} {w1.get('p50_ms', 0):<10.1f} {w1.get('max_ms', 0):<10.1f} {w2.get('p50_ms', 0):<10.1f} {w2.get('max_ms', 0):<10.1f} {rep['wall_time']:<10.2f}s")
        print("=" * 90)

        max_wall_time = max([r["wall_time"] for r in node_reports]) if node_reports else 0

        # Wave 1 Table
        print_table(
            "Wave 1: Cold-Start Synchronized Burst",
            all_w1_lats,
            num_nodes,
            args.processes,
            args.threads,
            total_w1_opens,
            total_w1_success,
            total_w1_fail,
            0
        )

        # Wave 2 Table
        print_table(
            f"Wave 2: Warm-Channel Burst (After {args.sleep_waves}s Sleep)",
            all_w2_lats,
            num_nodes,
            args.processes,
            args.threads,
            total_w2_opens,
            total_w2_success,
            total_w2_fail,
            0
        )

        # Combined Aggregate Table
        print_table(
            "Overall Cluster Aggregate (Both Waves Combined)",
            all_comb_lats,
            num_nodes,
            args.processes,
            args.threads,
            total_w1_opens + total_w2_opens,
            total_w1_success + total_w2_success,
            total_w1_fail + total_w2_fail,
            max_wall_time
        )

    # For autoresearch metric parser
    if all_comb_lats:
        max_sec = max(all_comb_lats) / 1000.0
        print(f"{max_sec:.3f}")
    else:
        print("999.000")

if __name__ == "__main__":
    main()
