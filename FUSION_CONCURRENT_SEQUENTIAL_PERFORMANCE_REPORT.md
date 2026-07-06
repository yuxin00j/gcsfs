# Fusion Performance Report: High-Throughput Mechanics in Concurrent-Sequential Workloads

## Executive Summary

This report outlines the structural and architectural reasons why **Fusion** (combining the FusionFS FUSE backend and Python `gcsfs` file system) achieves peak performance and near-linear scaling under **Concurrent-Sequential** workloads. 

Unlike **Parallel-Sequential** workloads—where multiple threads read disjoint, non-overlapping regions of a file (triggering state invalidations and seeks on GCS)—**Concurrent-Sequential** workloads involve multiple threads or processes reading the *exact same file offsets simultaneously*. By leveraging stateful gRPC stream preservation, asynchronous single-flighting, and multi-process shared-memory coalescing, Fusion converts potentially massive network and API overhead into ultra-fast local memory hits.

---

## 1. Backend Mechanics: gRPC Stream Continuity and GCS-Side Prefetching

The foundational performance differentiator for concurrent-sequential reads lies in how the backend interacts with the Google Cloud Storage (GCS) Bidirectional Read API.

### A. Perfect Stream Continuity vs. Constant Stream Seeks
*   **The Stateful gRPC Challenge:** The GCS Bidirectional Read API relies on active, stateful channels that continuously stream bytes sequentially. 
*   **Parallel-Sequential Bottleneck:** When multiple threads request different offsets (e.g., Job 0 reads `[0, 1M)`, Job 1 reads `[256M, 257M)`) multiplexed across a limited connection pool, GCS receives interleaved offset requests on the same gRPC channel. This forces the GCS-side stream coordinator to **invalidate its sequential read-ahead state**, perform physical backend seeks, and re-initialize streams constantly, decimating throughput.
*   **Concurrent-Sequential Continuity:** When all threads read the same offsets in lock-step, the outgoing gRPC streams carry perfectly continuous, monotonically ascending offset ranges. GCS never sees interleaved offset hops. Backend sequential prefetching remains engaged at $100\%$ efficiency, delivering maximum physical network throughput.

### B. Head-of-Line (HoL) Blocking Elimination
In parallel-sequential workloads, chunks for disparate offsets interleave. A delay in receiving a chunk for offset `256MB` blocks the pipeline for offset `0` sharing that connection (Head-of-Line blocking). In concurrent-sequential workloads, chunks arrive in a single, perfectly contiguous sequence. There is no HoL blocking, which eliminates client-side out-of-order buffer reassembly and memory pressure.

---

## 2. Client-Side Mechanics: The Dual-Layer Coalescing Strategy

To prevent redundant network round-trips when multiple local threads or processes request identical blocks, Fusion employs a dual-layer "Single-Flight" and memory-caching layer.

```
                  +-----------------------------------------+
                  |  Concurrent Sequential Read Requests    |
                  |  (e.g., N Threads or N OS Processes)    |
                  +--------------------+--------------------+
                                       |
                     [ GCSFS / FusionFS Coalescing Layer ]
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
        [ In-Process (Thread/Async) ]       [ Inter-Process (Multiprocessing) ]
           Shared asyncio.Future                 /dev/shm Memory-Backed Cache
                     |                                   |
                     v                                   v
         Only 1 Active Task Executes         Non-blocking fcntl.flock Retries
                     |                                   |
                     +-----------------+-----------------+
                                       |
                                       v
                        +----------------------------+
                        |  1 Stateful Network Fetch  |
                        |      to GCS via gRPC       |
                        +----------------------------+
```

### A. In-Process Single-Flighting (Thread & Asyncio Level)
When $N$ local threads or async coroutines submit identical range requests (e.g., `_cat_file_sequential` or `_concurrent_mrd_fetch` targeting `[start, end)`), the requests are intercepted by an in-memory single-flight map:
1.  **Future Coalescing:** The first thread to arrive registers an `asyncio.Future` in a shared `_in_process_reads` registry and initiates the actual fetch.
2.  **Immediate Subscription:** Subsequent threads detecting an active, matching key yield and await the existing future, entirely bypassing any duplicate network calls.
3.  **Simultaneous Resolution:** Once the fetch completes, the data is broadcast to all waiting readers in the same event loop instantly. **This reduces $N$ duplicate network calls to exactly $1$ fetch.**

### B. Inter-Process Shared-Memory Caching (OS Process Level)
For highly distributed workloads where separate OS processes (e.g., FIO concurrent jobs, Dask worker processes, or Ray actors) execute reads without sharing Python memory, Fusion utilizes an async-friendly, memory-backed file-system cache:
1.  **High-Speed RAM Drive (`/dev/shm`):** Cache chunks are written directly to Linux Shared Memory (`/dev/shm`), operating at RAM speeds ($GB/s$) and avoiding disk I/O bottlenecks.
2.  **Non-Blocking Coordinator (`fcntl.flock`):** When Process $A$ starts downloading a chunk, it acquires an exclusive non-blocking lock (`LOCK_EX | fcntl.LOCK_NB`) on a companion lock file. 
3.  **Asynchronous Backoff Spin-Lock:** Concurrent processes $B \dots Z$ attempting to read the same block will fail to acquire the lock. Instead of stalling the CPU or blocking OS threads in the executor, they asynchronously yield (`await asyncio.sleep(0.01)`).
4.  **Instant Cache Hits:** As soon as Process $A$'s atomic write finishes and releases the lock, waiting processes acquire the lock, detect the cached chunk in `/dev/shm`, read it instantly at memory speed, and return. **No duplicate egress bandwidth is consumed across processes.**

---

## 3. Quantitative / Behavioral Profile

| Metric / Dimension | Parallel-Sequential (`parallel-sequential`) | Concurrent-Sequential (`concurrent-sequential`) |
| :--- | :--- | :--- |
| **GCS Connection State** | Heavily thrashed; constant seeks and read-ahead invalidations. | Perfect streaming continuity; prefetch state fully preserved. |
| **Network Requests** | $N$ distinct, parallel requests fetching $N \times \text{Size}$ unique data. | **Exactly $1$ network fetch** serving $N$ clients via coalescing. |
| **Local I/O Speed** | Slowed by network throughput and gRPC stream scheduling. | Memory-backed speeds (up to tens of $GB/s$ via `/dev/shm` and Futures). |
| **Egress & API Costs** | Multiplied linearly by $N$ jobs. | Flat; equivalent to a single-threaded execution. |
| **Head-of-Line Blocking** | High; results in memory buffering and reordering overhead. | **None**; sequential ascending data is dispatched immediately. |

---

## Conclusion

Fusion performs exceptionally well in concurrent-sequential workloads because it **aligns client-side request coalescing with GCS server-side stream optimization**. By consolidating redundant requests into a single network call at both the **asyncio/thread level** (using shared Futures) and the **multiprocessing level** (using lock-synchronized shared memory under `/dev/shm`), Fusion bypasses the network entirely for concurrent readers. The remaining single active fetch proceeds down a state-preserved, sequential gRPC channel, allowing GCS to stream data at the absolute physical limits of the network pipeline.
