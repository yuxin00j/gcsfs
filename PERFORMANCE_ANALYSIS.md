# FusionFS Performance Analysis: Concurrent-Sequential vs. Parallel-Sequential

## Executive Summary
This report analyzes the performance divergence in FusionFS when running **Concurrent-Sequential** versus **Parallel-Sequential** read workloads using the FIO benchmark with the Google Cloud Storage (GCS) bidirectional streaming engine.

Under identical hardware and network conditions, **Concurrent-Sequential** workloads achieve significantly higher throughput and lower latency. This divergence is not caused by FUSE mount serialization, but by the stateful design of the GCS Bidirectional Read API and how FusionFS’s connection-pooling and round-robin multiplexing interact with these stateful streams.

---

## 1. Workload Comparison

The two FIO job profiles compared have the following parameters:

| Configuration Parameter | Parallel-Sequential (`parallel-sequential`) | Concurrent-Sequential (`concurrent-sequential`) |
| :--- | :--- | :--- |
| **Number of Jobs (`numjobs`)** | 8 | 8 |
| **Offset Increment (`offset_increment`)** | **`256m`** | **`0`** |
| **Read size per job (`io_size`)** | 256MB | 256MB |
| **Block Size (`bs`)** | 1MB | 1MB |
| **I/O Engine (`ioengine`)** | `libgcsfs_sync_fio_engine.so` | `libgcsfs_sync_fio_engine.so` |
| **GCS Prefetching (`use_prefetch`)**| Enabled (`1`) | Enabled (`1`) |
| **Cache Type (`cache_type`)** | None (`none`) | None (`none`) |

### Key Difference
* **`parallel-sequential`**: The 8 jobs read 8 **different, non-overlapping 256MB regions** of the same file concurrently (spanning offsets from `0` to `2GB`).
* **`concurrent-sequential`**: All 8 jobs read the **exact same 256MB range** (`[0, 256MB)`) of the file starting at offset 0 simultaneously.

---

## 2. Core Performance Divergence Factors

### A. gRPC Stream State Invalidation & GCS-Side Seeks (The Main Bottleneck)
The Google Cloud Storage (GCS) Bidirectional Read API is stateful. It is designed to stream bytes continuously and sequentially on a single active gRPC channel with server-side read-ahead.

* **In `parallel-sequential` (Constant Seek Overhead)**:
  Since we have 8 threads reading entirely different offsets multiplexed across 4 background gRPC streams, GCS receives interleaved ranges on each channel (e.g., `offset: 0` followed by `offset: 1024MB`). This forces GCS to constantly **invalidate its sequential read-ahead state**, perform physical seek operations on the backend, and re-initialize the stream state. This destroys GCS-side streaming optimizations and introduces massive latency.
* **In `concurrent-sequential` (Perfect Stream Continuity)**:
  GCS sees a perfectly sequential stream of ascending offsets on each gRPC stream because all threads are reading the same offsets in lock-step. No streams are invalidated, no state is thrashed, and GCS's high-speed sequential prefetching runs at maximum efficiency.

### B. Client-side Deduplication and Buffer Sharing
* **In `concurrent-sequential`**:
  Multiple FUSE reads targeting identical blocks arrive concurrently. The active prefetching engine coalesces these into a single network read. The first read-ahead populates the cache buffer, and subsequent concurrent requests hit this cache buffer directly. The net result is **1 network request serving 8 threads**.
* **In `parallel-sequential`**:
  With no active client-side block cache (since `cache_type=none`), the 8 concurrent streams compete for prefetching buffers, resulting in cache thrashing. The engine must fetch **2GB of unique data** from GCS concurrently, saturating client-side processing buffers and connection bandwidth.

### C. Head-of-Line (HoL) Blocking & Client-side Reassembly
* **In `parallel-sequential`**:
  Chunks for different offsets (e.g., Job 0 and Job 4) must interleave over the same gRPC connections. If a chunk for Job 4 is delayed, Job 0 is blocked. This introduces Head-of-Line blocking and forces FusionFS to manage out-of-order buffers in memory.
* **In `concurrent-sequential`**:
  Chunks arrive in a single contiguous sequence. There is no Head-of-Line blocking or reordering overhead, allowing FusionFS to dispatch the data immediately.

---

## 3. Detailed Code Analysis (FusionFS Internal Architecture)

The codebase of FusionFS reveals how concurrent requests are handled and where the performance bottlenecks are created.

### 1. Inode Resolution & Locking (`src/inode.rs`)
In many traditional file systems, concurrent reads on the same file contend for metadata locks. FusionFS optimizes this by using `parking_lot`'s raw reader-writer locks (`RawRwLock`) to protect inodes:

```rust
pub type InodeReadGuard = ArcRwLockReadGuard<RawRwLock, Inode>;
pub type InodeWriteGuard = ArcRwLockWriteGuard<RawRwLock, Inode>;
```

During both workloads, all 8 FIO threads can simultaneously acquire shared read-locks (`InodeReadGuard`) to resolve the file's path. This ensures **zero lock-contention** at the metadata layer.

### 2. Lock-Free FUSE Read Path (`src/fs.rs`)
The FUSE read implementation in FusionFS is entirely asynchronous and does not hold any synchronous locks or serialize requests at the mount level:

```rust
fn read(
    &self,
    header: &kernel::fuse_in_header,
    args: &kernel::fuse_read_in,
) -> impl Future<Output = kernel::Result<Vec<Bytes>>> + Send {
    // ...
    async move {
        // Path resolution via non-blocking inode_map loop
        let path = loop { ... };

        if fh != 0 {
            info!("FUSE read: calling backend.read_stream for nodeid={}, handle={}, path={}, offset={}, size={}", id, fh, path, offset, size);
            match backend.read_stream(id, fh, offset, size).await {
                Ok(data) => return Ok(data),
                // Fallback mechanics...
            }
        }
        // ...
    }
}
```
Because the read path spawns independent futures that run concurrently, the FUSE layer itself does not throttle the 8 threads. The bottleneck lies purely in the backend gRPC layer.

### 3. Bidirectional Multiplexing Bottleneck (`src/backend.rs`)
The core bottleneck is located in `GcsGrpcBidiBackend`. When a file is opened, FusionFS spawns a background stream pool of size `FUSIONFS_STREAMS_PER_INODE` (default: 4) per file:

```rust
let num_streams = std::env::var("FUSIONFS_STREAMS_PER_INODE")
    .unwrap_or_else(|_| "4".to_string())
    .parse::<usize>()
    .unwrap_or(4);
```

When concurrent read requests arrive, they are round-robin multiplexed across these 4 streams:

```rust
fn read_stream(&self, nodeid: u64, handle: u64, offset: u64, size: u32) -> impl std::future::Future<Output = io::Result<Vec<Bytes>>> + Send {
    // ...
    async move {
        // ...
        let (resp_tx, mut resp_rx) = mpsc::unbounded_channel();
        
        let req = BidiReadObjectRequest {
            read_ranges: vec![ReadRange {
                read_offset: offset as i64,
                read_length: size as i64,
                read_id,
            }],
            ..Default::default()
        };

        state.pending.lock().insert(read_id, PendingRead {
            sender: resp_tx,
            request: req.clone(),
            bytes_received: 0,
            forwarded: false,
        });

        // Round-robin index selection across the 4 physical gRPC streams
        let idx = state.next_stream.fetch_add(1, Ordering::Relaxed) % state.txs.len();
        state.txs[idx].send(req).await ...
        
        // Loop awaiting response chunks...
    }
}
```

#### Why `parallel-sequential` Slashes Performance in `read_stream`:
1. Because `numjobs = 8` and `FUSIONFS_STREAMS_PER_INODE = 4`, **multiple threads are forced to share the same physical stream**.
2. Since `parallel-sequential` threads have different offsets, `state.txs[idx]` interleaves non-sequential `ReadRange` requests (e.g. range `[0, 1M)` and range `[256M, 257M)`) on the **same gRPC stream**.
3. This interleaving invalidates the sequential caching state on GCS-side stream coordinators, triggering seeks and dropping streaming throughput to zero-sequential speeds.

---

## 4. Actionable Tuning Recommendations

To optimize these workloads, the following configuration adjustments are recommended:

### For Parallel-Sequential Optimization:
1. **Increase Stream Pool Size**: Set the environment variable `FUSIONFS_STREAMS_PER_INODE` to match the number of parallel threads (e.g. `8`). This ensures each parallel sequential thread gets its own dedicated gRPC stream to GCS, preventing range interleaving and seek invalidation.
2. **Enable Chunk Caching**: If possible, switch `cache_type` from `none` to an active block/page cache. This ensures that even if threads compete, they can read from memory caches instead of triggering round-trips.

### For Concurrent-Sequential Optimization:
1. **Enable Kernel Page Caching**: Ensure `direct=0` is set in your FIO profile and verify that FUSE kernel page caching is enabled. Once the first thread fetches a block, the remaining threads will read directly from kernel memory, bypassing FUSE and gRPC entirely.
2. **Increase Prefetch Block Size**: Increase `block_size` to `32MB` or `64MB` to maximize continuous network throughput for ascending read ranges.
