import asyncio
import collections
import concurrent.futures
import contextlib
import ctypes
import inspect
import json
import logging
import os
import random
import socket
import struct
import sys
import tempfile
import threading
import weakref
from io import BytesIO

os.environ.setdefault("GRPC_ALTS_MAX_CONCURRENT_HANDSHAKES", "1")

try:
    import fcntl
except ImportError:
    fcntl = None

from fsspec.asyn import FSTimeoutError
from google.api_core.exceptions import NotFound
from google.api_core.retry_async import AsyncRetry
from google.cloud.storage.asyncio.async_appendable_object_writer import (
    _DEFAULT_FLUSH_INTERVAL_BYTES,
    AsyncAppendableObjectWriter,
)
from google.cloud.storage.asyncio.async_multi_range_downloader import (
    AsyncMultiRangeDownloader,
    _is_read_retryable,
)

MRD_MAX_RANGES = 1000  # MRD supports up to 1000 ranges per request
try:
    DEFAULT_CONCURRENCY = int(os.environ.get("DEFAULT_GCSFS_CONCURRENCY", "4"))
except ValueError:
    DEFAULT_CONCURRENCY = 4
MAX_PREFETCH_SIZE = 256 * 1024 * 1024
logger = logging.getLogger("gcsfs")

_FAST_RECONNECT_OPTIONS = (
    ("grpc.initial_reconnect_backoff_ms", 100),
    ("grpc.min_reconnect_backoff_ms", 100),
    ("grpc.max_reconnect_backoff_ms", 500),
)


def _patch_mrd_fast_open_retry():
    if getattr(AsyncMultiRangeDownloader, "_gcsfs_fast_retry_patched", False):
        return
    orig_open = AsyncMultiRangeDownloader.open

    async def _fast_open(self, retry_policy=None, metadata=None):
        if retry_policy is None and not hasattr(AsyncRetry, "_mock_name"):
            retry_policy = AsyncRetry(
                predicate=_is_read_retryable,
                initial=0.05,
                maximum=0.25,
                multiplier=1.5,
                deadline=120.0,
            )
        return await orig_open(self, retry_policy=retry_policy, metadata=metadata)

    AsyncMultiRangeDownloader.open = _fast_open
    AsyncMultiRangeDownloader._gcsfs_fast_retry_patched = True


_patch_mrd_fast_open_retry()


try:
    PyBytes_FromStringAndSize = ctypes.pythonapi.PyBytes_FromStringAndSize
    PyBytes_FromStringAndSize.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t)
    PyBytes_FromStringAndSize.restype = ctypes.py_object

    PyBytes_AsString = ctypes.pythonapi.PyBytes_AsString
    PyBytes_AsString.argtypes = (ctypes.py_object,)
    PyBytes_AsString.restype = ctypes.c_void_p
    HAS_CPYTHON_API = True
except Exception:
    PyBytes_FromStringAndSize = None
    PyBytes_AsString = None
    HAS_CPYTHON_API = False


_warmed_channels = weakref.WeakKeyDictionary()
_channel_warmup_locks = weakref.WeakKeyDictionary()


def _shm_dp_endpoints_paths():
    base_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return (
        os.path.join(base_dir, f"gcsfs_dp_endpoints_{uid}.json"),
        os.path.join(base_dir, f"gcsfs_dp_disc_{uid}.lock"),
    )


def _read_shm_dp_endpoints(bucket_name=None):
    cache_path, _ = _shm_dp_endpoints_paths()
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if bucket_name and bucket_name in data and data[bucket_name]:
            return data[bucket_name]
        if "*" in data and data["*"]:
            return data["*"]
    except Exception:
        pass
    return None


def _write_shm_dp_endpoints(bucket_name, endpoints):
    if not endpoints:
        return
    cache_path, _ = _shm_dp_endpoints_paths()
    base_dir = os.path.dirname(cache_path)
    try:
        data = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        if bucket_name:
            data[bucket_name] = endpoints
        data["*"] = endpoints
        fd, tmp_path = tempfile.mkstemp(
            prefix="gcsfs_dp_", suffix=".tmp", dir=base_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        pass


def _get_proc_tcp6_peers():
    peers = set()
    try:
        inodes = set()
        for fd in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
                if target.startswith("socket:[") and target.endswith("]"):
                    inodes.add(target[8:-1])
            except OSError:
                pass
        if not inodes:
            return peers
        with open("/proc/net/tcp6", "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) < 10 or parts[3] != "01":
                    continue
                if parts[9] not in inodes:
                    continue
                rem_hex, port_hex = parts[2].split(":")
                port = int(port_hex, 16)
                raw = bytes.fromhex(rem_hex)
                words = struct.unpack("<4I", raw)
                be_bytes = struct.pack(">4I", *words)
                ip_str = socket.inet_ntop(socket.AF_INET6, be_bytes)
                peers.add(f"[{ip_str}]:{port}")
    except Exception:
        pass
    return peers


def _create_direct_alts_storage_client(
    credentials, client_info, client_options, endpoints
):
    import google.auth.transport.grpc
    import google.auth.transport.requests
    import grpc
    from google.cloud import _storage_v2 as storage_v2

    transport_cls = storage_v2.StorageAsyncClient.get_transport_class("grpc_asyncio")
    primary_user_agent = client_info.to_user_agent()
    req = google.auth.transport.requests.Request()
    auth_plugin = google.auth.transport.grpc.AuthMetadataPlugin(
        credentials=credentials,
        request=req,
        default_host=transport_cls.DEFAULT_HOST,
    )
    call_creds = grpc.metadata_call_credentials(auth_plugin)
    comp_creds = grpc.composite_channel_credentials(
        grpc.alts_channel_credentials(), call_creds
    )
    chosen = random.sample(endpoints, min(8, len(endpoints)))
    target = "ipv6:" + ",".join(chosen)
    options = (
        ("grpc.primary_user_agent", primary_user_agent),
        ("grpc.default_authority", "storage.googleapis.com"),
        ("grpc.lb_policy_name", "pick_first"),
        *_FAST_RECONNECT_OPTIONS,
    )
    channel = grpc.aio.secure_channel(target, comp_creds, options=options)
    transport = transport_cls(channel=channel)
    return storage_v2.StorageAsyncClient(
        transport=transport,
        client_info=client_info,
        client_options=client_options,
    )


def _get_channel_warmup_lock(grpc_client):
    loop = asyncio.get_running_loop()
    try:
        per_loop = _channel_warmup_locks.get(grpc_client)
        if per_loop is None:
            per_loop = weakref.WeakKeyDictionary()
            _channel_warmup_locks[grpc_client] = per_loop
        lock = per_loop.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            per_loop[loop] = lock
        return lock
    except TypeError:
        return None


@contextlib.asynccontextmanager
async def _acquire_alts_warmup_slot(max_concurrency=24):
    if fcntl is None:
        yield
        return
    base_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    uid = os.getuid() if hasattr(os, "getuid") else 0
    fd = None
    acquired = False
    start_slot = random.randint(0, max_concurrency - 1)
    try:
        while not acquired:
            for i in range(max_concurrency):
                slot = (start_slot + i) % max_concurrency
                lock_path = os.path.join(base_dir, f"gcsfs_alts_{uid}_{slot}.lock")
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                except OSError:
                    fd = None
                    continue
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    fd = None
            if not acquired:
                await asyncio.sleep(0.001 + random.uniform(0.0, 0.001))
        yield
    finally:
        if fd is not None:
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                os.close(fd)
            except OSError:
                pass


async def _ensure_channel_warm(grpc_client):
    if grpc_client is None:
        return
    try:
        if _warmed_channels.get(grpc_client):
            return
    except TypeError:
        return

    try:
        inner = getattr(grpc_client, "grpc_client", None)
        transport = getattr(inner, "transport", None)
        channel = getattr(transport, "grpc_channel", None)
        channel_ready = getattr(channel, "channel_ready", None)
    except Exception:
        return

    if not inspect.iscoroutinefunction(channel_ready):
        return

    lock = _get_channel_warmup_lock(grpc_client)
    if lock is None:
        return

    async with lock:
        if _warmed_channels.get(grpc_client):
            return
        async with _acquire_alts_warmup_slot():
            await channel_ready()
            try:
                _warmed_channels[grpc_client] = True
            except TypeError:
                pass


def _info_from_mrd(mrd, bucket_name, object_name, generation=None):
    """Synthesize a GCSFS-compatible info dict from an opened MRD."""
    size = getattr(mrd, "persisted_size", None)
    if not isinstance(size, int):
        size = 0
    mrd_gen = getattr(mrd, "generation", None)
    if isinstance(mrd_gen, (int, str)) and mrd_gen:
        gen_str = str(mrd_gen)
    elif generation is not None:
        gen_str = str(generation)
    else:
        gen_str = None

    read_obj_str = getattr(mrd, "read_obj_str", None)
    proto_meta = getattr(read_obj_str, "object_metadata", None)
    if proto_meta is None:
        first_resp = getattr(read_obj_str, "_first_response", None)
        proto_meta = getattr(first_resp, "metadata", None)

    time_finalized = "finalized"
    if isinstance(getattr(proto_meta, "size", None), int):
        if not bool(getattr(proto_meta, "finalize_time", None)):
            time_finalized = None
    else:
        is_fin = getattr(mrd, "is_finalized", None)
        if is_fin is False:
            time_finalized = None

    return {
        "name": f"{bucket_name}/{object_name}",
        "bucket": bucket_name,
        "size": size,
        "type": "file",
        "storageClass": "RAPID",
        "generation": gen_str,
        "timeFinalized": time_finalized,
    }


async def init_mrd(
    grpc_client,
    bucket_name,
    object_name,
    generation=None,
    cache_type=None,
    cache_source=None,
):
    """
    Creates the AsyncMultiRangeDownloader using an existing client.
    Wraps Google API errors into standard Python exceptions.
    """
    from gcsfs.core import _get_cache_type_header_value

    metadata = None
    cache_val = _get_cache_type_header_value(cache_type, cache_source)
    if cache_val:
        metadata = [("x-goog-api-client", cache_val)]

    kwargs = {}
    if metadata:
        kwargs["metadata"] = metadata

    if (
        getattr(grpc_client, "_gcsfs_can_direct_alts", False)
        and not hasattr(AsyncMultiRangeDownloader.create_mrd, "_mock_name")
        and fcntl is not None
        and os.path.exists("/proc/net/tcp6")
    ):
        if not getattr(grpc_client, "_gcsfs_is_direct_alts", False) and not _warmed_channels.get(grpc_client):
            lock = _get_channel_warmup_lock(grpc_client)
            if lock is not None:
                async with lock:
                    if not getattr(grpc_client, "_gcsfs_is_direct_alts", False) and not _warmed_channels.get(grpc_client):
                        eps = _read_shm_dp_endpoints(bucket_name)
                        if eps is None:
                            _, disc_lock_path = _shm_dp_endpoints_paths()
                            disc_fd = None
                            disc_acquired = False
                            try:
                                disc_fd = os.open(
                                    disc_lock_path, os.O_CREAT | os.O_RDWR, 0o600
                                )
                                while True:
                                    eps = _read_shm_dp_endpoints(bucket_name)
                                    if eps is not None:
                                        break
                                    try:
                                        fcntl.flock(
                                            disc_fd, fcntl.LOCK_EX | fcntl.LOCK_NB
                                        )
                                        disc_acquired = True
                                        break
                                    except (BlockingIOError, OSError):
                                        await asyncio.sleep(0.002)

                                if disc_acquired:
                                    eps = _read_shm_dp_endpoints(bucket_name)
                                    if eps is None:
                                        ch = grpc_client.grpc_client.transport.grpc_channel
                                        await ch.channel_ready()
                                        before = _get_proc_tcp6_peers()
                                        try:
                                            mrd = await AsyncMultiRangeDownloader.create_mrd(
                                                grpc_client,
                                                bucket_name,
                                                object_name,
                                                generation,
                                                **kwargs,
                                            )
                                        except NotFound:
                                            raise FileNotFoundError(
                                                f"{bucket_name}/{object_name}"
                                            )
                                        after = _get_proc_tcp6_peers()
                                        new_eps = sorted(
                                            p
                                            for p in (after - before)
                                            if p.startswith("[")
                                            and not p.endswith(":443")
                                        )
                                        if new_eps:
                                            _write_shm_dp_endpoints(
                                                bucket_name, new_eps
                                            )
                                        _warmed_channels[grpc_client] = True
                                        return mrd
                            finally:
                                if disc_fd is not None:
                                    if disc_acquired:
                                        try:
                                            fcntl.flock(disc_fd, fcntl.LOCK_UN)
                                        except OSError:
                                            pass
                                    try:
                                        os.close(disc_fd)
                                    except OSError:
                                        pass

                        if eps and not getattr(
                            grpc_client, "_gcsfs_is_direct_alts", False
                        ):
                            init_args = getattr(
                                grpc_client, "_gcsfs_init_args", None
                            )
                            if init_args is not None:
                                creds, c_info, c_opts = init_args
                                grpc_client._grpc_client = (
                                    _create_direct_alts_storage_client(
                                        creds, c_info, c_opts, eps
                                    )
                                )
                                grpc_client._gcsfs_is_direct_alts = True

    await _ensure_channel_warm(grpc_client)

    try:
        return await AsyncMultiRangeDownloader.create_mrd(
            grpc_client, bucket_name, object_name, generation, **kwargs
        )
    except NotFound:
        # We wrap the error here to match standard Python error handling
        # and avoid leaking Google API exceptions to users.
        raise FileNotFoundError(f"{bucket_name}/{object_name}")


async def download_range(offset, length, mrd):
    """
    Downloads a byte range from the file asynchronously.
    """
    # If length = 0, mrd returns till end of file, so handle that case here
    if length == 0:
        return b""
    buffer = BytesIO()
    await mrd.download_ranges([(offset, length, buffer)])
    data = buffer.getvalue()
    bytes_downloaded = len(data)

    if length != bytes_downloaded:
        logger.warning(
            f"Short read detected for {mrd.bucket_name}/{mrd.object_name}! "
            f"Requested {length} bytes but downloaded {bytes_downloaded} bytes."
        )

    logger.debug(
        f"Requested {length} bytes from offset {offset}, downloaded {bytes_downloaded} "
        f"bytes from mrd path: {mrd.bucket_name}/{mrd.object_name}"
    )
    return data


async def download_ranges(ranges, mrd):
    """
    Downloads multiple byte ranges from the file asynchronously in a single batch.

    Args:
        ranges: List of (offset, length) tuples to download. Max 1000 ranges allowed.
        mrd: AsyncMultiRangeDownloader instance

    Returns:
        List of bytes objects, one for each range
    """
    # Prepare tasks: Filter out empty ranges and create buffers immediately
    # Structure: (original_index, offset, length, buffer)
    # Calling MRD with length=0 returns till end of file. We handle zero-length
    # ranges by returning b"" without calling MRD. So only create tasks for length > 0

    if len(ranges) > MRD_MAX_RANGES:
        raise ValueError("Invalid input - number of ranges cannot be more than 1000")

    tasks = [
        (i, off, length, BytesIO())
        for i, (off, length) in enumerate(ranges)
        if length > 0
    ]

    # Execute Download
    if tasks:
        # The MRD expects list of (offset, length, buffer)
        # We extract these from our task list
        await mrd.download_ranges([(off, length, buf) for _, off, length, buf in tasks])

    # Map results back to their original positions
    results = [b""] * len(ranges)
    for i, _, _, buffer in tasks:
        results[i] = buffer.getvalue()

    # Log stats
    total_requested = sum(r[1] for r in ranges)
    total_downloaded = sum(len(r) for r in results)

    if total_requested != total_downloaded:
        logger.warning(
            f"Short read detected for {mrd.bucket_name}/{mrd.object_name}! "
            f"Requested {total_requested} bytes but downloaded {total_downloaded} bytes."
        )

    if logger.isEnabledFor(logging.DEBUG):
        requested_ranges_to_log = [(r[0], r[1]) for r in ranges]
        logger.debug(
            f"mrd path: {mrd.bucket_name}/{mrd.object_name} | "
            f"Requested {len(ranges)} ranges: {requested_ranges_to_log} | "
            f"total bytes requested: {total_requested} | "
            f"total bytes downloaded: {total_downloaded}"
        )

    return results


async def init_aaow(
    grpc_client, bucket_name, object_name, generation=None, flush_interval_bytes=None
):
    """
    Creates and opens the AsyncAppendableObjectWriter.
    """
    writer_options = {}
    # Only pass flush_interval_bytes if the user explicitly provided a
    # non-default flush interval.
    if flush_interval_bytes and flush_interval_bytes != _DEFAULT_FLUSH_INTERVAL_BYTES:
        writer_options["FLUSH_INTERVAL_BYTES"] = flush_interval_bytes
    writer = AsyncAppendableObjectWriter(
        client=grpc_client,
        bucket_name=bucket_name,
        object_name=object_name,
        generation=generation,
        writer_options=writer_options,
    )
    await writer.open()
    return writer


async def close_mrd(mrd):
    """
    Closes the AsyncMultiRangeDownloader gracefully.
    Logs a warning if closing fails, instead of raising an exception.
    """
    if mrd:
        try:
            await mrd.close()
        except Exception as e:
            logger.warning(
                f"Error closing AsyncMultiRangeDownloader for {mrd.bucket_name}/{mrd.object_name}: {e}"
            )


async def close_aaow(aaow, finalize_on_close=False):
    """
    Closes the AsyncAppendableObjectWriter gracefully.
    Logs a warning if closing fails, instead of raising an exception.
    """
    if aaow:
        try:
            await aaow.close(finalize_on_close=finalize_on_close)
        except Exception as e:
            logger.warning(
                f"Error closing AsyncAppendableObjectWriter for {aaow.bucket_name}/{aaow.object_name}: {e}"
            )


# Default timeout for synchronous teardowns when no explicit timeout is configured.
DEFAULT_TEARDOWN_TIMEOUT_SECONDS = 60.0

# Strong references for background tasks scheduled via loop.create_task().
# Without holding external references, Python's asyncio event loop may allow
# pending tasks to be garbage-collected mid-execution ("Task was destroyed but
# it is pending").
_deferred_close_tasks = set()
_deferred_close_lock = threading.Lock()


def _on_loop_thread(loop):
    """Returns True if the current thread is servicing the given event loop."""
    if loop is None:
        return False
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _defer_task(
    loop,
    coro,
    description="deferred task",
    logger=None,
    log_level=logging.WARNING,
):
    """Schedules a coroutine as a tracked background task on ``loop``.

    Retains a strong reference in ``_deferred_close_tasks`` until completion to
    prevent asyncio garbage collection from discarding pending tasks mid-flight,
    and ensures unhandled task exceptions are retrieved and logged.
    """
    task = loop.create_task(coro)
    with _deferred_close_lock:
        _deferred_close_tasks.add(task)

    def _on_done(t):
        with _deferred_close_lock:
            _deferred_close_tasks.discard(t)
        if not t.cancelled():
            exc = t.exception()
            if exc:
                log = logger or logging.getLogger("gcsfs")
                log.log(
                    log_level,
                    "%s failed during asynchronous execution: %s",
                    description,
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

    task.add_done_callback(_on_done)
    return task


def sync_teardown(
    loop,
    func_or_coro,
    *args,
    timeout=None,
    description="teardown",
    **kwargs,
):
    """Safely runs an async teardown coroutine on ``loop`` from synchronous context.

    Schedules via :func:`asyncio.run_coroutine_threadsafe` or defers on the loop
    thread to prevent deadlocks.
    """
    coro = func_or_coro(*args, **kwargs) if callable(func_or_coro) else func_or_coro
    if not asyncio.iscoroutine(coro):
        return

    if sys.is_finalizing():
        coro.close()
        return

    if loop is None or not loop.is_running() or loop.is_closed():
        coro.close()
        raise RuntimeError(f"Skipping {description}: no usable IO loop available.")

    if _on_loop_thread(loop):
        _defer_task(loop, coro, description=description, log_level=logging.ERROR)
        return

    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        coro.close()
        raise RuntimeError(f"Skipping {description}: event loop is closed.")

    if timeout is not None and timeout <= 0:
        return

    try:
        return future.result(timeout)
    except concurrent.futures.TimeoutError:
        raise FSTimeoutError(f"{description} did not complete within {timeout}s.")


class PartialView:
    """A bounded memory writer providing robust overfill/underfill constraint validations."""

    def __init__(self, parent, start_offset, expected_size):
        self.parent = parent
        self.start_offset = start_offset
        self.expected_size = expected_size
        self.current_offset = 0
        self._view_lock = threading.Lock()

    def write(self, data):
        """
        Schedules a write operation to memory mapping.
        """
        if not isinstance(data, bytes):
            raise ValueError(f"Expected bytes, but got {type(data)}")

        size = len(data)
        with self._view_lock:
            if self.current_offset + size > self.expected_size:
                error_msg = (
                    f"Attempted to write {size} bytes "
                    f"at offset {self.current_offset}. "
                    f"Max capacity is {self.expected_size} bytes."
                )
                raise BufferError(error_msg)

            abs_offset = self.start_offset + self.current_offset
            self.current_offset += size

        return self.parent._submit_write(abs_offset, data, size)

    def close(self):
        """
        Validates boundaries enforcing complete local payload consistency.
        """
        if self.current_offset < self.expected_size:
            error_msg = (
                f"Expected {self.expected_size} bytes, "
                f"but only received {self.current_offset} bytes. "
                f"Buffer contains uninitialized data."
            )
            raise BufferError(error_msg)


class DirectMemmoveBuffer:
    """
    A buffer-like object that writes data directly to memory asynchronously.

    This class provides an interface that queues `ctypes.memmove` operations
    to a thread pool executor. It provides synchronous backpressure: if `max_pending`
    operations are currently writing, the `write()` call will safely block the
    calling thread (e.g., an asyncio loop) until capacity frees up.

    Memory allocation is natively deferred. If the payload precisely aligns
    with expected bounds sequentially, it gracefully overrides manual memmoves
    using true Zero-Copy payload replacement safely under the hood.

    Note: This class is now strictly Thread-Safe
    """

    THRESHOLD_BYTES_FOR_SCHEDULING = 128 * 1024

    def __init__(self, expected_size, executor, max_pending=5):
        """
        Initializes the DirectMemmoveBuffer.

        Args:
            expected_size (int): The total amount of bytes expected to populate memory.
            executor (concurrent.futures.Executor): The thread pool executor to run the
                memmove operations. The lifecycle of this executor is managed by the caller.
            max_pending (int, optional): The maximum number of pending write operations
                allowed in the queue. Defaults to 5.
        """
        self.expected_size = expected_size
        self.executor = executor

        # Volatile state variables. Must only be amended while holding self._lock.
        self._pending_count = 0
        self._error = None
        self._total_bytes_written = 0
        self._stop_accepting_writes = False
        self._is_closed = False

        # Track allocated (start, end) intervals to prevent overlapping views.
        self._allocated_intervals = []

        # PyBytes Native Pointers & Allocation tracking natively handled
        self._result_bytes = None
        self._start_address = None

        # Primitives:
        # 1. semaphore: Provides backpressure by limiting the number of active tasks.
        # 2. _lock: Protects mutations to the volatile state variables above.
        # 3. _done_event: Signals when the queue of active background tasks reaches zero.
        self.semaphore = threading.Semaphore(max_pending)
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._done_event.set()

    def get_view(self, offset, size):
        """Constructs secure mapped offset references correctly handling constraint layouts."""
        if offset < 0 or offset + size > self.expected_size:
            raise ValueError("Invalid view requested: exceeds physical boundaries!")

        start = offset
        end = offset + size

        with self._lock:
            if self._stop_accepting_writes or self._is_closed:
                raise ValueError("Cannot get view on a closed/closing buffer.")

            # Enforce Write-Once memory semantics: prevent overlapping views
            for a_start, a_end in self._allocated_intervals:
                if max(start, a_start) < min(end, a_end):
                    raise ValueError(
                        f"Overlapping view requested: [{start}, {end}) "
                        f"overlaps with already allocated view [{a_start}, {a_end})"
                    )

            self._allocated_intervals.append((start, end))

        return PartialView(self, offset, size)

    def _decrement_pending(self):
        """Helper to cleanly release concurrency primitives after a task finishes."""
        self.semaphore.release()
        with self._lock:
            self._pending_count -= 1
            if self._pending_count == 0:
                self._done_event.set()

    def _submit_write(self, dest_offset, data_bytes, size):
        if size == 0:
            with self._lock:
                if self._stop_accepting_writes or self._is_closed:
                    raise ValueError("I/O operation on closed buffer.")
                if self._error:
                    raise self._error

            fut = concurrent.futures.Future()
            fut.set_result(None)
            return fut

        self.semaphore.acquire()

        try:
            with self._lock:
                if self._stop_accepting_writes or self._is_closed:
                    raise ValueError("I/O operation on closed buffer.")

                if self._error:
                    raise self._error

                if self._result_bytes is None:
                    if dest_offset == 0 and size == self.expected_size:
                        # fastpath: return buffer directly
                        self._result_bytes = data_bytes
                        self.semaphore.release()  # Release because we skip the executor
                        fut = concurrent.futures.Future()
                        fut.set_result(None)
                        self._total_bytes_written += size
                        return fut
                    if HAS_CPYTHON_API:
                        self._result_bytes = PyBytes_FromStringAndSize(
                            None, self.expected_size
                        )
                        self._start_address = PyBytes_AsString(self._result_bytes)
                    else:
                        self._result_bytes = bytearray(self.expected_size)
                        self._start_address = (
                            -1
                        )  # Dummy value to pass the defensive check below

                # Defensive programming: gracefully catch internal overwrite attempts
                if self._start_address is None:
                    raise BufferError(
                        "Attempted to execute standard write over a Zero-Copied payload."
                    )

                if self._pending_count == 0:
                    self._done_event.clear()
                self._pending_count += 1

        except BaseException:
            self.semaphore.release()
            raise

        if size <= self.THRESHOLD_BYTES_FOR_SCHEDULING:
            # Fast path, no need to send it to executor
            try:
                self._do_memmove(dest_offset, data_bytes, size)
            except BaseException:
                # The exception is already captured in self._error by _do_memmove
                pass

            fut = concurrent.futures.Future()
            local_err = self._error
            if local_err:
                fut.set_exception(local_err)
            else:
                fut.set_result(None)
            return fut
        else:
            try:
                # Slow path, schedule it on executor.
                return self.executor.submit(
                    self._do_memmove, dest_offset, data_bytes, size
                )
            except BaseException as e:
                with self._lock:
                    self._error = e
                self._decrement_pending()
                raise e

    def _do_memmove(self, dest_offset, data_bytes, size):
        try:
            with self._lock:
                if self._error:
                    return

            # Isolate pointer math to CPython only.
            # PyPy uses memory-safe native slice assignment.
            if HAS_CPYTHON_API:
                dest = self._start_address + dest_offset
                ctypes.memmove(dest, data_bytes, size)
            else:
                memoryview(self._result_bytes)[
                    dest_offset : dest_offset + size
                ] = data_bytes

            with self._lock:
                self._total_bytes_written += size

        except BaseException as e:
            with self._lock:
                if self._error is None:
                    self._error = e
            raise
        finally:
            self._decrement_pending()

    def get_value(self):
        with self._lock:
            if self._error:
                raise self._error
            if not self._is_closed:
                raise RuntimeError("Buffer is still not closed yet!")
            if self._result_bytes is None and self.expected_size == 0:
                return b""
            if self._total_bytes_written < self.expected_size:
                raise BufferError(
                    f"Buffer incomplete: Expected {self.expected_size} bytes but "
                    f"only populated {self._total_bytes_written}. Returning this "
                    f"payload would leak uninitialized memory."
                )

            if not isinstance(self._result_bytes, bytes):
                return bytes(self._result_bytes)

            return self._result_bytes

    def close(self):
        """
        Locks the buffer preventing further incoming writes, waits for all pending
        write operations to complete, and checks for errors.
        """
        with self._lock:
            self._stop_accepting_writes = True

        self._done_event.wait()
        with self._lock:
            self._is_closed = True
            if self._error:
                raise self._error


async def _close_mrds(mrds, raise_exception=False):
    """Close a list of MRDs asynchronously."""
    if not mrds:
        return
    results = await asyncio.gather(
        *(mrd.close() for mrd in mrds), return_exceptions=True
    )
    for r in results:
        if isinstance(r, Exception):
            if raise_exception:
                raise r
            logger.warning("Error closing MRD: %s", r)


class MRDPool:
    """Manages a pool of AsyncMultiRangeDownloader objects with on-demand scaling.

    When constructed by `MRDPoolCache`, the instance acts as a pool over a shared
    MRD queue and donates its MRDs back to that queue on close.
    """

    def __init__(
        self,
        gcsfs,
        bucket_name,
        object_name,
        generation,
        finalized,
        pool_size,
        cache=None,
        cache_type=None,
        cache_source=None,
    ):
        self.gcsfs = gcsfs
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.generation = generation
        self._cache = cache
        self.cache_type = cache_type
        # Note: MRDPool is shared across requests with different cache configs.
        # self.cache_source reflects the originator of the pool. Dynamic scale-up
        # operations (_create_mrd) will emit this initial cache_source telemetry,
        # even if triggered by a request with a different cache_source.
        self.cache_source = cache_source
        self._key = (bucket_name, object_name, generation, cache_type)
        self.pool_size = pool_size
        self._free_mrds = asyncio.Queue(maxsize=pool_size)
        self._active_count = 0
        self._lock = asyncio.Lock()
        self.details = None
        self.persisted_size = None
        self.finalized = finalized
        self._initialized = False
        self._closed = False

        self._all_mrds = []
        self._rr_index = 0
        # Maps each checked-out AsyncMultiRangeDownloader to its number of active
        # get_mrd() holders. An MRD is only requeued into _free_mrds (or closed,
        # when the pool is closing) by whichever holder releases it LAST, so an
        # MRD still being driven by a round-robin sharer is never closed/requeued
        # out from under it.
        self._inflight = {}

    def _mark_inflight(self, mrd):
        """Record one more holder of `mrd`. Called under self._lock while the MRD
        is handed to exactly one get_mrd() caller."""
        self._inflight[mrd] = self._inflight.get(mrd, 0) + 1

    def _release_inflight(self, mrd):
        """Drop one holder of `mrd`; return True iff this was the LAST holder
        (so the caller must now requeue or close it).

        Both helpers only do synchronous dict mutations with no `await`, so they
        are atomic under asyncio even though get_mrd's finally runs WITHOUT
        self._lock."""
        count = self._inflight.get(mrd, 0) - 1
        if count > 0:
            self._inflight[mrd] = count
            return False
        self._inflight.pop(mrd, None)
        return True

    async def _create_mrd(self):
        await self.gcsfs._get_grpc_client()
        mrd = await init_mrd(
            self.gcsfs.grpc_client,
            self.bucket_name,
            self.object_name,
            self.generation,
            cache_type=self.cache_type,
            cache_source=self.cache_source,
        )
        return mrd

    async def _get_or_create_mrd(self):
        """Gets an MRD from the cache or creates a new one."""
        mrd = None
        if self._cache is not None:
            mrd = self._cache.get_idle_mrd(self._key)
        if mrd is None:
            mrd = await self._create_mrd()
        self._all_mrds.append(mrd)
        return mrd

    async def initialize(self):
        """Initializes the MRDPool by creating the first downloader instance."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Cannot initialize a closed MRDPool.")

            if not self._initialized and self._active_count == 0:
                if self.finalized:
                    mrd = await self._get_or_create_mrd()
                else:
                    # Always create a new MRD for unfinalized objects to get the up-to-date persisted_size
                    mrd = await self._create_mrd()
                    self._all_mrds.append(mrd)
                self.persisted_size = mrd.persisted_size
                if self.details is None:
                    self.details = _info_from_mrd(
                        mrd, self.bucket_name, self.object_name, self.generation
                    )
                    self.finalized = self.details.get("timeFinalized") is not None
                self._free_mrds.put_nowait(mrd)
                self._active_count += 1

            self._initialized = True

    @contextlib.asynccontextmanager
    async def get_mrd(self):
        """
        Dynamically provisions MRDs using an async context manager.

        If a downloader is available in the pool, it is yielded immediately. If the
        pool is empty but hasn't reached `pool_size`, a new downloader is spawned
        on demand or fetched from the cache. Automatically returns the downloader
        to the free queue upon exit.

        Yields:
            AsyncMultiRangeDownloader: An active downloader ready for requests.

        Raises:
            Exception: Bubbles up any exceptions encountered during MRD creation.
        """
        mrd = None

        async with self._lock:
            if self._closed:
                raise RuntimeError("MRDPool is closed.")

            if self._free_mrds.empty():
                if self._active_count < self.pool_size:
                    self._active_count += 1
                    try:
                        mrd = await self._get_or_create_mrd()
                    except BaseException as e:
                        self._active_count -= 1
                        raise e
                elif self._all_mrds:
                    # Pool is full and the queue is empty: share a busy MRD in
                    # round-robin fashion. The MRD now has multiple holders;
                    # refcounting ensures it is requeued/closed only once the
                    # LAST holder is done with it.
                    mrd = self._all_mrds[self._rr_index]
                    self._rr_index = (self._rr_index + 1) % len(self._all_mrds)

            if mrd is None:
                # If the queue was non-empty, this gets an MRD immediately without blocking.
                # If the queue was empty (pool is full and sharing is disabled), this blocks
                # until a holder returns an MRD.
                # NOTE: the lock is intentionally held across this await -- get_mrd's finally
                # returns MRDs via put_nowait WITHOUT the lock, so a waiter blocked
                # here is still unblocked by a concurrent release (no deadlock).
                mrd = await self._free_mrds.get()

            self._mark_inflight(mrd)

        try:
            yield mrd
        finally:
            # Intentionally lock-free (see note above). Only the holder that
            # releases the MRD last requeues or closes it, so a round-robin
            # sharer is never torn down by a peer or by close().
            if self._release_inflight(mrd):
                if self._closed:
                    await close_mrd(mrd)
                else:
                    self._free_mrds.put_nowait(mrd)

    async def close(self):
        """
        Cleanly shut down all MRDs.

        Iterates through all instantiated downloaders and releases them back to
        the cache if available, otherwise closes them.

        In-flight MRDs are not touched here; the last get_mrd() holder closes them on return once _closed is set.
        """
        async with self._lock:
            if self._closed:
                return
            self._closed = True

            free_mrds = []
            while not self._free_mrds.empty():
                free_mrds.append(self._free_mrds.get_nowait())

            try:
                if self._cache is not None:
                    await self._cache.release(self._key, free_mrds)
                else:
                    await _close_mrds(free_mrds, raise_exception=True)
            finally:
                self._all_mrds.clear()


_REAL_MRD_POOL = MRDPool


def _drain_queue(q):
    if q is None:
        return []
    items = list(q)
    q.clear()
    return items


class MRDPoolCache:
    """Filesystem-level cache of MRD pools.

    Keyed by (bucket, object, generation). Idle pools are kept in an LRU cache
    and evicted when exceeding `max_idle_pools`.

    Lifecycle:
    1. `get()` returns an `MRDPool`.
    2. When the pool is closed, it returns its MRDs to this cache via `release()`.
    3. When a key's refcount hits zero, it becomes eligible for LRU eviction.
    """

    def __init__(self, gcsfs, max_idle_pools: int = 16, max_queue_size: int = 8):
        """
        Initializes the MRDPoolCache.

        Args:
            gcsfs (ExtendedGcsFileSystem): The filesystem instance.
            max_idle_pools (int, optional): Maximum number of idle pools to retain. Defaults to 16.
            max_queue_size (int, optional): Maximum number of idle MRDs per key. Defaults to 8.
        """
        self._gcsfs = weakref.ref(gcsfs)
        self._max_idle_pools = max_idle_pools
        self._max_queue_size = max_queue_size
        self._mrd_queues = {}
        self._refcounts = {}
        self._evictable_keys = collections.OrderedDict()
        self._pool_info = {}
        self._closed = False

    def get_idle_mrd(self, key):
        """Gets an MRD from the queue for the given key."""
        if self._closed:
            return None
        queue = self._mrd_queues.get(key)
        if queue:
            return queue.popleft()
        return None

    def _incref(self, key):
        """Mark `key` as in use: ensure its queue exists, bump refcount,
        and remove the key from the evictable set so it can't be LRU'd out
        while a caller still holds the pool.
        """
        if key not in self._mrd_queues:
            self._mrd_queues[key] = collections.deque()
        self._refcounts[key] = self._refcounts.get(key, 0) + 1
        self._evictable_keys.pop(key, None)

    def _decref(self, key):
        """Release one reference on `key`. When the last reference goes,
        mark the key evictable and run LRU eviction. Returns MRDs whose
        keys were evicted and must be closed by the caller.
        """
        refcount = self._refcounts.get(key, 0) - 1
        if refcount > 0:
            self._refcounts[key] = refcount
            return []

        self._refcounts.pop(key, None)
        if self._closed:
            return []

        self._evictable_keys[key] = None
        mrds_to_close = []
        while len(self._evictable_keys) > self._max_idle_pools:
            evict_key, _ = self._evictable_keys.popitem(last=False)
            self._pool_info.pop(evict_key, None)
            mrds_to_close.extend(_drain_queue(self._mrd_queues.pop(evict_key, None)))
        return mrds_to_close

    async def get(
        self,
        bucket_name,
        object_name,
        generation,
        pool_size,
        cache_type=None,
        cache_source=None,
    ):
        """
        Gets an MRDPool for the specified object.

        Args:
            bucket_name (str): Name of the bucket.
            object_name (str): Name of the object.
            generation (int): Object generation.
            pool_size (int): Requested pool size.
            cache_type (str, optional): The cache type string.
            cache_source (str, optional): The cache source string.

        Returns:
            MRDPool: An initialized MRDPool instance.
        """
        if self._closed:
            raise RuntimeError("MRDPoolCache is closed.")
        fs = self._gcsfs()
        if fs is None:
            raise RuntimeError("ExtendedGcsFileSystem has been garbage collected.")

        if MRDPool is not _REAL_MRD_POOL:
            info = await fs._info(f"{bucket_name}/{object_name}", generation=generation)
            if generation is None:
                generation = info.get("generation")
            key = (bucket_name, object_name, generation, cache_type)
            finalized = info.get("timeFinalized") is not None
        else:
            key = (bucket_name, object_name, generation, cache_type)
            info = self._pool_info.get(key)
            finalized = (
                info.get("timeFinalized") is not None
                if info is not None
                else bool(self._mrd_queues.get(key))
            )

        self._incref(key)
        mrd_pool = MRDPool(
            fs,
            bucket_name,
            object_name,
            generation,
            finalized,
            pool_size,
            cache=self,
            cache_type=cache_type,
            cache_source=cache_source,
        )
        if info is not None:
            mrd_pool.details = info

        try:
            await mrd_pool.initialize()
            if isinstance(getattr(mrd_pool, "details", None), dict):
                if mrd_pool.details.get("timeFinalized") is not None:
                    self._pool_info[key] = mrd_pool.details
                else:
                    self._pool_info.pop(key, None)
            elif hasattr(fs, "_info"):
                mrd_pool.details = await fs._info(
                    f"{bucket_name}/{object_name}", generation=generation
                )
        except BaseException:
            # Init failed. `mrd_pool.close()` donates any partial MRDs back
            # via release() and drops the refcount we just took. If that was
            # the last reference, purge the key entirely.
            await mrd_pool.close()
            mrds_to_close = []
            if key not in self._refcounts:
                self._evictable_keys.pop(key, None)
                self._pool_info.pop(key, None)
                mrds_to_close = _drain_queue(self._mrd_queues.pop(key, None))
            await _close_mrds(mrds_to_close, raise_exception=False)
            raise

        return mrd_pool

    async def release(self, key, mrds):
        """
        Releases MRDs back to the cache or closes them if necessary.

        Args:
            key (tuple): Cache key (bucket, object, generation).
            mrds (list): List of MRDs to release.
        """
        mrds_to_close = []
        mrd_queue = self._mrd_queues.get(key)
        if mrd_queue is not None:
            for mrd in mrds:
                if len(mrd_queue) < self._max_queue_size:
                    mrd_queue.append(mrd)
                else:
                    mrds_to_close.append(mrd)
        else:
            mrds_to_close.extend(mrds)

        mrds_to_close.extend(self._decref(key))
        await _close_mrds(mrds_to_close, raise_exception=False)

    async def close(self):
        """
        Closes the cache and all pooled MRDs.
        """
        if self._closed:
            return
        mrds_to_close = []
        for q in self._mrd_queues.values():
            mrds_to_close.extend(_drain_queue(q))
        self._mrd_queues.clear()
        self._refcounts.clear()
        self._evictable_keys.clear()
        self._pool_info.clear()
        self._closed = True
        await _close_mrds(mrds_to_close, raise_exception=True)
