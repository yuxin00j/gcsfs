"""Zero-copy deserializer for ``BidiReadObjectResponse`` messages.

The generated gRPC stub parses every response with proto-plus over upb, which
copies each data chunk twice (into the upb arena, then into a Python ``bytes``
for ``checksummed_data.content``) on the event-loop thread. Data responses are
structurally tiny apart from the chunk itself, so this module walks the wire
format directly and exposes ``content`` as a ``memoryview`` of the raw message.
Responses carrying object metadata (the first message of a stream opened without
a read handle) or anything unexpected fall back to the generated parser.
"""

import os

import google_crc32c
from google.cloud import _storage_v2 as storage_v2

_WT_VARINT, _WT_I64, _WT_LEN, _WT_I32 = 0, 1, 2, 5


class _ReadRange:
    __slots__ = ("read_offset", "read_length", "read_id")

    def __init__(self):
        self.read_offset = 0
        self.read_length = 0
        self.read_id = 0


class _ChecksummedData:
    __slots__ = ("content", "crc32c", "_has_crc32c")

    def __init__(self):
        self.content = b""
        self.crc32c = 0
        self._has_crc32c = False

    def HasField(self, name):
        if name == "crc32c":
            return self._has_crc32c
        raise ValueError(name)


class _ObjectRangeData:
    __slots__ = ("checksummed_data", "read_range", "range_end")

    def __init__(self):
        self.checksummed_data = _ChecksummedData()
        self.read_range = None
        self.range_end = False

    def HasField(self, name):
        if name == "read_range":
            return self.read_range is not None
        if name == "checksummed_data":
            return True
        raise ValueError(name)


class FastBidiReadObjectResponse:
    """Duck-typed stand-in for ``BidiReadObjectResponse`` without metadata."""

    __slots__ = ("object_data_ranges", "read_handle", "metadata")

    def __init__(self, ranges, read_handle):
        self.object_data_ranges = ranges
        self.read_handle = read_handle
        self.metadata = None

    @property
    def _pb(self):
        return self

    def HasField(self, name):
        if name == "read_handle":
            return self.read_handle is not None
        if name == "metadata":
            return False
        raise ValueError(name)


class _Malformed(ValueError):
    """Raised when the wire bytes do not look like a well-formed message."""


def _varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, i
        shift += 7
        if shift > 63:
            raise _Malformed("varint longer than 10 bytes")


def _length_prefixed(buf, i, end):
    """Return ``(start, stop)`` of a length-delimited field body at ``i``."""
    n, i = _varint(buf, i)
    stop = i + n
    if stop > end:
        raise _Malformed("field runs past its enclosing message")
    return i, stop


def _skip(buf, i, wt, end):
    if wt == _WT_VARINT:
        i = _varint(buf, i)[1]
    elif wt == _WT_LEN:
        i = _length_prefixed(buf, i, end)[1]
    elif wt == _WT_I64:
        i += 8
    elif wt == _WT_I32:
        i += 4
    else:
        raise _Malformed(f"unsupported wire type {wt}")
    if i > end:
        raise _Malformed("field runs past its enclosing message")
    return i


def _parse_read_range(buf, i, end):
    rr = _ReadRange()
    while i < end:
        tag, i = _varint(buf, i)
        f, wt = tag >> 3, tag & 7
        if wt == _WT_VARINT:
            v, i = _varint(buf, i)
            if f == 1:
                rr.read_offset = v
            elif f == 2:
                rr.read_length = v
            elif f == 3:
                rr.read_id = v
        else:
            i = _skip(buf, i, wt, end)
    if i != end:
        raise _Malformed("read_range overruns its length")
    return rr


def _parse_checksummed(buf, mv, i, end):
    cd = _ChecksummedData()
    while i < end:
        tag, i = _varint(buf, i)
        f, wt = tag >> 3, tag & 7
        if f == 1 and wt == _WT_LEN:
            start, i = _length_prefixed(buf, i, end)
            cd.content = mv[start:i]
        elif f == 2 and wt == _WT_I32:
            if i + 4 > end:
                raise _Malformed("crc32c runs past its enclosing message")
            cd.crc32c = int.from_bytes(buf[i : i + 4], "little")
            cd._has_crc32c = True
            i += 4
        else:
            i = _skip(buf, i, wt, end)
    if i != end:
        raise _Malformed("checksummed_data overruns its length")
    return cd


def _parse_range_data(buf, mv, i, end):
    rd = _ObjectRangeData()
    while i < end:
        tag, i = _varint(buf, i)
        f, wt = tag >> 3, tag & 7
        if wt == _WT_LEN:
            start, i = _length_prefixed(buf, i, end)
            if f == 1:
                rd.checksummed_data = _parse_checksummed(buf, mv, start, i)
            elif f == 2:
                rd.read_range = _parse_read_range(buf, start, i)
        elif f == 3 and wt == _WT_VARINT:
            v, i = _varint(buf, i)
            rd.range_end = bool(v)
        else:
            i = _skip(buf, i, wt, end)
    if i != end:
        raise _Malformed("object_data_ranges overruns its length")
    return rd


def _parse_handle(buf, i, end):
    handle = b""
    while i < end:
        tag, i = _varint(buf, i)
        f, wt = tag >> 3, tag & 7
        if f == 1 and wt == _WT_LEN:
            start, i = _length_prefixed(buf, i, end)
            handle = bytes(buf[start:i])
        else:
            i = _skip(buf, i, wt, end)
    if i != end:
        raise _Malformed("read_handle overruns its length")
    # A real message: the SDK stores the handle and passes it back in the
    # BidiReadObjectSpec of the next open, which only accepts the proto type.
    return storage_v2.BidiReadHandle(handle=handle)


_slow_deserialize = storage_v2.BidiReadObjectResponse.deserialize


def deserialize(buf):
    """Parse a serialized ``BidiReadObjectResponse``; chunk payloads alias ``buf``.

    Anything that is not a plain data response (object metadata present, or
    bytes that do not parse cleanly) is handed to the generated parser, which
    either produces the real message or raises its usual ``DecodeError``.
    """
    try:
        mv = memoryview(buf)
        n = len(buf)
        i = 0
        ranges = []
        handle = None
        while i < n:
            tag, i = _varint(buf, i)
            f, wt = tag >> 3, tag & 7
            if wt == _WT_LEN:
                start, end = _length_prefixed(buf, i, n)
                if f == 6:
                    ranges.append(_parse_range_data(buf, mv, start, end))
                elif f == 7:
                    handle = _parse_handle(buf, start, end)
                elif f == 4:
                    return _slow_deserialize(buf)
                i = end
            else:
                i = _skip(buf, i, wt, n)
        if i != n:
            return _slow_deserialize(buf)
        return FastBidiReadObjectResponse(ranges, handle)
    except Exception:
        return _slow_deserialize(buf)


_DISABLED_VALUES = ("0", "false", "no", "off")


def is_supported():
    """Whether the zero-copy parser may be installed in this process.

    The parser hands chunk payloads to the SDK as ``memoryview`` objects and the
    SDK's checksum verification feeds them to ``google_crc32c``; releases of
    google-crc32c that only accept ``bytes`` there would make every verified
    read fail, so the generated parser is kept in that case. Setting
    ``GCSFS_ZONAL_FAST_READ=0`` forces the generated parser as well.
    """
    if os.environ.get("GCSFS_ZONAL_FAST_READ", "").strip().lower() in _DISABLED_VALUES:
        return False
    try:
        google_crc32c.value(memoryview(b"\0"))
    except Exception:
        return False
    return True


def install(grpc_client):
    """Point ``grpc_client``'s BidiReadObject stub at :func:`deserialize`.

    ``_AsyncReadObjectStream`` looks the stub up through
    ``transport._wrapped_methods[transport.bidi_read_object]``, so both the stub
    cache and the wrapped-method table are rebound. Returns False (leaving the
    client untouched) if the transport does not look like the generated
    grpc_asyncio one.
    """
    try:
        import functools

        from google.api_core import grpc_helpers_async
        from google.api_core.gapic_v1.method_async import _GapicCallable

        transport = grpc_client._client._transport
        old = transport.bidi_read_object
        if getattr(old, "_gcsfs_fast_deserialize", False):
            return True
        wrapped = transport._wrapped_methods[old]
        # api-core < 2.34 keeps the metadata list; newer versions keep the
        # already-extracted default tuple. Either rebuilds the same callable.
        metadata = getattr(wrapped, "_metadata", None)
        if metadata is None:
            metadata = list(getattr(wrapped, "_default_metadata", ()))
        new = transport._logged_channel.stream_stream(
            "/google.storage.v2.Storage/BidiReadObject",
            request_serializer=storage_v2.BidiReadObjectRequest.serialize,
            response_deserializer=deserialize,
        )
        func = grpc_helpers_async.wrap_errors(new)
        new_wrapped = functools.wraps(func)(
            _GapicCallable(
                func,
                wrapped._retry,
                wrapped._timeout,
                wrapped._compression,
                metadata=metadata,
            )
        )
        new._gcsfs_fast_deserialize = True
        transport._stubs["bidi_read_object"] = new
        transport._wrapped_methods[new] = new_wrapped
        return True
    except Exception:
        return False
