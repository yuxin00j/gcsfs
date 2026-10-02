"""Tests for the zero-copy ``BidiReadObjectResponse`` parser."""

from unittest import mock

import google_crc32c
import pytest
from google.auth.credentials import AnonymousCredentials
from google.cloud import _storage_v2 as storage_v2
from google.protobuf.message import DecodeError

from gcsfs import _fast_bidi_read
from gcsfs.extended_gcsfs import ExtendedGcsFileSystem

PAYLOAD = bytes(range(256)) * 8


def _serialize(**fields):
    return storage_v2.BidiReadObjectResponse.serialize(
        storage_v2.BidiReadObjectResponse(**fields)
    )


def _data_response(content=PAYLOAD, crc32c=None, handle=None):
    checksummed = {"content": content}
    if crc32c is not None:
        checksummed["crc32c"] = crc32c
    fields = {
        "object_data_ranges": [
            storage_v2.ObjectRangeData(
                checksummed_data=storage_v2.ChecksummedData(**checksummed),
                read_range=storage_v2.ReadRange(
                    read_offset=4096, read_length=len(content), read_id=7
                ),
                range_end=True,
            )
        ]
    }
    if handle is not None:
        fields["read_handle"] = storage_v2.BidiReadHandle(handle=handle)
    return _serialize(**fields)


def _anonymous_storage_client():
    return storage_v2.StorageAsyncClient(credentials=AnonymousCredentials())


# --- deserialize -----------------------------------------------------------


def test_deserialize_data_response_round_trip():
    crc = google_crc32c.value(PAYLOAD)
    response = _fast_bidi_read.deserialize(_data_response(crc32c=crc))

    assert isinstance(response, _fast_bidi_read.FastBidiReadObjectResponse)
    assert not response.HasField("metadata")
    assert not response.HasField("read_handle")
    [range_data] = response.object_data_ranges
    assert range_data.HasField("read_range")
    assert range_data.read_range.read_offset == 4096
    assert range_data.read_range.read_length == len(PAYLOAD)
    assert range_data.read_range.read_id == 7
    assert range_data.range_end is True
    checksummed = range_data.checksummed_data
    assert isinstance(checksummed.content, memoryview)
    assert bytes(checksummed.content) == PAYLOAD
    assert checksummed.HasField("crc32c")
    assert checksummed.crc32c == crc


def test_deserialize_content_is_a_view_of_the_wire_buffer():
    buf = _data_response()
    content = _fast_bidi_read.deserialize(buf).object_data_ranges[0]
    assert content.checksummed_data.content.obj is buf


def test_deserialize_without_crc32c_reports_no_field():
    response = _fast_bidi_read.deserialize(_data_response())
    assert response.object_data_ranges[0].checksummed_data.HasField("crc32c") is False


def test_deserialize_metadata_response_uses_generated_parser():
    metadata = storage_v2.Object(name="o", bucket="projects/_/buckets/b", size=10)
    response = _fast_bidi_read.deserialize(_serialize(metadata=metadata))
    assert isinstance(response, storage_v2.BidiReadObjectResponse)
    assert response.metadata.size == 10


def test_deserialize_read_handle_is_a_bidi_read_handle():
    response = _fast_bidi_read.deserialize(_data_response(handle=b"opaque"))

    assert response.HasField("read_handle")
    assert isinstance(response.read_handle, storage_v2.BidiReadHandle)
    assert response.read_handle.handle == b"opaque"
    # The SDK feeds a captured handle straight back into the next open.
    spec = storage_v2.BidiReadObjectSpec(read_handle=response.read_handle)
    assert spec.read_handle.handle == b"opaque"


def test_deserialize_rejects_inner_length_past_enclosing_message():
    # object_data_ranges (field 6, 6 bytes) whose checksummed_data claims 8
    # bytes, followed by a well-formed read_handle. The generated parser
    # rejects this; the fast path must not hand back truncated content.
    buf = bytes([0x32, 0x06, 0x0A, 0x08]) + b"abcd" + bytes([0x3A, 0x02, 0x0A, 0x00])
    with pytest.raises(DecodeError):
        storage_v2.BidiReadObjectResponse.deserialize(buf)
    with pytest.raises(DecodeError):
        _fast_bidi_read.deserialize(buf)


def test_deserialize_rejects_overlong_varint():
    # A length encoded in 11 varint bytes is not valid protobuf.
    buf = bytes([0x32]) + b"\x80" * 10 + b"\x00"
    with pytest.raises(DecodeError):
        storage_v2.BidiReadObjectResponse.deserialize(buf)
    with pytest.raises(DecodeError):
        _fast_bidi_read.deserialize(buf)


def test_deserialize_rejects_truncated_message():
    buf = bytes([0x32, 0x10, 0x0A, 0x00])
    with pytest.raises(DecodeError):
        _fast_bidi_read.deserialize(buf)


# --- is_supported ----------------------------------------------------------


def test_is_supported_requires_crc32c_to_accept_memoryview(monkeypatch):
    def bytes_only(data):
        if type(data) is not bytes:
            raise TypeError("argument 1 must be read-only bytes-like object")
        return 0

    monkeypatch.setattr(google_crc32c, "value", bytes_only)
    assert _fast_bidi_read.is_supported() is False


def test_is_supported_when_crc32c_accepts_memoryview(monkeypatch):
    monkeypatch.setattr(google_crc32c, "value", lambda data: len(data))
    assert _fast_bidi_read.is_supported() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_is_supported_env_kill_switch(monkeypatch, value):
    monkeypatch.setattr(google_crc32c, "value", lambda data: len(data))
    monkeypatch.setenv("GCSFS_ZONAL_FAST_READ", value)
    assert _fast_bidi_read.is_supported() is False


# --- install ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_rebinds_stub_and_wrapped_method():
    # Building the grpc.aio channel needs a running event loop.
    client = _anonymous_storage_client()
    transport = client._client._transport
    before = transport._wrapped_methods[transport.bidi_read_object]

    assert _fast_bidi_read.install(client) is True

    stub = transport.bidi_read_object
    assert stub._gcsfs_fast_deserialize is True
    # _AsyncReadObjectStream resolves the rpc through this table.
    after = transport._wrapped_methods[stub]
    assert after is not before
    assert after._retry == before._retry
    assert after._timeout == before._timeout
    assert after._compression == before._compression

    # Installing again is a no-op.
    assert _fast_bidi_read.install(client) is True
    assert transport.bidi_read_object is stub


def test_install_leaves_unknown_clients_alone():
    assert _fast_bidi_read.install(object()) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [True, False])
async def test_get_grpc_client_installs_fast_parser_only_when_supported(supported):
    fs = ExtendedGcsFileSystem(project="p", token="anon", skip_instance_cache=True)
    grpc_client = mock.Mock(grpc_client=_anonymous_storage_client())
    with mock.patch(
        "gcsfs.extended_gcsfs.AsyncGrpcClient", return_value=grpc_client
    ), mock.patch.object(_fast_bidi_read, "is_supported", return_value=supported):
        await fs._get_grpc_client()

    transport = grpc_client.grpc_client._client._transport
    installed = getattr(transport.bidi_read_object, "_gcsfs_fast_deserialize", False)
    assert installed is supported
