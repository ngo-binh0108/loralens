import json
import struct

import numpy as np
import pytest

from loralens.st import SafetensorsError, Tensor, encode, load, save


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16", "F64"])
def test_round_trip(tmp_path, dtype):
    values = np.array([[0.0, 1.0, -2.5], [3.25, -0.125, 1024.0]], dtype=np.float32)
    path = tmp_path / "a.safetensors"
    save(str(path), {"w": Tensor.from_array(values, dtype), "s": Tensor.from_array(np.float32(4), dtype)},
         {"note": "hi"})
    tensors, metadata = load(str(path))
    assert list(tensors) == ["w", "s"]
    assert metadata == {"note": "hi"}
    assert tensors["w"].dtype == dtype and tensors["w"].shape == (2, 3)
    np.testing.assert_array_equal(tensors["w"].to_float32(), values)
    assert tensors["s"].shape == () and float(tensors["s"].to_float32()) == 4.0


def test_header_is_padded_to_eight_bytes(tmp_path):
    path = tmp_path / "a.safetensors"
    save(str(path), {"x": Tensor.from_array(np.ones(3, np.float32), "F32")})
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    assert n % 8 == 0
    assert json.loads(raw[8:8 + n])["x"]["data_offsets"] == [0, 12]


def test_bf16_rounds_to_nearest_even():
    halfway_down = np.uint32(0x3F808000).view(np.float32)  # tie, low bit 0 -> stays 0x3F80
    halfway_up = np.uint32(0x3F818000).view(np.float32)    # tie, low bit 1 -> 0x3F82
    above = np.uint32(0x3F808001).view(np.float32)
    bits = np.frombuffer(encode(np.array([1.0, halfway_down, halfway_up, above, np.nan]), "BF16"), "<u2")
    assert list(bits) == [0x3F80, 0x3F80, 0x3F82, 0x3F81, 0x7FC0]


def test_rejects_files_that_are_not_safetensors(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"abc")
    with pytest.raises(SafetensorsError, match="too small"):
        load(str(path))
    path.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(SafetensorsError, match="header length"):
        load(str(path))
    path.write_bytes(struct.pack("<Q", 4) + b"nope")
    with pytest.raises(SafetensorsError, match="not valid JSON"):
        load(str(path))


def test_rejects_offsets_that_dont_match_the_shape(tmp_path):
    header = json.dumps({"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    with pytest.raises(SafetensorsError, match="offsets"):
        load(str(path))


def test_non_float_dtypes_are_kept_but_not_converted(tmp_path):
    path = tmp_path / "a.safetensors"
    save(str(path), {"ids": Tensor("I64", (2,), struct.pack("<qq", 1, 2))})
    tensors, _ = load(str(path))
    assert bytes(tensors["ids"].data) == struct.pack("<qq", 1, 2)
    with pytest.raises(SafetensorsError, match="not a floating-point"):
        tensors["ids"].to_float32()
