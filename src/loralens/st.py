"""Read and write .safetensors files with numpy only.

The format is an 8-byte little-endian header length, a JSON header that maps
each tensor name to its dtype, shape and byte range, and then the raw data.
Tensors are read lazily from a memory map, and only the ones that change are
ever converted to numpy arrays.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import tempfile
from dataclasses import dataclass

import numpy as np

__all__ = ["FLOAT_DTYPES", "SafetensorsError", "Tensor", "encode", "load", "save"]

_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}
_NUMPY_FLOAT = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}
FLOAT_DTYPES = frozenset({"F16", "BF16", "F32", "F64"})
_MAX_HEADER = 100 * 1024 * 1024


class SafetensorsError(ValueError):
    """The file is not a valid safetensors file, or a tensor can't be converted."""


@dataclass
class Tensor:
    dtype: str
    shape: tuple[int, ...]
    data: bytes | memoryview

    @property
    def nbytes(self) -> int:
        return len(self.data)

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def to_float32(self) -> np.ndarray:
        if self.dtype == "BF16":
            bits = np.frombuffer(self.data, dtype="<u2").astype(np.uint32) << 16
            return bits.view(np.float32).reshape(self.shape)
        if self.dtype in _NUMPY_FLOAT:
            values = np.frombuffer(self.data, dtype=_NUMPY_FLOAT[self.dtype])
            return values.astype(np.float32).reshape(self.shape)
        raise SafetensorsError(f"dtype {self.dtype} is not a floating-point type")

    @classmethod
    def from_array(cls, array: np.ndarray, dtype: str) -> "Tensor":
        shape = tuple(int(n) for n in np.shape(array))
        return cls(dtype, shape, encode(array, dtype))


def encode(array: np.ndarray, dtype: str) -> bytes:
    """Encode a float array as `dtype` bytes (BF16 rounds to nearest even)."""
    values = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
    if dtype == "BF16":
        bits = values.view(np.uint32).astype(np.uint64)
        rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
        rounded[np.isnan(values)] = 0x7FC0
        return rounded.astype("<u2").tobytes()
    if dtype in _NUMPY_FLOAT:
        return values.astype(_NUMPY_FLOAT[dtype]).tobytes()
    raise SafetensorsError(f"can't write dtype {dtype}")


def load(path: str) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Return the tensors (in file order) and the string metadata of a file."""
    size = os.path.getsize(path)
    if size < 8:
        raise SafetensorsError(f"{path}: too small to be a safetensors file")
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        if header_len > _MAX_HEADER or 8 + header_len > size:
            raise SafetensorsError(f"{path}: invalid header length; is this a safetensors file?")
        try:
            header = json.loads(f.read(header_len).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SafetensorsError(f"{path}: the header is not valid JSON ({e})") from None
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path}: the header is not a JSON object")

    metadata = header.pop("__metadata__", None) or {}
    if not isinstance(metadata, dict) or not all(isinstance(v, str) for v in metadata.values()):
        raise SafetensorsError(f"{path}: __metadata__ must map names to strings")

    data = memoryview(mapped)[8 + header_len:]
    tensors: dict[str, Tensor] = {}
    for name, info in header.items():
        try:
            dtype = info["dtype"]
            shape = tuple(int(n) for n in info["shape"])
            start, end = (int(n) for n in info["data_offsets"])
        except (KeyError, TypeError, ValueError):
            raise SafetensorsError(f"{path}: bad header entry for {name!r}") from None
        if dtype not in _ITEMSIZE:
            raise SafetensorsError(f"{path}: {name}: unknown dtype {dtype}")
        expected = int(np.prod(shape, dtype=np.int64)) * _ITEMSIZE[dtype]
        if not 0 <= start <= end <= len(data) or end - start != expected:
            raise SafetensorsError(f"{path}: {name}: data offsets don't match its shape and dtype")
        tensors[name] = Tensor(dtype, shape, data[start:end])
    return tensors, dict(metadata)


def save(path: str, tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> None:
    """Write tensors in the given order. The file is replaced atomically."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    for name, tensor in tensors.items():
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".loralens-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            for tensor in tensors.values():
                f.write(tensor.data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
