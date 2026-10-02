"""Process-safe readers for contiguous HDF5 OpenFly pre-cache features.

The HDF5 file is deliberately a very small storage abstraction around the
existing global row index.  It does not own sampling, episode logic, or model
semantics.  Every DataLoader process opens the finalized file lazily in its
own PID, while the dataset continues to use features[index].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


H5_PRE_CACHE_FILENAMES = ("pre_cache.h5", "features.h5")


def _import_h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - environment gate
        raise RuntimeError(
            "HDF5 pre-cache support requires h5py in the training environment"
        ) from exc
    return h5py


def find_h5_pre_cache(cache_path: str | Path) -> Path | None:
    """Return the preferred finalized HDF5 feature file for a cache directory."""

    path = Path(cache_path).expanduser().resolve()
    if path.is_file():
        return path if path.suffix.lower() in {".h5", ".hdf5"} else None
    if not path.is_dir():
        return None
    for filename in H5_PRE_CACHE_FILENAMES:
        candidate = path / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _open_read_only(h5py: Any, path: Path) -> Any:
    """Open a finalized read-only file without retaining a forked handle."""

    try:
        return h5py.File(str(path), "r", locking=False)
    except TypeError:  # h5py versions before the locking keyword
        return h5py.File(str(path), "r")


def _complete_attr_is_true(value: Any) -> bool:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


class H5PreCache:
    """A lazy, per-process reader for features in contiguous HDF5.

    The constructor only opens the file briefly to validate shape, dtype and
    layout.  The persistent handle is created on the first __getitem__ in
    each DataLoader worker.  Array-index reads intentionally preserve order
    and duplicates by reading rows one at a time; this matches NumPy indexing
    semantics and avoids h5py restrictions on unsorted fancy indices.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_key: str = "features",
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: np.dtype[Any] | str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.dataset_key = str(dataset_key)
        self._pid: int | None = None
        self._file: Any | None = None
        self._data: Any | None = None
        self._shape: tuple[int, ...]
        self._dtype: np.dtype[Any]

        if not self.path.is_file():
            raise FileNotFoundError(f"HDF5 pre-cache is missing: {self.path}")

        h5py = _import_h5py()
        with _open_read_only(h5py, self.path) as handle:
            self._validate_handle(handle)
            data = handle[self.dataset_key]
            self._shape = tuple(int(value) for value in data.shape)
            self._dtype = np.dtype(data.dtype)
            if expected_shape is not None and self._shape != tuple(expected_shape):
                raise ValueError(
                    f"HDF5 pre-cache shape mismatch: {self._shape} != "
                    f"{tuple(expected_shape)}"
                )
            if expected_dtype is not None:
                expected = np.dtype(expected_dtype)
                if self._dtype != expected:
                    raise ValueError(
                        f"HDF5 pre-cache dtype mismatch: {self._dtype} != {expected}"
                    )

    def _validate_handle(self, handle: Any) -> None:
        if self.dataset_key not in handle:
            raise ValueError(
                f"HDF5 pre-cache has no dataset /{self.dataset_key}: {self.path}"
            )
        data = handle[self.dataset_key]
        if data.chunks is not None or data.compression is not None:
            raise ValueError(
                "OpenFly pre-cache must use contiguous, uncompressed HDF5; "
                f"got chunks={data.chunks!r}, compression={data.compression!r}"
            )
        if "complete" in handle.attrs and not _complete_attr_is_true(
            handle.attrs["complete"]
        ):
            raise ValueError(f"HDF5 pre-cache is not finalized: {self.path}")

    def _ensure_open(self) -> Any:
        pid = os.getpid()
        if self._file is not None and self._data is not None and self._pid == pid:
            return self._data

        self.close()
        h5py = _import_h5py()
        handle = _open_read_only(h5py, self.path)
        try:
            self._validate_handle(handle)
            data = handle[self.dataset_key]
            shape = tuple(int(value) for value in data.shape)
            dtype = np.dtype(data.dtype)
            if shape != self._shape or dtype != self._dtype:
                raise ValueError(
                    "HDF5 pre-cache changed after validation: "
                    f"shape={shape}, dtype={dtype}"
                )
        except BaseException:
            handle.close()
            raise
        self._pid = pid
        self._file = handle
        self._data = data
        return data

    def __len__(self) -> int:
        return self._shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    def __getitem__(self, index: Any) -> np.ndarray:
        data = self._ensure_open()
        if isinstance(index, (int, np.integer)):
            row = int(index)
            if row < 0:
                row += self._shape[0]
            if row < 0 or row >= self._shape[0]:
                raise IndexError(row)
            return np.asarray(data[row])
        if isinstance(index, slice):
            return np.asarray(data[index])

        indices = np.asarray(index)
        if indices.ndim == 0:
            return self.__getitem__(int(indices))
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise IndexError("HDF5 pre-cache indexing requires integer rows or a 1-D slice")
        if indices.size == 0:
            return np.empty((0, *self._shape[1:]), dtype=self._dtype)

        rows: list[np.ndarray] = []
        for raw_index in indices.tolist():
            row = int(raw_index)
            if row < 0:
                row += self._shape[0]
            if row < 0 or row >= self._shape[0]:
                raise IndexError(row)
            rows.append(np.asarray(data[row]))
        return np.stack(rows, axis=0)

    def close(self) -> None:
        handle = self._file
        self._file = None
        self._data = None
        self._pid = None
        if handle is not None:
            handle.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_data"] = None
        state["_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
