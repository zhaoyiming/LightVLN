"""PyTorch reader for the MiniVLA single-current-frame OpenFly TFDS export.

The TFDS builder emits ordinary TFRecord/tf.train.Example files so the
OpenFly reference stack can inspect them. Training MiniVLA does not import
TensorFlow: that would add a second CUDA runtime to the PyTorch process. This
module implements the stable subset of TFRecord and Example needed by the
flat TFDS schema and returns the same samples as the direct Parquet adapter.
"""

from __future__ import annotations

import io
import json
import struct
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from aerial_openvla.openfly.action_codec import OPENFLY_VALID_ACTION_IDS


OPENFLY_S0_TFDS_INDEX_FORMAT = "openfly_minivla_s0_current_tfds_index_v1"
_INDEX_DIRECTORY = "minivla_s0_index"
_MAX_PROTOBUF_VARINT_BYTES = 10


def _make_crc32c_table() -> tuple[int, ...]:
    table: list[int] = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value & 0xFFFFFFFF)
    return tuple(table)


_CRC32C_TABLE = _make_crc32c_table()


def crc32c(value: bytes) -> int:
    """Return the Castagnoli CRC used by TFRecord framing."""

    checksum = 0xFFFFFFFF
    for byte in value:
        checksum = _CRC32C_TABLE[(checksum ^ byte) & 0xFF] ^ (checksum >> 8)
    return (~checksum) & 0xFFFFFFFF


def masked_crc32c(value: bytes) -> int:
    checksum = crc32c(value)
    return (((checksum >> 15) | (checksum << 17)) & 0xFFFFFFFF) + 0xA282EAD8 & 0xFFFFFFFF


def iter_tfrecord_records(
    path: str | Path,
    *,
    verify_crc: bool = True,
    digest: Any | None = None,
) -> Iterator[tuple[int, int, bytes]]:
    """Yield record offset, payload length, and serialized Example bytes.

    When digest is supplied, it receives every on-disk framing byte while the
    shard is scanned. Index construction therefore gets a SHA-256 without a
    second pass over the generated TFDS files.
    """

    source = Path(path)
    with source.open("rb") as handle:
        offset = 0
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord length header in {source} at byte {offset}")
            if digest is not None:
                digest.update(header)
            length = struct.unpack("<Q", header[:8])[0]
            stored_length_crc = struct.unpack("<I", header[8:])[0]
            if verify_crc and stored_length_crc != masked_crc32c(header[:8]):
                raise ValueError(f"TFRecord length CRC mismatch in {source} at byte {offset}")
            payload = handle.read(length)
            footer = handle.read(4)
            if len(payload) != length or len(footer) != 4:
                raise ValueError(f"Truncated TFRecord payload in {source} at byte {offset}")
            if digest is not None:
                digest.update(payload)
                digest.update(footer)
            stored_payload_crc = struct.unpack("<I", footer)[0]
            if verify_crc and stored_payload_crc != masked_crc32c(payload):
                raise ValueError(f"TFRecord payload CRC mismatch in {source} at byte {offset}")
            yield offset, int(length), payload
            offset += 12 + int(length) + 4


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 7 * _MAX_PROTOBUF_VARINT_BYTES, 7):
        if offset >= len(payload):
            raise ValueError("Truncated protobuf varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
    raise ValueError("Protobuf varint is longer than 10 bytes")


def _iter_protobuf_fields(payload: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield basic protobuf wire fields without TensorFlow protobuf imports."""

    offset = 0
    while offset < len(payload):
        key, offset = _read_varint(payload, offset)
        field_number = key >> 3
        wire_type = key & 0x07
        if field_number <= 0:
            raise ValueError("Invalid protobuf field number")
        if wire_type == 0:
            value, offset = _read_varint(payload, offset)
        elif wire_type == 1:
            if offset + 8 > len(payload):
                raise ValueError("Truncated fixed64 protobuf field")
            value = payload[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = _read_varint(payload, offset)
            if offset + length > len(payload):
                raise ValueError("Truncated length-delimited protobuf field")
            value = payload[offset : offset + length]
            offset += length
        elif wire_type == 5:
            if offset + 4 > len(payload):
                raise ValueError("Truncated fixed32 protobuf field")
            value = payload[offset : offset + 4]
            offset += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        yield field_number, wire_type, value


def _decode_bytes_list(payload: bytes) -> list[bytes]:
    values: list[bytes] = []
    for field_number, wire_type, value in _iter_protobuf_fields(payload):
        if field_number == 1 and wire_type == 2:
            if not isinstance(value, bytes):
                raise RuntimeError("BytesList payload is not bytes")
            values.append(value)
    return values


def _decode_float_list(payload: bytes) -> list[float]:
    values: list[float] = []
    for field_number, wire_type, value in _iter_protobuf_fields(payload):
        if field_number != 1:
            continue
        if wire_type == 2:
            if not isinstance(value, bytes) or len(value) % 4:
                raise ValueError("Invalid packed FloatList payload")
            values.extend(struct.unpack(f"<{len(value) // 4}f", value))
        elif wire_type == 5:
            if not isinstance(value, bytes):
                raise RuntimeError("FloatList fixed32 payload is not bytes")
            values.append(struct.unpack("<f", value)[0])
        else:
            raise ValueError("Unsupported FloatList wire type")
    return values


def _decode_int64_list(payload: bytes) -> list[int]:
    values: list[int] = []
    for field_number, wire_type, value in _iter_protobuf_fields(payload):
        if field_number != 1:
            continue
        if wire_type == 2:
            if not isinstance(value, bytes):
                raise RuntimeError("Int64List payload is not bytes")
            offset = 0
            while offset < len(value):
                decoded, offset = _read_varint(value, offset)
                if decoded >= 1 << 63:
                    decoded -= 1 << 64
                values.append(decoded)
        elif wire_type == 0:
            if not isinstance(value, int):
                raise RuntimeError("Int64List varint payload is not an integer")
            if value >= 1 << 63:
                value -= 1 << 64
            values.append(value)
        else:
            raise ValueError("Unsupported Int64List wire type")
    return values


def _decode_feature(payload: bytes) -> list[bytes] | list[float] | list[int]:
    decoded: list[bytes] | list[float] | list[int] | None = None
    for field_number, wire_type, value in _iter_protobuf_fields(payload):
        if wire_type != 2 or not isinstance(value, bytes):
            raise ValueError("Invalid tf.train.Feature encoding")
        if field_number == 1:
            candidate = _decode_bytes_list(value)
        elif field_number == 2:
            candidate = _decode_float_list(value)
        elif field_number == 3:
            candidate = _decode_int64_list(value)
        else:
            continue
        if decoded is not None:
            raise ValueError("tf.train.Feature contains multiple value kinds")
        decoded = candidate
    if decoded is None:
        raise ValueError("tf.train.Feature has no supported value kind")
    return decoded


def decode_tf_example(payload: bytes) -> dict[str, list[bytes] | list[float] | list[int]]:
    """Decode the flat feature dictionary written by TFDS TFRecord adapter."""

    features_blob: bytes | None = None
    for field_number, wire_type, value in _iter_protobuf_fields(payload):
        if field_number == 1 and wire_type == 2 and isinstance(value, bytes):
            features_blob = value
            break
    if features_blob is None:
        raise ValueError("Serialized TFRecord is not a tf.train.Example with Features")

    features: dict[str, list[bytes] | list[float] | list[int]] = {}
    for field_number, wire_type, entry_blob in _iter_protobuf_fields(features_blob):
        if field_number != 1 or wire_type != 2 or not isinstance(entry_blob, bytes):
            continue
        key: str | None = None
        feature_blob: bytes | None = None
        for entry_field, entry_wire, entry_value in _iter_protobuf_fields(entry_blob):
            if entry_field == 1 and entry_wire == 2 and isinstance(entry_value, bytes):
                key = entry_value.decode("utf-8")
            elif entry_field == 2 and entry_wire == 2 and isinstance(entry_value, bytes):
                feature_blob = entry_value
        if key is None or feature_blob is None:
            raise ValueError("Malformed Features map entry")
        if key in features:
            raise ValueError(f"Duplicate TFDS feature key: {key}")
        features[key] = _decode_feature(feature_blob)
    if not features:
        raise ValueError("TFDS example has no decodable features")
    return features


def _bytes_feature(
    features: Mapping[str, list[bytes] | list[float] | list[int]], key: str
) -> list[bytes]:
    value = features.get(key)
    if value is None or any(not isinstance(item, bytes) for item in value):
        raise ValueError(f"TFDS example has no bytes feature {key!r}")
    return list(value)


def _int_feature(
    features: Mapping[str, list[bytes] | list[float] | list[int]], key: str
) -> list[int]:
    value = features.get(key)
    if value is None or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"TFDS example has no int64 feature {key!r}")
    return list(value)


@dataclass(frozen=True, slots=True)
class _DecodedEpisode:
    actions: tuple[int, ...]
    images: tuple[bytes, ...]
    instructions: tuple[str, ...]
    episode_id: str
    source_image_ids: tuple[str, ...]


class OpenFlyCurrentFrameTFDSDataset(Dataset[dict[str, Any]]):
    """Map-style S0 dataset backed by one current-RGB TFDS episode record."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        action_policy: str = "openfly_10",
        num_frames: int = 1,
        frame_history_stride: int = 1,
        action_chunk_horizon: int = 1,
        max_episodes: int = 0,
        cache_episodes: int = 32,
        verify_record_crc: bool = False,
    ) -> None:
        if action_policy != "openfly_10":
            raise ValueError("The S0 TFDS export preserves the full OpenFly 0..9 action space")
        if num_frames != 1:
            raise ValueError("OpenFly S0 TFDS contains exactly one current RGB frame per sample")
        if frame_history_stride != 1:
            raise ValueError("frame_history_stride must be 1 for a single-current-frame dataset")
        if action_chunk_horizon != 1:
            raise ValueError("OpenFly S0 TFDS does not support action chunks")
        if max_episodes < 0:
            raise ValueError("max_episodes must be non-negative")
        if cache_episodes <= 0:
            raise ValueError("cache_episodes must be positive")

        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        if not self.dataset_dir.is_dir():
            raise NotADirectoryError(f"OpenFly S0 TFDS directory is not a directory: {self.dataset_dir}")
        self.index_dir = self.dataset_dir / _INDEX_DIRECTORY
        metadata_path = self.index_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "OpenFly S0 TFDS has no MiniVLA index; run "
                "scripts/build_openfly_s0_tfds_index.py first: "
                f"{metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != OPENFLY_S0_TFDS_INDEX_FORMAT:
            raise ValueError(
                f"Unexpected OpenFly S0 TFDS index format in {metadata_path}: "
                f"{metadata.get('format')!r}"
            )

        shards_path = self.index_dir / "shards.json"
        self._shards = json.loads(shards_path.read_text(encoding="utf-8"))
        if not isinstance(self._shards, list) or not self._shards:
            raise ValueError(f"OpenFly S0 TFDS shard index is empty: {shards_path}")
        self._shard_paths = tuple(
            self.dataset_dir / str(item["relative_path"]) for item in self._shards
        )
        if any(not path.is_file() for path in self._shard_paths):
            missing = [str(path) for path in self._shard_paths if not path.is_file()]
            raise FileNotFoundError(f"OpenFly S0 TFDS shard missing: {missing[:3]}")

        self._row_starts = np.load(
            self.index_dir / "episode_row_starts.npy", mmap_mode="r", allow_pickle=False
        )
        self._shard_ids = np.load(
            self.index_dir / "episode_shard_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self._record_offsets = np.load(
            self.index_dir / "episode_record_offsets.npy", mmap_mode="r", allow_pickle=False
        )
        self._record_lengths = np.load(
            self.index_dir / "episode_record_lengths.npy", mmap_mode="r", allow_pickle=False
        )
        episode_count = len(self._shard_ids)
        if (
            len(self._row_starts) != episode_count + 1
            or len(self._record_offsets) != episode_count
            or len(self._record_lengths) != episode_count
        ):
            raise ValueError("OpenFly S0 TFDS index arrays have inconsistent episode lengths")
        if int(self._row_starts[0]) != 0 or np.any(np.diff(self._row_starts) <= 0):
            raise ValueError("OpenFly S0 TFDS index has non-contiguous/empty episodes")
        if max_episodes:
            episode_count = min(episode_count, int(max_episodes))
        self._episode_limit = episode_count
        self._row_limit = int(self._row_starts[episode_count])
        self._cache_episodes = int(cache_episodes)
        self._verify_record_crc = bool(verify_record_crc)
        self._episode_cache: OrderedDict[int, _DecodedEpisode] = OrderedDict()
        self._shard_handles: dict[int, BinaryIO] = {}
        self.num_actions = 10
        self.num_frames = 1
        self.frame_history_stride = 1
        self.action_chunk_horizon = 1

        expected_rows = metadata.get("rows")
        if max_episodes == 0 and expected_rows is not None and int(expected_rows) != self._row_limit:
            raise ValueError(
                f"OpenFly S0 TFDS index row count mismatch: metadata={expected_rows}, "
                f"arrays={self._row_limit}"
            )

    def __len__(self) -> int:
        return self._row_limit

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"OpenFly S0 TFDS row index out of range: {index}")
        episode_index = int(np.searchsorted(self._row_starts, index, side="right") - 1)
        if episode_index >= self._episode_limit:
            raise RuntimeError("OpenFly S0 TFDS index resolves outside selected episode range")
        step_id = int(index - int(self._row_starts[episode_index]))
        episode = self._load_episode(episode_index)
        if step_id >= len(episode.actions):
            raise RuntimeError("OpenFly S0 TFDS row index exceeds decoded episode length")

        with Image.open(io.BytesIO(episode.images[step_id])) as source:
            image = source.convert("RGB")
        action_id = episode.actions[step_id]
        previous_action = self.num_actions if step_id == 0 else episode.actions[step_id - 1]
        previous_actions_text: list[int | None] = [
            episode.actions[position] if position >= 0 else None
            for position in range(step_id - 3, step_id)
        ]
        return {
            "images": [image],
            "image": image,
            "frame_valid_mask": [True],
            "instruction": episode.instructions[step_id],
            "action_id": action_id,
            "episode_id": episode.episode_id,
            "step_id": step_id,
            "image_path": (
                f"{self._shard_paths[int(self._shard_ids[episode_index])]}@"
                f"{int(self._record_offsets[episode_index])}#{episode.source_image_ids[step_id]}"
            ),
            "prev_action": previous_action,
            "previous_actions_text": previous_actions_text,
            "source_action_id": action_id,
        }

    def _load_episode(self, episode_index: int) -> _DecodedEpisode:
        cached = self._episode_cache.pop(episode_index, None)
        if cached is not None:
            self._episode_cache[episode_index] = cached
            return cached

        shard_id = int(self._shard_ids[episode_index])
        if shard_id < 0 or shard_id >= len(self._shard_paths):
            raise ValueError(f"OpenFly S0 TFDS index references unknown shard {shard_id}")
        path = self._shard_paths[shard_id]
        handle = self._shard_handles.get(shard_id)
        if handle is None or handle.closed:
            handle = path.open("rb")
            self._shard_handles[shard_id] = handle
        offset = int(self._record_offsets[episode_index])
        expected_length = int(self._record_lengths[episode_index])
        handle.seek(offset)
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError(f"Cannot read TFRecord header from {path} at byte {offset}")
        length = int(struct.unpack("<Q", header[:8])[0])
        if length != expected_length:
            raise ValueError(
                f"OpenFly S0 TFDS index/TFRecord length mismatch at {path}:{offset}; "
                f"index={expected_length}, record={length}"
            )
        if self._verify_record_crc and struct.unpack("<I", header[8:])[0] != masked_crc32c(header[:8]):
            raise ValueError(f"TFRecord header CRC mismatch at {path}:{offset}")
        payload = handle.read(length)
        footer = handle.read(4)
        if len(payload) != length or len(footer) != 4:
            raise ValueError(f"Truncated TFRecord payload at {path}:{offset}")
        if self._verify_record_crc and struct.unpack("<I", footer)[0] != masked_crc32c(payload):
            raise ValueError(f"TFRecord payload CRC mismatch at {path}:{offset}")
        episode = self._decode_episode(payload)
        self._episode_cache[episode_index] = episode
        while len(self._episode_cache) > self._cache_episodes:
            self._episode_cache.popitem(last=False)
        return episode

    @staticmethod
    def _decode_episode(payload: bytes) -> _DecodedEpisode:
        features = decode_tf_example(payload)
        actions = _int_feature(features, "steps/action_id")
        images = _bytes_feature(features, "steps/observation/image")
        instructions = [
            value.decode("utf-8")
            for value in _bytes_feature(features, "steps/language_instruction")
        ]
        image_ids = [
            value.decode("utf-8")
            for value in _bytes_feature(features, "steps/source_image_id")
        ]
        episode_ids = _int_feature(features, "episode_metadata/episode_id")
        action_vectors = features.get("steps/action")
        if action_vectors is None or any(
            not isinstance(value, float) for value in action_vectors
        ):
            raise ValueError("OpenFly S0 TFDS record has no float action vectors")
        step_count = len(actions)
        if (
            step_count == 0
            or len(images) != step_count
            or len(instructions) != step_count
            or len(image_ids) != step_count
            or len(action_vectors) != step_count * 8
            or len(episode_ids) != 1
        ):
            raise ValueError("OpenFly S0 TFDS record has inconsistent flat step features")
        if any(action not in OPENFLY_VALID_ACTION_IDS for action in actions):
            raise ValueError("OpenFly S0 TFDS record contains an out-of-range action ID")
        if any(not image for image in images):
            raise ValueError("OpenFly S0 TFDS record contains an empty PNG payload")
        return _DecodedEpisode(
            actions=tuple(actions),
            images=tuple(images),
            instructions=tuple(instructions),
            episode_id=f"openfly-s0-{episode_ids[0]}",
            source_image_ids=tuple(image_ids),
        )

    def __getstate__(self) -> dict[str, Any]:
        """Do not inherit file descriptors or decoded bytes into data workers."""

        state = self.__dict__.copy()
        state["_episode_cache"] = OrderedDict()
        state["_shard_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in self._shard_handles.values():
            try:
                handle.close()
            except Exception:
                pass
