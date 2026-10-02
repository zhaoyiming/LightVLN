"""OpenFly's 8-D action-vector/token contract.

The original OpenFly VLA does not use a categorical action softmax.  It
normalizes an eight-dimensional control vector, discretizes each dimension
into one of 256 action tokens, and trains token-level cross entropy.  This
module makes that data contract explicit for the isolated MiniVLA adapter.

``action_tokens_v1`` retains the historical MiniVLA parallel head.  The
newer ``action_tokens_ar_v2`` uses this same immutable vector codec but
feeds its classes through OpenFly's reserved language-model token IDs, full
vocabulary CE, and greedy autoregressive generation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


OPENFLY_ACTION_HEAD_TYPE = "action_tokens_v1"
OPENFLY_ACTION_CODEC_FORMAT = "openfly_action_token_codec_v1"
OPENFLY_ACTION_DIM = 8
OPENFLY_ACTION_BINS = 256
OPENFLY_VALID_ACTION_IDS: tuple[int, ...] = tuple(range(10))
# The released training annotations contain no ID 6/7 rows.  The direct
# MiniVLA observed-action policy therefore trains only the eight IDs that
# actually occur, while preserving the original OpenFly IDs at the evaluator
# boundary.  This is deliberately not an AirVLN class map and does not alter
# OpenFly's 8-D vector specification below.
OPENFLY_OBSERVED8_ACTION_POLICY = "openfly_observed_8"
OPENFLY_OBSERVED8_ACTION_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 8, 9)
OPENFLY_OBSERVED8_CLASS_TO_ACTION_ID: dict[int, int] = dict(
    enumerate(OPENFLY_OBSERVED8_ACTION_IDS)
)
OPENFLY_OBSERVED8_ACTION_ID_TO_CLASS: dict[int, int] = {
    action_id: class_id
    for class_id, action_id in OPENFLY_OBSERVED8_CLASS_TO_ACTION_ID.items()
}
# ``-1`` and ``-2`` are annotation boundary sentinels, not executable
# controls.  They occur only before/after the valid action portion of a
# released episode and are intentionally excluded from supervision.
OPENFLY_SENTINEL_ACTION_IDS: tuple[int, ...] = (-2, -1)

OPENFLY_ACTION_NAMES: dict[int, str] = {
    0: "stop",
    1: "forward_3m",
    2: "turn_left",
    3: "turn_right",
    4: "up",
    5: "down",
    6: "move_left",
    7: "move_right",
    8: "forward_6m",
    9: "forward_9m",
}

# Exact action dictionary from OpenFly's vln/vln_norm builders and evaluator.
OPENFLY_ACTION_VECTORS: dict[int, tuple[float, ...]] = {
    0: (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    1: (0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    2: (0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    3: (0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0),
    4: (0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
    5: (0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0),
    6: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0),
    7: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0),
    8: (0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    9: (0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _normalise_counts(value: Mapping[int, int]) -> dict[str, int]:
    return {str(int(key)): int(count) for key, count in sorted(value.items())}


def _load_annotation_actions(annotation_path: Path) -> tuple[list[int], dict[int, int], dict[int, int]]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    episodes = raw.get("episodes", raw) if isinstance(raw, Mapping) else raw
    if not isinstance(episodes, list):
        raise ValueError("OpenFly annotation must be a list or an object with an 'episodes' list")
    actions: list[int] = []
    valid_counts = {action_id: 0 for action_id in OPENFLY_VALID_ACTION_IDS}
    sentinel_counts = {action_id: 0 for action_id in OPENFLY_SENTINEL_ACTION_IDS}
    for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"OpenFly annotation episode {episode_index} is not an object")
        values = episode.get("action", episode.get("actions"))
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"OpenFly annotation episode {episode_index} has no action array")
        for step_index, value in enumerate(values):
            try:
                action_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"OpenFly annotation episode {episode_index} step {step_index} has "
                    f"a non-integer action: {value!r}"
                ) from exc
            if action_id in OPENFLY_VALID_ACTION_IDS:
                actions.append(action_id)
                valid_counts[action_id] += 1
            elif action_id in OPENFLY_SENTINEL_ACTION_IDS:
                sentinel_counts[action_id] += 1
            else:
                raise ValueError(
                    f"Unsupported OpenFly action {action_id} at episode {episode_index} "
                    f"step {step_index}; expected 0..9 or boundary sentinel -1/-2"
                )
    if not actions:
        raise ValueError("OpenFly annotation contains no executable action rows")
    return actions, valid_counts, sentinel_counts


@dataclass(frozen=True)
class OpenFlyActionCodec:
    """Immutable OpenFly BOUNDS_Q99 action-token codec."""

    payload: dict[str, Any]
    path: Path | None = None

    @property
    def sha256(self) -> str:
        value = self.payload.get("sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("OpenFly action codec has no valid SHA-256")
        return value

    @property
    def q01(self) -> np.ndarray:
        return np.asarray(self.payload["q01"], dtype=np.float32)

    @property
    def q99(self) -> np.ndarray:
        return np.asarray(self.payload["q99"], dtype=np.float32)

    @property
    def mask(self) -> np.ndarray:
        return np.asarray(self.payload["mask"], dtype=np.bool_)

    @property
    def zero_dims(self) -> np.ndarray:
        return np.asarray(self.payload["zero_dims"], dtype=np.bool_)

    @property
    def bin_edges(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, OPENFLY_ACTION_BINS, dtype=np.float32)

    @property
    def bin_centers(self) -> np.ndarray:
        edges = self.bin_edges
        return (edges[:-1] + edges[1:]) / 2.0

    @property
    def action_vectors(self) -> np.ndarray:
        return np.asarray(
            [OPENFLY_ACTION_VECTORS[action_id] for action_id in OPENFLY_VALID_ACTION_IDS],
            dtype=np.float32,
        )

    @classmethod
    def build_from_annotation(cls, annotation_path: str | Path) -> "OpenFlyActionCodec":
        source = Path(annotation_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"OpenFly annotation does not exist: {source}")
        action_ids, valid_counts, sentinel_counts = _load_annotation_actions(source)
        vectors = np.asarray(
            [OPENFLY_ACTION_VECTORS[action_id] for action_id in action_ids],
            dtype=np.float32,
        )
        q01 = np.quantile(vectors, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(vectors, 0.99, axis=0).astype(np.float32)
        mins = vectors.min(axis=0).astype(np.float32)
        maxs = vectors.max(axis=0).astype(np.float32)
        payload: dict[str, Any] = {
            "format": OPENFLY_ACTION_CODEC_FORMAT,
            "action_head_type": OPENFLY_ACTION_HEAD_TYPE,
            "action_dim": OPENFLY_ACTION_DIM,
            "num_bins": OPENFLY_ACTION_BINS,
            "normalization": "bounds_q99",
            "source_annotation": str(source),
            "source_annotation_sha256": sha256_file(source),
            "valid_action_counts": _normalise_counts(valid_counts),
            "ignored_boundary_sentinel_counts": _normalise_counts(sentinel_counts),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "min": mins.tolist(),
            "max": maxs.tolist(),
            # OpenFly's dataset transform defaults to normalizing every action
            # dimension, then forces dimensions with min==max to zero.
            "mask": [True] * OPENFLY_ACTION_DIM,
            "zero_dims": (mins == maxs).tolist(),
            "action_codebook": {
                str(action_id): list(OPENFLY_ACTION_VECTORS[action_id])
                for action_id in OPENFLY_VALID_ACTION_IDS
            },
        }
        payload["sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        codec = cls(payload=payload, path=None)
        codec.validate()
        return codec

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str | None = None) -> "OpenFlyActionCodec":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"OpenFly action codec is missing: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenFly action codec is invalid JSON: {source}") from exc
        if not isinstance(payload, dict):
            raise ValueError("OpenFly action codec must be a JSON object")
        codec = cls(payload=payload, path=source)
        codec.validate()
        if expected_sha256 is not None and codec.sha256 != expected_sha256:
            raise ValueError(
                "OpenFly action codec hash mismatch: "
                f"expected {expected_sha256}, got {codec.sha256} ({source})"
            )
        return codec

    def validate(self) -> None:
        payload = self.payload
        required = {
            "format": OPENFLY_ACTION_CODEC_FORMAT,
            "action_head_type": OPENFLY_ACTION_HEAD_TYPE,
            "action_dim": OPENFLY_ACTION_DIM,
            "num_bins": OPENFLY_ACTION_BINS,
            "normalization": "bounds_q99",
        }
        mismatches = {key: (payload.get(key), value) for key, value in required.items() if payload.get(key) != value}
        if mismatches:
            raise ValueError(f"OpenFly action codec contract mismatch: {mismatches}")
        for field in ("q01", "q99", "min", "max", "mask", "zero_dims"):
            value = payload.get(field)
            if not isinstance(value, list) or len(value) != OPENFLY_ACTION_DIM:
                raise ValueError(f"OpenFly action codec {field!r} must have {OPENFLY_ACTION_DIM} values")
        codebook = payload.get("action_codebook")
        expected_codebook = {
            str(action_id): list(OPENFLY_ACTION_VECTORS[action_id])
            for action_id in OPENFLY_VALID_ACTION_IDS
        }
        if codebook != expected_codebook:
            raise ValueError("OpenFly action codec codebook differs from the released OpenFly contract")
        supplied_hash = payload.get("sha256")
        canonical = dict(payload)
        canonical.pop("sha256", None)
        computed_hash = hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()
        if supplied_hash != computed_hash:
            raise ValueError(
                "OpenFly action codec SHA-256 mismatch: "
                f"stored={supplied_hash!r}, computed={computed_hash}"
            )
        if np.any(self.q99 < self.q01):
            raise ValueError("OpenFly action codec has q99 < q01")
        if not np.array_equal(self.zero_dims, np.asarray(payload["min"], dtype=np.float32) == np.asarray(payload["max"], dtype=np.float32)):
            raise ValueError("OpenFly action codec zero_dims disagrees with min/max")

    def save_new(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite OpenFly action codec: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.inprogress")
        if temporary.exists():
            raise FileExistsError(f"OpenFly action codec temporary file exists: {temporary}")
        temporary.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target

    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float32)
        if values.shape[-1] != OPENFLY_ACTION_DIM:
            raise ValueError(f"Expected last action dimension {OPENFLY_ACTION_DIM}, got {values.shape}")
        normalized = 2.0 * (values - self.q01) / (self.q99 - self.q01 + 1.0e-8) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        normalized[..., self.zero_dims] = 0.0
        return normalized.astype(np.float32, copy=False)

    def encode_vectors(self, vectors: np.ndarray) -> np.ndarray:
        # This deliberately matches OpenFly's ActionTokenizer: ``digitize``
        # returns 1..256, which are represented here as CE class 0..255.
        normalized = self.normalize_vectors(vectors)
        targets = np.digitize(normalized, self.bin_edges).astype(np.int64) - 1
        return np.clip(targets, 0, OPENFLY_ACTION_BINS - 1)

    def encode_action_ids(self, action_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        ids = np.asarray(action_ids, dtype=np.int64)
        if np.any((ids < 0) | (ids >= len(OPENFLY_VALID_ACTION_IDS))):
            raise ValueError("OpenFly action IDs must be in 0..9")
        return self.encode_vectors(self.action_vectors[ids])

    def decode_token_classes(self, token_classes: np.ndarray) -> np.ndarray:
        classes = np.asarray(token_classes, dtype=np.int64)
        if classes.shape[-1] != OPENFLY_ACTION_DIM:
            raise ValueError("OpenFly token classes must end in the 8 action dimensions")
        centers = self.bin_centers
        normalized = centers[np.clip(classes, 0, len(centers) - 1)]
        return np.where(
            self.mask,
            0.5 * (normalized + 1.0) * (self.q99 - self.q01) + self.q01,
            normalized,
        ).astype(np.float32, copy=False)

    def decode_token_classes_to_actions(self, token_classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vectors = self.decode_token_classes(token_classes)
        rounded = np.rint(vectors).astype(np.int64)
        codebook = np.asarray(self.action_vectors, dtype=np.int64)
        matches = (rounded[..., None, :] == codebook).all(axis=-1)
        matched = matches.any(axis=-1)
        action_ids = matches.argmax(axis=-1).astype(np.int64)
        # OpenFly's evaluator defaults an unmatched continuous vector to STOP.
        action_ids = np.where(matched, action_ids, 0)
        return action_ids, rounded, matched

    def decode_token_ids_tensor(
        self,
        token_ids: torch.Tensor,
        *,
        action_vocab_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode full-vocabulary OpenFly generated IDs exactly as the reference VLA.

        The original ``predict_action`` subtracts generated IDs from the
        action vocabulary size and clips the resulting bin index to the 255
        available bin centers.  In particular, an out-of-range generated
        token is not silently re-ranked within the 256 action IDs.
        """
        if token_ids.ndim != 2 or token_ids.shape[1] != OPENFLY_ACTION_DIM:
            raise ValueError(
                "OpenFly generated token IDs must have shape [B,8], got "
                f"{tuple(token_ids.shape)}"
            )
        if action_vocab_size <= OPENFLY_ACTION_BINS:
            raise ValueError(
                f"OpenFly action vocabulary must exceed {OPENFLY_ACTION_BINS}, "
                f"got {action_vocab_size}"
            )
        # ActionTokenizer: token_id = vocab_size - digitize(value), then the
        # reference decoder uses clip(vocab_size - token_id - 1, 0, 254).
        bin_indices = (int(action_vocab_size) - token_ids.long() - 1).clamp(
            min=0, max=OPENFLY_ACTION_BINS - 2
        )
        centers = torch.linspace(
            -1.0 + 1.0 / (OPENFLY_ACTION_BINS - 1),
            1.0 - 1.0 / (OPENFLY_ACTION_BINS - 1),
            OPENFLY_ACTION_BINS - 1,
            dtype=torch.float32,
            device=token_ids.device,
        )
        normalized = centers[bin_indices]
        q01 = torch.as_tensor(self.q01, dtype=torch.float32, device=token_ids.device)
        q99 = torch.as_tensor(self.q99, dtype=torch.float32, device=token_ids.device)
        mask = torch.as_tensor(self.mask, dtype=torch.bool, device=token_ids.device)
        vectors = torch.where(
            mask,
            0.5 * (normalized + 1.0) * (q99 - q01) + q01,
            normalized,
        )
        rounded = torch.round(vectors).long()
        codebook = torch.as_tensor(
            self.action_vectors, dtype=torch.long, device=token_ids.device
        )
        matches = (rounded.unsqueeze(-2) == codebook).all(dim=-1)
        matched = matches.any(dim=-1)
        action_ids = matches.long().argmax(dim=-1)
        action_ids = torch.where(matched, action_ids, torch.zeros_like(action_ids))
        return action_ids, vectors, matched

    def targets_tensor(self, action_ids: torch.Tensor) -> torch.Tensor:
        if action_ids.dtype != torch.long:
            action_ids = action_ids.long()
        table = torch.as_tensor(
            self.encode_action_ids(np.arange(10, dtype=np.int64)),
            dtype=torch.long,
            device=action_ids.device,
        )
        if torch.any((action_ids < 0) | (action_ids >= 10)):
            raise ValueError("OpenFly action IDs must be in 0..9")
        return table[action_ids]

    def decode_logits_tensor(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.ndim != 3 or tuple(logits.shape[-2:]) != (OPENFLY_ACTION_DIM, OPENFLY_ACTION_BINS):
            raise ValueError(
                "OpenFly action-token logits must have shape [B,8,256], got "
                f"{tuple(logits.shape)}"
            )
        token_classes = logits.argmax(dim=-1)
        centers = torch.linspace(
            -1.0 + 1.0 / (OPENFLY_ACTION_BINS - 1),
            1.0 - 1.0 / (OPENFLY_ACTION_BINS - 1),
            OPENFLY_ACTION_BINS - 1,
            dtype=torch.float32,
            device=logits.device,
        )
        normalized = centers[token_classes.clamp(max=OPENFLY_ACTION_BINS - 2)]
        q01 = torch.as_tensor(self.q01, dtype=torch.float32, device=logits.device)
        q99 = torch.as_tensor(self.q99, dtype=torch.float32, device=logits.device)
        mask = torch.as_tensor(self.mask, dtype=torch.bool, device=logits.device)
        vectors = torch.where(
            mask,
            0.5 * (normalized + 1.0) * (q99 - q01) + q01,
            normalized,
        )
        rounded = torch.round(vectors).long()
        codebook = torch.as_tensor(
            self.action_vectors, dtype=torch.long, device=logits.device
        )
        matches = (rounded.unsqueeze(-2) == codebook).all(dim=-1)
        matched = matches.any(dim=-1)
        action_ids = matches.long().argmax(dim=-1)
        action_ids = torch.where(matched, action_ids, torch.zeros_like(action_ids))
        return action_ids, vectors, matched

    def codebook_probabilities_tensor(self, logits: torch.Tensor) -> torch.Tensor:
        """Return a 10-way diagnostic score over full codebook token tuples."""
        targets = torch.as_tensor(
            self.encode_action_ids(np.arange(10, dtype=np.int64)),
            dtype=torch.long,
            device=logits.device,
        )
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        expanded = targets.unsqueeze(0).expand(logits.shape[0], -1, -1)
        selected = torch.gather(
            log_probs.unsqueeze(1).expand(-1, 10, -1, -1),
            -1,
            expanded.unsqueeze(-1),
        ).squeeze(-1)
        return torch.softmax(selected.sum(dim=-1), dim=-1)


def load_openfly_action_codec(
    path: str | Path | None, *, expected_sha256: str | None
) -> OpenFlyActionCodec:
    if not path:
        raise ValueError("OpenFly action-token head requires openfly_action_codec_path")
    if not expected_sha256:
        raise ValueError("OpenFly action-token head requires openfly_action_codec_sha256")
    return OpenFlyActionCodec.load(path, expected_sha256=expected_sha256)
