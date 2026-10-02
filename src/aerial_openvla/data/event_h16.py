"""Deterministic Event-H16 history management shared by train and eval.

Event-H16 changes only which frozen Stage-1 memory tokens occupy the existing
16 history slots.  It never averages features: every representative is an
unaltered token produced by the Stage-1 memory tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EVENT_H16_CALIBRATION_FORMAT = "native1t_event_h16_cache_calibration_v1"
EVENT_H16_CALIBRATION_FILENAME = "event_h16_calibration.json"

try:
    import torch
except ImportError:  # pragma: no cover - preprocessing can run with NumPy only
    torch = None


def event_h16_calibration_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).resolve() / EVENT_H16_CALIBRATION_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_event_h16_cache_calibration(
    cache_dir: str | Path,
    *,
    tau_local: float | None = None,
    tau_anchor: float | None = None,
    max_event_length: int = 8,
    max_nodes: int = 16,
    recent_protected: int = 4,
    verify_cache_content: bool = True,
) -> dict[str, Any]:
    """Load and verify the cache-bound Event-H16 calibration sidecar.

    Supplying thresholds is allowed only as an explicit assertion that they
    equal the cache's calibration.  This makes a copied YAML unable to silently
    reuse another tokenizer cache's cosine scale.
    """
    cache_dir = Path(cache_dir).resolve()
    path = event_h16_calibration_path(cache_dir)
    if not path.is_file():
        raise FileNotFoundError(
            "Event-H16 requires a cache-specific calibration sidecar: "
            f"{path}"
        )
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if not calibration.get("complete") or calibration.get("format") != EVENT_H16_CALIBRATION_FORMAT:
        raise ValueError(f"invalid Event-H16 calibration sidecar: {path}")
    identity = calibration.get("target_identity", {})
    if Path(identity.get("cache_dir", "")).resolve() != cache_dir:
        raise ValueError("Event-H16 calibration is bound to a different cache")
    thresholds = calibration.get("thresholds", {})
    expected_static = {
        "max_event_length": int(max_event_length),
        "max_nodes": int(max_nodes),
        "recent_protected": int(recent_protected),
    }
    for key, value in expected_static.items():
        if thresholds.get(key) != value:
            raise ValueError(
                f"Event-H16 calibration {key} mismatch: "
                f"sidecar={thresholds.get(key)!r}, requested={value!r}"
            )
    for key, requested in (("tau_local", tau_local), ("tau_anchor", tau_anchor)):
        value = thresholds.get(key)
        if not isinstance(value, (float, int)) or not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid {key} in Event-H16 calibration")
        if requested is not None and not math.isclose(
            float(requested), float(value), rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                f"Event-H16 {key} must match the cache calibration: "
                f"requested={requested!r}, calibrated={value!r}"
            )
    metadata_path = cache_dir / "metadata.json"
    if identity.get("metadata_sha256") != _sha256(metadata_path):
        raise ValueError("Event-H16 calibration cache metadata hash mismatch")
    if verify_cache_content:
        features_path = cache_dir / "features.npy"
        if identity.get("features_sha256") != _sha256(features_path):
            raise ValueError("Event-H16 calibration feature hash mismatch")
        mapping_path = cache_dir / "manifest_to_cache.npy"
        expected_mapping_hash = identity.get("manifest_to_cache_sha256")
        actual_mapping_hash = _sha256(mapping_path) if mapping_path.exists() else None
        if expected_mapping_hash != actual_mapping_hash:
            raise ValueError("Event-H16 calibration mapping hash mismatch")
    calibration["_path"] = str(path)
    calibration["_sha256"] = _sha256(path)
    return calibration


@dataclass(slots=True)
class EventNode:
    anchor_idx: int
    repr_idx: int
    start_step: int
    end_step: int
    count: int


@dataclass(slots=True)
class EventUpdate:
    created_node: bool
    local_similarity: float | None
    anchor_similarity: float | None
    overflow_merges: int


def _cosine(left: Any, right: Any) -> float:
    """Float32 cosine with identical semantics for NumPy and Torch tokens."""
    if torch is not None and isinstance(left, torch.Tensor):
        left32 = left.detach().reshape(-1).float()
        right32 = right.detach().reshape(-1).float()
        denom = left32.norm() * right32.norm()
        if float(denom.item()) == 0.0:
            raise ValueError("Event-H16 received a zero-norm memory token")
        value = torch.dot(left32, right32) / denom
        result = float(value.item())
    else:
        left32 = np.asarray(left, dtype=np.float32).reshape(-1)
        right32 = np.asarray(right, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(left32) * np.linalg.norm(right32))
        if denom == 0.0:
            raise ValueError("Event-H16 received a zero-norm memory token")
        result = float(np.dot(left32, right32) / denom)
    if not np.isfinite(result):
        raise ValueError("Event-H16 cosine is not finite")
    return result


class EventMemoryManager:
    """Maintain at most ``max_nodes`` chronological visual event nodes."""

    def __init__(
        self,
        *,
        tau_local: float,
        tau_anchor: float,
        max_event_length: int = 8,
        max_nodes: int = 16,
        recent_protected: int = 4,
    ) -> None:
        if not -1.0 <= tau_local <= 1.0:
            raise ValueError("tau_local must lie in [-1, 1]")
        if not -1.0 <= tau_anchor <= 1.0:
            raise ValueError("tau_anchor must lie in [-1, 1]")
        if max_event_length < 1:
            raise ValueError("max_event_length must be positive")
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least two")
        if recent_protected < 0 or recent_protected >= max_nodes - 1:
            raise ValueError("recent_protected leaves no mergeable adjacent pair")
        self.tau_local = float(tau_local)
        self.tau_anchor = float(tau_anchor)
        self.max_event_length = int(max_event_length)
        self.max_nodes = int(max_nodes)
        self.recent_protected = int(recent_protected)
        self.reset()

    def reset(self) -> None:
        self.nodes: list[EventNode] = []
        self._tokens: dict[int, Any] = {}
        self._token_steps: dict[int, int] = {}
        self._previous_idx: int | None = None
        self._previous_step: int | None = None
        self.total_overflow_merges = 0

    def get_repr_indices(self) -> list[int]:
        return [node.repr_idx for node in self.nodes]

    def get_repr_steps(self) -> list[int]:
        return [self._token_steps[node.repr_idx] for node in self.nodes]

    def get_tokens(self) -> list[Any]:
        return [self._tokens[node.repr_idx] for node in self.nodes]

    def get_padded_indices(self, width: int | None = None) -> np.ndarray:
        width = self.max_nodes if width is None else int(width)
        indices = self.get_repr_indices()
        if len(indices) > width:
            raise RuntimeError("Event-H16 contains more nodes than output slots")
        result = np.full(width, -1, dtype=np.int32)
        if indices:
            result[-len(indices):] = np.asarray(indices, dtype=np.int32)
        return result

    def update(
        self,
        token: Any,
        step: int,
        *,
        token_idx: int | None = None,
    ) -> EventUpdate:
        step = int(step)
        token_idx = step if token_idx is None else int(token_idx)
        if token_idx in self._tokens:
            raise ValueError(f"duplicate Event-H16 token index: {token_idx}")
        if self._previous_step is not None and step != self._previous_step + 1:
            raise ValueError(
                f"Event-H16 requires consecutive episode steps, got "
                f"{self._previous_step} then {step}"
            )
        self._tokens[token_idx] = token
        self._token_steps[token_idx] = step

        local_similarity = None
        anchor_similarity = None
        created_node = not self.nodes
        if not self.nodes:
            self._create_node(token_idx, step)
        else:
            if self._previous_idx is None:
                raise RuntimeError("Event-H16 previous token is missing")
            last = self.nodes[-1]
            local_similarity = _cosine(token, self._tokens[self._previous_idx])
            anchor_similarity = _cosine(token, self._tokens[last.anchor_idx])
            stays_in_event = (
                local_similarity >= self.tau_local
                and anchor_similarity >= self.tau_anchor
                and last.count < self.max_event_length
            )
            if stays_in_event:
                last.repr_idx = token_idx
                last.end_step = step
                last.count += 1
            else:
                self._create_node(token_idx, step)
                created_node = True

        overflow_merges = 0
        while len(self.nodes) > self.max_nodes:
            self._merge_most_redundant_old_pair()
            overflow_merges += 1
        self.total_overflow_merges += overflow_merges
        self._previous_idx = token_idx
        self._previous_step = step
        return EventUpdate(
            created_node=created_node,
            local_similarity=local_similarity,
            anchor_similarity=anchor_similarity,
            overflow_merges=overflow_merges,
        )

    def _create_node(self, token_idx: int, step: int) -> None:
        self.nodes.append(EventNode(
            anchor_idx=token_idx,
            repr_idx=token_idx,
            start_step=step,
            end_step=step,
            count=1,
        ))

    def _merge_most_redundant_old_pair(self) -> None:
        # The final recent_protected nodes are untouchable.  With zero-based
        # indices, both j and j+1 must be inside nodes[:N-recent_protected].
        old_count = len(self.nodes) - self.recent_protected
        if old_count < 2:
            raise RuntimeError("Event-H16 overflow has no eligible old pair")
        similarities = [
            _cosine(
                self._tokens[self.nodes[index].repr_idx],
                self._tokens[self.nodes[index + 1].repr_idx],
            )
            for index in range(old_count - 1)
        ]
        # np.argmax deterministically resolves ties in favour of the oldest
        # eligible pair.
        merge_index = int(np.argmax(np.asarray(similarities, dtype=np.float32)))
        older = self.nodes[merge_index]
        newer = self.nodes[merge_index + 1]
        merged = EventNode(
            anchor_idx=older.anchor_idx,
            repr_idx=newer.repr_idx,
            start_step=older.start_step,
            end_step=newer.end_step,
            count=older.count + newer.count,
        )
        self.nodes[merge_index:merge_index + 2] = [merged]
