"""Hard contracts for the 32T Native1T current-frame reducer.

This module is shared by the two instruction-conditioned pooler launchers,
the Stage-1 cache extractor, and the Stage-2 gate.  Metadata saying
``visual_tokens_per_frame=32`` is not sufficient: the exact live classifier
source must contain the Native1T reducer call and every checkpoint/cache must
carry the same reducer identity.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


CLASSIFIER_RELATIVE_PATH = Path(
    "src/aerial_openvla/models/airminivla_classifier.py"
)
VISUAL_CONTRACT_KEYS = (
    "native1t_stage",
    "active_reducer",
    "current_raw_tokens",
    "current_projected_tokens",
    "current_output_tokens",
    "reducer_call",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classifier_source_path(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / CLASSIFIER_RELATIVE_PATH


def _is_native1t_condition(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Attribute)
        and test.attr == "use_native1t_postfusion"
        and isinstance(test.value, ast.Name)
        and test.value.id == "c"
    )


def verify_native1t_source(project_root: str | Path) -> dict[str, Any]:
    """Verify that the live Native1T forward contains the reducer call."""
    path = classifier_source_path(project_root)
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RuntimeError(f"Native1T classifier source is not parseable: {path}") from exc

    forward: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "AirMiniVLAActionClassifier":
            continue
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "forward":
                forward = child
                break
    if forward is None:
        raise RuntimeError("AirMiniVLAActionClassifier.forward was not found")

    native_reducer_calls = 0
    for node in ast.walk(forward):
        if not isinstance(node, ast.If) or not _is_native1t_condition(node.test):
            continue
        native_reducer_calls += sum(
            1
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "_reduce_visual_tokens"
        )
    if native_reducer_calls != 1:
        raise RuntimeError(
            "Native1T current-frame reducer is not wired exactly once in the live forward: "
            f"found {native_reducer_calls} call(s) inside the Native1T branch"
        )
    if "Native1T current reducer contract failed" not in source:
        raise RuntimeError(
            "Native1T forward lacks the post-reducer shape assertion; refusing to launch"
        )
    return {
        "classifier_path": str(path),
        "classifier_sha256": sha256_file(path),
        "native1t_reducer_callsite": "AirMiniVLAActionClassifier.forward/Native1T",
        "native1t_reducer_calls": native_reducer_calls,
        "shape_assertion": True,
    }


def validate_pooler32_config(cfg: Any, *, expected_stage: int) -> None:
    """Validate the exact new 32T Native1T pooler configuration."""
    mismatches: list[str] = []
    if not bool(getattr(cfg, "use_native1t_postfusion", False)):
        mismatches.append("use_native1t_postfusion must be true")
    if int(getattr(cfg, "native1t_stage", -1)) != int(expected_stage):
        mismatches.append(f"native1t_stage must be {expected_stage}")
    if int(getattr(cfg, "visual_tokens_per_frame", -1)) != 32:
        mismatches.append("visual_tokens_per_frame must be 32")
    if bool(getattr(cfg, "use_task_aware_pooler", False)):
        mismatches.append("legacy use_task_aware_pooler must be false")
    local = bool(getattr(cfg, "use_instruction_local_pooler", False))
    spatial = bool(getattr(cfg, "use_spatial_anchored_residual_pooler", False))
    if int(local) + int(spatial) != 1:
        mismatches.append(
            "exactly one of use_instruction_local_pooler/use_spatial_anchored_residual_pooler must be true"
        )
    if local and int(getattr(cfg, "instruction_local_pooler_tokens", -1)) != 32:
        mismatches.append("instruction_local_pooler_tokens must be 32")
    if spatial:
        if int(getattr(cfg, "spatial_anchored_pooler_tokens", -1)) != 32:
            mismatches.append("spatial_anchored_pooler_tokens must be 32")
        if int(getattr(cfg, "spatial_anchored_pooler_heads", -1)) != 8:
            mismatches.append("spatial_anchored_pooler_heads must be 8")
    if mismatches:
        raise ValueError("32T Native1T pooler contract failed: " + "; ".join(mismatches))


def build_native1t_visual_contract(
    cfg: Any,
    *,
    source_sha256: str | None = None,
    stage_override: int | None = None,
) -> dict[str, Any]:
    """Build the expected current-frame and multimodal visual geometry."""
    stage = int(stage_override if stage_override is not None else getattr(cfg, "native1t_stage", -1))
    target = int(getattr(cfg, "visual_tokens_per_frame", -1))
    local = bool(getattr(cfg, "use_instruction_local_pooler", False))
    spatial = bool(getattr(cfg, "use_spatial_anchored_residual_pooler", False))
    task_aware = bool(getattr(cfg, "use_task_aware_pooler", False))
    if local:
        reducer = "instruction_local_weighted32"
    elif spatial:
        reducer = "spatial_anchored_residual32"
    elif task_aware:
        reducer = f"task_aware_mha_{target}"
    elif target == 32:
        reducer = "avgpool32"
    elif target in {64, 128, 256}:
        reducer = f"avgpool{target}"
    else:
        reducer = f"unknown_{target}"
    memory_slots = 9 if stage == 1 else int(getattr(cfg, "native1t_history_length", -1))
    result: dict[str, Any] = {
        "format": "openfly_native1t_current_visual_contract_v2",
        "native1t_stage": stage,
        "active_reducer": reducer,
        "current_raw_tokens": 256,
        "current_projected_tokens": 256,
        "current_output_tokens": target,
        "memory_slots_in_sequence": memory_slots,
        "visual_tokens_in_sequence": target + memory_slots,
        "reducer_call": "native1t_postfusion_current_unconditional",
    }
    if source_sha256 is not None:
        result["source_classifier_sha256"] = str(source_sha256)
    return result


def require_visual_contract_compatible(
    actual: Any,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{label} has no native1t_current_visual_contract")
    mismatches = {
        key: (actual.get(key), expected.get(key))
        for key in VISUAL_CONTRACT_KEYS
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(f"{label} visual reducer contract mismatch: {mismatches}")


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value
