#!/usr/bin/env python3
"""Train MiniVLA on the eight OpenFly actions observed in released training data.

The public OpenFly action vocabulary has ten IDs, but the released train
annotations have no ``move_left`` (6) or ``move_right`` (7) samples.  This
adapter learns a compact eight-way head over the *observed* original IDs
``(0, 1, 2, 3, 4, 5, 8, 9)``.  It is not an AirVLN remap: the checkpoint pins
the compact-to-original mapping and the rollout adapter restores the original
OpenFly ID before the official evaluator executes it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _parse_adapter_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--openfly-action-policy",
        choices=("openfly_observed_8",),
        default="openfly_observed_8",
        help=(
            "Compact classifier over OpenFly IDs 0,1,2,3,4,5,8,9; "
            "the rollout adapter restores the original ID."
        ),
    )
    parser.add_argument(
        "--openfly-max-episodes",
        type=int,
        default=0,
        help="Bound completed TFDS episodes for a smoke run (0 means all).",
    )
    parser.add_argument(
        "--openfly-cache-episodes",
        type=int,
        default=4,
        help="Per-worker number of decoded TFDS episodes to cache in RAM.",
    )
    return parser.parse_known_args(argv)


def _require_observed8_contract(cfg: Any, *, action_policy: str) -> None:
    """Reject every setting outside direct one-frame observed-OpenFly CE."""

    from aerial_openvla.openfly.action_codec import (
        OPENFLY_OBSERVED8_ACTION_IDS,
        OPENFLY_OBSERVED8_ACTION_POLICY,
    )

    errors: list[str] = []
    if cfg.model_type != "airminivla":
        errors.append("model_type must be 'airminivla'")
    if cfg.input_mode != "multiframe_image":
        errors.append("input_mode must be 'multiframe_image'")
    if cfg.airminivla_variant != "s0":
        errors.append("airminivla_variant must be 's0'")
    if (
        action_policy != OPENFLY_OBSERVED8_ACTION_POLICY
        or cfg.openfly_action_policy != OPENFLY_OBSERVED8_ACTION_POLICY
    ):
        errors.append(
            "CLI and config action policy must both be 'openfly_observed_8'"
        )
    if cfg.openfly_action_head_type != "categorical":
        errors.append("openfly_action_head_type must be 'categorical'")
    if cfg.num_actions != len(OPENFLY_OBSERVED8_ACTION_IDS):
        errors.append("num_actions must be 8 for observed OpenFly IDs")
    if cfg.openfly_action_token_dim != 0 or cfg.openfly_action_token_bins != 0:
        errors.append("observed8 disables action-token dimensions and bins (both must be 0)")
    if cfg.openfly_action_codec_path or cfg.openfly_action_codec_sha256:
        errors.append("observed8 must not carry an action codec")
    if cfg.num_frames != 1 or cfg.frame_history_stride != 1:
        errors.append("current-frame TFDS observed8 requires num_frames=1 and stride=1")
    if cfg.sequence_length != 1 or cfg.action_chunk_horizon != 1:
        errors.append("observed8 requires sequence_length=1 and action_chunk_horizon=1")
    if cfg.visual_history_mode != "none":
        errors.append("observed8 requires visual_history_mode='none'")
    if cfg.use_prev_action or cfg.use_previous_actions_text:
        errors.append("observed8 does not use action-history input")
    if cfg.use_relative_state_text or cfg.use_relative_planar_state_text:
        errors.append("observed8 does not define AirVLN relative-state text")
    if cfg.use_memory or cfg.use_temporal_memory or cfg.use_native1t_postfusion:
        errors.append("observed8 is direct S0, not a memory/Native1T route")
    if cfg.use_h16_frozen_pool or cfg.use_omtrack or cfg.task_state_use_gru:
        errors.append("observed8 does not use H16, OmTrack, or TaskGRU")
    if cfg.use_motion_instruction_attention or cfg.task_state_mode != "none":
        errors.append("observed8 keeps the original non-SegTS MiniVLA surface")
    if (
        cfg.use_transition_aux_head
        or cfg.use_stop_aux_head
        or cfg.use_hierarchical_stop
        or cfg.use_progress_aux_head
    ):
        errors.append("observed8 uses only the primary eight-way cross-entropy loss")
    if errors:
        raise ValueError("OpenFly observed8 contract failed: " + "; ".join(errors))


def _tfds_dataset_dir(manifest_path: str) -> Path:
    """Resolve a completed current-frame TFDS version from its retained manifest."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"OpenFly observed8 retained manifest does not exist: {manifest}"
        )
    index_dir = manifest.parent
    if (
        manifest.name != "retained_episodes.jsonl"
        or index_dir.name != "minivla_s0_index"
        or not (index_dir / "metadata.json").is_file()
    ):
        raise ValueError(
            "train_manifest must be the immutable "
            "minivla_s0_index/retained_episodes.jsonl from a completed TFDS build"
        )
    return index_dir.parent


class _ObservedOpenFly8Dataset:
    """Map raw OpenFly IDs to compact head classes without mutating TFDS."""

    def __init__(self, base_dataset: Any) -> None:
        from aerial_openvla.openfly.action_codec import (
            OPENFLY_OBSERVED8_ACTION_ID_TO_CLASS,
        )

        self._base_dataset = base_dataset
        self._source_to_class = dict(OPENFLY_OBSERVED8_ACTION_ID_TO_CLASS)

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getattr__(self, name: str) -> Any:
        # Keep read-only TFDS metadata available to validation/cardinality code.
        return getattr(self._base_dataset, name)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._base_dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError(
                "OpenFly current-frame TFDS must yield mappings, got "
                f"{type(sample).__name__} at row {index}"
            )
        if "action_id" not in sample:
            raise KeyError(f"OpenFly TFDS sample {index} has no action_id")
        source_action_id = int(sample["action_id"])
        try:
            compact_action_id = self._source_to_class[source_action_id]
        except KeyError as error:
            raise ValueError(
                "Observed8 training encountered an unsupported OpenFly action "
                f"ID {source_action_id} at row {index}; expected only "
                f"{tuple(self._source_to_class)}"
            ) from error
        mapped = dict(sample)
        mapped["source_openfly_action_id"] = source_action_id
        mapped["action_id"] = compact_action_id
        return mapped


def main() -> None:
    adapter_args, trainer_argv = _parse_adapter_args(sys.argv[1:])
    if adapter_args.openfly_max_episodes < 0:
        raise ValueError("--openfly-max-episodes must be non-negative")
    if adapter_args.openfly_cache_episodes <= 0:
        raise ValueError("--openfly-cache-episodes must be positive")
    # The copied trainer owns its normal --config/--resume CLI. Remove only
    # this adapter's flags before it parses argv.
    sys.argv = [sys.argv[0], *trainer_argv]

    from aerial_openvla.core.config import TrainConfig, load_yaml_dataclass
    from aerial_openvla.data.multiframe_dataset import MultiFrameCollator
    from aerial_openvla.data.pipeline import DataPipeline, _cached_processor
    from aerial_openvla.openfly.tfds_current_frame import OpenFlyCurrentFrameTFDSDataset
    import aerial_openvla.training.train as trainer

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", required=True)
    config_args, _ = config_parser.parse_known_args(trainer_argv)
    direct_cfg = load_yaml_dataclass(config_args.config, TrainConfig)
    _require_observed8_contract(direct_cfg, action_policy=adapter_args.openfly_action_policy)

    def create_dataset(cfg: Any, manifest_path: str) -> _ObservedOpenFly8Dataset:
        _require_observed8_contract(cfg, action_policy=adapter_args.openfly_action_policy)
        # Keep the immutable TFDS loader on its raw OpenFly-ID contract; this
        # wrapper is the only place that compacts labels for the 8-logit head.
        base_dataset = OpenFlyCurrentFrameTFDSDataset(
            _tfds_dataset_dir(manifest_path),
            action_policy="openfly_10",
            num_frames=1,
            frame_history_stride=1,
            action_chunk_horizon=1,
            max_episodes=adapter_args.openfly_max_episodes,
            cache_episodes=adapter_args.openfly_cache_episodes,
        )
        return _ObservedOpenFly8Dataset(base_dataset)

    def create_pipeline(cfg: Any) -> DataPipeline:
        _require_observed8_contract(cfg, action_policy=adapter_args.openfly_action_policy)
        processor = _cached_processor(cfg)
        collator = MultiFrameCollator(
            processor,
            prompt_template=cfg.prompt_template,
            max_text_tokens=cfg.max_text_tokens,
            instruction_only_text_budget=cfg.instruction_only_text_budget,
            num_frames=1,
            include_instruction_token_mask=bool(
                getattr(cfg, "use_instruction_local_pooler", False)
                or getattr(cfg, "use_spatial_anchored_residual_pooler", False)
            ),
            include_segment_token_ids=False,
            use_previous_actions_text=False,
            use_relative_state_text=False,
            use_relative_planar_state_text=False,
            action_chunk_horizon=1,
            visual_history_mode="none",
        )
        return DataPipeline(
            dataset=create_dataset(cfg, cfg.train_manifest), collator=collator
        )

    trainer.create_dataset = create_dataset
    trainer.create_pipeline = create_pipeline
    trainer.main()


if __name__ == "__main__":
    main()
