"""Data pipeline factory: dispatch on ``cfg.input_mode`` to build
the right Dataset + Collator pair.

Usage::

    from aerial_openvla.data.pipeline import DataPipeline, create_pipeline

    pipeline = create_pipeline(cfg)
    train_loader = DataLoader(pipeline.dataset, collate_fn=pipeline.collator, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torch.utils.data import Dataset

from aerial_openvla.core.config import TrainConfig


@dataclass(frozen=True)
class DataPipeline:
    """A Dataset and its matching collate function."""

    dataset: Dataset[dict]
    collator: Callable[[list[dict]], dict[str, Any]]


def create_pipeline(cfg: TrainConfig) -> DataPipeline:
    """Create the data pipeline for the configured ``input_mode``.

    Parameters
    ----------
    cfg : TrainConfig
        Full training configuration.

    Returns
    -------
    DataPipeline
        ``.dataset`` is the PyTorch Dataset and ``.collator`` is the
        collate function suitable for DataLoader.
    """
    if getattr(cfg, "model_type", None) == "qwen35_nativemem":
        return _create_qwen35_nativemem_pipeline(cfg)
    if (
        getattr(cfg, "model_type", None) == "qwen35_native"
        and getattr(cfg, "task_state_mode", "none") != "none"
    ):
        return _create_qwen35_native_taskgru_pipeline(cfg)
    if getattr(cfg, "use_h16_frozen_pool", False):
        return _create_h16_frozen_pool_pipeline(cfg)
    if (
        getattr(cfg, "model_type", None) == "qwen35_native"
        and getattr(cfg, "use_native1t_postfusion", False)
    ):
        return _create_qwen35_native1t_pipeline(cfg)
    if getattr(cfg, "use_native1t_postfusion", False):
        return _create_native1t_pipeline(cfg)
    if getattr(cfg, "use_omtrack", False):
        return _create_omtrack_pipeline(cfg)
    input_mode = getattr(cfg, "input_mode", "image")

    if input_mode == "image":
        return _create_image_pipeline(cfg)

    if input_mode == "feature":
        return _create_feature_pipeline(cfg)

    if input_mode == "multiframe_image":
        return _create_multiframe_pipeline(cfg)

    raise ValueError(
        f"Unknown input_mode={input_mode!r}. "
        f"Supported modes: image, feature, multiframe_image."
    )


def create_dataset(cfg: TrainConfig, manifest_path: str) -> Dataset[dict]:
    """Create a dataset matching the configured training input contract."""
    if getattr(cfg, "model_type", None) == "qwen35_nativemem":
        from pathlib import Path
        from aerial_openvla.data.qwen35_nativemem_v2_dataset import Qwen35NativeMEMV2Dataset
        is_train = Path(manifest_path).resolve() == Path(cfg.train_manifest).resolve()
        cache_dir = (
            cfg.nativemem_premerger_train_cache
            if is_train else cfg.nativemem_premerger_val_cache
        )
        memory_cache_dir = (
            getattr(cfg, "nativemem_1t_train_cache", None)
            if is_train else getattr(cfg, "nativemem_1t_val_cache", None)
        )
        if not cache_dir:
            raise ValueError("NativeMEM requires a manifest-specific pre-merger cache")
        return Qwen35NativeMEMV2Dataset(
            manifest_path, cache_dir, stage=cfg.nativemem_stage,
            history_length=cfg.nativemem_history_length,
            stage1_frames=cfg.nativemem_stage1_frames,
            seed=cfg.seed,
            memory_cache_dir=memory_cache_dir,
            single_stage=bool(getattr(cfg, "nativemem_single_stage", False)),
            relative_state_mode=getattr(cfg, "qwen35_relative_state_mode", "none"),
            relative_state_cache_dir=getattr(
                cfg, "qwen35_relative_state_cache", None
            ),
        )
    if getattr(cfg, "use_h16_frozen_pool", False):
        from pathlib import Path
        from aerial_openvla.data.h16_frozen_pool_dataset import H16FrozenPoolDataset
        is_train = Path(manifest_path).resolve() == Path(cfg.train_manifest).resolve()
        cache_dir = (
            getattr(cfg, "h16_frozen_pool_train_cache", None)
            if is_train else getattr(cfg, "h16_frozen_pool_val_cache", None)
        )
        if not cache_dir:
            raise ValueError("H16-1T-FrozenPool requires a manifest-specific cache")
        return H16FrozenPoolDataset(
            manifest_path, cache_dir,
            history_length=getattr(cfg, "h16_frozen_pool_history_length", 16),
        )
    if getattr(cfg, "use_native1t_postfusion", False):
        from pathlib import Path
        from aerial_openvla.data.native1t_postfusion_dataset import (
            Native1TStage1Dataset, Native1TStage2Dataset,
        )
        if getattr(cfg, "native1t_stage", 2) == 1:
            return Native1TStage1Dataset(
                manifest_path,
                seed=cfg.seed,
                use_relative_state_text=getattr(
                    cfg, "use_relative_state_text", False
                ),
                use_relative_planar_state_text=getattr(
                    cfg, "use_relative_planar_state_text", False
                ),
            )
        is_train = Path(manifest_path).resolve() == Path(cfg.train_manifest).resolve()
        cache_dir = (
            getattr(cfg, "native1t_train_cache", None)
            if is_train else getattr(cfg, "native1t_val_cache", None)
        )
        if not cache_dir:
            raise ValueError("Native1T Stage 2 requires a manifest-specific cache")
        tokenizer_checkpoint = (
            getattr(cfg, "native1t_stage1_checkpoint", None)
            or cfg.init_checkpoint
        )
        if not tokenizer_checkpoint:
            raise ValueError("Native1T Stage 2 requires the Stage-1 init checkpoint")
        event_history_dir = (
            getattr(cfg, "native1t_event_train_indices", None)
            if is_train else getattr(cfg, "native1t_event_val_indices", None)
        )
        current_vision_cache_dir = None
        if cfg.model_type in {"qwen35_native", "qwen35_nativemem"}:
            current_vision_cache_dir = (
                getattr(cfg, "qwen35_premerger_train_cache", None)
                if is_train else getattr(cfg, "qwen35_premerger_val_cache", None)
            )
        return Native1TStage2Dataset(
            manifest_path,
            cache_dir,
            tokenizer_checkpoint,
            feature_dim=(
                1024 if getattr(cfg, "model_type", None) == "qwen35_native"
                else 2176
            ),
            tokenizer_filename=(
                "qwen35_native1t.pt"
                if getattr(cfg, "model_type", None) == "qwen35_native"
                else "native1t_postfusion.pt"
            ),
            history_length=getattr(cfg, "native1t_history_length", 16),
            history_mode=getattr(cfg, "native1t_history_mode", "fifo"),
            event_history_dir=event_history_dir,
            current_vision_cache_dir=current_vision_cache_dir,
            current_vision_cache_contract=getattr(
                cfg, "qwen35_premerger_cache_contract", None
            ),
            event_tau_local=getattr(
                cfg, "native1t_event_tau_local", 0.757488
            ),
            event_tau_anchor=getattr(
                cfg, "native1t_event_tau_anchor", 0.765
            ),
            event_max_length=getattr(
                cfg, "native1t_event_max_length", 8
            ),
            event_recent_protected=getattr(
                cfg, "native1t_event_recent_protected", 4
            ),
            action_chunk_horizon=getattr(cfg, "action_chunk_horizon", 1),
            use_action_history=getattr(cfg, "use_native1t_action_history", False),
            use_relative_state_text=getattr(cfg, "use_relative_state_text", False),
            use_relative_planar_state_text=getattr(
                cfg, "use_relative_planar_state_text", False
            ),
        )
    if getattr(cfg, "use_omtrack", False):
        from pathlib import Path
        from aerial_openvla.data.omtrack_dataset import OmTrackDataset

        is_train = Path(manifest_path).resolve() == Path(cfg.train_manifest).resolve()
        cache_dir = (
            getattr(cfg, "omtrack_train_sparse_cache", None)
            if is_train
            else getattr(cfg, "omtrack_val_sparse_cache", None)
        )
        if not cache_dir:
            raise ValueError("OmTrack requires a manifest-specific sparse cache directory")
        return OmTrackDataset(
            manifest_path,
            cache_dir,
            history_length=getattr(cfg, "omtrack_history_length", 31),
            history_tokens_per_frame=getattr(
                cfg, "omtrack_history_tokens_per_frame", 4
            ),
            action_chunk_horizon=getattr(cfg, "action_chunk_horizon", 8),
        )
    input_mode = getattr(cfg, "input_mode", "image")
    if input_mode == "image":
        if (
            getattr(cfg, "model_type", None) == "qwen35_native"
            and getattr(cfg, "qwen35_relative_state_mode", "none") != "none"
        ):
            from aerial_openvla.data.qwen35_relative_dataset import (
                Qwen35RelativeAppendPlanarClassificationDataset,
            )

            cache_dir = getattr(cfg, "qwen35_relative_state_cache", None)
            if not cache_dir:
                raise ValueError(
                    "Qwen3.5 append-planar Relative requires qwen35_relative_state_cache"
                )
            return Qwen35RelativeAppendPlanarClassificationDataset(
                manifest_path, cache_dir
            )
        if (
            getattr(cfg, "model_type", None) == "qwen35_native"
            and getattr(cfg, "use_relative_state_text", False)
        ):
            if getattr(cfg, "use_relative_planar_state_text", False):
                raise NotImplementedError(
                    "Single-frame Qwen3.5 Relative currently supports heading/altitude only"
                )
            from aerial_openvla.data.qwen35_relative_dataset import (
                Qwen35RelativeClassificationDataset,
            )

            return Qwen35RelativeClassificationDataset(manifest_path)
        from aerial_openvla.data.dataset import AerialVLNClassificationDataset

        return AerialVLNClassificationDataset(manifest_path)
    if input_mode == "multiframe_image":
        from aerial_openvla.data.multiframe_dataset import (
            EpisodeSequenceDataset,
            MultiFrameClassificationDataset,
            TaskGRUEpisodeStreamDataset,
        )

        num_frames = getattr(cfg, "num_frames", 4)
        history_stride = getattr(cfg, "frame_history_stride", 2)
        sequence_length = getattr(cfg, "sequence_length", 1)
        missing_history = getattr(cfg, "missing_history", "error")
        visual_history_mode = getattr(cfg, "visual_history_mode", "none")
        progress_annotations_dir = (
            getattr(cfg, "progress_annotations_dir", None)
            if getattr(cfg, "use_progress_aux_head", False)
            else None
        )
        if sequence_length > 1:
            if visual_history_mode != "none":
                raise ValueError("Recent-3/Keyframe-3 use action-only single-step training")
            if (
                getattr(cfg, "task_state_use_gru", False)
                and getattr(cfg, "task_gru_training_mode", "window")
                == "episode_stream"
            ):
                return TaskGRUEpisodeStreamDataset(
                    manifest_path,
                    num_frames=num_frames,
                    tbptt_steps=getattr(cfg, "task_gru_tbptt_steps", 4),
                    history_stride=history_stride,
                    missing_history=missing_history,
                    progress_annotations_dir=progress_annotations_dir,
                )
            return EpisodeSequenceDataset(
                manifest_path,
                num_frames=num_frames,
                sequence_length=sequence_length,
                history_stride=history_stride,
                missing_history=missing_history,
                progress_annotations_dir=progress_annotations_dir,
                burn_in=getattr(cfg, "task_gru_burn_in", 0),
                loss_window=(
                    getattr(cfg, "task_gru_loss_window", 0) or None
                ),
            )
        return MultiFrameClassificationDataset(
            manifest_path,
            num_frames=num_frames,
            history_stride=history_stride,
            missing_history=missing_history,
            visual_history_mode=visual_history_mode,
            progress_annotations_dir=progress_annotations_dir,
            action_chunk_horizon=getattr(cfg, "action_chunk_horizon", 1),
        )
    if input_mode == "feature":
        return _create_feature_pipeline(cfg).dataset
    raise ValueError(f"Unsupported input_mode={input_mode!r}")


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _create_image_pipeline(cfg: TrainConfig) -> DataPipeline:
    """Standard pipeline: read images from disk, process with VLM processor."""
    if cfg.model_type == "qwen35_native":
        from aerial_openvla.data.qwen35_native_collator import Qwen35NativeClassificationCollator

        processor = _cached_processor(cfg)
        dataset = create_dataset(cfg, cfg.train_manifest)
        return DataPipeline(
            dataset=dataset,
            collator=Qwen35NativeClassificationCollator(
                processor,
                prompt_template=cfg.prompt_template,
                max_text_tokens=cfg.max_text_tokens,
                use_relative_state_text=getattr(
                    cfg, "use_relative_state_text", False
                ),
                qwen35_relative_state_mode=getattr(
                    cfg, "qwen35_relative_state_mode", "none"
                ),
            ),
        )

    from aerial_openvla.data.collator import OpenVLAClassificationCollator

    processor = _cached_processor(cfg)
    dataset = create_dataset(cfg, cfg.train_manifest)
    collator = OpenVLAClassificationCollator(
        processor,
        prompt_template=cfg.prompt_template,
        max_text_tokens=cfg.max_text_tokens,
    )
    return DataPipeline(dataset=dataset, collator=collator)


def _create_qwen35_native_taskgru_pipeline(cfg: TrainConfig) -> DataPipeline:
    """Pure current-RGB Qwen3.5 SegTS-v6 TaskGRU pipeline.

    ``multiframe_image`` here selects the existing episode-contiguous sampler;
    ``num_frames=1`` keeps each model call strictly single-frame.
    """
    if cfg.input_mode != "multiframe_image" or cfg.num_frames != 1:
        raise ValueError("Qwen3.5 SegTS-v6 TaskGRU requires multiframe_image with num_frames=1")
    if cfg.sequence_length != cfg.task_gru_tbptt_steps:
        raise ValueError("Qwen3.5 TaskGRU sequence_length must equal task_gru_tbptt_steps")
    if cfg.use_native1t_postfusion or cfg.qwen35_relative_state_mode != "none":
        raise ValueError("Qwen3.5 SegTS-v6 is pure native single-frame, without H16/NativeMEM")
    if cfg.use_relative_state_text or cfg.use_relative_planar_state_text:
        raise ValueError("Qwen3.5 SegTS-v6 test must not add Relative state text")
    if not cfg.task_state_use_gru or not cfg.task_state_position_key:
        raise ValueError("Qwen3.5 SegTS-v6 requires TaskGRU and PosKey")
    from aerial_openvla.data.qwen35_native_collator import (
        Qwen35NativeClassificationCollator,
        Qwen35NativeTaskGRUEpisodeCollator,
    )

    processor = _cached_processor(cfg)
    per_step = Qwen35NativeClassificationCollator(
        processor,
        prompt_template=cfg.prompt_template,
        max_text_tokens=cfg.max_text_tokens,
        include_segment_token_ids=True,
        task_state_max_segments=cfg.task_state_max_segments,
        task_state_metadata_path=cfg.task_state_metadata_path,
    )
    return DataPipeline(
        dataset=create_dataset(cfg, cfg.train_manifest),
        collator=Qwen35NativeTaskGRUEpisodeCollator(
            per_step, sequence_length=cfg.sequence_length
        ),
    )


def _create_multiframe_pipeline(cfg: TrainConfig) -> DataPipeline:
    """Multi-frame pipeline: load F frames per step for AirMiniVLA."""
    from aerial_openvla.data.multiframe_dataset import (
        EpisodeSequenceCollator,
        MultiFrameCollator,
    )

    num_frames = getattr(cfg, "num_frames", 4)
    processor = _cached_processor(cfg)
    collator = MultiFrameCollator(
        processor,
        prompt_template=cfg.prompt_template,
        max_text_tokens=cfg.max_text_tokens,
        instruction_only_text_budget=getattr(
            cfg, "instruction_only_text_budget", False
        ),
        num_frames=num_frames,
        include_instruction_token_mask=(
            getattr(cfg, "use_instruction_local_pooler", False)
            or getattr(cfg, "use_motion_instruction_attention", False)
            or getattr(cfg, "task_state_mode", "none") != "none"
            or (
                getattr(cfg, "use_progress_aux_head", False)
                and getattr(cfg, "progress_mode", "direct") == "text_query"
            )
            or (
                getattr(cfg, "use_hierarchical_stop", False)
                and getattr(cfg, "hierarchical_stop_mode", "navigation") == "text_query"
            )
        ),
        include_segment_token_ids=(
            getattr(cfg, "task_state_mode", "none") != "none"
        ),
        task_state_max_segments=getattr(cfg, "task_state_max_segments", 10),
        task_state_metadata_path=getattr(cfg, "task_state_metadata_path", None),
        use_previous_actions_text=getattr(cfg, "use_previous_actions_text", False),
        use_relative_state_text=getattr(cfg, "use_relative_state_text", False),
        use_relative_planar_state_text=getattr(
            cfg, "use_relative_planar_state_text", False
        ),
        action_chunk_horizon=getattr(cfg, "action_chunk_horizon", 1),
        visual_history_mode=getattr(cfg, "visual_history_mode", "none"),
    )
    sequence_length = getattr(cfg, "sequence_length", 1)
    if sequence_length > 1:
        dataset = create_dataset(cfg, cfg.train_manifest)
        return DataPipeline(
            dataset=dataset,
            collator=EpisodeSequenceCollator(collator, sequence_length=sequence_length),
        )
    dataset = create_dataset(cfg, cfg.train_manifest)
    return DataPipeline(dataset=dataset, collator=collator)


def _create_omtrack_pipeline(cfg: TrainConfig) -> DataPipeline:
    from aerial_openvla.data.omtrack_dataset import OmTrackCollator

    processor = _cached_processor(cfg)
    return DataPipeline(
        dataset=create_dataset(cfg, cfg.train_manifest),
        collator=OmTrackCollator(
            processor,
            prompt_template=cfg.prompt_template,
            max_text_tokens=cfg.max_text_tokens,
        ),
    )


def _create_h16_frozen_pool_pipeline(cfg: TrainConfig) -> DataPipeline:
    from aerial_openvla.data.h16_frozen_pool_dataset import H16FrozenPoolCollator
    processor = _cached_processor(cfg)
    return DataPipeline(
        dataset=create_dataset(cfg, cfg.train_manifest),
        collator=H16FrozenPoolCollator(
            processor, cfg.prompt_template, cfg.max_text_tokens
        ),
    )


def _create_native1t_pipeline(cfg: TrainConfig) -> DataPipeline:
    from aerial_openvla.data.native1t_postfusion_dataset import Native1TCollator
    if getattr(cfg, "native1t_segts_mode", "none") != "none":
        incompatible = {
            "sequence_length != 1": getattr(cfg, "sequence_length", 1) != 1,
            "class weighting": getattr(cfg, "class_weights_path", None) is not None,
            "inflection weighting": getattr(cfg, "inflection_weight_coef", 1.0) != 1.0,
        }
        enabled = [name for name, value in incompatible.items() if value]
        if enabled:
            raise ValueError(
                "Native1T SegTS requires step-wise unweighted CE: " + ", ".join(enabled)
            )
    processor = _cached_processor(cfg)
    return DataPipeline(
        dataset=create_dataset(cfg, cfg.train_manifest),
        collator=Native1TCollator(
            processor,
            cfg.prompt_template,
            cfg.max_text_tokens,
            cfg.native1t_stage,
            include_instruction_token_mask=getattr(cfg, "use_instruction_local_pooler", False),
            instruction_only_text_budget=getattr(
                cfg, "instruction_only_text_budget", False
            ),
            include_segment_token_ids=(
                getattr(cfg, "task_state_mode", "none") != "none"
            ),
            task_state_max_segments=getattr(cfg, "task_state_max_segments", 10),
            task_state_metadata_path=getattr(
                cfg, "task_state_metadata_path", None
            ),
            use_relative_state_text=getattr(cfg, "use_relative_state_text", False),
            use_relative_planar_state_text=getattr(
                cfg, "use_relative_planar_state_text", False
            ),
            action_chunk_horizon=getattr(cfg, "action_chunk_horizon", 1),
        ),
    )


def _create_qwen35_native1t_pipeline(cfg: TrainConfig) -> DataPipeline:
    from aerial_openvla.data.qwen35_native_collator import (
        Qwen35Native1TStage1Collator,
        Qwen35Native1TStage2Collator,
    )

    processor = _cached_processor(cfg)
    collator_class = (
        Qwen35Native1TStage1Collator
        if cfg.native1t_stage == 1
        else Qwen35Native1TStage2Collator
    )
    return DataPipeline(
        dataset=create_dataset(cfg, cfg.train_manifest),
        collator=collator_class(
            processor,
            prompt_template=cfg.prompt_template,
            max_text_tokens=cfg.max_text_tokens,
            use_relative_state_text=getattr(
                cfg, "use_relative_state_text", False
            ),
        ),
    )


def _create_qwen35_nativemem_pipeline(cfg: TrainConfig) -> DataPipeline:
    from aerial_openvla.data.qwen35_nativemem_v2_collator import Qwen35NativeMEMV2Collator
    processor = _cached_processor(cfg)
    return DataPipeline(
        dataset=create_dataset(cfg, cfg.train_manifest),
        collator=Qwen35NativeMEMV2Collator(
            processor, prompt_template=cfg.prompt_template,
            stage=cfg.nativemem_stage,
            max_text_tokens=cfg.max_text_tokens,
            single_stage=bool(getattr(cfg, "nativemem_single_stage", False)),
            relative_state_mode=getattr(
                cfg, "qwen35_relative_state_mode", "none"
            ),
        ),
    )


def _create_feature_pipeline(cfg: TrainConfig) -> DataPipeline:
    """Pre-extracted feature pipeline: read ResNet features from LMDB."""
    raise NotImplementedError(
        "Feature pipeline is not yet implemented. "
        "Use `input_mode: image` or `train_cma.py` for full-sequence CMA training."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_processor_cache: dict[str, Any] = {}


def _cached_processor(cfg: TrainConfig) -> Any:
    """One processor per model path, shared by every pipeline.

    AirMiniVLA's ``load_airminivla_processor`` is a pure forward to
    ``load_openvla_processor``, so a single loader (and cache key) serves both
    model families.
    """
    key = cfg.model_name_or_path
    if key not in _processor_cache:
        if cfg.model_type in {"qwen35_native", "qwen35_nativemem"}:
            from transformers import AutoProcessor

            _processor_cache[key] = AutoProcessor.from_pretrained(key, local_files_only=True)
        else:
            from aerial_openvla.models.openvla_classifier import load_openvla_processor

            _processor_cache[key] = load_openvla_processor(key)
    return _processor_cache[key]
