#!/usr/bin/env python3
"""Distributed extraction of frozen pre-projector Native1T memory features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from aerial_openvla.core.config import TrainConfig, load_yaml_dataclass
from aerial_openvla.models.factory import create_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Frames(Dataset):
    def __init__(self, manifest: Path):
        self.manifest = manifest.resolve()
        # Cache extraction needs only image paths.  Retaining the repeated,
        # often very long instruction for 1.15M records multiplies host memory
        # across ranks and DataLoader workers for no benefit.
        self.image_paths: list[str] = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    image = str(json.loads(line)["image"])
                except Exception as exc:
                    raise ValueError(
                        f"Invalid image record at {self.manifest}:{line_number}"
                    ) from exc
                path = Path(image)
                if not path.is_absolute():
                    path = (self.manifest.parent / path).resolve()
                self.image_paths.append(str(path))
        if not self.image_paths:
            raise ValueError(f"Manifest contains no records: {self.manifest}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as source:
            return index, source.convert("RGB")


class StridedSampler(Sampler[int]):
    """Assign every manifest row to exactly one rank without padding."""

    def __init__(self, size: int, rank: int, world: int, start_offset: int = 0) -> None:
        self.size = int(size)
        self.rank = int(rank)
        self.world = int(world)
        self.start_offset = int(start_offset)

    def __iter__(self):
        return iter(range(
            self.rank + self.world * self.start_offset, self.size, self.world
        ))

    def __len__(self) -> int:
        return len(range(
            self.rank + self.world * self.start_offset, self.size, self.world
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        # Coordination carries no tensors; Gloo avoids creating an NCCL
        # communicator only at the final barrier after ranks finish at
        # different times.
        torch.distributed.init_process_group("gloo")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    cfg = load_yaml_dataclass(args.config, TrainConfig)
    if not cfg.use_native1t_postfusion or cfg.native1t_stage != 1:
        raise ValueError("Cache extraction requires a Native1T Stage-1 config")
    checkpoint = Path(args.checkpoint).resolve()
    qwen35_native = cfg.model_type == "qwen35_native"
    tokenizer_file = checkpoint / (
        "qwen35_native1t.pt" if qwen35_native else "native1t_postfusion.pt"
    )
    if not tokenizer_file.is_file():
        raise FileNotFoundError(tokenizer_file)
    processor, model = create_model(cfg, device)
    model.load_classifier_checkpoint(checkpoint)
    model.eval()
    model.native1t_memory_tokenizer.requires_grad_(False)
    feature_dim = int(model.native1t_memory_tokenizer.query.shape[-1])

    dataset = Frames(Path(cfg.train_manifest))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shard_indices_path = output / f"rank{rank:03d}.indices.npy"
    shard_features_path = output / f"rank{rank:03d}.features.npy"
    local_count = len(range(rank, len(dataset), world))
    write_offset = 0
    if args.resume and shard_indices_path.is_file() and shard_features_path.is_file():
        existing_indices = np.load(shard_indices_path, mmap_mode="r")
        existing_features = np.load(shard_features_path, mmap_mode="r")
        if existing_indices.shape != (local_count,):
            raise ValueError(f"Bad index shard shape: {existing_indices.shape}")
        if existing_features.shape != (local_count, feature_dim):
            raise ValueError(f"Bad feature shard shape: {existing_features.shape}")
        # Written rows contain their exact strided manifest index.  Locate the
        # first mismatch in chunks so a near-complete 5GB shard is not copied.
        for start in range(0, local_count, 100_000):
            stop = min(start + 100_000, local_count)
            expected = np.arange(
                rank + world * start, rank + world * stop, world, dtype=np.int64
            )
            mismatch = np.flatnonzero(np.asarray(existing_indices[start:stop]) != expected)
            if len(mismatch):
                write_offset = start + int(mismatch[0])
                break
            write_offset = stop
        del existing_indices, existing_features
        print(json.dumps({
            "rank": rank, "resume_rows": write_offset,
            "rank_total": local_count,
            "percent": round(100.0 * write_offset / local_count, 2),
        }), flush=True)
    elif args.resume and (shard_indices_path.is_file() or shard_features_path.is_file()):
        raise FileNotFoundError("Resume requires both index and feature shards")

    sampler = StridedSampler(len(dataset), rank, world, start_offset=write_offset)
    image_processor = getattr(processor, "image_processor", processor)

    def collate(rows):
        indices, images = zip(*rows)
        encoded = image_processor(list(images), return_tensors="pt")
        result = {
            "indices": torch.tensor(indices, dtype=torch.long),
            "pixel_values": encoded["pixel_values"],
        }
        if qwen35_native:
            if "image_grid_thw" not in encoded:
                raise RuntimeError("Qwen3.5 image processor omitted image_grid_thw")
            result["image_grid_thw"] = encoded["image_grid_thw"]
        return result

    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, collate_fn=collate,
    )
    shard_mode = "r+" if args.resume and shard_indices_path.is_file() else "w+"
    shard_indices = np.lib.format.open_memmap(
        shard_indices_path, mode=shard_mode, dtype=np.int64, shape=(local_count,)
    )
    shard_features = np.lib.format.open_memmap(
        shard_features_path, mode=shard_mode, dtype=np.float16,
        shape=(local_count, feature_dim),
    )
    initial_offset = write_offset
    with torch.inference_mode():
        extraction_started = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            indices = batch["indices"]
            pixels = batch["pixel_values"].to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if qwen35_native:
                    grid = batch["image_grid_thw"].to(device, non_blocking=True)
                    vision_output = model.backbone.model.get_image_features(
                        pixels, grid, return_dict=True
                    )
                    frame_features = vision_output.pooler_output
                    if len(frame_features) != len(indices):
                        raise RuntimeError(
                            "Qwen3.5 vision output count does not match cache batch"
                        )
                    token_counts = {int(features.shape[0]) for features in frame_features}
                    if len(token_counts) != 1:
                        raise RuntimeError("Qwen3.5 cache requires fixed 224x224 tokens")
                    raw = torch.stack(list(frame_features), dim=0)
                else:
                    raw = model.encode_frozen_vision(pixels)
                memory = model.native1t_memory_tokenizer(raw.unsqueeze(1)).squeeze(1)
            expected = (indices.shape[0], feature_dim)
            if tuple(memory.shape) != expected:
                raise RuntimeError(
                    f"Expected one [{feature_dim}] memory feature per RGB, got "
                    f"{tuple(memory.shape)} for indices {tuple(indices.shape)}"
                )
            count = indices.shape[0]
            shard_indices[write_offset:write_offset + count] = indices.numpy()
            shard_features[write_offset:write_offset + count] = (
                memory.to(dtype=torch.float16).cpu().numpy()
            )
            write_offset += count
            log_interval = 10 if qwen35_native else 100
            if batch_index % log_interval == 0 or write_offset == local_count:
                elapsed = max(time.perf_counter() - extraction_started, 1e-6)
                print(
                    json.dumps({
                        "rank": rank,
                        "rows": write_offset,
                        "rank_total": local_count,
                        "percent": round(100.0 * write_offset / local_count, 2),
                        "rows_per_second": round(
                            (write_offset - initial_offset) / elapsed,
                            2,
                        ),
                    }),
                    flush=True,
                )
    if write_offset != local_count:
        raise RuntimeError(f"Rank {rank} wrote {write_offset}/{local_count} rows")
    shard_indices.flush()
    shard_features.flush()
    del shard_indices, shard_features
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        features = np.lib.format.open_memmap(
            output / "features.npy", mode="w+", dtype=np.float16,
            shape=(len(dataset), feature_dim),
        )
        seen = np.zeros(len(dataset), dtype=np.uint8)
        for shard_rank in range(world):
            indices = np.load(output / f"rank{shard_rank:03d}.indices.npy", mmap_mode="r")
            shard = np.load(output / f"rank{shard_rank:03d}.features.npy", mmap_mode="r")
            for start in range(0, len(indices), 8192):
                stop = min(start + 8192, len(indices))
                chunk_indices = np.asarray(indices[start:stop])
                features[chunk_indices] = shard[start:stop]
                seen[chunk_indices] = 1
        features.flush()
        if not bool(seen.all()):
            raise RuntimeError(f"Native1T cache missing {(seen == 0).sum()} rows")
        metadata = {
            "complete": True,
            "shape": [len(dataset), feature_dim],
            "dtype": "float16",
            "feature_space": "frozen_memory_tokenizer_pre_projector",
            "manifest_path": str(dataset.manifest),
            "manifest_sha256": sha256(dataset.manifest),
            "tokenizer_checkpoint": str(checkpoint),
            "tokenizer_sha256": sha256(tokenizer_file),
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        for shard_rank in range(world):
            (output / f"rank{shard_rank:03d}.indices.npy").unlink()
            (output / f"rank{shard_rank:03d}.features.npy").unlink()
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
