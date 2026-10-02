from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from aerial_openvla.core.actions import ACTION_NAMES, NUM_ACTIONS, validate_action_id
from aerial_openvla.data.schema import ClassificationRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_DIGIT_RE = re.compile(r"(\d+)")


def natural_key(path: Path) -> tuple[object, ...]:
    parts = _DIGIT_RE.split(path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def candidate_frame_dirs(raw_root: Path, episode_id: str, trajectory_id: str) -> list[Path]:
    identifiers = [episode_id, trajectory_id]
    candidates: list[Path] = []
    for identifier in identifiers:
        candidates.extend([
            raw_root / "aerialvln-s" / identifier / "rgb",
            raw_root / "aerialvln-s" / identifier / "RGB",
            raw_root / "aerialvln-s" / identifier,
            raw_root / identifier / "rgb",
            raw_root / identifier / "RGB",
            raw_root / identifier,
            raw_root / "episodes" / identifier / "rgb",
            raw_root / "trajectories" / identifier / "rgb",
        ])
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def list_frames(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    frames = [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(frames, key=natural_key)


def resolve_frames(raw_root: Path, episode_id: str, trajectory_id: str) -> tuple[Path, list[Path]]:
    for frame_dir in candidate_frame_dirs(raw_root, episode_id, trajectory_id):
        frames = list_frames(frame_dir)
        if frames:
            return frame_dir, frames
    raise FileNotFoundError(
        f"No RGB frame directory found for episode={episode_id!r}, trajectory={trajectory_id!r} under {raw_root}"
    )


def load_episodes(annotation_path: Path) -> list[dict]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(episodes, list):
        raise ValueError(f"Expected a list or {{'episodes': [...]}} in {annotation_path}")
    return episodes


def iter_records(
    annotation_path: Path,
    raw_root: Path,
    split: str,
    frame_offset: int,
    strict: bool,
) -> Iterable[ClassificationRecord]:
    for episode in load_episodes(annotation_path):
        episode_id = str(episode["episode_id"])
        trajectory_id = str(episode.get("trajectory_id", episode_id))
        instruction_container = episode.get("instruction", {})
        instruction = (
            instruction_container.get("instruction_text", "")
            if isinstance(instruction_container, dict)
            else str(instruction_container)
        )
        actions = episode.get("actions")
        if not isinstance(actions, list):
            message = f"Episode {episode_id} has no action list"
            if strict:
                raise ValueError(message)
            print(f"[skip] {message}")
            continue

        try:
            _, frames = resolve_frames(raw_root, episode_id, trajectory_id)
        except FileNotFoundError:
            if strict:
                raise
            print(f"[skip] missing frames for episode {episode_id}")
            continue

        required = len(actions) + frame_offset
        if len(frames) < required:
            message = (
                f"Episode {episode_id}: {len(actions)} actions require at least {required} frames "
                f"with frame_offset={frame_offset}, but found {len(frames)}"
            )
            if strict:
                raise ValueError(message)
            print(f"[skip] {message}")
            continue

        for step_id, action in enumerate(actions):
            action_id = validate_action_id(int(action))
            image_path = frames[step_id + frame_offset].resolve()
            yield ClassificationRecord(
                episode_id=episode_id,
                trajectory_id=trajectory_id,
                step_id=step_id,
                image=str(image_path),
                instruction=instruction,
                action_id=action_id,
                split=split,
                scene_id=episode.get("scene_id"),
            )


def write_jsonl(records: Iterable[ClassificationRecord], output_path: Path) -> Counter[int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[int] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            counts[record.action_id] += 1
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return counts


def write_class_weights(counts: Counter[int], output_path: Path) -> None:
    total = sum(counts.values())
    weights: list[float] = []
    for action_id in range(NUM_ACTIONS):
        count = counts.get(action_id, 0)
        weights.append(0.0 if count == 0 else total / (NUM_ACTIONS * count))
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "counts": {str(index): counts.get(index, 0) for index in range(NUM_ACTIONS)},
                "weights": weights,
                "action_names": ACTION_NAMES,
            },
            handle,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build step-level AerialVLN-S classification manifests")
    parser.add_argument("--annotations", type=Path, required=True, help="AerialVLN-S split JSON")
    parser.add_argument("--raw-root", type=Path, required=True, help="Root containing per-episode RGB frames")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest")
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        choices=(0, 1),
        help="0 uses frame[t] for action[t]; 1 uses frame[t+1]. Verify against your frame extractor.",
    )
    parser.add_argument("--skip-bad-episodes", action="store_true")
    parser.add_argument("--class-weights-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = iter_records(
        annotation_path=args.annotations,
        raw_root=args.raw_root,
        split=args.split,
        frame_offset=args.frame_offset,
        strict=not args.skip_bad_episodes,
    )
    counts = write_jsonl(records, args.output)
    print(f"Wrote {sum(counts.values())} samples to {args.output}")
    for action_id in range(NUM_ACTIONS):
        print(f"  {action_id} {ACTION_NAMES[action_id]:>13}: {counts.get(action_id, 0)}")
    weights_path = args.class_weights_output or args.output.with_suffix(".class_weights.json")
    write_class_weights(counts, weights_path)
    print(f"Wrote class weights to {weights_path}")


if __name__ == "__main__":
    main()
