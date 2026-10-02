from __future__ import annotations

import copy
import importlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from aerial_openvla.core.actions import assert_matches_airvln, validate_action_id


class _DummyTokenizer:
    """Official AirVLN tokenizes instructions during environment construction.

    The OpenVLA policy consumes the raw instruction text directly, so a one-token placeholder is enough
    for AirVLN's internal observation formatting.
    """

    def encode_sentence(self, _: str) -> list[int]:
        return [0]


@contextmanager
def _temporary_argv(arguments: list[str]) -> Iterator[None]:
    original = sys.argv[:]
    sys.argv = [original[0], *arguments]
    try:
        yield
    finally:
        sys.argv = original


def prepare_airvln_runtime(project_root: Path, airvln_root: Path) -> Path:
    """Create AirVLN's compatibility tree under generated project output.

    AirVLN derives several legacy paths from ``<project_prefix>/DATA`` during
    import.  Keeping that tree below ``output/`` avoids recreating a top-level
    ``DATA`` directory while preserving its expected data and log locations.
    """
    runtime_root = project_root / "output" / "airvln_runtime"
    data_root = runtime_root / "DATA"
    data_root.mkdir(parents=True, exist_ok=True)
    # AirVLN reads annotations from DATA/data and pretrained encoder weights
    # (e.g. ddppo-models/gibson-2plus-resnet50.pth for the CMA depth encoder)
    # from DATA/models.
    for name in ("data", "models"):
        link = data_root / name
        upstream = airvln_root / "DATA" / name
        if link.exists() and not link.is_symlink():
            raise RuntimeError(f"AirVLN runtime path is not a symlink: {link}")
        if not link.exists() and upstream.is_dir():
            link.symlink_to(upstream, target_is_directory=True)
    return runtime_root


class AirVLNEvaluationAdapter:
    """Thin wrapper around the official AirVLN environment.

    It monkey-patches only the dataset loader so the evaluator can point at an explicit AerialVLN-S
    directory. Action execution and metrics remain inside the official AirVLN code.
    """

    def __init__(
        self,
        *,
        airvln_root: str | Path,
        project_root: str | Path,
        dataset_dir: str | Path,
        split: str,
        max_actions: int,
        simulator_port: int,
        eval_num: int = -1,
        image_height_rgb: int = 224,
        image_width_rgb: int = 224,
        reset_warmup_seconds: float = 0.0,
        seed: int = 7,
        episode_ids: set[str] | None = None,
    ) -> None:
        self.airvln_root = Path(airvln_root).resolve()
        self.project_root = Path(project_root).resolve()
        self.dataset_dir = Path(dataset_dir).resolve()
        self.split = split
        self.max_actions = max_actions
        self.simulator_port = simulator_port
        if reset_warmup_seconds < 0:
            raise ValueError("reset_warmup_seconds must be non-negative")
        self.reset_warmup_seconds = float(reset_warmup_seconds)
        self.seed = seed
        self.episode_ids = episode_ids

        if not self.airvln_root.is_dir():
            raise FileNotFoundError(f"AirVLN repository not found: {self.airvln_root}")
        split_path = self.dataset_dir / f"{split}.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"AerialVLN-S split not found: {split_path}")

        # AirVLN's logger derives its legacy DATA path during import.
        runtime_root = prepare_airvln_runtime(self.project_root, self.airvln_root)
        log_dir = runtime_root / "DATA" / "output" / "openvla_classifier" / "eval" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(self.airvln_root))

        synthetic_args = [
            "--project_prefix",
            str(runtime_root),
            "--name",
            "openvla_classifier",
            "--run_type",
            "eval",
            "--collect_type",
            "TF",
            "--batchSize",
            "1",
            "--maxAction",
            str(max_actions),
            "--simulator_tool_port",
            str(simulator_port),
            "--Image_Height_RGB",
            str(image_height_rgb),
            "--Image_Width_RGB",
            str(image_width_rgb),
            "--EVAL_NUM",
            "-1",
        ]
        with _temporary_argv(synthetic_args):
            param_module = importlib.import_module("src.common.param")
            env_module = importlib.import_module("src.vlnce_src.env")
            settings_module = importlib.import_module("airsim_plugin.airsim_settings")

            self._param_module = param_module
            self._env_module = env_module
            self._settings_module = settings_module
            assert_matches_airvln(settings_module.AirsimActions)

            def load_aerialvln_s(splits: list[str]) -> tuple[list[dict], dict]:
                episodes: list[dict] = []
                for split_spec in splits:
                    components = split_spec.split("@", maxsplit=1)
                    split_name = components[0]
                    limit = int(components[1]) if len(components) == 2 else -1
                    with (self.dataset_dir / f"{split_name}.json").open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    split_episodes = payload["episodes"] if isinstance(payload, dict) else payload
                    if self.episode_ids is not None:
                        split_episodes = [
                            episode
                            for episode in split_episodes
                            if str(episode.get("episode_id", "")) in self.episode_ids
                        ]
                    if limit > 0:
                        split_episodes = split_episodes[:limit]
                    episodes.extend(split_episodes)
                return episodes, {}

            env_module.load_my_datasets = load_aerialvln_s
            args = param_module.args
            args.project_prefix = str(runtime_root)
            args.run_type = "eval"
            args.collect_type = "TF"
            args.maxAction = int(max_actions)
            args.EVAL_NUM = -1
            args.batchSize = 1
            args.simulator_tool_port = int(simulator_port)
            args.Image_Height_RGB = int(image_height_rgb)
            args.Image_Width_RGB = int(image_width_rgb)
            args.tokenizer_use_bert = False
            args.ablate_rgb = False
            args.ablate_depth = False
            args.machines_info = [
                {
                    "MACHINE_IP": "127.0.0.1",
                    "SOCKET_PORT": int(simulator_port),
                    "MAX_SCENE_NUM": 16,
                    "open_scenes": [],
                }
            ]

            self.env = env_module.AirVLNENV(
                batch_size=1,
                split=split,
                seed=seed,
                tokenizer=_DummyTokenizer(),
                dataset_group_by_scene=True,
            )
        self.episodes = list(self.env.data)
        if eval_num > 0:
            self.episodes = self.episodes[:eval_num]
        self._current_index: int | None = None
        self._last_outputs: list[Any] | None = None

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def current_episode(self) -> dict:
        if self._current_index is None:
            raise RuntimeError("No episode is active")
        return self.episodes[self._current_index]

    @property
    def instruction(self) -> str:
        container = self.current_episode.get("instruction", {})
        if isinstance(container, dict):
            return str(container.get("instruction_text", ""))
        return str(container)

    @property
    def episode_id(self) -> str:
        return str(self.current_episode["episode_id"])

    def start_episode(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        if not 0 <= index < len(self.episodes):
            raise IndexError(index)
        self._current_index = index
        self.env.batch = [copy.deepcopy(self.episodes[index])]
        self.env.VectorEnvUtil.set_batch(self.env.batch)
        outputs = self.env.reset()
        # UE can return its first RGB frame before texture streaming, exposure,
        # and temporal rendering have settled—especially on a faster GPU.  Wait
        # briefly and fetch a fresh observation without advancing the vehicle.
        if self.reset_warmup_seconds > 0:
            time.sleep(self.reset_warmup_seconds)
            outputs = self.env.get_obs()
        self._last_outputs = outputs
        return self._decode_output(outputs[0])

    def step(self, action_id: int) -> tuple[Image.Image, dict[str, Any]]:
        action_id = validate_action_id(action_id)
        self.env.makeActions([action_id])
        outputs = self.env.get_obs()
        self._last_outputs = outputs
        return self._decode_output(outputs[0])

    def _decode_output(self, output: Any) -> tuple[Image.Image, dict[str, Any]]:
        observation, _reward, done, info = output
        rgb = observation.get("rgb")
        if rgb is None:
            raise RuntimeError("AirVLN observation has no RGB image")
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Unexpected RGB shape: {array.shape}")
        # AirVLN/AirSim returns BGR; AeroAct applies the same channel reversal in eval.py.
        image = Image.fromarray(array[..., ::-1].copy(), mode="RGB")
        decoded_info = dict(info)
        decoded_info["done"] = bool(decoded_info.get("done", done))
        decoded_info["episode_id"] = str(decoded_info.get("episode_id", self.episode_id))
        return image, decoded_info

    def close(self) -> None:
        if hasattr(self.env, "delete_VectorEnvUtil"):
            self.env.delete_VectorEnvUtil()

    def __enter__(self) -> "AirVLNEvaluationAdapter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
