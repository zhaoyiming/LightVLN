from __future__ import annotations

import importlib
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer


def _check_versions() -> None:
    """Warn about known version incompatibilities with OpenVLA dependencies."""
    checks = {
        "peft": ((0, 8, 0), None),
        "torch": ((2, 1, 0), None),
        "transformers": ((4, 30, 0), None),
    }
    for pkg, (min_ver, _max_ver) in checks.items():
        try:
            mod = importlib.import_module(pkg)
            raw = getattr(mod, "__version__", "0.0.0")
            parts = tuple(int(x) for x in raw.split(".")[:3])
        except Exception:
            continue
        if min_ver is not None and parts < min_ver:
            warnings.warn(
                f"{pkg}=={raw} is below recommended minimum {'.'.join(map(str, min_ver))}. "
                f"Some features may not work."
            )


_check_versions()


@dataclass(slots=True)
class ClassifierBuildConfig:
    model_name_or_path: str
    num_actions: int = 8
    pooling: str = "last_token"
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    train_projector: bool = False
    freeze_vision: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention_2: bool = True
    bf16: bool = True


def register_openvla_auto_classes() -> None:
    """Register local OpenVLA classes when third_party/OpenVLA is installed editable."""
    # Import the focused HF port without importing Prismatic's optional RLDS
    # training stack, which is not installed in the classification runtime.
    import sys
    import types

    if "prismatic" not in sys.modules:
        package = types.ModuleType("prismatic")
        package.__path__ = [str(Path(__file__).resolve().parents[3] / "third_party" / "OpenVLA" / "prismatic")]
        sys.modules["prismatic"] = package
    try:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    except ImportError:
        return

    registrations = [
        (AutoConfig, "openvla", OpenVLAConfig),
        (AutoImageProcessor, OpenVLAConfig, PrismaticImageProcessor),
        (AutoProcessor, OpenVLAConfig, PrismaticProcessor),
        (AutoModelForVision2Seq, OpenVLAConfig, OpenVLAForActionPrediction),
    ]
    for registry, key, value in registrations:
        try:
            registry.register(key, value)
        except ValueError:
            pass


def load_openvla_processor(model_name_or_path: str) -> Any:
    """Load PrismaticProcessor by manually assembling from saved components.
    
    AutoProcessor.from_pretrained is unreliable for locally-converted Prismatic models
    because it may fall back to returning the raw tokenizer. This bypasses the issue.
    """
    import json
    from pathlib import Path

    # Prismatic stub (bypasses TensorFlow/RLDS imports)
    import sys, types
    pkg = types.ModuleType("prismatic")
    pkg.__path__ = [str(Path(__file__).resolve().parents[3] / "third_party" / "OpenVLA" / "prismatic")]
    sys.modules["prismatic"] = pkg

    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    
    model_dir = Path(model_name_or_path).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True
    )
    
    preprocessor_config = json.loads((model_dir / "preprocessor_config.json").read_text())
    image_processor = PrismaticImageProcessor(
        use_fused_vision_backbone=preprocessor_config["use_fused_vision_backbone"],
        image_resize_strategy=preprocessor_config["image_resize_strategy"],
        input_sizes=preprocessor_config["input_sizes"],
        interpolations=preprocessor_config.get("interpolations"),
        means=preprocessor_config["means"],
        stds=preprocessor_config["stds"],
    )
    # Restore internal transform params
    image_processor.tvf_resize_params = preprocessor_config["tvf_resize_params"]
    image_processor.tvf_crop_params = preprocessor_config["tvf_crop_params"]
    image_processor.tvf_normalize_params = preprocessor_config["tvf_normalize_params"]
    image_processor.tvf_do_letterbox = preprocessor_config["tvf_do_letterbox"]
    image_processor.tvf_letterbox_fill = preprocessor_config["tvf_letterbox_fill"]

    processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
    
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return processor

def load_openvla_backbone(
    model_name_or_path: str,
    *,
    device: torch.device,
    use_flash_attention_2: bool,
    bf16: bool,
) -> nn.Module:
    register_openvla_auto_classes()
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if bf16 else torch.float32,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        backbone = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    except (ImportError, RuntimeError) as exc:
        if not use_flash_attention_2:
            raise
        warnings.warn(f"FlashAttention load failed ({exc}); retrying with default attention backend")
        kwargs.pop("attn_implementation", None)
        backbone = AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
    return backbone.to(device)


class OpenVLAActionClassifier(nn.Module):
    """Use OpenVLA as a fused image-language encoder and predict one of N actions.

    Bypasses OpenVLA's continuous-action tokenizer. Calls the public components
    (vision_backbone, projector, language_model) using the same image-token insertion
    order as Prismatic/OpenVLA: BOS, projected vision patches, remaining text tokens.
    """

    def __init__(self, backbone: nn.Module, config: ClassifierBuildConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        if config.pooling != "last_token":
            raise ValueError("Only pooling='last_token' is implemented")

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        _required = ["vision_backbone", "projector", "language_model"]
        for attr in _required:
            if not hasattr(self.backbone, attr):
                raise AttributeError(
                    f"Loaded model has no `{attr}`; expected an OpenVLA/Prismatic model."
                )

        if not config.freeze_vision:
            for parameter in self.backbone.vision_backbone.parameters():
                parameter.requires_grad = True

        if config.train_projector:
            for parameter in self.backbone.projector.parameters():
                parameter.requires_grad = True

        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
                init_lora_weights="gaussian",
            )
            self.backbone.language_model = get_peft_model(self.backbone.language_model, lora_config)
            if hasattr(self.backbone.language_model, "print_trainable_parameters"):
                self.backbone.language_model.print_trainable_parameters()
        else:
            for parameter in self.backbone.language_model.parameters():
                parameter.requires_grad = True

        if config.gradient_checkpointing:
            language_model = self.backbone.language_model
            if hasattr(language_model, "gradient_checkpointing_enable"):
                language_model.gradient_checkpointing_enable()
            if hasattr(language_model, "enable_input_require_grads"):
                language_model.enable_input_require_grads()
            if hasattr(language_model, "config"):
                language_model.config.use_cache = False

        hidden_size = int(self.backbone.config.text_config.hidden_size)
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, config.num_actions),
        )
        self.action_head.float()

    @property
    def vision_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.vision_backbone.parameters())

    def _get_internal_decoder(self, language_model: nn.Module) -> nn.Module:
        """Robustly locate the inner transformer decoder across PEFT wrappers.

        Tries multiple known attribute chains (Llama, Mistral, Gemma, etc.) and
        falls back to calling get_decoder() if the model exposes it.
        """
        base = language_model.get_base_model() if isinstance(language_model, PeftModel) else language_model

        for attr in ("model", "transformer", "decoder"):
            candidate = getattr(base, attr, None)
            if candidate is not None and hasattr(candidate, "layers"):
                return candidate

        if hasattr(base, "get_decoder"):
            return base.get_decoder()

        raise AttributeError(
            f"Cannot locate internal decoder transformer inside {type(base).__name__}. "
            f"Please check the model structure."
        )

    def _build_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeddings = self.backbone.get_input_embeddings()(input_ids)

        pixel_values = pixel_values.to(dtype=input_embeddings.dtype)

        if self.vision_is_frozen:
            with torch.no_grad():
                image_patches = self.backbone.vision_backbone(pixel_values)
        else:
            image_patches = self.backbone.vision_backbone(pixel_values)
        projected_patches = self.backbone.projector(image_patches)

        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1], projected_patches, input_embeddings[:, 1:]], dim=1
        )
        patch_mask = torch.ones(
            (attention_mask.shape[0], projected_patches.shape[1]),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1
        )
        return multimodal_embeddings, multimodal_attention_mask

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        language_model = self.backbone.language_model
        decoder = self._get_internal_decoder(language_model)

        outputs = decoder(
            input_ids=None,
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        last_valid_index = multimodal_attention_mask.long().sum(dim=1) - 1
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch_index, last_valid_index]
        return self.action_head(pooled.float())

    def trainable_parameter_groups(
        self,
        *,
        lora_lr: float,
        head_lr: float,
        projector_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []

        head_params = [p for p in self.action_head.parameters() if p.requires_grad]
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})

        proj_params = [p for p in self.backbone.projector.parameters() if p.requires_grad]
        if proj_params:
            groups.append({"params": proj_params, "lr": projector_lr, "weight_decay": weight_decay})

        lm_params = [p for p in self.backbone.language_model.parameters() if p.requires_grad]
        if lm_params:
            groups.append({"params": lm_params, "lr": lora_lr, "weight_decay": weight_decay})

        if not groups:
            raise RuntimeError("No trainable parameters found")
        return groups

    def save_classifier_checkpoint(
        self,
        output_dir: str | Path,
        processor: Any,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(output / "processor")
        torch.save(self.action_head.state_dict(), output / "action_head.pt")
        if self.config.train_projector:
            torch.save(self.backbone.projector.state_dict(), output / "projector.pt")
        if isinstance(self.backbone.language_model, PeftModel):
            self.backbone.language_model.save_pretrained(output / "lora")
        torch.save(
            {
                "model_name_or_path": self.config.model_name_or_path,
                "num_actions": self.config.num_actions,
                "pooling": self.config.pooling,
                "train_projector": self.config.train_projector,
                "freeze_vision": self.config.freeze_vision,
            },
            output / "classifier_meta.pt",
        )
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), output / "scheduler.pt")
        if extra:
            torch.save(extra, output / "training_state.pt")

    def load_classifier_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
    ) -> dict[str, Any] | None:
        checkpoint = Path(checkpoint_dir)
        head_path = checkpoint / "action_head.pt"
        if not head_path.is_file():
            raise FileNotFoundError(head_path)
        self.action_head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        projector_path = checkpoint / "projector.pt"
        if projector_path.is_file():
            self.backbone.projector.load_state_dict(
                torch.load(projector_path, map_location="cpu", weights_only=True)
            )
        # LoRA adapter: without this, resume silently restarts the adapter from
        # its random initialisation.  Load in place when already PEFT-wrapped.
        lora_path = checkpoint / "lora"
        if lora_path.is_dir():
            if isinstance(self.backbone.language_model, PeftModel):
                load_result = self.backbone.language_model.load_adapter(
                    str(lora_path), adapter_name="default", is_trainable=True
                )
                unexpected = getattr(load_result, "unexpected_keys", None)
                if unexpected:
                    raise RuntimeError(
                        f"LoRA weights in {lora_path} did not match the wrapped model; "
                        f"refusing a silent partial load (unexpected keys: {sorted(unexpected)[:5]})"
                    )
            else:
                self.backbone.language_model = PeftModel.from_pretrained(
                    self.backbone.language_model, lora_path, is_trainable=True
                )
        extra = None
        opt_path = checkpoint / "optimizer.pt"
        if opt_path.is_file() and optimizer is not None:
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
        sched_path = checkpoint / "scheduler.pt"
        if sched_path.is_file() and scheduler is not None:
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu", weights_only=True))
        state_path = checkpoint / "training_state.pt"
        if state_path.is_file():
            extra = torch.load(state_path, map_location="cpu", weights_only=False)
        return extra


def build_classifier(
    config: ClassifierBuildConfig,
    *,
    device: torch.device,
    lora_adapter_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[Any, OpenVLAActionClassifier]:
    processor = load_openvla_processor(config.model_name_or_path)
    backbone = load_openvla_backbone(
        config.model_name_or_path,
        device=device,
        use_flash_attention_2=config.use_flash_attention_2,
        bf16=config.bf16,
    )

    build_config = config
    if lora_adapter_dir is not None:
        build_config = ClassifierBuildConfig(**{**asdict(config), "use_lora": False})
    model = OpenVLAActionClassifier(backbone, build_config).to(device)

    if lora_adapter_dir is not None:
        adapter_path = Path(lora_adapter_dir)
        if adapter_path.is_dir():
            model.backbone.language_model = PeftModel.from_pretrained(
                model.backbone.language_model, adapter_path, is_trainable=False
            )
    if checkpoint_dir is not None:
        model.load_classifier_checkpoint(checkpoint_dir)
    return processor, model
