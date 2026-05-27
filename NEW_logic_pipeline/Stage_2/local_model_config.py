from __future__ import annotations

"""Environment-driven local Transformers configuration for atomization."""

import os

from NEW_logic_pipeline.Stage_2.model_backends import LocalTransformersConfig


def get_local_transformers_config() -> LocalTransformersConfig:
    mode = os.getenv("ATOMIZER_MODEL", "minicpm_hf")

    if mode == "minicpm_hf":
        return LocalTransformersConfig(
            mode="minicpm_hf",
            minicpm_model_id=os.getenv("MINICPM_MODEL_ID", "openbmb/MiniCPM5-1B"),
            max_new_tokens=int(os.getenv("ATOMIZER_MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("ATOMIZER_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("ATOMIZER_TOP_P", "0.95")),
            device_map=os.getenv("ATOMIZER_DEVICE_MAP", "auto"),
        )

    if mode == "qwen_hf_4bit":
        return LocalTransformersConfig(
            mode="qwen_hf_4bit",
            qwen_model_path=os.environ["QWEN_HF_MODEL_PATH"],
            max_new_tokens=int(os.getenv("ATOMIZER_MAX_NEW_TOKENS", "512")),
            temperature=float(os.getenv("ATOMIZER_TEMPERATURE", "0.0")),
            top_p=float(os.getenv("ATOMIZER_TOP_P", "0.95")),
            device_map=os.getenv("ATOMIZER_DEVICE_MAP", "auto"),
        )

    raise ValueError(f"Unknown ATOMIZER_MODEL={mode}")


__all__ = ["get_local_transformers_config"]
