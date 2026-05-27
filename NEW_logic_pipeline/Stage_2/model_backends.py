from __future__ import annotations

"""Local Hugging Face Transformers model backends for Stage 2 atomization."""

from dataclasses import dataclass
from typing import Any, Literal, Optional


ModelMode = Literal[
    "minicpm_hf",
    "qwen_hf_4bit",
]


@dataclass
class LocalTransformersConfig:
    mode: ModelMode

    # For MiniCPM:
    minicpm_model_id: str = "openbmb/MiniCPM5-1B"

    # For Qwen local HF folder:
    qwen_model_path: Optional[str] = None

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.95

    device_map: str = "auto"
    torch_dtype: str = "auto"


class LocalTransformersLLM:
    """
    Common local Transformers wrapper.

    Used by Stage 5 leaf_atomizer.py.

    Required interface:
        llm.generate(prompt: str) -> str
    """

    def __init__(self, config: LocalTransformersConfig):
        self.config = config

        if config.mode == "minicpm_hf":
            self.model_id_or_path = config.minicpm_model_id
            self.load_in_4bit = False

        elif config.mode == "qwen_hf_4bit":
            if not config.qwen_model_path:
                raise ValueError("qwen_model_path is required for qwen_hf_4bit mode.")
            self.model_id_or_path = config.qwen_model_path
            self.load_in_4bit = True

        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        self._load_model()

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path,
            trust_remote_code=True,
        )

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                trust_remote_code=True,
            )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
                trust_remote_code=True,
            )

        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        do_sample = self.config.temperature > 0

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generate_kwargs["temperature"] = self.config.temperature
            generate_kwargs["top_p"] = self.config.top_p

        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        input_len = inputs["input_ids"].shape[-1]
        new_tokens = outputs[0][input_len:]

        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()


def create_local_llm(config: LocalTransformersConfig) -> LocalTransformersLLM:
    return LocalTransformersLLM(config)


__all__ = [
    "LocalTransformersConfig",
    "LocalTransformersLLM",
    "ModelMode",
    "create_local_llm",
]
