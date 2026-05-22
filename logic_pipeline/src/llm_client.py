from __future__ import annotations

from datetime import datetime
from importlib import metadata
from pathlib import Path
import re
from threading import Thread
from typing import Protocol

from .config import PipelineConfig


class ChatModel(Protocol):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate assistant text from a system/user prompt pair."""


def create_chat_model(config: PipelineConfig) -> ChatModel:
    if config.model_provider == "huggingface":
        return HuggingFaceChatModel(config)
    if config.model_provider == "ollama":
        return OllamaChatModel(config)
    raise ValueError(f"Unsupported model_provider: {config.model_provider}")


class TraceWriter:
    """Writes LLM prompts and streamed outputs to terminal and a text file."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.path = Path(config.llm_trace_path) if config.llm_trace_path else None
        self.call_count = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def start_call(self, provider: str, model_name: str, system_prompt: str, user_prompt: str) -> int:
        self.call_count += 1
        call_id = self.call_count
        stamp = datetime.now().isoformat(timespec="seconds")
        text = (
            "\n"
            f"{'=' * 80}\n"
            f"LLM CALL {call_id} | {stamp} | provider={provider} | model={model_name}\n"
            f"{'=' * 80}\n"
            "[SYSTEM PROMPT]\n"
            f"{system_prompt}\n\n"
            "[USER PROMPT]\n"
            f"{user_prompt}\n\n"
            "[MODEL OUTPUT]\n"
        )
        self.write(text)
        return call_id

    def end_call(self, call_id: int) -> None:
        self.write(f"\n[END LLM CALL {call_id}]\n")

    def write(self, text: str) -> None:
        if not self.config.llm_live_trace:
            return
        if self.config.llm_trace_terminal:
            print(text, end="", flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text)
                f.flush()


class HuggingFaceChatModel:
    """
    Shared Hugging Face Transformers chat adapter.

    Qwen/Qwen3.5-4B is published as an image-text-to-text model, but the parser
    pipeline sends text-only chat messages. The processor chat template supports
    that text-only path while leaving room for future multimodal inputs.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.trace = TraceWriter(config)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        self._load()

        import torch

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        call_id = self.trace.start_call(
            "huggingface",
            self.config.model_name,
            system_prompt,
            user_prompt,
        )

        chat_template_kwargs = {
            "conversation": messages,
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": self.config.enable_thinking,
        }
        try:
            inputs = self.processor.apply_chat_template(**chat_template_kwargs)
        except TypeError as exc:
            if "enable_thinking" not in str(exc):
                raise
            chat_template_kwargs.pop("enable_thinking")
            inputs = self.processor.apply_chat_template(**chat_template_kwargs)
        inputs = inputs.to(self.model.device)

        resolved_temperature = self.config.temperature if temperature is None else temperature
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_new_tokens,
            "do_sample": resolved_temperature > 0,
        }
        if resolved_temperature > 0:
            generation_kwargs["temperature"] = resolved_temperature

        if self.config.llm_live_trace:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            def stream_generate():
                with torch.inference_mode():
                    self.model.generate(**inputs, **generation_kwargs, streamer=streamer)

            thread = Thread(
                target=stream_generate,
            )
            chunks = []
            thread.start()
            for chunk in streamer:
                chunks.append(chunk)
                self.trace.write(chunk)
            thread.join()

            self.trace.end_call(call_id)
            return "".join(chunks).strip()

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][prompt_length:]
        text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        self.trace.write(text)
        self.trace.end_call(call_id)
        return text

    def _load(self):
        if self.model is not None:
            return

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face backend requires transformers. Install requirements "
                "with `pip install -r requirements.txt`."
            ) from exc

        common_kwargs = {
            "trust_remote_code": self.config.hf_trust_remote_code,
            "local_files_only": self.config.hf_local_files_only,
        }
        model_kwargs = dict(common_kwargs)
        if self.config.hf_device_map is not None:
            model_kwargs["device_map"] = self.config.hf_device_map
        if self.config.hf_load_in_4bit:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit Hugging Face loading uses bitsandbytes and requires a CUDA GPU. "
                    "Run with `--no-4bit` or set hf_load_in_4bit=False if you need "
                    "CPU/full-precision loading."
                )
            self._validate_bitsandbytes()

            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "4-bit Hugging Face loading requires bitsandbytes support in transformers. "
                    "Install requirements with `pip install -r requirements.txt`."
                ) from exc

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._torch_dtype(
                    self.config.hf_bnb_4bit_compute_dtype,
                    default=torch.float16,
                ),
                bnb_4bit_quant_type=self.config.hf_bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.hf_bnb_4bit_use_double_quant,
            )
        elif self.config.hf_torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.config.hf_torch_dtype
        if self.config.hf_attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.hf_attn_implementation

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            **common_kwargs,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        if self.config.hf_device_map is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)

        self.model.eval()

    @staticmethod
    def _torch_dtype(value: str | None, *, default):
        if value is None or value == "auto":
            return default

        import torch

        aliases = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        try:
            return aliases[value.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported torch dtype: {value}") from exc

    @staticmethod
    def _validate_bitsandbytes() -> None:
        minimum = "0.46.1"
        try:
            installed = metadata.version("bitsandbytes")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "4-bit Hugging Face loading requires bitsandbytes>=0.46.1. "
                "Install it with `pip install -U \"bitsandbytes>=0.46.1\"`, "
                "or run `scripts/run_jsonl.py --no-4bit` to disable quantization."
            ) from exc

        if HuggingFaceChatModel._version_tuple(installed) < HuggingFaceChatModel._version_tuple(minimum):
            raise RuntimeError(
                f"4-bit Hugging Face loading requires bitsandbytes>={minimum}; "
                f"found bitsandbytes=={installed}. Upgrade with "
                f"`pip install -U \"bitsandbytes>={minimum}\"`, or run "
                "`scripts/run_jsonl.py --no-4bit` to disable quantization."
            )

    @staticmethod
    def _version_tuple(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in re.findall(r"\d+", value)[:3])


class OllamaChatModel:
    """Compatibility adapter for the previous Ollama backend."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = None
        self.trace = TraceWriter(config)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        self._load()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        call_id = self.trace.start_call(
            "ollama",
            self.config.model_name,
            system_prompt,
            user_prompt,
        )

        options = {
            "temperature": self.config.temperature if temperature is None else temperature,
            "seed": self.config.seed,
            "num_predict": max_new_tokens or self.config.max_new_tokens,
        }

        if self.config.llm_live_trace:
            chunks = []
            for part in self.client.chat(
                model=self.config.model_name,
                messages=messages,
                options=options,
                format="json",
                think=self.config.enable_thinking,
                stream=True,
            ):
                chunk = part.get("message", {}).get("content", "")
                if chunk:
                    chunks.append(chunk)
                    self.trace.write(chunk)
            self.trace.end_call(call_id)
            return "".join(chunks)

        response = self.client.chat(
            model=self.config.model_name,
            messages=messages,
            options=options,
            format="json",
            think=self.config.enable_thinking,
        )
        text = response["message"]["content"]
        self.trace.write(text)
        self.trace.end_call(call_id)
        return text

    def _load(self):
        if self.client is not None:
            return

        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError(
                "Ollama backend requires the ollama Python package. Install it or "
                "set model_provider='huggingface'."
            ) from exc

        self.client = Client(host=self.config.ollama_host)
