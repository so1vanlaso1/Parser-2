from dataclasses import dataclass
from typing import Literal


@dataclass
class PipelineConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    model_provider: Literal["huggingface", "ollama"] = "huggingface"

    # Ollama settings are kept for compatibility with older local runs.
    ollama_host: str = "http://localhost:11434"

    temperature: float = 0.0
    seed: int = 42
    max_new_tokens: int = 2048

    # LLM prompt/response tracing.
    llm_live_trace: bool = True
    llm_trace_terminal: bool = True
    llm_trace_path: str | None = "artifacts/llm_io.txt"

    # Hugging Face Transformers settings.
    hf_device_map: str | None = "auto"
    hf_torch_dtype: str | None = "auto"
    hf_trust_remote_code: bool = False
    hf_local_files_only: bool = False
    hf_attn_implementation: str | None = None
    hf_load_in_4bit: bool = True
    hf_bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    hf_bnb_4bit_compute_dtype: str = "float16"
    hf_bnb_4bit_use_double_quant: bool = True

    rag_top_k: int = 3
    max_repair_attempts: int = 2

    fail_on_unbound_variable: bool = True
    fail_on_unknown_node_type: bool = True
    fail_on_modal_as_negation: bool = True

    solver_target: str = "horn"  # "horn", "z3", "datalog"
