import pytest

from NEW_logic_pipeline.Stage_2.local_model_config import get_local_transformers_config
from NEW_logic_pipeline.Stage_2.model_backends import LocalTransformersConfig, LocalTransformersLLM


def test_default_local_config_uses_minicpm(monkeypatch):
    monkeypatch.delenv("ATOMIZER_MODEL", raising=False)
    monkeypatch.delenv("MINICPM_MODEL_ID", raising=False)

    config = get_local_transformers_config()

    assert config.mode == "minicpm_hf"
    assert config.minicpm_model_id == "openbmb/MiniCPM5-1B"
    assert config.max_new_tokens == 512
    assert config.temperature == 0.0


def test_qwen_local_config_uses_hf_folder(monkeypatch):
    monkeypatch.setenv("ATOMIZER_MODEL", "qwen_hf_4bit")
    monkeypatch.setenv("QWEN_HF_MODEL_PATH", r"D:\Exact logic parser\models\Qwen3.5-4B")
    monkeypatch.setenv("ATOMIZER_MAX_NEW_TOKENS", "128")
    monkeypatch.setenv("ATOMIZER_TEMPERATURE", "0.2")
    monkeypatch.setenv("ATOMIZER_TOP_P", "0.9")
    monkeypatch.setenv("ATOMIZER_DEVICE_MAP", "cpu")

    config = get_local_transformers_config()

    assert config.mode == "qwen_hf_4bit"
    assert config.qwen_model_path == r"D:\Exact logic parser\models\Qwen3.5-4B"
    assert config.max_new_tokens == 128
    assert config.temperature == 0.2
    assert config.top_p == 0.9
    assert config.device_map == "cpu"


def test_local_llm_selects_minicpm_backend(monkeypatch):
    monkeypatch.setattr(LocalTransformersLLM, "_load_model", lambda self: None)

    llm = LocalTransformersLLM(LocalTransformersConfig(mode="minicpm_hf"))

    assert llm.model_id_or_path == "openbmb/MiniCPM5-1B"
    assert llm.load_in_4bit is False


def test_local_llm_selects_qwen_4bit_backend(monkeypatch):
    monkeypatch.setattr(LocalTransformersLLM, "_load_model", lambda self: None)

    llm = LocalTransformersLLM(
        LocalTransformersConfig(
            mode="qwen_hf_4bit",
            qwen_model_path=r"D:\Exact logic parser\models\Qwen3.5-4B",
        )
    )

    assert llm.model_id_or_path == r"D:\Exact logic parser\models\Qwen3.5-4B"
    assert llm.load_in_4bit is True


def test_qwen_4bit_requires_model_path(monkeypatch):
    monkeypatch.setattr(LocalTransformersLLM, "_load_model", lambda self: None)

    with pytest.raises(ValueError, match="qwen_model_path is required"):
        LocalTransformersLLM(LocalTransformersConfig(mode="qwen_hf_4bit"))
