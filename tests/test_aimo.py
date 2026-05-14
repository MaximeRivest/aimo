import importlib

import pytest


SAMPLE_DATA = {
    "gpt-4o": {
        "provider": "openai",
        "context_window": 128000,
        "input_cost_per_token": 0.0000025,
        "output_cost_per_token": 0.00001,
        "supports_vision": True,
        "supports_function_calling": True,
        "capabilities": ["vision", "function_calling"],
    },
    "claude-3-5-sonnet-20240620": {
        "provider": "anthropic",
        "context_window": 200000,
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
    },
    "llama3-70b-8192": {
        "provider": "groq",
        "context_window": 8192,
    },
    "bedrock/us-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "provider": "bedrock",
        "context_window": 200000,
    },
}


@pytest.fixture()
def registry(monkeypatch):
    import aimo.models as models

    def fake_data(*, force_refresh=False, allow_network=True):
        return dict(SAMPLE_DATA)

    monkeypatch.setattr(models, "get_merged_model_data", fake_data)
    reg = models.AIMORegistry()
    reg.refresh(force=False, quiet=True, allow_network=False)
    return reg


def test_model_is_plain_string_with_metadata():
    from aimo.models import AIMOModel

    model = AIMOModel(
        "gpt-4o",
        provider="openai",
        context_window=128000,
        input_cost_per_token=0.0000025,
        output_cost_per_token=0.00001,
        id="upstream-id-field-does-not-break-property",
    )

    assert isinstance(model, str)
    assert str(model) == "gpt-4o"
    assert model.id == "gpt-4o"
    assert model.context == 128000
    assert model.input_price == 0.0000025
    assert "gpt-4o" in model.pretty()
    assert model.to_dict()["id"] == "gpt-4o"


def test_registry_builds_hierarchy_aliases_and_search(registry):
    assert str(registry.openai.gpt_4o) == "gpt-4o"
    assert registry.gpt4o is registry.openai.gpt_4o
    assert registry.anthropic.claude_3_5_sonnet_20240620.provider == "anthropic"
    assert registry.groq.llama3_70b_8192.context == 8192

    results = registry.search("gpt", provider="openai")
    assert results == [registry.openai.gpt_4o]


def test_bedrock_region_is_nested_when_region_is_present(registry):
    model = registry.bedrock.us_east_1.anthropic_claude_3_5_sonnet_20240620_v1_0
    assert str(model) == "bedrock/us-east-1/anthropic.claude-3-5-sonnet-20240620-v1:0"


def test_update_mutates_registry_in_place(monkeypatch):
    import aimo.models as models

    calls = []

    def fake_data(*, force_refresh=False, allow_network=True):
        calls.append(force_refresh)
        return dict(SAMPLE_DATA)

    monkeypatch.setattr(models, "get_merged_model_data", fake_data)
    reg = models.AIMORegistry()
    before = id(reg)
    returned = reg.update(force=True)

    assert id(returned) == before
    assert calls == [True]
    assert reg.gpt4o is reg.openai.gpt_4o


def test_package_level_getattr_delegates_to_global_registry(monkeypatch):
    import aimo
    import aimo.models as models

    def fake_data(*, force_refresh=False, allow_network=True):
        return dict(SAMPLE_DATA)

    monkeypatch.setattr(models, "get_merged_model_data", fake_data)
    aimo.aimo.refresh(force=False, quiet=True, allow_network=False)

    assert str(aimo.openai.gpt_4o) == "gpt-4o"
    assert aimo.gpt4o is aimo.openai.gpt_4o


def test_import_does_not_load_data(monkeypatch):
    import aimo.models as models

    def fail_if_called(*args, **kwargs):
        raise AssertionError("data should not load during module import")

    monkeypatch.setattr(models, "get_merged_model_data", fail_if_called)
    import aimo as package

    importlib.reload(package)
    assert package.__version__


def test_models_dev_failure_falls_back_to_litellm(monkeypatch):
    from aimo import data

    def fake_download(url, cache_path, *, force=False, allow_network=True):
        if url == data.LITELLM_URL:
            return {
                "gpt-4o": {
                    "litellm_provider": "openai",
                    "max_input_tokens": 128000,
                }
            }
        raise data.AIMODataError("models.dev unavailable")

    monkeypatch.setattr(data, "_download", fake_download)

    merged = data.get_merged_model_data()

    assert merged["gpt-4o"]["provider"] == "openai"
    assert merged["gpt-4o"]["context_window"] == 128000


def test_models_dev_provider_keyed_shape_is_parsed_with_limits_cost_and_provider_metadata():
    from aimo.data import _merge_data

    merged = _merge_data(
        {},
        {
            "anthropic": {
                "id": "anthropic",
                "name": "Anthropic",
                "api": "https://api.anthropic.com/v1",
                "models": {
                    "claude-3-5-sonnet-20240620": {
                        "id": "claude-3-5-sonnet-20240620",
                        "name": "Claude 3.5 Sonnet",
                        "attachment": True,
                        "reasoning": False,
                        "tool_call": True,
                        "temperature": True,
                        "release_date": "2024-06-20",
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                        "limit": {"context": 200000, "output": 8192},
                        "cost": {"input": 3, "output": 15},
                    }
                },
            }
        },
    )

    model = merged["claude-3-5-sonnet-20240620"]
    assert model["model_id"] == "claude-3-5-sonnet-20240620"
    assert model["provider"] == "anthropic"
    assert model["provider_name"] == "Anthropic"
    assert model["provider_logo"] == "https://models.dev/logos/anthropic.svg"
    assert model["context_window"] == 200000
    assert model["max_output_tokens"] == 8192
    assert model["input_cost_per_token"] == 0.000003
    assert model["output_cost_per_token"] == 0.000015
    assert "vision" in model["capabilities"]
    assert "function_calling" in model["capabilities"]


def test_models_dev_duplicate_model_ids_do_not_clobber_other_providers():
    from aimo.data import _merge_data

    merged = _merge_data(
        {},
        {
            "provider-a": {"models": {"same-id": {"id": "same-id", "limit": {"context": 1}}}},
            "provider-b": {"models": {"same-id": {"id": "same-id", "limit": {"context": 2}}}},
        },
    )

    assert merged["same-id"]["provider"] == "provider-a"
    assert merged["provider-b/same-id"]["provider"] == "provider-b"
    assert merged["provider-b/same-id"]["model_id"] == "same-id"


def test_lsp_stub_exposes_module_and_registry_attributes():
    from pathlib import Path

    stub = Path("aimo/__init__.pyi").read_text()

    assert "openai: _Openai" in stub
    assert "class _Openai:" in stub
    assert "gpt_4o: AIMOModel" in stub
    assert "class AIMORegistry:" in stub
    assert "    openai: _Openai" in stub
