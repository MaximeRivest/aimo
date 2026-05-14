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
    assert package.__version__ == "0.1.0"
