"""
Data loading and merging logic for aimo.

Combines:
- LiteLLM model_prices_and_context_window.json
- models.dev API (https://models.dev/api.json)

Network access is deliberately opt-in at package import time. Public callers load data lazily,
and ``update()`` can force a refresh.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
MODELS_DEV_URL = "https://models.dev/api.json"

CACHE_DIR = Path(os.environ.get("AIMO_CACHE_DIR", Path.home() / ".cache" / "aimo"))
LITELLM_CACHE = CACHE_DIR / "litellm_models.json"
MODELS_DEV_CACHE = CACHE_DIR / "models_dev.json"


class AIMODataError(RuntimeError):
    """Raised when model metadata cannot be loaded."""


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _download(url: str, cache_path: Path, *, force: bool = False, allow_network: bool = True) -> Dict[str, Any]:
    """Load JSON from cache, optionally refreshing it from the network.

    If a forced refresh fails but stale cache exists, the stale cache is returned. That makes
    ``aimo.update()`` resilient to transient upstream failures while still surfacing a clear
    error when no usable data exists.
    """
    if cache_path.exists() and not force:
        return _read_json(cache_path)

    if not allow_network:
        raise AIMODataError(
            f"No cached data at {cache_path}. Run `python -c 'import aimo; aimo.update()'` "
            "while online to populate the cache."
        )

    request = Request(url, headers={"User-Agent": "aimo-registry/0.1 (+https://github.com/MaximeRivest/aimo)"})
    try:
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        if cache_path.exists():
            return _read_json(cache_path)
        raise AIMODataError(f"Could not load model metadata from {url}: {exc}") from exc

    _write_json(cache_path, data)
    return data


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _normalise_litellm_price(pricing: Any, names: Iterable[str]) -> Any:
    if not isinstance(pricing, Mapping):
        return None
    return _first_present(pricing, names)


def _normalise_models_dev_cost(cost: Any, names: Iterable[str]) -> Any:
    """Return models.dev pricing as cost per token.

    models.dev publishes ``cost.input``/``cost.output`` as dollars per 1M tokens. aimo's
    public ``input_price``/``output_price`` aliases follow LiteLLM and are dollars per token.
    """
    if not isinstance(cost, Mapping):
        return None
    value = _first_present(cost, names)
    if value is None:
        return None
    try:
        return float(value) / 1_000_000
    except (TypeError, ValueError):
        return value


def _iter_models_dev_models(payload: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield model dictionaries from known models.dev API shapes.

    Historically examples of this API have appeared both as ``{"models": [...]}`` and
    as provider-keyed maps. This reader accepts both and annotates provider-keyed entries
    with their provider when the entry omits it.
    """
    top_models = payload.get("models")
    if isinstance(top_models, list):
        for item in top_models:
            if isinstance(item, Mapping):
                yield dict(item)
        return
    if isinstance(top_models, Mapping):
        for model_id, item in top_models.items():
            if isinstance(item, Mapping):
                model = dict(item)
                model.setdefault("id", model_id)
                yield model
        return

    for provider_id, provider_data in payload.items():
        if not isinstance(provider_data, Mapping):
            continue
        models = provider_data.get("models")
        provider_name = provider_data.get("name") or provider_id
        provider_api = provider_data.get("api")
        models_dev_provider = {
            "provider_id": provider_id,
            "provider_name": provider_name,
            "provider_api": provider_api,
            "provider_logo": f"https://models.dev/logos/{provider_id}.svg",
        }
        if isinstance(models, list):
            for item in models:
                if isinstance(item, Mapping):
                    model = dict(item)
                    model.setdefault("provider", provider_id)
                    model.update({k: v for k, v in models_dev_provider.items() if v is not None})
                    yield model
        elif isinstance(models, Mapping):
            for model_id, item in models.items():
                if isinstance(item, Mapping):
                    model = dict(item)
                    model.setdefault("id", model_id)
                    model.setdefault("provider", provider_id)
                    model.update({k: v for k, v in models_dev_provider.items() if v is not None})
                    yield model


def _merge_key(merged: Mapping[str, Dict[str, Any]], model_id: str, provider: str) -> str:
    """Return a stable key without conflating same-id models from different providers."""
    existing = merged.get(model_id)
    if existing is None:
        return model_id
    existing_provider = str(existing.get("provider", "unknown"))
    if existing_provider in {provider, "unknown"} or provider == "unknown":
        return model_id
    return f"{provider}/{model_id}"


def _models_dev_capabilities(model: Mapping[str, Any]) -> list[str]:
    capabilities = list(model.get("capabilities", []) or [])
    for field, name in [
        ("attachment", "vision"),
        ("reasoning", "reasoning"),
        ("tool_call", "function_calling"),
        ("temperature", "temperature"),
        ("open_weights", "open_weights"),
    ]:
        if model.get(field) is True and name not in capabilities:
            capabilities.append(name)
    return capabilities


def _models_dev_limits(model: Mapping[str, Any]) -> Tuple[Any, Any]:
    limit = model.get("limit")
    if isinstance(limit, Mapping):
        return limit.get("context"), limit.get("output")
    return _first_present(model, ["context_window", "context", "max_tokens"]), None


def _merge_data(litellm: Dict[str, Any], models_dev: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Merge LiteLLM and models.dev data into a unique-keyed metadata map.

    The metadata field ``model_id`` is the actual AI SDK identifier. The dictionary key is
    usually the same, but may be ``provider/model_id`` when different providers expose the
    same model id.
    """
    merged: Dict[str, Dict[str, Any]] = {}

    for model_id, raw_info in litellm.items():
        if model_id == "sample_spec" or not isinstance(raw_info, Mapping):
            continue
        info = dict(raw_info)
        provider = info.get("litellm_provider") or info.get("provider") or "unknown"
        merged[model_id] = {
            **info,
            "model_id": model_id,
            "provider": provider,
            "context_window": _first_present(info, ["max_input_tokens", "max_tokens", "context_window"]),
            "input_cost_per_token": info.get("input_cost_per_token"),
            "output_cost_per_token": info.get("output_cost_per_token"),
            "supports_vision": bool(info.get("supports_vision", False)),
            "supports_function_calling": bool(info.get("supports_function_calling", False)),
            "supports_prompt_caching": bool(info.get("supports_prompt_caching", False)),
            "source": "litellm",
        }

    for model in _iter_models_dev_models(models_dev):
        model_id = model.get("id") or model.get("name")
        if not model_id:
            continue
        provider = model.get("provider") or model.get("provider_id") or "unknown"
        context_window, output_tokens = _models_dev_limits(model)
        pricing = model.get("pricing")
        cost = model.get("cost")
        normalised = {
            **model,
            "model_id": str(model_id),
            "provider": provider,
            "context_window": context_window,
            "max_output_tokens": output_tokens,
            "input_cost_per_token": _normalise_models_dev_cost(cost, ["input", "prompt"])
            or _normalise_litellm_price(pricing, ["input", "input_cost_per_token", "prompt"]),
            "output_cost_per_token": _normalise_models_dev_cost(cost, ["output", "completion"])
            or _normalise_litellm_price(pricing, ["output", "output_cost_per_token", "completion"]),
            "capabilities": _models_dev_capabilities(model),
            "modalities": model.get("modalities", []),
            "release_date": model.get("release_date") or model.get("released") or model.get("releaseDate"),
            "last_updated": model.get("last_updated"),
        }
        key = _merge_key(merged, str(model_id), str(provider))
        if key in merged:
            # Preserve LiteLLM's pricing/provider fields unless models.dev adds previously missing data.
            existing = merged[key]
            for key, value in normalised.items():
                if value not in (None, [], {}) or key not in existing:
                    existing.setdefault(key, value)
            existing.update(
                {
                    "capabilities": normalised.get("capabilities", existing.get("capabilities", [])),
                    "modalities": normalised.get("modalities", existing.get("modalities", [])),
                    "release_date": normalised.get("release_date", existing.get("release_date")),
                    "source": "merged",
                }
            )
        else:
            normalised["source"] = "models.dev"
            merged[key] = normalised

    return merged


def get_merged_model_data(*, force_refresh: bool = False, allow_network: bool = True) -> Dict[str, Dict[str, Any]]:
    """Return merged model data.

    LiteLLM is the required primary source. models.dev is optional enrichment: if it is
    unavailable and no models.dev cache exists, aimo still works with LiteLLM metadata.
    ``force_refresh=True`` refreshes upstream files. ``allow_network=False`` is
    deterministic/offline-only and raises ``AIMODataError`` if the required LiteLLM cache is
    absent.
    """
    litellm_data = _download(LITELLM_URL, LITELLM_CACHE, force=force_refresh, allow_network=allow_network)
    try:
        models_dev_data = _download(MODELS_DEV_URL, MODELS_DEV_CACHE, force=force_refresh, allow_network=allow_network)
    except AIMODataError:
        models_dev_data = {}
    return _merge_data(litellm_data, models_dev_data)
