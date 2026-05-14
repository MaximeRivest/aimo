"""Core AIMOModel class and hierarchical registry builder."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .data import get_merged_model_data


class AIMOModel(str):
    """A model identifier that behaves like ``str`` and carries metadata.

    ``str(model)`` is always the provider model id, so instances can be passed to
    LiteLLM, DSPy, LangChain, SDKs, and any other API expecting a plain model string.
    Use ``repr(model)`` or ``model.pretty()`` for a human-readable metadata summary.
    """

    def __new__(cls, model_id: str, **metadata: Any) -> "AIMOModel":
        obj = super().__new__(cls, model_id)

        for key, value in metadata.items():
            # Keep string semantics and read-only properties intact. Provider metadata can
            # contain names such as "id"; those remain available through .raw/.to_dict().
            if key.isidentifier() and not hasattr(str, key) and not isinstance(getattr(cls, key, None), property):
                setattr(obj, key, value)

        obj.raw = dict(metadata)

        if not hasattr(obj, "context") and hasattr(obj, "context_window"):
            obj.context = obj.context_window
        if not hasattr(obj, "context") and hasattr(obj, "max_input_tokens"):
            obj.context = obj.max_input_tokens
        if not hasattr(obj, "input_price") and hasattr(obj, "input_cost_per_token"):
            obj.input_price = obj.input_cost_per_token
        if not hasattr(obj, "output_price") and hasattr(obj, "output_cost_per_token"):
            obj.output_price = obj.output_cost_per_token

        return obj

    @property
    def id(self) -> str:
        """Return the model id as a plain string."""
        return str.__str__(self)

    def __repr__(self) -> str:
        provider = getattr(self, "provider", "unknown")
        context = getattr(self, "context", None)
        return f"AIMOModel({self.id!r}, provider={provider!r}, context={context!r})"

    def __dir__(self) -> List[str]:
        """Make metadata attributes visible to IDE autocompletion."""
        return sorted(set(super().__dir__() + list(self.__dict__.keys())))

    def to_dict(self) -> Dict[str, Any]:
        """Return model metadata plus the canonical ``id`` field."""
        result = dict(self.raw)
        result["id"] = self.id
        result.setdefault("provider", getattr(self, "provider", None))
        result.setdefault("context", getattr(self, "context", None))
        result.setdefault("input_price", getattr(self, "input_price", None))
        result.setdefault("output_price", getattr(self, "output_price", None))
        return result

    def pretty(self) -> str:
        """Return a human-readable summary without changing string semantics."""
        provider = str(getattr(self, "provider", "unknown")).upper()
        context = getattr(self, "context", None)
        input_p = getattr(self, "input_price", None)
        output_p = getattr(self, "output_price", None)

        parts = [self.id, f"({provider})"]
        if isinstance(context, int) and context > 0:
            parts.append(f"{context // 1000}k context")
        if input_p is not None and output_p is not None:
            parts.append(f"${input_p * 1_000_000:.2f} / ${output_p * 1_000_000:.2f} per 1M tokens")
        return " • ".join(parts)


class AIMONamespace(SimpleNamespace):
    """Namespace node used for provider and nested provider attributes."""

    def __dir__(self) -> List[str]:
        return sorted(self.__dict__)

    def __repr__(self) -> str:
        names = ", ".join(sorted(self.__dict__)[:8])
        suffix = "..." if len(self.__dict__) > 8 else ""
        return f"AIMONamespace({names}{suffix})"


class AIMORegistry(AIMONamespace):
    """Lazy, refreshable model registry.

    Accessing a provider/model attribute loads cached metadata, or downloads it on first use
    when the cache is absent. Importing the package never performs network I/O.
    """

    def __init__(self) -> None:
        super().__init__()
        self._loaded = False
        self._model_data: Dict[str, Dict[str, Any]] = {}
        self._models_by_id: Dict[str, AIMOModel] = {}

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.refresh(force=False, quiet=True)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_") or name in {
            "refresh",
            "update",
            "search",
            "models",
            "model_ids",
            "to_dict",
            "_ensure_loaded",
            "__dict__",
            "__class__",
        }:
            return object.__getattribute__(self, name)
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            self._ensure_loaded()
            return object.__getattribute__(self, name)

    def __dir__(self) -> List[str]:
        self._ensure_loaded()
        return sorted(k for k in self.__dict__ if not k.startswith("_"))

    def refresh(self, *, force: bool = True, quiet: bool = False, allow_network: bool = True) -> "AIMORegistry":
        data = get_merged_model_data(force_refresh=force, allow_network=allow_network)
        self.__dict__.clear()
        self._loaded = True
        self._model_data = data
        self._models_by_id = {}
        _populate_registry(self, data)
        _attach_aliases(self, data)
        if not quiet:
            print("aimo model registry updated successfully.")
        return self

    def update(self, force: bool = True) -> "AIMORegistry":
        """Refresh cached upstream metadata and update this registry in place."""
        return self.refresh(force=force, quiet=False, allow_network=True)

    def search(self, query: str = "", provider: Optional[str] = None, limit: int = 20) -> List[AIMOModel]:
        """Search by model id and optional provider."""
        self._ensure_loaded()
        q = query.lower()
        matches: List[AIMOModel] = []
        for model_id, model in self._models_by_id.items():
            if provider and str(getattr(model, "provider", "")).lower() != provider.lower():
                continue
            if q and q not in model_id.lower():
                continue
            matches.append(model)
            if len(matches) >= limit:
                break
        return matches

    def models(self) -> List[AIMOModel]:
        """Return all known model objects."""
        self._ensure_loaded()
        return list(self._models_by_id.values())

    def model_ids(self) -> List[str]:
        """Return all known model ids."""
        self._ensure_loaded()
        return list(self._models_by_id)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Return the raw merged metadata keyed by model id."""
        self._ensure_loaded()
        return dict(self._model_data)


ALIASES = {
    "gpt4o": "gpt-4o",
    "gpt4omini": "gpt-4o-mini",
    "claude35": "claude-3-5-sonnet-20240620",
    "claude37": "claude-3-7-sonnet-20250219",
    "sonnet": "claude-3-5-sonnet-20240620",
    "opus": "claude-3-opus-20240229",
    "haiku": "claude-3-haiku-20240307",
    "o1": "o1",
    "o3": "o3",
    "llama3": "llama3-70b-8192",
    "llama31": "llama-3.1-70b-8192",
}


def _sanitize(name: str) -> str:
    """Convert arbitrary provider/model names to valid Python identifiers."""
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", name).strip("_").lower()
    if not safe:
        safe = "model"
    if safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def _unique_attr(namespace: AIMONamespace, preferred: str, model_id: str) -> str:
    existing = getattr(namespace, preferred, None)
    if existing is None or existing == model_id:
        return preferred
    suffix = _sanitize(model_id)[-12:] or "alt"
    candidate = f"{preferred}_{suffix}"
    counter = 2
    while hasattr(namespace, candidate):
        candidate = f"{preferred}_{suffix}_{counter}"
        counter += 1
    return candidate


def _model_metadata(model_id: str, info: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(info)
    metadata.setdefault("provider", info.get("provider", "unknown"))
    metadata.setdefault("context", info.get("context_window") or info.get("max_input_tokens"))
    metadata.setdefault("input_price", info.get("input_cost_per_token"))
    metadata.setdefault("output_price", info.get("output_cost_per_token"))
    metadata.setdefault("supports_vision", bool(info.get("supports_vision", False)))
    metadata.setdefault("supports_function_calling", bool(info.get("supports_function_calling", False)))
    metadata.setdefault("capabilities", info.get("capabilities", []))
    metadata["id"] = model_id
    return metadata


def _target_namespace(root: AIMORegistry, provider: str, model_id: str) -> AIMONamespace:
    safe_provider = _sanitize(provider)
    provider_ns = getattr(root, safe_provider, None)
    if provider_ns is None:
        provider_ns = AIMONamespace()
        setattr(root, safe_provider, provider_ns)

    # Support common Bedrock LiteLLM ids such as bedrock/us-east-1/... without assuming
    # all Bedrock ids have a region. Non-region ids remain directly below provider.
    parts = model_id.split("/")
    if provider in {"bedrock", "bedrock_converse"} and len(parts) >= 3 and re.match(r"^[a-z]{2}-[a-z]+-\d$", parts[1]):
        region_name = _sanitize(parts[1])
        region_ns = getattr(provider_ns, region_name, None)
        if region_ns is None:
            region_ns = AIMONamespace()
            setattr(provider_ns, region_name, region_ns)
        return region_ns
    return provider_ns


def _leaf_name(provider: str, model_id: str) -> str:
    if provider in {"bedrock", "bedrock_converse"}:
        return _sanitize(model_id.split("/")[-1])
    return _sanitize(model_id.split("/")[-1] if "/" in model_id else model_id)


def _populate_registry(root: AIMORegistry, data: Dict[str, Dict[str, Any]]) -> None:
    for model_id, info in data.items():
        provider = str(info.get("provider", "unknown"))
        model = AIMOModel(model_id, **_model_metadata(model_id, info))
        root._models_by_id[model_id] = model
        target = _target_namespace(root, provider, model_id)
        leaf = _unique_attr(target, _leaf_name(provider, model_id), model_id)
        setattr(target, leaf, model)


def _attach_aliases(root: AIMORegistry, data: Dict[str, Dict[str, Any]]) -> None:
    for alias, target_id in ALIASES.items():
        model = root._models_by_id.get(target_id)
        if model is None:
            info = data.get(target_id)
            if info is None:
                continue
            model = AIMOModel(target_id, **_model_metadata(target_id, info))
            root._models_by_id[target_id] = model
        setattr(root, alias, model)


aimo = AIMORegistry()


def update(force: bool = True) -> AIMORegistry:
    """Refresh the global registry in place and return it."""
    return aimo.update(force=force)


def search(query: str = "", provider: Optional[str] = None, limit: int = 20) -> List[AIMOModel]:
    """Search the global registry."""
    return aimo.search(query=query, provider=provider, limit=limit)
