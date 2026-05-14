"""
aimo - Smart AI Model Registry

A hierarchical model registry that combines LiteLLM and models.dev data.

Importing this package is intentionally cheap: model data is loaded lazily on first
registry access, and ``update()`` refreshes the local cache explicitly.
"""

from __future__ import annotations

from typing import Any, List

from .models import AIMOModel, AIMORegistry, aimo, search, update

__version__ = "0.1.2"
__all__ = ["aimo", "AIMOModel", "AIMORegistry", "update", "search"]


def __getattr__(name: str) -> Any:
    """Delegate provider/model attributes to the global registry.

    This makes the documented API work:

        import aimo
        model = aimo.openai.gpt_4o
    """
    return getattr(aimo, name)


def __dir__() -> List[str]:
    return sorted(set(__all__ + dir(aimo)))
