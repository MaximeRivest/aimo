# aimo

> **Proof of concept:** `aimo` is an experimental model-registry package. It is useful enough to try, but there is no clear long-term maintenance plan yet. Expect API changes, stale upstream metadata, and rough edges until the project proves its shape.

**Smart AI Model Registry** — Hierarchical access to AI model identifiers with rich metadata.

Combines metadata from [LiteLLM](https://github.com/BerriAI/litellm) and [models.dev](https://models.dev).

## Features

- ✨ Hierarchical access: `aimo.openai.gpt_4o`, `aimo.anthropic.claude_3_5_sonnet`
- 🧠 Rich objects that **act like strings**: pass them to DSPy, LiteLLM, LangChain, SDKs, etc.
- 📊 Metadata: context window, pricing, capabilities, vision support, and upstream raw fields
- 🔄 Explicit cache refresh from LiteLLM + models.dev via `aimo.update()`
- 🚀 Zero runtime dependencies; lazy startup with local caching
- 🧩 Static `.pyi` stubs for LSP autocomplete of common `aimo.openai.gpt_4o` paths

## Installation

```bash
pip install aimo-registry
```

## Usage

```python
import aimo

# Hierarchical access. The first registry access loads cached data, or downloads it
# if the cache has not been populated yet.
model = aimo.openai.gpt_4o
model = aimo.anthropic.claude_3_5_sonnet_20240620
model = aimo.groq.llama3_70b_8192

# Works as a string.
print(model)                    # "gpt-4o"
assert isinstance(model, str)

# Human-readable summary when you want one.
print(model.pretty())

# Common attributes from LiteLLM + models.dev are available when present.
print(model.context)            # e.g. 128000
print(model.input_price)        # e.g. 2.5e-06
print(model.output_price)       # e.g. 1e-05
print(model.supports_vision)    # e.g. True
print(model.capabilities)       # e.g. ['vision', 'function_calling']

# Full raw data.
print(list(model.raw.keys()))

# Use anywhere a string is expected.
# lm = dspy.LM(model)
# response = litellm.completion(model=model, messages=[...])
```

## Search & Aliases

```python
import aimo

# Search
models = aimo.search("claude")
models = aimo.search("gpt-4", provider="openai")
models = aimo.search(provider="groq")

# Popular aliases, when present in the upstream data
model = aimo.gpt4o
model = aimo.claude35
model = aimo.sonnet
model = aimo.haiku
model = aimo.o1
model = aimo.o3
```

## Updating the Model List

```python
import aimo

aimo.update()   # Force refresh from LiteLLM + models.dev and update the registry in place.
```

By default, `aimo` stores upstream JSON files in `~/.cache/aimo`. Set `AIMO_CACHE_DIR` to use a different cache directory.

## License

MIT
