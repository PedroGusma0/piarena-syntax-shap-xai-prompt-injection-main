---
title: Extending PIArena
slug: extending
category: guide
---

# Extending PIArena

PIArena uses registries for attacks, defenses, and XAI methods, so adding a new one is usually straightforward:

1. create a new class
2. give it a registry name
3. register it
4. import the module in the package `__init__.py`

## Add A New Attack

Create a file such as `piarena/attacks/my_attack.py`:

```python
from .base import BaseAttack, register_attack
from ..utils import inject


@register_attack
class MyAttack(BaseAttack):
    name = "my_attack"
    DEFAULT_CONFIG = {
        "prefix": "Ignore previous instructions.",
    }

    def execute(self, context, injected_task, inject_position="random", inject_times=1, **kwargs):
        injected_prompt = f"{self.config['prefix']} {injected_task}"
        return inject(context, injected_prompt, inject_position, inject_times)
```

Then import it in `piarena/attacks/__init__.py`:

```python
from . import my_attack  # noqa: F401
```

## Add A New Defense

Create a file such as `piarena/defenses/my_defense.py`:

```python
from .base import BaseDefense, register_defense


@register_defense
class MyDefense(BaseDefense):
    name = "my_defense"
    DEFAULT_CONFIG = {
        "threshold": 0.5,
    }

    def execute(self, target_inst, context):
        return {
            "detect_flag": False,
            "cleaned_context": context,
        }
```

Then import it in `piarena/defenses/__init__.py`:

```python
from . import my_defense  # noqa: F401
```

## Add A New XAI Method

Third plugin type (`XAI_REGISTRY`, `piarena/registry.py`), added to explain defense decisions — see `piarena/xai/kernelshap/` and `plans/xai-kernelshap-promptguard.md` for a full worked example (Kernel SHAP adapted for the `promptguard` classifier).

Create a file such as `piarena/xai/my_xai/xai_my_xai.py`:

```python
from ..base import BaseXAI, register_xai


@register_xai
class MyXAI(BaseXAI):
    name = "my_xai"
    DEFAULT_CONFIG = {}

    def explain(self, target_inst, context, injected_task, **kwargs) -> dict:
        return {"context": {"tokens": [], "values": []}}
```

Then import it in `piarena/xai/__init__.py`:

```python
from . import my_xai  # noqa: F401
```

Run it via `main.py --xai my_xai` (opt-in — omit `--xai` and nothing changes). Config, if any, goes through the `xai_config` YAML key (same pattern as `attack_config`/`defense_config` — no dedicated CLI flag).

## The Main Interfaces

### Attack interface

```python
def execute(self, context, injected_task, **kwargs) -> str:
    ...
```

An attack returns the injected context string.

### Defense interface

```python
def execute(self, target_inst, context) -> dict:
    ...
```

A defense usually returns one or more of:

- `detect_flag`
- `cleaned_context`
- `potential_injection`

### Full response path

Most defenses can also use the standard response helper:

```python
def get_response(self, target_inst, context, llm) -> dict:
    ...
```

If your defense fits the usual pattern, `BaseDefense.get_response()` and `BaseDefense.get_response_batch()` already provide a useful default.

### XAI interface

```python
def explain(self, target_inst, context, injected_task, **kwargs) -> dict:
    ...
```

An XAI method returns a dict keyed by the span(s) it explained (e.g. `"context"`, `"injected_task"`), each value itself method-specific (Kernel SHAP returns `tokens`/`values`/`base_value`/`full_value` — see [docs/xai/kernelshap.md](xai/kernelshap.md)).

## Keep New Docs Consistent

When you add a new public attack, defense, or XAI method, add a doc page under:

- `docs/attacks/`
- `docs/defenses/`
- `docs/xai/`

Use the same simple structure as the existing pages:

1. brief introduction
2. source link
3. how to use it
4. what it does
5. parameters
