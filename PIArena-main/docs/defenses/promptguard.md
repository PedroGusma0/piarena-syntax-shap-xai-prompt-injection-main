---
title: PromptGuard
slug: defenses/promptguard
category: defenses
---

# PromptGuard

`promptguard` is a lightweight classifier-based detection defense built on Meta Prompt Guard.

Source:
[piarena/defenses/promptguard/defense_promptguard.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/promptguard/defense_promptguard.py)

Model:
[Meta Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M)

## How To Use

```python
defense = get_defense("promptguard")
result = defense.get_response(
    target_inst=target_inst,
    context=context,
    llm=backend_llm,
)
```

## What It Does

It classifies the context as benign or suspicious. Suspicious inputs are blocked with a warning. Benign inputs are passed to the backend model.

## Parameters

This defense has no user-facing configuration in `DEFAULT_CONFIG`. It loads the Prompt Guard classifier directly.

## Windowed variant: `promptguard_windowed`

`promptguard` above feeds the whole context to Prompt Guard in a single pass (no truncation, no split). Prompt Guard has a 512-token context window, and Meta [recommends](https://huggingface.co/meta-llama/Prompt-Guard-86M) "splitting longer inputs into segments and scanning each in parallel to detect the presence of violations anywhere in longer prompts". `promptguard_windowed` does that:

```bash
python main.py --dataset lcc_long --attack direct --defense promptguard_windowed --window_tokens 512
```

The context is split into `window_tokens`-token windows (`piarena/windowing.py`, never cutting a word) and each window is scored on its own. The prompt is **blocked if any window has P(malign) >= `block_threshold`**, where P(malign) = P(INJECTION) + P(JAILBREAK). The original `promptguard` is unchanged.

`DEFAULT_CONFIG`: `window_tokens` (default `512`), `block_threshold` (default `0.5`), `window_batch_size` (default `16`).

`defense_result` additionally carries `window_p_malign`, `window_spans` (character offsets), `n_windows`, `max_p_malign`, `argmax_window` and `blocking_windows`, so per-window plots (`scripts/xai_window_anomaly.py`) work for every sample without running any XAI. Results are not comparable to `promptguard` runs: the decision rule differs (per-window P(malign) vs. argmax over the whole context). Source: `piarena/defenses/promptguard/defense_promptguard_windowed.py`.
