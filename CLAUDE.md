# CLAUDE.md — xai-in-pi-arena workspace

Guidance for an agent picking up this workspace. Read this first, then `PIArena-main/plans/xai-syntaxshap-promptguard.md` (full design/rationale) before touching the XAI code.

## Workspace layout

- `PIArena-main/` — the PIArena prompt-injection attack/defense benchmark (git repo root is actually one level up, at this workspace's own `.git`; `PIArena-main` has no `.git` of its own). Has its own `CLAUDE.md`/`AGENTS.md` — read those for PIArena's general architecture (registry pattern, Attack→Defense→[XAI]→Evaluation pipeline, etc.).
- `syntax-shap-main/` — the original SyntaxSHAP paper's code, vendored as a reference source (not used directly — see below). Local git repo with no remote, one commit — not a real upstream clone. There's a duplicate nested `syntax-shap-main/syntax-shap-main/` from a bad unzip; harmless, never cleaned up, ignore it.
- `papers-de-referencia/syntaxshap.md` — organized transcription of the SyntaxSHAP paper (arXiv:2402.09259), written during this session for quick reference (equations, evaluation setup, limitations). Read this before re-deriving anything about the paper's method.

## What this session did

Goal: use SyntaxSHAP as an XAI engine inside PIArena to explain the `promptguard` defense's decisions — `python main.py --dataset squad_v2 --attack direct --defense promptguard --xai syntaxshap`.

**Full design doc (read this for the real detail):** `PIArena-main/plans/xai-syntaxshap-promptguard.md`. It covers, with rationale for every decision:
- Why SyntaxSHAP had to be *adapted*, not used as-is (it's hard-wired to autoregressive generation via `.generate()` + captum teacher forcing; PromptGuard is a single-shot classifier).
- Why the core coalition/Shapley-value logic was **reimplemented** in a new `coalition.py`, not subclassed from the vendored `SyntaxExplainer` (avoids a captum dependency and a critical `O(2^M)` bug in the original `feature_exact` that made explaining long text impossible).
- Why the explained target is a **binary aggregate** `P(non-benign) = P(INJECTION) + P(JAILBREAK)` rather than the raw 3-class argmax (keeps the explained quantity consistent across every sample, matches what `PromptGuardDefense.detect_flag` actually checks).
- Why the **whole `context`** is explained (not just the isolated `injected_task` span) via per-sentence orchestration, and the cost tradeoffs of that.
- The full metrics design (Fidelity(t)/acc@1/injected-span-rank, adapted from the paper's Section 4 — div@K/Eq. 7 was implemented and then removed, since over the binary `[P(BENIGN), P(MALIGN)]` aggregate it collapses algebraically to `2·|Fidelity(t)|`, redundant by construction) and the `shap`-library saliency-map plotting design.

## What's implemented (in `PIArena-main/`)

- `piarena/xai/` — new third plugin type alongside attacks/defenses (`XAI_REGISTRY` in `piarena/registry.py`, `BaseXAI` in `piarena/xai/base.py`).
  - `piarena/xai/syntaxshap/coalition.py` — reimplemented, agnostic Shapley coalition math (no captum, no `2^M` blowup).
  - `piarena/xai/syntaxshap/classifier_explainer.py` — `ClassifierSyntaxExplainer`: `explain_row` (single span) and `explain_context` (whole multi-sentence `context`, per-sentence orchestration + `injected_span` localization).
  - `piarena/xai/syntaxshap/xai_syntaxshap.py` — `SyntaxShapXAI(BaseXAI)`, the registered `--xai syntaxshap` plugin. **Heavy imports (torch/transformers/spacy) are lazy** — importing `piarena.xai` must stay cheap so `--xai` stays opt-in; keep it that way if you touch this file.
  - `piarena/xai/syntaxshap/thirdparty/` — trimmed, **patched** vendored subset of `syntax-shap-main/syntaxshap` (masker, model wrapper, dependency-tree utils only). Two real bugs were found and fixed here by testing against a real tokenizer (see `_dependency_tree.py`'s inline comments): `[CLS]`/`[SEP]`-style special tokens weren't recognized (only `<s>`/`</s>` BPE-style was), and even after that fix they still leaked into the dependency tree as duplicate rows of the first/last real word.
  - `piarena/xai/metrics.py` + `scripts/xai_metrics.py` — pilot analysis (Fidelity/acc@1/injected-span-rank) and `shap.plots.text`/`shap.plots.bar` saliency-map rendering, run separately from `main.py` against its result JSON.
- `main.py` — added `--xai`, `--xai_config` (YAML-only, like `attack_config`/`defense_config`), `--limit` (new, for quick pilots — no dataset-size flag existed before). XAI phase runs right after Defense; result goes into `result_dp["xai_result"]`. Also now persists `injected_context` on each result (previously only in the separate attack-cache file) and appends `-{xai}` to the result filename so an `--xai` run can't silently collide with a prior non-XAI run under the same `--name`.
- `requirements.txt` — added `pandas`, `cloudpickle`, `shap`, `matplotlib` (all needed only for `--xai`/the metrics script; `spacy` was already listed — still need `python -m spacy download en_core_web_sm` separately).
- Docs updated: `docs/xai/` (new), `docs/extending.md`, `CLAUDE.md`/`AGENTS.md` (both, inside `PIArena-main/`), `CHANGELOG.md`.

## What's validated vs. not

This sandbox has **no GPU** (`main.py` hard-asserts `torch.cuda.device_count() > 0` at import — blocks running `main.py` here at all, unrelated to XAI) and `meta-llama/Prompt-Guard-86M` is **gated on HF** (needs the user's own token/access). So the actual pilot has never been run against the real model.

What *was* validated, in an isolated venv (`/tmp/xai_test_venv` — not cleaned up, harmless, recreate with `pip install numpy pandas transformers spacy && python -m spacy download en_core_web_sm && pip install torch scipy scikit-learn cloudpickle --index-url https://download.pytorch.org/whl/cpu` if useful) with a **mocked classifier** standing in for Prompt-Guard-86M plus a real (non-gated) BERT tokenizer + real spaCy parsing:
- `coalition.py`'s coalition construction matches a brute-force oracle exactly, and the `O(2^M)`-avoidance fix actually works (M=60 in 3ms vs. impossible before).
- `ClassifierSyntaxExplainer.explain_row`/`explain_context` run end-to-end correctly (found and fixed the two dependency-tree bugs above in the process), correctly locate `injected_span`, and correctly rank trigger-word tokens as most important.
- `piarena/xai/metrics.py`'s numeric functions.
- `piarena.xai`'s lazy-import property (importing it doesn't pull in torch/spacy/sklearn — confirmed empirically, not just by code inspection).

**Not yet done — next steps for whoever has GPU + HF access:**
1. `huggingface-cli login` (or `HF_TOKEN`) for the gated model.
2. `pip install -r requirements.txt` in the real `piarena` conda env (per `PIArena-main/CLAUDE.md`'s setup section) + `python -m spacy download en_core_web_sm`.
3. Confirm `AutoConfig.from_pretrained("meta-llama/Prompt-Guard-86M").id2label` matches the assumed 3-class `BENIGN`/`INJECTION`/`JAILBREAK` set (the code is written to be robust to this via substring matching on "benign", but it's never been checked against the real model).
4. Run the actual pilot — **do not blindly use `--limit 25`**, see "Computational cost bottleneck" below first; it takes the first N dataset rows in order, and the runtime across `squad_v2` rows varies by ~9 orders of magnitude, so a naive `--limit 25` risks picking a row that never finishes. Build a small curated-subset JSON (cheap rows only, see below) and pass it as `--dataset <path>` instead, e.g.: `python main.py --dataset PIArena-main/datasets/squad_v2_pilot_cheap.json --attack direct --defense promptguard --xai syntaxshap --name xai_pilot --seed 42`, then `python scripts/xai_metrics.py --result results/evaluation_results/xai_pilot/...json`.

## Computational cost bottleneck (investigated without GPU/model access)

Session question: "how long would the XAI module take to process one prompt end to end?" Answered analytically/empirically **without running the real pipeline** (no GPU, no gated model here — see constraints above) by reasoning about `piarena/xai/syntaxshap/coalition.py`'s own algorithm plus real (non-mocked) spaCy dependency parses and a real (non-gated) `microsoft/mdeberta-v3-base` tokenizer (same encoder family/tokenizer as `Prompt-Guard-86M`) run against this repo's actual `PIArena-main/datasets/squad_v2.json`. Method, cross-checked against the real `coalition.py` (30/30 random-tree brute-force trials matched exactly): derived a closed-form formula for exactly how many classifier forward passes `compute_shapley_values` issues for a sentence, given its dependency-tree's per-level token counts — avoids ever enumerating the coalitions themselves, which is what makes checking pathological cases (2^38-scale) tractable at all.

**Finding 1 — cost is driven by the widest single dependency-tree level, not by text length.** `explain_context` sums cost sentence-by-sentence (`piarena/xai/syntaxshap/classifier_explainer.py`'s per-sentence loop), and each sentence's cost is dominated by `2^(width of its widest level)` (level = BFS depth from the sentence's syntactic root, per the vendored `_dependency_tree.py`). Two `squad_v2` samples of nearly identical length (~700-750 chars, `injected_context` with `attack=direct`) differ by **8 orders of magnitude** in required forward passes: idx 3 (plain Neptune-facts prose, widest level = 12) needs 246,410 calls; idx 6 (a sentence with a long Korean-War parenthetical: `"(in South Korean Hangul: 한국전쟁, Hanja: 韓國戰爭, ...)"`, widest level = 38) needs ~18.76 trillion. `--limit N`/sample length are not usable proxies for runtime.

**Finding 2 — subtoken fragmentation of non-standard strings silently multiplies a level's effective width.** The vendored `get_token_dependency_tree` maps *every* subtoken of a word to that word's single tree level — this is the original SyntaxSHAP paper's own documented design (`papers-de-referencia/syntaxshap.md`: "quando um tokenizer quebra uma palavra em múltiplos tokens, SyntaxShap duplica o nó-palavra... tratando cada subtoken com o mesmo papel/nível do nó original"), not a bug introduced by this port, and it was fine at the paper's tested scale (short single sentences, GPT-2 BPE, common English words → usually 1 subtoken/word). It stops being fine here: `mdeberta-v3-base`'s tokenizer splits a URL like `http://secure-umayyadhistory.site` into 10 subtokens (`['▁http','://','secure','-','um','ayya','d','history','.','site']`), and non-Latin scripts (the Hangul/Hanja case above) fragment even harder — all landing on the *same* tree level as the original single word, inflating that level's width (and thus its `2^width` contribution) well beyond what the syntactic structure alone would suggest. `injected_task` strings in this dataset frequently contain attacker-inserted URLs (`https://...`) specifically to test this.

**Real cost distribution, 200 `squad_v2` contexts** (word-tree resolution; a lower bound — Finding 2 means the real subtoken-resolution numbers run higher, especially for the ~25% of rows whose `injected_task` contains a URL), forward passes needed for the whole `context` (`explain_targets=["context"]`, the default):

| percentile | # forward passes | est. time @ CPU ~20ms/call | est. time @ GPU ~5ms/call |
|---|---:|---:|---:|
| min | 2,353 | 47s | 12s |
| p25 | 31,615 | 10.5 min | 2.6 min |
| median | 103,050 | ~34 min | ~9 min |
| p90 | 4,057,384 | 22.5 h | 5.6 h |
| p99 | 776,957,273 | 180 days | 45 days |
| max | 1,594,710,110 | 369 days | 92 days |

(20ms/5ms-per-forward-pass are plausible ballpark figures for an ~86M-param transformer, CPU vs. GPU, batch=1 — matching `ClassifierSyntaxExplainer._score`'s current one-text-at-a-time `self.model([text])` call, no batching implemented. Not measured on real hardware here.)

**Curated cheap pilot subset (viable to actually run):** scanning the first 20 `squad_v2` rows at real subtoken resolution, indices **3, 7, 14, 10** are the only ones under ~450k forward passes each (next-cheapest jumps to 3.7M) — combined ~1.43M calls, ≈8h on CPU / ≈2h on GPU for all four. This is the subset step 4 above should use instead of a blind `--limit 25`. The one-off analysis scripts (`closed_form.py`, `pick_pilot_sample.py`) that produced these numbers were written to this session's ephemeral scratchpad, not committed — recreate them (or ask the session that has this conversation's history) before reusing this method rather than re-deriving it from scratch.

**Open question this raises for the plan doc:** is it worth adding a level-width cap/fallback (e.g. sample instead of enumerate a level beyond some width threshold) to `coalition.py`, given this would trade fidelity to the published SyntaxSHAP algorithm for tractability on real multi-sentence Wikipedia-style `context`s? Not yet decided — see `PIArena-main/plans/xai-syntaxshap-promptguard.md`'s "Riscos/suposições explícitos" section, this isn't logged there yet.
