# Changelog

This project does not currently use tagged releases consistently, so this changelog is maintained as a running record of notable repository-level changes.

## Unreleased

### Added

- Added root-level guidance in [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md) requiring implementation plans to be written under `plans/`.
- Added root docs trees for supported attacks and defenses under [`docs/attacks/`](docs/attacks/) and [`docs/defenses/`](docs/defenses/).
- Added a compact public docs page at [`docs/extending.md`](docs/extending.md) covering how to add new attacks and defenses.
- Added a docs migration and standardization plan at [`plans/docs-root-migration-and-standardization.md`](plans/docs-root-migration-and-standardization.md).
- Added merged AgentDyn benchmark assets to the vendored [`agents/agentdojo/`](agents/agentdojo/) tree, including new `shopping`, `github`, and `dailylife` suites plus the dynamic tool implementations they require.
- Added a third plugin type, XAI methods (`XAI_REGISTRY` in [`piarena/registry.py`](piarena/registry.py), `BaseXAI` in [`piarena/xai/base.py`](piarena/xai/base.py)), and a new optional `--xai`/`--xai_config`/`--limit` pipeline phase in [`main.py`](main.py) run right after the defense phase (opt-in — omitting `--xai` leaves existing behavior unchanged).
- Added the `syntaxshap` XAI method ([`piarena/xai/syntaxshap/`](piarena/xai/syntaxshap/)): adapts the [SyntaxSHAP paper](https://arxiv.org/abs/2402.09259) (originally for explaining next-token generation by autoregressive LLMs) to explain a text-classification defense's decision instead — currently `promptguard`. Reimplements the paper's syntax-tree-restricted Shapley coalition game (not a subclass of the vendored, generation-only `SyntaxExplainer`) against a binary `P(non-benign)` target, and fixes a combinatorial-explosion bug present in the original `feature_exact()` (unconditional `O(2^M)` enumeration even for the tree-restricted case) that would otherwise make explaining a whole multi-sentence `context` intractable. Supports `algorithm="syntax"` (tree-restricted) and `"shap"` (unrestricted, small texts only) — the paper's level-weighted variant (`"syntax-w"`/SyntaxSHAP-W) was implemented and then removed, since the experiment always explains with plain `"syntax"`. Full design in [`plans/xai-syntaxshap-promptguard.md`](plans/xai-syntaxshap-promptguard.md).
- Added `scripts/xai_metrics.py` / [`piarena/xai/metrics.py`](piarena/xai/metrics.py): computes Fidelity(t)/acc@1 (adapted from the SyntaxSHAP paper's Eqs. 6 and 8) and an injected-span rank percentile from a `--xai syntaxshap` result file, and renders `shap.plots.text`/`shap.plots.bar` saliency maps via the official `shap` package. (div@K/Eq. 7 was implemented and then removed: once the explained target is always the binary `[P(BENIGN), P(MALIGN)]` aggregate, div@K collapses algebraically to `2·|Fidelity(t)|` for every sample/threshold — redundant by construction, not merely correlated. A `spearman()` "syntax vs. syntax-w" concordance metric was also implemented and then removed along with `algorithm="syntax-w"` itself.)
- Added docs at [`docs/xai/`](docs/xai/) (index + `syntaxshap.md`) and an "Add A New XAI Method" section to [`docs/extending.md`](docs/extending.md).
- Vendored a trimmed, patched subset of `syntax-shap-main/syntaxshap` (masker, model-wrapper, and dependency-tree utilities only — not the generation-specific explainer/captum code) into [`piarena/xai/syntaxshap/thirdparty/`](piarena/xai/syntaxshap/thirdparty/), and added `pandas`, `cloudpickle`, `shap`, and `matplotlib` to [`requirements.txt`](requirements.txt) (`spacy` was already listed; also run `python -m spacy download en_core_web_sm`).
- Added per-phase progress visibility to the `--xai syntaxshap` pipeline, previously silent for the entire (potentially hours-long) XAI phase: [`main.py`](main.py) now prints Attack/Defense/XAI/Evaluation phase markers with elapsed time per sample; [`piarena/xai/syntaxshap/coalition.py`](piarena/xai/syntaxshap/coalition.py)'s `compute_shapley_values` computes its exact forward-pass budget upfront (from the already-built coalition set, no extra cost) and drives a `tqdm` bar off real `get_contribution_fn` calls; [`piarena/xai/syntaxshap/classifier_explainer.py`](piarena/xai/syntaxshap/classifier_explainer.py)'s `explain_context` prints which sentence (i/N) is running before each one, since cost varies wildly between sentences. On by default via a new `"progress"` key in `SyntaxShapXAI.DEFAULT_CONFIG` ([`piarena/xai/syntaxshap/xai_syntaxshap.py`](piarena/xai/syntaxshap/xai_syntaxshap.py)).
- Added [`scripts/estimate_xai_cost.py`](scripts/estimate_xai_cost.py): estimates `--xai syntaxshap` forward-pass cost for any dataset row using only a tokenizer + spaCy (no GPU, no gated-model access) — a closed-form reimplementation of the ad-hoc cost analysis in this file's "Computational cost bottleneck" section (formula derived from, and verified against, `coalition.py`'s actual coalition-counting logic on hand-worked small cases), packaged as a reusable tool instead of a one-off scratchpad script.
- Added [`scripts/run_xai_pilot_runpod.sh`](scripts/run_xai_pilot_runpod.sh): end-to-end runbook for a single-sample `--xai syntaxshap` pilot on a RunPod GPU pod (env setup incl. auto-installing Miniconda and accepting its channel ToS non-interactively, pinned-CUDA torch install, `huggingface-cli login`, a cost-check confirmation before the expensive run, `main.py`, then `scripts/xai_metrics.py`). Tuned against and partially validated on a real `runpod/pytorch:*-cu1281-torch280-*` pod.

### Changed

- Migrated the website to consume markdown directly from root [`docs/`](docs/) instead of maintaining a duplicate `website/docs/` tree.
- Reorganized public docs into a smaller structure centered on:
  - [`docs/getting-started.md`](docs/getting-started.md)
  - [`docs/evaluation.md`](docs/evaluation.md)
  - [`docs/attacks/`](docs/attacks/)
  - [`docs/defenses/`](docs/defenses/)
  - [`docs/extending.md`](docs/extending.md)
- Standardized attack and defense docs so each method page focuses on a brief introduction, source links, usage, behavior, and parameters.
- Updated the website docs sidebar in [`website/app.jsx`](website/app.jsx) to discover pages automatically from root docs and render a cleaner docs tree.
- Fixed inline docs link rendering in [`website/app.jsx`](website/app.jsx) so markdown links with code-formatted labels render correctly.
- Updated [`website/vite.config.js`](website/vite.config.js) to allow loading markdown from the repository root during website builds.
- Updated repository guidance in [`README.md`](README.md), [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md), and [`website/AGENTS.md`](website/AGENTS.md) to reflect the root-docs workflow.
- Varied docs and README examples so they do not repeatedly use `pisanitizer` as the default example defense.
- Expanded the vendored [`agents/agentdojo/`](agents/agentdojo/) integration so the existing PIArena defense adapter works for both classic AgentDojo suites and the merged AgentDyn suites.
- Updated [`main_agentdojo.py`](main_agentdojo.py) and [`scripts/run_agentdojo.py`](scripts/run_agentdojo.py) so one runner can execute classic AgentDojo suites, merged AgentDyn suites, PIArena defenses, and benchmark-native defenses from the same vendored benchmark tree.
- Commented out `torch`/`torchvision`/`torchaudio`/`vllm` in [`requirements.txt`](requirements.txt) (they were previously left uncommented despite this file's own setup docs already saying they should be installed separately) — an unpinned `vllm` here makes pip backtrack through 100+ releases looking for a resolvable combination, down to versions old enough to fail building from source against a modern CUDA/torch pod image. `torch` is not needed for `vllm` to be pinned/installed separately for the target CUDA version; `vllm` itself is not required for the `--defense promptguard --xai syntaxshap` pipeline (backend LLM loads via `transformers`, not `vllm`).

### Removed

- Removed the duplicate public docs copies from `website/docs/`.
- Removed the older flat public docs pages that were replaced by the new grouped attack and defense trees.
