#!/usr/bin/env python3
"""Standalone Kernel SHAP explanation of PromptGuard on NotInject (PIGuard's
over-defense set) — no `main.py`, no attack, no victim LLM, no evaluator.

Every NotInject prompt is benign, so this is the `--attack none` procedure:
*every* prompt is explained (blocked = false positive, not blocked = true
negative), then `scripts/xai_metrics.py` (confusion matrix, Fidelity/acc@1)
and `scripts/xai_shap_plots_test.py` (saliency/bar plots) run over the result.

The result JSON mimics `main.py`'s evaluation-result format and filename
convention (`{dataset}-{llm}-{attack}-{defense}-{xai}-{seed}.json`), which is
all those two scripts depend on:

  results/evaluation_results/<name>/notinject_<subset>-novictim-none-<defense>-kernelshap-<seed>.json
  results/evaluation_results/<name>/xai_metrics/...

Resumable: samples already in the result file are skipped.

Usage:
  python scripts/run_notinject_xai.py                       # all 3 subsets
  python scripts/run_notinject_xai.py --subsets one --limit 5 --name notinject_smoke
  python scripts/run_notinject_xai.py --defense piguard --name kernelshap_notinject_piguard
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SUBSETS = ["one", "two", "three"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subsets", nargs="+", default=SUBSETS, choices=SUBSETS)
    p.add_argument("--data-dir", default=os.path.join(ROOT, "datasets"),
                   help="Folder holding NotInject_<subset>.json.")
    p.add_argument("--name", default="kernelshap_notinject")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Only the first N prompts of each subset (smoke test).")
    p.add_argument("--defense", default="promptguard", choices=["promptguard", "piguard"],
                   help="Which defense's classifier to explain (piarena.xai.classifiers).")
    p.add_argument("--model-name", default=None,
                   help="Override the classifier of --defense (default: the one piarena.xai.classifiers maps it to).")
    p.add_argument("--xai-backend", default="shap", choices=["shap", "captum"])
    p.add_argument("--n-samples", default="auto", help="'auto' or an integer (Kernel SHAP coalition samples).")
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--no-metrics", action="store_true",
                   help="Only write the result JSON; skip xai_metrics.py and the plots.")
    return p.parse_args()


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # atomic: an interrupt never leaves a half-written result


def _make_xai(args):
    from piarena.xai.classifiers import resolve_classifier
    from piarena.xai.kernelshap import KernelShapXAI

    n_samples = args.n_samples if args.n_samples == "auto" else int(args.n_samples)
    classifier = resolve_classifier(args.defense, {"model_name": args.model_name} if args.model_name else None)
    return KernelShapXAI({
        **classifier,
        "explain_scope": "all",  # every NotInject prompt is explained, blocked or not
        "backend": args.xai_backend,
        "n_samples": n_samples,
        "max_length": args.max_length,
        "progress": False,
    })


def run_subset(xai, subset, args, out_dir):
    data = _load_json(os.path.join(args.data_dir, f"NotInject_{subset}.json"))
    if data is None:
        raise FileNotFoundError(f"NotInject_{subset}.json not found in {args.data_dir}")
    if args.limit is not None:
        data = data[: args.limit]

    result_path = os.path.join(out_dir, f"notinject_{subset}-novictim-none-{args.defense}-kernelshap-{args.seed}.json")
    result = _load_json(result_path, default={})

    for idx, item in enumerate(data):
        key = str(idx)
        if key in result:
            continue
        prompt = item["prompt"]
        t0 = time.time()
        sample = {
            "target_inst": "", "context": prompt, "injected_context": prompt,
            "injected_task": "", "target_task_answer": "", "injected_task_answer": "",
            "category": item.get("category"), "word_list": item.get("word_list"),
            "trigger_subset": subset,
            "defense": args.defense, "xai_model": xai.config["model_name"],
            "utility": None, "asr": None,
        }
        try:
            xai_result = xai.explain(target_inst="", context=prompt, injected_task=None, attack_name="none")
            ctx = xai_result["context"]
            # Same rule as PromptGuardDefense.execute: any non-"benign" argmax label blocks.
            sample["defense_result"] = {"detect_flag": "benign" not in str(ctx["predicted_label"]).lower()}
            sample["xai_result"] = xai_result
        except Exception as e:  # one bad sample must not kill the sweep
            sample["defense_result"] = {"detect_flag": None}
            sample["xai_result"] = {"error": f"{type(e).__name__}: {e}"}
        result[key] = sample
        _save_json(result, result_path)
        flag = sample["defense_result"]["detect_flag"]
        print(f"[{subset} {idx + 1}/{len(data)}] blocked={flag} ({time.time() - t0:.1f}s)", flush=True)

    n_blocked = sum(1 for s in result.values() if s["defense_result"]["detect_flag"] is True)
    print(f"[{subset}] done: {n_blocked}/{len(result)} blocked -> {result_path}")
    return result_path


def run_postprocessing(result_path, args):
    for cmd in (
        [sys.executable, "scripts/xai_metrics.py", "--result", result_path,
         "--no-wandb"],
        [sys.executable, "scripts/xai_shap_plots_test.py", "--result", result_path,
         "--min-count", "1"],  # NotInject trigger words are ~unique per prompt; the default 5 would hide them all
    ):
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    try:
        import numpy as np
        import torch
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    out_dir = os.path.join(ROOT, "results", "evaluation_results", args.name)
    os.makedirs(out_dir, exist_ok=True)

    xai = _make_xai(args)
    for subset in args.subsets:
        result_path = run_subset(xai, subset, args, out_dir)
        if not args.no_metrics:
            run_postprocessing(result_path, args)


if __name__ == "__main__":
    main()
