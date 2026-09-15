# -*- coding: utf-8 -*-

import torch
print("GPUs available:", torch.cuda.device_count())
assert torch.cuda.device_count() > 0, "No GPUs available"
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import argparse
import copy
import os
import time
import yaml
from tqdm import tqdm
from datasets import load_dataset, Dataset
from piarena.utils import save_json, load_json, setup_seeds, nice_print
from piarena.llm import Model
from piarena.attacks import get_attack
from piarena.defenses import get_defense
from piarena.xai import get_xai
from piarena.evaluations import (
    llm_judge, open_prompt_injection_utility,
    substring_match, longbench_metric_dict,
)


def load_config(config_path: str) -> dict:
    """Load experiment configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(prog='PIArena', description="PIArena evaluation pipeline")

    parser.add_argument('--config', type=str, default=None,
                        help="Path to YAML config file. CLI args override config values. "
                             "Supports attack_config and defense_config keys for component config.")

    # General args
    parser.add_argument('--dataset', type=str, default=None,
                        help="Path of dataset or subset in https://huggingface.co/datasets/sleeepeer/PIArena.")
    parser.add_argument('--backend_llm', type=str, default=None,
                        help="Name of the backend LLM to be used.")
    parser.add_argument('--attack', type=str, default=None,
                        help="Type of attack to be used.")
    parser.add_argument('--defense', type=str, default=None,
                        help="Type of defense to be used.")
    parser.add_argument('--attack_path', type=str, default=None,
                        help="Existing attack result to reuse.")
    parser.add_argument('--name', type=str, default=None,
                        help="Name of the experiment.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Seed for the experiment.")
    parser.add_argument('--xai', type=str, default=None,
                        help="Name of the XAI method to run after the defense phase "
                             "(opt-in; omit to skip explanation).")
    parser.add_argument('--no-wandb', action='store_true',
                        help="Disable W&B logging (on by default whenever --xai is set — see wandb.init() in main(); "
                             "has no effect without --xai). WANDB_MODE=offline unless already set in the environment.")
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit the number of dataset samples processed (for quick "
                             "pilots/smoke tests). Omit to process the full dataset "
                             "(default, unchanged behavior).")

    args = parser.parse_args()

    # Merge YAML config with CLI args (CLI takes priority)
    file_config = {}
    if args.config is not None:
        file_config = load_config(args.config)

    defaults = {
        'dataset': 'squad_v2',
        'backend_llm': 'Qwen/Qwen3-4B-Instruct-2507',
        'attack': 'combined',
        'defense': 'pisanitizer',
        'attack_path': None,
        'name': 'test',
        'seed': 42,
    }
    for key, default_val in defaults.items():
        cli_val = getattr(args, key)
        if cli_val is None:
            setattr(args, key, file_config.get(key, default_val))

    # Store component configs from YAML (attack_config, defense_config)
    args.attack_config = file_config.get("attack_config", None)
    args.defense_config = file_config.get("defense_config", None)

    # --xai has no CLI default (None = skip); --xai_config is YAML-only, same
    # pattern as attack_config/defense_config (no dedicated CLI flag).
    args.xai = args.xai or file_config.get("xai", None)
    args.xai_config = file_config.get("xai_config", None)

    print(args)
    return args


def main(args):
    # Load dataset
    try:
        dataset = load_json(args.dataset)
        dataset = Dataset.from_list(dataset)
        dataset_name = args.dataset.split('/')[-1].split('.')[0]
    except:
        dataset_name = args.dataset
        while True:
            try:
                dataset = load_dataset(
                    "sleeepeer/PIArena",
                    split=dataset_name,
                    download_mode="force_redownload"
                )
                break
            except Exception as e:
                if "429" in str(e):
                    print("Hit Hugging Face rate limit when loading dataset. Waiting 5 minutes...")
                    time.sleep(300)
                else:
                    raise e

    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"--limit {args.limit}: processing {len(dataset)} sample(s).")

    # Load cached attack results or instantiate attack
    try:
        attack_result = load_json(args.attack_path)
        assert len(attack_result) == len(dataset)
        print(f"Loaded existing attack result from {args.attack_path}.")
        attack_name = args.attack_path.split('/')[-1].split('.')[0]
        attack = None
    except:
        attack_result = {}
        attack_name = args.attack
        attack = get_attack(args.attack, config=args.attack_config)
        print(f"Initialized attack: {attack}")
        print("No existing attack result found, will run attack on-the-fly.")

    # Initialize backend LLM
    print(f"Loading backend LLM: {args.backend_llm}")
    llm = Model(args.backend_llm)

    # Instantiate defense
    defense = get_defense(args.defense, config=args.defense_config)
    print(f"Initialized defense: {defense}")

    # Instantiate XAI method once, outside the per-sample loop (opt-in — None skips it)
    xai = get_xai(args.xai, config=args.xai_config) if args.xai else None
    if xai is not None:
        print(f"Initialized xai: {xai}")

    # Result paths
    llm_name = args.backend_llm.replace('/', '-')
    # -{xai} suffix so a run with --xai can't silently collide with a prior
    # identical-config run lacking xai_result (the resume-skip logic below
    # would otherwise treat already-evaluated-without-XAI samples as done).
    xai_suffix = f"-{args.xai}" if args.xai else ""
    attack_result_path = f"results/evaluation_results/{args.name}/tmp_attack_results/{dataset_name}-{llm_name}-{attack_name}-{args.defense}-{args.seed}.json"
    evaluation_result_path = f"results/evaluation_results/{args.name}/{dataset_name}-{llm_name}-{attack_name}-{args.defense}{xai_suffix}-{args.seed}.json"

    # Select evaluators
    if "open_prompt_injection" in dataset_name:
        asr_evaluator = llm_judge
        utility_evaluator = open_prompt_injection_utility
    elif "sep" in dataset_name:
        asr_evaluator = llm_judge
        utility_evaluator = llm_judge
    elif "knowledge_corruption" in dataset_name:
        asr_evaluator = substring_match
        utility_evaluator = substring_match
    elif "_long" in dataset_name:
        asr_evaluator = llm_judge
        for metric in longbench_metric_dict.keys():
            if metric in dataset_name:
                utility_evaluator = longbench_metric_dict[metric]
                break
    else:
        asr_evaluator = llm_judge
        utility_evaluator = llm_judge

    # Load existing evaluation results
    try:
        evaluation_result = load_json(evaluation_result_path)
        print(f"Loaded existing evaluation result from {evaluation_result_path}.")
        if len(evaluation_result) == len(dataset):
            print("Evaluation result has the same length as the dataset, will quit evaluation.")
            return
    except:
        evaluation_result = {}
        print("No existing evaluation result found, will run evaluation on-the-fly.")

    # W&B logging (opt-in whenever --xai is set, --no-wandb to disable) —
    # per-sample cost/outcome logging below lets you watch progress from the
    # W&B dashboard without SSHing into the pod. Placed AFTER the "nothing
    # left to do" early-return above so resuming an already-fully-completed
    # run doesn't spin up an empty wandb run for it. WANDB_MODE defaults to
    # "offline" (writes locally, no live network dependency during the run)
    # unless already set in the environment — see wandb_sync_loop.sh for how
    # the offline data reaches the real wandb.ai dashboard.
    wandb_enabled = xai is not None and not args.no_wandb
    if wandb_enabled:
        os.environ.setdefault("WANDB_MODE", "offline")
        import wandb
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "piarena-xai-sweep"),
            group=attack_name,
            job_type="run",
            name=f"{attack_name}-{args.xai}-run",
            config={
                "attack": attack_name, "defense": args.defense, "xai": args.xai,
                "backend_llm": args.backend_llm, "seed": args.seed, "dataset": dataset_name,
            },
        )

    # Main evaluation loop
    for idx, dp in tqdm(enumerate(dataset)):
        print(f"=========={idx+1} / {len(dataset)}==========")
        if f"{idx}" in evaluation_result:
            print(f"Skipping index {idx} because it already exists in the evaluation result.")
            continue

        result_dp = copy.deepcopy(dp)
        target_inst = dp["target_inst"]
        context = dp["context"]
        injected_task = dp["injected_task"]
        target_task_answer = dp["target_task_answer"]
        injected_task_answer = dp["injected_task_answer"]

        # Attack phase
        try:
            injected_context = attack_result[idx]["injected_context"]
        except:
            injected_context = attack.execute(
                context=context,
                injected_task=injected_task,
                target_inst=target_inst,
                target_task_answer=target_task_answer,
                injected_task_answer=injected_task_answer,
            )
            attack_dp = copy.deepcopy(dp)
            attack_dp["injected_context"] = injected_context
            attack_result[idx] = attack_dp
            save_json(attack_result, attack_result_path)

        # Persisted unconditionally on the evaluation result (not just in the
        # attack cache file) so scripts/xai_metrics.py doesn't need to load
        # tmp_attack_results separately to know what text was explained.
        result_dp["injected_context"] = injected_context

        # Defense phase
        defense_result = defense.get_response(
            target_inst=target_inst,
            context=injected_context,
            llm=llm,
        )

        response = defense_result["response"]
        result_dp["defense_result"] = defense_result

        # XAI phase (opt-in via --xai), strictly after Defense and before Evaluation.
        # Only runs on samples the defense actually blocked (detect_flag=True) —
        # explaining a decision that let the prompt through is out of scope for
        # this experiment (the point is to explain *why the defense blocked*),
        # and skipping the rest is what makes the per-sample cost tractable.
        xai_phase_s = None  # captured here (not re-derived from phase_t0 later)
                             # so the W&B per-sample log below can report it.
        if xai is not None:
            if defense_result.get("detect_flag"):
                phase_t0 = time.time()
                print(f"[{idx+1}/{len(dataset)}] XAI phase ({args.xai})... this can take "
                      "from seconds to hours depending on the sample (see "
                      "plans/xai-kernelshap-promptguard.md) — progress prints below as it runs.")
                try:
                    result_dp["xai_result"] = xai.explain(
                        target_inst=target_inst,
                        context=injected_context,
                        injected_task=injected_task,
                        # `attack_name` (local var), not `args.attack` — when
                        # --attack_path loads a cached attack result, attack_name
                        # is derived from the cache file, and args.attack can be
                        # a stale/default value unrelated to what actually
                        # produced injected_context. Bug found while porting this
                        # gate to the SyntaxSHAP experiment.
                        attack_name=attack_name,
                    )
                    xai_phase_s = time.time() - phase_t0
                    print(f"[{idx+1}/{len(dataset)}] XAI phase done ({xai_phase_s:.1f}s).")
                except Exception as e:
                    # No algorithmic reason kernelshap should hang (its cost is
                    # bounded by n_samples, not text structure — see the plan's
                    # "Orçamento de amostras" section), but a single sample
                    # still shouldn't be able to take down an unattended,
                    # multi-attack sweep over the rest of the dataset (network
                    # hiccup, transient OOM, etc.) — mirrors the same try/except
                    # added to the syntaxshap experiment's main.py.
                    xai_phase_s = time.time() - phase_t0
                    result_dp["xai_result"] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"[{idx+1}/{len(dataset)}] XAI phase FAILED after {xai_phase_s:.1f}s "
                          f"({type(e).__name__}: {e}) — recorded as an error marker, continuing to the next sample.")
            else:
                result_dp["xai_result"] = None
                print(f"[{idx+1}/{len(dataset)}] XAI phase skipped (defense did not block this sample).")

        result_dp["utility"] = utility_evaluator(
            response,
            ground_truth=target_task_answer,
            task_prompt=f"{target_inst}\n\n{context}",
        )
        result_dp["asr"] = asr_evaluator(
            response,
            ground_truth=injected_task_answer,
            task_prompt=injected_task,
        )

        if wandb_enabled:
            # `.get()` chains throughout: `xai_result` is None (not blocked),
            # {"error": ...} (XAI phase failed), or a real dict with a
            # "context" key. syntaxshap-only fields (total_forward_passes/
            # widest_dependency_level/timed_out) simply come back None here —
            # this tree's ClassifierKernelExplainer never sets them.
            xai_result_dict = result_dp.get("xai_result") or {}
            xai_context = xai_result_dict.get("context") or {}
            wandb.log({
                "detect_flag": defense_result.get("detect_flag"),
                "asr": result_dp["asr"],
                "utility": result_dp["utility"],
                "xai_phase_s": xai_phase_s,
                "xai_error": xai_result_dict.get("error"),
                "xai_p_malign": xai_context.get("p_malign"),
                "xai_timed_out": xai_context.get("timed_out"),
                "xai_total_forward_passes": xai_context.get("total_forward_passes"),
                "xai_widest_dependency_level": xai_context.get("widest_dependency_level"),
                "xai_n_samples": xai_context.get("n_samples"),
                "xai_n_words": xai_context.get("n_words"),
            }, step=idx)

        evaluation_result[idx] = result_dp
        save_json(evaluation_result, evaluation_result_path)

        nice_print(f"Target Instruction: {target_inst}")
        print("\n")
        nice_print(f"Injected Task: {injected_task}")
        print("\n")
        nice_print(f"Response: {response}")
        print("\n")
        nice_print(f"Utility: {result_dp['utility']}, {round(sum([r['utility'] for r in evaluation_result.values()]) / len(evaluation_result), 2)}")
        nice_print(f"ASR: {result_dp['asr']}, {round(sum([r['asr'] for r in evaluation_result.values()]) / len(evaluation_result), 2)}")

    if wandb_enabled:
        # Final aggregate summary — mirrors scripts/xai_metrics.py's
        # _asr_utility_summary, visible on the W&B run's summary tab without
        # opening any file.
        all_samples = list(evaluation_result.values())
        blocked = [s for s in all_samples if s.get("defense_result", {}).get("detect_flag") is True]
        n_timed_out = sum(
            1 for s in blocked
            if ((s.get("xai_result") or {}).get("context") or {}).get("timed_out")
        )
        asr_vals = [float(s["asr"]) for s in all_samples if s.get("asr") is not None]
        utility_vals = [float(s["utility"]) for s in all_samples if s.get("utility") is not None]
        wandb.summary["n_samples"] = len(all_samples)
        wandb.summary["n_blocked"] = len(blocked)
        wandb.summary["n_timed_out"] = n_timed_out
        wandb.summary["asr_rate"] = sum(asr_vals) / len(asr_vals) if asr_vals else None
        wandb.summary["utility_mean"] = sum(utility_vals) / len(utility_vals) if utility_vals else None
        wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    setup_seeds(args.seed)
    torch.cuda.empty_cache()
    main(args)
