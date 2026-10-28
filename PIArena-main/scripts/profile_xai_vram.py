"""Profiles why `--xai kernelshap` (Prompt-Guard-86M) runs out of VRAM on the
*_long datasets, and projects how much VRAM / time each sample needs.

Meant to be run ONCE on a 24 GB cluster GPU (needs HF access to the gated
`meta-llama/Prompt-Guard-86M`), then send the printed summary + the JSON
file back. It never runs main.py / the backend LLM / Kernel SHAP itself: it
reproduces only the memory-dominant operation — one classifier forward pass
over `(B, L)` token ids, as `ClassifierKernelShapExplainer._forward_func`
does (fp32, `torch.no_grad()`, `AutoModelForSequenceClassification`).

Stages
  0. environment   — GPU, torch/transformers versions, model config
  1. tokens        — REAL tokenizer token counts per dataset (context + injected_task)
  2. grid          — peak VRAM of a forward pass for a grid of (L, B); OOMs recorded
  3. max batch     — largest B that fits for several L (binary search)
  4. fit           — peak_activation ~= c*B*L^2 + d*B*L + e
  5. real samples  — shortest/median/longest prompt of each dataset
  6. projection    — per-sample predicted peak for B in {1,2,4,8} using the fit,
                     how many samples exceed the budget, plus Kernel SHAP
                     n_samples and estimated time per sample

Usage (from PIArena-main/):
  python scripts/profile_xai_vram.py
  python scripts/profile_xai_vram.py --reserve-gb 8       # simulate the Qwen3-4B resident on the same GPU
  python scripts/profile_xai_vram.py --dtype both         # also compare fp16
  python scripts/profile_xai_vram.py --datasets lcc_long qasper_long --out /tmp/prof.json
"""
import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS = ["lcc_long", "passage_retrieval_en_long", "qasper_long", "hotpotqa_long"]
GIB = 1024 ** 3


def gb(x):
    return x / GIB


def free_cuda():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def load_dataset(name):
    with open(ROOT / "datasets" / f"{name}.json") as f:
        return json.load(f)


def make_ids(tokenizer, L, B, device):
    """Random content ids with CLS ... SEP, shape (B, L)."""
    import torch
    vocab = len(tokenizer)
    ids = torch.randint(1000, vocab - 1, (B, L), device=device)
    ids[:, 0] = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    ids[:, -1] = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    return ids


def forward_peak(model, ids):
    """One forward like the explainer. Returns peak-memory stats, or {"oom": True}."""
    import torch
    free_cuda()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            probs = torch.softmax(logits.float(), dim=-1)
            probs.sum(dim=-1).cpu()
        torch.cuda.synchronize()
        out = {
            "oom": False,
            "base_gb": gb(base),
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
            "peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
            "peak_act_gb": gb(torch.cuda.max_memory_allocated() - base),
            "seconds": time.time() - t0,
        }
    except (torch.OutOfMemoryError, RuntimeError) as e:
        # An OOM raised inside DeBERTa's TorchScript function (make_log_bucket_position)
        # surfaces as a plain RuntimeError instead of torch.OutOfMemoryError.
        if not isinstance(e, torch.OutOfMemoryError) and "out of memory" not in str(e).lower():
            raise
        out = {"oom": True}
    # drop the exception's frame references before freeing, or tensors stay alive
    logits = probs = None
    free_cuda()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Prompt-Guard-86M")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "both"], default="fp32",
                    help="fp32 matches the current explainer; 'both' also profiles fp16.")
    ap.add_argument("--reserve-gb", type=float, default=0.0,
                    help="Allocate a dummy tensor of this size first, to emulate other models "
                         "(e.g. Qwen3-4B ~8 GB) sharing the GPU during the Defense phase.")
    ap.add_argument("--perturbations-per-eval", type=int, default=2,
                    help="B used by --xai kernelshap (config default: 2).")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 16384, 20480])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-real-samples", type=int, default=3,
                    help="real samples per dataset in stage 5 (shortest, median, longest).")
    ap.add_argument("--out", default=str(ROOT / "results" / "profile_xai_vram.json"))
    args = ap.parse_args()

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit("No CUDA device — run this on the cluster GPU node.")

    report = {"args": vars(args)}
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    total_gb = gb(props.total_memory)

    # ---------------------------------------------------------------- stage 0
    print("=" * 78 + "\n[0] environment")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    cfg = model.config
    env = {
        "gpu": props.name,
        "gpu_total_gb": round(total_gb, 2),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "model": args.model,
        "model_type": cfg.model_type,
        "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "hidden_size": cfg.hidden_size,
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "position_buckets": getattr(cfg, "position_buckets", None),
        "relative_attention": getattr(cfg, "relative_attention", None),
        "pos_att_type": getattr(cfg, "pos_att_type", None),
        "attn_implementation": getattr(cfg, "_attn_implementation", None),
        "id2label": cfg.id2label,
        "tokenizer_model_max_length": tok.model_max_length,
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    report["env"] = env
    for k, v in env.items():
        print(f"  {k}: {v}")

    model.to(dev)
    reserved_tensor = None
    if args.reserve_gb > 0:
        reserved_tensor = torch.empty(int(args.reserve_gb * GIB), dtype=torch.uint8, device=dev)
        print(f"  reserved {args.reserve_gb:.1f} GB dummy tensor (emulating other resident models)")

    # ---------------------------------------------------------------- stage 1
    print("=" * 78 + "\n[1] token counts (real tokenizer, context + injected_task)")
    data = {}
    token_rows = []
    for name in args.datasets:
        try:
            ds = load_dataset(name)
        except FileNotFoundError:
            print(f"  {name}: not found, skipped")
            continue
        rows = []
        for i, x in enumerate(ds):
            enc = tok(x["context"] + "\n" + x["injected_task"], truncation=False)
            # M = number of Kernel SHAP word features (HF word ids)
            M = len({w for w in enc.word_ids(0) if w is not None})
            rows.append({"idx": i, "tokens": len(enc["input_ids"]), "words": M})
        data[name] = (ds, rows)
        Ls = np.array([r["tokens"] for r in rows])
        Ms = np.array([r["words"] for r in rows])
        token_rows.append({
            "dataset": name, "n": len(Ls), "mean": float(Ls.mean()), "median": float(np.median(Ls)),
            "p90": float(np.percentile(Ls, 90)), "p99": float(np.percentile(Ls, 99)),
            "min": int(Ls.min()), "max": int(Ls.max()),
            "words_mean": float(Ms.mean()), "words_max": int(Ms.max()),
        })
        print(f"  {name:28s} n={len(Ls)} mean={Ls.mean():7.0f} median={np.median(Ls):7.0f} "
              f"p90={np.percentile(Ls, 90):7.0f} max={Ls.max():6d} | words mean={Ms.mean():6.0f} max={Ms.max()}")
    all_L = np.array([r["tokens"] for _, rows in data.values() for r in rows])
    if len(all_L):
        print(f"  ALL ({len(all_L)} prompts): mean={all_L.mean():.0f} median={np.median(all_L):.0f} max={all_L.max()}")
    report["tokens"] = {
        "per_dataset": token_rows,
        "all_mean": float(all_L.mean()) if len(all_L) else None,
        "all_median": float(np.median(all_L)) if len(all_L) else None,
        "per_sample": {n: rows for n, (_, rows) in data.items()},
    }

    dtypes = ["fp32", "fp16"] if args.dtype == "both" else [args.dtype]
    report["profiles"] = {}
    for dt_name in dtypes:
        print("=" * 78 + f"\n##### dtype = {dt_name}")
        model = model.half() if dt_name == "fp16" else model.float()
        weights_gb = gb(sum(p.numel() * p.element_size() for p in model.parameters()))
        free_cuda()
        resident_gb = gb(torch.cuda.memory_allocated())
        print(f"  weights: {weights_gb:.2f} GB | resident now (incl. reserve): {resident_gb:.2f} GB")

        # -------------------------------------------------------------- stage 2
        print("=" * 78 + f"\n[2] peak VRAM grid ({dt_name}) — activation peak above resident, GB")
        grid = []
        print("  L \\ B " + "".join(f"{b:>10d}" for b in args.batches))
        for L in args.lengths:
            line = f"  {L:6d} "
            oom_seen = False
            for B in args.batches:
                if oom_seen:
                    grid.append({"L": L, "B": B, "oom": True, "skipped_after_oom": True})
                    line += f"{'OOM*':>10s}"
                    continue
                ids = make_ids(tok, L, B, dev)
                r = forward_peak(model, ids)
                del ids
                grid.append({"L": L, "B": B, **r})
                if r["oom"]:
                    oom_seen = True
                    line += f"{'OOM':>10s}"
                else:
                    line += f"{r['peak_act_gb']:>10.2f}"
            print(line)
        print("  (OOM* = not tried: a smaller B at this L already OOMed)")

        # -------------------------------------------------------------- stage 3
        print("=" * 78 + f"\n[3] largest batch that fits ({dt_name})")
        maxb = []
        for L in args.lengths:
            lo, hi, best_b = 1, 64, 0
            while lo <= hi:
                mid = (lo + hi) // 2
                ids = make_ids(tok, L, mid, dev)
                r = forward_peak(model, ids)
                del ids
                if r["oom"]:
                    hi = mid - 1
                else:
                    best_b, lo = mid, mid + 1
            maxb.append({"L": L, "max_B": best_b})
            print(f"  L={L:6d}: max B = {best_b}")

        # -------------------------------------------------------------- stage 4
        pts = [(g["L"], g["B"], g["peak_act_gb"] * GIB) for g in grid if not g.get("oom")]
        fit = None
        if len(pts) >= 4:
            y = np.array([m for _, _, m in pts], dtype=float)
            X = np.array([[B * L * L, B * L, 1.0] for L, B, _ in pts], dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            ss_res = float(((y - X @ coef) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
            fit = {"c_bytes_per_BL2": float(coef[0]), "d_bytes_per_BL": float(coef[1]),
                   "e_bytes": float(coef[2]), "r2": 1 - ss_res / ss_tot}
            print("=" * 78 + f"\n[4] fit ({dt_name}): peak_act ~= c*B*L^2 + d*B*L + e")
            print(f"  c = {coef[0]:.1f} bytes per B*L^2 (= {coef[0] / 48:.2f} x one (12,L,L) fp32 tensor; "
                  f"my estimate was ~3)")
            print(f"  d = {coef[1]:.0f} bytes per B*L, e = {gb(coef[2]):.3f} GB, R^2 = {fit['r2']:.4f}")

        def predict_gb(B, L):
            return gb(fit["c_bytes_per_BL2"] * B * L * L + fit["d_bytes_per_BL"] * B * L + fit["e_bytes"])

        # -------------------------------------------------------------- stage 5
        print("=" * 78 + f"\n[5] real samples ({dt_name}) — shortest / median / longest per dataset")
        real = []
        for name, (ds, rows) in data.items():
            order = sorted(rows, key=lambda r: r["tokens"])
            picks = [order[0], order[len(order) // 2], order[-1]][: args.max_real_samples]
            for r in picks:
                x = ds[r["idx"]]
                ids1 = tok(x["context"] + "\n" + x["injected_task"], return_tensors="pt",
                           truncation=False)["input_ids"].to(dev)
                for B in sorted({1, args.perturbations_per_eval}):
                    res = forward_peak(model, ids1.expand(B, -1).contiguous())
                    row = {"dataset": name, "idx": r["idx"], "L": r["tokens"], "B": B, **res}
                    if fit and not res["oom"]:
                        row["predicted_peak_act_gb"] = predict_gb(B, r["tokens"])
                    real.append(row)
                    if res["oom"]:
                        msg = "OOM"
                    else:
                        msg = (f"peak_act={res['peak_act_gb']:.2f} GB (alloc {res['peak_alloc_gb']:.2f}, "
                               f"reserved {res['peak_reserved_gb']:.2f}) {res['seconds']:.2f}s")
                        if "predicted_peak_act_gb" in row:
                            msg += f" | fit predicts {row['predicted_peak_act_gb']:.2f}"
                    print(f"  {name:28s} idx={r['idx']:3d} L={r['tokens']:6d} B={B}: {msg}")
                del ids1

        # -------------------------------------------------------------- stage 6
        print("=" * 78 + f"\n[6] projection ({dt_name}) — predicted peaks vs. free budget")
        budget = total_gb - resident_gb - 0.5  # slack for CUDA context / fragmentation
        t_by_L = {g["L"]: g["seconds"] for g in grid if g["B"] == 1 and not g.get("oom")}

        def sec_per_forward(L):
            if not t_by_L:
                return float("nan")
            Lk = min(t_by_L, key=lambda k: abs(k - L))
            return t_by_L[Lk] * (L / Lk) ** 1.5  # between linear and quadratic

        print(f"  budget for activations: {budget:.1f} GB")
        proj = []
        for name, (_, rows) in data.items():
            Ls = np.array([r["tokens"] for r in rows])
            Ms = np.array([r["words"] for r in rows])
            line = {"dataset": name}
            print(f"  {name}")
            if fit:
                for B in [1, 2, 4, 8]:
                    peaks = np.array([predict_gb(B, int(L)) for L in Ls])
                    line[f"B{B}_n_over_budget"] = int((peaks > budget).sum())
                    line[f"B{B}_mean_peak_gb"] = float(peaks.mean())
                    line[f"B{B}_max_peak_gb"] = float(peaks.max())
                    print(f"      B={B}: {int((peaks > budget).sum()):3d}/{len(Ls)} over budget "
                          f"(mean {peaks.mean():5.1f} GB, max {peaks.max():6.1f} GB)")
            ns = np.minimum(2 * Ms + 2048, 20000)  # Kernel SHAP n_samples ("auto")
            secs = np.array([n * sec_per_forward(int(L)) for n, L in zip(ns, Ls)])
            line.update({
                "kernelshap_n_samples_mean": float(ns.mean()),
                "kernelshap_est_min_per_sample_mean": float(np.nanmean(secs) / 60),
                "kernelshap_est_min_per_sample_max": float(np.nanmax(secs) / 60),
                "kernelshap_est_hours_per_100_samples": float(np.nansum(secs) / 3600),
            })
            print(f"      Kernel SHAP: n_samples mean {ns.mean():.0f}; ~{np.nanmean(secs) / 60:.1f} min/sample "
                  f"(max {np.nanmax(secs) / 60:.1f}); ~{np.nansum(secs) / 3600:.1f} h per 100 samples per attack "
                  "(B=1 timing, rough)")
            proj.append(line)

        report["profiles"][dt_name] = {
            "weights_gb": weights_gb, "resident_gb": resident_gb, "grid": grid,
            "max_batch": maxb, "fit": fit, "real_samples": real, "projection": proj,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("=" * 78 + f"\nDone. Full report: {args.out}\nSend me that JSON and this terminal output.")
    del reserved_tensor


if __name__ == "__main__":
    main()
