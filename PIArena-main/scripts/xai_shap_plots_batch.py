"""Batch driver for xai_shap_plots_test.py: for every raw `*-kernelshap-42.json`
under the `kernelshap_*` result folders, write `bar_global.png` and
`saliency_bars.html` into `<json dir>/xai_metrics/plots/<stem>/`.

Usage (from the repo root):
  python PIArena-main/scripts/xai_shap_plots_batch.py --only dolly_closed_qa
  python PIArena-main/scripts/xai_shap_plots_batch.py --skip-existing
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xai_shap_plots_test as T  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def discover(roots, only=None):
    found = []
    for root in roots:
        for f in glob.glob(os.path.join(glob.escape(root), "**", "*-kernelshap-42.json"), recursive=True):
            rel = os.path.relpath(f, REPO_ROOT).replace(os.sep, "/")
            parts = rel.split("/")
            if "xai_metrics" in parts or "tmp_attack_results" in parts or f.endswith("_metrics.json"):
                continue
            if "shap" in parts:  # squad_v2's `shap/` variant is excluded on purpose
                continue
            if only and only not in rel:
                continue
            found.append(f)
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None, help="default: every kernelshap_* dir in the repo root")
    ap.add_argument("--only", default=None, help="substring filter on the relative path, e.g. dolly_closed_qa")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--max-html-samples", type=int, default=None)
    args = ap.parse_args()

    roots = args.roots or sorted(glob.glob(os.path.join(glob.escape(REPO_ROOT), "kernelshap_*")))
    files = discover(roots, args.only)
    print(f"{len(files)} result file(s) found")

    summary = []
    for n, f in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(os.path.dirname(f), "xai_metrics", "plots", stem)
        label = os.path.relpath(f, REPO_ROOT)
        print(f"\n[{n}/{len(files)}] {label}")
        if args.skip_existing and all(os.path.exists(os.path.join(out, x)) for x in ("bar_global.png", "saliency_bars.html")):
            summary.append(("skipped (exists)", label))
            print("  skipped (already generated)")
            continue
        t0 = time.time()
        try:
            data = T.load_blocked(f)
            if not data:
                summary.append(("skipped (0 blocked with XAI)", label))
                print("  skipped: 0 blocked samples with XAI values")
                continue
            os.makedirs(out, exist_ok=True)
            T.plot_bar_global(data, out)
            T.write_saliency_bars_html(data, out, args.top_k, args.max_html_samples)
            summary.append((f"ok ({len(data)} samples, {time.time() - t0:.0f}s)", label))
        except Exception as e:  # keep going across files
            T.plt.close("all")
            summary.append((f"FAILED: {type(e).__name__}: {e}", label))
            print(f"  FAILED: {type(e).__name__}: {e}")
        finally:
            data = None

    print("\n=== summary ===")
    for status, label in summary:
        print(f"{status:40s} {label}")


if __name__ == "__main__":
    main()
