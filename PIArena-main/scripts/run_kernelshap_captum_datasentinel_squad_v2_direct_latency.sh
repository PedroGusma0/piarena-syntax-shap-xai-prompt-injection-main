#!/bin/bash
# squad_v2 + direct + datasentinel + kernelshap (backend captum), registrando latência/VRAM por fase
# (attack, defesa, xai) e o custo do Kernel SHAP por chamada ao detector. Depois roda xai_metrics.py.
# O DataSentinel não tem logits: o Kernel SHAP (captum) explica o detect_flag real dele, com todas as
# palavras do contexto como jogadores (piarena/xai/blackbox/; ver docs/xai/blackbox.md).
# Saídas: <out>/xai_metrics/<stem>_latency.json + <out>/latency.{json,txt} (inclui chamadas ao detector e s/chamada).
# Uso: bash scripts/run_kernelshap_captum_datasentinel_squad_v2_direct_latency.sh
#      LIMIT=5 (default) | LIMIT=2 para um teste ainda mais curto
#      N_SAMPLES=800 bash scripts/<este>.sh        (coalizões do Kernel SHAP; default auto = clip(16*palavras, 400, 3500))
#      BASELINE=neutral bash scripts/<este>.sh     (empty | neutral | clean_context)
# Cada coalizão é uma geração de um Mistral-7B: o tempo por amostra bloqueada pode passar de 10 min.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=squad_v2
ATTACK=direct
DEFENSE=datasentinel
XAI=kernelshap
BACKEND_LLM=${BACKEND_LLM:-Qwen/Qwen3-4B-Instruct-2507}
SEED=${SEED:-42}
LIMIT=${LIMIT:-5}
BASELINE=${BASELINE:-empty}
N_SAMPLES=${N_SAMPLES:-auto}
NAME=kernelshap_captum_datasentinel_latency_${DATASET}_${ATTACK}_${BASELINE}

OUT_ROOT="results/evaluation_results/${NAME}"
RESULT="${OUT_ROOT}/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${XAI}-${SEED}.json"
STEM="$(basename "${RESULT%.json}")"
CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CONFIG"' EXIT
mkdir -p logs
exec > >(tee -a "logs/${NAME}.log") 2>&1

# Run limpo: o resume do main.py pularia amostras e invalidaria os tempos.
rm -f "$RESULT" "${OUT_ROOT}/tmp_attack_results/${DATASET}-${BACKEND_LLM//\//-}-${ATTACK}-${DEFENSE}-${SEED}.json"

# Baseline e n_samples não têm flag no main.py: entram pelo xai_config do YAML.
cat > "$CONFIG" <<YAML
xai_config:
  baseline: ${BASELINE}
  n_samples: ${N_SAMPLES}
YAML

python main.py --config "$CONFIG" --dataset "$DATASET" --backend_llm "$BACKEND_LLM" --attack "$ATTACK" \
  --defense "$DEFENSE" --xai "$XAI" --xai_backend captum \
  --name "$NAME" --seed "$SEED" --no-wandb --limit "$LIMIT"

python - "$RESULT" "$OUT_ROOT" "$N_SAMPLES" "$BASELINE" <<'PYEOF'
import json, os, sys
import numpy as np

path, out_root, n_samples_cfg, baseline = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
all_samples = list(json.load(open(path, encoding="utf-8")).values())
samples = [s for s in all_samples if not (s.get("timing") or {}).get("cold_start")]
KEYS = ("attack_s", "promptguard_s", "xai_s", "defense_total_s")  # promptguard_s = tempo só do detector (detect_time_s)

def stats_of(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "total_s": float(v.sum()), "mean_s": float(v.mean()),
            "median_s": float(np.median(v)), "p90_s": float(np.percentile(v, 90)),
            "p99_s": float(np.percentile(v, 99)), "min_s": float(v.min()), "max_s": float(v.max())}

def stats(key):
    return stats_of([(s.get("timing") or {}).get(key) for s in samples])

# Custo do Kernel SHAP: chamadas ao detector e tempo por chamada (xai_s / n_detector_calls), só amostras explicadas.
explained = [s for s in all_samples
             if ((s.get("xai_result") or {}).get("context") or {}).get("explained")]
calls = [s["xai_result"]["context"].get("n_detector_calls") for s in explained]
per_call = [
    (s.get("timing") or {}).get("xai_s") / s["xai_result"]["context"]["n_detector_calls"]
    for s in explained
    if (s.get("timing") or {}).get("xai_s") and s["xai_result"]["context"].get("n_detector_calls")
]
n_blocked = sum(1 for s in all_samples if (s.get("defense_result") or {}).get("detect_flag") is True)
skips = {}
for s in all_samples:
    ctx = (s.get("xai_result") or {}).get("context")
    if ctx and not ctx.get("explained"):
        skips[ctx.get("skip_reason")] = skips.get(ctx.get("skip_reason"), 0) + 1

summary = {"result": path, "xai_backend": "captum", "xai": "kernelshap (detector booleano)",
           "n_samples_config": n_samples_cfg, "baseline": baseline,
           "n_dataset_samples": len(all_samples), "n_blocked": n_blocked, "n_explained": len(explained),
           "unexplained_by_reason": skips,
           "kernelshap_n_samples": stats_of([s["xai_result"]["context"].get("n_samples") for s in explained]),
           "n_detector_calls": stats_of(calls),
           "xai_seconds_per_detector_call": stats_of(per_call),
           **{k: stats(k) for k in KEYS}}
json.dump(summary, open(os.path.join(out_root, "latency.json"), "w"), indent=2)
lines = [f"xai=kernelshap backend=captum baseline={baseline} n_samples={n_samples_cfg} "
         f"| amostras={len(all_samples)} bloqueadas={n_blocked} explicadas={len(explained)} sem_explicacao={skips}",
         f"{'metrica':<28}{'n':>5}{'media':>10}{'mediana':>10}{'p90':>10}{'p99':>10}{'max':>10}{'total':>10}"]
for k, label in (("attack_s", "attack_s"), ("promptguard_s", "detector_s (detect_time_s)"),
                 ("xai_s", "xai_s"), ("defense_total_s", "defense_total_s")):
    s = summary[k]
    if s:
        lines.append(f"{label:<28}{s['n']:>5}{s['mean_s']:>10.4f}{s['median_s']:>10.4f}"
                     f"{s['p90_s']:>10.4f}{s['p99_s']:>10.4f}{s['max_s']:>10.4f}{s['total_s']:>10.1f}")
for label, key in (("chamadas ao detector", "n_detector_calls"), ("s por chamada (xai_s/chamadas)", "xai_seconds_per_detector_call")):
    s = summary[key]
    if s:
        lines.append(f"{label:<28}{s['n']:>5}{s['mean_s']:>10.4f}{s['median_s']:>10.4f}"
                     f"{s['p90_s']:>10.4f}{s['p99_s']:>10.4f}{s['max_s']:>10.4f}{s['total_s']:>10.1f}")
text = "\n".join(lines)
open(os.path.join(out_root, "latency.txt"), "w").write(text + "\n")
print(text)
PYEOF

python scripts/xai_metrics.py --result "$RESULT" --no-wandb

# Grava também a config do Kernel SHAP no JSON de latência do xai_metrics.
python3 - "${OUT_ROOT}/xai_metrics/${STEM}_latency.json" "$N_SAMPLES" "$BASELINE" <<'PY'
import json, sys
p, n_samples, baseline = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(p))
run = d.setdefault("run", {})
run.update({"xai_backend": "captum", "n_samples": n_samples, "baseline": baseline, "defense": "datasentinel"})
json.dump(d, open(p, "w"), indent=2)
PY
echo "Latência: ${OUT_ROOT}/latency.json, ${OUT_ROOT}/latency.txt e ${OUT_ROOT}/xai_metrics/${STEM}_latency.json"
