# XAI pilot report — kernelshap (meta-llama/Prompt-Guard-86M)

- Result file: `results/evaluation_results/xai_pilot_kernelshap/squad_v2-Qwen-Qwen3-4B-Instruct-2507-direct-promptguard-kernelshap-42.json`
- Samples analyzed: 25
- Runtime: 0.15s/sample (3.6s total)

## Fidelity(t) / acc@1

| t | Fidelity (lower=better) | acc@1 |
|---|---|---|
| 0.1 | 0.25741040354361755 | 0.6 |
| 0.2 | 0.29748857094731646 | 0.68 |
| 0.3 | 0.3342717859550612 | 0.64 |
| 0.5 | 0.21047142956755124 | 0.8 |

## Rank do span injetado (TP vs FN)

- Verdadeiros positivos: 15 (percentil médio: 0.6322875804570262)
- Falsos negativos: 10 (percentil médio: 0.6248650399128337)
