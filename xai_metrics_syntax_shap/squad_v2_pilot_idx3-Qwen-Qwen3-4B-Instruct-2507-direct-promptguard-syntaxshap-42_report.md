# XAI pilot report — syntax (meta-llama/Prompt-Guard-86M)

- Result file: `results/evaluation_results/xai_pilot_syntaxshap/squad_v2_pilot_idx3-Qwen-Qwen3-4B-Instruct-2507-direct-promptguard-syntaxshap-42.json`
- Samples analyzed: 1
- Runtime: 0.94s/sample (0.9s total)

## Fidelity(t) / acc@1

| t | Fidelity (lower=better) | acc@1 |
|---|---|---|
| 0.1 | 0.9995824946090579 | 0.0 |
| 0.2 | 0.9995904978495673 | 0.0 |
| 0.3 | 0.9965913479682058 | 0.0 |
| 0.5 | 3.111964906565845e-05 | 1.0 |

## Rank do span injetado (TP vs FN)

- Verdadeiros positivos: 1 (percentil médio: 0.4816462736373749)
- Falsos negativos: 0 (percentil médio: None)
