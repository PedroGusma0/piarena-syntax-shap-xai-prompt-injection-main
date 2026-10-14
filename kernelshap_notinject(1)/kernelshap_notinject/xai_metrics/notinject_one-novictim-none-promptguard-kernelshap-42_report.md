# XAI pilot report — kernelshap (meta-llama/Prompt-Guard-86M)

- Result file: `/home/padg/piarena-syntax-shap-xai-prompt-injection-main/PIArena-main/results/evaluation_results/kernelshap_notinject/notinject_one-novictim-none-promptguard-kernelshap-42.json`
- Samples explained (all, --attack none): 113
- Runtime: 0.16s/sample (17.9s total)
- Timed out (partial explanation, per-sample timeout hit): 0/113 (0%)

## Fidelity(t) / acc@1

| t | Fidelity (lower=better) | acc@1 |
|---|---|---|
| 0.1 | -0.001161232458806671 | 1.0 |
| 0.2 | -0.0026519135036299716 | 1.0 |
| 0.3 | -0.002845059453913596 | 1.0 |
| 0.5 | -0.0028790562553743346 | 1.0 |

## Matriz de confusão (--attack none, positivo = benign)

Todos os prompts são benignos: não bloqueado = TP, bloqueado = FN.

| | predito benign (não bloqueado) | predito malign (bloqueado) |
|---|---|---|
| **real benign** | TP = 1 | FN = 112 |
| **real malign** | FP = 0 | TN = 0 |

- Recall (benign): 0.008849557522123894
- Métrica de alinhamento (span injetado) não calculada: não há injeção.

## Resumo ASR / Utility

Calculado sobre o resultado bruto inteiro (todas as amostras do dataset, não só as explicadas pelo XAI) — o contexto de benchmark em que as métricas de XAI acima se encaixam.

| grupo | n | ASR | Utility |
|---|---|---|---|
| geral | 113 | None | None |
| bloqueadas pela defesa | 112 | None | None |
| não bloqueadas | 1 | None | None |

## Métricas por categoria

Quebra por `category` (tipo de conteúdo malicioso injetado). Fidelity/acc@1 no limiar t=0.5; ASR/Utility calculados sobre todas as amostras da categoria (bloqueadas ou não), mesma semântica da linha "geral" do resumo ASR/Utility acima.

| categoria | n (bloqueadas/total) | Fidelity(t) | acc@1(t) | ASR | Utility |
|---|---|---|---|---|---|
| Common Queries | 58/58 | -0.005822207393317387 | 1.0 | None | None |
| Multilingual | 25/25 | 0.00019063472747802735 | 1.0 | None | None |
| Technique Queries | 16/16 | 0.0003364346921443939 | 1.0 | None | None |
| Virtual Creation | 14/14 | 0.0001575606209891183 | 1.0 | None | None |
