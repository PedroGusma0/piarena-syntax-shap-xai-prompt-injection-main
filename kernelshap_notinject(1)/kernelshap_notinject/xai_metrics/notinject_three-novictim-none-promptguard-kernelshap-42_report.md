# XAI pilot report — kernelshap (meta-llama/Prompt-Guard-86M)

- Result file: `/home/padg/piarena-syntax-shap-xai-prompt-injection-main/PIArena-main/results/evaluation_results/kernelshap_notinject/notinject_three-novictim-none-promptguard-kernelshap-42.json`
- Samples explained (all, --attack none): 113
- Runtime: 0.13s/sample (14.9s total)
- Timed out (partial explanation, per-sample timeout hit): 0/113 (0%)

## Fidelity(t) / acc@1

| t | Fidelity (lower=better) | acc@1 |
|---|---|---|
| 0.1 | 0.0016502268546450454 | 1.0 |
| 0.2 | 0.00022073024142104966 | 1.0 |
| 0.3 | 0.00018075609629133107 | 1.0 |
| 0.5 | 0.00012202600462246785 | 1.0 |

## Matriz de confusão (--attack none, positivo = benign)

Todos os prompts são benignos: não bloqueado = TP, bloqueado = FN.

| | predito benign (não bloqueado) | predito malign (bloqueado) |
|---|---|---|
| **real benign** | TP = 0 | FN = 113 |
| **real malign** | FP = 0 | TN = 0 |

- Recall (benign): 0.0
- Métrica de alinhamento (span injetado) não calculada: não há injeção.

## Resumo ASR / Utility

Calculado sobre o resultado bruto inteiro (todas as amostras do dataset, não só as explicadas pelo XAI) — o contexto de benchmark em que as métricas de XAI acima se encaixam.

| grupo | n | ASR | Utility |
|---|---|---|---|
| geral | 113 | None | None |
| bloqueadas pela defesa | 113 | None | None |
| não bloqueadas | 0 | None | None |

## Métricas por categoria

Quebra por `category` (tipo de conteúdo malicioso injetado). Fidelity/acc@1 no limiar t=0.5; ASR/Utility calculados sobre todas as amostras da categoria (bloqueadas ou não), mesma semântica da linha "geral" do resumo ASR/Utility acima.

| categoria | n (bloqueadas/total) | Fidelity(t) | acc@1(t) | ASR | Utility |
|---|---|---|---|---|---|
| Common Queries | 19/19 | 0.00012518857654772306 | 1.0 | None | None |
| Multilingual | 29/29 | 1.6853727143386314e-07 | 1.0 | None | None |
| Technique Queries | 41/41 | 0.00016285442724460507 | 1.0 | None | None |
| Virtual Creation | 24/24 | 0.00019701818625132242 | 1.0 | None | None |
