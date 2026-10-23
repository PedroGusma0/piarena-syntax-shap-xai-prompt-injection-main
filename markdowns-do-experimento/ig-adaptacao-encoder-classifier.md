---
title: Integrated Gradients — adaptação para explicar um encoder classificador (PromptGuard)
---

# Integrated Gradients: como explicar um encoder classificador (PromptGuard)

> Baseado em `PIArena-main/piarena/xai/ig/` (`xai_ig.py`, `classifier_explainer.py`). Design completo em `PIArena-main/plans/xai-ig-promptguard.md`; uso em `PIArena-main/docs/xai/ig.md`. Complementa [[kernel-shap-adaptacao-encoder-classifier]] (o IG herda dessa classe) e [[custo-computacional-syntaxshap]] (o gargalo que motivou o método). **Estado: implementado e verificado só com um classificador sintético em CPU; ainda sem rodada real com o Prompt-Guard-86M (gated, sem GPU neste sandbox).** Nenhum resultado empírico é afirmado aqui.

## 1. Por que IG

O SyntaxSHAP custa `2^(largura do maior nível)` forward passes (de 2,3 mil a bilhões no `squad_v2`) e o KernelSHAP custa `min(2M+2048, max_n_samples)`. O Prompt-Guard-86M tem pesos abertos (gated), então um método **white box** é possível: o IG custa `n_steps` passes (padrão 50) em batch, independente da sintaxe e quase independente de `M`.

## 2. O que foi reaproveitado e o que é novo

`ClassifierIGExplainer` herda de `ClassifierKernelExplainer` e sobrescreve só `_explain_text`. Reaproveitado sem mudança: tokenização com `.word_ids()` (feature = palavra), alvo binário `P(malign)` (qualquer classe sem "benign" no rótulo), resolução do token de baseline, `make_rescorer` (as métricas Fidelity/acc@1 re-pontuam pelo mesmo mascaramento por palavra), e a localização de `injected_span` em `explain_context`.

Novo: `LayerIntegratedGradients` sobre `model.get_input_embeddings()` (os `input_ids` são discretos, então o gradiente é tomado na saída da camada de embeddings). O baseline mantém `[CLS]`/`[SEP]` e troca cada token de conteúdo pelo `baseline_token` (`mask` por padrão). A atribuição por token (soma sobre a dimensão da embedding) é somada por palavra via `feature_mask`.

## 3. Propriedades e diferenças em relação ao Shapley

- **Completude aproximada**: `Σ values ≈ full_value − base_value`, com erro dado por `ig_convergence_delta` (gravado em cada resultado). O Shapley é exato. Aumente `n_steps` se o delta for grande (o alvo satura perto de 1.0).
- **Sem noção sintática** e sem interações modeladas explicitamente (primeira ordem ao longo de um caminho linear).
- **Soma de subtokens**: palavras fragmentadas (URLs, CJK) acumulam mais termos; no KernelSHAP a palavra é um jogador único.
- **Sensível ao baseline** (`mask` vs `pad`).
- **Consequência para `shap.plots.text`**: o valor de saída exibido é `base + Σ values`; só coincide com `p_malign` até o delta. A escala de cores (normalizada pelo maior |valor| da amostra) não é comparável em magnitude com a do KernelSHAP, só em ranking.

## 4. Como comparar com os outros métodos

Por concordância de ranking (top-k, Spearman) e pelas métricas de fidelidade nas mesmas amostras (ex.: as 4 amostras curadas do piloto). Não por magnitude.

## 5. Verificação feita (sintética, CPU)

Modelo `DebertaV2ForSequenceClassification` minúsculo com pesos aleatórios (rótulos BENIGN/INJECTION/JAILBREAK) + tokenizer do `microsoft/mdeberta-v3-base`. Confirmado: `len(values) == len(tokens)`; `injected_span` correto; `explain_context`/`explain_row` não vazam chaves `_*`; `make_rescorer` com tudo mantido reproduz `p_malign`; `import piarena.xai` não carrega torch/transformers/captum; completude `Σ values ≈ full − base` com `n_steps=2000` (delta ~1e-6 a 1e-4). Observação importante: o delta depende muito de `n_steps` num modelo não-linear (semente 0: delta 1,76 com 50 passos, −0,47 com 300, −6e-5 com 2000), então o padrão de 50 deve ser conferido no Prompt-Guard real. Isso **não** valida nada sobre o Prompt-Guard-86M.
