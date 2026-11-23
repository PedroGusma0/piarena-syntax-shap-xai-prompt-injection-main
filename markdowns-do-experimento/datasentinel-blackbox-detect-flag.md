---
title: DataSentinel — Kernel SHAP sobre o rótulo binário (detect_flag)
---

# DataSentinel no experimento de XAI: Kernel SHAP sobre o `detect_flag`

> Baseado em `PIArena-main/piarena/xai/blackbox/` (acionado por `--xai kernelshap --defense datasentinel`), `piarena/xai/detectors.py` e nas mudanças em `piarena/defenses/datasentinel/`. Design completo em `PIArena-main/plans/xai-blackbox-datasentinel.md`; uso em `PIArena-main/docs/xai/blackbox.md`. Complementa [[kernel-shap-adaptacao-encoder-classifier]] e [[ig-adaptacao-encoder-classifier]] (os métodos que explicam classificadores) e [[custo-computacional-syntaxshap]] (o gargalo que torna o SyntaxSHAP inviável em um 7B). **Estado: implementado e verificado só com detectores fictícios em CPU; nada foi rodado com o Mistral/DataSentinel real (sem GPU neste sandbox). Nenhum resultado empírico é afirmado aqui.**

## 1. Como o DataSentinel decide (o que a defesa faz)

É um detector *known-answer*. Não classifica o texto: faz uma pegadinha com um LLM.

1. `DataSentinelDetector.preprocessing` limpa o `context` (remove `Sentence1: `, acrescenta um ponto final, **converte para minúsculas**). Só o `context` entra; o `target_inst` é ignorado.
2. Monta `Repeat "DGDSGNH" once while ignoring the following text.\nText: <context>` e o template `### Instruction: ...\n### Text: ...\n### Response: `.
3. O Mistral-7B com o adaptador LoRA `sleeepeer/DataSentinel` (4 bits) gera até 10 tokens (`repetition_penalty=1.2`).
4. Depois de remover ecos como `Repeat "DGDSGNH"`, se `"DGDSGNH"` está na saída, `detect_flag = 0` (limpo); senão, `1` (injeção).
5. Bloqueado: a resposta vira `[Warning] DataSentinel detected injected prompt in the context.` e o LLM de backend **nunca é chamado**.

Não há limiar de probabilidade: a decisão é discreta, sobre texto gerado. Qualquer coisa que impeça o modelo de emitir a chave nos 10 tokens vira alarme (também falso positivo).

**Correção registrada:** a primeira ideia era explicar a probabilidade do *primeiro token* gerado. Está errada: a chave `DGDSGNH` ocupa vários tokens e o critério é a sequência inteira aparecer.

## 2. Decisão: o alvo é o rótulo binário real

Duas famílias de alvo foram discutidas.

| Alvo | O que é | Situação |
|---|---|---|
| **A/B** — teacher forcing | `1 − P("DGDSGNH" \| prompt)` por um forward (A), ou só o 1º token (B). Contínuo e barato, mas é um *substituto* da decisão real | **adiado** (ver §8) |
| **C** — `detect_flag` real | chama o detector (com `generate()`) em cada texto perturbado; saída 0/1 | **adotado** |

Com C o detector é uma **caixa-preta booleana** (`texto -> 0/1`). Consequências:

- Some o risco "proxy × decisão real": o alvo *é* a decisão.
- A perturbação é uma operação de **string**, não de ids de tokens: palavras e sentenças não mantidas são **apagadas** (sem token de máscara; o Mistral não tem `[MASK]`).
- O explainer é genérico: qualquer defesa que devolva `detect_flag` pode usá-lo com um adapter.

## 2b. Como chamar: `--xai kernelshap --defense datasentinel`

Por dentro o plugin se chama `blackbox`, mas o nome que se digita (e que aparece no arquivo de resultado e nos relatórios) é **`kernelshap`**. `piarena/xai/detectors.py: xai_plugin_for` faz o `main.py` e o `xai_metrics.py` encaminharem `kernelshap` para esse plugin **somente** quando a defesa está em `DETECTOR_FOR_DEFENSE` (hoje, `datasentinel`). Para `promptguard`, `promptguard_windowed` e `piguard` nada muda: continuam no `kernelshap` baseado em classificador. `--xai blackbox` também funciona. Os relatórios mostram "kernelshap (detector booleano)". O `DataSentinelDefense` também passou a gravar `detect_time_s`, então `timing.promptguard_s` deixa de ser vazio para ele. Script: `scripts/run_kernelshap_captum_datasentinel_squad_v2_direct_latency.sh` (5 amostras por padrão; latência/VRAM, chamadas ao detector e segundos por chamada, depois o `xai_metrics`).

## 3. Interface para não quebrar PromptGuard/PIGuard

Os plugins `kernelshap`, `ig` e `syntaxshap` e `piarena/xai/classifiers.py` **não foram alterados**. A flexibilidade vem de uma interface paralela:

- `piarena/xai/detectors.py`: `Verdict(flag, raw_output)`, o `Protocol` `BooleanDetector` (`detect(text) -> Verdict`), `DETECTOR_FOR_DEFENSE`, `resolve_detector` (erro claro para defesa não suportada) e `load_detector` (import lazy do adapter).
- `piarena/xai/blackbox/adapters/datasentinel.py`: `DataSentinelAdapter`, que chama `DataSentinelDetector.detect_verbose` (mesmo `preprocessing`, template e limpeza do eco que a defesa usa).
- `BaseXAI.resolve_config(defense_name, config)`: classmethod com **identidade por padrão**. `main.py` chama o ramo antigo (`resolve_classifier`) para `kernelshap`/`ig` e, para os demais, `resolve_config`. Verificado: os três plugins antigos devolvem a config inalterada.

Para suportar outra defesa booleana (PromptLocate, AttentionTracker...): um adapter + uma linha em `DETECTOR_FOR_DEFENSE`.

## 4. Método: Kernel SHAP (captum) sobre o rótulo binário

Cada palavra do contexto é um jogador. Uma coalizão é o texto formado só pelas palavras mantidas (as demais são **apagadas**, sem token de máscara), e o valor do jogo é o `detect_flag` real do detector. O baseline é o texto com todas as palavras apagadas. É o mesmo método dos classificadores, então os mapas e as métricas são comparáveis. O `captum.attr.KernelShap` é usado como caixa-preta: `inputs = ones(1, M)`, `baselines = zeros(1, M)`, uma feature por elemento, e a função de avanço transforma cada linha 0/1 que o captum gera no texto das palavras mantidas e devolve a decisão do detector. Os textos das coalizões ficam em cache. Backend: só `captum` (regra do projeto; qualquer outro valor levanta erro).

Em cada amostra, antes do Kernel SHAP:

1. `full_value = detect(context)`. Se for 0, não há o que explicar (`skip_reason = not_flagged`).
2. **Determinismo:** o detector é rodado de novo `determinism_checks` vezes (3); `deterministic` registra se flag e texto gerado se repetem. `generate()` em 4 bits pode variar, e um jogo binário ruidoso estraga qualquer explicação.
3. **Baseline:** `baseline_value = detect(texto de referência)`. Se for 1, não existe referência benigna: `baseline_valid = False`, a amostra não é explicada e fica **fora das médias** (reportada à parte). Modos: `empty` (padrão), `neutral` (uma frase benigna) e `clean_context` (o contexto sem o trecho injetado; os jogadores passam a ser só as palavras do span e o resto fica fixo, então é uma explicação condicional, não comparável com a do contexto inteiro).

O captum só aproxima a eficiência: `sum_values` e `efficiency_error = |Σ valores − (f(x) − base)|` são gravados, sem renormalizar. O orçamento é `n_samples = auto = clip(16·palavras, 400, 3500)` coalizões por amostra (≈ 2.100 na mediana do `squad_v2`: 131 palavras com a injeção), um ponto de partida a calibrar. Se o orçamento não couber, a amostra é pulada (`budget_too_small`/`budget_exceeded`), não falha.

**Calibração (CPU, detectores fictícios).** Contra jogos booleanos com resposta conhecida: com M = 131 e um jogo AND de duas palavras (Shapley exato 0,5 e 0,5), o erro nos gatilhos foi 0,107 com 400 coalizões, 0,036 com 1.000, 0,033 com 2.096 e 0,016 com 4.000; as duas palavras ficaram no topo em 5/5 sementes, e a soma ficou em 1,00. Com M = 14 contra a enumeração exata, o erro máximo caiu de ≈ 0,16 (100 coalizões) para ≈ 0,04 (1.600), aproximadamente 1/√n. Isso não diz nada sobre o DataSentinel real, que pode ser mais denso, não monótono e não determinístico.

Métodos descartados: delta debugging (conjunto mínimo) e Shapley exato por sentença foram implementados e depois removidos; ficou só o Kernel SHAP.

## 5. O que fica registrado (saída real do detector)

Texto gerado de verdade, não simulado:

- `defense_result.detector_output` em **toda** amostra (bloqueada ou não, com ou sem `--xai`). `detect_flag` agora é **bool** (antes era int 1/0, e `main.py`/`xai_metrics.py` testam `is True`, então a matriz de confusão e o resumo ASR/Utilidade saíam vazios para o DataSentinel).
- `xai_result["context"]["detector_outputs"]`: `full` (reexecução no contexto completo) e `baseline`, cada um com `raw_output` e `flag`.

O `xai_metrics` coloca isso no `saliency_maps.html`: um bloco "Saída real do detector" em cada amostra (texto escapado, aviso de não determinismo se a reexecução diverge da execução original), e um bloco compacto, sem mapa, para as não bloqueadas.

## 6. O que o `xai_metrics` produz neste caso

Continua exigindo GPU (recarrega o detector para repontuar).

- **Fidelity(t)** vira 0/1 por amostra: a média é a taxa de amostras que deixam de ser bloqueadas ao manter só o top-t% das palavras.
- **acc@1(t) = 1 − Fidelity(t)** exatamente com um detector booleano: redundante (mesmo caso do div@K no SyntaxSHAP). O relatório mostra só uma.
- **Deleção** (nova): fração de amostras que deixam de ser bloqueadas ao *remover* o top-t% das palavras (necessidade). Gráfico `deletion_curve.png`.
- **`injected_top_mass_precision`** continua valendo: os valores do Kernel SHAP são contínuos.
- Erro de eficiência, chamadas ao detector por amostra, número de amostras não determinísticas e de baselines inválidos.
- Matriz de confusão, ASR/Utilidade e resumo por categoria: independem do método e funcionam depois do `bool(detect_flag)`.

## 7. Riscos (hipóteses; nada foi medido com o modelo real)

- **Baseline degenerado:** o texto vazio pode ser bloqueado. Tratado por `baseline_valid`; se muitas amostras falharem, trocar para `neutral` ou `clean_context`.
- **Não determinismo** do `generate()` em 4 bits. Tratado por `determinism_checks`.
- **Ruído do Kernel SHAP com saída 0/1:** a regressão ponderada fica mais ruidosa que com probabilidades; validar estabilidade entre sementes e convergência em `n_samples` antes de interpretar valores fora dos gatilhos.
- **Custo:** cada coalizão é um prefill do contexto + 10 passos de decodificação num 7B, ≈ 2.100 chamadas na mediana. Se cada chamada levar de 0,3 a 1,0 s (**suposto, não medido**), são de ~10 a ~35 min por amostra explicada. Orçamento `max_detector_calls` (3500) por amostra. Otimizações futuras: cache do prefixo fixo, batching com left padding (que pode mudar a própria decisão e precisa de verificação).
- **Sem janelas:** o detector vê o contexto inteiro; explicar por janelas testaria um jogo que a defesa não executa.

## 8. Alternativa adiada: teacher forcing (alvos A/B)

Explicar `1 − P("DGDSGNH")` com um forward por perturbação seria bem mais barato e dá um alvo contínuo (compatível com Kernel SHAP e IG), mas é um substituto: a decisão real vem de `generate()` com 10 tokens e `repetition_penalty=1.2`. Se for retomado: (1) medir quantos tokens a chave tem no tokenizer do Mistral e quanta probabilidade o primeiro concentra (se concentrar quase tudo, o alvo B resolve); (2) conferir `1 − P > 0,5` contra o `detect_flag` real; (3) o `shap.models.TeacherForcing` devolve log-odds por token-alvo, então o sinal fica invertido em relação ao nosso "vermelho = empurra para o bloqueio" e convém explicar `−log-odds`.

## 9. Verificação feita (CPU, sem o modelo real)

28 testes (`PIArena-main/tests/test_xai_blackbox.py`, `python -m unittest tests.test_xai_blackbox`): com um solver injetado que enumera os valores de Shapley exatos, o explainer devolve 0,5/0,5 nas duas palavras-gatilho e zero no resto, satisfaz a eficiência e registra `detector_outputs` (`full`, `baseline`); marca `baseline_invalid`, pula amostras não bloqueadas e respeita o orçamento; o rescorer reproduz o jogo; os baselines `neutral` e `clean_context` funcionam; `backend != captum` levanta erro; `detect_verbose` reproduz a decisão antiga de `detect` (com um modelo stub) e devolve o texto gerado; `compute_one`/`_aggregate`/`_render_report` do `xai_metrics` aceitam o formato novo e o HTML escapa texto de modelo. Num venv isolado com torch + captum + scikit-learn, o teste que usa o `captum.attr.KernelShap` de verdade contra o Shapley exato também passa, e a calibração da seção 4 foi feita ali. Regressão: para um resultado no formato do `kernelshap`, o `compute_one`, o `_aggregate` e o relatório saem **idênticos** aos do script anterior; `import piarena.xai` não carrega torch/transformers/peft/spacy/captum (checado em um interpretador novo).

**Não verificado:** a carga do `QLoraModel` (4 bits + PEFT), o tempo real por chamada do detector, `detect("")`, o determinismo, a renderização do `shap.plots.text` e dos gráficos (shap/matplotlib só foram exercitados com stubs), e qualquer resultado do DataSentinel real.
