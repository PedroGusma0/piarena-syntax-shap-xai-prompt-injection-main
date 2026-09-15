# Métricas do experimento SyntaxSHAP × PromptGuard (PI Arena)

Documento de referência sobre as métricas usadas para avaliar o `--xai syntaxshap` no PI Arena — o que cada uma mede, como é calculada, e em que ela difere da métrica correspondente no paper original do SyntaxSHAP (arXiv:2402.09259). Ver também `papers-de-referencia/syntaxshap.md` (transcrição do paper) e `PIArena-main/plans/xai-syntaxshap-promptguard.md` (design completo do experimento).

## Onde isso vive no código

- `PIArena-main/piarena/xai/metrics.py` — funções puras de cálculo (não dependem de torch/transformers/spacy).
- `PIArena-main/scripts/xai_metrics.py` — CLI que lê o JSON de resultado do `main.py --xai syntaxshap`, recarrega o classificador PromptGuard, roda as re-avaliações mascaradas e escreve `*_metrics.json` + `*_report.md` + saliency maps (`shap.plots.text`/`bar`).

As métricas **não** são calculadas dentro de `main.py` — ele só grava o `xai_result` bruto (tokens + valores SyntaxSHAP + `p_non_benign` + `predicted_label`). O cálculo das métricas é uma etapa de análise separada, porque Fidelity/acc@1 exigem rodar o classificador de novo várias vezes por amostra (uma vez por threshold `t`).

## Contexto necessário: o que é `f_ŷ(S)` aqui

No paper, `f_ŷ(S)` é a probabilidade de um LM autoregressivo prever um token-alvo específico dado um subconjunto mascarado `S` de tokens de entrada. Nesse experimento, o alvo não é geração — é o classificador `Prompt-Guard-86M`, que devolve 3 classes (`BENIGN`/`INJECTION`/`JAILBREAK`). Para manter uma quantidade explicada consistente entre amostras, o experimento agrega:

```
f_ŷ(S) = P(non-benign | S) = P(INJECTION | S) + P(JAILBREAK | S)
```

Todas as métricas abaixo giram em torno de comparar esse `f_ŷ` entre o texto completo e uma versão mascarada, seguindo o ranking de importância dado pelo SyntaxSHAP. Note que isso já implica que o experimento nunca compara o vetor de 3 classes cru — sempre o par agregado `[P(BENIGN), P(MALIGN)]` (`MALIGN = INJECTION + JAILBREAK`), o que tem uma consequência para a métrica div@K original do paper — ver seção "O que existia e foi removido" abaixo.

## 1. Fidelity — `fidelity()`

**O que mede:** o quanto a confiança do modelo na decisão muda quando se mantém só os tokens que a explicação apontou como mais importantes.

**Como é calculado** (Eq. 6 do paper, reinterpretada):
1. Para cada threshold `t ∈ {0.1, 0.2, 0.3, 0.5}`, seleciona os top-`t%` tokens por `|valor SyntaxSHAP|`.
2. Mascara todo o resto (substitui por token neutro).
3. Roda o PromptGuard no texto mascarado.
4. `Fid(t) = P(non-benign)_completo − P(non-benign)_mascarado`.

**Leitura:** quanto **mais perto de 0**, melhor — significa que os tokens mantidos já bastam sozinhos para reproduzir a confiança original na decisão.

**Diferença vs. paper:** a fórmula é idêntica (é a variante "keep" do Eq. 6); só muda o que `f_ŷ` representa — lá é a probabilidade do próximo token gerado, aqui é `P(non-benign)` do classificador.

## 2. acc@1 — `acc_at_1()`

**O que mede:** se a **decisão operacional** do PromptGuard (o mesmo corte que `PromptGuardDefense` usa: `P(non-benign) ≥ 0.5` → `detect_flag=True`) se mantém ao mascarar.

**Como é calculado:**
1. `full_decision = P(non-benign)_completo ≥ 0.5`.
2. Para cada `t`, `masked_decision = P(non-benign)_mascarado ≥ 0.5`.
3. `acc@1(t) = (masked_decision == full_decision)` — booleano por amostra, agregado como taxa (`acc_at_1_rate`) no relatório final.

**Por que "@1" e não "@K" como no paper:** o paper usa `acc@K` comparando overlap entre as top-K predições de um vocabulário enorme (K=10, porque várias continuações são plausíveis). Aqui o alvo é binário — só existe uma decisão para comparar, então o único `K` que não é trivial é `K=1`.

**Diferença vs. Fidelity:** Fidelity é a versão *contínua* dessa mesma pergunta (o quanto a confiança caiu); acc@1 é a versão *binarizada* (só importa se cruzou o corte de 0.5). Uma queda de Fidelity de 0.50 pode ou não virar um `acc@1=False`, dependendo de onde a probabilidade original estava.

## 3. Rank do span injetado (percentil) — `injected_span_percentile()`

**O que mede:** se o SyntaxSHAP aponta o trecho **realmente injetado** (`injected_task`, cuja posição já é conhecida — foi o próprio módulo de Attack que o inseriu) como importante, sem precisar de nenhuma anotação humana.

**Como é calculado:**
1. Rankeia todos os tokens do `context` por `|valor|` (ordem crescente).
2. Converte o rank em percentil (0 a 1, onde 1 = token mais importante do texto todo).
3. Tira a média dos percentis dos tokens que caem dentro do `injected_span`.

**Uso no relatório:** comparado entre amostras onde o PromptGuard **detectou** o ataque (`detect_flag=True`, verdadeiro positivo) vs. onde **não detectou** (`detect_flag=False`, falso negativo) — pergunta se, mesmo quando a defesa falha, a explicação ainda aponta corretamente onde está o ataque.

**Diferença vs. paper:** é a reinterpretação da **Semantic Alignment** (Seção 5.4). No paper, o "gabarito" era uma anotação humana manual (22 exemplos rotulados à mão, marcando a negação como o token que deveria importar). Aqui o gabarito é automático e objetivo: a posição do `injected_task` já é conhecida de antemão, sem precisar de rotulagem manual.

## O que existia e foi removido

### div@K

Havia uma métrica `div@K` (Eq. 7 do paper, `div_at_k()` em `metrics.py`) que comparava o vetor de probabilidade **das 3 classes cruas** (`[P(BENIGN), P(INJECTION), P(JAILBREAK)]`) via distância L1 entre o texto completo e o mascarado. Enquanto o vetor comparado era o cru de 3 classes, ela não era redundante com Fidelity/acc@1: esses dois só enxergam a **soma** `P(INJECTION)+P(JAILBREAK)`, então uma migração de massa de `INJECTION` para `JAILBREAK` que não mudasse essa soma passava despercebida por eles, mas não pelo `div@K` (que soma os módulos por classe *antes* de somar entre classes — não permite esse cancelamento). Exemplo: texto completo `[0.05, 0.90, 0.05]`, mascarado `[0.05, 0.40, 0.55]` → soma não-benigna igual (0.95 nos dois, `Fidelity=0`), mas `div@K = |0.05−0.05|+|0.90−0.40|+|0.05−0.55| = 1.00`.

Foi **removida** quando o experimento decidiu explicar e reportar sempre o par agregado `[P(BENIGN), P(MALIGN)]` (`MALIGN = INJECTION+JAILBREAK`) em vez do vetor cru — porque, com só 2 classes complementares (`P(BENIGN)+P(MALIGN)=1`), qualquer delta em `P(MALIGN)` é espelhado por um delta de mesma magnitude no sentido oposto em `P(BENIGN)`, e a fórmula colapsa algebricamente:

```
div@K(t) = |P(BENIGN)_full − P(BENIGN)_masked| + |P(MALIGN)_full − P(MALIGN)_masked|
         = 2 · |P(MALIGN)_full − P(MALIGN)_masked|
         = 2 · |Fidelity(t)|
```

Essa identidade vale exatamente — amostra por amostra, threshold por threshold, não é uma correlação observada empiricamente. No exemplo acima, recalculado sobre `[P(BENIGN), P(MALIGN)] = [0.05, 0.95]` nos dois casos, `div@K` cai pra `0`, idêntico à `Fidelity`. Reportar as duas lado a lado no relatório mostraria a mesma informação duas vezes com um fator de escala, então a métrica foi retirada em vez de recalculada — ver `plans/xai-syntaxshap-promptguard.md`, seção "Por que agregar em binário", para a versão completa do argumento.

### Concordância `syntax` vs. `syntax-w` (`spearman()`)

O paper também define uma variante ponderada por nível da árvore de dependência ("SyntaxSHAP-W", Seção 3.4, Eq. 5): em vez de cada palavra pesar igual no cálculo do Shapley value (`"syntax"`, Eq. 3), pondera por `1/nível` — palavras perto da raiz (sujeito/verbo) pesam mais que modificadores periféricos. Havia uma métrica extra (`spearman()` em `metrics.py`, mais `--result-w`/`syntax_vs_syntax_w_spearman` em `xai_metrics.py`) que rodava o pipeline duas vezes (uma com `--algorithm syntax`, outra com `--algorithm syntax-w`) e correlacionava (Spearman) o ranking de importância de token entre as duas, por amostra — perto de 1 = a ponderação por nível não muda muito o ranking; perto de 0 = muda bastante.

Foi **removida junto com o próprio `algorithm="syntax-w"`**: o experimento decidiu explicar sempre com o SyntaxSHAP padrão (`"syntax"`, sem ponderação), então não sobra uma segunda variante pra comparar — manter o parâmetro `weighted`/`algorithm="syntax-w"` em `coalition.py`/`classifier_explainer.py` só pra uma opção nunca de fato usada era superfície morta. Removido de `piarena/xai/syntaxshap/coalition.py` (o parâmetro `weighted` e o branch `"syntax-w"` em `compute_shapley_values`), `piarena/xai/syntaxshap/classifier_explainer.py` (`self.weighted`, os branches `algorithm in ("syntax", "syntax-w")`), `piarena/xai/syntaxshap/xai_syntaxshap.py` (comentário do `DEFAULT_CONFIG`), `piarena/xai/metrics.py` (`spearman()`) e `scripts/xai_metrics.py` (`--result-w`, `syntax_vs_syntax_w_spearman`).

### Trigger-word ranks

Havia uma sexta métrica (`trigger_word_ranks`, no `metrics.py`, e um bloco `semantic_alignment`/`--trigger-words` no `xai_metrics.py`) que buscava o rank de palavras-gatilho curadas (`"ignore"`, `"disregard"`, `"override"`, etc.) dentro do `injected_span`. Foi **removida** por ser redundante com a métrica 3 (`injected_span_percentile`, que já mede se o span inteiro — não palavra por palavra — é apontado como importante) e por depender de uma lista de palavras arbitrária e específica de idioma/domínio, que não generaliza bem entre datasets/ataques do PI Arena.

## O que existe no paper e não foi portado

- **Coherency** (Seção 5.3): compara pares de sentenças quase idênticas (com/sem uma pequena variação sintática) e verifica se a explicação muda de forma proporcional à mudança de comportamento do modelo. Não portado porque o PI Arena não fornece pares prontos desse tipo — exigiria gerar variantes perturbadas do `injected_task` como uma etapa nova de geração de dado, não só rodar `main.py` como está.
- **Semantic Alignment original** (Seção 5.4, baseada em negação + anotação humana): substituída pela métrica 4 acima, que usa o `injected_span` (já conhecido automaticamente) em vez de rótulos humanos.

## Resumo — tabela comparativa com o paper

| Métrica no experimento | Equivalente no paper | O que mudou |
|---|---|---|
| Fidelity(t) | Fid(t), Eq. 6 | alvo: `P(non-benign)` do classificador em vez de prob. do próximo token gerado |
| acc@1 | acc@K, Eq. 8 | K=10→K=1 (decisão binária em vez de overlap num vocabulário grande) |
| injected_span_percentile | Semantic Alignment, Seção 5.4 | gabarito automático (span conhecido) em vez de anotação humana manual |
| ~~div@K~~ | ~~div@K, Eq. 7~~ | removida — sobre o par agregado `[P(BENIGN), P(MALIGN)]` ela vira algebricamente `2·|Fidelity(t)|`, redundante por construção |
| ~~spearman (syntax vs. syntax-w)~~ | ~~—~~ | removida — `algorithm="syntax-w"` foi tirado do código; só resta `"syntax"`, sem segunda variante pra comparar |
| ~~trigger-word ranks~~ | ~~—~~ | removida (redundante com injected_span_percentile) |
| Coherency (Seção 5.3) | — | não portado (precisa de pares de sentenças perturbadas, ainda não gerados) |
