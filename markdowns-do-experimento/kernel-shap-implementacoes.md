# Kernel SHAP: `shap.KernelExplainer` vs. `captum.attr.KernelShap` — diferenças de implementação

> Nota de proveniência: as duas implementações foram lidas diretamente do código-fonte via `raw.githubusercontent.com` (não só da doc oficial). Para `captum.attr._core.kernel_shap.py` e `lime.py` o fetch retornou o **código-fonte literal completo** (classes `KernelShap`, `RandomBaselineKernelShap`, `Lime`, `LimeBase`), citado abaixo com confiança alta. Para `shap.explainers._kernel.KernelExplainer` o fetch retornou um **resumo/paráfrase** do código (feito por um modelo auxiliar do WebFetch), não o texto literal — os pontos marcados com "(via resumo)" têm confiança menor e merecem confirmação lendo `shap/explainers/_kernel.py` diretamente antes de qualquer decisão de implementação crítica. Complementa os Anexos A/B de [[kernel-shap]] (`kernel-shap.md`), que cobrem a API pública; este arquivo foca no *comportamento interno* e nas diferenças que importam para adaptar o motor a um encoder classificador tipo PromptGuard (contexto: `PIArena-main/plans/xai-syntaxshap-promptguard.md`).

## 1. Relação com o paper (Lundberg & Lee, 2017)

Ambas implementam o mesmo resultado teórico — Teorema 2 de [[kernel-shap]] (`kernel-shap.md#4.1`): os valores de Shapley são recuperáveis por regressão linear ponderada com o **Shapley kernel** `π(z') = (M−1) / [C(M,|z'|)·|z'|·(M−|z'|)]`. As duas bibliotecas literalmente implementam a mesma fórmula:

- `captum`, em `KernelShap.attribute()` (código-fonte lido):
  ```python
  num_features_list = torch.arange(num_interp_features, dtype=torch.float)
  denom = num_features_list * (num_interp_features - num_features_list)
  probs = torch.tensor((num_interp_features - 1)) / denom
  probs[0] = 0.0
  ```
  Isso é `p(k) = (M-1) / (k·(M-k))` — a probabilidade de escolher uma coalizão de tamanho `k`, exatamente a reparametrização do kernel de Shapley citada em `kernel_shap_perturb_generator`'s docstring, que também deriva isso passo a passo no código.
- `shap.KernelExplainer` resolve o mesmo problema via mínimos quadrados ponderados diretos, com os pesos vindos da mesma fórmula (via resumo — a forma exata de como o peso é aplicado internamente ao design matrix não foi confirmada literalmente, mas a doc/paper description bate).

Ou seja: **não há divergência teórica** entre as duas — a diferença está inteira na engenharia (o que cada API espera como entrada, como amostra coalizões, como lida com os casos extremos e como escala).

## 2. O que cada API espera como "modelo"

| | `shap.KernelExplainer` | `captum.attr.KernelShap` |
|---|---|---|
| Modelo | função Python `f(X) -> array`, `X` = `numpy.ndarray`/`DataFrame`/matriz esparsa, shape `(#amostras, #features)` | `forward_func: Callable[..., Tensor]` — recebe tensores PyTorch, roda sob `@torch.no_grad()` |
| Formato de saída aceito | vetor `(#amostras,)` ou matriz `(#amostras, #outputs)` (multi-classe tratado nativamente) | escalar por exemplo (via `target` para indexar saída multi-classe) ou escalar por batch |
| Quem escreve a ponte "texto → representação numérica"? | **você**, 100% manual — não existe abstração nativa de texto no `KernelExplainer` puro (isso vive só no `shap.Explainer`/`shap.maskers.Text` de nível mais alto, que por padrão usa o explicador *Partition*, não o Kernel puro) | também você, mas a ponte é sobre **tensores** (ex.: `input_ids`), não sobre strings — evita reconstrução de texto |

Implicação prática para o PromptGuard: com `captum`, o `forward_func` pode operar inteiramente em espaço de token id (mascarar = trocar um id de token por outro, ex. `[MASK]`), sem nunca precisar re-serializar texto e re-tokenizar a cada coalizão. Com `shap.KernelExplainer`, o wrapper que você escreve normalmente reconstrói uma string por linha da matriz de máscara e tokeniza de novo — mais barato de escrever, mas com risco de artefatos de espaçamento/pontuação na reconstrução.

## 3. Definição de "feature" e agrupamento

- **`captum`**: suporte nativo a `feature_mask` — um tensor do mesmo shape do input, valores inteiros `0..num_interp_features-1`, onde posições com o mesmo valor são perturbadas **juntas** (a doc do próprio Captum usa o exemplo de agrupar um bloco 2×2 de uma imagem 4×4 em 1 feature). Para texto: todos os subtokens de uma palavra recebem o mesmo id de grupo → resolve de graça a fragmentação de subtoken (Finding 2 do `CLAUDE.md` do workspace).
- **`shap.KernelExplainer`**: não tem `feature_mask`. `M` = número de colunas da matriz `X`/`data` que você fornece — se você quer que "palavra" seja a unidade, você já precisa ter feito esse agrupamento *antes* de chamar `shap_values`, dentro do seu próprio `f(X)` (cada coluna de `X` já representa uma palavra, e é sua função que sabe expandir isso de volta para posições de token).

Ou seja: `captum` empurra o agrupamento para dentro da biblioteca (declarativo, um tensor); `shap` empurra para fora (imperativo, dentro do seu wrapper). Resultado final equivalente, ergonomia diferente.

## 4. Estratégia de mascaramento ("o que substitui uma feature ausente")

- **`captum.attr.KernelShap`**: substituição por **baseline fixo** — escalar ou tensor (`baselines=None` → zero por padrão). A doc é explícita: *"missing features are represented by the values provided in `baselines`... Captum's KernelShap explains the model output relative to the chosen baseline values, rather than sampling missing features from a background distribution."* Existe uma variante, `RandomBaselineKernelShap` (subclasse), que amostra de uma **distribuição** de baselines (`baselines` vira um tensor com N amostras, `n_baseline_samples` controla quantas usar por coalizão, com média sobre elas — `_aggregate_model_outputs` faz `.view(...).mean(dim=1)`) — essa é a variante que mais se aproxima da noção "cheia" de background dataset do paper (Eq. 10/11 de [[kernel-shap]]).
- **`shap.KernelExplainer`**: pensado em torno de um **background dataset** (`data` no construtor) — a ideia original é imputar valores reais de um conjunto de referência, aproximando a expectativa condicional (Eq. 11). Para dados tabulares isso é natural (linha média, ou k-means do dataset). Para texto, não há "média de tokens" sensata — na prática o `data` degenera para uma única linha "tudo mascarado" (mesmo efeito de um baseline fixo do captum), o que é o padrão usado nos tutoriais oficiais de NLP do `shap`.

**Conclusão prática:** para texto, as duas bibliotecas convergem para a mesma semântica de fato (baseline fixo tipo `[MASK]`/pad), mesmo partindo de fundamentações nominalmente diferentes (baseline explícito vs. background dataset degenerado). A única forma de obter a semântica "cheia" do paper (integrar sobre uma distribuição real) é `RandomBaselineKernelShap` do captum — não há equivalente pronto no `shap.KernelExplainer` para texto (você teria que construir esse dataset de background você mesmo).

## 5. Geração de coalizões: amostragem pura vs. enumeração+amostragem

- **`captum`** (confirmado no código-fonte, `kernel_shap_perturb_generator`): **sempre** amostra por um processo em 3 passos —
  1. sempre gera primeiro a coalizão "tudo incluído" (`ones`) e "tudo excluído" (`zeros`) — os dois pontos extremos;
  2. depois, em loop infinito (até `n_samples` ser atingido pelo chamador), sorteia `k` da distribuição categórica `p(k) ∝ (M-1)/(k(M-k))`, gera um vetor normal aleatório de tamanho M e usa o `k`-ésimo maior valor como limiar para produzir um vetor binário com exatamente `k` uns.
  - É amostragem **pura** — nunca enumera exaustivamente nenhum subconjunto de cardinalidade além dos dois extremos, mesmo que `M` seja pequeno o suficiente para enumerar tudo exatamente.
- **`shap.KernelExplainer`** (via resumo, não confirmado literalmente): a doc/implementação é descrita como capaz de, para `M` pequeno, **enumerar exaustivamente** todas as coalizões de cardinalidade baixa/alta (que têm peso maior no kernel de Shapley) antes de completar o orçamento `nsamples` com amostragem aleatória do restante — essa é a razão histórica de existir `l1_reg="auto"` (mencionado na doc oficial como comportamento pré-0.47.0 ligado a "AIC para <20% do espaço de amostras enumerado"). **Isto precisa ser confirmado lendo o arquivo fonte diretamente** antes de assumir como fato de design — o resumo do WebFetch não deu certeza suficiente sobre o algoritmo exato de particionamento enumeração/amostragem.

Se essa diferença se confirmar, ela é relevante: para `M` moderado (algumas dezenas de palavras), `shap` pode dar uma estimativa *determinística* mais estável nas cardinalidades mais informativas (1, M-1, 2, M-2, ...), enquanto o `captum` é 100% estocástico exceto nos dois extremos — maior variância entre execuções com a mesma seed seria esperada no captum a paridade de `n_samples`.

## 6. Tratamento dos pesos infinitos nos extremos (|z'| ∈ {0, M})

O paper (Teorema 2) exige peso **literalmente infinito** nas coalizões vazia e completa, resolvido eliminando analiticamente essas duas variáveis (garante `local accuracy` exata: `f(x) = Σφᵢ`).

- **`captum`**: aproxima com peso grande porém **finito** — `self.inf_weight = 1_000_000.0`, atribuído em `kernel_shap_similarity_kernel` quando `num_selected_features in {0, M}`. É uma aproximação numérica padrão (evita singularidade), mas não é exata por construção — sobra um erro residual (tipicamente desprezível, mas existe).
- **`shap.KernelExplainer`**: trata os extremos de forma exata/analítica antes de rodar a regressão sobre o restante (via resumo — força `φ₀ = f_x(∅)` e fecha a soma pela restrição `f(x) = Σφᵢ`), o que é mais fiel à prescrição literal do Teorema 2.

## 7. Defaults de orçamento (`nsamples` / `n_samples`)

| Biblioteca | Default | Escala com M? |
|---|---|---|
| `shap.KernelExplainer.shap_values(..., nsamples='auto')` | `2·M + 2048` | Sim, linear em M, com piso de 2048 |
| `captum.attr.KernelShap.attribute(..., n_samples=25)` | **25**, fixo | **Não** — precisa ser ajustado manualmente para M grande |
| `captum.attr.RandomBaselineKernelShap.attribute(..., n_samples=25)` | 25, fixo | Não |

Isso é uma das diferenças mais importantes na prática: usar `captum.attr.KernelShap` com o default em um `context` de centenas de palavras (M grande) equivale a **subamostrar severamente** — nada trava (ao contrário do custo exponencial do SyntaxSHAP), mas a explicação sai com alta variância/baixa fidelidade se `n_samples` não for escalado manualmente (ex.: replicar a heurística do `shap`, `2M+2048`).

## 8. Batching de avaliações do modelo

- **`captum`**: `perturbations_per_eval` — processa várias coalizões perturbadas em um único forward batelado; para `DataParallel`, o batch é dividido entre os dispositivos disponíveis. Reduz diretamente o número de chamadas Python/overhead de forward, mesmo mantendo o mesmo `n_samples` total.
- **`shap.KernelExplainer`**: você mesmo escreve `f(X)` recebendo a matriz sintética inteira (`nsamples × M` linhas de uma vez, ou em blocos que você decide) — o batching é responsabilidade do seu wrapper, não uma opção declarativa da API.

Ambos permitem eliminar o padrão atual do `ClassifierSyntaxExplainer._score` (`self.model([text])`, uma chamada por vez, sem batch) — mas `captum` oferece isso pronto via parâmetro; `shap` exige que você implemente o batching dentro do wrapper.

## 9. Multi-classe / função de link

- **`shap.KernelExplainer`**: aceita `model` retornando matriz `(#amostras, #outputs)` nativamente — devolve SHAP values por classe. Tem `link="logit"` (vs. `"identity"`) para trabalhar em espaço log-odds, útil quando a saída satura perto de 0/1 (esperado num guard classifier bem calibrado).
- **`captum.attr.KernelShap`**: não tem função de link. Multi-classe é tratado via `target` (índice da classe a explicar, calculado uma vez por chamada de `.attribute()` — para explicar múltiplas classes, chama-se várias vezes). Se quiser trabalhar em log-odds, é preciso aplicar a transformação manualmente dentro do `forward_func` antes de retornar.

## 10. Resumo — tabela de decisão

| Critério | `shap.KernelExplainer` | `captum.attr.KernelShap` |
|---|---|---|
| Entrada nativa | função Python / numpy | tensores PyTorch |
| Agrupamento de features (palavra ⊃ subtokens) | manual, dentro do seu `f(X)` | nativo via `feature_mask` |
| Mascaramento | background dataset (degenera p/ baseline fixo em texto) | baseline fixo (ou distribuição, via `RandomBaselineKernelShap`) |
| Amostragem de coalizões | enumeração + amostragem (via resumo, não confirmado no código) | 100% amostragem estocástica (confirmado no código) |
| Extremos (∅, completo) | tratamento exato (via resumo) | aproximado (peso finito 1e6) |
| Orçamento default | `2M+2048` (linear, auto-seguro) | `25` (fixo — perigoso se não ajustado) |
| Batching | manual no wrapper | nativo (`perturbations_per_eval`) |
| Multi-classe / link | nativo, com `link="logit"` | via `target`; sem link, transformar manualmente |
| Melhor encaixe para PIArena/PromptGuard | menos código de biblioteca, mais glue code de texto | menos glue code (token-id direto), mas exige overrides explícitos de `n_samples` e ausência de `link` |

## 11. Nota de validação pendente

Os pontos marcados "(via resumo)" acima (comportamento de enumeração+amostragem do `shap.KernelExplainer.explain`, e o tratamento exato dos extremos) vieram de uma paráfrase do WebFetch sobre `shap/explainers/_kernel.py`, não do código lido literalmente — ao contrário do lado `captum`, onde o código-fonte completo de `kernel_shap.py`/`lime.py` foi obtido e citado diretamente. Antes de tomar qualquer decisão de implementação que dependa desses detalhes específicos do `shap`, vale ler `shap/explainers/_kernel.py` (métodos `explain`, `addsample`, `solve`) diretamente, do mesmo jeito que foi feito para o `captum`.
