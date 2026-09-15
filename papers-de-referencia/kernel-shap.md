# A Unified Approach to Interpreting Model Predictions (SHAP / Kernel SHAP)

Scott M. Lundberg, Su-In Lee — Paul G. Allen School of Computer Science, University of Washington.
NIPS 2017 (31st Conference on Neural Information Processing Systems), Long Beach, CA.

> Cópia de referência gerada a partir de `kernel-shapp.pdf` (10 páginas) para consulta rápida. Não é OCR literal — é uma transcrição organizada do conteúdo (texto, equações, achados), preservando a numeração de seções/equações/teoremas do original para permitir citar `kernel-shap.md#4.1` etc. Este é o paper original que introduz **SHAP** como framework unificado e **Kernel SHAP** como um dos seus métodos de estimação — não confundir com [[syntaxshap]] (`syntaxshap.md`), que é um método *derivado* que restringe as coalizões do Shapley clássico a uma árvore de dependência sintática.

## Abstract

Modelos complexos (ensemble, deep learning) atingem melhor acurácia mas são difíceis de interpretar — tensão acurácia vs. interpretabilidade. O paper propõe **SHAP (SHapley Additive exPlanations)**, um framework unificado que atribui a cada feature um valor de importância para uma predição específica. Contribuições: (1) identifica uma classe de métodos de atribuição aditiva de features que unifica 6 métodos existentes (LIME, DeepLIFT, Layer-Wise Relevance Propagation, Shapley regression/sampling values, Quantitative Input Influence); (2) mostra que há uma solução única nessa classe com um conjunto de propriedades desejáveis (valores de Shapley da teoria dos jogos); (3) propõe novos métodos de estimação (**Kernel SHAP**, **Deep SHAP**, **Max SHAP**, **Linear SHAP**) com melhor desempenho computacional e/ou melhor alinhamento com intuição humana.

## 1. Introduction

- Motivação: interpretar corretamente a saída de um modelo gera confiança apropriada, dá insight sobre como melhorar o modelo, e ajuda a entender o processo modelado.
- Três resultados principais: (1) qualquer explicação de uma predição pode ser vista como um modelo em si (o *explanation model*) → define a classe de **additive feature attribution methods** (§2); (2) resultados de teoria dos jogos garantindo solução única se aplicam a *toda* essa classe (§3), levando aos **SHAP values** como medida unificada de importância (§4); (3) novos métodos de estimação de SHAP values, mais alinhados com intuição humana e mais discriminativos entre classes de saída do modelo (§5).

## 2. Additive Feature Attribution Methods

A melhor explicação de um modelo simples é o próprio modelo. Para modelos complexos, usa-se um *explanation model* g — uma aproximação interpretável do modelo original f, local a uma predição f(x) sobre uma entrada x (perspectiva de LIME). Entradas simplificadas x' se relacionam à entrada original via x = h_x(x').

**Definição 1 (Additive feature attribution methods):** o explanation model é uma função linear de variáveis binárias:

```
g(z') = φ₀ + Σ_{i=1}^{M} φᵢ z'ᵢ
```

onde z' ∈ {0,1}^M, M = número de features simplificadas, φᵢ ∈ ℝ.

Seis métodos existentes se encaixam nessa definição:

### 2.1 LIME
`h_x` mapeia vetor binário (feature presente/ausente) → entrada original (ex.: bag-of-words: 1→contagem original, 0→zero; imagens: superpixels, 1→valor original, 0→média dos vizinhos). φ é encontrado minimizando `ξ = argmin_g L(f, g, π_x') + Ω(g)` (Eq. 2) — fidelidade via loss L ponderado pelo kernel local π_x', Ω penaliza complexidade de g. Com g linear e L = squared loss, resolve-se por regressão linear penalizada.

### 2.2 DeepLIFT
Atribui a cada input xᵢ um valor `C_{Δxᵢ Δy}` = efeito de setar esse input para um valor de referência vs. seu valor original. `h_x` mapeia 1→valor original, 0→valor de referência. Propriedade "summation-to-delta": `Σ C_{Δxᵢ Δo} = Δo`. Fazendo φᵢ = C_{Δxᵢ Δo} e φ₀ = f(r), casa com a Eq. 1.

### 2.3 Layer-Wise Relevance Propagation
Equivalente a DeepLIFT com ativações de referência fixadas em zero.

### 2.4 Classic Shapley Value Estimation
Três métodos usam equações clássicas de teoria dos jogos cooperativos:

- **Shapley regression values**: requer retreinar o modelo em todos os subconjuntos S ⊆ F de features. Compara `f_{S∪{i}}(x_{S∪{i}}) − f_S(x_S)` para todo S ⊆ F\{i}. Valor de Shapley (Eq. 4):

  ```
  φᵢ = Σ_{S⊆F\{i}}  [|S|!(|F|−|S|−1)!]/|F|!  · [f_{S∪{i}}(x_{S∪{i}}) − f_S(x_S)]
  ```

  `h_x` mapeia 1→feature incluída no modelo, 0→excluída; φ₀ = f_∅(∅).
- **Shapley sampling values**: aplica aproximação por amostragem à Eq. 4 e aproxima o efeito de remover uma variável integrando sobre amostras do dataset de treino — elimina a necessidade de retreinar e permite computar menos de 2^|F| diferenças.
- **Quantitative Input Influence**: framework mais amplo; propõe independentemente uma aproximação por amostragem quase idêntica a Shapley sampling values.

## 3. Simple Properties Uniquely Determine Additive Feature Attributions

Há uma **única** solução na classe de additive feature attribution methods que satisfaz três propriedades desejáveis — até então conhecidas só para os métodos clássicos de Shapley, não para os outros 5 métodos de §2.

**Propriedade 1 (Local accuracy):** `f(x) = g(x') = φ₀ + Σᵢ φᵢ x'ᵢ` (Eq. 5) — o explanation model deve bater exatamente com f(x) quando x = h_x(x'); φ₀ = f(h_x(0)) = saída do modelo com todos os inputs simplificados "desligados" (ausentes).

**Propriedade 2 (Missingness):** `x'ᵢ = 0 ⟹ φᵢ = 0` (Eq. 6) — features ausentes na entrada original não têm impacto atribuído. Todos os 6 métodos de §2 já obedecem isso.

**Propriedade 3 (Consistency):** sejam `f_x(z') = f(h_x(z'))` e `z'\i` = z' com z'ᵢ zerado. Para quaisquer dois modelos f e f', se `f'_x(z') − f'_x(z'\i) ≥ f_x(z') − f_x(z'\i)` para todo z' (Eq. 7), então `φᵢ(f',x) ≥ φᵢ(f,x)` — se a contribuição marginal de uma feature aumenta ou se mantém (independente das outras features), sua atribuição não pode diminuir.

**Teorema 1:** só existe um explanation model g que segue a Definição 1 e satisfaz as Propriedades 1–3 — e é exatamente a fórmula clássica dos **valores de Shapley** (Eq. 8):

```
φᵢ(f, x) = Σ_{z'⊆x'}  [|z'|!(M−|z'|−1)!]/M!  · [f_x(z') − f_x(z'\i)]
```

Segue de resultados de teoria dos jogos cooperativos (Shapley, 1953; Young, 1985 mostra que valores de Shapley são o único conjunto que satisfaz três axiomas similares a Prop. 1/3 + uma propriedade redundante nesse contexto; Prop. 2 é a peça extra necessária para adaptar as provas à classe de additive feature attribution methods).

**Implicação central:** métodos não baseados em valores de Shapley (LIME e DeepLIFT com seus parâmetros heurísticos originais, por exemplo) necessariamente violam local accuracy e/ou consistency.

## 4. SHAP (SHapley Additive exPlanation) Values

SHAP values = valores de Shapley de uma **função de expectativa condicional** do modelo original — i.e., a solução da Eq. 8 onde `f_x(z') = f(h_x(z')) = E[f(z) | z_S]`, S = índices não-nulos de z' (Figura 1: cada φᵢ é o quanto a expectativa condicional muda ao "revelar" a feature i, começando do valor-base `E[f(z)]` até chegar em `f(x)`; quando o modelo é não-linear ou features não são independentes, a ordem de revelação importa, e os SHAP values vêm da média sobre todas as ordens possíveis).

Como a maioria dos modelos não lida com padrões arbitrários de valores ausentes, aproxima-se `f(z_S)` por `E[f(z) | z_S]`, com duas simplificações opcionais em cadeia:

```
f(h_x(z')) = E[f(z) | z_S]                    (9)  — mapeamento exato
           = E_{z_S̄ | z_S}[f(z)]               (10) — expectativa condicional
          ≈ E_{z_S̄}[f(z)]                       (11) — assume independência de features
          ≈ f([z_S, E[z_S̄]])                    (12) — assume linearidade do modelo
```

### 4.1 Model-Agnostic Approximations

Assumindo independência de features (Eq. 11), SHAP values podem ser estimados via **Shapley sampling values** / Quantitative Input Influence (amostragem sobre uma versão de permutação da Eq. 8, estimativas separadas por feature) — mas **Kernel SHAP requer menos avaliações do modelo original para atingir acurácia de aproximação equivalente** (§5).

#### Kernel SHAP (Linear LIME + Shapley values)

LIME usa um modelo de explicação linear (Eq. 2) que, por ser um additive feature attribution method, tem os valores de Shapley como única solução possível que satisfaz Propriedades 1–3. Mas as escolhas heurísticas de LIME para L, π_x' e Ω **não** recuperam os valores de Shapley — violando local accuracy e/ou consistency, o que leva a comportamento não-intuitivo (§5).

**Teorema 2 (Shapley kernel):** as formas específicas de π_x', L e Ω que tornam a solução da Eq. 2 consistente com as Propriedades 1–3 são:

```
Ω(g) = 0

π_x'(z') = (M − 1) / [ C(M, |z'|) · |z'| · (M − |z'|) ]

L(f, g, π_x') = Σ_{z'∈Z} [f(h_x(z')) − g(z')]² · π_x'(z')
```

Nota importante: `π_x'(z') = ∞` quando `|z'| ∈ {0, M}` (coalizão vazia ou completa) — isso força `φ₀ = f_x(∅)` e `f(x) = Σ_{i=0}^{M} φᵢ` (soma exata = local accuracy exata). Na prática esses pesos infinitos são evitados eliminando analiticamente essas duas variáveis durante a otimização.

Como g(z') é linear e L é squared loss, a Eq. 2 com esse kernel ainda se resolve por **regressão linear ponderada** — ou seja, **valores de Shapley da teoria dos jogos podem ser computados via regressão linear ponderada**. Isso conecta LIME (regressão) com Shapley (teoria dos jogos): já que o mapeamento de entrada simplificada de LIME é equivalente à aproximação da Eq. 12, isso habilita estimação **model-agnostic, baseada em regressão**, dos SHAP values — estimar todos os SHAP values conjuntamente via regressão dá melhor eficiência amostral do que o uso direto das equações clássicas de Shapley (§5).

Intuição: a Eq. 8 é uma diferença de médias; como a média também é a melhor estimativa de mínimos quadrados para um conjunto de pontos, é natural buscar um kernel de ponderação que faça a regressão linear de mínimos quadrados recapitular os valores de Shapley — esse kernel (Shapley kernel, Fig. 2A) é distintamente diferente dos kernels escolhidos heuristicamente em LIME (cosine dist / L2 dist).

### 4.2 Model-Specific Approximations

- **Linear SHAP** (Corolário 1): para modelo linear `f(x) = Σ wⱼxⱼ + b`, assumindo independência de features: `φ₀ = b`, `φᵢ = wⱼ(xⱼ − E[xⱼ])` — direto dos coeficientes, sem precisar de regressão/amostragem.
- **Low-Order SHAP**: regressão via Teorema 2 tem complexidade `O(2^M + M³)` — eficiente só para M pequeno.
- **Max SHAP**: usando formulação por permutação, calcula a probabilidade de cada input aumentar o máximo sobre os outros, em ordem — permite computar Shapley values de uma função max com M inputs em `O(M²)` em vez de `O(M·2^M)`.
- **Deep SHAP (DeepLIFT + Shapley values)**: para redes profundas (composicionais), combina SHAP values de componentes simples (lineares, max pooling, ativação de 1 input — solúveis analiticamente) propagando multiplicadores de DeepLIFT (agora definidos em termos de SHAP values) via regra da cadeia (Eqs. 13–16) — evita ter que escolher heuristicamente como linearizar cada componente, como o DeepLIFT original fazia.

## 5. Computational and User Study Experiments

### 5.1 Computational Efficiency
Comparação Kernel SHAP (lasso debiased) vs. Shapley sampling values vs. LIME, em modelos de árvore de decisão densos (10 features) e esparsos (3 de 100 features), medindo importância de 1 feature vs. número de avaliações do modelo (Fig. 3). **Kernel SHAP converge para o valor de Shapley verdadeiro com muito menos avaliações do modelo** que Shapley sampling values, especialmente com regularização (lasso) — e os valores de LIME divergem significativamente dos valores de Shapley que satisfazem local accuracy/consistency.

### 5.2 Consistency with Human Intuition
User studies (Amazon Mechanical Turk) comparando LIME, DeepLIFT e SHAP contra explicações humanas em dois cenários: (A) score de doença mais alto quando só 1 de 2 sintomas presente; (B) alocação de lucro via função max (3 pessoas, crédito pelo maior score). SHAP teve concordância muito mais forte com explicações humanas que os outros métodos — inclusive resolvendo o problema em aberto de max pooling no DeepLIFT original.

### 5.3 Explaining Class Differences
Rede convolucional MNIST (2 conv + 2 dense + softmax 10 classes). Kernel SHAP e LIME rodados com 50k amostras. Máscara de 20% dos pixels (do 8→3) segundo cada método mostra Kernel SHAP com maior "change in log-odds" (Fig. 5B) que DeepLIFT original/novo e LIME — estimativas mais próximas de SHAP correlacionam com melhor performance na tarefa de mascaramento.

## 6. Conclusion

SHAP identifica a classe de additive feature importance methods (unificando 6 métodos prévios) e mostra que há solução única nessa classe respeitando propriedades desejáveis. Próximos passos sugeridos pelos autores: métodos de estimação mais rápidos e específicos por tipo de modelo com menos suposições, integrar estimativa de efeitos de interação da teoria dos jogos, e definir novas classes de explanation model.

---

## Anexo A — Kernel SHAP na prática: `shap.KernelExplainer` (biblioteca `shap`)

Referência: https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html

```python
class shap.KernelExplainer(model, data, feature_names=None, link='identity', **kwargs)
```

- **`model`**: função (ou `iml.Model`) que recebe uma matriz de amostras (# amostras × # features) e retorna a saída do modelo como vetor/matriz. Para PIArena isso é exatamente o papel do wrapper de classificador (equivalente ao `ClassifierSyntaxExplainer._score` atual).
- **`data`**: dataset de **background** para "apagar" features (imputação — não masking por token nulo). Features "ausentes" são simuladas substituindo pelos valores do background. Para problemas grandes, recomenda usar 1 valor de referência só ou sumarização via k-means (`shap.kmeans`).
- **`link`**: `"identity"` (default) ou `"logit"` — usar `"logit"` quando a saída do modelo é uma probabilidade (caso do PromptGuard).

```python
shap_values(X, nsamples='auto', l1_reg='num_features(10)', silent=False, gc_collect=False, **kwargs)
```

- **`nsamples`**: `'auto'` = `2·M + 2048` (M = nº de features/coalizões simplificadas) — **linear em M**, não exponencial. Valores maiores reduzem variância, custam mais avaliações do modelo.
- **`l1_reg`**: regularização de seleção de features na regressão (`"num_features(k)"`, `"aic"`/`"bic"`, float = alpha do Lasso do sklearn). Ligado à possibilidade de regularização mencionada no paper (§5.1, Fig. 3) para melhorar eficiência amostral.
- Retorno: valores de SHAP por amostra × feature; cada linha soma a diferença entre a predição da amostra e a saída esperada do modelo (local accuracy, Prop. 1, na prática).
- Sem menção explícita a uso em texto/NLP nem avisos de custo computacional específico na doc da API — o custo é regido por `nsamples`, controlável pelo usuário (diferente do `2^M` intrínseco do SyntaxSHAP vendored, ver `CLAUDE.md`).

## Anexo B — Kernel SHAP na prática: `captum.attr.KernelShap` (biblioteca `captum`, PyTorch)

Referência: https://captum.ai/api/kernel_shap.html

```python
class captum.attr.KernelShap(forward_func)
```

- Implementa Kernel SHAP via o framework do LIME (mesma conexão Teorema 2 do paper): "setting the loss function, weighting kernel and regularization terms appropriately in the LIME framework allows theoretically obtaining Shapley Values more efficiently than directly computing them."
- Diferença semântica chave vs. `shap.KernelExplainer`: features "ausentes" são substituídas por **valores de baseline** (fixos, ex. zero ou `[UNK]`/pad token), não amostradas de uma distribuição de background — mais parecido com o `hx` de DeepLIFT (§2.2 acima) que com a expectativa condicional "cheia" do SHAP original.

```python
attribute(inputs, baselines=None, target=None, feature_mask=None,
          n_samples=25, perturbations_per_eval=1, return_input_shape=True,
          show_progress=False)
```

- **`inputs`**: um exemplo por chamada (batch size = 1 recomendado para interpretabilidade por amostra) — relevante porque `explain_row`/`explain_context` do PIArena atual também operam amostra a amostra.
- **`baselines`**: valor(es) de referência para quando a feature interpretável = 0; default zero.
- **`target`**: índice(s) de saída do modelo a explicar — no caso do PromptGuard seria o índice/agregação correspondente a `P(non-benign)`, igual ao design já adotado no `classifier_explainer.py`.
- **`feature_mask`**: agrupa features de entrada em uma única unidade perturbável (ex.: todos os subtokens de uma palavra → 1 grupo). **Isso é diretamente relevante ao "Finding 2" do `CLAUDE.md`** (fragmentação de subtokens inflando artificialmente o custo do SyntaxSHAP vendored): com Kernel SHAP + `feature_mask`, cada palavra (ou até cada sentença) pode ser definida como 1 única feature simplificada, independente de quantos subtokens o tokenizer gerar — o custo deixa de depender da fragmentação do tokenizer.
- **`n_samples`** (default 25!): número de amostras (coalizões) usadas para treinar o surrogate linear — parâmetro explícito e livre, não derivado de nenhuma árvore de dependência ou nível — controla diretamente o trade-off custo/variância.
- **`perturbations_per_eval`**: quantas amostras são processadas por chamada ao forward — permite **batching real**, ao contrário do `self.model([text])` atual (uma chamada por vez) do `ClassifierSyntaxExplainer._score`.
- Fórmula de amostragem do tamanho de coalizão k: `p(k) = (M−1) / [k·(M−k)]` — é exatamente o Shapley kernel do Teorema 2 do paper (a mesma fórmula de `π_x'`, reparametrizada).
- Classe relacionada: `RandomBaselineKernelShap` — usa uma *distribuição* de baselines em vez de um único valor (parâmetro extra `n_baseline_samples`), mais próximo da expectativa sobre o background dataset do `shap.KernelExplainer`.
- Doc não trata explicitamente de custo computacional para texto/NLP nem dá exemplos NLP — exemplos são tensor denso genérico (imagem-like).
