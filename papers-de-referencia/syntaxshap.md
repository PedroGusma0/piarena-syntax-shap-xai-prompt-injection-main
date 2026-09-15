# SyntaxShap: Syntax-aware Explainability Method for Text Generation

Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady — Department of Computer Science, ETH Zurich.
arXiv:2402.09259v2 [cs.CL], 3 Jun 2024.

> Cópia de referência gerada a partir de `syntaxshap.pdf` (16 páginas) para consulta rápida. Não é OCR literal — é uma transcrição organizada do conteúdo (texto, equações, achados), preservando a numeração de seções/equações/figuras do original para permitir citar `syntaxshap.md#3.2` etc.

## Abstract

SyntaxShap é um método de explicabilidade local, model-agnostic, para geração de texto, que incorpora a sintaxe da sentença de entrada. Estende os valores de Shapley para respeitar dependências sintáticas via parsing (spaCy). Avaliado contra baselines SHAP-based em fidelidade, coerência e alinhamento semântico. Produz explicações mais fiéis e coerentes que métodos SHAP padrão para modelos autoregressivos (AR).

## 1. Introduction

- Motivação: interpretabilidade de LMs é importante para domínios safety-critical; SHAP é popular mas pouco explorado para tarefas sequence-to-sequence (next-token generation).
- Método (Fig. 1): dado uma sentença de entrada, um AR LM prevê o próximo token. A sintaxe é extraída via dependency parsing (spaCy). Para medir a importância de uma palavra para a predição do próximo token: (1) extrai múltiplas coalizões de palavras seguindo caminhos específicos na árvore de dependência, (2) mede a contribuição de adicionar a palavra a cada coalizão na mudança de probabilidade de prever o token-alvo, (3) faz a média dessas contribuições → valor SyntaxShap final.
- Contribuições: (1) SyntaxShap, método SHAP-based com árvore de dependência; (2) métricas quantitativas (fidelity variants) e qualitativas (coerência, alinhamento semântico); (3) avaliação em dois AR LMs (GPT-2, Mistral 7B).

## 2. Related Work

- **SHAP-based explainability em NLP**: métodos perturbation-based (LIME, SHAP) vs surrogate. Shapley values (Shapley, 1953) dão explicações locais atribuindo mudanças de predição a features.
- **Shapley values e dependências complexas**: SHAP assume independência de features. Frye et al. (2020) propõem *Asymmetric Shapley values* (derrubam a simetria, permitem incorporar dependência causal conhecida — base direta do SyntaxShap). Heskes et al. (2020): *Causal Shapley values*. Chen et al. (2019): coalizões baseadas em estrutura de grafo. HEDGE (Chen et al., 2020): clusteriza palavras via *cohesion score* minimizando perda de fidelidade — SyntaxShap propõe em vez disso coalizões que respeitam relações sintáticas da árvore de dependência.

## 3. SyntaxShap Methodology

### 3.1 Objective

Dada uma sentença de n palavras **x** = (x₁,...,xₙ) e **ŷ** = (ŷ₁,...,ŷₘ) as m palavras geradas por um AR LM f, o objetivo é avaliar a importância de cada token de entrada para a predição ŷ. O paper foca em explicar o *próximo token* (m=1).

> **Definição central (agnóstica ao tipo de tarefa):** seja `f_y(x)` a probabilidade predita pelo modelo de que a entrada x tem próximo token y. O método produz explicações locais.

Nota de leitura: essa definição é formalmente "probabilidade do modelo para uma classe-alvo y dada a entrada x" — a instanciação como "próximo token" é a escolha de aplicação do paper (título/escopo: AR LMs), não uma exigência da matemática do jogo de coalizão em si (ver 3.2–3.3).

### 3.2 Shapley values approach

Abordagem game-theoretic clássica (Shapley, 1953): para cada coalizão de features, computa a contribuição marginal da feature i (diferença de importância da coalizão com e sem i), agregada sobre todos os subconjuntos.

### 3.3 Syntax-aware coalition game

Diferente do Shapley clássico, SyntaxShap só considera coalizões **permitidas** (𝔖), restritas pela estrutura da árvore de dependência.

- Coalizão S: subconjunto de palavras {xᵢ, i∈[1,n]} da sentença x.
- Árvore de dependência com L níveis; lᵢ ∈ [1,L] = nível da palavra xᵢ; nₗ > 0 = número de palavras no nível l.
- SyntaxShap só considera as coalizões permitidas 𝔖 = ⋃_{l=0}^{L} 𝔖_l, onde 𝔖_l é o conjunto de coalizões permitidas no nível l. 𝔖₀ = {S₀}, S₀ = {} (coalizão nula).

**Propriedade** (cada coalizão S ∈ 𝔖_l respeita):
1. ∀i ∈ [1,n] t.q. lᵢ > l, xᵢ ∉ S.
2. ∀i ∈ [1,n] t.q. lᵢ < l, xᵢ ∈ S.

Ou seja: uma coalizão no nível l sempre contém todas as palavras de níveis anteriores e nenhuma de níveis posteriores — só varia quais palavras do próprio nível l entram.

**Contribuição da feature xᵢ no nível lᵢ (Eq. 3):**

```
φᵢ = (1/Nᵢ) · Σ_{S ∈ (⋃_{p=0}^{lᵢ-1} 𝔖_p) ∪ 𝔖_lᵢ\i}  [ f_ŷ(S ∪ {xᵢ}) − f_ŷ(S) ]
```

onde Nᵢ = número de coalizões permitidas no nível lᵢ que não contêm xᵢ, e 𝔖_l^{\i} = coalizões no nível l que excluem xᵢ = ⋃_{σ∈𝒫(X_l)} X_{<l} ∪ (σ\{xᵢ}).

**Propriedade (número de coalizões por nível, Eq. 4):**
```
N_l = Σ_{p=0}^{l-1} 2^{n_p} + 2^{n_l - 1} − l
```
(prova completa no Apêndice, derivada recursivamente a partir da raiz).

Ponto-chave para reuso do método: **nada em 3.2/3.3 depende do tipo de saída do modelo.** `f_ŷ(S)` é tratado como uma caixa-preta que devolve um score escalar (probabilidade) para uma classe/alvo ŷ fixado, dado um subconjunto mascarado S de tokens de entrada. A árvore de dependência e a enumeração de coalizões (𝔖_l, causal_ordering) dependem só da sentença de entrada, nunca da saída do modelo.

### 3.4 Weighted SyntaxShap (SyntaxShap-W)

Hipótese: palavras no topo da árvore (mais próximas da raiz) são sintaticamente mais fundamentais (sujeito, verbo) e devem pesar mais. Peso `w_l = 1/l` (inverso do nível). Contribuição ponderada (Eq. 5):

```
φᵢ = (w_{lᵢ}/Nᵢ) · Σ_{S ∈ (⋃_{p=0}^{lᵢ-1} 𝔖_p) ∪ 𝔖_lᵢ\i}  [ f_ŷ(S ∪ {xᵢ}) − f_ŷ(S) ]
```

## 4. Evaluation

### 4.1 Quantitative evaluation

- **Fidelity** (Eq. 6): mantém os top-t% tokens (por score de importância) e mede a mudança média na probabilidade predita do token originalmente previsto: `Fid(t) = (1/N) Σ [f_ŷ(xᵢ) − f_ŷ(x̃ᵢ^(t))]`. Palavras removidas são substituídas por token nulo (variante `Fid_rand`: por palavras aleatórias do vocabulário).
- **Probability divergence@K** (Eq. 7, "div@K"): diferença média nas top-K probabilidades preditas entre input completo e mascarado. K=10 (a maioria das sentenças aceita várias continuações sinônimas/coerentes).
- **Accuracy@K** (Eq. 8, "acc@K"): razão média de overlap entre as top-K predições do input completo vs. mascarado.

### 4.2 Qualitative evaluation

- **Coherency**: dado um par de sentenças com pequena variação sintática, se a predição do modelo é igual (ou muda pouco), espera-se explicações semelhantes; se a predição muda bastante (ex.: negação), espera-se explicações diferentes. Medido via similaridade de cosseno entre vetores de rank de atribuição.
- **Semantic alignment**: mede se o token que humanos consideram decisivo (ex.: a negação "not"/"no"/"without") de fato recebe alta importância do método — quando o modelo *ignora* a negação na predição, espera-se (do ponto de vista humano) baixa importância atribuída a esse token pela explicação.

## 5. Experiments

### 5.1 Setup

- **Datasets** (uma sentença por explicação, árvore de dependência via spaCy 3.7.2, relações Universal Dependencies): *Generics KB* (Generics), *ROCStories Winter2017*, *Inconsistent Dataset Negation* (Negation).
- Filtragem: remove sentenças com >15 tokens (custo computacional), sentenças com múltiplos spans, sentenças com pontuação. Ver Tabela 3 (tamanhos antes/depois do filtro) e Tabela 2 (distância de dependência média, tokens/sentença, tokens únicos).
- **Modelos AR**: GPT-2 (154M params) e Mistral 7B (7B params, "muito mais avançado" que GPT-2 segundo os autores).
- **Baselines**: Random, LIME (adaptado para geração de texto), FeatureAblation (leave-one-out, via Captum), SampleShapley (aprox. de SHAP por permutações aleatórias, via Captum), Partition (KernelSHAP hierárquico, versão rápida, baseada no HEDGE / lib `shap`).
- 4 seeds; média + variância reportadas.

### 5.2 Faithfulness

- SyntaxShap e SyntaxShap-W produzem explicações mais fiéis que Random, LIME e Partition (o SOTA shap-based para texto) em ambos os modelos (Fig. 2, Fig. 3).
- Para Mistral 7B, também superam FeatureAblation e SampleShapley.
- Para GPT-2 (Generics/ROCStories), ficam um pouco atrás de FeatureAblation/SampleShapley — atribuído a esses métodos ablacionarem tokens independentemente, "overlooking highly correlated tokens" (Miglani et al., 2023).
- Tabela 1 (div@10, t=0.5, mantendo 50% dos top tokens): SyntaxShap-W é o melhor ou próximo do melhor em quase todos os pares dataset×modelo, com destaque para Mistral 7B/Negation (0.449, o melhor de todos os métodos) e Mistral 7B/Generics (0.638) e ROCStories (0.612).
- **Conclusão da seção (destacada no paper):** "SyntaxShap(-W) generates more faithful explanations than the random baseline, LIME, and Partition. Although it does not beat the baselines FeatureAblation and SampleShapley for every language model, it considers the intrinsic nature of text data, accounting for token correlations and syntactic structure."

### 5.3 Coherency

- Amostra de 267 pares de sentenças (com/sem negação) do dataset Negation; para 90 pares o modelo prediz o mesmo próximo token.
- Fig. 5: SyntaxShap e SyntaxShap-W produzem atribuições mais similares para pares com predição igual, e mais diversas para pares com predição diferente — melhor que LIME, FeatureAblation, SampleShapley, Partition.
- **Conclusão destacada:** "Given a pair of sentences with and without a negation... the similarity of SyntaxShap's token attribution values for each sentence better reflect the degree of similarity of the next token predictions than LIME, FeatureAblation, SampleShapley and Partition."

### 5.4 Semantic alignment

- 22 instâncias rotuladas (Mistral 7B) onde o modelo erra semanticamente a negação (ex.: "A father is not a father").
- Fig. 6: SyntaxShap(-W) identifica a negação como o token *mais importante* em >80% dos casos — mesmo quando o modelo claramente não considerou a negação na sua predição.
- **Achado central do paper:** isso revela um **desalinhamento entre a fidelidade ao modelo e a expectativa semântica humana**. Explicações fiéis ao modelo não necessariamente combinam com a intuição humana — são critérios de avaliação distintos (model-focused vs. human-focused).

## 6. Discussion

- **Estocasticidade**: AR LMs usam top-k sampling; métricas tradicionais (fidelity, AOPC, log-odds) são determinísticas — daí as métricas propostas div@K/acc@K, que consideram as top-K predições.
- **Integração de conhecimento linguístico**: next-token prediction pode ser visto como classificação multi-classe com muitas classes, com tokens tendo papéis linguísticos diferentes (conteúdo vs. função). Futuro: avaliar fidelidade considerando mudança de predição entre categorias de tokens.
- **Considering humans**: LMs aprendem linguagem de forma diferente de humanos; explicações podem legitimamente divergir da intuição humana ("rationalization trap", Sevastjanova & El-Assady, 2022).

## 7. Conclusion

SyntaxShap: método local, model-agnostic, syntax-aware. "Specifically tailored for text data and meant to explain text generation tasks by AR LMs, whose interpretability in that context has not yet been addressed." Primeiro método SHAP-based a incorporar sintaxe via árvore de dependência na construção de coalizões. Mostra explicações mais fiéis e coerentes que baselines model-agnostic para AR LMs (GPT-2, Mistral 7B). O alinhamento semântico mostra que fidelidade e inteligibilidade humana são critérios distintos que nem sempre coincidem.

## Limitations (seção do paper)

- **Multiple sentences**: análise limitada a *uma* sentença de entrada por vez (uma árvore de dependência). Dizem que o método "can be scaled to text with multiple sentences or a paragraph by breaking it down into multiple dependency trees and running SyntaxShap in parallel" — mas isso perderia correlações entre sentenças.
- **Incorrect dependency tree**: o método depende inteiramente da árvore de dependência estar correta; spaCy às vezes produz dependências discutíveis do ponto de vista linguístico, e a acurácia cai fora do inglês. **"SyntaxShap is, for now, only meant to be used for English grammatically non-convoluted sentences to limit the uncertainty coming from the construction of the dependency tree."**

## Ethics Statement

Dados/modelos publicamente disponíveis; reconhece que LMs pré-treinados (GPT-2, Mistral 7B) podem herdar vieses, que podem se refletir também nas explicações geradas.

## Apêndice A — Textual data

- **A.1 Text generation**: tarefas de geração de texto = prever a próxima palavra numa sequência; pode ser enquadrado como seq2seq (tradução, QA). AR models (GPT) geram uma palavra de cada vez condicionando nas anteriores. O paper foca em next-token generation dado uma única sentença de entrada.
- **A.2 Dependency parsing**: técnica de NLP que identifica relações gramaticais entre palavras como uma árvore (head/dependent, rótulos via Universal Dependency Relations). Usam spaCy 3.7.2. Número de tokens varia de 5 a 15 nos datasets Generics/ROCStories.

## Apêndice B — SyntaxShap: characteristics and proofs

### B.1 SyntaxShap and the Shapley axioms

Os 4 axiomas clássicos de Shapley (efficiency, additivity, nullity, symmetry) não garantem que o valor computado seja o ideal para feature selection (Fryer et al., 2021); dois dos quatro **não** são satisfeitos por valores restritos à árvore:

- **Efficiency — NÃO satisfeito.** `v(S) = f_ŷ(S)` onde ŷ = argmax(f(x)). Por não-linearidade dos LMs, essa função de avaliação é não-monotônica (não necessariamente aumenta ao adicionar mais features). **Consequência prática: os valores SyntaxShap de cada palavra NÃO somam ao valor da sentença inteira** — são importâncias relativas, não uma decomposição aditiva exata.
- **Symmetry — satisfeito, mas só dentro do mesmo nível da árvore** (Eq. 9): duas features no mesmo nível que desempenham papéis equivalentes têm o mesmo valor SyntaxShap.
- **Nullity — satisfeito** (Eq. 10): se a feature não contribui em nenhuma coalizão, seu valor é zero.
- **Additivity — satisfeito** (Eq. 11): para dois modelos f e g, φᵢ(f+g) = φᵢ(f) + φᵢ(g).

### B.2 Computational complexity

- Shapley clássico: O(n·2ⁿ) (2^(n-1) coalizões por feature × n features).
- SyntaxShap: para uma árvore balanceada com L níveis, complexidade aproximada **O(n·L·2^(n/L))** — mais rápido que o Shapley exato porque a árvore particiona o espaço de coalizões.
- Fig. 10: robustez confirmada para sentenças de 5–15 tokens.

## Apêndice C — Data preprocessing

- Subtokenização: quando um tokenizer quebra uma palavra em múltiplos tokens, SyntaxShap duplica o nó-palavra na árvore de dependência, tratando cada subtoken com o mesmo papel/nível do nó original.
- Filtros aplicados: sentenças >15 tokens, sentenças com múltiplos spans, sentenças com pontuação (`!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`) são removidas. Tabela 3: tamanhos antes/depois do filtro por dataset (Negation 534→534, Generics 5777→3568/3328, ROCStories 2275→1592/1543 conforme filtro por GPT-2 ou Mistral).

## Apêndice D — Additional results

- **D.1 Masking strategies**: comparam duas estratégias de mascaramento — (i) zerar attention weight do token, (ii) substituir por token aleatório do vocabulário. Resultado: **conclusões relativas entre métodos são as mesmas** nas duas estratégias (Fig. 8, Fig. 9); para GPT-2, tokens aleatórios geram explicações ligeiramente menos fiéis que zerar atenção.
- **D.2 Number of tokens and faithfulness**: performance de todos os métodos é robusta ao variar 5–15 tokens (Fig. 10) — SyntaxShap "can be applied to a diverse range of sentence lengths" (dentro dessa faixa testada).
- **D.3 Dependency distance and faithfulness**: fidelidade é ~constante através das faixas de distância de dependência para Generics; leve queda para ROCStories em sentenças mais sintaticamente complexas — mas com poucas instâncias nos extremos (ex.: só 3 sentenças com d<0.5, 7 com d>3 — Tabela 4), então os autores pedem cautela na conclusão.

---

## Notas de aplicação ao projeto xai-in-pi-arena

Pontos relevantes para a decisão de usar SyntaxShap como XAI engine do PI Arena, especialmente mirando o `promptguard` (classificador, não gerador):

1. **A formulação matemática (Seção 3.1–3.3, Eq. 3–5) é genuinamente agnóstica ao tipo de saída.** `f_ŷ(S)` é só "probabilidade da classe-alvo ŷ dado o subconjunto mascarado S". Não há nada aí que exija geração autoregressiva — o requisito é só ter uma classe/alvo ŷ fixo e uma função que dê P(ŷ | S). Isso é estruturalmente idêntico a "probabilidade da classe 'injection' dada pelo PromptGuard para o texto mascarado S".
2. **O que é específico de geração é a instanciação do paper (título, abstract, conclusão) e a implementação vendorizada** (`explain_row` em [_syntax.py](../syntax-shap-main/syntaxshap/explainers/_syntax.py) usa `.generate()` para obter ŷ automaticamente e usa `captum.LLMAttribution` para computar `f_ŷ(S)` via teacher forcing) — não uma exigência do jogo de coalizão em si. Portanto adaptar para um classificador = trocar como `f_ŷ(S)` é obtido (forward pass direto do classificador em vez de log-prob de sequência via LLMAttribution) e eliminar a etapa de auto-derivar `ŷ` via `.generate()` (o classificador já dá a classe/prob diretamente) — mantendo intacta toda a lógica de árvore de dependência e coalizões por nível.
3. **Limitação declarada pelos autores — importante para o PI Arena**: método testado só com **uma sentença por vez**, ≤15 tokens, inglês, "grammatically non-convoluted". O `context` do PromptGuard no PI Arena costuma ser um parágrafo (múltiplas sentenças, muito mais que 15 tokens) — os próprios autores dizem que dá pra escalar quebrando em múltiplas árvores (uma por sentença) e rodando em paralelo, mas isso "loses sentence correlations", e não foi testado no paper. É um risco de validade a documentar no experimento, não um bloqueio.
4. **Efficiency axiom não vale**: os valores por palavra não somam ao valor da sentença — ao reportar/visualizar saliency maps, tratar como *importância relativa/ranking*, não como decomposição aditiva exata do score do PromptGuard.
5. **Custo computacional**: O(n·L·2^(n/L)) por sentença explicada — cresce rápido com o tamanho da sentença; para contextos longos do PI Arena, ou quebra por sentença (ver ponto 3) ou espera-se custo alto por amostra.
