---
title: Kernel SHAP — mudanças para funcionar sobre um encoder classificador (PromptGuard)
---

# Kernel SHAP: o que mudou para explicar um encoder classificador (PromptGuard)

> Baseado na análise de `piarena/xai/kernelshap/` (`xai_kernelshap.py`, `classifier_explainer.py`). Na época em que este arquivo foi escrito, esse código vivia num checkout paralelo (`piarena_xai_kernel_shap/PIArena-main/`), mantido separado de `PIArena-main/` só pra evitar que o desenvolvimento dos dois métodos XAI pisasse um no outro; os dois foram desde então mesclados de volta numa única árvore `PIArena-main/`, com `syntaxshap` e `kernelshap` coexistindo no mesmo `XAI_REGISTRY` (ver o `CHANGELOG.md` da árvore mesclada, seção "Changed"). Complementa [[kernel-shap-implementacoes]] (comparação `captum` vs. `shap` a nível de biblioteca) e `papers-de-referencia/kernel-shap.md` (base teórica, Lundberg & Lee 2017). Ver também [[custo-computacional-syntaxshap]] para o contraste com a experiência paralela do SyntaxSHAP, que precisou de uma adaptação muito mais pesada para o mesmo alvo.

## 1. O que o Kernel SHAP "de fábrica" pressupõe — e por que isso não serve direto

Kernel SHAP (Lundberg & Lee, 2017) é model-agnostic por definição: precisa só de uma função `f(x) -> score`. As duas implementações usadas aqui (`captum.attr.KernelShap` e `shap.KernelExplainer`) nascem para esse contrato genérico, tipicamente tabular ou de visão:

- `captum` documenta seu exemplo canônico de `feature_mask` agrupando um **bloco 2×2 de pixels** de uma imagem 4×4 — não há noção nativa de "palavra" ou "token".
- `shap.KernelExplainer` espera `X`/`data` como matriz `(#amostras, #features)` — o "background dataset" do paper (Eq. 10/11) é pensado para imputar valores plausíveis de linhas tabulares reais (ex.: a linha média, ou k-means do dataset).
- Nenhuma das duas sabe o que é um tokenizer, o que é um `[MASK]`, ou como uma string se transforma em `input_ids`.

Para explicar `PromptGuard` (`meta-llama/Prompt-Guard-86M`, um `AutoModelForSequenceClassification` baseado em mDeBERTa) sobre texto, cada uma dessas peças teve que ser construída — é isso que `ClassifierKernelExplainer` faz. Ao contrário do SyntaxSHAP (que teve que reimplementar a própria lógica de coalizão/Shapley porque o motor original estava hard-wired para `.generate()` autoregressivo via captum + teacher forcing), aqui a lógica matemática do Kernel SHAP em si **não precisou ser tocada** — as duas bibliotecas já resolvem a regressão ponderada corretamente para qualquer `f`. A adaptação inteira é de **engenharia de ponte**: texto → tensores de token, tensores → coalizões mascaradas, coalizões → um único score escalar.

## 2. Mudança 1 — de "modelo genérico" para "classificador de sequência carregado direto"

`_load_classifier()` (`xai_kernelshap.py:11-24`) carrega `AutoModelForSequenceClassification` + `AutoTokenizer` diretamente, **não** via o wrapper `pipeline()` de texto que `PromptGuardDefense` usa internamente. Motivo: Kernel SHAP via `captum` precisa de um `forward_func` que aceite e devolva **tensores**, sob `@torch.no_grad()`. Um `pipeline()` de classificação de texto só aceita string e devolve dict — cada coalizão perturbada teria que ser re-serializada para string e re-tokenizada, reintroduzindo exatamente o risco de artefato de espaçamento/pontuação que a nota de proveniência do arquivo de comparação de bibliotecas identifica como o preço de usar `shap.KernelExplainer` sem essa ponte.

## 3. Mudança 2 — a "feature" interpretável é a palavra, não o token, via `.word_ids()`

Um encoder moderno fragmenta palavras raras (URLs, scripts não-latinos) em várias subpalavras. Se cada subtoken fosse sua própria feature, a fragmentação inflaria artificialmente o espaço de coalizões e distorceria a importância relativa — exatamente o "Finding 2" documentado no `CLAUDE.md` do workspace para o SyntaxSHAP (onde a árvore de dependência duplicava o nó-palavra em cada subtoken, achatando tudo no mesmo nível).

A solução aqui (`_tokenize_content()`, `classifier_explainer.py:200-248`) usa `tokenizer.word_ids()` — recurso nativo de um *fast tokenizer* — que já devolve `None` para tokens especiais (`[CLS]`/`[SEP]`/`[PAD]`) e um índice de palavra 0-based estável para cada subtoken real. Isso resolve **de graça**, com uma chamada de biblioteca, duas coisas que o SyntaxSHAP teve que corrigir manualmente no código vendorizado (`_dependency_tree.py`): (a) detectar tokens especiais de qualquer convenção (`<s>`/`</s>` vs. `[CLS]`/`[SEP]`), e (b) impedir que eles vazem como linhas duplicadas na árvore. O comentário no topo de `classifier_explainer.py` chama isso explicitamente de "o equivalente moderno, fornecido pela biblioteca, do fix manual de detecção de CLS/SEP que o experimento SyntaxSHAP teve que fazer à mão".

No lado `captum`, esse agrupamento vira um `feature_mask` — um tensor do mesmo shape do input onde posições com o mesmo inteiro são perturbadas *juntas*; é o próprio mecanismo nativo da API (não uma reimplementação), só reaproveitado com granularidade de palavra em vez de bloco de pixel.

Para reconstruir o texto de cada palavra sem artefatos de detokenização, `_word_strings()` (linhas 250-262) não decodifica os ids de volta — faz slice do texto original usando o offset de caractere mínimo/máximo dos subtokens daquela palavra (via `return_offsets_mapping=True` do tokenizer).

## 4. Mudança 3 — mascaramento em espaço de token-id, nunca em string

`_score_with_mask_batch()` (linhas 290-308) opera inteiramente em tensores: recebe uma máscara booleana por palavra, expande para máscara por token via `feature_mask`, e troca (`torch.where`) os ids das palavras "ausentes" por um **token de baseline fixo**. Uma coalizão nunca é re-serializada para string e re-tokenizada — o mesmo ganho de engenharia que o comentário do módulo cita como diferencial de operar em "token-id tensors, not text".

### 4.1 Escolha do token de baseline (mask/pad/unk com fallback)

`_resolve_baseline_token_id()` (linhas 171-191) tenta `mask_token_id` primeiro, cai para `pad_token_id`, depois `unk_token_id` — e levanta erro só se o tokenizer não definir nenhum dos três. Isso importa porque nem todo tokenizer de classificador define `mask_token` (é um artefato do pré-treino MLM; um tokenizer treinado só para classificação pode não ter herdado esse token) — a cadeia de fallback evita que a adaptação quebre silenciosamente ou trave num tokenizer que a implementação genérica nunca tinha motivo para prever.

### 4.2 Uma única linha de "background" ao invés de um dataset de background real

O paper original do Kernel SHAP (Eq. 11) pensa o baseline como uma distribuição de fundo real, de onde se **amostra** valores plausíveis para imputar features "ausentes" — natural em dados tabulares (ex.: linha média do dataset). Para texto, não existe "token médio" sensato. A adaptação aqui converge as duas bibliotecas para a mesma semântica prática — um **baseline fixo** tipo `[MASK]`:

- No backend `captum`, isso é literal: `baselines=torch.full_like(content_ids, baseline_token_id)` (linha 345).
- No backend `shap`, que é pensado em torno de um "background dataset" de verdade, a adaptação degenera esse dataset para **uma única linha totalmente mascarada** (`background = np.zeros((1, M_words), dtype=bool)`, linha 387) — o padrão documentado nos próprios tutoriais de NLP do `shap`, mas que é uma simplificação deliberada da semântica "cheia" do paper, registrada explicitamente no código e em [[kernel-shap-implementacoes]] §4 como a forma de manter os dois backends comparáveis entre si.

## 5. Mudança 4 — o alvo explicado é o agregado P(malign), não o argmax de 3 classes

`_compute_malign_indices()` (linhas 153-169) não hardcoda nomes de classe (`"INJECTION"`/`"JAILBREAK"`) — filtra `id2label` por qualquer label cujo nome **não** contenha a substring `"benign"`, e soma os `softmax` dessas classes. Essa é a mesma escolha de design documentada para o SyntaxSHAP (explicar `P(non-benign) = P(INJECTION) + P(JAILBREAK)`), replicada aqui com o mesmo motivo: bater exatamente com o que `PromptGuardDefense.execute`'s `detect_flag` de fato checa, e ficar robusto caso o conjunto de labels real do modelo gated não seja exatamente o assumido (o comentário chama isso de "Risco #2 do plano").

`_forward_func()` (linhas 266-288) já devolve esse escalar `P(malign)` diretamente — nem `captum.attr.KernelShap.attribute()` precisa de um índice `target=` (que serviria pra selecionar uma classe entre várias), nem o `f(X)` do backend `shap` precisa lidar com múltiplas colunas de saída. Isso colapsa o problema multi-classe do classificador num único jogo de coalizão escalar, sem precisar da API `link="logit"`/`target=` que cada biblioteca oferece pra saída multi-classe genérica.

## 6. Mudança 5 — orçamento de amostragem nunca deixado no default da biblioteca

`captum.attr.KernelShap.attribute(n_samples=...)` tem default **fixo em 25**, independente de quantas features existem — perigoso para um `context` com dezenas ou centenas de palavras (M grande), porque nada trava (ao contrário da explosão exponencial do SyntaxSHAP), mas a explicação sai severamente subamostrada, com alta variância, sem nenhum erro visível.

`_resolve_n_samples()` (linhas 193-196) nunca deixa isso acontecer: com `n_samples="auto"` (o default de `DEFAULT_CONFIG`), resolve por chamada para `min(2*M_words + 2048, max_n_samples)` — a mesma heurística que `shap.KernelExplainer.shap_values(nsamples="auto")` já usa nativamente, replicada manualmente do lado `captum` porque ali não existe. Isso também dá **paridade de orçamento** entre os dois backends: mesmo `M`, mesmo `n_samples` resolvido, então os valores dos dois backends ficam comparáveis entre si (o objetivo citado no docstring de `n_samples` e em [[kernel-shap-implementacoes]] §7).

## 7. Mudança 6 — truncamento guardado contra o sentinela absurdo de `model_max_length`

Muitos tokenizers HF que nunca tiveram um `max_length` explícito configurado retornam um sentinela do tipo `1_000_000_000_000_000_019_884_624_838_656` em `model_max_length` em vez de um valor usável — um footgun conhecido do HF, não específico deste código. O construtor guardava contra isso: se `max_length` não fosse passado e `model_max_length` estourasse `_SENTINEL_MAX_LENGTH_THRESHOLD` (100.000), caía para `_FALLBACK_MAX_LENGTH = 512` em vez de tentar truncar para um número absurdo.

### 7.1 Correção — o fallback do guard virou um truncamento silencioso por padrão, divergindo do que a defesa realmente via

O guard contra o sentinela em si estava certo em existir — o problema era o que ele fazia quando `max_length` **não era passado**. `max_length or getattr(tokenizer, "model_max_length", None)` também caía no ramo do sentinela sempre que `max_length` era `None` (o valor de `DEFAULT_CONFIG`, ou seja, o caso comum de "ninguém pediu truncamento nenhum"), e nesse caso `model_max_length` de um tokenizer real (não o sentinela absurdo) — ~512 tokens para o tokenizer do `Prompt-Guard-86M` — virava o comprimento truncado usado em **toda** chamada, por padrão, silenciosamente.

Isso divergia do que `PromptGuardDefense.execute` (`defense_promptguard.py`) de fato faz: a chamada ao `pipeline()` da defesa não passa nenhum `truncation`/`max_length` — ela pontua o `context` inteiro, sem truncar. Para contextos longos onde o ataque insere a `injected_task` depois desse corte de ~512 tokens (comum em `dolly_closed_qa`/contextos estilo Wikipedia), a defesa via a injeção no texto completo e bloqueava corretamente (`detect_flag=True`), mas o Kernel SHAP explicava só o prefixo truncado — que nunca continha a injeção. Resultado: `predicted_label`/`p_malign` enganosos e `injected_span: null` para uma amostra que, na verdade, tinha sido corretamente bloqueada por causa de uma injeção que a explicação nunca chegava a mostrar.

Confirmado contra dados reais salvos, não só por leitura de código: em `kernelshap_dolly_closed_qa_captum/dolly_closed_qa-...-ignore-promptguard-kernelshap-42.json`, a amostra de chave `"1"` — contexto de 9047 caracteres, injeção no offset de caractere 5353 — tem a lista de palavras do próprio módulo de XAI parando na word 333 ("...large language models but"), centenas de palavras antes da injeção; `predicted_label` sai `"BENIGN"`, `p_malign=0.336`, `injected_span=null`. A amostra `"190"` do arquivo do ataque `direct` mostra o mesmo padrão.

Correção: `max_length=None` (o padrão) agora significa **sem truncamento nenhum** — `_tokenize_content` tokeniza o texto inteiro, batendo exatamente com o que `PromptGuardDefense.execute` pontua. Um `max_length` explícito continua disponível como opt-in para quem quiser limitar custo, mas agora dispara um `log.warning` avisando que a explicação resultante pode não cobrir o que a defesa realmente viu. Ver `plans/xai-kernelshap-truncation-fix.md` para o detalhamento completo da correção.

## 8. Mudança 7 — batching nativo em ambos os backends, ausente no SyntaxSHAP

`ClassifierSyntaxExplainer._score` (experimento paralelo) chama o modelo um texto por vez, sem batch. Aqui, `perturbations_per_eval` é reaproveitado como o parâmetro nativo do `captum` para lotes de coalizões perturbadas, e o mesmo valor é usado manualmente para dividir em chunks a matriz sintética que o backend `shap` entrega de uma vez em `f(X)` (`_compute_values_shap`, linhas 357-397) — o `shap.KernelExplainer` não faz batching por conta própria, então essa responsabilidade foi implementada aqui (`_score_with_mask_batch`, linhas 290-308, reaproveitado por ambos os caminhos).

## 9. Mudança 8 — dois backends coexistindo por trás do mesmo contrato

Diferente do SyntaxSHAP (uma implementação única, vendorizada e corrigida), o Kernel SHAP aqui foi construído com **dois motores de cálculo intercambiáveis** (`backend: "captum" | "shap"`, `xai_kernelshap.py:41-46`) atrás da mesma tokenização, mascaramento e formato de saída — trocar é mudança de config (`xai_config.yaml`), nunca de código. Isso existe porque, apesar de resolverem a mesma fórmula teórica (Shapley kernel, Teorema 2 do paper), as duas bibliotecas diferem em detalhes de engenharia que podem afetar o resultado prático em texto — documentado em [[kernel-shap-implementacoes]]:
- `captum` amostra coalizões 100% estocasticamente (confirmado no código-fonte); `shap` aparentemente combina enumeração exaustiva de cardinalidades extremas com amostragem do resto (não confirmado no código-fonte, só via resumo — ver nota de validação pendente nesse arquivo).
- `captum` aproxima os pesos infinitos dos extremos (coalizão vazia/completa) com um valor grande porém finito (`1e6`); `shap` trata os extremos de forma mais próxima do exato (via resumo).
- `shap` oferece `link="logit"` nativo para trabalhar em espaço log-odds (útil quando a saída satura perto de 0/1, esperado num guard classifier bem calibrado); `captum` não tem função de link — exigiria transformar manualmente dentro do `forward_func`.

`xai_result["context"]["backend"]` registra qual dos dois produziu um resultado dado, exatamente para permitir essa comparação no pilot (ver `experimento-kernel-shap/v1-captum-squad_v2-25amostras.md`, que roda com `backend=captum`).

## 10. O que NÃO precisou ser reimplementado (contraste direto com o SyntaxSHAP)

Vale registrar o que a adaptação para encoder **evitou** ter que fazer, porque contrasta com o custo de adaptação do SyntaxSHAP para o mesmo alvo:

- Nenhuma árvore de dependência sintática, nenhum spaCy, nenhuma orquestração por sentença — Kernel SHAP explica o `context` inteiro como **um único jogo de coalizão conjunto** (`explain_context` → uma chamada ao motor, cobrindo toda palavra de uma vez). O custo é o orçamento escolhido (`n_samples`), não uma propriedade estrutural do texto — não existe o problema de "nível mais largo da árvore" que domina o custo do SyntaxSHAP.
- Nenhuma lógica de coalizão/Shapley própria foi escrita — tanto `captum.attr.KernelShap` quanto `shap.KernelExplainer` já resolvem a regressão corretamente para qualquer `f`; não havia bug de complexidade (`O(2^M)`) a evitar, porque nenhuma das duas bibliotecas enumera coalizões exaustivamente por padrão.
- Nenhum vendoring de código terceiro foi necessário (ao contrário de `piarena/xai/syntaxshap/thirdparty/`).

Isso é consistente com o que o design de ambos os experimentos já registra: a explosão combinatória do SyntaxSHAP é uma propriedade do algoritmo específico (custo dominado por `2^largura`), não do problema "explicar um encoder classificador" em si — o Kernel SHAP ataca o mesmo problema com uma estratégia de custo fundamentalmente diferente (amostragem de orçamento fixo), e por isso a adaptação para PromptGuard foi inteiramente sobre a ponte texto↔tensor, nunca sobre a matemática de atribuição.

## Resumo

| Peça | Genérico (captum/shap "de fábrica") | Adaptado para PromptGuard (encoder classifier) |
|---|---|---|
| Modelo | função/tensor arbitrário | `AutoModelForSequenceClassification` carregado direto, sem `pipeline()` |
| Feature | pixel/coluna tabular arbitrária | palavra, via `tokenizer.word_ids()` — subtokens colapsam automaticamente |
| Máscara | baseline/background genérico | `[MASK]`→`[PAD]`→`[UNK]` com fallback; token id, nunca string |
| Alvo | score de saída arbitrário/multi-classe via `target`/link | escalar `P(malign)` agregado, calculado dentro do `forward_func` |
| Orçamento | `n_samples=25` fixo (captum) | `min(2M+2048, max_n_samples)`, nunca deixado no default |
| Truncamento | `tokenizer.model_max_length` direto | sem truncamento por padrão (bate com a defesa); opt-in via `max_length` explícito, com warning e guard contra o sentinela absurdo do HF (ver §7.1) |
| Batching | manual (shap) / nativo mas não usado por padrão | usado explicitamente nos dois backends |
| Backend | um só, por biblioteca | dois coexistindo atrás do mesmo contrato (`backend=`) |
