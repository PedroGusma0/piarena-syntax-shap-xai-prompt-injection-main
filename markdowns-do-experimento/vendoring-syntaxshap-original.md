# Do SyntaxSHAP original ao adaptador do PromptGuard — o que mudou

Relatório de referência sobre tudo que foi copiado, cortado, corrigido ou reimplementado ao levar `syntax-shap-main-oficial/` (código do paper SyntaxSHAP, arXiv:2402.09259, vendorizado sem alterações neste workspace) para dentro do PIArena como o plugin `--xai syntaxshap`. Ver também `papers-de-referencia/syntaxshap.md` (transcrição do paper) e `PIArena-main/plans/xai-syntaxshap-promptguard.md` (design completo, com todas as seções "Correções de arquitetura" / "Arquitetura da mudança" / "Riscos" citadas abaixo).

Comparação feita arquivo a arquivo com `diff -u` entre `syntax-shap-main-oficial/syntaxshap/` e `PIArena-main/piarena/xai/syntaxshap/thirdparty/` (mais os módulos novos que não têm equivalente no original).

## Resumo executivo

- O código original é hard-wired para **geração autoregressiva** (`.generate()` + `captum.attr.LLMAttribution` com teacher forcing). O PromptGuard é um **classificador de um único forward pass**. Isso obrigou a reescrever a camada de explicação, não só trocar o modelo.
- Só uma fração pequena do repo original foi vendorizada — **maskers de texto, wrapper de pipeline HuggingFace e utilitários de árvore de dependência**. Todo o resto (`explainers/`, geração de texto, dados tabulares, clustering hierárquico, scripts de CLI do próprio paper) foi descartado por não se aplicar a um classificador.
- O núcleo do algoritmo (`explainers/_syntax.py`, a lógica de coalizões + Shapley) **não foi importado nem herdado — foi reimplementado do zero** em `coalition.py`, porque (a) está amarrado ao captum/`.generate()`, e (b) tem um bug real de escalonamento: `feature_exact()` sempre constrói `range(2**M)` primeiro e só filtra depois, o que trava em qualquer M grande (um `context` inteiro do squad_v2 tem M ~100–180 tokens; `range(2**100)` nunca termina).
- Dos dois únicos arquivos vendorizados que precisaram de alteração de conteúdo (não só corte), `utils/_dependency_tree.py` recebeu **2 correções de bug reais**, encontradas testando contra um tokenizer de verdade (não achadas por leitura estática): tokens especiais estilo `[CLS]`/`[SEP]` (DeBERTa-v3, usado pelo Prompt-Guard-86M) não eram reconhecidos como especiais — só o estilo BPE `<s>`/`</s>` era — e, mesmo corrigindo isso, ainda vazavam pra árvore de dependência como linhas duplicadas da primeira/última palavra real.
- O alvo explicado mudou de "próximo token gerado" para um **agregado binário `P(não-benigno) = P(INJECTION) + P(JAILBREAK)`**, calculado por substring-match no rótulo (`"benign" not in label.lower()`) — a mesma checagem que `PromptGuardDefense.execute` já usa — em vez de nomes de classe fixos.
- O escopo do que é explicado também mudou: o paper explica frases isoladas curtas; aqui explica-se o **`context` inteiro** (o que o PromptGuard de fato recebe), via orquestração por sentença.
- Nenhuma dependência de `captum` sobrou. Em compensação, `pandas`, `shap` (biblioteca oficial de plot) e `matplotlib` foram adicionados ao `requirements.txt` do PIArena — usados só pela etapa de métricas/plot, não pelo cálculo dos valores SyntaxSHAP em si.

## Onde isso vive no código

```
PIArena-main/piarena/xai/
├── base.py                          # novo — BaseXAI, terceiro tipo de plugin (ver PIArena-main/CLAUDE.md)
├── __init__.py                      # novo — get_xai(), dispara @register_xai
├── metrics.py                       # novo — sem equivalente no paper (ver markdowns-do-experimento/metricas-xai-syntaxshap.md)
└── syntaxshap/
    ├── __init__.py                  # novo
    ├── _thirdparty_path.py          # novo — bootstrap de sys.path pro vendoring
    ├── coalition.py                 # novo — reimplementação de explainers/_syntax.py
    ├── classifier_explainer.py      # novo — reimplementação de SyntaxExplainer p/ classificador
    ├── xai_syntaxshap.py            # novo — plugin registrado (`--xai syntaxshap`)
    └── thirdparty/                  # cópia reduzida e parcialmente corrigida do original
        ├── _serializable.py         # idêntico
        ├── maskers/{_masker,_text}.py   # idênticos
        ├── models/{_model,_transformers_pipeline}.py  # idênticos
        └── utils/{_general,transformers}.py  # idênticos
        └── utils/_dependency_tree.py # PATCHED — 2 bugs corrigidos + dependência textdescriptives removida
```

## 1. O que foi vendorizado de `syntax-shap-main-oficial/syntaxshap/`

Só um subconjunto de arquivos foi copiado para `thirdparty/`, e sempre sem a subpasta `explainers/`.

| Original | Status no experimento | Motivo |
|---|---|---|
| `maskers/_masker.py` | ✅ mantido, byte-a-byte idêntico | Classe base do masker, agnóstica a geração vs. classificação |
| `maskers/_text.py` | ✅ mantido, byte-a-byte idêntico | Máscara de texto por token — reaproveitável como está |
| `maskers/_composite.py`, `_fixed.py`, `_fixed_composite.py`, `_output_composite.py`, `_tabular.py` | ❌ não copiados | Composição multi-masker e dados tabulares — sem uso aqui (só há um masker de texto) |
| `models/_model.py` | ✅ mantido, idêntico | Classe base de wrapper de modelo |
| `models/_transformers_pipeline.py` | ✅ mantido, idêntico | Wrapper de `pipeline()` do HuggingFace — serve tanto pra geração quanto pra classificação, sem mudança |
| `models/_teacher_forcing.py`, `_text_generation.py`, `_topk_lm.py` | ❌ não copiados | Específicos de geração autoregressiva (teacher forcing via captum, top-k de LM) |
| `utils/_dependency_tree.py` | ⚠️ mantido, **patched** | Ver seção 2 abaixo — 2 bugs corrigidos + dependência pesada removida |
| `utils/_general.py` | ✅ mantido, idêntico (conteúdo) — só `__init__.py` reexporta menos símbolos | `safe_isinstance`, `record_import_error`, `assert_import` |
| `utils/transformers.py` | ✅ mantido, idêntico | `parse_prefix_suffix_for_tokenizer` — necessário pra saber quantos tokens de prefixo/sufixo especiais o tokenizer do PromptGuard adiciona |
| `utils/_clustering.py`, `_masked_model.py`, `_parser.py`, `_show_progress.py`, `_legacy.py`, `_filter_data.py`, `_exceptions.py` | ❌ não copiados | Clustering hierárquico p/ dados tabulares, `MaskedModel` amarrado a geração, parsing de CLI do experimento do paper, etc. — nada disso se aplica |
| `_serializable.py` | ✅ mantido, idêntico | `Masker`/`Model` herdam de `Serializable` |
| `links.py` | ❌ não copiado | Não referenciado por nada do que sobrou (nem `coalition.py` nem `classifier_explainer.py` usam `links.identity`) |
| `explainers/` (inteiro: `_explainer.py`, `_syntax.py`, `_hedge.py`, `_partition.py`, `other/*`) | ❌ não copiado | Núcleo do algoritmo — **reimplementado** em `coalition.py`/`classifier_explainer.py`, não herdado (ver seção 3) |
| `datasets.py`, `embeddings.py`, `explain.py`, `main.py`, `model.py`, `save_predictions.py`, `save_statistics.py`, `svsampling.py`, `metrics.py` (nível raiz) | ❌ não copiados | Scripts do próprio experimento do paper (loaders IMDB/DynaHate, CLI, métricas do paper) — substituídos por `piarena/xai/metrics.py` + `scripts/xai_metrics.py`, desenhados especificamente pra saída do PIArena |

## 2. Patches reais nos arquivos vendorizados

O único arquivo vendorizado com mudança de **conteúdo** (além de cortes de import/`__all__`) é `utils/_dependency_tree.py`.

### 2.1 Dependência pesada removida

```diff
 import pandas as pd
 import re
 import spacy
-import textdescriptives as td
+
+# NOTE (PIArena vendoring): dropped `import textdescriptives as td` and the
+# `compute_dependency_distance` function that used it — neither is called by
+# `get_token_dependency_tree`, and textdescriptives pulls in a heavy dependency
+# chain (benepar, pyphen, ftfy, ...) we don't need.
```

A função `compute_dependency_distance` (que usava `textdescriptives`) também foi removida por inteiro — nada no experimento a chama; ela calculava uma métrica agregada de "distância de dependência" do paper original que não faz parte do pipeline do PIArena.

### 2.2 Bug real #1 — tokens especiais estilo `[CLS]`/`[SEP]` não eram reconhecidos

O código original só reconhecia tokens especiais no formato BPE (`<s>`, `</s>`, do GPT-2/Mistral usados no paper):

```diff
-            if not (decoded_word.startswith('<') and decoded_word.endswith('>')):
+            is_special_token = (
+                (decoded_word.startswith('<') and decoded_word.endswith('>'))
+                or (decoded_word.startswith('[') and decoded_word.endswith(']'))
+            )
+            if is_special_token:
+                pos_token_to_word[k] = -1
+            else:
                 word_len += len(decoded_word)
-            pos_token_to_word[k] = i
+                pos_token_to_word[k] = i
             k += 1
```

O tokenizer do Prompt-Guard-86M é DeBERTa-v3, que usa `[CLS]`/`[SEP]` — colchetes, não `< >`. Sem esse ramo, `[CLS]`/`[SEP]` eram contados como se fossem caracteres de palavra normais e o mapeamento posição-de-token → posição-de-palavra saía desalinhado a partir daí (risco listado no plano como "Risco #1", validado empiricamente numa venv isolada com tokenizer real).

### 2.3 Bug real #2 — mesmo depois do fix acima, o token especial ainda vazava pra árvore

Achado testando o fix #1 contra um tokenizer real: mesmo depois de reconhecer `[CLS]`/`[SEP]` como especiais, o código original ainda mapeava esse token pra `pos_token_to_word[k] = i` (o índice de palavra em que o loop externo estava no momento) — o que faz esse token se fundir espuriamente na linha da árvore de dependência dessa palavra, porque o join de `get_token_dependency_tree` usa só `word_position`, não o próprio token. Na prática, `[CLS]` aparecia como uma linha **duplicada** da primeira palavra real da frase.

A correção usa `-1` como sentinela — não bate com nenhuma linha real de `df_words`, então o inner join de `get_token_dependency_tree` descarta o token especial naturalmente, em vez de fundi-lo em outra palavra.

### 2.4 `compute_dependency_distance` removida por inteiro

Função inteira (usava `textdescriptives`, ver 2.1) apagada — não é chamada por `get_token_dependency_tree`, que é a única função deste arquivo que o resto do experimento usa.

### 2.5 Cortes nos `__init__.py` (reexportam só o que sobrou)

`maskers/__init__.py`, `models/__init__.py` e `utils/__init__.py` foram trimados pra reexportar só os símbolos dos arquivos que de fato foram vendorizados (`Masker`/`Text`; `Model`/`TransformersPipeline`; `assert_import`/`record_import_error`/`safe_isinstance`/`get_token_dependency_tree`/`create_dataframe_from_tree`/`spacy_doc_to_tree`) — sem mudança de comportamento nos símbolos que restaram, só remoção dos que não existem mais no diretório.

## 3. O núcleo do algoritmo — reimplementado, não herdado

`explainers/_syntax.py` (a classe `SyntaxExplainer` do paper, 371 linhas) **não foi vendorizado**. Em vez disso, a lógica pura de coalizões/Shapley foi reescrita em `piarena/xai/syntaxshap/coalition.py` (209 linhas). Motivo documentado na seção "Correções de arquitetura" do plano:

1. **Amarração a geração autoregressiva.** `SyntaxExplainer.get_contribution`/`_format_model_input` chamam `captum.attr.LLMAttribution` (`ShapleyValueSampling` + `_run_forward`) sobre `self.model_init.generate(...)`. Não há forma de reaproveitar isso pra um classificador de forward único sem reescrever a função de scoring inteira — nesse ponto, herdar a classe não economiza nada.
2. **Bug de escalonamento real em `feature_exact()`.** O original:

   ```python
   def feature_exact(M, asymmetric=False, causal_ordering=None):
       dt = pd.DataFrame({'id_combination': range(2**M)})
       list_combinations = [list(x) for x in chain(*[combinations(range(M), i) for i in range(M+1)])]
       ...
       if asymmetric:
           dt = dt[dt['features'].apply(lambda x: respects_order(x, causal_ordering))]
   ```

   constrói o **powerset completo de `range(M)`** incondicionalmente — mesmo no caminho restrito pela árvore sintática (`asymmetric=True`, os modos `"syntax"`/`"syntax-w"` que o paper de fato recomenda) — e só *depois* filtra pra o subconjunto que respeita a ordem causal. Para o M~15–20 tokens testado no paper isso é lento mas tolerável; para M~100–180 tokens (um `context` inteiro do squad_v2) é **literalmente impossível** — `range(2**100)` não termina.

   A correção em `coalition.py::build_allowed_coalitions` constrói o conjunto restrito **diretamente**, nível por nível da árvore de dependência (𝔖 = ⋃ₗ 𝔖ₗ, Eq. 4 do paper), sem nunca passar por `2^M`:

   ```python
   def build_allowed_coalitions(causal_ordering):
       coalitions = [[]]
       prefix = []
       for level_positions in causal_ordering:
           for r in range(1, len(level_positions) + 1):
               for combo in combinations(level_positions, r):
                   coalitions.append(prefix + list(combo))
           prefix = prefix + level_positions
       return coalitions
   ```

   Custo: Σₗ 2^(nₗ) (soma por nível da árvore), nunca 2^M inteiro. Testado contra um oráculo por força-bruta (`respects_order`, mantida como referência) em `M=60`: 3ms em vez de impossível.

3. **`compute_shapley_values`** (a versão reimplementada de `SyntaxExplainer.compute_shapley_values`) mantém exatamente a mesma lógica de acumulação de contribuição marginal do original — mesmo loop, mesma normalização final (`values / sum(values)`) — mas parametrizada por um callback `get_contribution_fn(mask) -> float` em vez de um método amarrado a captum, e operando em escalares (o original fazia `eval_diff[0, 0].item()` pra desempacotar tensor; aqui a função de score já devolve `float`).

O modo `"shap"` (Shapley exato/irrestrito, sem árvore sintática) foi preservado como está no original — continua O(2^M) por definição — mas só é computado quando `algorithm="shap"` é escolhido, nunca incondicionalmente como acontecia antes.

## 4. Da geração autoregressiva pra classificação binária

`classifier_explainer.py::ClassifierSyntaxExplainer` é uma classe própria, **sem herdar de nada do `thirdparty/`** — não é subclasse de `SyntaxExplainer` porque `__init__`/`explain_row` do original assumem em vários pontos um modelo de geração (`.get_outputs()`, `.generate()`, `LLMAttribution`); remover tudo isso não deixaria quase nada pra herdar de fato.

Principais diferenças de comportamento:

| | Original (`SyntaxExplainer`) | Adaptado (`ClassifierSyntaxExplainer`) |
|---|---|---|
| Alvo explicado | Próximo token gerado (`model.generate()`, 1 token de cada vez) | `P(não-benigno) = Σ P(classe)` pra toda classe cujo rótulo não contém `"benign"` |
| Mecanismo de score | `captum.attr.LLMAttribution` + `ShapleyValueSampling` sobre teacher forcing | Forward pass direto (`self.model([text])[0]`) via `thirdparty.models.TransformersPipeline` |
| Por que agregado binário, não argmax de 3 classes | N/A | Mantém a quantidade explicada consistente entre amostras e alinhada com o que `PromptGuardDefense.detect_flag` de fato checa (limiar sobre `P(não-benigno)`, não sobre a classe crua) |
| Como decide quais classes contam como "não-benignas" | N/A | `"benign" not in str(label).lower()` — mesma checagem por substring que `PromptGuardDefense.execute` já usa, em vez de fixar nomes como `INJECTION`/`JAILBREAK` (robusto a variações do `id2label` real do modelo, nunca confirmado por não ter acesso ao HF gated) |
| Alinhamento token↔máscara | Feito implicitamente pelo captum | `_rebase_dependency_tree`: `get_token_dependency_tree` indexa a tokenização crua (com `[CLS]` na posição 0); precisa subtrair `keep_prefix` pra alinhar com o array de máscara de M tokens de conteúdo — bug encontrado e corrigido nos testes com tokenizer real |
| Dependência de captum | Sim | Nenhuma |

## 5. `context` inteiro em vez de span isolado

O paper testa o método em frases curtas isoladas. O PromptGuard, na prática, recebe o `context` inteiro (várias sentenças). `classifier_explainer.py::explain_context` é orquestração nova, sem equivalente no original:

1. Segmenta `context` em sentenças via spaCy (`doc.sents`).
2. Roda o cálculo Shapley (via `compute_shapley_values`, seção 3) **por sentença**, com o resto do `context` mantido fixo como pano de fundo em toda avaliação de coalizão dessa sentença — ou seja, cada coalizão testada é pontuada contra o `context` completo, só os tokens dessa sentença variam.
3. Concatena os valores por sentença na ordem original de tokens do `context`.
4. Localiza `injected_span` via `context.find(injected_task)`, mapeado de caractere pra índice de token de conteúdo.

Trade-off documentado no plano: custo total de `explain_context` é a soma de ~6–10 chamadas do tamanho de uma sentença (M~10–20 cada, dentro da faixa testada pelo paper), não uma chamada única com M~150 — o que seria inviável mesmo com o fix da seção 3, já que o número de coalizões ainda cresce com o tamanho de cada nível da árvore.

## 6. Integração no PIArena (sem equivalente no repo original)

Tudo nesta seção é código novo, não uma adaptação de algo que já existia no SyntaxSHAP original:

- **`piarena/registry.py`**: novo `XAI_REGISTRY`, mecanicamente idêntico a `ATTACK_REGISTRY`/`DEFENSE_REGISTRY` — primeiro precedente de um terceiro tipo de plugin no PIArena.
- **`piarena/xai/base.py`**: `BaseXAI(ABC)` com `explain(target_inst, context, injected_task, **kwargs) -> dict` abstrato, mesmo padrão de `BaseAttack`/`BaseDefense`.
- **`piarena/xai/syntaxshap/xai_syntaxshap.py`**: `SyntaxShapXAI`, o plugin registrado como `--xai syntaxshap`. Carrega seu **próprio** pipeline HuggingFace (`top_k=None`, com fallback pra `return_all_scores=True` dependendo da versão de `transformers` instalada) — deliberadamente desacoplado de `PromptGuardDefense._detector` (não acessa estado privado da defesa).
- **`main.py`**: `--xai` (flag própria, como `--attack`/`--defense`), `--xai_config` (só YAML, como `attack_config`/`defense_config`), fase XAI rodando logo depois da fase Defense, resultado gravado em `result_dp["xai_result"]`. Também: `--limit` (nova, flag genérica de smoke-test — antes não existia nenhuma forma de limitar quantas amostras do dataset processar) e sufixo `-{xai}` no nome do arquivo de resultado, pra uma run com `--xai` não colidir silenciosamente com uma run sem XAI sob o mesmo `--name`.
- **`piarena/xai/metrics.py` + `scripts/xai_metrics.py`**: cálculo de Fidelity/acc@1/injected-span-rank e plots (`shap.plots.text`/`shap.plots.bar`) — reinterpretação das métricas do paper pra saída do PIArena, não uma cópia de `syntax-shap-main-oficial/syntaxshap/metrics.py` (esse arquivo do paper original mede coisas diferentes, amarradas a geração de texto, e não foi reaproveitado). `div@K` (Eq. 7) foi implementada e depois removida — sobre o par agregado `[P(BENIGN), P(MALIGN)]` ela vira algebricamente `2·|Fidelity(t)|`. Detalhe completo em `markdowns-do-experimento/metricas-xai-syntaxshap.md`.
- **`requirements.txt`**: `pandas`, `cloudpickle`, `shap`, `matplotlib` adicionados (só para `--xai`/o script de métricas); `spacy` já estava listado, mas passou a ser documentado que precisa de `python -m spacy download en_core_web_sm` separadamente. **Sem `captum`** — eliminado pela reimplementação da seção 3.

## 7. O que foi validado vs. não

Ambiente sem GPU e sem acesso ao Prompt-Guard-86M (gated no HF), então a pilotagem real (`main.py --xai syntaxshap` contra o modelo de verdade) **nunca rodou**. O que foi validado, numa venv isolada com um classificador mockado + tokenizer BERT real (não-gated) + parsing spaCy real:

- `coalition.py` bate exatamente com um oráculo por força-bruta, e o fix do bug O(2^M) funciona de fato (M=60 em 3ms, antes impossível).
- `ClassifierSyntaxExplainer.explain_row`/`explain_context` rodam ponta a ponta corretamente (foi rodando esses testes que os dois bugs da seção 2.2/2.3 foram encontrados e corrigidos), localizam `injected_span` corretamente, e rankeiam tokens-gatilho como mais importantes.
- As funções numéricas de `piarena/xai/metrics.py`.
- A propriedade de lazy-import de `piarena.xai` (importar o módulo não puxa torch/spacy/sklearn — confirmado empiricamente).

Não validado ainda: rodar contra o Prompt-Guard-86M real (precisa de `huggingface-cli login` + GPU), e confirmar se `id2label` do modelo real bate com o `BENIGN`/`INJECTION`/`JAILBREAK` assumido (mitigado pelo design por substring da seção 4, mas nunca checado contra o `config.json` real). Passo a passo completo em `PIArena-main/plans/xai-syntaxshap-promptguard.md`, seção "Riscos/suposições explícitos".
