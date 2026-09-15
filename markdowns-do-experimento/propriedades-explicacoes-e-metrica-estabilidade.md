# Propriedades de explicações XAI e uma métrica nova: Stability sob variação de ataque

Este documento resume uma taxonomia geral de "o que faz uma explicação boa" (propriedades de explicações individuais e de métodos de explicação), usa-a para mapear onde as três métricas já implementadas do experimento (`fidelity`, `acc@1`, `injected_span_percentile` — ver [`metricas-xai-syntaxshap.md`](metricas-xai-syntaxshap.md)) se encaixam, identifica lacunas, e propõe uma quarta métrica — **Stability sob variação de ataque**, via Rank-Biased Overlap (RBO) — para preenchar a lacuna mais barata e mais bem suportada pelos dados que o PI Arena já produz.

## Fontes

- Taxonomia de propriedades: capítulo sobre avaliação de explicações do livro *Interpretable Machine Learning* (C. Molnar), que por sua vez organiza a proposta de Robnik-Šikonja & Bohanec (2018).
- [Jørgensen et al., "Are Multilingual Sentiment Models Equally Right for the Right Reasons?" (BlackboxNLP 2022)](https://aclanthology.org/2022.blackboxnlp-1.11.pdf) — usa **Rank-Biased Overlap (RBO)** para comparar o ranking de importância de tokens dado por um modelo (saliência) contra um ranking-gabarito (rationale humano), justamente porque RBO tolera listas de tamanhos diferentes e pondera o topo do ranking mais que a cauda — é a peça que sustenta a métrica proposta abaixo.
- [Shen et al., "An Interpretability Evaluation Benchmark for Pre-trained Language Models" (2022)](https://arxiv.org/pdf/2207.13948) — inclui explicitamente, como um dos eixos do benchmark, checar **se a explicação/rationale do modelo permanece consistente quando o dado de entrada é perturbado**. É o precedente direto para tratar "mesmo `context`, ataque diferente" como o tipo de perturbação controlada a testar.

## Taxonomia: propriedades de explicações individuais

| Propriedade | Pergunta que responde |
|---|---|
| Accuracy | A explicação prevê bem dados não vistos, se usada no lugar do modelo? |
| **Fidelity** | A explicação aproxima bem a predição do modelo em si (não da realidade)? |
| Consistency | Duas explicações diferem entre **modelos** diferentes treinados na mesma tarefa com predições parecidas? |
| **Stability** | Duas explicações diferem entre **instâncias parecidas**, para um modelo fixo? |
| Comprehensibility | Um humano consegue entender a explicação? |
| Certainty | A explicação reflete a confiança do modelo na predição? |
| Degree of Importance | Fica claro qual parte da explicação é a mais importante? |
| Novelty | A explicação sinaliza quando a instância está fora da distribuição de treino? |
| Representativeness | Quantas instâncias a explicação cobre (uma só / um subconjunto / o modelo inteiro)? |

## Taxonomia: propriedades de métodos de explicação

| Propriedade | Pergunta que responde |
|---|---|
| Expressive Power | Em que "linguagem" o método produz explicações (regras, pesos, texto, árvore...)? |
| Translucency | O método olha para dentro do modelo (parâmetros) ou só para entradas/saídas? |
| Portability | Para quantos tipos de modelo o método serve? |
| Algorithmic Complexity | Qual o custo computacional de gerar a explicação? |

## Onde as métricas atuais do experimento já encaixam

| Métrica do experimento | Propriedade (individual) | Observação |
|---|---|---|
| `fidelity()` | **Fidelity** | Mapeamento direto — é literalmente a Eq. 6 do paper do SyntaxSHAP, reinterpretada para `P(non-benign)`. |
| `acc_at_1()` | Fidelity (versão binarizada) + **Degree of Importance** | Além de ser uma variante discreta de Fidelity, também responde "os tokens mantidos bastam pra manter a decisão operacional?", isto é, se a explicação captura o suficiente da importância real. |
| `injected_span_percentile()` | **Degree of Importance** (localizado no span certo) | É a métrica 3 do documento de métricas — mede se o SyntaxSHAP acerta *onde* está a importância dentro do texto, contra um gabarito automático (posição do `injected_task`). |

Nenhuma das três métricas atuais cobre **Stability**, **Certainty**, **Consistency**, **Comprehensibility**, **Novelty** ou **Representativeness**. Das seis, a maioria não é viável no experimento como está:

- **Consistency** exigiria um segundo classificador treinado na mesma tarefa que o PromptGuard — não existe no experimento (só há um modelo sendo explicado).
- **Comprehensibility** exigiria avaliação humana — fora de escopo (é justamente o motivo pelo qual a métrica 3 do documento de métricas substitui a Semantic Alignment original do paper, baseada em anotação humana, por um gabarito automático).
- **Novelty** exigiria uma noção de distribuição de treino do PromptGuard, que não é acessível a partir do PI Arena.
- **Representativeness** não se aplica bem — o SyntaxSHAP já é local por construção (uma explicação por amostra), então essa propriedade tem resposta trivial ("cobre 1 instância") em vez de ser uma métrica a calcular.
- **Certainty** é parcialmente observável de graça (`full_value`/`p_malign` já *é* a confiança do modelo, salvo em todo `xai_result`), mas transformar isso numa métrica de avaliação da explicação (não só do modelo) exigiria decidir o que "a explicação refletir a confiança" significaria concretamente — por exemplo, correlacionar a magnitude/dispersão dos `values` com a distância de `p_malign` a 0.5. É uma métrica plausível, mas mais barata e menos fundamentada nos dois papers de referência do que a proposta abaixo; fica anotada aqui como candidata futura, não desenvolvida.
- **Stability** é a única que sobra com (a) suporte direto nos dois papers linkados, e (b) dado que o PI Arena já produz sem precisar de nenhuma geração de dado nova.

## A lacuna escolhida: Stability

A definição de Stability é "o quanto a explicação muda entre instâncias parecidas, pra um modelo fixo — junto com o alerta explícito de que instabilidade pode vir de dois lugares: alta variância do próprio método de explicação, ou dependência de algo não-determinístico (ex.: uma etapa de amostragem de dados).

O SyntaxSHAP, do jeito que está implementado em `coalition.py`, é **exatamente determinístico** — sem amostragem, sem dependência de seed (é enumeração exata de coalizões restritas pela árvore de dependência, daí o próprio bug de custo documentado em `custo-computacional-syntaxshap.md`). Então a pergunta de "instabilidade por não-determinismo do método" não se aplica aqui; a pergunta que sobra, e que é a genuinamente interessante, é a outra metade da definição: **será que instâncias parecidas dão explicações parecidas?**

### Por que "ataque diferente, mesmo contexto" é o par de instâncias "parecidas" certo pra este experimento

O PI Arena já roda o mesmo par `(context, injected_task)` sob 5 ataques heurísticos diferentes (`direct`/`ignore`/`completion`/`combined`/`character`, `piarena/attacks/heuristic.py`), cada um só prefixando `injected_task` com uma string fixa (`PREFIX`) antes de injetar — o resto do `context` não muda. Melhor ainda: a posição de inserção (`inject_position="random"`) é resolvida por `random.seed(num_sentences)` dentro de `piarena/utils.py:inject()` — ou seja, é determinística **em função do tamanho do contexto**, não da seed da CLI nem do ataque. Na prática, isso significa que, pra uma mesma amostra do dataset, os 5 `injected_context` resultantes inserem o conteúdo injetado exatamente na mesma sentença-alvo, e todo o resto do texto permanece palavra-por-palavra idêntico.

Isso é exatamente o tipo de "instância parecida" que a definição de Stability pede — não duas amostras aleatoriamente parecidas do dataset, mas o mesmo texto-base com uma perturbação pequena e controlada (o prefixo do ataque) — e é literalmente o mesmo desenho de "checar se o rationale do modelo é robusto a perturbação do dado" do benchmark de Shen et al. (2022) citado acima, só que aqui a "perturbação" já vem pronta do próprio pipeline de ataque do PI Arena em vez de precisar ser gerada como um passo novo (ao contrário da métrica de Coherency do paper original do SyntaxSHAP, que a seção "O que existe no paper e não foi portado" do documento de métricas descarta justamente por exigir geração de pares perturbados que o PI Arena não fornece — aqui, o PI Arena já fornece).

### Por que Rank-Biased Overlap (RBO) e não Spearman

A métrica de concordância que já existia no experimento (`spearman()`, removida junto com `algorithm="syntax-w"`) comparava dois rankings de importância **sobre o mesmo texto tokenizado** — Spearman exige dois vetores do mesmo tamanho, alinhados posição-a-posição. Isso não serve para comparar explicações entre ataques diferentes: `ignore` insere `"Ignore previous instructions. "` antes de `injected_task`, `combined` insere um prefixo maior ainda — o texto final tem números de tokens diferentes, e a tokenização por subpalavra desloca offsets de forma não trivial. Não há um jeito direto de alinhar "token 42 do ranking de `direct`" com "token 42 do ranking de `ignore`".

RBO (Webber, Moffat & Zobel, 2010 — a métrica que Jørgensen et al. 2022 usa exatamente por este motivo) resolve isso porque compara duas listas ordenadas **por identidade de item**, não por posição, e não exige que as listas tenham o mesmo tamanho nem o mesmo conjunto de itens: mede sobreposição no topo-`d` de cada lista, para `d` crescente, ponderada por um fator `p` que decai geometricamente (então itens perto do topo do ranking pesam mais que a cauda):

```
RBO(p) = (1 − p) · Σ_{d=1}^∞ p^(d−1) · (|S_d ∩ T_d| / d)
```

onde `S_d`/`T_d` são os top-`d` itens de cada ranking. `RBO = 1` → mesmo topo de importância nas duas explicações; `RBO → 0` → topos disjuntos.

### O que usar como "item" comparável entre ataques

A unidade de comparação não pode ser o token de subpalavra (varia com o ataque, como acima) nem pode incluir o próprio span injetado (seu conteúdo textual muda entre ataques — `ignore` e `combined` têm mais palavras injetadas que `direct`). A unidade estável é a **palavra do `context` original, fora do `injected_span`** — que, pelo argumento da seção anterior, é textualmente idêntica e ocupa a mesma sentença em todos os 5 ataques. Concretamente:

1. Para cada ataque `a` já explicado (mesma amostra-base, mesmo `context`/`injected_task`), pegar `xai_result["context"]` (`tokens`, `values`, `injected_span`).
2. Agrupar os tokens de subpalavra de volta em palavras (via offsets do tokenizer) e excluir as palavras dentro de `injected_span`.
3. Ordenar as palavras remanescentes por `|valor|` agregado (soma ou máximo dos subtokens da palavra) — dá um ranking `R_a` de palavras do `context` original.
4. Para cada par de ataques `(a1, a2)` com resultado disponível na mesma amostra, calcular `RBO(R_{a1}, R_{a2})` (sugestão: `p = 0.9`, o mesmo valor comumente usado no paper original de RBO e em Jørgensen et al., que dá peso considerável aos top ~10 itens sem ignorar o resto).
5. Reportar a média de `RBO` sobre todos os pares de ataques disponíveis por amostra, e depois a média/distribuição dessa métrica por amostra no relatório agregado (mesmo padrão de agregação que `fidelity_acc_bars.png` já usa para Fidelity/acc@1).

**Leitura:** `stability_rbo` alto → o SyntaxSHAP aponta a mesma parte do `context` legítimo como (ir)relevante independentemente de qual prefixo de ataque foi usado, isto é, a explicação está lendo estrutura sintática real do texto, não artefato do prefixo específico. `stability_rbo` baixo é um sinal de alerta — sugeriria que a explicação está reagindo ao ataque (razoável, já que o ataque é *o que muda* a decisão) só que "vazando" importância pra palavras do `context` legítimo que não deveriam estar entre as mais importantes, de forma inconsistente entre ataques diferentes do mesmo conteúdo.

### Custo — não é grátis, mas é reaproveitável

Essa métrica exige ter `xai_result` computado para o mesmo conjunto de amostras sob **múltiplos** `--attack`, não só um — ou seja, multiplica o custo já documentado em `custo-computacional-syntaxshap.md` pelo número de ataques comparados (até 5×, se todos os heurísticos forem incluídos). Não precisa de nenhuma chamada extra ao classificador além das explicações que o próprio `main.py --xai syntaxshap` já produziria rodando cada ataque separadamente — ao contrário de Fidelity/acc@1, que re-rodam o classificador por amostra por threshold, `stability_rbo` é puro pós-processamento sobre `xai_result`s já gravados. Na prática, isso significa: continuar restrito ao subconjunto barato já identificado (os índices de custo viável do `squad_v2`, ver `custo-computacional-syntaxshap.md`), e não expandir para os 5 ataques de uma vez sem antes confirmar que o subconjunto barato para `direct` continua barato sob os prefixos dos outros ataques (prefixos como o de `combined` acrescentam palavras nas sentenças injetadas, o que pode mudar a árvore de dependência daquela sentença especificamente — mas não deveria mudar o custo das outras sentenças do `context`, já que estas permanecem intocadas).

## Resumo

| Propriedade coberta | Métrica proposta | Depende de dado novo? |
|---|---|---|
| Stability | `stability_rbo` — RBO entre rankings de palavras do `context` (fora do span injetado) sob ataques heurísticos diferentes, mesma amostra-base | Não — reaproveita rodadas de `main.py --xai syntaxshap` sob `--attack` diferentes na mesma amostra; nenhuma perturbação nova precisa ser gerada, graças a `random.seed(num_sentences)` em `piarena/utils.py:inject()` fixar a posição de inserção independentemente do ataque. |

Candidatas descartadas (por falta de dado/infra no experimento, não por falta de valor): Consistency (exigiria 2º modelo), Comprehensibility e Novelty (fora de escopo/sem dado de treino do PromptGuard acessível), Representativeness (trivial para um método já local). Certainty é a mais próxima de viável — fica registrada como possível quinta métrica futura, correlacionando magnitude/dispersão dos `values` com a distância de `p_malign` a 0.5, mas não desenvolvida aqui por ter suporte mais fraco nos dois papers usados como referência.
