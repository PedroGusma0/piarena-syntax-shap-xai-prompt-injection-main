# Gráficos oficiais do SHAP: `bar_global.png` e `saliency_bars.html`

> Estado: **teste exploratório**. Os scripts são separados do `scripts/xai_metrics.py` e ainda não foram integrados ao pipeline, nem documentados em `docs/xai/` ou no `CHANGELOG.md`. Complementa [[metricas-xai-syntaxshap]] (métricas Fidelity/acc@1/Alinhamento Semântico), cujos gráficos (`saliency_maps.html`, `alignment_histogram.png`, `fidelity_acc_bars.png`) continuam como estão.

## 1. O que foi criado

Dois gráficos novos por ataque, feitos **só com `shap.plots.bar`** (biblioteca oficial `shap`, sem matplotlib manual) e **sem nenhuma inferência**: leem apenas o JSON bruto do resultado, que já traz `tokens`, `values`, `base_value`, `full_value`, `p_malign` e `injected_span` em `xai_result["context"]`.

| Arquivo | O que mostra |
|---|---|
| `bar_global.png` | Importância média absoluta (`|SHAP|` médio) por token, agregada sobre todas as amostras bloqueadas do ataque. Só entram tokens com pelo menos 5 ocorrências (`n≥5`), e o `n` aparece no rótulo. |
| `saliency_bars.html` | Um `shap.plots.bar` **local** por amostra bloqueada: os 15 tokens de maior |valor| daquela amostra, com o restante agrupado em "Sum of N other features". Tokens dentro do span injetado levam o prefixo `▶`. Arquivo único, com os PNGs embutidos em base64. |

O `saliency_bars.html` cumpre o papel de explicar as instâncias individuais, como o `saliency_maps.html` (`shap.plots.text`), mas de forma quantitativa: mostra quais tokens carregam a decisão e o quanto.

## 2. Onde ficam

Dentro da pasta de cada ataque, junto do `saliency_maps.html`:

```
<dataset>/.../xai_metrics/plots/<dataset>-<llm>-<ataque>-promptguard-kernelshap-42/
  alignment_histogram.png
  fidelity_acc_bars.png
  saliency_maps.html      (já existia)
  bar_global.png          (novo)
  saliency_bars.html      (novo)
```

## 3. Cobertura (rodada de 2026-10-01)

Datasets processados, 4 ataques cada (combined, completion, direct, ignore), backend Kernel SHAP/captum, `Qwen3-4B-Instruct-2507` + `promptguard`, seed 42. Nº de amostras bloqueadas (e explicadas) = nº de blocos no HTML:

| Dataset | combined | completion | direct | ignore |
|---|---:|---:|---:|---:|
| dolly_closed_qa | 200 | 199 | 138 | 200 |
| dolly_summarization | 193 | 190 | 114 | 193 |
| gov_report_long | 7 | 5 | — | 7 |
| hotpotqa_rag | 81 | 71 | 49 | 81 |
| msmarco_rag | 94 | 88 | 64 | 94 |
| multi_news_long | 46 | 32 | 18 | 48 |
| nq_rag | 66 | 56 | 40 | 71 |
| squad_v2 (captum) | 200 | 199 | 134 | 200 |

Ressalvas:
- `gov_report_long` + `direct`: **0** amostras bloqueadas, então nenhum arquivo foi gerado.
- `gov_report_long` (demais ataques): só 5 a 7 amostras. O `bar_global.png` desses tem pouca força estatística, e o corte `n≥5` deixa poucos tokens.
- `squad_v2`: o conjunto `shap/` (biblioteca `shap`, em vez de captum) foi **excluído de propósito**; só o `captum/` foi processado.
- Alguns ataques têm amostras bloqueadas sem `xai_result` (ex.: dolly_summarization combined tem 195 bloqueadas e 193 explicadas). O loader as descarta, por isso o HTML tem o número de explicadas.
- Os gráficos dos datasets de documento longo (`multi_news_long`, `gov_report_long`) foram gerados, mas a legibilidade não foi inspecionada visualmente.

## 4. Como reproduzir

Scripts em `PIArena-main/scripts/`:
- `xai_shap_plots_test.py`: um resultado bruto por vez (`--result <json>`).
- `xai_shap_plots_batch.py`: varre todas as pastas `kernelshap_*` da raiz do repositório. Aceita `--only <substring>` (ex.: `dolly_closed_qa`), `--skip-existing`, `--top-k` e `--max-html-samples`. Ignora `xai_metrics/`, `tmp_attack_results/`, `*_metrics.json` e o diretório `shap/` do squad.

```bash
python PIArena-main/scripts/xai_shap_plots_batch.py --skip-existing
```

Dependências: `shap`, `matplotlib`, `numpy` (todos já em `requirements.txt`). Não precisa de GPU nem do modelo. A entrada é o JSON bruto (`...-kernelshap-42.json`), **não** o `..._metrics.json`, que só tem escalares.

## 5. Como ler

- **`bar_global.png`** no `ignore` do dolly_closed_qa: `ignore` (0.44), `instructions.` (0.18) e `previous` (0.15) dominam. Isso mostra que o PromptGuard reage ao padrão literal "ignore previous instructions". Tokens de baixa frequência aparecem quando fazem parte de um ataque específico (ex.: `conclude`, `simulate`, `emulate` nos ataques de template).
- **`saliency_bars.html`**: os tokens do span injetado (`▶`) normalmente ficam no topo, e o contexto legítimo aparece com valores pequenos, positivos ou negativos. Quando um token de contexto aparece no topo, é um candidato a atribuição espúria.
- Aditividade conferida em todos os runs: `|base_value + Σ values − full_value|` ≈ 1e-4 (máx. 8e-5 no dolly_closed_qa/ignore).

## 6. Decisões e descartes

Foram testados no `ignore` do dolly_closed_qa e **removidos** a pedido, depois de inspeção:
- **Waterfall**: redundante com o `saliency_bars.html`, que já explica cada instância. Além disso, com `P(malign)` ≈ 1 e valor base ≈ 0,001 quase sempre, o waterfall é sempre a mesma subida de 0 a 1.
- **Heatmap** (amostras × faixas de posição relativa): pouco intuitivo (linhas são posições, não tokens, e a ordenação é artificial). A diagonal vermelha apenas reflete onde o span foi inserido.
- **Beeswarm**: SHAP aqui é por token, sem features tabulares. A versão testada usava grupos de tokens como colunas e o tamanho do grupo como cor, uma adaptação forçada.

## 7. Limitações

- O corte `n≥5` e a normalização (`strip` + minúsculas) do `bar_global.png` são escolhas desta versão exploratória.
- Tokens CJK saem sem glifo nos PNGs (fonte DejaVu Sans), só estético.
- O HTML tem ~60 KB por amostra (PNG base64): ~11 MB com 200 amostras.
- Falta integrar ao `xai_metrics.py` (e então atualizar `docs/xai/` e o `CHANGELOG.md`, como pede o `CLAUDE.md` do PIArena).

## 8. Com Integrated Gradients (`--xai ig`)

`xai_shap_plots_test.py`, `xai_shap_plots_batch.py` e `xai_bar_global_benign.py` aceitam resultados do IG (mesmo formato por palavra). O batch e o bar global ganharam `--xai-method {kernelshap,ig}` (padrão `kernelshap`): troca o glob (`*-ig-42.json`) e as pastas (`ig_*`).

```bash
python PIArena-main/scripts/xai_shap_plots_batch.py --xai-method ig --skip-existing
```

Ressalvas ao ler: a aditividade só vale até `ig_convergence_delta` (o passo "aditividade conferida" da seção 5 deve reportar o delta em vez de ~1e-4), e o `|valor|` médio do `bar_global.png` é comparável em ranking, não em escala, com o do KernelSHAP. Ver [[ig-adaptacao-encoder-classifier]]. Ainda sem rodada real.
