# Tema Beamer CISSA

Tema Beamer que reproduz o template "PITCH DPP" (fundo escuro, DM Sans, azul e laranja, logos CISSA / CESAR School / FACEPE / MCTI / Embrapii).

## Arquivos

- `beamerthemeCISSA.sty`: o tema
- `assets/`: logos e formas decorativas (PNG)
- `fonts/`: DM Sans (Regular, Bold, Black, Italic, BoldItalic)
- `exemplo.tex`: os 5 slides do template recriados
- `compilar.sh`: compila duas vezes com XeLaTeX

Mantenha `.tex`, `.sty`, `assets/` e `fonts/` na mesma pasta. Compile com **XeLaTeX ou LuaLaTeX**, sempre **duas vezes** (as formas usam `remember picture`).

## Uso

```latex
\documentclass[aspectratio=169]{beamer}
\usetheme{CISSA}
\begin{document}

\ciCapa{Título da apresentação}{Subtítulo}          % capa

\begin{frame}{Título do slide}                      % layout padrão: título azul
  \ciRotulo{Rótulo:}
  \ciCitacao{Texto com seta azul, em itálico.}
  \ciDestaque{Objetivo:}{Texto na caixa arredondada}
\end{frame}

\cilayout{imagem}                                   % escolher ANTES do frame
\begin{frame}{Título com imagem}                    % título branco + imagem circular
  \begin{ciLista}
    \item \textbf{Termo} descrição
  \end{ciLista}
\end{frame}

\cilayout{secao}                                    % título centralizado
\begin{frame}{Fatores \ciAzul{determinantes} da robustez}\end{frame}

\ciFim{Obrigado!}                                   % encerramento
\end{document}
```

## Comandos

| Comando | Para que serve |
|---|---|
| `\ciCapa{título}{subtítulo}` | Slide de capa completo |
| `\ciFim{texto}` | Slide final completo |
| `\cilayout{conteudo\|imagem\|secao}` | Layout do **próximo** frame (padrão: `conteudo`) |
| `\ciRotulo{texto}` | Rótulo em negrito |
| `\ciCitacao{texto}` | Linha com seta azul, em itálico |
| `\ciDestaque[pos]{título}{texto}` | Caixa arredondada com gradiente (`pos` = altura, fração da página; padrão 0.722) |
| `ciLista` | Lista com bolinhas laranja grandes |
| `\ciAzul{texto}` / `\ciLaranja{texto}` | Cor de destaque no texto ou título |
| `\ciImagemLateral` | Imagem do layout `imagem` (padrão `img-cadeado-circulo.png`, em `assets/`; use imagem quadrada, é recortada em círculo) |

Para trocar a imagem: `\renewcommand{\ciImagemLateral}{minha-imagem.png}` (arquivo em `assets/`).

## Observações

- `\ciDestaque` é posicionada de forma absoluta (como no template): não deixe o texto do slide descer até ela.
- O layout `imagem` usa coluna estreita de texto (até ~45% da largura); use `ciLista` nele.
- Se `assets/` ou `fonts/` ficarem em outro lugar, defina antes de `\usetheme`: `\newcommand{\ciPastaAssets}{caminho/}` e `\newcommand{\ciPastaFontes}{caminho/}`.
