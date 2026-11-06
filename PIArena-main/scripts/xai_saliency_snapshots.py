"""PNG do mapa de saliência (bloco "inputs" do shap.plots.text) de cada amostra.

Lê o saliency_maps.html gerado por scripts/xai_metrics.py (que já desenha a borda
na tarefa injetada), recorta o bloco de texto de cada "Sample N", renderiza com o
Chrome headless e corta as margens brancas.

Uso:
    python scripts/xai_saliency_snapshots.py --html <plots>/<stem>/saliency_maps.html --out-dir <relatorio_v2>/saliency
    python scripts/xai_saliency_snapshots.py --html ... --out-dir ... --idx 0 3 --width 820

Saída: <out-dir>/idxN.png. Precisa de google-chrome/chromium no PATH e Pillow.
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

_SAMPLE = re.compile(r"<h3>Sample (\d+)\b")
_PAGE = """<!doctype html><html><head><meta charset='utf-8'><style>
body {{ margin: 12px; width: {width}px; background: #fff; font-family: Arial, Helvetica, sans-serif; font-size: 15px; line-height: 1.55; }}
</style></head><body>{body}</body></html>"""


def split_samples(html):
    """{idx: bloco HTML 'inputs'} — do <div align='center'> que contém 'inputs' até o fim da amostra."""
    marks = [(m.start(), m.group(1)) for m in _SAMPLE.finditer(html)]
    out = {}
    for (a, idx), (b, _) in zip(marks, marks[1:] + [(len(html), None)]):
        seg = html[a:b]
        j = seg.rfind("<div align='center'><div style=\"color: rgb(120,120,120); font-size: 12px; margin-top: -15px;\">inputs</div>")
        if j != -1:
            out[idx] = re.sub(r"</body>\s*</html>\s*$", "", seg[j:])
    return out


def trim(path, pad=6):
    im = Image.open(path).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:
        l, t, r, b = bbox
        im.crop((max(0, l - pad), max(0, t - pad), min(im.width, r + pad), min(im.height, b + pad))).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--html", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--idx", nargs="*", default=None)
    p.add_argument("--width", type=int, default=820, help="largura do texto em px")
    p.add_argument("--scale", type=float, default=2.0, help="device scale factor (nitidez)")
    a = p.parse_args()

    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if not chrome:
        raise SystemExit("google-chrome/chromium não encontrado no PATH")
    blocks = split_samples(a.html.read_text(encoding="utf-8"))
    idxs = a.idx or list(blocks)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for idx in idxs:
            if idx not in blocks:
                print(f"[skip] amostra {idx}: sem bloco 'inputs' no HTML")
                continue
            page = Path(tmp) / f"{idx}.html"
            page.write_text(_PAGE.format(width=a.width, body=blocks[idx]), encoding="utf-8")
            out = (a.out_dir / f"idx{idx}.png").resolve()
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                            f"--force-device-scale-factor={a.scale}", f"--window-size={a.width + 24},1400",
                            f"--screenshot={out}", page.resolve().as_uri()],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            trim(out)
            print(f"[ok] {out}")


if __name__ == "__main__":
    main()
