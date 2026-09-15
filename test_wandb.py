#!/usr/bin/env python3
"""Teste isolado e rápido: confirma que wandb está instalado, que o modo
offline grava local (sem precisar de GPU/HF/modelo nenhum) e, se
`wandb login` já foi feito, sincroniza pro wandb.ai de verdade e imprime o
link da run -- não roda nada do pipeline real (main.py etc.), só valida o
mecanismo em si.

Uso:
    python test_wandb.py
"""
import os
import subprocess
import sys

os.environ.setdefault("WANDB_MODE", "offline")

try:
    import wandb
except ImportError:
    sys.exit("wandb não está instalado neste ambiente/venv (pip install wandb).")

print(f"wandb {wandb.__version__} instalado. WANDB_MODE={os.environ['WANDB_MODE']}")

run = wandb.init(
    project=os.environ.get("WANDB_PROJECT", "piarena-xai-sweep"),
    job_type="smoke-test",
    name="test-wandb-conexao",
)

for step in range(5):
    wandb.log({"exemplo_metric": step * 1.5}, step=step)

wandb.summary["ok"] = True
run_dir = run.dir
run_path = run.path  # "<entity>/<project>/<run_id>"
wandb.finish()

print(f"\nRun offline gravada em: {run_dir}")
print("Tentando sincronizar agora (precisa de 'wandb login' feito antes)...")

result = subprocess.run(
    # `sys.executable -m wandb` (not a bare "wandb") — mais portável: usa o
    # mesmo interpretador/venv que já importou wandb acima, sem depender de
    # `wandb` (o executável) estar resolvível no PATH deste shell.
    [sys.executable, "-m", "wandb", "sync", os.path.dirname(run_dir)],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    print(
        "\nSync falhou -- provavelmente falta 'wandb login' "
        "(ou WANDB_API_KEY no ambiente). Os dados continuam salvos em "
        f"{os.path.dirname(run_dir)}, é só rodar 'wandb login' e depois "
        f"'wandb sync {os.path.dirname(run_dir)}' de novo."
    )
    sys.exit(1)

print(f"\nSincronizado! Confira em: https://wandb.ai/{run_path}")
