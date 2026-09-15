#!/usr/bin/env python3
"""Orquestra o sweep completo squad_v2 x {direct,ignore,combined,completion}
x {syntaxshap,kernelshap}: pra cada ataque, roda main.py (syntaxshap
primeiro, depois kernelshap), calcula scripts/xai_metrics.py assim que cada
run termina, e roda scripts/xai_compare_fidelity.py assim que os dois
métodos tiverem métricas prontas pra aquele ataque. Retomável: pula
qualquer etapa cuja saída já exista, a não ser que --force seja passado.

Os dois métodos XAI vivem na MESMA árvore PIArena-main/ desde que
piarena/xai/kernelshap/ foi mesclado de volta de um checkout paralelo
(piarena_xai_kernel_shap/PIArena-main/, removido) — uma única instalação
`piarena` no venv já resolve os dois, sem o malabarismo de PYTHONPATH que
uma versão anterior deste script precisava pra escolher entre duas árvores
com o mesmo nome de pacote.

Ver o plano completo (sweep completo squad_v2 x 4 ataques x 2 métodos XAI)
para o design/motivação de cada decisão aqui.

Uso:
    python run_full_sweep.py                   # os 4 ataques, os 2 métodos
    python run_full_sweep.py --attacks direct   # só um ataque
    python run_full_sweep.py --only-kernelshap  # pula o syntaxshap
    python run_full_sweep.py --force            # ignora saídas já existentes
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIARENA_DIR = ROOT / "PIArena-main"
LOG_DIR = ROOT / "logs" / "full_sweep"

ATTACKS = ["direct", "ignore", "combined", "completion"]
DATASET = "datasets/squad_v2.json"  # arquivo local (200 linhas) -- não o
                                     # atalho "squad_v2", que cai no branch de
                                     # main.py que baixa do HF Hub com
                                     # download_mode="force_redownload" toda
                                     # vez (main.py:99-106).
BACKEND_LLM = "Qwen/Qwen3-4B-Instruct-2507"
DEFENSE = "promptguard"
NAME = "full_sweep"
SEED = 42


def _dataset_name() -> str:
    return Path(DATASET).stem  # "squad_v2"


def _llm_name() -> str:
    return BACKEND_LLM.replace("/", "-")


def result_path(attack: str, xai: str) -> Path:
    """Reproduz exatamente o caminho que main.py monta (main.py:146-155)."""
    fname = f"{_dataset_name()}-{_llm_name()}-{attack}-{DEFENSE}-{xai}-{SEED}.json"
    return PIARENA_DIR / "results" / "evaluation_results" / NAME / fname


def metrics_path(result_file: Path) -> Path:
    """Default de --out-dir de scripts/xai_metrics.py: <result dir>/xai_metrics/."""
    return result_file.parent / "xai_metrics" / f"{result_file.stem}_metrics.json"


def compare_report_path(attack: str) -> Path:
    return ROOT / "results_comparison" / attack / "syntaxshap_vs_kernelshap_fidelity_comparison_report.md"


def _env() -> dict:
    env = os.environ.copy()
    # Offline by default (writes locally, no live network dependency during
    # the actual expensive run) — respects an already-exported WANDB_MODE
    # from the user's shell instead of overriding it. See wandb_sync_loop.sh
    # (spawned by main() below) for how the offline data reaches wandb.ai.
    env.setdefault("WANDB_MODE", "offline")
    return env


def run_logged(cmd: list[str], cwd: Path, log_name: str) -> bool:
    """Roda `cmd`, escrevendo stdout+stderr em logs/full_sweep/<log_name>.log
    em tempo real (dá pra `tail -f`). Retorna True se saiu com código 0."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name}.log"
    now = datetime.datetime.now()
    print(f"[{now:%H:%M:%S}] rodando: {' '.join(cmd)}  (cwd={cwd}, log={log_path})")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== {now.isoformat()} -- {' '.join(cmd)} ===\n")
        log_file.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, env=_env())
    ok = proc.returncode == 0
    status = "OK" if ok else f"FALHOU (exit {proc.returncode})"
    print(f"[{datetime.datetime.now():%H:%M:%S}] {status}: {log_name} -- ver {log_path}")
    return ok


def run_main(attack: str, xai: str, force: bool) -> Path | None:
    result_file = result_path(attack, xai)
    if not force and metrics_path(result_file).exists():
        print(f"pula main.py ({attack}/{xai}): metricas ja existem em {metrics_path(result_file)}")
        return result_file

    cmd = [
        sys.executable, "main.py",
        "--dataset", DATASET,
        "--backend_llm", BACKEND_LLM,
        "--attack", attack,
        "--defense", DEFENSE,
        "--xai", xai,
        "--name", NAME,
        "--seed", str(SEED),
    ]
    ok = run_logged(cmd, cwd=PIARENA_DIR, log_name=f"{attack}_{xai}_main")
    return result_file if ok else None


def run_metrics(attack: str, xai: str, result_file: Path, force: bool) -> Path | None:
    out_metrics = metrics_path(result_file)
    if not force and out_metrics.exists():
        print(f"pula xai_metrics.py ({attack}/{xai}): ja existe em {out_metrics}")
        return out_metrics

    # --xai-method não precisa ser passado -- scripts/xai_metrics.py infere
    # o método do próprio sufixo "-{xai}-" no nome do arquivo de resultado.
    cmd = [sys.executable, "scripts/xai_metrics.py", "--result", str(result_file)]
    ok = run_logged(cmd, cwd=PIARENA_DIR, log_name=f"{attack}_{xai}_metrics")
    return out_metrics if ok and out_metrics.exists() else None


def run_compare(attack: str, syntaxshap_metrics: Path, kernelshap_metrics: Path, force: bool) -> None:
    report = compare_report_path(attack)
    if not force and report.exists():
        print(f"pula xai_compare_fidelity.py ({attack}): ja existe em {report}")
        return

    cmd = [
        sys.executable, "scripts/xai_compare_fidelity.py",
        "--a-metrics", str(syntaxshap_metrics), "--a-label", "syntaxshap",
        "--b-metrics", str(kernelshap_metrics), "--b-label", "kernelshap",
        "--attack", attack,
        "--out-dir", str(report.parent),
    ]
    # xai_compare_fidelity.py é puro Python/numpy (sem torch/spacy) e vive em
    # PIArena-main/scripts/ -- roda a partir de lá.
    run_logged(cmd, cwd=PIARENA_DIR, log_name=f"{attack}_compare")


def process_attack(attack: str, force: bool, do_syntaxshap: bool, do_kernelshap: bool) -> None:
    print(f"\n===== {attack} =====")
    syntaxshap_metrics = kernelshap_metrics = None

    if do_syntaxshap:
        result_file = run_main(attack, "syntaxshap", force)
        if result_file is not None:
            syntaxshap_metrics = run_metrics(attack, "syntaxshap", result_file, force)
        else:
            print(f"{attack}/syntaxshap: main.py falhou, pulando metricas.")

    if do_kernelshap:
        result_file = run_main(attack, "kernelshap", force)
        if result_file is not None:
            kernelshap_metrics = run_metrics(attack, "kernelshap", result_file, force)
        else:
            print(f"{attack}/kernelshap: main.py falhou, pulando metricas.")

    if syntaxshap_metrics and kernelshap_metrics:
        run_compare(attack, syntaxshap_metrics, kernelshap_metrics, force)
    else:
        print(f"{attack}: pulando comparacao de Fidelity (falta metrica de algum dos dois metodos).")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attacks", default=",".join(ATTACKS),
                    help="Lista de ataques separada por virgula.")
    p.add_argument("--only-syntaxshap", action="store_true")
    p.add_argument("--only-kernelshap", action="store_true")
    p.add_argument("--force", action="store_true",
                    help="Roda de novo mesmo se a saida da etapa ja existir.")
    p.add_argument("--no-wandb-sync", action="store_true",
                    help="Nao sobe o wandb_sync_loop.sh em background (ele so importa se as "
                         "runs individuais tambem estiverem logando pro W&B -- --no-wandb nelas "
                         "ja deixa isso irrelevante).")
    return p.parse_args()


def _start_wandb_sync_loop() -> subprocess.Popen | None:
    """Sobe wandb_sync_loop.sh em background, sem bloquear -- assim o painel
    do wandb.ai fica quase ao vivo durante o sweep inteiro sem precisar de
    SSH nem de lembrar de rodar `wandb sync` manualmente (ver o script pra
    detalhes: um loop que roda `wandb sync --sync-all` dentro da árvore, a
    cada poucos minutos, `|| true` pra uma falha de sync nunca derrubar
    nada). Retorna None se o script nao existir (nao devia acontecer, mas
    nao trava o sweep por isso)."""
    script = ROOT / "wandb_sync_loop.sh"
    if not script.exists():
        print(f"aviso: {script} nao encontrado -- pulando o sync automatico do W&B.")
        return None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "wandb_sync.log"
    log_file = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(["bash", str(script)], cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"wandb_sync_loop.sh rodando em background (pid={proc.pid}, log={log_path}).")
    return proc


def main():
    args = parse_args()
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    do_syntaxshap = not args.only_kernelshap
    do_kernelshap = not args.only_syntaxshap

    sync_proc = None if args.no_wandb_sync else _start_wandb_sync_loop()
    try:
        for attack in attacks:
            process_attack(attack, args.force, do_syntaxshap, do_kernelshap)
    finally:
        if sync_proc is not None and sync_proc.poll() is None:
            sync_proc.terminate()
            print(f"wandb_sync_loop.sh (pid={sync_proc.pid}) encerrado.")


if __name__ == "__main__":
    main()
