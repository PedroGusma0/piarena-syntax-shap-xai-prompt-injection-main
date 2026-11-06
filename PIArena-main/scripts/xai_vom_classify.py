"""Classifica V/O/M (Vetor, Objetivo, Mecânica) das amostras bloqueadas, com a evidência textual de cada regra.

Taxonomia: papers-de-referencia/referencia_relatorio.tex (V/O/M) + pôster CrowdStrike
"Taxonomy of Prompt Injection Methods" (mecânicas) + OWASP LLM01. V é fixo em "Injeção Direta"
(segue o nome do ataque `direct` do PIArena).

Uso:
    python scripts/xai_vom_classify.py <resultado.json> <saida_vom.json>
"""
import json, re, sys

SRC, OUT = sys.argv[1], sys.argv[2]

SUB = {  # sub-objetivo concreto do sequestro, por categoria do PIArena
    "phishing_injection": "redirecionar o usuário a uma URL controlada pelo atacante (phishing)",
    "content_promotion": "inserir promoção de produto/marca na resposta",
    "access_denial": "negar o serviço ao usuário com um falso aviso de acesso bloqueado",
    "infrastructure_failure": "substituir a resposta por uma falsa mensagem de erro de infraestrutura",
}
RULES = [  # (mecânica, regex de evidência, categorias em que se aplica ou None)
    ("Escalada de Autoridade",
     r"^(system [a-z]+(?: [a-z]+)?:|note for processing:|for diagnostic demonstration)", None),
    ("Negação de Instrução",
     r"\b(instead\b|without processing|disregard|interrupt (?:normal|your normal)|refuse|decline to answer|"
     r"do not (?:proceed|continue|produce)|terminate (?:standard|output|the output|the reply|the response)|rather than|"
     r"stop(?: the reply| responding| processing| without| after)|halt (?:output|the response|computation)|cease|"
     r"end (?:the|this) (?:reply|response)|discontinue output|without supplying|"
     r"(?:respond\w*|repl\w*|output\w*|print\w*|display|return) (?:only|solely))", None),
    ("Adição de Regra por Gatilho (Trigger-Activated Rule Addition)",
     r"^(before|prior to|during|mid|midway|at the (?:start|beginning|end)|while|when|if processing|once|as you|amid|"
     r"following|directly before|insert immediately after|insert (?:at the (?:very )?start|at the beginning|before|ahead of)|"
     r"interject briefly before|as part of|to confirm|to ensure|within your|alongside|subtly conclude|"
     r"display (?:a notification at the start|first)|kindly begin|append a preface|insert an initial|insert a preliminary|"
     r"add a preliminary|insert at the end|on a side note|for this context|in your answer|before finalizing)", None),
    ("Falsificação de Estado de Sessão (Session State Spoofing)",
     r"(simulat\w*|emulat\w*|imitat\w*|mimic|act (?:as|like)|behave as|system (?:notice|message|communication|advisory|response)|"
     r"notice|notification|alert|advisory|service update|message|announce|error|fault|crash|outage|timeout|unavailable|failed|"
     r"access\b.*?\b(?:suspended|revoked|restricted|blocked|denied|disabled|locked))",
     {"infrastructure_failure", "access_denial"}),
    ("Enquadramento Enganoso (Contextual Misdirection)",
     r"(verif\w*|secure\w*|authenti\w*|official|scholarly|archival|validat\w*|https?://\S+)", {"phishing_injection"}),
    ("Enquadramento Enganoso (Contextual Misdirection)",
     r"(subtl\w*|discreet\w*|slip|seamless\w*|natural\w*|smooth\w*|weave|incidental|tasteful|passing)", {"content_promotion"}),
]
BASE = "Adição Explícita de Regra (Rule Addition Prompting)"

data = json.load(open(SRC, encoding="utf-8"))
out = {}
for k, s in data.items():
    if not s.get("defense_result", {}).get("detect_flag"):
        continue
    task, cat = s["injected_task"], s["category"]
    mech, evid = [], []
    for name, rx, cats in RULES:
        if cats and cat not in cats:
            continue
        m = re.search(rx, task, flags=re.I)
        if m and name not in mech:
            mech.append(name)
            evid.append(f'{name.split(" (")[0]}: "{m.group(0).strip()}"')
    mech = mech or [BASE]
    out[k] = {
        "V": "Injeção Direta",
        "O": "Sequestro de Objetivo (Goal Hijacking)",
        "O_detalhe": SUB[cat],
        "M": mech,
        "evidencia": evid or ["instrução explícita nova, sem gatilho/negação/autoridade"],
        "categoria": cat,
        "injected_task": task,
    }
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print(len(out)); print(Counter(m for v in out.values() for m in v["M"]))
