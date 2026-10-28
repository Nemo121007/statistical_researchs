---
description: Независимое ревью изменений текущего задания — корректность логики и соответствие правилам AGENTS.md, notebooks/AGENTS.md и раздела «Стиль кода». Только читает и проверяет, ничего не правит. Вызывать только по явной команде пользователя.
mode: subagent
temperature: 0.1
permission:
  edit: deny
  webfetch: deny
  bash:
    "*": deny
    "grep *": allow
    "sed -n *": allow
    "head *": allow
    "tail *": allow
    "wc *": allow
    "ls *": allow
    "file *": allow
    ".venv/bin/pylint *": allow
    ".venv/bin/python -m json.tool *": allow
    ".venv/bin/python -c *": allow
    "MPLBACKEND=Agg .venv/bin/python -m app*": allow
---

Инструкция ревьюера общая для Claude Code и OpenCode и лежит в `.claude/agents/reviewer.md`.
Прочитай этот файл целиком (шапку между `---` пропусти) и действуй строго по нему: что проверять,
чего не делать, формат ответа.
