"""Прогон блокнота в памяти через IPython без сохранения выходов в файл.

Запуск из каталога notebooks/:
    MPLBACKEND=Agg ../.venv/bin/python ../.claude/skills/run-notebook/run_notebook.py example_model.ipynb
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

HEAD_CHARS = 60
# Скрипт лежит в <корень>/.claude/skills/run-notebook/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _cell_head(source: str) -> str:
    """Первая непустая строка кода ячейки, обрезанная до ``HEAD_CHARS`` символов."""
    for line in source.splitlines():
        if line.strip():
            return line.strip()[:HEAD_CHARS]
    return ""


def _make_shell(project_root: Path) -> InteractiveShell:
    """Создаёт оболочку IPython для прогона без GUI и с корнем проекта в ``sys.path``.

    Args:
        project_root: Корень проекта, из которого импортируется пакет app.

    Returns:
        Настроенная оболочка.
    """
    sys.path.insert(0, str(project_root))
    shell = InteractiveShell.instance()
    # Голый InteractiveShell не умеет переключать GUI: без заглушки %matplotlib inline обрывает ячейку импортов.
    shell.enable_gui = lambda gui=None: None
    shell.run_line_magic("colors", "NoColor")
    shell.run_line_magic("xmode", "Plain")
    return shell


def run_notebook(notebook_path: Path, stop_on_error: bool) -> list[int]:
    """Выполняет code-ячейки блокнота по порядку и печатает статус каждой.

    Args:
        notebook_path: Путь к ``.ipynb``.
        stop_on_error: Остановиться на первой упавшей ячейке.

    Returns:
        Индексы упавших ячеек.
    """
    cells = json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
    shell = _make_shell(PROJECT_ROOT)
    failed: list[int] = []
    started = time.perf_counter()
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        cell_started = time.perf_counter()
        result = shell.run_cell(source, silent=False)
        elapsed = time.perf_counter() - cell_started
        status = "ok" if result.success else "FAIL"
        print(f"######## cell {index}: {elapsed:.1f} s, {status} | {_cell_head(source)}", flush=True)
        if result.success:
            continue
        failed.append(index)
        error = result.error_in_exec or result.error_before_exec
        print(f"######## cell {index} error: {type(error).__name__}: {error}", flush=True)
        if stop_on_error:
            break
    print(f"\nTOTAL {time.perf_counter() - started:.0f} s, failed cells: {failed}", flush=True)
    return failed


def main() -> int:
    """Разбирает аргументы командной строки и запускает прогон."""
    parser = argparse.ArgumentParser(description="Прогон блокнота в памяти без сохранения выходов.")
    parser.add_argument("notebook", type=Path, help="Путь к .ipynb")
    parser.add_argument("--stop-on-error", action="store_true", help="Остановиться на первой упавшей ячейке")
    args = parser.parse_args()
    if not args.notebook.is_file():
        print(f"Нет файла {args.notebook}", file=sys.stderr)
        return 2
    return 1 if run_notebook(args.notebook, args.stop_on_error) else 0


if __name__ == "__main__":
    sys.exit(main())
