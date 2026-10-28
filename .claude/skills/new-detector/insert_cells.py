"""Вставка блока нового детектора (markdown + code) в большой блокнот и регистрация в общей сводке.

Блокнот переписывается тем же форматированием (отступ 1, CRLF, без экранирования не-ASCII). Перед записью
скрипт проверяет, что исходный файл при таком форматировании воспроизводится побайтно; иначе прерывается,
чтобы не переформатировать блокнот целиком.

Пример (из корня проекта):
    .venv/bin/python .claude/skills/new-detector/insert_cells.py notebooks/example_model.ipynb \\
        --slug lof2 --title "LOF-2" --markdown /tmp/.../lof2.md --code /tmp/.../lof2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BEFORE_ID = "master-summary-md"
CATALOG_FUNCTION = "def _model_summary_catalog"
CATALOG_END = "    ]\n"


def _dump(notebook: dict, newline: str) -> str:
    """Сериализует блокнот в формате исходного файла."""
    return json.dumps(notebook, indent=1, ensure_ascii=False).replace("\n", newline) + newline


def _source_lines(text: str) -> list[str]:
    """Разбивает текст на строки ячейки в формате nbformat (с ``\\n`` у всех, кроме последней)."""
    lines = text.rstrip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _register_in_catalog(cells: list[dict], title: str, slug: str) -> None:
    """Добавляет строку модели в конец списка ``_model_summary_catalog``.

    Args:
        cells: Ячейки блокнота.
        title: Название модели в сводке.
        slug: Префикс переменных модели; сводка — ``<slug>_summary``.

    Raises:
        ValueError: Нет ячейки каталога или модель уже зарегистрирована.
    """
    for cell in cells:
        source = cell["source"]
        if cell["cell_type"] != "code" or not any(line.startswith(CATALOG_FUNCTION) for line in source):
            continue
        entry = f'        ("{title}", "{slug}_summary"),\n'
        if entry in source:
            raise ValueError(f"Модель {title} уже есть в _model_summary_catalog")
        start = next(i for i, line in enumerate(source) if line.startswith(CATALOG_FUNCTION))
        end = source.index(CATALOG_END, start)
        source.insert(end, entry)
        return
    raise ValueError("Не найдена ячейка с _model_summary_catalog")


def insert_detector(notebook_path: Path, slug: str, title: str, markdown: str, code: str) -> None:
    """Вставляет ячейки детектора перед общей сводкой и регистрирует модель в каталоге сводки.

    Args:
        notebook_path: Путь к ``.ipynb``.
        slug: Префикс переменных и id ячеек (``<slug>-theory-md``, ``<slug>-code-cell``).
        title: Название модели в общей сводке.
        markdown: Текст markdown-ячейки.
        code: Текст code-ячейки.

    Raises:
        ValueError: Формат файла не воспроизводится, id уже заняты или нет опорной ячейки.
    """
    raw = notebook_path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    notebook = json.loads(raw)
    if _dump(notebook, newline) not in (raw, raw + newline):
        raise ValueError("Блокнот не воспроизводится побайтно: запись переформатировала бы весь файл")
    trailing = raw.endswith(newline)

    cells = notebook["cells"]
    ids = {cell.get("id") for cell in cells}
    md_id, code_id = f"{slug}-theory-md", f"{slug}-code-cell"
    if md_id in ids or code_id in ids:
        raise ValueError(f"Ячейки {md_id} / {code_id} уже есть в блокноте")
    position = next((i for i, cell in enumerate(cells) if cell.get("id") == BEFORE_ID), None)
    if position is None:
        raise ValueError(f"Не найдена опорная ячейка {BEFORE_ID}")

    _register_in_catalog(cells, title, slug)
    cells[position:position] = [
        {"cell_type": "markdown", "id": md_id, "metadata": {}, "source": _source_lines(markdown)},
        {
            "cell_type": "code",
            "id": code_id,
            "metadata": {},
            "source": _source_lines(code),
            "outputs": [],
            "execution_count": None,
        },
    ]
    text = _dump(notebook, newline)
    notebook_path.write_bytes((text if trailing else text[: -len(newline)]).encode("utf-8"))


def main() -> int:
    """Разбирает аргументы командной строки и вставляет блок детектора."""
    parser = argparse.ArgumentParser(description="Вставка блока детектора в блокнот.")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--slug", required=True, help="Префикс переменных и id ячеек, например lof2")
    parser.add_argument("--title", required=True, help="Название модели в общей сводке")
    parser.add_argument("--markdown", required=True, type=Path, help="Файл с текстом markdown-ячейки")
    parser.add_argument("--code", required=True, type=Path, help="Файл с текстом code-ячейки")
    args = parser.parse_args()
    try:
        insert_detector(
            args.notebook,
            args.slug,
            args.title,
            args.markdown.read_text(encoding="utf-8"),
            args.code.read_text(encoding="utf-8"),
        )
    except ValueError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(
        f"Вставлены ячейки {args.slug}-theory-md и {args.slug}-code-cell перед {BEFORE_ID}; "
        f"модель «{args.title}» добавлена в сводку"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
