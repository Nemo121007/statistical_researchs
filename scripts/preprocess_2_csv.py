"""Предобработка data/2.csv: фильтрация координат и пометка аномалий по sat и (0, 0)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STATUS_ANOMALY = "anomaly"
MIN_SAT = 3


def find_project_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data" / "2.csv").exists() and (candidate / "app").exists():
            return candidate
    raise FileNotFoundError("Не найден корень проекта (ожидаются data/2.csv и app/).")


def preprocess_track_csv(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    rows_before = len(df)

    has_coords = df["lat"].notna() & df["lon"].notna()
    dropped_no_coords = int((~has_coords).sum())
    df = df.loc[has_coords].copy()

    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    sat = pd.to_numeric(df["sat"], errors="coerce")

    zero_coords = (lat == 0.0) & (lon == 0.0)
    low_sat = sat < MIN_SAT
    mark_anomaly = zero_coords | low_sat

    marked_zero = int(zero_coords.sum())
    marked_low_sat = int((low_sat & ~zero_coords).sum())
    df.loc[mark_anomaly, "status"] = STATUS_ANOMALY

    if "is_water" in df.columns:
        truthy = {True, "true", "True", "1", 1}
        df["is_water"] = df["is_water"].apply(lambda value: "true" if value in truthy else "false")

    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

    return {
        "rows_before": rows_before,
        "rows_after": len(df),
        "dropped_no_coords": dropped_no_coords,
        "marked_zero_coords": marked_zero,
        "marked_low_sat": marked_low_sat,
        "marked_anomaly_total": int(mark_anomaly.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Предобработка трека CSV (по умолчанию data/2.csv).")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Путь к CSV (по умолчанию <корень проекта>/data/2.csv).",
    )
    args = parser.parse_args()
    path = args.path or (find_project_root() / "data" / "2.csv")
    if not path.is_file():
        raise FileNotFoundError(path)

    stats = preprocess_track_csv(path)
    print(f"Файл: {path}")
    print(f"Строк до: {stats['rows_before']}, после: {stats['rows_after']}")
    print(f"Удалено без lat/lon: {stats['dropped_no_coords']}")
    print(f"Помечено anomaly (0, 0): {stats['marked_zero_coords']}")
    print(f"Помечено anomaly (sat < {MIN_SAT}): {stats['marked_low_sat']}")
    print(f"Всего помечено anomaly (условия 2–3): {stats['marked_anomaly_total']}")


if __name__ == "__main__":
    main()
