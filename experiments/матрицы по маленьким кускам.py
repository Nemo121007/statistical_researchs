from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pykalman import KalmanFilter
from tqdm import tqdm  # Импорт для прогресс-бара

from help_scripts.IOPs_geojson import IOPs_geojson
from settings.settings import DefaultLocate


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def discretize_df(df: pd.DataFrame, max_len_part: int) -> List[pd.DataFrame]:
    """
    Разделяет DataFrame на список DataFrame на основе столбца validate_point и максимальной длины.
    """
    if df.empty:
        return []

    # --- Шаг 1: Определение групп по validate_point ---
    valid_groups = df['validate_point'].replace(0, np.nan)
    valid_groups = valid_groups.ffill()
    valid_groups = valid_groups.fillna(0)

    mask = (valid_groups != valid_groups.shift(1))
    group_ids = mask.cumsum()

    # --- Шаг 2: Разбиение групп с учетом размера max_len_part ---
    df_list = []
    for _, group_df in df.groupby(group_ids):
        if len(group_df) <= max_len_part:
            df_list.append(group_df.reset_index(drop=True))
        else:
            for i in range(0, len(group_df), max_len_part):
                chunk = group_df.iloc[i:i + max_len_part]
                df_list.append(chunk.reset_index(drop=True))

    return df_list


def run_kalman_filter(data: np.ndarray, n_iter: int = 5):
    """
    Применяет EM-алгоритм и фильтр Калмана к данным.
    Обрабатывает NaN и Inf значения.
    """
    # 1. Проверка на наличие данных
    if len(data) == 0:
        return None, None, None, None

    # 2. Обработка NaN и Inf
    # Заменяем все inf на nan, чтобы pykalman мог их обработать как пропуски
    # Работаем с копией, чтобы не менять исходный массив (важно для графиков)
    data_clean = data.copy()
    data_clean[~np.isfinite(data_clean)] = np.nan

    # Ищем строки, где нет NaN (после очистки это значит, что точка валидна)
    valid_mask = ~np.isnan(data_clean).any(axis=1)
    valid_indices = np.where(valid_mask)[0]

    # Если в массиве вообще нет валидных точек
    if len(valid_indices) == 0:
        return None, None, None, None

    # Находим первое валидное значение для инициализации начального состояния
    first_valid_idx = valid_indices[0]
    initial_state = data_clean[first_valid_idx]

    # Инициализация параметров
    A_init = np.eye(2)
    Q_init = np.eye(2) * 0.1
    R_init = np.eye(2) * 0.1

    # Инициализация модели
    kf = KalmanFilter(
        n_dim_obs=2,
        n_dim_state=2,
        initial_state_mean=initial_state,
        initial_state_covariance=np.eye(2) * 0.1,
        transition_matrices=A_init,
        transition_covariance=Q_init,
        observation_matrices=np.eye(2),
        observation_covariance=R_init
    )

    # EM-алгоритм (передаем очищенные данные, где inf заменены на nan)
    kf = kf.em(data_clean, n_iter=n_iter)

    # Сглаживание
    smoothed_state_means, _ = kf.smooth(data_clean)

    return kf.transition_matrices, kf.transition_covariance, kf.observation_covariance, smoothed_state_means


def plot_results(smoothed_state_means: np.ndarray, data: np.ndarray, name_file: Path):
    """
    Строит и сохраняет график сравнения исходных и сглаженных данных.
    """
    plt.figure(figsize=(10, 6))

    valid_idx = ~np.isnan(data).any(axis=1)
    if np.any(valid_idx):
        plt.plot(data[valid_idx, 0], data[valid_idx, 1], 'ro', label='Наблюдения', alpha=0.5)

    plt.plot(smoothed_state_means[:, 0], smoothed_state_means[:, 1], 'b-', label='Сглаженные состояния')

    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.title('Сравнение измерений и сглаженной траектории')
    plt.legend()
    plt.grid()

    name_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(name_file)
    plt.close()


def calculate_matrix_stats(matrix_list: List[np.ndarray], name: str) -> str:
    """
    Вычисляет статистику для списка матриц и формирует строку отчета.
    """
    if not matrix_list:
        return f"--- {name} ---\nНет данных для расчета.\n"

    stack = np.stack(matrix_list)
    report_lines = [f"--- Статистика для {name} ---"]
    rows, cols = stack[0].shape

    for r in range(rows):
        for c in range(cols):
            elements = stack[:, r, c]
            mean_val = np.mean(elements)
            min_val = np.min(elements)
            max_val = np.max(elements)
            var_val = np.var(elements)

            report_lines.append(
                f"Элемент [{r},{c}]: Mean={mean_val:.6f}, Min={min_val:.6f}, Max={max_val:.6f}, Var={var_val:.6f}"
            )
    report_lines.append("")
    return "\n".join(report_lines)


def test_calman_filter(list_df: List[pd.DataFrame]):
    """
    Основная функция тестирования фильтра Калмана с прогресс-баром.
    """
    # 1. Инициализация логов
    log_true_mats = {'A': [], 'Q': [], 'R': []}
    log_false_mats = {'A': [], 'Q': [], 'R': []}

    text_log_true = ["Лог TRUE значений (validate_point = 1)"]
    text_log_false = ["Лог FALSE значений (validate_point = -1)"]

    counters = {'true': 0, 'false': 0}

    pict_dir = DefaultLocate.DATA_DIR / "pict"
    pict_dir.mkdir(parents=True, exist_ok=True)

    # 2. Цикл по элементам с прогресс-баром
    # Добавляем tqdm для визуализации
    pbar = tqdm(enumerate(list_df), total=len(list_df), desc="Анализ чанков", unit="chunk")

    for i, df in pbar:
        if df.empty:
            continue

        val_point = df['validate_point'].iloc[0]

        if val_point == 1:
            log_type = 'true'
            counters['true'] += 1
            current_idx = counters['true']
            target_mats = log_true_mats
            target_text = text_log_true
        elif val_point == -1:
            log_type = 'false'
            counters['false'] += 1
            current_idx = counters['false']
            target_mats = log_false_mats
            target_text = text_log_false
        else:
            continue

        # Обновляем описание прогресс-бара динамически
        pbar.set_postfix_str(f"Type={log_type}, Idx={current_idx}")

        data = df[['lon', 'lat']].values

        try:
            A_est, Q_est, R_est, smoothed = run_kalman_filter(data)

            if A_est is None:
                continue

            target_mats['A'].append(A_est)
            target_mats['Q'].append(Q_est)
            target_mats['R'].append(R_est)

            entry = (
                f"\n--- Запись #{current_idx} ---\n"
                f"A_estimated:\n{A_est}\n"
                f"Q_estimated:\n{Q_est}\n"
                f"R_estimated:\n{R_est}\n"
            )
            target_text.append(entry)

            plot_filename = pict_dir / f"{log_type}_{current_idx}.png"
            plot_results(smoothed, data, plot_filename)

        except Exception as e:
            # Используем pbar.write чтобы не ломать вывод прогресс-бара
            pbar.write(f"Ошибка при обработке chunk #{i} ({log_type}): {e}")
            continue

    # 6. Расчет статистики и сохранение логов
    stats_true = "\n".join([
        calculate_matrix_stats(log_true_mats['A'], "Matrix A (Transition)"),
        calculate_matrix_stats(log_true_mats['Q'], "Matrix Q (Transition Cov)"),
        calculate_matrix_stats(log_true_mats['R'], "Matrix R (Observation Cov)")
    ])
    text_log_true.append("\n\n=== ИТОГОВАЯ СТАТИСТИКА ===\n" + stats_true)

    stats_false = "\n".join([
        calculate_matrix_stats(log_false_mats['A'], "Matrix A (Transition)"),
        calculate_matrix_stats(log_false_mats['Q'], "Matrix Q (Transition Cov)"),
        calculate_matrix_stats(log_false_mats['R'], "Matrix R (Observation Cov)")
    ])
    text_log_false.append("\n\n=== ИТОГОВАЯ СТАТИСТИКА ===\n" + stats_false)

    save_dir = DefaultLocate.DATA_DIR

    with open(save_dir / "log_true.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(text_log_true))

    with open(save_dir / "log_false.txt", "w", encoding='utf-8') as f:
        f.write("\n".join(text_log_false))

    print(f"\nАнализ завершен. Логи сохранены в: {save_dir}")
    print(f"Графики сохранены в: {pict_dir}")


if __name__ == '__main__':
    path = DefaultLocate.DATA_POSTPROCESSED_DIR / "example.csv"

    if not path.exists():
        print(f"Файл не найден: {path}")
    else:
        df = load_csv(path)
        df_list = discretize_df(df, max_len_part=100)

        print(f"Всего частей: {len(df_list)}")
        if df_list:
            print(f"Размер первой части: {len(df_list[0])}")
            max_size = max(len(p) for p in df_list)
            print(f"Максимальный размер части: {max_size}")

            test_calman_filter(df_list)