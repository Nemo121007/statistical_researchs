from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from app.help_scripts.calculator_distances_length_large_circle import CalculatorDistancesLengthLargeCircle


class DataProcessor:
    """Класс для подготовки и обработки исходных данных."""

    LEN_LAT = 111132.0  # Длина одного градуса широты в метрах

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        """Загружает данные из CSV файла."""
        return pd.read_csv(path)

    def parse_intervals(self, df: pd.DataFrame, n: int = 300, distance_threshold: float = 500)\
            -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Разбивает DataFrame на списки валидных и невалидных интервалов.
        """
        list_valid_df = self._extend_intervals(df, target_point=1, n=n, distance_threshold=distance_threshold)
        list_invalid_df = self._extend_intervals(df, target_point=-1, n=n, distance_threshold=distance_threshold)
        return list_valid_df, list_invalid_df

    def _extend_intervals(self, df: pd.DataFrame, target_point: int = 1, n: int = 300,
                          distance_threshold: float = 500) -> List[pd.DataFrame]:
        """
        Внутренняя логика разбиения на интервалы с проверкой дистанции.
        """
        if df.empty:
            return []

        clean_df = df[df["validate_point"] == target_point].copy()
        if clean_df.empty:
            return []

        time_diff = clean_df["time"].diff()
        split_mask = time_diff > 10
        group_ids = split_mask.cumsum()

        result_list = []

        for _, group in clean_df.groupby(group_ids):
            if len(group) <= 1:
                continue

            # Логика обработки группы (разбиение на чанки и проверка дистанции)
            groups_to_process = []
            if len(group) > n:
                for i in range(0, len(group), n):
                    groups_to_process.append(group.iloc[i:i + n])
            else:
                groups_to_process.append(group)

            for chunk in groups_to_process:
                lon, lat = self.get_lon_lat(chunk)
                if len(lon) < 2:
                    continue

                # Проверка дистанции
                lon_ends = np.array([lon[0], lon[-1]])
                lat_ends = np.array([lat[0], lat[-1]])
                distance = float(
                    np.nansum(CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(lat_ends, lon_ends)))

                if distance < distance_threshold:
                    continue

                result_list.append(chunk)

        return result_list

    @staticmethod
    def get_lon_lat(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Извлекает массивы долгот и широт."""
        return df['lon'].to_numpy(), df['lat'].to_numpy()

    def convert_to_local_cartesian(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Переводит сферические координаты (lon, lat) в локальные прямоугольные (x, y) в метрах.
        """
        if len(lat) == 0:
            return np.array([]), np.array([])

        lat0 = lat[0]
        lon0 = lon[0]

        ky = self.LEN_LAT
        lat0_rad = np.radians(lat0)
        kx = self.LEN_LAT * np.cos(lat0_rad)

        x_local = (lon - lon0) * kx
        y_local = (lat - lat0) * ky

        return x_local, y_local

    @staticmethod
    def visualize_and_save(x_true: np.ndarray, y_true: np.ndarray,
                           x_filt: np.ndarray, y_filt: np.ndarray,
                           save_path: Path):
        """
        Строит и сохраняет график сравнения исходной и сглаженной траекторий.

        На графике отображаются:
        - Исходная траектория (синяя сплошная линия).
        - Сглаженная траектория (красная пунктирная линия).
        - Количество точек данных в заголовке.

        Args:
            x_true: Массив координат X исходной траектории (в метрах).
            y_true: Массив координат Y исходной траектории (в метрах).
            x_filt: Массив координат X сглаженной траектории (в метрах).
            y_filt: Массив координат Y сглаженной траектории (в метрах).
            save_path: Путь (объект pathlib.Path) для сохранения файла изображения.

        Returns:
            None. Функция сохраняет изображение на диск и закрывает фигуру.
        """
        num_points = len(x_true)

        plt.figure(figsize=(10, 8))
        plt.plot(x_true, y_true, c='blue', label=f'Исходные данные ({num_points} точек)', alpha=0.6)
        plt.plot(x_filt, y_filt, 'r--', label='Фильтр Калмана', linewidth=2)

        plt.xlabel('X (метры)')
        plt.ylabel('Y (метры)')
        plt.title(f'Траектория: {save_path.stem} | Точек: {num_points}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def process_track_list(df_list: List[pd.DataFrame],
                           save_dir: Path,
                           processor,  # DataProcessor
                           kalman_filter,  # Union[KalmanFilterCV, KalmanFilterRW]
                           label: str):
        """
        Обрабатывает список треков: фильтрует и сохраняет визуализацию.

        Args:
            df_list: Список DataFrame для обработки.
            save_dir: Папка для сохранения картинок.
            processor: Экземпляр класса DataProcessor.
            kalman_filter: Экземпляр класса фильтра (должен иметь метод filter).
            label: Название типа данных (для логирования).
        """
        print(f"Обработка {label}: {len(df_list)} шт.")

        for i, df in enumerate(df_list):
            lon, lat = processor.get_lon_lat(df)
            x, y = processor.convert_to_local_cartesian(lon, lat)
            time = df['time'].to_numpy()

            # Вызываем метод filter, который есть и у KalmanFilterCV, и у KalmanFilterRW
            x_filt, y_filt = kalman_filter.filter(x, y, time)

            save_path = save_dir / f'track_{i}.png'
            DataProcessor.visualize_and_save(x, y, x_filt, y_filt, save_path)

    @staticmethod
    def plot_array_and_hist(arr, bins=100, save_path: Path = None):
        """
        Рисует линейный график и гистограмму частот.
        Перед рисовкой выбросы обрезаются по правилу:
        всё, что меньше Q1 - IQR/2, становится равным Q1 - IQR/2.
        всё, что больше Q3 + IQR/2, становится равным Q3 + IQR/2.
        """
        # --- ЗАЩИТА ОТ NaN ---
        arr_np = np.asarray(arr).flatten()
        clean_arr = arr_np[~np.isnan(arr_np)]

        if len(clean_arr) == 0:
            print("Предупреждение: массив пуст или содержит только NaN. График не строится.")
            return

        # --- ПРЕОБРАЗОВАНИЕ ДАННЫХ (КЛИППИНГ) ---
        q1 = np.percentile(clean_arr, 25)  # 1 квартиль
        q3 = np.percentile(clean_arr, 75)  # 3 квартиль
        iqr = q3 - q1                       # Межквартильный размах (размах между 1 и 3 квартилем)

        lower_bound = q1 - (iqr / 2)
        upper_bound = q3 + (iqr / 2)

        # Принудительно ограничиваем значения
        corrected_arr = np.clip(clean_arr, lower_bound, upper_bound)

        # --- СОЗДАНИЕ ГРАФИКОВ ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Первый сабплот: Линейный график (из скорректированных данных) ---
        ax1.plot(corrected_arr, color='blue', linewidth=1.5)
        ax1.set_title('Линейный график (скорректированный)')
        ax1.set_xlabel('Индекс')
        ax1.set_ylabel('Значение')
        ax1.grid(True)

        # --- Второй сабплот: Гистограмма частот (из скорректированных данных) ---
        counts, bin_edges = np.histogram(corrected_arr, bins=bins)

        # Логика ограничения максимального класса (оставляем как было)
        if len(counts) > 1:
            sorted_counts = np.sort(counts)[::-1]
            max_count = sorted_counts[0]
            second_max_count = sorted_counts[1]

            if max_count > second_max_count + 1:
                target_height = second_max_count + 1
                counts[counts == max_count] = target_height
        elif len(counts) == 1 and counts[0] > 1:
            counts[0] = 1

        # Рисуем гистограмму вручную через bar
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + bin_width / 2

        ax2.bar(bin_centers, counts, width=bin_width, color='orange', edgecolor='black', alpha=0.7, align='center')

        ax2.set_title(f'Гистограмма частот RW (интервалы: {bins})\n(Макс. класс обрезан)', fontsize=10)
        ax2.set_xlabel('Значение')
        ax2.set_ylabel('Частота (скорректированная)')
        ax2.grid(True, linestyle='--', alpha=0.6, axis='y')

        plt.tight_layout()

        # --- Логика сохранения или отображения ---
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


if __name__ == '__main__':
    from app.working.kalman_filter_rw import KalmanFilterRW
    from app.working.kalman_filter_cv import KalmanFilterCV

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'data' / 'post_processing' / 'example.csv'

    processor = DataProcessor()
    kf = KalmanFilterCV(sigma_acc=0.0001 * 0.04, sigma_meas=1 * 2.4)

    df = processor.load_csv(data_path)
    df[['lon', 'lat']] = df[['lon', 'lat']].mask(df['satellites'] < 2, np.nan)

    print(len(df[df['validate_point'] == 1]))
    # Скобки обязательны! .notna() вместо != np.nan
    print(len(df[(df['validate_point'] == -1) & df['lon'].notna() & df['lat'].notna()]))

    chunk_size = 1000
    res_likelihood = []

    # Цикл разрезания изначального df на части по 1000 строк
    for i in range(0, len(df), chunk_size):
        # Берем сырой кусок
        chunk = df.iloc[i: i + chunk_size].copy()

        # Сбрасываем индексы, чтобы с нуля считать внутри куска (удобнее для срезов)
        chunk = chunk.reset_index(drop=True)

        # Ищем первый валидный индекс для lon и lat внутри ЭТОГО куска
        first_valid_lon = chunk['lon'].first_valid_index()
        first_valid_lat = chunk['lat'].first_valid_index()

        # Если хотя бы одна из колонок полностью пустая в этом куске — пропускаем
        if first_valid_lon is None or first_valid_lat is None:
            continue

        # Берем максимальный индекс (на случай если NaN в lon и lat начинаются не одновременно)
        start_idx = max(first_valid_lon, first_valid_lat)

        # Отрезаем "голову" с NaN
        clean_chunk = chunk.loc[start_idx:]

        # Если после отрезания осталось меньше 2 строк, фильтр не сработает
        if len(clean_chunk) < 2:
            continue

        # Каст к x/y
        lon, lat = processor.get_lon_lat(clean_chunk)
        x, y = processor.convert_to_local_cartesian(lon, lat)
        time = clean_chunk['time'].to_numpy()

        # Фильтр Калмана
        _, _, likelihood = kf.filter(x, y, time)

        # Сохраняем правдоподобие
        res_likelihood.append(float(np.nansum(likelihood)))

    save_path = project_root / 'CV_100'
    # Визуализация
    if res_likelihood:
        DataProcessor.plot_array_and_hist(res_likelihood, bins=100, save_path=None)
    else:
        print("Не собрано ни одного значения правдоподобия.")
