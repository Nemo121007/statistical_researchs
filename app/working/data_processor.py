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

            lons = group['lon'].to_numpy()
            lats = group['lat'].to_numpy()
            
            chunk_start_idx = 0
            for i in range(1, len(group)):
                lon_ends = np.array([lons[chunk_start_idx], lons[i]])
                lat_ends = np.array([lats[chunk_start_idx], lats[i]])
                distance = float(np.nansum(CalculatorDistancesLengthLargeCircle.vectorized_segment_distances(lat_ends, lon_ends)))
                
                if (i - chunk_start_idx + 1) >= n or distance >= distance_threshold:
                    result_list.append(group.iloc[chunk_start_idx:i + 1])
                    chunk_start_idx = i + 1

            if chunk_start_idx < len(group) - 1:
                result_list.append(group.iloc[chunk_start_idx:])

        return result_list

    @staticmethod
    def get_lon_lat(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Извлекает массивы долгот, широт и времени"""
        return df['lon'].to_numpy(), df['lat'].to_numpy(), df['time'].to_numpy()

    def convert_to_local_cartesian(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Переводит сферические координаты (lon, lat) в локальные прямоугольные (x, y) в метрах.
        """
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        
        if len(lat) == 0:
            return np.array([]), np.array([])

        valid_mask = ~np.isnan(lat) & ~np.isnan(lon)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return np.full_like(lon, np.nan, dtype=float), np.full_like(lat, np.nan, dtype=float)

        first_valid_idx = valid_indices[0]
        lat0 = lat[first_valid_idx]
        lon0 = lon[first_valid_idx]

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
        # --- ЗАЩИТА ОТ NaN ---
        arr_np = np.asarray(arr).flatten()
        clean_arr = arr_np[~np.isnan(arr_np)]

        if len(clean_arr) == 0:
            print("Предупреждение: массив пуст или содержит только NaN. График не строится.")
            return

        # --- СОЗДАНИЕ ГРАФИКОВ ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'График и гистограмма для {save_path.name if save_path else "данных"}\n'
                     f'Количество измерений: {len(clean_arr)}', fontsize=14)

        # --- Первый сабплот: Линейный график (из скорректированных данных) ---
        ax1.plot(clean_arr, linewidth=1.5)
        ax1.set_title('Линейный график (скорректированный)')
        ax1.set_xlabel('Индекс')
        ax1.set_ylabel('Значение')
        ax1.grid(True)

        # --- Второй сабплот: Гистограмма частот (из скорректированных данных) ---
        ax2.hist(clean_arr, bins=bins, edgecolor='black', alpha=0.7)
        ax2.set_title(f'Гистограмма частот RW (интервалы: {bins})', fontsize=10)
        ax2.set_xlabel('Значение')
        ax2.set_ylabel('Частота')
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
        lon, lat, time = processor.get_lon_lat(clean_chunk)
        x, y = processor.convert_to_local_cartesian(lon, lat)

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
