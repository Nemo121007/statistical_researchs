import datetime
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geomag.geomag import GeoMag
import math

from settings.settings import DefaultLocate

# Инициализируем модели один раз для эффективности
gm = GeoMag()
# Создаем один экземпляр geodesic для эффективности
g = geodesic()
# Вызываем measure один раз с фиктивными точками, чтобы инициализировать внутренний объект g.geod
g.measure((0, 0), (1, 1))


def main():
    path_file = DefaultLocate.DATA_PREPROCESSED_DIR
    files = [file.name for file in path_file.glob('*.csv')]

    results_list = []

    for file in files:
        print(f"Processing file: {path_file / file}")
        df = pd.read_csv(path_file / file)

        # Указываем дату для расчета магнитного склонения
        data = datetime.datetime(year=2025, month=6, day=1)

        # Предполагаем, что в df есть колонка 'heading' с истинным курсом
        if 'heading' not in df.columns:
            print(f"Пропускаем файл {file}, так как отсутствует колонка 'heading'.")
            continue

        # --- РАСЧЕТ АЗИМУТОВ ---

        # Создаем пары последовательных точек
        lat1 = df['lat'].shift(1)
        lon1 = df['lon'].shift(1)
        lat2 = df['lat']
        lon2 = df['lon']

        # Создаем DataFrame для итерации, удаляя первую строку без предыдущей точки
        pairs_df = pd.DataFrame({
            'lat1': lat1[1:], 'lon1': lon1[1:],
            'lat2': lat2[1:], 'lon2': lon2[1:]
        }).dropna()  # Удаляем строки, где shift() мог создать NaN

        geographic_azimuths = []
        magnetic_azimuths = []

        # Итерируемся по парам точек для расчета азимутов
        for index, row in pairs_df.iterrows():
            # Проверка на NaN уже есть благодаря dropna() выше, но оставим для надежности
            if pd.isna(row[['lat1', 'lon1', 'lat2', 'lon2']]).any():
                geographic_azimuths.append(np.nan)
                magnetic_azimuths.append(np.nan)
                continue

            # 1. Расчет географического азимута
            inverse_result = g.geod.Inverse(row['lat1'], row['lon1'], row['lat2'], row['lon2'])
            # Нормализуем азимут к диапазону [0, 360)
            geo_azimuth = (inverse_result['azi1'] + 360) % 360
            geographic_azimuths.append(geo_azimuth)

            # 2. Расчет магнитного азимута
            try:
                mag_result = gm.GeoMag(dlat=row['lat1'], dlon=row['lon1'], h=0, time=data.date())
                declination = mag_result.dec
                magnetic_azimuth = (geo_azimuth - declination + 360) % 360
                magnetic_azimuths.append(magnetic_azimuth)
            except Exception as e:
                print(f"Не удалось рассчитать магнитное склонение для точки ({row['lat1']}, {row['lon1']}): {e}")
                magnetic_azimuths.append(np.nan)

        # --- ПРИСВАИВАНИЕ И РАСЧЕТ СТАТИСТИК ---

        # Создаем временный DataFrame для удобства
        # Индекс будет совпадать с исходным df, т.к. pairs_df был создан из него
        temp_azimuth_df = pd.DataFrame({
            'geographic_azimuth': geographic_azimuths,
            'magnetic_azimuth': magnetic_azimuths
        })
        temp_azimuth_df.index = pairs_df.index
        # Объединяем с исходным df для расчета разниц
        # Используем 'left' join, чтобы сохранить все строки из df
        df = df.join(temp_azimuth_df, how='left')

        # Вычисляем разницу с истинным курсом
        # df['geo_diff'] = df['heading'] - df['geographic_azimuth']
        # df['mag_diff'] = df['heading'] - df['magnetic_azimuth']
        df['geo_diff'] = (df['heading'] - df['geographic_azimuth'] + 180) % 360 - 180
        df['mag_diff'] = (df['heading'] - df['magnetic_azimuth'] + 180) % 360 - 180

        # Создаем DataFrame только с колонками для статистики и удаляем строки с NaN
        stats_df = df[['geo_diff', 'mag_diff']].dropna()

        # Сохраняем количество обработанных (валидных) точек
        valid_points_count = len(stats_df)

        # Проверяем, есть ли валидные данные для расчета статистик
        if valid_points_count == 0:
            print(f"Пропускаем файл {file}, так как нет валидных точек для анализа.")
            continue

        # --- ВЫЧИСЛЕНИЕ СТАТИСТИК ---
        temp_result = {'filename': file}
        temp_result['valid_points_count'] = valid_points_count

        for col_name in ['geo_diff', 'mag_diff']:
            prefix = 'geographic' if 'geo' in col_name else 'magnetic'

            temp_result[f'{prefix}_diff_mean'] = stats_df[col_name].mean()
            temp_result[f'{prefix}_diff_min'] = stats_df[col_name].min()
            temp_result[f'{prefix}_diff_max'] = stats_df[col_name].max()

            for p in [25, 50, 75, 90, 95]:
                temp_result[f'{prefix}_diff_{p}p'] = stats_df[col_name].quantile(p / 100.0)

        print(f"  - Результаты для {file}: {temp_result}")
        print(df.head())
        results_list.append(temp_result)
        path = DefaultLocate.DATA_TEMP_DIR / f"course_{file}"
        df.to_csv(path)

    # --- Финальное сохранение ---
    if results_list:
        result_df = pd.DataFrame(results_list)
        print("\n--- Итоговые результаты по разнице курсов ---")
        # Устанавливаем 'filename' как индекс для лучшей читаемости
        result_df.set_index('filename', inplace=True)
        print(result_df)

        path = DefaultLocate.DATA_DIR / "results_course_comparison.csv"
        result_df.to_csv(path)
        print(f"\nРезультаты сохранены в файл: {path}")
    else:
        print("Нет файлов для обработки или не удалось получить результаты.")


if __name__ == '__main__':
    main()