import json
import logging
from pathlib import Path

import geojson
import psycopg2
from application.modules.bg_services.gps.core import Config

logging.basicConfig(level=logging.DEBUG)
# Параметры подключения к БД
config = Config.load()

DB_PARAMS = {
    "host": config.sync_db_settings.dsn.host,
    "port": config.sync_db_settings.dsn.port,
    "database": config.sync_db_settings.dsn.database,
    "user": config.sync_db_settings.dsn.user,
    "password": config.sync_db_settings.dsn.password,
}

def process_geojson(file_path):
    # Чтение GeoJSON файла
    with open(file_path, 'r', encoding='utf-8') as f:
        geojson_data = geojson.load(f)

    # Подключение к PostgreSQL
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()

    try:
        for feature in geojson_data.features:
            id_feature = feature['id'] or 0
            if id_feature < 5004:
                continue
            geom_type = feature['geometry']['type']
            properties = feature.get('properties', {})
            coordinates = feature['geometry']['coordinates']
            min_lon, min_lat, max_lon, max_lat = feature['boundingbox']
            rt_lat = max_lat
            rt_lon = max_lon
            lb_lat = min_lat
            lb_lon = min_lon
            name = properties.get('tags', {}).get('name', "Unnamed")

            if geom_type == 'Polygon':
                # Преобразование координат в WKT
                rings = []
                for ring in coordinates:
                    coords_str = ', '.join([f"{lon} {lat}" for lon, lat in ring])
                    rings.append(f"({coords_str})")
                wkt = f"POLYGON({', '.join(rings)})"
            elif geom_type == 'LineString':
                # Преобразование координат в WKT
                coords_str = ', '.join([f"{lon} {lat}" for lon, lat in coordinates])
                wkt = f"LINESTRING({coords_str})"

            # Начинаем транзакцию
            cursor.execute("BEGIN;")

            # Вставляем в vi_geofence
            cursor.execute(
                """
                INSERT INTO route_corrector.vi_geofence (name, rt_lat, rt_lon, lb_lat, lb_lon, sr_parent_id)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (name, rt_lat, rt_lon, lb_lat, lb_lon, None)
            )
            geofence_id = cursor.fetchone()[0]

            # Вставляем геометрию
            cursor.execute(
                """
                INSERT INTO route_corrector.vi_geofence_geometry (vi_geofence_id, geom)
                VALUES (%s, ST_GeomFromText(%s, 4326))
                """,
                (geofence_id, wkt)
            )

            # Фиксируем транзакцию
            cursor.execute("COMMIT;")

            logging.debug(f"Элемент с ID {id_feature} обработан.")
        logging.info("Данные успешно загружены!")

    except Exception as e:
        conn.rollback()
        print(f"Ошибка: {e}")
    finally:
        cursor.close()
        conn.close()

# Использование
if __name__ == "__main__":
    process_geojson(Path(__file__).parent.parent / "data" / "output" / "test_unity.geojson")