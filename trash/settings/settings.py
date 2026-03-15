# pylint: disable=too-few-public-methods
"""Конфигурационные параметры для обработки водных путей."""

from pathlib import Path


class DefaultLocate:
    """Стандартные пути к директориям и файлам данных.
    Attributes:
        DATA_DIR: Путь к директории с данными.
        GEOJSON_DIR: Путь к директории с GeoJSON файлами.
        RECORDS_DIR: Путь к директории с записями.
        GRAPHS_DIR: Путь к директории с графами.
        OUTPUT_DIR: Путь к директории для вывода результатов.
        DEFAULT_PATH_GRAPH: Путь к файлу стандартного графа.
        DEFAULT_PATH_RECORD: Путь к файлу стандартной записи.
        DEFAULT_PATH_GEOJSON: Путь к файлу стандартного GeoJSON.
    """

    DATA_DIR = Path(__file__).parent.parent / "data"
    OSM_DIR = DATA_DIR / "osm"
    GEOJSON_DIR = DATA_DIR / "geojson"
    RECORDS_DIR = DATA_DIR / "recorders"
    RECORDS_DIR_RAW = RECORDS_DIR / "raw"
    RECORDS_DIR_PROCESSED = RECORDS_DIR / "processed"
    GRAPHS_DIR = DATA_DIR / "graphs"
    OUTPUT_DIR = DATA_DIR / "output"
    DATA_RAW_DIR = DATA_DIR / "raw"
    DATA_ANALYZED_DIR = DATA_DIR / "analyzed"
    DATA_PREPROCESSED_DIR = DATA_DIR / "pre_processing"
    DATA_TEMP_DIR = DATA_DIR / "temp"
    DATA_POSTPROCESSED_DIR = DATA_DIR / "post_processing"
    DEFAULT_PATH_GRAPH = GRAPHS_DIR / "default_graph.svg"
    DEFAULT_PATH_RECORD = RECORDS_DIR / "default_record.csv"
    DEFAULT_PATH_GEOJSON = GEOJSON_DIR / "default_geojson.geojson"
    DATA_PATH_DATAFILE = DATA_DIR / "default_csv.csv"


class TagsOSM:
    """Теги OSM для фильтрации водных объектов.
    Attributes:
        AREA_TAGS_INCLUDE: Теги для фильтрации полигонов (Area).
        WAY_TAGS_INCLUDE: Теги для фильтрации линий (Way).
        AREA_BLACKLIST: Черный список тегов для полигонов.
        WAY_BLACKLIST: Черный список тегов для линий.
        SKIP_KEYWORDS: Ключевые слова для пропуска при обработке тегов.
        WHITE_LIST: Белый список значений тегов для включения.
    """

    AREA_TAGS_INCLUDE = {
        "natural": [
            "bay",  # Водоём
            "cape",  # Мыс
            "isthmus",  # Перешеек
            "strait",  # Пролив
            "water",  # Вода
        ],
        "place": [
            "island",  # Остров
            "islet",  # Островок
            "sea",  # Море
            "ocean",  # Океан
        ],
        "water": [
            "river",  # Река
            "oxbow",  # Старое русло реки
            "canal",  # Канал
            "lock",  # Шлюз
            "lake",  # Озеро
            "reservoir",  # Водохранилище
            "pond",  # Пруд
            "dam",  # Плотина
            "lagoon",  # Лагуна
            "fjord",  # Фьорд
            "bay",  # Бухта
            "sea",  # Море
        ],
        "waterway": [
            "river",  # Река
            "riverbank",  # Берег реки
            "canal",  # Канал
            "drain",  # Канавка
            "dam",  # Плотина
        ],
    }

    WAY_TAGS_INCLUDE = {
        "natural": [
            "bay",  # Водоём
            "coastline",  # Береговая линия
            "strait",  # Пролив
        ],
        "waterway": [
            "river",  # Река
            "riverbank",  # Берег реки
            "canal",  # Канал
            "drain",  # Канавка
            "dam",  # Плотина
            "ditch",  # Канава
            "weir",  # Плотина
            "lock_gate",  # Шлюзовые ворота
        ],
    }

    BLACKLIST = {
        "natural": ["isthmus", "scrub", "peninsula", "bay", "beach", "coastline"],
        "water": ["oxbow", "harbour"],
        "waterway": ["dam"],
        "building": ["yes", "ruins"],
        "landuse": ["industrial", "meadow"],
        "pond": ["yes"],
        "leisure": ["beach_resort"],
        "sport": ["swimming"],
        "nudism": ["yes"],
        "naturism": ["yes"],
        "supervised": ["no", "08:00-18:00", "yes"],
        "wheelchair": ["no"],
        "ship": ["no"],
        "magic_wand": ["yes"],
        "intermittent": ["yes"],
        "surface": ["sand", "gravel"],
        "tunnel": ["building_passage", "culvert", "yes"],
        # "" : [""],
    }


if __name__ == "__main__":
    print(DefaultLocate.DATA_DIR)
    print(TagsOSM.WAY_TAGS_INCLUDE)
