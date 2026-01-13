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
    DEFAULT_PATH_GRAPH = GRAPHS_DIR / "default_graph.svg"
    DEFAULT_PATH_RECORD = RECORDS_DIR / "default_record.csv"
    DEFAULT_PATH_GEOJSON = GEOJSON_DIR / "default_geojson.geojson"


class DefaultLocateRegion:
    """Пути к файлам OSM для различных федеральных округов.
    Attributes:
        south_fed_district: Путь к файлу OSM для Южного федерального округа.
        central_fed_district: Путь к файлу OSM для Центрального федерального округа.
        volga_fed_district: Путь к файлу OSM для Приволжского федерального округа.
    """

    central_fed_district = DefaultLocate.OSM_DIR / "central-fed-district.osm.pbf"
    crimean_fed_district = DefaultLocate.OSM_DIR / "crimean-fed-district.osm.pbf"
    far_eastern_fed_district = DefaultLocate.OSM_DIR / "far-eastern-fed-district.osm.pbf"
    north_caucasus_fed_district = DefaultLocate.OSM_DIR / "north-caucasus-fed-district.osm.pbf"
    northwestern_fed_district = DefaultLocate.OSM_DIR / "northwestern-fed-district.osm.pbf"
    siberian_fed_district = DefaultLocate.OSM_DIR / "siberian-fed-district.osm.pbf"
    south_fed_district = DefaultLocate.OSM_DIR / "south-fed-district.osm.pbf"
    ukraine = DefaultLocate.OSM_DIR / "ukraine.osm.pbf"
    ural_fed_district = DefaultLocate.OSM_DIR / "ural-fed-district.osm.pbf"
    volga_fed_district = DefaultLocate.OSM_DIR / "volga-fed-district.osm.pbf"


# # Создать директории при загрузке модуля
# for directory in [
#     DefaultLocate.GEOJSON_DIR,
#     DefaultLocate.RECORDS_DIR,
#     DefaultLocate.GRAPHS_DIR,
#     DefaultLocate.OUTPUT_DIR,
# ]:
#     if directory.exists():
#         logging.info(f"Подключена директория: {str(directory)}")
#     else:
#         directory.mkdir(parents=True, exist_ok=True)
#         logging.info(f"Создана директория: {str(directory)}")


class ExamplesSettings:
    """Пути к примерным файлам данных для тестирования и демонстрации."""

    PATH_EXAMPLES_CSV = DefaultLocate.RECORDS_DIR / "processed_7.csv"

    PATH_EXAMPLES_GEOJSON = DefaultLocate.GEOJSON_DIR / "south-fed-district-water.geojson"

    PATH_EXAMPLES_GEOJSON_OUTPUT = DefaultLocate.OUTPUT_DIR / "example_output.geojson"

    PATH_OSM = DefaultLocate.DATA_DIR / "osm" / "south-fed-district-251016.osm.pbf"


class DefaultLocateRegionGeoJson:
    """Пути к файлам GeoJSON для различных федеральных округов.
    Attributes:

    """

    central_fed_district = DefaultLocate.GEOJSON_DIR / "central-fed-district.geojson"
    crimean_fed_district = DefaultLocate.GEOJSON_DIR / "crimean-fed-district.geojson"
    far_eastern_fed_district = DefaultLocate.GEOJSON_DIR / "far-eastern-fed-district.geojson"
    north_caucasus_fed_district = DefaultLocate.GEOJSON_DIR / "north-caucasus-fed-district.geojson"
    northwestern_fed_district = DefaultLocate.GEOJSON_DIR / "northwestern-fed-district.geojson"
    siberian_fed_district = DefaultLocate.GEOJSON_DIR / "siberian-fed-district.geojson"
    south_fed_district = DefaultLocate.GEOJSON_DIR / "south-fed-district.geojson"
    ukraine = DefaultLocate.GEOJSON_DIR / "ukraine.geojson"
    ural_fed_district = DefaultLocate.GEOJSON_DIR / "ural-fed-district.geojson"
    volga_fed_district = DefaultLocate.GEOJSON_DIR / "volga-fed-district.geojson"


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
