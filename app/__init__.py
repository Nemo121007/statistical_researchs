"""Исследования и оценка качества GPS-траекторий судов.

Пакет app — рабочая кодовая база проекта: подготовка телеметрии, детекция аномалий в точках трека и сравнение
предсказаний модели с эталонной разметкой. Определения метрик и порядок их расчёта зафиксированы в report/Отчёт.tex
(геопространственные, временные, классификационные).

Подпакеты:

- evaluation — TrajectorySeries, расчёт метрик
    (ClassificationMetrics, TemporalMetrics, GeospatialMetrics, сводный TrajectoryMetrics).
- geo — геодезические расстояния, коллекторы участков карты, GeoJSON.
- viz — визуализация маршрута на фоне OpenStreetMap (plot_track_maps).
- filters, detectors — экспериментальные фильтры и алгоритмы детекции.

Импортируйте нужный подпакет явно, например
from app.evaluation import TrajectoryMetrics, TrajectorySeries.
"""
