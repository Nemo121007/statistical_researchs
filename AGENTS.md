# statistical_researchs — инструкции для агента

Исследование: детекция аномалий в GPS-телеметрии судов. Метка 0 — anomaly
(положительный класс), значение больше нуля — move.

## Канон

- Источник истины: `notebooks/example_model.ipynb` и `report/Отчёт.tex`.
- `docs/notes/results.md` и `notebooks/example_model_1/2.ipynb` — архив прошлых
  прогонов (другой размер выборки, другой набор моделей), не эталон.
- Правила оформления блокнотов — `notebooks/AGENTS.md`.

## Эксперименты

- Блокноты запускаются с рабочей директорией `notebooks/`, данные ожидаются
  в `../data/2.csv`.
- Новый детектор повторяет структуру блока ARIMA в `example_model.ipynb`.
- Порог: максимум MCC на train, затем полоса `MCC_BAND`, затем минимальный
  операционный distance loss; на test переносится тот же порог без подгонки.

## Окружение

- Python 3.12, виртуальное окружение `.venv`.
- Полный список зависимостей — `requirements.txt`. `pyproject.toml` отстаёт:
  в нём нет `pyro-ppl`, `statsmodels`, `hmmlearn`, `Pillow`.
- `data/*.csv` не хранятся в git: без исходных файлов эксперименты
  не воспроизводятся.

## Запреты

- Не коммитить `1.md`: в нём учётные данные прокси.
- Не править `report/Отчёт.tex` и `docs/notes/methods.md` без явного
  подтверждения.
- Не выносить логику эксперимента из блокнота в `app/` без просьбы.

## Проверка

- Тестов и CI нет. Минимум перед выводами — импорт затронутых модулей `app.*`
  и расчёт метрик на коротком ряду.
- Известные поломки: `import app.filters` (в `kalman_filter_cv` ссылка на
  отсутствующий `app.working`) и `__main__` в `classification_metrics`
  (вызывает несуществующий `plot_track`).

## Cursor Cloud specific instructions

- В базовом образе нет пакета `python3.12-venv`. Перед созданием окружения:
  `sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12-venv`.
- Виртуальное окружение — `.venv` в корне репозитория:
  `python3.12 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt 'Pillow>=10'`.
  Явный `Pillow` нужен, пока правка `requirements.txt` не попала в ветку по умолчанию: `app.viz.track_osm` импортирует `PIL`.
- Запускать Python из корня репозитория и задавать `MPLBACKEND=Agg`. Долгоживущего сервиса нет.
- Проверка окружения: импорт `app.evaluation`, `app.geo`, `app.detectors`, расчёт `TrajectoryMetrics.compute_all` на коротком синтетическом ряде и `plot_track_maps(..., use_basemap=False)`.
- `import app.filters` по-прежнему падает на `app.working`. `data/*.csv` в git нет, поэтому блокноты на `../data/2.csv` в чистом checkout не воспроизводятся.
