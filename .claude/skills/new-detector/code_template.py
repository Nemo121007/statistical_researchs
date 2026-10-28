# Шаблон code-ячейки детектора для notebooks/example_model.ipynb (структура блока ARIMA).
# Заменить: {{PREFIX}} — префикс констант (LOF, IF, ...), {{slug}} — префикс переменных (lof, if_, ...),
# {{TITLE}} — название модели в подписях. Строки-комментарии шаблона в ячейку не переносить.
# Общие хелперы (select_anomaly_threshold, evaluate_test, test_report_row, display_test_report_table,
# trajectory_series_pair, slice_trajectory_series, window_starts, ...) уже есть в общей ячейке — не дублировать.
# Импорты новых пакетов — только в ячейке импортов, не здесь.

{{PREFIX}}_PARAM = 1.0


def {{slug}}_fit(train: pd.DataFrame) -> object:
    """Обучает модель на обучающей части сценария.

    Args:
        train: Обучающая часть с колонками ``x``, ``n`` (ENU, м), ``time``, ``y``.

    Returns:
        Обученная модель.
    """
    raise NotImplementedError


def {{slug}}_scores(model: object, frame: pd.DataFrame) -> np.ndarray:
    """Score аномальности по точкам ``frame``: чем больше, тем аномальнее.

    Args:
        model: Модель из {{slug}}_fit.
        frame: Train или test сценария.

    Returns:
        Score той же длины, что ``frame``.
    """
    raise NotImplementedError


SCENARIOS = (
    (f"df1 (rho={TEST_SPLIT_1:.0%})", df1),
    (f"df2 (rho={TEST_SPLIT_2:.0%})", df2),
)

{{slug}}_rows: list[dict[str, object]] = []
scenario_results: list[dict[str, object]] = []

for scenario_name, bundle in SCENARIOS:
    train = bundle["train"]
    test = bundle["test"]
    anomaly_train = train["y"].astype(str).to_numpy() == "anomaly"

    print("=" * 72)
    print(scenario_name)
    model = {{slug}}_fit(train)
    train_scores = {{slug}}_scores(model, train)
    selection = select_anomaly_threshold(train, train_scores, MCC_BAND)
    if np.any(~anomaly_train) and np.any(anomaly_train):
        print(
            f"  медиана score на train: move {np.median(train_scores[~anomaly_train]):.4g}, "
            f"anomaly {np.median(train_scores[anomaly_train]):.4g}"
        )
    print(
        f"  порог {selection['tau']:.4g}: MCC train {selection['train_mcc']:.4f} "
        f"(лучший {selection['best_mcc']:.4f}, в полосе {int(selection['band_size'])}), "
        f"DL train {selection['train_dl_m'] / 1000:.4f} км"
    )

    test_scores = {{slug}}_scores(model, test)
    predicted_test = test_scores > selection["tau"]
    metrics = evaluate_test(test, predicted_test)

    {{slug}}_rows.append(test_report_row(scenario_name, metrics))
    scenario_results.append(
        {
            "name": scenario_name,
            "test": test,
            "predicted_test": predicted_test,
            "metrics": metrics,
        }
    )

{{slug}}_summary = pd.DataFrame({{slug}}_rows)
report_columns = ["Сценарий"] + [key for key, _ in TEST_REPORT_METRICS]
display_test_report_table(
    {{slug}}_summary.loc[:, report_columns],
    caption="{{TITLE}}: сводные метрики на test",
)

_ideal_test = scenario_results[0]["test"]
_map_tail = min(PLOT_TAIL, len(_ideal_test))
print(f"\nИдеальная разметка: последние {nfmt(_map_tail)} точек теста df1 (bbox: move + 5%)")
_dummy_move = np.zeros(len(_ideal_test), dtype=bool)
_truth_pred, _truth_gt = trajectory_series_pair(_ideal_test, _dummy_move)
_truth_pred = slice_trajectory_series(_truth_pred, _map_tail)
_truth_gt = slice_trajectory_series(_truth_gt, _map_tail)
plot_track_maps(
    _truth_pred,
    _truth_gt,
    map_panel="ground_truth",
    title="{{TITLE}} — идеальная разметка",
    figsize=MAP_FIGSIZE,
    annotate=False,
)

for item in scenario_results:
    tail = min(PLOT_TAIL, len(item["test"]))
    print(f"\n{item['name']}")
    display_test_report_table(
        pd.DataFrame([test_report_row(str(item["name"]), item["metrics"])]).loc[:, report_columns],
        caption=f"Метрики test: {item['name']}",
    )
    print(f"  Карта разметки модели: последние {nfmt(tail)} точек теста (bbox: move + 5%)")
    prediction, ground_truth = trajectory_series_pair(item["test"], item["predicted_test"])
    prediction = slice_trajectory_series(prediction, tail)
    ground_truth = slice_trajectory_series(ground_truth, tail)
    plot_track_maps(
        prediction,
        ground_truth,
        map_panel="model",
        title=f"{{TITLE}} — {item['name']}",
        figsize=MAP_FIGSIZE,
        annotate=False,
    )
