import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__":
    df = pd.read_csv("rusich-3.csv")
    print(df.head())
    df["time"] = df["v0"]
    df["value"] = df["v4"]

    # Исправление сброса счётчика
    max_counter_value = 65535
    df['value_shift'] = 0
    cumulative_shift = 0

    for i in range(1, len(df)):
        if df.loc[i, 'value'] < df.loc[i - 1, 'value']:
            cumulative_shift += max_counter_value
        df.loc[i, 'value_shift'] = cumulative_shift

    df['value_corrected'] = df['value'] + df['value_shift']
    df['value_diff'] = df['value_corrected'].diff()

    df['time'] = pd.to_datetime(df['time'])
    df['time_delta'] = df['time'].diff().dt.total_seconds()
    df['derivative'] = df['value_corrected'].diff() / df['time_delta']
    df['final_val'] = df['derivative'] * 3600 / 12

    print(df.head(20))
    print("\nMin-Max значения:")
    print(pd.DataFrame({
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True),
        'mean': df.mean(numeric_only=True),
        'mode': df.select_dtypes(include=['number']).mode().iloc[0]
    }))
    df.to_csv("result.csv", index=False)

    # Интерактивная визуализация с Plotly
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Временной ряд (с коррекцией сброса)', 'Производная временного ряда'),
        vertical_spacing=0.1
    )

    # График скорректированных значений
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['value_corrected'],
                   mode='lines', name='Скорректированные значения',
                   line=dict(color='blue')),
        row=1, col=1
    )

    # График производной
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['final_val'],
                   mode='lines', name='Производная',
                   line=dict(color='red')),
        row=2, col=1
    )

    # Настройка осей
    fig.update_xaxes(title_text="Время", row=1, col=1)
    fig.update_xaxes(title_text="Время", row=2, col=1)
    fig.update_yaxes(title_text="Значение", row=1, col=1)
    fig.update_yaxes(title_text="Производная (dValue/dTime)", row=2, col=1)

    # Настройка размера и экспорт в HTML
    fig.update_layout(height=800, showlegend=True)
    fig.write_html("rusich_interactive.html")
    print("\nИнтерактивный график сохранён в 'rusich_interactive.html'")

    # Опционально: открыть в браузере
    fig.show()
