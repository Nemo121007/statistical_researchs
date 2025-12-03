from pathlib import Path

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

def get_list_csv(path_dir: Path) -> list[Path]:
    return [file for file in path_dir.iterdir() if file.suffix == ".csv"]

def read_data(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    df = df.dropna()
    df.sort_values(by="time", inplace=True, ignore_index=True)
    data = df[['lon', 'lat']].to_numpy()
    return data

def print_results(smoothed_state_means: np.ndarray, data: np.ndarray, name_file: Path = None) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], 'ro', label='Наблюдения', alpha=0.5)
    plt.plot(smoothed_state_means[:, 0], smoothed_state_means[:, 1], 'b-', label='Сглаженные состояния')
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.title('Сравнение измерений и сглаженной траектории')
    plt.legend()
    plt.grid()

    if name_file:
        plt.savefig(name_file)
    else:
        plt.show()


def em_algorithm(data: np.ndarray, n_iter: int = 100):
    # Начальные приближения
    A_init = np.eye(2)
    Q_init = np.eye(2) * 0.1
    R_init = np.eye(2) * 0.1

    # Подготовка данных: все наблюдения, включая первое
    observations = data  # Используем все данные

    # Инициализация модели
    kf = KalmanFilter(
        n_dim_obs=2,  # Размерность наблюдений
        n_dim_state=2,  # Размерность состояния
        initial_state_mean=data[0],  # x_0 = data[0]
        initial_state_covariance=np.zeros((2, 2)),  # x_0 известно точно
        transition_matrices=A_init,
        transition_covariance=Q_init,
        observation_matrices=np.eye(2),  # H = I
        observation_covariance=R_init
    )

    # Обучение с помощью EM-алгоритма
    kf = kf.em(observations, n_iter=n_iter)

    A_estimated = kf.transition_matrices
    Q_estimated = kf.transition_covariance
    R_estimated = kf.observation_covariance

    smoothed_state_means, _ = kf.smooth(observations)

    return A_estimated, Q_estimated, R_estimated, smoothed_state_means


if __name__ == "__main__":
    # data_path = Path(__file__).parent / "output" / "example_0_0.csv"
    path_log = Path(__file__).parent / "em_algorithm_log.txt"
    list_files = get_list_csv(Path(__file__).parent / "output")
    for file_name in list_files:
        print(f"Обработка файла: {file_name.name}")
        data = read_data(file_name)

        # Запускаем EM-алгоритм
        A_est, Q_est, R_est, smoothed_state_means = em_algorithm(data=data)

        # Выводим результаты
        result = (f"\nРезультаты {file_name.name}:\n"
                  f"Оцененная матрица A:\n{A_est}\n"
                  f"Оцененная матрица Q:\n{Q_est}\n"
                  f"Оцененная матрица R:\n{R_est}\n")
        print(result)
        with path_log.open('a', encoding='utf-8') as f:
            f.write(result)

        print_results(smoothed_state_means=smoothed_state_means,
                      data=data,
                      name_file=file_name.parent.parent / "pictures" / f"{file_name.stem}_result.png")

        data = data[::-1]
        A_est, Q_est, R_est, smoothed_state_means = em_algorithm(data=data)
        result = (f"\nРезультаты (обратный порядок) {file_name.name}:\n"
                  f"Оцененная матрица A:\n{A_est}\n"
                  f"Оцененная матрица Q:\n{Q_est}\n"
                  f"Оцененная матрица R:\n{R_est}\n")
        print(result)
        with path_log.open('a', encoding='utf-8') as f:
            f.write(result)
        print_results(smoothed_state_means=smoothed_state_means,
                      data=data,
                      name_file=file_name.parent.parent / "pictures" / f"{file_name.stem}_result_reverce.png")

