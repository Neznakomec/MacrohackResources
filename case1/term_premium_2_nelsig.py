import numpy as np
import pandas as pd
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
from nelson_siegel_svensson import NelsonSiegelCurve


def nelson_siegel_interpolation(market_tenors, market_yields, target_tenors):
    """
    Интерполяция кривой доходности по модели Нельсона-Сигела.
    :param market_tenors: Массив имеющихся сроков (в годах), напр. [0.25, 1, 5, 10]
    :param market_yields: Массив соответствующих доходностей
    :param target_tenors: Сроки, для которых нужно рассчитать доходность
    """
    # Калибровка модели NS методом наименьших квадратов (OLS)
    curve, status = calibrate_ns_ols(market_tenors, market_yields)

    if status.success:
        # Получаем доходности для целевых теноров
        interpolated_yields = curve(target_tenors)
        return interpolated_yields, curve
    else:
        raise ValueError("Калибровка модели Нельсона-Сигела не удалась")


# --- Пример использования в задаче декомпозиции ---

# Рыночные данные (например, на конкретную дату)
tenors = np.array([1, 2, 3, 5, 7, 10])
yields = np.array([7.5, 8.2, 8.8, 9.5, 10.1, 10.5])

# Нам нужно узнать доходность для 4-х лет (которых нет в данных)
target = np.array([4.0])
interp_y, model_params = nelson_siegel_interpolation(tenors, yields, target)

print(f"Интерполированная доходность для 4Y: {interp_y[0]:.2f}%")
print(
    f"Параметры модели: Level={model_params.beta0:.2f}, Slope={model_params.beta1:.2f}, Curvature={model_params.beta2:.2f}")