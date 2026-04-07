import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


class TermStructureDecomposition:
    def __init__(self, tenors=[1, 2, 5, 10]):
        self.tenors = tenors  # Сроки в годах

    def forecast_short_rate(self, short_rate, steps):
        """
        Строит прогноз краткосрочной ставки (RS) на n шагов вперед.
        В реальном Кейсе 1 здесь может быть VAR или модель ACM.
        """
        # Обучаем простую модель AR(1), чтобы было что усреднять в будущем
        model = AutoReg(short_rate, lags=1).fit()
        forecast = model.predict(start=len(short_rate), end=len(short_rate) + steps - 1)
        return forecast

    def decompose(self, yields_df, short_rate_series):
        """
        Воспроизводит логику MATLAB: TPREM = RSG - Average(Forecast_RS)
        """
        results = pd.DataFrame(index=yields_df.index)

        # Для каждой даты в истории мы строим прогноз "вперед"
        for date in yields_df.index:
            history_up_to_date = short_rate_series.loc[:date]
            if len(history_up_to_date) < 5:  # Нужно хоть немного данных для AR
                continue

            for t in self.tenors:
                # 1. Сколько кварталов прогнозировать (t лет * 4)
                k_steps = t * 4

                # 2. Генерируем ожидаемую траекторию ставки
                future_expectations = self.forecast_short_rate(history_up_to_date, k_steps)
                expected_mean_rate = future_expectations.mean()

                # 3. Считаем премию для данного тенора
                yield_col = f'RSG{t}'
                if yield_col in yields_df.columns:
                    val_yield = yields_df.loc[date, yield_col]
                    results.loc[date, f'EW{t}Y_RS'] = expected_mean_rate
                    results.loc[date, f'TPREM{t}'] = val_yield - expected_mean_rate

        return results


# --- Пример с данными, которые не выдадут NaN ---
np.random.seed(42)
idx = pd.date_range(start='2018-01-01', periods=30, freq='Q')

mock_data = pd.DataFrame({
    'RS': 7 + np.cumsum(np.random.randn(30)),  # Короткая ставка (RUONIA)
    'RSG10': 9 + np.cumsum(np.random.randn(30)) * 0.5  # 10Y ОФЗ
}, index=idx)

# Запуск
model = TermStructureDecomposition(tenors=[10])
decomposition = model.decompose(mock_data[['RSG10']], mock_data['RS'])

# Теперь NaN будут только в самом начале (пока AR-модель не обучилась)
print("Результаты декомпозиции (последние 5 кварталов):")
print(decomposition.tail())