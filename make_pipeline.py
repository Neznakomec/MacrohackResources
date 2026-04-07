from sklearn.base import BaseEstimator, TransformerMixin
from arch import arch_model
from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd


def rolling_realized_volatility(series, window=22, multiply_to_annual = 12):
    """
    Рассчитывает реализованную волатильность в скользящем окне.

    Метод основан на стандартном отклонении логарифмических доходностей.

    Args:
        series (pd.Series): Временной ряд ценовых данных.
        window (int): Размер окна для расчета (количество периодов). По умолчанию 22.
        multiply_to_annual (int): Множитель для годового исчисления.
            Используйте 12 для месячных данных, 252 для дневных. По умолчанию 12.

    Returns:
        pd.Series: Ряд годовой реализованной волатильности.
    """
    log_returns = np.log(series / series.shift(1))

    # Считаем корень из скользящей суммы квадратов
    rolling_rv = np.sqrt((log_returns ** 2).rolling(window=window).sum())
    return rolling_rv * multiply_to_annual

def garch_realized_volatility(series, feature_as_is_without_returns=False):
    """
    Оценивает условную волатильность с использованием модели GARCH(1,1).

    Args:
        series (pd.Series): Временной ряд данных.
        feature_as_is_without_returns (bool): Если True, модель строится напрямую
            на значениях ряда без расчета доходностей. Если False, рассчитываются
            логарифмические доходности. По умолчанию False.

    Returns:
        pd.Series: Условная волатильность, предсказанная моделью GARCH.
    """
    returns = 100 * np.log(series / series.shift(1)).dropna()
    if feature_as_is_without_returns:
        returns = series.dropna()

    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
    model_fitted = model.fit(disp='off')

    # Получаем условную волатильность (conditional volatility), предсказанную моделью
    # Она уже в том же масштабе, что и доходности
    garch_vol = model_fitted.conditional_volatility
    return garch_vol

def add_lag_features_to_data(data, feature_name, lag_start=1, lag_end=7, test_size=0.15):
    # if feature_name not in data.columns:
    #     return data

    data = pd.DataFrame(data.copy())

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data[f"{feature_name}_lag_{i}"] = data[feature_name].shift(i)

    data = data.dropna()

    return data

def generate_volatility_features(data, feature_name, feature_as_is_without_returns = False):
    """
    Генерирует признаки волатильности (Rolling и GARCH) для указанной колонки.

    Создает две новые колонки в исходном DataFrame: '{feature_name}_RV'
    и '{feature_name}_GARCH'.

    Args:
        df (pd.DataFrame): Исходный датафрейм, в который будут добавлены признаки.
        feature_name (str): Название колонки, для которой считается волатильность.
        feature_as_is_without_returns (bool): Флаг для передачи в GARCH модель.
            По умолчанию False.

    Returns:
        None: Изменяет объект DataFrame на месте (in-place).
    """

    # считаем волатильность разными способами
    data[f'{feature_name}_RV_Rolling_10'] = rolling_realized_volatility(data[feature_name], window=10)
    data[f'{feature_name}_GARCH_1_1'] = garch_realized_volatility(data[feature_name], feature_as_is_without_returns)

    # изменение в будущем = будущее значение в датасете минус текущее значение
    data[f'{feature_name}_RV_change_1m'] = -1.0 * data[f'{feature_name}_RV_Rolling_10'].diff(-1)
    data[f'{feature_name}_RV_change_3m'] = -1.0 * data[f'{feature_name}_RV_Rolling_10'].diff(-3)
    data[f'{feature_name}_RV_change_6m'] = -1.0 * data[f'{feature_name}_RV_Rolling_10'].diff(-6)
    data[f'{feature_name}_RV_change_12m'] = -1.0 * data[f'{feature_name}_RV_Rolling_10'].diff(-12)

    data[f'{feature_name}_GARCH_change_1m'] = -1.0 * data[f'{feature_name}_GARCH_1_1'].diff(-1)
    data[f'{feature_name}_GARCH_change_3m'] = -1.0 * data[f'{feature_name}_GARCH_1_1'].diff(-3)
    data[f'{feature_name}_GARCH_change_6m'] = -1.0 * data[f'{feature_name}_GARCH_1_1'].diff(-6)
    data[f'{feature_name}_GARCH_change_12m'] = -1.0 * data[f'{feature_name}_GARCH_1_1'].diff(-12)

def generated_future_changes_of_feature(data, feature_name_current, feature_name_output):
    data[f'{feature_name_output}_change_1m'] = -1.0 * data[f'{feature_name_current}'].diff(-1)
    data[f'{feature_name_output}_change_3m'] = -1.0 * data[f'{feature_name_current}'].diff(-3)
    data[f'{feature_name_output}_change_6m'] = -1.0 * data[f'{feature_name_current}'].diff(-6)
    data[f'{feature_name_output}_change_12m'] = -1.0 * data[f'{feature_name_current}'].diff(-12)


class DatasetFillValues(BaseEstimator, TransformerMixin):
    """
    Pipeline-трансформер для подготовки макроэкономических данных.

    Выполняет загрузку, очистку, генерацию признаков и объединение
    различных источников данных (ключевая ставка, MPU, GDP и др.).
    """

    def __init__(self):
        """
        Инициализирует pipeline набором исходных данных.

        Args:
            keyrate_df (pd.DataFrame): Данные по ключевой ставке.
            mpu_df (pd.DataFrame): Индекс неопределенности (MPU).
            tradingview_df (pd.DataFrame): Данные TradingView (CPI, Output Gap).
            gdp_df (pd.DataFrame): Данные по промышленному производству.
            reer_df (pd.DataFrame): Данные по реальному эффективному курсу (REER).
        """
        # self.mpu_df = pd.read_csv('mpu_index_v1/mpu_index.csv')
        rnd_df = pd.read_csv('mpu_index_v2/rnd_mpu.csv')
        rwd_df = pd.read_csv('mpu_index_v2/rwd_mpu.csv')
        self.mpu_df = pd.merge(rnd_df, rwd_df, how='inner', on='Date')
        self.keyrate_df = pd.read_csv('cbr_key_rate.csv')
        self.gdp_df = pd.read_csv('gdp_proxy.csv')
        self.reer_df = pd.read_csv('reer_russia.csv')
        self.ruonia_df = pd.read_csv('ruonia_rate.csv')
        self.tradingview_df = pd.read_csv('full_hope.csv')
        self.news_mpu_df = pd.read_csv('RussiaNewsBasedUncertainty.csv')

    def fit(self, X, y=None):
        """
        Метод соответствия (fit) для интеграции в sklearn Pipeline.

        Args:
            X: Входные данные.
            y: Целевая переменная.

        Returns:
            self: Возвращает экземпляр класса.
        """
        return self

    def transform(self, X, y=None):
        """
        Основная логика трансформации и объединения данных.

        Выполняет расчет признаков волатильности для всех макропоказателей
        и собирает их в единый массив признаков (dataset).

        Args:
            X: Входные данные (не используются, так как данные переданы в __init__).

        Returns:
            pd.DataFrame: Объединенный датасет со всеми сгенерированными признаками.
        """

        # 1. Убеждаемся, что все колонки имеют тип datetime
        self.mpu_df['date'] = pd.to_datetime(self.mpu_df['Date'])
        self.keyrate_df['date'] = pd.to_datetime(self.keyrate_df['Дата'])
        self.tradingview_df['date'] = pd.to_datetime(self.tradingview_df[self.tradingview_df.columns[0]])

        # 2. Создаем временные колонки для объединения (формат 'YYYY-MM')
        # Используем dt.to_period('M'), это самый надежный способ для стыковки месяцев
        self.mpu_df['month_key'] = self.mpu_df['date'].dt.to_period('M')
        self.keyrate_df['month_key'] = self.keyrate_df['date'].dt.to_period('M')
        self.tradingview_df['month_key'] = self.tradingview_df['date'].dt.to_period('M')
        self.news_mpu_df['month_key'] = pd.to_datetime(self.news_mpu_df['Date']).dt.to_period('M')
        self.gdp_df['month_key'] = pd.to_datetime(self.gdp_df['Date']).dt.to_period('M')
        self.reer_df['month_key'] = pd.to_datetime(self.reer_df['Date']).dt.to_period('M')
        self.ruonia_df['month_key'] = pd.to_datetime(self.ruonia_df['Date']).dt.to_period('M')

        # Удаляем лишние колонки
        self.mpu_df.drop(columns=['Date', 'date'], inplace=True)
        self.keyrate_df.drop(columns=['Дата', 'date'], inplace=True)
        self.tradingview_df = self.tradingview_df[['month_key', 'CPI', 'OUTPUT_GAP']]
        self.news_mpu_df.drop(columns=['Date'], inplace=True)
        self.gdp_df.drop(columns=['Date'], inplace=True)
        self.reer_df.drop(columns=['Date'], inplace=True)
        self.ruonia_df.drop(columns=['Date'], inplace=True)

        self.news_mpu_df.rename(columns={'MPU_Index' : 'News_MPU_Index'}, inplace=True)
        # self.mpu_df = add_lag_features_to_data(data = self.mpu_df, feature_name='MPU_Index', lag_start=1, lag_end=7)
        self.mpu_df = add_lag_features_to_data(data=self.mpu_df, feature_name='RND_MPU_Index', lag_start=1, lag_end=7)
        self.mpu_df = add_lag_features_to_data(data=self.mpu_df, feature_name='RWD_MPU_Index', lag_start=1, lag_end=7)
        self.keyrate_df = self.keyrate_df.groupby(by='month_key').last()
        self.reer_df.rename(columns={'RBRUBIS' : 'REER'}, inplace=True)
        self.keyrate_df.rename(columns = {'Ключевая_ставка' : 'keyrate_end_of_month'}, inplace=True)
        self.ruonia_df.rename(columns= {'Realized_Vol' : 'ruonia_RV_Rolling_DaysInMonth', 'GARCH_Vol' : 'ruonia_GARCH_1_1'}, inplace=True)

        generate_volatility_features(self.keyrate_df, feature_name='keyrate_end_of_month')

        generate_volatility_features(self.tradingview_df, feature_name='CPI', feature_as_is_without_returns=True)

        # просто чтобы OUTPUT_GAP был в положительных числах около 100.0
        self.tradingview_df['OUTPUT_GAP'] = 100.0 + self.tradingview_df['OUTPUT_GAP']

        generate_volatility_features(self.tradingview_df, feature_name='OUTPUT_GAP')

        self.gdp_df['GDP_promproduction_clean'] = 100.0 + self.gdp_df['GDP_promproduction_clean']
        generate_volatility_features(self.gdp_df, feature_name='GDP_promproduction_clean')

        generate_volatility_features(self.reer_df, feature_name='REER')

        generated_future_changes_of_feature(self.ruonia_df, feature_name_current = 'ruonia_RV_Rolling_DaysInMonth', feature_name_output='ruonia_RV')
        generated_future_changes_of_feature(self.ruonia_df, feature_name_current='ruonia_GARCH_1_1', feature_name_output='ruonia_GARCH')

        dataset = pd.merge(self.keyrate_df, self.mpu_df, on='month_key', how='inner')
        dataset = pd.merge(dataset, self.tradingview_df, on='month_key', how='inner')
        dataset = pd.merge(dataset, self.gdp_df, on='month_key', how='inner')
        dataset = pd.merge(dataset, self.reer_df, on='month_key', how='inner')
        dataset = pd.merge(dataset, self.news_mpu_df, on='month_key', how='inner')
        dataset = pd.merge(dataset, self.ruonia_df, on='month_key', how='inner')
        dataset.set_index(keys='month_key')

        return dataset

pipe = Pipeline([
    ('make_df', DatasetFillValues()),
])

dataframe = pipe.fit_transform(pd.DataFrame())
# print(dataframe.tail())
print(dataframe.columns)

dataframe.to_excel('final_data.xlsx', index=False)
