import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

from make_df import make_df

warnings.filterwarnings("ignore")
data = make_df('2021-2022/Measured_data/ROME')

# применяем стандартизацию данных
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

# Преобразование отмасштабированных данных обратно в DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=['value'], index=data.index)

'''
    p (AR - Autoregression) - количество предыдущих значений временного ряда, 
    используемых для прогнозирования следующего значения в текущий момент времени.
    d (I - Integration) - количество раз, которое нужно взять разности между текущим значением временного ряда 
    и предыдущими значениями временного ряда для стабилизации дисперсии.
    q (MA - Moving Average) - количество предыдущих ошибок прогнозирования, используемых для прогнозирования текущего значения временного ряда. 
'''

p, d, q = 1, 0, 0

# обучаем модель ARIMA на обучающих данных
arima_model = ARIMA(scaled_df, order=(p, d, q))
arima_model_fit = arima_model.fit()

# сохранение модели
pickle.dump(arima_model_fit, open('arima_model.pkl', 'wb'))

arima_predictions = arima_model_fit.get_forecast(steps=256)

forecast_mean = arima_predictions.predicted_mean

# Преобразование прогнозных значений обратно в исходный масштаб
forecast_mean_inverse = scaler.inverse_transform(forecast_mean.values.reshape(-1, 1))

# Создание DataFrame с прогнозными значениями и установка правильного индекса
forecast_df = pd.DataFrame(forecast_mean_inverse, columns=['value'])
forecast_df.index = pd.date_range(start=data.index[-1], periods=len(forecast_df), freq='30T')

# Построение графика
plt.figure(figsize=(16, 8))
plt.plot(data[34400:].index, data[34400:]['value'], color='red', label='Исходные данные')
plt.plot(forecast_df.index, forecast_df['value'], color='blue', label='Прогнозируемые данные')

plt.legend()
plt.show()
