import warnings

import keras.callbacks
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from keras import Sequential
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from make_df import make_df

warnings.filterwarnings("ignore")
data = make_df('2021-2022/Measured_data/ROME')

print(data.dtypes)

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['date'], y=data['value'], name='main data'))
fig.update_layout(showlegend=True, title='station ROME 2021-2022')
fig.show()

train = data.loc[data['date'] <= '2022-10-31 23:30:00']
test_df = data.loc[data['date'] > '2022-10-31 23:30:00']
test_for_date = data.loc[data['date'] > '2022-10-31 23:30:00', ['date', 'value']]
test_for_date = test_for_date.reset_index(drop=True)  # Сбросить индексы и удалить старый столбец индексов
test_for_date['count'] = np.arange(len(test_for_date))  # Создать новый столбец "count" со значениями от 0 до конца

scaler = StandardScaler()
scaler = scaler.fit(train['value'].values.reshape(-1, 1))

train = scaler.transform(train['value'].values.reshape(-1, 1))
test = scaler.transform(test_df['value'].values.reshape(-1, 1))

TIME_STEPS = 30


def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])

    return np.array(Xs), np.array(ys)


train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

X_train, y_train = create_sequences(train_df, train_df)
X_test, y_test = create_sequences(test_df, test_df)

print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')

model = Sequential()
model.add(LSTM(64, input_shape=(TIME_STEPS, 1)))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(TIME_STEPS))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mae')
model.summary()

history = model.fit(X_train, y_train, epochs=1, batch_size=16, validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        mode='min')], shuffle=False)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

model.evaluate(X_test, y_test)

X_train_pred = model.predict(X_train, verbose=0)

'''нахождение порогового значения'''

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
# 1
plt.hist(train_mae_loss, bins=50)
plt.xlabel('Train MAE loss')
plt.ylabel('Number of Samples')
print("loss")
print(train_mae_loss)
threshold = np.percentile(train_mae_loss, 30)
print(f'Reconstruction error threshold: {threshold}')

# 2
# alpha = 0.1
# N = len(train_mae_loss)
# sample_std = np.std(train_mae_loss)
# t = stats.t.ppf(1 - alpha / 2, N - 1)
# threshold = t * sample_std

print("Threshold:", threshold)

X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

plt.hist(test_mae_loss)
plt.xlabel('Test calculated loss')
plt.ylabel('Number of samples')
plt.show()

''''''

test_score_df = pd.DataFrame(test_for_date[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['Close'] = test_for_date['value'][TIME_STEPS:]
test_score_df['date'] = test_for_date.loc[test_score_df.index, 'date'].values

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['date'], y=test_score_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df['date'], y=test_score_df['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()

anomalies = test_score_df.loc[
    test_score_df['anomaly'],
    ['Close', 'date']
]

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=test_score_df['date'], y=test_score_df['Close'].values.reshape(-1, 1).ravel(),
               name='Close price'))
fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['Close'].values.reshape(-1, 1).ravel(),
                         mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()
