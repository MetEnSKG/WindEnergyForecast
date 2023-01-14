import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# RMSE --> root mean squared error

data = pd.read_csv("ΕτήσιαΑιολικήΕνεργειακήΠαραγωγήΛέσβος.csv", parse_dates=["timS"], index_col="timS")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train, test = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]
model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
train = np.reshape(train, (train.shape[0], 1, train.shape[1]))
test = np.reshape(test, (test.shape[0], 1, test.shape[1]))

model.fit(train, epochs=100, batch_size=1, verbose=2)

train_predict = model.predict(train)
test_predict = model.predict(test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

train_score = np.sqrt(mean_squared_error(train_predict, train))
print('Apotelesma1: %.2f RMSE' % (train_score))
test_score = np.sqrt(mean_squared_error(test_predict, test))
