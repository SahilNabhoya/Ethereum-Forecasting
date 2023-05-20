import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import pickle
data = pd.read_csv('ETH-USD.csv')
data = data.fillna(method='ffill')
df = data.filter(['Close'])
dataset = df.values
training_data_len = math.ceil(len(dataset)*0.8)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<=60:
    print(x_train)
    print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(30, return_sequences=False, input_shape=(x_train.shape[1], 1)))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

model.fit(x_train, y_train, batch_size=1, epochs=1)



with open('lstm.pkl','wb') as f:
    pickle.dump(model,f)


test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len :, :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)

rmse = np.sqrt(np.mean(pred-y_test)**2)
print('RMSE : ',rmse)

train = df[:training_data_len]
valid = df[training_data_len:]
valid['Prediction'] = pred

plt.figure(figsize=(16,8))
plt.xlabel("Date")
plt.ylabel("Close price")
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Prediction']])
plt.legend(['Train', 'Val', 'Pred'])
plt.show()