import tensorflow as tf
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
import statsmodels.api as sm
from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pickle import dump

from numpy import *
from math import sqrt
from pandas import *
from datetime import datetime, timedelta

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Embedding, TimeDistributed, LeakyReLU
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.optimizers import Adam

import seaborn as sns

from matplotlib import pyplot
from pickle import load
##Load the data
allmodal = pd.read_csv('allmodal.csv',parse_dates=['time'])
###data preprocessing
allmodal = allmodal.set_index('time')
allmodal.index = pd.to_datetime(allmodal.index)
df_BTC_ext = allmodal.copy()
df_BTC_ext['Prediction'] = df_BTC_ext['close']

# Get the number of rows in the data
nrows = allmodal.shape[0]

# Convert the data to numpy values
np_data_unscaled = np.array(allmodal)
np_data = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = MinMaxScaler()
np_data_scaled = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(df_BTC_ext['close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

# Set the sequence length - this is the timeframe used to make a single prediction
sequence_length = 12

# Prediction Index
index_Close = allmodal.columns.get_loc("close")

# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 70% of the data
train_data_len = math.ceil(np_data_scaled.shape[0] * 0.7)
val_data_len = math.ceil(np_data_scaled.shape[0] * 0.2)
# Create the training and test data
train_data = np_data_scaled[0:train_data_len, :]
test_data = np_data_scaled[train_data_len:train_data_len+val_data_len, :]
val_data = np_data_scaled[train_data_len+val_data_len:, :]

# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, data):
    x, y = [], []
    data_len = data.shape[0]
    for i in range(sequence_length, data_len):
        x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columsn
        y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction

    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

# Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)
x_val, y_val = partition_dataset(sequence_length, val_data)
# Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_val.shape, y_val.shape)

# Validate that the prediction value and the input match up
# The last close price of the second input sample should equal the first prediction value
print(x_train[1][sequence_length-1][index_Close])
print(y_train[0])

#### Model structure
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Concatenate, Dot, Activation
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model

# Input layer for the CNN-LSTM model
input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))

# CNN layers for spatial feature extraction
cnn_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
cnn_layer = Flatten()(cnn_layer)

# LSTM layer for sequence processing
n_neurons = x_train.shape[1] * x_train.shape[2]
lstm_layer = LSTM(n_neurons, return_sequences=True)(input_layer)
lstm_layer = LSTM(n_neurons, return_sequences=False)(lstm_layer)

# Concatenate CNN output with LSTM output
combined = Concatenate(axis=-1)([cnn_layer, lstm_layer])

# Final prediction layer
output_layer = Dense(5)(combined)
output_layer = Dense(1)(output_layer)

# Create the CNN-LSTM model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='mse')

model.summary()

### Test the model and print the validation matrix
# Cross-validation
cvscores = []
kfold = KFold(n_splits=3, shuffle=True)
results = []
for train_index, test_index in kfold.split(x_train):
    # Train the model
    model.fit(x_train[train_index], y_train[train_index], epochs=30, batch_size=10)
    # Evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(scores)
    cvscores.append(scores *100)

print("%.6f%% (+/- %.6f%%)" % (np.mean(cvscores), np.std(cvscores)))

ypred_scaled = model.predict(x_val)
y_pred = scaler_pred.inverse_transform(ypred_scaled)
y_val_unscaled = scaler_pred.inverse_transform(y_val.reshape(-1, 1))

# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_val_unscaled, y_pred)
print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_val_unscaled, y_pred)/ y_val_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_val_unscaled, y_pred)/ y_val_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')