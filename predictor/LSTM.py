import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy
import math
import matplotlib.pyplot as plot
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# ----------------------------------------------------------------------------------------------------------------------
# Load the data-set.
filePath = os.path.join('..', 'all_data', 'Stocks', 'wmt.us.txt')
dataframe = read_csv(filePath, usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

# ----------------------------------------------------------------------------------------------------------------------
# Normalize the data-set.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# ----------------------------------------------------------------------------------------------------------------------
# Split into training and test data-sets.
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# ----------------------------------------------------------------------------------------------------------------------
# Function to convert an array of values into a data-set matrix.
def create_dataset_matrix(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# ----------------------------------------------------------------------------------------------------------------------
# Reshape into X = t
# Reshape into Y = t + 1
look_back = 1
trainX, trainY = create_dataset_matrix(train, look_back)
testX, testY = create_dataset_matrix(test, look_back)

# ----------------------------------------------------------------------------------------------------------------------
# Reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# ----------------------------------------------------------------------------------------------------------------------
# Create and fit the LSTM network model.
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

# ----------------------------------------------------------------------------------------------------------------------
# Make predictions.
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# ----------------------------------------------------------------------------------------------------------------------
# Invert prediction data to un-normalized it.
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# ----------------------------------------------------------------------------------------------------------------------
# Calculate the root mean squared error.
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % testScore)

# ----------------------------------------------------------------------------------------------------------------------
# Shift the train predictions for plotting.
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

# ----------------------------------------------------------------------------------------------------------------------
# Shift the test predictions for plotting.
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

# ----------------------------------------------------------------------------------------------------------------------
# Save my model after training is complete.
model.save("lstm_model.h5")
print("Saved model named 'lstm_model.h5' in the 'predictor' package.")

# ----------------------------------------------------------------------------------------------------------------------
# Plot baseline and predictions.
plot.plot(scaler.inverse_transform(dataset))
plot.plot(trainPredictPlot)
plot.plot(testPredictPlot)
plot.ylabel('Price (USD)')
plot.xlabel('Days')
plot.title('Stock Price Predictions')
plot.show()
