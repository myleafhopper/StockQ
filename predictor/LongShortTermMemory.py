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
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------------------------------------------------


class LstmModel:

    def __init__(self, file_name, test_set_divider, look_back_step, epochs):
        self.file_name = file_name
        self.test_set_divider = test_set_divider
        self.look_back_step = look_back_step
        self.epochs = epochs
        self.model_name = 'lstm_model.h5'
        self.model_path = os.path.join('./', self.model_name)

    # ------------------------------------------------------------------------------------------------------------------
    # Load and evaluate an existing model or create a new one
    def load_and_summarize_model(self):
        if os.path.isfile(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = Sequential()
            self.model.add(LSTM(4, input_shape=(1, self.look_back_step)))
            self.model.add(Dense(1))
            self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.summary()

    # ------------------------------------------------------------------------------------------------------------------
    # Load the data-set.
    def load_data_file(self):
        print('\n=> Beginning model training using: ' + self.file_name + '\n')
        filePath = os.path.join('..', 'all_data', 'Stocks', self.file_name)
        dataframe = read_csv(filePath, usecols=[1], engine='python')
        dataset = dataframe.values
        self.dataset = dataset.astype('float32')

    # ------------------------------------------------------------------------------------------------------------------
    # Normalize the data-set.
    def normalize_dataset(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.scaler.fit_transform(self.dataset)

    # ------------------------------------------------------------------------------------------------------------------
    # Split into training and test data-sets.
    def create_training_and_testing_data(self):
        train_size = int(len(self.dataset) * 0.8)
        self.training_data = self.dataset[0:train_size, :]
        self.testing_data = self.dataset[train_size:len(self.dataset), :]

    # ------------------------------------------------------------------------------------------------------------------
    # Convert an array of values into two data-set matrix.
    def create_dataset_matrix(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.look_back_step - 1):
            a = dataset[i:(i + self.look_back_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back_step, 0])
        return numpy.array(dataX), numpy.array(dataY)

    # ------------------------------------------------------------------------------------------------------------------
    # Reshape into X = t
    # Reshape into Y = t + 1
    # Reshape training data to be [samples, time steps, features] format.
    def reshape_datasets_and_fit_to_model(self):
        self.trainX, self.trainY = self.create_dataset_matrix(self.training_data)
        self.testX, self.testY = self.create_dataset_matrix(self.testing_data)
        self.trainX = numpy.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.testX = numpy.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))
        self.model.fit(self.trainX, self.trainY, self.epochs, 1, 2)

    # ------------------------------------------------------------------------------------------------------------------
    # Train the model.
    def initiate_model_training(self):
        self.trainPredict = self.model.predict(self.trainX)
        self.testPredict = self.model.predict(self.testX)

    # ------------------------------------------------------------------------------------------------------------------
    # Invert prediction data to un-normalized it.
    def invert_and_denormalize_predicted_data(self):
        self.trainPredict = self.scaler.inverse_transform(self.trainPredict)
        self.trainY = self.scaler.inverse_transform([self.trainY])
        self.testPredict = self.scaler.inverse_transform(self.testPredict)
        self.testY = self.scaler.inverse_transform([self.testY])

    # ------------------------------------------------------------------------------------------------------------------
    # Calculate the root mean squared error.
    def calculate_root_mean_and_squared_error(self):
        trainScore = math.sqrt(mean_squared_error(self.trainY[0], self.trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % trainScore)
        testScore = math.sqrt(mean_squared_error(self.testY[0], self.testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % testScore)

    # ------------------------------------------------------------------------------------------------------------------
    # Save my model after training is complete.
    def save_model(self):
        self.model.save(self.model_name)
        print("Saved model named 'lstm_model.h5' in the 'predictor' package.\n")

    # ------------------------------------------------------------------------------------------------------------------
    # Shift the test predictions for plotting.
    def shift_testing_data_prediction_for_plotting(self):
        self.testPredictPlot = numpy.empty_like(self.dataset)
        self.testPredictPlot[:, :] = numpy.nan
        start = len(self.trainPredict) + (self.look_back_step * 2) + 1
        end = len(self.dataset) - 1
        self.testPredictPlot[start:end, :] = self.testPredict

    # ------------------------------------------------------------------------------------------------------------------
    # Plot baseline and predictions.
    def graph_predictions(self):
        plot.plot(self.scaler.inverse_transform(self.dataset))
        plot.plot(self.testPredictPlot)
        plot.ylabel('Price (USD)')
        plot.xlabel('Days')
        plot.title('Stock Price Predictions')
        plot.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Train the model and display a graph showing the actual vs predicted data
    def train_and_display_graph(self):
        self.load_and_summarize_model()
        self.load_data_file()
        self.normalize_dataset()
        self.create_training_and_testing_data()
        self.reshape_datasets_and_fit_to_model()
        self.initiate_model_training()
        self.invert_and_denormalize_predicted_data()
        self.calculate_root_mean_and_squared_error()
        self.save_model()
        self.shift_testing_data_prediction_for_plotting()
        self.graph_predictions()


# ----------------------------------------------------------------------------------------------------------------------
# Train the model against every data file in the "data" directory
data_directory = os.path.join('./', 'data')
file_list = os.listdir(data_directory)

for file in file_list:
    lstmModel = LstmModel(file, 0.8, 1, 1)
    lstmModel.train_and_display_graph()
