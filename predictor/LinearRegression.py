import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.externals import joblib
import matplotlib.pyplot as plot

# ----------------------------------------------------------------------------------------------------------------------
# Load the data-set.
filePath = os.path.join('..', 'all_data', 'Stocks', 'wmt.us.txt')
data = pd.read_csv(filePath, sep=",")
data = data[["Open", "High", "Low", "Close", "Volume"]]
print(data.head(), '\n')

# ----------------------------------------------------------------------------------------------------------------------
# --- Also known as a label(s)
predict = "Close"

# --- Returns new data frame. This is all of our attributes.
attributes = np.array(data.drop([predict], 1))

# --- This is all of out labels.
labels = np.array(data[predict])

attributes_train, \
attributes_test, \
labels_train, \
labels_test = sklearn.model_selection.train_test_split(attributes, labels, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(attributes_train, labels_train)
accuracy = model.score(attributes_test, labels_test)

print("Accuracy: ", accuracy)
print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_, '\n')

predictions = model.predict(attributes_test)
percentages = []
totalPercentages = 0
print("Index => Prediction => => Actual => Accuracy %")

for x in range(len(predictions)):
    percentage = round(((predictions[x] / labels_test[x]) * 100), 2)
    percentages.append(percentage)
    totalPercentages = totalPercentages + percentage
    # print(x, ' => ', predictions[x], " => ", labels_test[x], " => ", percentage, "%")

print("\nAverage Accuracy % => ", round((totalPercentages / len(predictions)), 2), "%")

# ----------------------------------------------------------------------------------------------------------------------
# Save my model after training is complete.
joblib.dump(model, 'lr_model.sav')
print("Saved model named 'lr_model.sav' in the 'predictor' package.")

# ----------------------------------------------------------------------------------------------------------------------
# Plot baseline and predictions.
plot.plot(percentages)
plot.ylabel('Accuracy %')
plot.xlabel('Data Points')
plot.title('Stock Price Predictions')
plot.show()
