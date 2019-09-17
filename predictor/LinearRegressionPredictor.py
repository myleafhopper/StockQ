import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

data = pd.read_csv("data.csv", sep=",")
print(data.head(), '\n')

data = data[["open", "high", "low", "close", "volume"]]
print(data.head(), '\n')

# --- Also known as a label(s)
predict = "close"

# --- Returns new data frame. This is all of our attributes.
attributes = np.array(data.drop([predict], 1))

# --- This is all of out labels.
labels = np.array(data[predict])

attributes_train, \
attributes_test, \
labels_train, \
labels_test = sklearn.model_selection.train_test_split(attributes, labels, test_size = 0.1)

linear = linear_model.LinearRegression()
linear.fit(attributes_train, labels_train)
accuracy = linear.score(attributes_test, labels_test)

print("Accuracy: ", accuracy)
print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_, '\n')

predictions = linear.predict(attributes_test)

print("Index => Prediction => => Actual => Accuracy %")
for x in range(len(predictions)):
    percentage = round(((predictions[x] / labels_test[x]) * 100), 2)
    print(x, ' => ', predictions[x], " => ", labels_test[x], " => ", percentage, "%")
