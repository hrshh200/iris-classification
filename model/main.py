
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('IRIS.csv') #Collecting the dataset
X = dataset.drop(columns='species')
Y = dataset['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

models = RandomForestClassifier()
models.fit(X_train, Y_train)

model_prediction_train = models.predict(X_train)
accuracy_train = accuracy_score(model_prediction_train, Y_train)

model_prediction_test = models.predict(X_test)
accuracy_test = accuracy_score(model_prediction_test, Y_test)


input_data = (6.5,2.8,4.6,1.5) # One sample input dataset whose target value should be 1
input_array = np.asarray(input_data)
input_array_reshaped = input_array.reshape(1,-1)

prediction = models.predict(input_array_reshaped)
if prediction == 'Iris-virginica':
    print("The flower is Iris-Virginica")
elif prediction == 'Iris-setosa':
    print("The flower is Iris-Setosa")
else:
    print("The flower is Iris-versicolor")
