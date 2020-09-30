import numpy as np
import pandas as pd
import random
import math, time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

# store data
filename = 'diabetes.csv'
data = pd.read_csv(filename)
# print(data.info(),data.head())

# replace any missing values

data.loc[data['Insulin'] == 0, 'Insulin'] = data['Insulin'].mean()
data.loc[data['Glucose'] == 0, 'Glucose'] = data['Glucose'].mean()
data.loc[data['BloodPressure'] == 0, 'BloodPressure'] = data['BloodPressure'].mean()
data.loc[data['SkinThickness'] == 0, 'SkinThickness'] = data['SkinThickness'].mean()
data.loc[data['BMI'] == 0, 'BMI'] = data['BMI'].mean()

# Normalize the data by dividing each value by the max value in the column
data = data / data.max()


dataset = data.values
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)
fold_no = 1

cvscores = []
neuralnetworkresults = open('mainNN.txt', 'w')
nnstart = time.time()
for train, test in kfold.split(X, Y):
    #nnstart = time.time()
    model = Sequential([
        Dense(12, activation='relu', input_shape=(8,)),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(14, activation='relu'),
        Dense(1, activation='sigmoid')])

    # Compile the model
    model.compile(
        optimizer='Adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    # Generate a print
    # print('------------------------------------------------------------------------')
    # print(f'Training for fold {fold_no} ...')

    # Train the model
    hist = model.fit(X[train], Y[train], batch_size=57, epochs=500, validation_split=0.2)

    # Generate generalization metrics
    scores = model.evaluate(X[test], Y[test], verbose=0)  # This is where the test for this fold happens
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # evaluate the model

    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # cvscores.append(scores[1] * 100)

    # Increase fold number
    fold_no = fold_no + 1
nnexecutiontime = time.time() - nnstart
neuralnetworkresults.write('Total Time taken for Neural Network is : ' + str(nnexecutiontime) + "\n")
totalAcc = 0

for i in acc_per_fold:
    totalAcc = totalAcc + i
averageAcc = totalAcc / 10

print("Average Accuracy: ", averageAcc, "%")

neuralnetworkresults.write("Average Accuracy for the Neural Network is : " + str(averageAcc))
neuralnetworkresults.close()