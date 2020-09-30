import numpy as np
import pandas as pd
import random
import math
import time
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

# PSO parameters
inertiaW = 0.9
popSize = 20
maxIteration = 30
C1 = 2
C2 = 2
WMAX = 0.9
WMIN = 0.4
best_accuracy = -1
noFeatureSelected = -1
average_accuracy = 0
best_feature_vector = []
accuracy_list = []
features_list = []


def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    else:
        return 1 / (1 + math.exp(-x))


def fitness(position, trainX, trainy, testX, testy):
    indexes = []
    for i in range(position.shape[0]):
        if position[i] == 1:
            indexes.append(i)

    if np.shape(indexes)[0] == 0:
        return 1

    classifier = KNeighborsClassifier(n_neighbors=8)
    train_data = trainX[:, indexes]
    test_data = testX[:, indexes]
    classifier.fit(train_data, trainy)
    error = 1 - classifier.score(test_data, testy)

    return inertiaW * error + (1 - inertiaW) * (np.int_(sum(position)) / np.shape(position)[0])


# initalize population by choosing random subset k to be selected feature
def initialize(popSize, dim):
    population = np.zeros((popSize, dim))

    for i in range(popSize):
        no = random.randint(1, 6)
        pos = random.sample(range(0, dim - 1), no)
        for j in pos:
            population[i][j] = 1
    return population


def PSO(popSize, maxIteration, filename):

    # store data
    df = pd.read_csv(filename)

    # replace any missing values
    df.loc[df['Insulin'] == 0, 'Insulin'] = df['Insulin'].mean()
    df.loc[df['Glucose'] == 0, 'Glucose'] = df['Glucose'].mean()
    df.loc[df['BloodPressure'] == 0, 'BloodPressure'] = df['BloodPressure'].mean()
    df.loc[df['SkinThickness'] == 0, 'SkinThickness'] = df['SkinThickness'].mean()
    df.loc[df['BMI'] == 0, 'BMI'] = df['BMI'].mean()

    # Normalize the data by dividing each value by the max value in the column
    df = df / df.max()

    (a, b) = np.shape(df)

    trainX, testX, trainy, testy = train_test_split(df.values[:, 0:b - 1], df.values[:, b - 1], stratify=df.values[:, b - 1], test_size=0.2)
    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(trainX, trainy)
    accuracy = classifier.score(testX, testy)

    population = initialize(popSize, np.shape(df.values[:, 0:b - 1])[1])
    velocity = np.zeros((popSize, np.shape(df.values[:, 0:b - 1])[1]))

    gbestValue = 1000
    gbestVec = np.zeros(np.shape(population[0])[0])

    pbestValues = np.zeros(popSize)
    pbestVec = np.zeros(np.shape(population))

    for i in range(popSize):
        pbestValues[i] = 1000

    for currentIteration in range(maxIteration):
        newpopulation = np.zeros((popSize, np.shape(df.values[:, 0:b - 1])[1]))

        # set fitness for population
        accuracy = np.zeros(np.shape(population)[0])
        for i in range(popSize):
            accuracy[i] = fitness(population[i], trainX, trainy, testX, testy)
        fitnessPopList = accuracy

        # update pbest
        for i in range(popSize):
            if fitnessPopList[i] < pbestValues[i]:
                pbestValues[i] = fitnessPopList[i]
                pbestVec[i] = population[i].copy()
                print("pbest updated")

        # update gbest
        for i in range(popSize):
            if fitnessPopList[i] < gbestValue:
                gbestValue = fitnessPopList[i]
                gbestVec = population[i].copy()

        print("gbest: ", gbestValue,  np.int_(sum(gbestVec)))

        # update W
        W = WMAX - (currentIteration / maxIteration) * (WMAX - WMIN)

        for i in range(popSize):
            r1 = C1 * random.random()
            r2 = C2 * random.random()

            # update velocity
            velocity[i] = np.multiply(W, velocity[i]) + np.multiply(r1, (np.subtract(pbestVec[i], population[i]))) + np.multiply(r2, (np.subtract(gbestVec, population[i])))

            # update position
            newpopulation[i] = np.add(population[i], velocity[i])

            position1, position2 = np.array([]), np.array([])
            for j in range(np.shape(df.values[:, 0:b - 1])[1]): #dimension
                thresh = sigmoid(newpopulation[i][j])

                if thresh > 0.5:
                    position1 = np.append(position1, 1)
                else:
                    position1 = np.append(position1, 0)

                thresh = -sigmoid(newpopulation[i][j])
                if thresh > 0.5:
                    position2 = np.append(position2, 1)
                else:
                    position2 = np.append(position2, 0)

            position1fitness = fitness(position1, trainX, trainy, testX, testy)
            position2fitness = fitness(position2, trainX, trainy, testX, testy)

            if position1fitness < position2fitness:
                newpopulation[i] = position1.copy()
            else:
                newpopulation[i] = position2.copy()

        population = newpopulation.copy()

    gbestVecOutput = gbestVec.copy()

    indexes = []
    for i in range(gbestVecOutput .shape[0]):
        if gbestVecOutput [i] == 1:
            indexes.append(i)
    X_test = testX[:, indexes]
    X_train = trainX[:, indexes]
    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(X_train, trainy)
    accuracy = classifier.score(X_test, testy)

    return accuracy, gbestVecOutput


outfile = open('PSO_best_Features.txt', 'w')
outfile.write('|Best Accuracy      |Best Vector Features      |Best No Features |Time                 |\n')

for i in range(10):
    start = time.time()

    accuracy, gbestVec = PSO(popSize, maxIteration, filename)
    accuracy_list.append(accuracy)
    features_list.append(np.int_(sum(gbestVec)))

    if (accuracy == best_accuracy) and (np.int_(sum(gbestVec)) < noFeatureSelected):
        best_accuracy = accuracy
        best_feature_vector = gbestVec
        noFeatureSelected = np.int_(sum(gbestVec))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_feature_vector = gbestVec
        noFeatureSelected = np.int_(sum(gbestVec))

    execution_time = time.time() - start

    # write to file
    outfile.write(
        '|' + str(best_accuracy) + " " * (19 - len(str(best_accuracy))) + "|" + str(best_feature_vector) + " |" + str(
            noFeatureSelected) + " " * (17 - len(str(noFeatureSelected))) + "|" + str(execution_time) + " " * (
                    21 - len(str(execution_time))) + "|\n")

print('best: ', best_accuracy, noFeatureSelected, best_feature_vector)

outfile.close()

# new dataset
features_name = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                 "DiabetesPedigreeFunction", "Age"]
new_Features = []
c = 0  # count
for i in best_feature_vector:
    if i == 1:
        new_Features.append(features_name[c])
    # print(c)
    c = c + 1

new_Features.append("Outcome")
new_dataset = pd.read_csv("diabetes.csv", usecols=[i for i in new_Features])
# print(new_dataset.info(), new_dataset.head())

(a, b) = np.shape(new_dataset)
# print(a, b)
X = new_dataset.values[:, 0:b - 1]
Y = new_dataset.values[:, b - 1]

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)
fold_no = 1


# print("done")

cvscores = []
neuralnetworkresults = open('neuralnetworkresultsPSO.txt', 'w')

input_features_amount = noFeatureSelected
nnstart = time.time()
for train, test in kfold.split(X, Y):
    model = Sequential([
        Dense(12, activation='relu', input_shape=(noFeatureSelected,)),
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

    #nnstart = time.time()
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

