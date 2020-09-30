# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 20:47:27 2020

@author: shane
Genetic Algorithm for Feature Selection
"""
import numpy as np
import pandas as pd
import random
import math, time
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# ==================================================================
# Intializes the Popuplation
# Creates a 10 arrays with randomed 0's or 1's
def initialize(popSize, dim):
    population = np.zeros((popSize, dim))
    minn = 1
    maxx = math.floor(0.8 * dim)  # 6

    for i in range(popSize):
        random.seed(i ** 3 + 10 + time.time())
        no = random.randint(minn, maxx)
        if no == 0:
            no = 1
        random.seed(time.time() + 100)
        pos = random.sample(range(0, dim - 1), no)
        for j in pos:
            population[i][j] = 1

    return population


def fitness(solution, trainX, trainy, testX, testy):
    cols = np.flatnonzero(solution)
    val = 1
    if np.shape(cols)[0] == 0:
        return val
    clf = KNeighborsClassifier(n_neighbors=5)
    train_data = trainX[:, cols]
    test_data = testX[:, cols]
    clf.fit(train_data, trainy)
    error = 1 - clf.score(test_data, testy)

    # in case of multi objective  []
    featureRatio = (solution.sum() / np.shape(solution)[0])
    val = omega * error + (1 - omega) * featureRatio
    # print(error,featureRatio,val)
    return val


def allfit(population, trainX, trainy, testX, testy):
    x = np.shape(population)[0]
    acc = np.zeros(x)
    for i in range(x):
        acc[i] = fitness(population[i], trainX, trainy, testX, testy)
    return acc


def selectParentRoulette(popSize, fitnList):
    fitnList = np.array(fitnList)
    fitnList = 1 - fitnList / fitnList.sum()
    random.seed(time.time() + 19)
    val = random.uniform(0, fitnList.sum())
    for i in range(popSize):
        if val <= fitnList[i]:
            return i
        val -= fitnList[i]
    return -1

# ==================================================================
def geneticAlgo(dataset, popSize, maxIter, randomstate):
    # --------------------------------------------------------------------
    df = pd.read_csv(dataset)

    df.loc[df['Insulin'] == 0, 'Insulin'] = df['Insulin'].mean()
    df.loc[df['Glucose'] == 0, 'Glucose'] = df['Glucose'].mean()
    df.loc[df['BloodPressure'] == 0, 'BloodPressure'] = df['BloodPressure'].mean()
    df.loc[df['SkinThickness'] == 0, 'SkinThickness'] = df['SkinThickness'].mean()
    df.loc[df['BMI'] == 0, 'BMI'] = df['BMI'].mean()

    # Normalize the data by dividing each value by the max value in the column
    df = df / df.max()

    (a, b) = np.shape(df)
    print(a, b)
    data = df.values[:, 0:b - 1]
    label = df.values[:, b - 1]
    dimension = np.shape(data)[1]  # solution dimension
    # ---------------------------------------------------------------------

    cross = 5
    test_size = (1 / cross)
    trainX, testX, trainy, testy = train_test_split(data, label, stratify=label, test_size=test_size,
                                                    random_state=randomstate)  #
    print(np.shape(trainX), np.shape(trainy), np.shape(testX), np.shape(testy))

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX, trainy)
    val = clf.score(testX, testy)

    print("Total Acc: ", val)

    population = initialize(popSize, dimension)
    GBESTSOL = np.zeros(np.shape(population[0]))
    GBESTFIT = 1000

    start_time = datetime.now()  # Not Used

    for currIter in range(1, maxIter):
        newpop = np.zeros((popSize, dimension))
        fitList = allfit(population, trainX, trainy, testX, testy)
        arr1inds = fitList.argsort()
        population = population[arr1inds]  # Ranks chromosomes according to fitness
        fitList = fitList[arr1inds]

        bestInx = np.argmin(fitList)  # 0
        fitBest = min(fitList)  # Fittest Individual in the population
        print(currIter, 'best:', fitBest, " - Chromosome is ", population[bestInx], "Number of features are : ",
              population[bestInx].sum())
        # print(population[bestInx])
        if fitBest < GBESTFIT:
            GBESTSOL = population[bestInx].copy()
            GBESTFIT = fitBest

        for selectioncount in range(int(popSize / 2)):
            parent1 = selectParentRoulette(popSize, fitList)
            parent2 = parent1
            while parent2 == parent1:
                random.seed(time.time())
                parent2 = selectParentRoulette(popSize, fitList)

            parent1 = population[parent1].copy()
            parent2 = population[parent2].copy()

            # CrossOver between parent1 and parent2
            child1 = parent1.copy()
            child2 = parent2.copy()
            for i in range(dimension):
                random.seed(time.time())
                if random.uniform(0, 1) < crossoverprob:
                    child1[i] = parent2[i]
                    child2[i] = parent1[i]
            i = selectioncount
            j = int(i + (popSize / 2))
            # print(i,j)
            newpop[i] = child1.copy()
            newpop[j] = child2.copy()

        # Mutation
        mutationprob = muprobmin + (muprobmax - muprobmin) * (currIter / maxIter)
        for index in range(popSize):
            for i in range(dimension):
                random.seed(time.time() + dimension + popSize)
                if random.uniform(0, 1) < mutationprob:
                    newpop[index][i] = 1 - newpop[index][i]
        # for i in range(popSize):
        # print('before:',newpop[i].sum(),fitList[i])
        # newpop[i],fitList[i] = adaptiveBeta(newpop[i],fitList[i],trainX,trainy,testX,testy)
        # newpop[i],fitList[i] = deepcopy(mutation(newpop[i],fitList[i],trainX,trainy,testX,testy))
        # print('after:',newpop[i].sum(),fitList[i])

        population = newpop.copy()

    cols = np.flatnonzero(GBESTSOL)
    val = 1
    if np.shape(cols)[0] == 0:
        return GBESTSOL
    clf = KNeighborsClassifier(n_neighbors=5)
    train_data = trainX[:, cols]
    test_data = testX[:, cols]
    clf.fit(train_data, trainy)
    val = clf.score(test_data, testy)
    return GBESTSOL, val


# ========================================================================================================
def newDataset(selected_features):
    df = pd.read_csv("diabetes.csv")

    # Replacing the null values with the mean
    df.loc[df['Insulin'] == 0, 'Insulin'] = df['Insulin'].mean()
    df.loc[df['Glucose'] == 0, 'Glucose'] = df['Glucose'].mean()
    df.loc[df['BloodPressure'] == 0, 'BloodPressure'] = df['BloodPressure'].mean()
    df.loc[df['SkinThickness'] == 0, 'SkinThickness'] = df['SkinThickness'].mean()
    df.loc[df['BMI'] == 0, 'BMI'] = df['BMI'].mean()

    columns_drop = []
    for i in range(0, 8):
        if (selected_features[i] == 0):
            columns_drop.append(i)

    df = df.drop(df.columns[columns_drop], 1)
    df.to_csv('NewDataset.csv')
    dataset = df.values
    X = dataset[:, 0: (8 - len(columns_drop))]
    y = dataset[:, (8 - len(columns_drop))]
    return X, y


# ========================================================================================================
def NeuralNetwork(best_no_features, X, Y):
    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)

    fold_no = 1
    # cvscores = []
    input_features_amount = best_no_features

    start2 = time.time()
    for train, test in kfold.split(X, Y):
        model = Sequential([
            Dense(12, activation='relu', input_shape=(input_features_amount,)),
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
        model.fit(X[train], Y[train], batch_size=57, epochs=500, validation_split=0.2)

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
    end2 = time.time()
    neural_network_time = end2 - start2
    NNfile.write("Total Time for the Neural Network is : " + str(neural_network_time) + "\n")
    totalAcc = 0
    for i in acc_per_fold:
        totalAcc = totalAcc + i
    averageAcc = totalAcc / 10

    print("Average Accuracy: ", averageAcc, "%")
    # Saving the model
    model.save("model.h5")
    print("Model Saved!!!")
    return averageAcc


# Main=================================================================================
popSize = 10
maxIter = 20
omega = 0.9
crossoverprob = 0.6
muprobmin = 0.01
muprobmax = 0.3
dataset = "diabetes.csv"
randomstate = 5

best_accuracy = -100
best_no_features = 100
best_answer = []
accuList = []
featList = []
columns_drop = []

GAfile = open("GeneticAlgorithm.txt", "w")
NNfile = open("NeuralNetworkGA.txt", "w")
start = time.time()
# Running the GA for 10 iterations
for count in range(10):
    starttime = time.time()
    print(count)
    answer, testAcc = geneticAlgo(dataset, popSize, maxIter, randomstate)

    print(testAcc, answer.sum())
    GAfile.write("Accuracy for Iteration " + str(count) + " is = " + str(testAcc))     #This line is giving me an error
    GAfile.write(" Best Features"+str(answer) + " ")
    accuList.append(testAcc)
    featList.append(answer.sum())

    if testAcc >= best_accuracy and answer.sum() < best_no_features:
        best_accuracy = testAcc
        best_no_features = answer.sum()
        best_answer = answer.copy()
    if testAcc > best_accuracy:
        best_accuracy = testAcc
        best_no_features = answer.sum()
        best_answer = answer.copy()
    endtime = time.time()
    executiontime = endtime - starttime
    GAfile.write("Time Taken: " + str(executiontime) + "\n")

end = time.time()
genetic_algorithm_time = end - start
GAfile.write("Best Accuracy after 10 Iterations is : " + str(best_accuracy) + "\n")
GAfile.write("Total Time Taken for the Genetic Algorithm : " + str(genetic_algorithm_time) + "\n")
GAfile.close()
print(dataset, "best:", best_accuracy, best_no_features,
      answer)  # Best_no_features gives me the wrong answer for some reason
# print(totaltime)
# End of GA

# Loop for determining the number of selected features
selected_features = answer.astype(int)
number_features = 0
for i in range(0, 8):
    if (selected_features[i] == 1):
        number_features = number_features + 1

# Get the the new dataset after dropping the colomns
NN_Train_data, NNlabel = newDataset(selected_features)  # To go into the neural network
#start2 = time.time()
# Runs the neural network
NNAccuracy = NeuralNetwork(number_features, NN_Train_data, NNlabel)
#end2 = time.time()
#neural_network_time = end2 - start2
NNfile.write("Average Accuracy for the Neural Network is : " + str(NNAccuracy) + "\n")
#NNfile.write("Total Time for the Neural Network is : " + str(neural_network_time) + "\n")
NNfile.close()
