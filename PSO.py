import csv
import random
import math
import operator
from sklearn.model_selection import StratifiedKFold

def loadDataset(filename, split, dataset=[], trainingIdx=[], testIdx=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset[:] = list(lines)
        X = []
        Y = []
        # Convert String to Float
        for x in range(len(dataset)):
            for y in range(len(dataset[x])-1):
                dataset[x][y] = float(dataset[x][y])

        # Normalize
        # Calculate min and max for each column
        minmax = dataset_minmax(dataset)
        # Normalize columns
        normalize_dataset(dataset, minmax)
        # Split Classifier with others,X = Others, Y = Classifiers,
        for x in range(len(dataset)):
            X.append(dataset[x][:-1])
            Y.append(dataset[x][-1])
        # Get Idx of training and test set with StratifiedKFold

        kf = StratifiedKFold(n_splits=split)
        for train, test in kf.split(X, Y):
            trainingIdx.append(list(train))
            testIdx.append(list(test))

# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])-1):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def fitFunc(xVals):
    if xVals[1] == 0:
        xVals[1] = random.uniform(0.1,5)
    F = math.sqrt(xVals[0])-(1/xVals[1])
    return F


def initPosition(nParticles, nDimensions, xMin, xMax):
    Pos = [[random.uniform(xMin,xMax) for i in range(0, nDimensions)]
           for p in range(0, nParticles)]

    return Pos


def updatePosition(Pos, nParticles, nDimensions, xMin, xMax, V):

    for p in range(0, nParticles):
        for i in range(0, nDimensions):

            Pos[p][i] = Pos[p][i] + V[p][i]

            if Pos[p][i] > xMax:
                Pos[p][i] = xMax
            if Pos[p][i] < xMin:
                Pos[p][i] = xMin


def initVelocity(nParticles, nDimensions, vMin, vMax):
    V = [[random.uniform(-1,1) for i in range(0, nDimensions)]
         for p in range(0, nParticles)]

    return V


def updateVelocity(Pos, V, nParticles, nDimensions, vMin, vMax, k, pBestPos, gBestPos, c1, c2):

    for p in range(0, nParticles):
        for i in range(0, nDimensions):

            r1 = random.random()
            r2 = random.random()

            V[p][i] = k * (V[p][i] + r1*c1*(pBestPos[p][i]-Pos[p][i])
                           + r2*c2*(gBestPos[i] - Pos[p][i]))

def updateFitness(Pos, F, nParticles, pBestPos, pBestValue, gBestPos, gBestValue):

    for p in range(0, nParticles):
        F[p] = fitFunc(Pos[p])

        if F[p] > gBestValue:
            gBestValue = F[p]
            gBestPos = Pos[p]

        if F[p] > pBestValue[p]:
            pBestValue[p] = F[p]
            pBestPos[p] = Pos[p]
    
    return gBestValue,gBestPos


def main():
    dataset = []
    trainingIdx = []
    testIdx = []
    kfold = 2
    loadDataset('winequality-red.csv', kfold,dataset, trainingIdx, testIdx)
    # cara pake knn
    # KNN(kfold,dataset,trainingIdx,testIdx)

    nParticles = 3
    nDimensions = 2
    nIterations = 3
    # w = 1
    c1, c2 = 2.05, 2.05

    phi = c1+c2
    k = 2.0/abs(2.0-phi-math.sqrt(pow(phi, 2)-4*phi))

    xMin, xMax = 0, 5
    vMin, vMax = -xMin, xMax

    gBestValue = 0.0
    pBestValue = [0.0] * nParticles

    pBestPos = [[0.0]*nDimensions] * nParticles
    gBestPos = [0.0] * nDimensions

    history = []

    Pos = initPosition(nParticles, nDimensions, xMin, xMax)
    V = initVelocity(nParticles, nDimensions, vMin, vMax)
    F = [fitFunc(Pos[p]) for p in range(0, nParticles)]

    for _ in range(0, nIterations):       
    
        gBestValue,gBestPos = updateFitness(
            Pos, F, nParticles, pBestPos, pBestValue, gBestPos, gBestValue)

        print(gBestValue,gBestPos)
        history.append(gBestValue)

        updateVelocity(Pos, V, nParticles, nDimensions,
                       vMin, vMax, k, pBestPos, gBestPos, c1, c2)

        updatePosition(Pos, nParticles, nDimensions, xMin, xMax, V)

    nomor = 0
    for h in history:
        print(nomor, h)
        nomor += 1


main()
