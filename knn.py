# Example of kNN implemented from Scratch in Python
import csv
import random
import math
import operator
from sklearn.model_selection import StratifiedKFold

class KNN:
    def __init__(self,k,kfold,dataset,trainingIdx,testIdx):
        self.k = k
        self.kfold = kfold
        self.dataset = dataset
        self.trainingIdx = trainingIdx
        self.testIdx = testIdx


    class Distance:
        def euclideanDistance(self, instance1, instance2, length):
            distance = 0
            for x in range(length):
                distance += pow((instance1[x] - instance2[x]), 2)
            return math.sqrt(distance)

        def manhattanDistance(self, instance1, instance2, length):
            distance = 0
            for x in range(length):
                distance += abs(instance1[x] - instance2[x])
            return distance

        def cosineSimilarity(self, instance1, instance2, length):
            distance, sumv1, sumv2, sumv1v2 = 0, 0, 0, 0
            for i in range(length):
                x = instance1[i]
                y = instance2[i]
                sumv1 += x * x
                sumv2 += y * y
                sumv1v2 += x * y
            distance = sumv1v2 / (math.sqrt(sumv1) * math.sqrt(sumv2))
            return 1 - distance

        def minkowski_distance(self, p, q, n):
            return sum([abs(x-y) ** n for x, y in zip(p[:-1], q[:-1])]) ** 1/n

    def getDataset(self,dataset, trainingIdx, testIdx, trainingSet=[], testSet=[]):
        for i in trainingIdx:
            trainingSet.append(dataset[i])
        for i in testIdx:
            testSet.append(dataset[i])

    def getNeighbors(self,trainingSet, testInstance, k, mode=1, r=1):
        distances = []
        length = len(testInstance)-1

        for x in range(len(trainingSet)):
            d = self.Distance()
            if mode == 1:
                dist = d.euclideanDistance(testInstance, trainingSet[x], length)
            elif mode == 2:
                dist = d.manhattanDistance(testInstance, trainingSet[x], length)
            elif mode == 3:
                dist = d.cosineSimilarity(testInstance, trainingSet[x], length)
            elif mode == 4:
                dist = d.minkowski_distance(testInstance, trainingSet[x], r)

            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1), reverse=False)
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors

    def getResponse(self,neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(),
                            key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]


    def getAccuracy(self,testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            # print(testSet[x][-1],predictions[x])
            if testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0


    def main(self):
        # prepare data
        totalAccuracy = 0
        inDist = 1

        for i in range(self.kfold):
            trainingSet = []
            testSet = []
            self.getDataset(self.dataset, self.trainingIdx[i], self.testIdx[i], trainingSet, testSet)
            print('Train set: ' + str(len(trainingSet)))
            print('Test set: ' + str(len(testSet)))

            predictions = []
            for x in range(len(testSet)):
                neighbors = self.getNeighbors(trainingSet, testSet[x], self.k, mode=inDist)
                result = self.getResponse(neighbors)
                predictions.append(result)

            accuracy = self.getAccuracy(testSet, predictions)
            totalAccuracy += accuracy
            print('Accuracy: ' + str(accuracy) + '%')

        avgAcc = totalAccuracy/self.kfold
        print('\nTotal Accuracy: ' + str(avgAcc) + '%')
        return avgAcc
