import random
import math


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

            # if V[p][i] > vMax:
            #     V[p][i] = vMax
            # if V[p][i] < vMin:
            #     V[p][i] = vMin


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
    
    print('1.------------------------')
    print(Pos)
    print('---')
    print(V)
    print('---')
    print(F)
    print('-------------------------')

    for j in range(0, nIterations):
        print('2.------------------------')
       
        
        gBestValue,gBestPos = updateFitness(
            Pos, F, nParticles, pBestPos, pBestValue, gBestPos, gBestValue)

        print(gBestValue,gBestPos)
        history.append(gBestValue)

        updateVelocity(Pos, V, nParticles, nDimensions,
                       vMin, vMax, k, pBestPos, gBestPos, c1, c2)
        print('---')
        print(V)
        print('---')
        updatePosition(Pos, nParticles, nDimensions, xMin, xMax, V)
        print(Pos)
        print('-------------------------')
    nomor = 0
    for h in history:
        print(nomor, h)
        nomor += 1


main()
