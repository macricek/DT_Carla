import numpy as np
import random
import math

# Generating new pop
def genrpop(popsize, space):

    lstring = space.shape[1]
    newPop = np.zeros((popsize, lstring))
    for r in range(0, popsize):
        for s in range(0, lstring):
            d = space[1, s] - space[0, s]
            newPop[r, s] = random.uniform(0, 1) * d + space[0, s]
            if newPop[r, s] < space[0, s]:
                newPop[r, s] = space[0, s]
            if newPop[r, s] > space[1, s]:
                    newPop[r, s] = space[1, s]

    return newPop

############ SELECTIONS ############


def selsort(pop,fitPop,N):
    lstring = pop.shape[1]
    idxs = np.argsort(fitPop)
    newPop = np.zeros((N, lstring))
    for i in range(0, N):
        newPop[i, :] = pop[idxs[0, i], :]
    return newPop


def selrand(pop, fitPop, N):
    lpop = pop.shape[0]
    lstring = pop.shape[1]

    newPop = np.zeros((N, lstring))
    for i in range(0, N):
        j = math.floor(random.uniform(0, 1) * lpop)
        newPop[i, :] = pop[j, :]
    return newPop


def seltourn(pop, fitPop, N):

    lpop = pop.shape[0]
    lstring = pop.shape[1]
    newPop = np.zeros((N, lstring))

    for i in range(0, N):
        g1 = math.floor(random.uniform(0, 1) * lpop)
        g2 = math.floor(random.uniform(0, 1) * lpop)
        if g1 == g2:
            newPop[i, :] = pop[g1, :]
        elif fitPop[0, g1] <= fitPop[0, g2]:
            newPop[i, :] = pop[g1, :]
        else:
            newPop[i, :] = pop[g2, :]
    return newPop

############ MUTATIONS ############


def mutx(pop, factor, space):

    lpop = pop.shape[0]
    lstring = pop.shape[1]
    if factor > 1:
        factor = 1
    if factor < 0:
        factor = 0

    numOfToBeMutated = math.floor(random.uniform(0, 1) * lpop * lstring * factor)

    if numOfToBeMutated == 0:
        return pop

    newPop = np.copy(pop)

    for i in range(0, numOfToBeMutated):
        r = math.floor(random.uniform(0, 1) * lpop)
        s = math.floor(random.uniform(0, 1) * lstring)
        d = space[1, s] - space[0, s]
        newPop[r, s] = random.uniform(0, 1) * d + space[0, s]
        if newPop[r, s] < space[0, s]:
            newPop[r, s] = space[0, s]
        if newPop[r, s] > space[1, s]:
            newPop[r, s] = space[1, s]
    return newPop


def muta(pop, factor, amps, space):

    lpop = pop.shape[0]
    lstring = pop.shape[1]
    if factor > 1:
        factor = 1
    if factor < 0:
        factor = 0

    numOfToBeMutated = math.floor(random.uniform(0, 1) * lpop * lstring * factor)

    if numOfToBeMutated == 0:
        return pop

    newPop = np.copy(pop)

    for i in range(0, numOfToBeMutated):
        r = math.floor(random.uniform(0, 1) * lpop)
        s = math.floor(random.uniform(0, 1) * lstring)
        newPop[r, s] = pop[r, s] + random.uniform(-1, 1) * amps[0, s]
        if newPop[r, s] < space[0, s]:
            newPop[r, s] = space[0, s]
        if newPop[r, s] > space[1, s]:
            newPop[r, s] = space[1, s]
    return newPop

############ CROSSOVERS ############


def around(pop, typeOfSelection, alfa, Space):

    lpop = pop.shape[0]
    lstring = pop.shape[1]
    newPop = np.copy(pop)
    flag = np.zeros((1, lpop))
    number = math.floor(lpop/2)
    posG1 = 0
    posG2 = 0
    m = 0
    for i in range(0, number):
        if not typeOfSelection:
            while flag[0, posG1]:
                posG1 += 1
            flag[0, posG1] = 1 #prvy rodic

            posG2 = math.floor(lpop * random.uniform(0, 1))
            while flag[0, posG2]:
                posG2 = math.floor(lpop * random.uniform(0, 1))
            flag[0, posG2] = 2

        else:
            posG1 = 2 * i - 1
            posG2 = posG1 + 1

        b = np.zeros((1, lstring))
        d = np.zeros((1, lstring))
        c = np.zeros((1, lstring))

        for k in range(0, lstring):
            b = min(pop[posG1, k], pop[posG2, k])
            d = max(pop[posG1, k], pop[posG2, k]) - b
            c = (pop[posG1, k] + pop[posG2, k]) / 2
            npop = c + alfa * random.uniform(-1, 1) * d / 2
            if npop < Space[0, k]:
                npop = Space[0, k]
            if npop > Space[1, k]:
                npop = Space[1, k]
            newPop[m, k] = npop

            npop = c + alfa * random.uniform(-1, 1) * d / 2
            if npop < Space[0, k]:
                npop = Space[0, k]
            if npop > Space[1, k]:
                npop = Space[1, k]
            newPop[m+1, k] = npop
        m += 2
    return newPop


def createGenetic():
    popsize = 50
    zeros = np.zeros((1, 3))
    ones = np.ones((1, 3))
    space = np.concatenate((zeros, ones), axis=0)
    pop = genrpop(popsize, space)
    test = seltourn(pop, np.ones((1, popsize)), 6)
    mutatedTest = muta(test, 0.5, ones*0.1, space)
    arounded = around(test, False, 1.25, space)
    print(test - arounded)

