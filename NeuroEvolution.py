import numpy as np
import time
import genetic
from neuralNetwork import NeuralNetwork as NN
from PyQt5 import QtCore
from PyQt5.QtCore import QObject
import matplotlib as plt


class NeuroEvolution(QObject):
    def __init__(self, nInputs, nHidden, nOutputs):
        super(NeuroEvolution, self).__init__()
        popSize = 50
        self.numCycle = 10
        self.nn = NN(nInputs, nHidden, nOutputs)
        self.w, self.b = self.nn.getNumOfNeededElements()
        self.numParams = self.w + self.b
        bottom = np.ones((1, self.numParams)) * -3
        upper = np.ones((1, self.numParams)) * 3
        self.space = np.concatenate((bottom, upper), axis=0)
        self.amp = upper * 0.1
        initSpace = self.space * 0.1
        self.pop = genetic.genrpop(popSize, initSpace)
        self.minFit = []

    def fitness(self):
        fit = np.zeros((1, 50))
        for i in range(0, 50):
            fit[0, i] = 0
        return fit

    def perform(self):
        for i in range(0, self.numCycle):
            Best = genetic.selsort(self.pop, fit, 1)  # the best                                             [1]
            BestPop = genetic.selsort(self.pop, fit, 9)  # nine best                                         [10]
            WorkPop1 = genetic.selrand(self.pop, fit, 15)  # workPop                                         [25]
            WorkPop2 = genetic.seltourn(self.pop, fit, 15)  # work2                                          [40]
            NPop = genetic.genrpop(10, self.space)                                                         # [50]

            fit = self.fitness()
            self.minFit.append(min(fit))

            SortPop = genetic.around(WorkPop1, 0, 1.25, self.space)
            WorkPop = genetic.mutx(WorkPop2, 0.15, self.space)
            BestPop = genetic.muta(BestPop, 0.01, self.amp, self.space)
            self.pop = np.concatenate((Best, SortPop, BestPop, WorkPop, NPop), axis=0)
