import numpy as np
import time
import genetic
from neuralNetwork import NeuralNetwork as NN
from PyQt5.QtCore import QObject
import matplotlib as plt


class NeuroEvolution(QObject):
    '''
    Processing whole learning phase
    '''
    nBest: int
    nWork1: int
    nWork2: int
    nGenerate: int

    def __init__(self, nnConfig: dict):
        super(NeuroEvolution, self).__init__()

        # neural network params
        nInputs = int(nnConfig.get("ninput"))
        nHiddenOne = int(nnConfig.get("nhidden"))
        nHidden = []
        hiddenLayers = int(nnConfig.get("nhiddenlayers"))
        for _ in range(hiddenLayers):
            nHidden.append(nHiddenOne)
        nHidden = np.asarray(nHidden)
        nOutputs = int(nnConfig.get("noutput"))
        self.nn = NN(nInputs, nHidden, nOutputs)
        self.w, self.b = self.nn.getNumOfNeededElements()
        self.numParams = self.w + self.b

        # genetic algorithm params
        self.popSize = int(nnConfig.get("popsize"))
        self.numCycle = int(nnConfig.get("numcycles"))
        self.calculateParamsOfGeneticAlgorithm(nnConfig)
        bottom = np.ones((1, self.numParams)) * -3
        upper = np.ones((1, self.numParams)) * 3
        self.space = np.concatenate((bottom, upper), axis=0)
        self.amp = upper * 0.1
        initSpace = self.space * 0.1

        # initial params
        self.pop = genetic.genrpop(self.popSize, initSpace)
        self.minFit = []
        self.fit = np.zeros((1, 50))
        self.cycle = 0

    def fitness(self):
        self.minFit.append(min(self.fit))

    def perform(self):
        Best = genetic.selsort(self.pop, self.fit, 1)
        BestPop = genetic.selsort(self.pop, self.fit, self.nBest)
        WorkPop1 = genetic.selrand(self.pop, self.fit, self.nWork1)
        WorkPop2 = genetic.seltourn(self.pop, self.fit, self.nWork2)
        NPop = genetic.genrpop(self.nGenerate, self.space)

        SortPop = genetic.around(WorkPop1, 0, 1.25, self.space)
        WorkPop = genetic.mutx(WorkPop2, 0.15, self.space)
        BestPop = genetic.muta(BestPop, 0.01, self.amp, self.space)
        self.pop = np.concatenate((Best, SortPop, BestPop, WorkPop, NPop), axis=0)

        self.fitness()

    def calculateParamsOfGeneticAlgorithm(self, conf: dict):
        '''
        Loads config and sets values of pop
        :return:Nothing
        '''
        bestPecentage = float(conf.get("best"))
        work1Percentage = float(conf.get("work1"))
        work2Percentage = float(conf.get("work2"))

        popSizeWithoutBest = self.popSize - 1
        self.nBest = int(bestPecentage * popSizeWithoutBest)
        self.nWork1 = int(work1Percentage * popSizeWithoutBest)
        self.nWork2 = int(work2Percentage * popSizeWithoutBest)
        self.nGenerate = popSizeWithoutBest - self.nBest - self.nWork1 - self.nWork2
