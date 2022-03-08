import numpy as np
import time

import Vehicle
import genetic
from neuralNetwork import NeuralNetwork as NN
from PyQt5.QtCore import QObject
import matplotlib.pyplot as plt


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

        # for saving
        self.base = str(nnConfig.get("base"))
        self.rev = str(nnConfig.get("rev"))
        self.fileBest = self.base + f'best{self.rev}.csv'
        self.fileGraph = self.base + f"GA{self.rev}.png"

        # initial params
        self.pop = genetic.genrpop(self.popSize, initSpace)
        self.minFit = []
        self.fit = np.ones((1, self.popSize)) * np.inf

    def singleFit(self, vehicle: Vehicle.Vehicle):
        '''
        Calculate fit value of single vehicle solution
        :param vehicle: Vehicle object
        :return: Nothing
        '''
        at = vehicle.vehicleID
        crossings, errDec, collisions, penalty = vehicle.record()
        self.fit[0, at] = crossings * 2 + errDec * 0.1 + collisions * 10000 + penalty

    def perform(self):
        self.minFit.append(np.min(self.fit))

        Best = genetic.selsort(self.pop, self.fit, 1)
        BestPop = genetic.selsort(self.pop, self.fit, self.nBest)
        WorkPop1 = genetic.selrand(self.pop, self.fit, self.nWork1)
        WorkPop2 = genetic.seltourn(self.pop, self.fit, self.nWork2)
        NPop = genetic.genrpop(self.nGenerate, self.space)

        SortPop = genetic.around(WorkPop1, 0, 1.25, self.space)
        WorkPop = genetic.mutx(WorkPop2, 0.15, self.space)
        BestPop = genetic.muta(BestPop, 0.01, self.amp, self.space)
        self.pop = np.concatenate((Best, SortPop, BestPop, WorkPop, NPop), axis=0)

    def getNeuralNetwork(self, at):
        '''
        Set weights to NN at argument
        :param at: which part of population will fill the weights
        :return: NN object
        '''
        weights = self.pop[at, :]
        self.nn.setWeights(weights)
        return self.nn

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

    def plotEvolution(self):
        plot = plt.figure(1)
        plt.plot(np.asarray(self.minFit))
        plt.title("Priebeh evolúcie fitness funkcie")
        plt.xlabel("Cykly")
        plt.ylabel("Hodnota fitness funkcie")
        plt.savefig(self.fileGraph)
        plt.show()

    def finishNeuroEvolutionProcess(self):
        Best = genetic.selsort(self.pop, self.fit, 1)
        np.savetxt(self.fileBest, Best, delimiter=',')
        #data = np.loadtxt('data.csv', delimiter=',') // LOADING afterwards
        self.plotEvolution()
