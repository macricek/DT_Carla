from CarlaEnvironment import CarlaEnvironment
import genetic
import numpy as np
import time
import sys
from neuralNetwork import NeuralNetwork as NN

def fitness():
    fit = np.zeros((1, 50))
    for i in range(0, 50):
        fit[0, i] = 0
    return fit

#GLOBAL CONSTANTS
IM_WIDTH = 28
IM_HEIGHT = 28

#Parameters for Genetic algorithm
popSize = 50
numCycle = 10

#Parameters for Neural Network
inputLayer = IM_HEIGHT * IM_WIDTH
hiddenLayer = np.array((10, 50))
outputLayer = 2

#init NeuralNetwork and get needed number of elements
nn = NN(inputLayer, hiddenLayer, outputLayer)
a, b = nn.getNumOfNeededElements()

#init genetic algorithm
bottom = np.ones((1, a + b)) * -3
upper = np.ones((1, a + b)) * 3
space = np.concatenate((bottom, upper), axis=0)
initSpace = space * 0.1
amp = upper * 0.1
start = time.time()
fit = fitness()
pop = genetic.genrpop(popSize, initSpace)
Best = genetic.selsort(pop, fit, 1)              # the best                                        [1]
BestPop = genetic.selsort(pop, fit, 9)           # nine best                                      [10]
WorkPop1 = genetic.selrand(pop, fit, 15)         # workPop                                        [25]
WorkPop2 = genetic.seltourn(pop, fit, 15)        # work2                                          [40]
NPop = genetic.genrpop(10, space)                                                                #[50]
SortPop = genetic.around(WorkPop1, 0, 1.25, space)
WorkPop = genetic.mutx(WorkPop2, 0.15, space)
BestPop = genetic.muta(BestPop, 0.01, amp, space)
pop = np.concatenate((Best, SortPop, BestPop, WorkPop, NPop), axis=0)

end = time.time()
print(end - start)


def main():
    cc = CarlaEnvironment(1, False)


if __name__ == '__main__':
    main()
    print("It works!")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
