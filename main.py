from carlaDP import CarlaCar
import genetic
import numpy as np
from neuralNetwork import NeuralNetwork as NN


def test_genetic():
    genetic.createGenetic()


def test_neural():
    input = 10
    output = 2
    hidden = np.array((100, 100))
    nn = NN(input, hidden, output, None)
    a, b = nn.getNumOfNeededElements()
    zeros = np.ones((1, a+b))*-3
    ones = np.ones((1, a+b))*3
    space = np.concatenate((zeros, ones), axis=0)
    testW = genetic.genrpop(1, space)
    nn.setWeights(testW)
    nn.run(np.linspace(0, 255, 10))


def main():
    env = CarlaCar(True)
    env.__del__()


if __name__ == '__main__':
    test_neural()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
