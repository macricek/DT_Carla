import matplotlib.pyplot as plt
import numpy as np


def plotFastAIComparation():
    x = np.arange(1, 11)

    yRes = np.array([0.872, 0.902, 0.914, 0.92, 0.899, 0.9106, 0.922, 0.931, 0.934, 0.939])
    lossRes = 1 - yRes

    yFastAI = np.array([0.91, 0.917, 0.917, 0.91, 0.932, 0.924, 0.934, 0.946, 0.9471, 0.951])
    lossFastAI = 1 - yFastAI

    plot1 = plt.figure(1)
    plt.plot(x, yRes, label='ResNet34')
    plt.plot(x, yFastAI, label='FastAI')
    plt.legend()
    plt.title("Segmentation DiceMulti by epoch")
    plt.xlabel("Epochs")
    plt.ylabel("DiceMulti [0-1]")
    plt.xlim([1, 10])


def plotGeneticResults(numRevision):
    evolFile = f'results/{numRevision}/evol.csv'
    weightsFile = f'results/{numRevision}/best.csv'
    evol = np.loadtxt(evolFile, delimiter=',')
    weights = np.loadtxt(weightsFile, delimiter=',')

    plt.figure(0)
    plt.plot(evol)
    plt.title("Priebeh evolúcie fitness funkcie")
    plt.xlabel("Cykly")
    plt.ylabel("Hodnota fitness funkcie")
    plt.savefig(f'results/{numRevision}/genetic.png')

    plt.figure(1)
    biases = weights[210:-1]
    weights = weights[0:210]
    plt.plot(weights, '*', label='Váhy')
    plt.plot(biases, 'o', label='Biasy')
    plt.legend()
    plt.title("Vizualizácia váh")
    plt.xlabel("Poradové číslo váhy/biasu")
    plt.ylabel("Hodnota váhy/biasu")
    plt.savefig(f'results/{numRevision}/visWB.png')


if __name__ == '__main__':
    plotGeneticResults(61)
    plt.show()


