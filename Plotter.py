import matplotlib.pyplot as plt
import numpy as np
from neuralNetwork import NeuralNetwork as NN, loadNNParamsFromConfig
from CarlaConfig import CarlaConfig
from main import Results


def plotFastAIComparation():
    x = np.arange(1, 11)

    yRes = np.array([0.872, 0.902, 0.914, 0.92, 0.899, 0.9106, 0.922, 0.931, 0.934, 0.939])
    lossRes = 1 - yRes

    yFastAI = np.array([0.91, 0.917, 0.917, 0.91, 0.932, 0.924, 0.934, 0.946, 0.9471, 0.951])
    lossFastAI = 1 - yFastAI

    plot1 = plt.figure(1)
    plt.plot(x, yRes, label='Unet++')
    plt.plot(x, yFastAI, label='MobileNetV3Small')
    plt.legend()
    plt.title("Segmentation DiceMulti by epoch")
    plt.xlabel("Epochs")
    plt.ylabel("DiceMulti [0-1]")
    plt.xlim([1, 10])


def plotGeneticResults(numRevision):
    evolFile = f'results/{numRevision}/evol.csv'
    weightsFile = f'results/{numRevision}/best.csv'
    configFile = f'results/{numRevision}/config.ini'
    evol = np.loadtxt(evolFile, delimiter=',')
    weights = np.loadtxt(weightsFile, delimiter=',')
    config = CarlaConfig(path=configFile)

    plt.figure()
    plt.plot(evol)
    plt.title(f"Evolution of fitness function {numRevision.title()}")
    plt.xlabel("Cycles")
    plt.ylabel("Fitness function value")
    plt.savefig(f'results/{numRevision}/genetic.png')

    plt.figure()
    nInputs, nHidden, nOutputs = loadNNParamsFromConfig(config.loadNEData())
    nn = NN(nInputs, nHidden, nOutputs)
    numW, numB = nn.getNumOfNeededElements()

    biases = weights[numW:numB+numW]
    weights = weights[0:numW]
    plt.plot(weights, '*', label='Weights')
    plt.plot(biases, 'o', label='Biases')
    plt.legend()
    plt.title(f"Visualisation of weights/biases {numRevision.title()}")
    plt.xlabel("Number of weight/bias")
    plt.ylabel("Value of weight/bias")
    plt.savefig(f'results/{numRevision}/visWB.png')


def plotPath(numRevision, type=0):
    X = f'results/{numRevision}/X{type}.csv'
    Y = f'results/{numRevision}/Y{type}.csv'
    xNp = np.loadtxt(X, delimiter=',')
    yNp = np.loadtxt(Y, delimiter=',')

    xStart = np.array([xNp[0, 0]])
    yStart = np.array([yNp[0, 0]])
    xEnd, yEnd = getEnding(type)
    ran = xNp.shape[0]

    plt.figure()
    style = ['k', 'r--', 'b--', 'g--']
    labels = ['Real path', 'Right lane', 'Left lane', 'Optimal path']
    plt.title(f"Recorded path of vehicle [{numRevision.title()}]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xStart, yStart, 'c*', label="Start")
    plt.plot(xEnd, yEnd, 'y*', label="End")
    if type < 2:
        for i in range(ran):
            plt.plot(xNp[i, :], yNp[i, :], style[i], label=labels[i])
    else:
        plt.plot(xNp[0, :], yNp[0, :], style[0], label=labels[0])
        plt.plot(xNp[3, :], yNp[3, :], style[3], label=labels[3])
        plt.plot(xNp[1, np.where(xNp[1] != 0)], yNp[1, np.where(yNp[1] != 0)], 'r.')
        plt.plot(xNp[2, np.where(xNp[2] != 0)], yNp[2, np.where(yNp[2] != 0)], 'b.')
        plt.plot(xNp[1, 0], yNp[1, 0], 'r.', label=labels[1])
        plt.plot(xNp[2, 0], yNp[2, 0], 'b.', label=labels[2])

    plt.legend()
    plt.savefig(f'results/{numRevision}/path{type}.png')


def full(numRevision):
    plotGeneticResults(numRevision)
    plotPath(numRevision, 0)
    plotPath(numRevision, 1)
    plotPath(numRevision, 2)


def createGeneticForAllInOne():
    plt.figure()
    for mem in Results:
        if mem.value < 0:
            continue

        evolFile = f'results/{mem}/evol.csv'
        evol = np.loadtxt(evolFile, delimiter=',')

        plt.plot(evol, label=mem.title())

    plt.title(f"Comparasion of genetic algorithms")
    plt.xlabel("Cycles")
    plt.ylabel("Fitness function value")
    plt.legend()
    plt.savefig(f'results/compare.png')


def plotPathAllInOne(type):
    plt.figure()

    for mem in Results:
        if mem.value < 0:
            continue

        X = f'results/{mem}/X{type}.csv'
        Y = f'results/{mem}/Y{type}.csv'
        xNp = np.loadtxt(X, delimiter=',')
        yNp = np.loadtxt(Y, delimiter=',')

        if mem.value == 2:
            labels = [mem.title(), 'RightLane', 'LeftLine']
            for i in range(3):
                if i == 1:
                    if type == 2:
                        plt.plot(xNp[i, np.where(xNp[i] != 0)], yNp[i, np.where(yNp[i] != 0)], 'r.')
                    else:
                        plt.plot(xNp[i, :], yNp[i, :], 'r--', linewidth=1, label=labels[i])
                elif i == 2:
                    if type == 2:
                        plt.plot(xNp[i, np.where(xNp[i] != 0)], yNp[i, np.where(yNp[i] != 0)], 'b.')
                    else:
                        plt.plot(xNp[i, :], yNp[i, :], 'b--', linewidth=1, label=labels[i])
                else:
                    plt.plot(xNp[i, :], yNp[i, :], label=labels[i])

            if type == 2:
                plt.plot(xNp[1, 0], yNp[1, 0], 'r.', label=labels[1])
                plt.plot(xNp[2, 0], yNp[2, 0], 'b.', label=labels[2])
        else:
            for i in range(1):
                plt.plot(xNp[i, :], yNp[i, :], label=mem.title())

    xStart = np.array([xNp[0, 0]])
    yStart = np.array([yNp[0, 0]])

    xEnd, yEnd = getEnding(type)

    plt.title(f"Comparasion of paths of vehicles [Route {type}]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(xStart, yStart, 'c*', label="Start")
    plt.plot(xEnd, yEnd, 'y*', label="End")
    plt.legend()
    plt.savefig(f'results/compare{type}.png')


def getEnding(type):
    if type == 0:
        xEnd = np.array([-490])
        yEnd = np.array([174])
    elif type == 1:
        xEnd = np.array([211.2])
        yEnd = np.array([-392.1])
    elif type == 2:
        xEnd = np.array([220.5])
        yEnd = np.array([-169.5])

    return xEnd, yEnd


def actionForAllResults():
    for mem in Results:
        if mem.value < 0:
            continue
        full(mem)


if __name__ == '__main__':
    # res = Results.withoutLines
    # full(res)
    # plotGeneticResults(res)
    # plotPath(res)
    #
    #createGeneticForAllInOne()
    # plotPathAllInOne(0)
    # plotPathAllInOne(1)
    # plotPathAllInOne(2)
    actionForAllResults()
    plt.show()
