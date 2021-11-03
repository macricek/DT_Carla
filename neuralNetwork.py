import math
import tensorflow as tf
import numpy as np


class NeuralNetwork:
    nInput = 0
    nHiddenLayers = 0
    nHidden = None
    nOutput = 0
    szWeights = 0
    szBiases = 0
    weights = None

    def __init__(self, nInput, nHiddenInL, nOutput, weights=None):
        self.nInput = nInput
        self.nHiddenLayers = nHiddenInL.shape[0]
        self.nHidden = nHiddenInL
        self.nOutput = nOutput
        self.szWeights, self.szBiases = self.getNumOfNeededElements()
        self.weights = weights

    def setWeights(self, weights):
        self.weights = weights

    def getNumOfNeededElements(self):
        szW = self.nInput * self.nHidden[0] + self.nOutput * self.nHidden[-1]
        szB = self.nInput + self.nOutput + self.nHidden[-1]
        for i in range(0, self.nHiddenLayers - 1):
            szW += self.nHidden[i] * self.nHidden[i + 1]
            szB += self.nHidden[i]
        return szW, szB

    def parse(self):
        assert self.weights.shape[1] == self.szWeights + self.szBiases
        # weights index
        idxW1end = self.nInput * self.nHidden[0]
        idxW2start = idxW1end
        idxW2end = idxW2start + self.nHidden[0] * self.nHidden[1]
        idxW3start = idxW2end
        idxW3end = idxW3start + self.nHidden[1] * self.nOutput
        # biases index
        idxBIstart = idxW3end
        idxBIend = idxBIstart + self.nInput
        idxBH1start = idxBIend
        idxBH1end = idxBH1start + self.nHidden[0]
        idxBH2start = idxBH1end
        idxBH2end = idxBH2start + self.nHidden[1]
        idxBOstart = idxBH2end
        idxBOend = idxBOstart + self.nOutput
        v = self.weights.shape[1]
        assert idxBOend == self.weights.shape[1]
        # weights parsing
        W1 = np.reshape(self.weights[0, 0:idxW1end], (self.nInput, self.nHidden[0]))
        W2 = np.reshape(self.weights[0, idxW2start:idxW2end], (self.nHidden[0], self.nHidden[1]))
        W3 = np.reshape(self.weights[0, idxW3start:idxW3end], (self.nHidden[1], self.nOutput))
        # biases parsing
        BI = self.weights[0, idxBIstart:idxBIend]
        BH1 = self.weights[0, idxBH1start:idxBH1end]
        BH2 = self.weights[0, idxBH2start:idxBH2end]
        BO = self.weights[0, idxBOstart:idxBOend]
        return W1, W2, W3, BI, BH1, BH2, BO

    def run(self, inputs):
        #there needs to be preprocessing inputs
        assert inputs.shape[0] == self.nInput
        W1, W2, W3, BI, BH1, BH2, BO = self.parse()

        X = np.zeros((1, self.nInput))
        H1 = np.zeros((1, self.nHidden[0]))
        H2 = np.zeros((1, self.nHidden[1]))
        O = np.zeros((1, self.nOutput))

        for i in range(0, self.nInput):
            X[0, i] = math.tanh(inputs[i] + BI[i])

        tmp = X @ W1 + BH1
        for i in range(0, self.nHidden[0]):
            H1[0, i] = math.tanh(tmp[0, i])

        tmp = H1 @ W2 + BH2
        for i in range(0, self.nHidden[1]):
            H2[0, i] = math.tanh(tmp[0, i])

        tmp = H2 @ W3 + BO
        for i in range(0, self.nOutput):
            O[0, i] = math.tanh(tmp[0, i])

        return O

class TensorNeural:
    def __init__(self):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(8))
        model.add(tf.keras.layers.Dense(4))
        model.build((None, 16))
        len(model.weights)
        self.model = model