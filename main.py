import enum

from CarlaEnvironment import CarlaEnvironment
import sys
from PyQt5.QtCore import QObject, QThread, QCoreApplication
import time
sys.path.insert(0, "fastAI")
from fastAI.fastAI import get_image_array_from_fn, label_func
from fastAI.CameraGeometry import CameraGeometry


class Results(enum.IntEnum):
    none = -1
    all_possible_inputs = 2
    Metrics_Binary_Navigation = 5
    Metrics_Binary_Navigation_2 = 6
    Lines_Metrics_Binary_Navigation = 22
    Only_Lines = 61
    Lines_Radar_Agent = 103
    Metrics_Binary = 300
    All_Except_Navigation = 301

    def __str__(self):
        return self.value

    def title(self):
        return self.name.replace("_", " ")


class Mode(enum.IntEnum):
    trainingDefault = 0
    showTrainingDefault = 1
    runTestDefault = 2
    trainingAdvanced = 3
    showTrainingAdvanced = 4
    runTestAdvanced = 5


class Main(QCoreApplication):
    def __init__(self, data, mode):
        super(Main, self).__init__([])
        self.time = time.time()
        self.data = data
        self.mode = mode

        self.carlaEnvironment = CarlaEnvironment(self, data=data.value, debug=False)

    def terminate(self):
        print("Terminating MAIN!")
        try:
            self.carlaEnvironment.terminate()
        finally:
            sys.exit(0)

    def exec(self):
        # DEFAULT scenarios
        if self.mode == Mode.trainingDefault:
            self.defaultTraining()
        elif self.mode == Mode.showTrainingDefault:
            self.showTrainingDefaultResult()
        elif self.mode == Mode.runTestDefault:
            self.runTestDefault()
        # ADVANCED scenarios
        elif self.mode == Mode.trainingAdvanced:
            pass
        elif self.mode == Mode.showTrainingAdvanced:
            pass
        elif self.mode == Mode.runTestAdvanced:
            pass

        return super().exec()

    def defaultTraining(self):
        self.carlaEnvironment.train()

    def showTrainingDefaultResult(self):
        if self.data == Results.none:
            print("Need to pick some results")
            return
        self.carlaEnvironment.replayTrainingRide(data.value)

    def runTestDefault(self):
        if self.data == Results.none:
            print("Need to pick some results")
            return
        self.carlaEnvironment.testRide(data.value)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


if __name__ == '__main__':
    data = Results.Only_Lines
    mode = Mode.showTrainingDefault
    mainApp = Main(data, mode)
    code = mainApp.exec()
    sys.exit(code)
