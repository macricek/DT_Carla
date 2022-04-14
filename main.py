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


class Main(QCoreApplication):
    def __init__(self, data):
        super(Main, self).__init__([])
        self.time = time.time()
        self.data = data

        self.carlaEnvironment = CarlaEnvironment(self, data=data.value, debug=False)

    def terminate(self):
        print("Terminating MAIN!")
        try:
            self.carlaEnvironment.terminate()
        finally:
            sys.exit(0)

    def runTraining(self):
        self.carlaEnvironment.train()

    def showBestResult(self):
        if self.data == Results.none:
            print("Need to pick some results")
            return
        self.carlaEnvironment.replayTrainingRide(data.value)

    def runTest(self):
        if self.data == Results.none:
            print("Need to pick some results")
            return
        self.carlaEnvironment.testRide(data.value)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)


if __name__ == '__main__':
    data = Results.withoutLines

    mainApp = Main(data)
    # mainApp.runTraining()
    mainApp.showBestResult()
    # mainApp.runTest()
    code = mainApp.exec()
    sys.exit(code)
