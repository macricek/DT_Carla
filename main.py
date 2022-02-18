from CarlaEnvironment import CarlaEnvironment
import sys
from PyQt5.QtCore import QObject, QThread, QCoreApplication
import time
from fastAI import get_image_array_from_fn, label_func


class Main(QCoreApplication):
    def __init__(self):
        super(Main, self).__init__([])
        self.time = time.time()
        self.run = True
        self.bla = 0
        self.carlaEnvironment = CarlaEnvironment(1, True)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    def p(self):
        print("Sig")
        sys.exit(0)


if __name__ == '__main__':
    mainApp = Main()
    mainApp.exec()
    while True:
        mainApp.processEvents()
