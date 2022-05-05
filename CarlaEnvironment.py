import glob
import os
import sys
import threading
import time

from generate_traffic import generateTraffic

import pygame
from Vehicle import Vehicle
from CarlaConfig import CarlaConfig
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from NeuroEvolution import NeuroEvolution
import numpy as np
from fastAI.FALineDetector import FALineDetector

# from Carla doc
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class CarlaEnvironment(QObject):
    '''
    Carla Environment
    @author: Marko Chylik
    @Date: May, 2022
    '''
    debug: bool
    trainingMode: bool
    config: CarlaConfig
    NE: NeuroEvolution
    # lists
    vehicles = []
    # members
    client: carla.Client
    world: carla.World
    blueprints: carla.BlueprintLibrary
    trafficManager: carla.TrafficManager
    # PyQt signals
    done = pyqtSignal()

    def __init__(self, main, data=-1, debug=False):
        '''
        Create an object of Carla environment.
        :param main: reference to main (QCoreApplication)
        :param data: default = -1 -> create new directory based on "rev" from config.ini
                        if param is > 0 -> use the directory with the same name as data
        :param debug: bool
        '''
        super(CarlaEnvironment, self).__init__()
        self.id = 0
        self.debug = debug
        self.main = main
        self.clock = pygame.time.Clock()
        self.loadedData = True if data>0 else False
        self.whichPath = 0

        self.client = carla.Client('localhost', 2000)
        path = "config.ini" if data == -1 else os.path.join(f"results/{data}/config.ini")
        self.config = CarlaConfig(self.client, path)
        self.config.apply()
        self.NE = NeuroEvolution(self.config.loadNEData())
        self.faLineDetector = FALineDetector()
        self.MAX_ID = self.NE.popSize
        self.trainingMode = False
        self.trafficManager = self.client.get_trafficmanager()
        self.trafficManager.set_synchronous_mode(self.config.sync)
        self.world = self.client.get_world()
        self.blueprints = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.traffic = False

        print("CARLA ENVIRONMENT DONE")

    def tick(self):
        '''
        TICKING the world
        :return: num of tick
        '''
        self.clock.tick()
        tickNum = self.world.tick()
        return tickNum

    def replayTrainingRide(self, numRevision, traffic):
        '''
        Spawn one car and "simulate" ride through waypoints of training path
        :param numRevision: Which model will be used
        :param traffic: bool -> use traffic?
        :return: None
        '''
        self.trainingMode = False
        self.generateTraffic(traffic)
        spawnPoints = self.map.get_spawn_points()
        start = spawnPoints[99]
        self.whichPath = 0
        self.spawnVehicle(True, start, numRevision)
        print("Starting replay of training ride")
        if self.loop():
            self.main.terminate()

    def testRide(self, numRevision, traffic, path):
        '''
        Run test ride on one of the test paths.
        :param numRevision: Which model will be used
        :param traffic: bool
        :param path: which path is going to be used (1,2)
        :return:
        '''
        self.trainingMode = False
        self.generateTraffic(traffic)
        spawnPoints = self.map.get_spawn_points()
        point = 334 if path == 1 else 258

        start = spawnPoints[point]
        self.whichPath = path

        self.spawnVehicle(True, start, numRevision)
        if self.loop():
            self.main.terminate()

    def train(self, traffic):
        '''
        Run training process, don't show anything at all, just cover whole training process
        :param traffic: bool
        :return:
        '''
        self.trainingMode = True
        if not self.loadedData:
            self.config.incrementNE()
        self.generateTraffic(traffic)
        for i in range(self.NE.startCycle, self.NE.numCycle):
            print(f"Starting EPOCH {i+1}/{self.NE.numCycle}")
            # run one training epoch
            self.trainingRide(i)
            self.NE.perform()
            self.NE.finishNeuroEvolutionProcess()  # will probably block the thread
        self.main.terminate()

    def trainingRide(self, epoch):
        '''
        Handles one epoch of training
        :return: None
        '''
        print("Starting training")
        for i in range(self.MAX_ID):
            if epoch != 0 and i == 0:
                # do not run best solutions again!
                continue
            self.id = i
            spawnPoints = self.map.get_spawn_points()
            start = spawnPoints[99]
            self.whichPath = 0
            self.spawnVehicle(False, start)
            self.loop()

    def loop(self):
        '''
        main loop for any paths/scenario -> send tick to server and call runStep from vehicle
        :return:    True, if any vehicle ended
                    False, if any error has occured
        '''
        while True:
            try:
                tickNum = self.tick()
                listV = self.runStep(tickNum)
                if len(listV) > 0:
                    for veh in listV:
                        if self.trainingMode:
                            self.NE.singleFit(veh)
                        else:
                            self.storeVehicleResults(veh)
                        print(f"Vehicle {veh.vehicleID} done!")
                        self.deleteVehicle(veh)
                    return True
            except Exception as e:
                print(e)
                self.main.terminate()
                return False

    def path(self):
        '''
        load path based on whichPath
        :return: list of waypoints
        '''
        return self.config.loadPath(self.whichPath)

    def generateTraffic(self, generate):
        '''
        generate Traffic if desired
        :param generate: bool
        :return: None
        '''
        if not generate:
            return
        self.traffic = True
        self.trafficThread = threading.Thread(target=generateTraffic, args=(self.trafficGenerated,))
        self.trafficThread.start()

    def trafficGenerated(self):
        '''
        :return: bool: is traffic generated?
        '''
        return self.traffic

    def spawnVehicle(self, testRide, start, numRevision=0):
        '''
        Spawn vehicle to starting spot (start) and create it.
        :return: None
        :param testRide: is it test ride? (bool)
        :param start: point on map, where vehicle should be spawned
        :param numRevision: in case of test ride, which NN model should be used
        '''

        if not testRide:
            neuralNetwork = self.NE.getNeuralNetwork(self.id)
        else:
            weightsFile = f'results/{numRevision}/best.csv'
            weights = np.loadtxt(weightsFile, delimiter=',')
            neuralNetwork = self.NE.getNeuralNetworkToTest(weights)
        vehicle = Vehicle(self, spawnLocation=start, id=self.id, neuralNetwork=neuralNetwork)
        vehicle.applyConfig(testRide)
        self.vehicles.append(vehicle)

    def runStep(self, tickNum):
        '''
        Ask all available vehicles to do their job
        :return: list of ended vehicles
        '''
        endedVehicles = []
        for vehicle in self.vehicles:
            if not vehicle.run(tickNum):
                endedVehicles.append(vehicle)
        return endedVehicles

    def storeVehicleResults(self, vehicle: Vehicle):
        '''
        We can expect that vehicle will have 4 lists. Save this lists on disk with current revision
        :param vehicle: Vehicle object
        :return: None
        '''
        print("Storing results")
        pos, ll, rl, op = vehicle.returnVehicleResults()
        numberOfPositions = len(pos)

        x = np.zeros((4, numberOfPositions))
        y = np.zeros((4, numberOfPositions))
        for idx in range(numberOfPositions):
            x[0, idx] = pos[idx].x
            y[0, idx] = pos[idx].y

            x[3, idx] = op[idx].x
            y[3, idx] = op[idx].y

        for idx in range(len(ll)):
            x[1, idx] = ll[idx].x
            y[1, idx] = ll[idx].y

            x[2, idx] = rl[idx].x
            y[2, idx] = rl[idx].y

        rev = self.config.parser.get("NE", "rev")
        pathX = os.path.join(f"results/{rev}/X{self.whichPath}.csv")
        pathY = os.path.join(f"results/{rev}/Y{self.whichPath}.csv")
        np.savetxt(pathX, x, delimiter=',')
        np.savetxt(pathY, y, delimiter=',')

    def deleteVehicle(self, vehicle):
        '''
        When vehicle ends, we need to delete it also from our list
        :param vehicle: Vehicle
        :return: None
        '''
        for v in self.vehicles:
            if v == vehicle:
                try:
                    print(f"Deleting vehicle {vehicle.vehicleID}")
                    v.destroy()
                    self.vehicles.remove(vehicle)
                except:
                    print("Vehicle already out")

    def deleteAll(self):
        '''
        In case of error/ending the whole simulation, we wants to delete all vehicles.
        :return: None
        '''
        for vehicle in self.vehicles:
            try:
                vehicle.destroy()
                self.vehicles.remove(vehicle)
                del vehicle
                print("Removing vehicle!")
            except:
                print("Already deleted!")

    def terminate(self):
        '''
        In case of error/ending simulation, gives signal to main to end the program
        :return: None
        '''
        self.deleteAll()

        if self.traffic:
            self.traffic = False
            for _ in range(3):
                self.tick()
                time.sleep(0.5)

        print("Turning off sync mode")
        self.trafficManager.set_synchronous_mode(False)
        self.config.turnOffSync()

    def __del__(self):
        try:
            self.terminate()
        except:
            print("Error in deleting CarlaEnv")
