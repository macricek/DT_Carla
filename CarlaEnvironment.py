import glob
import os
import sys
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
        super(CarlaEnvironment, self).__init__()
        self.id = 0
        self.debug = debug
        self.main = main
        self.clock = pygame.time.Clock()
        self.loadedData = True if data>0 else False

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

        print("CARLA ENVIRONMENT DONE")

    def tick(self):
        '''
        TICKING the world
        :return: num of tick
        '''
        self.clock.tick()
        tickNum = self.world.tick()
        return tickNum

    def testRide(self, numRevision):
        '''
        Spawn one car and "simulate" ride through waypoints
        :return: Nothing
        '''
        self.trainingMode = False
        self.spawnVehicleToStart(True, numRevision)
        print("Starting test ride")
        while True:
            try:
                tickNum = self.tick()
                if len(self.runStep(tickNum)) > 0:
                    self.main.terminate()
                    break
            except Exception as e:
                print(e)
                self.main.terminate()

    def trainingRide(self, epoch):
        '''
        Handles one epoch of training!
        :return: Nothing
        '''
        print("Starting training")
        for i in range(self.MAX_ID):
            if epoch != 0 and i == 0:
                # do not run best solutions again!
                continue
            self.id = i
            self.spawnVehicleToStart(False)
            while True:
                try:
                    tickNum = self.tick()
                    listV = self.runStep(tickNum)
                    if len(listV) > 0:
                        for veh in listV:
                            self.NE.singleFit(veh)
                            print(f"Vehicle {veh.vehicleID} done!")
                            self.deleteVehicle(veh)
                        break
                except Exception as e:
                    print(e)
                    self.main.terminate()

    def train(self):
        self.trainingMode = True
        if not self.loadedData:
            self.config.incrementNE()
        for i in range(self.NE.startCycle, self.NE.numCycle):
            print(f"Starting EPOCH {i+1}/{self.NE.numCycle}")
            # run one training epoch
            self.trainingRide(i)
            self.NE.perform()
            self.NE.finishNeuroEvolutionProcess()  # will probably block the thread
        self.main.terminate()

    def spawnVehicleToStart(self, testRide, numRevision=0):
        '''
        Spawn vehicle to starting spot (spawnPoint[99]) and create it.
        :return: Nothing
        '''
        spawnPoints = self.map.get_spawn_points()
        start = spawnPoints[99]
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
        Ask all available vehicles to do their job.
        :return:
        '''
        endedVehicles = []
        for vehicle in self.vehicles:
            if not vehicle.run(tickNum):
                endedVehicles.append(vehicle)
        return endedVehicles

    def deleteVehicle(self, vehicle):
        for v in self.vehicles:
            if v == vehicle:
                try:
                    print(f"Deleting vehicle {vehicle.vehicleID}")
                    v.destroy()
                    self.vehicles.remove(vehicle)
                except:
                    print("Vehicle already out")

    def deleteAll(self):
        for vehicle in self.vehicles:
            try:
                vehicle.destroy()
                self.vehicles.remove(vehicle)
                del vehicle
                print("Removing vehicle!")
            except:
                print("Already deleted!")

    def terminate(self):
        self.deleteAll()
        print("Turning off sync mode")
        self.trafficManager.set_synchronous_mode(False)
        self.config.turnOffSync()

    def __del__(self):
        try:
            self.terminate()
        except:
            print("Error in deleting CarlaEnv")
