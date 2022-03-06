import glob
import os
import sys
import pygame
from Vehicle import Vehicle
from CarlaConfig import CarlaConfig
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from NeuroEvolution import NeuroEvolution
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

    def __init__(self, main, debug=False):
        super(CarlaEnvironment, self).__init__()
        self.id = 0
        self.debug = debug
        self.main = main
        self.clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.config = CarlaConfig(self.client)
        self.NE = NeuroEvolution(self.config.readSection("NE"))
        self.faLineDetector = FALineDetector()
        self.MAX_ID = self.NE.popSize

        self.trafficManager = self.client.get_trafficmanager()
        self.trafficManager.set_synchronous_mode(self.config.sync)
        self.world = self.client.get_world()
        self.blueprints = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        print("INIT DONE")

    def tick(self):
        '''
        TICKING the world
        :return:
        '''
        if self.debug:
            print("TICK!")
        self.world.tick()

    def testRide(self):
        self.spawnVehicleToStart(True)
        '''
        Spawn one car and "simulate" ride through waypoints
        :return: Nothing
        '''
        print("Starting test ride")
        while True:
            try:
                self.clock.tick()
                self.world.tick()
                if len(self.runStep()) > 0:
                    self.main.terminate()
                    break
            except:
                self.main.terminate()

    def trainingRide(self):
        '''
        Handles one epoch of training!
        :return: Nothing
        '''
        print("Starting training")
        for _ in range(1):
            self.spawnVehicleToStart(False)
            while True:
                try:
                    self.clock.tick()
                    self.world.tick()
                    listV = self.runStep()
                    if len(listV) > 0:
                        for veh in listV:
                            self.NE.singleFit(veh)
                            crosses, err, collisions = veh
                            print(f"Vehicle {veh.vehicleID} done!")
                            self.deleteVehicle(veh)
                        break
                except:
                    self.NE.finishNeuroEvolutionProcess()
                    self.main.terminate()

    def train(self):
        for i in range(self.NE.numCycle):
            print(f"Starting EPOCH {i}/{self.NE.numCycle-1}")
            # run one training epoch
            self.trainingRide()
            self.NE.perform()
        self.NE.finishNeuroEvolutionProcess()  # will probably block the thread

    def spawnVehicleToStart(self, vehDebug):
        '''
        Spawn vehicle to starting spot (spawnPoint[99]) and create it.
        :return: Nothing
        '''
        spawnPoints = self.map.get_spawn_points()
        start = spawnPoints[99]
        vehicle = Vehicle(self, start, id=self.id, neuralNetwork=self.NE.getNeuralNetwork(self.id))
        vehicle.debug = vehDebug
        self.vehicles.append(vehicle)
        self.handleVehicleId()

    def runStep(self):
        '''
        Ask all available vehicles to do their job.
        :return:
        '''
        endedVehicles = []
        for vehicle in self.vehicles:
            if not vehicle.run():
                endedVehicles.append(vehicle)
        return endedVehicles

    def handleVehicleId(self):
        if self.id < self.MAX_ID - 1:
            self.id += 1
        else:
            self.id = 0

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
