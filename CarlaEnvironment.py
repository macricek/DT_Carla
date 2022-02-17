import glob
import os
import random
import sys
import time
from Vehicle import Vehicle
from CarlaConfig import CarlaConfig
from PyQt5 import QtCore

# from Carla doc
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class CarlaEnvironment(QtCore.QObject):
    debug: bool
    config: CarlaConfig
    # lists
    vehicles = []
    maxId = 5
    # members
    client = carla.Client
    world = carla.World
    blueprints = carla.BlueprintLibrary

    def __init__(self, numVehicles, debug=False):
        super().__init__()
        self.id = 0
        self.debug = debug
        self.numVehicles = numVehicles
        self.readyVehicles = 0
        self.client = carla.Client('localhost', 2000)
        self.config = CarlaConfig(self.client)

        self.world = self.client.get_world()

        self.blueprints = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawnVehicles(numVehicles)
        print("INIT DONE")
        self.readyVehicles = self.numVehicles - 1
        self.tick()

#TODO: spravim to tak, ze ked ticknem, tak sa spusti thread vehiclu/ov, pocka sa na koniec a tak dokola...
    def tick(self):
        self.readyVehicles += 1
        print(f"Ready vehicles: {self.readyVehicles}/{self.numVehicles}")
        if self.readyVehicles == self.numVehicles:
            self.startThreads()
            print("TICK!")
            self.world.tick()
            self.readyVehicles = 0

    def spawnVehicles(self, numVehicles):
        for i in range(0, numVehicles):
            spawnPoints = self.map.get_spawn_points()
            start = spawnPoints[int(random.random()*len(spawnPoints))]
            vehicle = Vehicle(self, start, id=self.id)
            vehicle.sensorManager.readySignal.connect(self.tick)
            self.vehicles.append(vehicle)
            self.id += 1

    def startThreads(self):
        for vehicle in self.vehicles:
            vehicle.start()

    def run(self):
        for vehicle in self.vehicles:
            vehicle.run()

    def deleteVehicle(self, vehicle):
        for v in self.vehicles:
            if v == vehicle:
                try:
                    self.vehicles.remove(vehicle)
                    del vehicle
                except:
                    print("already out")

    def deleteAll(self):
        for vehicle in self.vehicles:
            try:
                vehicle.destroy()
                print("Removing vehicle!")
            except:
                print("Already deleted!")

    def __del__(self):
        self.deleteAll()
        print("Turning off sync mode")
        self.config.turnOffSync()
