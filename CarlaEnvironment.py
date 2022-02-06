import glob
import os
import sys
import time
from Vehicle import Vehicle
from CarlaConfig import CarlaConfig

# from Carla doc
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class CarlaEnvironment:
    debug: bool
    config: CarlaConfig
    # lists
    vehicles = []
    allFeatures = []
    maxId = 5
    # members
    client = carla.Client
    world = carla.World
    blueprints = carla.BlueprintLibrary

    def __init__(self, numVehicles, debug=False):
        self.id = 0
        self.debug = debug

        self.client = carla.Client('localhost', 2000)
        self.config = CarlaConfig(self.client)

        self.world = self.client.get_world()

        self.blueprints = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawnVehicles(numVehicles)
        self.startThreads()

    def spawnVehicles(self, numVehicles):
        for i in range(0, numVehicles):
            spawnPoints = self.map.get_spawn_points()
            start = spawnPoints[self.id+100]
            vehicle = Vehicle(self, start, id=self.id)
            self.vehicles.append(vehicle)
            self.allFeatures.append(vehicle)
            self.id += 1
            time.sleep(1)

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
                finally:
                    break

    def deleteAll(self):
        print("Destroying")
        for actor in self.allFeatures:
            try:
                actor.destroy()
                print("Removing utility!")
            except:
                print("Already deleted!")

    def __del__(self):
        print("Turning off sync mode")
        self.config.turnOffSync()
