import glob
import threading
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from carla import ColorConverter as cc

import LineDetection
from LineDetection import CNNLineDetector, transformImage

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
    debug = True
    editEnvironment = False
    # lists
    vehicles = []
    allFeatures = []
    maxId = 50
    #members
    client = None
    world = None
    blueprints = None
    model = None
    cnnLineDetector = None

    def __init__(self, numVehicles, debug=False):
        self.id = 0
        self.debug = debug
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.cnnLineDetector = CNNLineDetector(from_scratch=False, dataPath=LineDetection.data_path)
        if self.editEnvironment:
            self.setSimulation()

        self.blueprints = self.world.get_blueprint_library()
        self.model = self.blueprints.filter('model3')[0]
        self.map = self.world.get_map()
        self.spawnVehicles(numVehicles)
        self.startThreads()

    def spawnVehicles(self, numVehicles):
        for i in range(0, numVehicles):
            spawnPoints = self.map.get_spawn_points()
            start = spawnPoints[self.id]
            vehicle = Vehicle(self, start, id=self.id)
            self.vehicles.append(vehicle)
            self.allFeatures.append(vehicle)
            self.id += 1
            time.sleep(1)

    def setSimulation(self):
        self.world.unload_map_layer(carla.MapLayer.All) #make map easier
        self.world.load_map_layer(carla.MapLayer.Walls)
        self.originalSettings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def stopSimulation(self):
        self.world.load_map_layer(carla.MapLayer.All)
        self.world.apply_settings(self.originalSettings)

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
        if self.editEnvironment:
            self.stopSimulation()
        for actor in self.allFeatures:
            try:
                actor.destroy()
                print("Removing utility!")
            except:
                print("Already deleted!")
