import threading
import carla
import time
import random

import Sensors
from Sensors import *
from CarlaEnvironment import CarlaEnvironment
## global constants
MAX_TIME_CAR = 30


class Vehicle(threading.Thread):
    me = carla.Vehicle                           #ref to vehicle
    environment: CarlaEnvironment       #ref to environment upper

    camHeight = 512
    camWidth = 1024

    #states of vehicle
    location = None
    velocity = None

    #camera
    rgb = Sensors.Camera
    seg = Sensors.Camera

    #sensor
    collision = Sensors.CollisionSensor
    lidar = Sensors.LidarSensor
    radar = Sensors.RadarSensor
    obstacleDetector = Sensors.ObstacleDetector

    #other
    debug: bool


    sensors = []

    def __init__(self, environment, spawnLocation, id):
        threading.Thread.__init__(self)
        self.threadID = id  # threadOBJ
        self.environment = environment
        self.debug = self.environment.debug
        self.me = self.environment.world.spawn_actor(self.environment.model, spawnLocation)
        self.setupSensors()
        self.processMeasures()
        if self.debug:
            print("Vehicle {id} starting".format(id=self.threadID))

    def run(self):
        start = time.time()
        now = time.time()
        while now - start < MAX_TIME_CAR and not self.isCollided() and self.me:
            #there it will NN decide
            steer = random.uniform(-1, 1)
            throttle = random.uniform(0, 1)
            try:
                self.controlVehicle(throttle=throttle, steer=steer)
                self.processMeasures()
                if self.debug and self.rgb.isImageAvailable():
                    self.rgb.draw()
                if self.debug and self.seg.isImageAvailable():
                    self.seg.draw()
                time.sleep(0.05)
                now = time.time()
            except:
                None
        self.destroy()

    def controlVehicle(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        if self.debug:
            print("{id}:[Control] T: {th}, S: {st}, B: {b}".format(id=self.threadID, th=throttle, st=steer, b=brake))
        self.me.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                                        hand_brake=hand_brake, reverse=reverse))

    def isCollided(self):
        return self.collision.isCollided()

    def setupSensors(self):
        self.environment.allFeatures.append(self.me)
        self.rgb = Camera(self, self.camHeight, self.camWidth)
        #self.__seg = Camera(self, self.camHeight, self.camWidth, type='Semantic Segmentation')
        self.collision = CollisionSensor(self)
        #self.__radar = RadarSensor(self, True)
        #self.lidar = LidarSensor(self)
        self.obstacleDetector = ObstacleDetector(self)
        self.sensors.append(self.rgb)
        #self.sensors.append(self.__seg)
        self.sensors.append(self.collision)
        #self.sensors.append(self.__radar)
        #self.sensors.append(self.lidar)
        self.sensors.append(self.obstacleDetector)
        self.environment.allFeatures.extend(self.sensors)

    def processMeasures(self):
        self.location = self.me.get_location()
        self.velocity = self.me.get_velocity()
        if self.debug:
            self.print3Dvector(self.location, "Location")
            self.print3Dvector(self.velocity, "Velocity")

    def ref(self):
        return self.me

    def print3Dvector(self, vector, type):
        x = vector.x
        y = vector.y
        z = vector.z
        print("{id}:[{t}] X: {x}, Y: {y}, Z: {z}".format(id=self.threadID, t=type, x=x, y=y, z=z))

    def destroy(self):
        if self.debug:
            print("Destroying Vehicle {id}".format(id=self.threadID))
        try:
            for actor in self.sensors:
                actor.destroy()
            self.me.destroy()
        finally:
            self.environment.deleteVehicle(self)