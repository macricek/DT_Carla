import threading
import carla
import time
import random

from Sensors import *
import sys

sys.path.insert(0, "fastAI")
from FALineDetector import FALineDetector

## global constants
MAX_TIME_CAR = 30
CAM_HEIGHT = 512
CAM_WIDTH = 1024

class Vehicle(threading.Thread):
    me = carla.Vehicle  # ref to vehicle

    # states of vehicle
    location = None
    velocity = None

    # camera
    ldCam: LineDetectorCamera
    seg = Camera

    # sensor
    collision: CollisionSensor
    lidar: LidarSensor
    radar: RadarSensor
    obstacleDetector: ObstacleDetector

    # other
    debug: bool

    sensors = []

    def __init__(self, environment, spawnLocation, id):
        threading.Thread.__init__(self)
        self.threadID = id  # threadOBJ
        self.environment = environment
        self.debug = self.environment.debug
        self.fald = FALineDetector()
        self.me = self.environment.world.spawn_actor(self.environment.blueprints.filter('model3')[0], spawnLocation)
        self.setupSensors()
        self.processMeasures()
        if self.debug:
            print("Vehicle {id} starting".format(id=self.threadID))

    def run(self):
        start = time.time()
        now = time.time()
        while now - start < MAX_TIME_CAR and not self.collision.isCollided() and self.me:
            # there it will NN decide
            steer = random.uniform(-1, 1)
            throttle = random.uniform(0, 1)
            #try:
            self.controlVehicle(throttle=throttle)
            self.processMeasures()
            now = time.time()
            #except:
                #print("Run failed!")
                #pass
        self.destroy()

    def controlVehicle(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        if self.debug:
            print("{id}:[Control] T: {th}, S: {st}, B: {b}".format(id=self.threadID, th=throttle, st=steer, b=brake))
        self.me.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                                   hand_brake=hand_brake, reverse=reverse))

    def setupSensors(self):
        self.environment.allFeatures.append(self.me)
        self.ldCam = LineDetectorCamera(self)
        # self.seg = Camera(self, self.camHeight, self.camWidth, type='Semantic Segmentation')
        self.collision = CollisionSensor(self)
        # self.radar = RadarSensor(self, True)
        # self.lidar = LidarSensor(self)
        self.obstacleDetector = ObstacleDetector(self)
        self.sensors.append(self.ldCam)
        # self.sensors.append(self.seg)
        self.sensors.append(self.collision)
        # self.sensors.append(self.radar)
        # self.sensors.append(self.lidar)
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
