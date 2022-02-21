import threading
import carla
import time
import random
from PyQt5.QtCore import QObject, pyqtSignal
from Sensors import *
import sys

sys.path.insert(0, "fastAI")
from FALineDetector import FALineDetector

## global constants
MAX_TIME_CAR = 5
CAM_HEIGHT = 512
CAM_WIDTH = 1024


class Vehicle(QObject):
    me = carla.Vehicle  # ref to vehicle

    # states of vehicle
    location: carla.Vector3D
    velocity: carla.Vector3D

    # camera
    ldCam: LineDetectorCamera
    seg: Camera

    # sensor
    collision: CollisionSensor
    lidar: LidarSensor
    radar: RadarSensor
    obstacleDetector: ObstacleDetector

    # other
    debug: bool
    fald: FALineDetector

    # signals
    finished = pyqtSignal()

    def __init__(self, environment, spawnLocation, id):
        super(Vehicle, self).__init__()
        self.threadID = id  # threadOBJ
        self.environment = environment
        self.debug = self.environment.debug
        self.fald = FALineDetector()
        self.me = self.environment.world.spawn_actor(self.environment.blueprints.filter('model3')[0], spawnLocation)
        self.processMeasures()
        self.sensorManager = SensorManager(self, self.environment)
        if self.debug:
            print("Vehicle {id} ready".format(id=self.threadID))

    def run(self):
        if not self.me or self.sensorManager.isCollided():
            print("TERMINATE")
            self.sensorManager.destroy()
            self.destroy()

        # there will NN decide
        try:  # events that needs TICK
            self.environment.world.wait_for_tick()
            self.sensorManager.processSensors()
        except RuntimeError:
            print("Timeout, no tick!")
            print("TERMINATE")
            self.sensorManager.destroy()
            self.destroy()

        steer = random.uniform(-1, 1)
        throttle = random.uniform(0, 1)
        #self.controlVehicle(throttle=throttle)
        self.me.set_autopilot(True)
        #self.processMeasures()
        self.finished.emit()

    def controlVehicle(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        if self.debug:
            print("{id}:[Control] T: {th}, S: {st}, B: {b}".format(id=self.threadID, th=throttle, st=steer, b=brake))
        self.me.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                                   hand_brake=hand_brake, reverse=reverse))

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
            self.sensorManager.destroy()
            self.me.destroy()
            self.finished.emit()
        finally:
            self.environment.deleteVehicle(self)
