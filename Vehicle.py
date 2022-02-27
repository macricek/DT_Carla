import threading
import carla
import time
import random
from PyQt5.QtCore import QObject, pyqtSignal
from Sensors import *
from queue import Queue
import sys
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent

from fastAI.FALineDetector import FALineDetector

## global constants
MAX_TIME_CAR = 5
CAM_HEIGHT = 512
CAM_WIDTH = 1024


class Vehicle(QObject):
    me = carla.Vehicle  # ref to vehicle

    # states of vehicle
    location: carla.Location
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

    def __init__(self, environment, spawnLocation, id):
        super(Vehicle, self).__init__()
        self.threadID = id  # threadOBJ
        self.environment = environment
        self.debug = self.environment.debug
        self.fald = FALineDetector()
        self.me = self.environment.world.spawn_actor(self.environment.blueprints.filter('model3')[0], spawnLocation)
        self.speed = 0
        self.getLocation()

        self.initAgent(spawnLocation.location)
        self.sensorManager = SensorManager(self, self.environment)

        if self.debug:
            print("Vehicle {id} ready".format(id=self.threadID))

    def run(self):
        if not self.me or self.sensorManager.isCollided():
            self.terminate()
            return False
        # there will NN decide
        control = self.agent.run_step()
        control.manual_gear_shift = False
        print(f"Control: {control}")
        self.sensorManager.processSensors()

        self.me.apply_control(control)
        return True

    def agentAction(self):
        '''
        Usage of agent to reach expected speed.
        :return:
        '''
        if self.agent.done():
            if self.pts.empty():
                return -1
            self.goal = self.pts.get()
            print(f"New waypoint: {self.goal}")
            self.agent.set_destination(self.goal)
        print(f"Asked: {self.goal}")
        self.processMeasures()
        control = self.agent.run_step()
        control.manual_gear_shift = False
        print(f"Control: {control}")
        return control

    def initAgent(self, spawnLoc):
        '''
        Init BasicAgent - we will use agent's PID to regulate speed.
        :param spawnLoc: carla.Location -> starting Location of agent
        :return: nothing
        '''
        self.getLocation()
        while self.diffToLocation(spawnLoc) > 1:
            self.environment.tick()
            self.getLocation()
            time.sleep(0.01)
        self.agent = BasicAgent(self.me, target_speed=50)
        spawnPoints = self.environment.map.get_spawn_points()
        self.pts = Queue()
        self.pts.put(spawnPoints[133].location)
        self.pts.put(spawnPoints[129].location)
        self.goal = self.pts.get()
        self.agent.set_destination(self.goal)

    def terminate(self):
        print("TERMINATE")
        self.sensorManager.destroy()
        self.destroy()

    def controlVehicle(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        if self.debug:
            print("{id}:[Control] T: {th}, S: {st}, B: {b}".format(id=self.threadID, th=throttle, st=steer, b=brake))
        self.me.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake,
                                                   hand_brake=hand_brake, reverse=reverse))

    def getLocation(self):
        '''
        FILLS self.location with location of VEHICLE
        :return: carla.Location
        '''
        self.location = self.me.get_location()
        if self.debug:
            self.print3Dvector(self.location, "Location")
        return self.location

    def diffToLocation(self, location: carla.Location):
        dist = location.distance(self.location)
        return dist

    def getSpeed(self):
        '''
        FILLS   self.velocity with 3d vector of velocity
                self.speed with speed in km/h of vehicle
        :return: self.speed
        '''
        self.velocity = self.me.get_velocity()
        self.speed = 3.6 * math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2 + self.velocity.z ** 2)
        return self.speed

    def ref(self):
        '''
        :return: carla.Vehicle object of Vehicle
        '''
        return self.me

    def print3Dvector(self, vector, type):
        '''
        Method for debug write of carla.3Dvector
        :param vector: carla.3Dvector
        :param type: name, what is printed [location/velocity...]
        :return: nothing
        '''
        x = vector.x
        y = vector.y
        z = vector.z
        print("{id}:[{t}] X: {x}, Y: {y}, Z: {z}".format(id=self.threadID, t=type, x=x, y=y, z=z))

    def destroy(self):
        '''
        Destroy all sensors attached to vehicle and then vehicle on it's own
        :return: none
        '''
        if self.debug:
            print("Destroying Vehicle {id}".format(id=self.threadID))
        try:
            self.sensorManager.destroy()
            self.me.destroy()
        finally:
            self.environment.deleteVehicle(self)
