import carla
import time
import random
from PyQt5.QtCore import QObject, pyqtSignal
from Sensors import *
from queue import Queue
import sys
from agents.navigation.basic_agent import BasicAgent


# global constants
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
    goal: carla.Location
    path: queue.Queue
    done: bool

    def __init__(self, environment, spawnLocation, id):
        super(Vehicle, self).__init__()
        self.vehicleID = id
        self.environment = environment
        self.path = environment.config.loadPath()
        self.debug = self.environment.debug
        self.fald = self.environment.faLineDetector
        self.me = self.environment.world.spawn_actor(self.environment.blueprints.filter('model3')[0], spawnLocation)
        self.speed = 0
        self.vehicleStopped = 0
        self.done = False
        self.getLocation()

        self.initAgent(spawnLocation.location)
        self.sensorManager = SensorManager(self, self.environment)

        if self.debug:
            print("Vehicle {id} ready".format(id=self.vehicleID))

    def run(self):
        '''
        Do a tick response for vehicle object
        :return: True, if vehicle is alive; False, if ending conditions were MET
        '''
        if not self.me or self.sensorManager.isCollided() or self.checkGoal() or self.standing():
            return False
        # there will NN decide
        control = self.agentAction()
        self.sensorManager.processSensors()

        self.me.apply_control(control)
        return True

    def agentAction(self):
        '''
        Usage of agent to reach expected speed.
        :return: carla.Control
        '''
        if self.agent.done():
            if self.debug:
                print(f"New waypoint for agent: {self.goal}")
            self.agent.set_destination(self.goal)
        control = self.agent.run_step()
        control.manual_gear_shift = False
        if self.debug:
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
        self.goal = self.path.get()
        self.agent.set_destination(self.goal)

    def checkGoal(self):
        '''
        Checks, how far is vehicle from current goal. If goal is reached, new waypoint is loaded. When no more waypoints
        it will return True as signing that current task is done.
        :return: bool
        '''
        dist = self.diffToLocation(self.goal)
        print(f"Distance to goal is: {dist}")
        if dist < 0.2 or self.agent.done():
            if not self.path.empty():
                self.goal = self.path.get()
            else:
                return True

    def standing(self):
        '''
        check, if vehicle is moving or standing. If standing for longer period (100+ticks) returns True
        :return: bool
        '''
        speed = self.getSpeed()
        if speed < 5:
            self.vehicleStopped += 1
        else:
            self.vehicleStopped = 0

        if self.vehicleStopped >= 100:
            print("VEHICLE IS STOPPED!")
            return True
        else:
            return False

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
        self.getLocation()
        dist = location.distance(self.location)
        return dist

    @staticmethod
    def errInLocation(l1: carla.Location, l2: carla.Location):
        diffX = math.sqrt((l1.x - l2.x)**2)
        diffY = math.sqrt((l1.y - l2.y)**2)
        return diffX, diffY

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
        print("{id}:[{t}] X: {x}, Y: {y}, Z: {z}".format(id=self.vehicleID, t=type, x=x, y=y, z=z))

    def destroy(self):
        '''
        Destroy all sensors attached to vehicle and then vehicle on it's own
        :return: none
        '''
        if self.debug:
            print("Destroying Vehicle {id}".format(id=self.vehicleID))
        try:
            self.sensorManager.destroy()
            self.me.destroy()
            self.environment.tick()
        except:
            print(f"Error in destroying of {self.vehicleID}")
