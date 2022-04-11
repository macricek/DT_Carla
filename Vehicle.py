import copy
import datetime

import carla
import time
import random

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

import neuralNetwork
from Sensors import *
from queue import Queue
from collections import deque
import sys
from agents.navigation.basic_agent import BasicAgent

from CarlaConfig import InputsEnum


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
    toGoal: deque
    done: bool
    nn: neuralNetwork.NeuralNetwork

    # record quality of driving
    __crossings = 0
    __errDec = 0
    __collisions = 0
    __inCycle = 0
    __reachedGoals = 0
    __rangeDriven = 0

    def __init__(self, environment, spawnLocation, neuralNetwork, id):
        super(Vehicle, self).__init__()
        self.vehicleID = id
        self.environment = environment
        self.path = environment.path()
        self.askedInputs = environment.config.loadAskedInputs()[0]
        self.debug = self.environment.debug
        self.fald = self.environment.faLineDetector
        self.me = self.environment.world.spawn_actor(self.environment.blueprints.filter('model3')[0], spawnLocation)
        self.nn = neuralNetwork

        self.speed = 0
        self.vehicleStopped = 0
        self.steer = 0
        self.limit = 0.25
        self.defaultSteerMaxChange = 0.1
        self.numMeasure = 10
        self.lastLocation = self.getLocation()

        self.initAgent(spawnLocation.location)
        self.sensorManager = SensorManager(self, self.environment)
        currentDt = str(datetime.datetime.now())
        print("Vehicle {id} ready at {dt}".format(id=self.vehicleID, dt=currentDt))
        self.startTime = time.time()

        self.toGoal = deque(maxlen=10)
        self.metrics = deque(maxlen=2)

        self.__positionHistory = []
        self.__leftLinePlanner = []
        self.__rightLinePlanner = []
        self.__optimalPath = []

    def run(self, tickNum):
        '''
        Do a tick response for vehicle object
        :return: True, if vehicle is alive; False, if ending conditions were MET
        '''
        if not self.me or self.sensorManager.isCollided() or self.checkGoal():
            print("Collision or Goal")
            return False
        if not self.debug: #TRAINING MODE
            if self.standing() or self.inCycle():
                print("Standing/In cycle, penalization!")
                self.__inCycle += 1
                return False

        # there will NN decide
        self.sensorManager.processSensors()
        control = self.getControl()
        self.me.apply_control(control)
        if tickNum % self.numMeasure == 0:
            self.print(f"TN {tickNum}: {self.diffToLocation(self.goal)}")
            self.toGoal.append(self.diffToLocation(self.goal))
            self.metrics.append(control)
            if self.debug:
                self.__positionHistory.append(self.location)
                #self.agent.get_waypoints()
                #Here we need to locate somehow lines + path
        return True

    def agentAction(self):
        '''
        Usage of agent to reach expected speed.
        :return: carla.Control
        '''
        if self.agent.done():
            self.print(f"New waypoint for agent: {self.goal}")
            self.agent.set_destination(self.goal)
            self.toGoal.clear()
        control = self.agent.run_step()
        control.manual_gear_shift = False
        waypoints = self.agent.get_waypoints()[0]
        waypoint = waypoints.transform.location if waypoints else None
        self.print(f"Control: {control}")
        return control, waypoint

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

    def record(self):
        self.__crossings = self.sensorManager.laneInvasionDetector.crossings
        self.__collisions = 1 if self.sensorManager.isCollided() else 0
        retDict = {'crossings': self.__crossings,
                   'collisions': self.__collisions,
                   'inCycle': self.__inCycle,
                   'rangeDriven': self.__rangeDriven,
                   'reachedGoals': self.__reachedGoals,
                   'error': self.__errDec}

        return retDict

    def recordEachStep(self, agentSteer):
        self.__errDec += abs(agentSteer - self.steer)

        distLast = self.errInLocation(self.lastLocation, self.goal)
        self.lastLocation = self.getLocation()
        distNow = self.errInLocation(self.location, self.goal)
        difference = distLast - distNow
        self.__rangeDriven += difference if 10 > difference > -5 else 0

    def getControl(self, testingRide=False):
        maxSteerChange = self.dynamicMaxSteeringChange()
        control, waypoint = self.agentAction()
        agentSteer = control.steer

        if testingRide:
            self.steer = control.steer
            return control

        radar = self.sensorManager.radarMeasurement()
        left, right = self.sensorManager.lines()
        if np.sum(left) == 0 or np.sum(right) == 0:
            # Lines is not detected!
            self.steer = self.limitSteering(self.calcSteer(agentSteer, maxSteerChange))
        else:
            # Lines is detected!
            inputs = self.processInputs(left, right, radar, agentSteer, waypoint)
            outputNeural = self.nn.run(inputs, maxSteerChange)[0][0]
            self.steer = self.limitSteering(outputNeural)

        self.recordEachStep(agentSteer)
        control.steer = self.steer
        return control

    def limitSteering(self, askedChange):
        actualSteer = self.steer
        askedSteer = actualSteer + askedChange

        if askedSteer > self.limit:
            askedSteer = self.limit
        elif askedSteer < -self.limit:
            askedSteer = -self.limit

        return askedSteer

    def processInputs(self, left, right, radar, agentSteer, waypoint):
        inputs = np.array([])

        for asked in self.askedInputs:
            if asked == InputsEnum.linedetect:
                inputs = np.append(inputs, self.nn.normalizeLinesInputs(left, right))
            elif asked == InputsEnum.radar:
                inputs = np.append(inputs, self.nn.normalizeRadarInputs(radar))
            elif asked == InputsEnum.agent:
                inputs = np.append(inputs, self.nn.normalizeAgent(agentSteer))
            elif asked == InputsEnum.metrics:
                inputs = np.append(inputs, self.nn.normalizeMetrics(self.metrics, self.limit))
            elif asked == InputsEnum.binaryknowledge:
                inputs = np.append(inputs, self.nn.normalizeBinary(self.getBinaryKnowledge(left, right, agentSteer, radar)))
            elif asked == InputsEnum.navigation:
                inputs = np.append(inputs, self.nn.normalizeNavigation(self.location, waypoint))

        return inputs

    def returnVehicleResults(self):
        return self.__positionHistory, self.__leftLinePlanner, self.__rightLinePlanner, self.__optimalPath

    def calcSteer(self, agentSteer, maxChange):
        direction = 1 if agentSteer > self.steer else -1
        difference = min(abs(agentSteer - self.steer), maxChange*2)
        return direction * difference

    def dynamicMaxSteeringChange(self):
        # Speed will be 0 - 50, so max division will be 5
        speed = self.getSpeed() / 10
        return self.defaultSteerMaxChange / speed if speed > 1 else self.defaultSteerMaxChange

    def checkGoal(self):
        '''
        Checks, how far is vehicle from current goal. If goal is reached, new waypoint is loaded. When no more waypoints
        it will return True as signing that current task is done.
        :return: bool
        '''
        dist = self.diffToLocation(self.goal)
        self.print(f"Distance to goal is: {dist}")
        if dist < 0.2 or self.agent.done():
            self.__reachedGoals += 1
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

    def inCycle(self):
        now = time.time()
        timeBool = now > 300 + self.startTime  # gives timeout 5 min

        if len(self.toGoal) < self.toGoal.maxlen:
            dequeBool = False
        else:
            left = self.toGoal.popleft()
            right = self.toGoal.pop()
            self.toGoal.append(right)
            dequeBool = True if abs(left - right) < 5 else False
        if timeBool or dequeBool:
            return True
        else:
            return False

    def applyConfig(self, testing):
        self.debug = testing
        if testing:
            self.numMeasure = 8
            self.sensorManager.applyTesting()

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
        dist = l1.distance(l2)
        return dist

    def getBinaryKnowledge(self, left, right, agentSteer, radar) -> list:
        '''
        Process all needed datas and calc binary knowledge. More information in CarlaConfig.py
        :return: list(len=4) of binary knowledges (-1,0, or 1)
        '''
        left1 = np.abs(left[0])
        right1 = np.abs(right[0])
        '1: Based on lines detected (left line is closer than right -> turn right: 1)'
        if left1 > right1:
            one = 1
        elif left1 < right1:
            one = -1
        else:
            one = 0
        '2: Based on difference between agent steering and actual steering'
        if self.steer < agentSteer:
            two = -1
        elif self.steer > agentSteer:
            two = 1
        else:
            two = 0
        '3: Which radar measurement has the shortest range [-1 for left, 0 for center, 1 for right]'
        minRadar = np.min(radar)
        if minRadar == radar[0]:
            three = -1
        elif minRadar == radar[1]:
            three = 0
        else:
            three = 1
        '4: Based on actual speed towards goal'
        x = abs(self.velocity.x)
        y = abs(self.velocity.y)
        diffX = abs(self.goal.x - self.location.x)
        diffY = abs(self.goal.y - self.location.y)
        # We can expect that direction will be ok, so just use abs values
        if diffX > diffY and y > x:
            four = 1
        elif diffY > diffX and x > y:
            four = -1
        else:
            four = 0
        return [one, two, three, four]

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

    def print(self, message: str):
        if self.debug:
            print(message)

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
        self.print("Destroying Vehicle {id}".format(id=self.vehicleID))
        try:
            self.sensorManager.destroy()
            self.me.destroy()
            self.environment.tick()
        except:
            print(f"Error in destroying of {self.vehicleID}")
