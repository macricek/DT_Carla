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
    '''
    Vehicle implementation
    @author: Marko Chylik
    @Date: May, 2022
    '''
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
        '''
        Init vehicle and spawn it
        :param environment: reference to CarlaEnvironment
        :param spawnLocation: point on map, where Vehicle should be spawned
        :param neuralNetwork: NeuralNetwork object
        :param id: which vehicle from population of vehicles this is
        '''
        super(Vehicle, self).__init__()
        self.vehicleID = id
        self.environment = environment
        self.path = environment.path()
        self.askedInputs = environment.config.loadAskedInputs()[0]
        self.debug = self.environment.debug
        self.fald = self.environment.faLineDetector

        self.me = None
        while not self.me:
            self.me = self.environment.world.try_spawn_actor(self.environment.blueprints.filter('model3')[0], spawnLocation)
            if not self.me:
                for _ in range(10):
                    self.environment.tick()

        self.nn = neuralNetwork

        self.speed = 0
        self.vehicleStopped = 0
        self.steer = 0
        self.limit = 0.8
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
        :param tickNum: current tickNum from server
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
            self.storeCurrentData()
        return True

    def agentAction(self):
        '''
        Usage of agent to reach expected speed and navigation implementation
        :return: carla.Control, current waypoint from navigation
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
        :return: None
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
        '''
        Record the training ride and make a summary of it into dictionary.
        :return: dictionary
        '''
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
        '''
        Some of the parameters needs to be recorded at every step.
        :param agentSteer: steer suggested by agent
        :return: None
        '''
        self.__errDec += abs(agentSteer - self.steer)

        distLast = self.errInLocation(self.lastLocation, self.goal)
        self.lastLocation = self.getLocation()
        distNow = self.errInLocation(self.location, self.goal)
        difference = distLast - distNow
        self.__rangeDriven += difference if 10 > difference > -5 else 0

    def getControl(self, useAutopilot=False):
        '''
        calculate the steering:
        :param useAutopilot: use autopilot
        :return: carla.Control
        '''
        maxSteerChange = self.dynamicMaxSteeringChange()
        control, waypoint = self.agentAction()
        agentSteer = control.steer

        if useAutopilot:
            self.steer = control.steer
            return control

        radar = self.sensorManager.radarMeasurement()
        left, right = self.sensorManager.lines()

        inputs = self.processInputs(left, right, radar, agentSteer, waypoint)
        outputNeural = self.nn.run(inputs, maxSteerChange)[0][0]
        self.steer = self.limitSteering(outputNeural)

        self.recordEachStep(agentSteer)
        control.steer = self.steer
        return control

    def limitSteering(self, askedChange):
        '''
        block bigger changes - this should never happen, but we need to ensure it
        :param askedChange: change of steering
        :return: real change of steering
        '''
        actualSteer = self.steer
        askedSteer = actualSteer + askedChange

        if askedSteer > self.limit:
            askedSteer = self.limit
        elif askedSteer < -self.limit:
            askedSteer = -self.limit

        return askedSteer

    def processInputs(self, left, right, radar, agentSteer, waypoint):
        '''
        based on config, we want to use just some of inputs. This function will pick them for us
        :param left: left line from detector
        :param right: right line from detector
        :param radar: radar measures
        :param agentSteer: suggested agent steering
        :param waypoint: the next waypoint from navigation
        :return: ndarray of all inputs
        '''
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

    def storeCurrentData(self):
        '''
        when we are running test scenarios / showing the best on training, we want to store the history of movement of
        the vehicle, left, right lane and optimal path based on navigation - store it in global class members
        :return: None
        '''
        if self.debug:
            waypoints = self.agent.get_waypoints()[0]
            if waypoints:
                self.__positionHistory.append(self.location)
                ll = waypoints.get_left_lane()
                if ll:
                    self.__leftLinePlanner.append(ll.transform.location)
                rl = waypoints.get_right_lane()
                if rl:
                    self.__rightLinePlanner.append(rl.transform.location)
                self.__optimalPath.append(waypoints.transform.location)

    def returnVehicleResults(self):
        '''
        when we are running test scenarios / showing the best on training, return all eligible measures when vehicle
         ends
        :return: position, left line, right line, navigation path
        '''
        return self.__positionHistory, self.__leftLinePlanner, self.__rightLinePlanner, self.__optimalPath

    def dynamicMaxSteeringChange(self):
        '''
        Max steering change based on the current speed of vehicle.
        If speed is more than 10km/h, we will use 0,8/speed; else just 0,8.
        :return: float
        '''
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
        '''
        determine, that vehicle is not moving towards the goals and probably are stucked in cycle. It's used just in
        training scenarios.
        :return: bool
        '''
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
        '''
        If we are running test config, we want to change a few things - Record the vehicle's path more frequently
                                                                        apply test config for sensors/cameras
        :param testing: bool
        :return: None
        '''
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
        '''
        difference to location from current location of vehicle
        :param location: carla.Location
        :return: distance [float]
        '''
        self.getLocation()
        dist = location.distance(self.location)
        return dist

    @staticmethod
    def errInLocation(l1: carla.Location, l2: carla.Location):
        '''
        error from L1 to L2
        :param l1: carla.Location
        :param l2: carla.Location
        :return: distance [float]
        '''
        dist = l1.distance(l2)
        return dist

    def getBinaryKnowledge(self, left, right, agentSteer, radar) -> list:
        '''
        Process all needed datas and calc binary knowledge. More information in CarlaConfig.py
        :return: list(len=4) of binary knowledges (-1,0, or 1)
        '''
        try:
            left1 = np.abs(left[0])
            right1 = np.abs(right[0])
            '1: Based on lines detected (left line is closer than right -> turn right: 1)'
            if left1 > right1:
                one = 1
            elif left1 < right1:
                one = -1
            else:
                one = 0
        except:
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
