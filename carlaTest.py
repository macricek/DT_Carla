import glob
import threading
import os
import sys
import random
import time
import numpy as np
import cv2
import math

# from Carla doc
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


def setSimulation(self):
    self.world.unload_map_layer(carla.MapLayer.All)  # make map easier
    self.world.load_map_layer(carla.MapLayer.Walls)
    self.originalSettings = self.world.get_settings()
    settings = self.world.get_settings()
    settings.fixed_delta_seconds = 0.05
    self.world.apply_settings(settings)


def stopSimulation(self):
    self.world.load_map_layer(carla.MapLayer.All)
    self.world.apply_settings(self.originalSettings)