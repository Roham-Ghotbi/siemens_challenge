import os
import numpy as np

"""CONFIG FILE FOR TPC LEGO PROJECT"""

"""OPTIONS FOR DEMO"""
#whether to save rollouts
COLLECT_DATA = False

#whether to show plots/ask for success
QUERY = False

#whether to attempt multiple grasps from just 1 image; if true, susceptible to error from open loop control
CHAIN_GRASPS = False



"""TABLE SETUP SPECIFIC VALUES"""
#ordered by layout on floor (top to bottom with close row first)
HUES_TO_BINS = ["orange", "green-yellow", "cyan", "black", "red", "green", "blue", "yellow"]
labelclasses = ["Wrench", "Hammer", "Screwdriver", "Tape Measure", "Glue", "Tape"]

"""EMPIRICALLY TUNED PARAMETERs"""
#CONENCTED COMPONENTS ALG PARAMETERS
#number of pixels apart to be singulated
DIST_TOL = 5
#background range for thresholding the image
COLOR_TOL = 40
#number of pixels necssary for a cluster
SIZE_TOL = 350
#amount to scale image down by to run algorithm (for speed)
SCALE_FACTOR = 2

#ROBOT PARAMETERS
#distance grasp extends
LINE_SIZE = 38
#side length of square that checks grasp collisions
#increase range to reduce false positives
CHECK_RANGE = 2
#range around grasp point used to calculate average depth value
ZRANGE = 20

#HSV PARAMETERS
#cv2 range for HSV hue values
HUE_RANGE = 180.0
#cv2 range for HSV sat values
SAT_RANGE = 255.0
#cv2 range for HSV value values
VALUE_RANGE = 255.0
#fraction of saturation range that is white
WHITE_FACTOR = 0.15 #0.1
#fraction of value range that is black
BLACK_FACTOR = 0.3
#carving up HSV color space by lego-specific colors
#see https://en.wikipedia.org/wiki/HSL_and_HSV (scaled down from 360 to 180 degrees)
HUE_VALUES = {90: "cyan", 120: "blue", 0: "red", 10: "orange", 30: "yellow",
	60: "green", 35: "green-yellow"}
#include black as special case
ALL_HUE_VALUES = HUE_VALUES.copy()
ALL_HUE_VALUES[-1] = "black"

#singulation parameters
#factor to move start point by so it is not in the pile
SINGULATE_START_FACTOR = 1.2
#factor to move end point by towards start point
SINGULATE_END_FACTOR = 0.75

"""PATHS AND DATASET PARAMETERS"""
#convenience parameter to change paths based on machine
on_autolab = True
if on_autolab:
	ROOT_DIR = '/media/autolab/1tb/data/'
	DATA_PATH = ROOT_DIR + 'tpc/'
	IMG_MODULE = 'perception'
else:
	ROOT_DIR = '/Users/chrispowers/Documents/research/tpc/'
	DATA_PATH = ROOT_DIR + 'data/'
	IMG_MODULE = 'tpc.perception.image'
ROLLOUT_PATH = DATA_PATH+'rollouts-3-9/'
