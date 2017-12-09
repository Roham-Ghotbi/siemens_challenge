from hsrb_interface import geometry
import hsrb_interface
from geometry_msgs.msg import PoseStamped, Point, WrenchStamped
import geometry_msgs
import controller_manager_msgs.srv
import cv2
from cv_bridge import CvBridge, CvBridgeError
import IPython
from numpy.random import normal
import time
#import listener
import thread

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
from il_ros_hsr.core.joystick import  JoyStick

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
from tf import TransformListener
import tf
import rospy

from il_ros_hsr.core.grasp_planner import GraspPlanner
from il_ros_hsr.core.crane_gripper import Crane_Gripper

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys

from il_ros_hsr.p_pi.tpc.gripper import Lego_Gripper
from tpc.perception.cluster_registration import run_connected_components, visualize, has_multiple_objects
from tpc.perception.singulation import find_singulation, display_singulation
from tpc.perception.crop import crop_img
from perception import ColorImage, BinaryImage
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import il_ros_hsr.p_pi.bed_making.config_bed as cfg

from il_ros_hsr.core.rgbd_to_map import RGBD2Map

SINGULATE = True

#number of pixels apart to be singulated
DIST_TOL = 5
#background range for thresholding the image
COLOR_TOL = 40

if __name__ == "__main__":
    # c_img = cv2.imread("data/example_images/frame_40_10.png")
    c_img = cv2.imread("debug_imgs/c_img.png")
    mask = crop_img(c_img)
    c_img = ColorImage(c_img)
    c_img = c_img.mask_binary(mask)
    center_masses, directions, masks = run_connected_components(c_img, DIST_TOL, COLOR_TOL, viz=True)
    nums = [has_multiple_objects(m) for m in masks]
    print(nums)

    start, end = find_singulation(c_img, mask, masks[0])