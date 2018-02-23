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

from il_ros_hsr.core.crane_gripper import Crane_Gripper
from il_ros_hsr.core.grasp_planner import GraspPlanner

from il_ros_hsr.core.suction_gripper import Suction_Gripper
from il_ros_hsr.core.suction import Suction

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys


from il_ros_hsr.p_pi.tpc.gripper import Lego_Gripper
from tpc.perception.cluster_registration import run_connected_components, display_grasps, \
    grasps_within_pile, view_hsv, get_hsv_hist, hsv_classify
from tpc.perception.crop import crop_img
from tpc.data_manager import DataManager
from tpc.manipulation.primitives import GraspManipulator
from perception import ColorImage, BinaryImage
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import tpc.config.config_tpc as cfg

from il_ros_hsr.core.rgbd_to_map import RGBD2Map

import string
import random


"""
script used to find specific poses incrementally
"""
def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

if __name__ == "__main__":
    robot = hsrb_interface.Robot()
    rgbd_map = RGBD2Map()

    omni_base = robot.get('omni_base')
    whole_body = robot.get('whole_body')


    side = 'BOTTOM'

    cam = RGBD()
    com = COM()

    com.go_to_initial_state(whole_body)

    tt = TableTop()
    tt.find_table(robot)

    grasp_count = 0

    br = tf.TransformBroadcaster()
    tl = TransformListener()

    gp = GraspPlanner()
    gripper = Crane_Gripper(gp, cam, com.Options, robot.get('gripper'))
    suction = Suction_Gripper(gp, cam, com.Options, robot.get('suction'))
    gm = GraspManipulator(gp, gripper, suction, whole_body, omni_base, tt)
    gm.position_head()

    print "after thread"
    curr_offsets = np.array([0, 0, -0.5])
    curr_rot = np.array([0.0,0.0,1.57])

    while True:
        label = id_generator()
        tt.make_new_pose(curr_offsets,label,rot = curr_rot)
        whole_body.move_end_effector_pose(geometry.pose(z=-0.1), label)
        delta = raw_input()
        while not (delta in ["+x", "-x", "+y", "-y", "+z", "-z"]):
            print("not a valid delta")
            delta = raw_input()

        if delta == "+x":
            curr_offsets[0] += 0.05
        elif delta == "-x":
            curr_offsets[0] -= 0.05
        elif delta == "+y":
            curr_offsets[1] += .05
        elif delta == "-y":
            curr_offsets[1] -= .05
        elif delta == "+z":
            curr_offsets[2] += .05
        elif delta == "-z:":
            curr_offsets[2] -= .05
        
        print(curr_offsets)


