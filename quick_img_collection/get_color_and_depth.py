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
    grasps_within_pile, hsv_classify
from tpc.perception.groups import Group

from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.manipulation.primitives import GraspManipulator
from tpc.data_manager import DataManager
from il_ros_hsr.p_pi.bed_making.table_top import TableTop
from il_ros_hsr.core.rgbd_to_map import RGBD2Map

import tpc.config.config_tpc as cfg
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

class CollectData():

    def __init__(self):
        """
        Class to run HSR lego task

        """

        self.robot = hsrb_interface.Robot()
        self.rgbd_map = RGBD2Map()

        self.omni_base = self.robot.get('omni_base')
        self.whole_body = self.robot.get('whole_body')

        self.side = 'BOTTOM'

        self.cam = RGBD()
        self.com = COM()
        # if not DEBUG:
        self.com.go_to_initial_state(self.whole_body)

        #     self.tt = TableTop()
        #     self.tt.find_table(self.robot)

        self.grasp_count = 0

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()

        self.gp = GraspPlanner()
        self.gripper = Crane_Gripper(self.gp, self.cam, self.com.Options, self.robot.get('gripper'))
        self.suction = Suction_Gripper(self.gp, self.cam, self.com.Options, self.robot.get('suction'))

        self.gm = GraspManipulator(self.gp, self.gripper, self.suction, self.whole_body, self.omni_base, self.tl)

        self.collision_world = hsrb_interface.collision_world.CollisionWorld("global_collision_world")
        self.collision_world.remove_all()
        self.collision_world.add_box(x=.8,y=.9,z=0.5,pose=geometry.pose(y=1.4,z=0.15),frame_id='map')

        print "after thread"

    def collect_data(self):
        self.gm.position_head()

        time.sleep(3) #making sure the robot is finished moving
        i = 0
        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            cv2.imwrite("debug_imgs/data_chris/test" + str(i) + ".png", c_img)
            #can also save d_img
            IPython.embed() #re-arrange setup here
            i += 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False

    task = CollectData()
    task.collect_data()
