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
import os, sys


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
		""" Class to run HSR lego task """
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
		
		self.gm = GraspManipulator(self.gp, self.gripper, self.suction, self.whole_body, self.omni_base)
		
		self.collision_world = hsrb_interface.collision_world.CollisionWorld("global_collision_world")
		self.collision_world.remove_all()
		self.collision_world.add_box(x=.8,y=.9,z=0.5,pose=geometry.pose(y=1.4,z=0.15),frame_id='map')
		print("finished initializing collect data class")


	def collect_data(self):
		""" 
		Run this a few times to check that the rgb images are reasonable.
		If not, rearrange the setup and try again. Delete any images saved after
		that, the run this "for real".
		"""
		self.gm.position_head()
		IMDIR_RGB = 'image_rgb/'
		IMDIR_DEPTH = 'image_depth/'
		time.sleep(5) #making sure the robot is finished moving
		print("after calling gm.position_head() w/several second delay")
		print("\nwhole body joint positions:\n{}".format(self.whole_body.joint_positions))

		while True:
			num = len([x for x in os.listdir(IMDIR_RGB) if 'png' in x])
			c_img = self.cam.read_color_data()
			d_img = self.cam.read_depth_data()
			cv2.imshow('rgb/image_raw', c_img)
			cv2.imshow('depth/image_raw', d_img)
			fname1 = IMDIR_RGB+'rgb_raw_{}.png'.format(str(num).zfill(4))
			fname2 = IMDIR_DEPTH+'depth_raw_{}.png'.format(str(num).zfill(4))
			cv2.imwrite(fname1, c_img)
			cv2.imwrite(fname2, d_img)
			print("just saved {} and {}. NOW REARRANGE SETUP!!".format(fname1, fname2))
			IPython.embed() #re-arrange setup here


if __name__ == "__main__":
    task = CollectData()
    task.collect_data()

	## from earlier
	#bridge = CvBridge()
    #rospy.init_node('main', anonymous=True)
    #rospy.on_shutdown(OnShutdown_callback)

    ## TODO `depth_registered` topic doesn't produce readable output, also
    ## `depth` topic doesn't seem to be published.
    #hic_list = [HeadImage(), HeadImage()]
    #rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw",
    #                 Image, hic_list[0], queue_size=1)
    #rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image_raw",
    #                 Image, hic_list[1], queue_size=1)
    #isRunning = True
    #rospy.sleep(5)

    #while isRunning:
    #    # Obtain the images. For depth, cant use 'bgr8' I think.
    #    data = [hic.get_im() for hic in hic_list]
    #    img1 = bridge.imgmsg_to_cv2(data[0], "bgr8")
    #    img2 = bridge.imgmsg_to_cv2(data[1])
    #    print("img1.shape: {}, img2.shape: {}".format(img1.shape, img2.shape))

    #    # Show images, get file name, etc. Depth doesn't work yet.
    #    cv2.imshow('rgb/image_raw', img1)
    #    #cv2.imshow('depth/image_raw', img2) # Seems to be just black
    #    num = len([x for x in os.listdir(IMDIR_RGB) if 'png' in x])
    #    fname1 = IMDIR_RGB+'rgb_raw_{}.png'.format(str(num).zfill(4))
    #    #fname2 = IMDIR_DEPTH+'depth_raw_{}.png'.format(str(num).zfill(4))

    #    # Save images or exit if needed. Again, depth doesn't work yet.
    #    key = cv2.waitKey(0)
    #    if key in ESC_KEYS:
    #        print("Exiting now ...")
    #        sys.exit()
    #    cv2.imwrite(fname1, img1)
    #    print("saved: {}".format(fname1))
    #    #cv2.imwrite(fname2, img2)
    #    #print("saved: {}".format(fname2))
