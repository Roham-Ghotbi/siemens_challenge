#from hsrb_interface import geometry
#import hsrb_interface
#from geometry_msgs.msg import PoseStamped, Point, WrenchStamped
#import geometry_msgs
#import controller_manager_msgs.srv
import cv2
from cv_bridge import CvBridge, CvBridgeError
#import IPython
from numpy.random import normal
import time
#import listener
#import thread

#from geometry_msgs.msg import Twist
#from sensor_msgs.msg import Joy

#from il_ros_hsr.core.sensors import  RGBD, Gripper_Torque, Joint_Positions
#from il_ros_hsr.core.joystick import  JoyStick

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
#from tf import TransformListener
#import tf
#import rospy

#from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys

from tpc.python_labeler import Python_Labeler

from tpc.perception.cluster_registration import run_connected_components, display_grasps, has_multiple_objects, grasps_within_pile
from tpc.perception.singulation import find_singulation, display_singulation
from tpc.perception.crop import crop_img
from tpc.perception.bbox import bbox_to_mask, bbox_to_grasp
from perception import ColorImage, BinaryImage
#from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import tpc.config.config_tpc as cfg

#from il_ros_hsr.core.rgbd_to_map import RGBD2Map

if __name__ == "__main__":
    c_img = cv2.imread("debug_imgs/img7.png")
    d_img = self.cam.read_depth_data()
    main_mask = crop_img(c_img)
    col_img = ColorImage(c_img)
    workspace_img = col_img.mask_binary(main_mask)

    #compute clusters (can have 1 or multiple legos)
    center_masses, directions, masks = run_connected_components(workspace_img,
        cfg.DIST_TOL, cfg.COLOR_TOL, cfg.SIZE_TOL, viz=True)
    cluster_info = zip(center_masses, directions, masks)

    print "num masses", len(center_masses)
    if len(center_masses) == 0:
        print("cleared workspace")
    else:
        #for each cluster, compute grasps
        to_singulate = []
        to_grasp = []
        for c_info in cluster_info:
            grasp_cms, grasp_dirs, grasp_masks = grasps_within_pile(col_img.mask_binary(c_info[2]))
            IPython.embed()
            if len(grasp_cms) == 0:
                to_singulate.append(c_info)
            else:
                for i in range(len(grasp_cms)):
                    # pose,rot = self.gm.compute_grasp(grasp_cms[i], grasp_dirs[i], d_img)
                    # grasp_pose = self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data)
                    # class_num = hsv_classify(col_img.mask_binary(grasp_masks[i]))
                    # to_grasp.append((grasp_cms[i], grasp_dirs[i], grasp_masks[i], grasp_pose, class_num))
                    to_grasp.append((i))
        print("num grasps: " + str(len(to_grasp)))