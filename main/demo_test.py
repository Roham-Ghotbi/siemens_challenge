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

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys

from tpc.python_labeler import Python_Labeler

from tpc.perception.cluster_registration import run_connected_components, display_grasps, has_multiple_objects, grasps_within_pile
from tpc.perception.singulation import find_singulation, display_singulation
from tpc.perception.crop import crop_img
from tpc.perception.bbox import bbox_to_mask, bbox_to_grasp
from perception import ColorImage, BinaryImage
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import tpc.config.config_tpc as cfg

from il_ros_hsr.core.rgbd_to_map import RGBD2Map

if __name__ == "__main__":
    for imgnum in range(1164, 1171):
    # for imgnum in range(1170, 1171):
        c_img = cv2.imread("debug_imgs/singulation_tests/IMG_" + str(imgnum) + ".jpg")

        main_mask = crop_img(c_img)
        col_img = ColorImage(c_img)
        workspace_img = col_img.mask_binary(main_mask)

        center_masses, directions, masks = run_connected_components(workspace_img,
            cfg.DIST_TOL, cfg.COLOR_TOL, cfg.SIZE_TOL, viz=False)
        if len(center_masses) == 0:
            print("cleared workspace")
        else:
            has_multiple = [has_multiple_objects(col_img.mask_binary(m), alg="hsv") for m in masks]
            print "has multiple objects?:", has_multiple
            grasps = []
            viz_info = []
            for i in range(len(center_masses)):
                if not has_multiple[i]:
                    viz_info.append([center_masses[i], directions[i]])
                    grasps.append("a grasp")
                else:
                    new_cms, new_dirs = grasps_within_pile(col_img.mask_binary(masks[i]))
                    for j in range(len(new_cms)):
                        viz_info.append([new_cms[j], new_dirs[j]])
                        grasps.append("a grasp")
            if len(grasps) > 0:
                print("grasping")
                display_grasps(workspace_img, [v[0] for v in viz_info], [v[1] for v in viz_info],
                    name = "debug_imgs/out/grasps_" + str(imgnum))
            else:
                print("singulating")
                for pilenum in range(len(masks)):
                    curr_pile = masks[pilenum]
                    other_piles = masks[:pilenum] + masks[pilenum+1:]
                    start, end, rot, free_pix = find_singulation(col_img, main_mask, curr_pile,
                        other_piles, alg="border") 
                    display_singulation(start, end, rot, workspace_img, free_pix, 
                        name = "debug_imgs/out/singulate_" + str(imgnum) + "_" + str(pilenum))
