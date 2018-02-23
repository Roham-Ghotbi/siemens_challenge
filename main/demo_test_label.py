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

from tpc.perception.cluster_registration import run_connected_components, visualize
from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.perception.bbox import bbox_to_mask, bbox_to_grasp
from perception import ColorImage, BinaryImage
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import il_ros_hsr.p_pi.bed_making.config_bed as cfg

from il_ros_hsr.core.rgbd_to_map import RGBD2Map

if __name__ == "__main__":
    c_img = cv2.imread("debug_imgs/f1/c_img.png")

    labeler = Python_Labeler()
    data = labeler.label_image(c_img)

    grasp_boxes = []
    suction_boxes = []
    singulate_boxes = []

    for i in range(data['num_labels']):
        bbox = data['objects'][i]['box']
        classnum = data['objects'][i]['class']
        classname = ['grasp', 'singulate', 'suction', 'quit'][classnum]
        if classname == "grasp":
            grasp_boxes.append(bbox)
        elif classname == "suction":
            suction_boxes.append(bbox)
        elif classname == "singulate":
            singulate_boxes.append(bbox)
        elif classname == "quit":
        	break 

    main_mask = crop_img(c_img)
    col_img = ColorImage(c_img)
    workspace_img = col_img.mask_binary(main_mask)

    grasps = []
    viz_info = []
    for i in range(len(grasp_boxes)):
        bbox = grasp_boxes[i]
        center_mass, direction = bbox_to_grasp(bbox, c_img, d_img)

        viz_info.append([center_mass, direction])

    suctions = []
    for i in range(len(suction_boxes)):
        suctions.append("compute_suction?")

    if len(grasps) > 0 or len(suctions) > 0:
    	print("grasping")
        cv2.imwrite("grasps.png", visualize(workspace_img, [v[0] for v in viz_info], [v[1] for v in viz_info]))
    elif len(singulate_boxes) > 0:
        print("singulating")
        #should also consider non-singulate boxes here
        obj_masks = [bbox_to_mask(sbox, c_img) for sbox in singulate_boxes] + [bbox_to_mask(gbox, c_img) for gbox in singulate_boxes]
        start, end, rot = find_singulation(col_img, main_mask, obj_masks[0], obj_masks[1:], gripper_to_free = True, viz=True)
        IPython.embed()
        start, end, rot = find_singulation(col_img, main_mask, obj_masks[0], obj_masks[1:], viz=True)
        # IPython.embed()
    else:
        print("cleared workspace")