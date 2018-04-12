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
sys.path.append('/home/autolab/Workspaces/michael_working/hsr_web')
from web_labeler import Web_Labeler
from tpc.perception.connected_components import get_cluster_info, merge_groups

import tpc.config.config_tpc as cfg
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

class LabelDemo():

    def __init__(self):
        """
        Class to run HSR lego task

        """

        self.web = Web_Labeler()
        print "after thread"

    def bbox_to_fg(self, bbox, c_img, col_img):
        obj_mask = crop_img(c_img, bycoords = [bbox[1], bbox[3], bbox[0], bbox[2]])
        obj_workspace_img = col_img.mask_binary(obj_mask)
        fg = obj_workspace_img.foreground_mask(cfg.COLOR_TOL, ignore_black=True)
        return fg, obj_workspace_img

    def test_bbox_overlap(self, box_a, box_b):
        #bbox has format [xmin, ymin, xmax, ymax]
        if box_a[0] > box_b[2] or box_a[2] < box_b[0]:
            return False
        if box_a[1] > box_b[3] or box_a[3] < box_b[1]:
            return False 
        return True 

    def find_isolated_objects(self, obj_info):
        valid_objs = []
        for curr_ind in range(len(obj_info)):
            curr_obj = obj_info[curr_ind]
            curr_bbox = curr_obj[0]

            for test_ind in range(len(obj_info)):
                if curr_ind != test_ind:
                    test_obj = obj_info[test_ind]
                    test_bbox = test_obj[0]
                    if self.test_bbox_overlap(curr_bbox, test_bbox):
                        overlap = True
                        break 
            if not overlap:
                valid_objs.append(curr_obj)
        return valid_objs

    def select_first_obj(self, single_objs):
        #bbox has format [xmin, ymin, xmax, ymax]
        #obj has format (bbox, label)
        bottom_left_obj =  min(single_objs, key = lambda x: x[0][1] + x[0][2])
        return bottom_left_obj

    def run_grasp(self, bbox, class_label, c_img, col_img, workspace_img):
        #bbox has format [xmin, ymin, xmax, ymax]
        fg, obj_w = self.bbox_to_fg(bbox, c_img, col_img)
        cv2.imwrite("debug_imgs/test.png", obj_w.data)

        groups = get_cluster_info(fg)
        display_grasps(workspace_img, groups)

    def run_singulate(self, col_img, main_mask, to_singulate):
        singulator = Singulation(col_img, main_mask, [g.mask for g in to_singulate])
        waypoints, rot, free_pix = singulator.get_singulation()

        singulator.display_singulation()

    def label_demo(self):
        time.sleep(3) #making sure the robot is finished moving

        sample_data_paths = ["debug_imgs/data_chris/test" + str(i) + ".png" for i in range(3)]
        img_ind = 0
        c_img = cv2.imread(sample_data_paths[img_ind])

        while not (c_img is None):
            path = "/home/autolab/Workspaces/michael_working/siemens_challenge/debug_imgs/web.png"
            cv2.imwrite(path, c_img)
            time.sleep(2) #make sure new image is written before being read

            # print "\n new iteration"
            main_mask = crop_img(c_img, simple=True)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)

            labels = self.web.label_image(path)

            objs = labels['objects']
            obj_info = [(obj['box'], obj['class']) for obj in objs]
            single_objs = self.find_isolated_objects(obj_info)

            if len(single_objs) > 0:
                to_grasp = self.select_first_obj(single_objs)
                self.run_grasp(to_grasp[0], to_grasp[1], c_img, col_img, workspace_img)
            else:
                fg_imgs = [self.bbox_to_fg(obj[0], c_img, col_img) for obj in obj_info]
                groups = [get_cluster_info(fg[0])[0] for fg in fg_imgs]
                groups = merge_groups(groups, cfg.DIST_TOL)
                self.run_singulate(col_img, main_mask, groups)
            img_ind += 1
            c_img = cv2.imread(sample_data_paths[img_ind])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False

    task = LabelDemo()
    task.label_demo()