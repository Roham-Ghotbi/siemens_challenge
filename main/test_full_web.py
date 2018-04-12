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
from tpc.perception.cluster_registration import run_connected_components, display_grasps
from tpc.perception.groups import Group

from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.manipulation.primitives import GraspManipulator
from il_ros_hsr.core.rgbd_to_map import RGBD2Map
sys.path.append('/home/autolab/Workspaces/michael_working/hsr_web')
from web_labeler import Web_Labeler
from tpc.perception.connected_components import get_cluster_info, merge_groups
from tpc.perception.bbox import Bbox, find_isolated_objects, select_first_obj

import tpc.config.config_tpc as cfg
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

"""
This class is for use with the robot
Pipeline it tests is labeling all objects in an image using web interface,
then choosing between and predicting a singulation or a grasp,
then the robot executing the predicted motion
"""

class FullWebDemo():

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

        self.com.go_to_initial_state(self.whole_body)

        self.grasp_count = 0

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()

        self.gp = GraspPlanner()
        self.gripper = Crane_Gripper(self.gp, self.cam, self.com.Options, self.robot.get('gripper'))
        self.suction = Suction_Gripper(self.gp, self.cam, self.com.Options, self.robot.get('suction'))

        self.gm = GraspManipulator(self.gp, self.gripper, self.suction, self.whole_body, self.omni_base, self.tl)

        self.web = Web_Labeler()
        print "after thread"


    def run_grasp(self, bbox, c_img, col_img, workspace_img, d_img):
        print("grasping a " + cfg.labels[bbox.label])
        #bbox has format [xmin, ymin, xmax, ymax]
        fg, obj_w = bbox.to_mask(c_img, col_img)
        # cv2.imwrite("debug_imgs/test.png", obj_w.data)
        # cv2.imwrite("debug_imgs/test2.png", fg.data)
        groups = get_cluster_info(fg)
        curr_tol = cfg.COLOR_TOL 
        while len(groups) == 0 and curr_tol > 10:
            curr_tol -= 5
            #retry with lower tolerance- probably white object 
            fg, obj_w = bbox.to_mask(c_img, col_img, tol=curr_tol)
            groups = get_cluster_info(fg)

        if len(groups) == 0:
            print("No object within bounding box")
            return False

        display_grasps(workspace_img, groups)

        group = groups[0]
        pose,rot = self.gm.compute_grasp(group.cm, group.dir, d_img)
        grasp_pose = self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data)

        self.gm.execute_grasp(grasp_pose, class_num = bbox.label)

    def run_singulate(self, col_img, main_mask, to_singulate, d_img):
        print("singulating")
        singulator = Singulation(col_img, main_mask, [g.mask for g in to_singulate])
        waypoints, rot, free_pix = singulator.get_singulation()

        singulator.display_singulation()
        self.gm.singulate(waypoints, rot, col_img.data, d_img, expand=True)

    def full_web_demo(self):
        self.gm.position_head()

        time.sleep(3) #making sure the robot is finished moving

        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()

        while not (c_img is None or d_img is None):
            path = "/home/autolab/Workspaces/michael_working/siemens_challenge/debug_imgs/web.png"
            cv2.imwrite(path, c_img)
            time.sleep(2) #make sure new image is written before being read

            # print "\n new iteration"
            main_mask = crop_img(c_img, simple=True)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)

            labels = self.web.label_image(path)

            objs = labels['objects']
            bboxes = [Bbox(obj['box'], obj['class']) for obj in objs]
            single_objs = find_isolated_objects(bboxes)

            if len(single_objs) > 0:
                to_grasp = select_first_obj(single_objs)
                self.run_grasp(to_grasp, c_img, col_img, workspace_img, d_img)
            else:
                #for accurate singulation should have bboxes for all
                fg_imgs = [box.to_mask(c_img, col_img) for box in bboxes]
                groups = [get_cluster_info(fg[0])[0] for fg in fg_imgs]
                groups = merge_groups(groups, cfg.DIST_TOL)
                self.run_singulate(col_img, main_mask, groups, d_img)
            
            #reset to start
            self.whole_body.move_to_go()
            self.gm.move_to_home()
            self.gm.position_head()
            time.sleep(3)

            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False

    task = FullWebDemo()
    task.full_web_demo()