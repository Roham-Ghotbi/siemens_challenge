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

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys


from il_ros_hsr.p_pi.tpc.gripper import Lego_Gripper
from tpc.perception.cluster_registration import run_connected_components, display_grasps, \
    has_multiple_objects, grasps_within_pile, view_hsv, get_hsv_hist, hsv_classify
from tpc.perception.singulation import find_singulation, display_singulation
from tpc.perception.crop import crop_img
from tpc.data_manager import DataManager
from tpc.manipulation.primitives import GraspManipulator
from perception import ColorImage, BinaryImage
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import tpc.config.config_tpc as cfg

from il_ros_hsr.core.rgbd_to_map import RGBD2Map

class LegoDemo():

    def __init__(self):
        '''
        Class to run HSR lego task

        '''

        self.robot = hsrb_interface.Robot()
        self.rgbd_map = RGBD2Map()

        self.omni_base = self.robot.get('omni_base')
        self.whole_body = self.robot.get('whole_body')


        self.side = 'BOTTOM'

        self.cam = RGBD()
        self.com = COM()

        if not DEBUG:
            self.com.go_to_initial_state(self.whole_body)

            self.tt = TableTop()
            self.tt.find_table(self.robot)

        self.grasp_count = 0

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()

        self.gp = GraspPlanner()
        self.gripper = Crane_Gripper(self.gp, self.cam, self.com.Options, self.robot.get('gripper'))

        self.gm = GraspManipulator(self.gp, self.gripper, self.whole_body, self.omni_base, self.tt)

        print "after thread"

    def find_mean_depth(self,d_img):
        '''
        Evaluates the current policy and then executes the motion
        specified in the the common class
        '''

        indx = np.nonzero(d_img)

        mean = np.mean(d_img[indx])

        return

    def get_success(self, action):
        print("Was " + action + " successful? (y or n)")
        succ = ""
        succ = raw_input()
        while not (succ == "y" or succ == "n"):
            print("Enter only y or n to indicate success of " + action)
            succ = raw_input()
        return succ

    def lego_demo(self):

        self.dm = DataManager(cfg.COLLECT_DATA)
        self.get_new_grasp = True

        if not DEBUG:
            self.gm.position_head()

        time.sleep(1) #making sure the robot is finished moving
        i = 4
        while True:
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            cv2.imwrite("debug_imgs/hsv_testing/img" + str(i) + ".png", c_img)
            i += 1
            IPython.embed()
        while not (c_img == None or d_img == None):
            print "\n new iteration"

            self.dm.clear_traj()
            self.dm.update_traj("c_img", c_img)
            self.dm.update_traj("d_img", d_img)
            cv2.imwrite("debug_imgs/c_img.png", c_img)

            main_mask = crop_img(c_img)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)
            self.dm.update_traj("crop", workspace_img.data)

            a = time.time()
            center_masses, directions, masks = run_connected_components(workspace_img,
                cfg.DIST_TOL, cfg.COLOR_TOL, cfg.SIZE_TOL, viz=False)
            self.dm.update_traj("connected_components_time", time.time() - a)

            print "num masses", len(center_masses)
            if len(center_masses) == 0:
                print("cleared workspace")
                break

            has_multiple = [has_multiple_objects(col_img.mask_binary(m), alg="hsv") for m in masks]
            print "has multiple objects?:", has_multiple

            a = time.time()
            grasps = []
            grasp_cms_dirs_masks = []
            for i in range(len(center_masses)):
                curr_cms, curr_dirs = [center_masses[i]], [directions[i]]
                if has_multiple[i]:
                    curr_cms, curr_dirs = grasps_within_pile(col_img.mask_binary(masks[i]))
                for j in range(len(curr_cms)):
                    pose,rot = self.gm.compute_grasp(curr_cms[j], curr_dirs[j], d_img)
                    #should recompute masks for grasps within piles
                    class_num = hsv_classify(col_img.mask_binary(masks[i]))
                    grasp_cms_dirs_masks_classes.append((curr_cms[j], curr_dirs[j], masks[i], class_num))
                    grasps.append(self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data))
            self.dm.update_traj("compute_grasps_time", time.time() - a)

            #impose ordering on grasps (closest/highest y first)
            all_grasp_info = zip(grasp_cms_dirs_masks_classes, grasps)
            all_grasp_info.sort(key=lambda x:-x[0][0][0])
            grasp_cms_dirs_masks_classes, grasps = [list(t) for t in zip(*all_grasp_info)]

            if len(grasps) > 0:
                self.dm.update_traj("action", "grasp")
                self.dm.update_traj("info", [g for g in grasp_cms_dirs_masks_classes])

                print "running grasps"
                display_grasps(workspace_img, [g[0] for g in grasp_cms_dirs_masks_classes],
                    [g[1] for g in grasp_cms_dirs_masks], name = "grasps")
                IPython.embed()

                self.dm.update_traj("success", "n")
                #write here in case of failure causing crash
                self.dm.append_traj()

                a = time.time()
                for i, grasp in enumerate(grasps):
                    print "grasping", grasp
                    self.gm.execute_grasp(grasp, class_num)
                self.dm.update_traj("execute_time", time.time() - a)
                self.dm.update_traj("success", self.get_success("grasps"))
                self.dm.overwrite_traj()
            else:
                self.dm.update_traj("action", "singulate")
                print("singulating")
                a = time.time()
                #singulate smallest pile
                masks.sort(key=lambda m:len(m.nonzero_pixels()))
                waypoints, rot, free_pix = find_singulation(col_img, main_mask, masks[0],
                    masks[1:], alg="border")
                dm.update_traj("compute_singulate_time", time.time() - a)

                display_singulation(waypoints, workspace_img, free_pix,
                    name = "singulate")
                IPython.embed()

                self.dm.update_traj("info", (waypoints, rot, free_pix))
                self.dm.update_traj("success", "n")
                #write here in case of failure causing crash
                self.dm.append_traj()

                a = time.time()
                self.gm.singulate(waypoints, rot, c_img, d_img, expand=True)
                self.dm.update_traj("execute_time", time.time() - a)

                self.dm.update_traj("success", self.get_success("singulation"))
                self.dm.overwrite_traj()

            #also save timing data
            self.whole_body.move_to_go()
            self.gm.position_head()

            time.sleep(1) #making sure the robot is finished moving
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            self.dm.update_traj("c_img_result", c_img)
            self.dm.update_traj("d_img_result", d_img)
            self.dm.overwrite_traj()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False

    task = LegoDemo()
    task.lego_demo()
