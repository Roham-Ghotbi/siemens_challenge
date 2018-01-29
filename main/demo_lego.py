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
        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()

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

            #compute clusters (can have 1 or multiple legos)
            a = time.time()
            center_masses, directions, masks = run_connected_components(workspace_img,
                cfg.DIST_TOL, cfg.COLOR_TOL, cfg.SIZE_TOL, viz=False)
            cluster_info = zip(center_masses, directions, masks)
            self.dm.update_traj("connected_components_time", time.time() - a)

            print "num masses", len(center_masses)
            if len(center_masses) == 0:
                print("cleared workspace")
                break

            #for each cluster, compute grasps
            find_grasps_time = 0
            compute_grasps_time = 0

            to_singulate = []
            to_grasp = []
            for c_info in cluster_info:
                a = time.time()
                grasp_cms, grasp_dirs, grasp_masks = grasps_within_pile(col_img.mask_binary(c_info[2]))
                find_grasps_time += time.time() - a
                if len(grasp_cms) == 0:
                    to_singulate.append(c_info)
                else:
                    a = time.time()
                    for i in range(len(grasp_cms)):
                        pose,rot = self.gm.compute_grasp(grasp_cms[i], grasp_dirs[i], d_img)
                        grasp_pose = self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data)
                        class_num = hsv_classify(col_img.mask_binary(grasp_masks[i]))
                        to_grasp.append((grasp_cms[i], grasp_dirs[i], grasp_masks[i], grasp_pose, class_num))
                    compute_grasps_time += time.time() - a
            #impose ordering on grasps (by closest/highest y first)
            to_grasp.sort(key=lambda g:-1 * g[0][0])
            self.dm.update_traj("compute_grasps_time", compute_grasps_time)
            self.dm.update_traj("find_grasps_time", find_grasps_time)

            if len(to_grasp) > 0:
                self.dm.update_traj("action", "grasp")
                self.dm.update_traj("info", [(c[0], c[1], c[2].data, c[4]) for c in to_grasp])

                print "running grasps"
                display_grasps(workspace_img, [g[0] for g in to_grasp],
                    [g[1] for g in to_grasp])
                IPython.embed()

                successes = ["?" for i in range(len(to_grasp))]
                times = [0 for i in range(len(to_grasp))]

                self.dm.update_traj("success", successes)
                #write here in case of failure causing crash
                self.dm.append_traj()

                for i in range(len(to_grasp)):
                    print "grasping", to_grasp[i][3]
                    successes[i] = "n"
                    a = time.time()
                    self.gm.execute_grasp(to_grasp[i][3], to_grasp[i][4])
                    times[i] = time.time() - a
                    self.dm.update_traj("execute_time", times)
                    successes[i] = self.get_success("grasp")
                    self.dm.update_traj("success", successes)
                    #write here in case of failure causing crash
                    self.dm.overwrite_traj()
            else:
                self.dm.update_traj("action", "singulate")
                print("singulating")
                a = time.time()

                #singulate smallest pile
                masks = [c[2] for c in to_singulate]
                masks.sort(key=lambda m:len(m.nonzero_pixels()))
                waypoints, rot, free_pix = find_singulation(col_img, main_mask, masks[0],
                    masks[1:], alg="border")
                self.dm.update_traj("compute_singulate_time", time.time() - a)

                display_singulation(waypoints, workspace_img, free_pix,
                    name = "debug_imgs/singulate")
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
