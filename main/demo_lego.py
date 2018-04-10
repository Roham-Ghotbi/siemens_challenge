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

class LegoDemo():

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

        self.gm = GraspManipulator(self.gp, self.gripper, self.suction, self.whole_body, self.omni_base)

        self.collision_world = hsrb_interface.collision_world.CollisionWorld("global_collision_world")
        self.collision_world.remove_all()
        self.collision_world.add_box(x=.8,y=.9,z=0.5,pose=geometry.pose(y=1.4,z=0.15),frame_id='map')

        print "after thread"

    def find_mean_depth(self,d_img):
        """
        Evaluates the current policy and then executes the motion
        specified in the the common class
        """

        indx = np.nonzero(d_img)

        mean = np.mean(d_img[indx])

        return

    def get_success(self, action):
        """
        Parameters
        ----------
        action : str
            action to query success of
        Returns
        -------
        :str: y or n
        """
        if cfg.COLLECT_DATA and cfg.QUERY:
            print("Was " + action + " successful? (y or n)")
            succ = raw_input()
            while not (succ == "y" or succ == "n"):
                print("Enter only y or n to indicate success of " + action)
                succ = raw_input()
            return succ
        else:
            return "no data queried"

    def get_int(self):
        """
        Returns
        -------
        int
        """
        if cfg.COLLECT_DATA and cfg.QUERY:
            print("How many legos are on the table?")
            inp = raw_input()
            while not inp.isdigit():
                print("Enter an integer.")
                inp = raw_input()
            return inp
        else:
            return "no data queried"

    def get_str(self):
        """
        Returns
        -------
        :str: arbitary value
        """
        if cfg.COLLECT_DATA and cfg.QUERY:
            print("Any notes? (pushed legos off table, ran into table, etc.) (no if none)")
            inp = raw_input()
            return inp
        else:
            return "no data queried"

    def run_singulation(self, col_img, main_mask, d_img, to_singulate):
        """
        Parameters
        ----------
        col_img : `ColorImage`
        main_mask : `BinaryImage`
        d_img : 'DepthImage'
        to_singulate : list of `Group`
        """
        print("SINGULATING")

        self.dm.update_traj("action", "singulate")

        a = time.time()
        singulator = Singulation(col_img, main_mask, [g.mask for g in to_singulate])
        waypoints, rot, free_pix = singulator.get_singulation()
        self.dm.update_traj("compute_singulate_time", time.time() - a)

        singulator.display_singulation()

        self.dm.update_traj("singulate_info", (waypoints, rot, free_pix))
        self.dm.update_traj("singulate_successes", "crashed")
        self.dm.update_traj("execute_singulate_time", "crashed")
        self.dm.overwrite_traj()

        a = time.time()
        self.gm.singulate(waypoints, rot, col_img.data, d_img, expand=True)
        self.dm.update_traj("execute_singulate_time", time.time() - a)
        self.dm.update_traj("singulate_success", self.get_success("singulation"))
        self.dm.overwrite_traj()


    def run_grasps(self, workspace_img, to_grasp):
        """
        Parameters
        ----------
        workspace_img : `ColorImage`
        to_grasp : list of tuple of:
            (`Group`, grasp_pose, suction_pose, class_num)
        """

        print("GRASPING")

        #impose ordering on grasps (by closest/highest y first)
        to_grasp.sort(key=lambda g:-1 * g[0].cm[0])

        self.dm.update_traj("action", "grasp")

        self.dm.update_traj("all_grasps_info", [(c[0].cm, c[0].dir, c[0].mask.data, c[3]) for c in to_grasp])

        if not cfg.CHAIN_GRASPS:
            to_grasp = to_grasp[0:1]

        self.dm.update_traj("run_grasps_info", [(c[0].cm, c[0].dir, c[0].mask.data, c[3]) for c in to_grasp])

        display_grasps(workspace_img, [g[0] for g in to_grasp])

        successes = ["not attempted" for i in range(len(to_grasp))]
        correct_colors = ["not attempted" for i in range(len(to_grasp))]
        times = ["not attempted" for i in range(len(to_grasp))]


        for i in range(len(to_grasp)):
            print "grasping", to_grasp[i][1]

            successes[i] = "crashed"
            correct_colors[i] = "crashed"
            times[i] = "crashed"
            self.dm.update_traj("grasp_successes", successes)
            self.dm.update_traj("grasp_colors", correct_colors)
            self.dm.update_traj("execute_grasp_times", times)
            self.dm.overwrite_traj()

            a = time.time()
            self.gm.execute_grasp(to_grasp[i][1], class_num=to_grasp[i][3])
            # self.gm.execute_suction(to_grasp[i][2], to_grasp[i][3])

            times[i] = time.time() - a
            successes[i] = self.get_success("grasp")
            correct_colors[i] = self.get_success("correct color")

            self.dm.update_traj("grasp_successes", successes)
            self.dm.update_traj("grasp_colors", correct_colors)
            self.dm.update_traj("execute_grasp_times", times)
            self.dm.overwrite_traj()

    def clusters_to_actions(self, groups, col_img, d_img, workspace_img):
        """
        Parameters
        ----------
        groups : list of `Group`
        col_img : `ColorImage`
        d_img : `DepthImage`
        workspace_img : `ColorImage`

        Returns
        -------
        list of tuples of form:
            (`Group`, grasp_pose, suction_pose, class_num)
        list of `Group`
        """
        find_grasps_times = []
        compute_grasps_times = []
        find_hsv_times = []

        to_singulate = []
        to_grasp = []
        for group in groups:
            a = time.time()
            inner_groups = grasps_within_pile(col_img.mask_binary(group.mask))
            find_grasps_times.append(time.time() - a)

            if len(inner_groups) == 0:
                to_singulate.append(group)
            else:
                for in_group in inner_groups:
                    a = time.time()
                    pose,rot = self.gm.compute_grasp(in_group.cm, in_group.dir, d_img)
                    grasp_pose = self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data)
                    suction_pose = self.suction.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data)
                    compute_grasps_times.append(time.time() - a)

                    a = time.time()
                    class_num = hsv_classify(col_img.mask_binary(in_group.mask))
                    find_hsv_times.append(time.time() - a)

                    to_grasp.append((in_group, grasp_pose, suction_pose, class_num))
        self.dm.update_traj("compute_grasps_times", compute_grasps_times)
        self.dm.update_traj("find_grasps_times", find_grasps_times)
        self.dm.update_traj("find_hsv_times", find_hsv_times)

        return to_grasp, to_singulate

    def lego_demo(self):
        if not cfg.COLLECT_DATA:
            print("WARNING: NO DATA IS BEING COLLECTED")
            print("TO COLLECT DATA, CHANGE COLLECT_DATA IN config_tpc")

        self.dm = DataManager()
        self.get_new_grasp = True

        # DEBUG = False
        # if not DEBUG:
        #     self.gm.position_head()

        time.sleep(3) #making sure the robot is finished moving
        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()

        while not (c_img is None or d_img is None):
            print "\n new iteration"

            main_mask = crop_img(c_img, use_preset=True, arc=False)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)

            self.dm.clear_traj()
            self.dm.update_traj("c_img", c_img)
            self.dm.update_traj("d_img", d_img)
            self.dm.update_traj("stop_condition", "crash")
            cv2.imwrite("debug_imgs/c_img.png", c_img)
            self.dm.update_traj("crop", workspace_img.data)

            a = time.time()
            groups = run_connected_components(workspace_img, viz=True)
            self.dm.update_traj("connected_components_time", time.time() - a)

            self.dm.update_traj("num_legos", self.get_int())

            self.dm.append_traj()

            print "num masses", len(groups)
            if len(groups) == 0:
                print("cleared workspace")
                self.dm.update_traj("stop_condition", self.get_success("clearing table"))
                self.dm.overwrite_traj()
                time.sleep(5)
                break

            to_grasp, to_singulate = self.clusters_to_actions(groups, col_img, d_img, workspace_img)

            self.whole_body.collision_world = self.collision_world
            if len(to_grasp) > 0:
                self.run_grasps(workspace_img, to_grasp)
            else:
                self.run_singulation(col_img, main_mask, d_img, to_singulate)

            self.dm.update_traj("notes", self.get_str())

            #reset to start
            self.whole_body.move_to_go()
            self.gm.move_to_home()
            self.gm.position_head()

            time.sleep(8) #making sure the robot is finished moving
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            self.dm.update_traj("stop_condition", "none")
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
