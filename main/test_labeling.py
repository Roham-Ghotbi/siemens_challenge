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
from tpc.perception.connected_components import get_cluster_info

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


        self.web = Web_Labeler()
        print "after thread"

    def bbox_to_fg(self, bbox, c_img, col_img):
        obj_mask = crop_img(c_img, bycoords = [bbox[1], bbox[3], bbox[0], bbox[2]])
        obj_workspace_img = col_img.mask_binary(obj_mask)
        fg = obj_workspace_img.foreground_mask(cfg.COLOR_TOL, ignore_black=True)
        return fg, obj_workspace_img

    def label_demo(self):

        self.gm.position_head()
        time.sleep(3) #making sure the robot is finished moving

        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()

        while not (c_img is None or d_img is None):
            print "\n new iteration"
            cv2.imwrite("debug_imgs/c_img.png", c_img)
            main_mask = crop_img(c_img, simple=True)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)

            path = "/home/autolab/Workspaces/michael_working/siemens_challenge/debug_imgs/web.png"
            cv2.imwrite(path, workspace_img.data)
            # labels = self.web.label_image(path)

            # obj = labels[0]
            # bbox = obj['box']
            # class_label = obj['class']

            #xmin, ymin, xmax, ymax
            bbox = np.array([350, 200, 450, 400])
            fg, obj_w = self.bbox_to_fg(bbox, c_img, col_img)

            groups = get_cluster_info(fg)
            display_grasps(workspace_img, groups)
            IPython.embed()

            # bbox_cm = [(bbox[i] + bbox[i+2])/2.0 for i in [1, 0]] #y then x average
            # bbox_dir = np.array([1, 0])
            # display_grasps([(bbox_cm, bbox_dir)], workspace_img, as_tuple = True)

            group = groups[0]
            pose,rot = self.gm.compute_grasp(group.cm, group.dir, d_img)
            grasp_pose = self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data)
            self.gm.execute_grasp(grasp_pose)
            
            #reset to start
            self.whole_body.move_to_go()
            self.gm.move_to_home()
            self.gm.position_head()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False

    task = LabelDemo()
    task.label_demo()
