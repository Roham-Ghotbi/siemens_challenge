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
sys.path.append('/home/ron/siemens_sim/hsr_web')
from web_labeler import Web_Labeler
from tpc.perception.connected_components import get_cluster_info

import tpc.config.config_tpc as cfg
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

"""
This class is for use with the robot
Pipeline it tests is labeling a single object using the web interface,
then the robot grasping the object and placing it in the correct bin by AR marker
"""
POSE0 = (0.121594, 0.315173, 0.014000)
POSE1 = (0.004509, 0.500258, 0.024000)
POSE2 = (-0.107591, 0.515876, 0.024000)
POSE3 = (-0.191729, 0.358736, 0.024000)

POSES = [POSE0, POSE1, POSE2, POSE3]

def find_pose(pos):
    dist = 1000 
    grasp_pose = None
    for pose in POSES:
        if ((pos[0]-pose[0])**2 + (pos[1]-pose[1])**2 + (pos[2]-pose[2])**2 < dist):
            grasp_pose = pose
    return grasp_pose


class LabelDemo():

    def __init__(self):

        self.robot = hsrb_interface.Robot()
        self.br = tf.TransformBroadcaster()
        self.rgbd_map = RGBD2Map(self.br)
        # IPython.embed()

        self.omni_base = self.robot.get('omni_base')
        self.whole_body = self.robot.get('whole_body')
        self.side = 'BOTTOM'

        self.cam = RGBD()
        self.com = COM()

        self.com.go_to_initial_state(self.whole_body)
        self.grasp_count = 0
        self.tl = TransformListener()

        self.gp = GraspPlanner()
        self.gripper = Crane_Gripper(self.gp, self.cam, self.com.Options, self.robot.get('gripper'))
        self.suction = Suction_Gripper(self.gp, self.cam, self.com.Options, self.robot.get('suction'))
        self.gm = GraspManipulator(self.gp, self.gripper, self.suction, self.whole_body, self.omni_base, self.tl)
        self.web = Web_Labeler()

        thread.start_new_thread(self.broadcast_temp_bin,())

        time.sleep(3)
        print "after thread"

    def broadcast_temp_bin(self):
        while True:
            self.br.sendTransform((1, 1, 0.6),
                    tf.transformations.quaternion_from_euler(ai=0.0,aj=1.57,ak=0.0),
                    rospy.Time.now(),
                    'temp_bin',
                    # 'head_rgbd_sensor_link')
                    'map')
            rospy.sleep(1)

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

    def find_isolated_objects(self, bboxes, c_img):
        valid_bboxes = []
        for curr_ind in range(len(bboxes)):
            curr_bbox = bboxes[curr_ind]
            overlap = False  
            for test_ind in range(curr_ind + 1, len(bboxes)):
                test_bbox = bboxes[test_ind]
                if self.test_bbox_overlap(curr_bbox, test_bbox):
                    overlap = True 
                    break 
            if not overlap:
                valid_bboxes.append(curr_bbox)
        return valid_bboxes

    def label_demo(self):
        """ Main method which executes the stuff we're interested in.
        
        Should apply to both the physical and simulated HSRs. Call as `python
        main/test_labeling.py`.
        """
        self.gm.position_head()
        time.sleep(3) #making sure the robot is finished moving
        c_img = self.cam.read_color_data()
        d_img = self.cam.read_depth_data()

        path = "/home/ron/siemens_sim/siemens_challenge/debug_imgs/web.png"
        cv2.imwrite(path, c_img)
        time.sleep(2) #make sure new image is written before being read

        # print "\n new iteration"
        main_mask = crop_img(c_img, simple=True)
        col_img = ColorImage(c_img)
        workspace_img = col_img.mask_binary(main_mask)

        labels = self.web.label_image(path)

        obj = labels['objects'][0]
        bbox = obj['box']
        class_label = obj['class']

        #bbox has format [xmin, ymin, xmax, ymax]
        fg, obj_w = self.bbox_to_fg(bbox, c_img, col_img)
        cv2.imwrite("debug_imgs/test.png", obj_w.data)

        groups = get_cluster_info(fg)
        display_grasps(workspace_img, groups)

        group = groups[0]
        print(d_img)
        pose,rot = self.gm.compute_grasp(group.cm, group.dir, d_img)
        pose = find_pose(pose)
        if pose == None:
            print("unable to find corresponding item.")
            sys.exit()

        a = tf.transformations.quaternion_from_euler(ai=-2.355,aj=-3.14,ak=0.0)
        b = tf.transformations.quaternion_from_euler(ai=0.0,aj=0.0,ak=1.57)

        base_rot = tf.transformations.quaternion_multiply(a,b)

        print("now about to get grasp pose, w/pose: {}, rot: {}".format(pose, rot))
        thread.start_new_thread(self.gripper.loop_broadcast,(pose,base_rot,rot))
        time.sleep(5)
        print("now calling execute_grasp w/grasp_pose: {}".format(grasp_pose))
        # IPython.embed()
        self.gm.execute_grasp("grasp_0")
        self.whole_body.move_end_effector_pose(geometry.pose(),"temp_bin")
        self.gripper.open_gripper()



        #reset to start
        self.whole_body.move_to_go()
        # self.gm.move_to_home()
        self.gm.position_head()
        time.sleep(3)
        
if __name__ == "__main__":

    task = LabelDemo()
    task.label_demo()
