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

from tpc.python_labeler import Python_Labeler

from il_ros_hsr.p_pi.tpc.gripper import Lego_Gripper
from tpc.perception.cluster_registration import run_connected_components, visualize, has_multiple_objects
from tpc.perception.singulation import find_singulation, display_singulation
from tpc.perception.crop import crop_img
from tpc.perception.bbox import bbox_to_mask, bbox_to_grasp
from tpc.manipulation.primitives import GraspManipulator
from perception import ColorImage, BinaryImage
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import tpc.config.config_tpc as cfg

from fast_grasp_detect.core.data_saver import data_saver

from il_ros_hsr.core.rgbd_to_map import RGBD2Map

class BedMaker():

    def __init__(self):
        '''
        Initialization class for a Policy

        Parameters
        ----------
        yumi : An instianted yumi robot
        com : The common class for the robot
        cam : An open bincam class

        debug : bool

            A bool to indicate whether or not to display a training set point for
            debuging.

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

        self.wl = Python_Labeler(cam = self.cam)

        self.grasp_count = 0


        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()

        self.ds = data_saver('tpc_rollouts')

        self.gp = GraspPlanner()
        self.gripper = Crane_Gripper(self.gp, cam, options, robot.get('gripper'))

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

    def lego_demo(self):

        self.rollout_data = []
        self.get_new_grasp = True

        if not DEBUG:
            self.gm.position_head()

        while True:
            time.sleep(1) #making sure the robot is finished moving
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            cv2.imwrite("debug_imgs/c_img.png", c_img)
            print "\n new iteration"

            if(not c_img == None and not d_img == None):
                #label image
                data = self.wl.label_image(c_img)
                c_img = self.cam.read_color_data()


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
                        self.ds.save_rollout()
                        self.gm.move_to_home()
                self.ds.append_data_to_rollout(c_img,data)
                main_mask = crop_img(c_img)
                col_img = ColorImage(c_img)
                workspace_img = col_img.mask_binary(main_mask)

                grasps = []
                viz_info = []
                for i in range(len(grasp_boxes)):
                    bbox = grasp_boxes[i]
                    center_mass, direction = bbox_to_grasp(bbox, c_img, d_img)

                    viz_info.append([center_mass, direction])
                    pose,rot = self.gm.compute_grasp(center_mass,direction,d_img)
                    grasps.append(self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data))


                suctions = []
                for i in range(len(suction_boxes)):
                    suctions.append("compute_suction?")

                if len(grasps) > 0 or len(suctions) > 0:
                    cv2.imwrite("grasps.png", visualize(workspace_img, [v[0] for v in viz_info], [v[1] for v in viz_info]))
                    print "running grasps"

                    for grasp in grasps:
                        print "grasping", grasp
                        self.gm.execute_grasp(grasp)
                    print "running suctions"
                    for suction in suctions:
                        print "suctioning", suction
                        #execute suction
                elif len(singulate_boxes) > 0:
                    print("singulating")
                    bbox = singulate_boxes[0]
                    obj_mask = bbox_to_mask(bbox, c_img)
                    start, end = find_singulation(col_img, main_mask, obj_mask)
                    display_singulation(start, end, workspace_img)

                    self.gm.singulate(start, end, c_img, d_img)
                else:
                    print("cleared workspace")

                self.whole_body.move_to_go()
                self.gm.position_head()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False

    cp = BedMaker()
    cp.lego_demo()
