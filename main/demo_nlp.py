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
from math import pi
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

from il_ros_hsr.core.grasp_planner import GraspPlanner
from il_ros_hsr.core.crane_gripper import Crane_Gripper

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys



from il_ros_hsr.p_pi.tpc.gripper import Lego_Gripper
from tpc.perception.cluster_registration import run_connected_components, draw, count_size_blobs
from tpc.perception.singulation import find_singulation, display_singulation
from perception import ColorImage, BinaryImage
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import il_ros_hsr.p_pi.bed_making.config_bed as cfg


from il_ros_hsr.core.rgbd_to_map import RGBD2Map

SINGULATE = True

#number of pixels apart to be singulated
DIST_TOL = 5
#background range for thresholding the image
COLOR_TOL = 40

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
        
    
        self.grasp_count = 0
      

        self.br = tf.TransformBroadcaster()
        self.tl = TransformListener()



        self.gp = GraspPlanner()

        self.gripper = Crane_Gripper(self.gp,self.cam,self.com.Options,self.robot.get('gripper'))

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
            self.position_head()
        b = time.time()
        while True:

            

            time.sleep(1) #making sure the robot is finished moving

            a = time.time()
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            cv2.imwrite("debug_imgs/c_img.png", c_img)
            print "time to get images", time.time() - a
            print "\n new iteration"
            if(not c_img == None and not d_img == None):
  
                c_ms, dirs, _ = run_connected_components(c_img)
                img = draw(c_img,c_ms,dirs)

               
                # # IPython.embed()
                for c_m, direction in zip(c_ms, dirs):
                    pose,rot = self.compute_grasp(c_m,direction,d_img)
                    rot -= pi / 2.0
                    print "pose, rot:", pose, rot
               
                ####DETERMINE WHAT OBJECT TO GRASP


                grasp_name = self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=c_img)
                
                
                self.execute_grasp(grasp_name)
             
                
                self.whole_body.move_to_go()
                self.position_head()




 
    

    def execute_grasp(self,grasp_name):
        self.gripper.open_gripper()

        self.whole_body.end_effector_frame = 'hand_palm_link'
        
        self.whole_body.move_end_effector_pose(geometry.pose(),grasp_name)

        self.gripper.close_gripper()
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1),grasp_name)        
        
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1),'head_down')
        
    
        self.gripper.open_gripper()


    def compute_grasp(self, c_m, direction, d_img):

        if direction: 
            rot = 0.0
        else: 
            rot = 1.57

        x = c_m[1]
        y = c_m[0]

        z_box = d_img[y-20:y+20,x-20:x+20]

        z = self.gp.find_mean_depth(z_box)

        return [x,y,z],rot

    def singulate(self, start, end, c_img, d_img):
        # [355.9527559055119, 123.53543307086613, 977.26812500000005] 0.0
        rot = np.pi/2 + np.arctan2(end[0] - start[0], end[1] - start[1])

        self.gripper.close_gripper()
        # self.go_to_point(start, rot, c_img, d_img)
        # self.go_to_point(end, rot, c_img, d_img)

        y, x = start
        z_box = d_img[y-20:y+20, x-20:x+20]
        z = self.gp.find_mean_depth(z_box)
        # above_start_pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img)
        start_pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img)
        
        y, x = end
        z_box = d_img[y-20:y+20, x-20:x+20]
        z = self.gp.find_mean_depth(z_box)
        end_pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img)
        
        # raw_input("Click enter to move to " + above_start_pose_name)
        # self.whole_body.move_end_effector_pose(geometry.pose(), start_pose_name)
        # raw_input("Click enter to singulate from " + start_pose_name)
        print "singulating", start_pose_name
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.05), start_pose_name)
        self.whole_body.move_end_effector_pose(geometry.pose(z=-.01), start_pose_name)
        # raw_input("Click enter to singulate to " + end_pose_name)
        print "singulating", end_pose_name
        self.whole_body.move_end_effector_pose(geometry.pose(z=-.01), end_pose_name)

        self.gripper.open_gripper()
        
        
    def go_to_point(self, point, rot, c_img, d_img):
        y, x = point
        z_box = d_img[y-20:y+20, x-20:x+20]
        z = self.gp.find_mean_depth(z_box)
        print "singulation pose:", x,y,z
        pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img)
        raw_input("Click enter to move to " + pose_name)
        self.whole_body.move_end_effector_pose(geometry.pose(), pose_name)

    
    def position_head(self):

        self.tt.move_to_pose(self.omni_base,'lower_start')
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})

        




if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False
    
    cp = BedMaker()

    # cp.bed_make()
    cp.lego_demo()