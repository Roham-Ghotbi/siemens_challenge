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

from il_ros_hsr.core.grasp_planner import GraspPlanner

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys



from il_ros_hsr.p_pi.tpc.gripper import Lego_Gripper
from tpc.perception.TPC_singulate import run_connected_components, draw
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import il_ros_hsr.p_pi.bed_making.config_bed as cfg


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



        # if cfg.USE_WEB_INTERFACE:
        #     self.wl = Web_Labeler()
        # else:
        #     self.wl = Python_Labeler(cam = self.cam)


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


    def bed_make(self):

        self.rollout_data = []
        self.get_new_grasp = True

        self.position_head()
        while True:

            

            time.sleep(2)

            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()

            
            if(not c_img == None and not d_img == None):
  
                c_m, dirs = run_connected_components(c_img)
                draw(c_img,c_m,dirs)
                
                
                c_img = self.cam.read_color_data()
                d_img = self.cam.read_depth_data()
               

                grasp_name = self.gripper.get_grasp_pose(c_m[0],c_m[1],z,rot,c_img=c_img)
        
               
                
                pick_found,bed_pick = self.check_card_found()

                self.gripper.execute_grasp(bed_pick,self.whole_body,'head_down')
                
                self.grasp_count += 1
                self.whole_body.move_to_go()
                self.tt.move_to_pose(self.omni_base,'lower_start')
                time.sleep(1)
                self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})
 
    

    def execute_grasp(self,grasp_name):

        
        whole_body.end_effector_frame = 'hand_palm_link'
        
        whole_body.move_end_effector_pose(geometry.pose(),cards[0])


        self.gripper.open_gripper()
        
        whole_body.move_end_effector_pose(geometry.pose(z=-0.1),'head_down')
        
    
        self.com.close_gripper()
    
    def position_head(self):

        self.tt.move_to_pose(self.omni_base,'lower_start')
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})

        




if __name__ == "__main__":
   
    
    cp = BedMaker()

    cp.bed_make()

