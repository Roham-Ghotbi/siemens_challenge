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
from il_ros_hsr.core.crane_gripper import Crane_Gripper

from il_ros_hsr.p_pi.bed_making.com import Bed_COM as COM
import sys

from tpc.python_labeler import Python_Labeler

from il_ros_hsr.p_pi.tpc.gripper import Lego_Gripper
from tpc.perception.cluster_registration import run_connected_components, visualize, has_multiple_objects
from tpc.perception.singulation import find_singulation, display_singulation
from tpc.perception.crop import crop_img
from tpc.perception.groups import Group
from perception import ColorImage, BinaryImage
from il_ros_hsr.p_pi.bed_making.table_top import TableTop

import il_ros_hsr.p_pi.bed_making.config_bed as cfg


from il_ros_hsr.core.rgbd_to_map import RGBD2Map

SINGULATE = True

DIST_TOL = 5 #number of pixels apart to be singulated

COLOR_TOL = 40 #background range for thresholding the image

SIZE_TOL = 300 #number of pixels necssary for a cluster


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

    def bbox_to_mask(self, bbox, c_img):
        loX, loY, hiX, hiY = bbox 
        bin_img = np.zeros(c_img.shape[0:2])
        #don't include workspace points
        for x in range(loX, hiX):
            for y in range(loY, hiY):
                r, g, b = c_img[y][x]
                if r < 240 or g < 240 or b < 240:
                    bin_img[y][x] = 255

        return BinaryImage(bin_img.astype(np.uint8))
    def bbox_to_grasp(self, bbox, c_img, d_img):
        '''
        Computes center of mass and direction of grasp using bbox
        '''
        loX, loY, hiX, hiY = bbox 

        #don't include workspace points
        dpoints = []
        for x in range(loX, hiX):
            for y in range(loY, hiY):
                r, g, b = c_img[y][x]
                if r < 240 or g < 240 or b < 240:
                    dpoints.append([y, x])

        g = Group(1, points=dpoints)
        direction = g.orientation()
        center_mass = g.center_mass()

        #use x from color image
        x_center_points = [d for d in dpoints if abs(d[1] - center_mass[1]) < 4]

        #compute y using depth image
        dvals = [d_img[d[0], d[1]] for d in x_center_points]
        depth_vals = list(np.copy(dvals))
        depth_vals.sort()
        #use median to ignore depth noise
        middle_depth = depth_vals[len(depth_vals)/2]
        closest_ind = (np.abs(dvals - middle_depth)).argmin()
        closest_point = x_center_points[closest_ind]

        return closest_point, direction

    def lego_demo(self):

        self.rollout_data = []
        self.get_new_grasp = True

        if not DEBUG:
            self.position_head()
        while True:
            time.sleep(1) #making sure the robot is finished moving
            c_img = self.cam.read_color_data()
            d_img = self.cam.read_depth_data()
            cv2.imwrite("debug_imgs/c_img.png", c_img)
            print "\n new iteration"

            if(not c_img == None and not d_img == None):
                #label image
                data = self.wl.label_image(c_img)
                grasp_boxes = []
                suction_boxes = []
                singulate_boxes = []
                for i in range(data['num_labels']):
                    bbox = data['objects'][i]['box']
                    classnum = data['objects'][i]['class']
                    classname = ['grasp', 'singulate', 'suction'][classnum]
                    if classname == "grasp":
                        grasp_boxes.append(bbox)
                    elif classname == "suction":
                        suction_boxes.append(bbox)
                    elif classname == "singulate":
                        singulate_boxes.append(bbox)

                main_mask = crop_img(c_img)
                col_img = ColorImage(c_img)
                workspace_img = col_img.mask_binary(main_mask)

                grasps = []
                viz_info = []
                for i in range(len(grasp_boxes)):
                    bbox = grasp_boxes[i]
                    center_mass, direction = self.bbox_to_grasp(bbox, c_img, d_img)

                    viz_info.append([center_mass, direction])
                    pose,rot = self.compute_grasp(center_mass,direction,d_img)
                    grasps.append(self.gripper.get_grasp_pose(pose[0],pose[1],pose[2],rot,c_img=workspace_img.data))


                suctions = []
                for i in range(len(suction_boxes)):
                    suctions.append("compute_suction?")

                if len(grasps) > 0 or len(suctions) > 0:
                    cv2.imwrite("grasps.png", visualize(workspace_img, [v[0] for v in viz_info], [v[1] for v in viz_info]))
                    print "running grasps"
                    IPython.embed()
                    for grasp in grasps:
                        print "grasping", grasp
                        self.execute_grasp(grasp)
                    print "running suctions"
                    for suction in suctions:
                        print "suctioning", suction
                        #execute suction
                elif len(singulate_boxes) > 0:
                    print("singulating")
                    bbox = singulate_boxes[0]
                    obj_mask = self.bbox_to_mask(bbox, c_img)
                    start, end = find_singulation(col_img, main_mask, obj_mask)
                    display_singulation(start, end, workspace_img)
                    IPython.embed()
                    self.singulate(start, end, c_img, d_img)
                else:
                    print("cleared workspace")
                    break
                IPython.embed()
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
        #convert from image to world (flip x)
        dx = direction[1]
        dy = direction[0]
        dx *= -1
        #standardize to 1st/2nd quadrants
        if dy < 0:
            dx *= -1
            dy *= -1
        rot = np.arctan2(dy, dx)
        #convert to robot view (counterclockwise)
        rot = 3.14 - rot
        # IPython.embed()
        # if direction: 
        #     rot = 0.0
        # else: 
        #     rot = 1.57

        x = int(c_m[1])
        y = int(c_m[0])

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
        start_pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img.data)
        
        y, x = end
        z_box = d_img[y-20:y+20, x-20:x+20]
        z = self.gp.find_mean_depth(z_box)
        end_pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img.data)
        
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