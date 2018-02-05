import sys
import IPython
import tpc.config.config_tpc as cfg
from hsrb_interface import geometry
import numpy as np
from tpc.perception.cluster_registration import class_num_to_name

class GraspManipulator():
    def __init__(self, gp, gripper, whole_body, omni_base, tt):
        self.gp = gp
        self.gripper = gripper
        self.whole_body = whole_body
        self.omni_base = omni_base
        self.tt = tt

    def get_z(self, point, d_img):
        y = int(point[0])
        x = int(point[1])
        z_box = d_img[y-cfg.ZRANGE:y+cfg.ZRANGE, x-cfg.ZRANGE:x+cfg.ZRANGE]
        z = self.gp.find_mean_depth(z_box)
        return z

    def get_pose(self, point, rot, c_img, d_img):
        z = self.get_z(point, d_img)
        y, x = point
        pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img.data)
        return pose_name

    def singulate(self, waypoints, rot, c_img, d_img, expand=False):
        self.gripper.close_gripper()

        pose_names = [self.get_pose(waypoint, rot, c_img, d_img) for waypoint in waypoints]

        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.05), pose_names[0])

        for pose_name in pose_names:
            print "singulating", pose_name
            self.whole_body.move_end_effector_pose(geometry.pose(z=-0.01), pose_name)

        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.05), pose_names[-1])

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
        rot = np.pi - rot

        x = int(c_m[1])
        y = int(c_m[0])

        z_box = d_img[y-cfg.ZRANGE:y+cfg.ZRANGE,x-cfg.ZRANGE:x+cfg.ZRANGE]

        z = self.gp.find_mean_depth(z_box)

        return [x,y,z],rot

    def execute_grasp(self, grasp_name, class_num):
        if class_num not in range(8):
            raise ValueError("currently ony supports classes 0 to 7")
        self.gripper.half_gripper()

        self.whole_body.end_effector_frame = 'hand_palm_link'

        #before lowering gripper, go directly above grasp position
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1),grasp_name)
        self.whole_body.move_end_effector_pose(geometry.pose(),grasp_name)
        self.gripper.close_gripper()
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1),grasp_name)

        color_name = class_num_to_name(class_num)
        print("Identified lego: " + color_name)

        lego_class_num = cfg.HUES_TO_BINS.index(color_name)
        above_pose = "lego" + str(lego_class_num) + "above"
        below_pose = "lego" + str(lego_class_num) + "below"

        self.tt.move_to_pose(self.omni_base,'lego_prep')

        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1), above_pose)
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1), below_pose)
        self.gripper.open_gripper()
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1), above_pose)

    def go_to_point(self, point, rot, c_img, d_img):
        y, x = point
        z_box = d_img[y-cfg.ZRANGE:y+cfg.ZRANGE, x-cfg.ZRANGE:x+cfg.ZRANGE]
        z = self.gp.find_mean_depth(z_box)
        print "singulation pose:", x,y,z
        pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img)
        raw_input("Click enter to move to " + pose_name)
        self.whole_body.move_end_effector_pose(geometry.pose(), pose_name)

    def position_head(self):
        self.tt.move_to_pose(self.omni_base,'lower_start')
        self.whole_body.move_to_joint_positions({'head_tilt_joint':-0.8})

    def move_to_home(self):
        self.tt.move_to_pose(self.omni_base,'lower_mid')
        sys.exit()
