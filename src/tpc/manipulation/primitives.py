import sys
import IPython
import tpc.config.config_tpc as cfg

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

    def singulate(self, start, end, rot, c_img, d_img, expand=False):
        self.gripper.close_gripper()

        start_pose_name = self.get_pose(start, rot, c_img, d_img)

        mid = (start + end)/2.0
        mid_pose_name = self.get_pose(mid, rot, c_img, d_img)

        end_pose_name = self.get_pose(end, rot, c_img, d_img)

        # raw_input("Click enter to move to " + above_start_pose_name)
        # self.whole_body.move_end_effector_pose(geometry.pose(), start_pose_name)
        # raw_input("Click enter to singulate from " + start_pose_name)
        print "singulating", start_pose_name
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.05), start_pose_name)
        self.whole_body.move_end_effector_pose(geometry.pose(z=-.01), start_pose_name)
        # raw_input("Click enter to singulate to " + end_pose_name)
        print "singulating", mid_pose_name
        self.whole_body.move_end_effector_pose(geometry.pose(z=-.01), mid_pose_name)
        
        self.gripper.open_gripper()
        self.gripper.close_gripper()

        print "singulating", end_pose_name
        self.whole_body.move_end_effector_pose(geometry.pose(z=-.01), end_pose_name)

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
        rot = np.pi - rot


        x = int(c_m[1])
        y = int(c_m[0])

        z_box = d_img[y-cfg.ZRANGE:y+cfg.ZRANGE,x-cfg.ZRANGE:x+cfg.ZRANGE]

        z = self.gp.find_mean_depth(z_box)

        return [x,y,z],rot

    def execute_grasp(self, grasp_name):
        """
        todo
        1- need to modify to go to point above the grasp before executing the grasp (x, y before z)
            do some thing when lifting (z before x, y)
        2- need to add multiple end points
            pass in integer identifier (mapping to color bin)
            linearly space colors along side of the table
            put certain color in box
        """
 
        self.gripper.open_gripper()

        self.whole_body.end_effector_frame = 'hand_palm_link'

        self.whole_body.move_end_effector_pose(geometry.pose(),grasp_name)

        self.gripper.close_gripper()
        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1),grasp_name)

        self.whole_body.move_end_effector_pose(geometry.pose(z=-0.1),'head_down')

        self.gripper.open_gripper()

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
