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

    def singulate(self, start, end, c_img, d_img):
        # [355.9527559055119, 123.53543307086613, 977.26812500000005] 0.0
        rot = np.pi/2 + np.arctan2(end[0] - start[0], end[1] - start[1])

        self.gripper.close_gripper()
        # self.go_to_point(start, rot, c_img, d_img)
        # self.go_to_point(end, rot, c_img, d_img)

        y, x = start
        z_box = d_img[y-cfg.ZRANGE:y+cfg.ZRANGE, x-cfg.ZRANGE:x+cfg.ZRANGE]
        z = self.gp.find_mean_depth(z_box)
        # above_start_pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img)
        start_pose_name = self.gripper.get_grasp_pose(x,y,z,rot,c_img=c_img.data)

        y, x = end
        z_box = d_img[y-cfg.ZRANGE:y+cfg.ZRANGE, x-cfg.ZRANGE:x+cfg.ZRANGE]
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
