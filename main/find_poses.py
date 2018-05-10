import cv2
import IPython
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys

from tpc.perception.cluster_registration import run_connected_components, display_grasps
from tpc.perception.groups import Group
from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.manipulation.robot_actions import Robot_Actions
from tpc.perception.connected_components import get_cluster_info, merge_groups
from tpc.perception.bbox import Bbox, find_isolated_objects_by_overlap, select_first_obj, format_net_bboxes, draw_boxes, find_isolated_objects_by_distance
from tpc.helper import Helper
from tpc.data_logger import DataLogger
import tpc.config.config_tpc as cfg

import importlib
robot_module = importlib.import_module(cfg.ROBOT_MODULE)
Robot_Interface = getattr(robot_module, 'Robot_Interface')

sys.path.append('/home/autolab/Workspaces/michael_working/hsr_web')
from web_labeler import Web_Labeler

img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

from detection import Detector

"""
script used to find specific poses incrementally
"""

if __name__ == "__main__":
    robot = Robot_Interface()
    ra = Robot_Actions(robot)

    ra.go_to_start_pose()

    curr_offsets = np.array([0, 0, 0])
    curr_rot = np.array([0.0,0.0,1.57])

    while True:
        label = id_generator()
        pose_name = robot.create_grasp_pose(curr_offsets[0], curr_offsets[1], curr_offsets[2], curr_rot[2])
        robot.move_to_pose(pose_name, 0.1)

        delta = raw_input()
        while not (delta in ["+x", "-x", "+y", "-y", "+z", "-z"]):
            print("not a valid delta")
            delta = raw_input()

        if delta == "+x":
            curr_offsets[0] += 0.05
        elif delta == "-x":
            curr_offsets[0] -= 0.05
        elif delta == "+y":
            curr_offsets[1] += .05
        elif delta == "-y":
            curr_offsets[1] -= .05
        elif delta == "+z":
            curr_offsets[2] += .05
        elif delta == "-z:":
            curr_offsets[2] -= .05
        
        print(curr_offsets)


