import cv2
import IPython
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys

from tpc.perception.cluster_registration import run_connected_components, display_grasps, class_num_to_name, grasps_within_pile, hsv_classify
from tpc.perception.groups import Group
from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.perception.connected_components import get_cluster_info, merge_groups
from tpc.perception.bbox import Bbox, find_isolated_objects_by_overlap, select_first_obj, format_net_bboxes, draw_boxes, find_isolated_objects_by_distance
from tpc.manipulation.robot_actions import Robot_Actions
import tpc.config.config_tpc as cfg
# from tpc.detection.detector import Detector
from tpc.detection.maskrcnn_detection import detect

import importlib

img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

def test():
    path = 'debug_imgs/test.png'
    c_img = cv2.imread(path)

    col_img = ColorImage(c_img)

    output_dict = detect(path)

    bboxes = format_net_bboxes(output_dict, c_img.shape, maskrcnn=True)
    box_viz = draw_boxes(bboxes, c_img)
    cv2.imwrite("debug_imgs/box.png", box_viz)
    single_objs = find_isolated_objects_by_overlap(bboxes)
    if len(single_objs) == 0:
        single_objs = find_isolated_objects_by_distance(bboxes, col_img)

    to_grasp = select_first_obj(single_objs)
    print("Grasping a " + cfg.labels[to_grasp.label])
    try:
        group = to_grasp.to_group(c_img, col_img)
    except ValueError:
        return

    display_grasps(c_img, [group])

if __name__ == "__main__":
    test()
