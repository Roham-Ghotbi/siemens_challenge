import cv2
import IPython
from numpy.random import normal
import time
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
import sys

from tpc.perception.cluster_registration import run_connected_components, display_grasps, \
    grasps_within_pile, hsv_classify
from tpc.perception.groups import Group
from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.perception.image import ColorImage, BinaryImage

import tpc.config.config_tpc as cfg
from tpc.data_manager import DataManager

if __name__ == "__main__":
    c_img = cv2.imread("debug_imgs/singulate_fails/0/orig.png")

    main_mask = crop_img(c_img, use_preset=True)
    col_img = ColorImage(c_img)
    workspace_img = col_img.mask_binary(main_mask)

    groups = run_connected_components(workspace_img, viz=False)

    to_singulate = []
    to_grasp = []
    for group in groups:
        inner_groups = grasps_within_pile(col_img.mask_binary(group.mask))

        if len(inner_groups) == 0:
            to_singulate.append(group)
        else:
            for in_group in inner_groups:
                pose,rot = None, None
                grasp_pose = None
                suction_pose = None

                class_num = None

                to_grasp.append((in_group, grasp_pose, suction_pose, class_num))

    print("singulate: " + str(len(to_singulate)))
    print("grasp: " + str(len(to_grasp)))

    if len(to_grasp) > 0:
        to_grasp.sort(key=lambda g:-1 * g[0].cm[0])
        display_grasps(workspace_img, [g[0] for g in to_grasp])
    else:
        singulator = Singulation(col_img, main_mask, [g.mask for g in to_singulate])
        waypoints, rot, free_pix = singulator.get_singulation()
        singulator.display_singulation()
