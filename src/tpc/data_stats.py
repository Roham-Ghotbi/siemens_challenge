import tpc.config.config_tpc as cfg
import numpy as np
import IPython
import cv2
import os
import cPickle as pickle

from tpc.data_manager import DataManager

if __name__ == "__main__":
    dm = DataManager(True)
    rollout = dm.read_rollout(6)
    for traj in rollout:
        c_img = traj["c_img"]
        d_img = traj["d_img"]
        crop = traj["crop"]
        times = []
        times.append(traj["connected_components_time"])
        times.append(traj["compute_grasps_time"])
        action = traj["action"]
        info = traj["grasp_info"]
        succ = traj["success"]
        times.append(traj["execute_time"])

        if action == "grasp":
            for grasp in info:
                cm, di, mask, class_num = grasp
                # could add ability to go back and label masks with class nums
                # self.classes = cfg.CLASSES
        elif action == "singulate":
            times.append(traj["compute_singulate_time"])
            waypoints, rot, free_pix = info
        IPython.embed()