import tpc.config.config_tpc as cfg
import numpy as np
import IPython
import cv2
import os
import cPickle as pickle

from tpc.data_manager import DataManager

if __name__ == "__main__":
    dm = DataManager(True)
    rollout = dm.read_rollout(0)
    IPython.embed()
    for traj in rollout:
        c_img = traj["c_img"]
        d_img = traj["d_img"]
        c_after = traj["c_img_result"]
        d_after = traj["d_img_result"]

        crop = traj["crop"]
        times = []
        times.append(traj["connected_components_time"])
        times.append(traj["compute_grasps_time"])
        times.append(traj["find_grasps_time"])
        action = traj["action"]
        info = traj["info"]
        succ = traj["success"]
        # times.append(traj["execute_time"])

        if action == "grasp":
            for grasp in info:
                cm, di, mask, class_num = grasp
                # could add ability to go back and label masks with class nums
                # self.classes = cfg.CLASSES
        elif action == "singulate":
            times.append(traj["compute_singulate_time"])
            waypoints, rot, free_pix = info
