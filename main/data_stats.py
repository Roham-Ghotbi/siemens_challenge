import numpy as np
import IPython
import cv2
import os
import matplotlib.pyplot as plt
import cPickle as pickle
from tpc.perception.cluster_registration import display_grasps, color_to_binary
from tpc.perception.singulation import Singulation
from tpc.data_manager import DataManager
import tpc.config.config_tpc as cfg
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

def init_all_times():
    """
    For each time recorded, initialize a dictionary entry of the form [number of occurences, total time]
    """
    all_times = dict()
    for time_type in ["connected_components_time", "compute_grasps_times", "find_grasps_times", "find_hsv_times",
                "execute_grasp_times", "compute_singulate_time", "execute_singulate_time"]:
        all_times[time_type] = [0.0, 0.0]
    return all_times

def accumulate_all_times(all_times):
    """
    Print the average value of each time type
    """
    for time_type in all_times.keys():
        stored = all_times[time_type]
        avg_val = stored[1]/stored[0] if stored[0] != 0 else 0
        print("Average " + time_type + " was " + "%.1f" % avg_val)

def add_traj_times(traj, all_times):
    """
    Add all of a trajectories timing data to the dictionary
    Some times were stored as a list (if operation occured multiple times per trajectory)
    """
    for time_type in all_times.keys():
        if time_type in traj:
            curr = traj[time_type]
            if not(type(curr) is list):
                curr = [curr]
            for c in curr:
                stored = all_times[time_type]
                all_times[time_type] = [stored[0] + 1.0, stored[1] + c]
    return all_times

def init_all_successes():
    """
    For singulations and grasps, the ith element of the array is [#successes, #attempts] with i legos on the table
    """
    grasp_succ = [[0.0, 0.0] for i in range(1, 17)]
    singulate_succ = [[0.0, 0.0] for i in range(1, 17)]
    return grasp_succ, singulate_succ

def accumulate_all_successes(grasp_succ, singulate_succ):
    """
    Plot success rate by number of legos
    """
    grasp_percents = [100.0 * s[0]/s[1] if s[1] != 0 else 0 for s in grasp_succ]
    singulate_percents = [100.0 * s[0]/s[1] if s[1] != 0 else 0 for s in singulate_succ]
    x_axis = [i for i in range(1, 17)]

    print(grasp_succ)
    print(grasp_percents)
    plt.plot(x_axis, grasp_percents)
    plt.xlabel("Number of legos")
    plt.ylabel("Percent grasps successful")
    plt.savefig("grasp_succ.png")

    plt.figure()
    plt.plot(x_axis, singulate_percents)
    plt.xlabel("Number of legos")
    plt.ylabel("Percent singulations successful")
    plt.savefig("singulate_succ.png")

    grasp_combined = zip(*grasp_succ)
    total_grasp_percent = 100.0 * sum(grasp_combined[0])/sum(grasp_combined[1])
    print("%.1f percent of grasps succeeded" % total_grasp_percent)

    singulate_combined = zip(*singulate_succ)
    total_singulate_percent = 100.0 * sum(singulate_combined[0])/sum(singulate_combined[1])
    print("%.1f percent of singulations succeeded" % total_singulate_percent)

def add_traj_successes(traj, grasp_succ, singulate_succ):
    """
    Adds whether the trajectory succeeded or failed (by action type and number of legos)
    """
    if "action" in traj:
        n = int(traj["num_legos"]) - 1 #convert to 0-indexed
        if traj["action"] == "grasp":
            for t in traj["grasp_successes"]:
                stored = grasp_succ[n]
                succ_bool = 1.0 if t == "y" else 0.0
                grasp_succ[n] = [stored[0] + succ_bool, stored[1] + 1.0]
        elif traj["action"] == "singulate":
            stored = singulate_succ[n]
            succ_bool = 1.0 if traj["singulate_success"] == "y" else 0.0
            singulate_succ[n] = [stored[0] + succ_bool, stored[1] + 1.0]
    return grasp_succ, singulate_succ

if __name__ == "__main__":
    dm = DataManager()

    all_times = init_all_times()
    grasp_succ, singulate_succ = init_all_successes()
    for rnum in range(dm.num_rollouts):
        rollout = dm.read_rollout(rnum)
        for traj in rollout:
            all_times = add_traj_times(traj, all_times)
            grasp_succ, singulate_succ = add_traj_successes(traj, grasp_succ, singulate_succ)
    accumulate_all_times(all_times)
    accumulate_all_successes(grasp_succ, singulate_succ)
