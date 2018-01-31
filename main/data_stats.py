import tpc.config.config_tpc as cfg
import numpy as np
import IPython
import cv2
import os
import cPickle as pickle
from perception import ColorImage, BinaryImage
from tpc.perception.cluster_registration import display_grasps
from tpc.perception.singulation import display_singulation
from tpc.data_manager import DataManager

if __name__ == "__main__":
    connected_components_times = []
    compute_grasps_times = []
    find_grasps_times = []
    execute_grasp_times = []
    execute_singulation_times = []
    compute_singulation_times = []
    grasp_successes = 0
    grasp_attempts = 0
    singulation_successes = 0
    singulation_attempts = 0
    color_successes = 0
    color_attempts = 0

    to_save_imgs_num = 1

    dm = DataManager(False)
    for rnum in range(dm.num_rollouts):
        rollout = dm.read_rollout(rnum)
        trajnum = 0
        for traj in rollout:
            c_img = traj["c_img"]
            d_img = traj["d_img"]
            # c_after = traj["c_img_result"]
            # d_after = traj["d_img_result"]
            crop = traj["crop"]
            times = []
            connected_components_times.append(traj["connected_components_time"])
            compute_grasps_times.append(traj["compute_grasps_time"])
            find_grasps_times.append(traj["find_grasps_time"])
            action = traj["action"]
            info = traj["info"]
            succ = traj["success"]

            if action == "grasp":
                execute_grasp_times += traj["execute_time"]
                cms = []
                dis = []
                for grasp in info:
                    cm, di, mask, class_num = grasp
                    cms.append(cm)
                    dis.append(di)
                if rnum == to_save_imgs_num:
                    display_grasps(ColorImage(c_img), cms, dis, name="debug_imgs/rollout_imgs/r" + str(trajnum))
                for s in succ:
                    if s != "?":
                        grasp_attempts += 1
                        if s == "y":
                            grasp_successes += 1
                # for c in traj["color"]:
                #     if c != "?":
                #         color_attempts += 1
                #         if c == "y":
                #             color_successes += 1
            elif action == "singulate":
                compute_singulation_times.append(traj["compute_singulate_time"])
                # execute_singulation_times.append(traj["execute_time"])
                waypoints, rot, free_pix = info
                if rnum == to_save_imgs_num:
                    display_singulation(waypoints, ColorImage(crop), free_pix,
                        name = "debug_imgs/rollout_imgs/r" + str(trajnum))
                singulation_attempts += 1
                if succ == "y":
                    singulation_successes += 1
            trajnum += 1

    print("SUCCESS RATES")
    grasp_percent = str((100.0 * grasp_successes)/grasp_attempts)
    print("Succeded in " + str(grasp_successes) + " out of " + str(grasp_attempts) + " grasps (" + grasp_percent + "%)")
    singulation_percent = str((100.0 * singulation_successes)/singulation_attempts)
    print("Succeded in " + str(singulation_successes) + " out of " + str(singulation_attempts) + " singulations (" + singulation_percent + "%)")
    # color_percent = str((100.0 * color_successes)/color_attempts)
    # print("Succeded in " + str(color_successes) + " out of " + str(color_attempts) + " color identifications (" + color_percent + "%)")

    avg = lambda times: str(sum(times)/(1.0 * len(times)))
    print("TIMES")
    print("average connected components time: " + avg(connected_components_times))
    print("average compute grasps time: " + avg(compute_grasps_times))
    print("average find grasps time: " + avg(find_grasps_times))
    print("average execute grasp time: " + avg(execute_grasp_times))
    print("average execute singulation time: " + avg(execute_singulation_times))
    print("average compute singulation time: " + avg(compute_singulation_times))
