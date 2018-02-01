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

    grasp_crashes = 0
    singulate_crashes = 0
    completions = 0
    false_completions = 0
    total_stopped = 0

    grasps_per_rollout = []
    singulates_per_rollout = []

    actions_before_crash = []

    to_save_imgs_num = 1

    dm = DataManager(False)
    for rnum in range(dm.num_rollouts):
        rollout = dm.read_rollout(rnum)
        trajnum = 0
        rollout_grasps = 0
        rollout_singulates = 0
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


            if "stop_condition" in traj:
                total_stopped += 1
                stop = traj["stop_condition"]
                if stop == "y":
                    completions += 1
                elif stop == "n":
                    false_completions += 1
                elif stop == "crash":
                    if action == "grasp":
                        grasp_crashes += 1
                    elif action =="singulate":
                        singulate_crashes += 1

            if action == "grasp":
                rollout_grasps += 1
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
                for c in traj["color"]:
                    if c != "?":
                        color_attempts += 1
                        if c == "y":
                            color_successes += 1
            elif action == "singulate":
                rollout_singulates += 1
                compute_singulation_times.append(traj["compute_singulate_time"])
                execute_singulation_times.append(traj["execute_time"])
                waypoints, rot, free_pix = info
                if rnum == to_save_imgs_num:
                    display_singulation(waypoints, ColorImage(crop), free_pix,
                        name = "debug_imgs/rollout_imgs/r" + str(trajnum))
                singulation_attempts += 1
                if succ == "y":
                    singulation_successes += 1
            trajnum += 1
        grasps_per_rollout.append(rollout_grasps)
        singulates_per_rollout.append(rollout_singulates)

    print("SUCCESS RATES")
    percent = lambda succ, tot: "(" + str((100.0 * succ)/tot) + "%)"
    succ_rate = lambda succ, tot, name:
        "Succeded in " + str(succ) + " out of " + str(tot) + " " + name + " " + percent(succ, tot)
    print(succ_rate(grasp_successes, grasp_attempts, "grasps"))
    print(succ_rate(singulation_successes, singulation_attempts, "singulations"))
    print(succ_rate(color_successes, color_attempts, "color identifications"))

    print("STOPPING CONDITIONS")
    print("Out of " + str(total) + " rollouts, " +
        str(grasp_crashes) + " " + percent(grasp_crashes, total) + " ended in crashes after grasping, " +
        str(singulate_crashes) + " " + percent(singulate_crashes, total) + " ended in crashes after singulating, " +
        str(completions) + " " + percent(completions, total) + " cleared the table completely, and " +
        str(false_completions) + " " + percent(false_completions, total) + " stopped before clearing the table.")

    avg = lambda times: str(sum(times)/(1.0 * len(times)))
    print("ACTION BREAKDOWN")
    print("average grasps per rollout: " + avg(grasps_per_rollout))
    print("average singulations per rollout: " + avg(singulates_per_rollout))

    print("TIMES")
    print("average connected components time: " + avg(connected_components_times))
    print("average compute grasps time: " + avg(compute_grasps_times))
    print("average find grasps time: " + avg(find_grasps_times))
    print("average execute grasp time: " + avg(execute_grasp_times))
    print("average execute singulation time: " + avg(execute_singulation_times))
    print("average compute singulation time: " + avg(compute_singulation_times))
