import tpc.config.config_tpc as cfg
import numpy as np
import IPython
import cv2
import os
import cPickle as pickle
import IPython

class DataManager():
    def __init__(self):
        self.collect = cfg.COLLECT_DATA
        self.rollout_dir = cfg.ROLLOUT_PATH
        folder_name = "rollout"

        dirs = [item for item in os.listdir(self.rollout_dir) if os.path.isdir(os.path.join(self.rollout_dir, item))]

        #begin saving at the lowest available rollout number
        next_rollout_num = 0
        if len(dirs) > 0:
            next_rollout_num = max([int(di[len(folder_name):]) for di in dirs]) + 1

        self.num_rollouts = next_rollout_num
        self.curr_rollout_dir = self.rollout_dir + folder_name + str(next_rollout_num) + "/"
        self.curr_rollout_path = curr_rollout_dir + "rollout.p"
        self.curr_rollout = []
        self.curr_traj = {}

    def clear_traj(self):
        """
        empty trajectory
        """
        if self.collect:
            self.curr_traj = {}

    def update_traj(self, key, value):
        """
        add to or change trajectory values
        """
        if self.collect:
            self.curr_traj[key] = value

    def append_traj(self):
        """
        saves a trajectory at the end of the rollout
        """
        if self.collect:
            self.curr_rollout.append(self.curr_traj)
            if not os.path.exists(self.curr_rollout_dir):
                os.makedirs(self.curr_rollout_dir)
            pickle.dump(self.curr_rollout, open(self.curr_rollout_path, "wb"))

    def overwrite_traj(self):
        """
        overwrites the last trajectory of the rollout
        """
        if self.collect:
            if len(self.curr_rollout) > 0:
                self.curr_rollout[-1] = self.curr_traj
                pickle.dump(self.curr_rollout, open(self.curr_rollout_path, "wb"))
            else:
                raise AssertionError("Rollout must exist before being updated")

    def read_rollout(self, rollout_num):
        rollout_path = self.rollout_dir
        curr_rollout_dir = self.rollout_dir + "rollout" + str(rollout_num) + "/"
        curr_rollout_path = curr_rollout_dir + "rollout.p"
        print(curr_rollout_path)
        if not os.path.exists(curr_rollout_dir):
            raise AssertionError("Rollout number " + str(rollout_num) + " does not exist.")
        rollout = pickle.load(open(curr_rollout_path, "rb"))
        return rollout
