import tpc.config.config_tpc as cfg
import numpy as np
import IPython
import cv2
import os
import cPickle as pickle

class DataManager():
    def __init__(self):
        self.rollout_dir = cfg.ROLLOUT_PATH
        roots, dirs, files = os.walk(self.rollout_dir).next()
        #begin saving at the lowest available rollout number
        rollout_num = max([int(di[-1]) for di in dirs]) + 1
        curr_rollout_dir = self.rollout_dir + "rollout" + str(rollout_num) + "/"
        if not os.path.exists(curr_rollout_dir):
            os.makedirs(curr_rollout_dir)
        self.curr_rollout_path = curr_rollout_dir + "rollout.p"
        self.curr_rollout = []
        self.curr_traj = {}

    def clear_traj(self):
        #empty trajectory
        self.curr_traj = {}

    def update_traj(self, key, value):
        #add to or change trajectory values
        self.curr_traj[key] = value

    def append_traj(self):
        #saves a trajectory at the end of the rollout
        self.curr_rollout.append(self.curr_traj)
        pickle.dump(self.curr_rollout, open(self.curr_rollout_path, "wb"))

    def overwrite_traj(self):
        #overwrites the last trajectory of the rollout
        if len(self.curr_rollout) > 0:
            self.curr_rollout[-1] = self.curr_traj
            pickle.dump(self.curr_rollout, open(self.current_rollout_path, "wb"))
        else:
            raise AssertionError("Rollout must exist before being updated")

    def read_rollout(self, rollout_num):
        rollout_path = self.rollout_dir
        curr_rollout_dir = self.rollout_dir + "rollout" + str(rollout_num) + "/"
        curr_rollout_path = curr_rollout_dir + "rollout.p"
        if not os.path.exists(curr_rollout_dir):
            raise AssertionError("Rollout number " + str(rollout_num) + " does not exist.")
        rollout = pickle.read(open(curr_rollout_path, "rb"))
        return rollout
