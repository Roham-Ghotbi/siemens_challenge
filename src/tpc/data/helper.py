import sys
import IPython
import tpc.config.config_tpc as cfg
import numpy as np
from tpc.perception.cluster_registration import class_num_to_name
import time

class Helper():
    def __init__(self,config):
        self.config = config
        self.number_of_robots = config.NUM_ROBOTS_ON_NETWORK
        self.alpha_0 = -100
        self.alpha_1 = -10
        self.alpha_2 = -0.5
        self.asked = False

    def ask_for_help(self,bbox):

        if self.config.ASKING_FOR_HELP_POLICY == "NO_HELP":
            self.asked = False
            return False
        elif self.config.ASKING_FOR_HELP_POLICY == "SIMPLE":
            self.asked = True
            return True
        elif self.config.ASKING_FOR_HELP_POLICY == "MODEL_BASE":
            self.asked = False
            return False

    def start_timer(self):
        self.start_count = time.time()

    def stop_timer(self):
        self.duration = time.time() - self.start_count

    def get_reward(self,grasp_success,singulation_time):
        if self.asked:
            reward = self.alpha_0*(1.0-grasp_success)+self.alpha_1+self.alpha_2*(self.duration+singulation_time)
        else:
            reward = self.alpha_0*(1.0-grasp_success)+self.alpha_2*singulation_time

        return reward
