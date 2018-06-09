import cv2
import IPython
import time
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import sys

from tpc.perception.cluster_registration import run_connected_components, display_grasps
from tpc.perception.groups import Group
from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.manipulation.robot_actions import Robot_Actions
from tpc.perception.connected_components import get_cluster_info, merge_groups
from tpc.perception.bbox import Bbox, find_isolated_objects_by_overlap, select_first_obj, format_net_bboxes, draw_boxes, find_isolated_objects_by_distance
from tpc.helper import Helper
# from tpc.data_logger import DataLogger
import tpc.config.config_tpc as cfg
import importlib

import thread
import tf
import rospy
import os

if cfg.robot_name == "hsr":
    from core.hsr_robot_interface import Robot_Interface
elif cfg.robot_name == "fetch":
    from core.fetch_robot_interface import Robot_Interface 
elif cfg.robot_name is None:
    from tpc.offline.robot_interface import Robot_Interface

img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')


class DataCollection():

	def __init__(self):
		"""
		Class that runs decluttering task

		"""
		self.robot = Robot_Interface()
		self.helper = Helper(cfg)
		self.ra = Robot_Actions(self.robot)
		# self.dl = DataLogger("stats_data/model_base", cfg.EVALUATE)
	
		print "Finished init"

		self.ra.go_to_start_pose()
		time.sleep(2)

	def collect(self, dataset_size=200):

		for i in range(dataset_size):
			print(i)
			
			time.sleep(3)

			c_img, d_img = self.robot.get_img_data()

			cv2.imwrite('rgb_{}.png'.format(str(i).zfill(4)), c_img)
			cv2.imwrite('depth_{}.png'.format(str(i).zfill(4)), d_img)
			time.sleep(0.5)
			IPython.embed()

			


if __name__ == "__main__":
	DataCollection().collect()




