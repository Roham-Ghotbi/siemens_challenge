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
import json

if cfg.robot_name == "hsr":
    from core.hsr_robot_interface import Robot_Interface
elif cfg.robot_name == "fetch":
    from core.fetch_robot_interface import Robot_Interface 
elif cfg.robot_name is None:
    from tpc.offline.robot_interface import Robot_Interface

img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

sys.path.append('/home/zisu/simulator/siemens_challenge/sim_world')
from spawn_object_script import *
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetWorldProperties
from segmentation import *


class DataCollection():

	def __init__(self):
		"""
		Class that runs decluttering task

		"""
		self.robot = Robot_Interface()
		self.helper = Helper(cfg)
		self.ra = Robot_Actions(self.robot)
		# self.dl = DataLogger("stats_data/model_base", cfg.EVALUATE)

		self.dm, self.sm, self.om = setup_delete_spawn_service()
	
		print "Finished init"

		clean_floor(self.dm, self.om)

		self.ra.go_to_start_pose()
		time.sleep(2)

	def collect(self, dataset_size=90):

		IMDIR_RGB = 'sim_img_seg/rgb/'
		# IMDIR_DEPTH = 'sim_img_seg/depth/'

		num = len([x for x in os.listdir(IMDIR_RGB)])


		for i in range(dataset_size):
			print(i)
			if not os.path.exists(IMDIR_RGB+str(i+num)):
				os.makedirs(IMDIR_RGB+str(i+num))

			n = np.random.randint(5, 10)

			spawn_from_uniform(n, self.sm)
			labels = get_object_list(self.om)

			for j in range(len(labels)):
				c_img, d_img = self.robot.get_img_data()
				delete_object(labels[len(labels) - 1 - j], self.dm)
				cv2.imwrite(IMDIR_RGB+str(i+num)+'/rgb_{}.png'.format(str(len(labels) - 1 - j)), c_img)
				time.sleep(0.5)

			with open(IMDIR_RGB+str(i+num)+"/labels.json", 'w') as f:
				json.dump(labels, f)
			clean_floor(self.dm, self.om)

			find_item_masks(IMDIR_RGB+str(i+num))
			draw_masks(IMDIR_RGB+str(i+num))


			time.sleep(3)



			# for j in range(np.random.randint(5, 10)):
			# 	label = spawn_from_uniform(1, self.sm)
			# 	time.sleep(3)

			# 	c_img, d_img = self.robot.get_img_data()

			# 	cv2.imwrite(IMDIR_RGB+str(i)+'/rgb_{}.png'.format(str(j)), c_img)
			# 	# cv2.imwrite(IMDIR_DEPTH+'depth_{}.png'.format(str(num+i).zfill(4)), d_img)
			# 	labels.append(label)
			# 	time.sleep(0.5)

			# with open(IMDIR_RGB+str(i)+"/labels.json", 'w') as f:
			# 	json.dump(labels, f)
			# clean_floor(self.dm, self.om)
			# time.sleep(3)


if __name__ == "__main__":
	DataCollection().collect()




