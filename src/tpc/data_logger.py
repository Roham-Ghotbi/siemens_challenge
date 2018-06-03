import numpy as np
import IPython
import os

###Class created to store relevant information for learning at scale

class DataLogger():

	def __init__(self,file_path, log):
		self.log = log
		if self.log:
			self.data = []
			self.rollout_info = {}
			self.file_path = file_path

			
			if not os.path.exists(self.file_path):
				os.makedirs(self.file_path)

	def save_stat(self,stat_name,value,other_data=None):
		"""
		return: the String name of the next new potential rollout
		(i.e. do not overwrite another rollout)
		"""
		if self.log:
			i = 0

			file_path = self.file_path+'/'+stat_name

			if not os.path.exists(file_path):
				os.makedirs(file_path)

			file = file_path+'/rollout_'+str(i)+'.npy'

			while os.path.isfile(file):
				i += 1
				file = file_path + '/rollout_'+str(i) +'.npy'

			data = {'value':value, 'other_data':other_data}

			np.save(file,data)

	def record_success(self,stat_name,other_data=None):
		if self.log:
			while True:
				print "Was "+ stat_name + " successful?"
				ans =raw_input('(y/n): ')

				if ans == 'y':
					self.save_stat(stat_name,1,other_data=other_data)
					break;
				elif ans == 'n':
					self.save_stat(stat_name,0,other_data=other_data)
					break;
		return ans == "y"

	def record_reward(self, reward):
		self.save_stat("reward", reward)