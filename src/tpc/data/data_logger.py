import numpy as np
import IPython
import os

###Class created to store relevant information for learning at scale

class DataLogger():

	def __init__(self,file_path):

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
		i = 0

		file_path = self.file_path+'/'+stat_name

		if not os.path.exists(file_path):
			os.makedirs(file_path)

		stats_file_path = file_path+'/rollout_'+str(i)+'.npy'


		while os.path.isfile(stats_file_path):
			i += 1
			path = self.file_path + '/rollout_'+str(i) +'.npy'

		data = {'value':value, 'other_data':other_data}

		np.save(stats_file_path,data)




	def record_success(self,stat_name,other_data=None):

		while True:
			print("WAS "+ stat_name + "SUCCESUFL (Y/N)?")
			ans =raw_input('(y/n): ')

			if ans == 'y':
				self.save_stat(stat_name,1,other_data=other_data)
				break;
			elif ans == 'n':
				self.save_stat(stat_name,0,other_data=other_data)
				break;

		return
