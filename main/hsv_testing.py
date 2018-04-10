import numpy as np
import cv2
import IPython
from perception import ColorImage, BinaryImage
from tpc.perception.cluster_registration import view_hsv

"""
Script to run hsv segmentation on sample images
Can be used to fine tune values if necessary
Works best under consistent lighting (blinds closed) because white highlights on bricks
can be confused with the white background
"""

def get_img(ind):
	return cv2.imread("debug_imgs/data_chris/test" + str(ind) + ".png")

def write_img(img, ind):
	cv2.imwrite("debug_imgs/data_chris/test" + str(ind) + ".png", img)

def hsv_channels(ind):
	img = get_img(ind)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hue = hsv[:,:,0]
	sat = hsv[:,:,1]
	val = hsv[:,:,2]
	write_img(hue, str(ind) + "hue")
	write_img(sat, str(ind) + "sat")
	write_img(val, str(ind) + "val")

if __name__ == "__main__":
	for i in range(0, 6):
		hsv_channels(i)
		# img = get_img(i)
		# viz = view_hsv(ColorImage(img))
		# write_img(viz.data, i)
	# hsv_channels(12)