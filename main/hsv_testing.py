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

if __name__ == "__main__":
	get_img = lambda ind: cv2.imread("debug_imgs/hsv_testing/img" + str(ind) + ".png")
	write_img = lambda img, ind: cv2.imwrite("debug_imgs/hsv_testing/img_o" + str(ind) + ".png", img)
	for i in range(5):
		img = get_img(i)
		viz = view_hsv(ColorImage(img))
		write_img(viz.data, i)
