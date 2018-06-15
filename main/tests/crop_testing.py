import numpy as np
import cv2
import IPython
from perception import ColorImage, BinaryImage
from tpc.perception.crop import crop_img

"""
Script to run crop on sample images
Can be used to fine tune crop if necessary
"""

if __name__ == "__main__":
	get_img = lambda ind: cv2.imread("debug_imgs/new_setup_crop/img" + str(ind) + ".png")
	write_img = lambda img, ind: cv2.imwrite("debug_imgs/new_setup_crop/img_o" + str(ind) + ".png", img)
	for i in range(11, 12):
		img = get_img(i)
		crop_mask = crop_img(img, use_preset=True)
		viz = ColorImage(img).mask_binary(crop_mask)
		write_img(viz.data, i)
