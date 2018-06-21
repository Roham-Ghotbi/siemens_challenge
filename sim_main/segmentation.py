import cv2
import numpy as np
import json
from os import listdir, path
import re
import xml.etree.ElementTree as et

def read_img(img_path):
	return cv2.imread(img_path, 0)

def extract_mask(img):
	ret, mask = cv2.threshold(img, 253, 255, 1)
	kernel = np.ones((5, 5), np.uint8)
	# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	im2, contours, hierarchy = cv2.findContours(mask, 1, 1)
	areas = [cv2.contourArea(cnt) for cnt in contours]
	# print(areas)
	moments = [[cv2.moments(cnt), cnt] for cnt in contours if cv2.contourArea(cnt) > 0]
	centroids = [(int(M[0]['m10']/M[0]['m00']), int(M[0]['m01']/M[0]['m00'])) for M in moments]

	for i in range(len(moments)):
		if centroids[i][0] < 10 or mask.shape[1] - centroids[i][0] < 10 or centroids[i][1] < 10 or mask.shape[0] - centroids[i][1] < 10 or cv2.contourArea(moments[i][1]) < 10:
			cv2.drawContours(mask, [moments[i][1]], 0, (0, 0, 0), -1)
		else:
			cv2.drawContours(mask, [moments[i][1]], 0, (255, 255, 255), -1)

	cv2.imwrite("test.png", mask)
	return mask

def find_curr_img_mask(prev_img, curr_img):
	rows, cols = curr_img.shape
	shift_x = 0
	shift_y = 0
	shift_r = 0
	min_diff = np.sum(cv2.subtract(curr_img, prev_img))
	for i in range(10):
		for j in range(10):
			for r in range(6):
				if r > 2:
					M = cv2.getRotationMatrix2D((cols/2,rows/2),(r - 3) * 10,1)
				else:
					M = cv2.getRotationMatrix2D((cols/2,rows/2),(3 - r) * 10,0)
				shift_prev = cv2.warpAffine(prev_img,M,(cols,rows))
				m = np.float32([[1, 0, i-5], [0, 1, j-5]])
				shift_prev = cv2.warpAffine(shift_prev, m, (cols, rows))
				diff = cv2.subtract(curr_img, shift_prev)
				if np.sum(diff) < min_diff:
					shift_x = i - 5
					shift_y = j - 5
					shift_r = (r - 3) * 10
					min_diff = np.sum(diff)
	print(shift_x, shift_y, shift_r)
	kernel = np.ones((3, 3), np.uint8)
	if shift_r > 0: 
		M = cv2.getRotationMatrix2D((cols/2,rows/2),shift_r,1)
	else:
		M = cv2.getRotationMatrix2D((cols/2,rows/2),abs(shift_r),0)
	shift_prev = cv2.warpAffine(prev_img,M,(cols,rows))
	m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
	shift_prev = cv2.warpAffine(prev_img, m, (cols, rows))
	# cv2.imwrite("testorigin.png", shift_prev)
	# cv2.imwrite("testdilate.png", cv2.dilate(shift_prev, kernel))
	diff = cv2.subtract(curr_img, cv2.dilate(shift_prev, kernel))

	return shift_prev, diff

def reduce_noice(mask):
	im2, contours, hierarchy = cv2.findContours(mask, 1, 1)
	if len(contours) == 0:
		return mask

	areas = [cv2.contourArea(cnt) for cnt in contours]
	max_area = max(areas)

	for i in range(len(contours)):
		if areas[i] < max_area - 0.001:
			cv2.drawContours(mask, [contours[i]], 0, (0, 0, 0), -1)
		else:
			cv2.drawContours(mask, [contours[i]], 0, (255, 255, 255), -1)

	# cv2.imwrite("test.png", mask)
	return mask

def find_contour_and_bounding_box(mask):
	im2, contours, hierarchy = cv2.findContours(mask, 1, 1)
	if len(contours) != 1:
		return None, None
	x, y, w, h = cv2.boundingRect(contours[0])
	contours = [pt[0].tolist() for pt in contours[0]]

	return contours, [x, y, x + w, y + h]


def find_item_masks(folder_path):
	masks = []
	with open(folder_path+"/"+"labels.json") as f:
		lst = json.load(f)
	print(lst)
	item_num = len(lst)

	for i in range(item_num):
		curr_mask = extract_mask(read_img(folder_path+"/rgb_"+str(i)+".png"))
		for j in range(i):
			prev_mask = masks[j]
			prev_mask, curr_mask = find_curr_img_mask(prev_mask, curr_mask)
			masks[j] = reduce_noice(prev_mask)
		masks.append(reduce_noice(curr_mask))
	print(len(masks))

	for i in range(item_num):
		cv2.imwrite(folder_path+"/"+lst[i]+".png", masks[i])

def draw_masks(folder_path):
	with open(folder_path+"/"+"labels.json") as f:
		lst = json.load(f)
	item_num = len(lst)
	# print(folder_path+"/rgb_"+str(item_num - 1)+".png")
	img = cv2.imread(folder_path+"/rgb_"+str(item_num - 1)+".png", 1)
	# print(img)

	count = 1

	blank = np.zeros(img.shape, img.dtype)
	for filename in listdir(folder_path):
		if filename.split(".")[1] == "png" and filename.split("_")[0] != 'rgb':
			mask = cv2.imread(folder_path+"/"+filename, 0)
			color = np.zeros(img.shape, img.dtype)
			color[:,:] = (count % 3 * 127,(count // 3) % 3 * 127, (count // 9) % 3 * 127)
			colorMask = cv2.bitwise_and(color, color, mask=mask)
			cv2.addWeighted(colorMask, 1, blank, 1, 0, blank)
			# cv2.imwrite(folder_path+"/masked_"+filename, colorMask)
			count += 1
	cv2.imwrite(folder_path+"/masked_imgs.png", blank)
	cv2.addWeighted(blank, 1, img, 0.5, 0, img)
	cv2.imwrite(folder_path+"/compare_imgs2.png", img)


def create_segment_label(folder_path):
	with open(folder_path+"/"+"labels.json") as f:
		lst = json.load(f)
	item_num = len(lst)

	# set up segmentation json file
	seg_label = {
		"shapes":[],
		"lineColor": [
			0,
			255,
			0,
			128
		],
		"fillColor": [
			255,
			0,
			0,
			128
		], 
		"imagePath": "rgb_"+str(item_num-1)+".png"
	}

	# set up bounding box xml
	bbox_root = et.Element("annotate")

	et.SubElement(bbox_root, "folder").text = folder_path

	et.SubElement(bbox_root, "filename").text = path.abspath("rgb_"+str(item_num-1)+".png")

	source = et.SubElement(bbox_root, "source")
	et.SubElement(source, "database").text = "Unknown"

	size = et.SubElement(bbox_root, "size")
	et.SubElement(size, "width").text = "640"
	et.SubElement(size, "height").text = "480"
	et.SubElement(size, "depth").text = "3"

	et.SubElement(bbox_root, "segmented").text = "0"

	for filename in listdir(folder_path):
		if filename.split(".")[1] == "png" and filename.split("_")[0] != 'rgb':
			mask = cv2.imread(folder_path+"/"+filename, 0)
			seg, bb = find_contour_and_bounding_box(mask)
			class_label = re.split('(\d+)', filename)[0]

			if seg is None:
				continue

			# json add item
			single_item = {"label": class_label, "line_color": None, "fill_color": None, "points": seg}
			seg_label["shapes"].append(single_item)

			# xml add item
			obj_info = et.SubElement(bbox_root, "object")

			et.SubElement(obj_info, "name").text = class_label
			et.SubElement(obj_info, "pose").text = "Unspecified"
			et.SubElement(obj_info, "truncated").text = "0"
			et.SubElement(obj_info, "difficult").text = "0"

			bbox = et.SubElement(obj_info, "bndbox")
			et.SubElement(bbox, "xmin").text = str(bb[0])
			et.SubElement(bbox, "ymin").text = str(bb[1])
			et.SubElement(bbox, "xmax").text = str(bb[2])
			et.SubElement(bbox, "ymax").text = str(bb[3])


	# write to json
	with open(folder_path+"/"+"rgb_"+str(item_num-1)+".json", "w") as out:
		json.dump(seg_label, out)

	# write to xml tree
	tree = et.ElementTree(bbox_root)
	tree.write(folder_path+"/"+"rgb_"+str(item_num-1)+".xml")


