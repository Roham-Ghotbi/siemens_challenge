import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread("data/example_images/segment1.png")
# img = img[76:500,110:700]
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
#
# # Marker labelling
# im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
#
# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]
# cv2.imwrite("test.png", img)

# img = cv2.imread("data/example_images/segment2.png")
# img = img[76:500,110:700]
# kernel = np.ones((6, 6),np.float32)/36
# blurred = cv2.filter2D(img, -1, kernel)
# dst = cv2.pyrMeanShiftFiltering(blurred, 15, 50, 3)
# gray_image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
# bins = np.linspace(0, 255, 6)
# out = np.digitize(gray_image, bins) * 255/6
#
# color_counts = dict()
# for r in out:
#     for pix in r:
#         # pix = (p[0], p[1], p[2])
#         if pix not in color_counts:
#             color_counts[pix] = 0
#         color_counts[pix] += 1
# print(color_counts)
# colors = []
# thresh = 100
# for color in color_counts.keys():
#     if color_counts[color] > thresh:
#         colors.append((color, color_counts[color]))
# # for i, r in enumerate(out):
# #     for j, p in enumerate(r):
# #         if p == 127:
# #             out[i][j] = 255
# print(colors)
# #try applying slightly smaller mask to remove blurred noise
# print("Num objects found: " + str(len(colors) - 1))
# cv2.imwrite("test.png", out)
