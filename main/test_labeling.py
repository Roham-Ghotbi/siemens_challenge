import cv2
import IPython
from numpy.random import normal
import time
#import listener
import thread

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys


from tpc.perception.cluster_registration import run_connected_components, display_grasps, \
    grasps_within_pile, hsv_classify
from tpc.perception.groups import Group

from tpc.perception.singulation import Singulation
from tpc.perception.crop import crop_img
from tpc.data_manager import DataManager
import tpc.config.config_tpc as cfg
sys.path.append(cfg.WEB_PATH)
from web_labeler import Web_Labeler
from tpc.perception.connected_components import get_cluster_info
from tpc.perception.bbox import Bbox, find_isolated_objects_by_overlap, select_first_obj, format_net_bboxes, draw_boxes, find_isolated_objects_by_distance

import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

"""
This class is for use with the robot
Pipeline it tests is labeling a single object using the web interface,
then the robot grasping the object and placing it in the correct bin by AR marker
"""

class LabelDemo():

    def __init__(self):
        self.web = Web_Labeler(cfg.NUM_ROBOTS_ON_NETWORK)
        print "after thread"

    def bbox_to_fg(self, bbox, c_img, col_img):
        obj_mask = crop_img(c_img, bycoords = [bbox[1], bbox[3], bbox[0], bbox[2]])
        obj_workspace_img = col_img.mask_binary(obj_mask)
        fg = obj_workspace_img.foreground_mask(cfg.COLOR_TOL, ignore_black=True)
        return fg, obj_workspace_img

    def test_bbox_overlap(self, box_a, box_b):
        #bbox has format [xmin, ymin, xmax, ymax]
        if box_a[0] > box_b[2] or box_a[2] < box_b[0]:
            return False
        if box_a[1] > box_b[3] or box_a[3] < box_b[1]:
            return False
        return True

    def find_isolated_objects(self, bboxes, c_img):
        valid_bboxes = []
        for curr_ind in range(len(bboxes)):
            curr_bbox = bboxes[curr_ind]
            overlap = False
            for test_ind in range(curr_ind + 1, len(bboxes)):
                test_bbox = bboxes[test_ind]
                if self.test_bbox_overlap(curr_bbox, test_bbox):
                    overlap = True
                    break
            if not overlap:
                valid_bboxes.append(curr_bbox)
        return valid_bboxes

    def get_bboxes_from_web(self, path):
        labels = self.web.label_image(path)

        objs = labels['objects']
        bboxes = []
        vectors = []
        for obj in objs:
            if obj['motion'] == 1:
                coords = obj['coords']
                p0 = [coords[0], coords[1]]
                p1 = [coords[2], coords[3]]
                vectors.append(([p0, p1], obj['class']))
            else:
                bboxes.append(Bbox(obj['coords'], obj['class']))
        return bboxes, vectors

    def label_demo(self):

        while True:
            path = "/Users/chrispowers/Documents/research/hsr_web/data/images/frame_0.png"
            c_img = cv2.imread(path)
            # print "\n new iteration"
            main_mask = crop_img(c_img, simple=True)
            col_img = ColorImage(c_img)
            workspace_img = col_img.mask_binary(main_mask)

            boxes, vectors = self.get_bboxes_from_web(path)

            if len(boxes) > 0:
                single_objs = find_isolated_objects_by_distance(boxes, col_img)

                #bbox has format [xmin, ymin, xmax, ymax]
                fg, obj_w = self.bbox_to_fg(bbox, c_img, col_img)
                cv2.imwrite("debug_imgs/test.png", obj_w.data)

                groups = get_cluster_info(fg)
                display_grasps(workspace_img, groups)

                group = groups[0]
            elif len(vectors) > 0:
                waypoints, class_labels = vectors[0]
            else:
                print("No action")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEBUG = True
    else:
        DEBUG = False

    task = LabelDemo()
    task.label_demo()
