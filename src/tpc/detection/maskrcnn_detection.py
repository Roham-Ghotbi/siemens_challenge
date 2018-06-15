import matplotlib
matplotlib.use('Agg')
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import skimage
import json
import matplotlib
import matplotlib.pyplot as plt
import IPython

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


# Directory to save logs and trained model
ROOT_DIR = os.path.abspath("../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR2 = os.path.join(ROOT_DIR, "logs")
MODEL_PATH = os.path.join(MODEL_DIR2, "mask_rcnn_grasps.h5")
# IMAGE_DIR = os.path.join(ROOT_DIR, "samples/shapes/dataset/images")

class GraspsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "grasps"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    BATCH_SIZE = 1

config = GraspsConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)

class_names =  ['BG', 'scrap', 'tape', 'tube', 'screwdriver']

def detect(image_path):
    image = skimage.io.imread(image_path)
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    figsize=(16, 16)
    _, ax = plt.subplots(1, figsize=figsize)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], ax=ax)
    plt.savefig('debug_imgs/predictions.png')

if __name__ == '__main__':
    detect('dataset_val/images/rgb_raw_0089.png')
    # detect('dataset/images/rgb_raw_0128.png')
