# To run this test script run python src/tpc/perception/singulation.py 
# from the root of the repo

import matplotlib.pyplot as plt
import skimage as sk
from skimage.draw import polygon
import sys
from perception import ColorImage, BinaryImage
import numpy as np
from sklearn.decomposition import PCA
import cv2
import IPython

def display_border(image, border):
    """ helper to display the border between the segments

    Parameters
    ----------
    image : :obj:`ColorImage`
        image to display
    border : :obj:`numpy.ndarray`
        nx2 array of n border pixels
    """
    plt.imshow(image.data)
    plt.axis('off')
    plt.plot(border[:,1], border[:,0], 'o')
    plt.show()

def display_segments(image, segmented):
    """ helper to display the border between the segments

    Parameters
    ----------
    image : :obj:`ColorImage`
        image to display
    segmented : :obj:`SegmentedImage`
        the segments to be displayed
    """
    for i in range(segmented.num_segments):
        brick = image.mask_binary(segmented.segment_mask(i))
        plt.imshow(brick.data)
        plt.show()

def get_border_goal(img, focus_mask, brick_masks):
    """ Splits the bricks in the image into two segments, and 
    returns the pixels that border the two segments, for the 
    hsr to push across (approximately), and the goal pixel, 
    which is determined to be the point on the tray that is 
    most open

    Parameters
    ----------
    img :obj:`ColorImage`
        original image
    focus_mask :obj:`BinaryImage`
        crop of workspace
    brick_mask :list:obj:`BinaryImage`
        crops of object clusters (1st is one of interest)
    Returns
    -------
    :obj:`numpy.ndarray`
        nx2 array of n border pixels
    :obj:`numpy.ndarray`
        1x2 array of the goal pixel
    """
    bricks = img.mask_binary(brick_masks[0])
    segmented = bricks.segment_kmeans(.1, 2)
    border = segmented.border_pixels()

    # display_border(image, border)
    # display_segments(image, segmented)

    #Bricks and everything outside the tray
    #need all brick clusters here, not just one of interest
    binary_im_framed = focus_mask.inverse()
    for i in range(len(brick_masks)):
        binary_im_framed += brick_masks[i]
    goal_pixel = binary_im_framed.most_free_pixel()
    return border, goal_pixel

def get_direction(points, goal_pixel):
    """ Determines the direction which the robot should push 
    the pile to separate it the most by pushing along the 
    direction of the border pixels between the bricks while 
    pushing towards the goal pixel.  

    Parameters
    ----------
    points : :obj:`numpy.ndarray`
        nx2 array of n border pixels
    goal_pixel : :obj:`numpy.ndarray`
        1x2 array of the goal pixel
    
    Returns
    -------
    :obj:`numpy.ndarray`
        1x2 vector pointing in the direction in which the 
        robot should singulate
    :obj: int
        distance the robot should push to go through the 
        entire pile
    """
    _max_boundary_angle = np.pi/6.
    #optimal pushing direction
    pca = PCA(2)
    pca.fit(points)
    separation_dir = pca.components_[0]
    separation_dir = separation_dir / np.linalg.norm(separation_dir)
    max_distance = pca.explained_variance_[0]
    distance = .4*max_distance

    #optimal pushing location (most open space)
    push_center = np.mean(points, axis=0)
    goal_dir = goal_pixel - push_center
    goal_dir = goal_dir / np.linalg.norm(goal_dir)

    left_separation_dir = np.cos(_max_boundary_angle) * pca.components_[0] + \
                            np.sin(_max_boundary_angle) * pca.components_[1]
    right_separation_dir = np.cos(-_max_boundary_angle) * pca.components_[0] + \
                            np.sin(-_max_boundary_angle) * pca.components_[1]
        
    dot_prod = goal_dir.dot(separation_dir)
    goal_sep_angle = np.arccos(np.abs(dot_prod))
    if goal_sep_angle < _max_boundary_angle:
        # if within the cone, push directy toward the goal
        push_dir = goal_dir
    else:
        left_alignment = np.abs(goal_dir.dot(left_separation_dir))
        right_alignment = np.abs(goal_dir.dot(right_separation_dir))
        if left_alignment > right_alignment:
            push_dir = left_separation_dir
        else:
            push_dir = right_separation_dir

        # reverse direction if necessary
        if dot_prod < 0:
            push_dir = -push_dir
    return push_dir, distance

def find_singulation(img, focus_mask, brick_masks):
    """ Determines the direction which the robot should push 
    the pile to separate it the most by pushing along the 
    direction of the border pixels between the bricks while 
    pushing towards the goal pixel.  

    Parameters
    ----------
    img :obj:`ColorImage`
        original image
    focus_mask :obj:`BinaryImage`
        crop of workspace
    brick_mask :list:obj:`BinaryImage`
        crops of object cluster (1st is one of interest)
    
    Returns
    -------
    :obj:`numpy.ndarray`
        1x2 vector representing the start of the singulation
    :obj: `numpy.ndarray`
        1x2 vector representing the end of the singulation
    """
    #top of focus mask is out of range, so shouldn't be considered free space
    #for most free pixel, need to consider ALL brick masks
    middle_mask_pix_y = int(np.mean(focus_mask.nonzero_pixels(), axis=0)[0])
    shape = focus_mask.shape 
    ycoords = [i for i in range(middle_mask_pix_y, int(shape[0]))]
    xcoords = [j for j in range(0, int(shape[1]))]
    valid_pix = []
    for x in xcoords:
        for y in ycoords:
            valid_pix.append([y,x])
    focus_mask = focus_mask.mask_by_ind(np.array(valid_pix))
    # cv2.imwrite("smallmask.png", focus_mask.data)

    border, goal_pixel = get_border_goal(img, focus_mask, brick_masks)
    direction, distance = get_direction(border, goal_pixel)

    mean = np.mean(border, axis=0)
    brick_mask = brick_masks[0]
    low = brick_mask.closest_zero_pixel(mean, -1*direction, w=25)
    high = brick_mask.closest_zero_pixel(mean, direction)
    return low, high

def display_singulation(low, high, image):
    plt.figure()
    ax = plt.axes()
    ax.arrow(
            low[1], 
            low[0], 
            high[1] - low[1], 
            high[0] - low[0], 
            head_width=10, 
            head_length=10
        )
    plt.imshow(image.data)
    # plt.plot(goal_pixel[1], goal_pixel[0], 'bo')
    plt.axis('off')
    # plt.savefig("debug_imgs/single.png")
    # plt.show()
    plt.savefig("singulate.png")


if __name__ == "__main__":
    files = [
            "data/example_images/frame_40_10.png", 
            "data/example_images/frame_40_11.png", 
            "data/example_images/frame_40_12.png", 
            "data/example_images/frame_40_13.png", 
            "data/example_images/frame_40_14.png"
        ]
    for file in files:
        image = sk.img_as_ubyte(plt.imread(file))
        image = ColorImage(image)
        find_singulation(image)