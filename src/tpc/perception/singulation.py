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
import tpc.config.config_tpc as cfg

def display_border(img, border):
    """ helper to display the border between the segments

    Parameters
    ----------
    img : :obj:`ColorImage`
        image to display
    border : :obj:`numpy.ndarray`
        nx2 array of n border pixels
    """
    plt.imshow(img.data)
    plt.axis('off')
    plt.plot(border[:,1], border[:,0], 'o')
    plt.show()

def display_segments(img, segmented):
    """ helper to display the border between the segments

    Parameters
    ----------
    img : :obj:`ColorImage`
        image to display
    segmented : :obj:`SegmentedImage`
        the segments to be displayed
    """
    for i in range(segmented.num_segments):
        brick = img.mask_binary(segmented.segment_mask(i))
        plt.imshow(brick.data)
        plt.show()

def get_border(img, obj_mask):
    """ Splits the object pile into two segments, and
    returns the pixels that border the two segments to be
    used as the pushing direction

    Parameters
    ----------
    img :obj:`ColorImage`
        original image
    obj_mask :obj:`BinaryImage`
        crop of object cluster
    Returns
    -------
    :obj:`numpy.ndarray`
        nx2 array of n border pixels
    """
    bricks = img.mask_binary(obj_mask)
    segmented = bricks.segment_kmeans(.1, 2)
    border = segmented.border_pixels()

    # display_border(img, border)
    # display_segments(img, segmented)

    #rare case- 2 clusters aren't touching
    if len(border) == 0:
        print("no boundary points")
        border = obj_mask.nonzero_pixels()

    return border

def get_goal(img, focus_mask, other_objs):
    """ Finds the goal pixel, which is the point
    in the workspace furthest from all walls and
    other object clusters

    Parameters
    ----------
    img :obj:`ColorImage`
        original image
    focus_mask :obj:`BinaryImage`
        crop of workspace
    other_objs :list:obj:`BinaryImage`
        list of crops of other object clusters
    Returns
    -------
    :obj:`numpy.ndarray`
        1x2 array of the goal pixel
    """
    occupied_space = focus_mask.inverse()
    for i in range(len(other_objs)):
        occupied_space += other_objs[i]
    goal_pixel = occupied_space.most_free_pixel()
    return goal_pixel

def get_direction(border, goal_pixel, alg="border", max_angle=np.pi/6.0):
    """ Finds the direction in which to push the
    pile to separate it the most using the direction
    of the border pixels

    Parameters
    ----------
    border :obj:`numpy.ndarray`
        nx2 array of n border pixels
    goal_pixel :obj:`numpy.ndarray`
        1x2 array of the goal pixel
    alg :string
        `border`: push along border
        `free`: push along border with bias to
        free space
    max_angle :float
        maximum angle bias towards free space
    Returns
    -------
    :obj:`numpy.ndarray`
        1x2 vector pointing in the direction in which the
        robot should singulate
    :obj: int
        distance the robot should push to go through the
        entire pile
    """
    #optimal pushing direction
    pca = PCA(2)
    pca.fit(border)
    border_dir = pca.components_[0]
    border_dir /= np.linalg.norm(border_dir)
    max_distance = pca.explained_variance_[0]
    distance = .4*max_distance

    #optimal pushing location (most open space)
    push_center = np.mean(border, axis=0)
    goal_dir = goal_pixel - push_center
    goal_dir = goal_dir / np.linalg.norm(goal_dir)
    dot_prod = goal_dir.dot(border_dir)

    if alg == "border":
        push_dir = border_dir
        #prioritize upwards (high y to low in image)
        if push_dir[0] > 0:
            push_dir = -push_dir
    elif alg == "free":
        #define cone around optimal direction
        goal_sep_angle = np.arccos(np.abs(dot_prod))
        if goal_sep_angle < max_angle:
            # if within the cone, push directy toward the goal
            push_dir = goal_dir
        else:
            #if not within the cone, use the closest edge of the cone
            left_border_dir = np.cos(max_angle) * pca.components_[0] + \
                                np.sin(max_angle) * pca.components_[1]
            right_border_dir = np.cos(-max_angle) * pca.components_[0] + \
                                np.sin(-max_angle) * pca.components_[1]

            left_alignment = np.abs(goal_dir.dot(left_border_dir))
            right_alignment = np.abs(goal_dir.dot(right_border_dir))
            if left_alignment > right_alignment:
                push_dir = left_border_dir
            else:
                push_dir = right_border_dir
        # reverse direction if necessary
        if dot_prod < 0:
            push_dir = -push_dir
    else:
        raise ValueError("Unsupported algorithm specified. Use `border` or `free`.")

    return push_dir, distance

def find_singulation(img, focus_mask, obj_mask, other_objs, alg="border"):
    """ Finds the direction in which the robot should push
    the pile to separate it the most, and the gripper angle
    which will keep objects farthest from other piles

    Parameters
    ----------
    img :obj:`ColorImage`
        original image
    focus_mask :obj:`BinaryImage`
        crop of workspace
    obj_mask :obj:`BinaryImage`
        crop of object cluster
    other_objs :list:obj:`BinaryImage`
        list of crops of other object clusters
    alg :string
        `border`: push along border then to free space
        `free`: push along border with bias to free space
    Returns
    -------
    :obj:`numpy.ndarray`
        1x2 vector representing the start of the singulation
    :obj: `numpy.ndarray`
        1x2 vector representing the end of the singulation
    float
        angle of gripper, aligned towards free space
    :obj: `numpy.ndarray`
        1x2 vector representing the goal pixel
    :obj: `numpy.ndarray`
        1x2 vector representing the middle of singulation, if any
    """

    border = get_border(img, obj_mask)
    goal_pixel = get_goal(img, focus_mask, other_objs)

    direction, distance = get_direction(border, goal_pixel, alg=alg)

    mean = np.mean(border, axis=0)
    low = obj_mask.closest_zero_pixel(mean, -1*direction)
    high = obj_mask.closest_zero_pixel(mean, direction)

    #border should go towards free space (high closer to free pixel)
    low_d = np.linalg.norm(goal_pixel - low)
    high_d = np.linalg.norm(goal_pixel - high)
    if low_d < high_d:
        high, low = low, high

    waypoints = []

    #make sure starting point is not in pile
    low += (low - high)/np.linalg.norm(low-high) * 1.2

    waypoints.append(low)

    if alg == "free":
        waypoints.append(high)
        push_dir = high - low
        push_angle = np.arctan2(push_dir[0], push_dir[1])
        #want gripper perpendicular to push
        gripper_angle = push_angle + np.pi/2.0
    elif alg == "border":
        waypoints.append(low * 1.0/4.0 + high * 3.0/4.0)
        # goal_in_pile = goal_pixel in obj_mask.nonzero_pixels()
        # if not goal_in_pile:
        goal_dir = goal_pixel - mean
        goal_dir = goal_dir / np.linalg.norm(goal_dir)
        towards_free = obj_mask.closest_zero_pixel(mean, goal_dir, w=40)

        #want closer of goal pixel, and point closest to goal pixel on edge of mask
        closer_goal = min([goal_pixel, towards_free], key = lambda p: np.linalg.norm(p - mean))

        waypoints.append(closer_goal)
        gripper_angle = 0
    else:
        raise ValueError("Unsupported algorithm specified. Use `border` or `free`.")

    return waypoints, gripper_angle, goal_pixel

def display_singulation(waypoints, image, goal_pixel, name="debug_imgs/singulate"):
    """
    saves visualization of singulation trajectories

    Parameters
    ----------
    waypoints :list:`numpy.ndarray`
        1x2 points
        subsequent points connected by arrows
    image: obj:`ColorImage`
    goal_pixel :`numpy.ndarray`
        1x2 point to be marked with circle
    name :string
        name of saved image file
    """
    plt.figure()
    ax = plt.axes()
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i+1]
        ax.arrow(
            start[1], start[0],
            end[1] - start[1], end[0] - start[0],
            head_width = 10, head_length = 10
        )

    plt.imshow(image.data)
    plt.plot(goal_pixel[1], goal_pixel[0], 'bo')
    plt.axis('off')
    plt.savefig(name + ".png")  
    if cfg.QUERY:
        plt.show()

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
