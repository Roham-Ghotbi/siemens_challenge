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

def restrict_focus_mask(focus_mask):
    """cuts off top of focus mask to remove it from
    free space because robot range is estricted near top of workspace
    
    Parameters
    ----------
    focus_mask :obj:`BinaryImage`
        crop of workspace
    Returns
    -------
    :obj:`BinaryImage`
        crop of bottom half of workspace
    """
    middle_mask_pix_y = int(np.mean(focus_mask.nonzero_pixels(), axis=0)[0])
    shape = focus_mask.shape 
    ycoords = [i for i in range(middle_mask_pix_y, int(shape[0]))]
    xcoords = [j for j in range(0, int(shape[1]))]
    valid_pix = []
    for x in xcoords:
        for y in ycoords:
            valid_pix.append([y,x])
    focus_mask = focus_mask.mask_by_ind(np.array(valid_pix))
    cv2.imwrite("smallmask.png", focus_mask.data)
    return focus_mask

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
        `border`: push along border, use free
        space to bias gripper angle
        `free`: push along border with bias to
        free space, don't bias gripper angle
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
    focus_mask = restrict_focus_mask(focus_mask)

    border = get_border(img, obj_mask)
    goal_pixel = get_goal(img, focus_mask, other_objs)
    direction, distance = get_direction(border, goal_pixel, alg=alg)

    mean = np.mean(border, axis=0)
    low = obj_mask.closest_zero_pixel(mean, -1*direction, w=25)
    high = obj_mask.closest_zero_pixel(mean, direction)

    if alg == "free":
        push_dir = high - low 
        push_angle = np.arctan2(push_dir[0], push_dir[1])
        #want gripper perpendicular to push
        gripper_angle = push_angle + np.pi/2.0
        mid_point = None
    elif alg == "border":
        goal_dir = goal_pixel - mean
        goal_dir = goal_dir / np.linalg.norm(goal_dir)
        new_high = obj_mask.closest_zero_pixel(mean, goal_dir, w=25)
        # direction, distance = get_direction(border, goal_pixel, alg="free", max_angle=np.pi)

        # free_low = obj_mask.closest_zero_pixel(mean, -1*direction, w=25)
        # free_high = obj_mask.closest_zero_pixel(mean, direction)
        
        # free_push_dir = free_high - free_low 
        # #want gripper in line with free pixel direction
        # gripper_angle = np.arctan2(free_push_dir[0], free_push_dir[1])
        gripper_angle = 0
        mid_point = new_high
    else:
        raise ValueError("Unsupported algorithm specified. Use `border` or `free`.")

    return low, high, gripper_angle, goal_pixel, mid_point

def display_singulation(low, high, mid, image, goal_pixel, name="singulate"):
    plt.figure()
    ax = plt.axes()
    #push direction
    middle = low * 1.0/4.0 + high * 3.0/4.0
    high =  middle 

    ax.arrow(
            low[1], 
            low[0], 
            middle[1] - low[1], 
            middle[0] - low[0], 
            head_width=10, 
            head_length=10
        )
    #direction gripper would open 
    # grip_vector = np.array([np.sin(rot) * 10, np.cos(rot) * 10])
    # low_grip = middle - grip_vector 
    # high_grip = middle + grip_vector 
    high_grip = mid 
    # low_grip = middle - (mid - middle)
    low_grip = middle
    ax.arrow(
            low_grip[1], 
            low_grip[0], 
            high_grip[1] - low_grip[1], 
            high_grip[0] - low_grip[0], 
            head_width=10, 
            head_length=10
        )
    plt.imshow(image.data)
    plt.plot(goal_pixel[1], goal_pixel[0], 'bo')
    plt.axis('off')
    # plt.savefig("debug_imgs/single.png")
    # plt.show()
    plt.savefig(name + ".png")


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