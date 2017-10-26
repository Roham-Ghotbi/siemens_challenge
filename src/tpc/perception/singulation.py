# To run this test script run python src/tpc/perception/singulation.py 
# from the root of the repo

import matplotlib.pyplot as plt
import skimage as sk
from skimage.draw import polygon
import sys
from perception import ColorImage, BinaryImage
import numpy as np
from sklearn.decomposition import PCA

points = np.array([
        (462.90446650124068, 69.256823821339879), 
        (229.13771712158811, 71.797766749379605), 
        (194.83498759305209, 156.91935483870958), 
        (493.39578163771705, 159.46029776674931)
    ])

def get_focus_mask(shape):
    """ cuts out everything other than the stuff in the tray

    Parameters
    ----------
    image : :obj:`ColorImage`

    Returns
    -------
    :obj:`BinaryImage`
        mask for the tray
    """
    focus_mask = np.zeros(shape, dtype=np.uint8)
    rr, cc = polygon(points[:,1], points[:,0])
    focus_mask[rr,cc] = 255
    return BinaryImage(focus_mask)

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

def get_border_goal(image):
    """ Splits the bricks in the image into two segments, and 
    returns the pixels that border the two segments, for the 
    hsr to push across (approximately), and the goal pixel, 
    which is determined to be the point on the tray that is 
    most open

    Parameters
    ----------
    image : :obj:`ColorImage`
        image to display
    
    Returns
    -------
    :obj:`numpy.ndarray`
        nx2 array of n border pixels
    :obj:`numpy.ndarray`
        1x2 array of the goal pixel
    """
    focus_mask = get_focus_mask(image.data.shape[:2])
    focused = image.mask_binary(focus_mask)
    brick_mask = focused.foreground_mask(60)

    bricks = focused.mask_binary(brick_mask)
    segmented = bricks.segment_kmeans(.1, 2)
    border = segmented.border_pixels()

    # display_border(image, border)
    # display_segments(image, segmented)

    #Bricks and everything outside the tray
    binary_im_framed = brick_mask + focus_mask.inverse()
    plt.imshow(binary_im_framed.data)
    plt.show()
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
    distance = .25*max_distance

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

def find_singulation(image):
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
    border, goal_pixel = get_border_goal(image)
    direction, distance = get_direction(border, goal_pixel)

    mean = np.mean(border, axis=0)
    low = mean - distance*direction
    high = mean + distance*direction

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
    plt.plot(goal_pixel[1], goal_pixel[0], 'bo')
    plt.axis('off')
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