import numpy as np
import cv2
import IPython
from connected_components import get_cluster_info
from perception import ColorImage, BinaryImage
import matplotlib.pyplot as plt

def run_connected_components(img, dist_tol=5, color_tol=45, size_tol=300, viz=False):
    """ Generates mask for
    each cluster of objects
    Parameters
    ----------
    img :obj:`ColorImage`
        color image masked to
        white working area
    dist_tol : int
        minimum euclidean distance
        to be in same cluster
    color_tol : int
        minimum color distance
        to not threshold out 
    size_tol : int
        minimum cluster size
    viz : boolean
        if true, displays proposed grasps
    Returns
    -------
    :obj:tuple of 
        :obj:list of `numpy.ndarray`
            cluster center of masses
        :obj:list of `numpy.ndarray`
            cluster grasp angles
        :obj:list of `BinaryImage`
            cluster masks
    """

    fg = img.foreground_mask(color_tol, ignore_black=True)
    if viz:
        cv2.imwrite("mask.png", img.data)
    # img = cv2.medianBlur(img, 3)

    center_masses, directions, masks = get_cluster_info(fg, dist_tol, size_tol)
    if viz:
        cv2.imwrite("grasps.png", visualize(img, center_masses, directions))

    return center_masses, directions, masks

def visualize(img, center_masses, directions):
    """ Displays the proposed grasps
    Parameters
    ----------
    img :obj:`ColorImage`
        color image masked to
        white working area
    center_masses :obj:list of `numpy.ndarray`
    directions :obj:list of `numpy.ndarray`
    Returns
    -------
    :obj:`numpy.ndarray`
        visualization image
    """
    box_color = (255, 0, 0)
    box_size = 5
    line_color = box_color[::-1]
    line_size = 40
    img_data = img.data
    for i in range(len(center_masses)):
        cm = center_masses[i]
        d = directions[i] #True if y orientation

        img_data[int(cm[0] - box_size):int(cm[0] + box_size), 
            int(cm[1] - box_size):int(cm[1] + box_size)] = box_color
        p0 = tuple((cm - d * line_size/2)[::-1].astype('uint32'))
        p1 = tuple((cm + d * line_size/2)[::-1].astype('uint32'))
        cv2.line(img_data, p0, p1, line_color, 2)

    return img_data

def has_multiple_objects(img, algo="size"):
    """ Counts the objects in a cluster
    Parameters
    ----------
    img :obj:`BinaryImg`
        cluster mask
    algo :obj:`str`, optional
        Algorithm to use for counting (`size` or `color`)
    Returns
    -------
    :boolean
        True if multiple objects, else False
    """
    if algo == "size":
        #relies on objects being the same size
        n_pixels = len(img.nonzero_pixels())
        n_objs = int(n_pixels/550)
    elif algo == "color":
        bin_span = 128
        bins = [0 for i in range((256/bin_span)**3)]
        img_data = img.data
        for row in img_data:
            for pixel in row:
                b, g, r = pixel
                #put into bins
                b, g, r = int(b/bin_span), int(g/bin_span), int(r/bin_span)
                #calculate unique index for every combination of bins
                ind = int(b * (256/bin_span)**2 + g * (256/bin_span) + r)
                bins[ind] += 1
        n_objs = sum(np.array(bins) > 250) - 1 #ignore black
    else:
        raise ValueError("Unsupported algorithm specified. Use 'size' or 'color'")
    if n_objs == 0:
        raise ValueError("Cluster should have at least 1 object")
    return n_objs > 1
