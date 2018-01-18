import numpy as np
import math
import cv2
from scipy.ndimage.measurements import label
from scipy.misc import imresize
from union import UnionFind
from skimage.measure import block_reduce
from groups import Group
from perception import ColorImage, BinaryImage
import IPython

def generate_groups(img):
    """ Finds groups using adjacency
    (dist_tol = 0)
    Parameters
    ----------
    img :obj:`numpy.ndarray`
        binary foreground image
    Returns
    -------
    :obj:list of `Group`
    """
    #give each component a different integer label in the output matrix
    labeled_img = np.zeros(img.shape)

    #all ones --> 8 connectivity (use cross shape for 4 connectivity)
    struct = np.ones((3, 3))
    n_features = label(img, structure = struct, output = labeled_img)

    groups_by_label = {}
    groups = []

    #use labels to put pixels into groups
    for y in range(len(img)):
        for x in range(len(img[0])):
            #foreground pixels
            if img[y][x]:
                curr_label = labeled_img[y][x]

                #create the group if it does not yet exist
                if curr_label not in groups_by_label:
                    groups_by_label[curr_label] = Group(curr_label)
                    groups.append(groups_by_label[curr_label])

                groups_by_label[curr_label].add((y, x))

    return groups

def merge_groups(groups, dist_tol):
    """ Merges untill all groups have at least
    `dist_tol` distance between them
    Parameters
    ----------
    groups :obj:list of `Group`
        groups found using adjacency
    dist_tol : int
        minimum euclidean distance
        to be in same cluster
    Returns
    -------
    :obj:list of `Group`
    """
    #give groups initial labels = indexes
    for i in range(len(groups)):
        groups[i].updateLabel(i)

    uf = UnionFind(len(groups))

    #find labels to merge
    for i in range(len(groups)):
        curr_group = groups[i]
        #only look at groups farther in list
        for j in range(i, len(groups)):
            other_group = groups[j]

            #short circuit if already connected (minimize nearest neighbor calls)
            if not uf.find(curr_group.label, other_group.label):
                if Group.nearby(curr_group, other_group, dist_tol):
                    uf.union(curr_group.label, other_group.label)

    merged_groups = []
    unmerged_groups = []
    #iterate until all merges have been made
    while len(groups) > 0:
        curr_group = groups[0]
        merged_groups.append(curr_group)
        for j in range(1, len(groups)):
            other_group = groups[j]

            #on each iteration, one merged group moves to the new array
            if uf.find(curr_group.label, other_group.label):
                curr_group.merge(other_group)
            #all other groups are kept in the old array
            else:
                unmerged_groups.append(other_group)

        groups = unmerged_groups
        unmerged_groups = []

    return merged_groups

"""return a list of the smallest n groups"""
def get_smallest(groups, n):
    groups.sort()
    return groups[:min(n, len(groups))]

def get_cluster_info(img, dist_tol, size_tol):
    """ Generates mask for
    each cluster of objects
    Parameters
    ----------
    img :obj:`BinaryImage`
        mask of objects in working area
    dist_tol : int
        minimum euclidean distance
        to be in same cluster
    size_tol : int
        minimum cluster size
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
    scale_factor = 2

    img_data = img.data
    orig_shape = img_data.shape
    img_data = block_reduce(img_data, block_size = (scale_factor, scale_factor), func = np.mean)
    dist_tol = dist_tol/scale_factor

    #find groups of adjacent foreground pixels
    groups = generate_groups(img_data)
    groups = [g for g in groups if g.area >= size_tol/scale_factor]
    groups = merge_groups(groups, dist_tol)

    center_masses = [map(lambda x: x * scale_factor, g.center_mass()) for g in groups]
    directions = [g.orientation() for g in groups]
    masks = [BinaryImage(imresize(g.get_mask(img_data.shape),orig_shape)) for g in groups]
    return center_masses, directions, masks
