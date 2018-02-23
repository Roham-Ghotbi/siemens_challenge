import numpy as np
import cv2
import IPython
from connected_components import get_cluster_info
from groups import Group
from perception import ColorImage, BinaryImage
import matplotlib.pyplot as plt
import tpc.config.config_tpc as cfg
from sklearn.decomposition import PCA
import time

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
        cv2.imwrite("debug_imgs/mask.png", fg.data)
    # img = cv2.medianBlur(img, 3)

    center_masses, directions, masks = get_cluster_info(fg, dist_tol, size_tol)
    #would like to filter out clusters that are just lines here

    if viz:
        display_grasps(img, center_masses, directions)

    return center_masses, directions, masks

def draw_point(img, point):
    box_color = (255, 0, 0)
    box_size = 5
    img[int(point[0] - box_size):int(point[0] + box_size),
        int(point[1] - box_size):int(point[1] + box_size)] = box_color
    return img

def display_grasps(img, center_masses, directions,name="debug_imgs/grasps"):
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
    line_color = box_color[::-1]
    img_data = np.copy(img.data)
    for i in range(len(center_masses)):
        cm = center_masses[i]
        d = directions[i]

        img_data = draw_point(img_data, cm)

        p0 = tuple((cm - d * cfg.LINE_SIZE/2)[::-1].astype('uint32'))
        p1 = tuple((cm + d * cfg.LINE_SIZE/2)[::-1].astype('uint32'))
        cv2.line(img_data, p0, p1, line_color, 2)
    plt.figure()
    plt.imshow(img_data)
    cv2.imwrite(name + ".png", img_data)
    if cfg.QUERY:
        plt.show()

def dist_mod(m, a, b):
    """
    returns shortest distance between a and b under mod m
    """
    diff = abs(a-b)
    if diff < m/2:
        return diff
    else:
        return m - diff  

def get_hsv_hist(img):
    """ Separates image into bins by HSV and creates histograms
    Parameters
    ----------
    img :obj:`ColorImg`
        color image masked to relevant area
    Returns
    -------
    :obj:tuple of
        :obj:dict
            hsv value to count
        :obj:dict
            hsv value to list of pixel coordinates
    """
    hsv = cv2.cvtColor(img.data, cv2.COLOR_BGR2HSV)

    hue_counts = {}
    hue_pixels = {}

    for rnum, row in enumerate(hsv):
        for cnum, pix in enumerate(row):
            hue = pix[0]
            val = pix[2]
            sat = pix[1]
            #ignore white
            if not (sat < cfg.WHITE_FACTOR * cfg.SAT_RANGE):
                #black is its own bin 
                if val < cfg.BLACK_FACTOR * cfg.VALUE_RANGE:
                    bin_hue = -1
                else:
                    non_black_hues = cfg.HUE_VALUES.keys()
                    bin_hue = min(non_black_hues, key = lambda h:
                        dist_mod(cfg.HUE_RANGE, hue, h))

                if bin_hue not in hue_counts:
                    hue_counts[bin_hue] = 0
                    hue_pixels[bin_hue] = []
                hue_counts[bin_hue] += 1
                hue_pixels[bin_hue].append([rnum, cnum])

    return hue_counts, hue_pixels

def view_hsv(img):
    """ Separates image into bins by HSV and creates visualization
    Parameters
    ----------
    img :obj:`ColorImg`
    Returns
    -------
    :obj:`ColorImg`
    """
    _, pixels = get_hsv_hist(img)

    view = np.zeros(img.data.shape)
    for hsv_col in pixels.keys():
        points = pixels[hsv_col]
        for p in points:
            #black case
            if hsv_col == -1:
                view[p[0]][p[1]] = [0, 255, 255]
            else:
                view[p[0]][p[1]] = [hsv_col, 128, 128]
    view = view.astype(np.uint8)
    col = cv2.cvtColor(view, cv2.COLOR_HSV2BGR)
    return ColorImage(col)

def hsv_classify(img):
    """ Classifies the lego by HSV
    Parameters
    ----------
    img :obj:`ColorImg`
        mask of lego
    Returns
    -------
    :integer
        class number from 0 to 7
    """
    hue_to_count, _ = get_hsv_hist(img)
    dominant_hue = max(hue_to_count, key=hue_to_count.get)

    all_hues = cfg.ALL_HUE_VALUES.keys()
    all_hues.sort()

    class_num = all_hues.index(dominant_hue)
    return class_num

def class_num_to_name(class_num):
    """ Gets the color name for the index
    Parameters
    ----------
    class_num :integer 
        class number from 0 to 7
    Returns
    -------
    :obj:string
        color name
    """
    all_hues = cfg.ALL_HUE_VALUES.keys()
    all_hues.sort()
    hue = all_hues[class_num]
    color_name = cfg.ALL_HUE_VALUES[hue]
    return color_name

def is_valid_grasp(point, focus_mask):
    """ Checks that the point does not overlap
    another object in the cluster
    Parameters
    ----------
    point :obj:list of `numpy.ndarray`
        1x2 point
    focus_mask :obj:`BinaryImg`
        mask of object cluster
    Returns
    -------
    boolean
    """
    ymid = int(point[0])
    xmid = int(point[1])

    for y in range(ymid - cfg.CHECK_RANGE, ymid + cfg.CHECK_RANGE):
        for x in range(xmid - cfg.CHECK_RANGE, xmid + cfg.CHECK_RANGE):
            if focus_mask.data[y][x] != 0:
                return False
    return True

def grasps_within_pile(color_mask):
    """ Finds any grasps within the pile
    Parameters
    ----------
    color_mask :obj:`ColorImg`
        mask of object cluster
    Returns
    -------
    :obj:tuple of
        :obj:list of `numpy.ndarray`
            grasp center of masses
        :obj:list of `numpy.ndarray`
            grasp angles
    """
    # print "inside the grasps"
    a = time.time()
    hue_counts, hue_pixels = get_hsv_hist(color_mask)
    hsv_hist_time = time.time() - a

    individual_masks = []
    a = time.time()
    #color to binary
    # focus_mask = color_mask.to_binary() #this only looks at 1 channel so it doesn't always work correctly
    focus_mask = BinaryImage(np.sum(255 * (color_mask.data > 0), axis=2).astype(np.uint8))
    #this is too slow 
    # focus_mask = np.zeros(color_mask.data.shape[0:2])
    # for i, r in enumerate(color_mask.data):
    #    for j, c in enumerate(r):
    #        if c[0] > 0 or c[1] > 0 or c[2] > 0:
    #            focus_mask[i][j] = 255
    # focus_mask = BinaryImage(focus_mask.astype(np.uint8))
    col_to_bin_time = time.time() - a
    a = time.time()
    #segment by hsv
    for block_color in hue_counts.keys():
        #same threshold values for number of objects
        if hue_counts[block_color] > cfg.SIZE_TOL:
            valid_pix = hue_pixels[block_color]
            obj_mask = focus_mask.mask_by_ind(np.array(valid_pix))
            individual_masks.append(obj_mask)
    hsv_seg_time = time.time() - a
    obj_focus_mask = individual_masks[0]
    for im in individual_masks[1:]:
        obj_focus_mask += im

    #for each hsv block, again separate by connectivity
    a = time.time()
    all_center_masses = []
    all_directions = []
    all_masks = []
    for obj_mask in individual_masks:
        center_masses, directions, masks = get_cluster_info(obj_mask, cfg.DIST_TOL, cfg.SIZE_TOL)
        directions = [d/np.linalg.norm(d) for d in directions]

        for grasp_info in zip(center_masses, directions, masks):
            #matches endpoints of line in visualization
            grasp_top = grasp_info[0] + grasp_info[1] * cfg.LINE_SIZE/2
            grasp_bot = grasp_info[0] - grasp_info[1] * cfg.LINE_SIZE/2
            if is_valid_grasp(grasp_top, obj_focus_mask) and is_valid_grasp(grasp_bot, obj_focus_mask):
                all_center_masses.append(grasp_info[0])
                all_directions.append(grasp_info[1])
                all_masks.append(grasp_info[2])
    seperate_time = time.time() - a
    lis = [hsv_hist_time, col_to_bin_time, hsv_seg_time, seperate_time]

    return all_center_masses, all_directions, all_masks
