import numpy as np
import cv2
import IPython
from connected_components import get_cluster_info
from groups import Group
from perception import ColorImage, BinaryImage
import matplotlib.pyplot as plt
import tpc.config.config_tpc as cfg
from sklearn.decomposition import PCA

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

    #filter lines (marking workspace)
    filtered_centers, filtered_dirs, filtered_masks = [], [], []
    for info in zip(center_masses, directions, masks):
        pca = PCA(2)
        pca.fit(info[2].nonzero_pixels())

        #much more variance in one direction indicates a line
        v1, v2 = pca.explained_variance_[0], pca.explained_variance_[1]
        if v1 <= 10 * v2:
            filtered_centers.append(info[0])
            filtered_dirs.append(info[1])
            filtered_masks.append(info[2])
    center_masses, directions, masks = filtered_centers, filtered_dirs, filtered_masks

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
    cv2.imwrite(name + ".png", img_data)

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
            is_white = sat < 0.1 * cfg.SAT_RANGE
            if not is_white:
                #black is its own bin
                is_black = val < 0.3 * cfg.VALUE_RANGE
                if is_black:
                    bin_hue = -1
                else:
                    all_hues = cfg.HUE_VALUES.keys()
                    bin_hue = min(all_hues, key = lambda h:
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
    counts, _ = get_hsv_hist(img)
    color = max(counts, key=counts.get)

    all_hues = cfg.HUE_VALUES.keys() + [-1]
    all_hues.sort()

    class_num = all_hues.index(color)
    return class_num

def class_num_to_name(class_num):
    all_hues = cfg.HUE_VALUES.keys() + [-1]
    all_hues.sort()
    color = all_hues[class_num]
    if color == -1:
        color_name = "black"
    else:
        color_name = cfg.HUE_VALUES[color]
    return color_name

def has_multiple_objects(img, alg="hsv"):
    """ Counts the objects in a cluster
    Parameters
    ----------
    img :obj:`ColorImg`
        mask of object cluster
    algo :obj:`str`, optional
        Algorithm to use for counting (`size` or `color` or `hsv`)
    Returns
    -------
    :boolean
        True if multiple objects, else False
    """
    if alg == "size":
        #todo- modify to use projected image
        #relies on objects being the same size
        img = img.to_binary()
        n_pixels = len(img.nonzero_pixels())
        n_objs = int(n_pixels/550)
    elif alg == "color":
        img = img.to_binary()
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
    elif alg == "hsv":
        hues, _ = get_hsv_hist(img)
        n_objs = 0
        for block_color in hues.keys():
            if hues[block_color] > cfg.HSV_MAX:
                n_objs += 2
            elif hues[block_color] > cfg.HSV_MIN:
                n_objs += 1
    else:
        raise ValueError("Unsupported algorithm specified. Use 'size' or 'color' or 'hsv'")
    if n_objs == 0:
        raise ValueError("Cluster should have at least 1 object")
    return n_objs > 1

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
    #increase range to reduce false positives
    check_range = 2
    for y in range(ymid - check_range, ymid + check_range):
        for x in range(xmid - check_range, xmid + check_range):
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
    hue_counts, hue_pixels = get_hsv_hist(color_mask)

    individual_masks = []

    #color to binary
    focus_mask = np.zeros(color_mask.data.shape[0:2])
    for i, r in enumerate(color_mask.data):
        for j, c in enumerate(r):
            if c[0] > 0 or c[1] > 0 or c[2] > 0:
                focus_mask[i][j] = 255
    focus_mask = BinaryImage(focus_mask.astype(np.uint8))

    #segment by hsv
    for block_color in hue_counts.keys():
        #same threshold values for number of objects
        if hue_counts[block_color] > cfg.SIZE_TOL:
            valid_pix = hue_pixels[block_color]
            obj_mask = focus_mask.mask_by_ind(np.array(valid_pix))
            individual_masks.append(obj_mask)

    #for each hsv block, again separate by connectivity
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
            if is_valid_grasp(grasp_top, focus_mask) and is_valid_grasp(grasp_bot, focus_mask):
                all_center_masses.append(grasp_info[0])
                all_directions.append(grasp_info[1])
                all_masks.append(grasp_info[2])
    return all_center_masses, all_directions, all_masks
