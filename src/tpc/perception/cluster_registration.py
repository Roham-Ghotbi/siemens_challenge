import numpy as np
import cv2
import IPython
from connected_components import get_cluster_info
from groups import Group
from perception import ColorImage, BinaryImage
import matplotlib.pyplot as plt
import tpc.config.config_tpc as cfg
from tpc.perception.crop import crop_img

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
        display_grasps(img, center_masses, directions)

    return center_masses, directions, masks

def display_grasps(img, center_masses, directions,name="grasps"):
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
    img_data = np.copy(img.data)
    for i in range(len(center_masses)):
        cm = center_masses[i]
        d = directions[i] 

        img_data[int(cm[0] - box_size):int(cm[0] + box_size), 
            int(cm[1] - box_size):int(cm[1] + box_size)] = box_color
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
        class number from 1 to 8
    """
    counts, _ = get_hsv_hist(img, num_bins=groups)
    color = max(counts, key=counts.get)
    
    all_hues = cfg.HUE_VALUES.keys() + [-1]
    all_hues.sort()

    class_num = all_hues.index(color)
    return class_num

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
    check_range = 4 
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
    focus_mask = color_mask.to_binary()

    for block_color in hue_counts.keys():
        #same threshold values for number of objects
        if hue_counts[block_color] > cfg.HSV_MIN and hue_counts[block_color] < cfg.HSV_MAX:
            valid_pix = hue_pixels[block_color]
            obj_mask = focus_mask.mask_by_ind(np.array(valid_pix))
            individual_masks.append(obj_mask)    
    
    center_masses = []
    directions = []
    for obj_mask in individual_masks:
        g = Group(0, points = obj_mask.nonzero_pixels())
        center_mass = g.center_mass()
        direction = g.orientation()
        direction = direction/np.linalg.norm(direction)
        #matches endpoints of line in visualization
        grasp_top = center_mass + direction * cfg.LINE_SIZE/2
        grasp_bot = center_mass - direction * cfg.LINE_SIZE/2
        if is_valid_grasp(grasp_top, focus_mask) and is_valid_grasp(grasp_bot, focus_mask):
            center_masses.append(center_mass)
            directions.append(direction) 
    return center_masses, directions

if __name__ == "__main__":
    for j in range(1, 10):
        c_img = cv2.imread("debug_imgs/samples/test_" + str(j) + ".png")
        col_img = ColorImage(c_img)
        main_mask = crop_img(c_img)
        workspace_img = col_img.mask_binary(main_mask)

        center_masses, directions, masks = run_connected_components(workspace_img,
            cfg.DIST_TOL, cfg.COLOR_TOL, cfg.SIZE_TOL, viz=False)

        has_multiple = [has_multiple_objects(col_img.mask_binary(m), alg="hsv") for m in masks]
        print "has multiple objects?:", has_multiple

        grasps = []
        graspmasks = []
        viz_info = []
        for i in range(len(center_masses)):
            if not has_multiple[i]:
                cm = center_masses[i]
                di = directions[i]
                viz_info.append([cm, di])
                graspmasks.append(masks[i])
            else:
                new_cms, new_dirs = grasps_within_pile(col_img.mask_binary(masks[i]))
                for j in range(len(new_cms)):
                    new_cm = new_cms[j]
                    new_di = new_dirs[j]
                    viz_info.append([new_cm, new_di])
                    graspmasks.append(masks[i])

        if len(graspmasks) > 0:
            print "running grasps"
            display_grasps(workspace_img, [v[0] for v in viz_info], [v[1] for v in viz_info],
            name = "debug_imgs/samples/test_" + str(j) + "_h")