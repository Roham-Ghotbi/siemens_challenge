import numpy as np
import cv2
import IPython
from skimage.draw import polygon
from perception import BinaryImage
import tpc.config.config_tpc as cfg
from tpc.perception.cluster_registration import draw_point

def crop_img(img, viz=False):
    """ Crops image polygonally
    to white working area (WA)
    Parameters
    ----------
    img : :obj:`numpy.ndarray`
    viz : boolean
        if true, displays the crop
        polygon on the original image
    Returns
    -------
    :obj:`BinaryImage`
        Image with only WA in white
    """
    img = np.copy(img)

    #threshold to WA and flip colors
    mask_shape = (img.shape[0], img.shape[1])
    thresh_img = np.zeros(mask_shape).astype(np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for rnum, r in enumerate(hsv_img):
        for cnum, c in enumerate(r):
            sat = c[1]
            is_white = sat < 0.1 * cfg.SAT_RANGE
            if is_white:
                thresh_img[rnum][cnum] = 0
            else:
                thresh_img[rnum][cnum] = 255
    # return BinaryImage(thresh_img.astype(np.uint8))

    #remove noise by blurring and re-thresholding, then flip colors
    blur_amt = 20.0
    thresh_img = cv2.GaussianBlur(thresh_img, (21, 21), blur_amt)
    thresh_img[np.where(thresh_img > 3)] = 255
    thresh_img = 255 - thresh_img

    # return BinaryImage(thresh_img.astype(np.uint8))

    #get bounding points for largest contour
    (cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour=sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    #map bounding points to vertices
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    points = np.array([p[0] for p in approx])

    #get 4 corner vertices
    lower_left = min(points, key=lambda p:p[0] + p[1])
    upper_right = min(points, key=lambda p:-p[0] - p[1])
    lower_right = min(points, key=lambda p:-p[0] + p[1])
    upper_left = min(points, key=lambda p: p[0] - p[1])
    points = np.array([lower_left, lower_right, upper_right, upper_left])

    #blurring makes region smaller, so re-enlarge
    mean = np.mean(points, axis=0).astype("int32")
    points -= mean
    norms = np.apply_along_axis(np.linalg.norm, 1, points)
    points = (points * (1 + blur_amt/1.4/norms.reshape(-1,1))).astype("int32")
    points += mean

    #compute rectangle by stretching to outer points
    low_y = min(lower_left[1], lower_right[1])
    low_x = min(lower_left[0], upper_left[0])
    high_x = max(lower_right[0], upper_right[0])
    high_y = max(upper_left[1], upper_right[1])

    #compute arc (invisible to camera)
    upper_left = [low_x, high_y]
    mid_left = [low_x + 10, (low_y + high_y)/2.0]
    arc_left = [low_x * 3.0/4.0 + high_x * 1.0/4.0, low_y * 3.0/4.0 + high_y * 1.0/4.0]
    arc_mid = [(low_x + high_x)/2.0, low_y * 5.0/6.0 + high_y * 1.0/6.0]
    arc_right = [low_x * 1.0/4.0 + high_x * 3.0/4.0, low_y * 3.0/4.0 + high_y * 1.0/4.0]
    mid_right = [high_x - 10, (low_y + high_y)/2.0]
    upper_right = [high_x, high_y]

    points = np.array([upper_left, mid_left, arc_left, arc_mid, arc_right, mid_right, upper_right])

    # for p in points:
    #     img = draw_point(img, p[::-1])

    #create mask
    focus_mask = np.zeros(mask_shape, dtype=np.uint8)
    rr, cc = polygon(points[:,1], points[:,0])
    focus_mask[rr,cc] = 255
    if viz:
        ctrs = points.reshape((points.shape[0], 1, points.shape[1]))
        cv2.drawContours(img, [ctrs], -1, [0, 255, 0], 3)
        cv2.imwrite("crop.png", img)
        cv2.imwrite("mask.png", focus_mask)
    return BinaryImage(focus_mask.astype("uint8"))

if __name__ == "__main__":
    img = cv2.imread("sample.png")
    crop_img(img, viz=True)
