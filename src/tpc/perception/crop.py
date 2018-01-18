import numpy as np
import cv2
import IPython
from skimage.draw import polygon
from perception import BinaryImage

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
    #threshold to WA and flip colors
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = gray_img.copy()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #in hsv for cv2, white is (x, 0, 255)
    for rnum, r in enumerate(hsv_img):
        for cnum, c in enumerate(r):
            sat = c[1]
            val = c[2]
            if sat < 40 and val > 215:
                thresh_img[rnum][cnum] = 0
            else:
                thresh_img[rnum][cnum] = 255
    # thresh_img[np.where(gray_img > 250)] = 0
    # thresh_img[np.where(gray_img < 250)] = 255

    #remove noise by blurring and re-thresholding, then flip colors
    blur_amt = 20.0
    thresh_img = cv2.GaussianBlur(thresh_img, (21, 21), blur_amt)
    thresh_img[np.where(thresh_img > 3)] = 255
    thresh_img = 255 - thresh_img

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

    #create mask
    focus_mask = np.zeros(gray_img.shape, dtype=np.uint8)
    rr, cc = polygon(points[:,1], points[:,0])
    focus_mask[rr,cc] = 255
    if viz:
        ctrs = points.reshape((points.shape[0], 1, points.shape[1]))
        cv2.drawContours(img, [ctrs], -1, [0, 255, 0], 3)
        cv2.imwrite("crop.png", img)
        cv2.imwrite("mask.png", focus_mask)
    return BinaryImage(focus_mask.astype("uint8"))

if __name__ == "__main__":
    img = cv2.imread("frame_40_1.png")
    mask = crop_img(img, viz=True)
