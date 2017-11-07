import numpy as np
import cv2
import IPython
from connected_components import get_cluster_info
# from perception import ColorImage, BinaryImage
import matplotlib.pyplot as plt

def threshold_binarize(img, tolerance):
    #3 color channels
    dim = 3

    #create the image histogram
    split = cv2.split(img)
    hists = np.array([cv2.calcHist([split[i]], [0], None, [256], [0, 256]) for i in range(dim)])

    #find the threshold as the mode of the image
    modes = [np.argmax(hists[i]) for i in range(dim)]

    #add the tolerance to the bounds
    lb = np.array([modes[i] - tolerance for i in range(dim)])
    ub = np.array([modes[i] + tolerance for i in range(dim)])

    img = cv2.inRange(img, lb, ub)
    #switch black and white
    img[:,:,] = (255 - img[:,:,])
    return img


def run_connected_components(img, dist_tol=5, color_tol=45):
    orig_shape = img.shape
    img = np.copy(img)

    #crop image (change to detect automatically)
    lo_y, hi_y = 75, 145
    lo_x, hi_x = 240, 445
    crop = [slice(lo_y, hi_y), slice(lo_x, hi_x)]
    img = img[crop[0], crop[1]]
    cv2.imwrite("debug_imgs/crop.png", img)
    #threshhold the background and reduce noise
    img = threshold_binarize(img, color_tol)
    img = cv2.medianBlur(img, 3)

    center_masses, directions, masks = get_cluster_info(img, dist_tol)
    #transform centroids to uncropped image
    center_masses = [[c[0] + lo_y, c[1] + lo_x] for c in center_masses]
    #do the same for masks
    new_masks = []
    for m in masks:
        new_m = np.zeros(orig_shape[:2])
        new_m[lo_y:hi_y,lo_x:hi_x] = m
        new_masks.append(new_m)
    return center_masses, directions, new_masks

def draw(img, center_masses, directions):
    box_color = [255, 0, 0]
    box_size = 5
    line_color = box_color[::-1]
    line_size = 20

    for i in range(len(center_masses)):
        cm = center_masses[i]
        d = directions[i] #True if y orientation

        img[int(cm[0] - box_size):int(cm[0] + box_size),
            int(cm[1] - box_size):int(cm[1] + box_size)] = box_color
        if d:
            img[int(cm[0] - line_size):int(cm[0] + line_size),
            int(cm[1] - 1):int(cm[1] + 1)] = line_color
        else:
            img[int(cm[0] - 1):int(cm[0] + 1),
            int(cm[1] - line_size):int(cm[1] + line_size)] = line_color

    cv2.imshow('debug',img)
    cv2.waitKey(30)
    return img

#instead of color base on size, all lego bricks are the same size
def count_color_blobs(img):
    bin_span = 128
    bins = [0 for i in range((256/bin_span)**3)]
    for row in img:
        for pixel in row:
            b, g, r = pixel
            #put into bins
            b, g, r = int(b/bin_span), int(g/bin_span), int(r/bin_span)
            #calculate unique index for every combination of bins
            ind = int(b * (256/bin_span)**2 + g * (256/bin_span) + r)
            bins[ind] += 1
    return sum(np.array(bins) > 250) - 1 #ignore black

#this works as long as all legos are the same size
#can be improved with polygonal crop
def count_size_blobs(img):
    num_white = sum(sum([row > 0 for row in img]))
    return int(num_white/450)

if __name__ == "__main__":
    #number of pixels apart to be singulated
    dist_tol = 5
    #background range for threshholding the image
    color_tol = 60

    folder = "data/example_images/"

    for img_num in range(15):
        filen = "frame_40_" + str(img_num)
        img = cv2.imread(folder + filen + ".png")
        center_masses, directions, masks = run_connected_components(img, dist_tol, color_tol)

        # img_out = draw(img, center_masses, directions)
        # cv2.imwrite(folder + "out_" + filen + ".png", img_out)

        #want a list with elements of the form (cm, dir, mask, # of objects)
        #grasp if # of objects = 1, else singulate
        results = []
        for i in range(len(center_masses)):
            cm = center_masses[i]
            direction = directions[i]
            mask = masks[i]/255

            # masked_color = np.stack([mask,mask,mask],axis=-1) * img
            # cv2.imwrite(folder + "mask_" + filen + "_" + str(i) + ".png", masked_color)
            # num = count_color_blobs(masked_color)

            num = count_size_blobs(mask)
            results.append([cm, direction, mask, num])
        print("For image " + str(img_num) + " there were " + str(len(results)) +
            " clusters with " + str([r[3] for r in results]) + " objects respectively.")
