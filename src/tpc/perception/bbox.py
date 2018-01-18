from tpc.perception.groups import Group
import numpy as np
from perception import ColorImage, BinaryImage

def bbox_to_mask(bbox, c_img):
    loX, loY, hiX, hiY = bbox
    bin_img = np.zeros(c_img.shape[0:2])
    #don't include workspace points
    for x in range(loX, hiX):
        for y in range(loY, hiY):
            r, g, b = c_img[y][x]
            if r < 240 or g < 240 or b < 240:
                bin_img[y][x] = 255

    return BinaryImage(bin_img.astype(np.uint8))

def bbox_to_grasp(self, bbox, c_img, d_img):
    '''
    Computes center of mass and direction of grasp using bbox
    '''
    loX, loY, hiX, hiY = bbox

    #don't include workspace points
    dpoints = []
    for x in range(loX, hiX):
        for y in range(loY, hiY):
            r, g, b = c_img[y][x]
            if r < 240 or g < 240 or b < 240:
                dpoints.append([y, x])

    g = Group(1, points=dpoints)
    direction = g.orientation()
    center_mass = g.center_mass()

    #use x from color image
    x_center_points = [d for d in dpoints if abs(d[1] - center_mass[1]) < 4]

    #compute y using depth image
    dvals = [d_img[d[0], d[1]] for d in x_center_points]
    depth_vals = list(np.copy(dvals))
    depth_vals.sort()
    #use median to ignore depth noise
    middle_depth = depth_vals[len(depth_vals)/2]
    closest_ind = (np.abs(dvals - middle_depth)).argmin()
    closest_point = x_center_points[closest_ind]

    return closest_point, direction
