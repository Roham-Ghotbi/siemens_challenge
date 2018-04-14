from tpc.perception.groups import Group
import tpc.config.config_tpc as cfg
from tpc.perception.crop import crop_img
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')
import numpy as np 

def find_isolated_objects(bboxes):
    valid_bboxes = []
    for curr_ind in range(len(bboxes)):
        curr_bbox = bboxes[curr_ind]
        overlap = False 
        for test_ind in range(len(bboxes)):
            if curr_ind != test_ind:
                test_bbox = bboxes[test_ind]
                if curr_bbox.test_overlap(test_bbox):
                    overlap = True
                    break 
        if not overlap:
            valid_bboxes.append(curr_bbox)
    return valid_bboxes

def select_first_obj(bboxes):
    bottom_left_bbox =  min(bboxes, key = lambda x: x.xmin + x.ymin)
    return bottom_left_bbox

def format_net_bboxes(net_output, shape):
    #first filter by confidence
    scores = net_output['detection_scores'].round(2)
    classes = net_output['detection_classes']
    boxes = net_output['detection_boxes']
    num_valid = 0
    while scores[num_valid] > cfg.CONFIDENCE_THRESH:
        num_valid += 1
    filtered_output = []

    for i in range(num_valid):
        points = [boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]]
        box = Bbox(points, classes[i], scores[i])
        box.scale_from_net(shape)
        box.convert_label_from_net()
        filtered_output.append(box)

    return filtered_output

def draw_boxes(bboxes, img):
    img = np.copy(img)
    for b in bboxes:
        img = b.draw(img)
    return img

class Bbox:
    """
    class for object bounding boxes 
    """
    def __init__(self, points, label, confidence=None):
        #points should have format [xmin, ymin, xmax, ymax]
        #label should be an integer 
        self.xmin = points[0]
        self.ymin = points[1]
        self.xmax = points[2]
        self.ymax = points[3]
        self.label = label
        self.points = points 
        self.prob = confidence

    def test_overlap(self, other):
        if self.xmin > other.xmax or self.xmax < other.xmin:
            return False
        if self.ymin > other.ymax or self.ymax < other.ymin:
            return False 
        return True 

    def to_mask(self, c_img, col_img, tol=cfg.COLOR_TOL):
        obj_mask = crop_img(c_img, bycoords = [self.ymin, self.ymax, self.xmin, self.xmax])
        obj_workspace_img = col_img.mask_binary(obj_mask)
        # fg = obj_workspace_img.foreground_mask(cfg.COLOR_TOL, ignore_black=True)
        fg = obj_workspace_img.foreground_mask(tol, ignore_black=True)
        return fg, obj_workspace_img

    def scale_from_net(self, shape):
        h, w, dim = shape
        self.xmin = int(self.xmin * w)
        self.xmax = int(self.xmax * w)
        self.ymin = int(self.ymin * h)
        self.ymax = int(self.ymax * h)

    def convert_label_from_net(self):
        name = cfg.net_labels[self.label]
        self.label = cfg.labels.index(name)

    def draw(self, img):
        color = (255, 0, 0)
        new_img = np.copy(img)
        new_img[self.ymin:self.ymax, self.xmin:self.xmax] = color
        width = 3
        xlo, xhi = self.xmin + width, self.xmax - width 
        ylo, yhi = self.ymin + width, self.ymax - width
        new_img[ylo:yhi,xlo:xhi] = img[ylo:yhi,xlo:xhi]
        return new_img