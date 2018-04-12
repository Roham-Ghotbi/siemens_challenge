from tpc.perception.groups import Group
import tpc.config.config_tpc as cfg
from tpc.perception.crop import crop_img
import importlib
img = importlib.import_module(cfg.IMG_MODULE)
ColorImage = getattr(img, 'ColorImage')
BinaryImage = getattr(img, 'BinaryImage')

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

class Bbox:
    """
    class for object bounding boxes 
    """
    def __init__(self, points, label):
        #points should have format [xmin, ymin, xmax, ymax]
        #label should be an integer 
        self.xmin = points[0]
        self.ymin = points[1]
        self.xmax = points[2]
        self.ymax = points[3]
        self.label = label
        self.points = points 

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