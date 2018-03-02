import numpy as np
import IPython
import cv2
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.misc import imresize
from perception import ColorImage, BinaryImage
import tpc.config.config_tpc as cfg

class Group:
    """
    used to group pixels belonging to the same object clusters
    author: Chris Powers
    """
    def __init__(self, label, orig_shape, scaled_shape, points = None):
        self.ndim = 2
        if points == None:
            self.points = []
            self.low_coords = [-1 for d in range(self.ndim)]
            self.high_coords = [-1 for d in range(self.ndim)]
        else:
            self.points = points
            self.low_coords = [min(self.points, key=lambda x:x[d])[d] for d in range(self.ndim)]
            self.high_coords = [max(self.points, key=lambda x:x[d])[d] for d in range(self.ndim)]
        self.area = len(self.points)
        self.label = label

        self.orig_shape = orig_shape
        self.scaled_shape = scaled_shape

        self.cm = None 
        self.dir = None 
        self.mask = None 

    @staticmethod
    def checkDim(p1, p2):
        if len(p1) != len(p2) or len(p1) != self.ndim:
            raise ValueError("Cannot operate on points with different dimensions!")

    @staticmethod
    def squaredDist(p1, p2):
        """
        Euclidean l^2 norm
        """
        checkDim(p1, p2)
        return sum([(p1[dim] - p2[dim])**2 for dim in range(self.ndim)])

    @staticmethod
    def is_adj(p1, p2):
        """
        Checks for adjacency using 8-connectivity
        """
        checkDim(p1, p2)
        #check adjacency straight or on either diagonal for each dimension
        return all(p1[dim] - 1 <= p2[dim] <= p1[dim] + 1 for dim in range(self.ndim))

    @staticmethod
    def nearby(g1, g2, dist_tol):
        """
        Uses nearest neigbors to find if two groups are close enough to be merged
        """
        #see for reference: http://stackoverflow.com/questions/12923586/nearest-neighbor-search-python
        cluster_data = np.array(g1.points)
        query_data = np.array(g2.points)
        tree = cKDTree(cluster_data, leafsize = 8)

        for p in query_data:
            #format of result is (shortest dist, index in cluster_data of closest points)
            #if there are no points within tol, format is (inf, maxIndex + 1)
            result = tree.query(p, k = 1, distance_upper_bound = dist_tol + 1)
            #short circuits- just need one point within dist_tol
            if result[0] != float('inf'):
                return True
        return False

    def __lt__(self, other):
        return self.area < other.area

    def add(self, p):
        """
        Insert a new point into the group
        """
        n = len(self.points) * 1.0
        self.points.append(p)
        self.area += 1

        #update the bounding points
        for dim in range(self.ndim):
            if self.high_coords[dim] == -1 or self.high_coords[dim] < p[dim]:
                self.high_coords[dim] = p[dim]
            if self.low_coords[dim] == -1 or self.low_coords[dim] > p[dim]:
                self.low_coords[dim] = p[dim]

    def merge(self, other):
        """
        Merge one group into the other (other group can be discarded)
        """
        for dim in range(self.ndim):
            self.low_coords[dim] = min(self.low_coords[dim], other.low_coords[dim])
            self.high_coords[dim] = max(self.high_coords[dim], other.high_coords[dim])
        self.area = self.area + other.area
        n, m = len(self.points), len(other.points)
        self.points = self.points + other.points

    def get_bounds(self):
        """
        Returns [ly, hy, lx, hx], the bounding coordinates of the group
        """
        coords = []
        for dim in range(self.ndim):
            coords.append(self.low_coords[dim])
            coords.append(self.high_coords[dim])
        return coords

    def updateLabel(self, newLabel):
        self.label = newLabel

    def compute_info(self):
        self.compute_cm()
        self.compute_dir()
        self.compute_mask()

    def compute_mask(self):
        img = np.zeros(self.scaled_shape)
        for p in self.points:
            img[p[0]][p[1]] = 1

        self.mask = BinaryImage(imresize(img,self.orig_shape))

    def compute_cm(self):
        mean = np.mean(self.points, axis=0)
        self.cm = map(lambda x: x * cfg.SCALE_FACTOR, mean)

    def compute_dir(self):
        """
        Major axis vector
        """
        pca = PCA(n_components=2)
        pca.fit(self.points)
        # axis = pca.components_[1]
        #perpendicular to 1st component works better than second component
        axis = pca.components_[0]
        axis = [axis[1], -1*axis[0]]
        self.dir = axis/np.linalg.norm(axis)