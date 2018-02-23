# To run this test script run python src/tpc/perception/singulation.py
# from the root of the repo

import matplotlib.pyplot as plt
import skimage as sk
from skimage.draw import polygon
import sys
from perception import ColorImage, BinaryImage
import numpy as np
from sklearn.decomposition import PCA
import cv2
import IPython
import tpc.config.config_tpc as cfg

class Singulation():
    def __init__(self, img, focus_mask, workspace_img, obj_masks):
        """ 
        Parameters
        ----------
        img :obj:`ColorImage`
            original image
        focus_mask :obj:`BinaryImage`
            crop of workspace
        obj_masks :list:obj:`BinaryImage`
            list of crops of object clusters
        alg :string
            `border`: push along border then to free space
            `free`: push along border with bias to free space
        """
        self.img = img.copy() 
        self.focus_mask = focus_mask.copy()
        obj_masks = [o.copy() for o in obj_masks]
        self.workspace_img = workspace_img

        #singulate smallest pile first
        obj_masks.sort(key=lambda m:len(m.nonzero_pixels()))
        self.obj_mask = obj_masks[0]
        self.other_obj_masks = obj_masks[1:]

        #run computations in advance
        self.compute_singulation()

    def display_border(self, border):
        """ helper to display the border between the segments

        Parameters
        ----------
        border : :obj:`numpy.ndarray`
            nx2 array of n border pixels
        """
        plt.imshow(self.img.data)
        plt.axis('off')
        plt.plot(border[:,1], border[:,0], 'o')
        plt.show()

    def display_segments(self, segmented):
        """ helper to display the border between the segments

        Parameters
        ----------
        segmented : :obj:`SegmentedImage`
            the segments to be displayed
        """
        for i in range(segmented.num_segments):
            brick = self.img.mask_binary(segmented.segment_mask(i))
            plt.imshow(brick.data)
            plt.show()

    def get_border(self):
        """ Splits the object pile into two segments, and
        returns the pixels that border the two segments to be
        used as the pushing direction

        Returns
        -------
        :obj:`numpy.ndarray`
            nx2 array of n border pixels
        """
        bricks = self.img.mask_binary(self.obj_mask)
        segmented = bricks.segment_kmeans(.1, 2)
        border = segmented.border_pixels()

        # display_border(border)
        # display_segments(segmented)

        #rare case- 2 clusters aren't touching
        if len(border) == 0:
            print("no boundary points")
            border = self.obj_mask.nonzero_pixels()

        return border

    def get_goal_pixel(self):
        """ Finds the goal pixel, which is the point
        in the workspace furthest from all walls and
        other object clusters

        Returns
        -------
        :obj:`numpy.ndarray`
            1x2 array of the goal pixel
        """
        occupied_space = self.focus_mask.inverse()
        for i in range(len(self.other_obj_masks)):
            occupied_space += self.other_obj_masks[i]
        goal_pixel = occupied_space.most_free_pixel()
        return goal_pixel

    def get_direction(self, border, goal_pixel):
        """ Finds the direction in which to push the
        pile to separate it the most using the direction
        of the border pixels

        Parameters
        ----------
        border :obj:`numpy.ndarray`
            nx2 array of n border pixels
        goal_pixel :obj:`numpy.ndarray`
            1x2 array of the goal pixel
        Returns
        -------
        :obj:`numpy.ndarray`
            1x2 vector pointing in the direction in which the
            robot should singulate
        :obj: int
            distance the robot should push to go through the
            entire pile
        """
        #optimal pushing direction
        pca = PCA(2)
        pca.fit(border)
        border_dir = pca.components_[0]
        border_dir /= np.linalg.norm(border_dir)
        max_distance = pca.explained_variance_[0]
        distance = .4*max_distance

        push_dir = border_dir

        return push_dir, distance

    def get_border_endpoints(self, mean, direction):
        """ Finds the endpoints of the first push along the borer

        Parameters
        ----------
        mean :obj:`numpy.ndarray`
            1x2 vector, mean of border pixels
        direction :obj:`numpy.ndarray`
            1x2 vector pointing in the direction in which the
            robot should singulate
        """
        start_p = self.obj_mask.closest_zero_pixel(mean, -1*direction)
        end_p = self.obj_mask.closest_zero_pixel(mean, direction)

        #should be closer to goal pixel after border push than before it
        if np.linalg.norm(goal_p - start_p) < np.linalg.norm(goal_p - end_p):
            start_p, end_p = end_p, start_p

        #small adjustments (empirically does better)
        start_p += (start_p - end_p)/np.linalg.norm(start_p-end_p) * cfg.SINGULATE_START_FACTOR
        end_p = start_p * (1-cfg.SINGULATE_END_FACTOR) + end_p * cfg.SINGULATE_END_FACTOR

        return start_p, end_p

    def get_goal_waypoint(self, mean):
         """ Finds the endpoints of the first push along the borer

        Parameters
        ----------
        mean :obj:`numpy.ndarray`
            1x2 vector, mean of border pixels
        """
        goal_dir = self.goal_p - mean
        goal_dir = goal_dir / np.linalg.norm(goal_dir)
        towards_goal = self.obj_mask.closest_zero_pixel(mean, goal_dir, w=40)

        #don't want to push too far if goal pixel is inside object cluster
        closer_goal = min([self.goal_p, towards_goal], key = lambda p: np.linalg.norm(p - mean))

        return closer_goal

    def compute_singulation(self):
        """ Finds the direction in which the robot should push
        the pile to separate it the most, and the gripper angle
        which will keep objects farthest from other piles
        """
        border = self.get_border()
        mean = np.mean(border, axis=0)

        self.goal_p = self.get_goal_pixel()

        direction, _ = self.get_direction(border, self.goal_p)

        start_p, end_p = self.get_border_endpoints(mean, direction)
        towards_goal_p = self.get_goal_waypoint()
        self.waypoints = [start_p, end_p, towards_goal_p]

        self.gripper_angle = 0

    def get_singulation(self):
        """ 
        Returns
        -------
        :list:obj:`numpy.ndarray`
            list of 1x2 vectors representing the waypoints of the singulation
        float
            angle of gripper, aligned towards free space
        :obj: `numpy.ndarray`
            1x2 vector representing the goal pixel
        """

        return self.waypoints, self.gripper_angle, self.goal_p

    def display_singulation(self, name="debug_imgs/singulate"):
        """
        saves visualization of singulation trajectories
        """
        plt.figure()
        ax = plt.axes()
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]
            end = self.waypoints[i+1]
            ax.arrow(
                start[1], start[0],
                end[1] - start[1], end[0] - start[0],
                head_width = 10, head_length = 10
            )

        #BGR to RGB
        image = self.workspace_img
        rgb = np.fliplr(image.data.reshape(-1,3)).reshape(image.data.shape)
        plt.imshow(rgb)

        plt.plot(self.goal_p[1], self.goal_p[0], 'bo')
        plt.axis('off')
        plt.savefig(name + ".png")  
        if cfg.QUERY:
            plt.show()