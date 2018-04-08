"""
See the documentation in README.md for details.
By Daniel Seita. Thanks to Ron Bernstein for starter code.
"""
import argparse, cv2, sys, os, rospy, threading
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

IMDIR_RGB = 'images_rgb/'
IMDIR_DEPTH = 'images_depth/'
ESC_KEYS = [27, 1048603]


def OnShutdown_callback():
    global isRunning
    isRunning = False


class HeadImage(object):
    def __init__(self):
        # event that will block until the info is received
        self._event = threading.Event()
        # attribute for storing the rx'd message
        self._msg = None

    def __call__(self, headImage):
        # Uses __call__ so the object itself acts as the callback
        # save the data, trigger the event
        self._msg = headImage
        self._event.set()

    def get_im(self, timeout=None):
        """Blocks until the data is rx'd with optional timeout
        Returns the received message
        """
        # self._event.wait(timeout)
        return self._msg


if __name__ == '__main__':
    global isRunning
    bridge = CvBridge()
    rospy.init_node('main', anonymous=True)
    rospy.on_shutdown(OnShutdown_callback)

    # TODO `depth_registered` topic doesn't produce readable output, also
    # `depth` topic doesn't seem to be published.
    hic_list = [HeadImage(), HeadImage()]
    rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw",
                     Image, hic_list[0], queue_size=1)
    rospy.Subscriber("/hsrb/head_rgbd_sensor/depth_registered/image_raw",
                     Image, hic_list[1], queue_size=1)
    isRunning = True
    rospy.sleep(5)

    while isRunning:
        # Obtain the images. For depth, cant use 'bgr8' I think.
        data = [hic.get_im() for hic in hic_list]
        img1 = bridge.imgmsg_to_cv2(data[0], "bgr8")
        img2 = bridge.imgmsg_to_cv2(data[1])
        print("img1.shape: {}, img2.shape: {}".format(img1.shape, img2.shape))

        # Show images, get file name, etc. Depth doesn't work yet.
        cv2.imshow('rgb/image_raw', img1)
        #cv2.imshow('depth/image_raw', img2) # Seems to be just black
        num = len([x for x in os.listdir(IMDIR_RGB) if 'png' in x])
        fname1 = IMDIR_RGB+'rgb_raw_{}.png'.format(str(num).zfill(4))
        #fname2 = IMDIR_DEPTH+'depth_raw_{}.png'.format(str(num).zfill(4))

        # Save images or exit if needed. Again, depth doesn't work yet.
        key = cv2.waitKey(0)
        if key in ESC_KEYS:
            print("Exiting now ...")
            sys.exit()
        cv2.imwrite(fname1, img1)
        print("saved: {}".format(fname1))
        #cv2.imwrite(fname2, img2)
        #print("saved: {}".format(fname2))
