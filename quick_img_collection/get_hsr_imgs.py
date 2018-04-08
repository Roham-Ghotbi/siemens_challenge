"""
Quick script to get _some_ pipeline going. It is assumed that the HSR head is in
a sufficiently good spot. Just run this python script once. Rearrange stuff in
the setup, save, then rearrange again, save, etc. To save, press any key other
than ESC. To exit the program, press ESC.

By Daniel Seita. Thanks to Ron Bernstein for starter code.
"""
import argparse, cv2, sys, os, rospy, threading
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

IMDIR = 'images/'
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

    HIC1 = HeadImage()
    rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw",
                     Image, HIC1, queue_size=1)
    #rospy.Subscriber("/hsrb/head_r_stereo_camera/image_raw",
    #                 Image, HIC1, queue_size=1)
    isRunning = True
    rospy.sleep(3)

    while isRunning:
        im = bridge.imgmsg_to_cv2(HIC1.get_im(), "bgr8")
        cv2.imshow('main', im)
        num = len([x for x in os.listdir(IMDIR) if 'png' in x])
        fname = IMDIR+'rgb_raw_{}.png'.format(str(num).zfill(4))
        key = cv2.waitKey(0)
        if key in ESC_KEYS:
            print("Exiting now ...")
            sys.exit()
        cv2.imwrite(fname, im)
        print("just saved: {}".format(fname))
