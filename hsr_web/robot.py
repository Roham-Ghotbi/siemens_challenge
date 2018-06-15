import os
import Pyro4
import time
import cPickle as pickle
import cv2
import numpy as np
import thread

sharer = Pyro4.Proxy("PYRONAME:shared.server")

#robot interface
def label_image(img):
    global sharer

    sharer.set_img(img)
    sharer.set_img_ready(True)

    print("robot waiting")
    while not sharer.is_labeled():
        pass

    label = sharer.get_label_data()
    sharer.set_labeled(False)

    return label

n = 5
robots = np.arange(5)
np.random.shuffle(robots)
# sample = np.random.choice(5, 5)
correct_i = 2

for i in robots:

    # for i in range(2):
    #     p1 = thread.start_new_thread(label_image, (frame,))
    #
    # while True:
    #     joe = 1

    print "Robot Number: " + str(correct_i)
    print "Queued Robot: " + str(i)
    frame = "data/images/frame_" + str(i) + ".png"

    label_data = label_image(frame)
    print(label_data)
    print(float(label_data['time'])/1000.0)
    print(float(label_data['latency'])/1000.0)
    time.sleep(5)
    pickle.dump(label_data, open("data/labels/" + str(i) + ".p",'wb'))
    if i == correct_i:
        print "ROBOT LABELED"
        break;
