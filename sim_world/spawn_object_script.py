import rospy, tf
from gazebo_msgs.srv import DeleteModel, SpawnModel, GetWorldProperties
from geometry_msgs.msg import *

import random
import numpy as np
import re

LIMIT = {'x':(-0.2, 0.2), 'y':(0.8, 1.2), 'rad':(0, 3.14)}
# QUATER = tf.transformations.quaternion_from_euler(0,0,0)
# ORIENT = Quaternion(QUATER[0], QUATER[1], QUATER[2], QUATER[3])

MODEL_PATH = "/home/zisu/simulator/siemens_challenge/sim_world/toolbox/"
# MODEL_LIST = ["screwdriver1", "screwdriver2", "screwdriver3", "screwdriver4", "screwdriver5", "tape2", "tape3", "tube1", "hammer1", "wrench1"]#, "scrap1"]

MODEL_TYPE = {"lightbulb": 1, "gear": 2, "shoe": 3, "pear": 1, "nozzle": 1, "dolphin": 1, "bowl": 3, "screwdriver": 9, "mug": 3, "tape": 2, "can": 2, "bottle": 9, "elephant": 4, "barClamp": 1, "banana": 1, "scrap": 1, "combinationWrench": 15, "hammer": 1, "openEndWrench": 3, "socketwrench": 3, "adjustableWrench": 4, "hexagonalCylinder": 3, "cylinder": 26, "rectangularCube": 47, "cat": 2, "doll": 2, "cube": 1, "milkPitcher": 1, "fish": 1, "juiceBox": 1, "heart": 1, "ellipticalCylinder": 1, "oilCan": 2, "moon": 1, "pitcher": 1, "pony": 1, "seal": 1, "shampooBottle": 2, "tube": 1, "tortoise": 1}

# MODEL_LIST = ["screwdriver1", "screwdriver2", "screwdriver3", "screwdriver4", "screwdriver5", "tape2", "tape3", "tube1", "wrench1"]
# MODEL_TYPE = ["screwdriver", "tape", "tube", "scrap", "wrench"]


def setup_delete_spawn_service():
    """This method create rosservice call for spawning objects and deleting objects"""
    print("Waiting for gazebo services...")
    # rospy.init_node("spawn_products_in_bins")
    rospy.wait_for_service("gazebo/delete_model")
    rospy.wait_for_service("gazebo/spawn_sdf_model")
    rospy.wait_for_service("gazebo/get_world_properties")
    print("Got it.")
    delete_model = rospy.ServiceProxy("gazebo/delete_model", DeleteModel)
    spawn_model = rospy.ServiceProxy("gazebo/spawn_sdf_model", SpawnModel)

    object_monitor = rospy.ServiceProxy("gazebo/get_world_properties", GetWorldProperties)

    return delete_model, spawn_model, object_monitor

def get_object_list(object_monitor):
    lst = []
    rospy.wait_for_service("gazebo/get_world_properties")
    for name in object_monitor().model_names:
        ind = re.search("\d", name)
        if ind == None:
            continue
        else:
            ind = ind.start()

        tag = name[:ind]
        if tag in MODEL_TYPE:
            lst.append(name)
    return lst

def delete_object(name, delete_model):
    rospy.wait_for_service("gazebo/delete_model")
    print("Deleting Object.")
    delete_model(name)
    return name

def clean_floor(delete_model, object_monitor):
    object_lst = get_object_list(object_monitor)
    for obj in object_lst:
        delete_object(obj, delete_model)
        rospy.sleep(0.5)

def spawn_from_uniform(n, spawn_model):
    tags = []
    for i in range(n):
        # item
        model_tag = random.choice(MODEL_TYPE.keys())
        model_index = random.choice(range(1, MODEL_TYPE[model_tag]+1))
        model_paint = random.choice([0, 1])

        with open(MODEL_PATH+model_tag+str(model_index)+"_"+str(model_paint)+"/model.sdf", "r") as f:
            object_xml = f.read()

        # pose
        pt_x = np.random.uniform(LIMIT['x'][0], LIMIT['x'][1])
        pt_y = np.random.uniform(LIMIT['y'][0], LIMIT['y'][1])
        ei = np.random.uniform(LIMIT['rad'][0], LIMIT['rad'][1])
        ej = np.random.uniform(LIMIT['rad'][0], LIMIT['rad'][1])
        ek = np.random.uniform(LIMIT['rad'][0], LIMIT['rad'][1])
        quater = tf.transformations.quaternion_from_euler(ei, ej, ek)
        orient = Quaternion(quater[0], quater[1], quater[2], quater[3])

        object_pose = Pose(Point(x=pt_x, y=pt_y, z=0.5), orient)

        # spawn
        object_name = model_tag+"_"+str(i)
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        spawn_model(object_name, object_xml, "", object_pose, "world")
        rospy.sleep(0.5)
        tags.append(model_tag+str(model_index)+"_"+str(model_paint))
    return tags

def spawn_from_gaussian(n, spawn_model):
    for i in range(n):
        # item
        model_tag = random.choice(MODEL_LIST)

        with open(MODEL_PATH+model_tag+"/model.sdf", "r") as f:
            object_xml = f.read()

        # pose
        pt_x = np.random.normal((LIMIT['x'][0]+LIMIT['x'][1])/2, (LIMIT['x'][1]-LIMIT['x'][0])/8)
        pt_y = np.random.normal((LIMIT['y'][0]+LIMIT['y'][1])/2, (LIMIT['y'][1]-LIMIT['y'][0])/8)
        ei = np.random.normal((LIMIT['rad'][0]+LIMIT['rad'][1])/2, (LIMIT['rad'][1]-LIMIT['rad'][0])/8)
        ej = np.random.normal((LIMIT['rad'][0]+LIMIT['rad'][1])/2, (LIMIT['rad'][1]-LIMIT['rad'][0])/8)
        ek = np.random.normal((LIMIT['rad'][0]+LIMIT['rad'][1])/2, (LIMIT['rad'][1]-LIMIT['rad'][0])/8)
        quater = tf.transformations.quaternion_from_euler(ei, ej, ek)
        orient = Quaternion(quater[0], quater[1], quater[2], quater[3])

        object_pose = Pose(Point(x=pt_x, y=pt_y, z=0.5), orient)

        # spawn
        object_name = model_tag+"_"+str(i)
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        spawn_model(object_name, object_xml, "", object_pose, "world")
        rospy.sleep(0.5)

if __name__ == '__main__':
    delete_model, spawn_model, object_monitor = setup_delete_spawn_service()

    print(get_object_list(object_monitor))

    # spawn_from_uniform(10, spawn_model)
    clean_floor(delete_model, object_monitor)
    # spawn_from_gaussian(10, spawn_model)

