# Siemens Challenge

## Running the demo

The current version of the demo resides in `main/test_labeling.py`. Before running this script, follow the instructions on the hsr_web repository to start a server and run psiturk. Once the loading icon appears, you are ready to run the demo script. The robot's initial position should be sideways so its head can turn left to face the objects. 

During the demo, simply provide the bounding box and class label for the next object: the HSR will grasp it, return to its start position to get a clear view of the HSR markers, drop the object in the correct box, then return to its start position to begin the loop again. (note- in this version of the demo the HSR will only attempt grasps, so ensure the labeled object has sufficient clearance).

To restart the demo, run `endcomm.sh` to shut down all server code. Also reposition the robot correctly.

### Running in Simulation

Assume you have run the web demo described above, so you see the "Loading"
symbol in the web server.  This needs to be running beforehand.

If you want to run our demo in simulation, use Gazebo and a launch file. For
instance, on Ron's machine, in this directory:

```
/home/ron/siemens_sim/siemens_challenge/sim_world
```

we can run this `roslaunch` command if we have the right files:

```
roslaunch test_room.launch
```

And then you should see a world show up in Gazebo.

(If there's an `IPython.embed()` command in there, just do a CTRL+D and
continue, the Python code will proceed.)

#### Creating a simulated environment

You can start by copying the existing code in 

```
test_room.world
test_room.launch
```

which gives you an empty room with walls. 

Be aware of the parameters you use for your environment: to support accurate navigation and joint control, you must include
```
name="use_laser_odom" value="true"
```
in your launch file and make sure your environment is surrounded by obscures.

#### Running demo in simulator

You can launch an environment we provided by running 

```
roslaunch test_room2.launch
```

a room environment with a few cubes as grasping object and a basket as target would appear. 

Please run `sim_main/test_labeling.py` for a demo.

NOTE: This demo currently only runs on Ron's machine: I tweak frame transformation in Chris' existing code to fit in the simulator for now. More work is needed to clean this up, potentially includes a slightly modified version of hsr_core.


## Forming a Dataset

**Update: see email I wrote to the team on April 10 at around 9am. That has the
most up to date documentation.**

One part of this process will be to form a dataset of images (from the HSR's
camera) of the setup. The setup involves varying the amount and location of
objects that the HSR will need to pick up.

To do this:

- Ensure that the HSR is in the appropriate setup, e.g. by moving the joysticks.
  (EDIT: no, use the script `get_hsr_imgs.py` which should position it
  "sideways".)

- Specifically, our HSR has joint positions as computed from the `ihsrb` mode:

  ```
  In [1]: whole_body.joint_positions
  Out[1]: 
  {'arm_flex_joint': -0.005953039901891888,
   'arm_lift_joint': 1.408264702741982e-07,
   'arm_roll_joint': -1.5700016753088877,
   'base_l_drive_wheel_joint': -0.040536,
   'base_r_drive_wheel_joint': -0.014607,
   'base_roll_joint': -0.503419,
   'hand_l_spring_proximal_joint': -0.16177200000000003,
   'hand_motor_joint': 1.184772,
   'hand_r_spring_proximal_joint': -0.12877300000000003,
   'head_pan_joint': 1.500102405774593,
   'head_tilt_joint': -1.1500008266651047,
   'wrist_flex_joint': -1.569998402348724,
   'wrist_roll_joint': 0.0005735908368245113}
  ```

- Enter the `quick_img_collection` repository. Then after going into
  `hsrb_mode`, run:

  ```
  python get_hsr_imgs.py
  ```

  and repeatedly save images.  It uses IPython's embedding so you have to exit
  it, then it runs agai, etc.

- After you are done collecting images, we recommend moving the images in a new
  directory or saving it somewhere to avoid accidental deletion.
