# Siemens Challenge

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
