# Siemens Challenge

## Forming a Dataset

One part of this process will be to form a dataset of images (from the HSR's
camera) of the setup. The setup involves varying the amount and location of
objects that the HSR will need to pick up.

To do this:

- Ensure that the HSR is in the appropriate setup, e.g. by moving the joysticks.
  We put some tape to ensure a bit of uniformity in the setup.

- Specifically, our HSR has joint positions:

  ```
  In [7]: whole_body.joint_positions
  Out[7]: 
  {'arm_flex_joint': -0.005953039901891888,
   'arm_lift_joint': -7.033935297334412e-07,
   'arm_roll_joint': -1.4893346753088876,
   'base_l_drive_wheel_joint': 0.22499,
   'base_r_drive_wheel_joint': 9.566874,
   'base_roll_joint': 0.403055,
   'hand_l_spring_proximal_joint': 0.711545,
   'hand_motor_joint': -0.757545,
   'hand_r_spring_proximal_joint': 0.785545,
   'head_pan_joint': 6.405774592987967e-06,
   'head_tilt_joint': -1.159990826665105,
   'wrist_flex_joint': -1.905556402348724,
   'wrist_roll_joint': 0.0005735908368245113}
  
  In [8]: whole_body.get_end_effector_pose()
  Out[8]: Pose(pos=Vector3(x=0.15819081324595927, y=-0.05033104482254261, z=0.6274079510069306), ori=Quaternion(x=0.6005137262123358, y=-0.551253479214328, z=0.4243082889074291, w=0.39429093604738635))
  ```

  This setup may mean that the HSR's gripper is *slightly* visible in the
  camera, but it shouldn't obstruct the view of the board of objects and we can
  crop the image.

- Enter the `quick_img_collection` repository. Then after going into
  `hsrb_mode`, run:

  ```
  python get_hsr_imgs.py
  ```

  you should see the HSR's camera image pop up. Press any key (except ESC) to
  save the image in a specified directory. Images are named according to index
  to avoid overriding anything. 

- Repeat the following: move stuff in the setup. Then press any key other than
  ESC to save. When you are done collecting images, press ESC. You now have a
  collection of images.

- After you are done collecting images, we recommend moving the images in a new
  directory or saving it somewhere to avoid accidental deletion.
