import os
import sys
from os.path import join, dirname, abspath, isfile
import math
import numpy as np

from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition
from rlbench.action_modes.gripper_action_modes import GripperActionMode, Discrete

from rlbench.task_environment import TaskEnvironment
#from rlbench.backend.task import Task

CURRENT_DIR = dirname(abspath(__file__))


cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                          render_mode=RenderMode.OPENGL)
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.right_shoulder_camera = cam_config
obs_config.left_shoulder_camera = cam_config
obs_config.wrist_camera = cam_config
obs_config.front_camera = cam_config


#arm_action_mode = ArmActionMode()
joint_action_mode = JointPosition()
gripper_action_mode = Discrete()

act_mode = MoveArmThenGripper(joint_action_mode,gripper_action_mode)

env = Environment(action_mode = act_mode, obs_config= obs_config,robot_setup = 'ur3baxter')

env.launch()

print(env.action_shape)
print(env.get_scene_data)

task_env = env.get_task(env._string_to_task('slide_block_to_target.py'))
print(task_env.get_name())


task_env.reset()
for i in range (1000):
    action = np.array([math.pi,math.pi,math.pi,math.pi,math.pi,math.pi,1])
    observation, reward, done, info = task_env.step(action)
    #print(i)

#
# input("Type sth...")


env.shutdown()
