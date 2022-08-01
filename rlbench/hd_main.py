import os
import sys
from os.path import join, dirname, abspath, isfile
import math
import numpy as np

from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, JointPosition, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import GripperActionMode, Discrete

from rlbench.task_environment import TaskEnvironment
from rlbench.backend.observation import Observation
#from rlbench.backend.task import Task

CURRENT_DIR = dirname(abspath(__file__))


cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                          render_mode=RenderMode.OPENGL)
obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.joint_positions = True
obs_config.right_shoulder_camera = cam_config
obs_config.left_shoulder_camera = cam_config
obs_config.wrist_camera = cam_config
obs_config.front_camera = cam_config


#arm_action_mode = ArmActionMode()
joint_action_mode = JointPosition()
#joint_action_mode = EndEffectorPoseViaPlanning()
gripper_action_mode = Discrete()

act_mode = MoveArmThenGripper(joint_action_mode,gripper_action_mode)

env = Environment(action_mode = act_mode, obs_config= obs_config,robot_setup = 'ur3baxter')

env.launch()

print(env.action_shape)
print(env.get_scene_data)

task_env = env.get_task(env._string_to_task('slide_block_to_target.py'))
print(task_env.get_name())

task_env.reset()
#print(task_env.get_observation().get_low_dim_data())

joint0_1 = np.linspace(0, -2.5, num=50)
joint0_2 = np.linspace(-2.5, +2.5, num=100)
joint0_3 = np.linspace(+2.5, 0, num=50)
joint0 = np.concatenate((joint0_1,joint0_2,joint0_3))
print(joint0)

task_env.reset()

for _ in range(1):
    for i in range (200):

        action = np.array([joint0[i], -0.3, -0.164, -0.22, 1.884, -0.314,1])
        #action = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 1.0, 0.0, 1])
        observation, reward, done, info = task_env.step(action)

        print("Iter: ",i)
        print(observation.get_low_dim_data())
        print(reward)
        print(done)
        print(info)
        print(observation.front_rgb)
        print(observation.wrist_rgb)

env.shutdown()
