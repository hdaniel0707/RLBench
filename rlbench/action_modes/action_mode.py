from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode
from rlbench.action_modes.gripper_action_modes import GripperActionMode
from rlbench.backend.scene import Scene


class ActionMode(object):

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass


class MoveArmThenGripper(ActionMode):
    """The arm action is first applied, followed by the gripper action. """

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action(scene, arm_action)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))


# from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning

# class Primitives(ActionMode):
#     """
#     """

#     def __init__(self):
#         """
#         """

#         # utilities for primitive "move"
#         self._move_action_mode = EndEffectorPoseViaPlanning(
#             absolute_mode=True, frame='world',
#             collision_checking=False, linear_only=False)
        
#         # utilities for primitive "grasp"
#         self._grasp_
        
#         self.primitive_idx_to_name = {
#             0: "move",
#             1: "grasp",
#         }
#         self.primitive_name_to_func = dict(
#             move=self._move_action_mode.action,
#             grasp=self._top_grasp,
#         )
#         self.primitive_name_to_action_idx = dict(
#             move=[0, 1, 2],
#             grasp=[3, 4],
#         )
#         self.max_arg_len = 5
#         self.num_primitives = len(self.primitive_name_to_func)

#     def _top_grasp(self, scene: Scene, action: np.ndarray):
#         # change orientation with first parameter

#         # go down with second parameter

#         # grasp object

#     def break_apart_action(self, a):
#         broken_a = {}
#         for k, v in self.primitive_name_to_action_idx.items():
#             broken_a[k] = a[v]
#         return broken_a

#     def action(self, scene: Scene, action: np.ndarray):
#         primitive_idx, primitive_args = (
#             np.argmax(action[: self.num_primitives]),
#             action[self.num_primitives :],
#         )
#         primitive_name = self.primitive_idx_to_name[primitive_idx]
#         # if primitive_name != "no_op":
#         primitive_name_to_action_dict = self.break_apart_action(primitive_args)
#         primitive_action = primitive_name_to_action_dict[primitive_name]
#         primitive = self.primitive_name_to_func[primitive_name]
#         primitive(scene, primitive_action)

#         # success, terminate, info = scene.task.success()
#         # reward = scene.task.reward(success)
#         # stats = 
#         # return stats

#     def action_shape(self, scene: Scene) -> tuple:
#         return self.max_arg_len + self.num_primitives


# class Primitives(ArmActionMode):
#     """
#     """

#     def __init__(self, go_to_pose_iterations: int = 300):
#         """
#         """
#         self.primitive_idx_to_name = {
#             0: "move_delta_ee_pose",
#             1: "top_grasp",
#             2: "lift",
#             3: "drop",
#             4: "move_left",
#             5: "move_right",
#             6: "move_forward",
#             7: "move_backward",
#             # 8: "open_gripper",
#             # 9: "close_gripper",
#         }
#         self.primitive_name_to_func = dict(
#             move_delta_ee_pose=self.move_delta_ee_pose,
#             top_grasp=self.top_grasp,
#             lift=self.lift,
#             drop=self.drop,
#             move_left=self.move_left,
#             move_right=self.move_right,
#             move_forward=self.move_forward,
#             move_backward=self.move_backward,
#             # open_gripper=self.open_gripper,
#             # close_gripper=self.close_gripper,
#         )
#         self.primitive_name_to_action_idx = dict(
#             move_delta_ee_pose=[0, 1, 2],
#             top_grasp=3,
#             lift=4,
#             drop=5,
#             move_left=6,
#             move_right=7,
#             move_forward=8,
#             move_backward=9,
#             open_gripper=[],  # doesn't matter
#             close_gripper=[],  # doesn't matter
#         )
#         self.max_arg_len = 10
#         self.num_primitives = len(self.primitive_name_to_func)
#         self.go_to_pose_iterations = go_to_pose_iterations

#     def act(self, a):
#         primitive_idx, primitive_args = (
#             np.argmax(a[: self.num_primitives]),
#             a[self.num_primitives :],
#         )
#         primitive_name = self.primitive_idx_to_name[primitive_idx]
#         # if primitive_name != "no_op":
#         primitive_name_to_action_dict = self.break_apart_action(primitive_args)
#         primitive_action = primitive_name_to_action_dict[primitive_name]
#         primitive = self.primitive_name_to_func[primitive_name]
#         stats = primitive(primitive_action)
#         return stats

#     #robosuite
#     def move_delta_ee_pose(self, pose):
#         stats = self.goto_pose(self._eef_xpos + pose, grasp=True)
#         return stats

#     def move_left(self, x_dist):
#         x_dist = np.maximum(x_dist, 0.0)
#         stats = self.goto_pose(self._eef_xpos + np.array([0, -x_dist, 0.0]), grasp=True)
#         return stats

#      def goto_pose(self, pose, grasp=False):
#         total_reward, total_success = 0, 0
#         prev_delta = np.zeros_like(pose)
#         pose = np.clip(pose, self.workspace_low, self.workspace_high)
#         for _ in range(self.go_to_pose_iterations):
#             delta = pose - self._eef_xpos
#             if grasp:
#                 gripper = 1
#             else:
#                 gripper = -1
#             action = [*delta, 0, 0, 0, gripper]
#             if np.allclose(delta - prev_delta, 1e-4):
#                 break
#             policy_step = True
#             prev_delta = delta
#             for i in range(int(self.control_timestep / self.model_timestep)):
#                 self.sim.forward()
#                 self._pre_action(action, policy_step)
#                 self.sim.step()
#                 policy_step = False
#                 self.call_render_every_step()
#                 self.cur_time += self.control_timestep
#                 r = self.reward(action)
#                 total_reward += r
#                 total_success += float(self._check_success())
#         return np.array((total_reward, total_success))

#     #metaworld
#     def move_delta_ee_pose(self, pose):
#         stats = self.goto_pose(self.get_endeff_pos() + pose)
#         return stats

#     def move_left(self, x_dist):
#         x_dist = np.maximum(x_dist, 0.0)
#         stats = self.goto_pose(
#             self.get_endeff_pos() + np.array([-x_dist, 0.0, 0.0]), grasp=True
#         )
#         return stats

#     def goto_pose(self, pose, grasp=True):
#         total_reward, total_success = 0, 0
#         for _ in range(300):
#             delta = pose - self.get_endeff_pos()
#             gripper = self.sim.data.qpos[8:10]
#             if grasp:
#                 gripper = [1, -1]
#             self._set_action(
#                 np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, 0.0, *gripper])
#             )
#             self.data.set_mocap_quat("mocap", np.array([1, 0, 1, 0]))
#             self.sim.step()
#             self.call_render_every_step()
#             r, info = self.evaluate_state(self._get_obs(), [*delta, 0])
#             total_reward += r
#             total_success += info["success"]
#         return np.array((total_reward, total_success))
    
#     def action(self, scene: Scene, action: np.ndarray):
#         assert_action_shape(action, self.action_shape(scene))
#         scene.robot.arm.set_joint_target_velocities(action)
#         scene.step()
#         # TODO i need to call success, terminate, info = scene.task.success() to collect all rewards
#         scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

#     def action_shape(self, scene: Scene) -> tuple:
#         return self.max_arg_len + self.num_primitives

