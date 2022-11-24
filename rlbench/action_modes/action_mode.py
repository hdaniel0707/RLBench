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

    def set_control_mode(self, robot):
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

    def set_control_mode(self, robot):
        self.arm_action_mode.set_control_mode(robot)



from rlbench.action_modes.arm_action_modes import \
    EndEffectorPoseViaPlanning, EndEffectorPoseViaIK, FlatEndEffectorPoseViaPlanning, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete

from pyquaternion import Quaternion
# In RLBench the quaternion: A list containing the quaternion (x,y,z,w).
# In pyquaternion it is w,x,y,z
def pyquat2rlbench(quat): # (w,x,y,z) --> (x,y,z,w)
    return np.array([quat[1], quat[2], quat[3], quat[0]])
def rlbench2pyquat(quat): # (x,y,z,w) --> (w,x,y,z)
    return Quaternion(quat[3], quat[0], quat[1], quat[2])

class Primitives(ActionMode):
    """
    """

    def __init__(self, regrasping_strategy):
        """
        """

        # utilities for all primitives
        self._default_z = .87
        self._default_quaternion = [0,1,0,0]
        self._planning_ee_action_mode = EndEffectorPoseViaPlanning(
            absolute_mode=True, frame='world',
            collision_checking=False, linear_only=False)
        
        # utilities for primitive "grasp"
        assert regrasping_strategy in ["nothing", "release", "mask"]
        self._regrasping_strategy = regrasping_strategy
        self._gripper_action_mode = Discrete(
            attach_grasped_objects=True, detach_before_open=True)
        self._ik_ee_action_mode = EndEffectorPoseViaIK(
            absolute_mode=False, frame='world', collision_checking=False)
        
        self.primitive_idx_to_name = {
            0: "move",
            1: "grasp",
        }
        self.primitive_name_to_func = dict(
            move=self._go_to,
            grasp=self._top_grasp,
        )
        self.primitive_name_to_action_idx = dict(
            move=[0, 1],
            grasp=[2],
        )
        self.max_arg_len = 3
        self.num_primitives = len(self.primitive_name_to_func)

    def _primitive_mask(self, scene):
        # grasping primitive can't be done if we already have a grasped object
        mask = np.array([1.]*self.num_primitives)
        if self._regrasping_strategy == "mask" and self._robot_is_grasping(scene):
            mask[1] = 0
        return mask

    def _go_to(self, scene: Scene, action: np.ndarray):
        q = self._default_quaternion
        q = np.array(q) / np.linalg.norm(q)
        a = np.concatenate(([action[0], action[1], self._default_z], q))
        self._planning_ee_action_mode.action(scene, a)

    def _robot_is_grasping(self, scene):
        return len(scene.robot.gripper.get_grasped_objects()) > 0

    def _top_grasp(self, scene: Scene, action: np.ndarray):
        if self._robot_is_grasping(scene):
            if self._regrasping_strategy == "release":
                self._gripper_action_mode.action(scene, np.array([1.0]))
        else:
            current_pos = scene.robot.arm.get_tip().get_position()

            # open gripper
            # self._gripper_action_mode.action(scene, np.array([1.0]))

            # set initial height and orientation
            q = rlbench2pyquat(self._default_quaternion) # (x,y,z,w) --> (w,x,y,z)
            z_rot = Quaternion(axis=[0, 0, 1], angle=action[0]) # rotate with first parameter
            q = q * z_rot
            q = pyquat2rlbench(q) # (w,x,y,z) --> (x,y,z,w)  
            q = np.array(q) / np.linalg.norm(q)
            a = np.concatenate(([current_pos[0], current_pos[1], self._default_z], q))
            self._planning_ee_action_mode.action(scene, a)

            # change orientation with first parameter
            # target_pos = action[0]
            # a = scene.robot.arm.get_joint_positions()
            # a[-1] = target_pos
            # scene.robot.arm.set_joint_positions(a, disable_dynamics=True)
            # # while np.abs(scene.robot.arm.get_joint_positions()[-1] - target_pos) > 0.1:
            # #     scene.step()
            # scene.robot.arm.set_joint_target_positions(
            #     scene.robot.arm.get_joint_positions())

            # go down
            a = np.array([0.]*7)
            a[2] = -.1 # move -0.1 in the Z axis
            a[-1] = 1 # identity quaternion
            self._ik_ee_action_mode.action(scene, a)

            # go down
            # current_pose = scene.robot.arm.get_tip().get_pose()
            # current_pose[2] = DEFAULT_Z - 0.1
            # self._planning_ee_action_mode.action(scene, current_pose)

            # grasp object
            # print("gripper before", scene.robot.gripper.get_open_amount()) # TODO problem
            self._gripper_action_mode.action(scene, np.array([0.0]))
            # print("gripper after", scene.robot.gripper.get_open_amount())

            # go up
            a = np.array([0.]*7)
            a[2] = .1 # move -0.1 in the Z axis
            a[-1] = 1 # identity quaternion
            self._ik_ee_action_mode.action(scene, a)

            if not self._robot_is_grasping(scene):
                self._gripper_action_mode.action(scene, np.array([1.0]))

    def break_apart_action(self, a):
        broken_a = {}
        for k, v in self.primitive_name_to_action_idx.items():
            broken_a[k] = a[v]
        return broken_a

    def action(self, scene: Scene, action: np.ndarray):
        primitive_idx, primitive_args = (
            np.argmax(action[: self.num_primitives]*self._primitive_mask(scene)),
            action[self.num_primitives :],
        )
        primitive_name = self.primitive_idx_to_name[primitive_idx]
        # if primitive_name != "no_op":
        primitive_name_to_action_dict = self.break_apart_action(primitive_args)
        primitive_action = primitive_name_to_action_dict[primitive_name]
        primitive = self.primitive_name_to_func[primitive_name]
        primitive(scene, primitive_action)

        # success, terminate, info = scene.task.success()
        # reward = scene.task.reward(success)
        # stats = 
        # return stats

    def action_shape(self, scene: Scene) -> tuple:
        return self.max_arg_len + self.num_primitives

    def set_control_mode(self, robot):
        self._planning_ee_action_mode.set_control_mode(robot)
        self._ik_ee_action_mode.set_control_mode(robot)


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

