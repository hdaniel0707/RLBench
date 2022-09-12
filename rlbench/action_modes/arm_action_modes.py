from abc import abstractmethod

import numpy as np
from pyquaternion import Quaternion
from pyrep.const import ConfigurationPathAlgorithms as Algos, ObjectType
from pyrep.errors import ConfigurationPathError, IKError

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.const import SUPPORTED_ROBOTS


def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


def assert_unit_quaternion(quat):
    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise InvalidActionError('Action contained non unit quaternion!')


def calculate_delta_pose(robot: Robot, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
    new_rot = Quaternion(
        a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(new_rot)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return pose


class ArmActionMode(object):

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(True)


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


class JointVelocity(ArmActionMode):
    """Control the joint velocities of the arm.

    Similar to the action space in many continious control OpenAI Gym envs.
    """
    
    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        scene.robot.arm.set_joint_target_velocities(action)
        scene.step()
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(False)
        robot.arm.set_motor_locked_at_zero_velocity(True)


class JointPosition(ArmActionMode):
    """Control the target joint positions (absolute or delta) of the arm.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.
    """

    def __init__(self, absolute_mode: bool = True):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
        """
        self._absolute_mode = absolute_mode

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        a = action if self._absolute_mode else np.array(
            scene.robot.arm.get_joint_positions()) + action
        scene.robot.arm.set_joint_target_positions(a)
        scene.step()
        scene.robot.arm.set_joint_target_positions(
            scene.robot.arm.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],


class JointTorque(ArmActionMode):
    """Control the joint torques of the arm.
    """

    TORQUE_MAX_VEL = 9999

    def _torque_action(self, robot, action):
        tml = JointTorque.TORQUE_MAX_VEL
        robot.arm.set_joint_target_velocities(
            [(tml if t < 0 else -tml) for t in action])
        robot.arm.set_joint_forces(np.abs(action))

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        self._torque_action(scene.robot, action)
        scene.step()
        self._torque_action(scene.robot, scene.robot.arm.get_joint_forces())
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(False)


class EndEffectorPoseViaPlanning(ArmActionMode):
    """High-level action where target pose is given and reached via planning.

    Given a target pose, a linear path is first planned (via IK). If that fails,
    sample-based planning will be used. The decision to apply collision
    checking is a crucial trade off! With collision checking enabled, you
    are guaranteed collision free paths, but this may not be applicable for task
    that do require some collision. E.g. using this mode on pushing object will
    mean that the generated path will actively avoid not pushing the object.

    Note that path planning can be slow, often taking a few seconds in the worst
    case.

    This was the action mode used in:
    James, Stephen, and Andrew J. Davison. "Q-attention: Enabling Efficient
    Learning for Vision-based Robotic Manipulation."
    arXiv preprint arXiv:2105.14829 (2021).
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False,
                 linear_only: bool = False):
        """
        If collision check is enbled, and an object is grasped, then we

        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either 'world' or 'end effector'.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._robot_shapes = None
        self._linear_only = linear_only
        if frame not in ['world', 'end effector']:
            raise ValueError("Expected frame to one of: 'world, 'end effector'")

    def _quick_boundary_check(self, scene: Scene, action: np.ndarray):
        pos_to_check = action[:3]
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()
        if relative_to is not None:
            scene.target_workspace_check.set_position(pos_to_check, relative_to)
            pos_to_check = scene.target_workspace_check.get_position()
        if not scene.check_target_in_workspace(pos_to_check):
            raise InvalidActionError('A path could not be found because the '
                                     'target is outside of workspace.')

    def _pose_in_end_effector_frame(self, robot: Robot, action: np.ndarray):
        a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
        x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
        new_rot = Quaternion(
            a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
        qw, qx, qy, qz = list(new_rot)
        pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
        return pose

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()
        self._quick_boundary_check(scene, action)

        colliding_shapes = []
        if self._collision_checking:
            if self._robot_shapes is None:
                self._robot_shapes = scene.robot.arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # First check if we are colliding with anything
            colliding = scene.robot.arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = scene.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            scene.robot.arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]

        try:
            if self._linear_only:
                path = scene.robot.arm.get_linear_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=not self._collision_checking,
                    relative_to=relative_to,
                )
            else:    
                path = scene.robot.arm.get_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=not self._collision_checking,
                    relative_to=relative_to,
                    trials=100,
                    max_configs=10,
                    max_time_ms=10,
                    trials_per_goal=5,
                    algorithm=Algos.RRTConnect
                )
            [s.set_collidable(True) for s in colliding_shapes]
        except ConfigurationPathError as e:
            [s.set_collidable(True) for s in colliding_shapes]
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e
        done = False
        while not done:
            done = path.step()
            scene.step()
            success, terminate, info = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break

    def action_shape(self, scene: Scene) -> tuple:
        return 7,

class FlatEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):

    DEFAULT_Z = .77
    DEFAULT_QUATERNION = [0,1,0,0]

    def action(self, scene: Scene, action: np.ndarray):

        q = self.DEFAULT_QUATERNION
        q = np.array(q) / np.linalg.norm(q)

        action = np.concatenate((action,[self.DEFAULT_Z], q))

        return super().action(scene, action)

    def action_shape(self, scene: Scene) -> tuple:
        return 2,


class EndEffectorPoseViaIK(ArmActionMode):
    """High-level action where target pose is given and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either 'world' or 'end effector'.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        if frame not in ['world', 'end effector']:
            raise ValueError(
                "Expected frame to one of: 'world, 'end effector'")

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()

        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e
        done = False
        prev_values = None
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done:
            scene.step()
            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def action_shape(self, scene: Scene) -> tuple:
        return 7,
