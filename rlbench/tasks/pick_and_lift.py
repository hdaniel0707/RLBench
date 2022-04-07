from imp import is_builtin
from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors


class PickAndLift(Task):

    def init_task(self, reward="sparse", reward_scale=100) -> None:
        assert reward in ['sparse', 'dist', 'delta-dist']
        self._reward = reward
        self._reward_scale = reward_scale

        self.target_block = Shape('pick_and_lift_target')
        self.distractors = [
            Shape('stack_blocks_distractor%d' % i)
            for i in range(2)]
        self.register_graspable_objects([self.target_block])
        self.boundary = SpawnBoundary([Shape('pick_and_lift_boundary')])
        self.success_detector = ProximitySensor('pick_and_lift_success')

        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.target_block),
            DetectedCondition(self.target_block, self.success_detector)
        ])
        # cond_set = GraspedCondition(self.robot.gripper, self.target_block)
        self.register_success_conditions([cond_set])

    def init_episode(self, index: int) -> List[str]:

        block_color_name, block_rgb = colors[index]
        self.target_block.set_color(block_rgb)

        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        for i, ob in enumerate(self.distractors):
            name, rgb = colors[color_choices[int(i)]]
            ob.set_color(rgb)

        self.boundary.clear()
        self.boundary.sample(
            self.success_detector, min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0))
        for block in [self.target_block] + self.distractors:
            self.boundary.sample(block, min_distance=0.1)

        # utilities for reward()
        self._grasped = 0
        self._init_distance = self._distance_to_goal(self._grasped)
        self._prev_distance = self._init_distance

        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def variation_count(self) -> int:
        return len(colors)

    def get_low_dim_state(self) -> np.ndarray:
        return np.concatenate([
            np.array(self.target_block.get_position()),
            np.array(self.success_detector.get_position())
        ])

    def is_static_workspace(self) -> bool:
        return True

    def _distance_to_goal(self, index):
        targets = [self.target_block, self.success_detector]
        target = targets[index]
        tip_pos = self.robot.arm.get_tip().get_position()
        goal_pos = target.get_position()
        return np.linalg.norm(np.array(tip_pos) - np.array(goal_pos))

    def reward(self, terminate):
        if terminate:
            return 1

        # if block not grasped yet
        if self._grasped == 0:
            if len([ob for ob in self.robot.gripper.get_grasped_objects()
                    if self.target_block.get_handle() == ob.get_handle()]) > 0:
                self._grasped = 1
                self._init_distance = self._distance_to_goal(self._grasped)
                self._prev_distance = self._init_distance
                return 1

        # if block grasped
        if self._grasped == 1:
            if len([ob for ob in self.robot.gripper.get_grasped_objects()
                    if self.target_block.get_handle() == ob.get_handle()]) == 0:
                self._grasped = 0
                self._init_distance = self._distance_to_goal(self._grasped)
                self._prev_distance = self._init_distance
                return -1

        if self._reward == 'sparse':
            return 0
        elif self._reward == 'dist':
            distance = self._distance_to_goal(self._grasped)
            # return - distance  / (100 * self._init_distance)
            return 1 / (self._reward_scale * (1 + 10 * (distance  / self._init_distance)))
        elif self._reward == 'delta-dist':
            distance = self._distance_to_goal(self._grasped)
            return (self._prev_distance - distance) / (self._reward_scale * self._init_distance)
        else:
            raise ValueError

    @staticmethod
    def reward_from_demo(demo, reward="sparse", reward_scale=100):
        assert reward in ['sparse', 'dist', 'delta-dist']

        def distance(ob, i, grasp_index):
            if i <= grasp_index:
                return np.linalg.norm(
                    ob.gripper_pose[:3] - ob.task_low_dim_state[:3])
            else:
                return np.linalg.norm(
                    ob.gripper_pose[:3] - ob.task_low_dim_state[3:])

        grasp_index = len(demo)
        for i, ob in enumerate(demo):
            if distance(ob, 0, grasp_index) < 0.01:
                grasp_index = i
                break
        assert grasp_index < len(demo)

        if reward == 'sparse':
            rew = [0] * (len(demo) - 2) + [1]
        elif reward == 'dist':
            init_distance = distance(demo[0], 0, grasp_index)
            rew = [
                1 / (reward_scale * (1 + 10 * (distance(demo[i+1], i, grasp_index)  / init_distance)))
                for i in range(len(demo[1:-1]))] + [1]
        elif reward == 'delta-dist':
            init_distance = distance(demo[0], 0, grasp_index)
            rew = [
                (distance(demo[i], i, grasp_index) - distance(demo[i+1], i, grasp_index)) / (
                    reward_scale * init_distance)
                for i in range(len(demo[1:-1]))] + [1]
        else:
            raise ValueError
        rew[grasp_index] = 1
        return rew