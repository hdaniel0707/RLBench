from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition


class SlideBlockToTarget(Task):

    def init_task(self, reward="sparse", reward_scale=100) -> None:
        assert reward in ['sparse', 'dist', 'delta-dist']
        self._reward = reward
        self._reward_scale = reward_scale

        self.target_block = Shape('block')
        self.success_detector = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self.target_block, self.success_detector)])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index

        # utilities for reward()
        self._init_distance = self._distance_to_goal()
        self._prev_distance = self._init_distance

        return ['slide the block to target',
                'slide the block onto the target',
                'push the block until it is sitting on top of the target',
                'slide the block towards the green target',
                'cover the target with the block by pushing the block in its'
                ' direction']

    def variation_count(self) -> int:
        return 1

    def get_low_dim_state(self) -> np.ndarray:
        return np.concatenate([
            np.array(self.target_block.get_position()),
            np.array(self.success_detector.get_position())
        ])
        
    def _distance_to_goal(self):
        tip_pos = self.robot.arm.get_tip().get_position()
        block_pos = self.target_block.get_position()
        goal_pos = self.success_detector.get_position()
        return np.linalg.norm(np.array(tip_pos) - np.array(block_pos)) + \
               np.linalg.norm(np.array(block_pos) - np.array(goal_pos))

    def reward(self, terminate):
        if terminate:
            return 1

        if self._reward == 'sparse':
            return 0
        elif self._reward == 'dist':
            distance = self._distance_to_goal()
            # return - distance  / (100 * self._init_distance)
            return 1 / (self._reward_scale * (1 + 10 * (distance  / self._init_distance)))
        elif self._reward == 'delta-dist':
            distance = self._distance_to_goal()
            return (self._prev_distance - distance) / (self._reward_scale * self._init_distance)
        else:
            raise ValueError

    @staticmethod
    def reward_from_demo(demo, reward="sparse", reward_scale=100):
        assert reward in ['sparse', 'dist', 'delta-dist']

        def distance(ob):
            return np.linalg.norm(
                ob.gripper_pose[:3] - ob.task_low_dim_state[:3]) + np.linalg.norm(
                ob.task_low_dim_state[:3] - ob.task_low_dim_state[3:])

        if reward == 'sparse':
            return [0] * (len(demo) - 2) + [1]
        elif reward == 'dist':
            init_distance = distance(demo[0])
            return [
                1 / (reward_scale * (1 + 10 * (distance(ob)  / init_distance)))
                for ob in demo[1:-1]] + [1]
        elif reward == 'delta-dist':
            init_distance = distance(demo[0])
            return [
                (distance(demo[i]) - distance(demo[i+1])) / (reward_scale * init_distance)
                for i in range(len(demo[1:-1]))] + [1]
        else:
            raise ValueError