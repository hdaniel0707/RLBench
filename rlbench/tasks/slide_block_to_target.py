from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition


class SlideBlockToTarget(Task):

    def init_task(self) -> None:
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

    def reward(self, terminate):
        if terminate:
            return 1

        return 0
        # else
        distance = self._distance_to_goal()
        reward = (self._prev_distance - distance) / self._init_distance
        self._prev_distance = distance
        return reward

    @staticmethod
    def reward_from_demo(demo): # TODO integrate with reward
        return [0] * (len(demo) - 2) + [1]

    def get_low_dim_state(self) -> np.ndarray:
        return np.concatenate([
            np.array(self.target_block.get_position()),
            np.array(self.success_detector.get_position())
        ])
        
    def _distance_to_goal(self):
        tip_pos = self.target_block.get_position()
        goal_pos = self.success_detector.get_position()
        return np.linalg.norm(np.array(tip_pos) - np.array(goal_pos))