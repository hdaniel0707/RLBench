from typing import List
from pyrep.objects.shape import Shape
from rlbench.const import colors
from rlbench.backend.task import Task
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition
import numpy as np


class ReachButton(Task):

    def init_task(self, reward="sparse", reward_scale=100) -> None:
        assert reward in ['sparse', 'dist', 'delta-dist']
        self._reward = reward
        self._reward_scale = reward_scale

        self.target_button = Shape('push_button_target')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        self.variation_index = index
        button_color_name, button_rgb = colors[index]
        self.target_button.set_color(button_rgb)

        # utilities for reward()
        self._init_distance = self._distance_to_goal()
        self._prev_distance = self._init_distance

        return ['push the %s button' % button_color_name,
                'push down the %s button' % button_color_name,
                'press the button with the %s base' % button_color_name,
                'press the %s button' % button_color_name]

    def variation_count(self) -> int:
        return len(colors)

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target_button.get_position())

    def _distance_to_goal(self):
        tip_pos = self.robot.arm.get_tip().get_position()
        goal_pos = self.target_button.get_position()
        return np.linalg.norm(np.array(tip_pos) - np.array(goal_pos))

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
                ob.gripper_pose[:3] - ob.task_low_dim_state)

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