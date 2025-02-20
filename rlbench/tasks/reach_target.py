from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


class ReachTarget(Task):

    def init_task(self, reward="sparse", reward_scale=100, variations=[]) -> None:
        assert reward in ['sparse', 'dist', 'delta-dist']
        self._reward = reward
        self._reward_scale = reward_scale
        self._variations = variations if len(variations) > 1 else list(range(len(colors)))

        self.target = Shape('target')
        self.distractor0 = Shape('distractor0')
        self.distractor1 = Shape('distractor1')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])
        self._failure0_condition = DetectedCondition(
            self.robot.arm.get_tip(), ProximitySensor('failure0'))
        self._failure1_condition = DetectedCondition(
            self.robot.arm.get_tip(), ProximitySensor('failure1'))

    def init_episode(self, index: int) -> List[str]:
        self._index = index
        color_name, color_rgb = colors[index]
        print(color_name)
        self.target.set_color(color_rgb)
        # color_choices = np.random.choice(
        #     list(range(index)) + list(range(index + 1, len(colors))),
        #     size=2, replace=False)
        var_idx = self._variations.index(index)
        color_choices = np.random.choice(
            list(range(var_idx)) + list(range(var_idx + 1, len(self._variations))),
            size=2, replace=False)
        color_choices = [self._variations[i] for i in color_choices]

        self._distractor_index = color_choices
        for ob, i in zip([self.distractor0, self.distractor1], color_choices):
            name, rgb = colors[i]
            ob.set_color(rgb)
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        # utilities for reward()
        self._init_distance = self._distance_to_goal()
        self._prev_distance = self._init_distance

        return ['reach the %s target' % color_name,
                'touch the %s ball with the panda gripper' % color_name,
                'reach the %s sphere' %color_name]

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target.get_position())

    # def get_low_dim_state(self) -> np.ndarray:
    #     # One of the few tasks that have a custom low_dim_state function.
    #     if self._index == 0:
    #         return np.concatenate([
    #             np.array(self.target.get_position()),
    #             np.array(self.distractor0.get_position()),
    #             np.array(self.distractor1.get_position()),
    #         ])
    #     elif self._index == 1:
    #         return np.concatenate([
    #             np.array(self.distractor0.get_position()),
    #             np.array(self.target.get_position()),
    #             np.array(self.distractor1.get_position()),
    #         ])
    #     elif self._index == 2:
    #         return np.concatenate([
    #             np.array(self.distractor0.get_position()),
    #             np.array(self.distractor1.get_position()),
    #             np.array(self.target.get_position()),
    #         ])

    # def get_low_dim_state(self) -> np.ndarray:
    #     # One of the few tasks that have a custom low_dim_state function.
    #     return np.concatenate([
    #         np.array(self.target.get_position()),
    #         np.array(self.distractor0.get_position()),
    #         np.array(self.distractor1.get_position()),
    #     ])

    def is_static_workspace(self) -> bool:
        return True

    def _distance_to_goal(self):
        tip_pos = self.robot.arm.get_tip().get_position()
        goal_pos = self.target.get_position()
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
    def reward_from_demo(demo, reward="sparse", reward_scale=100, variations=[]):
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

    def success(self) -> Tuple[bool, bool]:
        """If the task is currently successful.

        :return: Tuple containing 2 bools: first specifies if the task is currently successful,
            second specifies if the task should terminate (either from success or from broken constraints).
        """
        if self._failure0_condition.condition_met()[0]:
            return False, True, {'reached_goal': self._distractor_index[0]}
        if self._failure1_condition.condition_met()[0]:
            return False, True, {'reached_goal': self._distractor_index[1]}

        all_met = np.all(
            [cond.condition_met()[0] for cond in self._success_conditions])
        should_terminate = all_met
        return all_met, should_terminate, {}