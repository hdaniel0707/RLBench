from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


class ReachTargetNoDistractors(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        # self.distractor0 = Shape('distractor0')
        # self.distractor1 = Shape('distractor1')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        # color_choices = np.random.choice(
        #     list(range(index)) + list(range(index + 1, len(colors))),
        #     size=2, replace=False)
        # for ob, i in zip([self.distractor0, self.distractor1], color_choices):
        #     name, rgb = colors[i]
        #     ob.set_color(rgb)
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target]:#, self.distractor0, self.distractor1]:
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

    def is_static_workspace(self) -> bool:
        return True

    # def set_reward(self, reward: str):
    #     assert reward in ['sparse', 'dist', 'delta-dist']
    #     self._reward = reward

    def reward(self, terminate):
        if terminate:
            return 1
            
        # return 0    
        distance = self._distance_to_goal()
        reward1 = (self._prev_distance - distance) / (100 * self._init_distance)
        # reward2 = - distance  / self._init_distance
        # reward3 = 1 / (1 + 10 * (distance  / self._init_distance))
        self._prev_distance = distance
        return reward1

    @staticmethod
    def reward_from_demo(demo): # TODO integrate with reward
        def distance(ob):
            return np.linalg.norm(
                ob.gripper_pose[:3] - ob.task_low_dim_state)

        # return [0] * (len(demo) - 2) + [1]

        init_distance = distance(demo[0])
        return [
            (distance(demo[i]) - distance(demo[i+1])) / (100 * init_distance)
            for i in range(len(demo[1:-1]))] + [1]
        return [
            1 / (1 + 10 * (distance(ob)  / init_distance))
            for ob in demo[1:-1]] + [1]

    # def _reward(tip_pos, goal_pos, init_distance):
    #     distance = self._distance(tip_pos, goal_pos)
    #     reward = 1 / (1 + 10 * (distance  / init_distance))
    #     return reward, distance

    def _distance_to_goal(self):
        tip_pos = self.robot.arm.get_tip().get_position()
        goal_pos = self.target.get_position()
        return np.linalg.norm(np.array(tip_pos) - np.array(goal_pos))