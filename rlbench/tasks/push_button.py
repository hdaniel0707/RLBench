from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition,ConditionSet

import numpy as np

# button top plate and wrapper will be be red before task completion
# and be changed to cyan upon success of task, so colors list used to randomly vary colors of
# base block will be redefined, excluding red and green
colors = [
    ('maroon', (0.5, 0.0, 0.0)),
    ('green', (0.0, 0.5, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
    ('navy', (0.0, 0.0, 0.5)),
    ('yellow', (1.0, 1.0, 0.0)),
    ('cyan', (0.0, 1.0, 1.0)),
    ('magenta', (1.0, 0.0, 1.0)),
    ('silver', (0.75, 0.75, 0.75)),
    ('gray', (0.5, 0.5, 0.5)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
]


class PushButton(Task):

    def init_task(self, reward="sparse", reward_scale=100) -> None:
        assert reward in ['sparse', 'dist', 'delta-dist']
        self._reward = reward
        self._reward_scale = reward_scale

        self.target_button = Shape('push_button_target')
        self.target_topPlate = Shape('target_button_topPlate')
        self.joint = Joint('target_button_joint')
        self.target_wrap = Shape('target_button_wrap')
        self.goal_condition = JointCondition(self.joint, 0.003)

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        self.target_topPlate.set_color([1.0, 0.0, 0.0])
        self.target_wrap.set_color([1.0, 0.0, 0.0])
        self.variation_index = index
        button_color_name, button_rgb = colors[index]
        self.target_button.set_color(button_rgb)
        self.register_success_conditions(
            [ConditionSet([self.goal_condition], True, False)])

        # utilities for reward()
        self._init_distance = self._distance_to_goal()
        self._prev_distance = self._init_distance

        return ['push the %s button' % button_color_name,
                'push down the %s button' % button_color_name,
                'press the button with the %s base' % button_color_name,
                'press the %s button' % button_color_name]

    def variation_count(self) -> int:
        return len(colors)

    def step(self) -> None:
        if self.goal_condition.condition_met() == (True, True):
            self.target_topPlate.set_color([0.0, 1.0, 0.0])
            self.target_wrap.set_color([0.0, 1.0, 0.0])

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self.target_button.get_position())

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

    def _distance_to_goal(self):
        tip_pos = self.robot.arm.get_tip().get_position()
        goal_pos = self.target_button.get_position()
        return np.linalg.norm(np.array(tip_pos) - np.array(goal_pos))