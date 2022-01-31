from typing import List
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition

import numpy as np


class PressSwitch(Task):

    def init_task(self) -> None:
        self._switch_joint = Joint('joint')
        self.register_success_conditions([JointCondition(self._switch_joint, 1.0)])

    def init_episode(self, index: int) -> List[str]:
        return ['press switch',
                'turn the switch on or off',
                'flick the switch']

    def variation_count(self) -> int:
        return 1

    def reward(self, terminate):
        if terminate:
            return 1

        return 0

    @staticmethod
    def reward_from_demo(demo): # TODO integrate with reward
        return [0] * (len(demo) - 2) + [1]

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.array(self._switch_joint.get_position())