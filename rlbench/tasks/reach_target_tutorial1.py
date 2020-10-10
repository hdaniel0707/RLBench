from typing import List, Tuple, Callable
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors

from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape

import numpy as np


class ReachTargetTutorial1(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.distractor0 = Shape('distractor0')
        self.distractor1 = Shape('distractor1')
        self.boundary = Shape('boundary')
        success_sensor = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self.robot.arm.get_tip(), success_sensor)
        ])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        color_choices = np.random.choice(list(range(index)) + list(
            range(index + 1, len(colors))), size=2, replace=False)
        self.target.set_color(color_rgb)
        for ob, i in zip([self.distractor0, self.distractor1],
                        color_choices):
            ob.set_color(colors[i][1])
        b = SpawnBoundary([self.boundary])
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(ob, min_distance=0.2, min_rotation=(0, 0, 0),
                    max_rotation=(0, 0, 0))
        return ['reach the %s target' % color_name,
                'reach the %s thing' % color_name]

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float],
                                            Tuple[float, float, float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True

