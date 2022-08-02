from typing import List
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from rlbench.const import colors
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition
#from rlbench.backend.spawn_boundary import SpawnBoundary
import random
import math


class HdYcbGrasp(Task):

    def init_task(self) -> None:
        self.workpiece = Shape('workpiece')
        self.target = Shape('target_container')
        self.source = Shape('source_container')
        #self.boundary = SpawnBoundary([Shape('source_container')])
        self.success_detector = ProximitySensor('success')
        self.register_success_conditions([
            DetectedCondition(self.workpiece, self.success_detector)])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        pass

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        pass

    def is_static_workspace(self) -> bool:
        return True
