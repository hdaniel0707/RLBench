from typing import List
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from rlbench.const import colors
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
import random
import math



class HdTest(Task):

    def init_task(self) -> None:
        self.block = Shape('block0')
        self.target = Shape('target')
        self.boundary = SpawnBoundary([Shape('boundary')])
        self.success_detector = ProximitySensor('success')
        self.register_waypoint_ability_start(0, self._push_from)
        self.register_success_conditions([
            DetectedCondition(self.block, self.success_detector)])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index

        block_color_name, block_rgb = colors[index]
        self.block.set_color(block_rgb)
        print(block_color_name)

        self.boundary.clear()
        self.boundary.sample(self.block)
        self.boundary.sample(self.target)

        return ['']

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        color_index = random.randint(0, len(colors)-1)
        # print(color_index)
        # block_color_name, block_rgb = colors[color_index]
        # self.block.set_color(block_rgb)
        # print(self.block.get_color())
        #
        # print(self.block.get_position())


    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        pass

    def _push_from(self, waypoint):
        block_pos = self.block.get_position()
        target_pos = self.target.get_position()

        print("--------------------------")
        print(block_pos)
        print(target_pos)

        dx =  target_pos[0] - block_pos[0]
        dy =  target_pos[1] - block_pos[1]

        print("--------------------------")
        print(dx)
        print(dy)

        alpha = math.atan(dy / dx)

        print("--------------------------")
        print(alpha)

        pos_x = math.cos(alpha) * 0.15
        pos_y = math.sin(alpha) * 0.15

        print("--------------------------")
        print(pos_x)
        print(pos_y)

        if block_pos[0] < target_pos[0]:
            pos_x = -pos_x

        if block_pos[1] < target_pos[1]:
            pos_y = -pos_y

        way_obj = waypoint.get_waypoint_object()
        #way_obj.set_position([pos_x,pos_y,0],relative_to=self.block)
        way_obj.set_position([pos_x + block_pos[0],pos_y + block_pos[1],block_pos[2]])
