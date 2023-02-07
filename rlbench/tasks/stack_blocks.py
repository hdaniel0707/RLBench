from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedSeveralCondition
from rlbench.backend.conditions import NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors

#MAX_STACKED_BLOCKS = 3
#DISTRACTORS = 4
CFG_FILE_PATH = "/home/daniel/rltrain/cfg_rlbench/config.yaml"

import yaml
import os
def load_yaml(file):
    if file is not None:
        with open(file) as f:
            return yaml.load(f, Loader=yaml.UnsafeLoader)
    return None


class StackBlocks(Task):

    def init_task(self) -> None:

        cfg_path = load_yaml(CFG_FILE_PATH)['path']
        config = load_yaml(cfg_path)
        self.config_params = config['environment']['task']['params']
        target_blocks_num = int(self.config_params[0])
        distractors_num = int(self.config_params[1])
        self.blocks_to_stack = int(self.config_params[2])
        #print(self.config_params)     

        self.blocks_stacked = 0
        self.target_blocks = [Shape('stack_blocks_target%d' % i)
                              for i in range(target_blocks_num)]

        cube_mass = 0.1
        #cube_mass = 0.01
        for obj in self.target_blocks:
            obj.set_mass(cube_mass)

        self.distractors = [
            Shape('stack_blocks_distractor%d' % i)
            for i in range(distractors_num)]
        
        for obj in self.distractors:
            obj.set_mass(cube_mass)

        # self.boundaries = [Shape('stack_blocks_boundary%d' % i)
        #                    for i in range(4)]
        self.boundaries = [Shape('hd_stack_blocks_boundary')]

        self.register_graspable_objects(self.target_blocks + self.distractors)

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoint_ability_start(3, self._move_above_drop_zone)
        self.register_waypoint_ability_start(5, self._is_last)
        self.register_waypoints_should_repeat(self._repeat)

    def init_episode(self, index: int) -> List[str]:
        # For each color, we want to have 2, 3 or 4 blocks stacked
        # color_index = int(index / MAX_STACKED_BLOCKS)
        # self.blocks_to_stack = 1 + index % MAX_STACKED_BLOCKS
        # self.blocks_to_stack = 3 + index % MAX_STACKED_BLOCKS
        color_index = int(index / self.blocks_to_stack)
        color_name, color_rgb = colors[color_index]
        for b in self.target_blocks:
            b.set_color(color_rgb)
        
        #self.target_blocks[0].set_color((1.0, 0.0, 1.0))

        success_detector = ProximitySensor(
            'stack_blocks_success')
        
        self._target_place = success_detector

        # print(type(self.target_blocks))
        # print(type(self.distractors))
        # print(type([success_detector]))
        # print(self.target_blocks)
        # print(self.distractors)
        # print(success_detector)

        self._observation = [success_detector] + self.target_blocks + self.distractors

        self.register_success_conditions([DetectedSeveralCondition(
            self.target_blocks, success_detector, self.blocks_to_stack),
            NothingGrasped(self.robot.gripper)
        ])

        self.blocks_stacked = 0
        color_choices = np.random.choice(
            list(range(color_index)) + list(
                range(color_index + 1, len(colors))),
            size=2, replace=False)
        for i, ob in enumerate(self.distractors):
            name, rgb = colors[color_choices[int(i / 4)]]
            ob.set_color(rgb)
        b = SpawnBoundary(self.boundaries)
        for block in self.target_blocks + self.distractors:
            b.sample(block, min_distance=0.1)

        return ['stack %d %s blocks' % (self.blocks_to_stack, color_name),
                'place %d of the %s cubes on top of each other'
                % (self.blocks_to_stack, color_name),
                'pick up and set down %d %s blocks on top of each other'
                % (self.blocks_to_stack, color_name),
                'build a tall tower out of %d %s cubes'
                % (self.blocks_to_stack, color_name),
                'arrange %d %s blocks in a vertical stack on the table top'
                % (self.blocks_to_stack, color_name),
                'set %d %s cubes on top of each other'
                % (self.blocks_to_stack, color_name)]

    def variation_count(self) -> int:
        return len(colors) * self.blocks_to_stack

    def _move_above_next_target(self, _):
        if self.blocks_stacked >= self.blocks_to_stack:
            raise RuntimeError('Should not be here.')
        w2 = Dummy('waypoint1')
        x, y, z = self.target_blocks[self.blocks_stacked].get_position()
        _, _, oz = self.target_blocks[self.blocks_stacked].get_orientation()
        ox, oy, _ = w2.get_orientation()
        w2.set_position([x, y, z])
        w2.set_orientation([ox, oy, -oz])

    def _move_above_drop_zone(self, waypoint):
        target = Shape('stack_blocks_target_plane')
        x, y, z = target.get_position()
        waypoint.get_waypoint_object().set_position(
            [x, y, z + 0.08 + 0.06 * self.blocks_stacked])

    def _is_last(self, waypoint):
        last = self.blocks_stacked == self.blocks_to_stack - 1
        waypoint.skip = last

    def _repeat(self):
        self.blocks_stacked += 1
        return self.blocks_stacked < self.blocks_to_stack
    
    def is_static_workspace(self) -> bool:
        return True