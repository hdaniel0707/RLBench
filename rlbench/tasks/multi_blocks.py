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

import os
import re
from os.path import dirname, abspath, join
from typing import List, Tuple, Callable, Union

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.dummy import Dummy
from pyrep.objects.force_sensor import ForceSensor
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object

from rlbench.backend.conditions import Condition
from rlbench.backend.exceptions import WaypointError
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.waypoints import Point, PredefinedPath, Waypoint
from rlbench.backend.conditions import DetectedCondition, ConditionSet, GraspedCondition


class MultiBlocks(Task):

    def init_task(self) -> None:
        self.circles_target = Shape('circles_target')
        self.circles_distractors = [
            Shape('circles_distractor%d' % i)
            for i in range(2)]
        self.blocks_target = Shape('blocks_target')
        self.blocks_distractors = [
            Shape('blocks_distractor%d' % i)
            for i in range(2)]
        self.register_graspable_objects([self.blocks_target])
        self.boundary = SpawnBoundary([Shape('boundary')])
        self.circles_success_sensor = ProximitySensor('circles_success')
        self.plane_success_sensor = ProximitySensor('plane_success')
        self.plane_target = Shape('plane_target')
        self._episode_waypoints = []

    def init_episode(self, index: int) -> List[str]:
        if index < len(colors):
            return self._reach_target(index % len(colors))
        elif index < 2 * len(colors):
            return self._pick_and_lift(index % len(colors))
        elif index < 3 * len(colors):
            return self._stack_block(index % len(colors))

    def _sample_items(self):
        self.boundary.clear()
        for block in [self.blocks_target] + self.blocks_distractors:
            self.boundary.sample(block, min_distance=0.1)
        for circle in [self.circles_target] + self.circles_distractors:
            self.boundary.sample(circle, min_distance=0.1,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        self.boundary.sample(self.plane_target, min_distance=0.1)

    def _pick_and_lift(self, index):
        self._episode_waypoints = [0,1,2,3]

        color_name, color_rgb = colors[index]
        self.blocks_target.set_color(color_rgb)
        self.circles_target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=4, replace=False)
        for i, ob in enumerate(self.blocks_distractors + self.circles_distractors):
            name, rgb = colors[color_choices[int(i)]]
            ob.set_color(rgb)

        self._sample_items()

        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.blocks_target),
            DetectedCondition(self.blocks_target, self.circles_success_sensor)
        ])
        self.register_success_conditions([cond_set])

        return ['pick up the %s block and lift it up to the target' %
                color_name,
                'grasp the %s block to the target' % color_name,
                'lift the %s block up to the target' % color_name]

    def _stack_block(self, index):
        self._episode_waypoints = [0,1,2,4,5,6]
        blocks_to_stack = 1

        color_name, color_rgb = colors[index]
        self.blocks_target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=5, replace=False)
        for i, ob in enumerate([self.circles_target] + self.blocks_distractors + self.circles_distractors):
            name, rgb = colors[color_choices[int(i)]]
            ob.set_color(rgb)

        self._sample_items()

        cond_set = ConditionSet([
            NothingGrasped(self.robot.gripper),
            DetectedCondition(self.blocks_target, self.plane_success_sensor)
        ])
        self.register_success_conditions([cond_set])

        return ['stack %d %s blocks' % (blocks_to_stack, color_name),
                'place %d of the %s cubes on top of each other'
                % (blocks_to_stack, color_name),
                'pick up and set down %d %s blocks on top of each other'
                % (blocks_to_stack, color_name),
                'build a tall tower out of %d %s cubes'
                % (blocks_to_stack, color_name),
                'arrange %d %s blocks in a vertical stack on the table top'
                % (blocks_to_stack, color_name),
                'set %d %s cubes on top of each other'
                % (blocks_to_stack, color_name)]

    def _reach_target(self, index):
        self._episode_waypoints = [3]

        color_name, color_rgb = colors[index]
        self.circles_target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=5, replace=False)
        for i, ob in enumerate([self.blocks_target] + self.blocks_distractors + self.circles_distractors):
            name, rgb = colors[color_choices[int(i)]]
            ob.set_color(rgb)

        self._sample_items()

        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), self.circles_success_sensor)])

        return ['reach the %s target' % color_name,
                'touch the %s ball with the panda gripper' % color_name,
                'reach the %s sphere' %color_name]

    # def _push_block(self):

    def variation_count(self) -> int:
        return 3 * len(colors)

    def is_static_workspace(self) -> bool:
        return True

    def _get_waypoints(self, validating=False) -> List[Waypoint]:
        waypoint_name = 'waypoint%d'
        waypoints = []
        additional_waypoint_inits = []
        i = 0
        for i in self._episode_waypoints:
            name = waypoint_name % i
            ob_type = Object.get_object_type(name)
            way = None
            if ob_type == ObjectType.DUMMY:
                waypoint = Dummy(name)
                start_func = None
                end_func = None
                if i in self._waypoint_abilities_start:
                    start_func = self._waypoint_abilities_start[i]
                if i in self._waypoint_abilities_end:
                    end_func = self._waypoint_abilities_end[i]
                way = Point(waypoint, self.robot,
                            start_of_path_func=start_func,
                            end_of_path_func=end_func)
            elif ob_type == ObjectType.PATH:
                cartestian_path = CartesianPath(name)
                way = PredefinedPath(cartestian_path, self.robot)
            else:
                raise WaypointError(
                    '%s is an unsupported waypoint type %s' % (
                        name, ob_type), self)

            if name in self._waypoint_additional_inits and not validating:
                additional_waypoint_inits.append(
                    (self._waypoint_additional_inits[name], way))
            waypoints.append(way)

        # Check if all of the waypoints are feasible
        feasible, way_i = self._feasible(waypoints)
        if not feasible:
            raise WaypointError(
                "Infeasible episode. Can't reach waypoint %d." % way_i, self)
        for func, way in additional_waypoint_inits:
            func(way)
        return waypoints