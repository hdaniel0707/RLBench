from imp import is_builtin
from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors


def modulo_90(angle: float):
    if angle < 0:
        angle += 2*np.pi
    k = 1
    while angle - k*np.pi/2 > -0.01:
        k += 1
    return angle - (k-1)*np.pi/2


class PickAndPlace(Task):

    def init_task(self, reward_kwargs={}, failure_when_drop=True) -> None:
        # assert reward in ['sparse', 'dist', 'delta-dist']
        # self._reward = reward
        # self._reward_scale = reward_scale
        self._reward_kwargs = reward_kwargs
        self._failure_when_drop = failure_when_drop

        # TODO remove
        pos = self.get_base().get_position()
        pos[0] += .05
        self.get_base().set_position(pos)
        # end TODO

        self.target_block = Shape('pick_and_place_target')
        # self.distractors = [
        #     Shape('stack_blocks_distractor%d' % i)
        #     for i in range(2)]
        self.register_graspable_objects([self.target_block])
        self.boundary = SpawnBoundary([Shape('pick_and_place_boundary')])
        self.success_detector = ProximitySensor('success')
        self.target = Shape('target')

        # print("min value", np.array(self.boundary._boundaries[0]._boundary.get_position()) + np.array(self.boundary._boundaries[0]._boundary_bbox.points[0]))
        # print("max value", np.array(self.boundary._boundaries[0]._boundary.get_position()) + np.array(self.boundary._boundaries[0]._boundary_bbox.points[-1]))

        self.register_success_conditions([
            DetectedCondition(self.target_block, self.success_detector)])

    def init_episode(self, index: int) -> List[str]:

        block_color_name, block_rgb = colors[index]
        self.target_block.set_color(block_rgb)

        # color_choices = np.random.choice(
        #     list(range(index)) + list(range(index + 1, len(colors))),
        #     size=2, replace=False)
        # for i, ob in enumerate(self.distractors):
        #     name, rgb = colors[color_choices[int(i)]]
        #     ob.set_color(rgb)

        # TODO remove
        # cur_pos = self.success_detector.get_position()
        # self.success_detector.set_position([cur_pos[0], cur_pos[1], .87])
        # end TODO
        self.boundary.clear()
        self.boundary.sample(
            self.target, min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0))
        for block in [self.target_block]:
            self.boundary.sample(block, min_distance=0.1)

        # utilities for reward()
        self._grasped = 0
        self._init_distance = self._distance_to_goal(self._grasped)
        self._prev_distance = self._init_distance

        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def variation_count(self) -> int:
        return len(colors)

    # def get_low_dim_state(self) -> np.ndarray:
    #     return np.concatenate([
    #         np.array(self.target_block.get_position()),
    #         np.array(self.success_detector.get_position())
    #     ])

    # TODO delete this    
    def get_low_dim_state(self) -> np.ndarray:
        angle = max([modulo_90(a) for a in self.target_block.get_orientation()])
        return np.concatenate([
            np.array(self.target_block.get_pose()[:2]),
            np.array([angle]), #np.array([self.target_block.get_pose()[5]]),
            np.array(self.success_detector.get_position()[:2])
        ])

    def is_static_workspace(self) -> bool:
        return True

    def _distance_to_goal(self, index):
        targets = [self.target_block, self.success_detector]
        target = targets[index]
        tip_pos = self.robot.arm.get_tip().get_position()
        goal_pos = target.get_position()
        return np.linalg.norm(np.array(tip_pos) - np.array(goal_pos))

    def reward(self, terminate):
        return self._reward(terminate, **self._reward_kwargs)

    def _reward(self, terminate, reward="sparse", scale=100):
        # assert reward in ['sparse', 'dist', 'delta-dist']

        if terminate:
            return 1

        # if block not grasped yet
        if self._grasped == 0:
            if len([ob for ob in self.robot.gripper.get_grasped_objects()
                    if self.target_block.get_handle() == ob.get_handle()]) > 0:
                self._grasped = 1
                self._init_distance = self._distance_to_goal(self._grasped)
                self._prev_distance = self._init_distance
                return 1

        # if block grasped
        if self._grasped == 1:
            if len([ob for ob in self.robot.gripper.get_grasped_objects()
                    if self.target_block.get_handle() == ob.get_handle()]) == 0:
                self._grasped = 0
                self._init_distance = self._distance_to_goal(self._grasped)
                self._prev_distance = self._init_distance
                return -1 # TODO

        if reward == 'sparse':
            return 0
        elif reward == 'dist':
            distance = self._distance_to_goal(self._grasped)
            # return - distance  / (100 * self._init_distance)
            return 1 / (scale * (1 + 10 * (distance  / self._init_distance)))
        elif reward == 'delta-dist':
            distance = self._distance_to_goal(self._grasped)
            return (self._prev_distance - distance) / (scale * self._init_distance)
        else:
            raise ValueError

    @staticmethod # we might call this before launching an env
    def reward_from_demo(demo, reward="sparse", scale=100):
        # assert reward in ['sparse', 'dist', 'delta-dist']

        def distance(ob, i, grasp_index):
            if i <= grasp_index:
                return np.linalg.norm(
                    ob.gripper_pose[:3] - ob.task_low_dim_state[:3])
            else:
                return np.linalg.norm(
                    ob.gripper_pose[:3] - ob.task_low_dim_state[3:])

        grasp_index = len(demo)
        for i, ob in enumerate(demo):
            if distance(ob, 0, grasp_index) < 0.01:
                grasp_index = i
                break
        assert grasp_index < len(demo)

        if reward == 'sparse':
            rew = [0] * (len(demo) - 2) + [1]
        elif reward == 'dist':
            init_distance = distance(demo[0], 0, grasp_index)
            rew = [
                1 / (scale * (1 + 10 * (distance(demo[i+1], i, grasp_index)  / init_distance)))
                for i in range(len(demo[1:-1]))] + [1]
        elif reward == 'delta-dist':
            init_distance = distance(demo[0], 0, grasp_index)
            rew = [
                (distance(demo[i], i, grasp_index) - distance(demo[i+1], i, grasp_index)) / (
                    scale * init_distance)
                for i in range(len(demo[1:-1]))] + [1]
        else:
            raise ValueError
        rew[grasp_index] = 1
        return rew

    def success(self) -> Tuple[bool, bool]:
        """If the task is currently successful.

        :return: Tuple containing 2 bools: first specifies if the task is currently successful,
            second specifies if the task should terminate (either from success or from broken constraints).
        """
        if self._failure_when_drop and self._grasped == 1 and len([
                ob for ob in self.robot.gripper.get_grasped_objects() 
                if self.target_block.get_handle() == ob.get_handle()]) == 0:
            return False, True, {}

        all_met = np.all(
            [cond.condition_met()[0] for cond in self._success_conditions])
        should_terminate = all_met
        return all_met, should_terminate, {}