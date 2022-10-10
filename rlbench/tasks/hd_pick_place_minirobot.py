import os
from typing import List, Tuple
from rlbench.backend.task import Task
from rlbench.const import colors
from rlbench.backend.conditions import ConditionSet, DetectedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy

import math

boundary_mins = [0.05, -0.15, 0.05]
boundary_maxs = [0.35, 0.25 , 0.1]

def sample_minirobot_parts(task_base):
    #assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../../3dmodels/minirobot/')

    assets_dir ="/home/daniel/sim2real_robotics/3dmodels/minirobot"
    #assets_dir ="/home/daniel/sim2real_robotics/3dmodels/ycb-tools/models/ycb"

    # samples = np.random.choice(
    #     os.listdir(assets_dir), num_samples, replace=False)


    #samples = ["red/base_red_2_tf","red/arm_red_2_tf","red/gripper_red_2_tf"]
    samples = ["red/base_red_tf","red/arm_red_tf","red/gripper_red_tf"]
    #samples = ["red/arm_red_tf_rot_2","red/arm_red_tf_rot_2","red/arm_red_tf_rot_2"]
    #samples = ["red/box_ref_100mm"]
    #samples = ["048_hammer/google_16k/textured","042_adjustable_wrench/google_16k/textured","043_phillips_screwdriver/google_16k/textured"]
    #samples = ["box","box","box"]

    created = []
    for s in samples:
        respondable = os.path.join(assets_dir, s + '.obj')
        #respondable = os.path.join(assets_dir, 'box_tr.obj')
        visual = os.path.join(assets_dir, s + '.obj')
        print(visual)
        resp = Shape.import_mesh(respondable, scaling_factor=0.05)
        vis = Shape.import_mesh(visual, scaling_factor=0.05)
        # resp = Shape.import_mesh(respondable, scaling_factor=1.0)
        # vis = Shape.import_mesh(visual, scaling_factor=1.0)
        resp = resp.get_convex_decomposition()
        #vis = vis.get_convex_decomposition()
        resp.set_renderable(False)
        vis.set_renderable(True)
        vis.set_parent(resp)
        vis.set_dynamic(False)
        vis.set_respondable(False)
        resp.set_dynamic(True)
        # resp.set_dynamic(False)
        resp.set_mass(1.0)
        resp.set_respondable(True)
        resp.set_model(True)
        # resp.set_respondable(False)
        # resp.set_model(False)
        resp.set_parent(task_base)
        created.append(resp)
    return created


class HdPickPlaceMinirobot(Task):

    def init_task(self) -> None:
        self.source_plane = Shape('source_plane')
        self.target_plane = Shape('target_plane')
        self.target_a = Shape('target_a')
        self.target_b = Shape('target_b')
        self.target_c = Shape('target_c')
        self.spawn_boundary = SpawnBoundary([Shape('spawn_boundary')])
        self.success_detector = ProximitySensor('success')
        # self.register_success_conditions([
        #     DetectedCondition(self.source_plane, self.success_detector)])

        # self.success_detector0 = ProximitySensor('success0')
        # self.success_detector1 = ProximitySensor('success1')

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index

        self.minirobot_parts = sample_minirobot_parts(self.get_base())

        self.x_pos = np.random.uniform(boundary_mins[0],boundary_maxs[0],3)
        self.y_pos = np.random.uniform(boundary_mins[1],boundary_maxs[1],3)
        #self.z_pos = np.random.uniform(boundary_mins[2],boundary_maxs[2],3)
        self.z_pos = np.array([0.03,0.01,0.01])
        #self.z_pos = np.array([0.1,0.1,0.1])
        self.register_graspable_objects(self.minirobot_parts)

        self.rot_1d = np.random.uniform(math.radians(0),math.radians(360),3)
        # self.y_rot = np.random.uniform(math.radians(0),math.radians(360),3)
        #self.z_rot = np.random.uniform(math.radians(0),math.radians(360),3)

        self.x_rot = np.array([math.radians(0),math.radians(0),math.radians(0)])
        self.y_rot = np.array([math.radians(90),math.radians(90),math.radians(90)])
        self.z_rot = np.array([math.radians(0),math.radians(0),math.radians(0)])

        self.spawn_boundary.clear()

        conditions = []
        i = 0
        for ob in self.minirobot_parts:
            ob.set_position([self.x_pos[i], self.y_pos[i], self.z_pos[i]], relative_to=self.source_plane,reset_dynamics=False)
            #ob.set_orientation([self.x_rot[i], self.y_rot[i], self.z_rot[i]],relative_to=self.source_plane, reset_dynamics=False)
            self.spawn_boundary.sample(ob, ignore_collisions=False, min_distance=0.1)
            #self.spawn_boundary.sample(ob)
            ob.set_orientation([self.x_rot[i], self.y_rot[i], self.z_rot[i]],relative_to=self.source_plane, reset_dynamics=False)
            #ob.rotate([self.rot_1d[i],0.0,0.0])
            ob.rotate([0.0,0.0,0.0])
            #ob.rotate([math.radians(270),0.0,0.0])
            i+=1
            conditions.append(DetectedCondition(ob, self.success_detector))

        self.register_success_conditions(
            [ConditionSet(conditions, simultaneously_met=True)])
        # TODO: This is called at the start of each episode.


    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def step(self) -> None:
        pass

    def cleanup(self) -> None:
        if self.minirobot_parts is not None:
            [ob.remove() for ob in self.minirobot_parts if ob.still_exists()]
            self.bin_objects = []
        self.minirobot_parts = []

    def is_static_workspace(self) -> bool:
        return True
