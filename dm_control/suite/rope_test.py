# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_env import specs
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
import random
import os
import math
from PIL import Image, ImageColor
from scipy.stats import linregress
from dm_control.suite.wrappers.modder import LightModder
from imageio import imsave

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()

CORNER_INDEX_ACTION = ['B3', 'B8', 'B10', 'B20']
GEOM_INDEX = ['G0_0', 'G0_8', 'G8_0', 'G8_8']


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    # return common.read_model('cloth_v0.xml'), common.ASSETS
    return common.read_model('rope_test_colored.xml'), common.ASSETS


W = 64


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
    """Returns the easy cloth task."""

    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Rope(randomize_gains=False, random=random, **kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, n_frame_skip=1, rope_task=True, **environment_kwargs)


class Physics(mujoco.Physics):
    """physics for the point_mass domain."""


class Rope(base.Task):
    """A point_mass `Task` to reach target with smooth reward."""

    def __init__(self, randomize_gains, random=None, init_flat=False, use_dr=False):
        """Initialize an instance of `PointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._randomize_gains = randomize_gains
        self._init_flat = init_flat
        self._n_geoms = 20
        self._use_dr = use_dr

        super(Rope, self).__init__(random=random)

    def action_spec(self, physics):
        return [specs.DiscreteArray(num_values=self._n_geoms),
                specs.BoundedArray(shape=(2,), dtype=np.float, minimum=[-1.0] * 2, maximum=[1.0] * 2)]

    def get_geoms(self, physics):
        geoms = [physics.named.data.geom_xpos['G{}'.format(i)][:2] for i in range(self._n_geoms)]
        return np.array(geoms)

    def initialize_episode(self, physics):
        if self._use_dr:
            self.dof_damping = np.concatenate([np.zeros((6)), np.ones(2 * (self._n_geoms - 1)) * 0.002], axis=0)
            self.body_mass = np.concatenate([np.zeros(1), np.ones(self._n_geoms) * 0.00563])
            self.body_inertia = np.concatenate(
                [np.zeros((1, 3)), np.tile(np.array([[4.58e-07, 4.58e-07, 1.8e-07]]), (self._n_geoms, 1))],
                axis=0)
            self.geom_friction = np.tile(np.array([[1, 0.005, 0.001]]), (self._n_geoms + 5, 1))
            self.cam_pos = np.array([0, 0, 0.75])
            self.cam_quat = np.array([1, 0, 0, 0])

            self.light_diffuse = np.array([0, 0, 0])
            self.light_specular = np.array([0, 0, 0])
            self.light_ambient = np.array([0, 0, 0])
            self.light_castshadow = np.array([1])
            self.light_dir = np.array([0, 0, -1])
            self.light_pos = np.array([0, 0, 1])

            self.apply_dr(physics)

        if not self._init_flat:
            physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :2] = np.random.uniform(-0.8, 0.8, size=8).reshape((4, 2))
        super(Rope, self).initialize_episode(physics)

    def apply_dr(self, physics):
        # visual randomization
        # light randomization
        lightmodder = LightModder(physics)
        ambient_value = self.light_ambient.copy() + np.random.uniform(-0.5, 0.5, size=3)
        lightmodder.set_ambient('light', ambient_value)

        shadow_value = self.light_castshadow.copy()
        lightmodder.set_castshadow('light', shadow_value + np.random.uniform(0, 40))
        diffuse_value = self.light_diffuse.copy()
        lightmodder.set_diffuse('light',diffuse_value+np.random.uniform(-0.1,0.1,))
        lightmodder.set_diffuse('light', np.array([0, 0, 0]))

        dir_value = self.light_dir.copy()
        lightmodder.set_dir('light', dir_value + np.random.uniform(-0.1, 0.1))
        pos_value = self.light_pos.copy()
        lightmodder.set_pos('light', pos_value + np.random.uniform(-0.1, 0.1))
        specular_value = self.light_specular.copy()
        lightmodder.set_specular('light', specular_value + np.random.uniform(-0.1, 0.1))

        # physics randomization

        # damping randomization

        physics.named.model.dof_damping[:] = np.random.uniform(0, 0.0001) + self.dof_damping

        # # friction randomization
        geom_friction = self.geom_friction.copy()
        physics.named.model.geom_friction[1:, 0] = np.random.uniform(-0.5, 0.5) + geom_friction[1:, 0]

        physics.named.model.geom_friction[1:, 1] = np.random.uniform(-0.002, 0.002) + geom_friction[1:, 1]

        physics.named.model.geom_friction[1:, 2] = np.random.uniform(-0.0005, 0.0005) + geom_friction[1:, 2]

        # # inertia randomization
        body_inertia = self.body_inertia.copy()
        physics.named.model.body_inertia[1:] = np.random.uniform(-0.5, 0.5) * 1e-07 + body_inertia[1:]

        # mass randomization
        body_mass = self.body_mass.copy()

        physics.named.model.body_mass[1:] = np.random.uniform(-0.0005, 0.0005) + body_mass[1:]


    def before_step(self, action, physics):
        physics.named.data.xfrc_applied[:, :3] = np.zeros((3,))
        physics.named.data.qfrc_applied[:2] = 0

        index, direction = action
        direction = direction * 0.075

        if self._use_dr and not self._per_traj:
            self.apply_dr(physics)

        if True:
            chosen_action = 'B{}'.format(index)
            chosen_geom = 'G{}'.format(index)

            position = direction + physics.named.data.geom_xpos[chosen_geom, :2]
            dist = position - physics.named.data.geom_xpos[chosen_geom, :2]

            loop = 0
            while np.linalg.norm(dist) > 0.025:
                loop += 1
                if loop > 40:
                    break
                physics.named.data.xfrc_applied[chosen_action, :2] = direction * 30
                physics.step()
                self.after_step(physics)
                dist = position - physics.named.data.geom_xpos[chosen_geom, :2]

    def get_termination(self, physics):
        return None

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        return obs

    def get_reward(self, physics):
        return 0
