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
_FIXED_ACTION_DIMS = 6
SUITE = containers.TaggedTasks()

CORNER_INDEX_ACTION = ['B3', 'B8', 'B10', 'B20']
GEOM_INDEX = ['G0_0', 'G0_8', 'G8_0', 'G8_8']


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    # return common.read_model('cloth_v0.xml'), common.ASSETS
    return common.read_model('rope_dr.xml'), common.ASSETS


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

    def __init__(self, randomize_gains, random=None, random_pick=True, init_flat=False,
                 use_dr=False, per_traj=False):
        """Initialize an instance of `PointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._randomize_gains = randomize_gains
        self._init_flat = init_flat
        self._random_pick = random_pick
        self._n_geoms = 25
        self._use_dr = use_dr
        self._per_traj = per_traj

        super(Rope, self).__init__(random=random)

    def action_spec(self, physics):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return specs.BoundedArray(
            shape=(_FIXED_ACTION_DIMS,),
            dtype=np.float,
            minimum=[-1.0] * _FIXED_ACTION_DIMS,
            maximum=[1.0] * _FIXED_ACTION_DIMS)

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

        render_kwargs = {}
        render_kwargs['camera_id'] = 0
        render_kwargs['width'] = W
        render_kwargs['height'] = W
        image = physics.render(**render_kwargs)
        self.image = image

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

        assert action.shape[0] == _FIXED_ACTION_DIMS
        d =_FIXED_ACTION_DIMS // 2
        if not self._random_pick:
            pick_location = (action[:2] * 0.5 + 0.5) * (W - 1)
        else:
            pick_location = self.current_loc

        goal_position = action[d:d + 2]
        goal_position = goal_position * 0.075

        if self._use_dr and not self._per_traj:
            self.apply_dr(physics)

        # computing the mapping from geom_xpos to location in image
        cam_fovy = physics.named.model.cam_fovy['fixed']
        f = 0.5 * W / math.tan(cam_fovy * math.pi / 360)
        cam_matrix = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]])
        cam_mat = physics.named.data.cam_xmat['fixed'].reshape((3, 3))
        cam_pos = physics.named.data.cam_xpos['fixed'].reshape((3, 1))
        cam = np.concatenate([cam_mat, cam_pos], axis=1)
        cam_pos_all = np.zeros((self._n_geoms, 3, 1))
        for i in range(self._n_geoms):
            geom_name = 'G{}'.format(i)
            geom_xpos_added = np.concatenate([physics.named.data.geom_xpos[geom_name], np.array([1])]).reshape((4, 1))
            cam_pos_all[i] = cam_matrix.dot(cam.dot(geom_xpos_added)[:3])

        cam_pos_xy = np.rint(cam_pos_all[:, :2].reshape((self._n_geoms, 2)) / cam_pos_all[:, 2])
        cam_pos_xy = cam_pos_xy.astype(int)
        cam_pos_xy[:, 1] = W - cam_pos_xy[:, 1]
        cam_pos_xy[:, [0, 1]] = cam_pos_xy[:, [1, 0]]

        dists = np.linalg.norm(cam_pos_xy - pick_location[None, :], axis=1)
        index = np.argmin(dists)

        if True:
            corner_action = 'B{}'.format(index)
            corner_geom = 'G{}'.format(index)

            position = goal_position + physics.named.data.geom_xpos[corner_geom, :2]
            dist = position - physics.named.data.geom_xpos[corner_geom, :2]

            loop = 0
            while np.linalg.norm(dist) > 0.025:
                loop += 1
                if loop > 40:
                    break
                physics.named.data.xfrc_applied[corner_action, :2] = dist * 30
                physics.step()
                self.after_step(physics)
                dist = position - physics.named.data.geom_xpos[corner_geom, :2]

    def get_termination(self, physics):
        if self.num_loc < 1:
            return 1.0
        else:
            return None

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        if not self._random_pick:
            pick_location = np.array([-1, 1])
            image = self.get_image(physics)
            mask = self.segment_image(image)
            self.image = image

            location_range = np.transpose(np.where(mask))
            self.location_range = location_range
            num_loc = np.shape(location_range)[0]
            self.num_loc = num_loc
        else:
            pick_location = self.sample_location(physics)
        self.current_loc = pick_location

        if self.current_loc is None:
            pick_location = np.array([-1, -1])

        obs['pick_location'] = np.tile(pick_location, 50).reshape(-1).astype('float32') / (W - 1)

        # generate random action as unif(0,1) * (hi - lo) + lo
        random_action = np.random.rand(_FIXED_ACTION_DIMS,) * 2 - 1
        random_action[:2] = pick_location.astype('float32') / (W - 1)
        random_action[2] = 0.  # todo: would be nice to use depth data
        obs['action_sample'] = random_action

        return obs

    def sample_location(self, physics):
        image = self.get_image(physics)
        self.image = image

        mask = self.segment_image(image)
        location_range = np.transpose(np.where(mask))
        self.location_range = location_range
        num_loc = np.shape(location_range)[0]
        self.num_loc = num_loc
        if num_loc == 0:
            return None
        index = np.random.randint(num_loc, size=1)
        location = location_range[index][0]

        return location

    def get_image(self, physics):
        render_kwargs = {}
        render_kwargs['camera_id'] = 0
        render_kwargs['width'] = W
        render_kwargs['height'] = W
        image = physics.render(**render_kwargs)
        return image

    def segment_image(self, image):
        return np.all(image > 150, axis=2)

    def get_reward(self, physics):
        reward_mask = self.segment_image(self.image).astype(int)
        line = np.linspace(0, 31, num=32) * (-0.5)
        column = np.concatenate([np.flip(line), line])
        reward = np.sum(reward_mask * np.exp(column).reshape((W, 1))) / 111.0
        return reward
