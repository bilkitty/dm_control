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

"""Planar Stacker domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import mujoco
from dm_env import specs
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools
from dm_control.suite.wrappers.modder import LightModder, MaterialModder, CameraModder

import os
from imageio import imsave
from PIL import Image, ImageColor
from lxml import etree
import numpy as np
import math

_TOL = 1e-13
_CLOSE = .01  # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = .02  # (Seconds)
_TIME_LIMIT = 30  # (Seconds)
_FIXED_ACTION_DIMS = 6
_ALL_PROPS = frozenset(["block", "ball"])

CORNER_INDEX_ACTION = ['B0_0', 'B0_8', 'B8_0', 'B8_8']
CORNER_INDEX_GEOM = ['G0_0', 'G0_8', 'G8_0', 'G8_8']

W = 64
SUITE = containers.TaggedTasks()


def make_model(prop_name=""):
  """Returns a tuple containing the model XML string and a dict of assets."""
  xml_string = common.read_model('cloth_block.xml')
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)

  # Select the desired prop.
  if prop_name == 'block':
    required_props = ['block']
  elif prop_name == 'ball':
    required_props = ['ball']
  elif prop_name == '':
    required_props = []
  else:
    assert False, f"unsupported prop: {prop_name}"

  # Remove unused props
  for unused_prop in _ALL_PROPS.difference(required_props):
    prop = xml_tools.find_element(mjcf, 'body', unused_prop)
    prop.getparent().remove(prop)

  return etree.tostring(mjcf, pretty_print=True), common.ASSETS

@SUITE.add('hard')
def hard(time_limit=_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
    """Returns stacker task with 2 boxes."""

    if "prop_name" in environment_kwargs.keys():
        physics = Physics.from_xml_string(*make_model(environment_kwargs["prop_name"]))
    else:
        physics = Physics.from_xml_string(*make_model())
    kwargs["prop_name"] = environment_kwargs["prop_name"]
    task = Cloth(randomize_gains=False, random=random, **kwargs)
    # hack: refresh env kwargs
    environment_kwargs = {}
    return control.Environment(
        physics, task, control_timestep=_CONTROL_TIMESTEP, special_task=True, time_limit=time_limit,
        **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""


class Cloth(base.Task):
    """A Stack `Task`: stack the boxes."""

    def __init__(self, randomize_gains, random=None, random_pick=True, init_flat=False, use_dr=False, texture_randomization=True, per_traj=False, prop_name=''):
        """Initialize an instance of `PointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._randomize_gains = randomize_gains
        self._random_pick = random_pick
        self._init_flat = init_flat
        self._use_dr = use_dr
        self._texture_randomization = texture_randomization
        self._per_traj = per_traj
        self._prop_name = prop_name

        super(Cloth, self).__init__(random=random)

    def action_spec(self, physics) -> specs.BoundedArray:
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return specs.BoundedArray(
            shape=(_FIXED_ACTION_DIMS,),
            dtype=np.float,
            minimum=[-1.0] * _FIXED_ACTION_DIMS,
            maximum=[1.0] * _FIXED_ACTION_DIMS)


    def initialize_episode(self, physics) -> None:
        physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
        physics.named.data.xfrc_applied['B4_4', :3] = np.array([0, 0, -2])


        render_kwargs = {}
        render_kwargs['camera_id'] = 0
        render_kwargs['width'] = W
        render_kwargs['height'] = W
        image = physics.render(**render_kwargs)
        self.image = image
        self.mask = self.segment_image(image).astype(int)

        if self._use_dr:
            # initialize the random parameters
            self.cam_pos = np.array([0, 0, 0.75])
            self.cam_quat = np.array([1, 0, 0, 0])

            self.light_diffuse = np.array([0.6, 0.6, 0.6])
            self.light_specular = np.array([0.3, 0.3, 0.3])
            self.light_ambient = np.array([0, 0, 0])
            self.light_castshadow = np.array([1])
            self.light_dir = np.array([0, 0, -1])
            self.light_pos = np.array([0, 0, 1])

            self.dof_damping = np.concatenate([np.zeros((6)), np.ones((160)) * 0.002], axis=0)
            self.body_mass = np.concatenate([np.zeros(1), np.ones(81) * 0.00309])
            self.body_inertia = np.concatenate(
                [np.zeros((1, 3)), np.tile(np.array([[2.32e-07, 2.32e-07, 4.64e-07]]), (81, 1))], axis=0)
            self.geom_friction = np.tile(np.array([[1, 0.005, 0.001]]), (86, 1))

            self.apply_dr(physics)

        if not self._init_flat:
            # DONE: create overlap between prop and cloth, see xml
            # TODO: randomly move object from underneath cloth
            physics.after_reset()
            #object_x = physics.named.data.site_xpos[self._prop_name, 'x']
            #object_z = physics.named.data.site_xpos[self._prop_name, 'z']
            #object_z_offset = object_z + 1
            #physics.named.data.geom_xpos['G3_4'] = object_x, object_z_offset, 0
            #physics.named.data.geom_xpos['G4_4'] = object_x, object_z_offset, 0

            physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :3] = np.random.uniform(-.3, .3, size=3)

        super(Cloth, self).initialize_episode(physics)

    def apply_dr(self, physics) -> None:
        if self._texture_randomization:
            physics.named.model.mat_texid[15] = np.random.choice(3, 1) + 9

        ### visual randomization

        #light randomization
        lightmodder = LightModder(physics)
        # ambient_value=lightmodder.get_ambient('light')
        ambient_value = self.light_ambient.copy() + np.random.uniform(-0.4, 0.4, size=3)
        lightmodder.set_ambient('light', ambient_value)

        # shadow_value=lightmodder.get_castshadow('light')
        shadow_value = self.light_castshadow.copy()
        lightmodder.set_castshadow('light', shadow_value + np.random.uniform(0, 40))
        # diffuse_value=lightmodder.get_diffuse('light')
        diffuse_value = self.light_diffuse.copy()
        lightmodder.set_diffuse('light', diffuse_value + np.random.uniform(-0.01, 0.01, ))
        # dir_value=lightmodder.get_dir('light')
        dir_value = self.light_dir.copy()
        lightmodder.set_dir('light', dir_value + np.random.uniform(-0.1, 0.1))
        # pos_value=lightmodder.get_pos('light')
        pos_value = self.light_pos.copy()
        lightmodder.set_pos('light', pos_value + np.random.uniform(-0.1, 0.1))
        # specular_value=lightmodder.get_specular('light')
        specular_value = self.light_specular.copy()
        lightmodder.set_specular('light', specular_value + np.random.uniform(-0.1, 0.1))

        # material randomization#
        #    Material_ENUM=['ground','wall_x','wall_y','wall_neg_x','wall_neg_x','wall_neg_y']
        #    materialmodder=MaterialModder(physics)
        #    for name in Material_ENUM:
        #     materialmodder.rand_all(name)

        # camera randomization
        # cameramodder=CameraModder(physics)
        # # fovy_value=cameramodder.get_fovy('fixed')
        # # cameramodder.set_fovy('fixed',fovy_value+np.random.uniform(-1,1))
        # # pos_value = cameramodder.get_pos('fixed')
        # pos_value=self.cam_pos.copy()
        # cameramodder.set_pos('fixed',np.random.uniform(-0.003,0.003,size=3)+pos_value)
        # # quat_value = cameramodder.get_quat('fixed')
        # quat_value=self.cam_quat.copy()
        # cameramodder.set_quat('fixed',quat_value+np.random.uniform(-0.01,0.01,size=4))

        ### physics randomization

        # damping randomization

        physics.named.model.dof_damping[:] = np.random.uniform(0, 0.0001) + self.dof_damping

        # # friction randomization
        geom_friction = self.geom_friction.copy()
        physics.named.model.geom_friction[5:, 0] = np.random.uniform(-0.5, 0.5) + geom_friction[5:, 0]
        #
        physics.named.model.geom_friction[5:, 1] = np.random.uniform(-0.002, 0.002) + geom_friction[5:, 1]
        #
        physics.named.model.geom_friction[5:, 2] = np.random.uniform(-0.0005, 0.0005) + geom_friction[5:, 2]
        #
        # # inertia randomization
        body_inertia = self.body_inertia.copy()
        physics.named.model.body_inertia[1:] = np.random.uniform(-0.5, 0.5) * 1e-07 + body_inertia[1:]
        #
        # mass randomization
        body_mass = self.body_mass.copy()

        physics.named.model.body_mass[1:] = np.random.uniform(-0.0005, 0.0005) + body_mass[1:]

    def before_step(self, action: np.ndarray, physics) -> None:
        """
        Sets the control signal for the actuators to values in `action`.
            applies 3d force to object goems
        """
        # Support legacy internal code.

        # clear previous xfrc_force
        physics.named.data.xfrc_applied[:, :3] = np.zeros((3,))

        if self._use_dr and not self._per_traj:
            self.apply_dr(physics)

        # scale the position to normal range and upscale to image size
        assert action.shape[0] == _FIXED_ACTION_DIMS
        d =_FIXED_ACTION_DIMS // 2
        if not self._random_pick:
            pick_location = (action[:2] * 0.5 + 0.5) * (W - 1)
            pick_location = np.round(pick_location).astype('int32')
        else:
            pick_location = self.current_loc

        # oddly enough, [0,2) are in pixel space, while the rest are in world coord frame?
        # TODO: treat entire vector as pixel space vector; i.e., formally convert these dims to world
        delta_position = action[d:d + 3]
        delta_position = delta_position * 0.1

        # project cloth points into image
        cam_fovy = physics.named.model.cam_fovy['fixed']
        f = 0.5 * W / math.tan(cam_fovy * math.pi / 360)
        cam_matrix = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]])
        cam_mat = physics.named.data.cam_xmat['fixed'].reshape((3, 3))
        cam_pos = physics.named.data.cam_xpos['fixed'].reshape((3, 1))
        cam = np.concatenate([cam_mat, cam_pos], axis=1)
        geoms_in_cam = np.zeros((81, 3, 1)) # assuming 9x9 geom mesh
        for i in range(81):
            geom_name = i
            geoms_in_world = np.concatenate([physics.named.data.geom_xpos[geom_name], np.array([1])]).reshape((4, 1))
            geoms_in_cam[i] = cam_matrix.dot(cam.dot(geoms_in_world)[:3])

        geoms_uv = np.rint(geoms_in_cam[:, :2].reshape((81, 2)) / geoms_in_cam[:, 2])
        geoms_d = geoms_in_cam[:, 2]
        geoms_uv = geoms_uv.astype(int)
        geoms_d = geoms_d.astype(int)

        # move origin to top left corner with +y oriented downward
        # move origin to bottom left corner?
        geoms_uv[:, 1] = W - geoms_uv[:, 1]
        geoms_uv[:, [0, 1]] = geoms_uv[:, [1, 0]]

        # select closest geom point projection to pick point in image space
        # hyperparameter epsilon=3(selecting joint in (2*eps)^2 box around pick pel)
        epsilon = 2
        possible_index = []
        possible_z = []
        for i in range(81):
            du = abs(geoms_uv[i][0] - pick_location[0])
            dv = abs(geoms_uv[i][1] - pick_location[1])
            if du < epsilon and dv < epsilon:
                possible_index.append(i)
                possible_z.append(physics.data.geom_xpos[i, 2])

        if possible_index != []:
            index = possible_index[possible_z.index(max(possible_z))]

            corner_action = index
            corner_geom = index

            # apply consecutive force to move the point to the target position
            target_xpos = delta_position + physics.named.data.geom_xpos[corner_geom]
            dist = target_xpos - physics.named.data.geom_xpos[corner_geom]

            # TODO: have a look at possible z, z target pos and z goem pos
            loop = 0
            while np.linalg.norm(dist) > 0.025:
                loop += 1
                if loop > 40:
                    break
                physics.named.data.xfrc_applied[corner_action, :3] = dist * 200
                physics.step()
                self.after_step(physics)
                dist = target_xpos - physics.named.data.geom_xpos[corner_geom]

    def get_observation(self, physics) -> dict:
        """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()

        image = self.get_image(physics)
        self.image = image
        self.current_loc = self.sample_location(physics) if self._random_pick else np.array([-1, 1])
        obs['pick_location'] = np.tile(self.current_loc, 50).reshape(-1).astype('float32') / (W - 1)

        # generate random action as unif(0,1) * (hi - lo) + lo
        random_action = np.random.rand(_FIXED_ACTION_DIMS,) * 2 - 1
        random_action[:2] = self.current_loc.astype('float32') / (W - 1)
        random_action[2] = 0.  # todo: would be nice to use depth data
        obs['action_sample'] = random_action

        return obs

    def get_image(self, physics) -> np.ndarray:
        render_kwargs = {}
        render_kwargs['camera_id'] = 0
        render_kwargs['width'] = W
        render_kwargs['height'] = W
        image = physics.render(**render_kwargs)
        return image

    def segment_image(self, image: np.ndarray) -> np.ndarray:
        image_dim_1 = image[:, :, [1]]
        image_dim_2 = image[:, :, [2]]
        mask = np.all(image> 200, axis=2) + np.all(image_dim_2 < 40, axis=2) + \
               (~np.all(image_dim_1 > 135, axis=2))
        return mask > 0

    def get_geoms(self, physics) -> np.ndarray:
        geoms = np.array([[physics.named.data.geom_xpos[f'G{i}_{j}', :2]
                           for j in range(9)]
                           for i in range(9)], dtype='float32')
        return geoms

    def sample_location(self, physics) -> np.ndarray:
        image = self.image
        location_range = np.transpose(np.where(self.segment_image(image)))
        num_loc = np.shape(location_range)[0]
        if num_loc == 0:
            return np.array([-1, -1])
        else:
            index = np.random.randint(num_loc)
            pick_location = location_range[index]

        return pick_location

    def get_reward(self, physics) -> float:
        current_mask = self.segment_image(self.image).astype(int)
        area = np.sum(current_mask * self.mask)
        reward = area / (np.sum(self.mask) + 1)
        return reward
