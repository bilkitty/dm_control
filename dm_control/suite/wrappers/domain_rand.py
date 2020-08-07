import dm_env
import numpy as np
from dm_control.suite.wrappers.modder import LightModder, CameraModder


class DomainRand(dm_env.Environment):
    """Wraps a control environment and adds a rendered pixel observation."""

    def __init__(self, env):
        """Initializes a new Domain Randomization Wrapper"""

        self.env = env
        self.physics = env.physics
        
        # Physics
        self.dof_damping = self.physics.named.model.dof_damping.copy()
        self.body_mass = self.physics.named.model.body_mass.copy()
        self.body_inertia = self.physics.named.model.body_inertia.copy()
        self.geom_friction = self.physics.named.model.geom_friction.copy()
        
        # Camera
        # self.camera_modder = CameraModder(self._env.physics)
        # self.cam_pos = np.array([0, 0, 0.75])
        # self.cam_quat = np.array([1, 0, 0, 0])

        # Lighting
        self.lightmodder = LightModder(self.physics)
        self.light_diffuse = self.lightmodder.get_diffuse('light').copy()
        self.light_specular = self.lightmodder.get_specular('light').copy()
        self.light_ambient = self.lightmodder.get_ambient('light').copy()
        self.light_castshadow = self.lightmodder.get_castshadow('light').copy()
        self.light_dir = self.lightmodder.get_dir('light').copy()
        self.light_pos = self.lightmodder.get_pos('light').copy()


    def reset(self):
        return self.env.reset()

    def step(self, action):
        # visual randomization

        # light randomization
        self.lightmodder.set_ambient('light', self.light_ambient + np.random.uniform(-0.5, 0.5, size=3))
        self.lightmodder.set_castshadow('light', self.light_castshadow + np.random.uniform(0, 40))
        self.lightmodder.set_diffuse('light', self.light_diffuse + np.random.uniform(-0.1, 0.1))
        self.lightmodder.set_dir('light', self.light_dir + np.random.uniform(-0.1, 0.1))
        self.lightmodder.set_pos('light', self.light_pos + np.random.uniform(-0.1, 0.1))
        self.lightmodder.set_specular('light', self.light_specular + np.random.uniform(-0.1, 0.1))

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

        # physics randomization

        # damping randomization
        self.physics.named.model.dof_damping[:] = np.random.uniform(0, 0.0001) + self.dof_damping

        # friction randomization
        self.physics.named.model.geom_friction[1:, 0] = np.random.uniform(-0.5, 0.5) + self.geom_friction[1:, 0]
        self.physics.named.model.geom_friction[1:, 1] = np.random.uniform(-0.002, 0.002) + self.geom_friction[1:, 1]
        self.physics.named.model.geom_friction[1:, 2] = np.random.uniform(-0.0005, 0.0005) + self.geom_friction[1:, 2]

        # inertia randomization
        self.physics.named.model.body_inertia[1:] = np.random.uniform(-0.5, 0.5) * 1e-05 + self.body_inertia[1:]

        # mass randomization
        self.physics.named.model.body_mass[1:] = np.random.uniform(-0.05, 0.05) + self.body_mass[1:]

        return self.env.step(action)

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def __getattr__(self, name):
        return getattr(self.env, name)
