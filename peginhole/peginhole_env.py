from collections import OrderedDict
import numpy as np
import os
import transformations as tr

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements
from robosuite import load_controller_config

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena, EmptyArena
from robosuite.models.objects import BoxObject
from robosuite.models.objects import CylinderObject, PlateWithHoleObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.wrappers.visualization_wrapper import VisualizationWrapper

from scipy.spatial.transform import Rotation as R
from objects import *

class PeginHole(SingleArmEnv):
    """
    This class corresponds to the peg-in-hole task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            table_full_size=(0.8, 0.8, 0.05),
            table_friction=(1., 5e-3, 1e-4),
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            peg_class=0,
            threshold=0.15
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.threshold = threshold
        
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        
        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        
        # object placement initializer
        self.placement_initializer = placement_initializer
        
        self.peg_class = peg_class
        
        self.headless = has_offscreen_renderer
        
        # set up controller, seen in peginhole_controller.json
        # Operational Space Control with variable stiffness is used as default
        ctrl_fpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "peginhole_controller.json")
        controller_configs = load_controller_config(custom_fpath=ctrl_fpath)
        # controller_configs = None
        # import pdb; pdb.set_trace()
        
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )
    
    def reward(self, _action=None):
        """
        Reward function for the task.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            _action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        cr = 1
        lmbd = 5
        ca = 5
        eps = 0.05
        eps_d = 5e-3
        eps_h = 5e-3
        _pos_to_hole = self.get_peg_pos_to_hole()
        r_s = np.linalg.norm(_pos_to_hole)
        r_xy = np.linalg.norm([_pos_to_hole[0], _pos_to_hole[1]])
        if self._check_success():
            reward = 10
        elif np.abs(_pos_to_hole[0]) <= eps_h and np.abs(_pos_to_hole[1]) <= eps_h and _pos_to_hole[2] <= 0:
            reward = 4 - 2*_pos_to_hole[2]/(self.threshold - eps_d)
        elif r_xy <= eps:
            reward = 2 - ca*r_xy
        else:
            reward = cr * (1 - (np.tanh(lmbd * r_s) + np.tanh(lmbd * r_xy))/2)
        
        if self.reward_scale:
            reward *= self.reward_scale/10
        
        return reward
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        # load model for table top workspace
        # mujoco_arena = TableArena(
        #     table_full_size=self.table_full_size,
        #     table_friction=self.table_friction,
        #     table_offset=self.table_offset,
        # )
        mujoco_arena = EmptyArena()
        
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        
        # initialize objects of interest
        self.peg_length = 0.08  # half length of peg
        self.hole_depth = 0.1  # depth from hole surface to object center
        
        # determine the clearance of the peg
        self.peg = Peg(name='peg', peg_class=self.peg_class)
        self.hole = Hole(name='hole', peg_class=self.peg_class)

        # load hole object
        hole_obj = self.hole.get_obj()
        hole_obj.set("pos", "-0.05 0 0.70")
        
        # load peg object
        peg_obj = self.peg.get_obj()
        peg_obj.set("pos", array_to_string((0, 0, self.peg_length)))
        
        # Append peg to robot ee
        robot_eef = self.robots[0].robot_model.eef_name
        robot_model = self.robots[0].robot_model
        robot_body = find_elements(robot_model.worldbody, tags="body", attribs={"name": robot_eef}, return_first=True)
        robot_body.append(peg_obj)
        
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.hole,
        )
        
        self.model.merge_assets(self.peg)
    
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        
        # Additional object references from this env
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
    
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        
        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            
            # peg hole-related observables
            @sensor(modality=modality)
            def peg_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.peg_body_id])
            
            @sensor(modality=modality)
            def peg_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.peg_body_id]), to="xyzw")
            
            @sensor(modality=modality)
            def hole_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.hole_body_id])
            
            @sensor(modality=modality)
            def hole_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.hole_body_id]), to="xyzw")
            
            sensors = [peg_pos, peg_quat]
            names = [s.__name__ for s in sensors]
            
            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        
        return observables
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
    
    def _check_success(self):
        """
        Check if peg is inserted into the hole.
        """
        # cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        # table_height = self.model.mujoco_arena.table_offset[2]
        
        # cube is higher than the table top above a margin
        # return cube_height > table_height + 0.04
        _pos_to_hole = self.get_peg_pos_to_hole()
        eps_h = 5e-3
        eps_d = 5e-3
        if np.abs(_pos_to_hole[0]) <= eps_h and np.abs(_pos_to_hole[1]) <= eps_h and self.threshold - eps_d <= -_pos_to_hole[2]:
            return True
        else:
            return False
    
    def get_robot_pose_6d(self):
        """
        Return the current robot pose in 6d (position + euler) in world frame
        """
        obs = self._get_observations()
        robot_pos = obs["robot0_eef_pos"]
        robot_quat = obs["robot0_eef_quat"]
        robot_euler = quat2axisangle(robot_quat)
        robot_6d_pose = np.hstack((robot_pos, robot_euler))
        return robot_6d_pose
    
    def get_peg_pos_to_hole(self):
        _peg_vec = np.array([0, 0, self.peg_length])
        _peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_id])
        _peg_quat = convert_quat(np.array(self.sim.data.body_xquat[self.peg_body_id]), to="xyzw")
        _peg_bottom_pos = R.from_quat(_peg_quat).as_matrix() @ _peg_vec + _peg_pos
        _hole_pos = np.array(self.sim.data.body_xpos[self.hole_body_id])
        _hole_pos[2] += self.hole_depth
        return _peg_bottom_pos - _hole_pos
        
        

if __name__ == "__main__":
    import pdb
    
    
    def get_policy_action(obs):
        low, high = env.action_spec
        return np.random.uniform(low, high)
    
    
    from robosuite.environments.manipulation.lift import Lift
    
    # env = Lift(robots="IIWA", initialization_noise=None, has_renderer=True, has_offscreen_renderer=False, use_camera_obs=False)
    env = PeginHole(robots=["Panda"], gripper_types=None, initialization_noise=None, has_renderer=True,
                    render_camera=None, has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True,
                    reward_scale=None, peg_class=1)
    env = VisualizationWrapper(env)
    obs = env.reset()
    done = False
    for i in range(1000):
        action = get_policy_action(obs)
        if env.env._check_success():
            action = np.zeros_like(action)
        obs, r, done, _ = env.step(action)
        print(env.env.get_peg_pos_to_hole())
        print(r)
        env.render()
        # pdb.set_trace()
    
    '''
    import numpy as np
    import robosuite as suite

    # create environment instance
    config = {"env_name": "Lift", "robots": "IIWA",}
    env = suite.make(**config, has_renderer=True, has_offscreen_renderer=False, use_camera_obs=False)

    # reset the environment
    env.reset()

    for i in range(1000):
        action = np.array([0,0,0,0,0,0,0,0])
        # action = np.random.randn(env.dof)  # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
    '''
