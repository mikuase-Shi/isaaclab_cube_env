import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg,AssetBaseCfg,RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm 
from isaaclab.managers import RewardTermCfg as RewTerm  
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm    
from isaaclab.scene import InteractiveSceneCfg  
from isaaclab.sensors import FrameTransformerCfg    
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import math
from . import mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

@configclass
class PushSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg=FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame=FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(
                    pos=(0.0,0.0,0.1034),
                    #learn this from 
                    #https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/joint_pos_env_cfg.py
                ),
            ),
        ],
    )

    object:RigidObjectCfg=RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5,0.0,0.15),
            rot=(1.0,0.0,0.0,0.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.15,0.15,0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            #give this cube a different height to distinguish 6 faces
            #it may help in the reward function designing
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1,0.8,0.1)),
        ),
    )

    goal: RigidObjectCfg = RigidObjectCfg(
     prim_path="{ENV_REGEX_NS}/Goal",
     init_state=RigidObjectCfg.InitialStateCfg(
         pos=(0.7, 0.0, 0.011), 
         rot=(1.0, 0.0, 0.0, 0.0)
        ),
     spawn=sim_utils.CuboidCfg(
         size=(0.15, 0.15, 0.01),
         collision_props=sim_utils.CollisionPropertiesCfg(
             collision_enabled=False
            ),
         rigid_props=sim_utils.RigidBodyPropertiesCfg(
             kinematic_enabled=True,
             disable_gravity=True
            ),
         mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
         visual_material=sim_utils.PreviewSurfaceCfg(
             diffuse_color=(0.0, 0.8, 0.0)
            ),
        ),
    )

    table=AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.7,0.0,0.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.2, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                #changable parameters,depend on inferences
                static_friction=0.5,
                dynamic_friction=0.5,
                ###
                restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5,0.3,0.1)),
        ),
    )

    plane=AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

@configclass
class ActionsCfg:
    arm_action:mdp.JointPositionActionCfg=mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.2,
        use_default_offset=True,
        #from https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/cabinet/config/franka/joint_pos_env_cfg.py
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos=ObsTerm(func=mdp.joint_pos_rel)
        joint_vel=ObsTerm(func=mdp.joint_vel_rel)
        actions=ObsTerm(func=mdp.last_action)

        rel_ee_object_distance=ObsTerm(func=mdp.rel_ee_object_distance)
        object_pos=ObsTerm(func=mdp.object_local_pos_obs, params={"asset_cfg":SceneEntityCfg("object")})
        object_quat=ObsTerm(func=mdp.root_quat_w,params={"asset_cfg":SceneEntityCfg("object")})

        object_to_goal_pos=ObsTerm(func=mdp.object_to_goal_pos_obs)
        object_to_goal_quat=ObsTerm(func=mdp.object_to_goal_quat_obs)

        def __post_init__(self):
            self.enable_corruption=True
            self.concatenate_terms=True
        
    policy:PolicyCfg=PolicyCfg()

@configclass
class EventCfg:
    reset_robot_joints=EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "position_range":(-0.05,0.05),
            "velocity_range":(0.0,0.0),
        },
    )
    reset_object_position=EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg":SceneEntityCfg("object"),
            "pose_range":{
                "x":(-0.1,0.1),
                "y":(-0.1,0.1),
                "roll":(-math.pi/4, math.pi/4), 
                "pitch":(-math.pi/4, math.pi/4),
                "yaw":(-math.pi, math.pi)
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
        },
    )
    reset_goal_position=EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg":SceneEntityCfg("goal"),
            "pose_range": {
             "x": (-0.3, 0.15),
             "y": (-0.45, 0.45),
             "z": (0.011, 0.011),
             "roll": (0.0, 0.0), 
             "pitch": (0.0, 0.0),
             "yaw": (-math.pi/2, math.pi/2)
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            },
        },
    )


@configclass
class RewardsCfg:
    reaching_reward = RewTerm(func=mdp.ms_reaching_reward, weight=5.0)

    goal_reaching_reward = RewTerm(func=mdp.ms_phased_goal_reward, weight=10.0)
    stationary_reward = RewTerm(func=mdp.ms_stationary_reward, weight=8.0)
    goal_pos_x_reward = RewTerm(func=mdp.ms_goal_pos_x_reward, weight=5.0)
    goal_pos_y_reward = RewTerm(func=mdp.ms_goal_pos_y_reward, weight=5.0)

    near_goal_vel_penalty = RewTerm(func=mdp.ms_near_goal_vel_penalty, weight=-2.0)
    overshoot_penalty = RewTerm(func=mdp.ms_overshoot_penalty, weight=-5.0)
    past_goal_penalty = RewTerm(func=mdp.ms_past_goal_penalty, weight=-8.0)

    z_stability_reward = RewTerm(func=mdp.ms_z_reward, weight=2.0)

    action_rate_penalty = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel_penalty = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_falling = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.1, "asset_cfg": SceneEntityCfg("object")}
    )
    task_success = DoneTerm(
        func=mdp.object_reached_goal,
        params={"pos_threshold": 0.008, "rot_threshold": 0.1}
    )


@configclass
class PushEnvCfg(ManagerBasedRLEnvCfg):
    scene: PushSceneCfg = PushSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2 
        self.episode_length_s = 8.0 
        self.viewer.eye = (1.5, 1.5, 1.5) 
        self.viewer.lookat = (0.5, 0.0, 0.0) 
        self.sim.dt = 1 / 120 
        self.sim.render_interval = self.decimation
        
        self.sim.physx.use_gpu = True