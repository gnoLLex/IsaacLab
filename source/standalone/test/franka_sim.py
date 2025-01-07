import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Franka")
parser.add_argument(
    "--num_envs", type=int, default=64, help="Number of environments to spawn."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import GroundPlaneCfg, DomeLightCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.controllers.config.rmp_flow import FRANKA_RMPFLOW_CFG
import omni.physics.tensors.impl.api as physx

import os
import torch
import theseus as th
from torchkin.forward_kinematics import Robot, get_forward_kinematics_fns

FRANKA_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

# x_y_z_quat
test_ee_poses = torch.tensor(
    [
        [0.6, -0.25, 0.15, 0.009, 0.72, -0.67, -0.014],
        [0.6, -0.25, 0.15, 0.7071, 0.7071, 0, 0],
        [0.6, -0.25, 0.15, 0.7071, 0.7071, 0, 0],
        [0.6, 0.25, 0.15, 0.009, 0.72, -0.67, -0.014],
        [0.6, 0.25, 0.15, 0.7071, 0.7071, 0, 0],
        [0.6, 0.25, 0.15, -0.7071, 0.7071, 0, 0],
        [0.6, 0.25, 0.65, 0.009, 0.72, -0.67, -0.014],
        [0.6, 0.25, 0.65, 0.7071, 0.7071, 0, 0],
        [0.6, 0.25, 0.65, -0.7071, 0.7071, 0, 0],
        [0.6, -0.25, 0.65, 0.009, 0.72, -0.67, -0.014],
        [0.6, -0.25, 0.65, 0.7071, 0.7071, 0, 0],
        [0.6, -0.25, 0.65, -0.7071, 0.7071, 0, 0],
        [0.35, -0.25, 0.15, 0.009, 0.72, -0.67, -0.014],
        [0.35, -0.25, 0.15, 0.7071, 0.7071, 0, 0],
        [0.35, -0.25, 0.15, -0.7071, 0.7071, 0, 0],
        [0.35, 0.25, 0.15, 0.009, 0.72, -0.67, -0.014],
        [0.35, 0.25, 0.15, 0.7071, 0.7071, 0, 0],
        [0.35, 0.25, 0.15, -0.7071, 0.7071, 0, 0],
        [0.35, 0.25, 0.45, 0.009, 0.72, -0.67, -0.014],
        [0.35, 0.25, 0.45, 0.7071, 0.7071, 0, 0],
        [0.35, 0.25, 0.45, -0.7071, 0.7071, 0, 0],
        [0.35, -0.25, 0.45, 0.009, 0.72, -0.67, -0.014],
        [0.35, -0.25, 0.45, 0.7071, 0.7071, 0, 0],
        [0.35, -0.25, 0.45, -0.7071, 0.7071, 0, 0],
    ],
    dtype=torch.float64,
    device="cuda"
)

test_ee_poses_se3 = th.SE3.x_y_z_unit_quaternion_to_SE3(
    test_ee_poses,
).tensor


def _min_jerk_spaces(
    N: int, T: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert N > 1, "Number of planning steps must be larger than 1."

    t_traj = torch.linspace(0, 1, N)
    p_traj = 10 * t_traj**3 - 15 * t_traj**4 + 6 * t_traj**5
    pd_traj = (30 * t_traj**2 - 60 * t_traj**3 + 30 * t_traj**4) / T
    pdd_traj = (60 * t_traj - 180 * t_traj**2 + 120 * t_traj**3) / (T**2)

    return p_traj, pd_traj, pdd_traj


def _compute_num_steps(time_to_go: float, hz: float):
    return int(time_to_go * hz)


def generate_joint_space_min_jerk(
    start: torch.Tensor, goal: torch.Tensor, time_to_go: float, hz: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = _compute_num_steps(time_to_go, hz)
    dt = 1.0 / hz

    p_traj, pd_traj, pdd_traj = _min_jerk_spaces(steps, time_to_go)

    D = goal - start
    q_traj = start[None, :] + D[None, :] * p_traj[:, None]
    qd_traj = D[None, :] * pd_traj[:, None]
    qdd_traj = D[None, :] * pdd_traj[:, None]

    time_from_start = torch.range(0, steps) * dt
    position = q_traj[:steps, :]
    velocity = qd_traj[:steps, :]
    acceleration = qdd_traj[:steps, :]

    return time_from_start, position, velocity, acceleration


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    franka: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/franka"
    )


class FrankaPanda:
    # TODO: torch device
    _isaac_robot: Articulation
    _theseus_robot: Robot
    _sim: sim_utils.SimulationContext

    def __init__(self, robot, sim: sim_utils.SimulationContext, scene, dtype=torch.float64):
        self._isaac_robot = robot
        self._sim = sim
        self._scene = scene
        self.dtype = dtype
        self.device = self._sim.cfg.device

        self.joint_ids, self.joint_names = self._isaac_robot.find_joints("panda_joint.*")
        self.finger_ids, self.finger_names = self._isaac_robot.find_joints("panda_finger_joint.*")

        self._theseus_robot = Robot.from_urdf_file(
            FRANKA_RMPFLOW_CFG.urdf_file, self.dtype, self.device
        )
        self._fk, self._jfk_b, self._jfk_s = get_forward_kinematics_fns(
            self._theseus_robot, ["panda_hand"]
        )

        self._joint_vel_limit = torch.tensor(
            [2.00, 2.00, 2.00, 2.00, 2.50, 2.50, 2.50],
            dtype=self.dtype,
            device=self.device,
        )
        self._joint_pos_min = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=self.dtype,
            device=self.device,
        )
        self._joint_pos_max = torch.tensor(
            [2.8973, 1.7628, 2.0, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=self.dtype,
            device=self.device,
        )


        self.sim_dt = self._sim.get_physics_dt()

    @property
    def dof(self):
        return self._theseus_robot.dof

    @property
    def joint_position(self):
        return self._isaac_robot.data.joint_pos.to(dtype=self.dtype, device=self.device)

    @property
    def ee_pose(self):
        return self.forward_kinematics(self.joint_position)

    def get_random_theta(self, B):
        return torch.rand(B, self.dof, dtype=self.dtype)

    def get_random_ee_pose(self, B):
        return th.SE3.randn(B, dtype=self.dtype, device=self.device).tensor

    def forward_kinematics(self, theta):
        return self._fk(theta)[0]

    def spatial_jacobian(self, theta=None):
        if theta is None:
            theta = self.joint_position
        return self._jfk_s(theta)[0][0]

    def body_jacobian(self, theta=None):
        if theta is None:
            theta = self.joint_position
        return self._jfk_s(theta)[0][0]

    def inverse_kinematics(
        self,
        pose: th.SE3,
        rtol: float = 1e-8,
        atol: float = 1e-3,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pose.ndim == 1:
            pose.unsqueeze(-1)

        def targeted_pose_error(optim_vars, aux_vars):
            (theta,) = optim_vars
            (targeted_pose,) = aux_vars
            pose = th.SE3(tensor=self._fk(theta.tensor)[0])
            return pose.local(targeted_pose)

        theta_opt = torch.zeros(
            pose.shape[0], self._theseus_robot.dof, dtype=self.dtype
        )
        optim_vars = (
            th.Vector(
                tensor=torch.zeros_like(
                    theta_opt, dtype=self.dtype, device=self.device
                ),
                name="theta_opt",
            ),
        )
        aux_vars = (th.SE3(tensor=pose, name="targeted_pose"),)

        cost_function = th.AutoDiffCostFunction(
            optim_vars,
            targeted_pose_error,
            6,
            aux_vars=aux_vars,
            name="targeted_pose_error",
            cost_weight=th.ScaleCostWeight(
                torch.ones([1], dtype=self.dtype, device=self.device)
            ),
            autograd_mode="vmap",
        )
        objective = th.Objective(dtype=self.dtype)
        objective.add(cost_function)
        optimizer = th.LevenbergMarquardt(
            objective.to(self.device),
            max_iterations=15,
            step_size=0.5,
            vectorize=True,
        )

        inputs = {
            "theta_opt": torch.zeros_like(
                theta_opt, dtype=self.dtype, device=self.device
            ),
            "targeted_pose": pose,
        }
        optimizer.objective.update(inputs)
        optimizer.optimize(verbose=verbose)
        theta = optim_vars[0].tensor

        achieved_pose = self.forward_kinematics(theta)
        is_close = torch.isclose(achieved_pose, pose, rtol=rtol, atol=atol)
        success = torch.all(is_close, dim=(1, 2))

        return theta, success

    def move_to_joint_theta(self, theta, max_steps=200):
        self._isaac_robot.set_joint_position_target(theta)
        for _ in range(max_steps):
            self._scene.write_data_to_sim()
            # perform step
            self._sim.step()
            # update buffers
            self._scene.update(self.sim_dt)

    def move_to_ee_pose(
        self,
        pose,
        max_steps=200,
        verbose: bool = False,
    ):
        target_theta, success = self.inverse_kinematics(pose, verbose=verbose)
        self.move_to_joint_theta(target_theta, max_steps=max_steps)
    
    def set_robot_joint_theta(self, theta):
        self._isaac_robot.write_joint_state_to_sim(theta, torch.zeros_like(theta))
        self._isaac_robot.set_joint_position_target(theta)
        self._isaac_robot.set_joint_velocity_target(torch.zeros_like(theta))
        for _ in range(10):
            self._scene.write_data_to_sim()
            # perform step
            self._sim.step()
            # update buffers
            self._scene.update(self.sim_dt)
    
    def set_robot_ee_pose(self, pose, verbose:bool = False):
        target_theta, success = self.inverse_kinematics(pose, verbose=verbose)
        self.set_robot_joint_theta(target_theta)

    def open_gripper(self):
        self._isaac_robot.set_joint_effort_target(torch.Tensor(4, device=self.device), joint_ids=self.finger_ids)
        self._scene.write_data_to_sim()

    def close_gripper(self):
        self._isaac_robot.set_joint_effort_target(torch.tensor(-20, self.device), joint_ids=self.finger_ids)
        self._scene.write_data_to_sim()


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.0, 0.0, 3.2), (0.0, 0.0, 0.5))

    scene = InteractiveScene(
        cfg=FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    )
    sim.reset()

    robot = FrankaPanda(scene["franka"], sim, scene)

    #target_ee_pose = robot.get_random_ee_pose(scene.num_envs)
    #robot.move_to_ee_pose(target_ee_pose)
    print(robot.spatial_jacobian().shape)
    print(robot.body_jacobian().shape)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
