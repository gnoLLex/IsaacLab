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

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.motion_generation")
from omni.isaac.lab.controllers.config.rmp_flow import FRANKA_RMPFLOW_CFG
from omni.isaac.lab.controllers.rmp_flow import RmpFlowController

from omni.isaac.lab_assets import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG
import omni.physics.tensors.impl.api as physx

from enum import Enum

import time, os
import torch
import theseus as th
from torchlie.functional import SE3 as SE3_Func
from torchkin.forward_kinematics import Robot, get_forward_kinematics_fns

def normalize_quat_in_pos_quat(pos_quat):
    quaternions = pos_quat[:, 3:]
    norms = torch.norm(quaternions, dim=1, keepdim=True)
    pos_quat[:, 3:] = quaternions / norms
    return pos_quat

# x_y_z_quat
hardcore_test_ee_poses = normalize_quat_in_pos_quat(torch.tensor(
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
))

#test_ee_poses = torch.tensor(
#    [
#        [0.305, -0.102, 0.548, 0.102, 0.221, 0.462, 0.843],
#        [0.312, 0.019, 0.536, 0.151, 0.431, 0.276, 0.839],
#        [0.271, -0.063, 0.491, -0.217, 0.561, 0.145, 0.788],
#        [0.353, -0.045, 0.516, 0.036, 0.688, 0.293, 0.664],
#        [0.285, 0.013, 0.470, 0.245, 0.531, -0.319, 0.773],
#        [0.318, -0.084, 0.502, 0.105, 0.380, 0.532, 0.755],
#        [0.337, -0.129, 0.474, 0.268, 0.490, -0.377, 0.738],
#        [0.309, 0.054, 0.515, 0.111, 0.602, 0.309, 0.725],
#        [0.328, -0.091, 0.462, 0.167, 0.423, 0.563, 0.692],
#        [0.354, -0.034, 0.528, 0.053, 0.641, 0.427, 0.634],
#        [0.289, -0.142, 0.455, 0.321, 0.549, 0.366, 0.718],
#        [0.312, 0.010, 0.503, 0.083, 0.553, 0.275, 0.786],
#        [0.336, -0.065, 0.472, 0.129, 0.438, 0.541, 0.703],
#        [0.318, 0.087, 0.483, 0.168, 0.394, 0.426, 0.766],
#        [0.307, -0.120, 0.492, 0.249, 0.493, 0.281, 0.792],
#        [0.298, 0.024, 0.537, 0.198, 0.451, 0.279, 0.832],
#        [0.334, -0.058, 0.502, 0.293, 0.569, 0.312, 0.741],
#        [0.347, -0.129, 0.470, 0.138, 0.562, 0.419, 0.704],
#        [0.332, 0.091, 0.494, 0.253, 0.421, 0.536, 0.693],
#        [0.317, -0.054, 0.502, 0.177, 0.381, 0.465, 0.747],
#        [0.336, -0.109, 0.475, 0.209, 0.465, 0.389, 0.753],
#        [0.310, -0.032, 0.539, 0.152, 0.412, 0.412, 0.749],
#        [0.323, -0.086, 0.510, 0.196, 0.518, 0.401, 0.749],
#        [0.339, 0.019, 0.469, 0.269, 0.391, 0.283, 0.813]
#    ],
#    dtype=torch.float64,
#    device="cuda"
#)
test_ee_poses = normalize_quat_in_pos_quat(torch.tensor(
    [
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.1, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.2, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.3, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.4, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.6, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.7, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.8, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.9, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.9, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.8, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.7, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.6, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.4, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.3, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.2, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.1, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
    device="cuda"
))


test_ee_poses_se3 = th.SE3.x_y_z_unit_quaternion_to_SE3(
    hardcore_test_ee_poses,
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
    franka: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/franka"
    )

class IK_Solver(Enum):
    LevenbergMarquardt = 1
    Jacobian = 2


class FrankaPanda:
    """
    INFO: all pos_quat are of form: [x, y, z, w, x, y, z]
    """
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
 
        print(f"loading theseus from {FRANKA_RMPFLOW_CFG.urdf_file}")
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

    
    def initialize(self):
        # reset joint state
        joint_pos = self._isaac_robot.data.default_joint_pos.clone()
        joint_vel = self._isaac_robot.data.default_joint_vel.clone()
        self._isaac_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._isaac_robot.reset()

        self._scene.write_data_to_sim()
        self._sim.step()
        self._scene.update(self.sim_dt)

    @property
    def dof(self):
        return self._theseus_robot.dof

    @property
    def joint_position(self):
        return self._isaac_robot.data.joint_pos.to(dtype=self.dtype, device=self.device)

    @property
    def ee_pos_quat(self):
        return self.forward_kinematics(self.joint_position)

    def get_random_theta(self, B=None):
        if B == None:
            B = self._scene.num_envs
        return torch.rand(B, self.dof, dtype=self.dtype)

    def get_random_ee_pos_quat(self, B=None):
        if B == None:
            B = self._scene.num_envs
        
        positions = torch.empty(B, 3, dtype=self.dtype, device=self.device).uniform_(0.4, 0.6) # x
        positions[:, 1] = torch.empty(B, dtype=self.dtype, device=self.device).uniform_(-0.1, 0.1) # y
        positions[:, 2] = torch.empty(B, dtype=self.dtype, device=self.device).uniform_(0.3, 0.7) # z

        quaternions = torch.randn(B, 4, dtype=self.dtype, device=self.device) # (B, [w, x, y, z])
        quaternions /= torch.norm(quaternions, dim=1, keepdim=True)
        
        pos_quat = torch.cat([positions, quaternions], dim=1)
        return pos_quat # th.SE3.randn(B, dtype=self.dtype, device=self.device).to_x_y_z_quaternion()

    def forward_kinematics(self, theta):
        return th.SE3(tensor=self._fk(theta)[0]).to_x_y_z_quaternion()

    def spatial_jacobian(self, theta=None):
        if theta is None:
            theta = self.joint_position
        return self._jfk_s(theta)[0][0]

    def body_jacobian(self, theta=None):
        if theta is None:
            theta = self.joint_position
        return self._jfk_b(theta)[0][0]

    def _inverse_kinematics_levenberg_marquardt(
        self,
        pos_quat: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        pose = th.SE3.x_y_z_unit_quaternion_to_SE3(pos_quat).tensor

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
            max_iterations=20,
            step_size=0.5,
            vectorize=True,
        )

        inputs = {
            # "theta_opt": torch.zeros_like(theta_opt, dtype=self.dtype, device=self.device),
            "theta_opt": self.joint_position,
            "targeted_pose": pose,
        }
        optimizer.objective.update(inputs)
        optimizer.optimize(verbose=False)#verbose)
        theta = optim_vars[0].tensor

        return theta

    def _inverse_kinematics_jacobian(
        self,
        pos_quat: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        step_size = 0.2
        def compute_delta_theta(jfk, theta, targeted_pose, use_body_jacobian):
            jac, poses = jfk(theta)
            pose_inv = SE3_Func.inv(poses[-1])
            error = (
                SE3_Func.log(
                    SE3_Func.compose(pose_inv, targeted_pose)
                    if use_body_jacobian
                    else SE3_Func.compose(targeted_pose, pose_inv)
                )
                .view(-1, 6, 1)
                .view(-1, 6, 1)
            )
            return (jac[-1].pinverse() @ error).view(-1, self.dof), error.norm().item()

        target_pose = th.SE3.x_y_z_unit_quaternion_to_SE3(pos_quat).tensor
        # theta_opt = torch.zeros_like(self.joint_position)
        theta_opt = self.joint_position
        for _ in range(50):
            delta_theta, error = compute_delta_theta(
                self._jfk_s, theta_opt, target_pose, False
            )
            if error < 1e-4:
                break
            theta_opt = theta_opt + step_size * delta_theta
        
        return theta_opt

    def inverse_kinematics(
        self,
        pos_quat: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        verbose: bool = False,
        ik_solver: IK_Solver = IK_Solver.LevenbergMarquardt,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_quat.ndim == 1:
            pos_quat.unsqueeze(-1)

        ik_solver_fn = None

        if ik_solver == IK_Solver.LevenbergMarquardt:
            ik_solver_fn = self._inverse_kinematics_levenberg_marquardt
        elif ik_solver == IK_Solver.Jacobian:
            ik_solver_fn = self._inverse_kinematics_jacobian


        start = time.time()
        computed_q = ik_solver_fn(pos_quat, verbose)
        if verbose:
            ms_elapsed = (time.time() - start) * 1000
            print(f"Franka IK: {ik_solver} took {ms_elapsed:04} ms")

        computed_pos_quat = self.forward_kinematics(computed_q)

        #print(f"IK difference: {computed_pos_quat - pos_quat}")

        is_close = torch.isclose(computed_pos_quat, pos_quat, rtol=rtol, atol=atol)
        success = torch.all(is_close, dim=-1)

        return computed_q, success

    def goto_joint_position(self, joint_position, max_steps=200, verbose=True):
        self._isaac_robot.set_joint_position_target(joint_position)
        self._isaac_robot.set_joint_velocity_target(torch.zeros_like(joint_position))

        for i in range(max_steps):
            if torch.allclose(joint_position, self.joint_position):
                return True

            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(self.sim_dt)
        
        return False
        

    def goto_ee_pos_quat(self, pos_quat, max_steps=200, verbose=False):
        target_q, s = self.inverse_kinematics(pos_quat, ik_solver=IK_Solver.LevenbergMarquardt, verbose=verbose)
        self._isaac_robot.set_joint_position_target(target_q)

        for i in range(max_steps):
            all_close = torch.allclose(pos_quat, self.ee_pos_quat)
            if all_close:
                return True
            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(self.sim_dt)
        return s
    
    def open_gripper(self):
        self._isaac_robot.set_joint_effort_target(4, joint_ids=self.finger_ids)
        self._scene.write_data_to_sim()

    def close_gripper(self):
        self._isaac_robot.set_joint_effort_target(-20, joint_ids=self.finger_ids)
        self._scene.write_data_to_sim()
    
    def let_run(self, steps=100):
        for i in range(steps):
            current_joint_pos = self._isaac_robot.data.joint_pos.clone()
            self._isaac_robot.set_joint_position_target(current_joint_pos)
            self._isaac_robot.set_joint_velocity_target(torch.zeros_like(current_joint_pos))

            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(self.sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((3.0, 0.0, 1), (0.0, 0.0, 0.0))

    scene = InteractiveScene(
        cfg=FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    )
    sim.reset()

    robot = FrankaPanda(scene["franka"], sim, scene)
    robot.initialize()

    for r in range(10):
        #print(robot.goto_ee_pos_quat(robot.get_random_ee_pos_quat(), max_steps=200, verbose=True))
        #print(robot.goto_ee_pos_quat(test_ee_poses, max_steps=200, verbose=True))
        if r % 2 == 0:
            robot.goto_ee_pos_quat(test_ee_poses[20].unsqueeze(0).repeat(scene.num_envs, 1), max_steps=200, verbose=True)
        else:
            robot.goto_ee_pos_quat(test_ee_poses[0].unsqueeze(0).repeat(scene.num_envs, 1), max_steps=200, verbose=True)
    robot.close_gripper()
    #robot.goto_pos_vel(pose.unsqueeze(0).repeat(scene.num_envs, 1))
    print("### LOOOPING ###")
    while True:
        robot.let_run()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
