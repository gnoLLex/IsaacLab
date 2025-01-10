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

from enum import Enum, auto

import time, os
import torch
import theseus as th
from torchlie.functional import SE3 as SE3_Func
from torchkin.forward_kinematics import Robot, get_forward_kinematics_fns

import pinocchio as pin
import numpy as np


def spherical_distance_quat(quat_1: torch.Tensor, quat_2: torch.Tensor):
    assert quat_1.shape[-1] == 4, f"quaternion 1 should have 4 components, but has {quat_1.shape[-1]}"
    assert quat_2.shape[-1] == 4, f"quaternion 2 should have 4 components, but has {quat_2.shape[-1]}"
    dot = torch.sum(quat_1 * quat_2, dim=-1)
    return 1 - dot * dot


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


class IkSolver(Enum):
    LEVENBERG_MARQUARDT = auto()
    JACOBIAN = auto()
    PINOCCHIO_CPU = auto()


class FrankaPanda:
    """
    INFO: all pos_quat are of form: [x, y, z, w, x, y, z]
    TODO: constraint joints
    """
    _isaac_robot: Articulation
    _theseus_robot: Robot
    _sim: sim_utils.SimulationContext

    def __init__(self, robot, sim: sim_utils.SimulationContext, scene, dtype=torch.float64, verbose=False):
        self._isaac_robot = robot
        self._sim = sim
        self._scene = scene
        self.dtype = dtype
        self.device = self._sim.cfg.device
        self.sim_dt = self._sim.get_physics_dt()
        self.verbose = verbose

        self.joint_ids, self.joint_names = self._isaac_robot.find_joints("panda_joint.*")
        self.finger_ids, self.finger_names = self._isaac_robot.find_joints("panda_finger_joint.*")

        # Theseus
        print(FRANKA_RMPFLOW_CFG.urdf_file)
        self._theseus_robot = Robot.from_urdf_file(
            FRANKA_RMPFLOW_CFG.urdf_file, self.dtype, self.device
        )
        self._fk, self._jfk_b, self._jfk_s = get_forward_kinematics_fns(
            self._theseus_robot, ["panda_end_effector"]
        )

        # Pinocchio
        self.pin_model = pin.buildModelFromUrdf(FRANKA_RMPFLOW_CFG.urdf_file)
        self.pin_data = self.pin_model.createData()
        self.pin_end_effector_frame_id = self.pin_model.getFrameId("panda_end_effector", pin.FrameType.FIXED_JOINT)

        # Limits
        # Hardware Constraints from [Polymetis](https://github.com/intuitive-robots/irl_polymetis/blob/main/polymetis/conf/robot_client/franka_hardware.yaml)
        # Can be adjusted, but for sim to real these limits are suggested as they are enforced on real hardware
        self._cartesian_position_lower_limit = torch.tensor(
            [0.1, -0.4, -0.05],
            dtype=self.dtype,
            device=self.device,
        )
        self._cartesian_position_upper_limit = torch.tensor(
            [1.0, 0.4, 1.0],
            dtype=self.dtype,
            device=self.device,
        )

        # Physical Limits
        # https://frankaemika.github.io/docs/control_parameters.html#limits-for-panda
        self._joint_position_lower_limit = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            dtype=self.dtype,
            device=self.device,
        )
        self._joint_position_upper_limit = torch.tensor(
            [2.8973, 1.7628, 2.0, -0.0698, 2.8973, 3.7525, 2.8973],
            dtype=self.dtype,
            device=self.device,
        )
        self._joint_velocity_limit = torch.tensor(
            [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
            dtype=self.dtype,
            device=self.device,
        )

    def reset(self):
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
        if B is None:
            B = self._scene.num_envs
        return torch.rand(B, self.dof, dtype=self.dtype)

    def get_random_ee_pos_quat(self, B=None):
        if B is None:
            B = self._scene.num_envs

        positions = torch.empty(B, 3, dtype=self.dtype, device=self.device).uniform_(0.4, 0.6)  # x
        positions[:, 1] = torch.empty(B, dtype=self.dtype, device=self.device).uniform_(-0.1, 0.1)  # y
        positions[:, 2] = torch.empty(B, dtype=self.dtype, device=self.device).uniform_(0.3, 0.7)  # z

        quaternions = torch.randn(B, 4, dtype=self.dtype, device=self.device)  # (B, [w, x, y, z])
        quaternions /= torch.norm(quaternions, dim=1, keepdim=True)

        pos_quat = torch.cat([positions, quaternions], dim=1)
        return pos_quat  # th.SE3.randn(B, dtype=self.dtype, device=self.device).to_x_y_z_quaternion()

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
            q: torch.Tensor,
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
            "theta_opt": q,
            "targeted_pose": pose,
        }
        optimizer.objective.update(inputs)
        optimizer.optimize(verbose=False)
        theta = optim_vars[0].tensor

        return theta

    def _inverse_kinematics_jacobian(
            self,
            pos_quat: torch.Tensor,
            q: torch.Tensor,
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
                .view(-1, 6, 1)  # TODO: ???? randomly appears in theseus example
            )
            return (jac[-1].pinverse() @ error).view(-1, self.dof), error.norm().item()

        target_pose = th.SE3.x_y_z_unit_quaternion_to_SE3(pos_quat).tensor
        theta_opt = q
        for _ in range(50):
            delta_theta, error = compute_delta_theta(
                self._jfk_s, theta_opt, target_pose, False
            )
            if error < 1e-4:
                break
            theta_opt = theta_opt + step_size * delta_theta

        return theta_opt

    def _inverse_kinematics_pinocchio_cpu(
            self,
            pos_quat: torch.Tensor,
            q: torch.Tensor,
    ) -> torch.Tensor:
        qs = q.cpu().numpy()

        computed_qs = qs.copy()

        for q_i in range(qs.shape[0]):
            q = qs[q_i]
            target_pos_quat = pos_quat[q_i]

            eps = 5e-4
            IT_MAX = 1000
            DT = 1e-1
            damp = 1e-12

            oMdes = th.SE3.x_y_z_unit_quaternion_to_SE3(target_pos_quat)
            rot = oMdes.rotation().tensor.cpu().numpy()[0]
            pos = oMdes.translation().tensor.cpu().numpy()[0]
            oMdes = pin.SE3(rot, pos)

            i = 0
            while True:
                # Calculate current end-effector position
                pin.forwardKinematics(self.pin_model, self.pin_data, q)
                pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
                pin.updateFramePlacements(self.pin_model, self.pin_data)
                oMd = self.pin_data.oMf[self.pin_end_effector_frame_id].actInv(oMdes)

                # Calculate error
                err = pin.log(oMd).vector

                if np.linalg.norm(err) < eps:
                    success = True
                    break

                # Check if max iterations reached
                if i >= IT_MAX:
                    success = False
                    break

                # Calculate Jacobian
                J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q, self.pin_end_effector_frame_id)
                J = -np.dot(pin.Jlog6(oMd.inverse()), J)
                H = J.dot(J.T) + damp * np.eye(J.shape[0])
                g = np.linalg.solve(H, err)
                v = -DT * J.T.dot(g)
                q = pin.integrate(self.pin_model, q, v)

                q = np.clip(q, self.pin_model.lowerPositionLimit, self.pin_model.upperPositionLimit)

                i += 1

            computed_qs[q_i] = q

        return torch.tensor(computed_qs, dtype=self.dtype, device=self.device)

    def inverse_kinematics(
            self,
            pos_quat: torch.Tensor,
            rtol: float = 0.0,
            atol: float = 1e-4,
            ik_solver: IkSolver = IkSolver.LEVENBERG_MARQUARDT,
            q=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_quat.ndim == 1:
            pos_quat.unsqueeze(-1)

        if q is None:
            q = self.joint_position

        ik_solver_fn = None

        if ik_solver == IkSolver.LEVENBERG_MARQUARDT:
            ik_solver_fn = self._inverse_kinematics_levenberg_marquardt
        elif ik_solver == IkSolver.JACOBIAN:
            ik_solver_fn = self._inverse_kinematics_jacobian
        elif ik_solver == IkSolver.PINOCCHIO_CPU:
            ik_solver_fn = self._inverse_kinematics_pinocchio_cpu
        else:
            print("### Unknown IK Solver")

        start = time.time()
        computed_q = ik_solver_fn(pos_quat, q)
        if self.verbose:
            ms_elapsed = (time.time() - start) * 1000
            print(f"Franka IK: {ik_solver} took {ms_elapsed:04} ms")
            print()

        computed_pos_quat = self.forward_kinematics(computed_q)
        positional_difference = computed_pos_quat[:, :3] - pos_quat[:, :3]
        rotational_difference = spherical_distance_quat(computed_pos_quat[:, 3:], pos_quat[:, 3:])

        print(f"Mean err pos: {positional_difference.abs().mean():04f}, rot: {rotational_difference.abs().mean():04f}")

        is_close = torch.isclose(computed_pos_quat, pos_quat, rtol=rtol, atol=atol)
        success = torch.all(is_close, dim=-1)

        return computed_q, success

    def is_within_joint_position_limits(self, joint_position):
        joint_position = joint_position[:, :7]
        joint_within_limits = (joint_position >= self._joint_position_lower_limit) & (
                    joint_position <= self._joint_position_upper_limit)

        all_within_limits = joint_within_limits.all(dim=-1)

        return all_within_limits

    def goto_joint_position(self, joint_position, max_steps=200):
        if not self.is_within_joint_position_limits(joint_position):
            print("Trying to goto out of limits joint position")
        self._isaac_robot.set_joint_position_target(joint_position)
        self._isaac_robot.set_joint_velocity_target(torch.zeros_like(joint_position))

        for i in range(max_steps):
            if torch.allclose(joint_position, self.joint_position):
                return True

            self._scene.write_data_to_sim()
            self._sim.step()
            self._scene.update(self.sim_dt)

        return False

    def set_joint_position(self, joint_position):
        if not torch.any(self.is_within_joint_position_limits(joint_position)):
            print("Trying to set out of limits joint position")
        joint_velocity = torch.zeros_like(joint_position)
        self._isaac_robot.write_joint_state_to_sim(joint_position, joint_velocity)
        self._isaac_robot.set_joint_position_target(joint_position)
        self._isaac_robot.write_data_to_sim()
        self.let_run(steps=1)

    def is_within_cartesian_limits(self, pos_quat):
        cartesian_position = pos_quat[:, :3]
        coordinate_within_limits = (cartesian_position >= self._cartesian_position_lower_limit) & (
                cartesian_position <= self._cartesian_position_upper_limit)

        all_within_limits = coordinate_within_limits.all(dim=-1)

        return all_within_limits


    def goto_ee_pos_quat(self, pos_quat, max_steps=200, ik_solver=IkSolver.LEVENBERG_MARQUARDT):
        if not self.is_within_cartesian_limits(pos_quat):
            print("Desired end effector position is out of limits")
        target_q, s = self.inverse_kinematics(pos_quat, ik_solver=ik_solver)
        self.goto_joint_position(target_q)
        return s

    def set_ee_pos_quat(self, pos_quat, ik_solver=IkSolver.LEVENBERG_MARQUARDT):
        target_q, s = self.inverse_kinematics(pos_quat, ik_solver=ik_solver)
        self.set_joint_position(target_q)

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
        [0.6, 0.25, 0.15, -0.7071, 0.7071, 0, 0],  # is this possible?
        [0.6, 0.25, 0.65, 0.009, 0.72, -0.67, -0.014],
        [0.6, 0.25, 0.65, 0.7071, 0.7071, 0, 0],
        [0.6, 0.25, 0.65, -0.7071, 0.7071, 0, 0],  # is this possible?
        [0.6, -0.25, 0.65, 0.009, 0.72, -0.67, -0.014],
        [0.6, -0.25, 0.65, 0.7071, 0.7071, 0, 0],
        [0.6, -0.25, 0.65, -0.7071, 0.7071, 0, 0],  # is this possible?
        [0.35, -0.25, 0.15, 0.009, 0.72, -0.67, -0.014],
        [0.35, -0.25, 0.15, 0.7071, 0.7071, 0, 0],
        [0.35, -0.25, 0.15, -0.7071, 0.7071, 0, 0],  # is this possible?
        [0.35, 0.25, 0.15, 0.009, 0.72, -0.67, -0.014],
        [0.35, 0.25, 0.15, 0.7071, 0.7071, 0, 0],
        [0.35, 0.25, 0.15, -0.7071, 0.7071, 0, 0],  # is this possible?
        [0.35, 0.25, 0.45, 0.009, 0.72, -0.67, -0.014],
        [0.35, 0.25, 0.45, 0.7071, 0.7071, 0, 0],
        [0.35, 0.25, 0.45, -0.7071, 0.7071, 0, 0],  # is this possible?
        [0.35, -0.25, 0.45, 0.009, 0.72, -0.67, -0.014],
        [0.35, -0.25, 0.45, 0.7071, 0.7071, 0, 0],
        [0.35, -0.25, 0.45, -0.7071, 0.7071, 0, 0],  # is this possible?
    ],
    dtype=torch.float64,
    device="cuda"
))

test_ee_poses = normalize_quat_in_pos_quat(torch.tensor(
    [
        [0.3, -0.6, 0.5, 0.0, 1.0, 0.0, 0.0],
        [0.3, -0.5, 0.5, 0.0, 1.0, 0.1, 0.0],
        [0.3, -0.4, 0.5, 0.0, 1.0, 0.2, 0.0],
        [0.3, -0.3, 0.5, 0.0, 1.0, 0.3, 0.0],
        [0.3, -0.2, 0.5, 0.0, 1.0, 0.4, 0.0],
        [0.3, -0.2, 0.5, 0.0, 1.0, 0.5, 0.0],
        [0.3, -0.1, 0.5, 0.0, 1.0, 0.6, 0.0],
        [0.3, -0.0, 0.5, 0.0, 1.0, 0.7, 0.0],
        [0.3, -0.1, 0.5, 0.0, 1.0, 0.8, 0.0],
        [0.3, -0.2, 0.5, 0.0, 1.0, 0.9, 0.0],
        [0.3, -0.3, 0.5, 0.0, 1.0, 1.0, 0.0],
        [0.3, -0.4, 0.5, 0.0, 0.9, 1.0, 0.0],
        [0.3, 0.1, 0.5, 0.0, 0.8, 1.0, 0.0],
        [0.3, 0.2, 0.5, 0.0, 0.7, 1.0, 0.0],
        [0.3, 0.3, 0.5, 0.0, 0.6, 1.0, 0.0],
        [0.3, 0.4, 0.5, 0.0, 0.5, 1.0, 0.0],
        [0.3, 0.5, 0.5, 0.0, 0.4, 1.0, 0.0],
        [0.3, 0.6, 0.5, 0.0, 0.3, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.2, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.1, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, -1.0, 0.0, 0.0],
        [0.3, 0.0, 0.5, 0.0, -1.0, -1.0, -1.0],
        [0.3, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ],
    dtype=torch.float64,
    device="cuda"
))


def resolve_ik_solver(name: str):
    default_name = "cpu"
    ik_solver_name_map = {
        "cpu": IkSolver.PINOCCHIO_CPU,
        "gpu": IkSolver.LEVENBERG_MARQUARDT,
    }

    if name not in ik_solver_name_map.keys():
        print(
            f'Unknown IK Solver of name: {name}. Available solvers: {ik_solver_name_map.keys()} Defaulting to "{default_name}".')
        return ik_solver_name_map[default_name]

    return ik_solver_name_map[name]


def main():
    iks = resolve_ik_solver("cpu")

    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((3.0, 0.25, 0.15), (-1.0, 0.0, 0.0))

    scene = InteractiveScene(
        cfg=FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    )
    sim.reset()

    robot = FrankaPanda(scene["franka"], sim, scene, verbose=False)
    robot.reset()

    pose = normalize_quat_in_pos_quat(torch.tensor(
        [
            [0.6, 0.25, 0.15, 0, 1, 0, 0],
            [0.6, 0.25, 0.15, -0.7071, -0.7071, 0, 0],
            [0.6, 0.25, 0.15, -0.7071, 0.7071, 0, 0],
        ],
        dtype=torch.float64,
        device="cuda"
    ))

    # robot.set_ee_pos_quat(pose, ik_solver=iks)

    for pose in hardcore_test_ee_poses:
        robot.set_ee_pos_quat(pose.unsqueeze(0).repeat(scene.num_envs, 1), ik_solver=iks)

    robot.close_gripper()
    while True:
        robot.let_run()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
