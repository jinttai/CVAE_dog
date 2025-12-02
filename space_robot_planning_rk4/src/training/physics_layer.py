import os
import sys
import importlib.util

import torch
from torch.func import vmap

# -------------------------------------------------------------------------
# 원본 프로젝트의 PhysicsLayer 를 파일 경로 기반으로 로드
#  - rk4 버전의 패키지 이름(`training`)과 충돌을 피하기 위해
#    모듈 이름이 아닌 파일 경로로 직접 import 한다.
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))    # .../space_robot_planning_rk4/src/training
src_root = os.path.dirname(current_dir)                     # .../space_robot_planning_rk4/src
project_root = os.path.dirname(os.path.dirname(src_root))   # .../CVAE

base_physics_path = os.path.join(
    project_root,
    "space_robot_planning",
    "src",
    "training",
    "physics_layer.py",
)


def _load_base_physics_layer():
    """원본 space_robot_planning PhysicsLayer 를 동적으로 로드."""
    spec = importlib.util.spec_from_file_location("base_physics_layer", base_physics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base PhysicsLayer from: {base_physics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PhysicsLayer


BasePhysicsLayer = _load_base_physics_layer()


class PhysicsLayer(BasePhysicsLayer):
    """
    Quaternion RK4 적분을 **학습부터** 기본으로 사용하는 PhysicsLayer 래퍼.

    - generate_trajectory, simulate_single, simulate_single_rk4 등은
      원본 PhysicsLayer 구현을 그대로 상속.
    - 단, calculate_loss 를 오버라이드하여 항상 RK4 기반 오차를 사용.
    """

    def calculate_loss(self, waypoints_flat, q0_init, q0_goal):
        """
        Batched Physics Simulation using vmap, but 내부는 simulate_single_rk4 사용.
        (원본의 calculate_loss_rk4 와 동일한 역할을 기본 loss 로 사용)
        """
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat)
        batch_sim_fn = vmap(self.simulate_single_rk4, in_dims=(0, 0, 0, 0))
        loss_batch = batch_sim_fn(q_traj, q_dot_traj, q0_init, q0_goal)
        return loss_batch.mean()


