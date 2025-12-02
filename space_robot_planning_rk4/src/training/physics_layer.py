import os
import sys

import torch
from torch.func import vmap

# -------------------------------------------------------------------------
# 원본 프로젝트(src)를 sys.path에 추가하여 기존 PhysicsLayer 를 재사용
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))    # .../space_robot_planning_rk4/src/training
src_root = os.path.dirname(current_dir)                     # .../space_robot_planning_rk4/src
project_root = os.path.dirname(os.path.dirname(src_root))   # .../CVAE

orig_src_dir = os.path.join(project_root, "space_robot_planning", "src")
if orig_src_dir not in sys.path:
    sys.path.append(orig_src_dir)

from training.physics_layer import PhysicsLayer as BasePhysicsLayer  # 원본 (Euler + RK4 보조)


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


