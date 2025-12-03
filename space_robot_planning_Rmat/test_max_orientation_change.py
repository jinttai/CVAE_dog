import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
import numpy as np
from torch.func import vmap

# === 경로 설정 ===
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../space_robot_planning_Rmat
project_root = os.path.dirname(current_dir)               # .../CVAE

orig_src_dir = os.path.join(project_root, "space_robot_planning", "src")
if orig_src_dir not in sys.path:
    sys.path.append(orig_src_dir)

rmat_src_dir = os.path.join(current_dir, "src")
if rmat_src_dir not in sys.path:
    sys.path.append(rmat_src_dir)

from training.physics_layer import PhysicsLayer   # Rmat 버전
from dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart


def quat_to_rot(q):
    """
    쿼터니언 q = [x, y, z, w] 를 회전행렬 R (3x3) 로 변환.
    vmap 호환: device는 텐서에서 자동 추출
    """
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    # out-of-place 연산으로 행렬 생성 (vmap 호환)
    R = torch.stack([
        torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)]),
        torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)]),
        torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)])
    ])
    return R


def skew(v):
    """
    3D 벡터 v에 대한 skew-symmetric 행렬 [v]_x
    vmap 호환: device는 텐서에서 자동 추출
    """
    vx, vy, vz = v
    zero = torch.zeros_like(vx)
    # out-of-place 연산으로 행렬 생성 (vmap 호환)
    M = torch.stack([
        torch.stack([zero, -vz, vy]),
        torch.stack([vz, zero, -vx]),
        torch.stack([-vy, vx, zero])
    ])
    return M


def rot_from_omega(wb, dt):
    """
    각속도 wb 에 대해 dt 동안의 회전을 나타내는 회전행렬 R_delta 계산.
    Rodrigues 공식을 사용.
    vmap 호환: device는 텐서에서 자동 추출
    """
    device = wb.device
    dtype = wb.dtype
    theta = torch.linalg.norm(wb) * dt
    
    axis = wb / (torch.linalg.norm(wb) + 1e-12)
    K = skew(axis)
    I = torch.eye(3, device=device, dtype=dtype)
    R_delta = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    return R_delta


def rot_to_euler(R):
    """
    회전 행렬 R (3x3)을 Euler angle (ZYX 순서, yaw-pitch-roll)로 변환.
    
    Args:
        R: [3, 3] 회전 행렬
    
    Returns:
        euler: [3] Euler angles [yaw, pitch, roll] in radians
    """
    # ZYX 순서 (yaw-pitch-roll)
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        yaw = torch.atan2(R[1, 0], R[0, 0])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = torch.atan2(R[2, 1], R[2, 2])
    else:
        yaw = torch.atan2(-R[0, 1], R[1, 1])
        pitch = torch.atan2(-R[2, 0], sy)
        roll = 0.0
    
    return torch.stack([yaw, pitch, roll])


def calculate_orientation_change(robot, dt, num_steps, q_traj, q_dot_traj, q0_init):
    """
    Trajectory parameter를 통해 body orientation 변화를 측정
    vmap 호환: physics 객체 대신 필요한 파라미터만 받음
    
    Args:
        robot: 로봇 딕셔너리
        dt: 시간 스텝 크기
        num_steps: 시뮬레이션 스텝 수
        q_traj: [num_steps, n_q] joint trajectory
        q_dot_traj: [num_steps, n_q] joint velocity trajectory
        q0_init: [4] 초기 quaternion
    
    Returns:
        angle_change: 초기 orientation에서 최종 orientation까지의 각도 변화 (radians)
    """
    device = q0_init.device
    
    # 기준 좌표계
    R0 = torch.eye(3, device=device, dtype=q0_init.dtype)
    r0 = torch.zeros(3, device=device, dtype=q0_init.dtype)
    
    # 초기 자세를 회전행렬로 변환
    R_init = quat_to_rot(q0_init)
    R_curr = R_init.clone()
    
    # Trajectory를 따라 시뮬레이션
    for t in range(num_steps):
        qm = q_traj[t]
        qd = q_dot_traj[t]
        
        # --- 1. SPART Dynamics Calculations ---
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = spart.inertia_projection(R0, RL, robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
        
        # --- 2. Non-holonomic Constraint Solver ---
        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device, dtype=q0_init.dtype)
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]  # Angular Velocity part
        
        # --- 3. Rotation Matrix Integration ---
        R_delta = rot_from_omega(wb, dt)
        R_curr = R_curr @ R_delta
    
    # --- 4. Orientation Change 계산 ---
    # 초기 orientation에서 최종 orientation까지의 상대 회전 행렬
    # tr(R) = 1 + 2cos(θ) 이므로, cos(θ) = (tr(R) - 1) / 2
    R_change = R_init.T @ R_curr
    trace = torch.clamp((torch.trace(R_change) - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    angle_change = torch.acos(trace)  # radians (회전행렬에서는 * 2.0 불필요)
    
    return angle_change


def calculate_orientation_change_batch(physics, q_traj, q_dot_traj, q0_init):
    """
    Batch 버전: [batch_size, num_steps, n_q] -> [batch_size]
    vmap을 사용하여 효율적으로 계산 (for loop 없이 완전 벡터화)
    """
    # vmap을 사용하여 배치 차원에 대해 병렬화
    # in_dims: (None, None, None, 0, 0, 0) -> robot, dt, num_steps는 공유, 나머지는 배치 차원
    batch_sim_fn = vmap(
        calculate_orientation_change, 
        in_dims=(None, None, None, 0, 0, 0)
    )
    angles = batch_sim_fn(
        physics.robot,
        physics.dt,
        physics.num_steps,
        q_traj,
        q_dot_traj,
        q0_init
    )
    return angles


def calculate_final_orientation(physics, q_traj, q_dot_traj, q0_init):
    """
    단일 trajectory에 대해 최종 회전 행렬과 Euler angle을 계산
    
    Args:
        physics: PhysicsLayer 인스턴스
        q_traj: [num_steps, n_q] joint trajectory
        q_dot_traj: [num_steps, n_q] joint velocity trajectory
        q0_init: [4] 초기 quaternion
    
    Returns:
        R_final: [3, 3] 최종 회전 행렬
        euler_final: [3] 최종 Euler angles [yaw, pitch, roll] in radians
    """
    device = q0_init.device
    
    # 기준 좌표계
    R0 = torch.eye(3, device=device, dtype=q0_init.dtype)
    r0 = torch.zeros(3, device=device, dtype=q0_init.dtype)
    
    # 초기 자세를 회전행렬로 변환
    R_init = quat_to_rot(q0_init)
    R_curr = R_init.clone()
    
    # Trajectory를 따라 시뮬레이션
    for t in range(physics.num_steps):
        qm = q_traj[t]
        qd = q_dot_traj[t]
        
        # --- 1. SPART Dynamics Calculations ---
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
        I0, Im = spart.inertia_projection(R0, RL, physics.robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
        H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)
        
        # --- 2. Non-holonomic Constraint Solver ---
        rhs = -H0m @ qd
        H0_damped = H0 + 1e-6 * torch.eye(6, device=device, dtype=q0_init.dtype)
        u0_sol = torch.linalg.solve(H0_damped, rhs)
        wb = u0_sol[:3]  # Angular Velocity part
        
        # --- 3. Rotation Matrix Integration ---
        R_delta = rot_from_omega(wb, physics.dt)
        R_curr = R_curr @ R_delta
    
    # 최종 Euler angle 계산
    euler_final = rot_to_euler(R_curr)
    
    return R_curr, euler_final


def monte_carlo_max_orientation_change(
    physics,
    num_samples=1000,
    waypoint_range=(-3.14, 3.14),
    q0_init=None,
    verbose=True
):
    """
    몬테카를로 방식으로 최대 orientation 변화를 찾는 함수
    
    Args:
        physics: PhysicsLayer 인스턴스
        num_samples: 몬테카를로 샘플 수
        waypoint_range: waypoint 값의 범위 (min, max)
        q0_init: 초기 quaternion [4] 또는 None (기본값: [0,0,0,1])
        verbose: 진행 상황 출력 여부
    
    Returns:
        max_angle_change: 최대 orientation 변화 (radians)
        max_angle_deg: 최대 orientation 변화 (degrees)
        best_waypoints: 최대 변화를 일으킨 waypoints
        all_angles: 모든 샘플의 orientation 변화 리스트
        euler_final: 최대 변화 시 최종 Euler angle [yaw, pitch, roll] (radians)
        euler_final_deg: 최대 변화 시 최종 Euler angle [yaw, pitch, roll] (degrees)
    """
    device = physics.device
    n_q = physics.n_q
    num_waypoints = physics.num_waypoints
    
    # 초기 orientation 설정
    if q0_init is None:
        q0_init = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    
    if q0_init.dim() == 1:
        q0_init = q0_init.unsqueeze(0)  # [1, 4]
    
    if verbose:
        print(f"=== 몬테카를로 최대 Orientation 변화 탐색 ===")
        print(f"샘플 수: {num_samples}")
        print(f"Waypoint 범위: {waypoint_range}")
        print(f"초기 Orientation: {q0_init[0].cpu().numpy()}")
        print(f"로봇 관절 수: {n_q}")
        print(f"Waypoint 수: {num_waypoints}")
        print()
    
    # 랜덤 waypoints 생성
    waypoints_flat = torch.rand(num_samples, num_waypoints * n_q, device=device)
    waypoints_flat = waypoints_flat * (waypoint_range[1] - waypoint_range[0]) + waypoint_range[0]
    
    # Trajectory 생성
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints_flat)
    
    # 각 샘플에 대해 orientation 변화 계산
    start_time = time.time()
    all_angles = calculate_orientation_change_batch(
        physics, q_traj, q_dot_traj, q0_init.repeat(num_samples, 1)
    )
    elapsed_time = time.time() - start_time
    
    # 최대값 찾기
    max_idx = torch.argmax(all_angles)
    max_angle_change = all_angles[max_idx].item()
    max_angle_deg = np.rad2deg(max_angle_change)
    best_waypoints = waypoints_flat[max_idx].clone()
    
    # 최대 orientation 변화를 일으킨 trajectory의 최종 Euler angle 계산
    best_q_traj = q_traj[max_idx]
    best_q_dot_traj = q_dot_traj[max_idx]
    best_q0_init = q0_init[0]  # 모든 샘플이 같은 초기 orientation 사용
    R_final, euler_final = calculate_final_orientation(
        physics, best_q_traj, best_q_dot_traj, best_q0_init
    )
    euler_final_deg = np.rad2deg(euler_final.cpu().numpy())
    
    if verbose:
        print(f"계산 완료 (소요 시간: {elapsed_time:.2f}초)")
        print(f"최대 Orientation 변화: {max_angle_change:.6f} rad ({max_angle_deg:.4f}°)")
        print(f"평균 Orientation 변화: {all_angles.mean().item():.6f} rad ({np.rad2deg(all_angles.mean().item()):.4f}°)")
        print(f"표준편차: {all_angles.std().item():.6f} rad ({np.rad2deg(all_angles.std().item()):.4f}°)")
        print(f"최대 변화를 일으킨 Waypoints:")
        print(f"  {best_waypoints.cpu().numpy()}")
        print(f"최대 변화 시 최종 Euler Angle (ZYX 순서, yaw-pitch-roll):")
        print(f"  Yaw:   {euler_final_deg[0]:.4f}° ({euler_final[0].item():.6f} rad)")
        print(f"  Pitch: {euler_final_deg[1]:.4f}° ({euler_final[1].item():.6f} rad)")
        print(f"  Roll:  {euler_final_deg[2]:.4f}° ({euler_final[2].item():.6f} rad)")
        print()
    
    return max_angle_change, max_angle_deg, best_waypoints, all_angles, euler_final, euler_final_deg


def plot_results(all_angles, save_path=None):
    """
    결과 히스토그램 및 통계 플롯
    """
    angles_deg = np.rad2deg(all_angles.cpu().numpy())
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 히스토그램
    axes[0].hist(angles_deg, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.rad2deg(all_angles.max().item()), color='r', linestyle='--', 
                    label=f'Max: {np.rad2deg(all_angles.max().item()):.4f}°')
    axes[0].axvline(np.rad2deg(all_angles.mean().item()), color='g', linestyle='--', 
                    label=f'Mean: {np.rad2deg(all_angles.mean().item()):.4f}°')
    axes[0].set_xlabel('Orientation Change (degrees)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Monte Carlo: Body Orientation Change Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 통계 정보
    stats_text = f"""
    Statistics:
    - Max: {np.rad2deg(all_angles.max().item()):.4f}°
    - Mean: {np.rad2deg(all_angles.mean().item()):.4f}°
    - Std: {np.rad2deg(all_angles.std().item()):.4f}°
    - Min: {np.rad2deg(all_angles.min().item()):.4f}°
    - Median: {np.rad2deg(all_angles.median().item()):.4f}°
    - 95th percentile: {np.rad2deg(torch.quantile(all_angles, 0.95).item()):.4f}°
    - 99th percentile: {np.rad2deg(torch.quantile(all_angles, 0.99).item()):.4f}°
    """
    axes[1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                 family='monospace', transform=axes[1].transAxes)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 몬테카를로 최대 Orientation 변화 탐색 (Rmat Physics) ===")
    print(f"Device: {device}\n")
    
    # 로봇 로드
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    # PhysicsLayer 설정
    NUM_WAYPOINTS = 3
    TOTAL_TIME = 1.0
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # 초기 orientation (기본값: identity quaternion)
    q0_init = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    
    # 몬테카를로 탐색
    num_samples = 100000  # 샘플 수 (필요에 따라 조정)
    waypoint_range = (-180.0 * np.pi / 180.0, 180.0 * np.pi / 180.0)  # -140도 ~ 140도 (라디안)
    
    max_angle_change, max_angle_deg, best_waypoints, all_angles, euler_final, euler_final_deg = \
        monte_carlo_max_orientation_change(
            physics,
            num_samples=num_samples,
            waypoint_range=waypoint_range,
            q0_init=q0_init,
            verbose=True
        )
    
    # 결과 저장
    save_dir = "results_rmat/monte_carlo_orientation"
    os.makedirs(save_dir, exist_ok=True)
    
    # 플롯 저장
    plot_path = os.path.join(save_dir, "orientation_change_distribution.png")
    plot_results(all_angles, save_path=plot_path)
    
    # 통계 저장
    stats_path = os.path.join(save_dir, "max_orientation_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("=== 몬테카를로 최대 Orientation 변화 탐색 결과 ===\n\n")
        f.write(f"샘플 수: {num_samples}\n")
        f.write(f"Waypoint 범위: {waypoint_range}\n")
        f.write(f"초기 Orientation: {q0_init.cpu().numpy()}\n\n")
        f.write(f"최대 Orientation 변화: {max_angle_change:.6f} rad ({max_angle_deg:.4f}°)\n")
        f.write(f"평균 Orientation 변화: {np.rad2deg(all_angles.mean().item()):.4f}°\n")
        f.write(f"표준편차: {np.rad2deg(all_angles.std().item()):.4f}°\n")
        f.write(f"최소값: {np.rad2deg(all_angles.min().item()):.4f}°\n")
        f.write(f"중앙값: {np.rad2deg(all_angles.median().item()):.4f}°\n")
        f.write(f"95th percentile: {np.rad2deg(torch.quantile(all_angles, 0.95).item()):.4f}°\n")
        f.write(f"99th percentile: {np.rad2deg(torch.quantile(all_angles, 0.99).item()):.4f}°\n\n")
        f.write(f"최대 변화를 일으킨 Waypoints:\n")
        f.write(f"{best_waypoints.cpu().numpy()}\n\n")
        f.write(f"최대 변화 시 최종 Euler Angle (ZYX 순서, yaw-pitch-roll):\n")
        f.write(f"  Yaw:   {euler_final_deg[0]:.4f}° ({euler_final[0].item():.6f} rad)\n")
        f.write(f"  Pitch: {euler_final_deg[1]:.4f}° ({euler_final[1].item():.6f} rad)\n")
        f.write(f"  Roll:  {euler_final_deg[2]:.4f}° ({euler_final[2].item():.6f} rad)\n")
    
    print(f"결과 저장 완료:")
    print(f"  - Plot: {plot_path}")
    print(f"  - Statistics: {stats_path}")


if __name__ == "__main__":
    main()
