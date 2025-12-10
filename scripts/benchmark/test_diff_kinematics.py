import torch
import sys
sys.stdout.reconfigure(encoding='utf-8')
print("Script started", flush=True)
import time
import sys
import os

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart

def test_correctness_and_speed():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 로봇 로드
    urdf_path = os.path.join(ROOT_DIR, "assets/a1_description/urdf/a1_bigfoot.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    
    # 테스트 입력 생성
    R0 = torch.eye(3, device=device)
    r0 = torch.zeros(3, device=device)
    qm = torch.zeros(robot['n_q'], device=device)
    
    # kinematics 실행
    RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
    
    # 1. 정확성 테스트 (새 버전 vs 기존 버전)
    if hasattr(spart, 'diff_kinematics_old'):
        print("\n--- Correctness Test ---")
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
        Bij_old, Bi0_old, P0_old, pm_old = spart.diff_kinematics_old(R0, r0, rL, e, g, robot)
        
        print("Bij match:", torch.allclose(Bij, Bij_old, atol=1e-6))
        if not torch.allclose(Bij, Bij_old, atol=1e-6):
            print("Bij Max Error:", (Bij - Bij_old).abs().max().item())
            
        print("Bi0 match:", torch.allclose(Bi0, Bi0_old, atol=1e-6))
        if not torch.allclose(Bi0, Bi0_old, atol=1e-6):
            print("Bi0 Max Error:", (Bi0 - Bi0_old).abs().max().item())

        print("pm match:", torch.allclose(pm, pm_old, atol=1e-6))
        if not torch.allclose(pm, pm_old, atol=1e-6):
            print("pm Max Error:", (pm - pm_old).abs().max().item())
    
    # 2. 속도 테스트
    print("\n--- Speed Test ---")
    num_iterations = 100
    
    # Warmup
    for _ in range(10):
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iterations):
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    print(f"diff_kinematics: {elapsed:.3f}s total, {elapsed/num_iterations*1000:.2f}ms/call")
    
    # 3. 전체 시뮬레이션 테스트
    print("\n=== Full Simulation Test ===")
    from src.training.physics_layer import PhysicsLayer
    
    physics = PhysicsLayer(robot, num_waypoints=3, total_time=10.0, device=device)
    
    # 단일 샘플 테스트
    q0_start = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    # A1 robot likely has 12 joints. Check robot n_q
    print(f"Robot n_q: {robot['n_q']}")
    
    q0_goal = torch.tensor([0.0789, 0.0941, 0.0789, 0.9893], device=device)
    
    waypoints = torch.zeros(1, 3 * robot['n_q'], device=device)
    q_traj, q_dot_traj = physics.generate_trajectory(waypoints)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    # Need to make sure input shapes match expected by simulate_single
    loss, q_final = physics.simulate_single(q_traj[0], q_dot_traj[0], q0_start, q0_goal)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    print(f"simulate_single: {time.time() - start:.3f}s")
    print(f"Loss: {loss.item():.6f}")

if __name__ == "__main__":
    test_correctness_and_speed()

