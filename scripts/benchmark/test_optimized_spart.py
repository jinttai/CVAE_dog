import torch
import time
import sys, os

# stdout UTF-8 인코딩 설정 (윈도우 환경 대응)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("Script started", flush=True)

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

from src.dynamics.urdf2robot_torch import urdf2robot
import src.dynamics.spart_functions_torch as spart

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    urdf_path = os.path.join(ROOT_DIR, "assets/a1_description/urdf/a1_bigfoot.urdf")
    robot, _ = urdf2robot(urdf_path, verbose_flag=False, device=device)
    
    R0 = torch.eye(3, device=device)
    r0 = torch.zeros(3, device=device)
    qm = torch.zeros(robot['n_q'], device=device)
    
    print("\n--- Correctness Test (generalized_inertia_matrix) ---")
    if hasattr(spart, 'generalized_inertia_matrix_old'):
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = spart.inertia_projection(R0, RL, robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
        
        # New
        H0, H0m, Hm = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
        # Old
        H0_old, H0m_old, Hm_old = spart.generalized_inertia_matrix_old(M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
        
        print("H0 match:", torch.allclose(H0, H0_old, atol=1e-6))
        if not torch.allclose(H0, H0_old, atol=1e-6):
            print("H0 max error:", (H0 - H0_old).abs().max().item())
            
        print("H0m match:", torch.allclose(H0m, H0m_old, atol=1e-6))
        if not torch.allclose(H0m, H0m_old, atol=1e-6):
            print("H0m max error:", (H0m - H0m_old).abs().max().item())
            
        print("Hm match:", torch.allclose(Hm, Hm_old, atol=1e-6))
        if not torch.allclose(Hm, Hm_old, atol=1e-6):
            print("Hm max error:", (Hm - Hm_old).abs().max().item())
    
    # Warmup
    for _ in range(10):
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = spart.inertia_projection(R0, RL, robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
        H0, H0m, Hm = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # 개별 함수 측정
    num_iter = 100
    
    RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
    
    # diff_kinematics
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iter):
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
    if device == "cuda": torch.cuda.synchronize()
    print(f"diff_kinematics: {(time.time()-t0)/num_iter*1000:.2f}ms/call")
    
    I0, Im = spart.inertia_projection(R0, RL, robot)
    M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
    
    # generalized_inertia_matrix
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iter):
        H0, H0m, Hm = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
    if device == "cuda": torch.cuda.synchronize()
    print(f"generalized_inertia_matrix: {(time.time()-t0)/num_iter*1000:.2f}ms/call")
    
    # 전체 파이프라인
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(num_iter):
        RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
        Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
        I0, Im = spart.inertia_projection(R0, RL, robot)
        M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
        H0, H0m, Hm = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, robot)
    if device == "cuda": torch.cuda.synchronize()
    print(f"전체 SPART 파이프라인: {(time.time()-t0)/num_iter*1000:.2f}ms/call")

if __name__ == "__main__":
    test()

