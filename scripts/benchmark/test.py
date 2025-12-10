# train_mlp.py 또는 별도 테스트 파일에서
import torch
import time

import os
import sys
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT_DIR)

import src.dynamics.spart_functions_torch as spart
from src.dynamics.urdf2robot_torch import urdf2robot
from src.training.physics_layer import PhysicsLayer
from src.models.cvae import MLP
#model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
robot_path = os.path.join(ROOT_DIR, "assets/a1_description/urdf/a1_bigfoot.urdf")
robot, _ = urdf2robot(robot_path, device=device)
CONDITION_DIM = 8
OUTPUT_DIM = 3 * robot["n_q"]
model = MLP(CONDITION_DIM, OUTPUT_DIM).to(device)




NUM_WAYPOINTS = 3
TOTAL_TIME = 10.0
physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)


# 단일 샘플 준비
q0_start = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
q0_goal = torch.tensor([0.0789, 0.0941, 0.0789, 0.9893], device=device)

waypoints_single = model(torch.cat([q0_start, q0_goal]).unsqueeze(0))
q_traj, q_dot_traj = physics.generate_trajectory(waypoints_single)
q_traj = q_traj[0]  # [100, n_q]
q_dot_traj = q_dot_traj[0]

# SPART 함수별 시간 측정
R0 = torch.eye(3, device=device)
r0 = torch.zeros(3, device=device)

t_kin, t_diff, t_inert, t_mass, t_gen = 0, 0, 0, 0, 0

for t in range(physics.num_steps):
    qm = q_traj[t]
    qd = q_dot_traj[t]
    
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, physics.robot)
    torch.cuda.synchronize() if device == 'cuda' else None
    t_kin += time.time() - t0
    
    t0 = time.time()
    Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, physics.robot)
    torch.cuda.synchronize() if device == 'cuda' else None
    t_diff += time.time() - t0
    
    t0 = time.time()
    I0, Im = spart.inertia_projection(R0, RL, physics.robot)
    torch.cuda.synchronize() if device == 'cuda' else None
    t_inert += time.time() - t0
    
    t0 = time.time()
    M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, physics.robot)
    torch.cuda.synchronize() if device == 'cuda' else None
    t_mass += time.time() - t0
    
    t0 = time.time()
    H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, physics.robot)
    torch.cuda.synchronize() if device == 'cuda' else None
    t_gen += time.time() - t0

print(f"kinematics:        {t_kin:.3f}s ({t_kin/physics.num_steps*1000:.1f}ms/step)")
print(f"diff_kinematics:   {t_diff:.3f}s ({t_diff/physics.num_steps*1000:.1f}ms/step)")
print(f"inertia_projection:{t_inert:.3f}s ({t_inert/physics.num_steps*1000:.1f}ms/step)")
print(f"mass_composite:    {t_mass:.3f}s ({t_mass/physics.num_steps*1000:.1f}ms/step)")
print(f"generalized_inertia:{t_gen:.3f}s ({t_gen/physics.num_steps*1000:.1f}ms/step)")
print(f"TOTAL:             {t_kin+t_diff+t_inert+t_mass+t_gen:.3f}s")


