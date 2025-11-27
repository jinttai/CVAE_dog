import sys
import os

# 현재 파일(physics_layer.py)의 부모 디렉토리(src)의 부모(root)를 경로에 추가
# 이렇게 하면 프로젝트 루트 기준으로 import가 가능해집니다.
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/training
src_dir = os.path.dirname(current_dir) # src
root_dir = os.path.dirname(src_dir) # space_robot_planning

sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, 'dynamics'))
import torch
import dynamics.spart_functions_torch as spart

class PhysicsLayer:
    def __init__(self, robot, num_waypoints, total_time, device):
        self.robot = robot
        self.n_q = robot['n_q']
        self.num_waypoints = num_waypoints
        self.total_time = total_time
        self.dt = 0.1  # Simulation step size
        self.num_steps = int(total_time / self.dt)
        self.device = device
        
        # Spline Basis 미리 생성 (Linear or Cubic)
        self.basis = self._get_spline_basis(self.num_waypoints, self.num_steps).to(device)

    def _get_spline_basis(self, num_points, num_steps):
        # 간단한 Linear Basis 예시 (실제로는 Cubic 추천)
        basis = torch.zeros((num_steps, num_points + 2))
        t = torch.linspace(0, 1, num_steps)
        segment_len = 1.0 / (num_points + 1)
        for i in range(num_points + 2):
            center = i * segment_len
            dist = torch.abs(t - center)
            val = torch.clamp(1 - dist / segment_len, 0, 1)
            basis[:, i] = val
        return basis

    def generate_trajectory(self, waypoints_flat):
        # waypoints_flat: [Batch, 24] -> [Batch, Steps, Joints]
        batch_size = waypoints_flat.size(0)
        w_mid = waypoints_flat.view(batch_size, self.num_waypoints, self.n_q)
        zeros = torch.zeros(batch_size, 1, self.n_q, device=self.device)
        w_full = torch.cat([zeros, w_mid, zeros], dim=1) # Start/End Zero
        
        # Interpolation
        q_traj = torch.einsum('st,btj->bsj', self.basis, w_full)
        
        # Velocity (Finite Difference)
        q_dot_traj = torch.zeros_like(q_traj)
        q_dot_traj[:, 1:, :] = (q_traj[:, 1:, :] - q_traj[:, :-1, :]) / self.dt
        
        return q_traj, q_dot_traj

    def calculate_loss(self, waypoints_flat, q0_init, q0_goal):
        """
        Differentiable Physics Simulation Loop
        """
        batch_size = waypoints_flat.size(0)
        q_traj, q_dot_traj = self.generate_trajectory(waypoints_flat)
        
        loss_batch = []
        
        # 현재 SPART가 Batch를 지원하지 않으므로 Loop 수행 (추후 최적화 필요)
        for b in range(batch_size):
            # 초기 상태 설정
            R0 = torch.eye(3, device=self.device)
            r0 = torch.zeros(3, device=self.device)
            q0_curr = q0_init[b]
            
            # 시간 적분 (Trajectory Following)
            for t in range(self.num_steps):
                qm = q_traj[b, t]
                qd = q_dot_traj[b, t]
                
                # --- SPART Physics Call ---
                # 1. Kinematics
                RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, self.robot)
                # 2. Diff Kinematics
                Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, self.robot)
                # 3. Inertia
                I0, Im = spart.inertia_projection(R0, RL, self.robot)
                M0_t, Mm_t = spart.mass_composite_body(I0, Im, Bij, Bi0, self.robot)
                # 4. GIM (H0, H0m)
                H0, H0m, _ = spart.generalized_inertia_matrix(M0_t, Mm_t, Bij, Bi0, P0, pm, self.robot)
                
                # 5. Base Velocity Calculation: wb = -H0_inv * H0m * qd
                rhs = -H0m @ qd
                u0_sol = torch.linalg.solve(H0, rhs)
                wb = u0_sol[:3] # Angular Velocity
                
                # 6. Quaternion Integration
                # q_dot = 0.5 * q * wb
                xyz = q0_curr[:3]
                w = q0_curr[3]
                d_xyz = 0.5 * (w * wb + torch.linalg.cross(wb, xyz))
                d_w = -0.5 * torch.dot(wb, xyz)
                
                q0_curr = q0_curr + torch.cat([d_xyz, d_w.unsqueeze(0)]) * self.dt
                q0_curr = q0_curr / torch.norm(q0_curr)
            
            # Final Error (Distance between quaternions)
            # Loss = 1 - <q_pred, q_goal>^2
            dot_prod = torch.sum(q0_curr * q0_goal[b])
            error = 1.0 - dot_prod**2
            loss_batch.append(error)
            
        return torch.stack(loss_batch).mean()