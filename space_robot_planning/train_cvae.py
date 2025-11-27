import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
from src.models.cvae import CVAE
from src.training.physics_layer import PhysicsLayer
from urdf2robot_torch import urdf2robot

# --- 시각화 헬퍼 함수 ---
def plot_trajectory(q_traj, q_dot_traj, epoch):
    """
    생성된 궤적을 Matplotlib 그림으로 변환하여 TensorBoard에 기록
    """
    q_traj = q_traj.detach().cpu().numpy() # [Steps, Joints]
    q_dot_traj = q_dot_traj.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    
    # 1. Joint Positions
    for i in range(q_traj.shape[1]):
        axes[0].plot(q_traj[:, i], label=f'J{i+1}')
    axes[0].set_title(f'Joint Angles (Epoch {epoch})')
    axes[0].set_ylabel('Rad')
    axes[0].grid(True)
    axes[0].legend(loc='right', fontsize='small')
    
    # 2. Joint Velocities
    for i in range(q_dot_traj.shape[1]):
        axes[1].plot(q_dot_traj[:, i], label=f'J{i+1}')
    axes[1].set_title('Joint Velocities')
    axes[1].set_ylabel('Rad/s')
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # 1. 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== CVAE Training Start on {device} ===")
    
    # 로봇 로드
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/cvae_debug_v1")
    
    # ==========================================
    # [중요] 디버깅용 파라미터 설정 (속도 향상)
    # ==========================================
    COND_DIM = 8
    NUM_WAYPOINTS = 4
    OUTPUT_DIM = NUM_WAYPOINTS * robot['n_q']
    LATENT_DIM = 8
    
    # CPU에서 돌릴 때는 이 값을 작게 유지하세요
    BATCH_SIZE = 32        # (기존 32 -> 2)
    TOTAL_TIME = 1.0      # (기존 5.0 -> 1.0) 시뮬레이션 시간 단축
    NUM_EPOCHS = 100
    
    # 2. 모델 및 물리 엔진 준비
    model = CVAE(COND_DIM, OUTPUT_DIM, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    # 시각화용 고정 테스트 셋
    fixed_start = torch.tensor([[0., 0., 0., 1.]], device=device) 
    fixed_goal = torch.tensor([[0., 0., 0.707, 0.707]], device=device) 
    fixed_cond = torch.cat([fixed_start, fixed_goal], dim=1)
    
    # 3. 학습 루프
    total_start_time = time.time()
    epoch_start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        # --- Training Step ---
        # 데이터 생성 (매번 랜덤 목표)
        q0_start = torch.randn(BATCH_SIZE, 4, device=device)
        q0_start /= torch.norm(q0_start, dim=1, keepdim=True)
        q0_goal = torch.randn(BATCH_SIZE, 4, device=device)
        q0_goal /= torch.norm(q0_goal, dim=1, keepdim=True)
        
        condition = torch.cat([q0_start, q0_goal], dim=1)
        
        optimizer.zero_grad()
        
        # Inference (Decoder Only for Physics Loss)
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        waypoints_pred = model.decode(condition, z)
        
        # Physics Simulation & Loss
        loss = physics.calculate_loss(waypoints_pred, q0_start, q0_goal)
        
        loss.backward()
        optimizer.step()
        
        # --- Logging (매 에폭마다 출력) ---
        writer.add_scalar('Loss/train', loss.item(), epoch)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss.item():.6f} | Time: {epoch_duration:.2f}s")
        epoch_start_time = time.time() # 타이머 리셋
        
        # --- Validation & Visualization (10 에폭마다) ---
        if (epoch+1) % 10 == 0:
            with torch.no_grad():
                z_vis = torch.randn(1, LATENT_DIM, device=device)
                wp_vis = model.decode(fixed_cond, z_vis)
                
                q_traj, q_dot_traj = physics.generate_trajectory(wp_vis)
                val_loss = physics.calculate_loss(wp_vis, fixed_start, fixed_goal)
                
                fig = plot_trajectory(q_traj[0], q_dot_traj[0], epoch+1)
                writer.add_figure('Trajectory/Fixed_Goal', fig, epoch)
                plt.close(fig) # 메모리 누수 방지
                
                print(f"   >>> Validation Loss: {val_loss.item():.6f}")

    print(f"Training Finished. Total Time: {time.time()-total_start_time:.2f}s")
    
    # 모델 저장
    if not os.path.exists("weights"):
        os.makedirs("weights")
    torch.save(model.state_dict(), "weights/cvae_debug.pth")
    writer.close()

if __name__ == "__main__":
    main()