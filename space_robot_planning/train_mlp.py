import torch
import torch.optim as optim
import time
from src.models.cvae import MLP
from src.training.physics_layer import PhysicsLayer
from urdf2robot_torch import urdf2robot

def main():
    # 1. 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    robot, _ = urdf2robot("assets/SC_ur10e.urdf", verbose_flag=False, device=device)
    
    # 파라미터
    COND_DIM = 8  # Start(4) + Goal(4)
    NUM_WAYPOINTS = 4
    OUTPUT_DIM = NUM_WAYPOINTS * robot['n_q']
    TOTAL_TIME = 5.0
    
    # 2. 모델 및 물리 엔진 준비
    model = MLP(COND_DIM, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    physics = PhysicsLayer(robot, NUM_WAYPOINTS, TOTAL_TIME, device)
    
    print(f"=== MLP Training Start on {device} ===")
    
    # 3. 학습 루프
    start_time = time.time()
    for epoch in range(100):
        # 데이터 생성 (Random Goals)
        q0_start = torch.randn(32, 4, device=device)
        q0_start /= torch.norm(q0_start, dim=1, keepdim=True)
        q0_goal = torch.randn(32, 4, device=device)
        q0_goal /= torch.norm(q0_goal, dim=1, keepdim=True)
        
        condition = torch.cat([q0_start, q0_goal], dim=1)
        
        # Forward & Loss
        optimizer.zero_grad()
        waypoints_pred = model(condition) # Deterministic
        
        loss = physics.calculate_loss(waypoints_pred, q0_start, q0_goal)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100] Loss: {loss.item():.6f}")
            
    print(f"Done. Time: {time.time()-start_time:.2f}s")
    
    # 모델 저장
    torch.save(model.state_dict(), "mlp_weights.pth")

if __name__ == "__main__":
    main()