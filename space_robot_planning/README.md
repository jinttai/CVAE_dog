v1

## Version 1 상세 정보

### 개요
- **CVAE (Conditional Variational Autoencoder)** 모델과 **MLP (Baseline)** 모델을 사용한 로봇 궤적 계획 학습
- 고정된 시작점에서 다양한 목표점으로의 궤적 생성 학습

### 주요 특징

#### 학습 설정
- **고정 시작점**: `q0_start = [0, 0, 0, 1]` (단위 쿼터니언)
- **랜덤 목표점**: 매 epoch마다 랜덤하게 생성된 정규화된 쿼터니언
- **물리 시뮬레이션**: PhysicsLayer를 통한 궤적 검증
- **Validation**: 10 epoch마다 고정된 목표점에 대한 검증 수행

#### CVAE 모델 설정
- **Batch Size**: 512
- **Epochs**: 15,000
- **Latent Dimension**: 8
- **Condition Dimension**: 8 (Start(4) + Goal(4))
- **Waypoints**: 4
- **Total Time**: 1.0초
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Hidden Dimension**: 256

#### MLP 모델 설정
- **Batch Size**: 256
- **Epochs**: 5,000
- **Condition Dimension**: 8 (Start(4) + Goal(4))
- **Waypoints**: 4
- **Total Time**: 1.0초
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Hidden Dimension**: 128

### 저장 파일

#### 학습 데이터
- **CSV 파일**: `plots/{model}_training_curve/v1.csv`
  - 컬럼: `epoch`, `train_loss`, `epoch_duration`, `val_loss`
- **플롯 이미지**: `plots/{model}_training_curve/v1.png`
  - Train Loss와 Validation Loss 곡선
- **모델 가중치**: 
  - CVAE: `weights/cvae_debug/v1.pth`
  - MLP: `weights/mlp_debug/v1.pth`

#### TensorBoard 로그
- **CVAE**: `runs/cvae_debug_v1/`
- **MLP**: `runs/mlp_debug_v1/`

### 모델 아키텍처

#### CVAE
- **Encoder**: Condition + Trajectory → (μ, log σ²)
- **Decoder**: Condition + Latent z → Waypoints
- **Inference**: Decoder만 사용하여 랜덤 샘플링된 z로 궤적 생성

#### MLP (Baseline)
- **구조**: 4층 MLP (128 hidden units)
- **입력**: Condition (Start + Goal)
- **출력**: Waypoints (결정론적)

### 학습 프로세스
1. 매 epoch마다 랜덤 목표점 생성
2. CVAE: 랜덤 샘플링된 z로 waypoints 예측
3. MLP: 조건부로 waypoints 예측
4. PhysicsLayer를 통한 궤적 시뮬레이션 및 손실 계산
5. 역전파 및 가중치 업데이트
6. 10 epoch마다 고정 목표점에 대한 검증 및 시각화

### 사용 방법
```bash
# CVAE 학습
python train_cvae.py

# MLP 학습
python train_mlp.py
```