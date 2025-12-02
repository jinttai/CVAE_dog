## 공간 로봇 계획 (Quaternion RK4 Variant)

이 폴더는 기존 `space_robot_planning` 프로젝트에서
**쿼터니언 미분 + Euler 적분** 대신 **쿼터니언 4차 Runge–Kutta(RK4) 적분**을
학습 단계부터 전부 사용하는 변형 버전입니다.

- 원본 물리 레이어: `space_robot_planning/src/training/physics_layer.py`
- RK4 학습용 래퍼: `space_robot_planning_rk4/src/training/physics_layer.py`

### 구성

- `train_cvae_rk4.py` : CVAE + RK4 물리로 학습, 결과는 `runs/`, `plots_rk4/`, `weights/cvae_rk4_debug/` 등에 저장
- `train_mlp_rk4.py`  : MLP  + RK4 물리로 학습
- `optimize_direct_rk4.py` : 초기 guess 없이 RK4 물리에서 직접 최적화
- `optimize_nn_rk4.py`     : CVAE warm start + LBFGS (RK4 물리)
- `evaluate_rk4.py`        : RK4 물리 기준으로 CVAE/MLP 성능 평가

모델/다이내믹스 코드는 원본 `space_robot_planning/src` 를 재사용하고,
물리 적분만 RK4 버전 `PhysicsLayer` 를 통해 대체합니다.


