clc; close all; clear;

%% 1. 데이터 정의 (실제 실험 데이터로 변경 필요)
% 예시: 연산 시간 (단위: 초)
% [Baseline 1, Baseline 2, Baseline 3, Proposed]
calc_time = [18.2, 12.1, 9.2, 3.1]; 

% x축 라벨 정의
method_names = {'Pure BFGS', 'Adam', 'MLP + BFGS', 'Proposed'};

%% 2. 그래프 그리기
figure('Position', [100, 100, 800, 600]);

% 막대 그래프 생성
b = bar(calc_time, 0.6); % 0.6은 막대 너비
hold on;

%% 3. 스타일 설정 (논문용 포맷팅)
% 3-1. 색상 설정: Proposed만 강조하고 나머지는 회색 처리
b.FaceColor = 'flat'; % 개별 색상 지정 활성화
base_color = [0.7, 0.7, 0.7]; % 회색 (Baselines)
prop_color = [0, 0.4470, 0.7410]; % 파란색 or 강조색 (Proposed)

% 1~3번째는 회색, 4번째(Proposed)는 강조색
b.CData(1,:) = base_color;
b.CData(2,:) = base_color;
b.CData(3,:) = base_color;
b.CData(4,:) = prop_color;

% 3-2. 축 및 라벨 설정
set(gca, 'XTickLabel', method_names, 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Computation Time [s]', 'FontSize', 14, 'FontWeight', 'bold');
title('Comparison of Calculation Time', 'FontSize', 16, 'FontWeight', 'bold');
grid on;
box on;       % 그래프 테두리 닫기
set(gca, 'LineWidth', 1.2); % 축 선 두께

% y축 범위 (데이터 최대값보다 20% 높게 잡아서 여유 공간 확보)
ylim([0, max(calc_time) * 1.2]); 

%% 4. 막대 위에 수치 텍스트 표시
% 각 막대 높이 바로 위에 텍스트 추가
for i = 1:length(calc_time)
    text(i, calc_time(i), sprintf('%.2f s', calc_time(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 12, 'FontWeight', 'bold');
end

hold off;