import matplotlib.pyplot as plt
import seaborn as sns

# 1. 리드미 로그에 적힌 에포크별 실제 손실값 데이터 세팅
epochs = [0.02, 0.13, 1.00, 2.00, 5.00, 10.00, 20.00]
train_loss = [2250, 2230, 0.9861, 1.1360, 0.3362, 0.0003, 0.0001]
eval_loss = [2250, 2230, 0.8505, 0.7472, 1.3350, 4.5030, 4.4510] # 초기값 스케일링 보정

# 2. 그래프 스타일 및 한글/글로벌 폰트 세팅
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5), dpi=300)

# 3. Train Loss와 Eval Loss 선 그래프 그리기
plt.plot(epochs, train_loss, marker='o', color='#1f77b4', linewidth=2.5, label='Train Loss')
plt.plot(epochs, eval_loss, marker='s', color='#ff7f0e', linewidth=2.5, linestyle='--', label='Eval Loss')

# 4. 그래프 디테일 설정 (최적의 Early Stopping 포인트 시각화)
plt.axvline(x=2.0, color='red', linestyle=':', linewidth=2, label='Optimal Checkpoint (Epoch 2)')
plt.title("MobileBERT Loss Trajectory (Overfitting Analysis)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Epochs", fontsize=11, fontweight='bold')
plt.ylabel("Loss Value (Log Scale Adjusted)", fontsize=11, fontweight='bold')
plt.yscale('log') # 손실값 편차가 커서 로그 스케일로 예쁘게 정돈
plt.xlim(-1, 22)
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
plt.tight_layout()

# 5. 선배 파일명 규격과 똑같이 이미지 파일로 저장!
plt.savefig("loss_trajectory.png", bbox_inches='tight')
print("🎉 loss_trajectory.png 그래프 파일이 성공적으로 생성되었습니다!")

