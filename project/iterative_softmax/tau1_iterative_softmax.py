import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 1. 微分方程式を定義する関数
# t: 時間, z: 各要素z_iの値の配列, w: 各要素の適応度w_iの配列, tau: 変化率
def replicator_dynamics(t, z, w, tau=1.0):
    """
    レプリケーターダイナミクスの微分方程式に変化率tauを導入します。
    dz_i/dt = tau * z_i * (w_i - Σz_j)
    """
    sum_z = np.sum(z)
    dzdt = tau * z * (w - sum_z)
    return dzdt

# 2. パラメータと初期条件の設定
# いろんな入力パターンを定義
input_patterns = {
#    'balanced': np.array([1.0, 1.0, 1.0]),  # 均等な入力
    'sparse': np.array([0.1, 0.2, 2.0]),    # スパースな入力
    'original': np.array([1.0, 0.8, 1.2]),  # 元の入力
#    'negative': np.array([-0.5, 0.2, 1.0]), # 負の値を含む入力
    'large_diff': np.array([0.1, 1.0, 5.0]), # 大きな差のある入力
    'five_dim': np.array([0.5, 1.0, 0.3, 2.0, 0.8]), # 5次元の入力
}

# z_i の初期値 (t=0 のときの値)
# 全ての要素が0から始まると変化しないため、小さな正の値を与える
def get_initial_values(n):
    return np.array([0.1] * n)

# 計算する時間の範囲 (0秒から10秒まで)
t_span = [0, 10]
# グラフ描画のために、計算結果を出力する時間点を指定
t_eval = np.linspace(t_span[0], t_span[1], 500)

# 変化率tauの設定
tau = 0.1

# 3. 各入力パターンでの比較実験
for pattern_name, w in input_patterns.items():
    # 初期値を設定
    z0 = get_initial_values(len(w))
    
    # 微分方程式を数値的に解く
    solution = solve_ivp(
        replicator_dynamics, 
        t_span, 
        z0, 
        args=(w, tau), 
        t_eval=t_eval,
        dense_output=True
    )

    # 結果をグラフに描画
    plt.figure(figsize=(12, 8))
    
    # サブプロット1: 正規化前の結果
    plt.subplot(2, 1, 1)
    for i in range(len(w)):
        plt.plot(solution.t, solution.y[i], label=f'$z_{i+1}$ (w = {w[i]:.1f})')
    plt.title(f'Simulation Results - {pattern_name} (Before Normalization)', fontsize=14)
    plt.xlabel('Time (t)')
    plt.ylabel('Value of each element (z)')
    plt.legend()
    plt.grid(True)
    
    # サブプロット2: 正規化後の結果とSoftmaxとの比較
    plt.subplot(2, 1, 2)
    z_normalized = solution.y / np.sum(solution.y, axis=0)
    for i in range(len(w)):
        plt.plot(solution.t, z_normalized[i], label=f'Iterative $z_{i+1}$ (w={w[i]:.1f}) - normalized')
    
    # wのSoftmax値（横一直線）
    w_softmax = F.softmax(torch.tensor(w, dtype=torch.float32), dim=0).numpy()
    for i in range(len(w)):
        plt.plot(t_eval, [w_softmax[i]]*len(t_eval), '--', alpha=0.7, 
                label=f'w{i+1} softmax ({w_softmax[i]:.3f})')
    
    plt.title(f'Iterative Softmax vs Softmax - {pattern_name} (Normalized)', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{pattern_name}_comparison_tau{tau}.png', dpi=150)
    plt.show()
    
    # 最終値と収束性の確認
    final_iterative = z_normalized[:, -1]
    print(f"\n=== {pattern_name.upper()} INPUT ===")
    print(f"Input w: {w}")
    print(f"Softmax result: {w_softmax}")
    print(f"Final iterative result: {final_iterative}")
    print(f"Difference: {np.abs(w_softmax - final_iterative)}")
    print(f"Max difference: {np.max(np.abs(w_softmax - final_iterative)):.6f}")

# 4. 全パターンの最終結果をまとめて比較
plt.figure(figsize=(15, 10))
subplot_idx = 1

for pattern_name, w in input_patterns.items():
    z0 = get_initial_values(len(w))
    solution = solve_ivp(replicator_dynamics, t_span, z0, args=(w, tau), t_eval=t_eval, dense_output=True)
    z_normalized = solution.y / np.sum(solution.y, axis=0)
    w_softmax = F.softmax(torch.tensor(w, dtype=torch.float32), dim=0).numpy()
    
    plt.subplot(2, 3, subplot_idx)
    for i in range(len(w)):
        plt.plot(solution.t, z_normalized[i], label=f'Iter z{i+1}')
        plt.axhline(y=w_softmax[i], color=f'C{i}', linestyle='--', alpha=0.7)
    
    plt.title(f'{pattern_name}')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    subplot_idx += 1

plt.tight_layout()
plt.savefig('all_patterns_comparison.png', dpi=150)
plt.show()