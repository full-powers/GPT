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
# w_i の値を設定 (入力)
# この例では3つの要素があり、3番目のw_3が最も大きい
w = np.array([1.0, 0.8, 1.2]) 

# z_i の初期値 (t=0 のときの値)
# 全ての要素が0から始まると変化しないため、小さな正の値を与える
z0 = np.array([0.1, 0.1, 0.1]) 

# 計算する時間の範囲 (0秒から20秒まで)
t_span = [0, 20]
# グラフ描画のために、計算結果を出力する時間点を指定
t_eval = np.linspace(t_span[0], t_span[1], 500)

# 3. 微分方程式を数値的に解く
# solve_ivp(方程式の関数, 時間範囲, 初期値, args=(方程式に渡す追加引数,), 評価時間)
solution = solve_ivp(
    replicator_dynamics, 
    t_span, 
    z0, 
    args=(w,), 
    t_eval=t_eval,
    dense_output=True # スムーズなプロットのために推奨
)

# 4. 結果をグラフに描画する
plt.figure(figsize=(10, 6))

# 各z_iについてプロット
for i in range(len(w)):
    plt.plot(solution.t, solution.y[i], label=f'$z_{i+1}$ (w = {w[i]})')

plt.title('simulation results', fontsize=16)
plt.xlabel('time (t)', fontsize=12)
plt.ylabel('value of each element (z)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# 図で比較
plt.figure(figsize=(10, 6))

# Iterative Softmax (replicator dynamics) の結果
for i in range(len(w)):
    plt.plot(solution.t, solution.y[i], label=f'Iterative $z_{i+1}$ (w={w[i]})')

# wのSoftmax値（横一直線）
w_softmax = F.softmax(torch.tensor(w, dtype=torch.float32), dim=0).numpy()
for i in range(3):
    plt.plot(t_eval, [w_softmax[i]]*len(t_eval), '--', label=f'w{i+1} softmax (const)')

plt.xlabel('Time / Input')
plt.ylabel('Value')
plt.title('Iterative Softmax vs Softmax')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('softmax_comparison.png')
plt.show()

tau_list = [0.1, 0.5, 1.0, 2.0]
for tau in tau_list:
    solution = solve_ivp(
        replicator_dynamics,
        t_span,
        z0,
        args=(w, tau),
        t_eval=t_eval,
        dense_output=True
    )
    plt.figure(figsize=(10, 6))
    for i in range(len(w)):
        plt.plot(solution.t, solution.y[i], label=f'Iterative $z_{i+1}$ (w={w[i]})')
    w_softmax = F.softmax(torch.tensor(w, dtype=torch.float32), dim=0).numpy()
    for i in range(3):
        plt.plot(t_eval, [w_softmax[i]]*len(t_eval), '--', label=f'w{i+1} softmax (const)')
    plt.xlabel('Time / Input')
    plt.ylabel('Value')
    plt.title(f'Iterative Softmax vs Softmax (tau={tau})')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'softmax_comparison_tau_{tau}.png')
    plt.close()