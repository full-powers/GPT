#include <stdio.h>

int main() {
    // パラメータ設定
    double w[3] = {2.0, 0.8, 1.2}; // 適応度
    double z[3] = {0.1, 0.1, 0.1}; // 初期値
    double tau = 1.0;
    double dt = 0.02;
    int steps = 50; // 0.02刻みで1秒分

    // ヘッダ出力
    printf("# t z0 z1 z2 z0_norm z1_norm z2_norm\n");

    for (int t = 0; t <= steps; ++t) {
        double sum_z = z[0] + z[1] + z[2];
        double dz[3];
        for (int i = 0; i < 3; ++i) {
            dz[i] = tau * z[i] * (w[i] - sum_z);
        }
        for (int i = 0; i < 3; ++i) {
            z[i] += dz[i] * dt;
        }
        sum_z = z[0] + z[1] + z[2]; // 更新後の合計
        double z_norm[3];
        for (int i = 0; i < 3; ++i) {
            z_norm[i] = z[i] / sum_z;
        }
        printf("%.4f %.6f %.6f %.6f %.6f %.6f %.6f\n", t*dt, z[0], z[1], z[2], z_norm[0], z_norm[1], z_norm[2]);
    }
    return 0;
}
