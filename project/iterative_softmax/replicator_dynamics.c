#include <stdio.h>
#include <stdint.h>
#include <math.h>

// s2.13
typedef int16_t fixed16_t;
#define FIXED_SCALE 8192  // 2^13
#define FIXED_ONE FIXED_SCALE

// double <-> fixed16_t
fixed16_t double_to_fixed(double x) {
    return (fixed16_t)(x * FIXED_SCALE);
}

double fixed_to_double(fixed16_t x) {
    return (double)x / FIXED_SCALE;
}

// fixed-point mul
fixed16_t fixed_mul(fixed16_t a, fixed16_t b) {
    return (fixed16_t)(((int32_t)a * b) >> 13);
}

// fixed-point div
fixed16_t fixed_div(fixed16_t a, fixed16_t b) {
    return (fixed16_t)(((int32_t)a << 13) / b);
}

int main() {
    // input, tau
    fixed16_t w[3] = {double_to_fixed(2.0), double_to_fixed(0.2), double_to_fixed(0.7)}; // input
    fixed16_t tau = double_to_fixed(1.0);
    
    // dt*steps = 1
    double dt_values[] = {0.005, 0.0078125, 0.01, 0.015625, 0.02, 0.05, 0.1, 0.2};
    int steps_values[] = {200, 128, 100, 64, 50, 20, 10, 5};
    int num_cases = 8;

    // softmax (theory)
    double w_theory[3] = {2.0, 0.2, 0.7};
    double theory[3];
    for (int i = 0; i < 3; ++i) {
        theory[i] = exp(w_theory[i]) / (exp(w_theory[0]) + exp(w_theory[1]) + exp(w_theory[2]));
    }
    printf("Theory (softmax): [%.6f %.6f %.6f]\n\n", theory[0], theory[1], theory[2]);

    for (int case_idx = 0; case_idx < num_cases; case_idx++) {
        // softmax (fixed-point)
        fixed16_t dt_fixed = double_to_fixed(dt_values[case_idx]);
        int steps = steps_values[case_idx];
        
        fixed16_t z_fixed[3] = {double_to_fixed(0.1), double_to_fixed(0.1), double_to_fixed(0.1)};

        for (int t = 0; t <= steps; ++t) {
            fixed16_t sum_z = z_fixed[0] + z_fixed[1] + z_fixed[2];
            fixed16_t dz[3];
            for (int i = 0; i < 3; ++i) {
                fixed16_t diff = w[i] - sum_z;
                fixed16_t temp = fixed_mul(z_fixed[i], diff);
                dz[i] = fixed_mul(tau, temp);
            }
            for (int i = 0; i < 3; ++i) {
                z_fixed[i] += fixed_mul(dz[i], dt_fixed);
            }
        }

        // softmax (double)
        double w_double[3] = {2.0, 0.2, 0.7};
        double z_double[3] = {0.1, 0.1, 0.1};
        double dt_double = dt_values[case_idx];
        
        for (int t = 0; t <= steps; ++t) {
            double sum_z_double = z_double[0] + z_double[1] + z_double[2];
            double dz_double[3];
            for (int i = 0; i < 3; ++i) {
                dz_double[i] = 1.0 * z_double[i] * (w_double[i] - sum_z_double);
            }
            for (int i = 0; i < 3; ++i) {
                z_double[i] += dz_double[i] * dt_double;
            }
        }
        
        // normalize
        fixed16_t sum_z_fixed = z_fixed[0] + z_fixed[1] + z_fixed[2];
        double sum_z_double = z_double[0] + z_double[1] + z_double[2];
        
        double z_norm_fixed[3], z_norm_double[3];
        for (int i = 0; i < 3; ++i) {
            z_norm_fixed[i] = fixed_to_double(fixed_div(z_fixed[i], sum_z_fixed));
            z_norm_double[i] = z_double[i] / sum_z_double;
        }
        
        // error
        double error_fixed = 0, error_double = 0;
        for (int i = 0; i < 3; ++i) {
            error_fixed += fabs(z_norm_fixed[i] - theory[i]);
            error_double += fabs(z_norm_double[i] - theory[i]);
        }
        
        printf("dt=%.3f: fixed=[%.6f %.6f %.6f] err=%.6f, double=[%.6f %.6f %.6f] err=%.6f\n", 
               dt_values[case_idx],
               z_norm_fixed[0], z_norm_fixed[1], z_norm_fixed[2], error_fixed,
               z_norm_double[0], z_norm_double[1], z_norm_double[2], error_double);
    }
    
    return 0;
}
