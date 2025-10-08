int i, j, k, l;
int l_start, l_end;
float query[n][d] = {0}, key[n][d] = {0}, value[n][d] = {0};
float query_input[n][d], key_input[n][d], value_input[n][d];
float w_q[d][d], w_k[d][d], w_v[d][d];
float q_landmarks[m][d] = {0}, k_landmarks[m][d] = {0};
float kernel_1[n][m] = {0}, kernel_2[m][m] = {0}, kernel_3[m][n] = {0};
float sum_row[m] = {0}, sum_column[m] = {0};
float V[m][m] = {0}, KV[m][m] = {0};
float matmul_1[m][m] = {0}, matmul_2[m][m] = {0}, matmul_3[m][m] = {0};
float output_1[n][m] = {0}, output_2[m][d] = {0}, out[n][d] = {0};

const int n = 256;
const int m = 4;
const int d = 256;

// 1. linear(query,key,value)
for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
        for (k = 0; k < d; k++) {
            query[i][j] += query_input[i][k] * w_q[k][j];
        }
    }
}
for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
        for (k = 0; k < d; k++) {
            key[i][j] += key_input[i][k] * w_k[k][j];
        }
    }
}
for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
        for (k = 0; k < d; k++) {
            value[i][j] += value_input[i][k] * w_v[k][j];
        }
    }
}

// 2. get_landmarks(query,key)
l_start = 0;
for (k = 0; k < m; k++) {
    l_end = l_start + n / m;
    for (j = 0; j < d; j++) {
        for (i = l_start; i < l_end; i++) {
            q_landmarks[k][j] += query[i][j];
        }
        q_landmarks[k][j] /= (n / m);
    }
    l_start = l_end;
}
l_start = 0;
for (k = 0; k < m; k++) {
    l_end = l_start + n / m;
    for (j = 0; j < d; j++) {
        for (i = l_start; i < l_end; i++) {
            k_landmarks[k][j] += key[i][j];
        }
        k_landmarks[k][j] /= (n / m);
    }
    l_start = l_end;
}

// 3. to softmax
// softmaxの実装は考え中
// kernel_1 (q * k_landmarks^T)
for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
        for (k = 0; k < d; k++) {
            kernel_1[i][j] += query[i][k] * k_landmarks[j][k];
        }
        kernel_1[i][j] /= 16.0f; // sqrt(d) = 16
    }
}
// softmax(kernel_1)
float kernel_1_softmax[n][m] = {0};
float z[m] = {0}, dz[m] = {0};
float sum_z;
const int steps = 64;
const float dt = 0.015625f;

for (i = 0; i < n; i++) {
    for (int t = 0; t <= steps; t++) {
        // init 
        if (t == 0) {
            for (j = 0; j < m; j++) {
                z[j] = 0.125f;
            }
            sum_z = 0.125f * m;
        } else {
            for (j = 0; j < m; j++) {
                dz[j] = z[j] * (kernel_1[i][j] - sum_z);
            }
            for (j = 0; j < m; j++) {
                z[j] += dz[j] * dt;
            }
            // update sum_z
            sum_z = 0.0f;
            for (j = 0; j < m; j++) {
                sum_z += z[j];
            }
        }
    }
    // normalize
    for (j = 0; j < m; j++) {
        kernel_1_softmax[i][j] = z[j] / sum_z;
    }
}

// kernel_2 (q_landmarks * k_landmarks^T)
for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
        for (k = 0; k < d; k++) {
            kernel_2[i][j] += q_landmarks[i][k] * k_landmarks[j][k];
        }
        kernel_2[i][j] /= 16.0f;
    }
}
// softmax(kernel_2)
float kernel_2_softmax[m][m] = {0};
float z[m] = {0}, dz[m] = {0};
float sum_z;
const int steps = 64;
const float dt = 0.015625f;

for (i = 0; i < m; i++) {
    for (int t = 0; t <= steps; t++) {
        // init 
        if (t == 0) {
            for (j = 0; j < m; j++) {
                z[j] = 0.125f;
            }
            sum_z = 0.125f * m;
        } else {
            for (j = 0; j < m; j++) {
                dz[j] = z[j] * (kernel_2[i][j] - sum_z);
            }
            for (j = 0; j < m; j++) {
                z[j] += dz[j] * dt;
            }
            // update sum_z
            sum_z = 0.0f;
            for (j = 0; j < m; j++) {
                sum_z += z[j];
            }
        }
    }
    // normalize
    for (j = 0; j < m; j++) {
        kernel_2_softmax[i][j] = z[j] / sum_z;
    }
}

// kernel_3 (q_landmarks * key)
for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
        for (k = 0; k < d; k++) {
            kernel_3[i][j] += q_landmarks[i][k] * key[j][k];
        }
        kernel_3[i][j] /= 16.0f;
    }
}
// softmax(kernel_3)
float kernel_3_softmax[m][n] = {0};
float z[n] = {0}, dz[n] = {0};
float sum_z;
const int steps = 64;
const float dt = 0.015625f;

for (i = 0; i < m; i++) {
    for (int t = 0; t <= steps; t++) {
        // init 
        if (t == 0) {
            for (j = 0; j < n; j++) {
                z[j] = 0.125f;
            }
            sum_z = 0.125f * n;
        } else {
            for (j = 0; j < n; j++) {
                dz[j] = z[j] * (kernel_3[i][j] - sum_z);
            }
            for (j = 0; j < n; j++) {
                z[j] += dz[j] * dt;
            }
            // update sum_z
            sum_z = 0.0f;
            for (j = 0; j < n; j++) {
                sum_z += z[j];
            }
        }
    }
    // normalize
    for (j = 0; j < n; j++) {
        kernel_3_softmax[i][j] = z[j] / sum_z;
    }
}

// 4. iterative_inv
// V0
for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
        sum_row[i] += abs(kernel_2[i][j]);
        sum_column[j] += abs(kernel_2[i][j]);
    }
}
float max_row = sum_row[0], max_col = sum_column[0];
for (i = 1; i < m; i++) {
    if (sum_row[i] > max_row) max_row = sum_row[i];
    if (sum_column[i] > max_col) max_col = sum_column[i];
}
for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
        V[i][j] = kernel_2[j][i] / (max_row * max_col);
    }
}

// iterative calculation
for (l = 0; l < 6; l++) {
    // KV
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            KV[i][j] = 0;
            for (k = 0; k < m; k++) {
                KV[i][j] += kernel_2[i][k] * V[k][j];
            }
        }
    }
    // KV * (7*I - KV)
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            matmul_1[i][j] = 0;
            for (k = 0; k < m; k++) {
                if (k == j) {
                    matmul_1[i][j] += KV[i][k] * (7 - KV[k][j]);
                } else {
                    matmul_1[i][j] -= KV[i][k] * KV[k][j];
                }
            }
        }
    }
    // KV * (15*I - (7*I - KV))
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            matmul_2[i][j] = 0;
            for (k = 0; k < m; k++) {
                if (k == j) {
                    matmul_2[i][j] += KV[i][k] * (15 - matmul_1[k][j]);
                } else {
                    matmul_2[i][j] -= KV[i][k] * matmul_1[k][j];
                }
            }
        }
    }
    // 0.25*V * (13*I - (15*I - (7*I - KV)))
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            matmul_3[i][j] = 0;
            for (k = 0; k < m; k++) {
                if (k == j) {
                    matmul_3[i][j] += (V[i][k] / 4.0f) * (13 - matmul_2[k][j]);
                } else {
                    matmul_3[i][j] -= (V[i][k] / 4.0f) * matmul_2[k][j];
                }
            }
        }
    }
    // V = matmul_3;
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
            V[i][j] = matmul_3[i][j];
}

// 5. output
// kernel_1 * kernel_2^T
for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
        output_1[i][j] = 0;
        for (k = 0; k < m; k++) {
            output_1[i][j] += kernel_1[i][k] * matmul_3[k][j];
        }
    }
}
// kernel_3 * Value
for (i = 0; i < m; i++) {
    for (j = 0; j < d; j++) {
        output_2[i][j] = 0;
        for (k = 0; k < n; k++) {
            output_2[i][j] += kernel_3[i][k] * value[k][j];
        }
    }
}
// output_1 * output_2
for (i = 0; i < n; i++) {
    for (j = 0; j < d; j++) {
        out[i][j] = 0;
        for (k = 0; k < m; k++) {
            out[i][j] += output_1[i][k] * output_2[k][j];
        }
    }
}