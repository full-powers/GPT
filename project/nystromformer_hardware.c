const int n = 256;
const int m = 4;
const int d = 256;

// 1. linear(query,key,value)
for (i=0, i<n, i++){
    for (j=0, j<d, j++){
        for (k=0, k<d, k++){
            query[i][j] += query_input[i][k] * w_q[k][j];
        }
    }
}
for (i=0, i<n, i++){
    for (j=0, j<d, j++){
        for (k=0, k<d, k++){
            key[i][j] += key_input[i][k] * w_k[k][j];
        }
    }
}
for (i=0, i<n, i++){
    for (j=0, j<d, j++){
        for (k=0, k<d, k++){
            value[i][j] += value_input[i][k] * w_v[k][j];
        }
    }
}

// 2. get_landmarks(query,key)
for (k=0, k<m, k++){
    l_end = l_start + n/m;
    for (j=0, j<d, j++){
        for (i=l_start, i<l_end, i++){
            q_landmarks[k][j] += query[i][j]
        }
        q_landmarks[k][j] /= n/m
    }
    l_start = l_end;
}
for (k=0, k<m, k++){
    l_end = l_start + n/m;
    for (j=0, j<d, j++){
        for (i=l_start, i<l_end, i++){
            k_landmarks[k][j] += key[i][j]
        }
        k_landmarks[k][j] /= n/m
    }
    l_start = l_end;
}


// 3. to softmax
// kernel_1 (q * k_landmarks^T)
for (i=0, i<n, i++){
    for (j=0, j<m, j++){
        for (k=0, k<d, k++){
            kernel_1[i][j] += query[i][k] * k_landmarks[j][k]
        }
    }
}
// kernel_2 (q_landmarks * k_landmarks^T)
for (i=0, i<m, i++){
    for (j=0, j<m, j++){
        for (k=0, k<d, k++){
            kernel_2[i][j] += q_landmarks[i][k] * k_landmarks[j][k]
        }
    }
}
// kernel_3 (q_landmarks * key)
for (i=0, i<m, i++){
    for (j=0, j<n, j++){
        for (k=0, k<d, k++){
            kernel_3[i][j] += q_landmarks[i][k] * key[j][k]
        }
    }
}

// softmaxの実装考える！！！
// スケーリングも

// 4. iterative_inv
// V0
for (i=0, i<m, i++){
    for (j=0, j<m, j++){
        sum_row[i] += abs(kernel_2[i][j])
        sum_column[j] += abs(kernel_2[i][j])
    }
}

for (i=0, i<m, i++){
    for (j=0, j<m, j++){
        V[i][j] = kernel_2[j][i] / (max(sum_row) * max(sum_column))
    }
}

// iterative calclation
for (l=0, l<6, l++){
    // KV
    for (i=0, i<m, i++){
        for (j=0, j<m, j++){
            for (k=0, k<m, k++){
                KV[i][j] += kernel_2[i][k] * V[k][j];
            }
        }
    }
    // KV * (7*I - KV)
    for (i=0, i<m, i++){
        for (j=0, j<m, j++){
            for (k=0, k<m, k++){
                if (k == j){
                    matmul_1[i][j] += KV[i][k] * (7 - KV[k][j])
                }
                else{
                    matmul_1[i][j] -= KV[i][k] * KV[k][j];
                }
            }
        }
    }
    // KV * (15*I - (7*I - KV))
    for (i=0, i<m, i++){
        for (j=0, j<m, j++){
            for (k=0, k<m, k++){
                if (k == j){
                    matmul_2[i][j] += KV[i][k] * (15 - matmul_1[k][j])
                }
                else{
                    matmul_2[i][j] -= KV[i][k] * matmul_1[k][j];
                }
            }
        }
    }
    // 0.25*V * (13*I - (15*I - (7*I - KV)))
    for (i=0, i<m, i++){
        for (j=0, j<m, j++){
            for (k=0, k<m, k++){
                if (k == j){
                    matmul_3[i][j] += (V[i][k] / 4) * (13 - matmul_2[k][j])
                }
                else{
                    matmul_3[i][j] -= (V[i][k] / 4) * matmul_2[k][j];
                }
            }
        }
    }
    V = matmul_3;
}

// 5. output
// kernel_1 * kernel_2^T
for (i=0, i<n, i++){
    for (j=0, j<m, j++){
        for (k=0, k<m, k++){
            output_1[i][j] += kernel_1[i][k] * matmul_3[k][j]
        }
    }
}
// kernel_3 * Value
for (i=0, i<m, i++){
    for (j=0, j<d, j++){
        for (k=0, k<n, k++){
            output_2[i][j] += kernel_3[i][k] * Value[k][j]
        }
    }
}
// output_1 * output_2
for (i=0, i<n, i++){
    for (j=0, j<d, j++){
        for (k=0, k<m, k++){
            out[i][j] += output_1[i][k] * output_2[k][j]
        }
    }
}