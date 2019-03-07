import numpy as np

def LS_oracle(X_test, Y_test):
    X_n = X_test
    H_inv = np.matmul(X_n.T, np.linalg.pinv(Y_test).T)
    return np.matmul(H_inv, Y_test.T).T
