import numpy as np

# 定义两个单应性矩阵
H1 = np.array([[1.52931434e+00, 8.72406417e-02, -1.78641929e+02],
               [-2.58135183e-02, 1.55964897e+00, -8.41393200e+01],
               [-4.80114961e-06, 5.09722643e-05, 1.00000000e+00]])

H2 = np.array([[1.56004648e+00, 8.61408870e-02, -1.86876977e+02],
               [-1.86268318e-02, 1.55864070e+00, -8.80253103e+01],
               [1.54560168e-05, 4.11948349e-05, 1.00000000e+00]])

# 计算Frobenius范数
frobenius_norm = np.linalg.norm(H1 - H2, 'fro')

# 计算相对误差
relative_error = frobenius_norm / (np.linalg.norm(H1, 'fro') + np.linalg.norm(H2, 'fro'))

print("Frobenius Norm:", frobenius_norm)
print("Relative Error:", relative_error)

