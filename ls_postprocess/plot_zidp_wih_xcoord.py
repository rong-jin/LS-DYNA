"""
Author: Rong Jin, University of Kentucky
Date: 04-15-2025
"""
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

plt.rcParams['font.family'] = 'Times New Roman'          # 设置字体，例如：'SimHei'、'Arial'、'Times New Roman'
# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
# 假设文件为 MATLAB 格式文件，如果文件名没有扩展名，请根据实际情况修改
Coordinates1 = np.genfromtxt('xcoord', dtype=float)  # 或者 'Coordinates1'，取决于文件格式
Displacements1 = np.genfromtxt('zdisp', dtype=float)
print(Coordinates1)
print(Displacements1)

# 从加载的数据字典中提取变量（变量名称需与MATLAB中保存时一致）
# Coordinates1 = coords_data['Coordinates1']
# Displacements1 = disps_data['Displacements1']

N_time = 51
N_node = Displacements1.shape[0] // N_time

# 提取基准坐标（第二列）和时间（第一列），注意 Python 索引从 0 开始
Coord0 = Coordinates1[0::N_time, 1] * 10.0
Time = Coordinates1[:N_time, 0].flatten() * 1e6

# ---------------------------
# 2. 构建各节点时程数据
# ---------------------------
# 为每个节点提取 N_time 个数据，构建矩阵
Disp1 = np.zeros((N_node, N_time))
Coord1 = np.zeros((N_node, N_time))
for i in range(N_node):
    # MATLAB中：(i-1)*N_time+1:i*N_time 对应 Python 的 i*N_time:(i+1)*N_time
    # 注意 MATLAB 中第二列对应 Python 的索引1
    Disp1[i, :] = -Displacements1[i * N_time:(i + 1) * N_time, 1].flatten() * 10.0
    Coord1[i, :] = Coordinates1[i * N_time:(i + 1) * N_time, 1].flatten() * 10.0

# ---------------------------
# 3. 数据排序与对称数据构造
# ---------------------------
# 对位移数据处理：
tmp_disp = np.hstack((Coord0.reshape(-1, 1), Disp1))
# 按第一列（Coord0）排序
tmp_disp_sorted = tmp_disp[np.argsort(tmp_disp[:, 0])]
Disp1_sorted = tmp_disp_sorted[:, 1:]
# 翻转除第一行外的部分（MATLAB中 flip(Disp1(2:end,:))）
Disp2 = np.flip(Disp1_sorted[1:, :], axis=0)
# 合并构造对称数据
Disp = np.vstack((Disp2, Disp1_sorted))

# 对坐标数据处理：
tmp_coord = np.hstack((Coord0.reshape(-1, 1), Coord1))
tmp_coord_sorted = tmp_coord[np.argsort(tmp_coord[:, 0])]
Coord1_sorted = tmp_coord_sorted[:, 1:]
Coord2 = np.flip(Coord1_sorted[1:, :], axis=0)
Coord = np.vstack((-Coord2, Coord1_sorted))

# ---------------------------
# 4. 设置绘图参数与绘图
# ---------------------------
N_ref = 11
# 实验数据时刻（MATLAB中 T_exp = [1 6 11 16 21]）
T_exp = np.array([1, 6, 11, 16, 21])
# 仿真时刻：T_sim = N_ref + 2*(T_exp - 1)  注意 MATLAB 下标从1开始，
# 转换到 Python 时需减 1
T_sim = (N_ref + 2 * (T_exp - 1)) - 1

# 设置颜色映射
cnum = 256
colors = plt.cm.hsv(np.linspace(0, 1, cnum))

plt.figure(figsize=(8, 6))

# 如果有实验数据（例如 Coord_exp 和 Disp_exp），可以先绘制实验曲线（虚线），
# 这里假设没有相应文件，所以只绘制仿真数据（实线）
for i in range(len(T_exp)):
    alpha = int(round(1 + i * (cnum - 1) / len(T_exp))) - 1  # 转换为 0-index
    # 提取仿真数据中在时刻 T_sim[i] 的数据
    tmp_x = Coord[:, T_sim[i]]
    tmp_y = Disp[:, T_sim[i]]
    # 去除位移为0的数据点
    nonzero = tmp_y != 0
    plt.plot(tmp_x[nonzero], tmp_y[nonzero], '-', color=colors[alpha],
             linewidth=3, label=f'{i} μs' if i < len(T_exp) else None)
    # 若需要区分实验数据（虚线）与仿真数据（实线），可添加相应绘图命令

plt.xlabel('Coordinate (mm)', fontsize=16, fontweight='bold')
plt.ylabel('Backface deflection (mm)', fontsize=16, fontweight='bold')
plt.xlim([-20, 20])
plt.xticks(np.arange(-20, 21, 5), fontsize=12)
plt.yticks(fontsize=12)
# plt.tick_params(width=2, length=8)
plt.legend(['0 μs', '1 μs', '2 μs', '3 μs', '4 μs'], loc='upper right', fontsize=12, frameon=True)
plt.show()
