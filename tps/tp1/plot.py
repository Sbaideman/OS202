import matplotlib.pyplot as plt
import numpy as np

# 1. 您的表格数据
threads = [1, 2, 4, 8, 16]
times = [7.64, 3.77, 3.01, 2.43, 1.94]
speedups = [1.00, 2.03, 2.54, 3.14, 3.94]

# --- 开始绘图 ---

# 创建一个图形窗口和一个主坐标轴 (ax1)
# figsize 用于调整图形的大小
fig, ax1 = plt.subplots(figsize=(10, 7))

# --- 在主坐标轴 ax1 (左Y轴) 上绘制 "执行时间" ---
# 为线条和标签选择一种颜色
color_time = 'tab:green'
ax1.set_xlabel('Nombre de Threads (n)', fontsize=14)
ax1.set_ylabel('Temps d\'exécution (s)', fontsize=14)
# 绘制执行时间曲线
line1 = ax1.plot(threads, times, color=color_time, marker='s', linestyle='-', label='Temps d\'exécution')
# 设置左Y轴刻度标签的颜色
ax1.tick_params(axis='y', labelcolor=color_time)
# 设置X轴刻度，确保所有数据点都清晰显示
ax1.set_xticks(threads)
ax1.grid(True, linestyle=':') # 添加网格线

# --- 创建一个共享X轴的次坐标轴 (ax2) 用于 "加速比" ---
ax2 = ax1.twinx()

# --- 在次坐标轴 ax2 (右Y轴) 上绘制 "加速比" ---
# 为线条和标签选择另一种颜色
color_speedup = 'tab:blue'
ax2.set_ylabel('Accélération', fontsize=14)
# 绘制实际加速比曲线
line2 = ax2.plot(threads, speedups, color=color_speedup, marker='o', linestyle='-', label='Accélération Réelle')
# 绘制理想加速比曲线作为对比
line3 = ax2.plot(threads, threads, color='tab:red', linestyle='--', marker='x', label='Accélération Idéale')
# 设置右Y轴刻度标签的颜色
ax2.tick_params(axis='y', labelcolor=color_speedup)

# --- 添加标题和图例 ---
plt.title('Performance de la Parallélisation OpenMP', fontsize=18)

# 合并两个坐标轴的图例
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', fontsize=12)

# 自动调整布局，防止标签被裁切
fig.tight_layout()

# 显示最终的图形
plt.show()