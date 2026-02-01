from mpi4py import MPI
import time
import numpy as np

comm = MPI.COMM_WORLD
nbp = comm.Get_size()  # 总进程数
rank = comm.Get_rank() # 当前进程编号

# 总样本数
total_samples = 1000000000

# 每个进程负责的样本数
samples_per_process = total_samples // nbp

if rank == 0:
    beg = time.time()

# --- 每个进程独立计算自己的部分 ---
# 注意：并行化后，每个进程创建的数组变小了，节省了单核内存压力
x = 2. * np.random.random_sample((samples_per_process,)) - 1.
y = 2. * np.random.random_sample((samples_per_process,)) - 1.
local_sum = np.sum(x*x + y*y < 1.)

# --- 汇总结果 ---
# 使用 MPI 的 reduce 操作将所有进程的 local_sum 相加，结果汇总到 rank 0
total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

if rank == 0:
    approx_pi = 4. * total_sum / (samples_per_process * nbp)
    end = time.time()
    print(f"Total Processes: {nbp}")
    print(f"Temps pour calculer pi : {end - beg} secondes")
    print(f"Pi vaut environ {approx_pi}")