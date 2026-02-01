import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
nbp = comm.Get_size()
rank = comm.Get_rank()

# 总维度
N = 16384

# 1. 计算 Nloc
if N % nbp != 0:
    if rank == 0:
        print("Error: N must be divisible by nbp")
    comm.Abort()

Nloc = N // nbp

# 2. 每个进程只组装属于自己的矩阵部分 (N 行 x Nloc 列)
# 该进程负责的列索引范围
start_col = rank * Nloc
end_col = (rank + 1) * Nloc

# 组装局部矩阵 A_local (形状: N x Nloc)
# 注意：j 对应行索引(0..N-1)，i 对应全局列索引(start..end)
A_local = np.array([[(i + j) % N + 1. for i in range(start_col, end_col)] 
                    for j in range(N)])

# 组装局部向量 u_local (长度: Nloc)
u_local = np.array([i + 1. for i in range(start_col, end_col)])

# 开始计时
comm.Barrier()
start_time = MPI.Wtime()

# 3. 计算局部贡献 (Partial Sum)
# 结果是一个长度为 N 的向量
v_local = A_local.dot(u_local)

# 4. 全局汇总：将所有进程的部分和相加，并将结果分发给所有人
v_final = np.zeros(N, dtype=np.double)
comm.Allreduce(v_local, v_final, op=MPI.SUM)

end_time = MPI.Wtime()


# 5. 输出结果
if rank == 0:
    print(f"Number of processes: {nbp}")
    print(f"Parallel computation time: {end_time - start_time:.6f} s")