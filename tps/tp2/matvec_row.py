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

# 2. 每个进程只组装属于自己的矩阵部分 (Nloc 行 x N 列)
# 该进程负责的行索引范围
start_row = rank * Nloc
end_row = (rank + 1) * Nloc

# 组装局部矩阵 A_local (形状: Nloc x N)
# 注意：j 对应局部行索引(从全局的 start..end)，i 对应列索引(0..N-1)
A_local = np.array([[(i + j) % N + 1. for i in range(N)] 
                    for j in range(start_row, end_row)])

# 组装完整的向量 u (每个进程都需要完整副本进行局部计算)
u = np.array([i + 1. for i in range(N)])

# 开始计时
comm.Barrier()
start_time = MPI.Wtime()

# 3. 计算局部结果片段 (Local Segment)
# 结果是一个长度为 Nloc 的向量，它是 v 的一部分
v_segment = A_local.dot(u)

# 4. 全局汇总：收集所有进程的片段，使每个人都拥有完整的 v (长度为 N)
v_final = np.empty(N, dtype=np.double)
# Allgather 会自动根据 rank 顺序拼接数据
comm.Allgather(v_segment, v_final)

end_time = MPI.Wtime()

# 5. 输出结果
if rank == 0:
    print(f"Number of processes: {nbp}")
    print(f"Parallel computation time: {end_time - start_time:.6f} s")