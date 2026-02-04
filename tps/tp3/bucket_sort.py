from mpi4py import MPI
import numpy as np
import time

def parallel_bucket_sort(N=100):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # --- 1. 数据准备与分发 ---
    if rank == 0:
        data = np.random.uniform(0, 100, N).astype(np.float64)
        print(f"Process 0 generated {N} random numbers.")
        # 将数据切分为大致相等的块
        chunks = np.array_split(data, size)
    else:
        chunks = None

    # 分发数据给每个进程
    local_data = comm.scatter(chunks, root=0)

    # --- 2. 局部排序 ---
    local_data.sort()

    # --- 3. 正规采样 (Regular Sampling) ---
    # 每个进程选取 size 个样本 (讲义第54页提到选取 nbp+1 个值)
    indices = np.linspace(0, len(local_data) - 1, size, dtype=int)
    local_samples = local_data[indices]

    # --- 4. 确定全局桶边界 (Pivots) ---
    all_samples = comm.gather(local_samples, root=0)

    pivots = None
    if rank == 0:
        # 将所有样本汇总并排序
        global_samples = np.concatenate(all_samples)
        global_samples.sort()
        # 选取 size-1 个枢轴(pivots)作为桶的边界
        # 比如有4个进程，我们需要3个边界来分出4个区间
        pivot_indices = np.linspace(0, len(global_samples) - 1, size + 1, dtype=int)[1:-1]
        pivots = global_samples[pivot_indices]
    
    # 广播边界给所有进程
    pivots = comm.bcast(pivots, root=0)

    # --- 5. 数据划分与全局交换 (All-to-all) ---
    # 每个进程根据 pivots 将本地数据分成 size 份
    # 使用 np.searchsorted 查找数据应该去哪个桶
    bucket_indices = np.searchsorted(pivots, local_data)
    
    # 准备要发送给各个进程的数据列表
    data_to_send = [local_data[bucket_indices == i] for i in range(size)]
    
    # 全局交换：每个进程发送数据给对应的桶进程，同时也接收属于自己的桶的数据
    received_chunks = comm.alltoall(data_to_send)

    # --- 6. 本地最终排序 ---
    # 将接收到的所有块合并并排序
    my_bucket = np.concatenate(received_chunks)
    my_bucket.sort()

    # --- 7. 汇总结果 ---
    sorted_all = comm.gather(my_bucket, root=0)

    if rank == 0:
        final_array = np.concatenate(sorted_all)
        return final_array
    return None

if __name__ == "__main__":
    # 测试数据量
    N_TOTAL = 100000000
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    start_time = time.time()
    result = parallel_bucket_sort(N_TOTAL)
    end_time = time.time()

    if rank == 0:
        print(f"Sorting complete! Time taken: {end_time - start_time:.4f} s")        
        # 验证是否有序
        is_sorted = np.all(result[:-1] <= result[1:])
        print(f"The result is sorted: {is_sorted}")