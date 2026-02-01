#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 函数：检查一个整数是否是2的幂
int is_power_of_two(int n) {
    if (n <= 0) return 0;
    // 如果一个数是2的幂，其二进制表示中只有一个'1'。
    // (n & (n - 1)) 这个操作会清除该数最低位的'1'。
    // 如果结果为0，说明原始的数只有一个'1'。
    return (n & (n - 1)) == 0;
}

// 函数：计算整数形式的 log2(n)
int int_log2(int n) {
    if (n <= 0) return -1;
    int count = 0;
    while (n > 1) {
        n >>= 1; // 相当于 n = n / 2
        count++;
    }
    return count;
}

int main(int argc, char** argv) {
    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    int nbp, rank;
    // 获取总进程数
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);
    // 获取当前进程的排名
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 验证进程总数是否为2的幂，这是构成超立方体的前提
    if (!is_power_of_two(nbp)) {
        if (rank == 0) {
            // 只让进程0打印错误信息，避免信息泛滥
            fprintf(stderr, "Error: The total number of processes (%d) must be a power of two to form a hypercube.\n", nbp);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // 计算超立方体的维度 d = log2(nbp)
    int dimension = int_log2(nbp);
    
    int token; // 用于存储广播数据的令牌

    // 记录广播开始的时间
    double start_time = MPI_Wtime();

    // ====================================================================
    // 核心广播算法
    // ====================================================================

    // 步骤1: 只有 rank 0 的进程初始化令牌
    if (rank == 0) {
        token = 123; // 可以设置为任何你想要的整数值
    }

    // 迭代 d 步，d 是超立方体的维度
    for (int i = 0; i < dimension; ++i) {
        // 计算在第 i 维上的通信伙伴
        // 使用异或(XOR)操作可以精确地翻转 rank 的第 i 位，而保持其他位不变
        int partner = rank ^ (1 << i);

        // 判断当前进程在这一步的角色：发送方还是接收方
        // 核心逻辑：在第 i 步，所有二进制表示中第 i 位为 0 的进程，
        // 如果它们已经拥有了数据，就需要向它们第 i 位为 1 的伙伴发送数据。
        
        // 检查当前 rank 的第 i 位
        if ((rank & (1 << i)) == 0) {
            // 如果第 i 位是0, 我是潜在的发送方
            MPI_Send(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
        } else {
            // 如果第 i 位是1, 我是潜在的接收方
            MPI_Recv(&token, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    // 记录广播结束的时间
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // 为了验证，让所有进程都打印它们最终收到的令牌值
    printf("Process %d received the final token value: %d\n", rank, token);
    
    // 使用一个屏障来同步所有进程，确保上面的打印都完成后，再打印总结信息
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 只让 rank 0 打印总结信息
    if (rank == 0) {
        printf("\n--- Summary ---\n");
        printf("Hypercube dimension: %d\n", dimension);
        printf("Total number of processes: %d\n", nbp);
        printf("Broadcast finished in: %f seconds\n", elapsed_time);
        
        // 与简单的线性广播进行理论性能对比
        // 线性广播需要 nbp-1 步通信
        // 超立方体广播只需要 dimension = log2(nbp) 步
        if (dimension > 0) {
             printf("Theoretical speedup vs linear broadcast: %.2f\n", (double)(nbp - 1) / dimension);
        }
        printf("---------------\n");
    }

    // 结束MPI环境
    MPI_Finalize();
    return EXIT_SUCCESS;
}