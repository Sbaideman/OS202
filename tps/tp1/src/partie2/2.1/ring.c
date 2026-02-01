#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    // 获取进程总数
    int nbp;
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);

    // 获取当前进程的排名
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 检查是否至少有两个进程，否则环无法形成
    if (nbp < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires at least 2 processes to run.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int token; // 用来存储令牌的变量

    // ====================================================================
    // 核心逻辑：根据进程的rank来决定行为
    // ====================================================================

    if (rank == 0) {
        // --- 进程0的行为 ---
        // 1. 初始化令牌为1
        token = 1;
        
        // 计算下一个进程的rank
        int next_rank = (rank + 1) % nbp;

        // 2. 将令牌发送给下一个进程 (rank 1)
        // 参数: 数据地址, 个数, 类型, 目标rank, 标签, 通信域
        MPI_Send(&token, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        printf("Process %d initialized token with value %d and sent it to process %d\n", rank, token, next_rank);

        // 6. 从最后一个进程接收最终的令牌
        int prev_rank = (rank - 1 + nbp) % nbp; // (0 - 1 + nbp) % nbp = nbp - 1
        // 参数: 数据地址, 个数, 类型, 来源rank, 标签, 通信域, 状态
        MPI_Recv(&token, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 7. 打印最终结果
        printf("Process %d received the final token with value: %d\n", rank, token);
        // 验证结果：最终值应该等于进程总数 nbp
        printf("Verification: Final value should be equal to nbp (%d). Is it? %s\n", nbp, (token == nbp ? "Yes" : "No"));

    } else {
        // --- 其他所有进程 (rank > 0) 的行为 ---
        // 计算上一个和下一个进程的rank
        int prev_rank = (rank - 1 + nbp) % nbp;
        int next_rank = (rank + 1) % nbp;

        // 1. 从上一个进程接收令牌
        MPI_Recv(&token, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token with value %d from process %d\n", rank, token, prev_rank);

        // 2. 将令牌的值加一
        token++;

        // 3. 将更新后的令牌发送给下一个进程
        MPI_Send(&token, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        printf("Process %d incremented token to %d and sent it to process %d\n", rank, token, next_rank);
    }

    // 结束MPI环境
    MPI_Finalize();

    return EXIT_SUCCESS;
}