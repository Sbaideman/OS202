#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Pi值
#ifndef M_PI
const double M_PI = 3.14159265358979323846;
#endif

// 主函数
int main(int argc, char** argv) {
    long long total_points;

    // 从命令行参数获取总投掷点数，默认为1亿
    if (argc > 1) {
        total_points = atoll(argv[1]);
    } else {
        total_points = 100000000; // 1亿
    }

    // 初始化MPI环境
    MPI_Init(&argc, &argv);

    int nbp, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ====================================================================
    // 1. 任务分解
    // ====================================================================
    long long points_per_process = total_points / nbp;
    // 为了确保总点数不变，让最后一个进程处理余数
    if (rank == nbp - 1) {
        points_per_process += total_points % nbp;
    }

    // 使用不同的随机数种子来确保每个进程生成不同的随机数序列
    // 这是一个简单但有效的方法
    srand(time(NULL) + rank);

    // 记录开始时间 (仅在rank 0进行)
    double start_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // ====================================================================
    // 2. 独立计算
    // ====================================================================
    long long local_points_in_circle = 0;
    for (long long i = 0; i < points_per_process; ++i) {
        // 生成 [-1, 1] 范围内的随机坐标 (x, y)
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;

        // 判断点是否在单位圆内 (x^2 + y^2 <= 1)
        if (x * x + y * y <= 1.0) {
            local_points_in_circle++;
        }
    }

    // ====================================================================
    // 3. 结果聚合 (使用 MPI_Reduce)
    // ====================================================================
    long long global_points_in_circle;

    // MPI_Reduce 会将所有进程的 local_points_in_circle 的值
    // 使用 MPI_SUM 操作进行相加，并将最终结果存放到 rank 0 进程的
    // global_points_in_circle 变量中。
    MPI_Reduce(&local_points_in_circle, &global_points_in_circle, 1, MPI_LONG_LONG, 
               MPI_SUM, 0, MPI_COMM_WORLD);

    // ====================================================================
    // 4. 最终计算与输出 (仅在 rank 0 进行)
    // ====================================================================
    if (rank == 0) {
        // 记录结束时间
        double end_time = MPI_Wtime();
        
        // 计算Pi的估算值
        double pi_estimate = 4.0 * global_points_in_circle / total_points;
        
        // 计算执行时间
        double elapsed_time = end_time - start_time;

        printf("Total points thrown: %lld\n", total_points);
        printf("Points inside circle: %lld\n", global_points_in_circle);
        printf("Estimated value of Pi: %f\n", pi_estimate);
        printf("Error from true Pi: %f\n", fabs(pi_estimate - M_PI));
        printf("Number of processes: %d\n", nbp);
        printf("Execution time: %f seconds\n", elapsed_time);
    }

    // 结束MPI环境
    MPI_Finalize();

    return EXIT_SUCCESS;
}