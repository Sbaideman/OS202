#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 自定义一个简单的线程安全随机数生成器 (LCG算法)
double get_random(unsigned int* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (double)(*seed & 0x7fffffff) / 2147483647.0;
}

int main(int argc, char* argv[]) {
    long long total_samples = 100000000; // 默认1亿次投掷
    if (argc > 1) {
        total_samples = atoll(argv[1]);
    }

    // 记录开始时间
    double start_time = omp_get_wtime();

    long long nbDarts = 0;

    // --- OpenMP 并行区域 ---
    // reduction(+:nbDarts) 自动处理每个线程的局部计数并最后汇总
    #pragma omp parallel reduction(+:nbDarts)
    {
        // 为每个线程准备一个独立的随机数种子，避免线程间的随机数竞争
        unsigned int seed = time(NULL) ^ omp_get_thread_num();

        #pragma omp for
        for (long long i = 0; i < total_samples; i++) {
            // 使用自定义的随机函数生成 [-1, 1] 之间的坐标
            double x = get_random(&seed) * 2.0 - 1.0;
            double y = get_random(&seed) * 2.0 - 1.0;

            if (x * x + y * y <= 1.0) {
                nbDarts++;
            }
        }
    }

    // 记录结束时间
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    // 计算结果
    double pi_estimate = 4.0 * (double)nbDarts / (double)total_samples;
    printf("Threads used      : %d\n", omp_get_max_threads());
    printf("Total samples     : %lld\n", total_samples);
    printf("Estimated Pi      : %.10f\n", pi_estimate);
    printf("Error             : %.10f\n", fabs(pi_estimate - M_PI));
    printf("Execution time    : %.6f seconds\n", elapsed_time);

    return 0;
}