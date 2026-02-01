#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
    for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
      for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
        C(i, j) += A(i, k) * B(k, j);
}

// 定义块的大小，这是一个需要调优的参数
const int szBlock = 64;
Matrix prodMatMatBlocked(const Matrix& A, const Matrix& B) {
    Matrix C(A.nbRows, B.nbCols, 0.0);
    // 三层新的循环，用于遍历块
    #pragma omp parallel for
    for (int iRowBlk = 0; iRowBlk < A.nbRows; iRowBlk += szBlock) {
        for (int jColBlk = 0; jColBlk < B.nbCols; jColBlk += szBlock) {
            for (int kColBlk = 0; kColBlk < A.nbCols; kColBlk += szBlock) {
                // 调用我们已有的函数来计算子块之间的乘积
                prodSubBlocks(iRowBlk, jColBlk, kColBlk, szBlock, A, B, C);
            }
        }
    }
    return C;
}
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  return prodMatMatBlocked(A, B);
}
