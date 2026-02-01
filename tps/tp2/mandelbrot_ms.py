import numpy as np
from mpi4py import MPI
from PIL import Image
import matplotlib.cm
from dataclasses import dataclass
from typing import Union
from math import log

# MandelbrotSet 类使用 1.1 的高效标量版
@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> Union[int, float]:
        if c.real*c.real+c.imag*c.imag < 0.0625: return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625: return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)): return self.max_iterations
        
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth: return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations
    
comm = MPI.COMM_WORLD
nbp = comm.Get_size()
rank = comm.Get_rank()

width, height = 1024, 1024
mandelbrot_set = MandelbrotSet(max_iterations=512)
scaleX, scaleY = 3./width, 2.25/height

# 信号常量
TAG_TASK = 1     # 发送任务
TAG_RESULT = 2   # 接收结果
TAG_DONE = 3     # 终止信号

if rank == 0:
    # --- MASTER 逻辑 ---
    full_image = np.empty((height, width), dtype=np.double)
    deb = MPI.Wtime()
    
    next_row = 0
    rows_received = 0
    
    # 1. 初始分发：给每个 Slave 发第一行
    for p in range(1, nbp):
        if next_row < height:
            comm.send(next_row, dest=p, tag=TAG_TASK)
            next_row += 1

    # 2. 动态分发循环
    while rows_received < height:
        status = MPI.Status()
        # 接收任何一个 Slave 传回的整行结果
        row_data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
        slave_p = status.Get_source()
        
        # 记录结果（row_data 包含 [行号, 该行数组]）
        row_idx, row_values = row_data
        full_image[row_idx] = row_values
        rows_received += 1
        
        # 如果还有任务，就发给刚空闲的这个 Slave
        if next_row < height:
            comm.send(next_row, dest=slave_p, tag=TAG_TASK)
            next_row += 1
        else:
            # 没任务了，让它休息
            comm.send(None, dest=slave_p, tag=TAG_DONE)

    fin = MPI.Wtime()
    
    print(f"Total number of processes: {nbp}")
    print(f"Total time spent on parallel computing (including communication): {fin - deb} seconds")
    # 图像处理
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(full_image)*255))
    image.show()
    input("Press Enter to close and exit...")

else:
    # --- SLAVE 逻辑 ---
    while True:
        status = MPI.Status()
        # 等待任务
        task = comm.recv(source=0, status=status)
        
        if status.Get_tag() == TAG_DONE:
            break
            
        y = task
        row_values = np.empty(width, dtype=np.double)
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            row_values[x] = mandelbrot_set.convergence(c)
        
        # 发回 [行号, 数据]
        comm.send([y, row_values], dest=0, tag=TAG_RESULT)

MPI.Finalize()