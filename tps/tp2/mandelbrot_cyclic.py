import numpy as np
from mpi4py import MPI
from PIL import Image
import matplotlib.cm
from math import log
from dataclasses import dataclass
from typing import Union

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

# --- MPI 初始化 ---
comm = MPI.COMM_WORLD
nbp = comm.Get_size()
rank = comm.Get_rank()

width, height = 1024, 1024
mandelbrot_set = MandelbrotSet(max_iterations=512, escape_radius=2.)

scaleX = 3./width
scaleY = 2.25/height

# --- 核心改进：循环划分逻辑 ---
# 每个进程计算的行：rank, rank+nbp, rank+2*nbp...
my_rows = range(rank, height, nbp)
local_convergence = []

comm.Barrier()
deb = MPI.Wtime()

my_rows_data = [] # 存储当前进程计算的所有行数据
for y in my_rows:
    row_values = np.empty(width, dtype=np.double)
    for x in range(width):
        c_scalar = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        # 调用 1.1 问那种逐个像素计算的函数
        row_values[x] = mandelbrot_set.convergence(c_scalar, smooth=True)
    my_rows_data.append(row_values)

local_convergence = np.array(my_rows_data)

# --- 结果汇总 ---
# 由于是循环分配，Gather 后的数组行顺序是乱的
# 做法：每个进程发送自己的数据，rank 0 根据索引拼回原图
if rank == 0:
    full_image = np.empty((height, width), dtype=np.double)
    # 将自己的部分放入
    for i, y in enumerate(my_rows):
        full_image[y] = local_convergence[i]
    
    # 接收其他进程的部分
    for p in range(1, nbp):
        # 接收数据
        buf = np.empty((len(range(p, height, nbp)), width), dtype=np.double)
        comm.Recv(buf, source=p)
        # 根据该进程对应的行号放回原位
        for i, y in enumerate(range(p, height, nbp)):
            full_image[y] = buf[i]
else:
    # 非0进程发送本地计算的所有行
    comm.Send(local_convergence, dest=0)

fin = MPI.Wtime()

if rank == 0:
    print(f"Total number of processes: {nbp}")
    print(f"Total time spent on parallel computing (including communication): {fin - deb} seconds")
    
    # 图像处理（注意转置回原来的逻辑）
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(full_image)*255))
    image.show()
    input("Press Enter to close and exit...")