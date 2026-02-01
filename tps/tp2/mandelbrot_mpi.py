import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
import matplotlib.cm
from mpi4py import MPI
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

# 计算每个进程负责的行数
local_height = height // nbp
y_start = rank * local_height
y_end = (rank + 1) * local_height

# 每个进程创建自己的局部缓冲区
# 注意：为了 Gather 方便，我们将 y 放在第一维
local_convergence = np.empty((local_height, width), dtype=np.double)

scaleX = 3./width
scaleY = 2.25/height

# --- 并行计算开始 ---
comm.Barrier() # 同步所有进程
deb_all = MPI.Wtime()

for y in range(y_start, y_end):
    for x in range(width):
        c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
        # 写入局部缓冲区的相对位置
        local_convergence[y - y_start, x] = mandelbrot_set.convergence(c, smooth=True)

# --- 汇总结果 ---
if rank == 0:
    full_convergence = np.empty((height, width), dtype=np.double)
else:
    full_convergence = None

# 使用 Gather 将所有 local_convergence 拼接到 rank 0 的 full_convergence
comm.Gather(local_convergence, full_convergence, root=0)

fin_all = MPI.Wtime()

# --- 只有进程 0 负责输出和显示 ---
if rank == 0:
    print(f"Total number of processes: {nbp}")
    print(f"Total time spent on parallel computing (including communication): {fin_all - deb_all} seconds")
    
    # 图像处理（注意转置回原来的逻辑）
    image_data = full_convergence # 此时已经是 (height, width)
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(image_data)*255))
    image.show()
    input("Press Enter to close and exit...")