import pygame as pg
import numpy as np
from mpi4py import MPI
from scipy.signal import convolve2d
import time
import sys
import warnings

# 禁用警告
warnings.filterwarnings("ignore")

# --- MPI 全局初始化 ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

class Grille:
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dim  # 全局维度 (ny, nx)
        self.ny, self.nx = dim
        
        # 确保高度能被进程数整除
        if self.ny % nbp != 0:
            if rank == 0: print(f"Error: ny ({self.ny}) must be divisible by nbp ({nbp})")
            comm.Abort()
            
        self.local_ny = self.ny // nbp
        
        # 本地网格：真实数据 + 上下各1行 ghost cells
        # 形状为 (local_ny + 2, nx)
        self.cells = np.zeros((self.local_ny + 2, self.nx), dtype=np.uint8)
        
        # Rank 0 初始化全局数据并分发
        if rank == 0:
            full_grid = np.zeros(self.dimensions, dtype=np.uint8)
            if init_pattern is not None:
                indices_i = [v[0] for v in init_pattern]
                indices_j = [v[1] for v in init_pattern]
                full_grid[indices_i, indices_j] = 1
            else:
                full_grid = np.random.randint(2, size=dim, dtype=np.uint8)
            
            chunks = np.array_split(full_grid, nbp, axis=0)
        else:
            chunks = None

        # 分发数据到本地有效区域 [1:-1]
        self.cells[1:-1, :] = comm.scatter(chunks, root=0)

        self.col_life = color_life
        self.col_dead = color_dead
        
        # 定义上下邻居
        self.top_neighbor = (rank - 1 + nbp) % nbp
        self.bottom_neighbor = (rank + 1) % nbp

    @staticmethod        
    def h(x):
        # 规则映射
        res = np.full(x.shape, -1, dtype=np.int8)
        res[x == 3] = 1
        res[x == 2] = 0
        return res

    def update_ghost_cells(self):
        if nbp > 1:
            # 垂直方向：进程间交换边界行
            # 向上发送 [1] 接收到 [0]；向下发送 [-2] 接收到 [-1]
            comm.Sendrecv(sendbuf=self.cells[1, :], dest=self.top_neighbor,
                        recvbuf=self.cells[0, :], source=self.top_neighbor)
            comm.Sendrecv(sendbuf=self.cells[-2, :], dest=self.bottom_neighbor,
                        recvbuf=self.cells[-1, :], source=self.bottom_neighbor)
        else:
            # 单进程回绕
            self.cells[0, :] = self.cells[-2, :]
            self.cells[-1, :] = self.cells[1, :]

    def compute_next_iteration(self):
        # 1. 同步虚细胞
        self.update_ghost_cells()

        # 2. 处理水平方向（左右）的环面边界
        # 为了让 convolve2d 在本地切片上正确执行左右回绕，我们需要手动对左右各加一列
        # 创建一个临时扩展数组 (local_ny+2, nx+2)
        padded = np.zeros((self.local_ny + 2, self.nx + 2), dtype=np.uint8)
        padded[:, 1:-1] = self.cells
        padded[:, 0] = self.cells[:, -1] # 最后一列拷到最左
        padded[:, -1] = self.cells[:, 0] # 第一列拷到最右

        # 3. 卷积计算
        C = np.ones((3, 3), dtype=np.uint8)
        C[1, 1] = 0
        # 使用 mode='valid' 获取准确的 (local_ny, nx) 邻居计数结果
        voisins = convolve2d(padded, C, mode='valid')
        
        # 4. 应用规则 (仅针对本地有效行)
        local_inner = self.cells[1:-1, :]
        diff = Grille.h(voisins)
        next_cells = np.clip(local_inner.astype(np.int8) + diff, 0, 1).astype(np.uint8)
        
        # 更新本地状态
        self.cells[1:-1, :] = next_cells

class App:
    def __init__(self, geometry, grid):
        self.grid = grid
        if rank == 0:
            pg.init()
            self.screen_resx = geometry[0]
            self.screen_resy = geometry[1]
            self.screen = pg.display.set_mode((self.screen_resx, self.screen_resy))

            # 颜色配置
            self.draw_color = pg.Color('lightgrey')
            self.boundary_color = pg.Color('red')

            # 汇总缓冲区
            self.full_grid_buffer = np.empty(grid.dimensions, dtype=np.uint8)
            # 绘图表面
            self.grid_surface = pg.Surface((grid.dimensions[1], grid.dimensions[0]))
            self.clock = pg.time.Clock()
        else:
            self.screen = None

    def draw(self):
        # 汇总各进程计算的 [1:-1] 区域
        all_chunks = comm.gather(self.grid.cells[1:-1, :], root=0)

        if rank == 0:
            self.full_grid_buffer = np.vstack(all_chunks)
            
            # 高效渲染
            img_array = np.zeros((self.grid.nx, self.grid.ny, 3), dtype=np.uint8)
            # 翻转并转置以匹配 Pygame 坐标系 (x,y)
            data = np.flip(self.full_grid_buffer, axis=0).T
            
            img_array[data == 1] = self.grid.col_life[:3]
            img_array[data == 0] = self.grid.col_dead[:3]
            
            pg.surfarray.blit_array(self.grid_surface, img_array)
            scaled_win = pg.transform.scale(self.grid_surface, (self.screen_resx, self.screen_resy))
            self.screen.blit(scaled_win, (0, 0))

            # 3. 动态计算线条位置，确保完美对齐
            nb_rows = self.grid.dimensions[0]
            nb_cols = self.grid.dimensions[1]
            local_height = nb_rows // nbp

            # 绘制横线（包含进程边界）
            for i in range(nb_rows + 1):
                # 使用比例计算 y 坐标，确保对齐 800 像素
                y_pos = int(i * (self.screen_resy / nb_rows))
                
                if i % local_height == 0 and i != 0 and i != nb_rows:
                    # 只有进程中间的分界线用红色，且加粗
                    pg.draw.line(self.screen, self.boundary_color, (0, y_pos), (self.screen_resx, y_pos), 3)
                else:
                    pg.draw.line(self.screen, self.draw_color, (0, y_pos), (self.screen_resx, y_pos), 1)

            # 绘制纵线
            for j in range(nb_cols + 1):
                x_pos = int(j * (self.screen_resx / nb_cols))
                pg.draw.line(self.screen, self.draw_color, (x_pos, 0), (x_pos, self.screen_resy), 1)
            
            pg.display.flip()

if __name__ == '__main__':
    # 模式字典
    dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }
    
    choice = sys.argv[1] if len(sys.argv) > 1 else 'glider'
    init_pattern = dico_patterns.get(choice, dico_patterns['glider'])
    
    grid = Grille(*init_pattern)
    appli = App((800, 800), grid)

    mustContinue = True
    while mustContinue:
        # 1. 计算
        t1 = time.time()
        grid.compute_next_iteration()
        t2 = time.time()
        # 2. 汇总并显示
        appli.draw()
        t3 = time.time()
        
        # 同步退出状态
        if rank == 0:
            for event in pg.event.get():
                if event.type == pg.QUIT: mustContinue = False
            mustContinue = comm.bcast(mustContinue, root=0)
            print(f"Temps calcul prochaine generation : {t2-t1:2.2e} secondes, temps affichage : {t3-t2:2.2e} secondes\r", end='');
        else:
            mustContinue = comm.bcast(None, root=0)

    if rank == 0: pg.quit()
    MPI.Finalize()