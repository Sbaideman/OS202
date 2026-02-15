import pygame as pg
import numpy as np
from mpi4py import MPI
import time
import sys
import warnings

# 禁用烦人的警告
warnings.filterwarnings("ignore")

# --- MPI 全局初始化 ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

class Grille:
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dim 
        # 确保行数能被进程数整除 (TD要求)
        self.local_height = self.dimensions[0] // nbp
        self.width = self.dimensions[1]
        
        # 本地网格：真实数据 + 上下各1行 ghost cells
        self.cells = np.zeros((self.local_height + 2, self.width), dtype=np.uint8)
        
        if rank == 0:
            full_grid = np.zeros(self.dimensions, dtype=np.uint8)
            if init_pattern is not None:
                indices_i = [v[0] for v in init_pattern]
                indices_j = [v[1] for v in init_pattern]
                full_grid[indices_i, indices_j] = 1
            else:
                full_grid = np.random.randint(2, size=dim, dtype=np.uint8)
            
            # 切分数据
            chunks = [full_grid[i*self.local_height:(i+1)*self.local_height, :] for i in range(nbp)]
        else:
            chunks = None

        # 使用小写 scatter 确保分发安全
        self.cells[1:-1, :] = comm.scatter(chunks, root=0)

        self.col_life = color_life
        self.col_dead = color_dead
        self.prev_rank = (rank - 1 + nbp) % nbp
        self.next_rank = (rank + 1) % nbp

    def update_ghost_cells(self):
        if nbp > 1:
            # 向上交换
            comm.Sendrecv(sendbuf=self.cells[1, :], dest=self.prev_rank,
                          recvbuf=self.cells[0, :], source=self.prev_rank)
            # 向下交换
            comm.Sendrecv(sendbuf=self.cells[-2, :], dest=self.next_rank,
                          recvbuf=self.cells[-1, :], source=self.next_rank)
        else:
            # 单进程回绕
            self.cells[0, :] = self.cells[-2, :]
            self.cells[-1, :] = self.cells[1, :]

    def compute_next_iteration(self):
        self.update_ghost_cells()
        nx = self.dimensions[1]
        # 向量化邻居计数 (Stencil 3x3)
        nb_neighbors = (
            self.cells[0:-2, 0:nx] + self.cells[0:-2, (np.arange(nx)+1)%nx] + self.cells[0:-2, (np.arange(nx)-1)%nx] +
            self.cells[1:-1, (np.arange(nx)+1)%nx] + self.cells[1:-1, (np.arange(nx)-1)%nx] +
            self.cells[2:,   0:nx] + self.cells[2:,   (np.arange(nx)+1)%nx] + self.cells[2:,   (np.arange(nx)-1)%nx]
        )
        
        # 应用规则到 [1:-1] 区域
        next_inner = np.zeros((self.local_height, self.width), dtype=np.uint8)
        birth = (self.cells[1:-1, :] == 0) & (nb_neighbors == 3)
        survive = (self.cells[1:-1, :] == 1) & ((nb_neighbors == 2) | (nb_neighbors == 3))
        next_inner[birth | survive] = 1
        self.cells[1:-1, :] = next_inner

class App:
    def __init__(self, geometry, grid):
        self.grid = grid
        if rank == 0:
            pg.init()
            # 统一使用传入的窗口分辨率 (800, 800)
            self.screen_resx = geometry[0]
            self.screen_resy = geometry[1]
            self.screen = pg.display.set_mode((self.screen_resx, self.screen_resy))
            
            # 颜色配置
            self.draw_color = pg.Color('lightgrey')
            self.boundary_color = pg.Color('red')
            
            # 汇总缓冲区
            self.full_grid_buffer = np.empty(grid.dimensions, dtype=np.uint8)
            # 绘图表面（保持与网格尺寸一致）
            self.grid_surface = pg.Surface((grid.dimensions[1], grid.dimensions[0]))
        else:
            self.screen = None

    def draw(self):
        # 汇总数据
        all_chunks = comm.gather(self.grid.cells[1:-1, :], root=0)

        if rank == 0:
            # 1. 填充背景并转换细胞数据
            self.full_grid_buffer = np.vstack(all_chunks)
            img_array = np.zeros((self.grid.dimensions[1], self.grid.dimensions[0], 3), dtype=np.uint8)
            
            # 翻转数据以匹配原始坐标系（逻辑 (0,0) 在左下）
            data = np.flip(self.full_grid_buffer, axis=0).T
            img_array[data == 1] = self.grid.col_life[:3]
            img_array[data == 0] = self.grid.col_dead[:3]
            
            # 2. 将细胞刷到屏幕（先刷细胞，再画线，防止线被覆盖）
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
    choice = 'glider'
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
        
        # 3. 只有 Rank 0 负责事件监听，并广播给所有人
        if rank == 0:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False
            print(f"Temps calcul prochaine generation : {t2-t1:2.2e} secondes, temps affichage : {t3-t2:2.2e} secondes\r", end='');
            # 广播退出标志
            mustContinue = comm.bcast(mustContinue, root=0)
        else:
            mustContinue = comm.bcast(None, root=0)

    if rank == 0: pg.quit()
    MPI.Finalize()