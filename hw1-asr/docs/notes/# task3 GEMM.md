# task3 GEMM
分块乘法
### Baseline实现：
输入两个矩阵（x[M,K], W[N,K]），按tile（默认16）分块计算其乘积.
#### 分层结构
默认 TILE = 16：
- 一个 block 有 16 × 16 = 256 个线程。
- 一个 block 负责输出 y 的一个 16 × 16 小块。
- 一个线程负责其中一个输出元素，结果累加在自己的寄存器 acc 中。
- 沿着 K 维，每次处理 16 个特征。
#### 流程
协作加载 x 和 W 的小块到 shared memory
                  ↓
           __syncthreads()
                  ↓
每个线程读取 shared memory，累加 16 个乘积
                  ↓
           __syncthreads()
                  ↓
           处理下一段 K