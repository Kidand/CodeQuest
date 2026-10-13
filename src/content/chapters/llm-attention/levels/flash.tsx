/**
 * L3-5 FlashAttention：HBM vs SRAM、online softmax、tiling 与 IO 复杂度、反向重计算、FA2/FA3/FA4。
 * 事实来源：FlashAttention（arXiv 2205.14135）Algorithm 0/1、Theorem 1/2、Fig. 1/2；FA2（arXiv 2307.08691）；
 * FA3（arXiv 2407.08608）；FA4（arXiv 2603.05451）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { IOCompareViz } from '../viz/IOCompareViz'
import { OnlineSoftmaxViz } from '../viz/OnlineSoftmaxViz'
import { TilingViz } from '../viz/TilingViz'

export default defineLevel({
  learn: [
    {
      id: 'io',
      title: '瓶颈不在算力，在显存读写',
      body: md`
        GPU 的存储是分层的。以 FlashAttention 论文中的 A100 为例：**HBM** 40–80 GB、带宽 1.5–2.0 TB/s；每个 SM 有 192 KB 片上 **SRAM**（共 108 个 SM），带宽约 19 TB/s——
        快一个数量级，但小好几个数量级。

        标准实现（论文 Algorithm 0）把注意力拆成几个独立的 kernel：

        1. 读 $Q, K$，算 $S = QK^\top$，**写回** HBM；
        2. 读 $S$，算 $P = \operatorname{softmax}(S)$，**写回** HBM；
        3. 读 $P, V$，算 $O = PV$，写回。

        $S, P$ 都是 $N \times N$。softmax、mask、dropout 这些逐元素操作计算量很小，时间几乎全花在读写上（memory-bound）。
        论文 Fig. 2 的实测（GPT-2 medium，$N=1024$、$d=64$、16 头、batch 64，A100，前向 + 反向）：

        | | 标准注意力 | FlashAttention |
        |---|---|---|
        | GFLOPs | 66.6 | 75.2 |
        | HBM 读写 | 40.3 GB | 4.4 GB |
        | 运行时间 | 41.7 ms | 7.3 ms |

        FlashAttention 的 FLOPs 反而**更多**（反向要重算），但 HBM 读写少了约 9 倍，于是快了约 5.7 倍。这就是 **IO-aware** 的含义。
      `,
      viz: IOCompareViz,
      after: md`
        > [!NOTE] 三个关键词
        > **tiling**（分块，把输入切成能放进 SRAM 的块，一个 kernel 内完成全部计算）、**online softmax**（不看整行也能算 softmax）、**recomputation**（反向时重算 $S, P$ 而不是从 HBM 读）。
        > 结果：**精确**注意力（不是近似），额外显存从 $O(N^2)$ 降到 $O(N)$。
      `,
      checkpoint: {
        id: 'cp-io',
        type: 'judge',
        prompt: md`FlashAttention 通过减少注意力的 FLOPs 来加速，代价是结果有轻微的近似误差。`,
        answer: false,
        explanation: md`两点都错。FlashAttention 计算的是**精确**注意力（论文 Theorem 1），FLOPs 不降反升（反向重计算）；加速来自 HBM 读写的大幅减少和 kernel 融合。`,
      },
    },
    {
      id: 'online',
      title: 'online softmax：不看完整行也能算 softmax',
      body: md`
        分块的难点在 softmax：归一化需要**整行**的最大值和指数和。解决办法是维护三个统计量，边读边修正：

        - $m$：目前见过的最大值；
        - $\ell$：以 $m$ 为基准的指数和 $\sum_j e^{x_j - m}$；
        - $\text{acc}$：以 $m$ 为基准的未归一化输出 $\sum_j e^{x_j - m} v_j$。

        读入新块 $x^{(b)}$ 时：

        $$
        m' = \max\big(m, \max_j x^{(b)}_j\big),\quad \alpha = e^{m - m'},\quad
        \ell' = \alpha\,\ell + \sum_j e^{x^{(b)}_j - m'},\quad
        \text{acc}' = \alpha\,\text{acc} + \sum_j e^{x^{(b)}_j - m'} v^{(b)}_j
        $$

        **为什么乘 $\alpha$**：旧的 $\ell$ 里每一项是 $e^{x_j - m}$，换成新基准要变成 $e^{x_j - m'} = e^{x_j - m}\cdot e^{m - m'}$——整体乘同一个因子即可，不需要回看旧数据。
        全部读完后 $\text{out} = \text{acc}/\ell$，与一次性 softmax **逐位相同**（只是浮点运算顺序不同）。

        这个技巧来自 Milakov & Gimelshein（2018）的 online normalizer；FlashAttention 把它推广到「同时累加输出」。
      `,
      viz: OnlineSoftmaxViz,
      after: md`
        > [!WARN] 边界：第一块全是 $-\infty$
        > 若 $m = -\infty$ 且新块最大值也是 $-\infty$（例如因果/padding 掩码遮住了整块），$\alpha = e^{-\infty - (-\infty)} = e^{\text{NaN}}$。实现时要特判：
        > $m'=-\infty$ 时令 $\alpha = 1$（或跳过该块）。实战第 1 题专门考这一点。

        > [!INTERVIEW] 记住这句
        > 「online softmax 的本质是：指数和可以换基准，换基准只需乘一个标量 $e^{m_{\text{old}} - m_{\text{new}}}$。」
      `,
      checkpoint: {
        id: 'cp-online',
        type: 'fill',
        prompt: md`对一行分数分两块处理：第一块 $[1, 2]$，第二块 $[4, 0]$。处理完两块后 $m =$ ____(1)____，$\ell =$ ____(2)____（保留 4 位小数）。`,
        blanks: [
          { label: '(1)', answers: ['4'], numeric: true, tolerance: 0 },
          { label: '(2)', answers: ['1.2034'], numeric: true, tolerance: 0.001 },
        ],
        explanation: md`第一块后 $m=2$，$\ell = e^{-1} + 1 = 1.3679$。第二块 $m'=4$，$\alpha = e^{2-4} = 0.1353$，$\ell' = 1.3679 \times 0.1353 + e^{0} + e^{-4} = 0.1851 + 1 + 0.0183 = 1.2034$。等于直接算 $\sum_j e^{x_j - 4}$。`,
      },
    },
    {
      id: 'tiling',
      title: '分块（tiling）与 IO 复杂度',
      body: md`
        把 $Q$ 切成 $T_r$ 个 $B_r \times d$ 的块，$K, V$ 切成 $T_c$ 个 $B_c \times d$ 的块。对每个 tile 在 SRAM 里算 $S_{ij} = Q_i K_j^\top$（$B_r\times B_c$），
        用 online softmax 更新第 $i$ 块行的 $m_i, \ell_i, O_i$。$N \times N$ 的 $S, P$ **从不写回 HBM**。

        **FA1（Algorithm 1）**：块大小 $B_c = \lceil M/4d \rceil$、$B_r = \min(\lceil M/4d\rceil, d)$（$M$ 为 SRAM 大小）；**外层遍历 K/V 块，内层遍历 Q 块**，
        所以每个内层步都要从 HBM 读写 $O_i, \ell_i, m_i$。

        **IO 复杂度**（论文 Theorem 2，$d \le M \le Nd$）：标准注意力 $\Theta(Nd + N^2)$ 次 HBM 访问，FlashAttention $\Theta(N^2 d^2 M^{-1})$。
        直觉：K/V 每块大小 $\Theta(M)$，共 $\Theta(Nd/M)$ 块，每块都要把 Q 扫一遍（$\Theta(Nd)$）。$d$ 取 64–128、$M$ 约 100 KB 时 $d^2 \ll M$。
        论文还证明了下界（Proposition 3）：不存在对所有 $M \in [d, Nd]$ 都只需 $o(N^2d^2M^{-1})$ 次访问的精确注意力算法。

        **FA2** 把循环顺序反过来：**外层遍历 Q 块**，$O_i$ 和统计量常驻片上，内层扫完所有 K/V 块后只写一次；不同 Q 块之间无需通信，可以分给不同 thread block
        （在 batch、head 之外再沿序列维并行）。**因果掩码**下，整块位于上三角的 tile 直接跳过，只有对角 tile 需要块内掩码——论文报告比不带掩码快约 1.7–1.8 倍。
      `,
      viz: TilingViz,
      after: md`
        > [!TIP] 数 tile
        > $N$ 被切成 $T$ 块（$B_r = B_c$）时，因果注意力只需计算 $T(T+1)/2$ 个 tile。$T=4$ 时是 10 个，跳过 6 个——实战第 2 题会让你返回这个计数。
      `,
      checkpoint: {
        id: 'cp-tiling',
        type: 'single',
        prompt: md`FlashAttention-2 把循环改成「外层 Q 块、内层 K/V 块」，最主要的好处是：`,
        options: [
          '减少了 FLOPs',
          '每个 Q 块的输出只需在最后写一次 HBM，且不同 Q 块完全独立，可沿序列维并行到不同 thread block',
          '可以跳过因果掩码中的上三角 tile',
          '让 softmax 变成近似计算以节省时间',
        ],
        answer: 1,
        explanation: md`FLOPs 不变（A 错）；跳过上三角两种循环顺序都能做（C 不是主要区别）；FA 系列都是精确计算（D 错）。外层 Q 让 $O_i$ 常驻片上、写一次，且输出互不依赖，长序列、小 batch 时也能吃满 SM。`,
      },
    },
    {
      id: 'backward',
      title: '反向：存 logsumexp，重算 P',
      body: md`
        反向需要 $P$。标准实现从 HBM 读回前向保存的 $N\times N$ 矩阵；FlashAttention 只保存输出 $O$ 和每行的统计量（FA1 存 $m, \ell$，FA2 只存 $L = m + \log \ell$），
        反向时按块重算：

        $$
        S_{ij} = Q_i K_j^\top/\sqrt{d},\qquad P_{ij} = \exp(S_{ij} - L_i)
        $$

        然后按链式法则（$dO$ 为上游梯度）：

        $$
        dV_j \mathrel{+}= P_{ij}^\top dO_i,\quad dP_{ij} = dO_i V_j^\top,\quad
        D_i = \operatorname{rowsum}(dO_i \circ O_i),\quad dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)
        $$

        $$
        dQ_i \mathrel{+}= dS_{ij} K_j/\sqrt{d},\qquad dK_j \mathrel{+}= dS_{ij}^\top Q_i/\sqrt{d}
        $$

        其中 $D_i = \sum_k P_{ik}\,dP_{ik}$ 用 $\operatorname{rowsum}(dO_i\circ O_i)$ 计算（二者相等，因为 $O_i = \sum_k P_{ik}V_k$），避免再扫一遍整行。

        **重计算多花了 FLOPs，为什么反而更快？** 因为瓶颈是 HBM 带宽：从 HBM 读 $N^2$ 的 $P$ 比在 SRAM 里重算慢得多。这和 gradient checkpointing 的思路相同，只是这里「省显存」与「提速」同时达成。
      `,
      checkpoint: {
        id: 'cp-bwd',
        type: 'multi',
        prompt: md`FlashAttention-2 前向结束后，为反向保存到 HBM 的量有哪些？（多选；$Q, K, V$ 本来就在）`,
        options: ['输出 $O$（$N\\times d$）', '每行的 logsumexp $L$（$N$ 个数）', '注意力概率 $P$（$N\\times N$）', '分数矩阵 $S$（$N\\times N$）', 'dropout 的掩码矩阵（$N\\times N$）'],
        answer: [0, 1],
        explanation: md`$P$ 和 $S$ 在反向中按块重算。dropout 掩码也不保存：保存随机数生成器的状态（种子与偏移），反向时重新生成——否则又回到 $O(N^2)$ 显存。`,
      },
    },
    {
      id: 'versions',
      title: 'FA2、FA3、FA4 各改了什么',
      body: md`
        | 版本 | 硬件 | 核心改进 | 论文报告的效果 |
        |---|---|---|---|
        | FA1（2022） | A100 | tiling + online softmax + 重计算，单 kernel 融合 | GPT-2（1K）训练 3× 于 HF/Megatron；注意力本身最高 7.6× |
        | FA2（2023） | A100 | ① 减少非矩阵乘 FLOPs（输出最后才除 $\ell$，只存 logsumexp）② 沿序列维并行 ③ warp 间按 Q 切分而非按 K/V（去掉 split-K 的共享内存通信） | 约 2× 于 FA1，达理论峰值 50–73%；GPT 训练最高 225 TFLOPs/s/A100（72% MFU） |
        | FA3（2024） | H100 | ① warp specialization，利用 Tensor Core 与 TMA 的异步性重叠计算与搬运 ② 块级 GEMM 与 softmax 交错流水 ③ FP8：块量化 + incoherent processing | 1.5–2.0× 于 FA2；FP16 最高 740 TFLOPs/s（75% 利用率），FP8 接近 1.2 PFLOPs/s，FP8 误差比基线 FP8 注意力低 2.6× |
        | FA4（2026） | B200 | ① 全异步 MMA 流水与更大 tile ② 软件模拟 exp + 条件式 softmax rescale，减少非矩阵乘操作 ③ 利用 tensor memory 与 2-CTA MMA 减少反向的共享内存流量与 atomic add；完全用 Python 中的 CuTe-DSL 编写 | BF16 最高 1613 TFLOPs/s（71% 利用率），最高 1.3× 于 cuDNN 9.13、2.7× 于 Triton |

        **FA2 为什么在意非矩阵乘 FLOPs**：A100 上 FP16/BF16 矩阵乘峰值 312 TFLOPs/s，而非矩阵乘的 FP32 只有 19.5 TFLOPs/s——每个非矩阵乘 FLOP 贵 16 倍。
        **FA4 的动机**相同而更极端：Blackwell 上 Tensor Core 吞吐翻倍，但指数单元与共享内存带宽没有同比增长（论文称之为「非对称的硬件扩展」）。
      `,
      after: md`
        > [!NEW] decode 阶段：Flash-Decoding / split-KV
        > decode 时每个序列只有 1 个 query，FA2 的「沿 Q 块并行」没有可并行的东西，batch 小时 SM 大量空闲。
        > Flash-Decoding（2023）把**长 KV 切成多段**并行计算，每段得到部分输出 $o_s$ 与 $\text{lse}_s$，最后用 logsumexp 合并：
        > $\text{lse} = \log\sum_s e^{\text{lse}_s}$，$o = \sum_s e^{\text{lse}_s - \text{lse}}\, o_s$——与 online softmax 是同一个数学。BOSS 关的实战会实现它。
      `,
      checkpoint: {
        id: 'cp-versions',
        type: 'single',
        prompt: md`下列哪一项是 **FlashAttention-3** 的改进？`,
        options: ['首次提出 online softmax 与 tiling', '沿序列长度维并行并改用外层 Q 循环', '利用 Hopper 的 TMA 与 Tensor Core 异步性做 warp specialization，并支持 FP8', '用软件模拟指数函数以缓解 Blackwell 上 exp 单元瓶颈'],
        answer: 2,
        explanation: md`A 是 FA1，B 是 FA2，D 是 FA4（2026，Blackwell）。FA3 面向 H100：warp specialization 重叠数据搬运与计算、GEMM 与 softmax 交错、FP8 块量化 + incoherent processing。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-exact',
      type: 'single',
      difficulty: 1,
      prompt: md`FlashAttention 与标准注意力相比：`,
      options: ['结果是近似的，但速度更快', '结果精确相等（至多浮点舍入差异），FLOPs 更少', '结果精确相等（至多浮点舍入差异），HBM 读写更少', '只能用于推理，不能用于训练'],
      answer: 2,
      explanation: md`FlashAttention 是精确算法（Theorem 1），只改变计算顺序；它减少的是 HBM 读写，FLOPs 因反向重计算反而略多（Fig. 2：75.2 vs 66.6 GFLOPs）。前向、反向都支持，训练是它最初的主要场景。`,
    },
    {
      id: 'q-flops',
      type: 'judge',
      difficulty: 1,
      prompt: md`FlashAttention 的额外显存（除输入输出外）是 $O(N)$。`,
      answer: true,
      explanation: md`Theorem 1：$O(N^2 d)$ FLOPs、$O(N)$ 额外显存——只需每行的 $m, \ell$（或 logsumexp $L$），$N\times N$ 的 $S, P$ 从不保存。`,
    },
    {
      id: 'q-mem',
      type: 'numeric',
      difficulty: 1,
      prompt: md`$N = 32768$，FP16，**单个头**的 $N \times N$ 注意力矩阵占多少 GiB？`,
      answer: 2,
      tolerance: 0.01,
      unit: 'GiB',
      explanation: md`$32768^2 \times 2 = 2^{31}$ 字节 $= 2$ GiB。乘以头数与层数（训练时还要为反向保存），标准实现在长序列下根本放不下。`,
    },
    {
      id: 'q-16x',
      type: 'numeric',
      difficulty: 2,
      prompt: md`FA2 论文：A100 的 FP16/BF16 矩阵乘峰值 312 TFLOPs/s，非矩阵乘 FP32 峰值 19.5 TFLOPs/s。一个非矩阵乘 FLOP 相当于多少个矩阵乘 FLOP 的时间？`,
      answer: 16,
      tolerance: 0.01,
      unit: '倍',
      explanation: md`$312 / 19.5 = 16$。所以 FA2 把输出的 $\operatorname{diag}(\ell)^{-1}$ 缩放推迟到最后一次性做、只存 logsumexp，尽量减少 exp、除法、max 等非矩阵乘操作。`,
    },
    {
      id: 'q-fa2',
      type: 'multi',
      difficulty: 2,
      prompt: md`以下哪些是 **FlashAttention-2** 相对 FA1 的改进？（多选）`,
      options: [
        '减少非矩阵乘 FLOPs：不在每步缩放输出，最后再除以 $\\ell$',
        '在 batch、head 之外，沿序列长度维并行',
        'thread block 内按 Q 在 warp 间切分，避免 split-K 的共享内存通信',
        '利用 TMA 与 warp specialization 做异步流水',
        '用 FP8 计算注意力并做 incoherent processing',
      ],
      answer: [0, 1, 2],
      explanation: md`D、E 是 FA3（Hopper）的改进。FA2 的三条正是论文摘要列出的：减少非矩阵乘 FLOPs、单个头也能跨 thread block 并行、warp 间更好的工作划分。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把版本与它的标志性改进配对：`,
      pairs: [
        ['FlashAttention（2022）', 'tiling + online softmax + 反向重计算'],
        ['FlashAttention-2（2023）', '外层 Q 循环，沿序列维并行'],
        ['FlashAttention-3（2024）', 'Hopper 异步：warp specialization + FP8'],
        ['FlashAttention-4（2026）', 'Blackwell：软件模拟 exp、条件 rescale'],
      ],
      explanation: md`每一代都在追同一个目标：让 Tensor Core 尽量忙，把非矩阵乘与数据搬运藏到矩阵乘后面。随着硬件变化（A100 → H100 → B200），瓶颈依次是 HBM 读写、warp 间通信、异步流水、exp 单元。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 FA2 前向中处理一个 tile $(i, j)$ 的步骤排成正确顺序（$O_i, m_i, \ell_i$ 常驻片上）：`,
      items: [
        '从 HBM 读入 $K_j, V_j$ 到 SRAM',
        '$S_{ij} = Q_i K_j^\\top / \\sqrt{d}$（对角 tile 加因果掩码）',
        '$m_i^{\\text{new}} = \\max(m_i, \\operatorname{rowmax}(S_{ij}))$',
        '$\\tilde P = \\exp(S_{ij} - m_i^{\\text{new}})$，$\\ell_i \\leftarrow e^{m_i - m_i^{\\text{new}}}\\ell_i + \\operatorname{rowsum}(\\tilde P)$',
        '$O_i \\leftarrow e^{m_i - m_i^{\\text{new}}} O_i + \\tilde P V_j$，$m_i \\leftarrow m_i^{\\text{new}}$',
      ],
      explanation: md`先有新最大值，才能算以它为基准的 $\tilde P$ 与缩放因子；$m_i$ 必须在用完旧值（算 $\alpha$）之后才更新。所有 $j$ 处理完后再做一次 $O_i / \ell_i$ 并写回 HBM。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的分块注意力（单头、非因果）结果与朴素实现对不上。点出 **2** 处 bug：`,
      code: py`
        def flash_attn(q, k, v, B=4):
            N, d = q.shape
            O = np.zeros_like(q)
            m = np.full(N, -np.inf)
            l = np.zeros(N)
            for j in range(0, k.shape[0], B):
                S = q @ k[j:j + B].T / np.sqrt(d)
                m_new = np.maximum(m, S.max(axis=1))
                P = np.exp(S - m_new[:, None])
                alpha = np.exp(m_new - m)
                l = alpha * l + P.sum(axis=1)
                O = O + P @ v[j:j + B]
                m = m_new
            return O / l[:, None]
      `,
      bugLines: [10, 12],
      explanation: md`第 10 行方向反了：应为 \`np.exp(m - m_new)\`（≤ 1），旧统计量要缩小到新基准；写反会放大旧值，第一块时还会得到 $e^{+\infty}$。
        第 12 行漏了对 \`O\` 的缩放：应为 \`O = alpha[:, None] * O + P @ v[j:j + B]\`——$\ell$ 和 $O$ 是以同一个 $m$ 为基准的，必须同步缩放。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`因果注意力，$N=8$、$B_r=B_c=2$，tile 网格为 4×4（行 = Q 块，列 = K/V 块）。点出**可以整块跳过**的 tile：`,
      items: Array.from({ length: 16 }, (_, n) => `S${Math.floor(n / 4)}${n % 4}`),
      cols: 4,
      rowLabels: ['Q0', 'Q1', 'Q2', 'Q3'],
      colLabels: ['KV0', 'KV1', 'KV2', 'KV3'],
      answer: [1, 2, 3, 6, 7, 11],
      explanation: md`tile $(i, j)$ 中最小列号 $2j$ 大于最大行号 $2i+1$ 时整块被遮，即 $j > i$——上三角 6 个。对角 tile（$i = j$）只被遮一部分，需要块内掩码；下三角无需掩码。共计算 $4\times5/2 = 10$ 个 tile。`,
    },
    {
      id: 'q-io',
      type: 'single',
      difficulty: 3,
      prompt: md`设序列长度 $N$、头维 $d$、SRAM 大小 $M$（$d \le M \le Nd$），FlashAttention 前向的 HBM 访问次数是：`,
      options: ['$\\Theta(Nd + N^2)$', '$\\Theta(N^2 d^2 M^{-1})$', '$\\Theta(N d^2 M^{-1})$', '$\\Theta(N^2 d)$'],
      answer: 1,
      explanation: md`论文 Theorem 2。K/V 分成 $\Theta(Nd/M)$ 块，每块都要把 Q（及 O）从 HBM 扫一遍（$\Theta(Nd)$），合计 $\Theta(N^2d^2/M)$。A 是标准注意力；D 是 FLOPs 的量级；C 漏了一个 $N$——FlashAttention 并没有把 HBM 访问降到线性。`,
    },
    {
      id: 'q-store',
      type: 'single',
      difficulty: 2,
      prompt: md`FA2 反向为什么只需要前向保存的 logsumexp $L_i = m_i + \log \ell_i$，而不需要分别保存 $m_i$ 与 $\ell_i$？`,
      options: [
        '因为反向不需要 softmax 概率',
        '因为 $P_{ij} = \\exp(S_{ij} - L_i)$，一个数就能把重算的分数直接变成归一化概率',
        '因为 $m_i$ 在反向中总是 0',
        '因为 logsumexp 可以从 $O$ 反推出来',
      ],
      answer: 1,
      explanation: md`$e^{S_{ij} - m_i}/\ell_i = e^{S_{ij} - (m_i + \log\ell_i)}$。存一个 $L_i$ 比存两个少一半统计量，重算时也少一次除法。$O$ 不能反推出 $L$（D 错）。`,
    },
    {
      id: 'q-decode',
      type: 'judge',
      difficulty: 2,
      prompt: md`decode 阶段（每个序列 1 个 query、KV 很长、batch 很小）直接使用 FA2 的前向 kernel，通常就能充分利用 GPU 的所有 SM。`,
      answer: false,
      explanation: md`FA2 的并行维度是 batch × head × Q 块；decode 时 Q 块只有 1 个，batch 小时 thread block 数远少于 SM 数。Flash-Decoding / split-KV 把 KV 维也切开并行，最后用 logsumexp 合并各段结果。`,
    },
  ],

  guided: [
    {
      id: 'g-flash',
      title: '一步步写出分块注意力（沿 K/V 分块的 online softmax）',
      intro: md`
        单头、非因果：\`q\`: \`(N, d)\`，\`k, v\`: \`(S, d)\`。沿 K/V 每次取 \`block\` 行，对**所有 query 同时**维护 \`m, l, O\`。
        这正是 FA2 内层循环在一个 Q 块上做的事。返回与 \`softmax(q k^T / sqrt(d)) v\` 相同的结果。
      `,
      template: py`
        import numpy as np

        def flash_attention(q, k, v, block=4):
            N, d = q.shape
            O = np.zeros((N, v.shape[1]))
            m = np.full(N, -np.inf)
            l = np.zeros(N)
            for j in range(0, k.shape[0], block):
                S = q @ k[j:j + block].T / np.sqrt(d)
                m_new = {{1}}
                P = {{2}}
                alpha = {{3}}
                l = {{4}}
                O = {{5}}
                m = m_new
            return O / l[:, None]
      `,
      steps: [
        {
          blank: 1,
          instruction: md`每行新的 running max：旧最大值与本块行最大值取大。`,
          answer: 'np.maximum(m, S.max(axis=1))',
          accept: ['np.maximum(m, S.max(1))', 'np.maximum(m, np.max(S, axis=1))', 'np.maximum(m, S.max(axis=-1))'],
          explain: md`\`np.maximum\` 是逐元素取大；不要写成 \`max(m, ...)\`（Python 内置 max 对数组会报错）。`,
        },
        {
          blank: 2,
          instruction: md`以新最大值为基准计算本块的指数。`,
          answer: 'np.exp(S - m_new[:, None])',
          accept: ['np.exp(S - m_new.reshape(-1, 1))', 'np.exp(S - m_new[:, np.newaxis])'],
          explain: md`\`m_new[:, None]\` 把 \`(N,)\` 变成 \`(N, 1)\` 以便按行广播。`,
        },
        {
          blank: 3,
          instruction: md`旧统计量换基准的缩放因子。`,
          answer: 'np.exp(m - m_new)',
          explain: md`$m \le m_{\text{new}}$，所以 $\alpha \in (0, 1]$。第一块时 $m=-\infty$、$\alpha = 0$，恰好把初始的 0 清掉（本题无全遮块，不会出现 $-\infty - (-\infty)$）。`,
        },
        {
          blank: 4,
          instruction: md`更新指数和。`,
          answer: 'alpha * l + P.sum(axis=1)',
          accept: ['alpha * l + P.sum(1)', 'l * alpha + P.sum(axis=1)', 'alpha * l + np.sum(P, axis=1)'],
          explain: md`旧和缩放到新基准，再加上本块。`,
        },
        {
          blank: 5,
          instruction: md`更新未归一化的输出累加器。`,
          answer: 'alpha[:, None] * O + P @ v[j:j + block]',
          accept: ['O * alpha[:, None] + P @ v[j:j + block]', 'alpha.reshape(-1, 1) * O + P @ v[j:j + block]'],
          explain: md`$O$ 与 $\ell$ 必须用同一个 $\alpha$ 同步缩放，最后 \`O / l\` 才是正确的 softmax 加权和。FA2 正是把除以 $\ell$ 推迟到循环结束，省掉每步的除法。`,
        },
      ],
      setup: py`
        import numpy as np

        def _naive(q, k, v):
            s = q @ k.T / np.sqrt(q.shape[1])
            p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            return p @ v
      `,
      tests: [
        {
          name: '与朴素实现一致',
          code: py`
            r = np.random.default_rng(0)
            q, k, v = r.normal(size=(5, 8)), r.normal(size=(12, 8)), r.normal(size=(12, 3))
            got = flash_attention(q, k, v, block=4)
            assert got.shape == (5, 3), f"形状应为 (5, 3)，得到 {got.shape}"
            assert np.allclose(got, _naive(q, k, v), atol=1e-10), f"最大误差 {np.abs(got - _naive(q, k, v)).max():.2e}"
          `,
        },
        {
          name: '块大小不整除 + 大数值',
          code: py`
            r = np.random.default_rng(1)
            q, k, v = r.normal(size=(6, 4)) * 20, r.normal(size=(11, 4)) * 20, r.normal(size=(11, 2))
            got = flash_attention(q, k, v, block=3)
            assert np.all(np.isfinite(got)), "出现 inf/NaN"
            assert np.allclose(got, _naive(q, k, v), atol=1e-8)
          `,
        },
        {
          name: '块大小 1（纯 online）',
          hidden: true,
          code: py`
            r = np.random.default_rng(2)
            q, k, v = r.normal(size=(3, 4)), r.normal(size=(7, 4)), r.normal(size=(7, 5))
            assert np.allclose(flash_attention(q, k, v, block=1), _naive(q, k, v), atol=1e-10)
          `,
        },
      ],
      outro: md`
        再在外面套一层「Q 分块」循环、加上对角 tile 的掩码与上三角 tile 的跳过，就是完整的 FA2 前向——实战第 2 题。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-online',
      title: '单遍 online logsumexp（处理全遮块与空块）',
      difficulty: 'medium',
      source: '大模型岗手撕 · online softmax',
      timeLimitMin: 15,
      description: md`
        同一行分数被切成若干连续片段，依次到来、**只能遍历一次**（可能是生成器）。实现

        \`\`\`python
        def online_logsumexp(blocks) -> tuple[float, float]:
            # 返回 (m, l)：m = 所有元素的最大值，l = Σ exp(x - m)
        \`\`\`

        - 每个块是一维 \`np.ndarray\`，可能为空，可能包含 \`-np.inf\`（被掩码的位置），甚至整块都是 \`-np.inf\`。
        - 全部元素都是 \`-inf\`（或根本没有元素）时，返回 \`(-inf, 0.0)\`。
        - 不得出现 NaN 或溢出（元素可能大到 $10^4$）。
        - 用一个 \`(m, l)\` 状态逐块更新，**不要**先把所有块拼起来（会检查源码中是否出现 \`concatenate\` / \`list(\`）。

        **示例**：\`[np.array([1., 2.]), np.array([4., 0.])]\` → \`(4.0, 1.2034...)\`
      `,
      starter: py`
        import numpy as np

        def online_logsumexp(blocks):
            m, l = -np.inf, 0.0
            for x in blocks:
                # TODO: 更新 (m, l)，注意空块与全 -inf 块
                pass
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def online_logsumexp(blocks):
            m, l = -np.inf, 0.0
            for x in blocks:
                if x.size == 0:
                    continue                               # 空块：max 会报错，直接跳过
                m_new = max(m, float(x.max()))
                if m_new == -np.inf:
                    continue                               # 迄今为止全被遮：-inf - (-inf) = NaN，跳过
                alpha = np.exp(m - m_new)                  # m = -inf 时为 0，正好清掉初始的 l
                l = alpha * l + float(np.exp(x - m_new).sum())
                m = m_new
            return m, l
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，按行批量处理的 online softmax 统计量）',
          code: py`
            import torch

            def online_lse(blocks):
                """blocks: 依次产生 (R, b) 的张量（R 行，各取一段列）。返回每行 (m, l)。"""
                m, l = None, None
                for x in blocks:
                    if x.numel() == 0:
                        continue
                    bm = x.max(dim=-1).values
                    m_new = bm if m is None else torch.maximum(m, bm)
                    safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)  # 全遮行的安全基准
                    p = torch.exp(x - safe[:, None])
                    alpha = torch.exp((m if m is not None else torch.full_like(m_new, float("-inf"))) - safe)
                    l = p.sum(-1) if l is None else alpha * l + p.sum(-1)
                    m = m_new
                return m, l
          `,
        },
      ],
      explanation: md`
        **更新公式**：$m' = \max(m, \max x)$，$\ell' = e^{m - m'}\ell + \sum e^{x - m'}$。

        **两个边界**
        1. 空块：\`x.max()\` 对空数组会抛异常，直接跳过；
        2. 迄今全为 $-\infty$：$m' = -\infty$ 时 \`np.exp(m - m_new)\` = \`exp(nan)\`，且 \`np.exp(x - m_new)\` 也是 NaN。FlashAttention 的实现里对应「整块被因果/padding 掩码遮住」的情况，通常用「若 $m' = -\infty$ 则把基准当作 0」来规避（见 PyTorch 版）。

        **复杂度**：$O(n)$ 时间、$O(1)$ 额外空间、单遍扫描。logsumexp 即 $m + \log \ell$。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`先写出无边界的版本：\`m_new = max(m, x.max())\`，\`l = exp(m - m_new) * l + exp(x - m_new).sum()\`。`,
        md`空块用 \`x.size == 0\` 判断后 \`continue\`。`,
        md`\`m_new == -inf\` 时也 \`continue\`：此时没有任何有效元素，状态不变。`,
      ],
      setup: py`
        import numpy as np, inspect

        def _ref_lse(blocks):
            xs = [b for b in blocks if b.size]
            if not xs:
                return -np.inf, 0.0
            x = np.concatenate(xs)
            m = x.max()
            if m == -np.inf:
                return -np.inf, 0.0
            return float(m), float(np.exp(x - m).sum())

        def _chk(blocks, what):
            exp_m, exp_l = _ref_lse(blocks)
            got = online_logsumexp(iter(blocks))       # 只能遍历一次
            assert isinstance(got, tuple) and len(got) == 2, f"{what}：应返回 (m, l)，得到 {got}"
            m, l = got
            assert not (np.isnan(m) or np.isnan(l)), f"{what}：出现 NaN：{got}"
            assert m == exp_m, f"{what}：m 期望 {exp_m}，得到 {m}"
            assert abs(l - exp_l) <= 1e-9 * max(1.0, abs(exp_l)), f"{what}：l 期望 {exp_l}，得到 {l}"
      `,
      tests: [
        { name: '示例', code: py`_chk([np.array([1., 2.]), np.array([4., 0.])], "示例")` },
        {
          name: '随机分块 = 一次性计算',
          code: py`
            r = np.random.default_rng(0)
            x = r.normal(size=37) * 3
            cuts = sorted(r.choice(np.arange(1, 37), size=6, replace=False).tolist())
            _chk(np.split(x, cuts), "随机分块")
          `,
        },
        { name: '大数值不溢出', code: py`_chk([np.array([1e4, 9999.]), np.array([10001., -1e4])], "大数值")` },
        { name: '第一块整块被遮', hidden: true, code: py`_chk([np.array([-np.inf, -np.inf]), np.array([0.5, -np.inf, 2.0])], "首块全 -inf")` },
        { name: '空块与全遮', hidden: true, code: py`_chk([np.array([]), np.array([-np.inf]), np.array([])], "全被遮")` },
        {
          name: '单遍、不拼接',
          hidden: true,
          code: py`
            src = inspect.getsource(online_logsumexp)
            assert "concatenate" not in src and "list(" not in src and "hstack" not in src, "请逐块更新状态，不要先把所有块拼起来"
            _chk([np.array([3.0]), np.array([]), np.array([-1.0, 7.0])], "含空块")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-merge',
          q: md`已知两段的 $(m_1, \ell_1, o_1)$ 与 $(m_2, \ell_2, o_2)$（$o$ 为各段归一化后的输出），怎么合并成整体输出？`,
          a: md`令 $m = \max(m_1, m_2)$，$\ell = e^{m_1-m}\ell_1 + e^{m_2-m}\ell_2$，$o = (e^{m_1-m}\ell_1 o_1 + e^{m_2-m}\ell_2 o_2)/\ell$。用 logsumexp 写更简洁：$\text{lse}_s = m_s + \log\ell_s$，$o = \sum_s e^{\text{lse}_s - \text{lse}} o_s$。这就是 Flash-Decoding、Ring Attention、前缀缓存（cascade attention）合并部分结果的方法。`,
        },
      ],
    },
    {
      id: 'c-flash',
      title: '手撕 FlashAttention-2 前向（Q/KV 双层分块 + 因果跳块）',
      difficulty: 'hard',
      source: '大模型岗手撕 · FlashAttention',
      timeLimitMin: 35,
      description: md`
        实现单头的 FA2 风格前向：

        \`\`\`python
        def flash_attention_forward(q, k, v, causal=False, Br=4, Bc=4) -> tuple[np.ndarray, np.ndarray, int]:
            # 返回 (O (Nq, dv), L (Nq,), n_tiles)
        \`\`\`

        - \`q\`: \`(Nq, d)\`，\`k\`: \`(Nk, d)\`，\`v\`: \`(Nk, dv)\`；缩放 $1/\sqrt{d}$。\`causal=True\` 时保证 \`Nq == Nk\`，query $i$ 只看 $j \le i$。
        - **外层**按 \`Br\` 行遍历 Q 块，**内层**按 \`Bc\` 行遍历 K/V 块；每个 tile 只计算 \`(Br, Bc)\` 的分数块，**不得**构造 \`(Nq, Nk)\` 的完整分数矩阵。
        - \`L[i]\` 为第 $i$ 行（掩码后、缩放后）分数的 logsumexp，供反向使用。
        - 因果时，整块满足「最小列号 > 最大行号」的 tile **必须跳过**；\`n_tiles\` 返回实际计算的 tile 数。
        - $N$ 不一定能被块大小整除。
      `,
      starter: py`
        import numpy as np

        def flash_attention_forward(q, k, v, causal=False, Br=4, Bc=4):
            Nq, d = q.shape
            Nk = k.shape[0]
            O = np.zeros((Nq, v.shape[1]))
            L = np.zeros(Nq)
            n_tiles = 0
            # TODO: for i0 in range(0, Nq, Br): 初始化该块的 m、l、acc
            #           for j0 in range(0, Nk, Bc): 因果跳块 → S 块 → 块内掩码 → online 更新
            #       最后 O[i0:i1] = acc / l，L[i0:i1] = m + log(l)
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def flash_attention_forward(q, k, v, causal=False, Br=4, Bc=4):
            Nq, d = q.shape
            Nk = k.shape[0]
            scale = 1.0 / np.sqrt(d)
            O = np.zeros((Nq, v.shape[1]))
            L = np.zeros(Nq)
            n_tiles = 0
            for i0 in range(0, Nq, Br):                       # 外层：Q 块（可并行到不同 thread block）
                i1 = min(i0 + Br, Nq)
                Qi = q[i0:i1]
                m = np.full(i1 - i0, -np.inf)
                l = np.zeros(i1 - i0)
                acc = np.zeros((i1 - i0, v.shape[1]))
                for j0 in range(0, Nk, Bc):                   # 内层：K/V 块
                    j1 = min(j0 + Bc, Nk)
                    if causal and j0 > i1 - 1:                # 整块在上三角：跳过（之后的块也都是）
                        break
                    n_tiles += 1
                    S = Qi @ k[j0:j1].T * scale               # (Br, Bc) 只在「片上」
                    if causal and j1 - 1 > i0:                # 对角 tile：块内因果掩码
                        rows = np.arange(i0, i1)[:, None]
                        cols = np.arange(j0, j1)[None, :]
                        S = np.where(cols <= rows, S, -np.inf)
                    m_new = np.maximum(m, S.max(axis=1))
                    P = np.exp(S - m_new[:, None])
                    alpha = np.exp(m - m_new)
                    l = alpha * l + P.sum(axis=1)
                    acc = alpha[:, None] * acc + P @ v[j0:j1]
                    m = m_new
                O[i0:i1] = acc / l[:, None]                   # 最后才归一化（FA2）
                L[i0:i1] = m + np.log(l)                      # 反向只需 logsumexp
            return O, L, n_tiles
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，算法结构与 CUDA kernel 相同，用于教学/对拍）',
          code: py`
            import math
            import torch

            def flash_attention_forward(q, k, v, causal=False, Br=64, Bc=64):
                Nq, d = q.shape
                Nk = k.size(0)
                O = torch.empty(Nq, v.size(1), dtype=q.dtype, device=q.device)
                L = torch.empty(Nq, dtype=torch.float32, device=q.device)
                for i0 in range(0, Nq, Br):                    # CUDA 中：每个 i0 是一个 thread block
                    i1 = min(i0 + Br, Nq)
                    Qi = q[i0:i1].float()
                    m = torch.full((i1 - i0,), float("-inf"), device=q.device)
                    l = torch.zeros(i1 - i0, device=q.device)
                    acc = torch.zeros(i1 - i0, v.size(1), device=q.device)
                    for j0 in range(0, Nk, Bc):
                        if causal and j0 > i1 - 1:
                            break
                        j1 = min(j0 + Bc, Nk)
                        S = Qi @ k[j0:j1].float().T / math.sqrt(d)
                        if causal and j1 - 1 > i0:
                            r = torch.arange(i0, i1, device=q.device)[:, None]
                            c = torch.arange(j0, j1, device=q.device)[None]
                            S = S.masked_fill(c > r, float("-inf"))
                        m_new = torch.maximum(m, S.amax(1))
                        P = torch.exp(S - m_new[:, None])
                        alpha = torch.exp(m - m_new)
                        l = alpha * l + P.sum(1)
                        acc = alpha[:, None] * acc + P @ v[j0:j1].float()
                        m = m_new
                    O[i0:i1] = (acc / l[:, None]).to(q.dtype)
                    L[i0:i1] = m + torch.log(l)
                return O, L
            # 生产中：from flash_attn import flash_attn_func 或 F.scaled_dot_product_attention（自动选择 flash 后端）
          `,
        },
      ],
      explanation: md`
        **结构**：外层 Q 块，块内维护 \`(m, l, acc)\`；内层每个 K/V 块做一次 online softmax 更新；外层结束时 \`O = acc / l\`、\`L = m + log l\`。

        **因果处理**：
        - tile 的最小列号 \`j0\` 大于最大行号 \`i1-1\` → 整块被遮，且后面的块列号更大，可以直接 \`break\`；
        - tile 的最大列号 \`j1-1\` 大于最小行号 \`i0\` → 跨越对角线，需要块内掩码；
        - 其余（完全在下三角）无需掩码。
        - 内层按 $j$ 从小到大，第一个 tile 一定包含第 0 列，每行至少有一个有效元素，不会出现全 $-\infty$ 的 NaN 问题。

        **tile 数**：$B_r = B_c = B$ 且 $N = TB$ 时因果为 $T(T+1)/2$，非因果为 $T^2$。

        **复杂度**：FLOPs 同标准注意力 $O(N^2 d)$，额外空间 $O(B_r B_c + N)$——这正是 FlashAttention 的意义。
      `,
      complexity: { time: 'O(Nq·Nk·d)', space: 'O(Br·Bc + Nq)' },
      hints: [
        md`先把引导练习的「沿 K/V 分块」写成内层循环，外面再包一层 Q 块循环，每个 Q 块单独维护 \`m, l, acc\`。`,
        md`跳块条件：\`causal and j0 > i1 - 1\`；块内掩码条件：\`causal and j1 - 1 > i0\`，掩码为 \`cols <= rows\`（用全局下标）。`,
        md`\`L[i0:i1] = m + np.log(l)\`；\`n_tiles\` 只在真正计算时加 1。`,
      ],
      setup: py`
        import numpy as np, inspect

        def _naive_attn(q, k, v, causal):
            s = q @ k.T / np.sqrt(q.shape[1])
            if causal:
                s = np.where(np.tril(np.ones(s.shape, bool)), s, -np.inf)
            m = s.max(1, keepdims=True)
            lse = (m + np.log(np.exp(s - m).sum(1, keepdims=True)))[:, 0]
            p = np.exp(s - m); p /= p.sum(1, keepdims=True)
            return p @ v, lse

        def _expected_tiles(Nq, Nk, Br, Bc, causal):
            n = 0
            for i0 in range(0, Nq, Br):
                i1 = min(i0 + Br, Nq)
                for j0 in range(0, Nk, Bc):
                    if causal and j0 > i1 - 1:
                        continue
                    n += 1
            return n

        def _cmp(seed, Nq, Nk, d, dv, causal, Br, Bc, scale=1.0):
            r = np.random.default_rng(seed)
            q, k, v = r.normal(size=(Nq, d)) * scale, r.normal(size=(Nk, d)) * scale, r.normal(size=(Nk, dv))
            O, L, n = flash_attention_forward(q, k, v, causal=causal, Br=Br, Bc=Bc)
            eO, eL = _naive_attn(q, k, v, causal)
            assert O.shape == eO.shape, f"O 形状应为 {eO.shape}，得到 {O.shape}"
            assert np.all(np.isfinite(O)), "O 中出现 NaN/inf"
            assert np.allclose(O, eO, atol=1e-9), f"O 最大误差 {np.abs(O - eO).max():.2e}"
            assert np.allclose(L, eL, atol=1e-9), f"logsumexp 最大误差 {np.abs(L - eL).max():.2e}"
            exp_n = _expected_tiles(Nq, Nk, Br, Bc, causal)
            assert n == exp_n, f"计算的 tile 数应为 {exp_n}，得到 {n}（因果时是否跳过了上三角整块？）"
      `,
      tests: [
        { name: '非因果，整除', code: py`_cmp(0, 8, 8, 4, 3, False, 4, 4)` },
        { name: '因果，Br = Bc = 2（应计算 10 个 tile）', code: py`_cmp(1, 8, 8, 4, 3, True, 2, 2)` },
        { name: '非因果，Nq ≠ Nk、不整除', code: py`_cmp(2, 7, 10, 6, 2, False, 3, 4)` },
        { name: '因果，Br ≠ Bc、不整除', hidden: true, code: py`_cmp(3, 11, 11, 4, 5, True, 4, 3)` },
        { name: '大数值稳定', hidden: true, code: py`_cmp(4, 9, 9, 8, 2, True, 2, 4, scale=12.0)` },
        {
          name: '不构造完整 N×N 分数矩阵',
          hidden: true,
          code: py`
            src = inspect.getsource(flash_attention_forward).replace(" ", "")
            for bad in ("q@k.T", "q@k.transpose", "np.tril(np.ones((Nq", "np.tril(np.ones((N"):
                assert bad not in src, f"发现 {bad}：请只按 tile 计算分数"
            _cmp(5, 6, 6, 4, 4, True, 1, 1)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-bwd',
          q: md`用前向返回的 \`L\`，反向如何得到第 $(i, j)$ 个 tile 的概率块？$dS$ 怎么算？`,
          a: md`$P_{ij} = \exp(Q_iK_j^\top/\sqrt d - L_i)$（对角 tile 同样加掩码）。$dP_{ij} = dO_i V_j^\top$，$D_i = \operatorname{rowsum}(dO_i \circ O_i)$（每行一个数，预先算好），$dS_{ij} = P_{ij}\circ(dP_{ij} - D_i)$，再累加 $dQ_i$、$dK_j$、$dV_j$。FA2 反向以 K/V 块为外层，$dQ$ 用 atomic add 累加。`,
        },
        {
          id: 'f-why-sram',
          q: md`块大小是越大越好吗？`,
          a: md`不是。块越大，K/V 被重复读取的次数越少（$\Theta(N^2d^2/M)$ 中 $M$ 越大越好），但受限于 SRAM/寄存器容量、occupancy（每个 SM 能同时驻留的 thread block 数）以及 head_dim。FA1 论文 Fig. 2 显示块大小超过 256 后运行时间不再下降，瓶颈转向算术运算；FA2 实际按 head_dim 与设备在 {64, 128} × {64, 128} 中选择。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-principle',
      freq: 3,
      q: md`FlashAttention 的原理是什么？为什么能加速？`,
      a: md`
        **结论**：IO-aware 的精确注意力——通过 tiling + online softmax 把整个注意力融合进一个 kernel，$N\times N$ 的中间矩阵只在 SRAM 里分块存在、从不写回 HBM。

        **原因**：标准实现中 softmax/mask/dropout 都是 memory-bound，读写 $N^2$ 矩阵的时间远超计算。FlashAttention 的 HBM 访问从 $\Theta(Nd + N^2)$ 降到 $\Theta(N^2d^2/M)$，额外显存从 $O(N^2)$ 降到 $O(N)$；反向用存下的 logsumexp 重算 $P$。FLOPs 反而略增，但论文实测 GPT-2 medium 上 HBM 读写从 40.3 GB 降到 4.4 GB，时间从 41.7 ms 降到 7.3 ms。

        **延伸**：它不改变 $O(N^2)$ 的计算量，长序列下仍要配合稀疏 / 线性注意力；推理 decode 用 split-KV（Flash-Decoding）；FA2/FA3/FA4 分别针对 A100/H100/B200 优化。
      `,
    },
    {
      id: 'qa-online',
      freq: 3,
      q: md`推导一下 online softmax。`,
      a: md`
        **结论**：维护 running max $m$ 与以 $m$ 为基准的指数和 $\ell$（以及输出累加 acc），新块到来时用 $\alpha = e^{m_{\text{old}} - m_{\text{new}}}$ 缩放旧量。

        **推导**：$\ell = \sum_j e^{x_j - m}$。若最大值变为 $m'$，则 $\sum_j e^{x_j - m'} = e^{m - m'}\sum_j e^{x_j - m}$，所以 $\ell' = \alpha\ell + \sum_{j\in\text{新块}} e^{x_j - m'}$；输出累加同理 $\text{acc}' = \alpha\,\text{acc} + \sum e^{x_j - m'}v_j$。最后 $o = \text{acc}/\ell$，与一次性 softmax 精确相等。

        **延伸**：两段结果的合并用 logsumexp：$o = \sum_s e^{\text{lse}_s - \text{lse}}o_s$，用于 Flash-Decoding、Ring Attention、前缀缓存的级联注意力。边界：全 $-\infty$ 块要特判，否则 $-\infty-(-\infty)$ 得 NaN。
      `,
    },
    {
      id: 'qa-bwd',
      freq: 2,
      q: md`FlashAttention 的反向怎么做？重计算为什么不会更慢？`,
      a: md`
        **结论**：前向只存 $O$ 和每行 logsumexp $L$；反向按块重算 $S = QK^\top/\sqrt d$、$P = \exp(S - L)$，再算梯度。

        **公式**：$dV = P^\top dO$，$dP = dO\,V^\top$，$D = \operatorname{rowsum}(dO \circ O)$，$dS = P\circ(dP - D)$，$dQ = dS\,K/\sqrt d$，$dK = dS^\top Q/\sqrt d$。

        **为什么更快**：瓶颈是 HBM 带宽，从 HBM 读 $N^2$ 的 $P$ 比在 SRAM 里重算慢得多；重算多出的 FLOPs 在 Tensor Core 上很便宜。dropout 也不存掩码，而是保存随机数状态重新生成。
      `,
    },
    {
      id: 'qa-versions',
      freq: 2,
      q: md`FlashAttention-2、3 相比第一代做了哪些改进？`,
      a: md`
        **FA2（A100）**：① 减少非矩阵乘 FLOPs——输出不在每步除以 $\ell$，最后统一缩放，反向只存 logsumexp（A100 上非矩阵乘 FLOP 比矩阵乘贵 16 倍）；② 循环改为外层 Q，沿序列维并行，长序列小 batch 也能占满 SM；③ warp 间按 Q 切分，避免 split-K 的共享内存通信。约 2× 于 FA1，达理论峰值 50–73%。

        **FA3（H100）**：warp specialization 利用 TMA 与 Tensor Core 的异步性重叠搬运与计算；GEMM 与 softmax 交错流水；FP8 块量化 + incoherent processing。1.5–2.0× 于 FA2，FP16 达 740 TFLOPs/s（75%）。

        **延伸**：FA4（2026，B200）针对 Tensor Core 吞吐翻倍而 exp 单元与共享内存带宽跟不上的问题，用软件模拟 exp、条件式 rescale、tensor memory 与 2-CTA MMA，BF16 达 1613 TFLOPs/s（71%）。
      `,
    },
    {
      id: 'qa-decode',
      freq: 2,
      q: md`推理 decode 阶段 FlashAttention 还有用吗？`,
      a: md`
        **结论**：有用，但需要 split-KV（Flash-Decoding）改造。

        **原因**：decode 每个序列只有 1 个 query，FA2 沿 Q 块并行无从下手，batch 小时大部分 SM 空闲；而且 decode 的瓶颈是读 KV cache。Flash-Decoding 把 KV 序列切成多段分给不同 thread block，各自算出部分输出与 logsumexp，再用一个小的归约 kernel 按 $e^{\text{lse}_s - \text{lse}}$ 加权合并。

        **延伸**：配合 PagedAttention（按块寻址 KV）、GQA/MLA（减少要读的字节）以及 FlashMLA、FlashInfer 等专用 decode kernel。
      `,
    },
  ],

  cheatsheet: md`
    **online softmax**（每行维护 $m, \ell, \text{acc}$）
    \`\`\`python
    m_new = np.maximum(m, S.max(1))
    P = np.exp(S - m_new[:, None])
    alpha = np.exp(m - m_new)           # ≤ 1，旧量换基准
    l = alpha * l + P.sum(1)
    acc = alpha[:, None] * acc + P @ Vj
    m = m_new                           # 结束：O = acc / l，L = m + log(l)
    \`\`\`

    - HBM 访问：标准 $\Theta(Nd+N^2)$ → Flash $\Theta(N^2d^2/M)$；额外显存 $O(N^2)\to O(N)$；FLOPs 不降
    - FA1：$B_c=\lceil M/4d\rceil$，外层 K/V；FA2：外层 Q、序列维并行、只存 logsumexp
    - 反向：$P=\exp(S-L)$，$D=\text{rowsum}(dO\circ O)$，$dS=P\circ(dP-D)$
    - 因果：$j>i$ 的 tile 跳过（$T(T+1)/2$ 个），对角 tile 块内掩码
    - 全 $-\infty$ 块：$\alpha$ 会变 NaN，要特判
    - A100：非矩阵乘 FLOP 贵 16×（312 vs 19.5 TFLOPs/s）
    - FA3（H100）：warp specialization + FP8，740 TFLOPs/s；FA4（B200）：1613 TFLOPs/s
    - decode：split-KV + lse 合并（Flash-Decoding）
  `,
})
