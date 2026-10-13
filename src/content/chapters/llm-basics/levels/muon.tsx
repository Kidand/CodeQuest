/**
 * llm-basics-muon · Muon 优化器
 * 事实来源：Keller Jordan 的 Muon 博客与开源实现；Moonlight（arXiv 2502.16982）；Kimi K2 技术报告（arXiv 2507.20534）；
 * GLM-4.5 技术报告（arXiv 2508.06471）；DeepSeek-V4（arXiv 2606.19348）§2.4 与 §3.4.1。
 * 文中 Newton-Schulz 多项式的收敛区间等数值为本关自行计算。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { NSIterViz } from '../viz/NSIterViz'
import { UpdateRMSViz } from '../viz/UpdateRMSViz'
import { QKClipViz } from '../viz/QKClipViz'

/** 跟练与实战共用的 Newton-Schulz 参考实现（放在 setup 中，名称带 _ref_ 前缀） */
const REF_NS = py`
  def _ref_ns(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
      a, b, c = coeffs
      X = G / (np.linalg.norm(G) + eps)
      tr = G.shape[0] > G.shape[1]
      if tr:
          X = X.T
      for _ in range(steps):
          A = X @ X.T
          X = a * X + (b * A + c * A @ A) @ X
      return X.T if tr else X
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'why',
      title: '为什么要对更新矩阵做“正交化”',
      body: md`
        AdamW 把每个参数当作独立的标量来更新。但 Transformer 里绝大多数参数是**矩阵**（$W_Q$、$W_{\text{up}}$……），矩阵有结构：它的更新 $G$ 可以做 SVD，$G = U\Sigma V^\top$。

        Keller Jordan 观察到：Transformer 中 SGD-动量 / Adam 给出的矩阵更新**条件数非常高**，近似低秩——几乎全部能量集中在少数几个奇异方向上。那些奇异值很小的“稀有方向”几乎得不到更新，尽管它们可能对学习很重要。

        **Muon**（MomentUm Orthogonalized by Newton-Schulz）的做法：把动量矩阵 $M$ 换成它的“正交化”版本

        $$
        \mathrm{Ortho}(M) = UV^\top \quad (\text{把所有非零奇异值都置为 } 1)
        $$

        - $UV^\top$ 是离 $M$ 最近的（半）正交矩阵（Frobenius 意义下），保留了更新的“方向结构”，丢掉了各方向的“强弱”；
        - 效果上等于**把被主导方向压制的稀有方向放大**，每个奇异方向走同样大小的一步；
        - 理论上可以理解为在谱范数约束下的最速下降（Bernstein 等人的分析）；
        - 对 $G$ 的整体缩放不敏感：$\mathrm{Ortho}(cG) = \mathrm{Ortho}(G)$，所以学习率要配合额外的缩放规则（后面的 update RMS 对齐）。

        直接做 SVD 太慢（且在 GPU 上不友好），Muon 用 **Newton-Schulz 迭代**只靠矩阵乘法近似计算 $UV^\top$——下一节的主角。

        > [!NEW] 2025–2026 的大模型在用 Muon
        > Moonlight（Kimi，3B 激活 / 16B 总参数，5.7T token）报告 Muon 在 compute-optimal 训练下约有 AdamW 的 **2 倍**计算效率；随后 Kimi K2（1.04T / 32B 激活）、GLM-4.5、DeepSeek-V4 都把 Muon 用于大部分矩阵参数。
      `,
      checkpoint: {
        id: 'cp-why',
        type: 'single',
        prompt: md`动量矩阵 $M = U\,\mathrm{diag}(10, 1, 0.01)\,V^\top$。$\mathrm{Ortho}(M)$ 是：`,
        options: [
          '$U\\,\\mathrm{diag}(1, 0.1, 0.001)\\,V^\\top$（除以最大奇异值）',
          '$U\\,\\mathrm{diag}(1, 1, 1)\\,V^\\top = UV^\\top$',
          '$M / \\|M\\|_F$',
          '单位矩阵 $I$',
        ],
        answer: 1,
        explanation: md`正交化把所有非零奇异值置为 1、保留奇异向量。A、C 只是整体缩放，奇异值的相对比例（条件数 1000）不变，稀有方向仍然几乎不动；D 丢掉了 $U$、$V$，与梯度方向无关。`,
      },
    },
    {
      id: 'ns',
      title: 'Newton-Schulz 迭代：只用矩阵乘法逼近 UVᵀ',
      body: md`
        对 $X = U\Sigma V^\top$，有 $XX^\top X = U\Sigma^3V^\top$。所以奇数次多项式迭代

        $$
        X \leftarrow aX + b\,(XX^\top)X + c\,(XX^\top)^2X
        $$

        **只改变奇异值**：$\sigma \leftarrow \varphi(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$，$U, V$ 保持不变。只要反复迭代能把 $[0, 1]$ 内的每个 $\sigma$ 都推到 1 附近，$X$ 就逼近 $UV^\top$。

        实现要点（Keller Jordan 的 \`zeropower_via_newtonschulz5\`）：
        1. **先归一化** $X = G/(\|G\|_F + \varepsilon)$：Frobenius 范数 $\ge$ 谱范数，保证所有 $\sigma \le 1$，落在 $\varphi$ 的收敛区域内；
        2. **宽矩阵**：若行数 > 列数先转置，使 $XX^\top$ 是较小的那个方阵（省计算），最后再转回来；
        3. **系数** $(a, b, c) = (3.4445, -4.7750, 2.0315)$，迭代 5 步，在 **bf16** 下运行。

        **系数怎么来的**：经典三次 Newton-Schulz $(1.5, -0.5)$ 能精确收敛到 1，但小奇异值每步只放大 1.5 倍，太慢。Keller 的思路是尽量**增大 $\varphi'(0) = a$**（让小奇异值增长最快），同时只要求最终落在 $[0.7, 1.3]$ 左右——他发现奇异值不精确等于 1 并不影响训练效果。代价是 $\varphi(1) = 0.701$：迭代不收敛到 1，而在一个区间内振荡（本关计算：充分迭代后约为 $[0.68, 1.13]$）。

        **DeepSeek-V4 的混合 NS**：共 10 步，前 8 步用 $(3.4445, -4.7750, 2.0315)$ 快速拉近，后 2 步换成 $(2, -1.5, 0.5)$——它满足 $\varphi(1) = 1$、$\varphi'(1) = 0$，在 1 附近收敛极快，把奇异值“精确地稳定在 1”（论文原话）。
      `,
      viz: NSIterViz,
      after: md`
        > [!NOTE] 开销
        > 每步 NS 是几次矩阵乘法。Keller 估算 Muon 的额外 FLOPs 不超过 $Tm/B$（$T$ 为 NS 步数，$m$ 为模型维度，$B$ 为以 token 计的 batch size）：NanoGPT speedrun 约 0.7%，Llama 405B 的设定约 0.5%。
      `,
      checkpoint: {
        id: 'cp-ns',
        type: 'judge',
        prompt: md`若不先把 $G$ 除以 $\|G\|_F$，直接对一个奇异值约为 5 的矩阵迭代 Muon 的五次多项式，奇异值仍会被推到 1 附近。`,
        answer: false,
        explanation: md`$\varphi(5) = 3.4445\times5 - 4.775\times125 + 2.0315\times3125 \approx 5770$，再迭代会立刻爆炸。$\varphi$ 只在 $[0, \approx 1.2]$ 附近表现良好，归一化是必需的。`,
      },
    },
    {
      id: 'algo',
      title: 'Muon 完整算法与“只用于二维隐藏层矩阵”',
      body: md`
        以 DeepSeek-V4 论文的 Algorithm 1 为例（与 Moonlight / Kimi K2 一致，只是 NS 与缩放常数不同），对每个**逻辑上独立**的矩阵 $W \in \mathbb{R}^{n\times m}$：

        $$
        \begin{aligned}
        M_t &= \mu M_{t-1} + G_t & &\text{动量} \\
        O'_t &= \mathrm{NS}(\mu M_t + G_t) & &\text{Nesterov + 正交化} \\
        O_t &= O'_t \cdot \sqrt{\max(n, m)} \cdot \gamma & &\text{update RMS 对齐} \\
        W_t &= W_{t-1}(1 - \eta\lambda) - \eta\, O_t & &\text{解耦权重衰减 + 更新}
        \end{aligned}
        $$

        | | Keller Jordan（开源实现） | Moonlight / Kimi K2 | GLM-4.5 | DeepSeek-V4 |
        |---|---|---|---|---|
        | NS | 五次，5 步 | 五次，5 步 | 5 步 | 混合 8 + 2 = 10 步 |
        | 动量 | 0.95，Nesterov | 0.95，Nesterov（Moonlight） | 0.95 | 0.95，Nesterov |
        | 缩放 | $\sqrt{\max(1, \text{out}/\text{in})}$ | $0.2\sqrt{\max(n,m)}$ | RMS 对齐到 0.2 | RMS 对齐到 0.18 |
        | 权重衰减 | 可选（默认 0） | 有（K2 为 0.1） | 0.1 | 0.1 |

        **哪些参数用 Muon**：只有**隐藏层的二维矩阵**。以下仍用 AdamW：
        - 输入 embedding 与输出头（LM head）——虽然是二维，但 Keller 明确建议交给 AdamW；
        - 所有一维参数：norm 的增益、偏置；DeepSeek-V4 还包括 mHC 的静态偏置与门控因子；
        - GLM-4.5 报告的划分是：除 word embedding、bias、RMSNorm 权重外都用 Muon。

        **“逻辑上独立”**：Q、K、V 要**分开**正交化（Keller 发现比对拼接的 QKV 整体正交化更好）；MoE 的每个专家单独处理（V4 把所有专家的同类矩阵展平后按专家切分）。卷积核可把后三维展平成矩阵。
      `,
      checkpoint: {
        id: 'cp-algo',
        type: 'multi',
        prompt: md`在一个用 Muon 训练的 LLaMA 式模型中，下列哪些参数**通常交给 AdamW** 而不是 Muon？（多选）`,
        options: ['`embed_tokens.weight`', '`self_attn.q_proj.weight`', '`input_layernorm.weight`', '`lm_head.weight`', '`mlp.down_proj.weight`'],
        answer: [0, 2, 3],
        explanation: md`embedding 与 LM head 虽然是矩阵，但它们的“行”对应离散的 token，梯度极度稀疏且语义不同，实践中都交给 AdamW（Keller 的建议与 Moonlight、K2、V4 的做法一致）；RMSNorm 增益是一维的。q_proj、down_proj 是隐藏层矩阵，用 Muon。`,
      },
    },
    {
      id: 'rms',
      title: 'Update RMS 对齐：让 Muon 复用 AdamW 的超参数',
      body: md`
        正交化矩阵 $O = UV^\top$（满秩，形状 $A\times B$）有 $\min(A, B)$ 个奇异值为 1，于是

        $$
        \mathrm{RMS}(O) = \sqrt{\frac{\|O\|_F^2}{AB}} = \sqrt{\frac{\min(A,B)}{AB}} = \frac{1}{\sqrt{\max(A, B)}}
        $$

        这带来两个问题：(1) 不同形状的层更新幅度不同；(2) 数值远小于 AdamW——Moonlight 论文观察到 AdamW 的 update RMS 通常在 0.2–0.4。

        Moonlight 把这列为 Muon 能扩展到大模型的**两个关键之一**（另一个是权重衰减）：

        $$
        W \leftarrow W - \eta\,\big(0.2\cdot O\cdot\sqrt{\max(A,B)} + \lambda W\big)
        $$

        乘上 $0.2\sqrt{\max(A,B)}$ 后每一层的 update RMS 都是 0.2，和 AdamW 处在同一量级，于是**可以直接沿用 AdamW 的学习率和权重衰减**，也能与负责 embedding / norm 的 AdamW 共用同一个学习率调度。Kimi K2、GLM-4.5 同样对齐到 0.2；DeepSeek-V4 对齐到 0.18。

        > [!WARN] 为什么必须加权重衰减
        > Moonlight 发现：原版 Muon（不加权重衰减）起初收敛更快，但长时间训练后部分权重变得过大——权重与层输出的 RMS 持续增长，超出 bf16 的高精度范围。加上解耦权重衰减后，在过训练（over-train）区间里 Muon 的验证 loss 同时低于原版 Muon 与 AdamW。
      `,
      viz: UpdateRMSViz,
      checkpoint: {
        id: 'cp-rms',
        type: 'numeric',
        prompt: md`一个 $4096 \times 14336$ 的 FFN 矩阵，正交化后（满秩）的 update RMS 是多少？（保留 4 位小数）`,
        answer: 0.0084,
        tolerance: 0.01,
        explanation: md`$1/\sqrt{\max(4096, 14336)} = 1/\sqrt{14336} \approx 0.00835$。要对齐到 0.2，需要乘 $0.2\sqrt{14336} \approx 23.9$。`,
      },
    },
    {
      id: 'clip',
      title: 'MuonClip / QK-Clip：Kimi K2 如何压住 logit 爆炸',
      body: md`
        Muon 让权重矩阵的更新“各方向一样大”，这也让 $W_Q, W_K$ 的谱范数更容易增长，attention logit 随之变大。Kimi K2 报告：在一个 9B 激活 / 53B 总参数的 MoE 上用原版 Muon，**最大 attention logit 很快超过 1000**，造成训练不稳定。

        已有方案为什么不够：
        - **logit soft-capping**：在算出 logit 之后再截断，$q\cdot k$ 本身仍可能过度增长；
        - **QK-Norm**：K2 使用 MLA，推理时 Key 矩阵不被完整显式构造，QK-Norm 无法施加。

        **QK-Clip**（逐头，发生在优化器更新**之后**，不改变当前步的前向与反向）：

        $$
        S_{\max}^h = \frac{1}{\sqrt{d}}\max_{X \in B}\max_{i,j} Q_i^h K_j^{h\top},\qquad
        \gamma_h = \min\!\Big(1, \frac{\tau}{S_{\max}^h}\Big)
        $$

        对 $S_{\max}^h > \tau$ 的头，把 $W_q, W_k$ 各乘 $\sqrt{\gamma_h}$——logit 恰好整体缩小 $\gamma_h$ 倍。在 MLA 中：
        - 头独有的 $q^C$、$k^C$（内容部分）各乘 $\sqrt{\gamma_h}$；
        - 头独有的 $q^R$（RoPE 部分）乘 $\gamma_h$；
        - **共享的 $k^R$ 不动**，以免影响其它头。

        **MuonClip** = Muon + 权重衰减 + update RMS 对齐（$0.2\sqrt{\max(n,m)}$）+ QK-Clip。K2 取 $\tau = 100$，权重衰减 0.1，用 15.5T token 完成预训练，报告称**全程没有一次 loss spike**；最大 logit 起初被压在 100，大约训练 30% 之后自然回落到正常范围，不再触发裁剪。
      `,
      viz: QKClipViz,
      checkpoint: {
        id: 'cp-clip',
        type: 'numeric',
        prompt: md`$\tau = 100$，某个头本步的 $S_{\max} = 400$。在 MLA 中，该头的 $W_{q}^{R}$（RoPE 部分的 query 投影）要乘以多少？`,
        answer: 0.25,
        tolerance: 0.001,
        explanation: md`$\gamma = 100/400 = 0.25$。$q^C, k^C$ 各乘 $\sqrt{0.25} = 0.5$；$q^R$ 乘 $\gamma = 0.25$；共享的 $k^R$ 不动。这样内容部分与 RoPE 部分的点积都恰好缩小到 1/4，新的最大 logit 为 100。`,
      },
    },
    {
      id: 'v4',
      title: 'DeepSeek-V4：混合 NS、不用 QK-Clip，以及工程实现',
      body: md`
        DeepSeek-V4（arXiv 2606.19348，§2.4）对“大部分模块”使用 Muon，并做了三点不同的选择：

        1. **混合 Newton-Schulz**：10 步，前 8 步 $(3.4445, -4.7750, 2.0315)$ 快速把奇异值拉到 1 附近，后 2 步 $(2, -1.5, 0.5)$ 把它们稳定在 1。
        2. **不用 QK-Clip**：V4 的 CSA / HCA 注意力在核心注意力计算前，对每个 query 头以及压缩后的 KV 条目都做了 **RMSNorm**——论文明确说这能有效防止 attention logit 爆炸，**因此不需要 QK-Clip**。这与 K2 正好相反：K2 的 MLA 无法加 QK-Norm，只好在优化器侧裁剪。
        3. **超参数**：动量 0.95、权重衰减 0.1、每个更新矩阵的 RMS 缩放到 0.18 以复用 AdamW 的学习率；AdamW 部分 $\beta = (0.9, 0.95)$、$\varepsilon = 10^{-20}$、权重衰减 0.1。V4-Flash 峰值学习率 $2.7\times10^{-4}$，V4-Pro $2.0\times10^{-4}$。

        **工程（§3.4.1）**：Muon 需要**完整的梯度矩阵**才能做 NS，和按元素切分的 ZeRO 冲突（Moonlight 为此提出 Distributed Muon：先 gather 完整矩阵，再只保留本地分片的更新）。V4 的做法：
        - 稠密参数限制 ZeRO 并行度，用**背包算法**把整块矩阵分配到各 rank 以均衡负载（padding 开销通常 < 10%）；
        - MoE 参数按专家独立优化，把所有专家的同类矩阵展平后均匀切分，且不切开任何一个逻辑矩阵；
        - 同形状的连续参数合并，**批量**执行 NS；观察到 NS 在 **BF16** 矩阵乘下保持稳定，并把要同步的 MoE 梯度以随机舍入量化到 BF16，通信量减半。

        **显存**：Muon 对每个矩阵只维护一个动量缓冲区，而 AdamW 要两个——优化器状态约为 AdamW 的一半（Moonlight）。
      `,
      checkpoint: {
        id: 'cp-v4',
        type: 'single',
        prompt: md`DeepSeek-V4 不使用 QK-Clip 的理由是：`,
        options: [
          '混合 Newton-Schulz 让 $W_Q, W_K$ 的谱范数不再增长',
          '注意力中对 query 头与压缩 KV 条目做了 RMSNorm，直接防止了 logit 爆炸',
          'V4 用的是 AdamW 而不是 Muon 训练注意力参数',
          'V4 的 MoE 使用 hash 路由，避免了 logit 爆炸',
        ],
        answer: 1,
        explanation: md`论文原文：V4 的注意力架构允许直接对 query 与 KV 条目施加 RMSNorm，有效防止 attention logit 爆炸，因此不在 Muon 中使用 QK-Clip。A：混合 NS 只是让正交化更精确；C 错：V4 对大部分模块用 Muon；D 与注意力 logit 无关。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-which',
      type: 'single',
      difficulty: 1,
      prompt: md`Muon 的设计用途是优化哪类参数？`,
      options: ['所有参数', '只有隐藏层的二维权重矩阵', '只有 embedding 与 LM head', '只有一维参数（norm、bias）'],
      answer: 1,
      explanation: md`正交化是矩阵运算，只对二维矩阵有意义；embedding / LM head 虽然是矩阵，但实践中（Keller、Moonlight、K2、V4）都交给 AdamW，一维参数更不必说。`,
    },
    {
      id: 'q-state',
      type: 'judge',
      difficulty: 1,
      prompt: md`和 Adam 一样，Muon 为每个矩阵参数维护一阶矩与二阶矩两个状态缓冲区。`,
      answer: false,
      explanation: md`Muon 只有一个动量缓冲区 $M$，没有二阶矩；“自适应”的作用由正交化承担。因此 Muon 的优化器状态约为 AdamW 的一半（Moonlight 论文）。`,
    },
    {
      id: 'q-scale',
      type: 'numeric',
      difficulty: 2,
      prompt: md`按 Moonlight 的做法把正交化更新的 RMS 对齐到 0.2，一个 $4096\times14336$ 的矩阵，NS 输出需要乘以多少？（保留两位小数）`,
      answer: 23.95,
      tolerance: 0.002,
      explanation: md`缩放因子 $0.2\sqrt{\max(A,B)} = 0.2\sqrt{14336} \approx 23.95$。乘之前 RMS 为 $1/\sqrt{14336} \approx 0.00835$，乘之后恰为 0.2。`,
    },
    {
      id: 'q-coef',
      type: 'fill',
      difficulty: 2,
      prompt: md`Muon 的五次 Newton-Schulz 迭代 $X \leftarrow aX + b(XX^\top)X + c(XX^\top)^2X$ 的系数为 $a$ = ____(1)____，$b$ = ____(2)____，$c$ = ____(3)____。`,
      blanks: [
        { label: '(1)', answers: ['3.4445'], numeric: true, tolerance: 0.0001 },
        { label: '(2)', answers: ['-4.7750'], numeric: true, tolerance: 0.0001 },
        { label: '(3)', answers: ['2.0315'], numeric: true, tolerance: 0.0001 },
      ],
      explanation: md`$(3.4445, -4.7750, 2.0315)$：$a$ 尽量大（小奇异值增长快），$a + b + c = 0.701$，所以迭代不精确收敛到 1，而是在约 $[0.68, 1.13]$ 间振荡。DeepSeek-V4 在最后 2 步改用 $(2, -1.5, 0.5)$ 精修。`,
    },
    {
      id: 'q-ns-props',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 Muon 中的 Newton-Schulz 迭代，正确的有（多选）：`,
      options: [
        '奇数次多项式迭代不改变 $X$ 的左、右奇异向量，只改变奇异值',
        '迭代前要先把矩阵除以 Frobenius 范数，保证奇异值不超过 1',
        '对行数大于列数的矩阵先转置，是为了让 $XX^\\top$ 更小、计算更省',
        '五次系数 $(3.4445, -4.7750, 2.0315)$ 迭代足够多步后，奇异值精确收敛到 1',
        'Keller 的实现与 DeepSeek-V4 都在 bf16 下执行 NS 迭代',
      ],
      answer: [0, 1, 2, 4],
      explanation: md`D 错：$\varphi(1) = 0.701 \ne 1$，1 不是不动点，迭代会在约 $[0.68, 1.13]$ 振荡；要精确到 1 需要像 V4 那样最后换成 $\varphi(1) = 1$ 的系数。E：Keller 的代码把 $G$ 转为 bfloat16 再迭代；V4 论文报告 NS 在 BF16 矩阵乘下保持稳定。`,
    },
    {
      id: 'q-qk',
      type: 'numeric',
      difficulty: 3,
      prompt: md`Kimi K2 的 QK-Clip，$\tau = 100$。某个 MHA 头本步 $S_{\max} = 400$，裁剪时 $W_q$ 应乘以多少？`,
      answer: 0.5,
      tolerance: 0.001,
      explanation: md`$\gamma = \tau/S_{\max} = 0.25$，$W_q, W_k$ 各乘 $\sqrt{\gamma} = 0.5$，logit 是二者之积，整体缩小 0.25 倍，新的最大值为 100。若只缩放 $W_q$ 就要乘 0.25——K2 选择对称地平分到 $W_q$ 与 $W_k$。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 Newton-Schulz 实现有 **2 处** bug（一处会让迭代发散，一处会在非方阵上直接报错）：`,
      code: py`
        def newton_schulz(G, steps=5, eps=1e-7):
            a, b, c = 3.4445, -4.7750, 2.0315
            X = G.astype(np.float32)
            transposed = G.shape[0] > G.shape[1]
            if transposed:
                X = X.T
            for _ in range(steps):
                A = X.T @ X
                B = b * A + c * A @ A
                X = a * X + B @ X
            if transposed:
                X = X.T
            return X
      `,
      bugLines: [3, 8],
      explanation: md`
        - 第 3 行没有归一化，应为 \`X = G / (np.linalg.norm(G) + eps)\`。奇异值大于约 1.2 时五次多项式会爆炸。
        - 第 8 行应为 \`A = X @ X.T\`：转置后 $X$ 是宽矩阵 $(r, c)$，\`X @ X.T\` 是 $(r, r)$，\`B @ X\` 形状才对得上；写成 \`X.T @ X\` 得到 $(c, c)$，与 $X$ 左乘时形状不匹配而报错；方阵时虽不报错，算出的却是 $(X^\top X)X$，它不等于 $(XX^\top)X$，结果静默地错误。
      `,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`按 DeepSeek-V4 论文 Algorithm 1 的顺序排列 Muon 的一步更新：`,
      items: [
        '计算梯度 $G_t$',
        '累积动量 $M_t = \\mu M_{t-1} + G_t$',
        '对 $\\mu M_t + G_t$ 做（混合）Newton-Schulz 正交化',
        '乘以 $\\sqrt{\\max(n,m)}\\cdot\\gamma$ 对齐 update RMS',
        '$W_t = W_{t-1}(1-\\eta\\lambda) - \\eta O_t$',
      ],
      explanation: md`Nesterov 技巧体现在第 3 步：正交化的对象是 $\mu M_t + G_t$ 而不是 $M_t$。正交化之后才缩放（因为 NS 输出与输入尺度无关）；权重衰减与更新在最后一步同时完成。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把模型 / 工作与它们在 Muon 上的关键做法配对：`,
      pairs: [
        ['Keller Jordan（NanoGPT speedrun）', '五次 NS、Nesterov，缩放 √max(1, out/in)'],
        ['Moonlight', '加权重衰减 + 0.2·√max(A,B) 对齐 RMS，约 2× 计算效率'],
        ['Kimi K2', 'MuonClip：逐头 QK-Clip，τ = 100'],
        ['DeepSeek-V4', '8 + 2 步混合 NS，RMS 0.18，不用 QK-Clip'],
        ['GLM-4.5', 'NS 5 步、动量 0.95、RMS 0.2，embedding / bias / RMSNorm 用 AdamW'],
      ],
      explanation: md`均来自各自的博客或技术报告。Keller 的博客报告 Muon 把 NanoGPT speedrun 的训练速度提升了约 1.35 倍；Moonlight 的 scaling 实验显示约 2 倍计算效率。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 3,
      prompt: md`拼出 Moonlight / V4 风格的 Muon 单步更新（\`newton_schulz\` 已实现）。有 2 行干扰项。`,
      lines: [
        'def muon_step(W, G, M, lr, mu=0.95, wd=0.1, rms=0.2):',
        '    M = mu * M + G',
        '    O = newton_schulz(mu * M + G)',
        '    O = O * rms * np.sqrt(max(W.shape))',
        '    W = W * (1 - lr * wd) - lr * O',
        '    return W, M',
      ],
      distractors: ['    O = newton_schulz(G) * np.sqrt(M)', '    W = W - lr * (O + wd * G)'],
      explanation: md`动量 → Nesterov 正交化 → RMS 对齐 → 解耦衰减与更新。干扰项 1 把 Adam 的“除以二阶矩”思路混了进来；干扰项 2 的衰减项乘的是梯度而不是权重。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`一个 LLaMA 式 decoder 层加上输入输出层，哪些参数会交给 **Muon**？点出全部。`,
      items: ['embed_tokens', 'q_proj', 'k_proj', 'o_proj', 'input_layernorm', 'gate_proj', 'down_proj', 'lm_head'],
      answer: [1, 2, 3, 5, 6],
      explanation: md`隐藏层的二维矩阵（q/k/o、gate/down，以及图中没列出的 v_proj、up_proj）用 Muon，并且 q、k、v 各自独立正交化。embedding、LM head 与 norm 增益交给 AdamW。`,
    },
    {
      id: 'q-v4',
      type: 'single',
      difficulty: 3,
      prompt: md`Kimi K2 需要 QK-Clip，而 DeepSeek-V4 不需要。最根本的区别在于：`,
      options: [
        'K2 的学习率更大',
        'V4 的注意力在点积前对 query 与（压缩）KV 条目做了 RMSNorm，K2 的 MLA 无法对 Key 施加 QK-Norm',
        'V4 用了更多的 NS 迭代步数',
        'K2 没有使用权重衰减',
      ],
      answer: 1,
      explanation: md`二者都面临 Muon 下 logit 增长的风险。V4 在架构上用 RMSNorm 把 $q$ 和 KV 条目的范数钉住，logit 天然有界；K2 的 MLA 推理时不显式构造完整 Key，没法加 QK-Norm，只能在优化器侧用 QK-Clip 直接缩放 $W_q, W_k$。C：NS 步数影响正交化精度，与 logit 增长无直接关系；D 错：K2 的权重衰减为 0.1。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-ns',
      title: '写出 Newton-Schulz 正交化',
      intro: md`
        实现 \`newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7)\`：输入任意形状的二维矩阵 \`G\`，返回同形状、奇异值被推到 1 附近的矩阵（近似 $UV^\top$）。
      `,
      template: py`
        import numpy as np

        def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
            a, b, c = coeffs
            X = {{1}}
            transposed = {{2}}
            if transposed:
                X = X.T
            for _ in range(steps):
                A = {{3}}
                B = {{4}}
                X = {{5}}
            if transposed:
                X = X.T
            return X
      `,
      steps: [
        {
          blank: 1,
          instruction: md`按 Frobenius 范数归一化，保证所有奇异值 $\le 1$。`,
          answer: 'G / (np.linalg.norm(G) + eps)',
          accept: ['G / (np.sqrt((G * G).sum()) + eps)', 'G / (np.linalg.norm(G, "fro") + eps)'],
          explain: md`$\|G\|_F = \sqrt{\sum\sigma_i^2} \ge \sigma_{\max}$。不用谱范数是因为求谱范数本身就要迭代；Frobenius 范数一步就能算。正交化对整体缩放不变，所以除以任何正数都不改变结果。`,
        },
        {
          blank: 2,
          instruction: md`判断是否需要转置：行数大于列数（“高”矩阵）时转置成“宽”矩阵。`,
          answer: 'G.shape[0] > G.shape[1]',
          accept: ['G.shape[0] > G.shape[-1]', 'X.shape[0] > X.shape[1]'],
          explain: md`宽矩阵 $(r, c)$、$r \le c$ 时，$XX^\top$ 是 $r\times r$，比 $X^\top X$（$c\times c$）小。数学结果相同，只是省计算。`,
        },
        {
          blank: 3,
          instruction: md`计算 $A = XX^\top$（较小的那个方阵）。`,
          answer: 'X @ X.T',
          accept: ['np.matmul(X, X.T)', 'X.dot(X.T)'],
          explain: md`$XX^\top = U\Sigma^2U^\top$，它与 $X$ 相乘只会改变奇异值。`,
        },
        {
          blank: 4,
          instruction: md`计算 $B = bA + cA^2$。`,
          answer: 'b * A + c * A @ A',
          accept: ['b * A + c * (A @ A)', 'b * A + c * np.matmul(A, A)'],
          explain: md`先合并成 $B$ 再乘 $X$，每步只需 3 次矩阵乘法（$XX^\top$、$A^2$、$BX$）。`,
        },
        {
          blank: 5,
          instruction: md`更新 $X \leftarrow aX + BX$。`,
          answer: 'a * X + B @ X',
          accept: ['a * X + np.matmul(B, X)'],
          explain: md`展开即 $aX + b(XX^\top)X + c(XX^\top)^2X$，对奇异值的作用为 $\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5$。`,
        },
      ],
      setup: REF_NS,
      tests: [
        {
          name: '形状保持（宽 / 高）',
          code: py`
            rng = np.random.default_rng(0)
            for shape in [(8, 16), (16, 8), (12, 12)]:
                G = rng.normal(size=shape)
                O = newton_schulz(G)
                assert O.shape == shape, f"形状应为 {shape}，得到 {O.shape}"
          `,
        },
        {
          name: '奇异值落在 [0.6, 1.25]',
          code: py`
            rng = np.random.default_rng(1)
            G = rng.normal(size=(16, 32))
            s = np.linalg.svd(newton_schulz(G), compute_uv=False)
            assert s.min() > 0.6 and s.max() < 1.25, f"奇异值范围 [{s.min():.3f}, {s.max():.3f}]，应在约 [0.68, 1.13]"
          `,
        },
        {
          name: '与 UVᵀ 方向一致 + 缩放不变',
          hidden: true,
          code: py`
            rng = np.random.default_rng(2)
            G = rng.normal(size=(24, 10))
            U, _, Vt = np.linalg.svd(G, full_matrices=False)
            O = newton_schulz(G)
            cos = (O * (U @ Vt)).sum() / (np.linalg.norm(O) * np.linalg.norm(U @ Vt))
            assert cos > 0.98, f"与 UVᵀ 的余弦相似度只有 {cos:.4f}"
            assert np.allclose(newton_schulz(1000.0 * G), O, atol=1e-6), "Ortho(cG) 应等于 Ortho(G)"
            assert np.allclose(O, _ref_ns(G), atol=1e-9), "与参考实现不一致"
          `,
        },
        {
          name: '经典系数迭代足够多步精确收敛',
          hidden: true,
          code: py`
            rng = np.random.default_rng(3)
            G = rng.normal(size=(6, 9))
            U, _, Vt = np.linalg.svd(G, full_matrices=False)
            O = newton_schulz(G, steps=40, coeffs=(1.5, -0.5, 0.0))
            assert np.allclose(O, U @ Vt, atol=1e-8), f"三次 NS 迭代 40 步后应收敛到 UVᵀ，最大误差 {np.abs(O - U @ Vt).max():.2e}"
          `,
        },
      ],
      outro: md`
        把 \`coeffs\` 换成 \`(2, -1.5, 0.5)\` 再迭代两步，就是 DeepSeek-V4 混合 NS 的“精修”阶段。实战第一题会在它的基础上写出完整的 Muon 更新。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-muon-step',
      title: 'Muon 单步更新（Nesterov + RMS 对齐 + 解耦衰减）',
      difficulty: 'medium',
      source: '大模型岗手撕 · 新必会',
      timeLimitMin: 20,
      description: md`
        实现 Moonlight / DeepSeek-V4 风格的 Muon 单步更新（starter 中已给出 \`newton_schulz\`，直接使用即可）：

        \`\`\`python
        def muon_step(W, G, M, lr, momentum=0.95, nesterov=True, weight_decay=0.1,
                      rms_scale=0.2, ns_steps=5) -> tuple[np.ndarray, np.ndarray]:
            """返回 (W_new, M_new)，不修改输入数组"""
        \`\`\`

        对形状为 $(n, m)$ 的矩阵 $W$：
        1. $M \leftarrow \mu M + G$
        2. $U = \mu M + G$（\`nesterov=True\`，用更新后的 $M$）或 $U = M$（\`nesterov=False\`）
        3. $O = \mathrm{NS}(U)\cdot\sqrt{\max(n, m)}\cdot\text{rms\_scale}$
        4. $W \leftarrow W(1 - \eta\lambda) - \eta\,O$
      `,
      starter: py`
        import numpy as np

        def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
            a, b, c = coeffs
            X = G / (np.linalg.norm(G) + eps)
            transposed = G.shape[0] > G.shape[1]
            if transposed:
                X = X.T
            for _ in range(steps):
                A = X @ X.T
                X = a * X + (b * A + c * A @ A) @ X
            return X.T if transposed else X

        def muon_step(W, G, M, lr, momentum=0.95, nesterov=True, weight_decay=0.1, rms_scale=0.2, ns_steps=5):
            # TODO: 动量 → (Nesterov) 正交化 → RMS 对齐 → 解耦衰减 + 更新
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
            a, b, c = coeffs
            X = G / (np.linalg.norm(G) + eps)
            transposed = G.shape[0] > G.shape[1]
            if transposed:
                X = X.T
            for _ in range(steps):
                A = X @ X.T
                X = a * X + (b * A + c * A @ A) @ X
            return X.T if transposed else X

        def muon_step(W, G, M, lr, momentum=0.95, nesterov=True, weight_decay=0.1, rms_scale=0.2, ns_steps=5):
            M = momentum * M + G                              # 1) 动量（新数组，不改输入）
            U = momentum * M + G if nesterov else M           # 2) Nesterov：向前多看一步
            O = newton_schulz(U, steps=ns_steps)              # 3) 正交化：奇异值 → ~1
            O = O * np.sqrt(max(W.shape)) * rms_scale         #    update RMS 对齐到 rms_scale
            W = W * (1.0 - lr * weight_decay) - lr * O        # 4) 解耦权重衰减 + 更新
            return W, M
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：Muon + AdamW 混合优化器的骨架）',
          code: py`
            import torch

            def newton_schulz(G: torch.Tensor, steps=5, eps=1e-7) -> torch.Tensor:
                a, b, c = 3.4445, -4.7750, 2.0315
                X = G.bfloat16()
                X = X / (X.norm() + eps)
                if G.size(0) > G.size(1):
                    X = X.T
                for _ in range(steps):
                    A = X @ X.T
                    X = a * X + (b * A + c * A @ A) @ X
                if G.size(0) > G.size(1):
                    X = X.T
                return X.to(G.dtype)

            class Muon(torch.optim.Optimizer):
                """只给二维隐藏层矩阵用；embedding / lm_head / norm / bias 放进另一个 AdamW"""
                def __init__(self, params, lr=2e-4, momentum=0.95, weight_decay=0.1, rms_scale=0.2):
                    super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rms_scale=rms_scale))

                @torch.no_grad()
                def step(self):
                    for g in self.param_groups:
                        for p in g["params"]:
                            if p.grad is None:
                                continue
                            buf = self.state[p].setdefault("M", torch.zeros_like(p))
                            buf.mul_(g["momentum"]).add_(p.grad)
                            O = newton_schulz(p.grad + g["momentum"] * buf)
                            O = O * (max(p.shape) ** 0.5) * g["rms_scale"]
                            p.mul_(1 - g["lr"] * g["weight_decay"]).add_(O, alpha=-g["lr"])

            # 参数划分示例：
            # muon_params  = [p for n, p in model.named_parameters() if p.ndim == 2 and "embed" not in n and "lm_head" not in n]
            # adamw_params = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "embed" not in n and "lm_head" not in n)]
          `,
        },
      ],
      explanation: md`
        **要点**
        - Nesterov：正交化的对象是 $\mu M_t + G_t$（用**更新后**的 $M_t$），这与 V4 论文 Algorithm 1 一致。Keller 的实现用 EMA 形式的动量（\`lerp\`），二者只差常数倍——而 NS 对整体缩放不敏感，所以正交化结果相同。
        - RMS 对齐在 NS **之后**：NS 输出的尺度与输入无关，只能事后缩放。
        - 衰减项乘的是 $W$ 本身，与 AdamW 的解耦衰减相同。

        **复杂度**：NS 每步 3 次矩阵乘法，$O(\text{steps}\cdot r^2 c)$（$r \le c$ 为两维中的较小 / 较大者）；额外状态只有一个 $M$。
      `,
      complexity: { time: 'O(steps · r²·c)', space: 'O(n·m)' },
      hints: [
        md`先 \`M = momentum * M + G\`（不要用 \`+=\`，否则会改到输入数组）。`,
        md`\`U = momentum * M + G if nesterov else M\`，再 \`O = newton_schulz(U, steps=ns_steps)\`。`,
        md`\`O *= np.sqrt(max(W.shape)) * rms_scale\`；\`W = W * (1 - lr * weight_decay) - lr * O\`。`,
      ],
      setup: REF_NS + '\n' + py`
        def _ref_muon(W, G, M, lr, mu=0.95, nesterov=True, wd=0.1, rms=0.2, steps=5):
            M = mu * M + G
            U = mu * M + G if nesterov else M
            O = _ref_ns(U, steps=steps) * np.sqrt(max(W.shape)) * rms
            return W * (1 - lr * wd) - lr * O, M
      `,
      tests: [
        {
          name: '单步对拍',
          code: py`
            rng = np.random.default_rng(0)
            W, G, M = rng.normal(size=(16, 32)), rng.normal(size=(16, 32)), np.zeros((16, 32))
            Wn, Mn = muon_step(W, G, M, lr=0.01)
            rW, rM = _ref_muon(W, G, M, lr=0.01)
            assert Wn.shape == W.shape and Mn.shape == M.shape, f"形状错误：{Wn.shape}, {Mn.shape}"
            assert np.allclose(Mn, rM), "动量 M 不正确"
            assert np.allclose(Wn, rW, atol=1e-9), f"W 最大误差 {np.abs(Wn - rW).max():.2e}"
          `,
        },
        {
          name: 'update RMS ≈ rms_scale',
          code: py`
            rng = np.random.default_rng(1)
            W, G = rng.normal(size=(32, 96)), rng.normal(size=(32, 96))
            lr, wd = 0.01, 0.1
            Wn, _ = muon_step(W, G, np.zeros_like(W), lr=lr, weight_decay=wd, rms_scale=0.2)
            O = (W * (1 - lr * wd) - Wn) / lr
            rms = np.sqrt((O ** 2).mean())
            assert 0.13 < rms < 0.23, f"update RMS 应接近 0.2（NS 输出奇异值约 0.68–1.13），得到 {rms:.4f}"
          `,
        },
        {
          name: '多步：Nesterov 与非 Nesterov',
          code: py`
            rng = np.random.default_rng(2)
            for nesterov in (True, False):
                W = rng.normal(size=(12, 20)); M = np.zeros_like(W)
                rW, rM = W.copy(), M.copy()
                for t in range(5):
                    G = rng.normal(size=W.shape)
                    W, M = muon_step(W, G, M, lr=0.02, nesterov=nesterov, weight_decay=0.05)
                    rW, rM = _ref_muon(rW, G, rM, lr=0.02, nesterov=nesterov, wd=0.05)
                assert np.allclose(W, rW, atol=1e-9), f"nesterov={nesterov}：5 步后最大误差 {np.abs(W - rW).max():.2e}"
          `,
        },
        {
          name: '不修改输入 + 高矩阵',
          hidden: true,
          code: py`
            rng = np.random.default_rng(3)
            W, G, M = rng.normal(size=(64, 16)), rng.normal(size=(64, 16)), rng.normal(size=(64, 16))
            snap = [a.copy() for a in (W, G, M)]
            Wn, Mn = muon_step(W, G, M, lr=0.01, rms_scale=0.18, ns_steps=5)
            for a, b in zip((W, G, M), snap):
                assert np.array_equal(a, b), "输入数组被原地修改了"
            rW, rM = _ref_muon(W, G, M, lr=0.01, rms=0.18)
            assert Wn.shape == (64, 16), f"形状应为 (64, 16)，得到 {Wn.shape}"
            assert np.allclose(Wn, rW, atol=1e-9), f"高矩阵结果错误，最大误差 {np.abs(Wn - rW).max():.2e}"
          `,
        },
        {
          name: '梯度尺度不影响更新（正交化的尺度不变性）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(4)
            W, G = rng.normal(size=(10, 30)), rng.normal(size=(10, 30))
            W1, _ = muon_step(W, G, np.zeros_like(W), lr=0.01)
            W2, _ = muon_step(W, 1e4 * G, np.zeros_like(W), lr=0.01)
            assert np.allclose(W1, W2, atol=1e-8), "梯度整体放大 1e4 倍后更新应几乎不变"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-zero',
          q: md`Muon 和 ZeRO-1 为什么不好结合？怎么解决？`,
          a: md`
            ZeRO-1 按元素把优化器状态切到各个数据并行 rank，适合 AdamW 这种逐元素的优化器；而 NS 需要**完整矩阵**。Moonlight 的 Distributed Muon：reduce-scatter 得到本地梯度分片后，先在 DP 组内 gather 出完整矩阵做 NS，再只保留本地分片对应的更新，通信量约为 Distributed AdamW 的 1–1.25 倍。DeepSeek-V4 则按“整块矩阵”分配：稠密参数用背包算法把矩阵分到各 rank、限制 ZeRO 并行度并允许冗余计算；MoE 按专家切分；同形状矩阵合并后批量做 NS。
          `,
        },
      ],
    },
    {
      id: 'c-qk-clip',
      title: 'QK-Clip（MLA 版）',
      difficulty: 'hard',
      source: 'Kimi K2 技术报告 · MuonClip',
      timeLimitMin: 25,
      description: md`
        实现 Kimi K2 的逐头 QK-Clip（MLA 版，省略 RoPE 旋转本身——它是线性的正交变换，不影响缩放关系）：

        \`\`\`python
        def qk_clip(X, Wq_c, Wk_c, Wq_r, Wk_r, tau=100.0):
            """返回 (Wq_c_new, Wk_c_new, Wq_r_new, Wk_r_new, S_max)"""
        \`\`\`

        - \`X\`：\`(T, d)\`，一段序列的隐藏状态；
        - \`Wq_c, Wk_c\`：\`(H, d, dc)\`，每个头的内容部分 query / key 投影；
        - \`Wq_r\`：\`(H, d, dr)\`，每个头的 RoPE 部分 query 投影；\`Wk_r\`：\`(d, dr)\`，**所有头共享**的 RoPE key 投影；
        - 第 $h$ 个头的 logit：$\ell_h(i, j) = \big(q^C_{h,i}\cdot k^C_{h,j} + q^R_{h,i}\cdot k^R_j\big)/\sqrt{d_c + d_r}$，其中 $q^C_h = XW^q_{c,h}$ 等；本题**不加因果 mask**，所有 $(i, j)$ 都参与；
        - $S_{\max}^h = \max_{i,j}\ell_h(i,j)$；对 $S_{\max}^h > \tau$ 的头，$\gamma_h = \tau/S_{\max}^h$：\`Wq_c[h]\`、\`Wk_c[h]\` 各乘 $\sqrt{\gamma_h}$，\`Wq_r[h]\` 乘 $\gamma_h$；\`Wk_r\` 不变；其余头不变。
        - 返回**新数组**（不修改输入）以及裁剪前的 \`S_max\`（形状 \`(H,)\`）。

        **自检**：裁剪后重新计算，被裁剪头的最大 logit 恰好等于 $\tau$。
      `,
      starter: py`
        import numpy as np

        def qk_clip(X, Wq_c, Wk_c, Wq_r, Wk_r, tau=100.0):
            # TODO: 1) 逐头计算 logits 与 S_max  2) 对超标的头按 γ 缩放  3) 返回新数组
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def qk_clip(X, Wq_c, Wk_c, Wq_r, Wk_r, tau=100.0):
            H, d, dc = Wq_c.shape
            dr = Wq_r.shape[-1]
            scale = 1.0 / np.sqrt(dc + dr)
            k_r = X @ Wk_r                                         # (T, dr)，所有头共享
            S_max = np.empty(H)
            for h in range(H):
                q_c, k_c, q_r = X @ Wq_c[h], X @ Wk_c[h], X @ Wq_r[h]
                logits = (q_c @ k_c.T + q_r @ k_r.T) * scale       # (T, T)
                S_max[h] = logits.max()

            Wq_c, Wk_c, Wq_r = Wq_c.copy(), Wk_c.copy(), Wq_r.copy()
            for h in range(H):
                if S_max[h] > tau:
                    gamma = tau / S_max[h]
                    Wq_c[h] *= np.sqrt(gamma)                      # 内容部分：q、k 各分一半
                    Wk_c[h] *= np.sqrt(gamma)
                    Wq_r[h] *= gamma                               # RoPE 部分：共享的 k_r 不动，γ 全给 q_r
            return Wq_c, Wk_c, Wq_r, Wk_r.copy(), S_max
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：在优化器 step 之后调用）',
          code: py`
            import torch

            @torch.no_grad()
            def qk_clip_(attn, s_max: torch.Tensor, tau: float = 100.0):
                """attn.wq_c / wk_c: (H, d, dc)，attn.wq_r: (H, d, dr)，s_max: 前向时记录的每头最大 logit"""
                gamma = torch.clamp(tau / s_max, max=1.0)                  # (H,)，未超标的头为 1
                attn.wq_c.mul_(gamma.sqrt()[:, None, None])
                attn.wk_c.mul_(gamma.sqrt()[:, None, None])
                attn.wq_r.mul_(gamma[:, None, None])                       # 共享的 wk_r 不动

            # 训练循环：loss.backward(); optimizer.step(); qk_clip_(layer.attn, layer.attn.s_max)
          `,
        },
      ],
      explanation: md`
        **为什么这样分配缩放**：$\ell = (q^C\cdot k^C + q^R\cdot k^R)/\sqrt{d}$。内容部分 $q^C, k^C$ 都是头独有的，各乘 $\sqrt\gamma$ 使乘积缩小 $\gamma$；RoPE 部分的 $k^R$ 被所有头共享，缩放它会波及其它头，所以把 $\gamma$ 全部放到头独有的 $q^R$ 上。两部分都缩小 $\gamma$ 倍，最大 logit 恰为 $\tau$。

        **K2 的细节**：$S_{\max}$ 在前向时顺带记录（取 batch 内最大值）；裁剪在优化器更新之后执行，不改变当前步的前向反向；只对超标的少数头生效。

        **复杂度**：$O(H T^2 (d_c + d_r) + H T d (d_c + d_r))$，本质上是一次注意力分数的计算。
      `,
      complexity: { time: 'O(H·T²·(dc+dr))', space: 'O(T²)' },
      hints: [
        md`共享的 \`k_r = X @ Wk_r\` 只算一次；每个头 \`logits = (q_c @ k_c.T + q_r @ k_r.T) / sqrt(dc + dr)\`。`,
        md`先 \`.copy()\` 再原地乘，避免修改输入。`,
        md`\`gamma = tau / S_max[h]\`：\`Wq_c[h]\`、\`Wk_c[h]\` 乘 \`sqrt(gamma)\`，\`Wq_r[h]\` 乘 \`gamma\`。`,
      ],
      setup: py`
        def _logits_max(X, Wq_c, Wk_c, Wq_r, Wk_r):
            H, d, dc = Wq_c.shape
            dr = Wq_r.shape[-1]
            k_r = X @ Wk_r
            out = []
            for h in range(H):
                L = ((X @ Wq_c[h]) @ (X @ Wk_c[h]).T + (X @ Wq_r[h]) @ k_r.T) / np.sqrt(dc + dr)
                out.append(L.max())
            return np.array(out)

        def _make(seed, T=6, d=8, H=4, dc=4, dr=2):
            rng = np.random.default_rng(seed)
            s = np.array([0.4, 1.0, 2.5, 4.0])[:H, None, None]           # 让不同头的 logit 量级不同
            Wq_c = rng.normal(size=(H, d, dc)) * s
            Wk_c = rng.normal(size=(H, d, dc)) * s
            Wq_r = rng.normal(size=(H, d, dr)) * s
            Wk_r = rng.normal(size=(d, dr))
            X = rng.normal(size=(T, d)) * 2
            return X, Wq_c, Wk_c, Wq_r, Wk_r
      `,
      tests: [
        {
          name: 'S_max 与形状',
          code: py`
            X, Wq_c, Wk_c, Wq_r, Wk_r = _make(0)
            out = qk_clip(X, Wq_c, Wk_c, Wq_r, Wk_r, tau=1e9)
            assert len(out) == 5, f"应返回 5 个值，得到 {len(out)}"
            exp = _logits_max(X, Wq_c, Wk_c, Wq_r, Wk_r)
            assert out[4].shape == (4,), f"S_max 形状应为 (4,)，得到 {out[4].shape}"
            assert np.allclose(out[4], exp), f"S_max 期望 {exp}，得到 {out[4]}"
            assert all(np.array_equal(a, b) for a, b in zip(out[:4], (Wq_c, Wk_c, Wq_r, Wk_r))), "τ 很大时不应修改任何权重"
          `,
        },
        {
          name: '裁剪后最大 logit 恰为 τ',
          code: py`
            X, Wq_c, Wk_c, Wq_r, Wk_r = _make(1)
            S = _logits_max(X, Wq_c, Wk_c, Wq_r, Wk_r)
            tau = float(np.median(S))
            nq, nk, nqr, nkr, _ = qk_clip(X, Wq_c, Wk_c, Wq_r, Wk_r, tau=tau)
            S2 = _logits_max(X, nq, nk, nqr, nkr)
            exp = np.minimum(S, tau)
            assert np.allclose(S2, exp), f"裁剪后每头最大 logit 期望 {exp}，得到 {S2}"
          `,
        },
        {
          name: '未超标的头保持不变',
          code: py`
            X, Wq_c, Wk_c, Wq_r, Wk_r = _make(2)
            S = _logits_max(X, Wq_c, Wk_c, Wq_r, Wk_r)
            tau = float(np.sort(S)[1] + 1e-6)
            nq, nk, nqr, _, _ = qk_clip(X, Wq_c, Wk_c, Wq_r, Wk_r, tau=tau)
            for h in range(4):
                if S[h] <= tau:
                    assert np.array_equal(nq[h], Wq_c[h]) and np.array_equal(nk[h], Wk_c[h]) and np.array_equal(nqr[h], Wq_r[h]), f"head {h}（S_max={S[h]:.2f} ≤ τ）不应被修改"
          `,
        },
        {
          name: '缩放因子分配（√γ / √γ / γ / 1）',
          hidden: true,
          code: py`
            X, Wq_c, Wk_c, Wq_r, Wk_r = _make(3)
            S = _logits_max(X, Wq_c, Wk_c, Wq_r, Wk_r)
            tau = float(S.min() * 0.5) if S.min() > 0 else 0.1
            nq, nk, nqr, nkr, _ = qk_clip(X, Wq_c, Wk_c, Wq_r, Wk_r, tau=tau)
            assert np.array_equal(nkr, Wk_r), "共享的 Wk_r 不能被修改"
            for h in range(4):
                if S[h] > tau:
                    g = tau / S[h]
                    assert np.allclose(nq[h], Wq_c[h] * np.sqrt(g)), f"head {h}：Wq_c 应乘 √γ"
                    assert np.allclose(nk[h], Wk_c[h] * np.sqrt(g)), f"head {h}：Wk_c 应乘 √γ"
                    assert np.allclose(nqr[h], Wq_r[h] * g), f"head {h}：Wq_r 应乘 γ（共享的 k_r 不动，γ 全部给 q_r）"
          `,
        },
        {
          name: '不修改输入',
          hidden: true,
          code: py`
            X, Wq_c, Wk_c, Wq_r, Wk_r = _make(4)
            snap = [a.copy() for a in (X, Wq_c, Wk_c, Wq_r, Wk_r)]
            qk_clip(X, Wq_c, Wk_c, Wq_r, Wk_r, tau=1.0)
            for a, b in zip((X, Wq_c, Wk_c, Wq_r, Wk_r), snap):
                assert np.array_equal(a, b), "输入数组被原地修改了"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-global',
          q: md`为什么 QK-Clip 要逐头做，而不是用全局最大 logit 统一缩放所有头？`,
          a: md`
            K2 报告观察到只有少数头的 logit 会爆炸。全局缩放会把大量正常的头也一起压小，改变它们的注意力分布（变得更平），属于不必要的干预；逐头裁剪只动超标的头，对其余头零影响。同理，MLA 中把 $\gamma$ 只施加在头独有的 $q^R$ 上，也是为了不让共享的 $k^R$ 波及其它头。
          `,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-muon',
      freq: 3,
      q: md`Muon 优化器是什么？它和 AdamW 有什么本质区别？`,
      a: md`
        **结论**：Muon 对二维隐藏层矩阵的动量做正交化：$O = \mathrm{NS}(\mu M + G) \approx UV^\top$，再缩放后作为更新；embedding、LM head、一维参数仍用 AdamW。

        **原因**：AdamW 逐元素自适应，看不到矩阵结构；Transformer 的矩阵更新通常近似低秩、条件数极高，少数方向主导，稀有方向几乎不更新。正交化把所有奇异值置为 1，让每个奇异方向都走同样的步长。它用 Newton-Schulz 五次迭代（系数 3.4445, −4.7750, 2.0315，5 步、bf16）代替 SVD，额外开销通常 < 1%。

        **延伸**：Moonlight 指出扩展到大模型的两个关键是**权重衰减**和 **update RMS 对齐**（$0.2\sqrt{\max(A,B)}$，从而复用 AdamW 超参数），并报告约 2× 计算效率；只有一个动量缓冲区，状态显存约为 AdamW 的一半；与 ZeRO 的结合需要特殊处理（需要完整矩阵）。
      `,
    },
    {
      id: 'qa-ns',
      freq: 2,
      q: md`Newton-Schulz 迭代为什么能正交化？五次系数为什么不收敛到 1？DeepSeek-V4 怎么改的？`,
      a: md`
        **结论**：奇数次矩阵多项式只作用在奇异值上：$\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5$，反复迭代把 $(0, 1]$ 内的奇异值推到 1 附近，$U, V$ 不变。

        **细节**：先除以 Frobenius 范数保证 $\sigma \le 1$；宽矩阵算 $XX^\top$ 更省。Keller 的五次系数以“最大化 $\varphi'(0)$”为目标（小奇异值增长最快），只要求结果落在约 $[0.7, 1.3]$；因为 $\varphi(1) = 0.701$，1 不是不动点，奇异值在约 $[0.68, 1.13]$ 间振荡——实验表明这不影响训练。

        **V4**：混合 NS 共 10 步，前 8 步用五次系数快速收敛，后 2 步用 $(2, -1.5, 0.5)$（$\varphi(1)=1$、$\varphi'(1)=0$）把奇异值精确稳定在 1。
      `,
    },
    {
      id: 'qa-clip',
      freq: 2,
      q: md`MuonClip / QK-Clip 解决什么问题？为什么 DeepSeek-V4 不需要它？`,
      a: md`
        **问题**：用 Muon 训练时 attention logit 增长更快，K2 在中等规模实验中观察到最大 logit 超过 1000，导致训练不稳定。soft-capping 是事后截断，$q\cdot k$ 本身仍在增长；QK-Norm 无法用于 MLA（推理时 Key 不被完整构造）。

        **QK-Clip**：优化器更新后，对最大 logit $S_{\max}^h > \tau$ 的头，令 $\gamma = \tau/S_{\max}^h$，$W_q, W_k$ 各乘 $\sqrt\gamma$；MLA 中头独有的 $q^C, k^C$ 各乘 $\sqrt\gamma$、$q^R$ 乘 $\gamma$，共享的 $k^R$ 不动。K2 取 $\tau = 100$，15.5T token 预训练无 loss spike。

        **V4**：注意力在点积前对每个 query 头与压缩 KV 条目做 RMSNorm，logit 天然有界，论文明确因此不使用 QK-Clip——从架构侧解决，而不是在优化器侧补救。
      `,
    },
    {
      id: 'qa-which',
      freq: 2,
      q: md`为什么 Muon 只用于隐藏层矩阵？embedding 和 LM head 为什么还用 AdamW？`,
      a: md`
        **结论**：正交化只对“把一个向量空间映射到另一个向量空间”的矩阵有意义，而 embedding / LM head 的每一行对应一个离散 token，本质上是查表。

        **原因**：embedding 每步只有出现过的 token 的行有梯度，梯度极度稀疏；对整张表做正交化会给没出现的 token 的行也分配同样大小的更新，这没有道理。LM head 也有类似问题，且输出 logits 的尺度敏感。Keller 的建议、Moonlight / K2 / V4 / GLM-4.5 的做法都把它们（以及所有一维参数）交给 AdamW。

        **延伸**：Q、K、V 要分开正交化；MoE 的每个专家各自正交化；卷积核展平成二维后可以用 Muon。
      `,
    },
  ],

  cheatsheet: md`
    **Muon 一步（V4 / Moonlight 风格）**
    \`\`\`python
    M = mu * M + G                              # mu = 0.95
    O = newton_schulz(mu * M + G)               # Nesterov
    O = O * np.sqrt(max(W.shape)) * 0.2         # V4: 0.18
    W = W * (1 - lr * wd) - lr * O              # wd = 0.1
    \`\`\`
    **NS**：\`X = G/‖G‖_F\`（高矩阵先转置），5 步 \`A = X@X.T; X = a*X + (b*A + c*A@A)@X\`

    | 数字 | |
    |---|---|
    | 五次系数 | (3.4445, −4.7750, 2.0315)，φ(1)=0.701，收敛带 ≈ [0.68, 1.13] |
    | V4 混合 NS | 8 步五次 + 2 步 (2, −1.5, 0.5)，共 10 步 |
    | 正交矩阵 RMS | $1/\sqrt{\max(A,B)}$；AdamW 约 0.2–0.4 |
    | QK-Clip | γ = τ/S_max，τ = 100；q^C、k^C ×√γ，q^R ×γ，k^R 不动 |

    - 只用于 2D 隐藏层矩阵；embedding / LM head / norm / bias → AdamW
    - Q、K、V 分开正交化；MoE 专家各自独立
    - 一个动量缓冲区 → 状态约为 AdamW 的一半
    - K2 用 QK-Clip（MLA 无法 QK-Norm）；V4 对 Q 与 KV 做 RMSNorm → 不用 QK-Clip
  `,
})
