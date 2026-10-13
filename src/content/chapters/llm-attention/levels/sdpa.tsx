/**
 * L3-1 缩放点积注意力（SDPA）：软寻址直觉、√d_k 的方差推导、因果 / padding 掩码与 NaN 陷阱。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { AttentionHeatmapViz } from '../viz/AttentionHeatmapViz'
import { ScaleSaturationViz } from '../viz/ScaleSaturationViz'
import { MaskViz } from '../viz/MaskViz'

export default defineLevel({
  learn: [
    {
      id: 'qkv',
      title: '注意力在做什么：可微的「软寻址」',
      body: md`
        把注意力想成一次**字典查询**：每个位置发出一个 query（我在找什么），所有位置各自挂出一个 key（我是什么）和一个 value（我携带的内容）。
        硬查询只返回匹配的那一个 value；注意力返回**所有 value 的加权平均**，权重由 query 与各 key 的相似度经 softmax 得到——这一步可微，所以能端到端训练。

        $$
        \text{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
        $$

        | 张量 | 形状 | 含义 |
        |---|---|---|
        | $Q$ | $(L, d_k)$ | $L$ 个 query |
        | $K$ | $(S, d_k)$ | $S$ 个 key（自注意力时 $S=L$） |
        | $V$ | $(S, d_v)$ | 与 key 一一对应的 value |
        | $QK^\top$ | $(L, S)$ | 第 $i$ 行 = query $i$ 对所有 key 的打分 |
        | 输出 | $(L, d_v)$ | 每个 query 得到一个 value 的凸组合 |

        $M$ 是加性掩码（允许的位置为 0，不允许的为 $-\infty$）。softmax **按行**（沿 key 维，\`axis=-1\`）归一化。
        下图的 Q、K 是手工设计的 4 维向量（维度含义：动物 / 物体 / 动作 / 功能词），权重由公式实算。点「它」这一行，看它把注意力放在哪个名词上。
      `,
      viz: AttentionHeatmapViz,
      after: md`
        > [!NOTE] 两个容易忽略的事实
        > 1. 输出是 value 的**凸组合**（权重非负、和为 1），所以单层注意力不会「放大」value，只能混合——非线性和放大由 FFN 负责。
        > 2. 自注意力本身对输入顺序是**置换等变**的：打乱 token 顺序，输出只是跟着打乱。位置信息必须靠位置编码（RoPE 等）或因果掩码引入。
      `,
      checkpoint: {
        id: 'cp-qkv',
        type: 'single',
        prompt: md`$Q \in \mathbb{R}^{L\times d_k}$、$K \in \mathbb{R}^{S\times d_k}$、$V \in \mathbb{R}^{S\times d_v}$，softmax 应该沿哪个维度做？`,
        options: ['沿 query 维（每一列和为 1）', '沿 key 维（每一行和为 1）', '对整个 $L\\times S$ 矩阵做', '沿 $d_k$ 维'],
        answer: 1,
        explanation: md`每个 query 要在**所有 key** 上分配权重，所以对 $(L,S)$ 矩阵的每一行（\`axis=-1\`）做 softmax。沿列归一化是经典 bug：输出不再是 value 的凸组合，因果模型还会把未来信息混进来。`,
      },
    },
    {
      id: 'scale',
      title: '为什么除以 √d_k：一行方差推导',
      body: md`
        假设 $q, k \in \mathbb{R}^{d_k}$ 的各分量相互独立、均值 0、方差 1（初始化附近大致如此）。则

        $$
        \mathbb{E}[q\cdot k] = \sum_{i=1}^{d_k}\mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0,\qquad
        \operatorname{Var}(q\cdot k) = \sum_{i=1}^{d_k}\operatorname{Var}(q_i k_i) = \sum_{i=1}^{d_k}\mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = d_k
        $$

        即点积的标准差是 $\sqrt{d_k}$。$d_k=128$ 时 logits 的典型差距有十几，softmax 会把几乎全部概率给最大的那个——**饱和**。
        softmax 的雅可比是 $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$，当 $p$ 接近 one-hot 时**每一项都趋于 0**，梯度几乎传不回 $Q, K$。

        除以 $\sqrt{d_k}$ 后 $\operatorname{Var}\big(q\cdot k/\sqrt{d_k}\big) = 1$，与维度无关。原论文（Vaswani et al., 2017）的说法正是：
        $d_k$ 较大时点积数值变大，把 softmax 推到梯度极小的区域。

        拖动下图的 $d_k$，先关掉缩放观察饱和，再打开对比。
      `,
      viz: ScaleSaturationViz,
      after: md`
        > [!INTERVIEW] 追问：为什么是 $\sqrt{d_k}$ 而不是 $d_k$？
        > 目标是让 logits 的**标准差**为常数，标准差是 $\sqrt{d_k}$，所以除以 $\sqrt{d_k}$。除以 $d_k$ 会让方差变成 $1/d_k$，大维度时 logits 几乎全为 0，softmax 退化成均匀分布，同样学不到东西。
        > 延伸：μP（Tensor Programs V）为了让超参数随宽度迁移，改用 $1/d$ 缩放；QK-Norm（Qwen3、OLMo 2、Gemma 3 等使用）直接对 q、k 做归一化来控制 logits 尺度。

        > [!WARN] 缩放因子用的是**单头**维度
        > 多头注意力中缩放用 $\sqrt{d_h}$（每个头的维度，例如 128），不是 $\sqrt{d_{\text{model}}}$。MLA 这类 q/k 维度与 v 不同的结构，缩放用的是 **q·k 的点积维度**（DeepSeek-V3 为 $\sqrt{128+64}$）。
      `,
      checkpoint: {
        id: 'cp-scale',
        type: 'numeric',
        prompt: md`$q, k$ 各分量独立、均值 0、方差 1，$d_k = 64$。**不缩放**时 $q\cdot k$ 的标准差是多少？`,
        answer: 8,
        tolerance: 0,
        explanation: md`$\operatorname{Var}(q\cdot k) = d_k = 64$，标准差 $\sqrt{64} = 8$。除以 $\sqrt{d_k} = 8$ 后标准差回到 1。`,
      },
    },
    {
      id: 'mask',
      title: '因果掩码与 padding 掩码：在 softmax 之前加 −∞',
      body: md`
        **因果掩码**（causal mask）：语言模型训练时一次性喂入整句，位置 $i$ 只能看 $j \le i$，否则就是偷看答案。实现为下三角布尔矩阵 \`np.tril(np.ones((T, T), bool))\`。

        **padding 掩码**：一个 batch 里序列长短不一，短的要补 \`<pad>\`。pad 不能被当作 key 看到，所以按 **key 维**遮掉 pad 列，形状通常是 $(B, 1, 1, S)$，广播到 $(B, h, L, S)$。

        两种掩码取**逻辑与**后，统一在 softmax **之前**把不允许的位置设成 $-\infty$（或很大的负数）：$e^{-\infty} = 0$，剩余位置自动重新归一化。
        **在 softmax 之后乘 0 是错的**——那一行的和不再是 1。

        **左 padding vs 右 padding**：训练时常用右 padding；批量**生成**时常用左 padding，这样每条序列的最后一个 token 都对齐在位置 $T-1$，新 token 可以整齐地追加在末尾。
        但左 padding 带来两个坑：位置编号必须从第一个真实 token 开始算（HF 用 \`cumsum(attention_mask) - 1\`）；pad query 行可能**整行被遮**。
      `,
      viz: MaskViz,
      after: md`
        > [!WARN] 整行被遮 → NaN → 传染
        > 左 padding + 因果时，开头的 pad query 一个可见位置都没有，softmax 变成 $0/0$。更糟的是：下一层里真实 token 对这些 pad 位置的权重虽然是 0，
        > 但 $0 \times \text{NaN} = \text{NaN}$，整个 batch 都会被污染。常见对策：掩码值用 \`torch.finfo(dtype).min\` 而不是 $-\infty$（HF 的做法）；
        > 或检测全遮行并把输出置 0；或让该行至少能看到自己。

        > [!NOTE] PyTorch 的两套相反约定
        > \`F.scaled_dot_product_attention\` 的布尔 \`attn_mask\`：**True = 参与注意力**；\`nn.MultiheadAttention\` 的 \`key_padding_mask\` / 布尔 \`attn_mask\`：**True = 屏蔽**。
        > 此外 SDPA 的 \`is_causal=True\` 在 $L \ne S$ 时是**左上角对齐**的下三角，且不能和 \`attn_mask\` 同时传（会报错）。
      `,
      checkpoint: {
        id: 'cp-mask',
        type: 'judge',
        prompt: md`先算 \`w = softmax(scores)\`，再执行 \`w = w * causal_mask\` 把未来位置置 0，与在 softmax 之前加 $-\infty$ 的结果等价。`,
        answer: false,
        explanation: md`不等价。softmax 之后再置 0，每行剩余权重之和小于 1，而且分母里混入了未来位置的 $e^{s_{ij}}$——未来信息仍然影响了输出的尺度。必须在 softmax **之前**屏蔽，让分母只包含允许的位置。`,
      },
    },
    {
      id: 'impl',
      title: '工程细节：数值稳定、复杂度与 L ≠ S',
      body: md`
        **数值稳定**：softmax 前先减去每行最大值 $\max_j s_{ij}$，避免 $e^{s}$ 溢出（FP16 最大约 65504，$e^{12}$ 就超了）。减常数不改变 softmax 结果。

        **复杂度**：$QK^\top$ 和 $PV$ 各需 $2LSd$ FLOPs，时间 $O(LSd)$；朴素实现要把 $(L, S)$ 的分数矩阵整个写出来，显存 $O(LS)$。
        $B=1$、32 个头、$L=S=8192$、FP16 时，仅分数矩阵就是 $32 \times 8192^2 \times 2$ 字节 $= 4$ GiB——这就是 FlashAttention 要解决的问题（本章第 5 关）。

        **$L \ne S$ 时的因果掩码**：带 KV cache 解码或 chunked prefill 时，query 只是序列的**最后** $L$ 个位置，key 却有 $S$ 个。
        正确的规则是**右下角对齐**：query $i$ 可以看 key $j \le i + (S - L)$，即 \`np.tril(ones((L, S)), k=S-L)\`。FlashAttention 从 2.1 版起采用右下角对齐；
        PyTorch SDPA 的 \`is_causal\` 却是左上角对齐，解码时直接传 \`is_causal=True\` 会出错（单 token 解码时只能看到第 0 个 key）。
      `,
      checkpoint: {
        id: 'cp-impl',
        type: 'fill',
        prompt: md`解码时 query 长度 $L=1$，缓存后 key 长度 $S=6$。右下角对齐的因果掩码中，这个 query 能看到 ____(1)____ 个 key；若误用左上角对齐的 \`tril(ones((1, 6)))\`，只能看到 ____(2)____ 个。`,
        blanks: [
          { label: '(1)', answers: ['6'], numeric: true },
          { label: '(2)', answers: ['1'], numeric: true },
        ],
        explanation: md`右下角对齐：$j \le 0 + (6-1) = 5$，6 个 key 全可见（新 token 是序列最后一个，本来就能看全部历史）。左上角对齐的 \`tril\` 第 0 行只有 $j=0$ 为 True，等于只看第一个 token。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-shape',
      type: 'single',
      difficulty: 1,
      prompt: md`多头自注意力中 \`q, k\` 的形状都是 \`(B, h, T, d_h)\`，\`scores = q @ k.transpose(-1, -2)\` 的形状是？`,
      options: ['`(B, h, T, T)`', '`(B, h, d_h, d_h)`', '`(B, T, T)`', '`(B, h, T, d_h)`'],
      answer: 0,
      explanation: md`矩阵乘在最后两维进行：$(T, d_h) \times (d_h, T) = (T, T)$，前面的 \`(B, h)\` 作为批维保留。\`(d_h, d_h)\` 是把转置写反（\`q.transpose @ k\`）的结果，那是「维度之间的相关」而不是 token 之间的注意力。`,
    },
    {
      id: 'q-why-scale',
      type: 'single',
      difficulty: 2,
      prompt: md`除以 $\sqrt{d_k}$ 的**根本**目的是：`,
      options: [
        '让 softmax 的输出之和为 1',
        '让 logits 的方差不随 $d_k$ 增长，避免 softmax 饱和导致梯度消失',
        '防止 $QK^\\top$ 在 FP32 下溢出',
        '让注意力对序列长度归一化',
      ],
      answer: 1,
      explanation: md`softmax 本身就保证和为 1（A 错）。FP32 在 $d_k=128$ 时不会溢出，数值稳定靠的是「减最大值」（C 错）。缩放与序列长度无关（D 错）。核心是 $\operatorname{Var}(q\cdot k)=d_k$，不缩放时 logits 尺度随维度增大，softmax 饱和、雅可比 $p_i(\delta_{ij}-p_j)$ 趋于 0。`,
    },
    {
      id: 'q-saturate',
      type: 'multi',
      difficulty: 2,
      prompt: md`$d_k = 1024$ 且**不做缩放**，训练初期会出现哪些现象？（多选）`,
      options: [
        'logits 标准差约为 32',
        '注意力分布接近 one-hot',
        '回传到 $Q, K$ 投影的梯度非常小',
        'softmax 输出出现负数',
        '注意力分布接近均匀分布',
      ],
      answer: [0, 1, 2],
      explanation: md`标准差 $\sqrt{1024}=32$；相邻 logits 的差动辄几十，$e^{30}$ 量级的差距让最大项吃掉几乎全部概率；one-hot 处雅可比各项趋于 0，梯度消失。softmax 输出永远非负（D 错）；接近均匀是**除以 $d_k$**（过度缩放）时的现象（E 错）。`,
    },
    {
      id: 'q-std',
      type: 'fill',
      difficulty: 1,
      prompt: md`$d_k = 128$，各分量独立、方差 1。不缩放时 $q\cdot k$ 的标准差约为 ____(1)____；缩放因子 $1/\sqrt{d_k}$ 约为 ____(2)____（保留 4 位小数）。`,
      blanks: [
        { label: '(1)', answers: ['11.3137'], numeric: true, tolerance: 0.01 },
        { label: '(2)', answers: ['0.0884'], numeric: true, tolerance: 0.01 },
      ],
      explanation: md`$\sqrt{128} \approx 11.3137$，$1/\sqrt{128} \approx 0.0884$。主流大模型的 head_dim 多为 128，所以面试中这两个数值得记住。`,
    },
    {
      id: 'q-mem',
      type: 'numeric',
      difficulty: 2,
      prompt: md`朴素注意力需要显式存储分数矩阵。$B=1$、32 个头、$L=S=8192$、FP16（2 字节）时，**单层**的分数矩阵占多少 GiB（$1\text{ GiB}=2^{30}$ 字节）？`,
      answer: 4,
      tolerance: 0.01,
      unit: 'GiB',
      explanation: md`$32 \times 8192^2 \times 2 = 32 \times 2^{26} \times 2 = 2^{32}$ 字节 $= 4$ GiB。训练时 softmax 后的 $P$ 还要为反向保存一份，长序列下这才是显存大头——FlashAttention 通过分块 + 重计算把它降到 $O(L)$。`,
    },
    {
      id: 'q-after-softmax',
      type: 'judge',
      difficulty: 1,
      prompt: md`因为 $e^{-\infty}=0$，被掩码的位置在 softmax 之后权重**恰好**为 0，其余位置的权重之和仍然为 1。`,
      answer: true,
      explanation: md`正确。掩码在 softmax 之前加 $-\infty$，分子为 0、分母只累加未被遮的项，所以剩余位置自动重新归一化。前提是这一行至少有一个未被遮的位置，否则是 $0/0$。`,
    },
    {
      id: 'q-finfo',
      type: 'judge',
      difficulty: 2,
      prompt: md`HF Transformers 常用 \`torch.finfo(dtype).min\`（一个有限的极小值）而不是 \`-inf\` 作为掩码值，一个重要原因是：当某一行被**全部**遮住时，softmax 仍能得到有限值，而不是 NaN。`,
      answer: true,
      explanation: md`正确。全为 $-\infty$ 时 softmax 是 $0/0=\text{NaN}$；全为同一个有限极小值时，减去行最大值后全为 0，得到均匀分布——数值无意义但有限，而且这一行（pad query）的输出最后会被丢弃。注意不要再在它上面加大负数，否则 FP16/BF16 下可能溢出成 $-\infty$。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 1,
      prompt: md`把带掩码的 SDPA 计算步骤按正确顺序排列：`,
      items: [
        '`scores = q @ k.T`',
        '`scores = scores / sqrt(d_k)`',
        '`scores = where(mask, scores, -inf)`',
        '`scores = scores - scores.max(-1, keepdims=True)`',
        '`p = exp(scores) / exp(scores).sum(-1, keepdims=True)`',
        '`out = p @ v`',
      ],
      explanation: md`缩放和加掩码的先后可以交换（$-\infty$ 缩放后仍是 $-\infty$），但掩码必须在 softmax 之前；减最大值要在掩码之后（否则最大值可能取自被遮的位置，虽不影响正确性但可能让可见位置全部下溢为 0）；最后才乘 $V$。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把掩码类型与其作用配对：`,
      pairs: [
        ['因果掩码（下三角）', '防止位置 i 看到 j > i 的未来 token'],
        ['padding 掩码（按 key 列）', '让真实 token 忽略补齐用的 pad'],
        ['滑动窗口掩码', '只看最近 W 个 token，降低长序列开销'],
        ['Prefix-LM 掩码', '前缀内部双向可见，生成部分因果'],
      ],
      explanation: md`四种掩码都只是 $(L, S)$ 上不同形状的布尔矩阵，统一在 softmax 前加 $-\infty$。滑动窗口（Mistral、Gemma 等）在「高效注意力」章节详讲；Prefix-LM 见于 T5/UL2 系列的非因果前缀。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`序列为 \`[A, B, C, <pad>, <pad>]\`（右 padding），同时使用**因果掩码 + padding 掩码**。点出所有被遮（设为 $-\infty$）的位置（行 = query，列 = key）。`,
      items: Array.from({ length: 25 }, (_, n) => `${Math.floor(n / 5)},${n % 5}`),
      cols: 5,
      rowLabels: ['q:A', 'q:B', 'q:C', 'q:pad', 'q:pad'],
      colLabels: ['A', 'B', 'C', 'pad', 'pad'],
      answer: [1, 2, 3, 4, 7, 8, 9, 13, 14, 18, 19, 23, 24],
      explanation: md`被遮条件：$j > i$（未来）**或** $j$ 是 pad。真实 token 行 A、B、C 是普通下三角；两个 pad query 行只能看 A、B、C 三列（$j \le i$ 且不是 pad），输出有限，最终被 loss mask 丢弃。注意第 3 行的对角线 (3,3) 也被遮——它自己是 pad。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的注意力实现有 **2 处** bug，点出它们所在的行：`,
      code: py`
        import numpy as np

        def attention(q, k, v, causal=False):
            d = q.shape[-1]
            scores = q @ k.swapaxes(-1, -2) / d
            if causal:
                T = q.shape[-2]
                mask = np.tril(np.ones((T, T), dtype=bool))
                scores = np.where(mask, scores, -np.inf)
            scores = scores - scores.max(axis=-1, keepdims=True)
            w = np.exp(scores)
            w = w / w.sum(axis=-2, keepdims=True)
            return w @ v
      `,
      bugLines: [5, 12],
      explanation: md`第 5 行应除以 \`np.sqrt(d)\`：除以 $d$ 会过度缩放，logits 方差变为 $1/d$，注意力趋于均匀。第 12 行应沿 \`axis=-1\`（key 维）归一化：沿 \`axis=-2\` 会让每**列**和为 1，输出不再是 value 的凸组合。第 10 行减最大值用的是 \`axis=-1\`，是对的。`,
    },
    {
      id: 'q-left-pad',
      type: 'single',
      difficulty: 3,
      prompt: md`批量生成时 batch 用**左 padding**，第 0 条序列为 \`[<pad>, <pad>, 我, 爱]\`。用「因果 & 非 pad」作为掩码直接跑朴素 SDPA（掩码值 $-\infty$），会发生什么？`,
      options: [
        '一切正常，pad 行的输出被丢弃即可',
        '前两行整行被遮，softmax 产生 NaN；下一层真实 token 对这些位置的权重是 0，但 0 × NaN 仍为 NaN，污染整条序列',
        '只有 pad 行的输出是 NaN，真实 token 的输出不受影响',
        '会报错，因为掩码形状不匹配',
      ],
      answer: 1,
      explanation: md`位置 0 只能看 $j \le 0$，而 $j=0$ 是 pad，整行 $-\infty$ → NaN；位置 1 同理。C 的错误在于忽略了下一层：真实 token 在 $P V$ 中要和 pad 位置的 value（已是 NaN）相乘，$0 \times \text{NaN} = \text{NaN}$。解决：用 \`finfo.min\` 作掩码值、全遮行置 0，或让 pad query 至少看自己。`,
    },
  ],

  guided: [
    {
      id: 'g-sdpa',
      title: '一步步写出带掩码的 SDPA',
      intro: md`
        实现 \`attention(q, k, v, mask=None)\`：
        - \`q\`: \`(..., L, d)\`，\`k\`: \`(..., S, d)\`，\`v\`: \`(..., S, dv)\`，前面的维度（batch、head）任意；
        - \`mask\`: 布尔数组，可广播到 \`(..., L, S)\`，**True 表示可以看**（与 PyTorch SDPA 一致）；
        - 返回 \`(..., L, dv)\`。本题不需要处理「整行被遮」。
      `,
      template: py`
        import numpy as np

        def attention(q, k, v, mask=None):
            d = q.shape[-1]
            scores = {{1}}
            if mask is not None:
                scores = {{2}}
            scores = {{3}}
            w = np.exp(scores)
            w = {{4}}
            return {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`计算缩放后的打分 $QK^\top/\sqrt{d}$。注意 \`k\` 可能带批维，转置要用 \`swapaxes(-1, -2)\` 而不是 \`.T\`。`,
          answer: 'q @ np.swapaxes(k, -1, -2) / np.sqrt(d)',
          accept: ['q @ k.swapaxes(-1, -2) / np.sqrt(d)', 'q @ np.swapaxes(k, -2, -1) / np.sqrt(d)', 'np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(d)', 'q @ k.swapaxes(-1, -2) / d ** 0.5'],
          hint: '`.T` 会把所有维度反转，(B, h, S, d) 会变成 (d, S, h, B)。',
          explain: md`对 4 维数组，\`k.T\` 的形状是 \`(d, S, h, B)\`，矩阵乘会报错或广播出错误结果。只交换最后两维才对。`,
        },
        {
          blank: 2,
          instruction: md`把 \`mask\` 为 False 的位置设为 $-\infty$（广播到 scores 的形状）。`,
          answer: 'np.where(mask, scores, -np.inf)',
          accept: ["np.where(mask, scores, float('-inf'))", 'np.where(mask, scores, -1e9)'],
          explain: md`\`np.where\` 会自动广播 \`(B, 1, 1, S)\` 这类 padding 掩码。用 \`-1e9\` 也可以，但 FP16 下 \`-1e9\` 会溢出成 $-\infty$。`,
        },
        {
          blank: 3,
          instruction: md`数值稳定：每行减去该行最大值。`,
          answer: 'scores - scores.max(axis=-1, keepdims=True)',
          accept: ['scores - np.max(scores, axis=-1, keepdims=True)', 'scores - scores.max(-1, keepdims=True)'],
          explain: md`\`keepdims=True\` 让最大值保持 \`(..., L, 1)\` 的形状以便广播。忘了它会把 \`(..., L)\` 和 \`(..., L, S)\` 按最后一维对齐，结果错位（$L \ne S$ 时直接报错，$L = S$ 时悄悄算错）。`,
        },
        {
          blank: 4,
          instruction: md`沿 key 维归一化，得到注意力权重。`,
          answer: 'w / w.sum(axis=-1, keepdims=True)',
          accept: ['w / np.sum(w, axis=-1, keepdims=True)', 'w / w.sum(-1, keepdims=True)'],
          explain: md`每个 query 的权重和为 1。被遮位置 $e^{-\infty}=0$，不进入分母。`,
        },
        {
          blank: 5,
          instruction: md`用权重对 value 加权求和。`,
          answer: 'w @ v',
          accept: ['np.matmul(w, v)'],
          explain: md`$(…, L, S) \times (…, S, d_v) = (…, L, d_v)$。`,
        },
      ],
      setup: py`
        import numpy as np

        def _ref_attn(q, k, v, mask=None):
            s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
            if mask is not None:
                s = np.where(mask, s, -np.inf)
            s = s - s.max(-1, keepdims=True)
            w = np.exp(s)
            w = w / w.sum(-1, keepdims=True)
            return w @ v
      `,
      tests: [
        {
          name: '无掩码',
          code: py`
            rng = np.random.default_rng(0)
            q, k, v = rng.normal(size=(4, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 3))
            out = attention(q, k, v)
            assert out.shape == (4, 3), f"形状应为 (4, 3)，得到 {out.shape}"
            assert np.allclose(out, _ref_attn(q, k, v), atol=1e-8), f"最大误差 {np.abs(out - _ref_attn(q, k, v)).max():.2e}"
          `,
        },
        {
          name: '因果掩码',
          code: py`
            rng = np.random.default_rng(1)
            q, k, v = (rng.normal(size=(2, 5, 4)) for _ in range(3))
            m = np.tril(np.ones((5, 5), dtype=bool))
            out = attention(q, k, v, m)
            assert np.allclose(out, _ref_attn(q, k, v, m), atol=1e-8)
            assert np.allclose(out[:, 0], v[:, 0]), "第 0 个 token 只能看自己，输出应等于 v[0]"
          `,
        },
        {
          name: '批量 + padding 掩码 + 大数值',
          hidden: true,
          code: py`
            rng = np.random.default_rng(2)
            q, k, v = (rng.normal(size=(2, 3, 6, 8)) * 30 for _ in range(3))
            m = np.ones((2, 1, 1, 6), dtype=bool)
            m[0, ..., 4:] = False
            out = attention(q, k, v, m)
            assert np.all(np.isfinite(out)), "出现 inf/NaN：是否忘了减最大值？"
            assert np.allclose(out, _ref_attn(q, k, v, m), atol=1e-6)
          `,
        },
      ],
      outro: md`
        这 6 行就是面试里「手撕 attention」的核心。实战题会在此基础上加上**全遮行不出 NaN**与 **$L \ne S$ 的右下角对齐因果掩码**——两个最常被追问的边界。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-sdpa',
      title: '手撕 scaled_dot_product_attention（含因果对齐与全遮行）',
      difficulty: 'medium',
      source: '大模型岗手撕 · 注意力',
      timeLimitMin: 20,
      description: md`
        用 NumPy 实现

        \`\`\`python
        def sdpa(q, k, v, attn_mask=None, is_causal=False) -> np.ndarray: ...
        \`\`\`

        - \`q\`: \`(..., L, d)\`，\`k\`: \`(..., S, d)\`，\`v\`: \`(..., S, dv)\`；前导维度相同（batch、head 等）。
        - \`attn_mask\`: 布尔数组或 \`None\`，可广播到 \`(..., L, S)\`，**True = 参与注意力**。
        - \`is_causal\`: 因果掩码按**右下角对齐**：query $i$ 只能看 key $j \le i + (S - L)$（$L=S$ 时就是普通下三角）。两种掩码可以同时给出，取逻辑与。
        - 缩放因子 $1/\sqrt{d}$。
        - 若某个 query 一个可见 key 都没有，该行输出**全 0**，整个结果中不得出现 NaN / inf。
        - 输入可能很大（logits 上千），需数值稳定。

        **示例**：\`L = 1, S = 4, is_causal=True\` 时，唯一的 query 能看到全部 4 个 key（它是序列的最后一个 token）。
      `,
      starter: py`
        import numpy as np

        def sdpa(q, k, v, attn_mask=None, is_causal=False):
            """q: (..., L, d)  k: (..., S, d)  v: (..., S, dv)  ->  (..., L, dv)"""
            # TODO 1: scores = q k^T / sqrt(d)
            # TODO 2: 组合因果掩码（右下角对齐）与 attn_mask
            # TODO 3: 稳定 softmax，全遮行输出 0
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def sdpa(q, k, v, attn_mask=None, is_causal=False):
            """q: (..., L, d)  k: (..., S, d)  v: (..., S, dv)  ->  (..., L, dv)"""
            L, S, d = q.shape[-2], k.shape[-2], q.shape[-1]
            scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)          # (..., L, S)

            allowed = np.ones((L, S), dtype=bool)
            if is_causal:
                allowed = np.tril(allowed, k=S - L)                   # 右下角对齐
            if attn_mask is not None:
                allowed = allowed & attn_mask                         # 广播到 (..., L, S)
            scores = np.where(allowed, scores, -np.inf)

            m = scores.max(axis=-1, keepdims=True)
            m = np.where(np.isfinite(m), m, 0.0)                      # 全遮行：避免 -inf - (-inf)
            w = np.exp(scores - m)                                    # 被遮位置 exp(-inf) = 0
            denom = w.sum(axis=-1, keepdims=True)
            w = w / np.where(denom > 0, denom, 1.0)                   # 全遮行：0 / 1 = 0
            return w @ v
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import math
            import torch

            def sdpa(q, k, v, attn_mask=None, is_causal=False):
                L, S = q.size(-2), k.size(-2)
                scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
                allowed = torch.ones(L, S, dtype=torch.bool, device=q.device)
                if is_causal:
                    allowed = allowed.tril(diagonal=S - L)          # 右下角对齐
                if attn_mask is not None:
                    allowed = allowed & attn_mask
                scores = scores.masked_fill(~allowed, float("-inf"))
                w = torch.softmax(scores, dim=-1)
                w = torch.nan_to_num(w, nan=0.0)                    # 全遮行 → 0
                return w @ v

            # 生产中直接调用（注意：is_causal 为左上角对齐，且不能与 attn_mask 同时传）：
            # torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
          `,
        },
      ],
      explanation: md`
        **三个要点**
        1. **掩码组合**：先构造 $(L, S)$ 的 \`allowed\`，因果部分用 \`np.tril(ones, k=S-L)\` 实现右下角对齐，再与 \`attn_mask\` 做逻辑与（自动广播）。
        2. **稳定 softmax**：减行最大值。全遮行的最大值是 $-\infty$，\`-inf - (-inf) = NaN\`，所以先把非有限的最大值替换为 0。
        3. **全遮行输出 0**：此时 \`w\` 全为 0、分母为 0，用 \`np.where(denom > 0, denom, 1)\` 避免除零，结果自然是 0。

        **复杂度**：时间 $O(LSd)$，额外空间 $O(LS)$（分数矩阵）。

        **常见错误**：用 \`.T\` 转置高维数组；softmax 沿错轴；\`is_causal\` 用 \`np.tril(ones((L, S)))\`（左上角对齐，解码时只能看第一个 token）；全遮行出现 NaN。
      `,
      complexity: { time: 'O(L·S·d)', space: 'O(L·S)' },
      hints: [
        md`先写出不带掩码的版本：\`q @ np.swapaxes(k, -1, -2) / np.sqrt(d)\` → softmax → \`@ v\`。`,
        md`右下角对齐的因果掩码：\`np.tril(np.ones((L, S), bool), k=S - L)\`。`,
        md`全遮行：行最大值为 \`-inf\`，用 \`np.where(np.isfinite(m), m, 0)\` 替换；分母为 0 时用 1 代替。`,
      ],
      setup: py`
        import numpy as np

        def _ref_sdpa(q, k, v, attn_mask=None, is_causal=False):
            q, k, v = np.asarray(q, float), np.asarray(k, float), np.asarray(v, float)
            L, S, d = q.shape[-2], k.shape[-2], q.shape[-1]
            allowed = np.ones((L, S), dtype=bool)
            for i in range(L):
                for j in range(S):
                    if is_causal and j > i + (S - L):
                        allowed[i, j] = False
            if attn_mask is not None:
                allowed = np.broadcast_to(allowed & attn_mask, q.shape[:-2] + (L, S))
            else:
                allowed = np.broadcast_to(allowed, q.shape[:-2] + (L, S))
            lead = q.shape[:-2]
            out = np.zeros(lead + (L, v.shape[-1]))
            for idx in np.ndindex(*lead):
                for i in range(L):
                    js = [j for j in range(S) if allowed[idx + (i, j)]]
                    if not js:
                        continue
                    s = np.array([q[idx + (i,)] @ k[idx + (j,)] for j in js]) / np.sqrt(d)
                    p = np.exp(s - s.max())
                    p /= p.sum()
                    out[idx + (i,)] = sum(pj * v[idx + (j,)] for pj, j in zip(p, js))
            return out

        def _check(got, exp, what):
            assert got.shape == exp.shape, f"{what}：形状应为 {exp.shape}，得到 {got.shape}"
            assert np.all(np.isfinite(got)), f"{what}：输出含 NaN/inf"
            err = np.abs(got - exp).max()
            assert err < 1e-6, f"{what}：最大误差 {err:.2e}"
      `,
      tests: [
        {
          name: '无掩码',
          code: py`
            rng = np.random.default_rng(0)
            q, k, v = rng.normal(size=(3, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 2))
            _check(sdpa(q, k, v), _ref_sdpa(q, k, v), "无掩码")
          `,
        },
        {
          name: '因果掩码（L = S）',
          code: py`
            rng = np.random.default_rng(1)
            q, k, v = (rng.normal(size=(2, 6, 4)) for _ in range(3))
            _check(sdpa(q, k, v, is_causal=True), _ref_sdpa(q, k, v, is_causal=True), "因果")
          `,
        },
        {
          name: 'padding 掩码广播 (B, 1, 1, S)',
          code: py`
            rng = np.random.default_rng(2)
            q, k, v = (rng.normal(size=(2, 2, 5, 3)) for _ in range(3))
            m = np.ones((2, 1, 1, 5), dtype=bool)
            m[1, ..., 3:] = False
            _check(sdpa(q, k, v, attn_mask=m), _ref_sdpa(q, k, v, attn_mask=m), "padding")
          `,
        },
        {
          name: 'L ≠ S：右下角对齐',
          hidden: true,
          code: py`
            rng = np.random.default_rng(3)
            q = rng.normal(size=(2, 3, 4))
            k, v = rng.normal(size=(2, 7, 4)), rng.normal(size=(2, 7, 5))
            _check(sdpa(q, k, v, is_causal=True), _ref_sdpa(q, k, v, is_causal=True), "L=3, S=7 因果")
            q1 = q[:, -1:]
            _check(sdpa(q1, k, v, is_causal=True), _ref_sdpa(q1, k, v), "单 token 解码应能看到全部 key")
          `,
        },
        {
          name: '全遮行输出 0 且无 NaN',
          hidden: true,
          code: py`
            rng = np.random.default_rng(4)
            q, k, v = (rng.normal(size=(4, 4, 3)) for _ in range(3))
            m = np.array([False, False, True, True])            # 左 padding：前两个 key 是 pad
            got = sdpa(q, k, v, attn_mask=m, is_causal=True)
            assert np.all(np.isfinite(got)), "全遮行产生了 NaN/inf"
            assert np.allclose(got[:, :2], 0), "整行被遮的 query 输出应为 0"
            _check(got, _ref_sdpa(q, k, v, attn_mask=m, is_causal=True), "因果 + 左 padding")
          `,
        },
        {
          name: '数值稳定（logits 上千）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(5)
            q, k, v = rng.normal(size=(2, 6, 16)) * 40, rng.normal(size=(2, 6, 16)) * 40, rng.normal(size=(2, 6, 3))
            _check(sdpa(q, k, v, is_causal=True), _ref_sdpa(q, k, v, is_causal=True), "大数值")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-dropout',
          q: md`训练时 attention dropout 加在哪里？推理时呢？`,
          a: md`加在 softmax 之后、乘 $V$ 之前：\`p = dropout(softmax(s))\`，保留的权重除以 $1-p_{\text{drop}}$（inverted dropout），所以每行和只在期望上为 1。推理时关闭。现代大模型预训练普遍把 attention dropout 设为 0。`,
        },
        {
          id: 'f-grad',
          q: md`写出 softmax 的反向：已知 $\partial \mathcal{L}/\partial P$，求 $\partial \mathcal{L}/\partial S$。`,
          a: md`$dS_{ij} = P_{ij}\big(dP_{ij} - \sum_k P_{ik}\, dP_{ik}\big)$，向量化为 \`dS = P * (dP - (P * dP).sum(-1, keepdims=True))\`。FlashAttention 反向中的 $D_i = \operatorname{rowsum}(dO_i \circ O_i)$ 正是 $\sum_k P_{ik} dP_{ik}$ 的高效算法。`,
        },
      ],
    },
    {
      id: 'c-mask-pos',
      title: '为左 / 右 padding 的批量构造注意力掩码与 position_ids',
      difficulty: 'medium',
      source: '大模型岗手撕 · 批量推理',
      timeLimitMin: 15,
      description: md`
        一个 batch 有 $B$ 条序列，真实长度为 \`lengths[b]\`，统一 pad 到长度 $T$。实现

        \`\`\`python
        def build_mask_and_positions(lengths, T, padding_side="right", causal=True):
            -> tuple[np.ndarray, np.ndarray]   # mask: (B, 1, T, T) bool,  pos: (B, T) int
        \`\`\`

        - **padding_side**：\`"right"\` 时真实 token 在前、pad 在后；\`"left"\` 时 pad 在前。
        - **mask**：\`mask[b, 0, i, j] = True\` 表示 query $i$ 可以看 key $j$。条件：$j$ 不是 pad，且（若 \`causal\`）$j \le i$。
          若某一行一个可见位置都没有（必然是 pad query），把该行的**对角线** \`[i, i]\` 设为 True，保证 softmax 有定义。
        - **pos**：真实 token 的位置编号从 0 开始连续递增（左 padding 时第一个真实 token 的 pos 也是 0）；pad 位置的值不检查。
        - 不许用 Python 三重循环逐元素填（$B \cdot T^2$ 次），请向量化（隐藏测试 $B=64, T=512$）。

        **示例**：\`lengths=[2, 3], T=3, padding_side="left"\`
        - 第 0 条：\`[pad, x, x]\`，\`pos = [*, 0, 1]\`；mask 第 0 行全 False → 只保留对角线 \`[0, 0]\`。
      `,
      starter: py`
        import numpy as np

        def build_mask_and_positions(lengths, T, padding_side="right", causal=True):
            # TODO: key_valid (B, T) → mask (B, 1, T, T)，以及 pos (B, T)
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def build_mask_and_positions(lengths, T, padding_side="right", causal=True):
            lengths = np.asarray(lengths)
            ar = np.arange(T)
            if padding_side == "right":
                valid = ar[None, :] < lengths[:, None]                  # (B, T)
            else:
                valid = ar[None, :] >= (T - lengths)[:, None]
            mask = np.broadcast_to(valid[:, None, :], (len(lengths), T, T)).copy()   # 按 key 列遮 pad
            if causal:
                mask &= np.tril(np.ones((T, T), dtype=bool))[None]
            empty = ~mask.any(axis=-1)                                   # (B, T) 全遮行
            b_idx, i_idx = np.nonzero(empty)
            mask[b_idx, i_idx, i_idx] = True                             # 让它至少看自己
            pos = np.cumsum(valid, axis=-1) - 1                          # 真实 token 从 0 计数
            pos = np.where(valid, pos, 0)
            return mask[:, None], pos
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def build_mask_and_positions(lengths: torch.Tensor, T: int, padding_side="right", causal=True):
                ar = torch.arange(T, device=lengths.device)
                valid = ar[None] < lengths[:, None] if padding_side == "right" else ar[None] >= (T - lengths)[:, None]
                mask = valid[:, None, :].expand(-1, T, -1).clone()
                if causal:
                    mask &= torch.ones(T, T, dtype=torch.bool, device=lengths.device).tril()
                empty = ~mask.any(-1)
                mask |= torch.diag_embed(empty)                        # 全遮行打开对角线
                pos = (valid.long().cumsum(-1) - 1).masked_fill(~valid, 0)   # HF 中 pad 位置常填 1
                return mask[:, None], pos
          `,
        },
      ],
      explanation: md`
        **思路**：先得到 \`valid (B, T)\`（哪些位置是真实 token），掩码 = 「key 有效」广播到每一行 & 因果下三角。
        全遮行用 \`~mask.any(-1)\` 找出来，用高级索引打开对角线。

        **position_ids**：\`cumsum(valid) - 1\` 让第一个真实 token 编号为 0。左 padding 若直接用 \`arange(T)\`，同一句话在不同 batch 里位置编号不同，RoPE 下结果会变——这是批量推理结果「和单条推理不一致」的经典原因。

        **复杂度**：$O(BT^2)$ 时间与空间，全部向量化。
      `,
      complexity: { time: 'O(B·T²)', space: 'O(B·T²)' },
      hints: [
        md`右 padding：\`valid = arange(T)[None] < lengths[:, None]\`；左 padding：\`arange(T)[None] >= (T - lengths)[:, None]\`。`,
        md`\`mask = valid[:, None, :] & tril(ones((T, T)))\`，形状 \`(B, T, T)\`，最后再 \`[:, None]\`。`,
        md`\`empty = ~mask.any(-1)\`；\`b, i = np.nonzero(empty)\`；\`mask[b, i, i] = True\`。位置用 \`np.cumsum(valid, -1) - 1\`。`,
      ],
      setup: py`
        import numpy as np, time

        def _ref_build(lengths, T, padding_side="right", causal=True):
            B = len(lengths)
            mask = np.zeros((B, 1, T, T), dtype=bool)
            pos = np.zeros((B, T), dtype=int)
            for b, n in enumerate(lengths):
                start = 0 if padding_side == "right" else T - n
                valid = [start <= j < start + n for j in range(T)]
                for j in range(T):
                    if valid[j]:
                        pos[b, j] = j - start
                for i in range(T):
                    for j in range(T):
                        mask[b, 0, i, j] = valid[j] and (not causal or j <= i)
                    if not mask[b, 0, i].any():
                        mask[b, 0, i, i] = True
            return mask, pos

        def _cmp(lengths, T, side, causal):
            m, p = build_mask_and_positions(lengths, T, side, causal)
            em, ep = _ref_build(lengths, T, side, causal)
            assert m.shape == em.shape, f"mask 形状应为 {em.shape}，得到 {m.shape}"
            assert m.dtype == bool, f"mask 应为 bool，得到 {m.dtype}"
            bad = np.argwhere(m != em)
            assert len(bad) == 0, f"lengths={lengths}, side={side}, causal={causal}：mask 在 {bad[0].tolist()} 处不一致"
            assert p.shape == ep.shape, f"pos 形状应为 {ep.shape}，得到 {p.shape}"
            for b, n in enumerate(lengths):
                start = 0 if side == "right" else T - n
                got = np.asarray(p[b, start:start + n]).tolist()
                assert got == list(range(n)), f"第 {b} 条真实 token 的 pos 应为 {list(range(n))}，得到 {got}"
      `,
      tests: [
        { name: '示例：左 padding + 因果', code: py`_cmp([2, 3], 3, "left", True)` },
        { name: '右 padding + 因果', code: py`_cmp([4, 1, 3], 4, "right", True)` },
        { name: '右 padding 非因果（编码器）', code: py`_cmp([2, 5], 5, "right", False)` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(7)
            for _ in range(30):
                T = int(rng.integers(1, 9))
                lengths = rng.integers(1, T + 1, size=int(rng.integers(1, 5))).tolist()
                _cmp(lengths, T, str(rng.choice(["left", "right"])), bool(rng.integers(0, 2)))
          `,
        },
        {
          name: '规模：B=64, T=512（需向量化）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(8)
            lengths = rng.integers(1, 513, size=64).tolist()
            t0 = time.perf_counter()
            m, p = build_mask_and_positions(lengths, 512, "left", True)
            dt = time.perf_counter() - t0
            assert m.shape == (64, 1, 512, 512)
            assert dt < 1.5, f"耗时 {dt:.2f}s，请向量化"
            b = 3; n = lengths[b]
            assert not m[b, 0, -1, : 512 - n].any(), "最后一个 query 不应看到 pad"
            assert m[b, 0, -1, 512 - n:].all(), "最后一个 query 应看到全部真实 token"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-left',
          q: md`为什么 decoder-only 模型批量生成要用左 padding？`,
          a: md`生成时每步取**最后一个位置**的 logits 并在末尾追加新 token。右 padding 时短序列的最后一个真实 token 不在位置 $T-1$，新 token 会被追加在 pad 后面，中间夹着 pad；左 padding 让所有序列的末尾对齐，追加操作是统一的 \`concat\`。代价是要正确处理 position_ids 与掩码。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-scale',
      freq: 3,
      q: md`Attention 为什么要除以 $\sqrt{d_k}$？不除会怎样？为什么不除以 $d_k$？`,
      a: md`
        **结论**：为了让 logits 的方差与维度无关，避免 softmax 饱和导致梯度消失。

        **原因**：设 $q, k$ 各分量独立、零均值、单位方差，则 $\operatorname{Var}(q\cdot k) = d_k$，标准差 $\sqrt{d_k}$。不缩放时 $d_k=128$ 的 logits 差距可达十几，softmax 接近 one-hot，雅可比 $p_i(\delta_{ij}-p_j)$ 趋于 0，Q/K 投影学不动。除以 $\sqrt{d_k}$ 让方差回到 1。除以 $d_k$ 则方差变为 $1/d_k$，logits 几乎全为 0，注意力退化为均匀平均。

        **延伸**：这只是初始化时的近似论证。训练中 logits 仍可能变大（attention logit growth），所以有 QK-Norm（Qwen3、Gemma 3 等）、logit soft-capping（Gemma 2）、MuonClip/QK-Clip（Kimi K2）等进一步控制手段；μP 则用 $1/d$ 缩放以便超参数随宽度迁移。
      `,
    },
    {
      id: 'qa-mask',
      freq: 3,
      q: md`因果掩码和 padding 掩码分别怎么实现？两者怎么合并？有什么坑？`,
      a: md`
        **结论**：都是 $(L, S)$ 上的布尔矩阵，取逻辑与后在 softmax **之前**把不允许的位置设为 $-\infty$（或 dtype 最小值）。

        **实现**：因果用 \`tril(ones(T, T))\`；padding 掩码由 \`attention_mask (B, S)\` 扩成 \`(B, 1, 1, S)\` 按 key 列广播。

        **坑**：① softmax 之后再乘 0 是错的；② 左 padding + 因果会产生整行被遮的 pad query，用 $-\infty$ 会得到 NaN 并经 $0\times\text{NaN}$ 传染，HF 用 \`finfo.min\` 规避；③ $L \ne S$（KV cache 解码、chunked prefill）时因果掩码要右下角对齐，PyTorch SDPA 的 \`is_causal\` 是左上角对齐；④ PyTorch SDPA 的布尔掩码 True=参与，\`nn.MultiheadAttention\` 的 True=屏蔽，方向相反。
      `,
    },
    {
      id: 'qa-complexity',
      freq: 2,
      q: md`自注意力的时间和空间复杂度是多少？长序列的瓶颈在哪里？`,
      a: md`
        **结论**：时间 $O(T^2 d)$，朴素实现空间 $O(T^2)$（每个头每层一个 $T\times T$ 矩阵）。

        **原因**：$QK^\top$ 与 $PV$ 各 $2T^2d$ FLOPs；分数矩阵与 softmax 结果都要写回显存，训练时还要为反向保存。对比 FFN 的 $O(Td^2)$：$T$ 超过约 $d$ 量级后注意力占主导。

        **延伸**：FlashAttention 不改变 $O(T^2 d)$ 的计算量，但通过分块 + online softmax 把额外显存降到 $O(T)$、大幅减少 HBM 读写；要降低计算量本身需要稀疏注意力（滑窗、NSA/DSA）或线性注意力。
      `,
    },
    {
      id: 'qa-qk',
      freq: 2,
      q: md`Q 和 K 为什么要用两个不同的投影矩阵？共享会怎样？`,
      a: md`
        **结论**：分开投影让相似度 $x_i^\top W_Q W_K^\top x_j$ 可以是**非对称**的双线性形式；共享则矩阵 $W W^\top$ 对称半正定。

        **原因**：对称意味着「i 关注 j」与「j 关注 i」打分相同，且 $x_i^\top W W^\top x_i = \|W^\top x_i\|^2 \ge$ 与他人的点积上界之一，模型倾向于关注自己；而语言中的关系（主语找谓语、代词找先行词）往往是非对称的。

        **延伸**：$W_Q W_K^\top$ 的秩至多 $d_h$，多头就是用多个低秩双线性形式拼出丰富的关系；MLA 的「矩阵吸收」正是利用了 $W_Q W_K^\top$ 可以合并成一个矩阵这一点。
      `,
    },
  ],

  cheatsheet: md`
    $$\text{softmax}\Big(\tfrac{QK^\top}{\sqrt{d_k}} + M\Big)V,\quad \operatorname{Var}(q\cdot k)=d_k$$

    \`\`\`python
    def sdpa(q, k, v, mask=None):            # mask: True = 可看
        s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        w = np.exp(s); w /= w.sum(-1, keepdims=True)
        return w @ v
    \`\`\`

    - softmax 沿 **key 维**（\`axis=-1\`）；掩码在 softmax **之前**
    - 缩放用**单头**维度 $\sqrt{d_h}$；head_dim=128 → $1/\sqrt{128}\approx 0.0884$
    - $L\ne S$ 因果：\`tril(ones((L,S)), k=S-L)\`（右下角对齐）；PyTorch \`is_causal\` 是左上角
    - 左 padding + 因果 → 全遮行 → NaN；用 \`finfo.min\` 或置 0
    - PyTorch SDPA 布尔掩码 True=参与；\`nn.MultiheadAttention\` True=屏蔽
    - 分数矩阵显存：$B\cdot h\cdot L\cdot S\cdot$ bytes
  `,
})
