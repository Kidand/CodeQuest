/**
 * L3-2 多头注意力：为什么多头、拆头 reshape/transpose 的维度推导、W_O 与参数量 / FLOPs。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { HeadPatternsViz } from '../viz/HeadPatternsViz'
import { ShapeFlowViz } from '../viz/ShapeFlowViz'
import { ParamFlopsViz } from '../viz/ParamFlopsViz'


export default defineLevel({
  learn: [
    {
      id: 'why-multi',
      title: '为什么要多头：一个 softmax 只能表达一种分布',
      body: md`
        单头注意力里，每个 query 只有**一个** softmax 分布。如果「它」既要找先行词「垫子」、又要看前一个词，单头只能把两者平均成一个模糊的分布。

        多头的做法：把 $D$ 维切成 $h$ 份，每个头在自己的 $d_h = D/h$ 维子空间里独立做注意力，最后拼接再用 $W_O$ 混合：

        $$
        \text{head}_i = \operatorname{softmax}\!\Big(\frac{Q_i K_i^\top}{\sqrt{d_h}}\Big)V_i,\qquad
        \text{MHA}(X) = \operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O
        $$

        其中 $Q_i = X W_Q^{(i)}$，$W_Q^{(i)}$ 是 $W_Q$ 的第 $i$ 个**列块**（$D \times d_h$）。

        **代价几乎为零**：$h$ 个 $d_h$ 维的头与一个 $D$ 维的单头相比，投影参数完全相同（都是 $4D^2$），注意力 FLOPs 也相同（$h \cdot 2T^2 d_h = 2T^2 D$）。
        多出来的只是 $h$ 个独立的 softmax。

        下图四个头的 logits 由简单规则生成（前一个 token / 首 token / 指代 / 局部平均），再经因果 softmax 实算——真实模型中可以观察到类似的头。
      `,
      viz: HeadPatternsViz,
      after: md`
        > [!NOTE] 头越多越好吗？
        > 不是。$D$ 固定时头越多 $d_h$ 越小，每个头的 $W_Q^{(i)} W_K^{(i)\top}$ 秩不超过 $d_h$，表达能力受限（Bhojanapalli et al., 2020 称之为 low-rank bottleneck）。
        > 主流大模型的 $d_h$ 几乎都取 **128**（Llama 2/3、Qwen2.5、DeepSeek-V3 的 nope 部分等），再按 $D$ 决定头数。另有研究发现训练好的模型中相当一部分头可以剪掉而几乎不掉点（Michel et al., 2019）。
      `,
      checkpoint: {
        id: 'cp-why',
        type: 'judge',
        prompt: md`$D$ 固定时，把头数 $h$ 从 8 增加到 32，MHA 的参数量和注意力 FLOPs 都会增加。`,
        answer: false,
        explanation: md`都不变。参数量始终是 $W_Q, W_K, W_V, W_O$ 四个 $D\times D$ 矩阵；注意力 FLOPs 为 $h \cdot 2T^2 \cdot (D/h) = 2T^2D$。改变的只是每个头的维度 $d_h$。`,
      },
    },
    {
      id: 'shapes',
      title: '拆头的维度推导：reshape 再 transpose',
      body: md`
        手撕 MHA 最常错的就是拆头。正确流程（$B$ 批、$T$ 长度、$D = h \cdot d_h$）：

        | 步骤 | 代码 | 形状 |
        |---|---|---|
        | 投影 | \`q = x @ Wq\` | $(B, T, D)$ |
        | 切分最后一维 | \`q.reshape(B, T, h, d_h)\` | $(B, T, h, d_h)$ |
        | 头维提前 | \`.transpose(0, 2, 1, 3)\` | $(B, h, T, d_h)$ |
        | 打分 | \`q @ k.transpose(0, 1, 3, 2) / sqrt(d_h)\` | $(B, h, T, T)$ |
        | 加权 | \`softmax(s) @ v\` | $(B, h, T, d_h)$ |
        | 合头 | \`.transpose(0, 2, 1, 3).reshape(B, T, D)\` | $(B, T, D)$ |
        | 输出投影 | \`@ Wo\` | $(B, T, D)$ |

        **为什么不能直接 \`reshape(B, h, T, d_h)\`？** reshape 只按**内存顺序**重新切分。$(B, T, D)$ 在内存里是「token 0 的 D 维、token 1 的 D 维……」，
        直接切成 $(h, T, d_h)$ 时，「头 0」拿到的是前 $T \cdot d_h$ 个元素——即前几个 token 的**全部**维度。形状对、不报错、结果全错。

        逐步播放下图，再切换到「错误写法」对比。
      `,
      viz: ShapeFlowViz,
      after: md`
        > [!WARN] PyTorch 的 \`view\` 与 \`contiguous\`
        > \`transpose\` 只改 stride，不搬数据，结果是**非连续**张量。合头时 \`o.transpose(1, 2).view(B, T, D)\` 会报错，
        > 必须写 \`o.transpose(1, 2).contiguous().view(B, T, D)\` 或 \`o.transpose(1, 2).reshape(B, T, D)\`（reshape 必要时自动拷贝）。
        > NumPy 的 \`reshape\` 总能成功（必要时拷贝），所以不会暴露这个问题。

        > [!TIP] 记忆口诀
        > **拆头**：先 reshape（切最后一维）后 transpose（头维提前）；**合头**：先 transpose（头维放回）后 reshape（拼回最后一维）。两者互为逆操作。
      `,
      checkpoint: {
        id: 'cp-shape',
        type: 'single',
        prompt: md`$x$ 形状 $(2, 10, 512)$，$h = 8$。\`q = (x @ Wq).reshape(2, 10, 8, 64).transpose(0, 2, 1, 3)\` 之后 \`q[1, 3]\` 的形状和含义是？`,
        options: ['`(10, 64)`：第 2 个样本、第 4 个头里全部 10 个 token 的 query', '`(8, 64)`：第 2 个样本、第 4 个 token 的全部 8 个头', '`(64,)`：第 4 个 token 的第 2 个头', '`(10, 512)`：第 2 个样本的完整 query'],
        answer: 0,
        explanation: md`transpose 之后形状为 $(B, h, T, d_h) = (2, 8, 10, 64)$，\`q[1, 3]\` 取第 1 个样本、第 3 个头（0 起），剩下 $(T, d_h) = (10, 64)$。B 选项描述的是 transpose **之前** \`q[1, 3]\` 的含义。`,
      },
    },
    {
      id: 'params',
      title: 'W_O、参数量与 FLOPs',
      body: md`
        **$W_O$ 不是可有可无的**。把 $W_O$ 按行切成 $h$ 块 $W_O^{(i)} \in \mathbb{R}^{d_h \times D}$：

        $$
        \operatorname{Concat}(o_1,\dots,o_h)\,W_O = \sum_{i=1}^{h} o_i\, W_O^{(i)}
        $$

        即每个头的输出各自映射回 $D$ 维后**相加**——这是各头信息在本层内唯一的交汇点。这个「按头分块求和」的视角在 MLA 一关会再次用到（$W_{UV}$ 吸收进 $W_O$）。

        **参数量**（无 bias）：$4D^2$；带 bias 再加 $4D$。$D = 4096$ 时约 67.1M，一个 32 层模型的注意力投影共约 2.1B。

        **前向 FLOPs**（$B=1$，1 次乘加记 2 FLOPs）：

        | 部分 | FLOPs |
        |---|---|
        | Q、K、V 投影 | $3 \times 2TD^2 = 6TD^2$ |
        | $QK^\top$（所有头） | $2T^2D$ |
        | $PV$（所有头） | $2T^2D$ |
        | $W_O$ 投影 | $2TD^2$ |

        合计 $8TD^2 + 4T^2D$，两者相等于 $T = 2D$。拖动下图滑块观察：改变 $h$ 时所有数字都不变。
      `,
      viz: ParamFlopsViz,
      after: md`
        > [!INTERVIEW] 常见计算题
        > 「GPT-2 small（$D=768$，带 bias）每层注意力多少参数？」答：$4 \times 768^2 + 4 \times 768 = 2{,}362{,}368$。
        > 「Llama 类模型用 GQA 后呢？」$W_K, W_V$ 变成 $D \times (n_{kv} d_h)$，参数量 $2D^2 + 2D\,n_{kv}d_h$——下一关详讲。
      `,
      checkpoint: {
        id: 'cp-params',
        type: 'numeric',
        prompt: md`$D = 2048$、$h = 16$、无 bias 的 MHA 层，参数量是多少（单位：M，$1\text{M} = 10^6$）？`,
        answer: 16.777216,
        tolerance: 0.01,
        unit: 'M',
        explanation: md`$4 \times 2048^2 = 16{,}777{,}216 \approx 16.78$M。头数 16 是干扰信息——它不影响参数量。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-dh',
      type: 'single',
      difficulty: 1,
      prompt: md`$d_{\text{model}} = 4096$、32 个头的 MHA，每个头的维度 $d_h$ 与 softmax 前的缩放因子分别是？`,
      options: ['$128$，$1/\\sqrt{128}$', '$128$，$1/\\sqrt{4096}$', '$4096$，$1/\\sqrt{4096}$', '$32$，$1/\\sqrt{32}$'],
      answer: 0,
      explanation: md`$d_h = 4096/32 = 128$；每个头内部的点积维度是 $d_h$，所以缩放 $1/\sqrt{d_h}$。用 $\sqrt{D}$ 缩放会让 logits 过小（相当于温度放大 $\sqrt{32}$ 倍）。`,
    },
    {
      id: 'q-params',
      type: 'numeric',
      difficulty: 1,
      prompt: md`GPT-2 small：$D = 768$，12 个头，Q/K/V/O 投影都带 bias。**单层**注意力的参数量是多少？`,
      answer: 2362368,
      tolerance: 0,
      explanation: md`权重 $4 \times 768^2 = 2{,}359{,}296$，bias $4 \times 768 = 3{,}072$，合计 $2{,}362{,}368$。头数不影响参数量。`,
    },
    {
      id: 'q-heads-judge',
      type: 'judge',
      difficulty: 1,
      prompt: md`多头注意力中，每个头都有自己独立的、形状为 $D \times D$ 的 $W_Q$。`,
      answer: false,
      explanation: md`每个头的 $W_Q^{(i)}$ 形状是 $D \times d_h$，它们是同一个 $D \times D$ 矩阵 $W_Q$ 的不同列块。实现上用一次大矩阵乘 \`x @ Wq\` 再 reshape 拆头，而不是循环 $h$ 次。`,
    },
    {
      id: 'q-view',
      type: 'single',
      difficulty: 2,
      prompt: md`PyTorch 中 \`o\` 形状为 \`(B, h, T, d_h)\`，执行 \`o.transpose(1, 2).view(B, T, h * d_h)\` 会怎样？`,
      options: [
        '正常得到 `(B, T, D)`',
        '报错：transpose 后张量不连续，`view` 无法在不拷贝的情况下改变形状',
        '不报错，但各头的输出被交错拼接',
        '报错：维度乘积不匹配',
      ],
      answer: 1,
      explanation: md`\`transpose\` 只交换 stride，不移动数据；\`view\` 要求新形状能用原 stride 表达，非连续张量合并 \`h\` 与 \`d_h\` 维时做不到，会抛 RuntimeError。改用 \`.contiguous().view(...)\` 或 \`.reshape(...)\`。元素个数是一致的，所以 D 错。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 MHA 前向**不会报错**，但结果是错的。点出有问题的 **2** 行：`,
      code: py`
        def mha(x, Wq, Wk, Wv, Wo, h):
            B, T, D = x.shape
            d = D // h
            q = (x @ Wq).reshape(B, h, T, d)
            k = (x @ Wk).reshape(B, T, h, d).transpose(0, 2, 1, 3)
            v = (x @ Wv).reshape(B, T, h, d).transpose(0, 2, 1, 3)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d)
            p = softmax(s, axis=-1)
            o = (p @ v).reshape(B, T, D)
            return o @ Wo
      `,
      bugLines: [4, 9],
      explanation: md`第 4 行：q 直接 \`reshape(B, h, T, d)\`，「头 0」拿到的是前几个 token 的全部维度，应为 \`.reshape(B, T, h, d).transpose(0, 2, 1, 3)\`。
        第 9 行：\`(p @ v)\` 形状是 \`(B, h, T, d)\`，必须先 \`.transpose(0, 2, 1, 3)\` 再 reshape，否则同样按内存顺序错拼。k、v 的写法是对的。`,
    },
    {
      id: 'q-wo',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于输出投影 $W_O$，下列说法正确的是：（多选）`,
      options: [
        '$h \\cdot d_v = D$ 时 $W_O$ 的形状是 $D \\times D$',
        '$\\operatorname{Concat}(o_1,\\dots,o_h)W_O = \\sum_i o_i W_O^{(i)}$，$W_O^{(i)}$ 是 $W_O$ 的第 $i$ 个行块',
        '若去掉 $W_O$，各头的输出在本层注意力内无法相互混合',
        '$W_O$ 是冗余的，可以直接合并进 $W_V$ 且不增加任何参数',
        '$h = 1$ 时拼接结果已是 $D$ 维，因此单头注意力不需要 $W_O$',
      ],
      answer: [0, 1, 2],
      explanation: md`A、B、C 正确。D：每个头的 $W_V^{(i)} W_O^{(i)}$ 确实可以合并成一个 $D\times D$ 矩阵（所谓 OV circuit），但 $h$ 个头就是 $hD^2$ 参数，远多于原来的 $2D^2$——低秩分解正是为了省参数。E：$W_O$ 的作用不是「升维」，单头时它仍提供一个额外的线性变换；标准 Transformer 在任何头数下都保留它。`,
    },
    {
      id: 'q-flops',
      type: 'fill',
      difficulty: 2,
      prompt: md`$B=1$、$T=4096$、$D=4096$ 的自注意力（不跳过因果上三角），所有头的 $QK^\top$ 共 ____(1)____ FLOPs；四个投影共 ____(2)____ FLOPs。（1 次乘加 = 2 FLOPs，可填表达式如 \`2*4096^3\`）`,
      blanks: [
        { label: '(1)', answers: ['137438953472'], numeric: true, tolerance: 0.01 },
        { label: '(2)', answers: ['549755813888'], numeric: true, tolerance: 0.01 },
      ],
      explanation: md`(1) $2T^2D = 2 \times 4096^3 \approx 1.37\times10^{11}$。(2) $8TD^2 = 8 \times 4096^3 \approx 5.50\times10^{11}$。此时 $T = D < 2D$，投影仍是大头；序列长到 8192 以上注意力矩阵乘才反超。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 MHA 中 query 张量的形状变化按顺序排列：`,
      items: ['`(B, T, D)` 投影后', '`(B, T, h, d_h)` reshape', '`(B, h, T, d_h)` transpose', '`(B, h, T, T)` 与 k 打分', '`(B, h, T, d_h)` 乘 v', '`(B, T, h, d_h)` transpose 回来', '`(B, T, D)` reshape 合头'],
      explanation: md`拆头 = reshape + transpose，合头 = transpose + reshape，二者互逆；中间的 $(B, h)$ 被当作批维做批量矩阵乘。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`$B=2$、$T=10$、$D=512$、$h=8$，把张量与形状配对：`,
      pairs: [
        ['拆头后的 q', '`(2, 8, 10, 64)`'],
        ['注意力分数 scores', '`(2, 8, 10, 10)`'],
        ['投影矩阵 W_Q', '`(512, 512)`'],
        ['合头后、W_O 之前的输出', '`(2, 10, 512)`'],
      ],
      explanation: md`$d_h = 512/8 = 64$；分数是每个头的 $T\times T$；$W_Q$ 一次性投影全部头；合头后回到 $(B, T, D)$。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`投影后的 q 形状为 $(T, D) = (2, 8)$（两个 token A、B）。如果**错误地**执行 \`q.reshape(2, 2, 4)\` 当作 $(h, T, d_h)$，那么「头 0」实际包含哪些元素？点出全部。`,
      items: Array.from({ length: 16 }, (_, n) => `${'AB'[Math.floor(n / 8)]}${n % 8}`),
      cols: 8,
      rowLabels: ['A', 'B'],
      colLabels: ['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7'],
      answer: [0, 1, 2, 3, 4, 5, 6, 7],
      explanation: md`reshape 按内存（行优先）顺序切分：前 $T \cdot d_h = 8$ 个元素就是 token A 的全部 8 维，于是「头 0」= token A，「头 1」= token B。正确拆分时头 0 应该是两个 token 的前 4 维（A0–A3、B0–B3）。`,
    },
    {
      id: 'q-why',
      type: 'single',
      difficulty: 3,
      prompt: md`在总维度 $D$ 相同的前提下，多头注意力相比单头注意力的**本质**优势是：`,
      options: [
        '参数更多，模型容量更大',
        '每个头有独立的 softmax，同一个 query 能在不同子空间里同时以不同模式关注不同位置',
        'FLOPs 更少，计算更快',
        '多头可以去掉对位置编码的依赖',
      ],
      answer: 1,
      explanation: md`参数量与 FLOPs 都与单头相同（A、C 错）。单头每个 query 只有一个分布，多个关注目标只能被平均；多头相当于 $h$ 个秩为 $d_h$ 的双线性相似度各自 softmax，再由 $W_O$ 组合。位置信息与头数无关（D 错）。代价是 $d_h$ 变小带来的低秩瓶颈。`,
    },
    {
      id: 'q-contig',
      type: 'judge',
      difficulty: 3,
      prompt: md`在 NumPy 中，\`x.reshape(B, T, h, d_h).transpose(0, 2, 1, 3)\` 会立即拷贝一份数据。`,
      answer: false,
      explanation: md`两步都返回 **view**：连续数组的 reshape 只改形状，transpose 只改 strides，都不拷贝。真正的拷贝发生在之后需要连续内存的操作（如合头时对非连续数组 reshape，或 \`np.ascontiguousarray\`）。矩阵乘 \`@\` 能直接处理任意 stride。`,
    },
  ],

  guided: [
    {
      id: 'g-mha',
      title: '一步步写出因果 MHA（NumPy）',
      intro: md`
        实现 \`mha(x, Wq, Wk, Wv, Wo, n_heads)\`：\`x\` 形状 \`(B, T, D)\`，四个权重都是 \`(D, D)\`（行向量约定 \`x @ W\`），因果自注意力，无 bias。
        已提供 \`softmax(x, axis=-1)\`。
      `,
      template: py`
        import numpy as np

        def softmax(x, axis=-1):
            x = x - x.max(axis=axis, keepdims=True)
            e = np.exp(x)
            return e / e.sum(axis=axis, keepdims=True)

        def mha(x, Wq, Wk, Wv, Wo, n_heads):
            B, T, D = x.shape
            d_h = D // n_heads

            def split(t):          # (B, T, D) -> (B, h, T, d_h)
                return {{1}}

            q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)
            s = {{2}}
            causal = np.tril(np.ones((T, T), dtype=bool))
            s = np.where(causal, s, -np.inf)
            o = {{3}}
            o = {{4}}
            return {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`拆头：先切最后一维，再把头维提到 T 之前。`,
          answer: 't.reshape(B, T, n_heads, d_h).transpose(0, 2, 1, 3)',
          accept: ['t.reshape(B, T, n_heads, d_h).swapaxes(1, 2)', 'np.transpose(t.reshape(B, T, n_heads, d_h), (0, 2, 1, 3))'],
          hint: '直接 reshape(B, n_heads, T, d_h) 形状也对，但内容错了。',
          explain: md`reshape 只能切分**相邻**维度：$D \to (h, d_h)$ 必须先在原位置切，再用 transpose 把 $h$ 挪到 $T$ 前面。`,
        },
        {
          blank: 2,
          instruction: md`每个头独立打分并缩放，形状 \`(B, h, T, T)\`。`,
          answer: 'q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_h)',
          accept: ['q @ k.swapaxes(-1, -2) / np.sqrt(d_h)', 'q @ np.swapaxes(k, -1, -2) / np.sqrt(d_h)', 'q @ k.swapaxes(2, 3) / np.sqrt(d_h)'],
          explain: md`缩放用 $\sqrt{d_h}$。若写成 \`np.sqrt(D)\`，测试会失败：等价于温度放大 $\sqrt{h}$ 倍。`,
        },
        {
          blank: 3,
          instruction: md`softmax 后对 v 加权，得到每个头的输出 \`(B, h, T, d_h)\`。`,
          answer: 'softmax(s) @ v',
          accept: ['softmax(s, axis=-1) @ v', 'np.matmul(softmax(s), v)'],
          explain: md`softmax 默认沿最后一维（key 维）。`,
        },
        {
          blank: 4,
          instruction: md`合头：回到 \`(B, T, D)\`。`,
          answer: 'o.transpose(0, 2, 1, 3).reshape(B, T, D)',
          accept: ['o.swapaxes(1, 2).reshape(B, T, D)', 'np.transpose(o, (0, 2, 1, 3)).reshape(B, T, D)'],
          explain: md`拆头的逆操作：先 transpose 把 $T$ 放回第 1 维，再把 $(h, d_h)$ 合并成 $D$。顺序反了同样是「形状对、内容错」。`,
        },
        {
          blank: 5,
          instruction: md`输出投影。`,
          answer: 'o @ Wo',
          accept: ['np.matmul(o, Wo)'],
          explain: md`$W_O$ 把各头的输出混合起来。`,
        },
      ],
      setup: py`
        import numpy as np

        def _ref_mha(x, Wq, Wk, Wv, Wo, h):
            B, T, D = x.shape
            dh = D // h
            out = np.zeros((B, T, D))
            for b in range(B):
                heads = []
                for i in range(h):
                    sl = slice(i * dh, (i + 1) * dh)
                    q, k, v = x[b] @ Wq[:, sl], x[b] @ Wk[:, sl], x[b] @ Wv[:, sl]
                    s = q @ k.T / np.sqrt(dh)
                    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
                    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
                    heads.append(p @ v)
                out[b] = np.concatenate(heads, -1) @ Wo
            return out

        def _rand(seed, B, T, D):
            r = np.random.default_rng(seed)
            return r.normal(size=(B, T, D)), *(r.normal(size=(D, D)) / np.sqrt(D) for _ in range(4))
      `,
      tests: [
        {
          name: 'B=1, T=4, D=8, h=2',
          code: py`
            x, Wq, Wk, Wv, Wo = _rand(0, 1, 4, 8)
            out = mha(x, Wq, Wk, Wv, Wo, 2)
            assert out.shape == (1, 4, 8), f"形状应为 (1, 4, 8)，得到 {out.shape}"
            exp = _ref_mha(x, Wq, Wk, Wv, Wo, 2)
            assert np.allclose(out, exp, atol=1e-8), f"最大误差 {np.abs(out - exp).max():.2e}"
          `,
        },
        {
          name: 'B=2, T=6, D=16, h=4',
          code: py`
            x, Wq, Wk, Wv, Wo = _rand(1, 2, 6, 16)
            out = mha(x, Wq, Wk, Wv, Wo, 4)
            exp = _ref_mha(x, Wq, Wk, Wv, Wo, 4)
            assert np.allclose(out, exp, atol=1e-8), f"最大误差 {np.abs(out - exp).max():.2e}"
          `,
        },
        {
          name: 'h=1 退化为单头',
          hidden: true,
          code: py`
            x, Wq, Wk, Wv, Wo = _rand(2, 1, 5, 6)
            assert np.allclose(mha(x, Wq, Wk, Wv, Wo, 1), _ref_mha(x, Wq, Wk, Wv, Wo, 1), atol=1e-8)
          `,
        },
      ],
      outro: md`
        这就是面试里最常见的「手撕 MHA」。实战题会加上 bias、返回注意力权重，以及 encoder-decoder 的 cross-attention 与 key padding。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-mha',
      title: '手撕 Multi-Head Attention（返回注意力权重）',
      difficulty: 'medium',
      source: '大模型岗手撕 · MHA',
      timeLimitMin: 20,
      description: md`
        用 NumPy 实现多头自注意力：

        \`\`\`python
        def mha_forward(x, params, n_heads, causal=True) -> tuple[np.ndarray, np.ndarray]:
            # 返回 (out, attn)：out (B, T, D)，attn (B, n_heads, T, T)
        \`\`\`

        - \`x\`: \`(B, T, D)\`；\`params\` 是 dict，含 \`Wq, Wk, Wv, Wo\`（均为 \`(D, D)\`，约定 \`x @ W\`），以及**可选**的 \`bq, bk, bv, bo\`（形状 \`(D,)\`，不存在则视为无 bias）。
        - 第 $i$ 个头使用 \`Wq[:, i*d_h:(i+1)*d_h]\` 等列块；缩放 $1/\sqrt{d_h}$。
        - \`attn\` 为 softmax 之后的权重；\`causal=True\` 时上三角为 0。
        - **不允许**对头做 Python 循环（用 reshape/transpose 批量计算）。

        **示例**：\`n_heads = 1\` 时结果应与单头 SDPA + 输出投影完全一致。
      `,
      starter: py`
        import numpy as np

        def mha_forward(x, params, n_heads, causal=True):
            B, T, D = x.shape
            d_h = D // n_heads
            # TODO: 投影(+bias) → 拆头 → 打分/掩码/softmax → 加权 → 合头 → 输出投影(+bias)
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def mha_forward(x, params, n_heads, causal=True):
            B, T, D = x.shape
            d_h = D // n_heads

            def proj(name):
                y = x @ params["W" + name]
                b = params.get("b" + name)
                return y if b is None else y + b

            def split(t):                                         # (B, T, D) -> (B, h, T, d_h)
                return t.reshape(B, T, n_heads, d_h).transpose(0, 2, 1, 3)

            q, k, v = split(proj("q")), split(proj("k")), split(proj("v"))
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_h)          # (B, h, T, T)
            if causal:
                s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            attn = np.exp(s)
            attn /= attn.sum(axis=-1, keepdims=True)
            o = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)  # 合头
            out = o @ params["Wo"]
            if params.get("bo") is not None:
                out = out + params["bo"]
            return out, attn
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import math
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MultiHeadAttention(nn.Module):
                def __init__(self, d_model: int, n_heads: int, bias: bool = False):
                    super().__init__()
                    assert d_model % n_heads == 0
                    self.h, self.d_h = n_heads, d_model // n_heads
                    self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)   # 融合 QKV 投影
                    self.o = nn.Linear(d_model, d_model, bias=bias)

                def forward(self, x: torch.Tensor, causal: bool = True):
                    B, T, D = x.shape
                    q, k, v = self.qkv(x).split(D, dim=-1)
                    q, k, v = (t.view(B, T, self.h, self.d_h).transpose(1, 2) for t in (q, k, v))
                    s = q @ k.transpose(-2, -1) / math.sqrt(self.d_h)
                    if causal:
                        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                        s = s.masked_fill(~mask, float("-inf"))
                    attn = s.softmax(dim=-1)
                    o = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)   # 注意 contiguous
                    return self.o(o), attn
                    # 生产中：o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
          `,
        },
      ],
      explanation: md`
        **步骤**：投影（+bias）→ \`reshape(B, T, h, d_h).transpose(0, 2, 1, 3)\` 拆头 → $(B, h, T, T)$ 打分、缩放 $\sqrt{d_h}$、因果掩码、稳定 softmax → \`attn @ v\` → \`transpose(0, 2, 1, 3).reshape(B, T, D)\` 合头 → $W_O$（+bias）。

        **复杂度**：时间 $O(BTD^2 + BT^2D)$，额外空间 $O(BhT^2)$（注意力权重）。

        **易错点**：拆/合头的 reshape 与 transpose 顺序；缩放用 $\sqrt{D}$；bias 的 \`None\` 判断；返回的 \`attn\` 必须是 softmax 之后的权重。
      `,
      complexity: { time: 'O(B·T·D² + B·T²·D)', space: 'O(B·h·T²)' },
      hints: [
        md`先写一个 \`split(t)\`：\`t.reshape(B, T, n_heads, d_h).transpose(0, 2, 1, 3)\`。`,
        md`bias 用 \`params.get("bq")\` 取，可能为 \`None\`。`,
        md`合头：\`(attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)\`，最后 \`@ Wo\` 再加 \`bo\`。`,
      ],
      setup: py`
        import numpy as np, inspect

        def _ref(x, P, h, causal=True):
            B, T, D = x.shape
            dh = D // h
            out = np.zeros((B, T, D)); att = np.zeros((B, h, T, T))
            g = lambda n: P.get(n) if P.get(n) is not None else 0.0
            for b in range(B):
                heads = []
                for i in range(h):
                    sl = slice(i * dh, (i + 1) * dh)
                    q = x[b] @ P["Wq"][:, sl] + (g("bq")[sl] if P.get("bq") is not None else 0)
                    k = x[b] @ P["Wk"][:, sl] + (g("bk")[sl] if P.get("bk") is not None else 0)
                    v = x[b] @ P["Wv"][:, sl] + (g("bv")[sl] if P.get("bv") is not None else 0)
                    s = q @ k.T / np.sqrt(dh)
                    if causal:
                        s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
                    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
                    att[b, i] = p
                    heads.append(p @ v)
                out[b] = np.concatenate(heads, -1) @ P["Wo"] + g("bo")
            return out, att

        def _params(seed, D, bias=False):
            r = np.random.default_rng(seed)
            P = {n: r.normal(size=(D, D)) / np.sqrt(D) for n in ("Wq", "Wk", "Wv", "Wo")}
            if bias:
                P.update({n: r.normal(size=D) * 0.1 for n in ("bq", "bk", "bv", "bo")})
            return P

        def _check(seed, B, T, D, h, bias=False, causal=True):
            x = np.random.default_rng(seed + 100).normal(size=(B, T, D))
            P = _params(seed, D, bias)
            out, att = mha_forward(x, P, h, causal)
            eo, ea = _ref(x, P, h, causal)
            assert out.shape == (B, T, D), f"out 形状应为 {(B, T, D)}，得到 {out.shape}"
            assert att.shape == (B, h, T, T), f"attn 形状应为 {(B, h, T, T)}，得到 {att.shape}"
            assert np.allclose(att, ea, atol=1e-8), f"注意力权重不一致，最大误差 {np.abs(att - ea).max():.2e}（检查拆头与缩放）"
            assert np.allclose(out, eo, atol=1e-8), f"输出不一致，最大误差 {np.abs(out - eo).max():.2e}"
      `,
      tests: [
        { name: 'B=1, T=4, D=8, h=2', code: py`_check(0, 1, 4, 8, 2)` },
        { name: '带 bias', code: py`_check(1, 2, 5, 12, 3, bias=True)` },
        {
          name: '因果：上三角权重为 0',
          code: py`
            x = np.random.default_rng(5).normal(size=(1, 6, 8))
            _, att = mha_forward(x, _params(5, 8), 4)
            assert np.allclose(np.triu(att[0, 0], 1), 0), "因果掩码失效：上三角权重不为 0"
            assert np.allclose(att.sum(-1), 1), "每行权重之和应为 1"
          `,
        },
        { name: '非因果 + h=1', hidden: true, code: py`_check(2, 2, 7, 6, 1, causal=False)` },
        { name: '较大规模 B=2, T=16, D=64, h=8', hidden: true, code: py`_check(3, 2, 16, 64, 8, bias=True)` },
        {
          name: '不对头做 Python 循环',
          hidden: true,
          code: py`
            src = inspect.getsource(mha_forward)
            assert "range(n_heads)" not in src and "range(h)" not in src, "请用 reshape/transpose 批量处理所有头，不要循环"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-fused',
          q: md`为什么工程实现里常把 Q、K、V 三个投影融合成一个 \`(D, 3D)\` 的线性层？`,
          a: md`一次大矩阵乘比三次小矩阵乘更能吃满 GPU（更少的 kernel 启动、更好的访存局部性）。输出再 \`split\` 成三份即可，数学上完全等价。GQA 时融合矩阵形状变为 $(D, (h + 2n_{kv})d_h)$。`,
        },
      ],
    },
    {
      id: 'c-cross',
      title: '带 key padding 的多头 Cross-Attention',
      difficulty: 'medium',
      source: '大模型岗手撕 · Encoder-Decoder / 多模态',
      timeLimitMin: 20,
      description: md`
        Encoder-decoder（T5、Whisper）与很多多模态模型（文本 query 看图像 token）都用 cross-attention：**query 来自一个序列，key/value 来自另一个序列**。实现

        \`\`\`python
        def cross_attention(x_q, x_kv, params, n_heads, kv_lengths) -> np.ndarray:   # (B, Tq, D)
        \`\`\`

        - \`x_q\`: \`(B, Tq, D)\`；\`x_kv\`: \`(B, Tk, D)\`（$T_q$ 与 $T_k$ 可以不同）。
        - \`params\`：\`Wq, Wk, Wv, Wo\`，均为 \`(D, D)\`，无 bias。
        - \`kv_lengths[b]\`：第 $b$ 个样本中前 \`kv_lengths[b]\` 个 key 有效，其余为 padding（\`kv_lengths[b] >= 1\`）。
        - **非因果**；padding key 的注意力权重必须为 0。
        - 输出与 padding 位置的 \`x_kv\` 取值无关（隐藏测试会把 padding 位置改成随机大数）。
      `,
      starter: py`
        import numpy as np

        def cross_attention(x_q, x_kv, params, n_heads, kv_lengths):
            # TODO: q 来自 x_q，k/v 来自 x_kv；按 kv_lengths 构造 (B, 1, 1, Tk) 的 key 掩码
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def cross_attention(x_q, x_kv, params, n_heads, kv_lengths):
            B, Tq, D = x_q.shape
            Tk = x_kv.shape[1]
            d_h = D // n_heads

            def split(t, T):
                return t.reshape(B, T, n_heads, d_h).transpose(0, 2, 1, 3)

            q = split(x_q @ params["Wq"], Tq)                      # (B, h, Tq, d_h)
            k = split(x_kv @ params["Wk"], Tk)                     # (B, h, Tk, d_h)
            v = split(x_kv @ params["Wv"], Tk)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_h)          # (B, h, Tq, Tk)
            valid = np.arange(Tk)[None, :] < np.asarray(kv_lengths)[:, None]   # (B, Tk)
            s = np.where(valid[:, None, None, :], s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            w = np.exp(s)
            w /= w.sum(axis=-1, keepdims=True)
            o = (w @ v).transpose(0, 2, 1, 3).reshape(B, Tq, D)
            return o @ params["Wo"]
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch
            import torch.nn.functional as F

            def cross_attention(x_q, x_kv, Wq, Wk, Wv, Wo, n_heads, kv_lengths):
                B, Tq, D = x_q.shape
                Tk, d_h = x_kv.size(1), D // n_heads
                split = lambda t: t.view(B, -1, n_heads, d_h).transpose(1, 2)
                q, k, v = split(x_q @ Wq), split(x_kv @ Wk), split(x_kv @ Wv)
                valid = torch.arange(Tk, device=x_q.device)[None] < kv_lengths[:, None]   # (B, Tk)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])  # True = 参与
                return o.transpose(1, 2).reshape(B, Tq, D) @ Wo
          `,
        },
      ],
      explanation: md`
        与自注意力的唯一区别是 q 与 k/v 的来源和长度不同：分数矩阵是 $(B, h, T_q, T_k)$。key padding 掩码由 \`arange(Tk) < kv_lengths\` 得到 $(B, T_k)$，扩成 $(B, 1, 1, T_k)$ 广播到所有头与所有 query。

        **为什么输出与 padding 处的取值无关**：被遮位置的权重恰为 0，$0 \times$ 有限值 $= 0$。（若 padding 处是 NaN/inf 则会污染——这也是数据管线里 pad 值要填 0 的原因。）

        **推理优化**：cross-attention 的 K、V 只依赖 encoder 输出，解码每一步都相同，可以在第一步算好后一直缓存（Whisper、T5 解码都这么做）。
      `,
      complexity: { time: 'O(B·(Tq+Tk)·D² + B·Tq·Tk·D)', space: 'O(B·h·Tq·Tk)' },
      hints: [
        md`拆头函数要接收长度参数：q 用 $T_q$，k/v 用 $T_k$。`,
        md`\`valid = np.arange(Tk)[None, :] < np.asarray(kv_lengths)[:, None]\`，再 \`valid[:, None, None, :]\` 广播。`,
      ],
      setup: py`
        import numpy as np

        def _ref_cross(xq, xkv, P, h, lens):
            B, Tq, D = xq.shape
            dh = D // h
            out = np.zeros((B, Tq, D))
            for b in range(B):
                n = lens[b]
                heads = []
                for i in range(h):
                    sl = slice(i * dh, (i + 1) * dh)
                    q = xq[b] @ P["Wq"][:, sl]
                    k = xkv[b, :n] @ P["Wk"][:, sl]
                    v = xkv[b, :n] @ P["Wv"][:, sl]
                    s = q @ k.T / np.sqrt(dh)
                    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
                    heads.append(p @ v)
                out[b] = np.concatenate(heads, -1) @ P["Wo"]
            return out

        def _case(seed, B, Tq, Tk, D, h, lens):
            r = np.random.default_rng(seed)
            P = {n: r.normal(size=(D, D)) / np.sqrt(D) for n in ("Wq", "Wk", "Wv", "Wo")}
            return r.normal(size=(B, Tq, D)), r.normal(size=(B, Tk, D)), P

        def _cmp(seed, B, Tq, Tk, D, h, lens):
            xq, xkv, P = _case(seed, B, Tq, Tk, D, h, lens)
            got = cross_attention(xq, xkv, P, h, lens)
            exp = _ref_cross(xq, xkv, P, h, lens)
            assert got.shape == (B, Tq, D), f"形状应为 {(B, Tq, D)}，得到 {got.shape}"
            assert np.allclose(got, exp, atol=1e-8), f"最大误差 {np.abs(got - exp).max():.2e}"
      `,
      tests: [
        { name: 'Tq ≠ Tk，无 padding', code: py`_cmp(0, 1, 3, 5, 8, 2, [5])` },
        { name: '有 padding', code: py`_cmp(1, 2, 4, 6, 8, 4, [6, 2])` },
        {
          name: 'padding 位置的取值不影响输出',
          hidden: true,
          code: py`
            xq, xkv, P = _case(2, 2, 3, 7, 12, 3, [4, 7])
            a = cross_attention(xq, xkv, P, 3, [4, 7])
            xkv2 = xkv.copy()
            xkv2[0, 4:] = np.random.default_rng(9).normal(size=(3, 12)) * 1e3
            b = cross_attention(xq, xkv2, P, 3, [4, 7])
            assert np.allclose(a, b, atol=1e-8), "改变 padding 位置的 x_kv 后输出变了：padding key 没有被正确屏蔽"
          `,
        },
        { name: '随机对拍', hidden: true, code: py`_cmp(3, 3, 5, 9, 16, 4, [1, 9, 5])` },
      ],
      followUps: [
        {
          id: 'f-cache',
          q: md`解码器做 cross-attention 时，KV cache 有什么不同？`,
          a: md`cross-attention 的 K、V 来自 encoder 输出，生成过程中**不随步数增长**，算一次即可复用全部步；而 self-attention 的 KV cache 每步追加一个 token。所以 encoder-decoder 模型的 cache 分两部分：静态的 cross KV 与动态增长的 self KV。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-split',
      freq: 3,
      q: md`手撕 MHA 时拆头为什么要 reshape 之后再 transpose？直接 reshape 成 \`(B, h, T, d_h)\` 行不行？`,
      a: md`
        **结论**：不行，形状对但语义错。

        **原因**：reshape 按内存的行优先顺序切分。$(B, T, D)$ 中同一个 token 的 $D$ 维是连续存放的，所以只能先把**最后一维** $D$ 切成 $(h, d_h)$ 得到 $(B, T, h, d_h)$，此时第 $i$ 个头对应每个 token 的第 $i$ 段维度；再 transpose 把 $h$ 提到 $T$ 前面。直接 reshape 成 $(B, h, T, d_h)$，「头 0」拿到的是前 $T/h$ 个 token 的全部维度。

        **延伸**：合头是逆过程（先 transpose 再 reshape）；PyTorch 中 transpose 后张量不连续，\`view\` 会报错，需要 \`.contiguous()\` 或 \`.reshape()\`。
      `,
    },
    {
      id: 'qa-why-heads',
      freq: 3,
      q: md`为什么用多头注意力？头数越多越好吗？`,
      a: md`
        **结论**：多头让模型在多个子空间中同时关注不同位置；头数并非越多越好。

        **原因**：单头时每个 query 只有一个 softmax 分布，多种关系只能被平均；多头把 $D$ 维切成 $h$ 份，每个头是一个秩至多 $d_h$ 的双线性相似度，各自 softmax，再由 $W_O$ 组合。参数量与 FLOPs 与单头相同。头太多则 $d_h$ 太小，单头表达能力受限（low-rank bottleneck）；经验上 $d_h$ 取 64–128，现代大模型几乎都是 128。

        **延伸**：多个头的 KV 要缓存，推理时 KV cache 与 KV 头数成正比——这引出了 MQA / GQA（共享 KV 头）与 MLA（低秩压缩 KV）。
      `,
    },
    {
      id: 'qa-params',
      freq: 2,
      q: md`MHA 的参数量和 FLOPs 怎么算？什么时候注意力本身成为计算瓶颈？`,
      a: md`
        **结论**：参数 $4D^2$（+$4D$ bias），与头数无关；前向 FLOPs $8TD^2 + 4T^2D$（$B=1$）。

        **原因**：Q/K/V/O 四个投影各 $2TD^2$；$QK^\top$ 与 $PV$ 在所有头上合计各 $2T^2D$。两者相等于 $T = 2D$，所以 $D=4096$ 的模型在上下文超过约 8K 后，注意力矩阵乘才超过投影（还没算 FFN，整层看交叉点更靠后）。

        **延伸**：因果注意力实际可以跳过一半；训练 FLOPs 约为前向 3 倍。估算整模型时常用 $6ND$（训练）与 $2N$/token（推理）再加上注意力项约 $6\, n_{\text{layer}} T D$ FLOPs/token（训练、因果、按位置平均）。
      `,
    },
    {
      id: 'qa-wo',
      freq: 2,
      q: md`MHA 里的 $W_O$ 起什么作用？能去掉吗？`,
      a: md`
        **结论**：$W_O$ 是各头输出在本层内唯一的混合点，不能简单去掉。

        **原因**：$\operatorname{Concat}(o_i) W_O = \sum_i o_i W_O^{(i)}$，每个头经自己的行块映射回 $D$ 维后相加。没有 $W_O$，拼接后的向量里不同头的信息分处不同维度、互不交互，只能等后面的 FFN 混合；且每个头的输出只能落在固定的 $d_h$ 维子空间里。

        **延伸**：$W_V^{(i)} W_O^{(i)}$ 合起来是一个秩 $\le d_h$ 的 $D\times D$ 矩阵（「OV 电路」）。MLA 推理时把 $W_{UV}$ 吸收进 $W_O$，用的就是这种「按头分块的矩阵乘可以结合」的性质。
      `,
    },
  ],

  cheatsheet: md`
    \`\`\`python
    split = lambda t: t.reshape(B, T, h, d_h).transpose(0, 2, 1, 3)   # 拆头
    q, k, v = split(x @ Wq), split(x @ Wk), split(x @ Wv)              # (B,h,T,d_h)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_h)                     # (B,h,T,T)
    o = softmax(masked(s)) @ v                                         # (B,h,T,d_h)
    o = o.transpose(0, 2, 1, 3).reshape(B, T, D)                       # 合头
    out = o @ Wo
    \`\`\`

    - 拆头 reshape→transpose，合头 transpose→reshape；PyTorch 合头要 \`.contiguous()\`
    - 缩放用 $\sqrt{d_h}$；$d_h$ 现代大模型几乎都是 128
    - 参数 $4D^2$（与 $h$ 无关）；FLOPs $8TD^2 + 4T^2D$，交叉点 $T=2D$
    - $\text{Concat}(o_i)W_O = \sum_i o_i W_O^{(i)}$（MLA 吸收的基础）
    - GPT-2 small 每层注意力参数：2,362,368
  `,
})

