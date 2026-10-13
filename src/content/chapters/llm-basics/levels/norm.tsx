/**
 * llm-basics-norm · LayerNorm 与 RMSNorm
 * 公式与可学习参数、LN vs BN、RMSNorm、Pre-LN vs Post-LN、LN 反向、QK-Norm。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { NormDistViz, NormDistRmsViz } from '../viz/NormDistViz'
import { NormAxisViz } from '../viz/NormAxisViz'
import { PrePostViz } from '../viz/PrePostViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'ln',
      title: 'LayerNorm：每个 token 自己做标准化',
      body: md`
        深层网络里，每层输入的尺度会随参数更新不断漂移，后面的层被迫追着前面的层重新适应；激活过大还会让 softmax、指数运算溢出。
        归一化层把每个 token 的隐藏向量 $x \in \mathbb{R}^d$ 拉回固定尺度：

        $$
        \mu = \frac{1}{d}\sum_{i} x_i,\quad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2,\quad
        y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
        $$

        - 统计量沿**最后一维**（隐藏维 $d$）计算，每个 token 一组，与 batch 中其它 token 无关；
        - 方差是**有偏**的（除以 $d$，不是 $d - 1$）——\`torch.nn.LayerNorm\` 与 \`np.var\` 默认都是如此；
        - $\varepsilon$ 加在**根号里**（PyTorch 默认 $10^{-5}$），防止常数向量 $\sigma = 0$ 时除零；
        - 可学习参数 $\gamma, \beta \in \mathbb{R}^d$，共 $2d$ 个，初始化为 $\gamma = 1, \beta = 0$。它们让网络在需要时恢复原来的尺度与偏移，归一化不会损失表达能力。

        拖动输入的平移与缩放，验证 LayerNorm 的输出不变。
      `,
      viz: NormDistViz,
      after: md`
        > [!NOTE] 不变性 = 梯度的“盲区”
        > 因为 $\mathrm{LN}(a x + b) = \mathrm{LN}(x)$（$a>0$），损失对 $x$ 的梯度**没有沿 $\mathbf{1}$ 方向和沿 $\hat{x}$ 方向的分量**。后面推导反向时会看到这一点：$\sum_j \mathrm{d}x_j = 0$。
      `,
      checkpoint: {
        id: 'cp-ln',
        type: 'fill',
        prompt: md`对 $x = [1, 2, 3, 4]$ 做 LayerNorm（$\gamma=1, \beta=0, \varepsilon=0$）：$\mu$ = ____(1)____，$\sigma^2$ = ____(2)____，第一个元素的输出 $y_1$ = ____(3)____（保留两位小数）。`,
        blanks: [
          { label: '(1)', answers: ['2.5'], numeric: true, tolerance: 0.001 },
          { label: '(2)', answers: ['1.25'], numeric: true, tolerance: 0.001 },
          { label: '(3)', answers: ['-1.34'], numeric: true, tolerance: 0.005 },
        ],
        explanation: md`$\mu = 2.5$；$\sigma^2 = \frac{1}{4}(2.25 + 0.25 + 0.25 + 2.25) = 1.25$（有偏方差，不是 $5/3$）；$y_1 = (1 - 2.5)/\sqrt{1.25} \approx -1.34$。`,
      },
    },
    {
      id: 'axis',
      title: 'LN vs BN：只差一个轴，命运截然不同',
      body: md`
        对形状 \`(B, T, d)\` 的激活：

        | | BatchNorm | LayerNorm |
        |---|---|---|
        | 统计范围 | 同一通道、跨 batch 与位置（$B \times T$ 个数） | 同一 token、跨全部通道（$d$ 个数） |
        | 依赖 batch | 是：输出取决于同 batch 的其它样本 | 否 |
        | 推理时 | 用训练时累积的滑动平均 | 与训练完全相同 |
        | 变长 / padding | padding 会污染统计量 | 不受影响 |

        语言模型几乎不用 BN，原因很具体：
        1. **自回归解码时 batch 统计量没有意义**：一次只生成一个 token，batch 里是不同请求，统计量和训练时完全不同；
        2. **变长序列与 padding**：padding 位置会被算进均值方差（见下图开关），需要额外的 mask 逻辑；
        3. **大模型每卡 micro-batch 很小**，BN 的统计量噪声大，还要跨卡同步（SyncBN）带来通信；
        4. 训练与推理行为不一致，本身就是 bug 的温床。

        点击矩阵中的格子，比较两种归一化各用哪些元素求统计量。
      `,
      viz: NormAxisViz,
      checkpoint: {
        id: 'cp-axis',
        type: 'single',
        prompt: md`激活形状为 \`(B, T, d) = (8, 1024, 4096)\`。LayerNorm 一共计算多少组 $(\mu, \sigma)$？`,
        options: ['4096 组', '8 组', '8 × 1024 = 8192 组', '1 组'],
        answer: 2,
        explanation: md`LayerNorm 对每个 token 单独统计，共 $B \times T = 8192$ 组，每组用 $d = 4096$ 个数。A 是 BatchNorm 的组数（每个通道一组）。`,
      },
    },
    {
      id: 'rms',
      title: 'RMSNorm：去掉减均值，省一半统计量',
      body: md`
        RMSNorm（Zhang & Sennrich, 2019）只做缩放，不做平移：

        $$
        y = \gamma \odot \frac{x}{\mathrm{RMS}(x)},\qquad \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}
        $$

        - **没有 $\beta$**，参数只有 $\gamma \in \mathbb{R}^d$；
        - 只需一次归约（平方和），不用先求均值再求方差，kernel 更简单更快；
        - 论文的论点是：LayerNorm 的收益主要来自**缩放不变性**，而“减均值”带来的平移不变性可有可无。

        LLaMA 系列、Qwen、Mistral、DeepSeek、Kimi 等主流开源 LLM 都使用 RMSNorm（T5 更早就用了不减均值、无 bias 的简化 LN）。工程细节：
        - LLaMA 的实现会先把输入转成 **fp32** 求平方均值，再转回 bf16（平方和在 bf16 下容易损失精度）；
        - Gemma 把权重参数化为 $(1 + \gamma)$，$\gamma$ 初始化为 0，便于对它施加权重衰减。

        下图默认把输入整体平移了 $b = 3$：RMSNorm 的输出被共同的偏移量占据，原本的起伏被压扁——它**不具备平移不变性**。把 $b$ 拖回 0，它就和 LayerNorm 几乎一样。实际网络里激活的均值通常接近 0，所以这不构成问题。
      `,
      viz: NormDistRmsViz,
      checkpoint: {
        id: 'cp-rms',
        type: 'judge',
        prompt: md`RMSNorm 满足 $\mathrm{RMSNorm}(a x) = \mathrm{RMSNorm}(x)$（$a > 0$，忽略 $\varepsilon$），但一般不满足 $\mathrm{RMSNorm}(x + b\mathbf{1}) = \mathrm{RMSNorm}(x)$。`,
        answer: true,
        explanation: md`缩放时分子分母同乘 $a$；平移时分子变成 $x + b$，而 $\mathrm{RMS}(x + b\mathbf{1}) \ne \mathrm{RMS}(x)$，输出改变。只有 LayerNorm 同时具备两种不变性。`,
      },
    },
    {
      id: 'prepost',
      title: 'Pre-LN vs Post-LN：归一化放在残差的哪一侧',
      body: md`
        $$
        \text{Post-LN:}\ x_{l+1} = \mathrm{LN}\big(x_l + F(x_l)\big)\qquad
        \text{Pre-LN:}\ x_{l+1} = x_l + F\big(\mathrm{LN}(x_l)\big)
        $$

        原始 Transformer 与 BERT 用 Post-LN；GPT-2 起改为 Pre-LN，并在最后输出头之前加一个 **final norm**，此后的 LLM 几乎都是 Pre-Norm。

        **为什么 Pre-LN 更好训**：Pre-LN 的残差路径上没有任何非线性，$\partial x_L / \partial x_l$ 里始终有一项单位矩阵，梯度可以直达浅层。Xiong 等（2020）分析了初始化时的梯度：Post-LN 靠近输出层的参数梯度很大，必须用学习率 warmup 才能稳定；Pre-LN 的梯度在各层分布均匀，去掉 warmup 也能训练。

        **Pre-LN 的代价**：残差流只加不减，范数随深度增长（下图 ≈ $\sqrt{l}$），于是第 $l$ 层的改动只占残差流的 ≈ $1/\sqrt{l}$——越深的层“话语权”越小。Post-LN 每层都重新归一化，没有这个问题，但深了就难训练。

        后续的折中方案：DeepNorm（Post-LN + 放大残差分支，训练到 1000 层）、Sandwich/Peri-LN（子层输入输出都做 norm，如 Gemma 2/3）、OLMo 2 的“reordered norm”（只对子层输出做 norm：$x + \mathrm{Norm}(F(x))$）。
      `,
      viz: PrePostViz,
      after: md`
        > [!INTERVIEW] 面试官想听到的
        > “Pre-LN 的恒等残差路径让梯度稳定、可以不依赖 warmup，所以大模型都用 Pre-Norm；代价是残差流范数随深度增长，深层的相对贡献变小，并且最后一定要加 final norm，否则输出头看到的是没归一化、范数很大的残差流。”
      `,
      checkpoint: {
        id: 'cp-prepost',
        type: 'single',
        prompt: md`Pre-LN 结构中，如果忘了在 LM head 之前加 final norm，最直接的后果是：`,
        options: [
          '参数量减少导致模型能力下降',
          'LM head 的输入是范数随层数增长、未归一化的残差流，logits 尺度失控',
          '注意力无法计算',
          '反向传播时梯度变为 0',
        ],
        answer: 1,
        explanation: md`Pre-LN 每一层只往残差流里加东西，最后一层的残差流从未被归一化过，范数大且随深度变化；final norm 的作用就是给输出头一个尺度稳定的输入。参数量只差 $d$ 个（A 不是关键）；C、D 都不会发生。`,
      },
    },
    {
      id: 'backward',
      title: 'LayerNorm 的反向：手撕高频题',
      body: md`
        记 $r = (\sigma^2 + \varepsilon)^{-1/2}$，$\hat{x} = (x - \mu)\,r$，$y = \gamma \odot \hat{x} + \beta$，上游梯度 $\mathrm{d}y$。

        **参数梯度**（对所有 token 求和）：$\mathrm{d}\beta = \sum \mathrm{d}y$，$\mathrm{d}\gamma = \sum \mathrm{d}y \odot \hat{x}$。

        **输入梯度**：令 $g = \mathrm{d}y \odot \gamma$（即 $\partial\mathcal{L}/\partial\hat{x}$）。关键是 $\mu$ 和 $\sigma$ 也依赖 $x$：

        $$
        \frac{\partial \hat{x}_i}{\partial x_j} = r\Big(\delta_{ij} - \frac{1}{d} - \frac{\hat{x}_i \hat{x}_j}{d}\Big)
        $$

        三项分别来自：$x$ 本身、$\mu$ 对 $x$ 的依赖、$\sigma$ 对 $x$ 的依赖。代入链式法则：

        $$
        \mathrm{d}x = r \Big( g - \overline{g} - \hat{x} \odot \overline{g \odot \hat{x}} \Big)
        $$

        上划线表示沿隐藏维求平均。**自检**：因为 $\sum_j \hat{x}_j = 0$，所以 $\sum_j \mathrm{d}x_j = 0$——正对应前向的平移不变性。

        RMSNorm 没有 $\mu$，第二项消失：$\mathrm{d}x = r\big(g - \hat{x} \odot \overline{g \odot \hat{x}}\big)$，其中 $r = (\overline{x^2} + \varepsilon)^{-1/2}$，$\hat{x} = x r$。

        > [!WARN] 两个最常见的错误
        > 1. 把 $\mu, \sigma$ 当常数，得到 $\mathrm{d}x = r\,g$——少了两项，数值梯度检查一跑就露馅；
        > 2. $\mathrm{d}\gamma, \mathrm{d}\beta$ 忘了对 batch 与序列维求和，形状对不上 \`(d,)\`。
      `,
      checkpoint: {
        id: 'cp-backward',
        type: 'numeric',
        prompt: md`$x = [1, 2, 3, 4]$，$\gamma = 1$，$\varepsilon = 0$，上游梯度 $\mathrm{d}y = [1, 0, 0, 0]$。求 $\mathrm{d}x_1$（保留 3 位小数）。`,
        answer: 0.268,
        tolerance: 0.01,
        explanation: md`
          $r = 1/\sqrt{1.25} = 0.894$，$\hat{x} = [-1.342, -0.447, 0.447, 1.342]$，$g = [1, 0, 0, 0]$，$\overline{g} = 0.25$，$\overline{g\hat{x}} = -1.342/4 = -0.335$。
          $\mathrm{d}x_1 = 0.894 \times (1 - 0.25 - (-1.342)(-0.335)) = 0.894 \times 0.30 = 0.268$。完整的 $\mathrm{d}x = [0.268, -0.358, -0.089, 0.179]$，和为 0。
        `,
      },
    },
    {
      id: 'qknorm',
      title: 'QK-Norm：给注意力 logits 加一道保险',
      body: md`
        注意力 logit $q \cdot k / \sqrt{d_h}$ 的大小没有上界：训练中 $W_Q, W_K$ 的范数一旦变大，logits 就会爆炸，softmax 退化成 one-hot，loss 出现尖刺甚至发散。
        **QK-Norm** 在计算点积前，对每个头的 $q$ 和 $k$ 各做一次 RMSNorm / LayerNorm：

        $$
        \mathrm{logit} = \frac{\mathrm{Norm}(q)\cdot \mathrm{Norm}(k)}{\sqrt{d_h}}
        $$

        归一化后 $\|q\|, \|k\|$ 只由 norm 的增益 $\gamma$ 决定，logit 的量级被牢牢控制住。

        | 模型 | 归一化设计 |
        |---|---|
        | ViT-22B（2023） | 引入 QK LayerNorm，解决大规模训练中 logit 发散 |
        | OLMo 2 | QK-Norm + 子层输出端 norm（reordered norm） |
        | Gemma 3 | 用 QK-Norm 替代 Gemma 2 的 logit soft-capping；子层输入、输出都有 RMSNorm |
        | Qwen3 | 去掉 Qwen2 的 QKV bias，引入 QK-Norm |
        | DeepSeek-V4 | 对 CSA/HCA 的每个 query 头与压缩后的 KV 条目做 RMSNorm，论文明确说因此**不需要** QK-Clip |
        | Kimi K2（MLA） | 不能用 QK-Norm：MLA 推理时不显式构造完整的 Key 矩阵，因此改用 MuonClip（QK-Clip），见「Muon 优化器」一关 |

        > [!NEW] 为什么 2025 年 QK-Norm 突然重要
        > Muon 等优化器让权重矩阵的更新更“满秩”，logit 增长更快——Kimi 在中等规模的 Muon 实验中观察到最大 logit 超过 1000。QK-Norm（或 QK-Clip）成了大规模训练的标配保险。
      `,
      checkpoint: {
        id: 'cp-qknorm',
        type: 'single',
        prompt: md`Kimi K2 为什么没有用 QK-Norm，而是设计了 QK-Clip？`,
        options: [
          'QK-Norm 会显著增加参数量',
          'K2 用 MLA，推理时不显式构造完整的 Key 矩阵，QK-Norm 无法施加在 Key 上',
          'QK-Norm 与 RoPE 不兼容',
          'QK-Norm 只适用于 Post-LN 结构',
        ],
        answer: 1,
        explanation: md`K2 技术报告原文的理由：QK-Norm 不适用于 MLA，因为推理时其 Key 矩阵没有被完整地显式构造（KV 以压缩的潜向量缓存，并通过矩阵吸收参与计算）。参数量（A）只增加每头 $2d_h$ 个；C、D 不成立（Qwen3、Gemma 3 都是 RoPE + Pre-Norm + QK-Norm）。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-axis',
      type: 'single',
      difficulty: 1,
      prompt: md`Transformer 中对形状 \`(B, T, d)\` 的激活做 LayerNorm，均值和方差沿哪个维度计算？`,
      options: ['B（batch）', 'T（序列）', 'd（隐藏维）', 'B 和 T 一起'],
      answer: 2,
      explanation: md`LayerNorm 对每个 token 的隐藏向量单独统计。沿 B（或 B、T 一起）统计的是 BatchNorm 的做法；沿 T 统计会让输出依赖序列中其它位置（含未来 token），破坏因果性。`,
    },
    {
      id: 'q-rms-param',
      type: 'judge',
      difficulty: 1,
      prompt: md`RMSNorm 与 LayerNorm 一样有可学习的缩放 $\gamma$ 和偏置 $\beta$。`,
      answer: false,
      explanation: md`RMSNorm 只有 $\gamma$（$d$ 个参数），没有 $\beta$，也不减均值。LLaMA、Qwen、DeepSeek 等模型的 norm 层都只有一个 \`weight\`。`,
    },
    {
      id: 'q-count',
      type: 'numeric',
      difficulty: 2,
      prompt: md`LLaMA-2-7B：32 层，$d = 4096$，每层有 2 个 RMSNorm（注意力前、FFN 前），最后还有 1 个 final norm。所有 RMSNorm 的参数共多少个？`,
      answer: 266240,
      tolerance: 0,
      explanation: md`$(32 \times 2 + 1) \times 4096 = 65 \times 4096 = 266{,}240$。相对 67 亿参数可以忽略，但它们通常**不做权重衰减**、在 Muon 训练中也交给 AdamW 处理（一维参数）。`,
    },
    {
      id: 'q-bn',
      type: 'multi',
      difficulty: 2,
      prompt: md`语言模型很少使用 BatchNorm，原因包括（多选）：`,
      options: [
        '自回归解码时一次只处理一个 token，batch 统计量与训练时不一致',
        '变长序列的 padding 会污染 batch 统计量',
        'BatchNorm 无法求导',
        '大模型每卡 micro-batch 小，统计量噪声大，跨卡同步还有通信开销',
        'BatchNorm 的参数量是 LayerNorm 的 $T$ 倍',
      ],
      answer: [0, 1, 3],
      explanation: md`A、B、D 都是真实的工程问题。C 错：BN 完全可导。E 错：BN 和 LN 的可学习参数都是每通道一个 $\gamma$ 和一个 $\beta$，参数量相同（BN 另有不参与训练的滑动均值与方差缓冲区）。`,
    },
    {
      id: 'q-rms-val',
      type: 'fill',
      difficulty: 2,
      prompt: md`对 $x = [1, 2, 3, 4]$ 做 RMSNorm（$\gamma = 1$，$\varepsilon = 0$）：$\mathrm{RMS}(x)$ = ____(1)____，输出的第 4 个元素 = ____(2)____。（保留三位小数）`,
      blanks: [
        { label: '(1)', answers: ['2.739'], numeric: true, tolerance: 0.002 },
        { label: '(2)', answers: ['1.461'], numeric: true, tolerance: 0.002 },
      ],
      explanation: md`$\mathrm{RMS} = \sqrt{(1 + 4 + 9 + 16)/4} = \sqrt{7.5} \approx 2.739$，$4 / 2.739 \approx 1.461$。对比 LayerNorm 的第 4 个输出 $1.342$：RMSNorm 不减均值，输出整体偏正。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 LayerNorm 前向（\`x\` 形状 \`(N, d)\`）有 **2 处** bug，点出来：`,
      code: py`
        def layer_norm(x, gamma, beta, eps=1e-5):
            mu = x.mean(axis=0, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            xhat = (x - mu) / np.sqrt(var) + eps
            return gamma * xhat + beta
      `,
      bugLines: [2, 4],
      explanation: md`
        - 第 2 行沿 \`axis=0\`（batch）求均值，变成了 BatchNorm 式的统计，且与第 3 行的方差轴不一致。应为 \`axis=-1\`。
        - 第 4 行 \`eps\` 加在了除法**之后**，常数向量时 \`var = 0\` 仍会除零。应为 \`(x - mu) / np.sqrt(var + eps)\`。
        第 3 行 \`x.var\` 默认 \`ddof=0\`（有偏），正是 LayerNorm 需要的，没有问题。
      `,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 LLaMA 式 Pre-Norm decoder block 的前向步骤排好序（输入为残差流 $x$）：`,
      items: [
        '$h = \\mathrm{RMSNorm}_1(x)$',
        '$x = x + \\mathrm{Attention}(h)$',
        '$h = \\mathrm{RMSNorm}_2(x)$',
        '$x = x + \\mathrm{FFN}(h)$',
        '（全部层之后）$\\mathrm{FinalNorm}(x)$ 送入 LM head',
      ],
      explanation: md`Pre-Norm：每个子层先对残差流做 norm，子层输出再**加回**残差流；残差流本身从不被归一化，直到最后的 final norm。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把归一化设计与采用它的代表模型配对：`,
      pairs: [
        ['Post-LN', '原始 Transformer、BERT'],
        ['Pre-LN + final LN', 'GPT-2'],
        ['Pre-Norm + RMSNorm', 'LLaMA'],
        ['子层输入、输出都做 RMSNorm', 'Gemma 2 / Gemma 3'],
        ['去掉 QKV bias，加 QK-Norm', 'Qwen3'],
      ],
      explanation: md`这些是各自技术报告 / 论文中写明的设计。共同趋势：从 Post-LN 走向 Pre-Norm，从 LayerNorm 走向 RMSNorm，并在注意力内部（QK）与子层输出端追加 norm 以提升大规模训练的稳定性。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LLaMA 风格的 RMSNorm 前向（输入可能是 bf16，内部用 fp32 计算）。有 2 行干扰项。`,
      lines: [
        'def rms_norm(x, weight, eps=1e-6):',
        '    dtype = x.dtype',
        '    x32 = x.astype(np.float32)',
        '    ms = np.mean(x32 * x32, axis=-1, keepdims=True)',
        '    y = x32 / np.sqrt(ms + eps)',
        '    return (y * weight).astype(dtype)',
      ],
      distractors: ['    ms = np.mean(x32, axis=-1, keepdims=True)', '    y = (x32 - ms) / np.sqrt(ms + eps)'],
      explanation: md`RMSNorm 求的是**平方**的均值，并且不减均值。干扰项 1 求的是普通均值；干扰项 2 把 LayerNorm 的减均值混了进来，而且减的还是均方值。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`激活展平为 4 个 token × 3 个通道。训练模式下的 **BatchNorm** 在归一化 ★ 这个元素（token 2、通道 c2）时，用哪些元素求均值和方差？点出全部。`,
      items: ['·', '·', '·', '·', '·', '·', '·', '·', '★', '·', '·', '·'],
      cols: 3,
      rowLabels: ['token 0', 'token 1', 'token 2', 'token 3'],
      colLabels: ['c0', 'c1', 'c2'],
      answer: [2, 5, 8, 11],
      explanation: md`BatchNorm 按**通道**统计：c2 这一列的全部 token（含 ★ 本身）。若是 LayerNorm，则用 token 2 这一行（下标 6、7、8）。`,
    },
    {
      id: 'q-bwd-wrong',
      type: 'single',
      difficulty: 3,
      prompt: md`某同学推 LayerNorm 反向时把 $\mu$ 和 $\sigma$ 当作常数，得到 $\mathrm{d}x = r \cdot g$（$g = \mathrm{d}y \odot \gamma$）。下列哪项可以**一眼**判断这个结果是错的？`,
      options: [
        '它的形状与 $x$ 不同',
        '它一般不满足 $\\sum_j \\mathrm{d}x_j = 0$，而 LN 的平移不变性要求梯度沿全 1 方向的分量为 0',
        '它对 $\\gamma$ 不可导',
        '当 $g$ 全为正时它会变成负数',
      ],
      answer: 1,
      explanation: md`$\mathrm{LN}(x + b\mathbf{1}) = \mathrm{LN}(x)$，所以 $\mathcal{L}$ 沿 $\mathbf{1}$ 方向不变，$\langle \nabla_x \mathcal{L}, \mathbf{1}\rangle = 0$。$r \cdot g$ 一般不满足，说明漏掉了 $\mu$ 的贡献（$-\overline{g}$ 项）与 $\sigma$ 的贡献（$-\hat{x}\,\overline{g\hat{x}}$ 项）。A、C、D 都不成立。`,
    },
    {
      id: 'q-bwd-num',
      type: 'numeric',
      difficulty: 3,
      prompt: md`接上题的正确公式。$x = [1, 2, 3, 4]$，$\gamma = 1$，$\varepsilon = 0$，$\mathrm{d}y = [0, 1, 0, 0]$（只对第 2 个输出有梯度）。求 $\mathrm{d}x_2$（保留 3 位小数）。`,
      answer: 0.626,
      tolerance: 0.01,
      explanation: md`
        $r = 1/\sqrt{1.25} = 0.8944$，$\hat{x} = [-1.342, -0.447, 0.447, 1.342]$，$g = [0, 1, 0, 0]$，$\overline{g} = 0.25$，$\overline{g\hat{x}} = -0.4472/4 = -0.1118$。
        $\mathrm{d}x_2 = 0.8944 \times \big(1 - 0.25 - (-0.4472)(-0.1118)\big) = 0.8944 \times (0.75 - 0.05) = 0.626$。
        常见错误：漏掉第三项会得到 $0.8944 \times 0.75 = 0.671$；把 $\mu,\sigma$ 都当常数会得到 $r = 0.894$。
      `,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-rmsnorm',
      title: 'RMSNorm 的前向与反向',
      intro: md`
        先在更简单的 RMSNorm 上走一遍“前向缓存 → 反向推导”的完整流程，实战题再挑战 LayerNorm 的反向。
        约定：\`x\` 形状 \`(..., d)\`，归一化沿最后一维；\`gamma\` 形状 \`(d,)\`。前向返回输出和反向需要的缓存。
      `,
      template: py`
        import numpy as np

        def rms_norm_forward(x, gamma, eps=1e-6):
            ms = {{1}}
            rstd = {{2}}
            xhat = x * rstd
            return xhat * gamma, (xhat, rstd)

        def rms_norm_backward(dy, gamma, cache):
            xhat, rstd = cache
            d = xhat.shape[-1]
            dgamma = {{3}}
            g = dy * gamma
            dx = {{4}}
            return dx, dgamma
      `,
      steps: [
        {
          blank: 1,
          instruction: md`沿最后一维求**平方的均值**，保留维度。`,
          answer: 'np.mean(x * x, axis=-1, keepdims=True)',
          accept: ['np.mean(x ** 2, axis=-1, keepdims=True)', '(x * x).mean(axis=-1, keepdims=True)', 'np.mean(np.square(x), axis=-1, keepdims=True)'],
          explain: md`不减均值——这是 RMSNorm 与 LayerNorm 的唯一区别。`,
        },
        {
          blank: 2,
          instruction: md`计算 $r = 1/\sqrt{\overline{x^2} + \varepsilon}$。`,
          answer: '1.0 / np.sqrt(ms + eps)',
          accept: ['1 / np.sqrt(ms + eps)', '(ms + eps) ** -0.5'],
          explain: md`缓存 $r$（而不是 RMS 本身）可以让反向只做乘法。$\varepsilon$ 一定在根号里面。`,
        },
        {
          blank: 3,
          instruction: md`$\gamma$ 的梯度：$\mathrm{d}y \odot \hat{x}$ 对除最后一维以外的所有维求和，结果形状为 \`(d,)\`。`,
          answer: '(dy * xhat).reshape(-1, d).sum(axis=0)',
          accept: ['np.sum((dy * xhat).reshape(-1, d), axis=0)', '(dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))'],
          hint: '先 reshape 成 (-1, d)，再对第 0 维求和。',
          explain: md`同一个 $\gamma$ 被所有 token 共用，所以要把所有 token 的贡献加起来。`,
        },
        {
          blank: 4,
          instruction: md`输入梯度：$\mathrm{d}x = r\big(g - \hat{x} \odot \overline{g \odot \hat{x}}\big)$，均值沿最后一维、保留维度。`,
          answer: 'rstd * (g - xhat * np.mean(g * xhat, axis=-1, keepdims=True))',
          accept: ['rstd * (g - xhat * (g * xhat).mean(axis=-1, keepdims=True))'],
          hint: '第二项来自 RMS 对 x 的依赖。',
          explain: md`
            推导：$\partial \hat{x}_i / \partial x_j = r\,\delta_{ij} - x_i r^3 x_j / d = r(\delta_{ij} - \hat{x}_i\hat{x}_j/d)$，
            于是 $\mathrm{d}x_j = r\big(g_j - \hat{x}_j \cdot \frac{1}{d}\sum_i g_i\hat{x}_i\big)$。LayerNorm 只是多了 $\mu$ 带来的 $-\overline{g}$ 一项。
          `,
        },
      ],
      setup: py`
        def _ref_rms(x, gamma, eps=1e-6):
            return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma

        def _num_grad(f, x, h=1e-6):
            g = np.zeros_like(x)
            for idx in np.ndindex(*x.shape):
                old = x[idx]
                x[idx] = old + h; fp = f(x)
                x[idx] = old - h; fm = f(x)
                x[idx] = old
                g[idx] = (fp - fm) / (2 * h)
            return g
      `,
      tests: [
        {
          name: '前向',
          code: py`
            x = np.array([[1.0, 2.0, 3.0, 4.0]])
            y, _ = rms_norm_forward(x, np.ones(4), eps=0.0)
            exp = np.array([[0.36514837, 0.73029674, 1.09544512, 1.46059349]])
            assert np.allclose(y, exp), f"期望 {exp}，得到 {y}"
          `,
        },
        {
          name: '反向：数值梯度检查',
          code: py`
            rng = np.random.default_rng(0)
            x = rng.normal(size=(2, 3, 5))
            gamma = rng.normal(size=5)
            dy = rng.normal(size=(2, 3, 5))
            y, cache = rms_norm_forward(x, gamma)
            dx, dgamma = rms_norm_backward(dy, gamma, cache)
            assert dx.shape == x.shape and dgamma.shape == (5,), f"形状错误：dx {dx.shape}, dgamma {dgamma.shape}"
            ndx = _num_grad(lambda z: float((_ref_rms(z, gamma) * dy).sum()), x.copy())
            ndg = _num_grad(lambda g_: float((_ref_rms(x, g_) * dy).sum()), gamma.copy())
            assert np.allclose(dx, ndx, atol=1e-6), f"dx 最大误差 {np.abs(dx - ndx).max():.2e}"
            assert np.allclose(dgamma, ndg, atol=1e-6), f"dgamma 最大误差 {np.abs(dgamma - ndg).max():.2e}"
          `,
        },
        {
          name: '缩放不变 ⇒ dx 与 x 正交',
          hidden: true,
          code: py`
            rng = np.random.default_rng(1)
            x = rng.normal(size=(4, 8))
            dy = rng.normal(size=(4, 8))
            _, cache = rms_norm_forward(x, np.ones(8), eps=0.0)
            dx, _ = rms_norm_backward(dy, np.ones(8), cache)
            dots = (dx * x).sum(axis=-1)
            assert np.allclose(dots, 0, atol=1e-10), f"每行 <dx, x> 应为 0，得到 {dots}"
          `,
        },
      ],
      outro: md`
        RMSNorm 对 $x$ 缩放不变，所以梯度与 $x$ 正交（$\langle \mathrm{d}x, x\rangle = 0$，$\varepsilon=0$ 时严格成立）。LayerNorm 还多一个平移不变性，于是梯度还与 $\mathbf{1}$ 正交——实战题的测试会检查这一点。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-norm-forward',
      title: 'LayerNorm 与 RMSNorm 前向',
      difficulty: 'medium',
      source: '大模型岗手撕 · 归一化',
      timeLimitMin: 15,
      description: md`
        实现两个函数，归一化沿**最后一维**，支持任意前导维度（如 \`(B, T, d)\`）：

        \`\`\`python
        def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray: ...
        def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray: ...
        \`\`\`

        - 与 \`torch.nn.LayerNorm\` / LLaMA 的 \`RMSNorm\` 语义一致：方差为**有偏**估计，$\varepsilon$ 加在根号内。
        - \`gamma\`、\`beta\` 形状为 \`(d,)\`。
        - 数值稳健：输入可能均值很大、方差很小（如 $10^6 + \text{噪声}$），也可能是常数向量。

        **示例**：\`layer_norm([[1, 2, 3, 4]], ones, zeros, eps=0)\` → \`[[-1.342, -0.447, 0.447, 1.342]]\`
      `,
      starter: py`
        import numpy as np

        def layer_norm(x, gamma, beta, eps=1e-5):
            # TODO: 沿最后一维求均值与有偏方差，标准化后做仿射变换
            raise NotImplementedError

        def rms_norm(x, gamma, eps=1e-6):
            # TODO: 除以均方根（不减均值），再乘 gamma
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def layer_norm(x, gamma, beta, eps=1e-5):
            x = np.asarray(x, dtype=np.float64)
            mu = x.mean(axis=-1, keepdims=True)
            xc = x - mu                                   # 先中心化再求方差（两遍法，避免抵消误差）
            var = (xc * xc).mean(axis=-1, keepdims=True)  # 有偏方差
            return xc / np.sqrt(var + eps) * gamma + beta

        def rms_norm(x, gamma, eps=1e-6):
            x = np.asarray(x, dtype=np.float64)
            ms = (x * x).mean(axis=-1, keepdims=True)
            return x / np.sqrt(ms + eps) * gamma
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch
            import torch.nn as nn

            class RMSNorm(nn.Module):          # LLaMA 风格
                def __init__(self, dim: int, eps: float = 1e-6):
                    super().__init__()
                    self.eps = eps
                    self.weight = nn.Parameter(torch.ones(dim))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x32 = x.float()             # 在 fp32 中求均方
                    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
                    return y.type_as(x) * self.weight

            def layer_norm(x, gamma, beta, eps=1e-5):
                return torch.nn.functional.layer_norm(x, (x.shape[-1],), gamma, beta, eps)
          `,
        },
      ],
      explanation: md`
        **要点**
        - 方差用有偏估计 \`ddof=0\`；\`np.var\` 默认就是，\`torch.var\` 默认是**无偏**的（\`unbiased=True\` / \`correction=1\`），手写 PyTorch 版时要注意。
        - **不要**用 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ 求方差：当均值远大于标准差时（如 $10^6 + 10^{-3}$ 量级的噪声），两个大数相减发生灾难性抵消，结果可能为负，开根号得到 NaN。先中心化再平方（两遍法）或 Welford 在线算法才稳健；GPU kernel 通常用 Welford。
        - 常数向量：$\sigma = 0$，靠 $\varepsilon$ 避免除零，输出等于 $\beta$。

        **复杂度**：$O(N d)$。
      `,
      complexity: { time: 'O(Nd)', space: 'O(Nd)' },
      hints: [
        md`统计量都用 \`axis=-1, keepdims=True\`，结果可以直接与 \`x\` 广播。`,
        md`方差先算 \`xc = x - mu\`，再 \`(xc * xc).mean(-1)\`；不要用 \`E[x²] − E[x]²\`。`,
        md`RMSNorm：\`x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * gamma\`。`,
      ],
      setup: py`
        def _ref_ln(x, g, b, eps=1e-5):
            x = np.asarray(x, dtype=np.float64)
            mu = x.mean(-1, keepdims=True)
            var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) / np.sqrt(var + eps) * g + b

        def _ref_rms(x, g, eps=1e-6):
            x = np.asarray(x, dtype=np.float64)
            return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g
      `,
      tests: [
        {
          name: '示例',
          code: py`
            got = layer_norm(np.array([[1.0, 2.0, 3.0, 4.0]]), np.ones(4), np.zeros(4), eps=0.0)
            exp = np.array([[-1.34164079, -0.4472136, 0.4472136, 1.34164079]])
            assert got.shape == (1, 4), f"形状应为 (1, 4)，得到 {got.shape}"
            assert np.allclose(got, exp), f"期望 {exp}，得到 {got}"
            got2 = rms_norm(np.array([[1.0, 2.0, 3.0, 4.0]]), np.ones(4), eps=0.0)
            exp2 = np.array([[0.36514837, 0.73029674, 1.09544512, 1.46059349]])
            assert np.allclose(got2, exp2), f"rms_norm 期望 {exp2}，得到 {got2}"
          `,
        },
        {
          name: '有偏方差（d = 2）',
          code: py`
            got = layer_norm(np.array([[0.0, 2.0]]), np.ones(2), np.zeros(2), eps=0.0)
            assert np.allclose(got, [[-1.0, 1.0]]), f"有偏方差下应为 [[-1, 1]]，得到 {got}（用了 ddof=1？）"
          `,
        },
        {
          name: '三维输入 + 仿射参数',
          code: py`
            rng = np.random.default_rng(0)
            x = rng.normal(size=(2, 3, 8)) * 3 + 1
            g, b = rng.normal(size=8), rng.normal(size=8)
            got = layer_norm(x, g, b)
            assert got.shape == (2, 3, 8), f"形状应为 (2, 3, 8)，得到 {got.shape}"
            assert np.allclose(got, _ref_ln(x, g, b)), f"最大误差 {np.abs(got - _ref_ln(x, g, b)).max():.2e}"
            got2 = rms_norm(x, g)
            assert np.allclose(got2, _ref_rms(x, g)), f"rms_norm 最大误差 {np.abs(got2 - _ref_rms(x, g)).max():.2e}"
          `,
        },
        {
          name: '常数向量不出 NaN',
          hidden: true,
          code: py`
            got = layer_norm(np.full((2, 5), 3.0), np.ones(5), np.full(5, 0.5))
            assert np.all(np.isfinite(got)), f"出现 NaN：{got}"
            assert np.allclose(got, 0.5), f"常数输入应输出 beta，得到 {got}"
            got2 = rms_norm(np.zeros((1, 4)), np.ones(4))
            assert np.all(np.isfinite(got2)) and np.allclose(got2, 0), f"零向量应输出 0，得到 {got2}"
          `,
        },
        {
          name: '不变性',
          hidden: true,
          code: py`
            rng = np.random.default_rng(1)
            x = rng.normal(size=(3, 16))
            g, b = rng.normal(size=16), rng.normal(size=16)
            base = layer_norm(x, g, b, eps=0.0)
            assert np.allclose(layer_norm(5.0 * x - 7.0, g, b, eps=0.0), base), "LayerNorm 应对 a·x + b 不变"
            assert np.allclose(rms_norm(4.0 * x, g, eps=0.0), rms_norm(x, g, eps=0.0)), "RMSNorm 应对缩放不变"
          `,
        },
        {
          name: '大均值小方差（数值稳健）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(2)
            x = 1e6 + rng.normal(size=(4, 32)) * 1e-3
            got = layer_norm(x, np.ones(32), np.zeros(32), eps=1e-12)
            exp = _ref_ln(x, np.ones(32), np.zeros(32), eps=1e-12)
            assert np.all(np.isfinite(got)), "出现 NaN：是不是用了 E[x²] − E[x]² 求方差？"
            err = np.abs(got - exp).max()
            assert err < 1e-3, f"最大误差 {err:.2e}：E[x²] − E[x]² 在大均值下会发生灾难性抵消"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-welford',
          q: md`GPU kernel 里 LayerNorm 的均值方差通常怎么算？`,
          a: md`
            一个 block 负责一行（或若干行），用 **Welford 在线算法**（或其并行合并形式）一遍扫描同时得到均值与 $M_2$，再做 warp/block 级归约。它避免了 $\mathbb{E}[x^2]-\mathbb{E}[x]^2$ 的抵消误差，又只读一次显存。
            LayerNorm 是典型的访存受限算子，因此常与残差加法、dropout、量化等**融合**成一个 kernel（如 fused add + RMSNorm）。
          `,
        },
      ],
    },
    {
      id: 'c-ln-backward',
      title: 'LayerNorm 反向传播',
      difficulty: 'hard',
      source: '大模型岗手撕 · 高频',
      timeLimitMin: 25,
      description: md`
        实现 LayerNorm 的反向：

        \`\`\`python
        def layer_norm_backward(dy: np.ndarray, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
        \`\`\`

        - 前向为 $y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta$，归一化沿最后一维，\`x\` 与 \`dy\` 形状 \`(..., d)\`。
        - 返回 \`(dx, dgamma, dbeta)\`：\`dx\` 与 \`x\` 同形状；\`dgamma\`、\`dbeta\` 形状 \`(d,)\`（对所有前导维求和）。
        - 需要自己在函数内重算前向所需的中间量。

        **自检**：每一行 \`dx\` 的和应为 0。
      `,
      starter: py`
        import numpy as np

        def layer_norm_backward(dy, x, gamma, eps=1e-5):
            # TODO: 1) 重算 mu、rstd、xhat
            #       2) dbeta、dgamma（对前导维求和）
            #       3) dx = rstd * (g - mean(g) - xhat * mean(g * xhat))，其中 g = dy * gamma
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def layer_norm_backward(dy, x, gamma, eps=1e-5):
            x = np.asarray(x, dtype=np.float64)
            dy = np.asarray(dy, dtype=np.float64)
            d = x.shape[-1]
            mu = x.mean(axis=-1, keepdims=True)
            xc = x - mu
            rstd = 1.0 / np.sqrt((xc * xc).mean(axis=-1, keepdims=True) + eps)
            xhat = xc * rstd

            dbeta = dy.reshape(-1, d).sum(axis=0)
            dgamma = (dy * xhat).reshape(-1, d).sum(axis=0)

            g = dy * gamma                                            # dL/dxhat
            dx = rstd * (g - g.mean(axis=-1, keepdims=True)
                         - xhat * (g * xhat).mean(axis=-1, keepdims=True))
            return dx, dgamma, dbeta
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：自定义 autograd.Function）',
          code: py`
            import torch

            class LayerNormFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, gamma, beta, eps=1e-5):
                    mu = x.mean(-1, keepdim=True)
                    rstd = torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
                    xhat = (x - mu) * rstd
                    ctx.save_for_backward(xhat, rstd, gamma)
                    return xhat * gamma + beta

                @staticmethod
                def backward(ctx, dy):
                    xhat, rstd, gamma = ctx.saved_tensors
                    d = xhat.shape[-1]
                    g = dy * gamma
                    dx = rstd * (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True))
                    dgamma = (dy * xhat).reshape(-1, d).sum(0)
                    dbeta = dy.reshape(-1, d).sum(0)
                    return dx, dgamma, dbeta, None
          `,
        },
      ],
      explanation: md`
        **推导**：$\hat{x} = (x - \mu) r$，$r = (\sigma^2 + \varepsilon)^{-1/2}$。利用
        $\partial\mu/\partial x_j = 1/d$、$\partial\sigma^2/\partial x_j = 2(x_j - \mu)/d$，得

        $$
        \frac{\partial \hat{x}_i}{\partial x_j} = r\Big(\delta_{ij} - \frac{1}{d} - \frac{\hat{x}_i\hat{x}_j}{d}\Big),\qquad
        \mathrm{d}x = r\Big(g - \overline{g} - \hat{x}\odot\overline{g\odot\hat{x}}\Big)
        $$

        其中 $g = \mathrm{d}y \odot \gamma$。实现时缓存 $\hat{x}$ 与 $r$，反向只需两次行内归约。

        **常见错误**：漏掉 $\mu$ 或 $\sigma$ 对 $x$ 的依赖；\`dgamma\` 写成 \`dy * x\`（应是 \`dy * xhat\`）；忘了对前导维求和；方差用了无偏估计。

        **复杂度**：$O(Nd)$ 时间；需要缓存 $\hat{x}$（$O(Nd)$）与 $r$（$O(N)$）。工业实现常常只缓存 $\mu, r$，反向时重算 $\hat{x}$ 以节省显存。
      `,
      complexity: { time: 'O(Nd)', space: 'O(Nd)' },
      hints: [
        md`先写参数梯度：\`dbeta = dy.reshape(-1, d).sum(0)\`，\`dgamma = (dy * xhat).reshape(-1, d).sum(0)\`。`,
        md`令 \`g = dy * gamma\`，它是对 $\hat{x}$ 的梯度。$\mu$ 贡献 $-\overline{g}$，$\sigma$ 贡献 $-\hat{x}\,\overline{g\hat{x}}$。`,
        md`最终：\`dx = rstd * (g - g.mean(-1, keepdims=True) - xhat * (g * xhat).mean(-1, keepdims=True))\`。`,
      ],
      setup: py`
        def _ref_ln(x, g, b, eps=1e-5):
            mu = x.mean(-1, keepdims=True)
            var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) / np.sqrt(var + eps) * g + b

        def _num_grad(f, x, h=1e-6):
            g = np.zeros_like(x)
            for idx in np.ndindex(*x.shape):
                old = x[idx]
                x[idx] = old + h; fp = f(x)
                x[idx] = old - h; fm = f(x)
                x[idx] = old
                g[idx] = (fp - fm) / (2 * h)
            return g
      `,
      tests: [
        {
          name: '手算示例',
          code: py`
            dx, dg, db = layer_norm_backward(np.array([[1.0, 0.0, 0.0, 0.0]]), np.array([[1.0, 2.0, 3.0, 4.0]]), np.ones(4), eps=0.0)
            exp = np.array([[0.26832816, -0.35777088, -0.08944272, 0.17888544]])
            assert dx.shape == (1, 4), f"dx 形状应为 (1, 4)，得到 {dx.shape}"
            assert np.allclose(dx, exp), f"期望 {exp}，得到 {dx}"
            assert np.allclose(db, [1, 0, 0, 0]), f"dbeta 期望 [1,0,0,0]，得到 {db}"
            assert np.allclose(dg, [-1.34164079, 0, 0, 0]), f"dgamma 期望 [-1.342,0,0,0]，得到 {dg}"
          `,
        },
        {
          name: '形状与 dx 行和为 0',
          code: py`
            rng = np.random.default_rng(0)
            x = rng.normal(size=(2, 3, 6))
            dy = rng.normal(size=(2, 3, 6))
            dx, dg, db = layer_norm_backward(dy, x, rng.normal(size=6))
            assert dx.shape == (2, 3, 6) and dg.shape == (6,) and db.shape == (6,), f"形状错误：{dx.shape}, {dg.shape}, {db.shape}"
            assert np.allclose(dx.sum(axis=-1), 0, atol=1e-10), f"每行 dx 之和应为 0，得到 {dx.sum(axis=-1)}"
          `,
        },
        {
          name: '数值梯度检查（dx）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(1)
            x = rng.normal(size=(3, 5)) * 2 + 1
            gamma, beta = rng.normal(size=5), rng.normal(size=5)
            dy = rng.normal(size=(3, 5))
            dx, _, _ = layer_norm_backward(dy, x, gamma)
            ndx = _num_grad(lambda z: float((_ref_ln(z, gamma, beta) * dy).sum()), x.copy())
            err = np.abs(dx - ndx).max()
            assert err < 1e-6, f"dx 与数值梯度最大误差 {err:.2e}（是否漏了 μ 或 σ 对 x 的依赖？）"
          `,
        },
        {
          name: '数值梯度检查（dgamma / dbeta）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(2)
            x = rng.normal(size=(2, 4, 3))
            gamma, beta = rng.normal(size=3), rng.normal(size=3)
            dy = rng.normal(size=(2, 4, 3))
            _, dg, db = layer_norm_backward(dy, x, gamma)
            ndg = _num_grad(lambda g_: float((_ref_ln(x, g_, beta) * dy).sum()), gamma.copy())
            ndb = _num_grad(lambda b_: float((_ref_ln(x, gamma, b_) * dy).sum()), beta.copy())
            assert np.allclose(dg, ndg, atol=1e-6), f"dgamma 最大误差 {np.abs(dg - ndg).max():.2e}"
            assert np.allclose(db, ndb, atol=1e-6), f"dbeta 最大误差 {np.abs(db - ndb).max():.2e}"
          `,
        },
        {
          name: '大规模对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(3)
            x = rng.normal(size=(4, 16, 64)) * 3
            gamma = rng.normal(size=64)
            dy = rng.normal(size=(4, 16, 64))
            dx, dg, db = layer_norm_backward(dy, x, gamma, eps=1e-5)
            mu = x.mean(-1, keepdims=True); r = 1 / np.sqrt(x.var(-1, keepdims=True) + 1e-5); xh = (x - mu) * r
            gg = dy * gamma
            ref = r * (gg - gg.mean(-1, keepdims=True) - xh * (gg * xh).mean(-1, keepdims=True))
            assert np.allclose(dx, ref, atol=1e-10), f"最大误差 {np.abs(dx - ref).max():.2e}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-memory',
          q: md`反向需要缓存哪些量？能不能少存一点？`,
          a: md`
            最少只需缓存输入 $x$（本来就为上一层的反向保存着）以及每行的 $\mu$ 和 $r$（各 $O(N)$），反向时重算 $\hat{x} = (x - \mu) r$。
            Apex / Megatron 的 fused LayerNorm 都只存 $\mu$ 与 $r$；配合激活重计算还能进一步省显存。
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
      id: 'qa-ln-bn',
      freq: 3,
      q: md`LayerNorm 和 BatchNorm 有什么区别？为什么 Transformer 用 LN？`,
      a: md`
        **结论**：两者只差统计轴：BN 对每个通道跨 batch（和位置）统计，LN 对每个样本（token）跨通道统计。Transformer 用 LN 是因为它与 batch 无关。

        **原因**：(1) 自回归推理时 batch 统计量与训练时完全不同，BN 只能用滑动平均，训练推理不一致；(2) 变长序列的 padding 会污染 BN 统计量；(3) 大模型每卡 micro-batch 很小，BN 统计噪声大，跨卡同步还要通信；(4) NLP 中不同 token 的激活尺度差异大，按 token 归一化更合理。

        **延伸**：CV 里 BN 仍然常见（大 batch、固定尺寸）；ViT 用 LN；LLM 进一步换成 RMSNorm。
      `,
    },
    {
      id: 'qa-pre-post',
      freq: 3,
      q: md`Pre-LN 和 Post-LN 有什么区别？各自的优缺点？`,
      a: md`
        **结论**：Post-LN 是 $\mathrm{LN}(x + F(x))$，Pre-LN 是 $x + F(\mathrm{LN}(x))$。现在的 LLM 几乎都用 Pre-Norm，并在输出前加 final norm。

        **原因**：Pre-LN 的残差路径是纯恒等映射，梯度可以无衰减地传到浅层，训练稳定，对 warmup 不敏感（Xiong 等 2020 分析了 Post-LN 初始化时输出层附近梯度过大、需要 warmup）。Post-LN 的梯度必须穿过每个 LN，深层时容易梯度消失或不稳定，但它的表示能力往往更强（每层输出都被重新归一化，各层贡献均衡）。

        **延伸**：Pre-LN 的问题是残差流范数随深度近似按 $\sqrt{l}$ 增长，深层的相对贡献变小；因此出现了 DeepNorm（Post-LN + 放大残差）、Sandwich/Peri-LN（Gemma 2/3 在子层输入输出都加 norm）、OLMo 2 的子层输出 norm 等折中方案。
      `,
    },
    {
      id: 'qa-rms',
      freq: 2,
      q: md`RMSNorm 相比 LayerNorm 改了什么？为什么大模型都用它？`,
      a: md`
        **结论**：去掉减均值和 $\beta$，只除以均方根再乘 $\gamma$。

        **原因**：(1) 少一次归约、少一组参数，kernel 更简单，速度更快；(2) 原论文的实验表明 LN 的收益主要来自缩放不变性，去掉平移不变性对效果几乎无损；(3) 网络中激活的均值本来就接近 0。

        **延伸**：工程上 RMSNorm 要在 fp32 下求均方（LLaMA 实现先 \`.float()\` 再转回）；Gemma 用 $(1+\gamma)$ 参数化。在 Muon 训练中，RMSNorm 的 $\gamma$ 这类一维参数仍交给 AdamW 更新。
      `,
    },
    {
      id: 'qa-qknorm',
      freq: 2,
      q: md`什么是 QK-Norm？它解决什么问题？有没有不能用的情况？`,
      a: md`
        **结论**：在注意力点积之前，对每个头的 $q$、$k$ 做 RMSNorm / LayerNorm，控制 attention logit 的量级，防止 logit 爆炸导致的 loss 尖刺与训练发散。

        **原因**：logit $= q\cdot k/\sqrt{d_h}$ 的大小取决于 $W_Q, W_K$ 的范数，没有任何约束；归一化后 $q, k$ 的范数只由 norm 增益决定。ViT-22B 用它解决大规模训练不稳定；OLMo 2、Gemma 3、Qwen3 都采用；DeepSeek-V4 对 query 与压缩 KV 条目做 RMSNorm，论文明确说因此不需要 QK-Clip。

        **限制**：MLA 推理时 Key 不被完整显式构造（缓存的是潜向量并做矩阵吸收），QK-Norm 无法施加在 Key 上——Kimi K2 因此改用 QK-Clip：根据每个头的最大 logit 直接缩放 $W_Q, W_K$。
      `,
    },
  ],

  cheatsheet: md`
    | | LayerNorm | RMSNorm |
    |---|---|---|
    | 公式 | $\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta$ | $\gamma\odot\frac{x}{\sqrt{\overline{x^2}+\varepsilon}}$ |
    | 参数 | $2d$ | $d$ |
    | 不变性 | 平移 + 缩放 | 仅缩放 |

    **LN 反向**（$g = \mathrm{d}y\odot\gamma$，$r = (\sigma^2+\varepsilon)^{-1/2}$）
    \`\`\`python
    dbeta  = dy.reshape(-1, d).sum(0)
    dgamma = (dy * xhat).reshape(-1, d).sum(0)
    dx = r * (g - g.mean(-1, keepdims=True) - xhat * (g * xhat).mean(-1, keepdims=True))
    \`\`\`
    RMSNorm：去掉 \`- g.mean(...)\` 一项。自检：LN 的 \`dx.sum(-1) == 0\`。

    - 方差有偏（ddof=0），$\varepsilon$ 在根号内；别用 $\mathbb{E}[x^2]-\mathbb{E}[x]^2$
    - Pre-LN：$x + F(\mathrm{LN}(x))$ + final norm；残差流范数 ≈ $\sqrt{l}$ 增长
    - QK-Norm：Qwen3 / Gemma 3 / OLMo 2；DeepSeek-V4 对 Q 与压缩 KV 做 RMSNorm → 不用 QK-Clip
    - MLA 不能用 QK-Norm → Kimi K2 用 QK-Clip
  `,
})
