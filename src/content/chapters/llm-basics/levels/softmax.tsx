/**
 * llm-basics-softmax · Softmax 与交叉熵
 * 稳定 softmax / log-sum-exp / log_softmax、CE 的梯度 p − q、ignore_index、label smoothing、SFT loss mask。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { SoftmaxTempViz } from '../viz/SoftmaxTempViz'
import { OverflowViz } from '../viz/OverflowViz'
import { CEGradViz } from '../viz/CEGradViz'
import { LossMaskViz } from '../viz/LossMaskViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'def',
      title: 'softmax：把任意实数变成概率分布',
      body: md`
        语言模型最后一层输出的是 **logits** $z \in \mathbb{R}^V$（$V$ 是词表大小，Llama 3 为 128256）。softmax 把它变成概率：

        $$
        p_i = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
        $$

        三个必须脱口而出的性质：

        1. **平移不变**：$\mathrm{softmax}(z + c\mathbf{1}) = \mathrm{softmax}(z)$。分子分母同乘 $e^{c}$ 后约掉——概率只取决于 logit 的**差值**。这条性质是下一节“减最大值”合法的根据。
        2. **温度**：采样时用 $\mathrm{softmax}(z / T)$。$T \to 0$ 时趋向 argmax 的 one-hot（greedy），$T \to \infty$ 时趋向均匀分布；熵随 $T$ 单调增加（$\mathrm{d}H/\mathrm{d}T = \mathrm{Var}_p(z)/T^3 \ge 0$）。
        3. **不是“软的 max”而是“软的 argmax”**：真正光滑近似 max 的是 log-sum-exp，$\max_i z_i \le \mathrm{LSE}(z) \le \max_i z_i + \log V$。

        拖动温度与平移常数，验证这两条性质。
      `,
      viz: SoftmaxTempViz,
      after: md`
        > [!NOTE] 温度不改变排序
        > 对任意 $T>0$，$z/T$ 与 $z$ 的大小顺序相同，所以 argmax 不变；温度只改变“多自信”。这也是为什么 greedy 解码与温度无关。
      `,
      checkpoint: {
        id: 'cp-def',
        type: 'single',
        prompt: md`logits $z = [2, 1, 0]$，温度 $T = 0.5$。下面哪组 logits 在 $T = 1$ 下得到**完全相同**的概率分布？`,
        options: ['`[1, 0.5, 0]`', '`[4, 2, 0]`', '`[2.5, 1.5, 0.5]`', '`[1, 2, 3]`'],
        answer: 1,
        explanation: md`$\mathrm{softmax}(z/T)$ 中 $z/0.5 = [4, 2, 0]$。C 是 $z + 0.5$，与 $T=1$ 时的 $z$ 等价，但不等价于 $T = 0.5$；A 是 $z \cdot T$，方向反了；D 改变了排序。`,
      },
    },
    {
      id: 'overflow',
      title: '数值稳定：为什么一定要减最大值',
      body: md`
        直接按定义写 \`np.exp(z) / np.exp(z).sum()\` 有两种失败方式：

        | 情形 | 例子（float32） | 结果 |
        |---|---|---|
        | **上溢** | $z = [101, 100.8, \dots]$，$e^{101} > 3.4\times 10^{38}$ | $\infty / \infty = \text{NaN}$ |
        | **下溢** | $z = [-1000, -1001, \dots]$，$e^{-1000}$ 在 float32 中为 0 | $0 / 0 = \text{NaN}$ |

        $e^x$ 的溢出阈值是 $\ln(\text{dtype 最大值})$：float16 约 $11.09$（最大值 65504），float32 / bfloat16 约 $88.72$，float64 约 $709.78$。
        训练中 logits 到几十并不罕见，**混合精度下用 fp16 直接 exp 几乎必炸**。

        修正：令 $m = \max_i z_i$，利用平移不变性

        $$
        p_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}
        $$

        此时最大的那一项恰为 $e^0 = 1$，其余都在 $(0, 1]$，**分母 $\ge 1$**：上溢不可能发生，也不会出现 $0/0$。
        个别很小的项仍可能下溢为 0，但那只是“概率确实小到不可表示”，结果依然正确。
      `,
      viz: OverflowViz,
      after: md`
        > [!WARN] 批量计算时 max 必须按行取
        > 对形状 \`(B, V)\` 的 logits，要用 \`x.max(axis=-1, keepdims=True)\` 按**每一行**减。若减的是全局最大值，数学上仍正确，但 logits 整体偏小的那一行会全部下溢成 0，再次得到 $0/0$。\`keepdims=True\` 则保证 \`(B, 1)\` 能正确广播。
      `,
      checkpoint: {
        id: 'cp-overflow',
        type: 'judge',
        prompt: md`对任意有限的 logits $z$，稳定版 softmax 的分母 $\sum_j e^{z_j - \max(z)}$ 都满足 $1 \le \text{分母} \le V$。`,
        answer: true,
        explanation: md`最大项为 $e^0 = 1$，其余每项都在 $(0, 1]$（或下溢为 0），所以和在 $[1, V]$ 之间。正是这个下界保证了不会出现除以 0。`,
      },
    },
    {
      id: 'logsoftmax',
      title: 'log-sum-exp 与 log_softmax：训练时只用对数域',
      body: md`
        交叉熵要的是 $\log p_y$。如果先算 softmax 再取 log，一旦 $p_y$ 下溢为 0，就得到 $\log 0 = -\infty$，梯度随之爆掉。正确做法是在**对数域**里直接算：

        $$
        \log p_i = z_i - \underbrace{\Big(m + \log \sum_j e^{z_j - m}\Big)}_{\mathrm{LSE}(z)}
        $$

        $\mathrm{LSE}$（log-sum-exp）同样先减最大值。对比两种算法（float32，$z = [0, -200]$）：

        | 算法 | $\log p_2$ |
        |---|---|
        | \`np.log(softmax(z))[1]\` | $\log(0) = -\infty$ |
        | \`(z - lse(z))[1]\` | $-200$（精确） |

        这就是 PyTorch 的 \`F.cross_entropy\` **接收 logits 而不是概率**的原因：内部是 \`log_softmax\` + \`nll_loss\`，全程不显式算出 $p$。同理，\`log_softmax\` 的反向只需要 $\mathrm{softmax}$，即 $\exp(\log p)$，也不会出问题。

        > [!TIP] 一个 LSE 走天下
        > logsumexp 在大模型里随处可见：交叉熵、attention 的 online softmax（FlashAttention 里每块保存的就是行最大值 $m$ 与和 $\ell$，等价于 LSE）、PaLM 的 z-loss（惩罚 $\log^2 Z$，其中 $\log Z$ 就是 LSE）、DPO 里的序列对数概率。
      `,
      checkpoint: {
        id: 'cp-logsoftmax',
        type: 'single',
        prompt: md`为什么训练时要用 \`log_softmax\`，而不是 \`np.log(softmax(z))\`？`,
        options: [
          'log_softmax 计算更快，但数值上两者等价',
          'softmax 的小概率会下溢为 0，取 log 得到 −inf；log_softmax 在对数域计算，结果有限且精确',
          'softmax 的输出之和不为 1，需要 log 修正',
          'log_softmax 可以避免上溢，而 softmax 无法避免上溢',
        ],
        answer: 1,
        explanation: md`问题出在**下溢**：$p_i$ 小于 dtype 能表示的最小正数时变成 0，$\log 0 = -\infty$。稳定版 softmax 同样不会上溢（D 错）；两者数学上相等但数值上不等价（A 错）；C 是胡说。`,
      },
    },
    {
      id: 'ce',
      title: '交叉熵与它的梯度：p − y',
      body: md`
        单个样本、目标类 $y$ 的交叉熵：

        $$
        \mathcal{L} = -\log p_y = -z_y + \mathrm{LSE}(z)
        $$

        对 logits 求导：$\partial(-z_y)/\partial z_i = -[i = y]$，$\partial\,\mathrm{LSE}/\partial z_i = e^{z_i}/\sum_j e^{z_j} = p_i$，于是

        $$
        \frac{\partial \mathcal{L}}{\partial z} = p - \mathrm{onehot}(y)
        $$

        简洁到令人惊讶：**预测分布减去目标分布**。推论：梯度各分量之和为 0；$|g_i| \le 1$，交叉熵对 logits 的梯度天然有界。

        **Label smoothing**（\`label_smoothing=ε\`）把目标换成 $q = (1-\varepsilon)\,\mathrm{onehot}(y) + \varepsilon / C$（$C$ 为类别数，均匀部分**包括**真实类），
        损失 $\mathcal{L} = -\sum_i q_i \log p_i$，梯度为 $p - q$。最优解不再是 $p_y = 1$，而是 $p_y = 1 - \varepsilon + \varepsilon / C$：当 $z_y$ 过大时，梯度反过来把它往回拉。

        **batch 平均**：对 $N$ 个样本取 mean，梯度要再除以 $N$；被 \`ignore_index\` 跳过的样本梯度为 0，**也不计入分母**。
      `,
      viz: CEGradViz,
      after: md`
        > [!INTERVIEW] 面试官想听到的
        > “softmax 和交叉熵要融合起来算：前向用 LSE 避免溢出，反向直接得到 $p - y$，既省掉 $V \times V$ 的 softmax 雅可比，又数值稳定。”
        > 追问 label smoothing 时，记得说清代价：它让模型系统性地“不够自信”。LLM 预训练通常不用它（目标就是拟合真实的下一个 token 分布），在分类、机器翻译中更常见。
      `,
      checkpoint: {
        id: 'cp-ce',
        type: 'fill',
        prompt: md`3 分类，$p = [0.7, 0.2, 0.1]$，目标 $y = 0$，不做 label smoothing。$\partial\mathcal{L}/\partial z_0$ = ____(1)____，$\partial\mathcal{L}/\partial z_2$ = ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['-0.3'], numeric: true, tolerance: 0.001 },
          { label: '(2)', answers: ['0.1'], numeric: true, tolerance: 0.001 },
        ],
        explanation: md`$g = p - \mathrm{onehot}(0) = [0.7 - 1, 0.2, 0.1] = [-0.3, 0.2, 0.1]$，三者之和为 0。`,
      },
    },
    {
      id: 'mask',
      title: 'ignore_index 与 SFT 的 loss mask',
      body: md`
        语言模型的 loss 是**逐 token 的交叉熵**：位置 $t$ 的 logits 预测第 $t+1$ 个 token，所以要把 \`logits[:, :-1]\` 与 \`labels[:, 1:]\` 错位对齐（HF Transformers 在模型内部做这一步）。

        SFT 时我们只想学“怎么回答”，不想学“怎么提问”，于是：

        - \`labels = input_ids.clone()\`，把 **prompt（含 system / user / assistant 起始标记）与 padding** 位置设为 **−100**；
        - \`F.cross_entropy(..., ignore_index=-100)\`（默认值就是 −100）会跳过这些位置：**既不计入分子，也不计入分母**。

        归约方式会改变样本权重：

        | 归约 | 公式 | 效果 |
        |---|---|---|
        | batch 内 token 平均（默认） | $\frac{\sum_{\text{valid}} \ell_t}{N_{\text{valid}}}$ | 每个 token 等权，长回复的样本权重大 |
        | 先样本内平均再跨样本平均 | $\frac{1}{B}\sum_b \frac{\sum_{t \in b} \ell_t}{n_b}$ | 每个样本等权，短回复的 token 权重大 |

        两种都有人用，但**必须前后一致**。最常见的坑是梯度累积：每个 micro-batch 先各自做 token 平均再求平均，就不再等于大 batch 的 token 平均（2024 年 10 月 Unsloth 指出 HF Transformers 的梯度累积存在这个问题，随后修复）。正确做法是累加 loss 之和，最后除以**全部 micro-batch 的有效 token 总数**。
      `,
      viz: LossMaskViz,
      after: md`
        > [!WARN] 别忘了 EOS
        > 回复末尾的 \`<eos>\`（或 \`<|im_end|>\`）**要算 loss**——否则模型学不会停下来，推理时会一直生成。
      `,
      checkpoint: {
        id: 'cp-mask',
        type: 'single',
        prompt: md`一个 batch 有两条样本，回复部分分别有 2 个和 8 个 token（prompt 已设为 −100）。按 batch 内 token 平均，第一条样本在 loss 中的总权重是：`,
        options: ['10%', '20%', '50%', '80%'],
        answer: 1,
        explanation: md`token 平均时每个有效 token 权重为 $1/10$，第一条样本共 2 个 token，总权重 $2/10 = 20\%$。若先样本内平均再跨样本平均，则两条样本各 50%。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-shift',
      type: 'single',
      difficulty: 1,
      prompt: md`$\mathrm{softmax}([1, 2, 3])$ 与 $\mathrm{softmax}([101, 102, 103])$ 的关系是：`,
      options: ['完全相同', '后者更尖锐（更接近 one-hot）', '后者更平坦', '后者每个元素都是前者的 $e^{100}$ 倍'],
      answer: 0,
      explanation: md`平移不变性：分子分母同乘 $e^{100}$ 后约掉。改变“尖锐程度”的是**缩放**（温度），不是平移。D 只对分子成立，忘了分母也乘了同样的因子。`,
    },
    {
      id: 'q-temp',
      type: 'judge',
      difficulty: 2,
      prompt: md`对一组不全相等的 logits，温度 $T$ 越高，$\mathrm{softmax}(z/T)$ 的熵越大（单调不减）。`,
      answer: true,
      explanation: md`可以证明 $\frac{\mathrm{d}H}{\mathrm{d}T} = \frac{\mathrm{Var}_{p}(z)}{T^3} \ge 0$，logits 不全相等时方差大于 0，熵严格递增：从 $T \to 0$ 时的 0（one-hot）增加到 $T \to \infty$ 时的 $\log V$（均匀）。`,
    },
    {
      id: 'q-init',
      type: 'numeric',
      difficulty: 1,
      prompt: md`词表大小 $V = 32000$ 的语言模型刚初始化时输出接近均匀分布。此时每个 token 的交叉熵 loss 约为多少？（自然对数，保留两位小数）`,
      answer: 10.37,
      tolerance: 0.005,
      unit: 'nats',
      explanation: md`均匀分布下 $p_y = 1/V$，$\mathcal{L} = \ln V = \ln 32000 \approx 10.37$。这是检查训练脚本的常用 sanity check：初始 loss 若远大于 $\ln V$，通常是初始化过大（logits 太尖）或 label 错位。`,
    },
    {
      id: 'q-thresh',
      type: 'fill',
      difficulty: 2,
      prompt: md`不减最大值时，\`exp(x)\` 在 float16 中当 $x >$ ____(1)____ 时溢出为 inf，在 float32 中当 $x >$ ____(2)____ 时溢出。（保留两位小数）`,
      blanks: [
        { label: '(1)', answers: ['11.09'], numeric: true, tolerance: 0.002 },
        { label: '(2)', answers: ['88.72'], numeric: true, tolerance: 0.002 },
      ],
      explanation: md`阈值为 $\ln(\text{最大可表示值})$：float16 最大值 65504，$\ln 65504 \approx 11.09$；float32 最大值约 $3.40 \times 10^{38}$，$\ln \approx 88.72$。bfloat16 的指数位与 float32 相同，阈值也约为 88.7——这是 bf16 训练比 fp16 更不容易溢出的原因之一。`,
    },
    {
      id: 'q-lse',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 log-sum-exp 与 log_softmax，以下说法正确的是（多选）：`,
      options: [
        '$\\log\\mathrm{softmax}(z) = z - \\mathrm{LSE}(z)$',
        '在 float32 下，`np.log(softmax(z))` 与稳定版 `log_softmax(z)` 总是给出相同结果',
        '$\\max_i z_i \\le \\mathrm{LSE}(z) \\le \\max_i z_i + \\log V$',
        '只要减去了最大值，所有 exp 项都不会下溢',
        'PyTorch 的 `F.cross_entropy` 接收 logits，内部等价于 `log_softmax` + `nll_loss`',
      ],
      answer: [0, 2, 4],
      explanation: md`A 是定义。C：LSE 至少等于最大项的贡献，至多是 $V$ 个最大项之和的对数。B 错：小概率下溢为 0 后取 log 得到 $-\infty$，稳定版给出有限值。D 错：减最大值只消除**上溢**和 $0/0$，远小于最大值的项仍会下溢为 0（但这是无害的）。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的批量 softmax（输入形状 \`(B, V)\`，$B \ne V$）有 **2 处** bug：一处在某些输入上产生 NaN，一处在所有输入上出错。点出这两行。`,
      code: py`
        import numpy as np

        def softmax(x):
            """x: (B, V)，对每一行做 softmax"""
            m = np.max(x)
            e = np.exp(x - m)
            s = e.sum(axis=-1)
            return e / s
      `,
      bugLines: [5, 7],
      explanation: md`
        - 第 5 行减的是**全局**最大值。数学上仍对，但如果第一行 logits 在 1000 附近、第二行在 0 附近，第二行全部下溢为 0，得到 $0/0$。应改为 \`m = np.max(x, axis=-1, keepdims=True)\`。
        - 第 7 行少了 \`keepdims=True\`：\`s\` 形状为 \`(B,)\`，\`e / s\` 会按**最后一维**广播——$B \ne V$ 时直接报错，$B = V$ 时静默地除错了数。应改为 \`e.sum(axis=-1, keepdims=True)\`。
      `,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出支持 \`ignore_index\` 的数值稳定交叉熵（logits 形状 \`(N, C)\`，返回有效 token 上的平均 loss）。有 2 行干扰项。`,
      lines: [
        'def cross_entropy(logits, targets, ignore_index=-100):',
        '    m = logits.max(axis=-1, keepdims=True)',
        '    logp = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))',
        '    valid = targets != ignore_index',
        '    safe_t = np.where(valid, targets, 0)',
        '    nll = -logp[np.arange(len(targets)), safe_t]',
        '    return (nll * valid).sum() / max(valid.sum(), 1)',
      ],
      distractors: [
        '    logp = np.log(np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True))',
        '    return nll.mean()',
      ],
      explanation: md`
        先在对数域算 \`log_softmax\`；\`safe_t\` 把 −100 换成任意合法下标，避免越界索引（这些位置随后会被 \`valid\` 乘成 0）；
        分母是**有效 token 数**。干扰项 1 是会溢出且会 \`log(0)\` 的朴素写法；干扰项 2 把被忽略的位置也算进了分母（而且分子里还混入了它们的 loss）。
      `,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把左侧的概念与右侧的作用配对：`,
      pairs: [
        ['`ignore_index=-100`', '该位置既不计入 loss 分子，也不计入分母'],
        ['`label_smoothing=ε`', '目标变为 (1−ε)·onehot + ε/C'],
        ['z-loss', '惩罚 log²Z，让 logsumexp 保持在 0 附近'],
        ['温度 T', '采样前把 logits 除以 T，改变分布的尖锐程度'],
        ['减最大值', '防止 exp 上溢和 0/0'],
      ],
      explanation: md`z-loss 来自 PaLM：辅助损失 $10^{-4} \cdot \log^2 Z$（$Z$ 为 softmax 分母），让 logits 不随训练整体漂移，有助于训练稳定。其余几项见本关正文。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 SFT 计算 loss 的步骤按正确顺序排列：`,
      items: [
        '用 chat template 拼接 prompt 与回复，得到 input_ids',
        '复制 input_ids 作为 labels，把 prompt 与 padding 位置设为 −100',
        '前向得到 logits，把 logits[:, :-1] 与 labels[:, 1:] 错位对齐',
        '对 logits 做 log_softmax，按 labels 取出目标 token 的对数概率',
        '在 label ≠ −100 的位置上取平均得到 loss',
      ],
      explanation: md`顺序的关键是：mask 在构造 labels 时就打好（与模型无关）；错位对齐发生在 logits 与 labels 之间；最后的平均只对有效位置。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 1,
      prompt: md`一条 SFT 样本 tokenize 后如下（已 padding）。构造 labels 时，哪些位置**保留原 token id**（即不设为 −100）？点出全部。`,
      items: ['<user>', '你好', '<asst>', '您好', '！', '<eos>', '<pad>', '<pad>'],
      answer: [3, 4, 5],
      explanation: md`只有回复部分 \`您好\`、\`！\` 以及结尾的 \`<eos>\` 算 loss。\`<asst>\` 起始标记属于 prompt（模板固定，不需要学）；\`<eos>\` 必须算，否则模型学不会停止；padding 一律 −100。`,
    },
    {
      id: 'q-ls-grad',
      type: 'single',
      difficulty: 3,
      prompt: md`PyTorch 语义的 label smoothing（$C$ 类，系数 $\varepsilon$），单样本 loss 对 logits 的梯度是：`,
      options: [
        '$p - \\mathrm{onehot}(y)$，与 ε 无关',
        '$p - \\big[(1-\\varepsilon)\\,\\mathrm{onehot}(y) + \\varepsilon / C\\big]$',
        '$(1-\\varepsilon)\\,(p - \\mathrm{onehot}(y))$',
        '$p - \\big[(1-\\varepsilon)\\,\\mathrm{onehot}(y) + \\varepsilon / (C-1)\\,(1 - \\mathrm{onehot}(y))\\big]$',
      ],
      answer: 1,
      explanation: md`
        PyTorch 的 loss 为 $(1-\varepsilon)(-\log p_y) + \varepsilon \cdot \frac{1}{C}\sum_c(-\log p_c)$，等价于对目标 $q = (1-\varepsilon)\,\mathrm{onehot}(y) + \varepsilon/C$ 的交叉熵，梯度为 $p - q$。
        C 漏掉了均匀部分的梯度；D 是“把 ε 平分给其余 $C-1$ 类”的另一种约定（有些论文这么写），**不是** PyTorch 的实现——面试中写清楚你用的是哪种。
      `,
    },
    {
      id: 'q-accum',
      type: 'numeric',
      difficulty: 3,
      prompt: md`
        梯度累积 2 个 micro-batch：第 1 个有 10 个有效 token，loss 之和为 20；第 2 个有 30 个有效 token，loss 之和为 30。
        与“一次性处理整个 batch、按 token 平均”等价的 loss 是多少？
      `,
      answer: 1.25,
      tolerance: 0.001,
      explanation: md`
        正确值 = 全部 loss 之和 / 全部有效 token 数 $= (20 + 30) / (10 + 30) = 1.25$。
        常见的错误做法是“每个 micro-batch 先求平均再平均”：$(20/10 + 30/30)/2 = 1.5$——这等于给 token 少的 micro-batch 更大的权重，和大 batch 训练不一致。
      `,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-softmax',
      title: '写出任意 axis 的稳定 softmax 与 log_softmax',
      intro: md`
        实现 \`softmax(x, axis=-1)\` 与 \`log_softmax(x, axis=-1)\`（NumPy）。要求：对任意形状、任意 axis 工作；logits 大到 1000 或小到 −1000 时都不出现 NaN / inf。
      `,
      template: py`
        import numpy as np

        def softmax(x, axis=-1):
            x = np.asarray(x, dtype=np.float64)
            m = {{1}}
            e = {{2}}
            return {{3}}

        def log_softmax(x, axis=-1):
            x = np.asarray(x, dtype=np.float64)
            m = np.max(x, axis=axis, keepdims=True)
            lse = {{4}}
            return x - lse
      `,
      steps: [
        {
          blank: 1,
          instruction: md`沿 \`axis\` 求最大值，并保留被归约的维度，便于广播。`,
          answer: 'np.max(x, axis=axis, keepdims=True)',
          accept: ['x.max(axis=axis, keepdims=True)', 'np.amax(x, axis=axis, keepdims=True)'],
          hint: '别忘了 keepdims=True。',
          explain: md`\`keepdims=True\` 让形状 \`(B, V)\` 的最大值变成 \`(B, 1)\` 而不是 \`(B,)\`，减法才能按行广播。axis 不是最后一维时同样成立。`,
        },
        {
          blank: 2,
          instruction: md`利用平移不变性，对 \`x - m\` 取指数。`,
          answer: 'np.exp(x - m)',
          explain: md`最大项变成 $e^0 = 1$，不可能上溢。`,
        },
        {
          blank: 3,
          instruction: md`沿同一个 axis 归一化。`,
          answer: 'e / np.sum(e, axis=axis, keepdims=True)',
          accept: ['e / e.sum(axis=axis, keepdims=True)'],
          explain: md`分母 $\ge 1$，不会出现 $0/0$。`,
        },
        {
          blank: 4,
          instruction: md`写出 log-sum-exp：$\mathrm{LSE}(x) = m + \log\sum e^{x - m}$（保持维度）。`,
          answer: 'm + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))',
          accept: ['m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))'],
          hint: '把 m 加回来。',
          explain: md`\`log\` 的参数在 $[1, V]$ 之间，既不会是 0 也不会溢出；所以 \`log_softmax\` 即使在某个概率下溢为 0 时也返回精确的有限值。`,
        },
      ],
      setup: py`
        def _ref_softmax(x, axis=-1):
            x = np.asarray(x, dtype=np.float64)
            e = np.exp(x - x.max(axis=axis, keepdims=True))
            return e / e.sum(axis=axis, keepdims=True)

        def _ref_log_softmax(x, axis=-1):
            x = np.asarray(x, dtype=np.float64)
            m = x.max(axis=axis, keepdims=True)
            return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
      `,
      tests: [
        {
          name: '示例与大 logits',
          code: py`
            p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]))
            assert p.shape == (2, 3), f"形状应为 (2, 3)，得到 {p.shape}"
            assert np.all(np.isfinite(p)), f"出现 NaN/inf：{p}"
            assert np.allclose(p[0], p[1]), f"平移不变：两行应相同，得到 {p}"
            assert np.allclose(p[0], [0.09003057, 0.24472847, 0.66524096]), f"得到 {p[0]}"
          `,
        },
        {
          name: 'axis=0',
          code: py`
            x = np.array([[1.0, 5.0], [3.0, 5.0]])
            got = softmax(x, axis=0)
            exp = _ref_softmax(x, axis=0)
            assert np.allclose(got, exp), f"期望 {exp}，得到 {got}"
            assert np.allclose(got.sum(axis=0), 1), "沿 axis=0 求和应为 1"
          `,
        },
        {
          name: 'log_softmax 极端值',
          code: py`
            x = np.array([0.0, -200.0, -1000.0])
            got = log_softmax(x)
            assert np.all(np.isfinite(got)), f"log_softmax 不应出现 -inf：{got}"
            assert np.allclose(got, [0.0, -200.0, -1000.0], atol=1e-9), f"得到 {got}"
          `,
        },
        {
          name: '随机对拍（3 维、各种 axis）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(0)
            x = rng.normal(size=(2, 3, 5)) * 50
            for ax in (0, 1, 2, -1):
                assert np.allclose(softmax(x, axis=ax), _ref_softmax(x, axis=ax)), f"softmax axis={ax} 不一致"
                assert np.allclose(log_softmax(x, axis=ax), _ref_log_softmax(x, axis=ax)), f"log_softmax axis={ax} 不一致"
          `,
        },
      ],
      outro: md`
        这两个函数就是本章后面所有 loss 的地基。下一步：把 \`log_softmax\` 和 gather 组合成交叉熵，并亲手推出 $p - q$ 的梯度。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-masked-softmax',
      title: '带掩码的稳定 softmax',
      difficulty: 'medium',
      source: '大模型岗手撕 · 注意力 / padding mask',
      timeLimitMin: 15,
      description: md`
        实现带布尔掩码的 softmax（注意力里的 padding mask、因果 mask 都要用到）：

        \`\`\`python
        def masked_softmax(x: np.ndarray, mask: np.ndarray | None = None, axis: int = -1) -> np.ndarray: ...
        \`\`\`

        - \`x\`：任意形状的浮点数组；\`mask\`：布尔数组，可**广播**到 \`x\` 的形状，\`True\` 表示保留、\`False\` 表示屏蔽；\`None\` 表示不屏蔽。
        - 沿 \`axis\` 做 softmax；被屏蔽的位置输出**严格为 0**。
        - 若某一行（沿 \`axis\`）**全部被屏蔽**，该行输出全 0（而不是 NaN）。
        - 数值稳定：logits 的绝对值可达 $10^4$。

        **示例**
        - \`masked_softmax([[1, 2, 3]], [[True, True, False]])\` → \`[[0.2689, 0.7311, 0.0]]\`
        - \`masked_softmax([[1, 2]], [[False, False]])\` → \`[[0.0, 0.0]]\`
      `,
      starter: py`
        import numpy as np

        def masked_softmax(x, mask=None, axis=-1):
            x = np.asarray(x, dtype=np.float64)
            # TODO: 1) 屏蔽位置填 -inf  2) 减去最大值（注意全屏蔽行的最大值是 -inf）
            #       3) exp 并归一化（注意全屏蔽行的分母是 0）
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def masked_softmax(x, mask=None, axis=-1):
            x = np.asarray(x, dtype=np.float64)
            if mask is not None:
                mask = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
                x = np.where(mask, x, -np.inf)          # 屏蔽位置 → -inf，exp 后恰为 0
            m = np.max(x, axis=axis, keepdims=True)
            m = np.where(np.isfinite(m), m, 0.0)        # 全屏蔽行的 max 为 -inf，避免 -inf - (-inf) = nan
            e = np.exp(x - m)
            s = np.sum(e, axis=axis, keepdims=True)
            return e / np.where(s == 0, 1.0, s)         # 全屏蔽行分母为 0 → 输出全 0
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def masked_softmax(x: torch.Tensor, mask: torch.Tensor | None = None, dim: int = -1) -> torch.Tensor:
                if mask is not None:
                    x = x.masked_fill(~mask, float("-inf"))
                m = x.amax(dim=dim, keepdim=True)
                m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
                e = torch.exp(x - m)
                s = e.sum(dim=dim, keepdim=True)
                return e / s.clamp_min(1e-30) * (s > 0)   # 全屏蔽行 → 0
                # 注意：直接 torch.softmax 在全 -inf 行会返回 NaN
          `,
        },
      ],
      explanation: md`
        **思路**：屏蔽位置填 $-\infty$，\`exp\` 之后恰好是 0，既不影响分母也让输出严格为 0。

        **两个边界**：
        1. 全屏蔽行的最大值是 $-\infty$，\`x - m\` 变成 $-\infty - (-\infty) = \text{NaN}$。把这种行的 \`m\` 换成 0 即可（换成任何有限数都行）。
        2. 全屏蔽行的分母是 0，用 \`np.where(s == 0, 1, s)\` 防止 $0/0$。

        **为什么不用 \`x * mask\` 或 \`-1e9\`**：乘以 0 只会把 logit 置为 0，这个位置照样分到概率；\`-1e9\` 在 fp16 下会溢出成 \`-inf\`（fp16 最小值约 −65504），而且当整行都是 \`-1e9\` 时会退化成均匀分布，而不是全 0。

        **复杂度**：$O(\text{numel})$。
      `,
      complexity: { time: 'O(N)', space: 'O(N)' },
      hints: [
        md`用 \`np.where(mask, x, -np.inf)\` 把屏蔽位置变成 $-\infty$，\`np.broadcast_to\` 处理形状不同的 mask。`,
        md`全屏蔽行的 \`max\` 是 \`-inf\`，用 \`np.where(np.isfinite(m), m, 0)\` 修正。`,
        md`全屏蔽行的分母是 0：除之前用 \`np.where(s == 0, 1, s)\` 替换。`,
      ],
      setup: py`
        def _ref_masked_softmax(x, mask=None, axis=-1):
            x = np.asarray(x, dtype=np.float64)
            if mask is None:
                mask = np.ones(x.shape, dtype=bool)
            mask = np.broadcast_to(np.asarray(mask, dtype=bool), x.shape)
            xm = np.moveaxis(np.where(mask, x, 0.0), axis, -1)
            mk = np.moveaxis(mask, axis, -1)
            out = np.zeros(xm.shape)
            for idx in np.ndindex(*xm.shape[:-1]):
                row, keep = xm[idx], mk[idx]
                if keep.any():
                    v = row[keep]
                    e = np.exp(v - v.max())
                    out[idx][keep] = e / e.sum()
            return np.moveaxis(out, -1, axis)
      `,
      tests: [
        {
          name: '示例',
          code: py`
            got = masked_softmax(np.array([[1.0, 2.0, 3.0]]), np.array([[True, True, False]]))
            exp = np.array([[0.26894142, 0.73105858, 0.0]])
            assert got.shape == (1, 3), f"形状应为 (1, 3)，得到 {got.shape}"
            assert np.allclose(got, exp), f"期望 {exp}，得到 {got}"
            assert got[0, 2] == 0.0, f"被屏蔽位置必须严格为 0，得到 {got[0, 2]}"
          `,
        },
        {
          name: '无 mask + 大 logits',
          code: py`
            got = masked_softmax(np.array([1000.0, 1001.0, 1002.0]))
            assert np.all(np.isfinite(got)), f"出现 NaN/inf：{got}"
            assert np.allclose(got, [0.09003057, 0.24472847, 0.66524096]), f"得到 {got}"
          `,
        },
        {
          name: '整行被屏蔽 → 全 0',
          code: py`
            x = np.array([[1.0, 2.0], [3.0, 4.0]])
            m = np.array([[False, False], [True, True]])
            got = masked_softmax(x, m)
            assert np.all(np.isfinite(got)), f"不应出现 NaN：{got}"
            assert np.allclose(got[0], 0.0), f"全屏蔽行应为全 0，得到 {got[0]}"
            assert np.allclose(got[1], [0.26894142, 0.73105858]), f"第二行得到 {got[1]}"
          `,
        },
        {
          name: '广播 mask（注意力 padding mask）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(1)
            B, H, T = 2, 3, 5
            scores = rng.normal(size=(B, H, T, T)) * 30
            lengths = np.array([5, 3])
            pad = (np.arange(T)[None, :] < lengths[:, None])[:, None, None, :]   # (B,1,1,T)
            got = masked_softmax(scores, pad)
            exp = _ref_masked_softmax(scores, pad)
            assert got.shape == scores.shape, f"形状应为 {scores.shape}，得到 {got.shape}"
            assert np.allclose(got, exp), f"最大误差 {np.abs(got - exp).max():.3e}"
            assert np.all(got[1, ..., 3:] == 0), "padding 位置必须严格为 0"
          `,
        },
        {
          name: '任意 axis + 随机 mask',
          hidden: true,
          code: py`
            rng = np.random.default_rng(2)
            x = rng.normal(size=(4, 6, 3)) * 100
            mk = rng.random(size=(4, 6, 3)) < 0.6
            for ax in (0, 1, 2):
                got = masked_softmax(x, mk, axis=ax)
                exp = _ref_masked_softmax(x, mk, axis=ax)
                assert np.all(np.isfinite(got)), f"axis={ax} 出现 NaN"
                assert np.allclose(got, exp), f"axis={ax} 最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: '极端 logits（±1e4）',
          hidden: true,
          code: py`
            x = np.array([[1e4, -1e4, 0.0, 1e4 - 1.0], [-1e4, -1e4 - 2.0, -1e4 - 1.0, -1e4]])
            mk = np.array([[True, True, True, True], [True, True, False, True]])
            got = masked_softmax(x, mk)
            exp = _ref_masked_softmax(x, mk)
            assert np.all(np.isfinite(got)), f"出现 NaN/inf：{got}"
            assert np.allclose(got, exp, atol=1e-12), f"期望 {exp}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-neg-big',
          q: md`很多代码用 \`scores.masked_fill(~mask, -1e9)\` 而不是 \`-inf\`，有什么区别？`,
          a: md`
            \`-1e9\` 的好处是全屏蔽行不会产生 NaN（会退化为均匀分布，再由后续 mask 抹掉）；坏处是：fp16 中 \`-1e9\` 直接变成 \`-inf\`（fp16 最小有限值约 −65504），优势消失；且结果依赖“足够负”的假设。
            更稳妥的做法是用 \`torch.finfo(dtype).min\`，或者用 \`-inf\` 并显式处理全屏蔽行（FlashAttention 等 kernel 就是这么做的）。
          `,
        },
      ],
    },
    {
      id: 'c-cross-entropy',
      title: '交叉熵：ignore_index + label smoothing + 手推梯度',
      difficulty: 'medium',
      source: '大模型岗手撕 · loss',
      timeLimitMin: 20,
      description: md`
        实现与 PyTorch \`F.cross_entropy(logits, targets, ignore_index, label_smoothing, reduction="mean")\` 语义一致的交叉熵，并返回对 logits 的梯度：

        \`\`\`python
        def cross_entropy(logits: np.ndarray, targets: np.ndarray,
                          ignore_index: int = -100, label_smoothing: float = 0.0) -> tuple[float, np.ndarray]: ...
        \`\`\`

        - \`logits\`：形状 \`(N, C)\`；\`targets\`：形状 \`(N,)\` 的整数，取值 $[0, C)$ 或 \`ignore_index\`。
        - 对有效位置（\`targets != ignore_index\`），单样本 loss 为
          $$\ell_n = (1-\varepsilon)\big(-\log p_{n, y_n}\big) + \varepsilon \cdot \frac{1}{C}\sum_{c}\big(-\log p_{n,c}\big)$$
        - 返回 \`(loss, dlogits)\`：\`loss\` 是有效位置上 $\ell_n$ 的**平均值**（float）；\`dlogits\` 形状 \`(N, C)\`，是 \`loss\` 对 \`logits\` 的梯度，被忽略的行梯度为 0。
        - 若**没有任何有效位置**，返回 \`(0.0, 全 0 梯度)\`（PyTorch 此时返回 NaN，这里约定为 0）。
        - 数值稳定：logits 的绝对值可达 $10^4$。

        **示例**：\`logits = [[2, 1, 0], [0, 0, 0]]\`，\`targets = [0, 2]\`，$\varepsilon = 0$ →
        loss $= \frac{1}{2}\big(0.4076 + 1.0986\big) = 0.7531$。
      `,
      starter: py`
        import numpy as np

        def cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.0):
            logits = np.asarray(logits, dtype=np.float64)
            targets = np.asarray(targets)
            # TODO: 1) 稳定 log_softmax  2) 构造目标分布 q（label smoothing）
            #       3) 只在有效行上求平均  4) 梯度 (p - q) / n_valid，忽略行为 0
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.0):
            logits = np.asarray(logits, dtype=np.float64)
            targets = np.asarray(targets)
            N, C = logits.shape
            valid = targets != ignore_index                     # (N,)
            n_valid = int(valid.sum())
            if n_valid == 0:
                return 0.0, np.zeros_like(logits)

            # 1) 稳定的 log_softmax
            m = logits.max(axis=-1, keepdims=True)
            logp = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
            p = np.exp(logp)

            # 2) 目标分布 q = (1-ε)·onehot + ε/C（被忽略的行先随便填一个合法类别）
            eps = label_smoothing
            safe_t = np.where(valid, targets, 0)
            q = np.full((N, C), eps / C)
            q[np.arange(N), safe_t] += 1.0 - eps

            # 3) 每行 loss = -Σ q·logp，只在有效行上平均
            per_row = -(q * logp).sum(axis=-1)
            loss = float(per_row[valid].sum() / n_valid)

            # 4) 梯度：(p - q) / n_valid，忽略行置 0
            grad = (p - q) * valid[:, None] / n_valid
            return loss, grad
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch
            import torch.nn.functional as F

            def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index=-100, label_smoothing=0.0):
                logits = logits.detach().double().requires_grad_(True)
                loss = F.cross_entropy(logits, targets, ignore_index=ignore_index,
                                       label_smoothing=label_smoothing, reduction="mean")
                loss.backward()          # 全部被忽略时 loss 为 NaN
                return loss.item(), logits.grad
          `,
        },
      ],
      explanation: md`
        **前向**：在对数域算 $\log p = z - \mathrm{LSE}(z)$；label smoothing 等价于对目标分布 $q = (1-\varepsilon)\,\mathrm{onehot}(y) + \varepsilon/C$ 求交叉熵 $-\sum_c q_c \log p_c$。

        **反向**：单行 $\partial \ell / \partial z = p - q$（因为 $\sum_c q_c = 1$）。mean 归约再除以有效行数 $n_{\text{valid}}$，被忽略的行对 loss 没有贡献，梯度为 0。

        **易错点**
        - 分母用 $N$ 而不是 $n_{\text{valid}}$（与 PyTorch 不一致）；
        - 用 \`targets\` 直接做花式索引，−100 会被当成“倒数第 100 列”（NumPy 负索引！），不报错但结果错；
        - label smoothing 的均匀部分写成 $\varepsilon/(C-1)$ 且不包含真实类——那是另一种约定。

        **复杂度**：$O(NC)$ 时间与空间。
      `,
      complexity: { time: 'O(NC)', space: 'O(NC)' },
      hints: [
        md`先写稳定的 \`log_softmax\`，再构造目标分布 \`q\`，loss 就是 \`-(q * logp).sum(-1)\`。`,
        md`被忽略的行：用 \`safe_t = np.where(valid, targets, 0)\` 避免 −100 的负索引陷阱，最后用 \`valid\` 把它们乘成 0。`,
        md`梯度 = \`(p - q) * valid[:, None] / n_valid\`。`,
      ],
      setup: py`
        def _ref_ce(logits, targets, ignore_index=-100, label_smoothing=0.0):
            logits = np.asarray(logits, dtype=np.float64)
            N, C = logits.shape
            total, cnt = 0.0, 0
            for n in range(N):
                t = int(targets[n])
                if t == ignore_index:
                    continue
                z = logits[n]
                lse = z.max() + np.log(np.exp(z - z.max()).sum())
                logp = z - lse
                total += (1 - label_smoothing) * (-logp[t]) + label_smoothing * (-logp.mean())
                cnt += 1
            return total / cnt if cnt else 0.0

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
          name: '示例',
          code: py`
            loss, g = cross_entropy(np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), np.array([0, 2]))
            assert abs(float(loss) - 0.75311) < 1e-4, f"期望 0.75311，得到 {loss}"
            assert g.shape == (2, 3), f"梯度形状应为 (2, 3)，得到 {g.shape}"
            exp_g = np.array([[-0.16738, 0.12236, 0.04502], [0.16667, 0.16667, -0.33333]])
            assert np.allclose(g, exp_g, atol=1e-4), f"期望梯度 {exp_g}，得到 {g}"
          `,
        },
        {
          name: 'ignore_index 不计入分母',
          code: py`
            x = np.array([[2.0, 1.0, 0.0], [5.0, -3.0, 1.0], [0.0, 0.0, 0.0]])
            t = np.array([0, -100, 2])
            loss, g = cross_entropy(x, t)
            exp = _ref_ce(x, t)
            assert abs(float(loss) - exp) < 1e-9, f"期望 {exp:.6f}（分母为 2），得到 {loss}"
            assert np.allclose(g[1], 0), f"被忽略行的梯度应为 0，得到 {g[1]}"
          `,
        },
        {
          name: 'label smoothing',
          code: py`
            x = np.array([[2.0, 1.0, 0.0, -1.0]])
            loss, g = cross_entropy(x, np.array([1]), label_smoothing=0.1)
            exp = _ref_ce(x, [1], label_smoothing=0.1)
            assert abs(float(loss) - exp) < 1e-9, f"期望 {exp:.6f}，得到 {loss}"
            assert abs(g.sum()) < 1e-9, f"每行梯度之和应为 0，得到 {g.sum()}"
          `,
        },
        {
          name: '全部被忽略',
          code: py`
            loss, g = cross_entropy(np.ones((2, 3)), np.array([-100, -100]))
            assert float(loss) == 0.0, f"期望 0.0，得到 {loss}"
            assert np.all(g == 0), "梯度应全为 0"
          `,
        },
        {
          name: '大 logits 稳定',
          hidden: true,
          code: py`
            x = np.array([[1e4, 0.0, -1e4], [-1e4, -1e4 + 3.0, -1e4 + 1.0]])
            t = np.array([2, 1])
            loss, g = cross_entropy(x, t, label_smoothing=0.05)
            exp = _ref_ce(x, t, label_smoothing=0.05)
            assert np.isfinite(loss) and np.all(np.isfinite(g)), f"出现 NaN/inf：loss={loss}"
            assert abs(float(loss) - exp) < 1e-6 * max(1.0, abs(exp)), f"期望 {exp}，得到 {loss}"
          `,
        },
        {
          name: '数值梯度检查',
          hidden: true,
          code: py`
            rng = np.random.default_rng(3)
            x = rng.normal(size=(5, 4)) * 2
            t = np.array([0, 3, -100, 1, 2])
            _, g = cross_entropy(x, t, label_smoothing=0.2)
            ng = _num_grad(lambda z: _ref_ce(z, t, label_smoothing=0.2), x.copy())
            err = np.abs(g - ng).max()
            assert err < 1e-6, f"解析梯度与数值梯度最大误差 {err:.3e}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(4)
            for trial in range(30):
                N, C = rng.integers(1, 8), rng.integers(2, 10)
                x = rng.normal(size=(N, C)) * 5
                t = rng.integers(0, C, size=N)
                t[rng.random(N) < 0.3] = -100
                ls = float(rng.choice([0.0, 0.1, 0.3]))
                loss, _ = cross_entropy(x, t, label_smoothing=ls)
                exp = _ref_ce(x, t, label_smoothing=ls)
                assert abs(float(loss) - exp) < 1e-9, f"N={N}, C={C}, ε={ls}：期望 {exp}，得到 {loss}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-vocab-mem',
          q: md`词表 152K、序列 32K 时，logits 本身就要多少显存？怎么省？`,
          a: md`
            fp32 的 logits 为 $32768 \times 151936 \times 4\,\text{B} \approx 19.9\,\text{GB}$（单条序列！），反向还要同样大的梯度。
            常见做法：**分块计算交叉熵**（按序列或词表切块，前向时直接算出 LSE 与梯度，不保存完整 logits，如 Liger Kernel 的 fused linear cross entropy、Apple 的 Cut Cross-Entropy）；或在 TP 下做词表并行交叉熵，每张卡只算自己那一段词表，再 all-reduce 最大值与和。
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
      id: 'qa-max',
      freq: 3,
      q: md`softmax 为什么要减最大值？减最小值、减均值行不行？`,
      a: md`
        **结论**：减任何常数结果都不变（平移不变性），但只有减**最大值**能同时保证不上溢、不出现 $0/0$。

        **原因**：减最大值后指数的最大项恰为 $e^0 = 1$，所有项在 $(0, 1]$，分母 $\in [1, V]$。减最小值时最大项变成 $e^{\max - \min}$，范围大的时候照样上溢；减均值也一样。

        **延伸**：同样的技巧出现在 log-sum-exp、online softmax（FlashAttention 维护运行最大值 $m$，遇到更大的值时用 $e^{m_{\text{old}} - m_{\text{new}}}$ 修正之前的和）、以及带 mask 的注意力（全屏蔽行的最大值是 $-\infty$，要特判）。
      `,
    },
    {
      id: 'qa-grad',
      freq: 3,
      q: md`推导 softmax + 交叉熵对 logits 的梯度。为什么框架要把两者融合？`,
      a: md`
        **结论**：$\partial\mathcal{L}/\partial z = p - \mathrm{onehot}(y)$。

        **推导**：$\mathcal{L} = -z_y + \log\sum_j e^{z_j}$，第一项对 $z_i$ 的导数为 $-[i=y]$，第二项为 $p_i$。

        **为什么融合**：(1) 数值：分开算要先得到 $p$ 再取 log，小概率下溢后 $\log 0 = -\infty$；融合后前向走 LSE，反向直接用 $p - y$。(2) 计算：单独的 softmax 反向是向量-雅可比积 $p \odot (g - \langle g, p\rangle)$，融合后省掉这一步。(3) 显存：大词表下可以分块融合，不必保存完整的概率矩阵。

        **延伸**：梯度有界（每个分量绝对值 ≤ 1）且和为 0，这也是 logits 层训练比较稳定的原因之一；而 logits 整体漂移（$\log Z$ 变大）不受这个梯度约束，PaLM 因此加了 z-loss。
      `,
    },
    {
      id: 'qa-ls',
      freq: 2,
      q: md`label smoothing 的作用和代价是什么？大模型预训练为什么一般不用？`,
      a: md`
        **结论**：它把 one-hot 目标换成 $(1-\varepsilon)\,\mathrm{onehot} + \varepsilon/C$，防止模型把正确类 logit 推向无穷大，起正则作用，常能改善分类任务的泛化与校准。

        **代价**：最优解是 $p_y = 1 - \varepsilon + \varepsilon/C$，模型被**系统性地要求“不够自信”**；它还会让同类样本的表示更紧，丢失“类间相似度”信息（Müller 等 2019 指出这会损害知识蒸馏的效果）。

        **为什么预训练不用**：语言建模的目标是拟合真实的下一个 token 分布，困惑度就是评估指标，人为加入均匀分布等于主动偏离真实分布；并且大词表下 $\varepsilon/C$ 极小，正则效果主要体现为压低正确 token 的概率。需要抑制 logit 漂移时，大模型更常用 z-loss 或对输出层做归一化。
      `,
    },
    {
      id: 'qa-sft',
      freq: 3,
      q: md`SFT 为什么只在回复上算 loss？多轮对话、packing、梯度累积时 loss 要注意什么？`,
      a: md`
        **结论**：prompt 是给定条件，不是要学习的输出；把它算进 loss 会让模型学着“生成用户问题”，并且 prompt 的高 loss 会稀释回复部分的梯度。

        **细节**：
        - 多轮对话：每一轮 assistant 的回复（含结尾的 \`<|im_end|>\` / \`<eos>\`）都算，user / system / tool 输出都设为 −100。
        - packing：多条样本拼成一条时，除了 labels mask，还要让注意力不跨样本（block-diagonal mask 或 \`cu_seqlens\`），position id 每条样本从 0 重新开始。
        - 归约：token 平均让长回复占更大权重，样本平均则相反，需要和训练目标一致。
        - 梯度累积：要用“全部 micro-batch 的 loss 之和 ÷ 全部有效 token 数”，而不是对每个 micro-batch 的平均 loss 再求平均——否则等价于给 token 少的 micro-batch 加权。
      `,
    },
  ],

  cheatsheet: md`
    **稳定 softmax / log_softmax**
    \`\`\`python
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    p = e / e.sum(axis=-1, keepdims=True)
    logp = x - m - np.log(e.sum(axis=-1, keepdims=True))
    \`\`\`

    | 公式 | |
    |---|---|
    | 交叉熵 | $\mathcal{L} = -z_y + \mathrm{LSE}(z)$ |
    | 梯度 | $\partial\mathcal{L}/\partial z = p - q$，和为 0 |
    | label smoothing | $q = (1-\varepsilon)\,\mathrm{onehot} + \varepsilon/C$ |
    | 均匀初始 loss | $\ln V$（32K 词表 ≈ 10.37） |
    | exp 溢出阈值 | fp16 ≈ 11.09，fp32/bf16 ≈ 88.72，fp64 ≈ 709.78 |

    - max 要**按行**取并 \`keepdims=True\`
    - \`ignore_index=-100\`：不计入分子也不计入分母；NumPy 里小心 −100 被当成负索引
    - SFT：prompt 与 padding 设 −100，\`<eos>\` 要算 loss；logits 与 labels 错位一位
    - 梯度累积：总 loss 和 ÷ 总有效 token 数
    - 全屏蔽行：max 为 −inf → 特判，否则 NaN
  `,
})
