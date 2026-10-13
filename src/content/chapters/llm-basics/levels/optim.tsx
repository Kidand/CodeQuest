/**
 * llm-basics-optim · 优化器与学习率
 * SGD 动量、Adam 偏差修正、AdamW 解耦权重衰减、全局范数梯度裁剪、warmup + cosine / WSD、优化器显存。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { BiasCorrectionViz } from '../viz/BiasCorrectionViz'
import { DecayCompareViz } from '../viz/DecayCompareViz'
import { ClipViz } from '../viz/ClipViz'
import { LRScheduleViz } from '../viz/LRScheduleViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'sgd',
      title: '从 SGD 到动量：为什么要“记住”过去的梯度',
      body: md`
        **SGD**：$\theta \leftarrow \theta - \eta\, g_t$。问题有二：mini-batch 梯度噪声大；在“峡谷”形的损失面上，陡峭方向来回震荡、平缓方向进展缓慢。

        **动量**（heavy ball）：用梯度的累积作为更新方向

        $$
        v_t = \mu\, v_{t-1} + g_t,\qquad \theta_t = \theta_{t-1} - \eta\, v_t
        $$

        这是 PyTorch \`SGD(momentum=μ)\` 的写法。震荡方向上正负梯度相互抵消，一致方向上不断累积——稳态下有效步长约放大 $1/(1-\mu)$ 倍（$\mu = 0.9$ 时为 10 倍）。

        另一种写法是**指数滑动平均（EMA）**：$m_t = \beta m_{t-1} + (1-\beta) g_t$。它与上式只差一个常数因子 $(1-\beta)$，可以被学习率吸收——Adam 用的就是 EMA 形式。

        **Nesterov 动量**：在“先按动量走一步之后的位置”看梯度，等价的常用实现是把更新方向换成 $g_t + \mu v_t$（向前多看一步）。后面的 Muon 默认就用 Nesterov 形式。

        > [!NOTE] 为什么 LLM 不用纯 SGD
        > Transformer 不同参数的梯度尺度相差几个数量级（embedding、LayerNorm 增益、注意力投影……），单一学习率无法兼顾。需要**逐坐标自适应**的步长——这就是 Adam。
      `,
      checkpoint: {
        id: 'cp-sgd',
        type: 'numeric',
        prompt: md`PyTorch 形式的动量 $v_t = \mu v_{t-1} + g_t$，$\mu = 0.9$，梯度恒为 $g$。稳态时 $v_t$ 是 $g$ 的多少倍？`,
        answer: 10,
        tolerance: 0,
        explanation: md`稳态 $v = \mu v + g \Rightarrow v = g/(1-\mu) = 10g$。若用 EMA 形式 $m = \beta m + (1-\beta) g$，稳态则是 $m = g$——两者只差常数倍。`,
      },
    },
    {
      id: 'adam',
      title: 'Adam：一阶矩、二阶矩与偏差修正',
      body: md`
        $$
        \begin{aligned}
        m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, & v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
        \hat{m}_t &= \frac{m_t}{1-\beta_1^t}, & \hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
        \theta_t &= \theta_{t-1} - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
        \end{aligned}
        $$

        - $m$ 估计梯度均值（方向），$v$ 估计梯度平方的均值（尺度）。$\hat{m}/\sqrt{\hat{v}}$ 是**无量纲**的：梯度整体乘以 1000，更新量不变。每个坐标的步长大约是 $\eta$，与该参数梯度的绝对大小无关。
        - **偏差修正**：$m_0 = v_0 = 0$，早期的 EMA 被严重低估。对恒定梯度，$m_t = (1-\beta_1^t)g$，除以 $1-\beta_1^t$ 正好还原。$t$ 从 **1** 开始计数。
        - 默认值 $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}$；LLM 预训练普遍把 $\beta_2$ 调到 **0.95**（GPT-3、Llama 2、DeepSeek-V4 的 AdamW 部分），让 $v$ 更快跟上梯度尺度的变化，减少 loss 尖刺。

        下图回答：如果不做偏差修正，前期步长会偏离多少？
      `,
      viz: BiasCorrectionViz,
      after: md`
        > [!WARN] β2 越大，越离不开偏差修正
        > $\beta_1 = 0.9, \beta_2 = 0.999$ 时，未修正的步长在第 1 步是目标值的 $0.1/\sqrt{0.001} \approx 3.16$ 倍，第 12 步附近达到约 6.6 倍，到第 1000 步仍有 1.26 倍。
        > 训练初期参数最“脆弱”，这正是不修正就容易发散的时候。
      `,
      checkpoint: {
        id: 'cp-adam',
        type: 'single',
        prompt: md`所有参数的梯度同时乘以 100（例如 loss 被放大了 100 倍），Adam 的更新量会怎样（忽略 $\varepsilon$）？`,
        options: ['放大 100 倍', '放大 10 倍', '基本不变', '缩小 100 倍'],
        answer: 2,
        explanation: md`$\hat{m}$ 放大 100 倍，$\sqrt{\hat{v}}$ 也放大 100 倍，比值不变。这是 Adam 对梯度尺度不敏感的根源，也是混合精度 loss scaling 不影响 Adam 更新的原因（前提是 unscale 正确、$\varepsilon$ 相对很小）。SGD 则会放大 100 倍。`,
      },
    },
    {
      id: 'adamw',
      title: 'AdamW：为什么 Adam + L2 ≠ 权重衰减',
      body: md`
        对 SGD，L2 正则与权重衰减等价：在 loss 中加 $\frac{\lambda}{2}\|\theta\|^2$，梯度多出 $\lambda\theta$，更新为 $\theta \leftarrow \theta - \eta(g + \lambda\theta) = (1 - \eta\lambda)\theta - \eta g$。

        但在 Adam 里，$\lambda\theta$ 混进梯度后会**一起被 $\sqrt{\hat{v}}$ 除**：

        $$
        \text{Adam + L2:}\quad \theta \leftarrow \theta - \eta\,\frac{\widehat{(g + \lambda\theta)}}{\sqrt{\hat{v}} + \varepsilon}
        $$

        于是某个参数的有效衰减强度 $\approx \eta\lambda / \sqrt{\hat{v}}$：**历史梯度大的参数几乎不衰减，梯度小的参数被过度衰减**——和“对大权重施加更强约束”的初衷相反。

        **AdamW**（Loshchilov & Hutter, ICLR 2019）把衰减从梯度里拿出来，直接作用在参数上：

        $$
        \theta \leftarrow \theta - \eta\left(\frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon} + \lambda\,\theta\right)
        $$

        PyTorch 实现为先 \`p.mul_(1 - lr * weight_decay)\`，再做 Adam 更新。注意衰减项**乘了学习率**，所以学习率调度也会调度衰减强度。LLM 常用 $\lambda = 0.1$；偏置与 norm 的增益通常不做衰减。
      `,
      viz: DecayCompareViz,
      after: md`
        > [!INTERVIEW] 一句话说清楚
        > “Adam 的更新被 $\sqrt{\hat v}$ 归一化，L2 的梯度项也被归一化了，所以梯度大的参数衰减弱、梯度小的参数衰减强；AdamW 把衰减解耦出来，所有参数都按 $(1-\eta\lambda)$ 等比例缩小，正则效果与梯度尺度无关。”
      `,
      checkpoint: {
        id: 'cp-adamw',
        type: 'judge',
        prompt: md`对**不带动量的 SGD**，L2 正则（系数 $\lambda$）与解耦权重衰减（系数 $\lambda$）的更新完全相同；对 Adam 则不同。`,
        answer: true,
        explanation: md`SGD：$\theta - \eta(g + \lambda\theta) = (1-\eta\lambda)\theta - \eta g$，两者一致。带动量的 SGD 已有细微差别（L2 项会进入动量缓冲区），而 Adam 的差别最大，因为它会除以 $\sqrt{\hat{v}}$。`,
      },
    },
    {
      id: 'clip',
      title: '全局范数梯度裁剪：防止一步走飞',
      body: md`
        大模型训练中偶尔会出现梯度尖刺（坏数据、注意力 logit 爆炸、低精度溢出）。裁剪把整体梯度的范数限制在 \`max_norm\` 以内：

        $$
        \text{total} = \sqrt{\sum_i \|g_i\|_2^2},\qquad c = \min\!\Big(1, \frac{\text{max\_norm}}{\text{total} + 10^{-6}}\Big),\qquad g_i \leftarrow c\cdot g_i
        $$

        这正是 \`torch.nn.utils.clip_grad_norm_\` 的实现：先把**所有参数的梯度看成一个大向量**求 L2 范数，再统一乘系数。

        - **为什么用全局范数**：所有张量乘同一个系数，梯度**方向不变**，只是步子变小。逐张量裁剪会改变各部分的相对比例，等于换了一个下降方向。
        - **位置**：\`backward()\` 之后、\`optimizer.step()\` 之前；混合精度 fp16 训练要先 \`scaler.unscale_(optimizer)\` 再裁剪，否则裁的是放大后的梯度。
        - **数值**：GPT-3、Llama 2 都是 1.0。注意它对 Adam 的影响有限：裁剪系数在 $\hat{m}/\sqrt{\hat{v}}$ 中会部分抵消，主要作用是阻止异常梯度污染 $m$ 和 $v$。
        - 分布式（TP / PP / FSDP）下各卡只持有部分梯度，要先在各卡算平方和，再 all-reduce 得到全局范数。
      `,
      viz: ClipViz,
      checkpoint: {
        id: 'cp-clip',
        type: 'fill',
        prompt: md`两个参数的梯度为 $g_1 = [3, 4]$、$g_2 = [12]$，\`max_norm = 1\`。全局范数 = ____(1)____，裁剪后 $g_2$ = ____(2)____（保留 3 位小数，忽略 $10^{-6}$）。`,
        blanks: [
          { label: '(1)', answers: ['13'], numeric: true, tolerance: 0.001 },
          { label: '(2)', answers: ['0.923'], numeric: true, tolerance: 0.002 },
        ],
        explanation: md`total $= \sqrt{9 + 16 + 144} = 13$，系数 $1/13$，$g_2 = 12/13 \approx 0.923$，$g_1 = [0.231, 0.308]$。裁剪后全局范数恰为 1。`,
      },
    },
    {
      id: 'schedule',
      title: '学习率调度：warmup + cosine，以及 WSD',
      body: md`
        **warmup**：前 $W$ 步把学习率从接近 0 线性升到峰值。原因：初始时 Adam 的 $\hat{v}$ 只基于很少的样本，自适应步长方差大（RAdam 论文的分析），参数又处在随机初始化附近，大步长容易把模型推进坏区域；Post-LN 结构尤其依赖 warmup。

        **cosine 衰减**（warmup 之后，进度 $p \in [0, 1]$）：

        $$
        \eta(p) = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\big(1 + \cos(\pi p)\big)
        $$

        通常 $\eta_{\min} = 10\%\,\eta_{\max}$（GPT-3、Llama 2）。缺点：形状依赖**事先确定的总步数**，中途想多训一点就得重新规划。

        **WSD（Warmup–Stable–Decay）**：warmup 后长时间保持峰值，只在最后一小段（如 10%–20%）快速衰减（MiniCPM 推广了这种调度）。好处是稳定段的任何 checkpoint 接一段衰减就是一个“训练完”的模型，适合续训和做 scaling 实验。

        | 模型 | 调度（技术报告原文） |
        |---|---|
        | Llama 2 | warmup 2000 步，cosine 衰减到峰值的 10% |
        | GLM-4.5 | warmup 到 2.5e-4，cosine 衰减到 2.5e-5（报告明确说没用 WSD） |
        | DeepSeek-V4-Flash | warmup 2000 步，大部分时间保持 2.7e-4，临近结束按 cosine 衰减到 2.7e-5 |
      `,
      viz: LRScheduleViz,
      checkpoint: {
        id: 'cp-schedule',
        type: 'numeric',
        prompt: md`cosine 调度，$\eta_{\max} = 3\times10^{-4}$，$\eta_{\min} = 3\times10^{-5}$，warmup 之后的进度恰好过半（$p = 0.5$）。此时学习率是多少？（用科学计数法，如 1.5e-4）`,
        answer: 1.65e-4,
        tolerance: 0.005,
        explanation: md`$p = 0.5$ 时 $\cos(\pi/2) = 0$，$\eta = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) = 3\times10^{-5} + 1.35\times10^{-4} = 1.65\times10^{-4}$，即最大值与最小值的中点。`,
      },
    },
    {
      id: 'memory',
      title: '优化器状态占多少显存',
      body: md`
        Adam 为每个参数维护 $m$ 和 $v$ 两个状态，通常是 **fp32**。混合精度训练时每个参数的显存账：

        | 项 | 字节 / 参数 |
        |---|---|
        | bf16 权重（前向反向用） | 2 |
        | bf16 梯度 | 2 |
        | fp32 主权重（master weights） | 4 |
        | fp32 一阶矩 $m$ | 4 |
        | fp32 二阶矩 $v$ | 4 |
        | **合计** | **16** |

        7B 模型仅这些就要 $7\times10^9 \times 16\,\text{B} = 112\,\text{GB}$，还没算激活。省显存的方向：

        - **ZeRO / FSDP**：把优化器状态、梯度、参数切分到数据并行的各卡（ZeRO-1/2/3）；
        - **8-bit Adam**（Dettmers 等）：$m$、$v$ 分块量化到 8 bit；
        - **Adafactor**：把二阶矩矩阵分解为行、列两个向量；
        - **Muon**：隐藏层矩阵只需一个动量缓冲区，优化器状态约为 AdamW 的一半（见「Muon 优化器」一关）。
      `,
      checkpoint: {
        id: 'cp-memory',
        type: 'numeric',
        prompt: md`一个 70B 参数的模型用混合精度 AdamW 训练（每参数 16 字节），不切分，权重 + 梯度 + 优化器状态共需多少 GB？（1 GB = $10^9$ 字节）`,
        answer: 1120,
        tolerance: 0.001,
        unit: 'GB',
        explanation: md`$70\times10^9 \times 16 = 1.12\times10^{12}$ 字节 = 1120 GB，需要至少 14 张 80 GB 卡才放得下——这就是 ZeRO-3 / FSDP 必不可少的原因。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-mv',
      type: 'single',
      difficulty: 1,
      prompt: md`Adam 中的 $m_t$ 和 $v_t$ 分别是什么的指数滑动平均？`,
      options: ['梯度、梯度的平方（逐元素）', '参数、参数的平方', '梯度、梯度的 L2 范数', '损失、损失的平方'],
      answer: 0,
      explanation: md`$m$ 是梯度的 EMA（一阶矩），$v$ 是逐元素平方梯度的 EMA（二阶原点矩），因此每个参数都有自己的 $v$，实现逐坐标自适应。C 的“整体范数”是 LAMB / 裁剪里的概念。`,
    },
    {
      id: 'q-adamw-judge',
      type: 'judge',
      difficulty: 1,
      prompt: md`AdamW 把 $\lambda\theta$ 加到梯度上，再一起送入 $m$、$v$ 的更新。`,
      answer: false,
      explanation: md`那是 Adam + L2。AdamW 的衰减**不进入** $m$、$v$，而是直接作用在参数上：$\theta \leftarrow \theta(1 - \eta\lambda)$，再做 Adam 更新。`,
    },
    {
      id: 'q-nobias',
      type: 'numeric',
      difficulty: 2,
      prompt: md`$\beta_1 = 0.9$，$\beta_2 = 0.999$，若**不做**偏差修正，第 1 步的更新量 $|m_1/\sqrt{v_1}|$ 是做了修正时（$=1$）的多少倍？（忽略 $\varepsilon$，保留两位小数）`,
      answer: 3.16,
      tolerance: 0.005,
      explanation: md`$m_1 = 0.1g$，$v_1 = 0.001g^2$，$m_1/\sqrt{v_1} = 0.1/\sqrt{0.001} \approx 3.16$；修正后 $\hat{m}_1 = g$，$\hat{v}_1 = g^2$，比值为 1。之后几步这个倍数还会继续上升到约 6.6。`,
    },
    {
      id: 'q-mem',
      type: 'numeric',
      difficulty: 2,
      prompt: md`7B 模型、混合精度 AdamW（bf16 权重与梯度 + fp32 主权重、$m$、$v$），不做任何切分。仅权重、梯度与优化器状态就需要多少 GB？（1 GB = $10^9$ 字节）`,
      answer: 112,
      tolerance: 0.001,
      unit: 'GB',
      explanation: md`每参数 $2 + 2 + 4 + 4 + 4 = 16$ 字节，$7\times10^9 \times 16 = 112$ GB。其中优化器相关（主权重 + $m$ + $v$）占 12 字节，是 ZeRO-1 首先切分的对象。`,
    },
    {
      id: 'q-clip-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 \`clip_grad_norm_(params, max_norm=1.0)\`，正确的是（多选）：`,
      options: [
        '它计算所有参数梯度拼成一个向量后的 L2 范数',
        '当全局范数小于 max_norm 时，梯度保持不变',
        '它分别把每个参数张量的范数裁剪到 max_norm',
        '它应在 `loss.backward()` 之后、`optimizer.step()` 之前调用',
        'fp16 + GradScaler 时，应先 `scaler.unscale_(optimizer)` 再裁剪',
      ],
      answer: [0, 1, 3, 4],
      explanation: md`C 是“逐张量裁剪”，会改变梯度方向，不是 \`clip_grad_norm_\` 的行为。E：不 unscale 的话，梯度仍乘着 loss scale（可能是 $2^{16}$），总范数被严重高估，几乎每一步都会被错误地大幅裁剪。`,
    },
    {
      id: 'q-wsd-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`warmup 1000 步、峰值 $\eta_{\max} = 6\times 10^{-4}$，第 $t$ 步（从 0 开始）的 warmup 学习率取 $\eta_{\max}\cdot(t+1)/1000$。第 249 步的学习率 = ____(1)____；之后 cosine 衰减到 $\eta_{\min} = 0$，衰减阶段进度 $p = 1/3$ 时学习率 = ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['1.5e-4'], numeric: true, tolerance: 0.002 },
        { label: '(2)', answers: ['4.5e-4'], numeric: true, tolerance: 0.002 },
      ],
      explanation: md`(1) $6\times10^{-4} \times 250/1000 = 1.5\times10^{-4}$。(2) $\frac{1}{2}(1 + \cos(\pi/3)) = \frac{1}{2}(1 + 0.5) = 0.75$，$0.75 \times 6\times10^{-4} = 4.5\times10^{-4}$。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面号称实现 AdamW 的单步更新（$t$ 从 1 开始），有 **2 处** bug，点出来：`,
      code: py`
        def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.1):
            g = g + wd * p
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g
            m_hat = m / (1 - b1 ** t)
            v_hat = v / (1 - b2 ** t)
            p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
            return p, m, v
      `,
      bugLines: [2, 4],
      explanation: md`
        - 第 2 行把衰减加到梯度上，这是 **Adam + L2**，不是 AdamW。应删掉这一行，改为在更新前（或后）执行 \`p = p * (1 - lr * wd)\`。
        - 第 4 行二阶矩少了平方，应为 \`(1 - b2) * g * g\`。否则 $v$ 可能为负，\`np.sqrt\` 得到 NaN。
        第 5–7 行的偏差修正与 $\varepsilon$ 位置（加在根号外，与 PyTorch 一致）都是对的。
      `,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把一个标准训练步按正确顺序排列：`,
      items: [
        '`optimizer.zero_grad()`',
        '前向计算 `loss = model(batch)`',
        '`loss.backward()`',
        '`clip_grad_norm_(model.parameters(), 1.0)`',
        '`optimizer.step()`',
        '`scheduler.step()`',
      ],
      explanation: md`清梯度 → 前向 → 反向 → 裁剪（必须在梯度算出之后、更新之前）→ 更新参数 → 更新学习率（为下一步准备）。\`zero_grad\` 放在 \`step\` 之后也可以，关键是不能夹在 backward 与 step 之间。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把 LLM 预训练中 AdamW 的常见超参数与典型取值配对（Llama 2 的设置）：`,
      pairs: [
        ['β1', '0.9'],
        ['β2', '0.95'],
        ['weight decay', '0.1'],
        ['梯度裁剪 max_norm', '1.0'],
        ['warmup', '2000 步'],
      ],
      explanation: md`Llama 2 论文：AdamW（$\beta_1 = 0.9, \beta_2 = 0.95, \varepsilon = 10^{-5}$），cosine 调度、warmup 2000 步、最终衰减到峰值的 10%，weight decay 0.1，梯度裁剪 1.0。这组设置几乎成了开源 LLM 预训练的默认值。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出“线性 warmup + cosine 衰减”的学习率函数。有 2 行干扰项。`,
      lines: [
        'def lr_at(step, max_lr, min_lr, warmup, total):',
        '    if step < warmup:',
        '        return max_lr * (step + 1) / warmup',
        '    p = min(1.0, (step - warmup) / (total - warmup))',
        '    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))',
      ],
      distractors: ['    p = min(1.0, step / total)', '    return max_lr * 0.5 * (1 + math.cos(math.pi * p))'],
      explanation: md`进度要从 warmup 结束时算起，否则 warmup 结束时 lr 会突然掉下来（干扰项 1）；干扰项 2 忽略了 \`min_lr\`，最终衰减到 0。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`按常见做法（如 nanoGPT：二维及以上的权重做衰减，其余不做），下列参数中哪些**不做**权重衰减？`,
      items: ['q_proj.weight', 'mlp.up_proj.weight', 'input_layernorm.weight', 'q_proj.bias', 'lm_head.weight', 'final_norm.bias'],
      answer: [2, 3, 5],
      explanation: md`norm 的增益与偏置、各种 bias 都是一维参数，衰减它们会把 norm 的增益拉向 0、限制表达力，几乎所有代码库都把它们排除在外。矩阵（含 \`lm_head\`）做衰减。embedding 是否衰减各代码库不一（nanoGPT 衰减，minGPT 不衰减）。`,
    },
    {
      id: 'q-why-not-l2',
      type: 'single',
      difficulty: 3,
      prompt: md`Adam + L2（系数 $\lambda$）中，参数 $\theta_i$ 的历史梯度二阶矩 $\hat{v}_i$ 很大。与 AdamW 相比，它受到的有效衰减：`,
      options: ['更强，约为 $\\eta\\lambda\\sqrt{\\hat v_i}$', '相同', '更弱，约为 $\\eta\\lambda / \\sqrt{\\hat v_i}$', '为 0'],
      answer: 2,
      explanation: md`L2 项 $\lambda\theta_i$ 进入梯度后与真实梯度一起被 $\sqrt{\hat v_i}$ 除，衰减量约 $\eta\lambda\theta_i/\sqrt{\hat v_i}$；$\hat v_i$ 大则衰减弱。AdamW 的衰减量是 $\eta\lambda\theta_i$，与 $\hat v_i$ 无关。`,
    },
    {
      id: 'q-warmup',
      type: 'single',
      difficulty: 3,
      prompt: md`对 Adam 类优化器，学习率 warmup 最主要的动机是：`,
      options: [
        '让偏差修正项 $1-\\beta^t$ 尽快趋于 1',
        '训练初期 $\\hat v$ 只基于极少数样本，自适应步长方差大、不可靠，小学习率可以避免早期更新把参数推到坏区域',
        '节省显存',
        '增强权重衰减的效果',
      ],
      answer: 1,
      explanation: md`RAdam（Liu 等）从“自适应学习率的方差”角度解释了 warmup 的必要性；另外初始化附近的损失面曲率大、Post-LN 初始梯度大，也需要小步长。A：偏差修正本身已处理 EMA 的低估，warmup 不是为了它；C、D 与 warmup 无关（AdamW 的衰减乘了 lr，warmup 反而会**减弱**早期衰减）。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-schedule',
      title: '写一个 warmup + cosine 学习率函数',
      intro: md`
        实现 \`lr_at(step, max_lr, warmup, total, min_lr=0.0)\`，\`step\` 从 0 开始：
        - \`step < warmup\`：线性 warmup，返回 $\eta_{\max}\cdot(\text{step}+1)/\text{warmup}$；
        - \`step >= total\`：返回 \`min_lr\`；
        - 其余：cosine 从 \`max_lr\` 衰减到 \`min_lr\`，进度从 warmup 结束时算起。
      `,
      template: py`
        import math

        def lr_at(step, max_lr, warmup, total, min_lr=0.0):
            if step < warmup:
                return {{1}}
            if step >= total:
                return {{2}}
            progress = {{3}}
            coeff = {{4}}
            return min_lr + coeff * (max_lr - min_lr)
      `,
      steps: [
        {
          blank: 1,
          instruction: md`线性 warmup：第 0 步就有一个很小但非零的学习率，第 \`warmup - 1\` 步达到峰值。`,
          answer: 'max_lr * (step + 1) / warmup',
          accept: ['max_lr * (step + 1.0) / warmup', '(step + 1) / warmup * max_lr'],
          explain: md`用 \`step + 1\` 而不是 \`step\`，避免第 0 步学习率为 0（那一步等于白算）。`,
        },
        {
          blank: 2,
          instruction: md`训练步数超过计划时，保持在最低学习率。`,
          answer: 'min_lr',
          explain: md`防止进度大于 1 时 cosine 又开始“回升”。`,
        },
        {
          blank: 3,
          instruction: md`计算衰减阶段的进度 $p \in [0, 1)$：从 warmup 结束时开始算。`,
          answer: '(step - warmup) / (total - warmup)',
          accept: ['(step - warmup) / max(1, total - warmup)'],
          explain: md`若写成 \`step / total\`，warmup 结束的那一刻 lr 会从峰值突然掉到 cosine 曲线上对应 \`warmup/total\` 的位置。`,
        },
        {
          blank: 4,
          instruction: md`cosine 系数：$p=0$ 时为 1，$p=1$ 时为 0。`,
          answer: '0.5 * (1 + math.cos(math.pi * progress))',
          accept: ['0.5 * (1.0 + math.cos(math.pi * progress))', '(1 + math.cos(math.pi * progress)) / 2'],
          explain: md`$\frac{1}{2}(1 + \cos\pi p)$ 在两端导数为 0：开头缓降、中段最快、末尾趋平。`,
        },
      ],
      setup: py`
        def _ref_lr(step, max_lr, warmup, total, min_lr=0.0):
            if step < warmup:
                return max_lr * (step + 1) / warmup
            if step >= total:
                return min_lr
            p = (step - warmup) / (total - warmup)
            return min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (max_lr - min_lr)
      `,
      tests: [
        {
          name: 'warmup 段',
          code: py`
            assert abs(lr_at(0, 1e-3, 100, 1000) - 1e-5) < 1e-12, f"第 0 步应为 1e-5，得到 {lr_at(0, 1e-3, 100, 1000)}"
            assert abs(lr_at(99, 1e-3, 100, 1000) - 1e-3) < 1e-12, f"第 99 步应达到峰值，得到 {lr_at(99, 1e-3, 100, 1000)}"
          `,
        },
        {
          name: 'cosine 段与终点',
          code: py`
            got = lr_at(550, 1e-3, 100, 1000, min_lr=1e-4)
            assert abs(got - 5.5e-4) < 1e-12, f"进度过半应为 (max+min)/2 = 5.5e-4，得到 {got}"
            assert lr_at(1000, 1e-3, 100, 1000, min_lr=1e-4) == 1e-4, "超过 total 应返回 min_lr"
            assert abs(lr_at(100, 1e-3, 100, 1000) - 1e-3) < 1e-12, "warmup 结束时应仍为峰值（无跳变）"
          `,
        },
        {
          name: '全程对拍',
          hidden: true,
          code: py`
            for s in range(0, 1300, 7):
                got, exp = lr_at(s, 3e-4, 200, 1200, 3e-5), _ref_lr(s, 3e-4, 200, 1200, 3e-5)
                assert abs(got - exp) < 1e-15, f"step={s}：期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        想改成 WSD：在 cosine 之前插入一段“stable”——\`if step < decay_start: return max_lr\`，然后让进度从 \`decay_start\` 开始计算。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-adamw',
      title: '手写 AdamW 单步更新',
      difficulty: 'medium',
      source: '大模型岗手撕 · 高频',
      timeLimitMin: 15,
      description: md`
        实现与 \`torch.optim.AdamW\` 语义一致的单步更新：

        \`\`\`python
        def adamw_step(param, grad, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
            """返回 (new_param, new_m, new_v)；t 为当前步数，从 1 开始；不要原地修改输入数组"""
        \`\`\`

        更新规则（PyTorch 顺序）：
        1. 解耦权重衰减：$\theta \leftarrow \theta\,(1 - \eta\lambda)$
        2. $m \leftarrow \beta_1 m + (1-\beta_1) g$，$v \leftarrow \beta_2 v + (1-\beta_2) g^2$
        3. $\hat m = m/(1-\beta_1^t)$，$\hat v = v/(1-\beta_2^t)$
        4. $\theta \leftarrow \theta - \eta\,\hat m / (\sqrt{\hat v} + \varepsilon)$

        \`param\`、\`grad\`、\`m\`、\`v\` 为同形状的 float 数组。
      `,
      starter: py`
        import numpy as np

        def adamw_step(param, grad, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
            # TODO: 解耦衰减 → 更新一阶/二阶矩 → 偏差修正 → 更新参数
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def adamw_step(param, grad, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
            b1, b2 = betas
            p = param * (1.0 - lr * weight_decay)            # 1) 解耦权重衰减（不进入 m、v）
            m = b1 * m + (1.0 - b1) * grad                   # 2) 一阶矩
            v = b2 * v + (1.0 - b2) * grad * grad            #    二阶矩（逐元素平方）
            m_hat = m / (1.0 - b1 ** t)                      # 3) 偏差修正，t 从 1 开始
            v_hat = v / (1.0 - b2 ** t)
            p = p - lr * m_hat / (np.sqrt(v_hat) + eps)      # 4) eps 加在根号外
            return p, m, v
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：最小 AdamW 优化器）',
          code: py`
            import torch

            class MiniAdamW(torch.optim.Optimizer):
                def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
                    super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

                @torch.no_grad()
                def step(self):
                    for group in self.param_groups:
                        b1, b2 = group["betas"]
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            st = self.state[p]
                            if not st:
                                st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                            st["t"] += 1
                            t, m, v = st["t"], st["m"], st["v"]
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            m.lerp_(p.grad, 1 - b1)                       # m = b1*m + (1-b1)*g
                            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                            denom = (v / (1 - b2 ** t)).sqrt_().add_(group["eps"])
                            p.addcdiv_(m, denom, value=-group["lr"] / (1 - b1 ** t))
          `,
        },
      ],
      explanation: md`
        **关键点**
        - 衰减作用在参数上、不进入 $m, v$——这是与 Adam + L2 的唯一区别，也是最常被问的点。
        - 偏差修正的 $t$ 从 1 开始；从 0 开始会除以 0。
        - $\varepsilon$ 加在 $\sqrt{\hat v}$ 之外（PyTorch 约定）。
        - 第 1 步：$\hat m = g$，$\hat v = g^2$，更新量约为 $\eta\cdot\mathrm{sign}(g)$，与梯度大小无关——测试会检查这一点。

        **复杂度**：$O(n)$ 时间，额外状态 $2n$（$m$、$v$）。
      `,
      complexity: { time: 'O(n)', space: 'O(n)' },
      hints: [
        md`先 \`p = param * (1 - lr * weight_decay)\`，衰减与梯度无关。`,
        md`\`v\` 用 \`grad * grad\`；偏差修正分母是 \`1 - b ** t\`。`,
        md`最后 \`p - lr * m_hat / (np.sqrt(v_hat) + eps)\`，返回 \`(p, m, v)\`。`,
      ],
      setup: py`
        def _ref_adamw(p, g, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.01):
            b1, b2 = betas
            p = p * (1 - lr * wd)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            return p - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps), m, v
      `,
      tests: [
        {
          name: '第 1 步：步长约等于 lr',
          code: py`
            p0 = np.array([1.0, -2.0, 0.5])
            g = np.array([1e-3, 50.0, -3.0])
            p, m, v = adamw_step(p0, g, np.zeros(3), np.zeros(3), t=1, lr=0.01, weight_decay=0.0)
            exp = p0 - 0.01 * np.sign(g)
            assert p.shape == (3,), f"形状应为 (3,)，得到 {p.shape}"
            assert np.allclose(p, exp, atol=1e-6), f"偏差修正后第 1 步更新应为 lr·sign(g)：期望 {exp}，得到 {p}"
          `,
        },
        {
          name: '零梯度时只做解耦衰减',
          code: py`
            p0 = np.array([2.0, -4.0])
            p, m, v = adamw_step(p0, np.zeros(2), np.zeros(2), np.zeros(2), t=1, lr=0.1, weight_decay=0.5)
            assert np.allclose(p, p0 * (1 - 0.1 * 0.5)), f"期望 {p0 * 0.95}，得到 {p}"
          `,
        },
        {
          name: '多步对拍',
          code: py`
            rng = np.random.default_rng(0)
            p, m, v = rng.normal(size=5), np.zeros(5), np.zeros(5)
            rp, rm, rv = p.copy(), m.copy(), v.copy()
            for t in range(1, 21):
                g = rng.normal(size=5)
                p, m, v = adamw_step(p, g, m, v, t, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
                rp, rm, rv = _ref_adamw(rp, g, rm, rv, t, lr=3e-3, betas=(0.9, 0.95), wd=0.1)
            assert np.allclose(p, rp, atol=1e-12), f"20 步后参数最大误差 {np.abs(p - rp).max():.2e}"
            assert np.allclose(m, rm) and np.allclose(v, rv), "m / v 与参考实现不一致"
          `,
        },
        {
          name: '解耦：衰减与梯度尺度无关',
          hidden: true,
          code: py`
            # 两个参数初值相同、梯度方向相同但尺度差 1e4：AdamW 下它们的结果应完全相同（忽略 eps）
            p0 = np.array([1.0, 1.0])
            m, v = np.zeros(2), np.zeros(2)
            p = p0.copy()
            for t in range(1, 11):
                g = np.array([1e-2, 1e2]) * (1 if t % 2 else -1)
                p, m, v = adamw_step(p, g, m, v, t, lr=0.01, weight_decay=0.1)
            assert abs(p[0] - p[1]) < 1e-6, f"衰减被梯度尺度影响了（Adam + L2？）：{p}"
          `,
        },
        {
          name: '不修改输入',
          hidden: true,
          code: py`
            p0, g = np.array([1.0, 2.0]), np.array([0.5, -0.5])
            m0, v0 = np.array([0.1, 0.1]), np.array([0.2, 0.2])
            c = [a.copy() for a in (p0, g, m0, v0)]
            adamw_step(p0, g, m0, v0, t=3)
            for a, b in zip((p0, g, m0, v0), c):
                assert np.array_equal(a, b), "输入数组被原地修改了"
          `,
        },
        {
          name: '高维对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(5)
            p, m, v = rng.normal(size=(4, 6)), np.zeros((4, 6)), np.zeros((4, 6))
            rp, rm, rv = p.copy(), m.copy(), v.copy()
            for t in range(1, 51):
                g = rng.normal(size=(4, 6)) * rng.choice([1e-3, 1.0, 1e3])
                p, m, v = adamw_step(p, g, m, v, t, lr=1e-3, weight_decay=0.05)
                rp, rm, rv = _ref_adamw(rp, g, rm, rv, t, lr=1e-3, wd=0.05)
            assert np.allclose(p, rp, atol=1e-12), f"最大误差 {np.abs(p - rp).max():.2e}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-lr-wd',
          q: md`PyTorch 的 AdamW 衰减项乘了学习率（$\eta\lambda$），这有什么影响？`,
          a: md`
            衰减强度随学习率调度一起变化：warmup 期间衰减很弱，cosine 末期也很弱。有效的“衰减时间尺度”约为 $1/(\eta\lambda)$ 步。
            原论文的 AdamW 写作 $\theta \leftarrow \theta - \eta_t(\ldots + \lambda\theta)$，其中 $\eta_t$ 是调度乘子；也有实现使用不乘学习率的“完全解耦”版本，此时 $\lambda$ 的合理取值要小几个数量级。换优化器框架时要核对这一点，否则同样的 \`weight_decay=0.1\` 含义完全不同。
          `,
        },
      ],
    },
    {
      id: 'c-clip',
      title: '全局范数梯度裁剪 clip_grad_norm_',
      difficulty: 'easy',
      source: '大模型岗手撕 · 训练稳定性',
      timeLimitMin: 10,
      description: md`
        实现与 \`torch.nn.utils.clip_grad_norm_\` 语义一致的函数：

        \`\`\`python
        def clip_grad_norm_(grads: list[np.ndarray], max_norm: float, norm_type: float = 2.0) -> float: ...
        \`\`\`

        - 把所有梯度看成一个大向量，计算其 \`norm_type\` 范数 \`total\`（支持 \`2.0\` 与 \`float('inf')\`，后者为所有元素绝对值的最大值）。
        - 系数 \`coef = max_norm / (total + 1e-6)\`；若 \`coef < 1\`，把**每个**梯度**原地**乘以 \`coef\`（\`g *= coef\`），否则不变。
        - 返回裁剪**前**的 \`total\`（Python float）。空列表返回 \`0.0\`。

        **示例**：\`grads = [[3, 4], [12]]\`，\`max_norm = 1\` → 返回 \`13.0\`，梯度变为 \`[[0.2308, 0.3077], [0.9231]]\`。
      `,
      starter: py`
        import numpy as np

        def clip_grad_norm_(grads, max_norm, norm_type=2.0):
            # TODO: 1) 全局范数  2) 系数  3) 原地缩放  4) 返回裁剪前的范数
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def clip_grad_norm_(grads, max_norm, norm_type=2.0):
            if len(grads) == 0:
                return 0.0
            if norm_type == float("inf"):
                total = max(float(np.abs(g).max()) if g.size else 0.0 for g in grads)
            else:
                # 先求每个张量的范数，再对这些范数求范数（与 PyTorch 实现一致）
                per = np.array([np.sum(np.abs(g) ** norm_type) ** (1.0 / norm_type) for g in grads])
                total = float(np.sum(per ** norm_type) ** (1.0 / norm_type))
            coef = max_norm / (total + 1e-6)
            if coef < 1.0:
                for g in grads:
                    g *= coef          # 原地修改，所有张量乘同一个系数
            return total
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            @torch.no_grad()
            def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
                grads = [p.grad for p in parameters if p.grad is not None]
                if not grads:
                    return torch.tensor(0.0)
                if norm_type == float("inf"):
                    total = torch.stack([g.abs().max() for g in grads]).max()
                else:
                    total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, norm_type) for g in grads]), norm_type)
                coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
                for g in grads:
                    g.mul_(coef)
                return total
          `,
        },
      ],
      explanation: md`
        **要点**：范数是“所有参数拼成一个向量”的范数，等价于“各张量范数的范数”；所有张量乘**同一个**系数，保持梯度方向。
        PyTorch 的写法是 \`coef_clamped = clamp(coef, max=1)\` 后无条件相乘，效果相同。

        **易错点**：逐张量裁剪；返回裁剪后的范数（应返回裁剪前，日志里看到的 \`grad_norm\` 就是它）；忘了原地修改（\`g = g * coef\` 只是重新绑定局部变量）。

        **分布式**：TP / FSDP 下各卡只有部分梯度，要先本地求 $\sum\|g\|^2$，all-reduce 后再开方。

        **复杂度**：$O(n)$。
      `,
      complexity: { time: 'O(n)', space: 'O(k)' },
      hints: [
        md`L2 情形：\`total = sqrt(sum(np.sum(g * g) for g in grads))\`。`,
        md`原地修改用 \`g *= coef\`，不要写 \`g = g * coef\`。`,
        md`\`norm_type=inf\` 时 total 是所有元素绝对值的最大值。`,
      ],
      setup: py`
        def _ref_total(grads, p):
            if p == float("inf"):
                return max(float(np.abs(g).max()) for g in grads)
            return float(sum(np.sum(np.abs(g) ** p) for g in grads) ** (1 / p))
      `,
      tests: [
        {
          name: '示例',
          code: py`
            gs = [np.array([3.0, 4.0]), np.array([12.0])]
            total = clip_grad_norm_(gs, 1.0)
            assert abs(total - 13.0) < 1e-9, f"应返回裁剪前的范数 13，得到 {total}"
            assert np.allclose(gs[0], [3 / 13, 4 / 13], atol=1e-6) and np.allclose(gs[1], [12 / 13], atol=1e-6), f"原地裁剪结果错误：{gs}"
          `,
        },
        {
          name: '未超过阈值不变',
          code: py`
            gs = [np.array([0.3, -0.4])]
            total = clip_grad_norm_(gs, 1.0)
            assert abs(total - 0.5) < 1e-9, f"期望 0.5，得到 {total}"
            assert np.allclose(gs[0], [0.3, -0.4]), f"不应修改，得到 {gs[0]}"
          `,
        },
        {
          name: '无穷范数',
          code: py`
            gs = [np.array([1.0, -5.0]), np.array([[2.0, 3.0]])]
            total = clip_grad_norm_(gs, 2.5, norm_type=float("inf"))
            assert abs(total - 5.0) < 1e-9, f"无穷范数应为 5，得到 {total}"
            assert np.allclose(gs[0], [0.5, -2.5], atol=1e-6), f"得到 {gs[0]}"
          `,
        },
        {
          name: '方向保持（全局而非逐张量）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(0)
            gs = [rng.normal(size=(3, 4)), rng.normal(size=7) * 50, rng.normal(size=(2, 2, 2))]
            orig = np.concatenate([g.ravel() for g in gs])
            total = clip_grad_norm_(gs, 1.0)
            new = np.concatenate([g.ravel() for g in gs])
            cos = orig @ new / (np.linalg.norm(orig) * np.linalg.norm(new))
            assert abs(total - np.linalg.norm(orig)) < 1e-6, f"范数应为 {np.linalg.norm(orig)}，得到 {total}"
            assert cos > 1 - 1e-9, f"方向被改变了（余弦 {cos}）——是不是逐张量裁剪？"
            assert abs(np.linalg.norm(new) - 1.0) < 1e-5, f"裁剪后总范数应为 1，得到 {np.linalg.norm(new)}"
          `,
        },
        {
          name: '随机对拍 + 空列表',
          hidden: true,
          code: py`
            assert clip_grad_norm_([], 1.0) == 0.0, "空列表应返回 0.0"
            rng = np.random.default_rng(1)
            for trial in range(20):
                gs = [rng.normal(size=rng.integers(1, 6)) * rng.choice([0.01, 1, 100]) for _ in range(rng.integers(1, 5))]
                p = float(rng.choice([2.0, float("inf")]))
                exp = _ref_total(gs, p)
                got = clip_grad_norm_(gs, 1.0, norm_type=p)
                assert abs(got - exp) < 1e-9 * max(1, exp), f"norm_type={p}：期望 {exp}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-nan',
          q: md`如果某一步的梯度里出现 NaN / inf，\`clip_grad_norm_\` 会怎样？训练中怎么处理？`,
          a: md`
            total 变成 NaN/inf，系数也变成 NaN 或 0，裁剪不仅救不了，还会把所有梯度变成 NaN。PyTorch 提供 \`error_if_nonfinite=True\` 直接报错。
            实践中：fp16 训练由 GradScaler 检测 inf 并**跳过这一步**、降低 loss scale；bf16 训练通常监控 \`grad_norm\`，出现尖刺时跳过该 batch，严重时回滚到更早的 checkpoint 并跳过对应数据（PaLM 报告过：从尖刺前约 100 步的 checkpoint 重启，并跳过约 200–500 个 batch）。
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
      id: 'qa-adamw',
      freq: 3,
      q: md`AdamW 和 Adam + L2 正则有什么区别？为什么不等价？`,
      a: md`
        **结论**：Adam + L2 把 $\lambda\theta$ 加进梯度，AdamW 把衰减直接作用在参数上：$\theta \leftarrow \theta - \eta(\hat m/(\sqrt{\hat v}+\varepsilon) + \lambda\theta)$。

        **原因**：Adam 会把梯度除以 $\sqrt{\hat v}$。L2 项混进梯度后也被除，于是有效衰减 $\approx \eta\lambda\theta/\sqrt{\hat v}$：梯度历史大的参数衰减弱，梯度小的参数衰减强，而且衰减还会进入动量。SGD 没有这个除法，所以二者在 SGD 下等价。AdamW 让所有参数按同一比例 $(1-\eta\lambda)$ 收缩，正则强度与梯度尺度解耦，实验上泛化更好、学习率与衰减系数也更容易分开调。

        **延伸**：PyTorch 的 AdamW 衰减乘了学习率；LLM 常用 $\lambda=0.1$，并排除 bias 与 norm 增益。Muon 训练（Moonlight、Kimi K2、DeepSeek-V4）同样对矩阵参数使用解耦权重衰减，Moonlight 论文把它列为 Muon 能扩展到大规模的两个关键之一。
      `,
    },
    {
      id: 'qa-bias',
      freq: 2,
      q: md`Adam 为什么要做偏差修正？不做会怎样？`,
      a: md`
        **结论**：$m, v$ 从 0 初始化，EMA 在早期被低估，修正因子 $1/(1-\beta^t)$ 把它还原成无偏估计。

        **原因**：对恒定梯度，$m_t = (1-\beta_1^t)g$，$v_t = (1-\beta_2^t)g^2$。不修正时实际步长是目标的 $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$ 倍：$\beta_2 = 0.999$ 时第 1 步约 3.2 倍、第 12 步左右约 6.6 倍、第 1000 步仍约 1.26 倍。训练初期最脆弱，这样的放大容易导致发散。

        **延伸**：$\beta_2$ 越接近 1 越需要修正；$\beta_2 = 0.95$ 时影响很小（最大约 1.1 倍）。warmup 在一定程度上也掩盖了这个问题，但偏差修正几乎零成本，没有理由去掉。
      `,
    },
    {
      id: 'qa-schedule',
      freq: 3,
      q: md`为什么要 warmup？cosine 和 WSD 怎么选？`,
      a: md`
        **结论**：warmup 用小学习率度过“自适应统计量不可靠 + 参数刚初始化”的阶段；衰减阶段让 loss 降到噪声地板以下。cosine 简单稳定，WSD 更灵活。

        **原因**：初期 $\hat v$ 方差大（RAdam 的分析），大步长容易让 loss 尖刺或发散。衰减则是为了收敛：恒定学习率下 SGD 类方法会在最优点附近以与学习率成正比的幅度震荡。cosine 需要事先知道总步数；WSD 在稳定段保持峰值，最后集中衰减，任意 checkpoint 都能“补一段衰减”得到可用模型，方便续训与做 scaling law 实验。

        **延伸**：Llama 2 用 2000 步 warmup + cosine 到 10%；GLM-4.5 明确选择 cosine 而非 WSD；DeepSeek-V4 是 warmup 2000 步、长时间恒定、临近结束再 cosine 衰减——本质上是 WSD 形状。还要提一句：batch size 也常做 warmup（DeepSeek-V3/V4、GLM-4.5 都逐步增大 batch）。
      `,
    },
    {
      id: 'qa-clip',
      freq: 2,
      q: md`梯度裁剪是怎么做的？为什么用全局范数？对 Adam 有用吗？`,
      a: md`
        **结论**：把所有梯度拼成一个向量求 L2 范数，超过阈值就整体乘 $\text{max\_norm}/\text{total}$；LLM 常用阈值 1.0。

        **原因**：全局缩放保持梯度方向，只限制步长；逐张量裁剪会改变各部分比例，相当于换了方向。

        **对 Adam 的作用**：由于 $\hat m/\sqrt{\hat v}$ 对尺度不敏感，裁剪对单步更新量的影响被部分抵消；它的主要价值是防止一次异常大的梯度污染 $m$ 和 $v$（尤其 $v$，会在很多步内压低有效学习率）。

        **延伸**：\`grad_norm\` 是最重要的训练监控指标之一，尖刺往往预示 loss spike；分布式下需要 all-reduce 平方和；fp16 要先 unscale 再裁剪。
      `,
    },
    {
      id: 'qa-mem',
      freq: 2,
      q: md`训练一个 7B 模型，优化器相关的显存是多少？有哪些办法降低？`,
      a: md`
        **结论**：混合精度 AdamW 每参数约 16 字节（bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + $m$ 4 + $v$ 4），7B 模型约 112 GB，其中优化器相关 12 字节/参数 = 84 GB。

        **办法**：(1) ZeRO-1/2/3 或 FSDP 把状态切到数据并行的各卡；(2) 8-bit Adam 量化 $m, v$；(3) Adafactor 用行列分解近似 $v$；(4) CPU offload；(5) 换优化器——Muon 对隐藏层矩阵只保留一个动量缓冲区，状态约为 AdamW 的一半。

        **延伸**：激活显存通常比优化器状态更大，需要配合激活重计算（activation checkpointing）、序列并行等手段。
      `,
    },
  ],

  cheatsheet: md`
    **AdamW（PyTorch 顺序，t 从 1 开始）**
    \`\`\`python
    p *= 1 - lr * wd                       # 解耦衰减
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    p -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
    \`\`\`

    **全局范数裁剪**：\`total = sqrt(Σ‖g‖²)\`，\`coef = min(1, max_norm/(total+1e-6))\`，所有 g 乘 coef

    **warmup + cosine**：$\eta = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})(1+\cos\pi p)$，$p$ 从 warmup 结束算起

    | 数字 | |
    |---|---|
    | LLM 常用 | β=(0.9, 0.95)，wd 0.1，clip 1.0 |
    | 不修正偏差（β2=0.999） | 第 1 步 3.16×，峰值约 6.6× |
    | 混合精度 AdamW | 16 B/参数（7B → 112 GB） |

    - Adam + L2：衰减被 $\sqrt{\hat v}$ 除 → 大梯度参数几乎不衰减
    - bias / norm 增益不做衰减
    - fp16：先 unscale 再裁剪
  `,
})
