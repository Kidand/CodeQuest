/**
 * llm-basics-boss · BOSS：基础算子综合
 * 复盘 + 跨关混合题 + 两道综合手撕：从零训练小 MLP（CE + 反向 + 裁剪 + AdamW + warmup/cosine）、分块的 linear + CE（不物化 logits）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { TrainLoopViz } from '../viz/TrainLoopViz'

export default defineLevel({
  /* ================================================================ */
  /* 复盘                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'review',
      title: '复盘：一个训练 step 里的全部零件',
      body: md`
        本章的每一关都是训练循环中的一个零件。把它们按执行顺序串起来：

        | 阶段 | 零件 | 关键点（易错点） |
        |---|---|---|
        | 前向 | RMSNorm / LayerNorm | 沿最后一维、有偏方差、$\varepsilon$ 在根号内；Pre-Norm + final norm |
        | 前向 | SwiGLU FFN | 激活在 gate 分支；$h \approx \frac{8}{3}d$ 向上对齐 |
        | 前向 | logits → loss | 稳定 log_softmax（减最大值 / LSE）；SFT 只在回复上算、\`ignore_index=-100\` |
        | 反向 | softmax-CE | $\mathrm{d}z = (p - y)/N_{\text{valid}}$ |
        | 反向 | Linear / ReLU / Norm | 形状对齐；偏置梯度求和；LN 反向别漏 $\mu, \sigma$ 的依赖 |
        | 更新前 | 全局范数裁剪 | 所有参数一起算范数，乘同一个系数；fp16 先 unscale |
        | 更新 | AdamW / Muon | 偏差修正 $t$ 从 1 开始；衰减解耦；Muon 只管二维隐藏层矩阵并对齐 update RMS |
        | 更新后 | 学习率调度 / QK-Clip | warmup + cosine / WSD；K2 在 step 之后做 QK-Clip |

        **NaN 排查清单**（几乎都能在本章找到原因）：fp16 下直接 \`exp\`（> 11.09 溢出）、\`log(softmax)\` 取到 $\log 0$、方差用 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ 后开根号、全屏蔽行的 softmax（$-\infty - (-\infty)$）、Adam 的 $v$ 少了平方变负数、偏差修正从 $t = 0$ 开始除以 0、梯度尖刺没有裁剪、attention logit 爆炸。

        下面这个可视化在浏览器里真实地训练一个 2→16→2 的 MLP，用的就是你在本章写过的全部公式。试试：关掉 warmup 与裁剪并调大学习率；把 AdamW 换成 SGD 并保持同样的学习率。
      `,
      viz: TrainLoopViz,
    },
  ],

  /* ================================================================ */
  /* 综合练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'b-nan',
      type: 'multi',
      difficulty: 2,
      prompt: md`fp16 / bf16 混合精度训练中出现了 NaN。下列哪些**都可能**是原因？（多选）`,
      options: [
        'softmax 没有减最大值，logit 超过 11.09 时在 fp16 中 exp 溢出',
        'LayerNorm 用 $\\mathbb{E}[x^2]-\\mathbb{E}[x]^2$ 求方差，得到负数后开根号',
        '对整行都被 mask 的注意力做 softmax',
        '用 `log_softmax` 代替了 `log(softmax(x))`',
        'Adam 的偏差修正从 $t = 0$ 开始计数',
      ],
      answer: [0, 1, 2, 4],
      explanation: md`A、B、C、E 都会产生 inf/NaN：E 中 $1 - \beta^0 = 0$，除以 0。D 恰恰是**修复**手段：\`log_softmax\` 在对数域计算，不会出现 $\log 0$。`,
    },
    {
      id: 'b-ce-grad',
      type: 'single',
      difficulty: 1,
      prompt: md`交叉熵对 logits 的梯度 $p - y$ 中，$p$ 是什么？`,
      options: ['logits 本身', '$\\mathrm{softmax}(\\text{logits})$', '$\\log\\mathrm{softmax}(\\text{logits})$', '标签的 one-hot'],
      answer: 1,
      explanation: md`$\partial(-z_y + \mathrm{LSE}(z))/\partial z = \mathrm{softmax}(z) - \mathrm{onehot}(y)$。C 是前向计算 loss 用的量；D 是 $y$。`,
    },
    {
      id: 'b-params',
      type: 'numeric',
      difficulty: 3,
      prompt: md`LLaMA-7B：32 层，$d = 4096$，标准 MHA（Q、K、V、O 都是 $d\times d$），SwiGLU 隐藏维 11008，无 bias。只算各层的注意力与 FFN 矩阵（不含 embedding、LM head、norm），参数共多少？（可输入表达式，如 \`32*(4*4096**2 + ...)\`）`,
      answer: 6476005376,
      tolerance: 0.002,
      explanation: md`每层注意力 $4d^2 = 67{,}108{,}864$，FFN $3 \cdot 4096 \cdot 11008 = 135{,}266{,}304$，合计 $202{,}375{,}168$；乘 32 层 = $6{,}476{,}005{,}376 \approx 6.48$B。再加上 embedding 与 LM head（各 $32000 \times 4096 \approx 131$M）约 6.74B，就是“7B”的由来。`,
    },
    {
      id: 'b-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把一个 SFT 训练 step 的操作按正确顺序排列：`,
      items: [
        '构造 labels：prompt 与 padding 位置设为 −100',
        '前向：Pre-Norm decoder 层 → final norm → LM head 得到 logits',
        '错位对齐后，用稳定 log_softmax 计算有效 token 上的平均交叉熵',
        '反向传播得到所有参数的梯度',
        '全局范数梯度裁剪',
        '优化器更新（AdamW / Muon），然后更新学习率',
      ],
      explanation: md`labels 在数据侧就准备好；裁剪必须在梯度算出之后、更新之前；学习率调度在更新之后推进到下一步。`,
    },
    {
      id: 'b-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把技术与它要解决的问题配对：`,
      pairs: [
        ['减最大值 / log-sum-exp', 'softmax 的上溢与 0/0'],
        ['Pre-LN', '深层网络的梯度传播与训练稳定性'],
        ['AdamW 解耦权重衰减', '衰减强度被 √v̂ 扭曲'],
        ['全局范数梯度裁剪', '偶发的梯度尖刺'],
        ['QK-Norm / QK-Clip', 'attention logit 爆炸'],
        ['SwiGLU 隐藏维取 8/3·d', '三个矩阵时保持与 4d FFN 相同的参数量'],
      ],
      explanation: md`每一项都对应本章的一关。面试中被问到“为什么要 X”时，先说它解决了什么问题，再说代价。`,
    },
    {
      id: 'b-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 PyTorch 训练循环（\`y\` 已经错位对齐好）有 **2 处** bug，点出来：`,
      code: py`
        for step, (x, y) in enumerate(loader):
            logits = model(x)                                   # (B, T, V)
            probs = F.softmax(logits, dim=-1)
            loss = F.cross_entropy(probs.view(-1, V), y.view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()
            optimizer.zero_grad()
      `,
      bugLines: [4, 7],
      explanation: md`
        - 第 4 行把**概率**传给了 \`F.cross_entropy\`，它内部还会再做一次 \`log_softmax\`，等于对概率又做了一次 softmax，loss 和梯度都错了，且梯度被严重压扁。应直接传 \`logits\`（第 3 行可以删掉）。
        - 第 7 行裁剪放在了 \`optimizer.step()\` **之后**，对本步更新毫无作用（下一步开始前又被 \`zero_grad\` 清掉）。应移到 \`loss.backward()\` 与 \`optimizer.step()\` 之间。
      `,
    },
    {
      id: 'b-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`PyTorch 的 AdamW 在 Adam 更新之前对参数做 \`p ← p · (1 − ____(1)____)\`（用 \`lr\`、\`wd\` 表示）；Muon 把正交化后的更新乘以 \`0.2 · sqrt(____(2)____)\` 对齐 RMS（用 \`n\`、\`m\` 表示矩阵形状）。`,
      blanks: [
        { label: '(1)', answers: ['lr*wd', 'wd*lr', 'lr * wd', 'wd * lr'] },
        { label: '(2)', answers: ['max(n,m)', 'max(m,n)', 'max(n, m)', 'max(m, n)'] },
      ],
      explanation: md`AdamW 的衰减量乘了学习率；Muon 的 $0.2\sqrt{\max(n,m)}$ 让满秩正交更新的 RMS 恰为 0.2（Moonlight），从而复用 AdamW 的学习率与衰减。`,
    },
    {
      id: 'b-judge',
      type: 'judge',
      difficulty: 1,
      prompt: md`在参数量相同的前提下，SwiGLU FFN 的隐藏维比 ReLU FFN 的隐藏维小。`,
      answer: true,
      explanation: md`ReLU FFN 两个矩阵、隐藏维 $4d$；SwiGLU 三个矩阵，为了参数相同隐藏维取 $\frac{8}{3}d$。`,
    },
    {
      id: 'b-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`下列算子中，哪些的反向需要用到**沿某一维的统计量**（而不只是逐元素的输入或输出值）？`,
      items: ['softmax', 'ReLU', 'LayerNorm', 'SiLU', 'RMSNorm', '逐元素加法'],
      answer: [0, 2, 4],
      explanation: md`softmax 的 VJP 需要 $\langle \mathrm{d}p, p\rangle$（沿词表维的内积）；LayerNorm 需要 $\mu, 1/\sigma$ 以及 $\overline{g}$、$\overline{g\hat x}$；RMSNorm 需要 $1/\mathrm{RMS}$ 与 $\overline{g\hat x}$。ReLU、SiLU、加法的反向都是逐元素的。`,
    },
    {
      id: 'b-parsons',
      type: 'parsons',
      difficulty: 3,
      prompt: md`按词表分块计算每行的 log-sum-exp（online softmax 的思路，不物化完整 logits）。拼出正确代码，有 2 行干扰项。`,
      lines: [
        'def chunked_lse(h, W, chunk):',
        '    m = np.full(h.shape[0], -np.inf)',
        '    s = np.zeros(h.shape[0])',
        '    for a in range(0, W.shape[1], chunk):',
        '        z = h @ W[:, a:a + chunk]',
        '        m_new = np.maximum(m, z.max(axis=1))',
        '        s = s * np.exp(m - m_new) + np.exp(z - m_new[:, None]).sum(axis=1)',
        '        m = m_new',
        '    return m + np.log(s)',
      ],
      distractors: ['        s = s + np.exp(z - m_new[:, None]).sum(axis=1)', '        m = z.max(axis=1)'],
      explanation: md`
        维护运行最大值 $m$ 与以 $m$ 为基准的和 $s$。最大值变大时，旧的和要乘 $e^{m_{\text{old}} - m_{\text{new}}}$ 重新换算到新基准——干扰项 1 漏了这一步；干扰项 2 只记住了当前块的最大值。这正是 FlashAttention 与 Liger / Cut Cross-Entropy 等“分块交叉熵”的核心。
      `,
    },
    {
      id: 'b-clip',
      type: 'numeric',
      difficulty: 2,
      prompt: md`三个参数张量的梯度范数分别为 2、3、6，\`max_norm = 1\`。全局范数裁剪的系数是多少？（保留 4 位小数，忽略 1e-6）`,
      answer: 0.1429,
      tolerance: 0.002,
      explanation: md`全局范数 $\sqrt{4 + 9 + 36} = 7$，系数 $1/7 \approx 0.1429$，三个张量都乘这个系数，裁剪后范数分别为 0.286、0.429、0.857，平方和开根号为 1。`,
    },
    {
      id: 'b-muon-mem',
      type: 'numeric',
      difficulty: 3,
      prompt: md`
        LLaMA-7B：隐藏层矩阵约 6.476B 参数，embedding + LM head 约 0.262B（norm 可忽略）。
        原本全部用 AdamW（每参数 2 个状态）；现在隐藏层矩阵改用 Muon（每参数 1 个状态），其余仍用 AdamW。
        优化器状态的元素总数变为原来的多少倍？（保留两位小数）
      `,
      answer: 0.52,
      tolerance: 0.03,
      explanation: md`原来 $2 \times 6.738 = 13.476$B 个状态；现在 $6.476 \times 1 + 0.262 \times 2 = 7.0$B 个，比值约 $0.52$——接近一半，因为绝大部分参数都在隐藏层矩阵里。`,
    },
  ],

  guided: [],

  /* ================================================================ */
  /* 综合实战                                                           */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-train-mlp',
      title: '从零训练一个 MLP：CE + 反向 + 裁剪 + AdamW + warmup/cosine',
      difficulty: 'hard',
      source: '综合手撕 · 训练循环',
      timeLimitMin: 35,
      description: md`
        只用 NumPy 写一个完整的训练循环：

        \`\`\`python
        def train(params, X, y, steps=200, max_lr=1e-2, warmup=20, min_lr_ratio=0.1,
                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, max_norm=1.0) -> tuple[list[float], dict]: ...
        \`\`\`

        - \`params\`：\`W1 (D, H)\`、\`b1 (H,)\`、\`W2 (H, C)\`、\`b2 (C,)\`；**不要修改**传入的数组。\`X (N, D)\`，\`y (N,)\` 为整数类别。
        - 返回 \`(losses, new_params)\`：\`losses[t]\` 是第 $t$ 步**更新前**的 loss（长度为 \`steps\`）。

        第 $t$ 步（$t$ 从 0 开始），**严格按以下顺序**：
        1. 前向 + 反向：$h = \mathrm{ReLU}(XW_1 + b_1)$，$s = hW_2 + b_2$，$L = $ 平均交叉熵（数值稳定）；ReLU 在 0 处梯度为 0。
        2. 全局 L2 范数裁剪：四个梯度一起求范数 \`total\`，系数 \`min(1, max_norm / (total + 1e-6))\`。
        3. 学习率：$t <$ \`warmup\` 时 $\eta_{\max}\cdot(t+1)/\text{warmup}$；否则从 $\eta_{\max}$ 按 cosine 衰减到 $\eta_{\max}\cdot$\`min_lr_ratio\`，进度 $p = (t - \text{warmup})/(\text{steps} - \text{warmup})$。
        4. AdamW（PyTorch 语义，Adam 的步数为 $t + 1$）：先 $\theta \leftarrow \theta(1 - \eta\lambda)$（**只对 W1、W2**），再做带偏差修正的 Adam 更新（四个参数都更新）。
      `,
      starter: py`
        import math
        import numpy as np

        def train(params, X, y, steps=200, max_lr=1e-2, warmup=20, min_lr_ratio=0.1,
                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, max_norm=1.0):
            P = {k: v.copy() for k, v in params.items()}
            losses = []
            # TODO: 初始化 m、v；循环 steps 次：前向反向 → 裁剪 → 学习率 → AdamW
            return losses, P
      `,
      solution: py`
        import math
        import numpy as np

        def _loss_and_grads(P, X, y):
            N = X.shape[0]
            z1 = X @ P["W1"] + P["b1"]
            h = np.maximum(z1, 0.0)
            s = h @ P["W2"] + P["b2"]
            m = s.max(axis=1, keepdims=True)
            logp = s - m - np.log(np.exp(s - m).sum(axis=1, keepdims=True))
            loss = float(-logp[np.arange(N), y].mean())
            ds = np.exp(logp)
            ds[np.arange(N), y] -= 1.0
            ds /= N
            dh = ds @ P["W2"].T
            dz1 = dh * (z1 > 0)
            return loss, {"W1": X.T @ dz1, "b1": dz1.sum(0), "W2": h.T @ ds, "b2": ds.sum(0)}

        def _lr_at(t, max_lr, warmup, steps, min_lr):
            if t < warmup:
                return max_lr * (t + 1) / warmup
            p = (t - warmup) / (steps - warmup)
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))

        def train(params, X, y, steps=200, max_lr=1e-2, warmup=20, min_lr_ratio=0.1,
                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, max_norm=1.0):
            P = {k: v.copy() for k, v in params.items()}               # 不修改输入
            M = {k: np.zeros_like(v) for k, v in P.items()}
            S = {k: np.zeros_like(v) for k, v in P.items()}
            b1, b2 = betas
            losses = []
            for t in range(steps):
                loss, G = _loss_and_grads(P, X, y)                     # 1) 前向 + 反向
                losses.append(loss)
                total = math.sqrt(sum(float((g * g).sum()) for g in G.values()))
                coef = min(1.0, max_norm / (total + 1e-6))             # 2) 全局范数裁剪
                lr = _lr_at(t, max_lr, warmup, steps, max_lr * min_lr_ratio)   # 3) 调度
                for k in P:                                            # 4) AdamW
                    g = G[k] * coef
                    if k in ("W1", "W2"):
                        P[k] = P[k] * (1 - lr * weight_decay)
                    M[k] = b1 * M[k] + (1 - b1) * g
                    S[k] = b2 * S[k] + (1 - b2) * g * g
                    m_hat = M[k] / (1 - b1 ** (t + 1))
                    v_hat = S[k] / (1 - b2 ** (t + 1))
                    P[k] = P[k] - lr * m_hat / (np.sqrt(v_hat) + eps)
            return losses, P
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import math
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            def train(model: nn.Module, X, y, steps=200, max_lr=1e-2, warmup=20, min_lr_ratio=0.1, weight_decay=0.01):
                decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
                no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
                opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                         {"params": no_decay, "weight_decay": 0.0}], lr=max_lr, betas=(0.9, 0.95))
                def lr_lambda(t):
                    if t < warmup:
                        return (t + 1) / warmup
                    p = (t - warmup) / (steps - warmup)
                    return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * p))
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
                losses = []
                for t in range(steps):
                    loss = F.cross_entropy(model(X), y)
                    losses.append(loss.item())
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                return losses
          `,
        },
      ],
      explanation: md`
        这道题把本章五关串在了一起：稳定 softmax-CE（Softmax 关）→ MLP 反向（反向传播关）→ 全局范数裁剪、warmup + cosine、AdamW（优化器关）。

        **最容易错的地方**
        - 偏差修正的步数要用 $t + 1$；
        - 衰减只作用于矩阵，且在 Adam 更新**之前**乘 $(1 - \eta\lambda)$（与 PyTorch 一致）；
        - 裁剪系数作用在梯度上，必须在更新 $m, v$ 之前；
        - 每一步都要用**当前**参数重新前向，\`losses[t]\` 记录的是更新前的 loss；
        - 不要修改传入的 \`params\`（先 copy）。

        **复杂度**：每步 $O(N(DH + HC))$。
      `,
      complexity: { time: 'O(steps · N · (D·H + H·C))', space: 'O(D·H + H·C + N·H)' },
      hints: [
        md`先写 \`_loss_and_grads(P, X, y)\`：就是反向传播关的 \`mlp_loss_and_grads\`。`,
        md`裁剪：\`total = sqrt(Σ (g*g).sum())\`，\`coef = min(1, max_norm / (total + 1e-6))\`，然后 \`g = G[k] * coef\`。`,
        md`学习率函数就是优化器关跟练里的 \`lr_at\`（min_lr = max_lr · min_lr_ratio）。`,
        md`AdamW：矩阵先 \`P[k] *= 1 - lr * wd\`，再 \`M = b1*M + (1-b1)*g\`、\`S = b2*S + (1-b2)*g*g\`，偏差修正用 \`t + 1\`。`,
      ],
      setup: py`
        def _ref_lg(P, X, y):
            N = X.shape[0]
            z1 = X @ P["W1"] + P["b1"]; h = np.maximum(z1, 0)
            s = h @ P["W2"] + P["b2"]; m = s.max(1, keepdims=True)
            logp = s - m - np.log(np.exp(s - m).sum(1, keepdims=True))
            ds = np.exp(logp); ds[np.arange(N), y] -= 1; ds /= N
            dz1 = (ds @ P["W2"].T) * (z1 > 0)
            return float(-logp[np.arange(N), y].mean()), {"W1": X.T @ dz1, "b1": dz1.sum(0), "W2": h.T @ ds, "b2": ds.sum(0)}

        def _ref_train(params, X, y, steps=200, max_lr=1e-2, warmup=20, min_lr_ratio=0.1,
                       betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, max_norm=1.0):
            P = {k: v.copy() for k, v in params.items()}
            M = {k: np.zeros_like(v) for k, v in P.items()}; S = {k: np.zeros_like(v) for k, v in P.items()}
            out = []
            for t in range(steps):
                loss, G = _ref_lg(P, X, y); out.append(loss)
                total = math.sqrt(sum(float((g * g).sum()) for g in G.values()))
                c = min(1.0, max_norm / (total + 1e-6))
                if t < warmup:
                    lr = max_lr * (t + 1) / warmup
                else:
                    p = (t - warmup) / (steps - warmup)
                    lr = max_lr * min_lr_ratio + 0.5 * (max_lr - max_lr * min_lr_ratio) * (1 + math.cos(math.pi * p))
                for k in P:
                    g = G[k] * c
                    if k in ("W1", "W2"):
                        P[k] = P[k] * (1 - lr * weight_decay)
                    M[k] = betas[0] * M[k] + (1 - betas[0]) * g
                    S[k] = betas[1] * S[k] + (1 - betas[1]) * g * g
                    P[k] = P[k] - lr * (M[k] / (1 - betas[0] ** (t + 1))) / (np.sqrt(S[k] / (1 - betas[1] ** (t + 1))) + eps)
            return out, P

        def _data(seed=0, N=120, H=16, scale=1.0):
            rng = np.random.default_rng(seed)
            centers = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=float)
            lab = np.array([0, 0, 1, 1])
            idx = np.arange(N) % 4
            X = centers[idx] + 0.3 * rng.normal(size=(N, 2))
            y = lab[idx]
            params = {"W1": rng.normal(size=(2, H)) * scale, "b1": np.zeros(H),
                      "W2": rng.normal(size=(H, 2)) * np.sqrt(2 / H) * scale, "b2": np.zeros(2)}
            return params, X, y
      `,
      tests: [
        {
          name: '第 0 步 loss（前向正确）',
          code: py`
            params, X, y = _data(0)
            losses, _ = train(params, X, y, steps=1)
            assert isinstance(losses, list) and len(losses) == 1, f"losses 应为长度 1 的 list，得到 {type(losses)} / {len(losses) if hasattr(losses, '__len__') else '?'}"
            exp = _ref_lg(params, X, y)[0]
            assert abs(losses[0] - exp) < 1e-9, f"第 0 步 loss 期望 {exp:.6f}，得到 {losses[0]}"
          `,
        },
        {
          name: '一步更新后参数一致',
          code: py`
            params, X, y = _data(1)
            _, P = train(params, X, y, steps=1, warmup=1)
            _, R = _ref_train(params, X, y, steps=1, warmup=1)
            for k in R:
                err = np.abs(P[k] - R[k]).max()
                assert err < 1e-10, f"{k} 一步后最大误差 {err:.2e}（检查：梯度 / 裁剪 / 学习率 / 偏差修正 / 衰减只作用于矩阵）"
          `,
        },
        {
          name: '20 步完整对拍',
          code: py`
            params, X, y = _data(2)
            L, P = train(params, X, y, steps=20, max_lr=0.05, warmup=5)
            RL, R = _ref_train(params, X, y, steps=20, max_lr=0.05, warmup=5)
            assert len(L) == 20, f"losses 长度应为 20，得到 {len(L)}"
            assert np.allclose(L, RL, atol=1e-9), f"loss 序列不一致：第一个差异在第 {int(np.argmax(np.abs(np.array(L) - np.array(RL)) > 1e-9))} 步"
            for k in R:
                assert np.allclose(P[k], R[k], atol=1e-9), f"20 步后 {k} 不一致"
          `,
        },
        {
          name: 'loss 下降且学会 XOR',
          hidden: true,
          code: py`
            params, X, y = _data(3)
            L, P = train(params, X, y, steps=300, max_lr=0.05, warmup=20)
            assert L[-1] < 0.25 * L[0], f"loss 应显著下降：初始 {L[0]:.3f}，最终 {L[-1]:.3f}"
            h = np.maximum(X @ P["W1"] + P["b1"], 0)
            acc = ((h @ P["W2"] + P["b2"]).argmax(1) == y).mean()
            assert acc > 0.95, f"训练准确率应 > 95%，得到 {acc:.2%}"
          `,
        },
        {
          name: '梯度尖刺时裁剪生效',
          hidden: true,
          code: py`
            params, X, y = _data(4, scale=30.0)
            L, P = train(params, X, y, steps=5, max_lr=0.1, warmup=2, max_norm=0.5)
            RL, R = _ref_train(params, X, y, steps=5, max_lr=0.1, warmup=2, max_norm=0.5)
            assert np.allclose(L, RL, atol=1e-8), "大梯度下 loss 序列与参考不一致（裁剪是否在更新 m、v 之前？）"
            for k in R:
                assert np.allclose(P[k], R[k], atol=1e-8), f"{k} 不一致"
          `,
        },
        {
          name: '不修改输入 params',
          hidden: true,
          code: py`
            params, X, y = _data(5)
            snap = {k: v.copy() for k, v in params.items()}
            train(params, X, y, steps=3)
            for k in params:
                assert np.array_equal(params[k], snap[k]), f"params['{k}'] 被原地修改了"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-bf16',
          q: md`如果把这个训练循环搬到 bf16 混合精度，哪些量必须保持 fp32？`,
          a: md`
            主权重（master weights）与 Adam 的 $m, v$ 保持 fp32（bf16 只有 8 位尾数，小的更新量 $\eta\hat m/\sqrt{\hat v}$ 加到大权重上会被舍入掉）；softmax / log_softmax、LayerNorm / RMSNorm 的统计量、loss 的归约都在 fp32 中计算；梯度的 all-reduce 通常也用 fp32（或 bf16 + 随机舍入，如 DeepSeek-V4 对 MoE 梯度的处理）。前向反向的矩阵乘可以用 bf16。
          `,
        },
      ],
    },
    {
      id: 'c-chunked-ce',
      title: '分块 Linear + 交叉熵：不物化 (N, V) logits',
      difficulty: 'hard',
      source: '综合手撕 · 大词表显存优化',
      timeLimitMin: 30,
      description: md`
        词表很大时，\`logits = h @ W\` 本身就是显存瓶颈（$N = 32\text{K}$、$V = 152\text{K}$ 的 fp32 logits 约 20 GB）。实现**按词表分块**的前向与反向：

        \`\`\`python
        def chunked_linear_ce(h: np.ndarray, W, targets: np.ndarray, chunk_size: int
                              ) -> tuple[float, np.ndarray, np.ndarray]: ...
        \`\`\`

        - \`h\`：\`(N, d)\`；\`targets\`：\`(N,)\`，取值 $[0, V)$（本题没有 ignore）。
        - \`W\`：形状 \`(d, V)\` 的权重，**只能**通过 \`W[:, a:b]\` 按列切片读取，且每次 \`b - a <= chunk_size\`（测试会传入一个只支持这种访问的对象；\`W.shape\` 可用）。
        - 返回 \`(loss, dh, dW)\`：\`loss\` 为平均交叉熵（float），\`dh\` 形状 \`(N, d)\`，\`dW\` 形状 \`(d, V)\`（梯度本身可以是完整数组）。
        - 数值稳定：logits 绝对值可能很大。

        **提示思路**：第一遍分块扫描，用 online softmax 求每行的 $\mathrm{LSE}$ 与目标 logit；第二遍分块重算 logits，得到 $g = (\mathrm{softmax} - \mathrm{onehot})/N$，累加 $\mathrm{d}h \mathrel{+}= gW_c^\top$，写入 $\mathrm{d}W_c = h^\top g$。
      `,
      starter: py`
        import numpy as np

        def chunked_linear_ce(h, W, targets, chunk_size):
            N, d = h.shape
            V = W.shape[1]
            # TODO: 第一遍：online softmax 求 lse 与目标 logit
            #       第二遍：重算每块 logits，求 dh 与 dW
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def chunked_linear_ce(h, W, targets, chunk_size):
            N, d = h.shape
            V = W.shape[1]
            rows = np.arange(N)
            # ---- 第一遍：online softmax，得到每行的 LSE 与目标 logit ----
            m = np.full(N, -np.inf)
            s = np.zeros(N)
            tgt = np.zeros(N)
            for a in range(0, V, chunk_size):
                b = min(a + chunk_size, V)
                z = h @ W[:, a:b]                                   # (N, b-a)
                m_new = np.maximum(m, z.max(axis=1))
                s = s * np.exp(m - m_new) + np.exp(z - m_new[:, None]).sum(axis=1)
                m = m_new
                inside = (targets >= a) & (targets < b)
                tgt[inside] = z[rows[inside], targets[inside] - a]
            lse = m + np.log(s)
            loss = float(np.mean(lse - tgt))
            # ---- 第二遍：重算每块 logits，g = (softmax - onehot) / N ----
            dh = np.zeros((N, d))
            dW = np.zeros((d, V))
            for a in range(0, V, chunk_size):
                b = min(a + chunk_size, V)
                Wc = W[:, a:b]
                g = np.exp(h @ Wc - lse[:, None])
                inside = (targets >= a) & (targets < b)
                g[rows[inside], targets[inside] - a] -= 1.0
                g /= N
                dh += g @ Wc.T
                dW[:, a:b] = h.T @ g
            return loss, dh, dW
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：按序列分块 + 重计算，思路同 Liger 的 fused linear CE）',
          code: py`
            import torch
            import torch.nn.functional as F

            def chunked_linear_ce(h: torch.Tensor, W: torch.Tensor, targets: torch.Tensor, chunk: int = 4096):
                """按 token 分块：每块只物化 (chunk, V) 的 logits，立刻算出 dh 与 dW 的贡献后释放"""
                N = h.shape[0]
                loss = h.new_zeros(())
                dh = torch.empty_like(h)
                dW = torch.zeros_like(W)
                for a in range(0, N, chunk):
                    hc, tc = h[a:a + chunk], targets[a:a + chunk]
                    logits = (hc @ W).float()
                    loss += F.cross_entropy(logits, tc, reduction="sum")
                    g = torch.softmax(logits, -1)
                    g[torch.arange(len(tc)), tc] -= 1
                    g = (g / N).to(h.dtype)
                    dh[a:a + chunk] = g @ W.T
                    dW += hc.T @ g
                return loss / N, dh, dW
          `,
        },
      ],
      explanation: md`
        **两遍扫描**：交叉熵需要每行的归一化常数 $\mathrm{LSE}$，而它依赖整行 logits。第一遍用 online softmax（运行最大值 $m$ + 以 $m$ 为基准的和 $s$，最大值更新时乘 $e^{m_{\text{old}} - m_{\text{new}}}$ 修正）分块求出 LSE 与目标 logit；第二遍**重算**每块 logits（用计算换显存），此时 $p = e^{z - \mathrm{LSE}}$ 已知，直接得到梯度块。

        **显存**：峰值只有 $(N, \text{chunk})$ 的 logits，而不是 $(N, V)$；代价是 $hW$ 算了两遍。工业实现（Liger Kernel 的 fused linear cross entropy、Apple 的 Cut Cross-Entropy）把这件事写成融合 kernel，并按 token 或词表分块。

        **复杂度**：时间 $O(2NdV)$，额外显存 $O(N\cdot\text{chunk})$。
      `,
      complexity: { time: 'O(N·d·V)（两遍）', space: 'O(N·chunk) 额外' },
      hints: [
        md`第一遍维护 \`m\`（初值 $-\infty$）与 \`s\`（初值 0）：\`m_new = max(m, z.max(1))\`，\`s = s*exp(m - m_new) + exp(z - m_new).sum(1)\`。`,
        md`目标 logit：\`inside = (targets >= a) & (targets < b)\`，\`tgt[inside] = z[rows[inside], targets[inside] - a]\`。loss = \`mean(lse - tgt)\`。`,
        md`第二遍：\`g = exp(h @ Wc - lse[:, None])\`，目标位置减 1，再除以 N；\`dh += g @ Wc.T\`，\`dW[:, a:b] = h.T @ g\`。`,
      ],
      setup: py`
        class _ChunkedW:
            """只允许 W[:, a:b] 且 b - a <= chunk 的只读权重"""
            def __init__(self, W, chunk):
                self._W = np.asarray(W, dtype=np.float64)
                self.shape = self._W.shape
                self._chunk = chunk
                self.reads = 0
            def __getitem__(self, key):
                assert isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None) and isinstance(key[1], slice), "只能用 W[:, a:b] 的方式读取"
                a, b, step = key[1].indices(self.shape[1])
                assert step == 1, "不支持带步长的切片"
                assert b - a <= self._chunk, f"一次读取了 {b - a} 列，超过 chunk_size={self._chunk}（不能物化完整 logits）"
                self.reads += 1
                return self._W[:, a:b].copy()

        def _ref_full(h, W, t):
            z = h @ W
            m = z.max(1, keepdims=True)
            lse = (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]
            N = len(t)
            loss = float(np.mean(lse - z[np.arange(N), t]))
            g = np.exp(z - lse[:, None]); g[np.arange(N), t] -= 1; g /= N
            return loss, g @ W.T, h.T @ g
      `,
      tests: [
        {
          name: '小例子对拍',
          code: py`
            rng = np.random.default_rng(0)
            h, W, t = rng.normal(size=(4, 3)), rng.normal(size=(3, 10)), np.array([0, 9, 4, 5])
            loss, dh, dW = chunked_linear_ce(h, _ChunkedW(W, 3), t, 3)
            el, edh, edW = _ref_full(h, W, t)
            assert abs(float(loss) - el) < 1e-10, f"loss 期望 {el:.6f}，得到 {loss}"
            assert dh.shape == (4, 3) and dW.shape == (3, 10), f"形状错误：dh {dh.shape}，dW {dW.shape}"
            assert np.allclose(dh, edh) and np.allclose(dW, edW), f"梯度不一致：dh 误差 {np.abs(dh - edh).max():.2e}，dW 误差 {np.abs(dW - edW).max():.2e}"
          `,
        },
        {
          name: '块大小不整除 / 大于词表',
          code: py`
            rng = np.random.default_rng(1)
            h, W = rng.normal(size=(5, 4)), rng.normal(size=(4, 11))
            t = rng.integers(0, 11, size=5)
            el, edh, edW = _ref_full(h, W, t)
            for c in (4, 7, 11, 50):
                loss, dh, dW = chunked_linear_ce(h, _ChunkedW(W, c), t, c)
                assert abs(float(loss) - el) < 1e-10, f"chunk={c}：loss 期望 {el:.6f}，得到 {loss}"
                assert np.allclose(dh, edh) and np.allclose(dW, edW), f"chunk={c}：梯度不一致"
          `,
        },
        {
          name: '大 logits 数值稳定',
          code: py`
            rng = np.random.default_rng(2)
            h, W = rng.normal(size=(3, 4)) * 40, rng.normal(size=(4, 9)) * 40
            t = np.array([1, 8, 3])
            loss, dh, dW = chunked_linear_ce(h, _ChunkedW(W, 2), t, 2)
            el, edh, edW = _ref_full(h, W, t)
            assert np.isfinite(loss) and np.all(np.isfinite(dh)), f"出现 NaN/inf：loss={loss}"
            assert abs(float(loss) - el) < 1e-6 * max(1, abs(el)), f"loss 期望 {el}，得到 {loss}"
            assert np.allclose(dW, edW, atol=1e-8), "大 logits 下 dW 不一致"
          `,
        },
        {
          name: '随机对拍（较大词表）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(3)
            N, d, V, c = 8, 6, 257, 32
            h, W = rng.normal(size=(N, d)), rng.normal(size=(d, V))
            t = rng.integers(0, V, size=N)
            cw = _ChunkedW(W, c)
            loss, dh, dW = chunked_linear_ce(h, cw, t, c)
            el, edh, edW = _ref_full(h, W, t)
            assert abs(float(loss) - el) < 1e-10 and np.allclose(dh, edh) and np.allclose(dW, edW), "与完整计算不一致"
            assert cw.reads <= 2 * int(np.ceil(V / c)) + 2, f"读取了 {cw.reads} 次权重块，预期两遍扫描约 {2 * int(np.ceil(V / c))} 次"
          `,
        },
        {
          name: '目标恰在块边界',
          hidden: true,
          code: py`
            rng = np.random.default_rng(4)
            h, W = rng.normal(size=(6, 3)), rng.normal(size=(3, 12))
            t = np.array([0, 3, 4, 7, 8, 11])                      # chunk=4 时的块首、块尾
            loss, dh, dW = chunked_linear_ce(h, _ChunkedW(W, 4), t, 4)
            el, edh, edW = _ref_full(h, W, t)
            assert abs(float(loss) - el) < 1e-10, f"块边界上的目标处理错误：期望 {el:.6f}，得到 {loss}"
            assert np.allclose(dh, edh) and np.allclose(dW, edW), "块边界上的梯度处理错误"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-why-two-pass',
          q: md`能不能只扫一遍就同时得到 loss 和梯度？`,
          a: md`
            梯度 $g = \mathrm{softmax} - \mathrm{onehot}$ 需要最终的 LSE，而 LSE 要看完所有块才知道。$\mathrm{d}h$ 可以像 FlashAttention 累加输出那样“边扫边修正”：每行用运行最大值算未归一化的贡献，最大值变化时把该行已累加的部分乘 $e^{m_{\text{old}} - m_{\text{new}}}$，最后除以 $s$。但 $\mathrm{d}W_c = h^\top g_c$ 是对**所有行**求和的结果，每行的修正系数各不相同，事后无法用一个系数统一修正，只能保留未归一化的概率块——相当于又把 logits 物化了。所以常见做法要么是两遍扫描（本题），要么改为**按 token 分块**：每块拿到完整的行，一次就能得到精确的 LSE 与梯度（Liger 的 fused linear cross entropy 即按 token 分块）。
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
      id: 'qa-spike',
      freq: 3,
      q: md`大模型预训练中 loss 突然出现尖刺（spike）或 NaN，你会怎么排查和处理？`,
      a: md`
        **先止血**：回滚到尖刺前的 checkpoint，跳过对应的数据 batch（PaLM 的做法）；必要时降低学习率。

        **排查**（从本章的零件出发）：(1) 看 \`grad_norm\` 曲线，尖刺通常先于 loss spike 出现；检查梯度裁剪是否生效、是否在 unscale 之后；(2) 看各层 attention 最大 logit——logit 爆炸是常见根因，对策是 QK-Norm（Qwen3、Gemma 3、OLMo 2）或 QK-Clip（Kimi K2）；(3) 检查 softmax / CE / norm 是否在 fp32 中计算，方差是否稳定；(4) 检查 Adam 的 $\varepsilon$ 与 $\beta_2$（$\beta_2 = 0.95$ 比 0.999 更能跟上梯度尺度变化）；(5) 检查数据：极端长度、重复 token、错误的 label mask。

        **预防**：z-loss（PaLM）、合理的 warmup、bf16 而非 fp16、对异常 batch 自动跳过；K2 报告用 MuonClip 实现了全程无 loss spike。
      `,
    },
    {
      id: 'qa-step',
      freq: 2,
      q: md`如果让你不依赖框架、从零写一个训练 step，有哪些数值细节必须注意？`,
      a: md`
        **前向**：softmax / log_softmax 减最大值；交叉熵直接从 logits 计算并对有效 token 求平均；LayerNorm 用两遍法求方差、$\varepsilon$ 在根号内、统计量用 fp32。

        **反向**：softmax-CE 用 $p - y$；Linear 用形状对齐；偏置梯度对 batch 求和；LayerNorm 反向不能漏 $\mu, \sigma$ 的依赖；写完用 float64 数值梯度检查。

        **更新**：全局范数裁剪在 step 之前；AdamW 偏差修正从 $t = 1$ 开始、衰减解耦且只对矩阵；学习率 warmup + 衰减；混合精度下主权重与优化器状态保持 fp32。

        **大词表**：logits 显存巨大，用分块（按 token 或词表）的融合 linear + CE，前向求 LSE、反向重算。
      `,
    },
  ],

  cheatsheet: md`
    **训练 step 顺序**：前向（Norm → Attn/FFN → final norm → logits）→ 稳定 CE → 反向 → 全局裁剪 → AdamW / Muon → 调度

    \`\`\`python
    # online LSE（分块交叉熵 / FlashAttention 的核心）
    m_new = np.maximum(m, z.max(1))
    s = s * np.exp(m - m_new) + np.exp(z - m_new[:, None]).sum(1)
    m = m_new                      # 最后 lse = m + log(s)
    \`\`\`

    | 本章关键数字 | |
    |---|---|
    | exp 溢出 | fp16 11.09，fp32/bf16 88.72 |
    | SwiGLU | $h = \frac83 d$ → LLaMA-7B 11008 |
    | AdamW | β=(0.9, 0.95)，wd 0.1，clip 1.0，16 B/参数 |
    | Muon | NS (3.4445, −4.7750, 2.0315)，RMS 0.2·√max(A,B)；V4 8+2 步、0.18 |
    | QK-Clip | τ = 100，q^C/k^C ×√γ，q^R ×γ |

    - NaN 清单：exp 溢出、log 0、负方差、全屏蔽行、v 少平方、t 从 0 开始、未裁剪的尖刺、logit 爆炸
  `,
})
