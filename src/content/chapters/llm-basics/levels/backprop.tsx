/**
 * llm-basics-backprop · 手写反向传播
 * 计算图与链式法则、Linear / ReLU / softmax / softmax-CE 的反向、两层 MLP、数值梯度检查、激活显存。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { CompGraphViz } from '../viz/CompGraphViz'
import { LinearGradViz } from '../viz/LinearGradViz'
import { GradCheckViz } from '../viz/GradCheckViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'graph',
      title: '计算图与链式法则：反向传播就是逆拓扑序的局部求导',
      body: md`
        任何网络都可以写成一个有向无环图（DAG）：节点是运算，边是张量。训练需要标量 loss $L$ 对每个参数的梯度。

        **反向传播 = 反向模式自动微分**：
        1. **前向**：按拓扑序计算每个节点，并**缓存**反向需要的中间值；
        2. **反向**：从 $\partial L/\partial L = 1$ 开始，按**逆拓扑序**访问节点；每个节点拿到上游梯度 $\bar{y} = \partial L/\partial y$，乘以自己的局部导数，传给每个输入：$\bar{x} \mathrel{+}= \bar{y}\cdot \partial y/\partial x$。

        三条“局部规则”覆盖了大部分情形：
        - **加法**把梯度原样分发给所有输入；
        - **乘法**把梯度乘以“另一个输入”后交换着传回去；
        - **一个变量被多处使用**（扇出）时，各路梯度**相加**——多元链式法则，也是 PyTorch 里 \`.grad\` 默认累加的原因。

        **为什么用反向模式**：输出是 1 个标量、输入（参数）有几十亿个。反向模式一次遍历就得到所有参数的梯度，代价约为前向的 2 倍；前向模式则需要每个输入方向各遍历一次。
      `,
      viz: CompGraphViz,
      after: md`
        > [!NOTE] 和拓扑排序的关系
        > autograd 记录前向时构建的图，\`loss.backward()\` 按逆拓扑序执行每个节点的 \`backward\`。经典算法路线的“BFS/DFS 与拓扑排序”一关里有一道标量 mini-autograd 的实战，正是这里的实现。
      `,
      checkpoint: {
        id: 'cp-graph',
        type: 'numeric',
        prompt: md`$L = (wx)^2 + wx$，$w = 2$，$x = 3$。求 $\partial L/\partial w$。（提示：$u = wx$ 被用了两次，梯度要相加）`,
        answer: 39,
        tolerance: 0,
        explanation: md`$u = 6$，$\partial L/\partial u = 2u + 1 = 13$，$\partial L/\partial w = 13 \times x = 39$。若只算其中一条路径，会得到 36 或 3。`,
      },
    },
    {
      id: 'linear',
      title: 'Linear 层的反向：靠形状对齐写公式',
      body: md`
        前向 $Y = XW + b$，$X \in \mathbb{R}^{B\times n_{in}}$，$W \in \mathbb{R}^{n_{in}\times n_{out}}$。给定 $\mathrm{d}Y = \partial L/\partial Y$（与 $Y$ 同形状）：

        $$
        \mathrm{d}X = \mathrm{d}Y\, W^\top,\qquad \mathrm{d}W = X^\top \mathrm{d}Y,\qquad \mathrm{d}b = \textstyle\sum_{\text{batch}} \mathrm{d}Y
        $$

        **记忆法**：梯度和对应变量形状相同，而能拼出这个形状的矩阵乘法只有一种。
        $\mathrm{d}W$ 要是 $(n_{in}, n_{out})$，手里有 $X: (B, n_{in})$、$\mathrm{d}Y: (B, n_{out})$，只能是 $X^\top \mathrm{d}Y$——batch 维被求和掉，正好对应“同一个 $W$ 被所有样本共用，梯度要累加”。

        **对三维输入** $(B, T, n_{in})$：先 reshape 成 $(B\cdot T, n_{in})$ 再套公式，$\mathrm{d}W$ 对 $B$ 和 $T$ 两个维度都求和。

        **计算量**：前向一次矩阵乘法 $2Bn_{in}n_{out}$ FLOPs；反向要算 $\mathrm{d}X$ 和 $\mathrm{d}W$ 两次同样大小的矩阵乘法，约为前向的 **2 倍**。于是训练每个 token 约 $2N$（前向）$+ 4N$（反向）$= 6N$ FLOPs——这就是 Chinchilla 等估算里 $6ND$ 的来历。
      `,
      viz: LinearGradViz,
      checkpoint: {
        id: 'cp-linear',
        type: 'fill',
        prompt: md`$X$ 形状 \`(32, 512)\`，$W$ 形状 \`(512, 256)\`。$\mathrm{d}W = X^\top\mathrm{d}Y$ 的形状是 \`(____(1)____, ____(2)____)\`；$\mathrm{d}X$ 的第二维是 ____(3)____。`,
        blanks: [
          { label: '(1)', answers: ['512'], numeric: true, tolerance: 0 },
          { label: '(2)', answers: ['256'], numeric: true, tolerance: 0 },
          { label: '(3)', answers: ['512'], numeric: true, tolerance: 0 },
        ],
        explanation: md`$\mathrm{d}W$ 与 $W$ 同形 \`(512, 256)\`；$\mathrm{d}X = \mathrm{d}Y W^\top$：\`(32, 256) @ (256, 512) = (32, 512)\`，与 $X$ 同形。`,
      },
    },
    {
      id: 'softmax',
      title: '逐元素激活、softmax 与 softmax-CE 的反向',
      body: md`
        **ReLU**：$y = \max(0, x)$，$\mathrm{d}x = \mathrm{d}y \odot [x > 0]$。$x = 0$ 处不可导，约定梯度为 0（PyTorch 如此）。需要缓存输入（或只缓存布尔 mask，省显存）。

        **softmax**：$p = \mathrm{softmax}(z)$ 的雅可比是 $V\times V$ 矩阵

        $$
        \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)\quad\Longleftrightarrow\quad J = \mathrm{diag}(p) - pp^\top
        $$

        但反向从来不需要显式构造 $J$（词表 15 万时它有 $2\times10^{10}$ 个元素）。我们只要**向量-雅可比积**（VJP）：

        $$
        \mathrm{d}z = J^\top \mathrm{d}p = p \odot \big(\mathrm{d}p - \langle \mathrm{d}p, p\rangle\big)
        $$

        只需 $O(V)$ 的计算——autograd 里每个算子实现的都是 VJP，而不是雅可比本身。

        **softmax + 交叉熵**：上游 $\mathrm{d}p_i = -y_i/p_i$（one-hot $y$），代入得 $\mathrm{d}z = p \odot(-y/p + 1) = p - y$。两步合在一起的梯度极其简单，这就是框架把它们融合成一个算子的原因（见「Softmax 与交叉熵」一关）。

        **batch 均值**：loss 取 mean 时，每个样本的梯度都要再除以 $N$——手写 MLP 时最容易漏。
      `,
      checkpoint: {
        id: 'cp-softmax',
        type: 'numeric',
        prompt: md`$p = \mathrm{softmax}(z) = [0.5, 0.3, 0.2]$。雅可比元素 $\partial p_1/\partial z_2$（下标从 1 开始，即第 1 个概率对第 2 个 logit）是多少？`,
        answer: -0.15,
        tolerance: 0.001,
        explanation: md`$i \ne j$ 时 $\partial p_i/\partial z_j = -p_ip_j = -0.5 \times 0.3 = -0.15$；对角元 $\partial p_1/\partial z_1 = p_1(1-p_1) = 0.25$。每一行之和为 0（softmax 对平移不变）。`,
      },
    },
    {
      id: 'mlp',
      title: '把它们串起来：两层 MLP 的完整反向',
      body: md`
        前向（$X: (N, D)$，标签 $y: (N,)$）：

        | 步骤 | 前向 | 反向 |
        |---|---|---|
        | 1 | $z_1 = XW_1 + b_1$ | $\mathrm{d}W_1 = X^\top \mathrm{d}z_1$，$\mathrm{d}b_1 = \sum \mathrm{d}z_1$ |
        | 2 | $h = \mathrm{ReLU}(z_1)$ | $\mathrm{d}z_1 = \mathrm{d}h \odot [z_1 > 0]$ |
        | 3 | $s = hW_2 + b_2$ | $\mathrm{d}W_2 = h^\top \mathrm{d}s$，$\mathrm{d}b_2 = \sum \mathrm{d}s$，$\mathrm{d}h = \mathrm{d}s\, W_2^\top$ |
        | 4 | $L = \frac{1}{N}\sum_n \mathrm{CE}(s_n, y_n)$ | $\mathrm{d}s = (\mathrm{softmax}(s) - \mathrm{onehot}(y))/N$ |

        反向从表格最下面一行往上执行。需要从前向缓存的量：$X$、$z_1$（或 ReLU mask）、$h$、$\mathrm{softmax}(s)$。

        > [!TIP] 面试手撕的检查清单
        > 1. 每个梯度的形状与对应变量一致；
        > 2. 偏置梯度是对 batch 维**求和**（不是平均——平均已经在 $\mathrm{d}s$ 的 $1/N$ 里了）；
        > 3. $\mathrm{d}h$ 用 $W_2^\top$；
        > 4. 写完用数值梯度检查一遍。
      `,
      checkpoint: {
        id: 'cp-mlp',
        type: 'single',
        prompt: md`上表中，若 loss 用 mean 归约，$\mathrm{d}b_2$ 应该是：`,
        options: ['`ds.mean(axis=0)`，其中 ds = p − onehot', '`ds.sum(axis=0)`，其中 ds = (p − onehot) / N', '`ds.sum(axis=1)`', '`ds.sum()`（一个标量）'],
        answer: 1,
        explanation: md`$1/N$ 已经放进了 $\mathrm{d}s$，偏置梯度对 batch 维求和即可。A 在数值上与 B 相同，但混淆了“谁负责除以 N”，一旦 loss 的归约方式变了（如 sum 或加权）就会出错；C 的轴错了；D 把所有类别加在了一起，形状不对。`,
      },
    },
    {
      id: 'gradcheck',
      title: '数值梯度检查：手写反向的唯一可靠验收',
      body: md`
        **中心差分**：$\dfrac{\partial L}{\partial \theta_i} \approx \dfrac{L(\theta + h e_i) - L(\theta - h e_i)}{2h}$，截断误差 $O(h^2)$，比前向差分的 $O(h)$ 好得多。

        **相对误差**：$\dfrac{|g_a - g_n|}{\max(|g_a|, |g_n|, \text{tiny})}$。经验阈值（CS231n）：$> 10^{-2}$ 基本是错的；$10^{-4}$ 附近要警惕（除非有 ReLU 拐点）；float64 下正确实现通常能到 $10^{-7}$ 以下。

        **实践要点**：
        - **必须用 float64**：舍入误差约 $\varepsilon_{\text{mach}}|L|/h$，float32 的 $\varepsilon \approx 1.2\times10^{-7}$。下图这个简单函数上 float32 最好也只能到约 $10^{-5}$，且要把 $h$ 放大到 $10^{-2}$ 附近（对有拐点的网络，这么大的 $h$ 很容易跨过拐点）；真实网络的 loss 是大量项之和，误差只会更大；
        - $h$ 取 $10^{-5}$～$10^{-6}$；
        - 关掉 dropout 等随机性，固定数据；
        - 参数很多时只抽查若干随机坐标；
        - 注意不可导点（ReLU 在 0、max 的并列）：$\pm h$ 跨过拐点时数值梯度会“平均”两侧斜率。
      `,
      viz: GradCheckViz,
      checkpoint: {
        id: 'cp-gradcheck',
        type: 'single',
        prompt: md`手写反向后做梯度检查，得到相对误差 $3\times10^{-3}$（float64，$h = 10^{-5}$，网络里没有 ReLU）。最合理的判断是：`,
        options: ['精度很好，可以继续', '大概率有 bug，但可能只是某个小项（如漏了 1/N 以外的系数或某个分支）', '一定是 h 取得太小导致的舍入误差', '应该改用 float32 重新检查'],
        answer: 1,
        explanation: md`float64、合理的 $h$、无拐点时，正确实现的误差应在 $10^{-7}$ 附近；$10^{-3}$ 说明解析梯度有系统性偏差。C：$h = 10^{-5}$ 时 float64 的舍入误差约 $10^{-11}$，不是主因。D 会让误差更大。`,
      },
    },
    {
      id: 'memory',
      title: '反向要缓存什么：激活显存与重计算',
      body: md`
        反向需要前向的中间结果：Linear 要输入 $X$ 来算 $\mathrm{d}W$，ReLU 要 mask，softmax 要 $p$，LayerNorm 要 $\mu, r$……这些**激活**在前向时就要保存，直到反向用完才能释放。

        对 Transformer 训练，激活显存与 $B \times T \times d \times L$ 成正比，长序列下往往比参数 + 优化器状态还大。常见的省法：

        | 方法 | 思路 | 代价 |
        |---|---|---|
        | 激活重计算（activation checkpointing） | 只保存每层的输入，反向时重新前向一次 | 多约 1 次前向计算（总计算约 +33%） |
        | 选择性重计算 | 只重算“显存大、计算便宜”的部分（如注意力分数） | 额外计算很少 |
        | 只存 mask / 低精度 | ReLU 只存 1 bit mask；某些激活存 FP8 | 实现复杂度 |
        | 融合算子 | 把 softmax-CE、LayerNorm 等融合，中间结果不落显存 | 需要写 kernel |

        > [!INTERVIEW] 常见追问
        > “为什么 in-place 操作有时会让 autograd 报错？”——因为它覆盖了反向需要的缓存值。PyTorch 用版本计数器检测这种情况。
      `,
      checkpoint: {
        id: 'cp-memory',
        type: 'judge',
        prompt: md`对 $y = \mathrm{ReLU}(x)$，反向只需要知道 $x > 0$ 的布尔 mask，不需要保存 $x$ 的具体数值。`,
        answer: true,
        explanation: md`$\mathrm{d}x = \mathrm{d}y \odot [x>0]$，只依赖符号。实际上保存输出 $y$ 也行（$y > 0 \Leftrightarrow x > 0$），PyTorch 的 \`relu\` 就是保存输出来做反向的，这样 in-place ReLU 也能正确求导。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-chain',
      type: 'single',
      difficulty: 1,
      prompt: md`$y = f(g(x))$，链式法则给出 $\mathrm{d}y/\mathrm{d}x$ 为：`,
      options: ["$f'(x)\\,g'(x)$", "$f'(g(x))\\,g'(x)$", "$f'(g'(x))$", "$f(g'(x))$"],
      answer: 1,
      explanation: md`外层导数要在**内层的值** $g(x)$ 处求值，再乘内层导数。这也是反向传播要缓存前向中间值的原因：$f'$ 需要 $g(x)$。`,
    },
    {
      id: 'q-fanout',
      type: 'judge',
      difficulty: 1,
      prompt: md`一个张量在前向中被两个不同的运算使用，反向时它的梯度是两条路径传回来的梯度之和。`,
      answer: true,
      explanation: md`多元链式法则：$\frac{\partial L}{\partial x} = \sum_k \frac{\partial L}{\partial y_k}\frac{\partial y_k}{\partial x}$。残差连接 $x + F(x)$ 就是典型：$x$ 的梯度 = 恒等路径的梯度 + 经过 $F$ 的梯度。`,
    },
    {
      id: 'q-flops',
      type: 'numeric',
      difficulty: 2,
      prompt: md`$X$: \`(32, 512)\`，$W$: \`(512, 256)\`。前向 $XW$ 约 $2 \times 32 \times 512 \times 256$ FLOPs。反向计算 $\mathrm{d}X$ 与 $\mathrm{d}W$ 一共约多少 FLOPs？`,
      answer: 16777216,
      tolerance: 0.001,
      explanation: md`前向 $2\cdot32\cdot512\cdot256 = 8{,}388{,}608$。$\mathrm{d}X = \mathrm{d}YW^\top$ 与 $\mathrm{d}W = X^\top\mathrm{d}Y$ 各需要同样多的乘加，合计 $16{,}777{,}216$，即前向的 2 倍。前向 + 反向 = 3 倍前向 → 训练每 token 约 $6N$ FLOPs。`,
    },
    {
      id: 'q-linear-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`前向 \`Y = X @ W\`，补全反向：\`dX = dY @ ____(1)____\`，\`dW = ____(2)____ @ dY\`。（写 NumPy 表达式）`,
      blanks: [
        { label: '(1)', answers: ['W.T', 'W.transpose()', 'np.transpose(W)', 'W.swapaxes(0,1)'] },
        { label: '(2)', answers: ['X.T', 'X.transpose()', 'np.transpose(X)', 'X.swapaxes(0,1)'] },
      ],
      explanation: md`形状对齐：\`dX\` 要 \`(B, in)\` = \`(B, out) @ (out, in)\`；\`dW\` 要 \`(in, out)\` = \`(in, B) @ (B, out)\`。`,
    },
    {
      id: 'q-cache',
      type: 'multi',
      difficulty: 2,
      prompt: md`为了完成反向传播，前向时需要缓存（或能重算出）哪些量？（多选）`,
      options: [
        'Linear 层的输入 $X$（用于计算 $\\mathrm{d}W$）',
        'ReLU 的输入符号（mask）',
        'softmax 的输出 $p$（用于 VJP 或 $p - y$）',
        'Linear 层的输出 $Y$（用于计算 $\\mathrm{d}X$）',
        'LayerNorm 每行的 $\\mu$ 与 $1/\\sigma$',
      ],
      answer: [0, 1, 2, 4],
      explanation: md`D 错：$\mathrm{d}X = \mathrm{d}YW^\top$ 只需要权重 $W$ 和上游梯度，不需要 Linear 的输出 $Y$（$Y$ 可能被下一层当作输入缓存，但那是另一回事）。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`前向为 \`z1 = X @ W1 + b1; h = relu(z1); logits = h @ W2 + b2\`，\`dlogits\` 已包含 1/N。下面的反向有 **2 处** bug：`,
      code: py`
        def mlp_backward(X, z1, h, W2, dlogits):
            dW2 = h.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ W2
            dz1 = dh * (z1 > 0)
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=1)
            return dW1, db1, dW2, db2
      `,
      bugLines: [4, 7],
      explanation: md`
        - 第 4 行应为 \`dlogits @ W2.T\`：\`dlogits\` 是 \`(N, C)\`，\`W2\` 是 \`(H, C)\`，要得到 \`(N, H)\` 必须转置（当 $H = C$ 时不会报错，却静默地算错）。
        - 第 7 行偏置梯度应对 batch 维求和：\`dz1.sum(axis=0)\`，结果形状 \`(H,)\`。
        第 5 行用 \`z1 > 0\` 作为 ReLU mask 是对的（等价于 \`h > 0\`）。
      `,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`两层 MLP（Linear → ReLU → Linear → softmax-CE）的反向计算，按执行顺序排列：`,
      items: [
        '$\\mathrm{d}s = (p - \\mathrm{onehot}(y))/N$',
        '$\\mathrm{d}W_2 = h^\\top\\mathrm{d}s$，$\\mathrm{d}b_2 = \\sum\\mathrm{d}s$',
        '$\\mathrm{d}h = \\mathrm{d}s\\,W_2^\\top$',
        '$\\mathrm{d}z_1 = \\mathrm{d}h \\odot [z_1 > 0]$',
        '$\\mathrm{d}W_1 = X^\\top\\mathrm{d}z_1$，$\\mathrm{d}b_1 = \\sum\\mathrm{d}z_1$',
      ],
      explanation: md`逆拓扑序：从 loss 往输入方向。第 2、3 步互不依赖，可以交换；但第 4 步必须在第 3 步之后。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把前向运算与它的反向（VJP）配对：`,
      pairs: [
        ['$Y = XW$', '$\\mathrm{d}X = \\mathrm{d}Y\\,W^\\top$'],
        ['$y = \\mathrm{ReLU}(x)$', '$\\mathrm{d}x = \\mathrm{d}y \\odot [x > 0]$'],
        ['$y = x + b$（$b$ 广播到 batch）', '$\\mathrm{d}b = \\sum_{\\text{batch}} \\mathrm{d}y$'],
        ['$s = \\sum_i x_i$', '$\\mathrm{d}x_i = \\mathrm{d}s$（广播）'],
        ['$p = \\mathrm{softmax}(z)$', '$\\mathrm{d}z = p \\odot (\\mathrm{d}p - \\langle\\mathrm{d}p, p\\rangle)$'],
      ],
      explanation: md`一个有用的对偶：**前向广播 ⇔ 反向求和**，**前向求和 ⇔ 反向广播**。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出一个逐元素的中心差分数值梯度函数（会原地扰动 \`x\`，结束时必须恢复）。有 2 行干扰项。`,
      lines: [
        'def numerical_grad(f, x, h=1e-5):',
        '    g = np.zeros_like(x)',
        '    for idx in np.ndindex(*x.shape):',
        '        old = x[idx]',
        '        x[idx] = old + h',
        '        fp = f(x)',
        '        x[idx] = old - h',
        '        fm = f(x)',
        '        x[idx] = old',
        '        g[idx] = (fp - fm) / (2 * h)',
        '    return g',
      ],
      distractors: ['        g[idx] = (fp - fm) / h', '        x[idx] = old - 2 * h'],
      explanation: md`中心差分的分母是 $2h$（干扰项 1 写成 $h$，结果会大一倍）。两次扰动都以原值 \`old\` 为基准：干扰项 2 把第二次扰动写成 \`old - 2 * h\`，两次求值点的间隔变成 $3h$，与分母 $2h$ 不匹配。最后一定要把 \`x[idx]\` 恢复。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 1,
      prompt: md`ReLU 的输入 $z$ 如下（2 × 4）。反向时哪些位置的梯度能“流过去”（不被置 0）？按 PyTorch 约定，$z = 0$ 处梯度为 0。`,
      items: ['1.5', '-0.2', '0', '3.0', '-2.1', '0.01', '0', '-0.5'],
      cols: 4,
      rowLabels: ['样本 0', '样本 1'],
      colLabels: ['h0', 'h1', 'h2', 'h3'],
      answer: [0, 3, 5],
      explanation: md`只有 $z > 0$ 的位置梯度为 $\mathrm{d}y$：1.5、3.0、0.01。恰好为 0 的位置按约定取 0 梯度（次梯度可以取 $[0,1]$ 中任意值，框架选了 0）。`,
    },
    {
      id: 'q-softmax-jac',
      type: 'numeric',
      difficulty: 3,
      prompt: md`$p = \mathrm{softmax}(z) = [0.5, 0.3, 0.2]$，上游梯度 $\mathrm{d}p = [1, 0, 0]$。用 VJP 公式 $\mathrm{d}z = p\odot(\mathrm{d}p - \langle \mathrm{d}p, p\rangle)$ 求 $\mathrm{d}z_1$（第一个分量）。`,
      answer: 0.25,
      tolerance: 0.001,
      explanation: md`$\langle \mathrm{d}p, p\rangle = 0.5$；$\mathrm{d}z = [0.5\times0.5,\ 0.3\times(-0.5),\ 0.2\times(-0.5)] = [0.25, -0.15, -0.10]$，正好是雅可比的第一行，且和为 0。`,
    },
    {
      id: 'q-why-reverse',
      type: 'single',
      difficulty: 3,
      prompt: md`为什么神经网络训练用**反向模式**而不是前向模式自动微分？`,
      options: [
        '前向模式无法处理非线性函数',
        '输出是一个标量 loss、输入参数极多：反向模式一次遍历得到全部梯度，前向模式需要每个参数方向各遍历一次',
        '反向模式不需要缓存中间结果，更省显存',
        '前向模式的数值精度更差',
      ],
      answer: 1,
      explanation: md`对 $f: \mathbb{R}^n \to \mathbb{R}^m$，前向模式每次得到雅可比的一列（代价 ∝ $n$），反向模式每次得到一行（代价 ∝ $m$）。训练时 $m = 1$、$n$ 达数十亿。C 恰好相反：反向模式需要缓存前向的中间结果，这正是激活显存的来源。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-linear',
      title: 'Linear 层的前向与反向',
      intro: md`
        实现 \`linear_backward(dY, cache)\`：前向为 \`Y = X @ W + b\`，\`X\` 形状 \`(N, in)\`，\`W\` 形状 \`(in, out)\`，\`b\` 形状 \`(out,)\`，\`cache = (X, W)\`。返回 \`(dX, dW, db)\`，形状分别与 \`X, W, b\` 相同。
      `,
      template: py`
        import numpy as np

        def linear_forward(X, W, b):
            return X @ W + b, (X, W)

        def linear_backward(dY, cache):
            X, W = cache
            dX = {{1}}
            dW = {{2}}
            db = {{3}}
            return dX, dW, db
      `,
      steps: [
        {
          blank: 1,
          instruction: md`输入梯度：\`(N, out)\` 乘什么能得到 \`(N, in)\`？`,
          answer: 'dY @ W.T',
          accept: ['np.dot(dY, W.T)', 'dY @ W.transpose()', 'np.matmul(dY, W.T)'],
          explain: md`$\mathrm{d}X_{n,i} = \sum_j \mathrm{d}Y_{n,j} W_{i,j}$，即 $\mathrm{d}Y W^\top$。`,
        },
        {
          blank: 2,
          instruction: md`权重梯度：要得到 \`(in, out)\`，并把所有样本的贡献加起来。`,
          answer: 'X.T @ dY',
          accept: ['np.dot(X.T, dY)', 'X.transpose() @ dY', 'np.matmul(X.T, dY)'],
          explain: md`矩阵乘法里对 batch 维的求和，正好对应“同一个 $W$ 被 $N$ 个样本共用”。`,
        },
        {
          blank: 3,
          instruction: md`偏置梯度：前向中 \`b\` 被广播到每个样本，反向就对 batch 维求和。`,
          answer: 'dY.sum(axis=0)',
          accept: ['np.sum(dY, axis=0)'],
          explain: md`前向广播 ⇔ 反向求和。写成 \`mean\` 是常见错误——平均（如果有）应由 loss 的归约负责。`,
        },
      ],
      setup: py`
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
          name: '形状',
          code: py`
            rng = np.random.default_rng(0)
            X, W, b = rng.normal(size=(5, 3)), rng.normal(size=(3, 4)), rng.normal(size=4)
            Y, cache = linear_forward(X, W, b)
            dX, dW, db = linear_backward(np.ones_like(Y), cache)
            assert dX.shape == (5, 3) and dW.shape == (3, 4) and db.shape == (4,), f"形状错误：{dX.shape}, {dW.shape}, {db.shape}"
            assert np.allclose(db, 5.0), f"dY 全为 1 时 db 应全为 N=5，得到 {db}"
          `,
        },
        {
          name: '数值梯度检查',
          code: py`
            rng = np.random.default_rng(1)
            X, W, b = rng.normal(size=(4, 3)), rng.normal(size=(3, 2)), rng.normal(size=2)
            dY = rng.normal(size=(4, 2))
            _, cache = linear_forward(X, W, b)
            dX, dW, db = linear_backward(dY, cache)
            L = lambda X_, W_, b_: float(((X_ @ W_ + b_) * dY).sum())
            assert np.allclose(dX, _num_grad(lambda z: L(z, W, b), X.copy()), atol=1e-6), "dX 与数值梯度不一致"
            assert np.allclose(dW, _num_grad(lambda z: L(X, z, b), W.copy()), atol=1e-6), "dW 与数值梯度不一致"
            assert np.allclose(db, _num_grad(lambda z: L(X, W, z), b.copy()), atol=1e-6), "db 与数值梯度不一致"
          `,
        },
      ],
      outro: md`
        小技巧：测试里把 loss 设成 $L = \sum (Y \odot \mathrm{d}Y)$，则 $\partial L/\partial Y = \mathrm{d}Y$——这是检查任意层 VJP 的通用方法，实战题会反复用到。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-mlp',
      title: '两层 MLP：前向 + 反向',
      difficulty: 'medium',
      source: '大模型岗手撕 · 反向传播',
      timeLimitMin: 25,
      description: md`
        实现两层 MLP 分类器的损失与梯度：

        \`\`\`python
        def mlp_loss_and_grads(params: dict, X: np.ndarray, y: np.ndarray) -> tuple[float, dict]: ...
        \`\`\`

        - \`params\`：\`W1 (D, H)\`、\`b1 (H,)\`、\`W2 (H, C)\`、\`b2 (C,)\`；\`X (N, D)\`；\`y (N,)\` 为整数类别。
        - 前向：$h = \mathrm{ReLU}(XW_1 + b_1)$，$s = hW_2 + b_2$，$L = \frac{1}{N}\sum_n -\log\mathrm{softmax}(s_n)_{y_n}$。
        - 返回 \`(loss, grads)\`，\`grads\` 的键与 \`params\` 相同、形状相同。ReLU 在 0 处梯度取 0。
        - softmax 要数值稳定（logits 可能很大）。不要修改 \`params\`。
      `,
      starter: py`
        import numpy as np

        def mlp_loss_and_grads(params, X, y):
            W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
            # TODO: 前向（缓存 z1, h, p）→ loss → 反向
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def mlp_loss_and_grads(params, X, y):
            W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
            N = X.shape[0]
            # ---- 前向 ----
            z1 = X @ W1 + b1                      # (N, H)
            h = np.maximum(z1, 0.0)               # ReLU
            s = h @ W2 + b2                       # (N, C)
            m = s.max(axis=1, keepdims=True)
            logp = s - m - np.log(np.exp(s - m).sum(axis=1, keepdims=True))
            loss = float(-logp[np.arange(N), y].mean())
            # ---- 反向 ----
            ds = np.exp(logp)                     # softmax
            ds[np.arange(N), y] -= 1.0
            ds /= N                               # mean 归约
            dW2 = h.T @ ds
            db2 = ds.sum(axis=0)
            dh = ds @ W2.T
            dz1 = dh * (z1 > 0)                   # z1 == 0 处梯度为 0
            dW1 = X.T @ dz1
            db1 = dz1.sum(axis=0)
            return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：用 autograd 对拍）',
          code: py`
            import torch
            import torch.nn.functional as F

            def mlp_loss_and_grads(params: dict, X: torch.Tensor, y: torch.Tensor):
                p = {k: v.detach().clone().double().requires_grad_(True) for k, v in params.items()}
                h = F.relu(X.double() @ p["W1"] + p["b1"])
                loss = F.cross_entropy(h @ p["W2"] + p["b2"], y)
                loss.backward()
                return loss.item(), {k: v.grad for k, v in p.items()}
          `,
        },
      ],
      explanation: md`
        **反向顺序**：$\mathrm{d}s = (p - \mathrm{onehot})/N$ → $\mathrm{d}W_2 = h^\top\mathrm{d}s$、$\mathrm{d}b_2 = \sum\mathrm{d}s$ → $\mathrm{d}h = \mathrm{d}sW_2^\top$ → $\mathrm{d}z_1 = \mathrm{d}h\odot[z_1>0]$ → $\mathrm{d}W_1 = X^\top\mathrm{d}z_1$、$\mathrm{d}b_1 = \sum\mathrm{d}z_1$。

        **易错点**：忘记除以 $N$；偏置梯度用了 mean 或错误的轴；\`dh\` 没转置 $W_2$；softmax 不稳定；ReLU mask 用 \`>=\`（与约定不符，隐藏测试会构造恰好为 0 的预激活）。

        **复杂度**：$O(N(DH + HC))$，与前向同阶（约 2 倍）。
      `,
      complexity: { time: 'O(N·(D·H + H·C))', space: 'O(N·(H + C))' },
      hints: [
        md`前向时保存 \`z1\`、\`h\`、\`logp\`；\`p = exp(logp)\`。`,
        md`\`ds = p.copy(); ds[arange(N), y] -= 1; ds /= N\`。`,
        md`\`dh = ds @ W2.T\`，\`dz1 = dh * (z1 > 0)\`，偏置梯度都是 \`.sum(axis=0)\`。`,
      ],
      setup: py`
        def _ref_loss(params, X, y):
            h = np.maximum(X @ params["W1"] + params["b1"], 0)
            s = h @ params["W2"] + params["b2"]
            m = s.max(1, keepdims=True)
            logp = s - m - np.log(np.exp(s - m).sum(1, keepdims=True))
            return float(-logp[np.arange(len(y)), y].mean())

        def _make(seed, N=6, D=4, H=5, C=3, scale=1.0):
            rng = np.random.default_rng(seed)
            params = {"W1": rng.normal(size=(D, H)) * scale, "b1": rng.normal(size=H) * 0.1,
                      "W2": rng.normal(size=(H, C)) * scale, "b2": rng.normal(size=C) * 0.1}
            return params, rng.normal(size=(N, D)), rng.integers(0, C, size=N)

        def _num_grad_param(params, X, y, key, h=1e-6):
            p = {k: v.copy() for k, v in params.items()}
            g = np.zeros_like(p[key])
            for idx in np.ndindex(*p[key].shape):
                old = p[key][idx]
                p[key][idx] = old + h; fp = _ref_loss(p, X, y)
                p[key][idx] = old - h; fm = _ref_loss(p, X, y)
                p[key][idx] = old
                g[idx] = (fp - fm) / (2 * h)
            return g
      `,
      tests: [
        {
          name: 'loss 数值与梯度形状',
          code: py`
            params, X, y = _make(0)
            loss, grads = mlp_loss_and_grads(params, X, y)
            exp = _ref_loss(params, X, y)
            assert abs(float(loss) - exp) < 1e-9, f"loss 期望 {exp:.6f}，得到 {loss}"
            for k in params:
                assert k in grads, f"grads 缺少键 {k}"
                assert grads[k].shape == params[k].shape, f"{k} 的梯度形状应为 {params[k].shape}，得到 {grads[k].shape}"
          `,
        },
        {
          name: '数值梯度检查（全部参数）',
          code: py`
            params, X, y = _make(1)
            _, grads = mlp_loss_and_grads(params, X, y)
            for k in params:
                ng = _num_grad_param(params, X, y, k)
                err = np.abs(grads[k] - ng).max()
                assert err < 1e-6, f"{k} 与数值梯度最大误差 {err:.2e}"
          `,
        },
        {
          name: '不修改 params',
          code: py`
            params, X, y = _make(2)
            snap = {k: v.copy() for k, v in params.items()}
            mlp_loss_and_grads(params, X, y)
            for k in params:
                assert np.array_equal(params[k], snap[k]), f"params['{k}'] 被修改了"
          `,
        },
        {
          name: '大 logits 稳定',
          hidden: true,
          code: py`
            params, X, y = _make(3, scale=30.0)
            loss, grads = mlp_loss_and_grads(params, X, y)
            assert np.isfinite(loss), f"loss 不是有限值：{loss}"
            assert all(np.all(np.isfinite(g)) for g in grads.values()), "梯度中出现 NaN/inf"
            assert abs(float(loss) - _ref_loss(params, X, y)) < 1e-6 * max(1, abs(loss)), "大 logits 下 loss 不正确"
          `,
        },
        {
          name: 'ReLU 在 0 处梯度为 0',
          hidden: true,
          code: py`
            # 构造恰好为 0 的预激活：X 的某一列为 0，对应 b1 为 0
            params = {"W1": np.array([[1.0, -1.0], [0.0, 0.0]]), "b1": np.array([0.0, 0.5]),
                      "W2": np.array([[1.0, -2.0], [0.5, 1.0]]), "b2": np.zeros(2)}
            X = np.array([[0.0, 3.0], [2.0, 1.0]])
            y = np.array([0, 1])
            _, grads = mlp_loss_and_grads(params, X, y)
            # 第 0 个样本 z1 = [0, 0.5]：第 0 个隐藏单元恰为 0，不应贡献梯度
            z1 = X @ params["W1"] + params["b1"]
            assert z1[0, 0] == 0.0
            # 只有第 1 个样本 (z1=[2, -1.5]) 的第 0 个隐藏单元有梯度
            h = np.maximum(z1, 0); s = h @ params["W2"]
            p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            ds = p.copy(); ds[np.arange(2), y] -= 1; ds /= 2
            dh = ds @ params["W2"].T
            exp = dh[1, 0]
            assert abs(grads["b1"][0] - exp) < 1e-12, f"db1[0] 期望 {exp}（z1=0 的位置不应传梯度），得到 {grads['b1'][0]}"
          `,
        },
        {
          name: '更大规模对拍',
          hidden: true,
          code: py`
            params, X, y = _make(4, N=16, D=6, H=8, C=5)
            _, grads = mlp_loss_and_grads(params, X, y)
            for k in ("W1", "b2"):
                ng = _num_grad_param(params, X, y, k)
                assert np.abs(grads[k] - ng).max() < 1e-6, f"{k} 梯度不一致"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-init',
          q: md`这个 MLP 如果把 $W_1, W_2$ 全部初始化为 0 会怎样？`,
          a: md`
            $W_1 = 0$ 时所有隐藏单元的输出相同（都等于 $\mathrm{ReLU}(b_1)$），反向时它们收到的梯度也完全相同，更新后仍然相同——**对称性无法打破**，等价于只有一个隐藏单元。若 $b_1 \le 0$，ReLU 全为 0，$W_1$ 根本收不到梯度。所以权重要随机初始化（如 Kaiming / Xavier，方差约 $2/\text{fan\_in}$ 或 $1/\text{fan\_in}$），偏置可以为 0。
          `,
        },
      ],
    },
    {
      id: 'c-vjp',
      title: '数值梯度与 softmax 的 VJP',
      difficulty: 'medium',
      source: '大模型岗手撕 · 自动微分',
      timeLimitMin: 15,
      description: md`
        实现两个函数：

        \`\`\`python
        def numerical_grad(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray: ...
        def softmax_backward(p: np.ndarray, dp: np.ndarray, axis: int = -1) -> np.ndarray: ...
        \`\`\`

        - \`numerical_grad\`：\`f\` 接收与 \`x\` 同形状的数组、返回标量。用**中心差分**逐元素估计梯度；可以原地扰动 \`x\`，但**返回前必须恢复原值**。
        - \`softmax_backward\`：已知 \`p = softmax(z, axis)\` 与上游梯度 \`dp = ∂L/∂p\`，返回 \`dz = ∂L/∂z\`。**不要**构造雅可比矩阵，复杂度应为 $O(\text{numel})$。

        **示例**：\`softmax_backward([0.5, 0.3, 0.2], [1, 0, 0])\` → \`[0.25, -0.15, -0.10]\`
      `,
      starter: py`
        import numpy as np

        def numerical_grad(f, x, h=1e-5):
            # TODO: 中心差分，记得恢复 x
            raise NotImplementedError

        def softmax_backward(p, dp, axis=-1):
            # TODO: dz = p * (dp - sum(dp * p, axis))
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def numerical_grad(f, x, h=1e-5):
            g = np.zeros_like(x, dtype=np.float64)
            for idx in np.ndindex(*x.shape):
                old = x[idx]
                x[idx] = old + h
                fp = f(x)
                x[idx] = old - h
                fm = f(x)
                x[idx] = old                      # 恢复
                g[idx] = (fp - fm) / (2 * h)
            return g

        def softmax_backward(p, dp, axis=-1):
            p = np.asarray(p, dtype=np.float64)
            dp = np.asarray(dp, dtype=np.float64)
            inner = np.sum(dp * p, axis=axis, keepdims=True)   # <dp, p>，沿 softmax 的轴
            return p * (dp - inner)
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：自定义 softmax 的 backward）',
          code: py`
            import torch

            class Softmax(torch.autograd.Function):
                @staticmethod
                def forward(ctx, z, dim=-1):
                    p = torch.softmax(z, dim=dim)
                    ctx.save_for_backward(p)
                    ctx.dim = dim
                    return p

                @staticmethod
                def backward(ctx, dp):
                    (p,) = ctx.saved_tensors
                    return p * (dp - (dp * p).sum(dim=ctx.dim, keepdim=True)), None

            # 验证：torch.autograd.gradcheck(lambda z: Softmax.apply(z), (torch.randn(3, 5, dtype=torch.double, requires_grad=True),))
          `,
        },
      ],
      explanation: md`
        **VJP 推导**：$\mathrm{d}z_j = \sum_i \mathrm{d}p_i\, p_i(\delta_{ij} - p_j) = p_j\mathrm{d}p_j - p_j\sum_i \mathrm{d}p_i p_i = p_j(\mathrm{d}p_j - \langle\mathrm{d}p, p\rangle)$。

        **数值梯度**：中心差分 + 恢复原值；\`f\` 可能依赖 \`x\` 的全部元素，所以每次只扰动一个元素。用 float64。

        **复杂度**：VJP 为 $O(n)$；数值梯度需要 $2n$ 次函数求值，只适合小规模验证。
      `,
      complexity: { time: 'VJP O(n)；数值梯度 O(n·cost(f))', space: 'O(n)' },
      hints: [
        md`\`numerical_grad\`：\`for idx in np.ndindex(*x.shape)\`，扰动 \`x[idx]\` 为 \`old ± h\`，最后 \`x[idx] = old\`。`,
        md`\`softmax_backward\`：\`p * (dp - (dp * p).sum(axis=axis, keepdims=True))\`。`,
      ],
      setup: py`
        def _sm(z, axis=-1):
            e = np.exp(z - z.max(axis=axis, keepdims=True))
            return e / e.sum(axis=axis, keepdims=True)
      `,
      tests: [
        {
          name: '示例',
          code: py`
            got = softmax_backward(np.array([0.5, 0.3, 0.2]), np.array([1.0, 0.0, 0.0]))
            assert np.allclose(got, [0.25, -0.15, -0.10]), f"期望 [0.25, -0.15, -0.10]，得到 {got}"
          `,
        },
        {
          name: 'numerical_grad 正确且恢复 x',
          code: py`
            x = np.array([[1.0, -2.0], [0.5, 3.0]])
            x0 = x.copy()
            g = numerical_grad(lambda z: float((z ** 3).sum()), x)
            assert np.allclose(g, 3 * x0 ** 2, atol=1e-6), f"期望 {3 * x0 ** 2}，得到 {g}"
            assert np.array_equal(x, x0), "numerical_grad 返回后 x 没有被恢复"
          `,
        },
        {
          name: 'VJP 与数值梯度一致（axis=-1）',
          code: py`
            rng = np.random.default_rng(0)
            z = rng.normal(size=(3, 5))
            dp = rng.normal(size=(3, 5))
            got = softmax_backward(_sm(z), dp)
            ng = numerical_grad(lambda t: float((_sm(t) * dp).sum()), z.copy())
            assert np.allclose(got, ng, atol=1e-7), f"最大误差 {np.abs(got - ng).max():.2e}"
          `,
        },
        {
          name: '任意 axis',
          hidden: true,
          code: py`
            rng = np.random.default_rng(1)
            z = rng.normal(size=(2, 4, 3))
            dp = rng.normal(size=(2, 4, 3))
            for ax in (0, 1, 2):
                got = softmax_backward(_sm(z, ax), dp, axis=ax)
                ng = numerical_grad(lambda t: float((_sm(t, ax) * dp).sum()), z.copy())
                assert np.allclose(got, ng, atol=1e-7), f"axis={ax} 最大误差 {np.abs(got - ng).max():.2e}"
          `,
        },
        {
          name: '不构造雅可比（大词表）',
          hidden: true,
          code: py`
            import time
            V = 200_000
            rng = np.random.default_rng(2)
            p = _sm(rng.normal(size=(1, V)))
            dp = rng.normal(size=(1, V))
            t0 = time.perf_counter()
            got = softmax_backward(p, dp)
            dt = time.perf_counter() - t0
            assert got.shape == (1, V), f"形状应为 (1, {V})，得到 {got.shape}"
            assert abs(got.sum()) < 1e-8, f"每行 dz 之和应为 0，得到 {got.sum()}"
            assert dt < 0.5, f"耗时 {dt:.2f}s——是不是构造了 V×V 的雅可比？"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-jvp',
          q: md`VJP 和 JVP 有什么区别？什么时候用 JVP？`,
          a: md`
            VJP（向量-雅可比积，$v^\top J$）对应反向模式，一次得到“一个输出方向”对所有输入的梯度；JVP（雅可比-向量积，$Jv$）对应前向模式，一次得到所有输出沿“一个输入方向”的方向导数。
            输入少、输出多时用 JVP 更划算，例如计算方向导数、Hessian-向量积（对梯度再做一次 JVP，即 forward-over-reverse）、某些隐式微分与科学计算场景；PyTorch 的 \`torch.func.jvp\` / \`jacfwd\` 提供这种能力。
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
      id: 'qa-linear',
      freq: 3,
      q: md`推导全连接层 $Y = XW + b$ 的反向传播。训练一个 token 为什么是 $6N$ FLOPs？`,
      a: md`
        **结论**：$\mathrm{d}X = \mathrm{d}YW^\top$，$\mathrm{d}W = X^\top\mathrm{d}Y$，$\mathrm{d}b = \sum_{\text{batch}}\mathrm{d}Y$。

        **推导**：$Y_{nj} = \sum_i X_{ni}W_{ij} + b_j$，所以 $\partial L/\partial W_{ij} = \sum_n \mathrm{d}Y_{nj}X_{ni}$，写成矩阵即 $X^\top\mathrm{d}Y$；$\mathrm{d}X$ 同理。也可以用形状对齐直接写出。

        **6N**：每个参数在前向中参与 1 次乘加（2 FLOPs/token）；反向要算 $\mathrm{d}X$ 与 $\mathrm{d}W$ 两个同样大小的矩阵乘（4 FLOPs/token）。合计 $6N$ FLOPs/token，训练 $D$ 个 token 约 $6ND$。开启激活重计算后要再加一次前向，约 $8ND$。
      `,
    },
    {
      id: 'qa-autodiff',
      freq: 2,
      q: md`反向传播和自动微分是什么关系？PyTorch 的 autograd 是怎么工作的？`,
      a: md`
        **结论**：反向传播就是反向模式自动微分在神经网络上的应用。

        **机制**：前向时每个运算创建一个 \`grad_fn\` 节点，记录输入与反向所需的缓存，形成 DAG；\`loss.backward()\` 从 loss 开始按逆拓扑序调用每个节点的 backward（计算 VJP），把梯度累加到叶子张量的 \`.grad\` 上（所以要 \`zero_grad\`）。

        **为什么是反向模式**：标量输出、海量参数，反向模式一次遍历得到全部梯度，代价约为前向的 2 倍。

        **延伸**：缓存的中间值就是激活显存；in-place 修改被缓存的张量会被版本计数器检测并报错；\`torch.no_grad()\` 不建图，推理时省显存。
      `,
    },
    {
      id: 'qa-gradcheck',
      freq: 2,
      q: md`手写了一个算子的反向，怎么验证它是对的？`,
      a: md`
        **结论**：用 float64 做中心差分数值梯度检查，比较相对误差。

        **做法**：构造随机输入与随机上游梯度 $G$，令 $L = \sum(\mathrm{op}(x)\odot G)$，则解析梯度就是你的 backward 输出；数值梯度用 $(L(x+he_i) - L(x-he_i))/2h$，$h \approx 10^{-5}$～$10^{-6}$。相对误差 $\lesssim 10^{-7}$ 说明正确，$> 10^{-3}$ 基本有 bug。

        **注意**：float32 的舍入误差会让检查失效；关闭 dropout 等随机性；不可导点附近（ReLU 的 0、max 并列）会有假阳性；参数多时随机抽查部分坐标。PyTorch 提供 \`torch.autograd.gradcheck\` 自动完成这件事。
      `,
    },
    {
      id: 'qa-act-mem',
      freq: 2,
      q: md`训练时激活为什么占那么多显存？激活重计算是怎么回事？`,
      a: md`
        **结论**：反向需要前向的中间结果，这些激活从前向一直保留到对应层反向完成，总量随 batch、序列长度、隐藏维和层数线性增长，长序列时常常超过参数与优化器状态。

        **重计算**：前向时只保存每层（或每个 block）的输入，丢弃内部激活；反向到这一层时先重新跑一遍前向恢复激活，再做反向。代价是额外约一次前向计算（总训练计算约多 1/3），换来激活显存从 $O(L)$ 份降到接近 1 份 block 内激活 + $L$ 个 block 输入。

        **延伸**：选择性重计算（只重算注意力 softmax 等“显存大、计算少”的部分，Megatron 的 selective recompute）、FlashAttention 本身就在反向时重算注意力矩阵而不保存它。
      `,
    },
  ],

  cheatsheet: md`
    | 前向 | 反向（VJP） |
    |---|---|
    | $Y = XW + b$ | $\mathrm{d}X = \mathrm{d}YW^\top$，$\mathrm{d}W = X^\top\mathrm{d}Y$，$\mathrm{d}b = \sum_{\text{batch}}\mathrm{d}Y$ |
    | ReLU | $\mathrm{d}x = \mathrm{d}y\odot[x>0]$ |
    | softmax | $\mathrm{d}z = p\odot(\mathrm{d}p - \langle\mathrm{d}p,p\rangle)$ |
    | softmax + CE（mean） | $\mathrm{d}z = (p - \mathrm{onehot})/N$ |
    | 广播 ⇔ 求和 | 前向广播的维度，反向求和 |

    \`\`\`python
    def numerical_grad(f, x, h=1e-5):          # float64！
        g = np.zeros_like(x)
        for i in np.ndindex(*x.shape):
            old = x[i]; x[i] = old + h; fp = f(x)
            x[i] = old - h; fm = f(x); x[i] = old
            g[i] = (fp - fm) / (2 * h)
        return g
    \`\`\`

    - 形状对齐写公式；偏置梯度对 batch **求和**
    - 扇出的梯度相加；逆拓扑序执行
    - 训练 ≈ $6N$ FLOPs/token（前向 2N + 反向 4N）
    - 相对误差 < 1e-7 正确，> 1e-3 有 bug
  `,
})
