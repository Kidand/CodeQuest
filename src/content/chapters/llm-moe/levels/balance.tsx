/**
 * llm-moe-balance：负载均衡与容量
 * 事实来源：Switch Transformer (arXiv 2101.03961) §2.2 / §2.4；ST-MoE (arXiv 2202.08906) §3；
 *          HuggingFace transformers Mixtral load_balancing_loss_func；DeepSeek-V3 技术报告 (arXiv 2412.19437)；
 *          V-MoE (Batch Prioritized Routing)、MegaBlocks（dropless MoE）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { CollapseViz } from '../viz/CollapseViz'
import { AuxLossViz } from '../viz/AuxLossViz'
import { ZLossViz } from '../viz/ZLossViz'
import { CapacityViz } from '../viz/CapacityViz'

const LOSS_SETUP = py`
  import numpy as np

  def _ref_lse(x):
      m = x.max(axis=-1, keepdims=True)
      return (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))[:, 0]

  def _ref_parts(logits, k):
      T, E = logits.shape
      lse = _ref_lse(logits)
      p = np.exp(logits - lse[:, None])
      idx = np.argsort(-p, axis=-1, kind="stable")[:, :k]
      f = np.bincount(idx.ravel(), minlength=E) / (T * k)
      return lse, p, f

  def _ref_losses(logits, k, alpha, zc):
      T, E = logits.shape
      lse, p, f = _ref_parts(logits, k)
      aux = alpha * E * np.sum(f * p.mean(0))
      z = zc * np.mean(lse ** 2)
      g_aux = (alpha * E / T) * p * (f[None, :] - (p * f[None, :]).sum(-1, keepdims=True))
      g_z = zc * (2.0 / T) * lse[:, None] * p
      return aux, z, g_aux + g_z

  def _frozen_total(logits, f, alpha, zc):
      """f 冻结时的 aux + z（用于数值梯度）"""
      T, E = logits.shape
      lse = _ref_lse(logits)
      p = np.exp(logits - lse[:, None])
      return alpha * E * np.sum(f * p.mean(0)) + zc * np.mean(lse ** 2)

  def _num_grad(logits, k, alpha, zc, eps=1e-6):
      _, _, f = _ref_parts(logits, k)
      g = np.zeros_like(logits)
      for i in range(logits.shape[0]):
          for j in range(logits.shape[1]):
              a = logits.copy(); a[i, j] += eps
              b = logits.copy(); b[i, j] -= eps
              g[i, j] = (_frozen_total(a, f, alpha, zc) - _frozen_total(b, f, alpha, zc)) / (2 * eps)
      return g
`

const CAP_SETUP = py`
  import math
  import numpy as np

  def _ref_dispatch(topk_idx, num_experts, capacity_factor, scores=None):
      topk_idx = np.asarray(topk_idx)
      T, k = topk_idx.shape
      C = math.ceil(k * T * capacity_factor / num_experts)
      if scores is None:
          order = list(range(T))
      else:
          order = sorted(range(T), key=lambda t: (-scores[t], t))
      fill = [0] * num_experts
      slot = -np.ones((T, k), dtype=int)
      for r in range(k):
          for t in order:
              e = int(topk_idx[t, r])
              if fill[e] < C:
                  slot[t, r] = fill[e]
                  fill[e] += 1
      return slot, C

  def _rand_topk(rng, T, E, k, skew=1.0):
      logits = rng.normal(size=(T, E)) + skew * np.linspace(1, -1, E)
      return np.argsort(-logits, axis=-1)[:, :k], logits.max(-1)
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'collapse',
      title: '专家坍缩：为什么 MoE 必须做负载均衡',
      body: md`
        router 是和专家一起训练的，这形成一个**正反馈**：

        1. 某个专家因为初始化碰巧被多选了一些 token；
        2. 它得到更多梯度更新，变得比别的专家「更好用」；
        3. 主损失的梯度推动 router 更多地选它 → 回到 1。

        最终大部分 token 挤进少数几个专家，其余专家几乎拿不到数据、也就得不到训练——**路由坍缩（routing collapse）**。后果有三：
        - **容量浪费**：总参数里大部分专家等于没用，MoE 退化成一个小稠密模型；
        - **系统瓶颈**：专家并行时，过载专家所在的卡成为木桶的短板，其它卡空等；
        - **token 丢弃**：有容量限制时，过载专家装不下的 token 直接被丢（本关第 4 节）。

        下图是一个确定性的玩具模拟（top-1、8 个专家）：「富者愈富」用两条规则建模——router 被推向当前选中的专家，专家拿到的数据越多就越强。切换辅助损失系数 α 看负载演化。
      `,
      viz: CollapseViz,
      after: md`
        > [!NOTE] 模拟的边界
        > 这是为了展示机制的玩具模型，α 的数值不能直接类比真实模型（Switch Transformer 用 $\alpha = 10^{-2}$）。真实训练中坍缩的速度取决于学习率、初始化、数据分布等，但「不加约束就会集中」这一结论是普遍的。
      `,
      checkpoint: {
        id: 'cp-collapse',
        type: 'single',
        prompt: md`下面哪一项**不是**路由坍缩带来的问题？`,
        options: [
          '大部分专家得不到训练，实际可用容量远小于总参数',
          '专家并行时过载的卡拖慢整个 step',
          '每个 token 的 FLOPs 成倍增加',
          '有容量限制时大量 token 被丢弃',
        ],
        answer: 2,
        explanation: md`每个 token 仍然只经过 $k$ 个专家，FLOPs 不变；变的是 FLOPs **在专家之间的分布**。A、B、D 都是坍缩的直接后果。`,
      },
    },
    {
      id: 'auxloss',
      title: 'Switch 辅助损失：为什么是 f·P',
      body: md`
        Switch Transformer 的可导负载均衡损失（$N$ 个专家、batch 内 $T$ 个 token、top-1）：

        $$
        \mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i,\qquad
        f_i = \frac{1}{T}\sum_{x} \mathbb{1}\{\arg\max p(x) = i\},\qquad
        P_i = \frac{1}{T}\sum_{x} p_i(x)
        $$

        - $f_i$：**实际**分到专家 $i$ 的 token 比例——由 argmax 计数得到，**不可导**；
        - $P_i$：分给专家 $i$ 的平均路由概率——**可导**；
        - 原文：「P 向量可导，f 向量不可导」；$\alpha = 10^{-2}$；完全均匀时取最小值，此时 $f_i = P_i = 1/N$，$\mathcal{L}_{\text{aux}} = \alpha$。

        **为什么要乘 f？** 把 $f$ 当常数求导：$\partial \mathcal{L}/\partial P_i = \alpha N f_i$——**负载越高的专家，它的概率被压得越狠**。$f$ 是「哪里过载」的测量，$P$ 是「能动的旋钮」。
        如果只用 $P$（比如惩罚 $N\sum P_i^2$），router 可以把每个 token 的分布都压得很平、却让 argmax 全落在同一个专家上：$P$ 看起来均匀，实际负载完全坍缩。

        **系数的口径不止一种**：
        - Switch（top-1）：上式，均衡时值为 $\alpha$；
        - HuggingFace Mixtral 实现：\`tokens_per_expert\` 是 top-k 派发的比例（每 token 计 k 次，和为 $k$），均衡时损失为 $k$（再乘 \`router_aux_loss_coef\`）；
        - DeepSeek 序列级损失：$f_i$ 乘 $\frac{N_r}{K_r T}$，均衡时 $\sum f_i P_i = 1$。
      `,
      viz: AuxLossViz,
      after: md`
        > [!WARN] 均衡损失不是免费的
        > 它的梯度和语言模型损失的梯度方向不一致：$\alpha$ 太小压不住坍缩，太大则强迫 router 忽视 token 的真实偏好、损害效果——这就是 DeepSeek 提出「无辅助损失」均衡的动机（第三关）。
      `,
      checkpoint: {
        id: 'cp-aux',
        type: 'judge',
        prompt: md`对任意的 $f$ 和 $P$（各自非负、和为 1），都有 $N\sum_i f_i P_i \ge 1$，所以 Switch 辅助损失的最小值一定在均匀分布处取到。`,
        answer: false,
        explanation: md`错误。若 $f$ 与 $P$ 「错开」，例如 $N=2$、$f=(1,0)$、$P=(0,1)$，则 $N\sum f_iP_i = 0$。但这种情况在真实路由中不会出现：$f$ 是对 $p$ 取 argmax 计数得到的，二者正相关。当 $f = P$ 时由柯西不等式 $N\sum P_i^2 \ge (\sum P_i)^2 = 1$，等号在均匀时取到。`,
      },
    },
    {
      id: 'zloss',
      title: 'Router z-loss 与路由数值稳定',
      body: md`
        ST-MoE 发现 MoE 训练不稳定的一大来源是 router：logits 进入指数函数，而 bf16 只有 8 位有效精度，**logits 绝对值越大舍入误差越大**，经过 exp 被放大。它提出 router z-loss：

        $$
        \mathcal{L}_z(x) = \frac{1}{B}\sum_{i=1}^{B}\Big(\log \sum_{j=1}^{N} e^{x_j^{(i)}}\Big)^2,\qquad
        \mathcal{L} = \mathcal{L}_{\text{CE}} + c_B\,\mathcal{L}_B + c_z\,\mathcal{L}_z
        $$

        其中 $B$ 是 token 数、$N$ 是专家数、$x \in \mathbb{R}^{B\times N}$ 是 router logits；ST-MoE 取 $c_z = 0.001$，并报告它稳定了训练且没有损失质量（甚至略有提升）。

        - 它惩罚的是 $\log Z$（logsumexp）偏离 0：softmax 对整体平移不变，平移本身不影响概率，却影响舍入误差——z-loss 把 logits 的整体水平拉回 0 附近；
        - 梯度 $\partial \mathcal{L}_z / \partial x_j^{(i)} = \frac{2}{B}\,\log Z^{(i)}\cdot p_j^{(i)}$：概率大的 logit 被压得多，所以它也抑制了过大的 logit 幅度；
        - 论文指出**裁剪 logits 不能替代** z-loss：裁剪发生在舍入之后，误差已经产生。
        - 配套的工程做法：Switch 的 **selective precision**——只在 router 函数内部把输入转为 float32。
      `,
      viz: ZLossViz,
      after: md`
        > [!TIP] 读图
        > 把「整体偏移」拖到 0 附近：两组概率几乎重合。拖到 ±80：logits 只能取 0.5 的整数倍，原本 0.25 的差距被舍入成 0 或 0.5，概率误差可达几个百分点——在 top-k 选择附近，这足以让路由来回翻转。
      `,
      checkpoint: {
        id: 'cp-z',
        type: 'single',
        prompt: md`router z-loss $\big(\log\sum_j e^{x_j}\big)^2$ 对 logit $x_j$ 的梯度是：`,
        options: ['$2\,x_j$', '$2\,\log Z \cdot p_j$（$p = \operatorname{softmax}(x)$）', '$2\,p_j(1-p_j)$', '$0$，因为 softmax 对平移不变'],
        answer: 1,
        explanation: md`链式法则：$\frac{\partial (\log Z)^2}{\partial x_j} = 2\log Z \cdot \frac{\partial \log Z}{\partial x_j} = 2\log Z\cdot p_j$。D 错：softmax 的**输出**对平移不变，但 $\log Z$ 不是，z-loss 正是针对 $\log Z$。`,
      },
    },
    {
      id: 'capacity',
      title: 'Capacity factor 与 token 丢弃',
      body: md`
        TPU / 静态图实现需要固定形状，于是给每个专家一个固定大小的缓冲区（Switch Transformer）：

        $$
        \text{expert capacity} = \frac{\text{tokens per batch}}{\text{number of experts}} \times \text{capacity factor}
        $$

        top-$k$ 时常写成 $C = \lceil k\,T\cdot \mathrm{CF} / E \rceil$。超出容量的派发被**丢弃**：这个 token 在该专家处不计算，Switch 原文的做法是让它的表示**经残差连接直接传到下一层**。

        - **CF 小**（如 1.0）：负载稍有不均就丢 token，影响效果；
        - **CF 大**（如 2.0）：空槽也要参与计算与通信（padding），浪费算力和显存；
        - **排队顺序**：GShard 式实现先处理所有 token 的第 1 选择、再处理第 2 选择；同一轮内默认按序列位置，**靠后的 token 更容易被丢**。V-MoE 的 Batch Prioritized Routing 改为按 router 分数排队，优先保留最自信的 token。

        **dropless MoE**：MegaBlocks 用块稀疏矩阵乘（grouped GEMM）处理变长的专家 batch，不再需要容量上限；DeepSeek-V3 依靠均衡，训练与推理都**不丢 token**。
      `,
      viz: CapacityViz,
      after: md`
        > [!INTERVIEW] 加分点：推理时丢 token 的隐患
        > 有容量上限时，一个 token 会不会被丢，取决于**同一个 batch 里还有谁**。于是同一个请求在不同负载下可能得到不同输出——批次不变性被破坏。这是推理侧倾向 dropless 实现的重要原因。
      `,
      checkpoint: {
        id: 'cp-cap',
        type: 'numeric',
        prompt: md`一个 batch 有 2048 个 token，32 个专家，top-1 路由，capacity factor = 1.5。每个专家的容量是多少个 token？`,
        answer: 96,
        tolerance: 0,
        unit: '个',
        explanation: md`$C = 2048 / 32 \times 1.5 = 64 \times 1.5 = 96$。总槽位 $32 \times 96 = 3072$，是 token 数的 1.5 倍：即使完全均衡，也有三分之一的槽位是空的 padding。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-f-diff',
      type: 'judge',
      difficulty: 1,
      prompt: md`Switch 辅助损失中的 $f_i$（分到专家 $i$ 的 token 比例）对 router 参数是可导的，梯度主要通过它传递。`,
      answer: false,
      explanation: md`错误。$f_i$ 由 argmax / top-k 计数得到，是分段常数，不可导。梯度只经 $P_i$（平均路由概率）传递：$\partial\mathcal{L}/\partial P_i = \alpha N f_i$，$f$ 在这里起「按实际负载加权」的作用。`,
    },
    {
      id: 'q-aux-val',
      type: 'numeric',
      difficulty: 2,
      prompt: md`$N = 4$ 个专家，$f = [0.5, 0.25, 0.25, 0]$，$P = [0.4, 0.3, 0.2, 0.1]$，$\alpha = 0.01$。按 Switch 公式 $\alpha N \sum_i f_i P_i$ 计算辅助损失。`,
      answer: 0.013,
      tolerance: 0.001,
      explanation: md`$\sum f_iP_i = 0.2 + 0.075 + 0.05 + 0 = 0.325$；乘 $N=4$ 得 1.3；乘 $\alpha$ 得 0.013。完全均衡时为 $\alpha = 0.01$，所以这个 batch 的不均衡让损失高了 30%。`,
    },
    {
      id: 'q-cap',
      type: 'numeric',
      difficulty: 2,
      prompt: md`$T = 4096$ 个 token，$E = 64$ 个专家，top-2，capacity factor 1.25，按 $C = \lceil kT\cdot\mathrm{CF}/E\rceil$ 计算每个专家的容量。`,
      answer: 160,
      tolerance: 0,
      unit: '个',
      explanation: md`$2 \times 4096 \times 1.25 / 64 = 160$。注意 top-2 时总派发数是 $2T$，忘记乘 $k$ 会得到 80，导致大量丢弃。`,
    },
    {
      id: 'q-collapse',
      type: 'single',
      difficulty: 1,
      prompt: md`「路由坍缩」形成正反馈的根本原因是：`,
      options: [
        'router 用了 softmax 而不是 sigmoid',
        '被选得多的专家得到更多训练、变得更好，router 又因此更倾向于选它',
        '专家数太多，router 无法区分',
        'top-k 的 k 太大',
      ],
      answer: 1,
      explanation: md`router 与专家**共同**训练，选择与能力互相强化。A：打分函数换成 sigmoid 也会坍缩（所以 DeepSeek-V3 仍需 bias 均衡）；C、D 不是主因，k 大反而让负载更分散。`,
    },
    {
      id: 'q-z-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 ST-MoE 的 router z-loss，正确的是（多选）：`,
      options: [
        '它惩罚每个 token 的 router logsumexp 的平方',
        'ST-MoE 选用的系数 $c_z = 0.001$',
        '它的作用是减小进入指数函数的数值的舍入误差，提升训练稳定性',
        '它可以完全替代负载均衡损失',
        '直接裁剪 router logits 与 z-loss 效果等价',
      ],
      answer: [0, 1, 2],
      explanation: md`A、B、C 均为原文内容。D 错：ST-MoE 的总损失同时包含均衡损失 $c_B\mathcal{L}_B$ 与 $c_z\mathcal{L}_z$，二者目标不同。E 错：论文指出裁剪发生在舍入之后，不能消除已经产生的舍入误差。`,
    },
    {
      id: 'q-fill-coef',
      type: 'fill',
      difficulty: 1,
      prompt: md`Switch Transformer 辅助损失的系数 $\alpha$ = ____(1)____；ST-MoE 的 z-loss 系数 $c_z$ = ____(2)____；DeepSeek-V3 序列级均衡损失的 $\alpha$ = ____(3)____。`,
      blanks: [
        { label: '(1)', answers: ['0.01'], numeric: true, tolerance: 0 },
        { label: '(2)', answers: ['0.001'], numeric: true, tolerance: 0 },
        { label: '(3)', answers: ['0.0001'], numeric: true, tolerance: 0 },
      ],
      explanation: md`$10^{-2}$、$10^{-3}$、$10^{-4}$。DeepSeek 的系数比 Switch 小两个数量级，因为它的主力均衡手段是 bias，序列级损失只用来兜底。`,
    },
    {
      id: 'q-order-cap',
      type: 'order',
      difficulty: 2,
      prompt: md`把带容量限制的 top-2 分发（GShard 式）的步骤按正确顺序排列：`,
      items: [
        '计算容量 $C = \\lceil kT\\cdot\\mathrm{CF}/E \\rceil$，每个专家的已用槽位清零',
        '按 token 顺序处理所有 token 的第 1 选择，有空槽就占位，否则丢弃',
        '再按同样顺序处理所有 token 的第 2 选择',
        '每个专家对自己缓冲区里的 token 计算 FFN（空槽为 padding）',
        '把专家输出按门控权重加回各 token；被丢弃的派发贡献为 0（token 经残差直通）',
      ],
      explanation: md`先处理所有第 1 选择再处理第 2 选择，保证每个 token「最想去」的专家优先被满足。丢弃的派发不计算、不贡献输出，token 至少还有残差连接。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把技术与它要解决的问题配对：`,
      pairs: [
        ['Switch 辅助损失 $N\\sum f_iP_i$', '负载不均与路由坍缩'],
        ['router z-loss', 'router logits 过大导致的舍入误差与训练不稳'],
        ['selective precision（router 用 fp32）', 'bf16 下 router 计算精度不足'],
        ['capacity factor', '静态形状下每个专家缓冲区的大小'],
        ['Batch Prioritized Routing', '容量不足时优先保留高置信度 token'],
      ],
      explanation: md`前三项针对「均衡」与「稳定」，后两项针对「容量」。注意 z-loss 与 selective precision 解决的是同一类问题（数值精度）的两面：一个让数值变小，一个让精度变高。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 Switch 辅助损失（top-1）有 **2 处** bug，点出有问题的行：`,
      code: py`
        def switch_aux_loss(logits, alpha=0.01):
            T, E = logits.shape
            z = logits - logits.max(-1, keepdims=True)
            probs = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
            top1 = probs.argmax(-1)
            f = np.bincount(top1) / T
            P = probs.mean(axis=1)
            return alpha * E * np.sum(f * P)
      `,
      bugLines: [6, 7],
      explanation: md`第 6 行缺 \`minlength=E\`：若编号最大的几个专家没分到 token，\`bincount\` 的长度小于 $E$，与 \`P\` 的形状不匹配（广播报错或错位）。第 7 行应对 token 维（\`axis=0\`）求平均得到 $(E,)$ 的 $P$；\`axis=1\` 得到的是每个 token 概率之和（全为 1）的平均，形状还是 $(T,)$。`,
    },
    {
      id: 'q-pick-drop',
      type: 'pick',
      difficulty: 2,
      prompt: md`8 个 token、2 个专家、top-1，capacity factor 0.75（$C = \lceil 8 \times 0.75 / 2\rceil = 3$），按位置排队。各 token 的目标专家如下。点出**被丢弃**的 token：`,
      items: ['t0→E0', 't1→E1', 't2→E0', 't3→E0', 't4→E0', 't5→E1', 't6→E0', 't7→E1'],
      cols: 4,
      answer: [4, 6],
      explanation: md`E0 依次收到 t0、t2、t3，三个槽位已满，t4、t6 被丢；E1 收到 t1、t5、t7，恰好装满。被丢的 t4、t6 只能经残差连接直通到下一层。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出数值稳定的 router z-loss（\`logits\` 形状 $(T, E)$；有 2 行干扰项）：`,
      lines: [
        'def router_z_loss(logits, z_coef=1e-3):',
        '    m = logits.max(axis=-1, keepdims=True)',
        '    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))',
        '    return z_coef * np.mean(lse ** 2)',
      ],
      distractors: ['    lse = np.log(np.exp(logits).sum(axis=-1))', '    return z_coef * np.mean(lse) ** 2'],
      explanation: md`干扰项 1 在 logits 很大时 \`exp\` 溢出成 inf——而 z-loss 恰恰在 logits 大时最重要；干扰项 2 先平均再平方，正负 $\log Z$ 会相互抵消，不再是逐 token 的惩罚。`,
    },
    {
      id: 'q-hf',
      type: 'single',
      difficulty: 3,
      prompt: md`HuggingFace transformers 的 Mixtral \`load_balancing_loss_func\` 中，\`tokens_per_expert\` 是 top-k 派发次数按 token 数归一化（每个 token 计 $k$ 次），\`router_prob_per_expert\` 是平均 softmax 概率，返回 \`sum(tokens_per_expert * router_prob_per_expert) * num_experts\`。Mixtral（$k=2$）在**完全均衡**时这个函数返回多少（乘 \`router_aux_loss_coef\` 之前）？`,
      options: ['1', '2', '8', '0.25'],
      answer: 1,
      explanation: md`完全均衡时每个专家的 \`tokens_per_expert\` $= k/E = 2/8$，平均概率 $= 1/8$；求和 $= 8 \times \frac{2}{8}\cdot\frac{1}{8} = \frac{1}{4}$；乘 \`num_experts\` $=8$ 得 2，即 $k$。Switch 原式（top-1）的最小值是 1；DeepSeek 的序列级损失通过 $\frac{N_r}{K_rT}$ 归一化后最小值也是 1。不同实现的「基线值」不同，比较 loss 曲线时要注意。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-losses',
      title: '实现 Switch 辅助损失 + router z-loss',
      intro: md`
        实现 \`router_losses(logits, k=1, alpha=0.01, z_coef=0.001)\`，返回 \`(aux, z)\`：
        - $f_i$ = 专家 $i$ 被 top-k 选中的次数 / $(T\cdot k)$（和为 1）；$P_i$ = 平均 softmax 概率；
        - \`aux\` $= \alpha E \sum_i f_i P_i$；\`z\` $= c_z \cdot \text{mean}_t(\operatorname{lse}_t^2)$；
        - logsumexp 必须数值稳定（测试里有 $10^3$ 量级的 logits）。
      `,
      template: py`
        import numpy as np

        def router_losses(logits, k=1, alpha=0.01, z_coef=0.001):
            T, E = logits.shape
            m = logits.max(axis=-1, keepdims=True)
            lse = {{1}}                                      # (T,) 稳定的 logsumexp
            probs = np.exp(logits - lse[:, None])            # softmax
            idx = np.argsort(-probs, axis=-1)[:, :k]
            f = {{2}}                                        # (E,) 实际派发比例
            P = {{3}}                                        # (E,) 平均路由概率
            aux = {{4}}
            z = {{5}}
            return aux, z
      `,
      steps: [
        {
          blank: 1,
          instruction: md`用减最大值的技巧计算每个 token 的 logsumexp，结果形状 $(T,)$。`,
          answer: 'm[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))',
          accept: ['m.squeeze(-1) + np.log(np.exp(logits - m).sum(axis=-1))', 'm[:, 0] + np.log(np.sum(np.exp(logits - m), axis=-1))'],
          explain: md`$\operatorname{lse}(x) = m + \log\sum_j e^{x_j - m}$。\`m\` 是 $(T,1)$，要取 \`[:, 0]\` 变成 $(T,)$。有了 lse，softmax 就是 \`exp(logits - lse)\`，一举两得。`,
        },
        {
          blank: 2,
          instruction: md`统计每个专家被选中的次数，除以总派发数 $T\cdot k$。`,
          answer: 'np.bincount(idx.ravel(), minlength=E) / (T * k)',
          accept: ['np.bincount(idx.reshape(-1), minlength=E) / (T * k)', 'np.bincount(idx.flatten(), minlength=E) / (T * k)'],
          explain: md`这是不可导的计数。除以 $Tk$ 让 $\sum f_i = 1$，于是完全均衡时 $f_i = P_i = 1/E$，\`aux\` $= \alpha$。`,
        },
        {
          blank: 3,
          instruction: md`对 token 维求平均，得到每个专家的平均路由概率。`,
          answer: 'probs.mean(axis=0)',
          accept: ['np.mean(probs, axis=0)', 'probs.sum(axis=0) / T'],
          explain: md`轴别写错：\`axis=0\` 是对 token 平均。$P$ 是梯度的唯一通路。`,
        },
        {
          blank: 4,
          instruction: md`组合成 Switch 辅助损失。`,
          answer: 'alpha * E * np.sum(f * P)',
          accept: ['alpha * E * (f * P).sum()', 'alpha * E * np.dot(f, P)', 'alpha * E * f @ P'],
          explain: md`乘 $E$ 让均衡时的值与专家数无关（恒为 $\alpha$）。在 PyTorch 里 $f$ 要么来自 \`argmax\`（天然无梯度），要么显式 \`.detach()\`。`,
        },
        {
          blank: 5,
          instruction: md`router z-loss：每个 token 的 lse 平方后再平均，乘系数。`,
          answer: 'z_coef * np.mean(lse ** 2)',
          accept: ['z_coef * (lse ** 2).mean()', 'z_coef * np.mean(np.square(lse))'],
          explain: md`先平方再平均。先平均再平方会让正负 lse 抵消，失去逐 token 惩罚的意义。`,
        },
      ],
      setup: py`
        def _g_ref(logits, k=1, alpha=0.01, zc=0.001):
            T, E = logits.shape
            m = logits.max(-1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logits - m).sum(-1))
            p = np.exp(logits - lse[:, None])
            idx = np.argsort(-p, -1, kind="stable")[:, :k]
            f = np.bincount(idx.ravel(), minlength=E) / (T * k)
            return alpha * E * np.sum(f * p.mean(0)), zc * np.mean(lse ** 2)
      `,
      tests: [
        {
          name: '完全均衡时 aux = α',
          code: py`
            logits = np.array([[3.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 3.0, 0], [0, 0, 0, 3.0]])
            aux, z = router_losses(logits)
            assert abs(aux - 0.01) < 1e-12, f"完全均衡时 aux 应为 alpha=0.01，得到 {aux}"
            ea, ez = _g_ref(logits)
            assert abs(z - ez) < 1e-12, f"z 期望 {ez:.6f}，得到 {z}"
          `,
        },
        {
          name: '随机 + top-2',
          code: py`
            r = np.random.default_rng(0)
            logits = r.normal(size=(16, 6)) * 2
            got = router_losses(logits, k=2, alpha=0.02, z_coef=0.005)
            exp = _g_ref(logits, 2, 0.02, 0.005)
            assert np.allclose(got, exp), f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '大 logits 数值稳定',
          hidden: true,
          code: py`
            logits = np.array([[1000.0, 999.0, 0.0], [5.0, 1200.0, 3.0]])
            aux, z = router_losses(logits)
            assert np.isfinite(aux) and np.isfinite(z), f"出现 inf/nan：aux={aux}, z={z}"
            ea, ez = _g_ref(logits)
            assert np.isclose(aux, ea) and np.isclose(z, ez), f"期望 {(ea, ez)}，得到 {(aux, z)}"
          `,
        },
      ],
      outro: md`
        训练时总损失 = LM loss + \`aux\` + \`z\`（每个 MoE 层各算一份再求和或平均）。实战第一题会要求你手推这两项对 logits 的梯度——面试里「f 为什么不可导、梯度怎么走」的标准追问。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-aux-grad',
      title: '辅助损失与 z-loss：前向 + 手推梯度',
      difficulty: 'hard',
      source: '大模型岗手撕 · 负载均衡',
      timeLimitMin: 30,
      description: md`
        实现 router 的两个正则项及其对 logits 的梯度：

        \`\`\`python
        def router_aux_z_loss(logits: np.ndarray, k: int, alpha: float, z_coef: float):
            """返回 (aux, z, grad)，grad 形状与 logits 相同 (T, E)"""
        \`\`\`

        - $p = \operatorname{softmax}(\text{logits})$；每个 token 取 top-$k$；$f_i = \text{count}_i / (Tk)$；$P_i = \frac1T\sum_t p_{t,i}$
        - \`aux\` $= \alpha E \sum_i f_i P_i$；\`z\` $= c_z \cdot \frac{1}{T}\sum_t \operatorname{lse}_t^2$，其中 $\operatorname{lse}_t = \log\sum_j e^{\text{logits}_{t,j}}$
        - \`grad\` $= \partial(\text{aux} + \text{z}) / \partial\,\text{logits}$，**把 $f$ 当作常数**（它来自不可导的计数）

        **要求**：logsumexp / softmax 数值稳定（隐藏测试有 $10^3$ 量级的 logits）；不允许用数值差分（会检查源码中是否出现 \`eps\`）。测试数据中 top-k 没有并列。
      `,
      starter: py`
        import numpy as np

        def router_aux_z_loss(logits, k, alpha, z_coef):
            # 前向：lse、softmax、top-k 计数 f、平均概率 P → aux、z
            # 反向：softmax 雅可比 + 链式法则（f 为常数）
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def router_aux_z_loss(logits, k, alpha, z_coef):
            T, E = logits.shape
            m = logits.max(axis=-1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))     # (T,)
            p = np.exp(logits - lse[:, None])                            # (T, E) softmax
            idx = np.argsort(-p, axis=-1)[:, :k]
            f = np.bincount(idx.ravel(), minlength=E) / (T * k)          # 常数（不可导）
            P = p.mean(axis=0)

            aux = alpha * E * np.sum(f * P)
            z = z_coef * np.mean(lse ** 2)

            # aux：dL/dP_i = αE f_i；dP_i/dx_tj = p_ti(δ_ij − p_tj)/T
            #   ⇒ dL/dx_tj = (αE/T) · p_tj · (f_j − Σ_i f_i p_ti)
            g_aux = (alpha * E / T) * p * (f[None, :] - (p @ f)[:, None])
            # z：d(lse_t²)/dx_tj = 2·lse_t·p_tj，再除以 T
            g_z = z_coef * (2.0 / T) * lse[:, None] * p
            return aux, z, g_aux + g_z
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，梯度交给 autograd）',
          code: py`
            import torch
            import torch.nn.functional as F

            def router_aux_z_loss(logits: torch.Tensor, k: int, alpha: float, z_coef: float):
                T, E = logits.shape
                logits = logits.float()                                  # router 损失用 fp32
                probs = logits.softmax(dim=-1)
                idx = probs.topk(k, dim=-1).indices
                f = F.one_hot(idx, E).sum(dim=(0, 1)).float() / (T * k)  # 计数，天然无梯度
                P = probs.mean(dim=0)
                aux = alpha * E * (f * P).sum()
                z = z_coef * torch.logsumexp(logits, dim=-1).pow(2).mean()
                return aux, z

            # loss = lm_loss + aux + z; loss.backward()
          `,
        },
      ],
      explanation: md`
        **前向**：一次稳定的 logsumexp 同时得到 softmax（\`exp(logits − lse)\`）和 z-loss 需要的 $\operatorname{lse}$。

        **aux 的梯度**：$f$ 为常数，$\frac{\partial \mathcal{L}}{\partial P_i} = \alpha E f_i$；softmax 雅可比 $\frac{\partial p_{ti}}{\partial x_{tj}} = p_{ti}(\delta_{ij} - p_{tj})$，所以
        $$\frac{\partial \mathcal{L}_{\text{aux}}}{\partial x_{tj}} = \frac{\alpha E}{T}\sum_i f_i\,p_{ti}(\delta_{ij} - p_{tj}) = \frac{\alpha E}{T}\,p_{tj}\Big(f_j - \sum_i f_i p_{ti}\Big)$$
        直觉：若专家 $j$ 的负载 $f_j$ 高于该 token 视角下的「平均负载」$\sum_i f_ip_{ti}$，梯度为正，下降一步就会压低 $x_{tj}$。

        **z 的梯度**：$\frac{\partial}{\partial x_{tj}}\frac{c_z}{T}\operatorname{lse}_t^2 = \frac{2c_z}{T}\operatorname{lse}_t\,p_{tj}$。

        **复杂度**：$O(TE\log E)$（排序），其余 $O(TE)$。**易错点**：忘记 $1/T$；把 $f$ 也求导；\`p @ f\` 写成 \`f @ p\`；朴素 \`exp\` 溢出。
      `,
      complexity: { time: 'O(T·E·log E)', space: 'O(T·E)' },
      hints: [
        md`先写前向：\`lse = m + log(sum(exp(logits - m)))\`，\`p = exp(logits - lse)\`。`,
        md`aux 对 $P_i$ 的偏导是 $\alpha E f_i$；对 $x_{tj}$ 再乘 softmax 雅可比并除以 $T$。`,
        md`向量化：\`(alpha * E / T) * p * (f[None, :] - (p @ f)[:, None])\`；z 项：\`z_coef * 2 / T * lse[:, None] * p\`。`,
      ],
      setup: LOSS_SETUP,
      tests: [
        {
          name: '前向：均衡时 aux = α',
          code: py`
            logits = np.eye(4) * 4.0
            aux, z, g = router_aux_z_loss(logits, 1, 0.01, 0.001)
            assert abs(aux - 0.01) < 1e-12, f"完全均衡时 aux 应为 0.01，得到 {aux}"
            ea, ez, eg = _ref_losses(logits, 1, 0.01, 0.001)
            assert abs(z - ez) < 1e-12, f"z 期望 {ez}，得到 {z}"
            assert np.shape(g) == (4, 4), f"grad 形状应为 (4, 4)，得到 {np.shape(g)}"
          `,
        },
        {
          name: '梯度与数值差分一致（f 冻结）',
          code: py`
            r = np.random.default_rng(1)
            logits = r.normal(size=(6, 5)) * 1.5
            aux, z, g = router_aux_z_loss(logits, 2, 0.05, 0.01)
            ng = _num_grad(logits, 2, 0.05, 0.01)
            err = np.abs(np.asarray(g) - ng).max()
            assert err < 1e-7, f"梯度最大误差 {err:.3e}（是不是把 f 也求导了，或漏了 1/T？）"
          `,
        },
        {
          name: '不使用数值差分',
          code: py`
            import inspect
            src = inspect.getsource(router_aux_z_loss)
            assert "eps" not in src, "请手推解析梯度，不要用数值差分"
          `,
        },
        {
          name: '随机对拍（含 top-1/2/3）',
          hidden: true,
          code: py`
            r = np.random.default_rng(2)
            for _ in range(30):
                T, E = r.integers(2, 20), r.integers(3, 10)
                k = int(r.integers(1, min(3, E - 1) + 1))
                logits = r.normal(size=(T, E)) * 2
                got = router_aux_z_loss(logits, k, 0.01, 0.001)
                exp = _ref_losses(logits, k, 0.01, 0.001)
                assert np.isclose(got[0], exp[0]) and np.isclose(got[1], exp[1]), f"前向不一致：期望 {exp[:2]}，得到 {got[:2]}"
                assert np.allclose(got[2], exp[2], atol=1e-12), f"梯度最大误差 {np.abs(np.asarray(got[2]) - exp[2]).max():.3e}"
          `,
        },
        {
          name: '大 logits 数值稳定',
          hidden: true,
          code: py`
            logits = np.array([[1000.0, 998.0, 0.0, -5.0], [3.0, 1500.0, 2.0, 1.0], [-800.0, -790.0, -805.0, -801.0]])
            aux, z, g = router_aux_z_loss(logits, 1, 0.01, 0.001)
            assert np.isfinite(aux) and np.isfinite(z) and np.all(np.isfinite(g)), "出现 inf/nan：logsumexp 需要减最大值"
            ea, ez, eg = _ref_losses(logits, 1, 0.01, 0.001)
            assert np.isclose(z, ez) and np.allclose(g, eg), f"z 期望 {ez:.4f}，得到 {z:.4f}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-sign',
          q: md`aux 梯度 $\frac{\alpha E}{T}p_{tj}(f_j - \sum_i f_ip_{ti})$ 对所有 $j$ 求和等于多少？说明了什么？`,
          a: md`$\sum_j p_{tj}f_j - \sum_j p_{tj}\sum_i f_ip_{ti} = \bar f_t - \bar f_t = 0$。梯度对每个 token 的 logits 之和为 0——它只在专家之间「重新分配」概率，不改变 logits 的整体水平（这正是 softmax 平移不变性的体现）；调整整体水平的是 z-loss。`,
        },
      ],
    },
    {
      id: 'c-capacity',
      title: '按容量分发：计算每个派发的槽位',
      difficulty: 'hard',
      source: '大模型岗手撕 · GShard / Switch 分发',
      timeLimitMin: 25,
      description: md`
        实现带容量上限的 MoE 分发（GShard / Switch 风格）：

        \`\`\`python
        def capacity_dispatch(topk_idx, num_experts, capacity_factor, scores=None):
            """返回 (slot, C)：slot 形状 (T, k)，slot[t, r] 为第 t 个 token 的第 r 个选择在专家缓冲区中的位置，被丢弃为 -1"""
        \`\`\`

        - \`topk_idx\`：$(T, k)$ 整数数组，第 $r$ 列是每个 token 的第 $r$ 选择；
        - 容量 \`C = math.ceil(k * T * capacity_factor / num_experts)\`（就按这个 Python 表达式算）；
        - **处理顺序**：先处理所有 token 的第 0 列，再第 1 列……；同一列内，\`scores is None\` 时按 token 下标升序，否则按 \`scores\`（形状 $(T,)$）**降序**、分数相同按下标升序（Batch Prioritized Routing）；
        - 每个专家的槽位从 0 开始依次分配，满 $C$ 个后，后来的派发被丢弃（\`-1\`）。

        **示例**：\`topk_idx=[[0],[1],[0],[0],[0]]\`，\`num_experts=2\`，\`capacity_factor=1.0\` → $C = \lceil 5/2 \rceil = 3$，\`slot=[[0],[0],[1],[2],[-1]]\`。
      `,
      starter: py`
        import math
        import numpy as np

        def capacity_dispatch(topk_idx, num_experts, capacity_factor, scores=None):
            topk_idx = np.asarray(topk_idx)
            T, k = topk_idx.shape
            C = math.ceil(k * T * capacity_factor / num_experts)
            # TODO：按「列优先 + 队列顺序」分配槽位
            return -np.ones((T, k), dtype=int), C
      `,
      solution: py`
        import math
        import numpy as np

        def capacity_dispatch(topk_idx, num_experts, capacity_factor, scores=None):
            topk_idx = np.asarray(topk_idx)
            T, k = topk_idx.shape
            C = math.ceil(k * T * capacity_factor / num_experts)
            if scores is None:
                order = np.arange(T)
            else:
                order = np.lexsort((np.arange(T), -np.asarray(scores)))   # 分数降序，平局按下标
            slot = -np.ones((T, k), dtype=int)
            fill = np.zeros(num_experts, dtype=int)                       # 每个专家已占用的槽位
            for r in range(k):                                            # 先满足所有 token 的第 1 选择
                e = topk_idx[order, r]                                    # 这一轮按队列顺序的目标专家
                onehot = np.eye(num_experts, dtype=int)[e]                # (T, E)
                pos = fill[e] + (np.cumsum(onehot, axis=0)[np.arange(T), e] - 1)   # 队列中的名次 + 已占用
                keep = pos < C
                slot[order[keep], r] = pos[keep]
                fill += onehot.sum(axis=0)
                fill = np.minimum(fill, C)
            return slot, C
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：GShard 式 one-hot + cumsum，按位置排队）',
          code: py`
            import math
            import torch
            import torch.nn.functional as F

            def capacity_dispatch(topk_idx: torch.Tensor, num_experts: int, capacity_factor: float):
                T, k = topk_idx.shape
                C = math.ceil(k * T * capacity_factor / num_experts)
                slot = torch.full((T, k), -1, dtype=torch.long)
                filled = torch.zeros(num_experts, dtype=torch.long)
                for r in range(k):
                    mask = F.one_hot(topk_idx[:, r], num_experts)              # (T, E)
                    pos = torch.cumsum(mask, dim=0) - 1 + filled                # 每个派发在专家内的位置
                    pos = (pos * mask).sum(-1)                                  # 取出自己那一列
                    keep = pos < C
                    slot[keep, r] = pos[keep]
                    filled = torch.clamp(filled + mask.sum(0), max=C)
                return slot, C
          `,
        },
      ],
      explanation: md`
        **思路**：逐列（第 1 选择、第 2 选择……）处理，每列内按队列顺序给每个派发编号。某个派发在专家 $e$ 中的位置 = 该专家已占用的槽位数 + 本列中排在它前面、也去 $e$ 的派发数。后者用 one-hot 的**前缀和**一次性算出（GShard 的经典写法），位置 $\ge C$ 的派发丢弃。

        **为什么要列优先**：保证每个 token 的「最想去」的专家先被满足；若按 token 逐个处理它的 k 个选择，前面 token 的第 2 选择可能挤掉后面 token 的第 1 选择。

        **复杂度**：$O(kTE)$（one-hot 写法）或 $O(kT)$（逐个计数）。

        **易错点**：容量忘记乘 $k$；\`fill\` 累加时没有截断到 $C$；按分数排序时用了升序；分数排序后写回 \`slot\` 时没有映射回原 token 下标。
      `,
      complexity: { time: 'O(k·T·E)', space: 'O(T·E)' },
      hints: [
        md`最直接的写法：两层循环 \`for r in range(k): for t in order:\`，维护每个专家的已用槽位数 \`fill[e]\`。`,
        md`\`order\`：\`scores is None\` 时是 \`range(T)\`；否则按 \`(-score, t)\` 排序。`,
        md`向量化：本列的 one-hot \`(T, E)\` 做 \`cumsum(axis=0)\`，取每行自己那一列再减 1，加上之前已占用的 \`fill[e]\`，就是槽位。`,
      ],
      setup: CAP_SETUP,
      tests: [
        {
          name: '示例',
          code: py`
            slot, C = capacity_dispatch(np.array([[0], [1], [0], [0], [0]]), 2, 1.0)
            assert C == 3, f"容量应为 3，得到 {C}"
            assert np.asarray(slot).tolist() == [[0], [0], [1], [2], [-1]], f"期望 [[0],[0],[1],[2],[-1]]，得到 {np.asarray(slot).tolist()}"
          `,
        },
        {
          name: 'top-2：先满足所有第 1 选择',
          code: py`
            idx = np.array([[0, 1], [0, 1], [1, 0], [0, 1]])
            slot, C = capacity_dispatch(idx, 2, 0.75)          # C = ceil(2*4*0.75/2) = 3
            exp, _ = _ref_dispatch(idx, 2, 0.75)
            assert C == 3, f"容量应为 3，得到 {C}"
            assert np.asarray(slot).tolist() == exp.tolist(), f"期望 {exp.tolist()}，得到 {np.asarray(slot).tolist()}"
          `,
        },
        {
          name: '按分数排队（BPR）',
          code: py`
            idx = np.array([[0], [0], [0], [1]])
            scores = np.array([0.3, 0.9, 0.5, 0.8])
            slot, C = capacity_dispatch(idx, 2, 1.0, scores=scores)   # C = 2
            assert np.asarray(slot).tolist() == [[-1], [0], [1], [0]], f"期望 [[-1],[0],[1],[0]]，得到 {np.asarray(slot).tolist()}"
          `,
        },
        {
          name: '随机对拍（位置 / 分数两种排队）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(3)
            for _ in range(60):
                T, E, k = int(rng.integers(1, 40)), int(rng.integers(2, 9)), int(rng.integers(1, 3))
                k = min(k, E)
                idx, sc = _rand_topk(rng, T, E, k, skew=float(rng.uniform(0, 2)))
                cf = float(rng.choice([0.5, 0.75, 1.0, 1.25, 2.0]))
                for s in (None, sc):
                    got, C = capacity_dispatch(idx, E, cf, scores=s)
                    exp, EC = _ref_dispatch(idx, E, cf, scores=s)
                    assert C == EC, f"容量期望 {EC}，得到 {C}"
                    assert np.array_equal(np.asarray(got), exp), f"T={T},E={E},k={k},cf={cf}：\n期望 {exp.tolist()}\n得到 {np.asarray(got).tolist()}"
          `,
        },
        {
          name: '槽位合法性 + 规模',
          hidden: true,
          code: py`
            rng = np.random.default_rng(4)
            idx, sc = _rand_topk(rng, 4000, 16, 2, skew=1.5)
            slot, C = capacity_dispatch(idx, 16, 1.0, scores=sc)
            slot = np.asarray(slot)
            for e in range(16):
                s = np.sort(slot[(idx == e) & (slot >= 0)])
                assert s.tolist() == list(range(len(s))), f"专家 {e} 的槽位应为 0..n-1 且不重复"
                assert len(s) <= C, f"专家 {e} 超出容量"
            exp, _ = _ref_dispatch(idx, 16, 1.0, scores=sc)
            assert np.array_equal(slot, exp), "大规模结果与参考实现不一致"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-dropless',
          q: md`如果完全不想丢 token，有哪些做法？各自代价是什么？`,
          a: md`① 把 capacity factor 设得很大（≥ E/k 时绝不丢）：大量空槽 padding，显存与计算浪费；② dropless 实现（MegaBlocks 的块稀疏 / grouped GEMM，按专家排序后变长计算）：需要专门的 kernel，形状动态；③ 靠均衡（DeepSeek-V3 的 bias 均衡 + 部署时冗余专家）让负载本身接近均匀，即使有容量也几乎不触发丢弃。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-aux',
      freq: 3,
      q: md`讲一下 MoE 的负载均衡损失。为什么是 $f\cdot P$ 而不是只用 $P$？`,
      a: md`
        **结论**：Switch 的 $\mathcal{L}_{\text{aux}} = \alpha N\sum_i f_iP_i$，$f_i$ 是专家 $i$ 实际分到的 token 比例，$P_i$ 是平均路由概率，$\alpha = 10^{-2}$，均匀时取最小值 $\alpha$。

        **原因**：$f$ 来自 argmax 计数，不可导；$P$ 可导但只看 $P$ 测不到真实负载——router 可以让每个 token 的分布很平、argmax 却全落在同一个专家上。乘积形式下 $\partial\mathcal{L}/\partial P_i = \alpha N f_i$：用不可导的 $f$ 指出「谁过载」，用可导的 $P$ 去「调」。

        **延伸**：top-k 实现中 $f$ 的归一化各不相同（HF Mixtral 均衡时为 $k$，DeepSeek 序列级损失乘 $N_r/(K_rT)$ 后为 1）；$\alpha$ 太大会干扰主损失，DeepSeek 因此改用无辅助损失的 bias 均衡，只保留 $10^{-4}$ 的序列级损失兜底。
      `,
    },
    {
      id: 'qa-zloss',
      freq: 2,
      q: md`什么是 router z-loss？为什么 MoE 需要它？`,
      a: md`
        **结论**：$\mathcal{L}_z = \frac1B\sum_i(\log\sum_j e^{x_j^{(i)}})^2$，惩罚 router logits 的 logsumexp 偏离 0，ST-MoE 取 $c_z = 0.001$。

        **原因**：router logits 进入 exp，bf16 只有 8 位有效精度，绝对值越大舍入误差越大，经指数放大后导致路由抖动、loss spike。z-loss 让 logits 整体保持小数值，ST-MoE 报告它稳定训练且不损失质量；裁剪 logits 不能替代，因为舍入已经发生。

        **延伸**：配套做法是 router 内部用 fp32（Switch 的 selective precision）。注意 z-loss 与均衡损失目标不同，二者通常同时使用。
      `,
    },
    {
      id: 'qa-capacity',
      freq: 2,
      q: md`capacity factor 怎么选？token 被丢弃后会发生什么？`,
      a: md`
        **结论**：容量 $C = \lceil kT\cdot\mathrm{CF}/E\rceil$；CF 小省显存和计算但丢 token，CF 大浪费 padding。训练常用 1.0–1.25，评估/推理常设得更大或不设上限。

        **原因**：被丢弃的派发在该专家处不计算，token 只经残差连接传到下一层（top-2 时可能只丢一个选择）。排队顺序决定谁被丢：按位置时序列末尾的 token 吃亏，Batch Prioritized Routing 按 router 分数优先保留自信的 token。

        **延伸**：推理时有容量上限会让输出依赖同 batch 的其它请求，破坏批次不变性；MegaBlocks 等 dropless 实现和 DeepSeek-V3（训练推理都不丢 token）代表了现在的主流方向。
      `,
    },
    {
      id: 'qa-collapse',
      freq: 2,
      q: md`什么是专家坍缩？除了辅助损失还有什么办法？`,
      a: md`
        **结论**：router 与专家共同训练形成正反馈，负载集中到少数专家，其余专家得不到训练。

        **办法**：① 辅助损失（Switch / GShard 的 $f\cdot P$）；② 噪声门控（Shazeer 2017 的 noisy top-k，给 logits 加可学习噪声鼓励探索）；③ 容量上限（硬性限制单专家负载）；④ 无辅助损失的 bias 调度（DeepSeek-V3）；⑤ 更好的初始化与 router 数值稳定（z-loss、fp32 router）；⑥ 前几层用固定的 Hash 路由（DeepSeek-V4，路由与学习无关，天然不会坍缩）。

        **延伸**：负载均衡与专家专精存在张力：均衡粒度越细（如序列级）约束越强，专精越难，所以 DeepSeek-V3 主要做 batch 级均衡。
      `,
    },
  ],

  cheatsheet: md`
    **Switch aux loss**：$\alpha N\sum_i f_iP_i$，$\alpha=10^{-2}$，均匀时 $=\alpha$；$\partial/\partial P_i = \alpha N f_i$
    **z-loss**：$\frac1B\sum(\operatorname{lse})^2$，$c_z=10^{-3}$；梯度 $\frac{2}{B}\operatorname{lse}\cdot p$
    \`\`\`python
    m = logits.max(-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(-1))
    p = np.exp(logits - lse[:, None])
    idx = np.argsort(-p, -1)[:, :k]
    f = np.bincount(idx.ravel(), minlength=E) / (T * k)   # 不可导
    aux = alpha * E * np.sum(f * p.mean(0))
    z = z_coef * np.mean(lse ** 2)
    g_aux = alpha * E / T * p * (f - (p @ f)[:, None])
    \`\`\`

    - 容量 $C = \lceil kT\cdot\mathrm{CF}/E\rceil$；溢出 → 残差直通；先第 1 选择再第 2 选择
    - HF Mixtral aux 均衡值 = $k$；DeepSeek 序列级 = 1（乘 $N_r/(K_rT)$）
    - router 用 fp32（selective precision）；裁剪 logits ≠ z-loss
    - dropless：MegaBlocks、DeepSeek-V3（不丢 token）
  `,
})
