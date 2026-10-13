/**
 * llm-moe-boss：BOSS · MoE 综合
 * 综合实战：① 完整 DeepSeekMoE 层（共享专家 + 分组受限路由 + 序列级均衡损失）；② 排序式分发（permute / unpermute）。
 */
import { defineLevel, md, py } from '@/content/helpers'

const LAYER_SETUP = py`
  import numpy as np

  def _swiglu(h, W):
      Wg, Wu, Wd = W
      a = h @ Wg
      return ((a / (1.0 + np.exp(-a))) * (h @ Wu)) @ Wd

  def _ref_layer(x, W_r, bias, shared, routed, k, n_group, topk_group, route_scale, alpha):
      T, d = x.shape
      E = W_r.shape[1]
      s = 1.0 / (1.0 + np.exp(-(x @ W_r)))
      choice = s + bias
      per = k // topk_group
      g = choice.reshape(T, n_group, E // n_group)
      gscore = -np.sort(-g, -1)[:, :, :per].sum(-1)
      keep = np.argsort(-gscore, -1, kind="stable")[:, :topk_group]
      gm = np.zeros((T, n_group), dtype=bool)
      np.put_along_axis(gm, keep, True, -1)
      masked = np.where(np.repeat(gm, E // n_group, 1), choice, -np.inf)
      idx = np.argsort(-masked, -1, kind="stable")[:, :k]
      w = np.take_along_axis(s, idx, -1)
      w = w / w.sum(-1, keepdims=True) * route_scale
      y = np.zeros_like(x)
      for W in shared:
          y = y + _swiglu(x, W)
      for t in range(T):
          for j in range(k):
              y[t] += w[t, j] * _swiglu(x[t:t + 1], routed[idx[t, j]])[0]
      load = np.bincount(idx.ravel(), minlength=E)
      f = E / (k * T) * load
      P = (s / s.sum(-1, keepdims=True)).mean(0)
      return y, load, alpha * np.sum(f * P)

  def _make(seed, T=12, d=6, E=8, f=8, n_shared=1):
      r = np.random.default_rng(seed)
      mk = lambda: (r.normal(size=(d, f)) / np.sqrt(d), r.normal(size=(d, f)) / np.sqrt(d), r.normal(size=(f, d)) / np.sqrt(f))
      return r.normal(size=(T, d)), r.normal(size=(d, E)), r.normal(scale=0.1, size=E), [mk() for _ in range(n_shared)], [mk() for _ in range(E)]

  def _cmp(got, exp, tag=""):
      assert isinstance(got, tuple) and len(got) == 3, f"{tag}应返回 (y, load, seq_aux)"
      y, load, aux = got
      assert np.shape(y) == exp[0].shape, f"{tag}y 形状应为 {exp[0].shape}，得到 {np.shape(y)}"
      assert np.asarray(load).tolist() == exp[1].tolist(), f"{tag}负载期望 {exp[1].tolist()}，得到 {np.asarray(load).tolist()}"
      assert np.allclose(y, exp[0], atol=1e-8), f"{tag}输出最大误差 {np.abs(np.asarray(y) - exp[0]).max():.3e}"
      assert np.isclose(aux, exp[2]), f"{tag}seq_aux 期望 {exp[2]:.6f}，得到 {aux}"
`

const PERM_SETUP = py`
  import numpy as np

  class _Rec:
      """记录每次调用的输入；输出 tanh(h @ W)"""
      def __init__(self, W):
          self.W = W
          self.inputs = []
      def __call__(self, h):
          h = np.asarray(h)
          if h.shape[0] > 0:
              self.inputs.append(np.array(h, copy=True))
          return np.tanh(h @ self.W)

  def _ref_perm(x, idx, w, experts):
      T, k = idx.shape
      flat = idx.ravel()
      order = np.argsort(flat, kind="stable")
      y = np.zeros_like(x)
      for t in range(T):
          for j in range(k):
              y[t] += w[t, j] * np.tanh(x[t] @ experts[idx[t, j]].W)
      return y, order // k, np.bincount(flat, minlength=len(experts))

  def _pcase(seed, T=10, d=5, E=6, k=2, skew=0.0):
      r = np.random.default_rng(seed)
      logits = r.normal(size=(T, E)) + skew * np.linspace(1, -1, E)
      idx = np.argsort(-logits, axis=-1)[:, :k]
      w = r.uniform(0.1, 1.0, size=(T, k))
      return r.normal(size=(T, d)), idx, w, [_Rec(r.normal(size=(d, d))) for _ in range(E)]
`

export default defineLevel({
  /* ================================================================ */
  /* 复盘                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'review',
      title: '复盘：一个 MoE 层的完整生命周期',
      body: md`
        把本章串起来，一个现代 MoE 层（以 DeepSeek 系为例）从输入到输出要经过：

        | 阶段 | 做什么 | 关键细节 | 出处 |
        |---|---|---|---|
        | 打分 | $s = \sigma(xW_r)$（V3）/ $\sqrt{\operatorname{softplus}}$（V4）/ softmax（Mixtral） | router 用 fp32；z-loss 抑制大 logits | 第 1、2、4 关 |
        | 选择 | top-k on $s + b$；V3 先按节点受限；V4 前 3 层查 \`tid2eid\` | bias 只管选择 | 第 3、4 关 |
        | 权重 | 原始 $s$ 在 top-k 内归一化 × \`routed_scaling_factor\` | k=1 时归一化会切断 router 梯度 | 第 1、3 关 |
        | 分发 | 按专家循环 / 带容量 / 排序 + grouped GEMM | 展平 scatter 要用 \`add.at\` | 第 1、2 关、本关 |
        | 计算 | 每个专家一次批量 SwiGLU；共享专家全量计算 | 小专家 GEMM 效率低 | 第 1、3 关 |
        | 合并 | 加权 scatter-add 回原 token，加共享专家输出 | 被丢弃的派发经残差直通 | 第 1、2 关 |
        | 均衡 | aux loss（Switch）/ bias（V3）/ QB（K3）+ 小 α 序列级损失 | 新 bias 只用于下一步 | 第 2、3、4 关 |

        **三组必背数字**
        - DeepSeek-V3：256 路由 + 1 共享，top-8，节点 ≤ 4，专家中间维 2048，$\gamma = 0.001$，序列级 $\alpha = 10^{-4}$，671B / 37B；
        - DeepSeek-V4：Flash 256+1 / Pro 384+1，均 top-6，前 3 个 MoE 层 Hash 路由，Sqrt(Softplus)，284B/13B 与 1.6T/49B；
        - Kimi K3：896 选 16 + 2 共享，latent 维 0.5d，Quantile Balancing，2.8T / 104B。

        > [!INTERVIEW] 压轴题的答题框架
        > 「手撕 MoE」时先写最朴素的按专家循环版本，跑通后主动提出三个升级点：① 换成 DeepSeek 式 sigmoid + bias 选择；② 加负载均衡（aux loss 或 bias 更新，并说明为什么 $f$ 不可导）；③ 生产实现用排序 + grouped GEMM、专家并行 all-to-all。能把这三点讲清楚，基本就是这道题的满分答案。
      `,
    },
  ],

  /* ================================================================ */
  /* 综合练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'b-k1',
      type: 'single',
      difficulty: 2,
      prompt: md`你实现了一个 top-1 MoE，并按 Mixtral 的习惯在 top-k 内重归一化门控权重。训练时发现 router 参数几乎不更新。最可能的原因是：`,
      options: [
        '学习率太小',
        'top-1 重归一化后门控权重恒为 1，主损失对 router 没有梯度',
        'softmax 没有减最大值',
        '没有加 z-loss',
      ],
      answer: 1,
      explanation: md`k=1 时 $w = p_i/p_i = 1$，与 router 参数无关。Switch Transformer 因此直接用 $p_i \cdot E_i(x)$；若坚持重归一化，router 只能从辅助损失学习。A、C、D 都不会让梯度恰好为 0。`,
    },
    {
      id: 'b-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`下列说法正确的是（多选）：`,
      options: [
        'MoE 节省的是每 token 计算量，而不是显存',
        'Switch 辅助损失中梯度只经过平均路由概率 P 传递',
        'DeepSeek-V3 的 bias 均衡会改变被选中专家的门控权重',
        'Kimi K3 的路由专家在 0.5d 维的 latent 空间中计算',
        'DeepSeek-V4 的 Hash 路由层仍用 router 分数计算门控权重',
      ],
      answer: [0, 1, 3, 4],
      explanation: md`C 错：bias 只影响 top-k 选择，权重用原始亲和度归一化。其余均正确：MoE 全部专家需常驻显存；$f$ 不可导；K3 的 $\ell = 3584 = 0.5 \times 7168$；V4 hash 层的权重从 \`sqrt(softplus)\` 分数 gather。`,
    },
    {
      id: 'b-num-v3',
      type: 'numeric',
      difficulty: 2,
      prompt: md`DeepSeek-V3 的一个 MoE 层（$d = 7168$，专家中间维 2048，SwiGLU，8 个路由专家 + 1 个共享专家被激活），每个 token 激活的专家参数是多少百万（M，$10^6$）？`,
      answer: 396.361728,
      tolerance: 0.005,
      unit: 'M',
      explanation: md`单个专家 $3 \times 7168 \times 2048 = 44{,}040{,}192$；激活 9 个：$396{,}361{,}728 \approx 396.4\text{M}$。乘以 58 个 MoE 层约 23.0B，占 V3 每 token 37B 激活参数的六成左右。`,
    },
    {
      id: 'b-num-grad',
      type: 'numeric',
      difficulty: 3,
      prompt: md`Switch 辅助损失 $\mathcal{L} = \alpha N\sum_i f_iP_i$，$\alpha = 0.01$，$N = 8$。某专家的实际派发比例 $f_i = 0.25$（均衡时应为 0.125）。$\partial\mathcal{L}/\partial P_i$ 等于多少？`,
      answer: 0.02,
      tolerance: 0.001,
      explanation: md`把 $f$ 当常数：$\partial\mathcal{L}/\partial P_i = \alpha N f_i = 0.01 \times 8 \times 0.25 = 0.02$，是均衡专家（$0.01$）的两倍——负载越高，压低其概率的力度越大。`,
    },
    {
      id: 'b-drop',
      type: 'judge',
      difficulty: 1,
      prompt: md`DeepSeek-V3 训练时设置了 capacity factor，超出容量的 token 会被丢弃并经残差直通。`,
      answer: false,
      explanation: md`错误。V3 技术报告明确说明：凭借有效的负载均衡，V3 训练时不丢弃任何 token，推理时也通过部署策略保证不丢。容量与 token 丢弃是 GShard / Switch 时代的做法。`,
    },
    {
      id: 'b-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`DeepSeek-V3 的 bias 更新速度 $\gamma$ = ____(1)____；DeepSeek-V4 在前 ____(2)____ 个 MoE 层使用 Hash 路由；Kimi K3 每 token 从 ____(3)____ 个路由专家中激活 16 个。`,
      blanks: [
        { label: '(1)', answers: ['0.001'], numeric: true, tolerance: 0 },
        { label: '(2)', answers: ['3'], numeric: true, tolerance: 0 },
        { label: '(3)', answers: ['896'], numeric: true, tolerance: 0 },
      ],
      explanation: md`V3：$\gamma = 0.001$（前 14.3T token），之后为 0；V4：\`num_hash_layers: 3\`；K3：896 选 16，另有 2 个共享专家。`,
    },
    {
      id: 'b-order',
      type: 'order',
      difficulty: 2,
      prompt: md`按时间先后排列这些 MoE 工作 / 模型：`,
      items: ['Switch Transformer（top-1 + 辅助损失）', 'ST-MoE（router z-loss）', 'Mixtral 8×7B', 'DeepSeek-V3（无辅助损失均衡）', 'DeepSeek-V4（Hash 路由、Sqrt(Softplus)）', 'Kimi K3（Quantile Balancing）'],
      explanation: md`Switch（arXiv 2021-01）→ ST-MoE（2022-02）→ Mixtral（2023-12 发布）→ DeepSeek-V3（2024-12）→ DeepSeek-V4（2026）→ Kimi K3（2026-07）。`,
    },
    {
      id: 'b-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把问题与解决它的技术配对：`,
      pairs: [
        ['路由坍缩', 'Switch 辅助损失 $N\\sum f_iP_i$'],
        ['bf16 下 router logits 舍入误差', 'router z-loss'],
        ['辅助损失梯度干扰主任务', 'bias 只参与选择的无辅助损失均衡'],
        ['跨节点 IB 通信量', '节点受限路由（M = 4）'],
        ['浅层学习路由难以训练', 'Hash 路由（按 token id 查表）'],
        ['近千专家时 sign 规则步长难调', 'Quantile Balancing'],
      ],
      explanation: md`六个问题与六种技术一一对应，基本覆盖了本章的主线。`,
    },
    {
      id: 'b-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 MoE 层把派发对展平后按专家处理（\`idx\`、\`w\` 形状 $(T,k)$），有 **2 处** bug。点出有问题的行：`,
      code: py`
        def moe_layer(x, W_r, bias, shared_ffn, experts, k):
            s = 1 / (1 + np.exp(-(x @ W_r)))
            idx = np.argsort(-(s + bias), axis=-1)[:, :k]
            w = np.take_along_axis(s, idx, axis=-1)
            w = w / w.sum(axis=-1, keepdims=True)
            y = shared_ffn(x)
            flat_e = idx.ravel()
            flat_w = w.ravel()
            flat_t = np.tile(np.arange(x.shape[0]), k)
            for e in range(len(experts)):
                m = flat_e == e
                out = experts[e](x[flat_t[m]])
                y[flat_t] += flat_w[m][:, None] * out
            return y
      `,
      bugLines: [9, 13],
      explanation: md`第 9 行：\`idx.ravel()\` 是行优先展开（$t_0$ 的 k 个、$t_1$ 的 k 个…），对应的 token 下标应为 \`np.repeat(np.arange(T), k)\`；\`np.tile\` 得到 $[0,1,\dots,T-1,0,1,\dots]$，错位。第 13 行应为 \`y[flat_t[m]] += ...\`（只写回本专家的 token；同一专家内无重复下标，花式 \`+=\` 安全）。`,
    },
    {
      id: 'b-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`下列机制中，哪些**只影响「选哪些专家」、不改变被选中专家的门控权重计算**？`,
      items: ['DeepSeek-V3 的均衡 bias', 'DeepSeek-V4 前 3 层的 Hash 路由', 'Kimi K3 Quantile Balancing 的 bias', 'Switch 辅助损失', 'routed_scaling_factor', 'router z-loss'],
      cols: 3,
      answer: [0, 1, 2],
      explanation: md`前三者只改选择，权重仍由原始 router 分数归一化得到。辅助损失和 z-loss 通过梯度改变 router 参数，从而同时影响选择与权重；\`routed_scaling_factor\` 直接缩放权重。`,
    },
    {
      id: 'b-parsons',
      type: 'parsons',
      difficulty: 3,
      prompt: md`拼出「按专家排序」的分发与合并（\`idx\`、\`w\` 为 $(T,k)$，\`wf = w.ravel()\`，\`y\` 已初始化为 0；有 2 行干扰项）：`,
      lines: [
        'flat = idx.ravel()',
        'order = np.argsort(flat, kind="stable")',
        'perm_tok = order // k',
        'offsets = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=E))])',
        'for e in range(E):',
        '    a, b = offsets[e], offsets[e + 1]',
        '    out = experts[e](x[perm_tok[a:b]])',
        '    np.add.at(y, perm_tok[a:b], wf[order[a:b], None] * out)',
      ],
      distractors: ['perm_tok = order % k', 'offsets = np.cumsum(np.bincount(flat, minlength=E))'],
      explanation: md`展平位置 $p = t\cdot k + r$，所以 token 下标是 \`p // k\`（\`% k\` 得到的是第几选择）。\`offsets\` 必须以 0 开头，专家 $e$ 的段是 \`[offsets[e], offsets[e+1])\`。排序后同一专家的 token 连续，正适合 grouped GEMM。`,
    },
    {
      id: 'b-decode',
      type: 'single',
      difficulty: 3,
      prompt: md`在线服务一个 DeepSeek-V3 规模的 MoE，decode 阶段 batch 很小。下列哪一项最主导每步延迟？`,
      options: [
        'router 的 matmul 计算量',
        '从显存读取被触碰到的专家权重（访存瓶颈），以及专家并行的 all-to-all 通信',
        '负载均衡 bias 的更新',
        'softmax / sigmoid 的计算',
      ],
      answer: 1,
      explanation: md`小 batch decode 的算术强度极低，batch 内 token 会触碰到大量专家，读权重的字节数远大于计算量；跨卡专家并行还要做 dispatch / combine 两次 all-to-all。bias 推理时冻结，A、D 的计算量可忽略。缓解手段：大 batch、低比特专家权重（V4 的 FP4）、通信优化（DeepEP）、投机解码。`,
    },
  ],

  guided: [],

  /* ================================================================ */
  /* 综合实战                                                           */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-dsmoe-layer',
      title: '完整 DeepSeekMoE 层：共享专家 + 分组路由 + 序列级均衡损失',
      difficulty: 'hard',
      source: '大模型岗手撕 · MoE 压轴',
      timeLimitMin: 40,
      description: md`
        对**一条序列**实现 DeepSeek-V3 风格的 MoE 层前向，并返回负载与序列级均衡损失：

        \`\`\`python
        def deepseek_moe_layer(x, W_r, bias, shared, routed, k, n_group, topk_group, route_scale, alpha):
            """返回 (y, load, seq_aux)"""
        \`\`\`

        - \`x\`：$(T, d)$；\`W_r\`：$(d, E)$；\`bias\`：$(E,)$
        - \`shared\` / \`routed\`：专家权重列表，每个元素是 SwiGLU 三元组 \`(Wg, Wu, Wd)\`，形状 $(d,f),(d,f),(f,d)$；FFN 为 $(\operatorname{silu}(hW_g)\odot hW_u)W_d$
        - **路由**（同第三关）：$s = \sigma(xW_r)$；选择分数 $s+b$；组分数 = 组内最大的 \`k // topk_group\` 个之和，保留前 \`topk_group\` 组；组内外屏蔽后取 top-$k$；权重 = 原始 $s$ 在 $k$ 个内归一化 × \`route_scale\`
        - **输出**（不含残差）：$y = \sum_{\text{共享}} \mathrm{FFN}(x) + \sum_{\text{选中}} g\cdot\mathrm{FFN}(x)$；每个路由专家应**批量**处理分给它的 token
        - \`load\`：$(E,)$ 每个专家被选中的次数
        - \`seq_aux\` $= \alpha\sum_i f_iP_i$，$f_i = \frac{E}{kT}\,\text{load}_i$，$P_i = \frac1T\sum_t \frac{s_{t,i}}{\sum_j s_{t,j}}$（在**全部** $E$ 个专家上归一化）

        测试数据中没有并列。
      `,
      starter: py`
        import numpy as np

        def deepseek_moe_layer(x, W_r, bias, shared, routed, k, n_group, topk_group, route_scale, alpha):
            # 1) 路由：sigmoid → s+b → 分组屏蔽 → top-k → 原始 s 归一化 × route_scale
            # 2) 共享专家 + 按专家批量计算路由专家并加权合并
            # 3) load 与序列级均衡损失
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def _ffn(h, W):
            Wg, Wu, Wd = W
            a = h @ Wg
            return ((a / (1.0 + np.exp(-a))) * (h @ Wu)) @ Wd       # SwiGLU

        def deepseek_moe_layer(x, W_r, bias, shared, routed, k, n_group, topk_group, route_scale, alpha):
            T, d = x.shape
            E = W_r.shape[1]
            # ---- 路由 ----
            s = 1.0 / (1.0 + np.exp(-(x @ W_r)))                     # (T, E) 亲和度
            choice = s + bias                                         # 只用于选择
            grp = choice.reshape(T, n_group, E // n_group)
            gscore = -np.sort(-grp, axis=-1)[:, :, : k // topk_group].sum(-1)
            keep = np.argsort(-gscore, axis=-1)[:, :topk_group]
            gmask = np.zeros((T, n_group), dtype=bool)
            np.put_along_axis(gmask, keep, True, axis=-1)
            masked = np.where(np.repeat(gmask, E // n_group, axis=1), choice, -np.inf)
            idx = np.argsort(-masked, axis=-1)[:, :k]                 # (T, k)
            w = np.take_along_axis(s, idx, axis=-1)
            w = w / w.sum(axis=-1, keepdims=True) * route_scale

            # ---- 计算：共享专家全量 + 路由专家按专家批量 ----
            y = np.zeros_like(x)
            for W in shared:
                y += _ffn(x, W)
            for e in range(E):
                tok, slot = np.nonzero(idx == e)
                if tok.size:
                    y[tok] += w[tok, slot][:, None] * _ffn(x[tok], routed[e])

            # ---- 负载与序列级均衡损失 ----
            load = np.bincount(idx.ravel(), minlength=E)
            f = E / (k * T) * load
            P = (s / s.sum(axis=-1, keepdims=True)).mean(axis=0)
            return y, load, alpha * np.sum(f * P)
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：nn.Module 形式）',
          code: py`
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class Expert(nn.Module):
                def __init__(self, d, f):
                    super().__init__()
                    self.w1, self.w3 = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False)
                    self.w2 = nn.Linear(f, d, bias=False)
                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))

            class DeepSeekMoE(nn.Module):
                def __init__(self, d, f, n_routed, n_shared, k, n_group, topk_group, route_scale, alpha):
                    super().__init__()
                    self.k, self.ng, self.tg, self.scale, self.alpha = k, n_group, topk_group, route_scale, alpha
                    self.gate = nn.Linear(d, n_routed, bias=False)
                    self.register_buffer("bias", torch.zeros(n_routed))      # 无辅助损失均衡用，不进优化器
                    self.experts = nn.ModuleList(Expert(d, f) for _ in range(n_routed))
                    self.shared = Expert(d, f * n_shared)                    # 多个共享专家合并成一个宽 FFN

                def forward(self, x):                                         # x: (T, d) 一条序列
                    T, E = x.size(0), len(self.experts)
                    s = self.gate(x).float().sigmoid()
                    choice = s + self.bias
                    g = choice.view(T, self.ng, -1).topk(self.k // self.tg, -1).values.sum(-1)
                    gmask = torch.zeros_like(g).scatter_(1, g.topk(self.tg, -1).indices, 1)
                    choice = choice.masked_fill(gmask.repeat_interleave(E // self.ng, 1) == 0, float("-inf"))
                    idx = choice.topk(self.k, -1).indices
                    w = s.gather(1, idx)
                    w = (w / w.sum(-1, keepdim=True) * self.scale).to(x.dtype)
                    y = self.shared(x)
                    for e in range(E):
                        tok, slot = torch.where(idx == e)
                        if tok.numel():
                            y.index_add_(0, tok, self.experts[e](x[tok]) * w[tok, slot, None])
                    load = torch.bincount(idx.flatten(), minlength=E)
                    f = E / (self.k * T) * load.float()                       # 无梯度
                    P = (s / s.sum(-1, keepdim=True)).mean(0)
                    return y, load, self.alpha * (f * P).sum()
          `,
        },
      ],
      explanation: md`
        **结构**：路由（选择用 $s+b$、分组屏蔽；权重用原始 $s$）→ 共享专家对全部 token 计算 → 路由专家按专家批量计算并加权 scatter-add → 统计负载与序列级损失。

        **序列级损失的两个归一化**：$f_i$ 乘 $\frac{E}{kT}$ 使完全均衡时 $f_i = 1$；$P_i$ 中的 $s'$ 在**全部**专家上归一化（不是只在 top-k 内）。完全均衡时 $\sum f_iP_i = 1$，所以 \`seq_aux\` 的基线是 $\alpha$。

        **复杂度**：路由 $O(TdE + TE\log E)$；专家计算总共 $T\cdot k$ 次路由 FFN + $T \cdot N_s$ 次共享 FFN。

        **易错点**：bias 进入权重；组分数用错；$P$ 用 top-k 内归一化的权重代替全量归一化的 $s'$；$f$ 忘记乘 $E/(kT)$；共享专家乘了门控权重。
      `,
      complexity: { time: 'O(T·d·E + T·k·C_ffn)', space: 'O(T·E + T·d)' },
      hints: [
        md`先复用第三关的分组门控得到 \`idx\`、\`w\`。`,
        md`共享专家：\`y += ffn(x, W)\`（无门控）；路由专家：对每个 \`e\` 用 \`np.nonzero(idx == e)\` 批量计算。`,
        md`\`load = np.bincount(idx.ravel(), minlength=E)\`；\`f = E / (k*T) * load\`；\`P = (s / s.sum(-1, keepdims=True)).mean(0)\`。`,
      ],
      setup: LAYER_SETUP,
      tests: [
        {
          name: '小规模：E=8 分 4 组保留 2 组，top-2，1 个共享专家',
          code: py`
            x, W_r, bias, shared, routed = _make(0)
            got = deepseek_moe_layer(x, W_r, bias, shared, routed, 2, 4, 2, 2.5, 1e-4)
            _cmp(got, _ref_layer(x, W_r, bias, shared, routed, 2, 4, 2, 2.5, 1e-4))
          `,
        },
        {
          name: '负载之和 = T·k，且每 token 至多触达 topk_group 个组',
          code: py`
            x, W_r, bias, shared, routed = _make(1, T=20, E=16)
            y, load, aux = deepseek_moe_layer(x, W_r, bias, shared, routed, 4, 4, 2, 1.0, 1e-3)
            assert int(np.sum(load)) == 80, f"负载之和应为 T·k = 80，得到 {int(np.sum(load))}"
            _cmp((y, load, aux), _ref_layer(x, W_r, bias, shared, routed, 4, 4, 2, 1.0, 1e-3))
          `,
        },
        {
          name: '无共享专家 + 不分组',
          hidden: true,
          code: py`
            x, W_r, bias, shared, routed = _make(2, n_shared=0)
            _cmp(deepseek_moe_layer(x, W_r, bias, [], routed, 3, 1, 1, 1.0, 0.01), _ref_layer(x, W_r, bias, [], routed, 3, 1, 1, 1.0, 0.01), "无共享：")
          `,
        },
        {
          name: '两个共享专家 + 大 bias（选择被改变，权重不受影响）',
          hidden: true,
          code: py`
            x, W_r, bias, shared, routed = _make(3, n_shared=2)
            bias = np.zeros(8)
            bias[5] = 3.0
            got = deepseek_moe_layer(x, W_r, bias, shared, routed, 2, 2, 1, 2.5, 1e-4)
            exp = _ref_layer(x, W_r, bias, shared, routed, 2, 2, 1, 2.5, 1e-4)
            _cmp(got, exp, "大 bias：")
            assert int(np.asarray(got[1])[5]) == 12, "bias 很大的专家 5 应被所有 token 选中"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            for sd in range(10, 22):
                E = [8, 16][sd % 2]
                ng = [2, 4][sd % 2]
                x, W_r, bias, shared, routed = _make(sd, T=6 + sd % 7, E=E)
                k = 2 * (1 + sd % 2)
                tg = 2 if k % 2 == 0 else 1
                _cmp(deepseek_moe_layer(x, W_r, bias, shared, routed, k, ng, tg, 2.5, 1e-4),
                     _ref_layer(x, W_r, bias, shared, routed, k, ng, tg, 2.5, 1e-4), f"seed={sd}：")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-ep',
          q: md`把这个层改成专家并行（EP，8 张卡各放 E/8 个专家），前向要增加哪些通信？`,
          a: md`路由后做一次 **dispatch all-to-all**：每张卡把自己的 token 按目标专家所在的卡分组发送（每个派发对一份隐状态，带上权重或在本地保留权重）；各卡对收到的 token 做本地专家计算；再做一次 **combine all-to-all** 把结果发回 token 所在的卡，按权重 scatter-add。共享专家在本地计算，可与通信重叠。节点受限路由限制的就是 dispatch 跨节点的份数。`,
        },
      ],
    },
    {
      id: 'c-permute',
      title: '排序式分发：permute → grouped 计算 → unpermute',
      difficulty: 'hard',
      source: '大模型岗手撕 · MoE 工程实现',
      timeLimitMin: 30,
      description: md`
        生产级 MoE（MegaBlocks、各类 grouped GEMM kernel）不逐个专家做布尔筛选，而是把 $T\cdot k$ 个派发对**按专家排序**成一个连续缓冲区。实现：

        \`\`\`python
        def moe_permute_forward(x, topk_idx, topk_w, experts):
            """返回 (y, perm_tok, tokens_per_expert)"""
        \`\`\`

        - \`x\`：$(T, d)$；\`topk_idx\`、\`topk_w\`：$(T, k)$；\`experts\`：$E$ 个可调用对象，$(n, d) \to (n, d)$
        - **permute**：把展平后的派发（位置 $p = t\cdot k + r$）按专家 id 做**稳定**排序；\`perm_tok[j]\` 是排序后第 $j$ 个派发对应的 token 下标（长度 $T\cdot k$）
        - \`tokens_per_expert\`：$(E,)$ 每个专家的派发数；它的前缀和给出每个专家在缓冲区中的连续段
        - **grouped 计算**：每个专家**恰好调用一次**（没有派发的专家不调用），输入**必须是** \`x[perm_tok[a:b]]\`（其连续段，顺序一致）
        - **unpermute**：把每个派发的输出乘对应权重，累加回 \`y[t]\`

        隐藏测试会记录每个专家收到的输入，检查它是否恰好等于排序缓冲区中的对应连续段。
      `,
      starter: py`
        import numpy as np

        def moe_permute_forward(x, topk_idx, topk_w, experts):
            # 1) 展平 + 稳定排序  2) 每个专家一个连续段  3) 加权累加回原 token
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def moe_permute_forward(x, topk_idx, topk_w, experts):
            T, k = topk_idx.shape
            E = len(experts)
            flat_e = topk_idx.ravel()                                  # 位置 p = t*k + r
            order = np.argsort(flat_e, kind="stable")                  # 按专家排序（稳定）
            perm_tok = order // k                                      # 排序后每个派发对应的 token
            perm_w = topk_w.ravel()[order]
            tokens_per_expert = np.bincount(flat_e, minlength=E)
            offsets = np.concatenate([[0], np.cumsum(tokens_per_expert)])

            buf = x[perm_tok]                                          # permute：(T*k, d) 连续缓冲区
            out = np.empty_like(buf)
            for e in range(E):                                         # grouped GEMM 的 NumPy 模拟
                a, b = offsets[e], offsets[e + 1]
                if b > a:
                    out[a:b] = experts[e](buf[a:b])

            y = np.zeros_like(x)
            np.add.at(y, perm_tok, perm_w[:, None] * out)              # unpermute：重复下标要用 add.at
            return y, perm_tok, tokens_per_expert
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def moe_permute_forward(x, topk_idx, topk_w, experts):
                T, k = topk_idx.shape
                flat_e = topk_idx.flatten()
                order = torch.argsort(flat_e, stable=True)
                perm_tok = order // k
                tokens_per_expert = torch.bincount(flat_e, minlength=len(experts))
                buf = x[perm_tok]                                      # permute
                chunks = torch.split(buf, tokens_per_expert.tolist())  # 每个专家一个连续段
                out = torch.cat([experts[e](c) for e, c in enumerate(chunks)])   # 实际中是一次 grouped GEMM
                out = out * topk_w.flatten()[order, None]
                y = torch.zeros_like(x).index_add_(0, perm_tok, out)  # unpermute
                return y, perm_tok, tokens_per_expert
          `,
        },
      ],
      explanation: md`
        **三步**：
        1. **permute**：\`order = argsort(topk_idx.ravel(), stable)\`，\`perm_tok = order // k\`，\`buf = x[perm_tok]\`——同一专家的派发在缓冲区里连续，段边界是 \`tokens_per_expert\` 的前缀和；
        2. **grouped 计算**：每个专家处理自己的连续段（GPU 上是一次 grouped GEMM，不需要 padding，也就不需要容量上限——dropless）；
        3. **unpermute**：输出乘权重后按 \`perm_tok\` 累加回原位置。同一 token 有 k 个派发，**必须**用 \`np.add.at\` / \`index_add_\`，\`y[perm_tok] += ...\` 会丢贡献。

        **为什么用稳定排序**：保证同一专家内 token 的顺序确定（按原位置），便于复现与调试；EP 下这个顺序也决定 all-to-all 的打包布局。

        **复杂度**：排序 $O(Tk\log(Tk))$（也可以用计数排序做到 $O(Tk + E)$）；计算与按专家循环相同，但内存访问是连续的。
      `,
      complexity: { time: 'O(T·k·log(T·k) + T·k·C_expert)', space: 'O(T·k·d)' },
      hints: [
        md`\`order = np.argsort(topk_idx.ravel(), kind="stable")\`，token 下标是 \`order // k\`，权重是 \`topk_w.ravel()[order]\`。`,
        md`\`offsets = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=E))])\`，专家 $e$ 的段是 \`[offsets[e], offsets[e+1])\`。`,
        md`合并：\`np.add.at(y, perm_tok, perm_w[:, None] * out)\`。`,
      ],
      setup: PERM_SETUP,
      tests: [
        {
          name: '示例：输出、permutation 与段长',
          code: py`
            x, idx, w, ex = _pcase(0)
            y, perm, tpe = moe_permute_forward(x, idx, w, ex)
            ey, eperm, etpe = _ref_perm(x, idx, w, ex)
            assert np.shape(y) == x.shape, f"y 形状应为 {x.shape}，得到 {np.shape(y)}"
            assert np.allclose(y, ey, atol=1e-10), f"输出最大误差 {np.abs(np.asarray(y) - ey).max():.3e}"
            assert np.asarray(perm).tolist() == eperm.tolist(), f"perm_tok 期望 {eperm.tolist()}，得到 {np.asarray(perm).tolist()}"
            assert np.asarray(tpe).tolist() == etpe.tolist(), f"tokens_per_expert 期望 {etpe.tolist()}，得到 {np.asarray(tpe).tolist()}"
          `,
        },
        {
          name: '每个专家恰好收到排序缓冲区的连续段',
          code: py`
            x, idx, w, ex = _pcase(1, T=16, E=5, k=2)
            y, perm, tpe = moe_permute_forward(x, idx, w, ex)
            _, eperm, etpe = _ref_perm(x, idx, w, ex)
            off = np.concatenate([[0], np.cumsum(etpe)])
            for e, rec in enumerate(ex):
                if etpe[e] == 0:
                    assert len(rec.inputs) == 0, f"专家 {e} 没有派发，不应被调用"
                    continue
                assert len(rec.inputs) == 1, f"专家 {e} 应恰好调用 1 次，实际 {len(rec.inputs)} 次"
                seg = x[eperm[off[e]:off[e + 1]]]
                assert rec.inputs[0].shape == seg.shape and np.array_equal(rec.inputs[0], seg), f"专家 {e} 的输入应为 x[perm_tok[{off[e]}:{off[e + 1]}]]"
          `,
        },
        {
          name: 'k=1 与 k=3',
          hidden: true,
          code: py`
            for k in (1, 3):
                x, idx, w, ex = _pcase(2 + k, T=12, E=6, k=k)
                y, perm, tpe = moe_permute_forward(x, idx, w, ex)
                ey, eperm, etpe = _ref_perm(x, idx, w, ex)
                assert np.allclose(y, ey), f"k={k}：输出最大误差 {np.abs(np.asarray(y) - ey).max():.3e}"
                assert np.asarray(perm).tolist() == eperm.tolist(), f"k={k}：perm_tok 不对"
          `,
        },
        {
          name: '有空闲专家的偏斜路由',
          hidden: true,
          code: py`
            x, idx, w, ex = _pcase(7, T=20, E=8, k=2, skew=4.0)
            y, perm, tpe = moe_permute_forward(x, idx, w, ex)
            ey, eperm, etpe = _ref_perm(x, idx, w, ex)
            assert (etpe == 0).any(), "测试数据应包含空闲专家"
            assert np.allclose(y, ey), f"输出最大误差 {np.abs(np.asarray(y) - ey).max():.3e}（同一 token 的多个贡献是否都累加了？）"
            for e in np.flatnonzero(etpe == 0):
                assert len(ex[e].inputs) == 0, f"空闲专家 {e} 不应被调用"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            for sd in range(20, 35):
                x, idx, w, ex = _pcase(sd, T=5 + sd % 9, E=3 + sd % 6, k=1 + sd % 3)
                y, perm, tpe = moe_permute_forward(x, idx, w, ex)
                ey, eperm, etpe = _ref_perm(x, idx, w, ex)
                assert np.allclose(y, ey) and np.asarray(perm).tolist() == eperm.tolist() and np.asarray(tpe).tolist() == etpe.tolist(), f"seed={sd} 不一致"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-counting',
          q: md`能不能不用 $O(N\log N)$ 的 argsort 完成 permute？`,
          a: md`可以。专家 id 是 $[0, E)$ 的小整数，用计数排序：先 \`bincount\` 得到每个专家的计数与前缀和（段起点），再对每个派发 $p$ 计算「它在本专家内的名次」（one-hot 的列前缀和，或原子加计数器），位置 = 段起点 + 名次，整体 $O(Tk + E)$。GPU 上的 permute kernel 基本都是这个思路。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-full',
      freq: 3,
      q: md`从头到尾讲一遍 DeepSeek-V3 的 MoE 层，以及它和 Mixtral 的区别。`,
      a: md`
        **结论**：V3 = 细粒度 + 共享专家（256 路由 + 1 共享，top-8，专家中间维 2048）+ sigmoid 亲和度 + bias 只管选择的无辅助损失均衡（$\gamma=0.001$）+ 节点受限路由（≤4 节点）+ $\alpha=10^{-4}$ 的序列级均衡损失，训练推理都不丢 token。

        **与 Mixtral 的区别**：Mixtral 8 个大专家选 2、softmax(TopK) 即重归一化、无共享专家、用辅助损失均衡；V3 专家多而小、组合数巨大，用 sigmoid + 归一化 × 2.5，均衡不靠梯度，并为跨节点通信设计了路由约束。

        **延伸**：V4 在此基础上换 Sqrt(Softplus)、去节点限制、前 3 层 Hash 路由；K3 走向 896 选 16，用 LatentMoE 与 Quantile Balancing 应对超稀疏。
      `,
    },
    {
      id: 'qa-impl',
      freq: 2,
      q: md`生产环境的 MoE 前向是怎么实现的？和面试手撕版有什么不同？`,
      a: md`
        **结论**：手撕版按专家循环 + 布尔筛选；生产版是 **permute（按专家排序）→ grouped GEMM → unpermute**，多卡时再加 dispatch / combine 两次 all-to-all。

        **原因**：排序后每个专家的 token 连续，一次 grouped GEMM 处理所有专家、无需 padding（dropless）；按专家循环会启动 $E$ 次小 kernel，专家数上百时开销巨大。

        **延伸**：DeepSeek 开源的 DeepEP 专门优化 EP 的 all-to-all（训练 / prefill 的高吞吐内核与 decode 的低延迟内核）；推理侧还会做冗余专家部署来平衡热点专家的负载。
      `,
    },
  ],

  cheatsheet: md`
    **MoE 层全流程**：打分 → 选择（$s+b$，分组/查表）→ 权重（原始 $s$ 归一化 × scale）→ permute → 专家 → unpermute → + 共享专家
    \`\`\`python
    order = np.argsort(idx.ravel(), kind="stable")    # permute
    perm_tok, perm_w = order // k, w.ravel()[order]
    off = np.concatenate([[0], np.cumsum(np.bincount(idx.ravel(), minlength=E))])
    for e in range(E):
        a, b = off[e], off[e + 1]
        if b > a: out[a:b] = experts[e](x[perm_tok[a:b]])
    np.add.at(y, perm_tok, perm_w[:, None] * out)    # unpermute
    \`\`\`

    | 均衡手段 | 作用于 | 关键数字 |
    |---|---|---|
    | Switch aux | 梯度 | $\alpha=10^{-2}$，$\alpha N\sum fP$ |
    | z-loss | 梯度 | $c_z=10^{-3}$ |
    | V3 bias | 选择 | $\gamma=10^{-3}$，序列级 $\alpha=10^{-4}$ |
    | K3 QB | 选择 | 分位数 $1-k/n$，无步长 |
    | V4 Hash | 选择 | 前 3 个 MoE 层 |
  `,
})
