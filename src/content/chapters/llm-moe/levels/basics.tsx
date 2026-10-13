/**
 * llm-moe-basics：MoE 基础与手撕
 * 事实来源：Mixtral of Experts (arXiv 2401.04088)；DeepSeek-V2/V3 HF config；DeepSeek-V3 技术报告 (arXiv 2412.19437)；
 *          Switch Transformer (arXiv 2101.03961)；Qwen3-235B-A22B / gpt-oss-120b 官方 config 与 model card。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { MoeFlowViz } from '../viz/MoeFlowViz'
import { RouterGateViz } from '../viz/RouterGateViz'
import { DispatchViz } from '../viz/DispatchViz'
import { ParamCalcViz } from '../viz/ParamCalcViz'

/* 实战 1 的隐藏辅助代码：参考实现 + 专家工厂 + 调用计数器 */
const MOE_SETUP = py`
  import numpy as np

  def _ref_softmax(z):
      z = z - z.max(axis=-1, keepdims=True)
      e = np.exp(z)
      return e / e.sum(axis=-1, keepdims=True)

  def _ref_moe(x, W_r, experts, k, norm_topk_prob=True):
      probs = _ref_softmax(x @ W_r)
      idx = np.argsort(-probs, axis=-1, kind="stable")[:, :k]
      w = np.take_along_axis(probs, idx, axis=-1)
      if norm_topk_prob:
          w = w / w.sum(-1, keepdims=True)
      outs = np.stack([f(x) for f in experts], axis=1)          # (T, E, d) 稠密地全算一遍，仅用于对拍
      sel = np.take_along_axis(outs, idx[:, :, None], axis=1)   # (T, k, d)
      return (w[:, :, None] * sel).sum(axis=1)

  def _linear_experts(rng, E, d):
      Ws = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(E)]
      return [(lambda W: (lambda h: h @ W))(W) for W in Ws]

  def _swiglu_experts(rng, E, d, f):
      def make(Wg, Wu, Wd):
          def ffn(h):
              a = h @ Wg
              return ((a / (1.0 + np.exp(-a))) * (h @ Wu)) @ Wd
          return ffn
      return [make(rng.normal(size=(d, f)) / np.sqrt(d), rng.normal(size=(d, f)) / np.sqrt(d), rng.normal(size=(f, d)) / np.sqrt(f)) for _ in range(E)]

  class _Counter:
      """包装专家：统计每个专家被调用的次数与处理的行数（只统计非空调用）"""
      def __init__(self, experts):
          self.calls = [0] * len(experts)
          self.rows = [0] * len(experts)
          self.fns = [self._wrap(i, f) for i, f in enumerate(experts)]
      def _wrap(self, i, f):
          def g(h):
              h = np.asarray(h)
              if h.shape[0] > 0:
                  self.calls[i] += 1
                  self.rows[i] += h.shape[0]
              return f(h)
          return g

  def _case(seed, T, d, E):
      rng = np.random.default_rng(seed)
      x = rng.normal(size=(T, d))
      W_r = rng.normal(size=(d, E))
      return rng, x, W_r
`

const PARAM_SETUP = py`
  import random

  def _ref_count(cfg):
      d = cfg["d_model"]
      emb = cfg["vocab_size"] * d * (1 if cfg["tie_embeddings"] else 2)
      attn = cfg["n_layers"] * cfg["attn_params"]
      dense = cfg["n_dense_layers"] * 3 * d * cfg["d_ff_dense"]
      moe_layers = cfg["n_layers"] - cfg["n_dense_layers"]
      expert = 3 * d * cfg["d_expert"]
      router = moe_layers * d * cfg["n_routed"]
      common = emb + attn + dense + router + moe_layers * cfg["n_shared"] * expert
      total = common + moe_layers * cfg["n_routed"] * expert
      active = common + moe_layers * min(cfg["top_k"], cfg["n_routed"]) * expert
      return total, active

  MIXTRAL = dict(vocab_size=32000, d_model=4096, n_layers=32, n_dense_layers=0, d_ff_dense=0,
                 attn_params=4096 * 4096 * 2 + 4096 * 1024 * 2, n_routed=8, n_shared=0, top_k=2,
                 d_expert=14336, tie_embeddings=False)
  DSV3 = dict(vocab_size=129280, d_model=7168, n_layers=61, n_dense_layers=3, d_ff_dense=18432,
              attn_params=187105280, n_routed=256, n_shared=1, top_k=8, d_expert=2048, tie_embeddings=False)
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'why',
      title: '为什么要 MoE：把参数量和计算量解耦',
      body: md`
        稠密 Transformer 里，每个 token 都要经过**全部**参数，于是「模型更大」= 「每个 token 更贵」。
        **混合专家**（Mixture-of-Experts, MoE）把每层的 FFN 换成 $E$ 个并列的 FFN（**专家**）加一个**路由器（router）**，
        每个 token 只送进其中 $k$ 个专家：

        $$
        p(x) = \operatorname{softmax}(x W_r) \in \mathbb{R}^{E},\qquad
        y = \sum_{i \in \operatorname{TopK}(p(x),\,k)} g_i(x)\, \mathrm{FFN}_i(x)
        $$

        这样**总参数**随 $E$ 增长（容量、知识），**每 token 计算量**只随 $k$ 增长。为什么替换的是 FFN 而不是注意力？
        1. FFN 是参数大头：标准配置下注意力约 $4d^2$、FFN 约 $8d^2$，FFN 占每层约 2/3；
        2. FFN 是**逐 token 独立**的计算，天然可以按 token 分流；注意力要在 token 之间交互，拆成专家会破坏 KV 的共享。

        | 模型 | 路由专家：总数 选 k | 共享专家 | 总参数 / 激活参数 |
        |---|---|---|---|
        | Mixtral 8×7B | 8 选 2 | 0 | 47B / 13B |
        | Qwen3-235B-A22B | 128 选 8 | 0 | 235B / 22B |
        | gpt-oss-120b | 128 选 4 | 0 | 117B / 5.1B |
        | DeepSeek-V3 | 256 选 8 | 1 | 671B / 37B |
        | Kimi K2 | 384 选 8 | 1 | 约 1T / 32B |

        下图：8 个 token 经 router 选专家。拖动 **专家数 E** 和 **top-k**，看右侧每个专家分到多少 token。
      `,
      viz: MoeFlowViz,
      after: md`
        > [!TIP] 读图的关键
        > 总派发量恒为 $T \cdot k$，但它在专家之间**怎么分**完全由 router 决定。即使 router 是随机初始化的，负载也明显不均——
        > 有的专家被挤爆、有的颗粒无收。训练中这种不均会自我强化（被选得多 → 训练得好 → 更容易被选），这就是第二关的「专家坍缩」。

        > [!WARN] MoE 不是免费午餐
        > 省的是**计算**，不省**显存**：所有专家都得常驻显存（或分布在多卡上，做专家并行 EP，引入 all-to-all 通信）。
        > 小 batch 解码时还是访存瓶颈——一个 batch 里的 token 会把绝大多数专家都「点」一遍。
      `,
      checkpoint: {
        id: 'cp-why',
        type: 'single',
        prompt: md`一个 MoE 层有 $E=64$ 个与原稠密 FFN **同尺寸**的专家、top-$k=2$。忽略 router，它每个 token 的 FFN 计算量约是原稠密 FFN 的多少倍？`,
        options: ['1 倍', '2 倍', '32 倍', '64 倍'],
        answer: 1,
        explanation: md`每个 token 只经过 $k=2$ 个专家，FLOPs 是单个 FFN 的 2 倍，与 $E$ 无关。$E$ 只决定**参数量**（这里是 64 倍）。这也是 Mixtral「8×7B」只激活约 13B 的原因：2 个专家 + 共享的注意力与嵌入。`,
      },
    },
    {
      id: 'router',
      title: 'Router 与 top-k：要不要重归一化',
      body: md`
        router 就是一个线性层 $W_r \in \mathbb{R}^{d \times E}$。拿到分数后，主流做法有三种，**区别全在 top-k 之后的权重**：

        | 做法 | 公式 | 代表 |
        |---|---|---|
        | softmax → top-k，**不归一** | $g_i = p_i$（$i \in$ TopK） | Switch（k=1）、DeepSeek-V2 |
        | softmax → top-k → **重归一化** | $g_i = p_i / \sum_{j \in \text{TopK}} p_j$ | Mixtral、Qwen3 |
        | sigmoid → top-k → 归一化 × 缩放 | $g_i = \gamma\, s_i / \sum_{j \in \text{TopK}} s_j$ | DeepSeek-V3（$\gamma=2.5$） |

        Mixtral 论文写的是 $G(x) = \operatorname{softmax}(\operatorname{TopK}(x W_g))$：先在 logits 上取 top-k、其余置 $-\infty$，再 softmax。
        它与第二行**完全等价**——子集上的 softmax $= e^{z_i}/\sum_{j\in S} e^{z_j} = p_i / \sum_{j \in S} p_j$，全集的归一化常数被约掉了。
        HuggingFace 的实现就是「全量 softmax → topk → 除以和」。

        **重归一化的利弊**
        - 利：权重和恒为 1，输出幅度不受 router「自信程度」和专家数 $E$ 影响；
        - 弊：**k = 1 时权重恒为 1**，主损失对 router 的梯度为 0（router 只能靠辅助损失学习）。所以 Switch Transformer（top-1）直接用 $p_i$ 乘专家输出，梯度经 $p_i$ 回传给 router。
        - 不归一化时权重和 = top-k 概率的覆盖率，$E$ 越大越小。DeepSeek-V2 的 config 是 \`norm_topk_prob: false\`、\`routed_scaling_factor: 16.0\`，这个放大系数可以理解为对此的补偿（官方未展开解释）。
      `,
      viz: RouterGateViz,
      after: md`
        > [!INTERVIEW] 高频追问：「你写的 MoE 要不要 renormalize？」
        > 先说清是哪种模型的约定（Mixtral / Qwen3 是 \`norm_topk_prob=True\`，DeepSeek-V2 是 False），再指出 k=1 时重归一化会让 router 失去主损失梯度。
        > 最后补一句：DeepSeek-V3 改用 sigmoid，各专家的亲和度互相独立，必须归一化，并乘 \`routed_scaling_factor=2.5\`（见 HF config）。
      `,
      checkpoint: {
        id: 'cp-router',
        type: 'judge',
        prompt: md`对同一组 logits，$\operatorname{softmax}(\operatorname{TopK}(z))$（未选中的置 $-\infty$）与「先对全部 $z$ 做 softmax，取 top-k 概率后再除以它们的和」得到的权重完全相同。`,
        answer: true,
        explanation: md`正确。设选中集合为 $S$，后者为 $\frac{e^{z_i}/Z}{\sum_{j\in S} e^{z_j}/Z} = \frac{e^{z_i}}{\sum_{j\in S} e^{z_j}}$，正是前者。全集归一化常数 $Z$ 被约掉，所以二者只是实现顺序不同。`,
      },
    },
    {
      id: 'dispatch',
      title: 'Dispatch 与 Combine：按专家循环，而不是按 token 循环',
      body: md`
        路由结果是 \`idx\`（形状 $(T, k)$，每个 token 选了哪些专家）和 \`w\`（对应权重）。接下来要把 token **分发**给专家、算完再**加权合并**。三种常见实现：

        1. **按专家循环 + gather / scatter**（面试手撕首选）：对每个专家 $e$，找出所有选了它的 token，拼成一个小 batch 一次算完，再乘权重加回 $y$。循环 $E$ 次，每次是一个稠密矩阵乘。
        2. **one-hot 分发张量 + einsum**（GShard / Switch 的 TPU 写法）：构造 $(T, E, C)$ 的 dispatch / combine 张量（$C$ 为专家容量，下一关讲），全是静态形状，但内存是 $O(T \cdot E \cdot C)$。
        3. **按专家排序 + grouped GEMM**（MegaBlocks 等 GPU 实现，「dropless」）：把 $T \cdot k$ 个 (token, 专家) 对按专家 id 排序，切成连续段，一次 grouped GEMM 算完，再按原顺序还原（unpermute）。BOSS 关会手撕它。

        **千万不要**写成「对每个 token 循环、对它的 k 个专家各算一次」——那是 $T \cdot k$ 次向量-矩阵乘，GPU 利用率极低。
      `,
      viz: DispatchViz,
      after: md`
        > [!WARN] \`y[tok] += …\` 什么时候安全
        > NumPy 的花式索引 \`y[tok] += v\` 遇到**重复下标只会生效一次**。按专家循环时，同一个专家内一个 token 至多出现一次（top-k 选的是 k 个不同专家），所以安全；
        > 但如果你把全部 $T \cdot k$ 个派发对展平后一次性 \`y[flat_tok] += contrib\`，同一 token 的 k 份贡献只会留下最后一份——必须用 \`np.add.at\`（PyTorch 用 \`index_add_\` / \`scatter_add_\`）。
      `,
      checkpoint: {
        id: 'cp-dispatch',
        type: 'single',
        prompt: md`把所有派发对展平：\`flat_tok\`（长度 $T\cdot k$）、\`contrib\`（$(T\cdot k, d)$），想把贡献累加回 \`y\`。下面哪种写法是对的？`,
        options: ['`y[flat_tok] += contrib`', '`np.add.at(y, flat_tok, contrib)`', '`y[flat_tok] = contrib`', '`y += contrib[flat_tok]`'],
        answer: 1,
        explanation: md`\`flat_tok\` 中每个 token 出现 $k$ 次。A 的缓冲式花式索引对重复下标只累加一次；C 直接覆盖；D 的形状和语义都不对。\`np.add.at\` 是无缓冲的逐元素累加，能正确处理重复下标。`,
      },
    },
    {
      id: 'params',
      title: '总参数 vs 激活参数：逐项算一遍',
      body: md`
        设 SwiGLU 专家的中间维为 $d_e$，单个专家参数 $= 3\,d\,d_e$（gate、up、down 三个矩阵）。对有 $L_{\text{moe}}$ 个 MoE 层的模型：

        $$
        \begin{aligned}
        N_{\text{total}} &= \underbrace{N_{\text{emb}} + N_{\text{attn}} + N_{\text{dense}}}_{\text{所有 token 共用}} + L_{\text{moe}}\big[(E_r + E_s)\cdot 3 d d_e + d\,E_r\big] \\
        N_{\text{active}} &= N_{\text{emb}} + N_{\text{attn}} + N_{\text{dense}} + L_{\text{moe}}\big[(k + E_s)\cdot 3 d d_e + d\,E_r\big]
        \end{aligned}
        $$

        **DeepSeek-V3 验算**（$d=7168$，61 层，前 3 层稠密 $d_{ff}=18432$，58 个 MoE 层 × (256 路由 + 1 共享)，$d_e=2048$，MLA 每层约 187.1M，词表 129280）：
        - 单个专家 $3 \times 7168 \times 2048 \approx 44.0\text{M}$；全部专家 $58 \times 257 \times 44.0\text{M} \approx 656.5\text{B}$；
        - 注意力 $61 \times 187.1\text{M} \approx 11.4\text{B}$，嵌入 + 输出头 $2 \times 129280 \times 7168 \approx 1.85\text{B}$，稠密 FFN $\approx 1.19\text{B}$，router $\approx 0.11\text{B}$；
        - 合计 $\approx 671.0\text{B}$；激活 $= 58 \times 9 \times 44.0\text{M} + 14.6\text{B} \approx 37.6\text{B}$。与官方 671B / 37B 吻合。

        可见：**九成以上的参数在路由专家里**，激活参数里专家部分只占约六成。
      `,
      viz: ParamCalcViz,
      after: md`
        > [!NOTE] 激活参数 ≠ 推理成本
        > 激活参数决定每 token 的 FLOPs（前向 ≈ $2N_{\text{active}}$）。但 decode 是访存瓶颈：batch 内 $B$ 个 token 各选 $k$ 个专家，
        > 若路由近似均匀独立，一层里被触碰的专家数期望为 $E\,[1-(1-k/E)^B]$。DeepSeek-V3 在 $B=64$ 时约为 222 个（共 256 个）——几乎要把整层专家权重都读一遍。
        > 这就是为什么 MoE 推理必须靠大 batch + 专家并行把权重读取摊薄。
      `,
      checkpoint: {
        id: 'cp-params',
        type: 'numeric',
        prompt: md`某 MoE 层：$d = 4096$，每个专家是 SwiGLU、$d_e = 1408$，64 个路由专家选 6 个，另有 2 个共享专家。单层中**每 token 激活的专家参数**有多少百万（M）？（1M = $10^6$）`,
        answer: 138.412032,
        tolerance: 0.01,
        unit: 'M',
        explanation: md`激活的专家 = 6 个路由 + 2 个共享 = 8 个；每个 $3 \times 4096 \times 1408 = 17{,}301{,}504$；合计 $8 \times 17.30\text{M} \approx 138.4\text{M}$。常见错误：漏掉共享专家（得 103.8M），或把 SwiGLU 当成两个矩阵（得 92.3M）。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-ffn',
      type: 'single',
      difficulty: 1,
      prompt: md`主流 MoE 大模型把哪一部分换成了专家？最主要的理由是什么？`,
      options: [
        '注意力层；因为注意力计算量随序列长度平方增长',
        'FFN 层；因为它参数占比最大且逐 token 独立计算，易于按 token 分流',
        '嵌入层；因为词表越大越需要分片',
        'LayerNorm；因为它在每层都出现',
      ],
      answer: 1,
      explanation: md`FFN 约占每层 2/3 参数，并且对每个 token 独立计算，把 token 分给不同 FFN 不影响其它 token。注意力需要 token 间交互，拆分会破坏 KV 共享；嵌入层是查表，没有计算可省；LayerNorm 参数可以忽略不计。`,
    },
    {
      id: 'q-memory',
      type: 'judge',
      difficulty: 1,
      prompt: md`MoE 模型推理时，只需要把每个 token 激活的那几个专家放进显存，因此显存占用与激活参数量成正比。`,
      answer: false,
      explanation: md`错误。哪个 token 选哪个专家要到运行时才知道，而一个 batch 内不同 token 会选到几乎所有专家，所以**全部专家权重**都必须常驻显存（或通过专家并行分布在多卡上）。MoE 省的是计算，不省显存。`,
    },
    {
      id: 'q-renorm',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 top-k 之后「是否重归一化门控权重」，下列说法正确的是（多选）：`,
      options: [
        'Mixtral 的 softmax(TopK(logits)) 等价于先 softmax 再对 top-k 概率重归一化',
        'top-1 路由若重归一化，router 无法从语言模型主损失得到梯度',
        '不重归一化时，权重之和等于 top-k 概率之和，专家数越多通常越小',
        'DeepSeek-V3 用 softmax 打分，且不做归一化',
        '重归一化会让每个专家的输出被放大 E 倍',
      ],
      answer: [0, 1, 2],
      explanation: md`A：子集 softmax 等于概率除以子集概率和。B：k=1 时归一化后权重恒为 1，与 router 参数无关，梯度为 0（Switch 因此直接用 $p_i$）。C：不归一化时 $\sum g = \sum_{\text{TopK}} p$，概率被更多专家摊薄。D 错：V3 用 **sigmoid** 打分，并对选中的分数**归一化**后乘 2.5。E 错：归一化后权重和为 1，与 $E$ 无关。`,
    },
    {
      id: 'q-mixtral-active',
      type: 'numeric',
      difficulty: 2,
      prompt: md`
        按 Mixtral 8×7B 的公开配置计算**每 token 激活参数**（单位 B = $10^9$，忽略 norm）：
        词表 32000、$d=4096$、32 层；注意力为 GQA（32 个 Q 头、8 个 KV 头、$d_h=128$，无 bias）；每层 8 个 SwiGLU 专家选 2 个，$d_e = 14336$；
        router 为 $d \times 8$ 的线性层；输入嵌入与输出头不共享，都计入激活参数。
      `,
      answer: 12.879659008,
      tolerance: 0.01,
      unit: 'B',
      explanation: md`
        - 注意力/层：$W_Q, W_O$ 各 $4096^2$，$W_K, W_V$ 各 $4096 \times 1024$，合计 $41.94\text{M}$；32 层 $= 1.342\text{B}$
        - 激活专家：$32 \times 2 \times 3 \times 4096 \times 14336 = 11.274\text{B}$
        - 嵌入 + 输出头：$2 \times 32000 \times 4096 = 0.262\text{B}$；router：$32 \times 4096 \times 8 \approx 0.001\text{B}$

        合计 $\approx 12.88\text{B}$，与官方「约 13B 激活」一致（总参数同法可得 46.70B）。常见错误是把 K/V 按 32 个头算，多出约 0.8B。
      `,
    },
    {
      id: 'q-touched',
      type: 'numeric',
      difficulty: 3,
      prompt: md`decode 时一个 batch 有 $B=32$ 个 token，某层 $E=64$ 个专家、top-$k=2$。假设每个 token 独立地**均匀随机**选 2 个不同专家，这一层期望有多少个专家至少被一个 token 选中（需要读取其权重）？`,
      answer: 40.8285,
      tolerance: 0.01,
      unit: '个',
      explanation: md`某个专家被单个 token 选中的概率为 $k/E = 1/32$，不被 32 个 token 中任何一个选中的概率为 $(1-1/32)^{32} \approx 0.3620$。期望被触碰数 $= 64 \times (1 - 0.3620) \approx 40.83$。也就是说只有 32 个 token，却要读约 64% 的专家权重——这是 MoE 小 batch 解码访存效率差的根源。`,
    },
    {
      id: 'q-fill-v3',
      type: 'fill',
      difficulty: 1,
      prompt: md`DeepSeek-V3 每个 MoE 层有 ____(1)____ 个路由专家、____(2)____ 个共享专家，每个 token 激活 ____(3)____ 个路由专家；它的打分函数是 ____(4)____。`,
      blanks: [
        { label: '(1)', answers: ['256'], numeric: true },
        { label: '(2)', answers: ['1'], numeric: true },
        { label: '(3)', answers: ['8'], numeric: true },
        { label: '(4)', answers: ['sigmoid', 'Sigmoid', 'σ'] },
      ],
      explanation: md`见 DeepSeek-V3 技术报告 §4.2：每个 MoE 层 1 个共享专家 + 256 个路由专家，专家中间维 2048，每 token 激活 8 个路由专家，且最多发往 4 个节点。亲和度用 sigmoid（V2 用 softmax）。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 1,
      prompt: md`把一个 MoE 层的前向步骤按正确顺序排列：`,
      items: [
        '计算 router logits：`logits = x @ W_r`',
        '打分（softmax 或 sigmoid）并取 top-k 专家下标',
        '得到门控权重（按约定决定是否在 top-k 内归一化）',
        '按专家分组：把选了同一专家的 token 拼成小 batch',
        '每个专家对自己的小 batch 做一次 FFN',
        '乘门控权重，累加回各 token 的输出位置',
      ],
      explanation: md`路由（打分 → 选择 → 权重）在前，分发（dispatch）→ 专家计算 → 合并（combine）在后。共享专家（若有）与路由专家并行计算，最后相加。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把模型与它的路由配置配对：`,
      pairs: [
        ['Mixtral 8×7B', '8 个专家选 2，无共享专家'],
        ['DeepSeek-V3', '256 个路由专家选 8 + 1 个共享专家'],
        ['Qwen3-235B-A22B', '128 个专家选 8，无共享专家'],
        ['gpt-oss-120b', '128 个专家选 4'],
      ],
      explanation: md`均来自官方 config / 技术报告。注意 Qwen3 的 MoE **没有**共享专家（Qwen2-MoE 与 Qwen3-Next / Qwen3.5 有），gpt-oss-120b 每 token 只激活 5.1B（总 117B）。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出「按专家循环」的分发与合并（\`idx\`、\`w\` 形状均为 $(T,k)$，已重归一化；有 2 行干扰项）：`,
      lines: [
        'y = np.zeros_like(x)',
        'for e in range(E):',
        '    tok, slot = np.nonzero(idx == e)',
        '    if tok.size == 0:',
        '        continue',
        '    out = experts[e](x[tok])',
        '    y[tok] += w[tok, slot][:, None] * out',
        'return y',
      ],
      distractors: ['    out = experts[e](x)', '    y[tok] = w[tok, slot][:, None] * out'],
      explanation: md`干扰项 1 让每个专家处理**所有** token，计算量变成稠密的 $E$ 倍；干扰项 2 用赋值覆盖，会丢掉同一 token 在先前专家处累加的贡献（每个 token 有 k 份贡献）。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的路由函数有 **2 处** bug（\`probs\` 形状为 $(T, E)$）。点出有问题的行：`,
      code: py`
        import numpy as np

        def route(x, W_r, k):
            logits = x @ W_r
            z = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
            idx = np.argsort(probs, axis=-1)[:, :k]
            w = np.take_along_axis(probs, idx, axis=-1)
            w = w / w.sum()
            return idx, w
      `,
      bugLines: [7, 9],
      explanation: md`第 7 行 \`argsort\` 是**升序**，取前 k 个是概率**最小**的专家，应为 \`np.argsort(-probs, axis=-1)[:, :k]\`（或 \`argpartition(-probs, k-1, axis=-1)[:, :k]\`）。第 9 行没有指定轴，除的是**全局**和，应为 \`w / w.sum(axis=-1, keepdims=True)\`。第 5–6 行是标准的数值稳定 softmax，没有问题。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`某 token 的 router softmax 概率如下，top-$k=3$。点出被选中的专家：`,
      items: ['E0 · 0.08', 'E1 · 0.21', 'E2 · 0.05', 'E3 · 0.17', 'E4 · 0.02', 'E5 · 0.19', 'E6 · 0.16', 'E7 · 0.12'],
      cols: 4,
      answer: [1, 3, 5],
      explanation: md`最大的三个是 E1(0.21)、E5(0.19)、E3(0.17)；E6(0.16) 只差一点落选。重归一化后权重为 $0.21/0.57 \approx 0.368$、$0.19/0.57 \approx 0.333$、$0.17/0.57 \approx 0.298$；不归一化时权重和只有 0.57。`,
    },
    {
      id: 'q-switch',
      type: 'single',
      difficulty: 3,
      prompt: md`Switch Transformer 用 top-1 路由，输出写成 $y = p_i(x)\,E_i(x)$（$i$ 为概率最大的专家），而不是直接 $y = E_i(x)$。最关键的原因是：`,
      options: [
        '$p_i < 1$ 可以起到缩放输出、防止激活爆炸的作用',
        '只有乘上 $p_i$，主损失才能通过 $p_i$ 把梯度传给 router；直接用 $E_i(x)$ 时 argmax 不可导，router 学不到东西',
        '为了让输出与稠密模型的数值范围对齐',
        '为了在推理时省掉 softmax 的计算',
      ],
      answer: 1,
      explanation: md`top-k 选择本身（argmax）不可导。router 唯一的可导通路是门控权重 $p_i(x)$：$\partial L/\partial p_i = \langle \partial L/\partial y, E_i(x)\rangle$。若输出不含 $p_i$（或 top-1 再重归一化使权重恒为 1），router 的梯度只能来自辅助损失。A、C 不是主要动机；D 错，推理仍要算 softmax 才能取 $p_i$。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-moe',
      title: '一步步写出 NumPy MoE 前向（SwiGLU 专家）',
      intro: md`
        实现 \`moe_forward(x, W_r, W_gate, W_up, W_down, k, norm_topk_prob=True)\`：
        - \`x\`：$(T, d)$；\`W_r\`：$(d, E)$；\`W_gate\`、\`W_up\`：$(E, d, f)$；\`W_down\`：$(E, f, d)$
        - 专家 $e$ 的 FFN：$\big(\operatorname{silu}(h W_{\text{gate}}[e]) \odot h W_{\text{up}}[e]\big) W_{\text{down}}[e]$
        - softmax 打分 → top-k → （可选）重归一化 → 按专家分组计算 → 加权累加。
      `,
      template: py`
        import numpy as np

        def silu(v):
            return v / (1.0 + np.exp(-v))

        def moe_forward(x, W_r, W_gate, W_up, W_down, k, norm_topk_prob=True):
            T, d = x.shape
            E = W_r.shape[1]
            logits = x @ W_r                                    # (T, E)
            z = {{1}}
            probs = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
            topk_idx = {{2}}
            topk_w = np.take_along_axis(probs, topk_idx, axis=-1)   # (T, k)
            if norm_topk_prob:
                topk_w = {{3}}
            y = np.zeros_like(x)
            for e in range(E):
                tok, slot = {{4}}
                if tok.size == 0:
                    continue
                h = x[tok]
                out = (silu(h @ W_gate[e]) * (h @ W_up[e])) @ W_down[e]
                {{5}}
            return y
      `,
      steps: [
        {
          blank: 1,
          instruction: md`数值稳定的 softmax：先减去每行最大值。`,
          answer: 'logits - logits.max(axis=-1, keepdims=True)',
          accept: ['logits - np.max(logits, axis=-1, keepdims=True)', 'logits - logits.max(-1, keepdims=True)'],
          explain: md`softmax 对整体平移不变，减去行最大值后 $e^{z} \le 1$，不会溢出。\`keepdims=True\` 保证形状 $(T,1)$ 能广播。`,
        },
        {
          blank: 2,
          instruction: md`取每个 token 概率最大的 $k$ 个专家下标，形状 $(T, k)$。`,
          answer: 'np.argsort(-probs, axis=-1)[:, :k]',
          accept: ['np.argsort(probs, axis=-1)[:, ::-1][:, :k]', 'np.argpartition(-probs, k - 1, axis=-1)[:, :k]', 'np.argsort(-probs, axis=1)[:, :k]'],
          hint: 'argsort 默认升序，取负号或者翻转。',
          explain: md`\`argsort\` 是 $O(E\log E)$；\`argpartition\` 是 $O(E)$ 但前 k 个不保证有序——对 MoE 来说顺序无关紧要，二者都可以。PyTorch 用 \`torch.topk\`。`,
        },
        {
          blank: 3,
          instruction: md`在选中的 $k$ 个专家内重归一化，使每行权重和为 1。`,
          answer: 'topk_w / topk_w.sum(axis=-1, keepdims=True)',
          accept: ['topk_w / topk_w.sum(-1, keepdims=True)', 'topk_w / np.sum(topk_w, axis=-1, keepdims=True)'],
          explain: md`这就是 Mixtral / Qwen3 的 \`norm_topk_prob=True\`。漏写 \`axis\` 会除以全局和，是最常见的 bug。`,
        },
        {
          blank: 4,
          instruction: md`找出所有选择了专家 \`e\` 的 (token 下标, 第几选择)。`,
          answer: 'np.nonzero(topk_idx == e)',
          accept: ['np.where(topk_idx == e)', '(topk_idx == e).nonzero()'],
          explain: md`\`topk_idx == e\` 是 $(T,k)$ 的布尔矩阵，\`nonzero\` 返回行下标（token）和列下标（slot）。由于每个 token 的 k 个专家互不相同，\`tok\` 里没有重复。`,
        },
        {
          blank: 5,
          instruction: md`把专家输出乘以对应的门控权重，累加回 \`y\` 的对应行。`,
          answer: 'y[tok] += topk_w[tok, slot][:, None] * out',
          accept: ['np.add.at(y, tok, topk_w[tok, slot][:, None] * out)', 'y[tok] += topk_w[tok, slot, None] * out'],
          explain: md`\`topk_w[tok, slot]\` 取出每个派发对的权重，\`[:, None]\` 变成 $(n,1)$ 广播到 $(n,d)$。这里用 \`+=\` 是安全的，因为 \`tok\` 在同一专家内不重复；跨专家的累加发生在不同的循环轮次里。`,
        },
      ],
      setup: py`
        def _ref_guided(x, W_r, Wg, Wu, Wd, k, norm=True):
            z = x @ W_r
            z = z - z.max(-1, keepdims=True)
            p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
            idx = np.argsort(-p, axis=-1, kind="stable")[:, :k]
            w = np.take_along_axis(p, idx, axis=-1)
            if norm:
                w = w / w.sum(-1, keepdims=True)
            y = np.zeros_like(x)
            for t in range(x.shape[0]):
                for j in range(k):
                    e = idx[t, j]
                    a = x[t] @ Wg[e]
                    y[t] += w[t, j] * (((a / (1 + np.exp(-a))) * (x[t] @ Wu[e])) @ Wd[e])
            return y

        def _gcase(seed, T=10, d=6, E=5, f=8):
            r = np.random.default_rng(seed)
            return (r.normal(size=(T, d)), r.normal(size=(d, E)), r.normal(size=(E, d, f)) / 3,
                    r.normal(size=(E, d, f)) / 3, r.normal(size=(E, f, d)) / 3)
      `,
      tests: [
        {
          name: 'top-2 + 重归一化',
          code: py`
            args = _gcase(0)
            got = moe_forward(*args, k=2)
            exp = _ref_guided(*args, k=2)
            assert got.shape == exp.shape, f"形状应为 {exp.shape}，得到 {got.shape}"
            assert np.allclose(got, exp, atol=1e-8), f"最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: '不归一化 + k=3',
          code: py`
            args = _gcase(1)
            got = moe_forward(*args, k=3, norm_topk_prob=False)
            exp = _ref_guided(*args, k=3, norm=False)
            assert np.allclose(got, exp, atol=1e-8), f"最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            for s in range(2, 8):
                args = _gcase(s, T=7 + s, E=4 + s % 3)
                for k in (1, 2):
                    got = moe_forward(*args, k=k)
                    exp = _ref_guided(*args, k=k)
                    assert np.allclose(got, exp, atol=1e-8), f"seed={s}, k={k}：最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
      ],
      outro: md`
        这就是面试里「手撕 MoE」的骨架。追问通常有三个方向：
        ① 加**共享专家**（直接 \`y += shared_ffn(x)\`）；② 加**负载均衡 loss**（下一关）；③ 改成 DeepSeek-V3 的 **sigmoid + bias 选择**（第三关）。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-moe-forward',
      title: '手撕 MoE 层前向（专家按批调用）',
      difficulty: 'medium',
      source: '大模型岗手撕 · MoE',
      timeLimitMin: 25,
      description: md`
        实现一个 top-k MoE 层的前向。专家以**可调用对象**给出（每个把 $(n, d)$ 映射到 $(n, d)$），这样测试可以统计你调用专家的方式。

        \`\`\`python
        def moe_forward(x: np.ndarray, W_r: np.ndarray, experts: list, k: int,
                        norm_topk_prob: bool = True) -> np.ndarray: ...
        \`\`\`

        - \`x\`：$(T, d)$ float64；\`W_r\`：$(d, E)$；\`experts[e](h)\` 返回 \`h\` 经专家 $e$ 的输出
        - 打分 \`probs = softmax(x @ W_r, axis=-1)\`，每个 token 选概率最大的 $k$ 个专家（测试中没有并列）
        - 权重为对应概率；\`norm_topk_prob=True\` 时在这 $k$ 个之内重归一化
        - 返回 $y_t = \sum_j w_{t,j}\, \text{experts}[\text{idx}_{t,j}](x_t)$，形状 $(T, d)$

        **效率要求（隐藏测试会检查）**
        - 每个专家**至多调用一次**，一次性处理所有分给它的 token；没有 token 的专家不要调用（或只用 0 行调用）；
        - 所有专家处理的总行数必须恰好是 $T \cdot k$——不能先让每个专家算全部 token 再挑选。
      `,
      starter: py`
        import numpy as np

        def moe_forward(x, W_r, experts, k, norm_topk_prob=True):
            # 1) softmax 打分  2) top-k  3) 权重（可选重归一化）
            # 4) 按专家分组，每个专家只调用一次  5) 加权累加
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def moe_forward(x, W_r, experts, k, norm_topk_prob=True):
            T, d = x.shape
            E = W_r.shape[1]
            logits = x @ W_r
            z = logits - logits.max(axis=-1, keepdims=True)          # 数值稳定
            probs = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
            idx = np.argsort(-probs, axis=-1)[:, :k]                  # (T, k) top-k 专家
            w = np.take_along_axis(probs, idx, axis=-1)               # (T, k) 门控权重
            if norm_topk_prob:
                w = w / w.sum(axis=-1, keepdims=True)

            y = np.zeros((T, d), dtype=np.result_type(x, np.float64))
            for e in range(E):
                tok, slot = np.nonzero(idx == e)                      # 选了 e 的 (token, slot)
                if tok.size == 0:
                    continue                                          # 空闲专家不计算
                out = experts[e](x[tok])                              # 一次批量计算
                y[tok] += w[tok, slot][:, None] * out                 # tok 在同一专家内不重复
            return y
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class SwiGLUExpert(nn.Module):
                def __init__(self, d, f):
                    super().__init__()
                    self.gate = nn.Linear(d, f, bias=False)
                    self.up = nn.Linear(d, f, bias=False)
                    self.down = nn.Linear(f, d, bias=False)

                def forward(self, x):
                    return self.down(F.silu(self.gate(x)) * self.up(x))

            class MoE(nn.Module):
                def __init__(self, d, f, n_experts, k, norm_topk_prob=True):
                    super().__init__()
                    self.k, self.norm = k, norm_topk_prob
                    self.router = nn.Linear(d, n_experts, bias=False)
                    self.experts = nn.ModuleList(SwiGLUExpert(d, f) for _ in range(n_experts))

                def forward(self, x):                      # x: (B, S, d)
                    shape = x.shape
                    x = x.reshape(-1, shape[-1])           # (T, d)
                    probs = self.router(x).float().softmax(dim=-1)   # router 用 fp32
                    w, idx = probs.topk(self.k, dim=-1)    # (T, k)
                    if self.norm:
                        w = w / w.sum(dim=-1, keepdim=True)
                    w = w.to(x.dtype)
                    y = torch.zeros_like(x)
                    for e, expert in enumerate(self.experts):
                        tok, slot = torch.where(idx == e)
                        if tok.numel() == 0:
                            continue
                        out = expert(x[tok]) * w[tok, slot, None]
                        y.index_add_(0, tok, out)          # 等价于 y[tok] += out
                    return y.reshape(shape), probs         # probs 留给负载均衡 loss
          `,
        },
      ],
      explanation: md`
        **思路**：路由 → 分组 → 批量计算 → 加权合并。

        1. 数值稳定 softmax 后用 \`argsort(-probs)[:, :k]\`（或 \`argpartition\`）取 top-k，\`take_along_axis\` 取出对应概率；
        2. 对每个专家 \`e\`：\`tok, slot = np.nonzero(idx == e)\`，把 \`x[tok]\` 一次送进专家；
        3. \`y[tok] += w[tok, slot][:, None] * out\`——同一专家内 \`tok\` 不重复，所以花式索引的 \`+=\` 是安全的。

        **复杂度**：路由 $O(TdE)$，专家计算总共处理 $T\cdot k$ 行（每行一次 FFN），循环 $E$ 次。

        **常见错误**
        - \`argsort\` 忘记取负，选成了概率最小的专家；
        - 重归一化漏写 \`axis\`；
        - 每个专家对全部 token 计算后再 mask——结果对，但计算量是 $E/k$ 倍（隐藏测试会抓）；
        - 把所有派发对展平后 \`y[flat_tok] += ...\`——重复下标只加一次，结果错；要用 \`np.add.at\`。
      `,
      complexity: { time: 'O(T·d·E + T·k·C_expert)', space: 'O(T·(E + d))' },
      hints: [
        md`先得到 \`idx\`（$(T,k)$）和 \`w\`（$(T,k)$），再考虑分发。`,
        md`对每个专家 \`e\`，\`np.nonzero(idx == e)\` 同时给出 token 下标和它在 top-k 中的位置 slot。`,
        md`\`out = experts[e](x[tok])\`，然后 \`y[tok] += w[tok, slot][:, None] * out\`。`,
      ],
      setup: MOE_SETUP,
      tests: [
        {
          name: '示例：top-2 + 重归一化',
          code: py`
            rng, x, W_r = _case(0, T=6, d=4, E=4)
            ex = _linear_experts(rng, 4, 4)
            got = moe_forward(x, W_r, ex, 2)
            exp = _ref_moe(x, W_r, ex, 2)
            assert isinstance(got, np.ndarray) and got.shape == (6, 4), f"形状应为 (6, 4)，得到 {getattr(got, 'shape', type(got))}"
            assert np.allclose(got, exp, atol=1e-8), f"最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: '不重归一化',
          code: py`
            rng, x, W_r = _case(1, T=8, d=5, E=6)
            ex = _linear_experts(rng, 6, 5)
            got = moe_forward(x, W_r, ex, 2, norm_topk_prob=False)
            exp = _ref_moe(x, W_r, ex, 2, norm_topk_prob=False)
            assert np.allclose(got, exp, atol=1e-8), f"最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: 'k=1 且重归一化：权重恒为 1',
          code: py`
            rng, x, W_r = _case(2, T=5, d=3, E=4)
            ex = _linear_experts(rng, 4, 3)
            got = moe_forward(x, W_r, ex, 1, norm_topk_prob=True)
            best = np.argmax(x @ W_r, axis=-1)
            exp = np.stack([ex[e](x[t:t + 1])[0] for t, e in enumerate(best)])
            assert np.allclose(got, exp, atol=1e-8), f"k=1 归一化后应直接等于最优专家输出，最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: '效率：每个专家只调用一次、总行数 = T·k',
          hidden: true,
          code: py`
            rng, x, W_r = _case(3, T=40, d=6, E=8)
            cnt = _Counter(_swiglu_experts(rng, 8, 6, 12))
            got = moe_forward(x, W_r, cnt.fns, 3)
            probs = _ref_softmax(x @ W_r)
            idx = np.argsort(-probs, axis=-1, kind="stable")[:, :3]
            load = np.bincount(idx.ravel(), minlength=8)
            assert max(cnt.calls) <= 1, f"每个专家应至多调用 1 次，实际调用次数 {cnt.calls}"
            assert sum(cnt.rows) == 40 * 3, f"专家处理的总行数应为 T·k = 120，实际 {sum(cnt.rows)}（是不是让专家算了全部 token？）"
            assert cnt.rows == load.tolist(), f"各专家处理行数应等于其负载 {load.tolist()}，实际 {cnt.rows}"
          `,
        },
        {
          name: '随机对拍（SwiGLU 专家）',
          hidden: true,
          code: py`
            for s in range(10, 22):
                T, d, E = 5 + s, 4 + s % 3, 4 + s % 5
                for k in (1, 2, 3):
                    if k > E:
                        continue
                    rng, x, W_r = _case(s, T, d, E)
                    ex = _swiglu_experts(rng, E, d, 2 * d)
                    for norm in (True, False):
                        got = moe_forward(x, W_r, ex, k, norm_topk_prob=norm)
                        exp = _ref_moe(x, W_r, ex, k, norm_topk_prob=norm)
                        assert got.shape == exp.shape, f"形状应为 {exp.shape}，得到 {got.shape}"
                        assert np.allclose(got, exp, atol=1e-8), f"seed={s}, k={k}, norm={norm}：最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: 'k = E：退化为按概率加权的稠密混合',
          hidden: true,
          code: py`
            rng, x, W_r = _case(99, T=7, d=4, E=3)
            ex = _linear_experts(rng, 3, 4)
            got = moe_forward(x, W_r, ex, 3, norm_topk_prob=False)
            probs = _ref_softmax(x @ W_r)
            exp = sum(probs[:, e:e + 1] * ex[e](x) for e in range(3))
            assert np.allclose(got, exp, atol=1e-8), f"k=E 时应等于全部专家按概率加权，最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-shared',
          q: md`如果要加 DeepSeek 式的共享专家，代码怎么改？它会不会参与 top-k？`,
          a: md`共享专家不经过 router，所有 token 都过：\`y = y_routed + sum(shared_e(x) for shared_e in shared)\`。它不参与 top-k，也没有门控权重（权重恒为 1）。在实现上通常把多个共享专家合并成一个中间维更大的 FFN，一次矩阵乘算完。`,
        },
        {
          id: 'f-fp32',
          q: md`为什么 router 常在 fp32 下计算（PyTorch 版里的 \`.float()\`）？`,
          a: md`router logits 进入 exp，bf16 只有 8 位有效精度，大 logits 的舍入误差会被指数放大，导致路由抖动、训练不稳。Switch Transformer 提出「selective precision」：只在 router 内部把输入转成 float32，其余保持 bf16。下一关的 router z-loss 也是为了解决同一问题。`,
        },
      ],
    },
    {
      id: 'c-param-count',
      title: 'MoE 模型的总参数与激活参数',
      difficulty: 'easy',
      source: '大模型岗计算题',
      timeLimitMin: 12,
      description: md`
        给定一个（可能含稠密层的）MoE decoder 配置，返回 \`(total, active)\` 两个整数。

        \`\`\`python
        def moe_param_count(cfg: dict) -> tuple[int, int]: ...
        \`\`\`

        \`cfg\` 字段：\`vocab_size, d_model, n_layers, n_dense_layers, d_ff_dense, attn_params, n_routed, n_shared, top_k, d_expert, tie_embeddings\`

        **计数约定**（忽略所有 norm 与 bias）
        - 嵌入：\`vocab_size * d_model\`，\`tie_embeddings=False\` 时输出头再算一份；
        - 每层注意力参数直接给出：\`attn_params\`；
        - 前 \`n_dense_layers\` 层是稠密 SwiGLU FFN：\`3 * d_model * d_ff_dense\`；其余为 MoE 层；
        - 每个专家（路由或共享）是 SwiGLU：\`3 * d_model * d_expert\`；每个 MoE 层的 router 为 \`d_model * n_routed\`；
        - **激活参数**：嵌入（含输出头）、全部注意力、稠密层、router、共享专家，加上每个 MoE 层的 \`min(top_k, n_routed)\` 个路由专家。

        **示例**：Mixtral 8×7B（词表 32000、$d$=4096、32 层、\`attn_params = 41943040\`、8 选 2、\`d_expert=14336\`、不共享嵌入）
        → \`(46702526464, 12879659008)\`。
      `,
      starter: py`
        def moe_param_count(cfg: dict) -> tuple[int, int]:
            # TODO：按约定逐项累加
            return 0, 0
      `,
      solution: py`
        def moe_param_count(cfg: dict) -> tuple[int, int]:
            d = cfg["d_model"]
            emb = cfg["vocab_size"] * d * (1 if cfg["tie_embeddings"] else 2)
            attn = cfg["n_layers"] * cfg["attn_params"]
            dense = cfg["n_dense_layers"] * 3 * d * cfg["d_ff_dense"]
            moe_layers = cfg["n_layers"] - cfg["n_dense_layers"]
            expert = 3 * d * cfg["d_expert"]                      # SwiGLU：gate / up / down
            router = moe_layers * d * cfg["n_routed"]
            shared = moe_layers * cfg["n_shared"] * expert

            common = emb + attn + dense + router + shared          # 每个 token 都会用到
            total = common + moe_layers * cfg["n_routed"] * expert
            active = common + moe_layers * min(cfg["top_k"], cfg["n_routed"]) * expert
            return total, active
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：从真实模型统计）',
          code: py`
            import torch.nn as nn

            def count_from_module(model: nn.Module, n_routed: int, top_k: int):
                total = sum(p.numel() for p in model.parameters())
                # 约定：路由专家的参数名里含 "experts."（各框架命名不同，按实际调整）
                routed = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
                active = total - routed + routed * top_k // n_routed
                return total, active
          `,
        },
      ],
      explanation: md`
        把参数分成「每个 token 都用」和「只有被选中才用」两类：后者只有路由专家。于是
        $\text{active} = \text{total} - L_{\text{moe}}(E_r - k)\cdot 3 d d_e$。

        **验算**：DeepSeek-V3（MLA 每层 187,105,280 个参数）得到约 671.0B / 37.6B，与官方 671B / 37B 一致。
        **易错点**：SwiGLU 是 3 个矩阵不是 2 个；共享专家算激活；不共享嵌入时输出头要再算一份；稠密层不乘专家数。
      `,
      complexity: { time: 'O(1)', space: 'O(1)' },
      hints: [
        md`先算出一个专家的参数量 \`3 * d * d_expert\` 和 MoE 层数 \`n_layers - n_dense_layers\`。`,
        md`\`total\` 与 \`active\` 只差在路由专家的个数：\`n_routed\` vs \`min(top_k, n_routed)\`。`,
      ],
      setup: PARAM_SETUP,
      tests: [
        {
          name: 'Mixtral 8×7B',
          code: py`
            got = moe_param_count(dict(MIXTRAL))
            assert tuple(got) == (46702526464, 12879659008), f"期望 (46702526464, 12879659008)，得到 {got}"
          `,
        },
        {
          name: '稠密模型（Llama-2-7B 形状）：total == active',
          code: py`
            cfg = dict(vocab_size=32000, d_model=4096, n_layers=32, n_dense_layers=32, d_ff_dense=11008,
                       attn_params=4 * 4096 * 4096, n_routed=0, n_shared=0, top_k=0, d_expert=0, tie_embeddings=False)
            got = moe_param_count(cfg)
            assert tuple(got) == (6738149376, 6738149376), f"期望 (6738149376, 6738149376)，得到 {got}"
          `,
        },
        {
          name: 'DeepSeek-V3：约 671B / 37.6B',
          hidden: true,
          code: py`
            got = moe_param_count(dict(DSV3))
            exp = _ref_count(DSV3)
            assert tuple(got) == exp, f"期望 {exp}，得到 {got}"
            assert 670e9 < got[0] < 672e9, f"总参数应约 671B，得到 {got[0] / 1e9:.2f}B"
          `,
        },
        {
          name: '共享嵌入 + top_k > n_routed 的边界',
          hidden: true,
          code: py`
            cfg = dict(vocab_size=1000, d_model=64, n_layers=4, n_dense_layers=1, d_ff_dense=256,
                       attn_params=4 * 64 * 64, n_routed=2, n_shared=1, top_k=4, d_expert=32, tie_embeddings=True)
            got = moe_param_count(cfg)
            exp = _ref_count(cfg)
            assert tuple(got) == exp, f"期望 {exp}，得到 {got}"
            assert got[0] == got[1], "top_k ≥ n_routed 时所有专家都激活，total 应等于 active"
          `,
        },
        {
          name: '随机配置对拍',
          hidden: true,
          code: py`
            rnd = random.Random(5)
            for _ in range(200):
                L = rnd.randint(1, 80)
                cfg = dict(vocab_size=rnd.randint(1000, 200000), d_model=rnd.choice([512, 1024, 4096, 7168]),
                           n_layers=L, n_dense_layers=rnd.randint(0, L), d_ff_dense=rnd.randint(1, 30000),
                           attn_params=rnd.randint(10**5, 3 * 10**8), n_routed=rnd.randint(1, 512),
                           n_shared=rnd.randint(0, 2), top_k=rnd.randint(1, 16), d_expert=rnd.randint(64, 4096),
                           tie_embeddings=rnd.random() < 0.3)
                got = moe_param_count(cfg)
                exp = _ref_count(cfg)
                assert tuple(got) == exp, f"cfg={cfg}：期望 {exp}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-flops',
          q: md`训练 DeepSeek-V3 这样的模型，按 $6ND$ 估算训练 FLOPs 时 $N$ 该用总参数还是激活参数？`,
          a: md`用**激活参数**（每 token 实际参与的矩阵乘参数），约 37B，而不是 671B。$6N$ 的来源是每个参数在前向做 1 次乘加（2 FLOPs）、反向 2 次（4 FLOPs），只有被激活的专家参与。注意力的 $O(T^2)$ 项另算。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-handwrite',
      freq: 3,
      q: md`手撕一个 MoE 层，你会怎么写？关键点有哪些？`,
      a: md`
        **结论**：router（线性层）→ softmax/sigmoid 打分 → top-k → 门控权重（按约定是否重归一化）→ **按专家循环**做 gather / 批量 FFN / 加权 scatter-add → 加上共享专家（若有）→ 返回输出和路由概率（给负载均衡 loss 用）。

        **关键点**：① softmax 要减最大值，router 最好用 fp32；② top-k 取最大（\`argsort(-p)\` / \`torch.topk\`）；③ 按专家循环而不是按 token 循环，每个专家一次批量计算；④ 累加用 \`index_add_\`，展平写法必须处理重复下标；⑤ 返回 \`probs\` 与 \`idx\`，用于 aux loss 与负载统计。

        **延伸**：生产实现用排序 + grouped GEMM（dropless）或带容量的 dispatch；专家并行时分发变成 all-to-all。
      `,
    },
    {
      id: 'qa-vs-dense',
      freq: 3,
      q: md`MoE 相比同等激活参数的稠密模型，好处和代价分别是什么？`,
      a: md`
        **结论**：同样的每 token 计算量下，MoE 有多得多的参数（容量），通常能以更少训练 FLOPs 达到同等或更好的效果；代价在显存、通信、均衡与稳定性。

        **代价**：① 所有专家都要常驻显存，总参数决定部署成本；② 专家并行引入 all-to-all 通信，容易成为瓶颈；③ 负载不均衡会导致专家坍缩、木桶效应（最慢的卡拖住全局）、token 丢弃；④ router 训练不稳定（大 logits、bf16 舍入），需要 z-loss / fp32 router；⑤ 小 batch 解码时几乎要读全部专家，访存效率差。

        **延伸**：这也是为什么 DeepSeek-V3 要做无辅助损失均衡、节点受限路由、DualPipe 与 DeepEP 等一整套系统优化。
      `,
    },
    {
      id: 'qa-renorm',
      freq: 2,
      q: md`top-k 之后要不要对门控权重做归一化？不同模型怎么选的？`,
      a: md`
        **结论**：没有标准答案，是模型约定：Mixtral / Qwen3 归一化（\`norm_topk_prob=True\`），DeepSeek-V2 不归一化（配合 \`routed_scaling_factor=16\`），DeepSeek-V3 用 sigmoid 后必须归一化并乘 2.5。

        **原因**：归一化让输出幅度与 router 置信度、专家数解耦，训练更稳；不归一化保留了「router 不自信时输出变小」的信号，router 梯度路径也更直接。**k=1 时绝不能归一化**，否则权重恒为 1，router 没有主损失梯度（Switch 因此用 $p_i \cdot E_i(x)$）。

        **延伸**：sigmoid 打分时各专家分数不和为 1、量级随专家数变化不大，所以 V3 先归一化再用一个固定缩放因子调回合适的幅度。
      `,
    },
    {
      id: 'qa-why-ffn',
      freq: 2,
      q: md`为什么 MoE 一般只放在 FFN 上？有没有注意力的 MoE？`,
      a: md`
        **结论**：FFN 参数占比大（约 2/3）、逐 token 独立，最适合条件计算；注意力需要 token 间交互，按 token 路由会让同一序列的 K/V 分散在不同专家里，难以共享与缓存。

        **延伸**：确实有 attention 头级别的专家化研究（如 Mixture of Attention Heads、SwitchHead），但主流开源大模型（Mixtral、DeepSeek、Qwen、Kimi、gpt-oss）都只在 FFN 上做 MoE。另外 DeepSeek-V3 的前 3 层仍保留稠密 FFN，V4 把它们换成了 Hash 路由的 MoE（见「前沿」一关）。
      `,
    },
    {
      id: 'qa-decode',
      freq: 2,
      q: md`为什么说 MoE 在小 batch 解码时效率差？怎么缓解？`,
      a: md`
        **结论**：decode 是访存瓶颈。batch 里 $B$ 个 token 各选 $k$ 个专家，均匀路由下一层被触碰的专家数期望为 $E[1-(1-k/E)^B]$，$B$ 稍大就接近 $E$——读取的权重接近总参数，计算却只有激活参数那么多，算术强度很低。

        **缓解**：① 增大 batch（连续批处理），让每个专家一次处理更多 token；② 专家并行，把专家分散到多卡，每卡只读自己的专家；③ 低比特专家权重（如 DeepSeek-V4 的 FP4 路由专家）直接减少读取字节；④ 投机解码一次验证多个 token。
      `,
    },
  ],

  cheatsheet: md`
    **MoE 层**：$y = \sum_{i\in \text{TopK}} g_i\,\mathrm{FFN}_i(x)$（+ 共享专家）
    \`\`\`python
    p = softmax(x @ W_r)                        # (T, E)
    idx = np.argsort(-p, -1)[:, :k]             # (T, k)
    w = np.take_along_axis(p, idx, -1)
    w = w / w.sum(-1, keepdims=True)            # norm_topk_prob
    for e in range(E):
        tok, slot = np.nonzero(idx == e)
        if tok.size: y[tok] += w[tok, slot, None] * experts[e](x[tok])
    \`\`\`

    | 门控 | 权重 | 模型 |
    |---|---|---|
    | softmax→top-k | $p_i$ | Switch(k=1)、DeepSeek-V2 |
    | softmax→top-k→归一 | $p_i/\sum_S p$ | Mixtral、Qwen3 |
    | sigmoid→top-k→归一×γ | $2.5\,s_i/\sum_S s$ | DeepSeek-V3 |

    - 参数：专家 $3dd_e$；total 含全部 $E_r$，active 只含 $k$ 个；V3 ≈ 671B / 37B
    - k=1 + 归一化 ⇒ router 无主损失梯度
    - 展平 scatter 用 \`np.add.at\` / \`index_add_\`，别用 \`y[idx] +=\`
    - 期望触碰专家数 $E[1-(1-k/E)^B]$ → 小 batch 解码访存差
  `,
})
