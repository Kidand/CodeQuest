/**
 * llm-moe-frontier：2026 MoE 前沿
 * 事实来源：DeepSeek-V4 (arXiv 2606.19348) §2.1、§4.2；DeepSeek-V4-Flash / V4-Pro 官方 HF config 与 inference/model.py；
 *          Kimi K3 (arXiv 2607.24653 v2) §2.3 与 Table 1；Hash Layers (Roller et al., 2021)；
 *          Mixtral / DeepSeek-V2/V3 / Qwen3 / Qwen3.5 / Kimi K2 / gpt-oss 官方论文、config 或 model card。
 * Quantile Balancing 按原文 §2.3.3 的描述转述；大规模直方图估计（附录 D）等细节以原文为准。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { ActivationFnViz } from '../viz/ActivationFnViz'
import { HashRoutingViz } from '../viz/HashRoutingViz'
import { QuantileBalanceViz } from '../viz/QuantileBalanceViz'
import { ActivationTrendViz } from '../viz/ActivationTrendViz'

const V4_SETUP = py`
  import numpy as np

  def _ref_v4(logits, input_ids, layer_id, tid2eid, bias, k, n_hash_layers=3, route_scale=1.5):
      scores = np.sqrt(np.logaddexp(0.0, logits))
      if layer_id < n_hash_layers:
          idx = tid2eid[input_ids]
      else:
          idx = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :k]
      w = np.take_along_axis(scores, idx, axis=-1)
      return idx, w / w.sum(-1, keepdims=True) * route_scale

  def _canon(idx, w):
      idx = np.asarray(idx)
      w = np.asarray(w, dtype=float)
      o = np.argsort(idx, axis=-1)
      return np.take_along_axis(idx, o, -1), np.take_along_axis(w, o, -1)

  def _check(got, exp, tag=""):
      assert isinstance(got, tuple) and len(got) == 2, f"{tag}应返回 (idx, weights)"
      gi, gw = _canon(*got)
      ei, ew = _canon(*exp)
      assert gi.shape == ei.shape, f"{tag}idx 形状应为 {ei.shape}，得到 {gi.shape}"
      assert np.array_equal(gi, ei), f"{tag}选中的专家不对：\n期望 {ei.tolist()}\n得到 {gi.tolist()}"
      assert np.all(np.isfinite(gw)), f"{tag}权重出现 inf/nan（softplus 是否数值稳定？）"
      assert np.allclose(gw, ew, atol=1e-9), f"{tag}权重最大误差 {np.abs(gw - ew).max():.3e}"

  def _case(seed, T=6, E=16, V=50, k=4):
      r = np.random.default_rng(seed)
      tid2eid = np.stack([r.choice(E, size=k, replace=False) for _ in range(V)])
      return r.normal(size=(T, E)), r.integers(0, V, size=T), tid2eid, r.normal(scale=0.2, size=E)
`

const QB_SETUP = py`
  import numpy as np

  def _ref_qb(scores, bias, k):
      m, n = scores.shape
      q = m * k // n
      choice = scores + bias
      alpha = -np.sort(-choice, axis=1)[:, k]
      margin = scores - alpha[:, None]
      thr = -np.sort(-margin, axis=0)[q]
      bh = -thr
      return bh - bh.mean()

  def _loads(scores, bias, k):
      idx = np.argsort(-(scores + bias), axis=1, kind="stable")[:, :k]
      return np.bincount(idx.ravel(), minlength=scores.shape[1])

  def _maxvio(load):
      return (load.max() - load.mean()) / load.mean()

  def _stream(seed, steps, m, n):
      r = np.random.default_rng(seed)
      pop = np.linspace(0.9, -0.9, n)
      return 1 / (1 + np.exp(-(pop + r.normal(size=(steps, m, n)))))

  S8 = np.array([[0.56, 0.9, 0.32, 0.52], [0.8, 0.67, 0.34, 0.51], [0.52, 0.25, 0.24, 0.44], [0.6, 0.82, 0.53, 0.45],
                 [0.9, 0.62, 0.59, 0.26], [0.7, 0.84, 0.72, 0.48], [0.76, 0.63, 0.73, 0.15], [0.67, 0.57, 0.47, 0.7]])
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'v4',
      title: 'DeepSeek-V4 的 MoE：四处改动',
      body: md`
        DeepSeek-V4（2026）在 DeepSeekMoE 的基础上做了四处改动（论文 §2.1）：

        1. **亲和度激活**：$s_{i,t} = \operatorname{Sigmoid}(\cdot)$ 改为 $\sqrt{\operatorname{Softplus}(\cdot)}$；
        2. **均衡**：继续用无辅助损失策略（bias 更新速度 0.001），再加一个**很小的序列级均衡损失**（权重 0.0001）防止单条序列内的极端不均衡；
        3. **去掉节点限制**：取消「每个 token 最多发往 $M$ 个节点」的约束，同时重新设计并行策略来维持训练效率；
        4. **Hash 路由**：把最前面几层的**稠密 FFN** 换成 MoE，这些层的专家由「关于输入 token id 的预定义哈希函数」决定——配置里是**前 3 个 MoE 层**。

        | | V4-Flash | V4-Pro |
        |---|---|---|
        | 层数 / 隐藏维 | 43 / 4096 | 61 / 7168 |
        | 路由 + 共享专家 | 256 + 1 | 384 + 1 |
        | 每 token 激活路由专家 | 6 | 6 |
        | 专家中间维 | 2048 | 3072 |
        | \`routed_scaling_factor\`（config） | 1.5 | 2.5 |
        | 总参数 / 激活参数 | 284B / 13B | 1.6T / 49B |

        官方推理代码的门控：\`scores = softplus(logits).sqrt()\`；bias 只加在选择分数上；权重从原始 scores gather 后归一化再乘 \`route_scale\`——与 V3 的套路一致，只换了激活函数。
        路由专家权重以 FP4 存储（config 中 \`expert_dtype: "fp4"\`）。
      `,
      viz: ActivationFnViz,
      after: md`
        > [!WARN] 关于「为什么换成 Sqrt(Softplus)」
        > V4 论文**没有给出**这个改动的动机或消融。上图只是函数形状的对比：sigmoid 在正区间饱和、导数趋于 0；$\sqrt{\operatorname{softplus}}$ 在正区间约为 $\sqrt{x}$、持续可分，在负区间约为 $e^{x/2}$、比 sigmoid 平缓。面试中如被问到，应先说明「论文未解释」，再给出这类分析，而不是编一个官方理由。
      `,
      checkpoint: {
        id: 'cp-v4',
        type: 'multi',
        prompt: md`相对 DeepSeek-V3，DeepSeek-V4 的 MoE 做了哪些改动？（多选）`,
        options: [
          '亲和度激活从 Sigmoid 改为 Sqrt(Softplus)',
          '取消每个 token 的路由目标节点数限制',
          '前 3 个 MoE 层使用基于 token id 的 Hash 路由，替代原来的稠密 FFN 层',
          '放弃无辅助损失策略，改用 Switch 式辅助损失（α = 0.01）',
          '去掉共享专家',
        ],
        answer: [0, 1, 2],
        explanation: md`A、B、C 均为论文原文。D 错：V4 仍用无辅助损失策略，只额外加了权重 0.0001 的序列级均衡损失。E 错：V4-Flash / Pro 每层都有 1 个共享专家。`,
      },
    },
    {
      id: 'hash',
      title: 'Hash 路由：用查表代替学习',
      body: md`
        Hash 路由的思想来自 *Hash Layers For Large Sparse Models*（Roller et al., 2021）：专家不由 router 学出来，而由 token id 经固定哈希决定。
        DeepSeek-V4 官方推理代码中的实现非常直接：

        \`\`\`python
        # layer_id < n_hash_layers（=3）时
        indices = self.tid2eid[input_ids]          # (T, k)：词表大小 × 激活专家数 的固定查找表
        weights = original_scores.gather(1, indices)
        weights /= weights.sum(dim=-1, keepdim=True)   # 权重仍来自 router 的 sqrt(softplus) 分数
        weights *= self.route_scale
        \`\`\`

        - \`tid2eid\` 是 \`requires_grad=False\` 的 int32 参数，数值随 checkpoint 提供；hash 层没有均衡 bias；
        - **只有「选谁」是固定的**，「权重多大」仍由可学习的 router 决定。

        **好处**（分析）：选择与训练无关，不会坍缩，也不需要均衡机制；路由在拿到 token id 时就已确定，通信可以提前规划。
        **代价**：同一个 token 永远去同一组专家，不能按上下文区分（「bank」在金融和河岸里走同样的专家），所以只放在最浅的几层——那里的隐状态最接近 token 本身。
        **负载**：token 频率服从 Zipf 分布，「词表上均匀」不等于「流量上均匀」；V4 论文只说是「预定义的哈希函数」，**\`tid2eid\` 如何构造未作说明**。
      `,
      viz: HashRoutingViz,
      after: md`
        > [!NOTE] 为什么放在前几层
        > V3 的前 3 层是稠密 FFN（\`first_k_dense_replace: 3\`），通常的解释是浅层路由学不好、容易不均衡。V4 把它们换成 Hash 路由的 MoE：既获得了 MoE 的参数容量，又绕开了浅层的学习路由问题。这一解读是分析而非论文原话。
      `,
      checkpoint: {
        id: 'cp-hash',
        type: 'judge',
        prompt: md`在 DeepSeek-V4 的 Hash 路由层中，每个 token 被送往的专家只取决于它的 token id，而门控权重仍由 router 的分数计算。`,
        answer: true,
        explanation: md`正确。官方推理代码中 hash 层的 \`indices = tid2eid[input_ids]\`，但 \`weights\` 仍从 \`sqrt(softplus)\` 的 router 分数中 gather 并归一化。因此 router 在 hash 层依然会被训练（学的是权重，不是选择）。`,
      },
    },
    {
      id: 'k3',
      title: 'Kimi K3：Stable LatentMoE，896 选 16',
      body: md`
        Kimi K3（arXiv 2607.24653，2026-07）：2.8T 总参数、每 token 约 104B 激活（Table 1 记为 2.78T / 104.2B），1M 上下文。MoE 部分称为 **Stable LatentMoE**：

        | 项 | 值 |
        |---|---|
        | 路由专家 / 每 token 激活 | 896 / 16（稀疏度 56） |
        | 共享专家 | 2 个，全宽 |
        | 隐藏维 $d$ / latent 维 $\ell$ | 7168 / 3584（$0.5\times$） |
        | 每个专家的 MoE 隐藏维 | 3072 |

        **LatentMoE**：路由专家不在全宽 $d$ 上算，而是先降维到 latent 空间：

        $$
        z = W_{\downarrow} x \in \mathbb{R}^{\ell},\qquad u = \sum_{i \in \mathcal{T}_k(x)} p_i\, E^{\text{routed}}_i(z),\qquad
        y = \sum_{j=1}^{N_s} E^{\text{shared}}_j(x) + W_{\uparrow}\operatorname{RMSNorm}(u)
        $$

        router 仍作用在全宽 $x$ 上（$s = \operatorname{Sigmoid}(W_r x)$）。论文的说法是：每个被选中的专家不必接收完整的 $d$ 维 token，于是高路由倍数（16 个专家）的代价可控。

        **「Stable」解决两个失效模式**（论文 §2.3）：
        1. 路由支路串了约 4 个连续矩阵乘，病态，在 2.8T 规模下内部激活爆炸 → 在聚合与上投影之间加 **RMSNorm**，并用有界激活 **SiTU-GLU**：$\big[\beta_1\tanh(W_g x/\beta_1)\odot\sigma(W_g x)\big]\odot\big[\beta_2\tanh(W_u x/\beta_2)\big]$，$\beta_1=4,\ \beta_2=25$，标量响应 $|f|\le \beta_1\beta_2 = 100$；
        2. 在近 $10^3$ 个专家上，sign 规则的 bias 更新不再表现良好 → **Quantile Balancing**（下一节）。
      `,
      checkpoint: {
        id: 'cp-k3',
        type: 'single',
        prompt: md`Kimi K3 的 LatentMoE 中，下列哪一项的输入是全宽的 $x$（维度 $d$），而**不是** latent $z$？`,
        options: ['路由专家 $E^{\\text{routed}}_i$', 'router（$s = \\operatorname{Sigmoid}(W_r x)$）和共享专家', '$W_{\\uparrow}$ 之前的 RMSNorm', '所有专家都用 latent $z$'],
        answer: 1,
        explanation: md`router 与 2 个共享专家都直接作用于全宽 $x$；只有路由专家在 $\ell = 3584$ 维的 latent 空间中计算，它们的加权和 $u$ 经 RMSNorm 后由 $W_\uparrow$ 投回 $d$ 维。`,
      },
    },
    {
      id: 'qb',
      title: 'Quantile Balancing：没有步长的 bias 均衡',
      body: md`
        K3 沿用「bias 只管选择、不进权重」的框架：$\mathcal{T}_i = \operatorname{argtop}_k(s_i + b)$，权重 $p_{i,j} = s_{i,j}/\sum_{r\in\mathcal{T}_i} s_{i,r}$。区别在 bias 怎么更新。

        sign 规则 $b_j \leftarrow b_j + \gamma\,\operatorname{sign}(\bar\ell - \ell_j)$ 的问题：$\gamma$ 在「适应慢」与「负载震荡」之间权衡。QB 改为**直接解出**让每个专家恰好分到目标负载 $q = mk/n$ 的 bias（$m$ 个 token、$n$ 个专家）：

        1. 对选择分数取 Top-$(k+1)$：前 $k$ 个是实际路由，第 $k+1$ 个是 token $i$ 的 **cutoff** $\alpha_i$——专家分数要超过它才能进入 top-k；
        2. **margin** $= s_{i,j} - \alpha_i$。若固定各 cutoff，专家 $j$ 的负载 $= \#\{i : s_{i,j} + b_j > \alpha_i\}$，随 $b_j$ 单调；
        3. 要让负载恰为 $q$（无并列时），$-b_j$ 应取 margin 的**第 $q+1$ 大**，即 margins 的 $(1-k/n)$ 分位数：$\hat b_j = -\operatorname{quantile}_{1-k/n}\big(s_{:,j} - \alpha\big)$；
        4. 减去均值：$b = \hat b - \operatorname{mean}(\hat b)$（公共偏移不改变 top-k）。

        **因果性**：新 bias 只在**下一步**生效，一个 batch 永远不会用由它自己算出的 bias 路由；推理时 bias 冻结。
        **规模化**：全局精确分位数不现实，论文用每个专家 margin 的**直方图**估计，一次 all-reduce 汇总各 rank 的分桶计数（每个专家「几百个桶」）——具体桶数与误差界见原文附录。
      `,
      viz: QuantileBalanceViz,
      after: md`
        > [!NEW] 和 V3 的 sign 规则比
        > sign 规则每步只走 $\gamma$，偏差大时追得慢、偏差小时又来回跳；QB 每步直接「解」出让负载达标的 bias，没有步长超参，一步就能消掉大部分不均衡。
        > 一步之后通常不是严格均衡——所有专家的 bias 同时变化会让 cutoff 也随之移动——但在数据流上会迅速收敛到采样噪声水平。
        > 本节公式按论文 §2.3.3 的描述转述，分位数的精确取法与直方图细节**以原文为准**。
      `,
      checkpoint: {
        id: 'cp-qb',
        type: 'numeric',
        prompt: md`一个 batch 有 $m = 4096$ 个 token，$n = 896$ 个路由专家，每 token 选 $k = 16$ 个。QB 中每个专家的目标负载 $q = mk/n$ 是多少？（保留两位小数）`,
        answer: 73.14,
        tolerance: 0.001,
        unit: '个 token',
        explanation: md`$q = 4096 \times 16 / 896 = 65536/896 \approx 73.14$。对应分位数水平 $1 - k/n = 1 - 16/896 \approx 0.982$：只有约 1.8% 的 token 的 margin 应高于阈值。实际训练的全局 batch 远大于此，所以要用直方图近似分位数。`,
      },
    },
    {
      id: 'trend',
      title: '激活比例越来越低？算清楚再说',
      body: md`
        「MoE 越来越稀疏」这句话要分两个口径：

        | 模型 | 时间 | 总参数 | 激活参数 | 激活比例 | 路由专家 k / N |
        |---|---|---|---|---|---|
        | Mixtral 8×7B | 2023-12 | 47B | 13B | 27.7% | 2 / 8 |
        | DeepSeek-V2 | 2024-05 | 236B | 21B | 8.9% | 6 / 160 |
        | DeepSeek-V3 | 2024-12 | 671B | 37B | 5.5% | 8 / 256 |
        | Qwen3-235B-A22B | 2025-04 | 235B | 22B | 9.4% | 8 / 128 |
        | Kimi K2 | 2025-07 | 1T | 32B | 3.2% | 8 / 384 |
        | gpt-oss-120b | 2025-08 | 117B | 5.1B | 4.4% | 4 / 128 |
        | Qwen3.5-397B-A17B | 2026-02 | 397B | 17B | 4.3% | 10 / 512 |
        | DeepSeek-V4-Flash | 2026 | 284B | 13B | 4.6% | 6 / 256 |
        | DeepSeek-V4-Pro | 2026 | 1.6T | 49B | 3.1% | 6 / 384 |
        | Kimi K3 | 2026-07 | 2.8T | 104B | 3.7% | 16 / 896 |

        - **路由稀疏度 $k/N$** 确实一路走低：25% → 3.1%（V3）→ 1.56%（V4-Pro）/ 1.79%（K3）；
        - **整体激活比例**在 2025 年后基本停在 3%–5%：注意力、嵌入、共享专家是每个 token 都要算的「固定开销」，路由专家再稀疏也压不掉它们。K3 的 3.7% 就高于 K2 的 3.2%。

        越稀疏的代价：每个专家拿到的 token 更少（GEMM 更小、更难用满算力）、均衡更难（于是有了 QB）、all-to-all 派发更碎（于是有了 LatentMoE 这类降低派发维度的设计）。
      `,
      viz: ActivationTrendViz,
      after: md`
        > [!INTERVIEW] 回答「MoE 的趋势」时的加分点
        > 不要只说「越来越稀疏」。给出两个口径的数字，指出整体激活比例的平台期及其原因（固定开销），再讲超稀疏带来的三个工程问题：小 GEMM 效率、均衡、通信——以及 V4（Hash 路由、去节点限制）和 K3（LatentMoE、QB）分别如何应对。
      `,
      checkpoint: {
        id: 'cp-trend',
        type: 'numeric',
        prompt: md`DeepSeek-V4-Pro 总参数 1.6T、每 token 激活 49B。激活比例是百分之多少？`,
        answer: 3.0625,
        tolerance: 0.01,
        unit: '%',
        explanation: md`$49 / 1600 = 3.0625\%$。对比 V3 的 $37/671 \approx 5.5\%$，总参数增加 2.4 倍，激活参数只增加 1.3 倍。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-v4-config',
      type: 'fill',
      difficulty: 2,
      prompt: md`DeepSeek-V4-Pro 每个 MoE 层有 ____(1)____ 个路由专家、1 个共享专家，每 token 激活 ____(2)____ 个路由专家，专家中间维 ____(3)____；前 ____(4)____ 个 MoE 层使用 Hash 路由。`,
      blanks: [
        { label: '(1)', answers: ['384'], numeric: true, tolerance: 0 },
        { label: '(2)', answers: ['6'], numeric: true, tolerance: 0 },
        { label: '(3)', answers: ['3072'], numeric: true, tolerance: 0 },
        { label: '(4)', answers: ['3'], numeric: true, tolerance: 0 },
      ],
      explanation: md`论文 §4.2.1：V4-Pro 每层 1 个共享 + 384 个路由专家，中间维 3072，每 token 激活 6 个；V4-Flash 为 256 个路由专家、中间维 2048，同样激活 6 个。两者都在前 3 个 MoE 层使用 Hash 路由。`,
    },
    {
      id: 'q-k3-ratio',
      type: 'numeric',
      difficulty: 1,
      prompt: md`Kimi K3 每 token 从 896 个路由专家中激活 16 个。路由专家的激活比例 $k/N$ 是百分之多少？（保留两位小数）`,
      answer: 1.79,
      tolerance: 0.005,
      unit: '%',
      explanation: md`$16/896 = 1/56 \approx 1.79\%$，论文称稀疏度（sparsity）为 56。对比 V3 的 $8/256 = 3.125\%$、V4-Pro 的 $6/384 = 1.5625\%$。`,
    },
    {
      id: 'q-hash-judge',
      type: 'judge',
      difficulty: 2,
      prompt: md`因为 Hash 路由层的专家选择由 token id 查表决定，所以这些层不需要 router，也就没有任何可学习的门控参数。`,
      answer: false,
      explanation: md`错误。V4 的官方推理代码中，hash 层的 \`indices\` 来自 \`tid2eid[input_ids]\`，但门控 \`weights\` 仍从 router 的 \`sqrt(softplus)\` 分数中 gather、归一化并乘 \`route_scale\`。固定的只是「选谁」，router 权重依然参与训练。`,
    },
    {
      id: 'q-qb-quantile',
      type: 'single',
      difficulty: 2,
      prompt: md`Quantile Balancing 中，$m$ 个 token、$n$ 个专家、top-$k$。专家 $j$ 的新 bias 取 margin $s_{:,j} - \alpha$ 的哪个分位数的相反数？`,
      options: ['中位数（0.5 分位）', '$k/n$ 分位数', '$1 - k/n$ 分位数', '$1 - 1/n$ 分位数'],
      answer: 2,
      explanation: md`目标负载 $q = mk/n$，即只有 $q/m = k/n$ 比例的 token 的 margin 应高于阈值，阈值就是 $1-k/n$ 分位数（第 $q+1$ 大的 margin）。B 把方向弄反了（那会让 $1-k/n$ 比例的 token 都选它）；D 忽略了 $k$。`,
    },
    {
      id: 'q-qb-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 Quantile Balancing 一次更新的步骤按顺序排列：`,
      items: [
        '用当前 bias 计算选择分数 $s + b$，取 Top-$(k+1)$',
        '前 $k$ 个作为本步路由；第 $k+1$ 个作为每个 token 的 cutoff $\\alpha_i$',
        '计算 margin $s_{i,j} - \\alpha_i$',
        '对每个专家取 margin 的 $1-k/n$ 分位数（大规模时用直方图 + all-reduce 估计），取相反数得 $\\hat b$',
        '减去均值得到新 bias，供下一步使用',
      ],
      explanation: md`本步路由用的是旧 bias；新 bias 只影响下一步（因果性）。减均值不改变 top-k，只去掉一个不可辨识的公共偏移。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把模型与它 MoE 设计中最有代表性的特点配对：`,
      pairs: [
        ['Mixtral 8×7B', '8 选 2，softmax 后在 top-k 内重归一化'],
        ['DeepSeek-V3', 'sigmoid 亲和度 + sign 规则 bias 均衡 + 节点受限路由'],
        ['DeepSeek-V4', 'Sqrt(Softplus) 亲和度 + 前 3 个 MoE 层 Hash 路由'],
        ['Kimi K3', 'LatentMoE（路由专家在 0.5d 的 latent 空间计算）+ Quantile Balancing'],
        ['Qwen3-235B-A22B', '128 选 8，没有共享专家'],
      ],
      explanation: md`这五个设计基本串起了 MoE 的演进：门控归一化 → 无辅助损失均衡 → 去掉浅层学习路由 → 超稀疏下的降维与无步长均衡。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面是 V4 风格的门控（\`n_hash_layers=3\` 表示第 0、1、2 层用 Hash 路由），有 **2 处** bug。点出有问题的行：`,
      code: py`
        import numpy as np

        def v4_gate(logits, input_ids, layer_id, tid2eid, bias, k, n_hash_layers=3, route_scale=1.5):
            scores = np.sqrt(np.logaddexp(0.0, logits))         # sqrt(softplus)
            if layer_id <= n_hash_layers:
                idx = tid2eid[input_ids]                         # 查表
            else:
                idx = np.argsort(-(scores + bias), axis=-1)[:, :k]
            w = np.take_along_axis(scores + bias, idx, axis=-1)
            w = w / w.sum(axis=-1, keepdims=True)
            return idx, w * route_scale
      `,
      bugLines: [5, 9],
      explanation: md`第 5 行应为 \`layer_id < n_hash_layers\`：层号从 0 开始，\`<=\` 会让第 3 层（第 4 个 MoE 层）也走查表（官方代码为 \`layer_id < args.n_hash_layers\`）。第 9 行应从原始 \`scores\` 取权重，bias 只用于选择。第 4 行用 \`np.logaddexp(0, x)\` 计算 softplus 是数值稳定的正确写法。`,
    },
    {
      id: 'q-pick-ratio',
      type: 'pick',
      difficulty: 2,
      prompt: md`点出**整体激活比例（激活参数 / 总参数）低于 4%** 的模型：`,
      items: ['Mixtral 8×7B（47B/13B）', 'DeepSeek-V3（671B/37B）', 'Kimi K2（1T/32B）', 'gpt-oss-120b（117B/5.1B）', 'Qwen3.5-397B-A17B', 'DeepSeek-V4-Flash（284B/13B）', 'DeepSeek-V4-Pro（1.6T/49B）', 'Kimi K3（2.8T/104B）'],
      cols: 2,
      answer: [2, 6, 7],
      explanation: md`K2 3.2%、V4-Pro 3.06%、K3 3.71% 低于 4%；gpt-oss-120b 4.36%、Qwen3.5 $17/397 \approx 4.28\%$、V4-Flash 4.58%、V3 5.5%、Mixtral 27.7% 都不低于 4%。注意 V4-Flash 的比例反而高于 V3 之后的大多数模型——小模型里固定开销占比更大。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 Quantile Balancing 的一次 bias 更新（\`scores\` 形状 $(m, n)$，测试数据可整除；有 2 行干扰项）：`,
      lines: [
        'def qb_update(scores, bias, k):',
        '    m, n = scores.shape',
        '    q = m * k // n',
        '    alpha = -np.sort(-(scores + bias), axis=1)[:, k]',
        '    margin = scores - alpha[:, None]',
        '    b_hat = np.sort(-margin, axis=0)[q]',
        '    return b_hat - b_hat.mean()',
      ],
      distractors: ['    alpha = -np.sort(-(scores + bias), axis=1)[:, k - 1]', '    return b_hat'],
      explanation: md`\`np.sort(-margin, axis=0)[q]\` 是「每列第 $q+1$ 大的 margin 取负」，正是 $\hat b$。干扰项 1 取的是第 $k$ 大（最后一个被选中的分数），不是 cutoff；干扰项 2 漏掉减均值——不影响本步 top-k，但 bias 会随时间漂移。`,
    },
    {
      id: 'q-plateau',
      type: 'single',
      difficulty: 3,
      prompt: md`路由稀疏度 $k/N$ 从 Mixtral 的 25% 降到 K3 的 1.79%，但整体激活比例在 2025 年后停在 3%–5% 附近。最主要的原因是：`,
      options: [
        '新模型的路由专家变大了',
        '注意力、嵌入 / 输出头、共享专家对每个 token 都要计算，构成固定开销',
        '新模型都改用了稠密 FFN',
        '激活参数的统计口径改变了',
      ],
      answer: 1,
      explanation: md`激活参数 = 固定部分 + $k$ 个路由专家。$N$ 再大也只减少总参数里「没被激活」的部分，固定部分不会变小；当路由专家的激活份额已经很小时，固定开销主导了比例。A 不是主因（专家其实更细）；C 错；D 不成立，各家口径差异不足以解释这一趋势。`,
    },
    {
      id: 'q-qb-mean',
      type: 'single',
      difficulty: 3,
      prompt: md`QB 在得到 $\hat b$ 后还要减去其均值 $b = \hat b - \operatorname{mean}(\hat b)$。这一步：`,
      options: [
        '会改变本步每个 token 的 top-k 选择，用于进一步均衡',
        '不改变任何 token 的 top-k 选择（所有专家加同一个常数，排序不变），只是去掉一个公共偏移',
        '让 bias 进入门控权重时和为 0',
        '等价于 sign 规则中的步长 γ',
      ],
      answer: 1,
      explanation: md`对所有专家加同一常数，每个 token 的 $s + b$ 排序不变，top-k 不变；减均值让 bias 保持「零均值」，避免整体漂移、数值可辨识。C 错：bias 根本不进权重；D 与此无关，QB 没有步长。`,
    },
    {
      id: 'q-latent',
      type: 'multi',
      difficulty: 3,
      prompt: md`关于 Kimi K3 的 Stable LatentMoE，下列说法正确的是（多选）：`,
      options: [
        '路由专家在 $\\ell = 3584$ 维（$0.5d$）的 latent 空间中计算',
        '路由专家输出的加权和经 RMSNorm 后再由 $W_\\uparrow$ 投回 $d$ 维',
        'SiTU-GLU 用 tanh 对门支路与 up 支路做软截断，使标量响应有界（$|f| \\le 100$）',
        'K3 的 router 使用 softmax 打分',
        'K3 用 sign 规则更新 bias，步长 γ = 0.001',
      ],
      answer: [0, 1, 2],
      explanation: md`A、B、C 来自论文 §2.3。D 错：$s = \operatorname{Sigmoid}(W_r x)$。E 错：K3 用 Quantile Balancing，论文指出近千个专家时 sign 规则的 bias 更新不再表现良好，且 QB 没有步长 γ。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-v4-gate',
      title: '写出 DeepSeek-V4 的门控（含 Hash 层）',
      intro: md`
        按官方推理代码的逻辑实现 \`v4_gate\`：
        - 分数 $s = \sqrt{\operatorname{softplus}(\text{logits})}$（用 \`np.logaddexp(0, x)\` 保证数值稳定）；
        - \`layer_id < n_hash_layers\`：专家 = \`tid2eid[input_ids]\`；否则按 $s + b$ 取 top-k；
        - 权重从原始 $s$ 取，归一化后乘 \`route_scale\`。
      `,
      template: py`
        import numpy as np

        def v4_gate(logits, input_ids, layer_id, tid2eid, bias, k, n_hash_layers=3, route_scale=1.5):
            scores = {{1}}                                  # (T, E) sqrt(softplus)
            if layer_id < n_hash_layers:
                idx = {{2}}                                 # (T, k) 查表
            else:
                idx = {{3}}                                 # bias 只用于选择
            w = {{4}}                                       # 原始分数
            return idx, {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`计算 $\sqrt{\operatorname{softplus}(x)}$，softplus 要数值稳定。`,
          answer: 'np.sqrt(np.logaddexp(0.0, logits))',
          accept: ['np.sqrt(np.logaddexp(0, logits))', 'np.sqrt(np.maximum(logits, 0) + np.log1p(np.exp(-np.abs(logits))))'],
          explain: md`$\operatorname{softplus}(x) = \log(1+e^x)$；直接写 \`np.log(1 + np.exp(x))\` 在 $x > 709$ 时溢出。\`logaddexp(0, x)\` 等价且稳定。`,
        },
        {
          blank: 2,
          instruction: md`Hash 层：按 token id 查表。`,
          answer: 'tid2eid[input_ids]',
          explain: md`\`tid2eid\` 形状 (词表大小, k)，花式索引直接得到 (T, k)。与 logits、bias 都无关。`,
        },
        {
          blank: 3,
          instruction: md`普通层：用 $s + b$ 取 top-k。`,
          answer: 'np.argsort(-(scores + bias), axis=-1)[:, :k]',
          accept: ['np.argsort(-(scores + bias), axis=1)[:, :k]', 'np.argpartition(-(scores + bias), k - 1, axis=-1)[:, :k]'],
          explain: md`V4 去掉了节点限制，所以这里不再需要 V3 那样的分组屏蔽。`,
        },
        {
          blank: 4,
          instruction: md`从原始分数中取出被选中专家的值。`,
          answer: 'np.take_along_axis(scores, idx, axis=-1)',
          accept: ['np.take_along_axis(scores, idx, -1)', 'np.take_along_axis(scores, idx, axis=1)'],
          explain: md`Hash 层与普通层共用这一行：两种层的权重都来自 router 分数。`,
        },
        {
          blank: 5,
          instruction: md`归一化并乘缩放因子。`,
          answer: 'w / w.sum(axis=-1, keepdims=True) * route_scale',
          accept: ['w / w.sum(-1, keepdims=True) * route_scale', 'route_scale * w / w.sum(axis=-1, keepdims=True)'],
          explain: md`V4-Flash 的 \`routed_scaling_factor\` 为 1.5，V4-Pro 为 2.5（官方 config）。`,
        },
      ],
      setup: V4_SETUP,
      tests: [
        {
          name: 'Hash 层：只看 token id',
          code: py`
            logits, ids, table, bias = _case(0)
            idx1, _ = v4_gate(logits, ids, 0, table, bias, 4)
            idx2, _ = v4_gate(-logits * 3, ids, 2, table, bias * 0, 4)
            assert np.array_equal(np.asarray(idx1), table[ids]), "hash 层的专家应等于 tid2eid[input_ids]"
            assert np.array_equal(np.asarray(idx1), np.asarray(idx2)), "hash 层的选择不应依赖 logits 或 bias"
            _check(v4_gate(logits, ids, 1, table, bias, 4), _ref_v4(logits, ids, 1, table, bias, 4))
          `,
        },
        {
          name: '普通层 + 数值稳定',
          code: py`
            logits, ids, table, bias = _case(1)
            logits[0, :3] = [800.0, 750.0, -900.0]
            _check(v4_gate(logits, ids, 3, table, bias, 4, route_scale=2.5), _ref_v4(logits, ids, 3, table, bias, 4, route_scale=2.5))
          `,
        },
      ],
      outro: md`
        对比 V3 的门控：激活函数换了、分组屏蔽没了、前几层多了一个查表分支——其余（bias 只管选择、top-k 内归一化、缩放）完全一样。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-v4-gate',
      title: 'DeepSeek-V4 门控：Sqrt(Softplus) + Hash 路由层',
      difficulty: 'medium',
      source: '前沿模型手撕 · DeepSeek-V4',
      timeLimitMin: 20,
      description: md`
        实现 DeepSeek-V4 风格的 MoE 门控（参照官方 \`inference/model.py\` 的 \`Gate.forward\`）：

        \`\`\`python
        def v4_moe_gate(logits, input_ids, layer_id, tid2eid, bias, k, n_hash_layers=3, route_scale=1.5):
            """返回 (idx, weights)，形状均为 (T, k)"""
        \`\`\`

        - \`logits\`：$(T, E)$；\`input_ids\`：$(T,)$ 整数；\`tid2eid\`：$(V, k)$ 整数查找表；\`bias\`：$(E,)$
        - 分数 $s = \sqrt{\operatorname{softplus}(\text{logits})}$，**必须数值稳定**（隐藏测试有 $|x| \approx 10^3$ 的 logits）
        - \`layer_id < n_hash_layers\`：\`idx = tid2eid[input_ids]\`（不看 logits，也不用 bias）
        - 否则：按 $s + b$ 取 top-$k$
        - 两种层的权重都是：选中专家的**原始** $s$，在 $k$ 个内归一化后乘 \`route_scale\`

        \`idx\` 每行内的顺序不限。
      `,
      starter: py`
        import numpy as np

        def v4_moe_gate(logits, input_ids, layer_id, tid2eid, bias, k, n_hash_layers=3, route_scale=1.5):
            # TODO：sqrt(softplus) 分数；hash 层查表 / 普通层 s+b 取 top-k；原始分数归一化 × route_scale
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def v4_moe_gate(logits, input_ids, layer_id, tid2eid, bias, k, n_hash_layers=3, route_scale=1.5):
            scores = np.sqrt(np.logaddexp(0.0, logits))              # 稳定的 sqrt(softplus)
            if layer_id < n_hash_layers:
                idx = np.asarray(tid2eid)[np.asarray(input_ids)]      # Hash 层：只由 token id 决定
            else:
                idx = np.argsort(-(scores + bias), axis=-1)[:, :k]    # bias 只用于选择
            w = np.take_along_axis(scores, idx, axis=-1)              # 权重始终来自原始分数
            w = w / w.sum(axis=-1, keepdims=True) * route_scale
            return idx, w
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，对应官方 Gate.forward 的逻辑）',
          code: py`
            import torch
            import torch.nn.functional as F

            def v4_moe_gate(x, weight, input_ids, layer_id, tid2eid, bias, k, n_hash_layers=3, route_scale=1.5):
                scores = F.linear(x.float(), weight.float())          # router 在 fp32 下计算
                scores = F.softplus(scores).sqrt()                    # sqrt(softplus)
                original_scores = scores
                if layer_id < n_hash_layers:
                    indices = tid2eid[input_ids]                      # (T, k) 固定查找表
                else:
                    if bias is not None:
                        scores = scores + bias                        # 只影响 topk
                    indices = scores.topk(k, dim=-1)[1]
                weights = original_scores.gather(1, indices)
                weights = weights / weights.sum(dim=-1, keepdim=True)
                return weights * route_scale, indices
          `,
        },
      ],
      explanation: md`
        **两个分支只在「选谁」上不同**：hash 层查表，普通层按 $s+b$ 取 top-k；「权重多大」两者一致，都用原始分数归一化再缩放。

        **数值稳定**：$\operatorname{softplus}(x) = \max(x, 0) + \log(1 + e^{-|x|})$，NumPy 里直接 \`np.logaddexp(0, x)\`；朴素的 \`np.log(1 + np.exp(x))\` 在 $x > 709$ 时得到 inf。

        **易错点**：hash 层也加 bias 或按 logits 选；层号边界写成 \`<=\`；权重用了 $s + b$。

        **复杂度**：$O(TE\log E)$（普通层），hash 层 $O(Tk)$。
      `,
      complexity: { time: 'O(T·E·log E)', space: 'O(T·E)' },
      hints: [
        md`分数：\`np.sqrt(np.logaddexp(0.0, logits))\`。`,
        md`\`if layer_id < n_hash_layers: idx = tid2eid[input_ids]\`，否则 \`argsort(-(scores + bias))[:, :k]\`。`,
        md`权重：\`np.take_along_axis(scores, idx, -1)\`，再除以行和、乘 \`route_scale\`。`,
      ],
      setup: V4_SETUP,
      tests: [
        {
          name: 'Hash 层：专家 = tid2eid[input_ids]',
          code: py`
            logits, ids, table, bias = _case(10)
            idx, w = v4_moe_gate(logits, ids, 0, table, bias, 4)
            assert np.array_equal(np.asarray(idx), table[ids]), f"hash 层的专家应为 tid2eid[input_ids]：\n期望 {table[ids].tolist()}\n得到 {np.asarray(idx).tolist()}"
            _check((idx, w), _ref_v4(logits, ids, 0, table, bias, 4), "hash 层：")
          `,
        },
        {
          name: '普通层：bias 选择、原始分数加权',
          code: py`
            logits, ids, table, bias = _case(11)
            _check(v4_moe_gate(logits, ids, 5, table, bias, 4, route_scale=2.5), _ref_v4(logits, ids, 5, table, bias, 4, route_scale=2.5), "普通层：")
          `,
        },
        {
          name: '层号边界：layer_id == n_hash_layers 是普通层',
          code: py`
            logits, ids, table, bias = _case(12)
            _check(v4_moe_gate(logits, ids, 3, table, bias, 4), _ref_v4(logits, ids, 3, table, bias, 4), "layer_id=3：")
          `,
        },
        {
          name: '数值稳定（|logits| ≈ 1e3）',
          hidden: true,
          code: py`
            logits, ids, table, bias = _case(13)
            logits[:, 0] = 1000.0
            logits[:, 1] = -1000.0
            logits[2, 5] = 900.0
            for layer in (0, 4):
                _check(v4_moe_gate(logits, ids, layer, table, bias, 4), _ref_v4(logits, ids, layer, table, bias, 4), f"layer={layer}：")
          `,
        },
        {
          name: 'Hash 层不受 bias / logits 影响',
          hidden: true,
          code: py`
            logits, ids, table, bias = _case(14)
            a = np.asarray(v4_moe_gate(logits, ids, 1, table, bias, 4)[0])
            b = np.asarray(v4_moe_gate(logits[:, ::-1].copy(), ids, 1, table, bias + 5.0, 4)[0])
            assert np.array_equal(a, b), "hash 层的选择不应依赖 logits 或 bias"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            for s in range(20, 40):
                k = 2 + s % 4
                logits, ids, table, bias = _case(s, T=8, E=24, V=40, k=k)
                for layer in (0, 2, 3, 7):
                    _check(v4_moe_gate(logits, ids, layer, table, bias, k, route_scale=1.5), _ref_v4(logits, ids, layer, table, bias, k), f"seed={s}, layer={layer}：")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-hash-balance',
          q: md`Hash 层天然没有坍缩问题，那它一定是负载均衡的吗？`,
          a: md`不一定。查表只保证「词表上」的分配是固定的；实际流量取决于 token 频率，而自然语言服从 Zipf 分布，高频 token（标点、虚词）会把它们所在的专家挤爆。表要结合词频构造（例如按频率贪心装箱），而单个超高频 token 的流量无法再拆，只能靠每 token 多个专家（k）摊薄。V4 论文没有说明 \`tid2eid\` 的构造方法。`,
        },
      ],
    },
    {
      id: 'c-qb',
      title: 'Kimi K3 的 Quantile Balancing 更新',
      difficulty: 'hard',
      source: '前沿模型手撕 · Kimi K3',
      timeLimitMin: 25,
      description: md`
        实现一次 Quantile Balancing 的 bias 更新（按 K3 论文 §2.3.3 的描述；论文在大规模训练中用直方图近似分位数，**细节以原文为准**，本题约定精确取值）：

        \`\`\`python
        def quantile_balance_update(scores: np.ndarray, bias: np.ndarray, k: int) -> np.ndarray:
            """scores: (m, n) 亲和度；bias: (n,) 当前 bias；返回新的 bias (n,)"""
        \`\`\`

        1. 目标负载 $q = mk/n$（测试保证整除）；
        2. 选择分数 $c = s + b$；每个 token 的 cutoff $\alpha_i$ = 该行**第 $k+1$ 大**的 $c$；
        3. margin $M_{i,j} = s_{i,j} - \alpha_i$（注意是原始分数减 cutoff）；
        4. 对每个专家 $j$：$\hat b_j = -(\text{第 } q+1 \text{ 大的 } M_{:,j})$；
        5. 返回 $\hat b - \operatorname{mean}(\hat b)$。

        **示例**：论文图 5 的设定（$m=8,\ n=4,\ k=1,\ q=2$）。隐藏测试会检查：与参考实现一致；新 bias 均值为 0；在冷热不均的数据流上，逐步用 QB 更新（每步用上一步算出的 bias 路由）后，负载不均衡度迅速下降，且快于步长 0.01 的 sign 规则。
      `,
      starter: py`
        import numpy as np

        def quantile_balance_update(scores, bias, k):
            m, n = scores.shape
            # TODO：cutoff α（第 k+1 大的 s+b）→ margin → 每列第 q+1 大 → 取负、去均值
            return np.zeros(n)
      `,
      solution: py`
        import numpy as np

        def quantile_balance_update(scores, bias, k):
            m, n = scores.shape
            q = m * k // n                                         # 每个专家的目标负载
            choice = scores + bias
            alpha = -np.sort(-choice, axis=1)[:, k]                # (m,) 第 k+1 大的选择分数 = cutoff
            margin = scores - alpha[:, None]                       # (m, n)
            b_hat = -(-np.sort(-margin, axis=0)[q])                # 每列第 q+1 大的 margin，取负
            return b_hat - b_hat.mean()                            # 公共偏移不改变 top-k
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：单卡精确版；大规模时按论文用直方图 + all-reduce 估计分位数）',
          code: py`
            import torch

            @torch.no_grad()
            def quantile_balance_update(scores: torch.Tensor, bias: torch.Tensor, k: int) -> torch.Tensor:
                m, n = scores.shape
                q = m * k // n
                alpha = (scores + bias).topk(k + 1, dim=1).values[:, k]     # cutoff
                margin = scores - alpha[:, None]
                thr = margin.topk(q + 1, dim=0).values[q]                   # 每列第 q+1 大
                b_hat = -thr
                return b_hat - b_hat.mean()                                 # 下一步才使用
          `,
        },
      ],
      explanation: md`
        **为什么是分位数**：固定 cutoff 后，专家 $j$ 被 token $i$ 选中 ⇔ $s_{i,j} + b_j > \alpha_i$ ⇔ $M_{i,j} > -b_j$。负载是 $-b_j$ 的单调递减函数；令 $-b_j$ 等于第 $q+1$ 大的 margin，恰好 $q$ 个 margin 在它之上（无并列时）——也就是 margin 的 $1-k/n$ 分位数。

        **为什么一步后不一定严格均衡**：所有专家的 bias 同时改变，cutoff $\alpha_i$ 也会跟着变，上面的「固定 cutoff」只是近似；但在数据流上反复应用会很快收敛到采样噪声水平——与 sign 规则相比没有步长 $\gamma$ 需要调。

        **复杂度**：排序 $O(mn\log n + nm\log m)$。大规模时精确分位数不现实，论文按专家维护 margin 直方图，一次 all-reduce 汇总。

        **易错点**：cutoff 取成第 $k$ 大；margin 用 $s+b$ 减 $\alpha$（多算了一次 bias）；忘记取负或去均值；\`np.quantile\` 默认线性插值，不等于「第 $q+1$ 大」。
      `,
      complexity: { time: 'O(m·n·log(mn))', space: 'O(m·n)' },
      hints: [
        md`cutoff：\`alpha = -np.sort(-(scores + bias), axis=1)[:, k]\`（降序第 $k$ 个下标 = 第 $k+1$ 大）。`,
        md`margin：\`scores - alpha[:, None]\`；每列降序后取下标 $q$。`,
        md`\`b_hat = -(那个值)\`，返回 \`b_hat - b_hat.mean()\`。`,
      ],
      setup: QB_SETUP,
      tests: [
        {
          name: '示例：m=8, n=4, k=1',
          code: py`
            b = quantile_balance_update(S8, np.zeros(4), 1)
            exp = np.array([-0.0475, -0.0875, 0.0625, 0.0725])
            assert np.shape(b) == (4,), f"返回形状应为 (4,)，得到 {np.shape(b)}"
            assert np.allclose(b, exp), f"期望 {exp.tolist()}，得到 {np.round(b, 4).tolist()}"
            load = _loads(S8, b, 1)
            assert load.tolist() == [2, 2, 2, 2], f"用新 bias 重新路由后负载应为 [2,2,2,2]，得到 {load.tolist()}"
          `,
        },
        {
          name: '新 bias 均值为 0',
          code: py`
            r = np.random.default_rng(0)
            s = r.uniform(size=(64, 8))
            b = quantile_balance_update(s, r.normal(scale=0.1, size=8), 2)
            assert abs(np.mean(b)) < 1e-12, f"bias 均值应为 0，得到 {np.mean(b):.3e}"
          `,
        },
        {
          name: '随机对拍（含非零旧 bias、k>1）',
          hidden: true,
          code: py`
            r = np.random.default_rng(1)
            for _ in range(40):
                n = int(r.choice([4, 8, 16]))
                k = int(r.integers(1, 4))
                m = n * int(r.integers(2, 20))
                s = r.uniform(size=(m, n))
                old = r.normal(scale=0.1, size=n)
                got = quantile_balance_update(s, old, k)
                exp = _ref_qb(s, old, k)
                assert np.allclose(got, exp), f"m={m}, n={n}, k={k}：最大误差 {np.abs(np.asarray(got) - exp).max():.3e}"
          `,
        },
        {
          name: '数据流上迅速收敛，且快于 sign 规则',
          hidden: true,
          code: py`
            data = _stream(7, 12, 512, 16)
            b_qb = np.zeros(16)
            b_sg = np.zeros(16)
            v_qb, v_sg = [], []
            for t in range(12):
                l_qb = _loads(data[t], b_qb, 2)
                l_sg = _loads(data[t], b_sg, 2)
                v_qb.append(_maxvio(l_qb))
                v_sg.append(_maxvio(l_sg))
                b_qb = quantile_balance_update(data[t], b_qb, 2)       # 只给下一步用
                b_sg = b_sg + 0.01 * np.sign(l_sg.mean() - l_sg)
            assert np.mean(v_qb[1:6]) < 0.5 * v_qb[0], f"QB 前几步 MaxVio 应迅速下降：{np.round(v_qb, 2).tolist()}"
            assert np.mean(v_qb[1:6]) < np.mean(v_sg[1:6]), f"QB 应比 sign(γ=0.01) 收敛更快：QB {np.round(v_qb[1:6], 2).tolist()} vs sign {np.round(v_sg[1:6], 2).tolist()}"
          `,
        },
        {
          name: '不能用 np.quantile 的默认插值',
          hidden: true,
          code: py`
            r = np.random.default_rng(3)
            s = r.uniform(size=(40, 8))
            got = quantile_balance_update(s, np.zeros(8), 2)
            assert np.allclose(got, _ref_qb(s, np.zeros(8), 2)), "阈值应取「第 q+1 大」的 margin（np.quantile 默认的线性插值会得到不同结果）"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-qb-hist',
          q: md`在数千张卡上训练时，QB 的分位数怎么算？`,
          a: md`每个专家的 margin 分布在所有数据并行 rank 上，精确分位数需要全局排序，不现实。K3 论文的做法：每个 rank 为每个专家维护 margin 的直方图（论文称「每个专家几百个桶」），一次 all-reduce 把各 rank 的分桶计数加起来，再在全局直方图上读出 $1-k/n$ 分位数。误差界见原文附录。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-v4',
      freq: 2,
      q: md`DeepSeek-V4 的 MoE 相比 V3 改了什么？`,
      a: md`
        **结论**：四处——亲和度 Sigmoid → Sqrt(Softplus)；保留无辅助损失均衡（bias 速度 0.001）并加权重 0.0001 的序列级均衡损失；去掉路由目标节点数限制；前 3 个 MoE 层用基于 token id 的 Hash 路由，替代 V3 前几层的稠密 FFN。

        **配置**：V4-Pro 384 路由 + 1 共享、激活 6、专家中间维 3072、1.6T / 49B；V4-Flash 256 + 1、激活 6、2048、284B / 13B；路由专家权重 FP4。

        **延伸**：论文没有解释换 Sqrt(Softplus) 的动机，可以从函数形状分析（正区间不饱和、负区间更平缓），但要说明是推测。去掉节点限制依赖重新设计的并行策略。
      `,
    },
    {
      id: 'qa-hash',
      freq: 2,
      q: md`什么是 Hash 路由？DeepSeek-V4 为什么只在前几层用？`,
      a: md`
        **结论**：专家由 token id 经固定哈希（V4 中是 \`tid2eid\` 查找表）决定，不靠学习；V4 前 3 个 MoE 层使用，门控权重仍来自 router 分数。

        **原因**（分析）：浅层隐状态主要由 token 本身决定，按 token id 路由损失不大；固定路由不会坍缩、不需要均衡机制，且在拿到 token id 时就能确定通信。深层需要按上下文区分语义，固定路由会限制表达力。

        **延伸**：Hash Layers（Roller et al., 2021）最早系统研究了这一思路；负载均衡取决于表的构造与 token 频率（Zipf），V4 未说明表的构造方法。
      `,
    },
    {
      id: 'qa-qb',
      freq: 2,
      q: md`Kimi K3 的 Quantile Balancing 与 DeepSeek-V3 的 bias 均衡有什么区别？`,
      a: md`
        **结论**：框架相同（bias 只管 top-k 选择、不进权重、只影响下一步），区别在更新规则：V3 每步按负载符号走固定步长 $\gamma$；QB 直接把 bias 设成让每个专家负载恰为 $q = mk/n$ 的值——margin（$s - $ 第 $k+1$ 大的选择分数）的 $1-k/n$ 分位数取负，再减均值。

        **原因**：K3 有 896 个专家，论文指出在近千专家下 sign 规则的 bias 更新不再表现良好；$\gamma$ 必须在「慢」与「震荡」之间权衡，而 QB 没有步长超参。

        **延伸**：大规模时用每专家 margin 直方图 + 一次 all-reduce 估计分位数；一步后不严格均衡（cutoff 会随 bias 变化），但收敛很快。
      `,
    },
    {
      id: 'qa-trend',
      freq: 2,
      q: md`MoE 越来越稀疏，激活比例会一直降下去吗？超稀疏有什么代价？`,
      a: md`
        **结论**：路由稀疏度 $k/N$ 在持续下降（Mixtral 25% → V3 3.1% → V4-Pro 1.56% / K3 1.79%），但整体激活比例在 3%–5% 附近进入平台：注意力、嵌入、共享专家是固定开销。

        **代价**：① 每个专家分到的 token 少，GEMM 小、算力难用满，需要大 batch 与专家并行；② 均衡更难（K3 改用 Quantile Balancing）；③ 每 token 派发更多专家，通信碎片化（K3 用 LatentMoE 把派发维度降到 $0.5d$；V4 去掉节点限制后重新设计并行）；④ 显存与总参数成正比，推理部署门槛高。

        **延伸**：答题时给出两个口径的具体数字，比泛泛说「越来越稀疏」更有说服力。
      `,
    },
  ],

  cheatsheet: md`
    **DeepSeek-V4**：$s=\sqrt{\operatorname{softplus}(x)}$；bias 速度 0.001 + 序列级损失 1e-4；无节点限制；前 3 个 MoE 层 Hash 路由
    \`\`\`python
    s = np.sqrt(np.logaddexp(0.0, logits))
    idx = tid2eid[ids] if layer < 3 else np.argsort(-(s + b), -1)[:, :k]
    w = np.take_along_axis(s, idx, -1); w = w / w.sum(-1, keepdims=True) * scale
    \`\`\`
    **Kimi K3 QB**：$\alpha_i$ = 第 $k+1$ 大的 $s+b$；$\hat b_j = -\,$第 $q{+}1$ 大的 $(s_{:,j}-\alpha)$，$q = mk/n$；$b = \hat b - \bar{\hat b}$
    \`\`\`python
    alpha = -np.sort(-(s + b), 1)[:, k]
    b_hat = np.sort(-(s - alpha[:, None]), 0)[q]
    b = b_hat - b_hat.mean()        # 下一步才生效
    \`\`\`

    | | 路由 + 共享 | 激活 | 总 / 激活 |
    |---|---|---|---|
    | V4-Flash | 256 + 1 | 6 | 284B / 13B |
    | V4-Pro | 384 + 1 | 6 | 1.6T / 49B |
    | K3 | 896 + 2 | 16 | 2.8T / 104B |

    - K3 LatentMoE：路由专家在 $\ell = 0.5d = 3584$ 维计算，RMSNorm 后 $W_\uparrow$；router 与共享专家用全宽 $x$
    - 激活比例平台 3%–5%；$k/N$：25% → 3.1% → 1.56% / 1.79%
  `,
})
