/**
 * llm-moe-deepseek：DeepSeekMoE 与无辅助损失均衡
 * 事实来源：DeepSeekMoE (arXiv 2401.06066)；DeepSeek-V3 技术报告 (arXiv 2412.19437) §2.1.2、§3.2、§4.2；
 *          Auxiliary-Loss-Free Load Balancing (Wang et al., arXiv 2408.15664)；DeepSeek-V2 / V3 官方 HF config。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { FineGrainedViz } from '../viz/FineGrainedViz'
import { GateBiasViz } from '../viz/GateBiasViz'
import { BiasBalanceViz } from '../viz/BiasBalanceViz'
import { NodeLimitViz } from '../viz/NodeLimitViz'

const GATE_SETUP = py`
  import numpy as np

  def _ref_v3_gate(logits, bias, k, n_group, topk_group, route_scale):
      T, E = logits.shape
      s = 1.0 / (1.0 + np.exp(-logits))
      choice = s + bias
      per = k // topk_group
      g = choice.reshape(T, n_group, E // n_group)
      gscore = -np.sort(-g, axis=-1)[:, :, :per].sum(-1)              # (T, n_group)
      keep = np.argsort(-gscore, axis=-1, kind="stable")[:, :topk_group]
      gmask = np.zeros((T, n_group), dtype=bool)
      np.put_along_axis(gmask, keep, True, axis=-1)
      choice = np.where(np.repeat(gmask, E // n_group, axis=1), choice, -np.inf)
      idx = np.argsort(-choice, axis=-1, kind="stable")[:, :k]
      w = np.take_along_axis(s, idx, axis=-1)
      return idx, w / w.sum(-1, keepdims=True) * route_scale

  def _canon(idx, w):
      idx = np.asarray(idx)
      w = np.asarray(w, dtype=float)
      o = np.argsort(idx, axis=-1)
      return np.take_along_axis(idx, o, -1), np.take_along_axis(w, o, -1)

  def _check(got, exp, tag=""):
      assert isinstance(got, tuple) and len(got) == 2, f"{tag}应返回 (idx, weights) 二元组"
      gi, gw = got
      assert np.shape(gi) == exp[0].shape, f"{tag}idx 形状应为 {exp[0].shape}，得到 {np.shape(gi)}"
      ci, cw = _canon(gi, gw)
      ei, ew = _canon(*exp)
      assert np.array_equal(ci, ei), f"{tag}选中的专家不对：\n期望 {ei.tolist()}\n得到 {ci.tolist()}"
      assert np.allclose(cw, ew, atol=1e-9), f"{tag}权重最大误差 {np.abs(cw - ew).max():.3e}"
`

const SIM_SETUP = py`
  import numpy as np

  def _ref_sim(score_batches, k, gamma):
      S, T, E = score_batches.shape
      b = np.zeros(E)
      loads = np.zeros((S, E), dtype=int)
      for t in range(S):
          idx = np.argsort(-(score_batches[t] + b), axis=-1, kind="stable")[:, :k]
          load = np.bincount(idx.ravel(), minlength=E)
          loads[t] = load
          b = b + gamma * np.sign(load.mean() - load)
      return loads, b

  def _skewed(seed, S, T, E):
      r = np.random.default_rng(seed)
      pop = np.linspace(0.9, -0.9, E)
      return 1.0 / (1.0 + np.exp(-(pop + r.normal(size=(S, T, E)))))

  def _maxvio(load):
      m = load.mean()
      return (load.max() - m) / m
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'finegrained',
      title: '细粒度专家 + 共享专家：为什么要拆小、为什么要共享',
      body: md`
        DeepSeekMoE（2024-01）认为传统「8 或 16 个大专家选 2」有两个问题：
        - **知识混杂（hybridity）**：专家太少，每个专家被迫同时覆盖很多种知识，难以专精；
        - **知识冗余（redundancy）**：各专家都要学一遍通用知识（语法、常见搭配），参数被重复浪费。

        对应两招：
        1. **细粒度切分**：把每个专家的 FFN 中间维缩小到 $1/m$，专家数变为 $mN$，激活数变为 $mK$——**总参数与每 token 计算量都不变**，但可选的专家组合数暴涨。论文的例子：$N=16$ 选 2 只有 $\binom{16}{2}=120$ 种组合；$m=4$ 后 64 选 8 有 $\binom{64}{8} = 4{,}426{,}165{,}368$ 种。
        2. **共享专家隔离**：从中拿出 $K_s$ 个作为共享专家，**每个 token 都必经**，专门吸收通用知识；路由专家变为 $mN - K_s$ 个、激活 $mK - K_s$ 个。

        $$
        h_t' = u_t + \sum_{i=1}^{N_s} \mathrm{FFN}^{(s)}_i(u_t) + \sum_{i=1}^{N_r} g_{i,t}\,\mathrm{FFN}^{(r)}_i(u_t)
        $$

        | 模型 | 共享 | 路由（选几个） | 专家中间维 |
        |---|---|---|---|
        | DeepSeekMoE 16B | 2 | 64（选 6） | 标准 FFN 的 0.25 倍 |
        | DeepSeek-V2 | 2 | 160（选 6） | 1536 |
        | DeepSeek-V3 | 1 | 256（选 8） | 2048（前 3 层稠密 FFN 为 18432） |
      `,
      viz: FineGrainedViz,
      after: md`
        > [!WARN] 代价
        > 专家越小，每次专家 GEMM 越「瘦」，GPU 效率越低；每个 token 派发给更多专家，all-to-all 通信量随激活专家数增长；router 的输出维也变大。
        > DeepSeek-V3 用节点受限路由（本关第 4 节）和 DeepEP 等通信优化来消化这些代价。
      `,
      checkpoint: {
        id: 'cp-fine',
        type: 'fill',
        prompt: md`从「16 个标准专家选 2」出发，切分倍数 $m=4$，并隔离 1 个共享专家。router 要从 ____(1)____ 个路由专家中选 ____(2)____ 个。`,
        blanks: [
          { label: '(1)', answers: ['63'], numeric: true },
          { label: '(2)', answers: ['7'], numeric: true },
        ],
        explanation: md`切分后共 $mN = 64$ 个小专家、每 token 激活 $mK = 8$ 个；拿出 $K_s=1$ 个做共享专家后，路由专家 $64-1=63$ 个、选 $8-1=7$ 个。每 token 仍是 8 个小专家 = 2 个标准 FFN 的计算量。`,
      },
    },
    {
      id: 'gate',
      title: 'V3 门控：sigmoid 亲和度 + top-k 后归一化',
      body: md`
        DeepSeek-V3 的门控（技术报告式 13–16）：

        $$
        s_{i,t} = \operatorname{Sigmoid}(u_t^\top e_i),\qquad
        g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \operatorname{TopK}(\{s_{j,t} + b_j\},\,K_r) \\ 0, & \text{otherwise}\end{cases},\qquad
        g_{i,t} = \frac{g'_{i,t}}{\sum_j g'_{j,t}}
        $$

        三个要点：
        1. **sigmoid 而不是 softmax**：每个专家的亲和度独立计算，不在 256 个专家之间「抢」概率质量。Loss-Free Balancing 论文的实验里 sigmoid 基线也优于 softmax 基线。V2 用的是 softmax。
        2. **只在选中的专家内归一化**：sigmoid 分数之和不是 1，必须归一化才能控制输出幅度。官方 config 另有 \`routed_scaling_factor: 2.5\`，即最终权重和为 2.5（报告正文没有讨论这个系数）。
        3. **偏置 $b_i$ 只出现在「选择」里**：$g'_{i,t}$ 取的是原始 $s_{i,t}$，不是 $s_{i,t}+b_i$。bias 不进入前向输出，于是也不参与梯度计算——这正是它能「无辅助损失」地调负载的关键。
      `,
      viz: GateBiasViz,
      after: md`
        > [!INTERVIEW] 「bias 加在哪？」是必问题
        > 正确答案：只加在 top-k 的**排序分数**上；门控权重用原始亲和度。如果把 bias 也算进权重，bias 的每次调整都会直接改变模型输出，均衡机制就从「调度」变成了「干预」。
        > HF 的 \`noaux_tc\` 实现也是这样：\`scores_for_choice = scores + e_score_correction_bias\` 只用于 topk，权重从原始 \`scores\` gather。
      `,
      checkpoint: {
        id: 'cp-gate',
        type: 'single',
        prompt: md`按官方 config（\`norm_topk_prob: true\`、\`routed_scaling_factor: 2.5\`），DeepSeek-V3 每个 token 的 8 个路由专家门控权重之和是多少？`,
        options: ['1', '2.5', '8', '取决于 sigmoid 分数，不固定'],
        answer: 1,
        explanation: md`先在选中的 8 个专家内归一化（和为 1），再乘缩放因子 2.5，所以和恒为 2.5。若不归一化，和就是 8 个 sigmoid 分数之和，随 token 变化——这正是 D 选项描述的未归一化情形。`,
      },
    },
    {
      id: 'lossfree',
      title: '无辅助损失均衡：用 bias 调度，而不是用梯度惩罚',
      body: md`
        Switch 式辅助损失有个两难：$\alpha$ 小了压不住不均衡；$\alpha$ 大了，辅助损失的梯度会和语言模型损失「打架」（论文称为 interference gradients），拉低效果。
        Loss-Free Balancing（Wang et al., 2024）提出完全不用辅助损失：给每个专家一个偏置 $b_i$，**只用来调整 top-k 选择**，每步按负载更新：

        $$
        b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}\big(\bar{c} - c_i\big)
        $$

        其中 $c_i$ 是专家 $i$ 在当前 batch 的负载，$\bar c$ 是平均负载：**过载就调低 bias，欠载就调高**。

        - **不走梯度**：$b_i$ 不是可学习参数，不进入前向输出，没有任何梯度干扰。
        - **因果性**：第 $t$ 步的路由用的是之前各步累积的 bias。论文指出若用**当前序列**的负载来决定路由，会泄露未来 token 的信息，破坏语言建模的因果约束。
        - **V3 的取值**：$\gamma = 0.001$ 用于前 14.3T token，最后 500B token 设为 0（冻结 bias）。
        - **sign 还是按比例**：论文对比过 $b_i \leftarrow b_i + u\,e_i$（不取 sign）：均衡度略好（MaxVio 0.036 vs 0.044），但困惑度并未更好（9.51 vs 9.50），最终保留 sign 版本。
      `,
      viz: BiasBalanceViz,
      after: md`
        > [!TIP] 读图
        > 虚线是不做均衡时的 MaxVio（最大负载超出均值的比例），始终在 1 以上——最热的专家拿到两倍以上的平均负载。
        > 切换 γ：0.001 太慢、0.1 来回震荡；sign 规则下 bias 每步只能走一个固定步长，这是它稳定的原因，也是它慢的原因。
        > 第五关（前沿）会看到 Kimi K3 的 **Quantile Balancing** 如何去掉这个步长超参。
      `,
      checkpoint: {
        id: 'cp-lossfree',
        type: 'single',
        prompt: md`某步结束时，平均负载 $\bar c = 100$，专家 3 的负载 $c_3 = 140$，$\gamma = 0.001$。它的 bias 如何变化？`,
        options: ['$b_3$ 增加 0.001', '$b_3$ 减少 0.001', '$b_3$ 减少 0.04', '$b_3$ 不变，改为增大辅助损失'],
        answer: 1,
        explanation: md`$\operatorname{sign}(\bar c - c_3) = \operatorname{sign}(-40) = -1$，所以 $b_3 \leftarrow b_3 - 0.001$。sign 规则只看方向不看大小，所以不是 C（那是比例更新的思路）。bias 变小后专家 3 在下一步更难进入 top-k。`,
      },
    },
    {
      id: 'nodelimit',
      title: '节点受限路由：给通信量设上限',
      body: md`
        细粒度专家意味着每个 token 要派发到很多专家，而这些专家分布在不同机器上。DeepSeek-V3 的部署：每层的路由专家**均匀分布在 8 个节点的 64 张 GPU 上**。
        跨节点走 InfiniBand（约 50 GB/s），节点内走 NVLink（约 160 GB/s，是 IB 的约 3.2 倍）。

        **节点受限路由**：每个 token 最多发往 $M=4$ 个节点。做法是先给每个节点打分——**该节点上最高的 $K_r/M = 2$ 个亲和度之和**——选出前 $M$ 个节点，再只在这些节点的专家里取 top-$K_r$。

        - 一个 token 进入某节点后，可以经 NVLink 再转发给该节点内的多个专家；报告据此估算每个 token 平均可在每个节点选约 3.2 个专家而不增加额外开销，最多可扩展到约 13 个专家（4 节点 × 3.2）。
        - HF config 中对应 \`n_group: 8, topk_group: 4\`；V2 是 \`n_group: 8, topk_group: 3\`（设备受限路由）。
        - 代价：可能排除全局最优的专家（见下图），但只要同时做好负载均衡，损失很小。
        - DeepSeek-V4 **去掉了**这个节点数限制，并重新设计了并行策略（前沿一关）。
      `,
      viz: NodeLimitViz,
      after: md`
        > [!WARN] 易错点
        > 节点分数不是「节点内最大值」，也不是「节点内所有分数之和」，而是**节点内最高的 $K_r/M$ 个**之和；而且打分用的是**加了 bias 的**选择分数（HF 实现里 \`scores_for_choice\` 在分组前就加上了 bias）。
      `,
      checkpoint: {
        id: 'cp-node',
        type: 'numeric',
        prompt: md`按 DeepSeek-V3 报告的估算：NVLink 带宽约为 IB 的 3.2 倍，每个 token 最多发往 4 个节点。在不增加通信开销的前提下，每个 token 最多大约可以选多少个路由专家？（四舍五入到整数）`,
        answer: 13,
        tolerance: 0,
        unit: '个',
        explanation: md`每个节点平均可容纳约 3.2 个专家而不增加 NVLink 开销，$4 \times 3.2 = 12.8 \approx 13$。V3 实际只用 8 个路由专家，通信还有余量。`,
      },
    },
    {
      id: 'seqaux',
      title: '补充的序列级均衡损失，以及「不丢 token」',
      body: md`
        bias 保证的是**整个 batch** 上的均衡。但单条序列内部仍可能极端不均（例如一段代码把同一批专家打满）。V3 为此加了一个**很小的**序列级辅助损失：

        $$
        \mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i,\quad
        f_i = \frac{N_r}{K_r T}\sum_{t=1}^{T} \mathbb{1}\big(s_{i,t} \in \operatorname{TopK}\big),\quad
        s'_{i,t} = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}},\quad
        P_i = \frac{1}{T}\sum_{t=1}^{T} s'_{i,t}
        $$

        - $T$ 是**单条序列**的 token 数；$\frac{N_r}{K_r T}$ 的归一化让完全均衡时 $f_i = 1$、$\sum_i f_i P_i = 1$；
        - $s'$ 是把 sigmoid 分数在**全部** $N_r$ 个专家上归一化（不只是选中的）；
        - $\alpha = 0.0001$——比 Switch 的 0.01 小两个数量级，只防极端情况，基本不干扰主损失。

        **为什么不全靠序列级约束？** V3 报告讨论过：batch 级均衡的约束更宽松，允许专家在不同领域（代码、数学、语言）上专精；
        他们也观察到无辅助损失的模型表现出更明显的专家专精模式。

        **不丢 token**：凭借良好的均衡，V3 训练时**不丢弃任何 token**，推理时也通过部署策略保证不丢——下一关讲的 capacity factor 与 token 丢弃在 V3 中不存在。
      `,
      checkpoint: {
        id: 'cp-seqaux',
        type: 'numeric',
        prompt: md`
          一条序列 $T=4$ 个 token，$N_r = 4$、$K_r = 2$（bias 为 0）。各 token 在全部专家上归一化后的分数 $s'$ 为：
          $t_0$：[0.4, 0.3, 0.2, 0.1]，$t_1$：[0.4, 0.1, 0.4, 0.1]，$t_2$：[0.5, 0.3, 0.1, 0.1]，$t_3$：[0.3, 0.2, 0.1, 0.4]。
          求 $\sum_i f_i P_i$（即 $\alpha = 1$ 时的 $\mathcal{L}_{\text{Bal}}$）。
        `,
        answer: 1.2125,
        tolerance: 0.001,
        explanation: md`
          各 token 的 top-2：$t_0\{0,1\}$、$t_1\{0,2\}$、$t_2\{0,1\}$、$t_3\{3,0\}$ → 计数 [4, 2, 1, 1]；$f = \frac{4}{2\cdot 4}\times$ 计数 $= [2, 1, 0.5, 0.5]$。
          $P$ = 各列均值 $= [0.4, 0.225, 0.2, 0.175]$。$\sum f_i P_i = 0.8 + 0.225 + 0.1 + 0.0875 = 1.2125$，比完全均衡的 1 高出 21%。
        `,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-motive',
      type: 'single',
      difficulty: 1,
      prompt: md`DeepSeekMoE 提出「细粒度专家切分」，在**不改变总参数与每 token 计算量**的前提下，它直接带来的好处是：`,
      options: [
        '每个专家的 GEMM 更大，GPU 利用率更高',
        '可选的专家组合数大幅增加，专家能更专精',
        '不再需要负载均衡',
        '减少了 all-to-all 通信量',
      ],
      answer: 1,
      explanation: md`切分后专家数 $\times m$、激活数 $\times m$，组合数从 $\binom{N}{K}$ 变为 $\binom{mN}{mK}$，知识可以拆得更细、组合更灵活。A 恰好相反（专家变小、GEMM 变瘦）；C 错，专家越多越需要均衡；D 错，每个 token 要派发给更多专家，通信量通常**增加**。`,
    },
    {
      id: 'q-comb',
      type: 'numeric',
      difficulty: 2,
      prompt: md`DeepSeekMoE 论文的例子：16 个专家选 2 共 120 种组合；把每个专家切成 4 份后，64 个专家选 8 个，共有多少种组合？（可输入表达式或科学计数法）`,
      answer: 4426165368,
      tolerance: 0.001,
      explanation: md`$\binom{64}{8} = \frac{64!}{8!\,56!} = 4{,}426{,}165{,}368 \approx 4.4\times 10^9$，是 120 的三千多万倍。注意这里没有隔离共享专家；隔离 1 个后是 $\binom{63}{7} = 553{,}270{,}671$。`,
    },
    {
      id: 'q-bias-weight',
      type: 'judge',
      difficulty: 1,
      prompt: md`DeepSeek-V3 中，无辅助损失均衡的偏置 $b_i$ 会同时加在 top-k 选择分数和门控权重上，以便更快地调整负载。`,
      answer: false,
      explanation: md`错误。$b_i$ **只**用于 top-k 选择；门控权重 $g_{i,t}$ 用原始亲和度 $s_{i,t}$ 归一化。这样 bias 不改变选中专家的输出权重，也不产生梯度干扰。`,
    },
    {
      id: 'q-lossfree-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于无辅助损失负载均衡（Loss-Free Balancing / DeepSeek-V3），下列说法正确的是（多选）：`,
      options: [
        'bias 是通过反向传播学习的参数',
        '过载专家的 bias 在每步结束时减小 γ，欠载专家增加 γ',
        '用当前序列自身的负载来决定当前 token 的路由，会泄露未来 token 的信息',
        'V3 在前 14.3T token 用 γ = 0.001，最后 500B token 设为 0',
        '有了 bias 之后 V3 完全不需要任何辅助损失',
      ],
      answer: [1, 2, 3],
      explanation: md`A 错：bias 按负载做规则更新，不走梯度。B、C、D 均为原文表述。E 错：V3 另加了 $\alpha = 0.0001$ 的**序列级**均衡损失，防止单条序列内的极端不均衡。`,
    },
    {
      id: 'q-fill-node',
      type: 'fill',
      difficulty: 2,
      prompt: md`DeepSeek-V3 每层的路由专家均匀分布在 ____(1)____ 个节点的 64 张 GPU 上；每个 token 最多发往 ____(2)____ 个节点；节点按其上最高的 ____(3)____ 个亲和度之和排序。`,
      blanks: [
        { label: '(1)', answers: ['8'], numeric: true },
        { label: '(2)', answers: ['4'], numeric: true },
        { label: '(3)', answers: ['2'], numeric: true },
      ],
      explanation: md`8 个节点 × 8 张 GPU = 64 张；$M = 4$；节点分数为最高的 $K_r/M = 8/4 = 2$ 个亲和度之和（HF config：\`n_group=8, topk_group=4\`）。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 DeepSeek-V3 一个 MoE 层的路由步骤按正确顺序排列：`,
      items: [
        '计算亲和度 $s = \\operatorname{sigmoid}(u_t^\\top e_i)$',
        '加上 bias 得到选择分数 $s + b$',
        '按节点内最高 2 个选择分数之和选出 4 个节点',
        '在这 4 个节点的专家中按选择分数取 top-8',
        '取被选中专家的原始 $s$，归一化后乘 2.5 作为门控权重',
        '步末统计负载，按 sign 规则更新 bias',
      ],
      explanation: md`选择相关的步骤（加 bias、选节点、top-k）都用 $s+b$；权重只用原始 $s$；bias 的更新发生在这一步路由**之后**，只影响后续步骤。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把 DeepSeekMoE / V3 的设计与它要解决的问题配对：`,
      pairs: [
        ['共享专家', '通用知识在各专家间重复学习'],
        ['细粒度切分', '专家太少、知识混杂难以专精'],
        ['偏置 b（无辅助损失）', '辅助损失梯度干扰主损失'],
        ['节点受限路由', '跨节点 IB 通信开销'],
        ['序列级均衡损失（α=1e-4）', '单条序列内的极端不均衡'],
      ],
      explanation: md`五者各管一件事：共享专家去冗余、细粒度切分提专精、bias 管 batch 级均衡且不污染梯度、节点受限管通信、小 α 序列损失兜底极端情况。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面是 V3 风格的路由 + bias 更新（\`logits\` 为 $(T,E)$，\`bias\` 为 $(E,)$），有 **2 处** bug。点出有问题的行：`,
      code: py`
        import numpy as np

        def v3_route(logits, bias, k, gamma, route_scale=2.5):
            s = 1.0 / (1.0 + np.exp(-logits))
            choice = s + bias
            idx = np.argsort(-choice, axis=-1)[:, :k]
            w = np.take_along_axis(choice, idx, axis=-1)
            w = w / w.sum(axis=-1, keepdims=True) * route_scale
            load = np.bincount(idx.ravel(), minlength=bias.shape[0])
            bias = bias + gamma * np.sign(load - load.mean())
            return idx, w, bias
      `,
      bugLines: [7, 10],
      explanation: md`第 7 行应从原始亲和度取权重：\`np.take_along_axis(s, idx, axis=-1)\`，否则 bias 混进了输出。第 10 行符号反了：过载（\`load > mean\`）时 bias 应**减小**，应为 \`bias + gamma * np.sign(load.mean() - load)\`；写反会让热门专家越来越热，加速坍缩。`,
    },
    {
      id: 'q-pick-node',
      type: 'pick',
      difficulty: 2,
      prompt: md`4 个节点、每节点 4 个专家，各专家的选择分数如下（bias 已加入）。节点受限路由取 $M=2$ 个节点（节点分 = 节点内最高 2 个分数之和），再选 top-4。点出最终被选中的专家：`,
      items: ['0.90', '0.30', '0.25', '0.10', '0.70', '0.68', '0.40', '0.20', '0.85', '0.20', '0.15', '0.05', '0.66', '0.64', '0.60', '0.10'],
      cols: 4,
      rowLabels: ['N0', 'N1', 'N2', 'N3'],
      colLabels: ['e0', 'e1', 'e2', 'e3'],
      answer: [4, 5, 12, 13],
      explanation: md`节点分：N0 = 1.20、N1 = 1.38、N2 = 1.05、N3 = 1.30 → 选 N1、N3。在它们的 8 个专家里取 top-4：0.70、0.68、0.66、0.64。注意全局最高的 0.90（N0）和 0.85（N2）**都被排除了**——节点分看的是「前 2 名之和」，单个尖峰撑不起一个节点。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出一次「用 bias 选择 → 统计负载 → 更新 bias」的训练步（有 2 行干扰项）：`,
      lines: [
        'def route_and_update(s, bias, k, gamma):',
        '    idx = np.argsort(-(s + bias), axis=-1)[:, :k]',
        '    w = np.take_along_axis(s, idx, axis=-1)',
        '    w = w / w.sum(axis=-1, keepdims=True)',
        '    load = np.bincount(idx.ravel(), minlength=s.shape[1])',
        '    new_bias = bias + gamma * np.sign(load.mean() - load)',
        '    return idx, w, new_bias',
      ],
      distractors: ['    w = np.take_along_axis(s + bias, idx, axis=-1)', '    new_bias = bias + gamma * (load - load.mean())'],
      explanation: md`干扰项 1 让 bias 进入权重；干扰项 2 既没有 sign、方向也反了（过载时反而增大 bias）。注意 \`new_bias\` 只给**下一步**使用，本步的 \`idx\` 已经用旧 bias 算好——这就是因果性。`,
    },
    {
      id: 'q-sigmoid',
      type: 'single',
      difficulty: 3,
      prompt: md`DeepSeek-V3 把亲和度从 V2 的 softmax 换成 sigmoid。下列哪一项**不是**这一改动的合理推论？`,
      options: [
        '各专家的亲和度互不竞争，加 bias 调整某个专家不会改变其它专家的原始分数',
        'sigmoid 分数之和不为 1，因此必须在 top-k 内归一化（并乘缩放因子）来控制输出幅度',
        '换成 sigmoid 后不再需要任何负载均衡机制',
        '专家数很大（256）时，softmax 会把概率摊得很薄，sigmoid 不存在这个问题',
      ],
      answer: 2,
      explanation: md`C 错：打分函数与均衡是两回事，V3 依然需要 bias 均衡 + 序列级损失。A：sigmoid 逐专家独立；B：这正是 V3 归一化 + \`routed_scaling_factor\` 的原因；D：softmax 的归一化在 256 个专家上摊薄概率，sigmoid 的值域不受专家数影响。报告未详细说明换 sigmoid 的动机，但 Loss-Free Balancing 论文报告 sigmoid 基线优于 softmax。`,
    },
    {
      id: 'q-causal',
      type: 'single',
      difficulty: 3,
      prompt: md`为什么 bias 必须用**之前**步骤的负载来更新，而不能「先看本 batch 的负载，再决定本 batch 的路由」？`,
      options: [
        '因为本 batch 的负载要等反向传播结束才知道',
        '因为在序列内根据整体负载决定某个 token 的路由，相当于让它的计算依赖了同序列后面 token 的信息，破坏因果性（推理时也无法复现）',
        '因为这样会让 bias 的更新步长过大',
        '因为 all-to-all 通信要求 bias 在各卡之间同步',
      ],
      answer: 1,
      explanation: md`负载是对整条序列 / 整个 batch 的统计量；若当前 token 的路由依赖它，就隐含使用了未来 token 的信息——训练时「偷看」答案，推理逐 token 生成时又拿不到，训练推理不一致。Loss-Free Balancing 论文明确指出了这一点。A 错，负载在前向时就能统计；C、D 与问题无关。Kimi K3 的 Quantile Balancing 同样强调「新 bias 只在下一步生效」。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-v3-route',
      title: '写出 DeepSeek-V3 的门控与 bias 更新',
      intro: md`
        实现两个函数（不考虑节点受限）：
        - \`v3_route(logits, bias, k, route_scale=2.5)\`：sigmoid 亲和度 → 用 $s+b$ 选 top-k → 用原始 $s$ 归一化 → 乘缩放因子，返回 \`(idx, w)\`；
        - \`update_bias(bias, idx, gamma)\`：统计每个专家被选中的次数，按 $b \leftarrow b + \gamma\,\mathrm{sign}(\bar c - c)$ 更新。
      `,
      template: py`
        import numpy as np

        def v3_route(logits, bias, k, route_scale=2.5):
            s = {{1}}                                   # (T, E) sigmoid 亲和度
            choice = {{2}}                              # 只用于选择
            idx = np.argsort(-choice, axis=-1)[:, :k]   # (T, k)
            w = {{3}}                                   # 取原始亲和度
            w = {{4}} * route_scale                     # top-k 内归一化
            return idx, w

        def update_bias(bias, idx, gamma):
            E = bias.shape[0]
            load = {{5}}                                # (E,) 每个专家的负载
            return {{6}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`计算 sigmoid 亲和度。`,
          answer: '1.0 / (1.0 + np.exp(-logits))',
          accept: ['1 / (1 + np.exp(-logits))'],
          explain: md`每个专家独立打分，值域 (0,1)，和不为 1。大规模实现中 router 通常在 fp32 下计算。`,
        },
        {
          blank: 2,
          instruction: md`得到选择分数（加上 bias）。`,
          answer: 's + bias',
          accept: ['bias + s'],
          explain: md`\`bias\` 形状 $(E,)$，会广播到每个 token。它只出现在这里——决定「选谁」。`,
        },
        {
          blank: 3,
          instruction: md`从**原始**亲和度中取出被选中专家的分数，形状 $(T,k)$。`,
          answer: 'np.take_along_axis(s, idx, axis=-1)',
          accept: ['np.take_along_axis(s, idx, -1)', 'np.take_along_axis(s, idx, axis=1)'],
          explain: md`这里若写成 \`choice\`，bias 就混进了门控权重——面试最常见的错误。`,
        },
        {
          blank: 4,
          instruction: md`在选中的 $k$ 个之内归一化。`,
          answer: 'w / w.sum(axis=-1, keepdims=True)',
          accept: ['w / w.sum(-1, keepdims=True)', 'w / np.sum(w, axis=-1, keepdims=True)'],
          explain: md`归一化后再乘 \`route_scale\`（V3 为 2.5），每个 token 的权重和恒为 \`route_scale\`。`,
        },
        {
          blank: 5,
          instruction: md`统计每个专家在这个 batch 中被选中的次数。`,
          answer: 'np.bincount(idx.ravel(), minlength=E)',
          accept: ['np.bincount(idx.reshape(-1), minlength=E)', 'np.bincount(idx.flatten(), minlength=E)'],
          explain: md`\`minlength=E\` 保证没被选中的专家也有一个 0，否则数组长度可能小于 $E$。`,
        },
        {
          blank: 6,
          instruction: md`按 sign 规则返回新的 bias：过载减 $\gamma$、欠载加 $\gamma$、恰好等于均值不变。`,
          answer: 'bias + gamma * np.sign(load.mean() - load)',
          accept: ['bias - gamma * np.sign(load - load.mean())'],
          explain: md`\`np.sign(0) = 0\`，负载恰为均值的专家 bias 不动。整体上 bias 之和的变化取决于过载 / 欠载专家的个数，不必为 0——只有相对大小影响 top-k。`,
        },
      ],
      setup: py`
        def _ref_route(logits, bias, k, route_scale=2.5):
            s = 1 / (1 + np.exp(-logits))
            idx = np.argsort(-(s + bias), axis=-1, kind="stable")[:, :k]
            w = np.take_along_axis(s, idx, -1)
            return idx, w / w.sum(-1, keepdims=True) * route_scale

        def _same(a, b):
            ia, wa = a
            ib, wb = b
            oa, ob = np.argsort(ia, -1), np.argsort(ib, -1)
            return (np.array_equal(np.take_along_axis(ia, oa, -1), np.take_along_axis(ib, ob, -1))
                    and np.allclose(np.take_along_axis(wa, oa, -1), np.take_along_axis(wb, ob, -1)))
      `,
      tests: [
        {
          name: 'bias 改变选择但不进入权重',
          code: py`
            r = np.random.default_rng(0)
            logits = r.normal(size=(6, 8))
            bias = np.array([-0.5, 0.0, 0.3, 0.0, 0.1, -0.2, 0.4, 0.0])
            got = v3_route(logits, bias, 3)
            assert _same(got, _ref_route(logits, bias, 3)), "路由结果与参考实现不一致（检查 bias 是否进入了权重）"
            assert np.allclose(np.asarray(got[1]).sum(-1), 2.5), "每个 token 的权重和应为 route_scale = 2.5"
          `,
        },
        {
          name: 'bias 更新方向',
          code: py`
            idx = np.array([[0, 1], [0, 2], [0, 1], [3, 0]])
            got = update_bias(np.zeros(4), idx, 0.01)
            exp = np.array([-0.01, 0.0, 0.01, 0.01])
            assert np.allclose(got, exp), f"负载 [4,2,1,1]、均值 2：期望 {exp.tolist()}，得到 {np.round(got, 4).tolist()}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            r = np.random.default_rng(7)
            for _ in range(20):
                T, E, k = r.integers(3, 12), r.integers(4, 10), r.integers(1, 4)
                logits, bias = r.normal(size=(T, E)), r.normal(scale=0.3, size=E)
                assert _same(v3_route(logits, bias, k, 1.5), _ref_route(logits, bias, k, 1.5))
          `,
        },
      ],
      outro: md`
        真实实现（HF \`noaux_tc\`）还多一步分组：把 \`choice\` reshape 成 \`(T, n_group, E/n_group)\`，组内取前 2 名求和，保留前 \`topk_group\` 组，其余置 0 后再 topk。实战第一题就是它。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-v3-gate',
      title: 'DeepSeek-V3 门控：sigmoid + bias 选择 + 节点受限',
      difficulty: 'medium',
      source: '大模型岗手撕 · DeepSeekMoE',
      timeLimitMin: 25,
      description: md`
        实现 DeepSeek-V3 风格的门控（对应 HF \`topk_method="noaux_tc"\`）：

        \`\`\`python
        def deepseek_v3_gate(logits, bias, k, n_group, topk_group, route_scale):
            """返回 (idx, weights)，形状均为 (T, k)"""
        \`\`\`

        - \`logits\`：$(T, E)$ router 输出；\`bias\`：$(E,)$ 均衡偏置；$E$ 能被 \`n_group\` 整除，第 $g$ 组是专家 $[g\cdot E/n_{group},\,(g+1)\cdot E/n_{group})$
        1. $s = \mathrm{sigmoid}(\text{logits})$，选择分数 $c = s + b$；
        2. **组分数** = 组内最大的 \`k // topk_group\` 个 $c$ 之和（V3：$8 // 4 = 2$）；每个 token 只保留组分数最高的 \`topk_group\` 个组；
        3. 在保留组的专家里按 $c$ 取 top-$k$；
        4. 权重 = 选中专家的**原始** $s$，在 $k$ 个内归一化后乘 \`route_scale\`。

        \`idx\` 每行内的顺序不限（测试会按专家编号排序后比较，权重要随之对应）。测试数据没有并列。

        **示例**（\`n_group=1, topk_group=1\` 时退化为不分组）：V3 为 \`E=256, n_group=8, topk_group=4, k=8, route_scale=2.5\`。
      `,
      starter: py`
        import numpy as np

        def deepseek_v3_gate(logits, bias, k, n_group, topk_group, route_scale):
            # 1) sigmoid + bias  2) 组分数 = 组内 top-(k // topk_group) 之和
            # 3) 屏蔽未选中的组后 top-k  4) 原始 s 归一化 × route_scale
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def deepseek_v3_gate(logits, bias, k, n_group, topk_group, route_scale):
            T, E = logits.shape
            s = 1.0 / (1.0 + np.exp(-logits))                    # 原始亲和度：只用于权重
            choice = s + bias                                     # 选择分数：只用于排序
            # 组分数：每组最大的 k // topk_group 个选择分数之和
            g = choice.reshape(T, n_group, E // n_group)
            per = k // topk_group
            group_score = -np.sort(-g, axis=-1)[:, :, :per].sum(axis=-1)          # (T, n_group)
            keep = np.argsort(-group_score, axis=-1)[:, :topk_group]              # (T, topk_group)
            group_mask = np.zeros((T, n_group), dtype=bool)
            np.put_along_axis(group_mask, keep, True, axis=-1)
            expert_mask = np.repeat(group_mask, E // n_group, axis=1)            # (T, E)
            masked = np.where(expert_mask, choice, -np.inf)
            idx = np.argsort(-masked, axis=-1)[:, :k]                             # (T, k)
            w = np.take_along_axis(s, idx, axis=-1)
            w = w / w.sum(axis=-1, keepdims=True) * route_scale
            return idx, w
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，与 HF DeepseekV3TopkRouter 等价）',
          code: py`
            import torch

            def deepseek_v3_gate(logits, bias, k, n_group, topk_group, route_scale):
                T, E = logits.shape
                scores = logits.float().sigmoid()                          # (T, E)
                choice = scores + bias                                     # e_score_correction_bias
                group_scores = choice.view(T, n_group, -1).topk(k // topk_group, dim=-1)[0].sum(-1)
                group_idx = group_scores.topk(topk_group, dim=-1)[1]       # (T, topk_group)
                group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
                score_mask = group_mask.unsqueeze(-1).expand(T, n_group, E // n_group).reshape(T, E)
                masked = choice.masked_fill(score_mask == 0, float("-inf"))   # HF 里填 0.0（sigmoid+bias 通常 > 0）
                idx = masked.topk(k, dim=-1)[1]
                w = scores.gather(1, idx)
                w = w / w.sum(dim=-1, keepdim=True) * route_scale
                return idx, w
          `,
        },
      ],
      explanation: md`
        **思路**：「选择」和「加权」用两套分数——选择用 $s+b$（含分组屏蔽），加权用原始 $s$。

        - 组分数：\`choice.reshape(T, n_group, E/n_group)\` 后组内降序取前 \`k // topk_group\` 个求和；
        - 保留前 \`topk_group\` 组：构造 $(T, n_{group})$ 掩码，\`np.repeat\` 展开到 $(T, E)$，未保留的专家置 $-\infty$；
        - top-k 后从 \`s\`（不是 \`choice\`）gather 权重，归一化再乘缩放。

        **复杂度**：$O(T E \log E)$（用 \`argpartition\` 可到 $O(TE)$）。

        **易错点**：组分数用最大值或全组求和；用 \`choice\` 算权重；忘记归一化或缩放；屏蔽时用 0 而 \`choice\` 可能为负（bias 很负时）——用 $-\infty$ 最稳妥（HF 用 0.0，依赖 sigmoid+bias 通常为正）。
      `,
      complexity: { time: 'O(T·E·log E)', space: 'O(T·E)' },
      hints: [
        md`先写不分组的版本：\`idx = argsort(-(s + bias))[:, :k]\`，权重从 \`s\` 取。`,
        md`分组：\`g = choice.reshape(T, n_group, -1)\`，组分数 \`-np.sort(-g, -1)[:, :, :k // topk_group].sum(-1)\`。`,
        md`用 \`np.put_along_axis\` 构造组掩码，\`np.repeat(mask, E // n_group, axis=1)\` 展开到专家维，再 \`np.where(mask, choice, -np.inf)\`。`,
      ],
      setup: GATE_SETUP,
      tests: [
        {
          name: '示例：8 个专家分 4 组、保留 2 组、选 4 个',
          code: py`
            logits = np.array([[2.0, -1.0, 0.5, 0.4, 1.8, 1.7, -0.5, 0.0],
                               [0.1, 0.2, 1.5, 1.4, -1.0, 0.0, 2.2, -2.0]])
            bias = np.zeros(8)
            got = deepseek_v3_gate(logits, bias, 4, 4, 2, 2.5)
            _check(got, _ref_v3_gate(logits, bias, 4, 4, 2, 2.5))
          `,
        },
        {
          name: '不分组时等于普通 top-k',
          code: py`
            r = np.random.default_rng(1)
            logits, bias = r.normal(size=(5, 12)), r.normal(scale=0.2, size=12)
            _check(deepseek_v3_gate(logits, bias, 3, 1, 1, 1.0), _ref_v3_gate(logits, bias, 3, 1, 1, 1.0))
          `,
        },
        {
          name: 'bias 只影响选择，不影响权重',
          code: py`
            logits = np.array([[1.0, 0.9, 0.8, -3.0]])
            bias = np.array([0.0, 0.0, 0.0, 5.0])       # 强行把专家 3 拉进 top-2
            idx, w = deepseek_v3_gate(logits, bias, 2, 1, 1, 1.0)
            ci, cw = _canon(idx, w)
            assert ci.tolist() == [[0, 3]], f"应选中专家 [0, 3]，得到 {ci.tolist()}"
            s = 1 / (1 + np.exp(-np.array([1.0, -3.0])))
            assert np.allclose(cw[0], s / s.sum()), f"权重应由原始 sigmoid 分数归一化：期望 {np.round(s / s.sum(), 4)}，得到 {np.round(cw[0], 4)}"
          `,
        },
        {
          name: '每个 token 最多触达 topk_group 个组',
          hidden: true,
          code: py`
            r = np.random.default_rng(2)
            logits, bias = r.normal(size=(40, 64)), r.normal(scale=0.1, size=64)
            idx, w = deepseek_v3_gate(logits, bias, 8, 8, 4, 2.5)
            groups = np.asarray(idx) // 8
            n = max(len(set(row)) for row in groups.tolist())
            assert n <= 4, f"有 token 触达了 {n} 个组（上限 4）"
            assert np.allclose(np.asarray(w).sum(-1), 2.5), "权重和应为 route_scale"
          `,
        },
        {
          name: 'V3 规模随机对拍（E=256, 8 组选 4, top-8）',
          hidden: true,
          code: py`
            r = np.random.default_rng(3)
            for _ in range(5):
                logits, bias = r.normal(size=(16, 256)), r.normal(scale=0.05, size=256)
                _check(deepseek_v3_gate(logits, bias, 8, 8, 4, 2.5), _ref_v3_gate(logits, bias, 8, 8, 4, 2.5), "V3 规模：")
          `,
        },
        {
          name: '负 bias 下的屏蔽（选择分数可能为负）',
          hidden: true,
          code: py`
            r = np.random.default_rng(4)
            logits = r.normal(size=(10, 16))
            bias = np.full(16, -2.0) + r.normal(scale=0.1, size=16)
            _check(deepseek_v3_gate(logits, bias, 4, 4, 2, 1.0), _ref_v3_gate(logits, bias, 4, 4, 2, 1.0), "负选择分数：")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-why-top2',
          q: md`组分数为什么用「组内前 $K_r/M$ 个之和」，而不是组内最大值？`,
          a: md`token 进入一个节点后平均要在里面选 $K_r/M$ 个专家，节点的「价值」应由它能提供的前 $K_r/M$ 个专家共同决定。只看最大值会被单个尖峰误导：选了一个只有一个好专家的节点，剩下的名额只能用很差的专家填。`,
        },
        {
          id: 'f-ep',
          q: md`节点受限路由和专家并行（EP）的 all-to-all 是什么关系？`,
          a: md`EP 下每个 token 要被发送到持有其目标专家的 GPU。不加限制时一个 token 可能发往 8 个不同节点，跨节点 IB 流量随之线性增长；限制为 $M$ 个节点后，token 只需跨 IB 发送 $\le M$ 份，再在节点内经 NVLink 转发给多个专家（DeepEP 就是围绕这种两级转发设计的）。`,
        },
      ],
    },
    {
      id: 'c-aux-free-sim',
      title: '模拟无辅助损失均衡：多步 bias 更新',
      difficulty: 'medium',
      source: '大模型岗手撕 · 负载均衡',
      timeLimitMin: 20,
      description: md`
        给定 $S$ 个 batch 的亲和度（已经过 sigmoid），模拟 DeepSeek-V3 的无辅助损失均衡：

        \`\`\`python
        def simulate_aux_free(score_batches: np.ndarray, k: int, gamma: float):
            """score_batches: (S, T, E)；返回 (loads, bias)：loads 为 (S, E) 的整数数组，bias 为最终的 (E,)"""
        \`\`\`

        - bias 初始为全 0；
        - 第 $t$ 步：用**当前** bias 计算选择分数 $s + b$，每个 token 取 top-$k$；记录各专家负载 $c$（被选中次数）到 \`loads[t]\`；
        - 然后更新 $b \leftarrow b + \gamma\cdot\mathrm{sign}(\bar c - c)$，供**下一步**使用（$\bar c = Tk/E$）。

        隐藏测试会检查：结果与参考实现一致；在冷热不均的数据上，后期的负载不均衡度（MaxVio）显著低于第一步。
      `,
      starter: py`
        import numpy as np

        def simulate_aux_free(score_batches, k, gamma):
            S, T, E = score_batches.shape
            # TODO：逐步路由 → 记录负载 → 更新 bias
            return np.zeros((S, E), dtype=int), np.zeros(E)
      `,
      solution: py`
        import numpy as np

        def simulate_aux_free(score_batches, k, gamma):
            S, T, E = score_batches.shape
            bias = np.zeros(E)
            loads = np.zeros((S, E), dtype=int)
            for t in range(S):
                choice = score_batches[t] + bias                    # 用「上一步更新后」的 bias
                idx = np.argsort(-choice, axis=-1)[:, :k]
                load = np.bincount(idx.ravel(), minlength=E)
                loads[t] = load
                bias = bias + gamma * np.sign(load.mean() - load)   # 过载减、欠载加；只影响下一步
            return loads, bias
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：训练循环中的 bias 更新）',
          code: py`
            import torch

            @torch.no_grad()
            def update_router_bias(bias: torch.Tensor, topk_idx: torch.Tensor, gamma: float):
                """bias: (E,) buffer（不是 nn.Parameter）；topk_idx: 本步所有 token 的 (N, k)"""
                load = torch.bincount(topk_idx.flatten(), minlength=bias.numel()).float()
                # 多卡训练时先 all-reduce 负载，保证各卡 bias 一致
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(load)
                bias.add_(gamma * torch.sign(load.mean() - load))

            # 在 optimizer.step() 之后调用；bias 用 register_buffer 注册，不进优化器
          `,
        },
      ],
      explanation: md`
        **关键是顺序**：先用旧 bias 路由并统计负载，再更新 bias——新 bias 只影响下一步（因果性）。

        **为什么能均衡**：过载专家的 bias 每步减 $\gamma$，直到它的选择分数降到与其它专家相当；sign 规则让每步步长恒定，稳定但收敛速度受 $\gamma$ 限制。
        bias 不进入权重、也不是可学习参数，所以不会产生梯度干扰。

        **复杂度**：每步 $O(TE\log E)$。

        **易错点**：先更新再路由（用到了本步负载）；sign 方向写反；\`bincount\` 漏 \`minlength\`；把 bias 加到权重上。
      `,
      complexity: { time: 'O(S·T·E·log E)', space: 'O(S·E)' },
      hints: [
        md`循环 $S$ 步；每步 \`idx = argsort(-(scores[t] + bias))[:, :k]\`。`,
        md`\`load = np.bincount(idx.ravel(), minlength=E)\`，先存进 \`loads[t]\` 再更新 bias。`,
        md`更新：\`bias = bias + gamma * np.sign(load.mean() - load)\`。`,
      ],
      setup: SIM_SETUP,
      tests: [
        {
          name: '示例：2 步、3 个专家、top-1',
          code: py`
            sb = np.array([[[0.9, 0.5, 0.1], [0.8, 0.6, 0.2], [0.7, 0.3, 0.65]],
                           [[0.9, 0.5, 0.1], [0.8, 0.6, 0.2], [0.7, 0.3, 0.65]]])
            loads, bias = simulate_aux_free(sb, 1, 0.25)
            assert np.asarray(loads).shape == (2, 3), f"loads 形状应为 (2, 3)，得到 {np.shape(loads)}"
            assert np.asarray(loads).tolist() == [[3, 0, 0], [0, 2, 1]], f"期望 [[3,0,0],[0,2,1]]，得到 {np.asarray(loads).tolist()}"
            assert np.allclose(bias, [0.0, 0.0, 0.25]), f"最终 bias 期望 [0, 0, 0.25]，得到 {np.round(bias, 4).tolist()}"
          `,
        },
        {
          name: 'gamma = 0：bias 不变、负载等于普通 top-k',
          code: py`
            sb = _skewed(1, 3, 50, 6)
            loads, bias = simulate_aux_free(sb, 2, 0.0)
            exp, _ = _ref_sim(sb, 2, 0.0)
            assert np.array_equal(np.asarray(loads), exp), f"期望 {exp.tolist()}，得到 {np.asarray(loads).tolist()}"
            assert np.allclose(bias, 0), "gamma=0 时 bias 应保持为 0"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            for seed in range(5):
                sb = _skewed(10 + seed, 12, 64, 8)
                got_l, got_b = simulate_aux_free(sb, 2, 0.01)
                exp_l, exp_b = _ref_sim(sb, 2, 0.01)
                assert np.array_equal(np.asarray(got_l), exp_l), f"seed={seed}：第一处不同的步 {int(np.argmax((np.asarray(got_l) != exp_l).any(1)))}"
                assert np.allclose(got_b, exp_b), f"seed={seed}：最终 bias 最大误差 {np.abs(got_b - exp_b).max():.3e}"
          `,
        },
        {
          name: '因果性：第一步必须用全 0 bias',
          hidden: true,
          code: py`
            sb = _skewed(20, 4, 100, 8)
            loads, _ = simulate_aux_free(sb, 2, 0.05)
            plain = np.bincount(np.argsort(-sb[0], -1, kind="stable")[:, :2].ravel(), minlength=8)
            assert np.asarray(loads)[0].tolist() == plain.tolist(), "第一步的负载应等于不加 bias 的 top-k——是不是先更新了 bias 再路由？"
          `,
        },
        {
          name: '均衡效果：后期 MaxVio 显著下降',
          hidden: true,
          code: py`
            sb = _skewed(30, 80, 256, 8)
            loads, bias = simulate_aux_free(sb, 2, 0.01)
            loads = np.asarray(loads)
            first = _maxvio(loads[0])
            late = np.mean([_maxvio(l) for l in loads[-20:]])
            assert late < 0.4 * first, f"第一步 MaxVio={first:.2f}，最后 20 步平均 {late:.2f}，下降不足"
            assert bias[0] < bias[-1], "天然最热的专家 0 的 bias 应低于最冷的专家 7"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-freeze',
          q: md`V3 为什么在最后 500B token 把 $\gamma$ 设为 0？`,
          a: md`训练末期通常伴随学习率退火，模型和路由趋于稳定；继续以固定步长调 bias 只会带来不必要的抖动。冻结 bias 让最后阶段的路由与推理时使用的 bias 一致（推理时 bias 固定）。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-lossfree',
      freq: 3,
      q: md`DeepSeek-V3 为什么能做到「无辅助损失」的负载均衡？具体怎么做的？`,
      a: md`
        **结论**：把「均衡」从优化目标里拿出来，变成路由时的一个调度偏置：每个专家一个 bias $b_i$，只加在 top-k 的选择分数上；每步按负载 $b_i \leftarrow b_i + \gamma\,\mathrm{sign}(\bar c - c_i)$ 更新（V3：$\gamma = 0.001$，最后 500B token 冻结）。

        **原因**：辅助损失的梯度与语言模型损失冲突，$\alpha$ 越大效果越差；bias 不进入门控权重、不是可学习参数，没有任何梯度干扰。更新只用历史负载，保持因果性。

        **延伸**：① 「无辅助损失」不是绝对的——V3 仍有 $\alpha=10^{-4}$ 的序列级均衡损失兜底；② batch 级均衡比序列级更宽松，专家更易专精；③ V4 沿用该策略；Kimi K3 用 Quantile Balancing 去掉了步长 $\gamma$。
      `,
    },
    {
      id: 'qa-fine-shared',
      freq: 3,
      q: md`DeepSeekMoE 的细粒度专家和共享专家分别解决什么问题？代价是什么？`,
      a: md`
        **结论**：细粒度切分解决「专家少 → 知识混杂、难以专精」，共享专家解决「通用知识在各专家里重复 → 冗余」。

        **原因**：切分 $m$ 倍后总参数和每 token FLOPs 不变，但组合数从 $\binom{16}{2}=120$ 变成 $\binom{64}{8}\approx 4.4\times10^9$；共享专家每个 token 必经，专门承载通用能力，让路由专家去学差异化知识。

        **代价**：小专家 GEMM 效率低；每 token 派发给更多专家，all-to-all 通信增加；router 输出维变大、均衡更难。V3 用节点受限路由、DeepEP、无辅助损失均衡把这些代价压下去。V3 配置：1 共享 + 256 路由选 8，专家中间维 2048。
      `,
    },
    {
      id: 'qa-node',
      freq: 2,
      q: md`讲讲 DeepSeek-V3 的节点受限路由。`,
      a: md`
        **结论**：每个 token 最多发往 $M = 4$ 个节点：先按「节点内最高 $K_r/M = 2$ 个选择分数之和」选节点，再在选中节点的专家里取 top-8。

        **原因**：路由专家分布在 8 个节点 64 张 GPU 上；跨节点 IB 约 50 GB/s，节点内 NVLink 约 160 GB/s。限制节点数等于给最慢链路上的流量设上限，token 进入节点后再经 NVLink 分发给多个专家。

        **延伸**：V2 有类似的设备受限路由（config 中 \`topk_group=3\`）；DeepSeek-V4 去掉了路由目标节点数的限制，并重新设计了并行策略来维持训练效率。
      `,
    },
    {
      id: 'qa-sigmoid',
      freq: 2,
      q: md`为什么 DeepSeek-V3 用 sigmoid 而不是 softmax 计算亲和度？`,
      a: md`
        **结论**：sigmoid 让每个专家的亲和度独立、不互相竞争，更适合「256 个专家 + 外部 bias 调度」的设定；代价是必须在 top-k 内归一化，并用 \`routed_scaling_factor\`（V3 为 2.5）调回输出幅度。

        **原因**：softmax 在大量专家上把概率摊得很薄，某个专家分数变化会牵动所有专家的概率；sigmoid 的分数只取决于自身 logit。Loss-Free Balancing 论文的实验中 sigmoid 门控基线优于 softmax，且对不均衡更不敏感。

        **延伸**：V3 报告本身没有专门论证这一选择；DeepSeek-V4 又进一步把 sigmoid 换成了 $\sqrt{\mathrm{softplus}(\cdot)}$（见前沿一关）。
      `,
    },
  ],

  cheatsheet: md`
    **DeepSeek-V3 门控**
    $$s=\sigma(u^\top e_i),\ \ \text{选择}: \operatorname{TopK}(s+b),\ \ g_i = 2.5\cdot\frac{s_i}{\sum_{j\in S}s_j}$$
    \`\`\`python
    s = 1 / (1 + np.exp(-logits))
    idx = np.argsort(-(s + bias), -1)[:, :k]          # bias 只用于选择
    w = np.take_along_axis(s, idx, -1)                # 权重用原始 s
    w = w / w.sum(-1, keepdims=True) * 2.5
    load = np.bincount(idx.ravel(), minlength=E)
    bias += gamma * np.sign(load.mean() - load)       # 只影响下一步
    \`\`\`

    | 配置 | DeepSeekMoE 16B | V2 | V3 |
    |---|---|---|---|
    | 共享 + 路由（选） | 2 + 64（6） | 2 + 160（6） | 1 + 256（8） |
    | 打分 | softmax | softmax | sigmoid |
    | 受限 | — | 3 组 / 8 | 4 节点 / 8 |

    - γ = 0.001（前 14.3T token），之后 0；序列级损失 α = 1e-4，$f_i$ 乘 $N_r/(K_rT)$
    - 节点分 = 节点内 top-$K_r/M$（=2）选择分数之和；NVLink ≈ 3.2× IB
    - V3 训练与推理都不丢 token；细粒度：$\binom{16}{2}=120 \to \binom{64}{8}\approx4.4\times10^9$
  `,
})
