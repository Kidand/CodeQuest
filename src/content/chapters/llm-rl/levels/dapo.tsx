/**
 * llm-rl-dapo：DAPO（arXiv 2503.14476）——Clip-Higher、动态采样、token 级 loss、超长奖励整形、去 KL
 */
import { defineLevel, md, py } from '@/content/helpers'
import { ClipRegionDualViz } from '../viz/ClipRegionViz'
import { ClipHigherViz } from '../viz/ClipHigherViz'
import { DynamicSamplingViz } from '../viz/DynamicSamplingViz'
import { OverlongViz } from '../viz/OverlongViz'
import { PY_MAKE_BATCH, PY_REF_ADV } from '../shared'

const PY_REF_DAPO = py`
  def _ref_overlong(correct, lengths, L_max=20480, L_cache=4096):
      base = np.where(np.asarray(correct, dtype=bool), 1.0, -1.0)
      L = np.asarray(lengths, dtype=np.float64)
      soft = L_max - L_cache
      r_len = np.where(L <= soft, 0.0, np.where(L <= L_max, (soft - L) / L_cache, -1.0))
      return base + r_len

  def _ref_dapo_keep(correct, group_ids):
      c = np.asarray(correct, dtype=np.float64)
      _, inv = np.unique(np.asarray(group_ids), return_inverse=True)
      inv = inv.reshape(-1)
      n = np.bincount(inv).astype(np.float64)[inv]
      k = np.bincount(inv, weights=c)[inv]
      return (k > 0) & (k < n)

  def _ref_dapo_loss(logp_new, logp_old, rewards, correct, group_ids, mask, eps_low=0.2, eps_high=0.28, adv_eps=1e-6):
      keep = _ref_dapo_keep(correct, group_ids)
      m = np.asarray(mask, dtype=np.float64) * keep[:, None]
      A = _ref_group_adv(rewards, group_ids, "group", adv_eps)[:, None]
      ratio = np.exp(np.where(m > 0, logp_new - logp_old, 0.0))
      surr1 = ratio * A
      surr2 = np.clip(ratio, 1 - eps_low, 1 + eps_high) * A
      n = m.sum()
      if n == 0:
          return 0.0, {"keep": keep, "n_tokens": 0, "clip_frac": 0.0}
      loss = float((-np.minimum(surr1, surr2) * m).sum() / n)
      return loss, {"keep": keep, "n_tokens": int(n), "clip_frac": float(((surr2 < surr1) * m).sum() / n)}

  def _dapo_batch(seed, n_groups=4, G=4, T=8):
      ln, lo, lr, _, gid, m = _make_batch(seed, n_groups=n_groups, G=G, T=T)
      rng = np.random.default_rng(seed + 999)
      correct = rng.integers(0, 2, size=n_groups * G).astype(bool)
      correct[:G] = True                 # 第 0 组全对 → 过滤
      if n_groups > 2:
          correct[2 * G:3 * G] = False   # 第 2 组全错 → 过滤
      lengths = rng.integers(10000, 22000, size=n_groups * G)
      rewards = _ref_overlong(correct, lengths)
      return ln, lo, rewards, correct, gid, m
`

const SETUP = PY_MAKE_BATCH + '\n\n' + PY_REF_ADV + '\n\n' + PY_REF_DAPO

export default defineLevel({
  learn: [
    {
      id: 'why',
      title: '朴素 GRPO 在长 CoT 上的四个病灶',
      body: md`
        DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，字节 Seed 与清华 AIR，2025-03）的起点很朴素：
        用朴素 GRPO 从 **Qwen2.5-32B base** 做 RL，AIME 2024 只到 **30 分**，远低于 DeepSeek 报告的 R1-Zero-Qwen-32B（47 分）。
        他们定位到四个问题，各配一个修补：

        | 现象 | 原因 | DAPO 的修补 |
        |---|---|---|
        | **熵坍缩**：同组回答几乎一样，探索停止 | 对称 clip 的上界卡住了低概率 token 的上涨 | **Clip-Higher**：$\varepsilon_{low}=0.2,\ \varepsilon_{high}=0.28$ |
        | 有效样本越来越少、梯度噪声变大 | 全对（或全错）组优势为 0 | **动态采样**：过滤掉准确率为 0 或 1 的题 |
        | 长回答里的胡言乱语罚不动，熵与长度不健康地增长 | 样本级 loss 让长回答的每个 token 权重更小 | **token 级 loss** |
        | 截断样本带来奖励噪声 | 好推理只因太长被截断也拿惩罚 | **超长过滤 / 软超长惩罚** |

        另外两个设定：**去掉 KL**（长 CoT 训练中策略本就会远离初始模型，这个约束不必要），以及**规则奖励** $R=1$（与标准答案等价）否则 $R=-1$。

        逐项消融（AIME24 avg@32）：朴素 GRPO **30** → +超长过滤 **36** → +Clip-Higher **38** → +软超长惩罚 **41** → +token 级 loss **42** → +动态采样 **50**。
        最终 50 分超过 R1-Zero-Qwen-32B 的 47 分，且只用了其 **50%** 的训练步数。
      `,
      checkpoint: {
        id: 'cp-why',
        type: 'single',
        prompt: md`DAPO 消融中，带来**最大单步提升**的技术是：`,
        options: ['超长过滤（30 → 36）', 'Clip-Higher（36 → 38）', 'token 级 loss（41 → 42）', '动态采样（42 → 50）'],
        answer: 3,
        explanation: md`动态采样带来 +8，最大；其次是超长过滤 +6。token 级 loss 只 +1，论文说它的主要价值是**训练稳定**、让长度增长更健康，而不是分数。`,
      },
    },
    {
      id: 'clip-higher',
      title: 'Clip-Higher：把上下界解耦',
      body: md`
        PPO 的上界是**乘性**的：A>0 时一轮最多把概率推到 $(1+\varepsilon)\pi_{old}$。论文的例子：$\varepsilon=0.2$ 时，
        $\pi_{old}=0.01$ 的 token 最多涨到 $0.012$，$\pi_{old}=0.9$ 的 token 上界 $1.08$ 等于不设限。
        结果是“利用”（高概率 token 更高）很容易，“探索”（低概率 token 抬起来）极难；论文观测到**被上界裁剪的 token 平均概率 < 0.2**。

        DAPO 把 clip 写成 $\mathrm{clip}(r_{i,t},\,1-\varepsilon_{low},\,1+\varepsilon_{high})$，只调大 $\varepsilon_{high}$：

        - $\varepsilon_{high}=0.28$：给低概率 token 更多上涨空间，熵不再快速坍缩；
        - $\varepsilon_{low}$ **保持 0.2**：放宽下界会把低概率 token 的概率一路压到 0，导致采样空间坍缩。

        在下面的图里切到 “A>0”，拖动 $\varepsilon_{high}$，看可信域右边界如何移动；再看概率空间的版本。
      `,
      viz: ClipRegionDualViz,
      after: md`
        > [!NOTE] 这不是 DAPO 独有的思路
        > MiniMax-M1 认为“Wait / However / Recheck”这类反思 token 正是低概率的“分叉点”，被 clip 掉之后长 CoT 学不出来；
        > 他们的 CISPO 干脆不丢任何 token 的梯度（见 GSPO 一关）。GSPO 实验里 GRPO 基线也用了非对称的 0.2 / 0.27。
      `,
      checkpoint: {
        id: 'cp-ch',
        type: 'numeric',
        prompt: md`A>0，某 token $\pi_{old}=0.05$。在 $\varepsilon_{high}=0.28$ 下，本轮更新中它的概率最多涨到多少仍有梯度？`,
        answer: 0.064,
        tolerance: 0.01,
        explanation: md`上界 $(1+0.28)\times0.05=0.064$；对称 $\varepsilon=0.2$ 时只有 $0.06$。超过上界后 min 选中常数项，梯度为 0。`,
      },
    },
    {
      id: 'clip-prob',
      title: '概率空间里看 Clip-Higher',
      body: md`
        把上面的“ratio 空间”换成“概率空间”更直观：每一行是一个 token，灰色是 $\pi_{old}$，橙色是 $\varepsilon=0.2$ 允许的增量，绿色是 $\varepsilon_{high}$ 额外放出的空间。
        横轴用对数刻度——否则 0.01 这样的 token 在图上根本看不见，而它们恰恰是“Wait”“However”这类决定推理走向的 token。
      `,
      viz: ClipHigherViz,
      checkpoint: {
        id: 'cp-cp',
        type: 'judge',
        prompt: md`把 $\varepsilon_{low}$ 也从 0.2 调到 0.28，可以进一步提升探索。`,
        answer: false,
        explanation: md`下界控制 A<0 时概率能被压多低。放宽它会让低概率 token 更快被压到 0，采样空间收窄——与提升探索的目标相反。DAPO 明确只放宽上界。`,
      },
    },
    {
      id: 'dynamic',
      title: '动态采样：凑满“有梯度”的题',
      body: md`
        GRPO 中一组回答若全部答对，优势全为 0、梯度为 0。论文图 3b 显示：训练中准确率为 1 的题的比例**持续上升**，
        于是每个 batch 的有效题数不断下降，梯度方差变大。DAPO 的做法是**过采样 + 过滤**：

        $$
        \text{s.t.}\quad 0<\big|\{o_i\mid \texttt{is\_equivalent}(a,o_i)\}\big|<G
        $$

        即只保留**有对有错**的题，持续采样直到 batch 填满。注意过滤条件是**准确率**而不是奖励——加了长度惩罚后，全对组的奖励也可能各不相同。

        代价是 rollout 量变得不固定。论文的辩护：同步 RL 系统里生成时间主要由**长尾样本**决定，多采几道题对墙钟时间影响有限；而且达到同样性能所需的训练步数更少（图 6）。
      `,
      viz: DynamicSamplingViz,
      after: md`
        > [!INTERVIEW] 和 Dr.GRPO 的“难度偏差”是什么关系？
        > 动态采样处理的是 $p\in\{0,1\}$ 的**两个端点**（优势恰为 0）；Dr.GRPO 讨论的是端点附近（$p$ 接近 0 或 1 但不等于）的题被 std 归一化**放大权重**。
        > 前者扔掉没信号的题，后者纠正有信号的题之间的权重。两者可以同时用。
      `,
      checkpoint: {
        id: 'cp-dyn',
        type: 'pick',
        prompt: md`G=4，下列 6 道题的回答正确性如下。动态采样会**保留**哪些题？`,
        items: ['✓✓✓✓', '✓✗✗✗', '✗✗✗✗', '✓✓✓✗', '✗✓✗✓', '✗✗✗✗（但长度惩罚不同）'],
        answer: [1, 3, 4],
        explanation: md`保留条件是 $0<$ 答对数 $<G$。全对（①）、全错（③）都被过滤；⑥ 虽然加了长度惩罚后奖励各不相同，但准确率仍为 0，DAPO 的约束按 \`is_equivalent\` 计数，同样过滤。`,
      },
    },
    {
      id: 'token-loss',
      title: 'token 级 loss：让每个 token 等权',
      body: md`
        GRPO 原式先在每条回答内对 token 平均（$\frac1{|o_i|}\sum_t$），再对回答平均——**每条回答等权**，于是长回答里每个 token 的权重被稀释。
        DAPO 改为对整个 batch 的所有 token 一起平均：

        $$
        \mathcal J_{DAPO}=\mathbb E\Big[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\big(r_{i,t}\hat A_{i,t},\ \mathrm{clip}(r_{i,t},1-\varepsilon_{low},1+\varepsilon_{high})\hat A_{i,t}\big)\Big]
        $$

        论文给出的两点理由：① 高质量的长回答里，推理模式能被充分学到；② **过长样本里常有胡言乱语和重复**，样本级 loss 罚不动它们，
        导致熵和长度不健康地上涨（论文图 4）。“一个生成模式无论出现在多长的回答里，都被同等地鼓励或抑制。”

        > [!WARN] token-mean 并不完全无偏
        > Dr.GRPO 指出，按“batch 内 token 总数”归一化（trl / verl 的 token-mean）仍会让梯度尺度随 batch 的长度分布变化；
        > 它推荐除以一个**全局常数**（如最大生成长度）。下一关专门比较这几种聚合。
      `,
      checkpoint: {
        id: 'cp-tok',
        type: 'numeric',
        prompt: md`一组只有 2 条回答，长度 100 和 1000。长回答中**每个 token** 的权重，token 级 loss 是样本级 loss 的多少倍？`,
        answer: 1.818,
        tolerance: 0.01,
        explanation: md`样本级：$\frac1G\cdot\frac1{|o_i|}=\frac1{2\times1000}=0.0005$；token 级：$\frac1{100+1000}=0.000909$；比值 $=2000/1100\approx1.818$。短回答则反过来：$\frac{1/1100}{1/200}=0.18$ 倍。`,
      },
    },
    {
      id: 'overlong',
      title: '超长奖励整形：别因为“太长”惩罚好推理',
      body: md`
        生成时设了最大长度，超出就截断。默认给截断样本一个惩罚奖励，会引入噪声：一段合理的推理仅因为长就被判负。
        DAPO 先试了 **Overlong Filtering**（直接 mask 掉截断样本的 loss），发现训练明显更稳、分数更高；
        再提出 **Soft Overlong Punishment**，加到正确性奖励上：

        $$
        R_{length}(y)=\begin{cases}0, & |y|\le L_{max}-L_{cache}\\[2pt] \dfrac{(L_{max}-L_{cache})-|y|}{L_{cache}}, & L_{max}-L_{cache}<|y|\le L_{max}\\[6pt] -1, & L_{max}<|y|\end{cases}
        $$

        论文取期望最大长度 $16{,}384$、缓冲区 $L_{cache}=4{,}096$，所以生成上限 $L_{max}=20{,}480$。
        缓冲区内惩罚从 0 线性降到 −1，给模型一个“快超了”的平滑信号，而不是悬崖。
      `,
      viz: OverlongViz,
      after: md`
        **其余超参**：verl 框架，AdamW、恒定学习率 $10^{-6}$、前 20 个 rollout step 线性 warm-up；每步 512 道题 × 每题 16 个回答；
        mini-batch 512，即每个 rollout step 做 **16 次**梯度更新（所以 clip 很关键）；评测 AIME 重复 32 次取 avg@32，温度 1.0、top-p 0.7。
      `,
      checkpoint: {
        id: 'cp-ol',
        type: 'fill',
        prompt: md`按论文设置（$L_{max}=20480,\ L_{cache}=4096$），一个**答对**、长度 18432 的回答：$R_{length}$ = ____(1)____，总奖励 = ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['-0.5'], numeric: true, tolerance: 0.001 },
          { label: '(2)', answers: ['0.5'], numeric: true, tolerance: 0.001 },
        ],
        explanation: md`$L_{max}-L_{cache}=16384<18432\le20480$，处在缓冲区：$R_{length}=(16384-18432)/4096=-0.5$；总奖励 $=1-0.5=0.5$。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-four',
      type: 'multi',
      difficulty: 1,
      prompt: md`以下哪些是 DAPO 提出 / 采用的做法？（多选）`,
      options: ['非对称裁剪 ε_low=0.2、ε_high=0.28', '过滤准确率为 0 或 1 的题', '优势只减均值、不除 std', 'token 级 policy gradient loss', '去掉 KL 正则', '用过程奖励模型给每步打分'],
      answer: [0, 1, 3, 4],
      explanation: md`C 是 Dr.GRPO 的修改（DAPO 仍用组内 $(r-\text{mean})/\text{std}$）；F 不是：DAPO 用规则奖励 ±1，并明确以避免 reward hacking 为由不用奖励模型。`,
    },
    {
      id: 'q-eps',
      type: 'numeric',
      difficulty: 2,
      prompt: md`A>0，$\pi_{old}=0.01$。$\varepsilon_{high}=0.28$ 时该 token 在本轮最多被推到多少（仍有梯度）？`,
      answer: 0.0128,
      tolerance: 0.01,
      explanation: md`$(1+0.28)\times0.01=0.0128$。对称 $\varepsilon=0.2$ 时为 $0.012$——差别看似很小，但对 0.01 这种 token 是相对 +28% vs +20% 的上涨空间，累积多步后差距显著。`,
    },
    {
      id: 'q-low',
      type: 'single',
      difficulty: 2,
      prompt: md`DAPO 为什么只放宽上界、保持 $\varepsilon_{low}=0.2$？`,
      options: [
        '下界对 A>0 的 token 不起作用，调了也没用',
        '放宽下界会把低概率 token 的概率压到 0，导致采样空间坍缩',
        '为了与 PPO 的默认值兼容',
        '下界越大 KL 越大，而 DAPO 去掉了 KL',
      ],
      answer: 1,
      explanation: md`下界作用于 A<0：允许概率被压得更低。放宽它会让本就稀少的 token 更快消失，与“保持探索”背道而驰。A 的前半句对，但不是原因；C、D 与论文无关。`,
    },
    {
      id: 'q-dyn-cost',
      type: 'judge',
      difficulty: 2,
      prompt: md`动态采样要多采样很多题，因此在同步 RL 系统中必然显著拖慢整体训练。`,
      answer: false,
      explanation: md`论文的论点：同步、非流水线的系统中生成时间主要被**长尾样本**主导，多采一些短样本对墙钟影响不大；且因为每步都是有效样本，收敛所需步数更少，整体时间甚至更短（图 6）。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`按 DAPO 消融表的叠加顺序排列（AIME24 avg@32 从 30 涨到 50）：`,
      items: ['朴素 GRPO（30）', '+ 超长过滤（36）', '+ Clip-Higher（38）', '+ 软超长惩罚（41）', '+ token 级 loss（42）', '+ 动态采样（50）'],
      explanation: md`论文表 1 的顺序与数字。注意这是**累加**消融，单项贡献受叠加顺序影响，不能简单理解为“动态采样单独值 8 分”。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把 DAPO 的技术与它针对的问题配对：`,
      pairs: [
        ['Clip-Higher', '熵坍缩、探索不足'],
        ['动态采样', '优势为 0 的组越来越多'],
        ['token 级 loss', '长回答中的重复与胡言乱语罚不动'],
        ['软超长惩罚', '截断样本的奖励噪声'],
      ],
      explanation: md`四个技术各对一个病灶；去 KL 与规则奖励是额外的设定。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`$L_{max}=20480,\ L_{cache}=4096$。下列长度中，哪些回答的 $R_{length}$ **严格小于 0**？`,
      items: ['8192', '16384', '16385', '20480', '20481', '12000'],
      answer: [2, 3, 4],
      explanation: md`阈值 $L_{max}-L_{cache}=16384$：$|y|\le16384$ 时为 0（16384 恰好不罚）；16385 为 $-1/4096$；20480 为 $-1$；超过 20480 为 $-1$。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的软超长惩罚实现有 **2 处** bug（约定同论文），点出来：`,
      code: py`
        def overlong_reward(correct, lengths, L_max=20480, L_cache=4096):
            base = np.where(correct, 1.0, -1.0)
            L = np.asarray(lengths, dtype=np.float64)
            soft = L_max - L_cache
            r_len = np.zeros_like(L)
            buf = (L > soft) & (L <= L_max)
            r_len[buf] = (L[buf] - soft) / L_cache
            r_len[L > L_max] = -1.0
            r_len[L < soft] = 0.0
            return base * r_len
      `,
      bugLines: [7, 10],
      explanation: md`第 7 行符号反了：应为 \`(soft - L[buf]) / L_cache\`（缓冲区内从 0 降到 −1），写反变成了**奖励变长**。第 10 行应为相加 \`base + r_len\`：长度惩罚是加到正确性奖励上的，相乘会让短回答（r_len=0）全部得 0 分。第 9 行冗余但无害。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 DAPO 的动态采样过滤（\`correct\` 为 \`(n_groups, G)\` 的 0/1 数组，返回保留的组下标；有 2 行干扰项）：`,
      lines: [
        'def keep_groups(correct):',
        '    k = correct.sum(axis=1)',
        '    G = correct.shape[1]',
        '    keep = (k > 0) & (k < G)',
        '    return np.flatnonzero(keep)',
      ],
      distractors: ['    keep = (k >= 0) & (k <= G)', '    k = correct.sum(axis=0)'],
      explanation: md`对每组（axis=1）数答对个数，保留 $0<k<G$。用 \`>=\`/\`<=\` 什么都过滤不掉；axis=0 是在跨组统计同一位置。`,
    },
    {
      id: 'q-kl',
      type: 'single',
      difficulty: 1,
      prompt: md`DAPO 去掉 KL 项的理由是：`,
      options: [
        'k3 估计有偏，无法使用',
        '训练长 CoT 时模型分布本就会大幅偏离初始模型，这个约束没有必要',
        'KL 项的计算太耗显存',
        '去掉 KL 后可以用更大的学习率',
      ],
      answer: 1,
      explanation: md`论文原意：RLHF 中 KL 用来防止偏离初始模型，但长 CoT 推理训练中模型分布会显著偏离初始模型，这一限制并不必要。A 不是理由（k3 本身无偏）；C、D 论文未提。`,
    },
    {
      id: 'q-tok',
      type: 'single',
      difficulty: 3,
      prompt: md`一个 batch 内，一条 8000 token 的**错误**回答里混着大段重复文本。与样本级 loss 相比，token 级 loss 对这些重复 token 的作用是：`,
      options: [
        '惩罚更弱，因为 token 越多每个 token 分到的权重越小',
        '惩罚更强：每个 token 与其它回答的 token 等权，不再被 1/|o_i| 稀释',
        '完全一样，两种聚合只差一个全局常数',
        '不惩罚，因为错误回答的优势在 token 级 loss 中被置零',
      ],
      answer: 1,
      explanation: md`样本级下这条回答整体只占 $1/G$ 的权重，每个 token 再除以 8000；token 级下每个 token 的权重是 $1/\sum|o_i|$，与短回答的 token 相同，长回答的总影响随长度增长。A 描述的正是样本级的问题；C 只有所有回答等长时才成立。`,
    },
    {
      id: 'q-reward',
      type: 'judge',
      difficulty: 1,
      prompt: md`DAPO 的规则奖励是：答对 1 分，答错 0 分。`,
      answer: false,
      explanation: md`DAPO 式 (7)：与标准答案等价得 $+1$，否则 $-1$。（Dr.GRPO 用的是 1/0。）组内标准化后两者给出的优势相同，但不除 std 的算法会受影响。`,
    },
  ],

  guided: [
    {
      id: 'g-overlong',
      title: '实现 DAPO 的奖励：±1 + 软超长惩罚',
      intro: md`
        实现 \`overlong_reward(correct, lengths, L_max, L_cache)\`：\`correct\` 是 \`(B,)\` 的布尔数组，\`lengths\` 是 \`(B,)\` 的回答长度。
        返回 \`(B,)\` 的总奖励 = 正确性奖励（答对 +1，答错 −1）+ $R_{length}$。全部用向量化的 \`np.where\` 写。
      `,
      template: py`
        import numpy as np

        def overlong_reward(correct, lengths, L_max=20480, L_cache=4096):
            base = {{1}}                                   # 答对 +1，答错 -1
            L = np.asarray(lengths, dtype=np.float64)
            soft = L_max - L_cache                         # 期望最大长度
            r_len = np.where(L <= soft, 0.0, {{2}})        # 缓冲区内线性扣分
            r_len = {{3}}                                  # 超过 L_max 扣满 1
            return {{4}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`把布尔数组映射成 ±1。`,
          answer: 'np.where(np.asarray(correct, dtype=bool), 1.0, -1.0)',
          accept: ['np.where(correct, 1.0, -1.0)', '2.0 * np.asarray(correct, dtype=np.float64) - 1.0'],
          explain: md`DAPO 用 ±1 而不是 1/0。组内标准化下两者等价，但这里的长度惩罚是**加法**，零点位置会影响“错且长”与“对且长”的相对排序。`,
        },
        {
          blank: 2,
          instruction: md`缓冲区内：$\frac{(L_{max}-L_{cache})-|y|}{L_{cache}}$，从 0 线性降到 −1。`,
          answer: '(soft - L) / L_cache',
          explain: md`在 $|y|=$ soft 时为 0，在 $|y|=L_{max}$ 时为 −1，连续无跳变。`,
        },
        {
          blank: 3,
          instruction: md`长度超过 $L_{max}$ 的位置统一设为 −1（否则线性公式会给出比 −1 更低的值）。`,
          answer: 'np.where(L > L_max, -1.0, r_len)',
          accept: ['np.maximum(r_len, -1.0)'],
          explain: md`截断样本实际长度就是 $L_{max}$，但在“长度预算课程”等场景下长度可能超出，显式兜底更安全。用 \`np.maximum(r_len, -1)\` 等价。`,
        },
        {
          blank: 4,
          instruction: md`长度惩罚**加到**正确性奖励上。`,
          answer: 'base + r_len',
          explain: md`缓冲区内答对的回答奖励在 $[0, 1]$，答错的在 $[-2, -1]$：正确性仍然是主导信号，长度只是平滑的次要信号。`,
        },
      ],
      setup: SETUP,
      tests: [
        {
          name: '论文设置下的典型长度',
          code: py`
            c = np.array([True, True, True, False, False])
            L = np.array([8000, 18432, 20480, 18432, 30000])
            got = overlong_reward(c, L)
            exp = np.array([1.0, 0.5, 0.0, -1.5, -2.0])
            assert np.allclose(got, exp), f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '边界与随机对拍',
          code: py`
            rng = np.random.default_rng(0)
            for Lm, Lc in [(20480, 4096), (8192, 1024), (4096, 4096)]:
                L = np.concatenate([[Lm - Lc, Lm - Lc + 1, Lm, Lm + 1], rng.integers(1, Lm + 2000, size=50)])
                c = rng.integers(0, 2, size=len(L)).astype(bool)
                got = overlong_reward(c, L, Lm, Lc)
                exp = _ref_overlong(c, L, Lm, Lc)
                assert np.allclose(got, exp), f"L_max={Lm}, L_cache={Lc}: 最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
      ],
      outro: md`
        超长整形只改**奖励**，不改 loss。另一种做法“超长过滤”则是改 **loss mask**：截断样本整条不参与梯度。论文两者都做了实验，消融表中先加过滤（+6）、再加软惩罚（+3）。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-dapo-loss',
      title: '手写 DAPO loss：动态采样过滤 + Clip-Higher + token 级聚合',
      difficulty: 'hard',
      source: '大模型岗手撕 · DAPO',
      timeLimitMin: 25,
      description: md`
        \`\`\`python
        def dapo_loss(logp_new, logp_old, rewards, correct, group_ids, mask,
                      eps_low=0.2, eps_high=0.28, adv_eps=1e-6) -> tuple[float, dict]: ...
        \`\`\`

        - \`logp_new / logp_old / mask\`：\`(B, T)\`；padding 位置的 logp 是垃圾值；
        - \`rewards\`：\`(B,)\`，已含超长惩罚；\`correct\`：\`(B,)\` 布尔，是否答对；\`group_ids\`：\`(B,)\`，同 id 为同一题（可乱序）。

        **步骤**
        1. **动态采样过滤**：一组中答对数 $k$ 满足 $0<k<G$ 才保留（按 \`correct\` 判断，不按 rewards）；
        2. 组内优势 $(r-\text{mean})/(\text{std}_{ddof=1}+\texttt{adv\_eps})$（对所有组计算即可，被过滤的组不参与 loss）；
        3. $-\min(r\hat A,\ \mathrm{clip}(r,1-\varepsilon_{low},1+\varepsilon_{high})\hat A)$，**无 KL**；
        4. **token 级聚合**：对所有保留行的有效 token 求和，除以这些 token 的总数。若没有任何保留的 token，loss 返回 \`0.0\`。

        **返回** \`(loss, stats)\`，\`stats\` 含 \`"keep"\`（\`(B,)\` 布尔）、\`"n_tokens"\`（参与 loss 的 token 数，int）、
        \`"clip_frac"\`（参与 loss 的 token 中 \`surr2 < surr1\` 的比例，无 token 时为 0.0）。
      `,
      starter: py`
        import numpy as np

        def dapo_loss(logp_new, logp_old, rewards, correct, group_ids, mask,
                      eps_low=0.2, eps_high=0.28, adv_eps=1e-6):
            # TODO 1: 按 group_ids 统计每组答对数，得到 keep (B,)
            # TODO 2: 组内优势
            # TODO 3: 非对称 clip 的 token 级 loss（无 KL），只统计保留行的有效 token
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def dapo_loss(logp_new, logp_old, rewards, correct, group_ids, mask,
                      eps_low=0.2, eps_high=0.28, adv_eps=1e-6):
            r = np.asarray(rewards, dtype=np.float64)
            c = np.asarray(correct, dtype=np.float64)
            _, inv = np.unique(group_ids, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv).astype(np.float64)

            # 1) 动态采样：0 < 答对数 < G
            k = np.bincount(inv, weights=c)
            keep = (k[inv] > 0) & (k[inv] < cnt[inv])

            # 2) 组内优势（ddof=1）
            dev = r - (np.bincount(inv, weights=r) / cnt)[inv]
            std = np.sqrt(np.bincount(inv, weights=dev ** 2) / (cnt - 1))
            A = (dev / (std[inv] + adv_eps))[:, None]

            # 3) Clip-Higher，无 KL
            m = np.asarray(mask, dtype=np.float64) * keep[:, None]
            ratio = np.exp(np.where(m > 0, logp_new - logp_old, 0.0))
            surr1 = ratio * A
            surr2 = np.clip(ratio, 1 - eps_low, 1 + eps_high) * A

            # 4) token 级聚合：Σ_i Σ_t / Σ_i |o_i|
            n = m.sum()
            if n == 0:
                return 0.0, {"keep": keep, "n_tokens": 0, "clip_frac": 0.0}
            loss = float((-np.minimum(surr1, surr2) * m).sum() / n)
            clip_frac = float(((surr2 < surr1) * m).sum() / n)
            return loss, {"keep": keep, "n_tokens": int(n), "clip_frac": clip_frac}
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，连续分组：rewards / correct 可 view 成 (n_groups, G)）',
          code: py`
            import torch

            def dapo_loss_torch(logp_new, logp_old, rewards, correct, mask, G,
                                eps_low=0.2, eps_high=0.28, adv_eps=1e-6):
                r = rewards.view(-1, G)
                k = correct.view(-1, G).sum(-1, keepdim=True)
                keep = ((k > 0) & (k < G)).expand(-1, G).reshape(-1)          # (B,)
                adv = (r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + adv_eps)
                A = adv.reshape(-1, 1).detach()

                m = mask.float() * keep.unsqueeze(-1)
                ratio = torch.exp((logp_new - logp_old.detach()) * m)
                surr = torch.min(ratio * A, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * A)
                return -(surr * m).sum() / m.sum().clamp(min=1)               # token-mean
          `,
        },
      ],
      explanation: md`
        **与 GRPO 的差异只有四处**：过滤、非对称 clip、去 KL、token-mean。写成一个函数时最容易错的是：

        - 过滤要**按准确率**（\`correct\`）判断。rewards 已含长度惩罚，全对组的奖励也可能不同，按 rewards 判断会漏过滤。
        - token-mean 的分母是**保留行的有效 token 总数**，不是 \`B*T\`，也不是保留行数。
        - 真实系统中过滤发生在**采样阶段**（不够就继续采），这里在 loss 里用 mask 模拟同样的效果。

        **复杂度**：$O(BT)$。
      `,
      complexity: { time: 'O(B·T)', space: 'O(B·T)' },
      hints: [
        md`\`np.bincount(inv, weights=correct)\` 得到每组答对数 \`k\`，再和每组样本数比较。`,
        md`把 \`keep\` 乘进 mask：\`m = mask * keep[:, None]\`，后面所有统计都用 \`m\`。`,
        md`loss = \`(-min(surr1, surr2) * m).sum() / m.sum()\`；记得处理 \`m.sum() == 0\`。`,
      ],
      setup: SETUP,
      tests: [
        {
          name: 'keep 与字段',
          code: py`
            ln, lo, r, c, gid, m = _dapo_batch(0)
            loss, st = dapo_loss(ln, lo, r, c, gid, m)
            keep = np.asarray(st["keep"])
            assert keep.shape == (16,), f"keep 形状应为 (16,)，得到 {keep.shape}"
            assert not keep[:4].any(), "第 0 组全对，应被过滤"
            assert not keep[8:12].any(), "第 2 组全错，应被过滤"
            assert keep.tolist() == _ref_dapo_keep(c, gid).tolist(), f"keep 不正确：{keep.tolist()}"
          `,
        },
        {
          name: '按 correct 而不是 rewards 过滤',
          code: py`
            ln, lo, r, c, gid, m = _dapo_batch(1)
            r = r.copy(); r[:4] = [1.0, 0.7, 0.2, 0.9]          # 全对组但长度惩罚不同
            _, st = dapo_loss(ln, lo, r, c, gid, m)
            assert not np.asarray(st["keep"])[:4].any(), "全对组（即使奖励各不相同）也应被过滤"
          `,
        },
        {
          name: '全部被过滤时 loss = 0',
          code: py`
            ln, lo, r, c, gid, m = _dapo_batch(2)
            c = np.ones_like(c)
            loss, st = dapo_loss(ln, lo, r, c, gid, m)
            assert loss == 0.0 and st["n_tokens"] == 0, f"得到 loss={loss}, n_tokens={st['n_tokens']}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            for seed in range(30):
                ln, lo, r, c, gid, m = _dapo_batch(50 + seed, n_groups=5, G=4, T=9)
                loss, st = dapo_loss(ln, lo, r, c, gid, m)
                el, es = _ref_dapo_loss(ln, lo, r, c, gid, m)
                assert abs(loss - el) < 1e-8, f"seed={seed}: 期望 {el:.8f}，得到 {loss:.8f}"
                assert st["n_tokens"] == es["n_tokens"], f"seed={seed}: n_tokens 期望 {es['n_tokens']}，得到 {st['n_tokens']}"
                assert abs(st["clip_frac"] - es["clip_frac"]) < 1e-9, f"seed={seed}: clip_frac 期望 {es['clip_frac']}，得到 {st['clip_frac']}"
          `,
        },
        {
          name: '非对称 clip 与乱序分组',
          hidden: true,
          code: py`
            ln, lo, r, c, gid, m = _dapo_batch(9, n_groups=4, G=6, T=8)
            ln = lo + 3 * (ln - lo)
            perm = np.random.default_rng(1).permutation(len(r))
            ids = np.array([5, 11, 2, 40])[gid]
            args = (ln[perm], lo[perm], r[perm], c[perm], ids[perm], m[perm])
            for el_, eh_ in [(0.2, 0.28), (0.1, 0.4)]:
                loss, _ = dapo_loss(*args, eps_low=el_, eps_high=eh_)
                el, _ = _ref_dapo_loss(*args, eps_low=el_, eps_high=eh_)
                assert abs(loss - el) < 1e-8, f"eps=({el_},{eh_}): 期望 {el:.8f}，得到 {loss:.8f}"
          `,
        },
        {
          name: 'token 级聚合（不是行平均）',
          hidden: true,
          code: py`
            ln, lo, r, c, gid, m = _dapo_batch(13, n_groups=4, G=4, T=10)
            loss, _ = dapo_loss(ln, lo, r, c, gid, m)
            el, _ = _ref_dapo_loss(ln, lo, r, c, gid, m)
            assert abs(loss - el) < 1e-8, f"期望 token-mean 结果 {el:.8f}，得到 {loss:.8f}（是不是先行内平均了？）"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-async',
          q: md`如果改成异步 / 流水线式 RL 系统，动态采样还“几乎免费”吗？`,
          a: md`不一定。论文的论据建立在“同步、非流水线、生成时间被长尾主导”上。异步系统中 rollout 吞吐是瓶颈，过滤掉的样本是实打实的浪费；常见做法是提前用历史通过率筛题（离线难度过滤），或把被过滤的题留到后面复用。`,
        },
        {
          id: 'f-entropy',
          q: md`除了 Clip-Higher，还有什么办法对付熵坍缩？`,
          a: md`熵正则（加 $-\alpha H$ 到 loss，但系数敏感、容易熵爆炸）；Clip-Cov / KL-Cov（只限制“概率与优势协方差高”的 token 的更新）；只在高熵“分叉 token”上算梯度；提高采样温度；CISPO 保留所有 token 的梯度。详见「训练动力学」一关。`,
        },
      ],
    },
    {
      id: 'c-fill-batch',
      title: '动态采样的“凑 batch”循环',
      difficulty: 'medium',
      source: 'DAPO Algorithm 1',
      timeLimitMin: 12,
      description: md`
        DAPO 的 Algorithm 1：不断采样、过滤掉全对/全错的组、放入缓冲区，直到缓冲区里有 \`batch_size\` 个组。

        \`\`\`python
        def fill_batch(group_stream, batch_size: int) -> tuple[list[int], int]: ...
        \`\`\`

        - \`group_stream\`：**可迭代对象（可能是生成器，没有 len）**，依次产出每组的正确性数组（1-D，0/1）；
        - 按顺序消费，保留 $0<\sum<G$ 的组；凑满 \`batch_size\` 个后**立即停止**（不能多消费，采样很贵）；
        - 返回 \`(kept, consumed)\`：被保留组在流中的下标列表（按顺序），以及实际消费的组数。流耗尽时返回已有的结果。
      `,
      starter: py`
        def fill_batch(group_stream, batch_size):
            kept, consumed = [], 0
            # TODO: 逐个消费 group_stream，凑满就停
            return kept, consumed
      `,
      solution: py`
        def fill_batch(group_stream, batch_size):
            kept, consumed = [], 0
            if batch_size <= 0:
                return kept, consumed
            for idx, group in enumerate(group_stream):
                consumed += 1
                k, G = int(sum(group)), len(group)
                if 0 < k < G:                 # 有对有错 → 有梯度
                    kept.append(idx)
                    if len(kept) == batch_size:
                        break                 # 立即停止，不多采一组
            return kept, consumed
      `,
      solutionAlt: [
        {
          label: 'PyTorch / 训练框架中的写法（仅展示）',
          code: py`
            # verl 的 DAPO recipe 思路：每轮生成一个 batch → 过滤 → 不够就继续生成并拼接
            buffer = []
            while len(buffer) < train_batch_size:
                batch = rollout(next(prompt_iter))            # 每题采 G 个
                acc = batch["correct"].view(-1, G).float().mean(-1)
                ok = (acc > 0) & (acc < 1)
                buffer.extend(batch.select_groups(ok))
            buffer = buffer[:train_batch_size]
          `,
        },
      ],
      explanation: md`
        要点只有两个：**惰性消费**（用 for 迭代，不要 \`list(stream)\`），以及**凑满即停**。真实系统里每“消费”一组都是一次 G 路采样，多一组就是多一份 GPU 时间。

        复杂度 $O(\text{consumed}\cdot G)$。
      `,
      hints: [md`用 \`for idx, g in enumerate(group_stream)\`，每拿到一组就 \`consumed += 1\`。`, md`保留条件 \`0 < sum(g) < len(g)\`；\`len(kept) == batch_size\` 时 \`break\`。`],
      setup: py`
        class _Stream:
            """只能迭代一次的流，记录被消费了多少组"""
            def __init__(self, groups):
                self.groups = list(groups)
                self.taken = 0
            def __iter__(self):
                for g in self.groups:
                    self.taken += 1
                    yield g
      `,
      tests: [
        {
          name: '示例',
          code: py`
            s = _Stream([[1, 1, 1], [1, 0, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]])
            kept, consumed = fill_batch(s, 2)
            assert kept == [1, 3] and consumed == 4, f"期望 ([1, 3], 4)，得到 {(kept, consumed)}"
          `,
        },
        {
          name: '流耗尽',
          code: py`
            kept, consumed = fill_batch(_Stream([[1, 1], [0, 0], [1, 0]]), 5)
            assert kept == [2] and consumed == 3, f"期望 ([2], 3)，得到 {(kept, consumed)}"
          `,
        },
        {
          name: '不能多消费（惰性）',
          hidden: true,
          code: py`
            import random
            rnd = random.Random(3)
            groups = [[rnd.randint(0, 1) for _ in range(4)] for _ in range(500)]
            s = _Stream(groups)
            kept, consumed = fill_batch(s, 10)
            exp_kept = [i for i, g in enumerate(groups) if 0 < sum(g) < 4][:10]
            assert kept == exp_kept, f"期望 {exp_kept}，得到 {kept}"
            assert consumed == exp_kept[-1] + 1, f"应在第 {exp_kept[-1] + 1} 组停止，实际 consumed={consumed}"
            assert s.taken == consumed, f"实际从流中拿走了 {s.taken} 组，但报告 consumed={consumed}——是不是先 list() 了整个流？"
          `,
        },
        {
          name: 'batch_size = 0 与生成器输入',
          hidden: true,
          code: py`
            kept, consumed = fill_batch(iter([[1, 0]]), 0)
            assert kept == [] and consumed == 0, f"batch_size=0 时不应消费，得到 {(kept, consumed)}"
            gen = ([i % 2, 1] for i in range(6))
            kept, consumed = fill_batch(gen, 2)
            assert kept == [0, 2] and consumed == 3, f"期望 ([0, 2], 3)，得到 {(kept, consumed)}"
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-dapo',
      freq: 3,
      q: md`DAPO 相对 GRPO 改了哪几处？每处解决什么问题？`,
      a: md`
        **结论**：四个技术 + 两个设定。Clip-Higher（ε_low=0.2、ε_high=0.28）防熵坍缩；动态采样过滤全对/全错题，保持有效 batch；token 级 loss 让长回答中的每个 token 等权；超长过滤 / 软超长惩罚去除截断带来的奖励噪声。另外去掉 KL、用 ±1 规则奖励。

        **原因**：朴素 GRPO 从 Qwen2.5-32B base 训练只有 AIME 30 分，逐项加上后到 50 分（超过 R1-Zero-Qwen-32B 的 47，步数减半）。

        **延伸**：动态采样贡献最大（+8）；token 级 loss 分数只 +1，但稳定了熵与长度。这些技术后来被 TRL、verl 等框架作为选项内置。
      `,
    },
    {
      id: 'qa-cliphigher',
      freq: 3,
      q: md`为什么对称的 clip 会导致熵坍缩？Clip-Higher 为什么只调上界？`,
      a: md`
        **结论**：上界是乘性的，$\pi_{old}$ 小的 token 一轮只能涨一点点，探索性 token 永远抬不起来；而高概率 token 几乎不受限，策略迅速变得确定、熵下降。

        **原因**：A>0 时允许的最大概率是 $(1+\varepsilon)\pi_{old}$；论文观测被上界裁剪的 token 平均概率低于 0.2。调大 $\varepsilon_{high}$ 给这些 token 更多空间。下界作用于 A<0，放宽它会把低概率 token 压到 0，采样空间坍缩，所以保持 0.2。

        **延伸**：CISPO 更进一步——不丢任何 token 的梯度，只裁剪 IS 权重；Clip-Cov / KL-Cov 则针对“高协方差”token 精确限制。
      `,
    },
    {
      id: 'qa-overlong',
      freq: 2,
      q: md`超长样本怎么处理？软超长惩罚的公式是什么？`,
      a: md`
        **结论**：两种：超长过滤（截断样本整条 mask 掉 loss）；软超长惩罚（在 $[L_{max}-L_{cache}, L_{max}]$ 内从 0 线性降到 −1，超过 $L_{max}$ 记 −1，加到正确性奖励上）。

        **原因**：截断样本可能是“推理正确但太长”，直接给负奖励会让模型困惑；缓冲区给出平滑信号。论文用期望长度 16,384、缓冲 4,096、上限 20,480。

        **延伸**：长度控制后来发展为“推理强度”档位——DeepSeek-V4 为不同推理档位训练专家时使用了不同的长度惩罚与上下文窗口。
      `,
    },
    {
      id: 'qa-dyn',
      freq: 2,
      q: md`动态采样的代价是什么？有没有替代方案？`,
      a: md`
        **结论**：代价是每步 rollout 量不固定、需要过采样；替代方案是离线/在线难度筛选。

        **原因**：训练后期全对题越来越多，要采更多题才能凑满 batch。论文认为同步系统中长尾样本主导生成时间，代价可接受，且收敛步数更少。

        **延伸**：工程上常结合历史通过率提前剔除过易/过难的题（如只保留通过率在 (0,1) 的题）、或按难度分桶做课程；在异步系统中更依赖这些离线手段。
      `,
    },
  ],

  cheatsheet: md`
    **DAPO = GRPO + 4 招 − KL**
    $$\mathcal J=\frac{1}{\sum_i|o_i|}\sum_i\sum_t\min\big(r\hat A,\ \mathrm{clip}(r,1-\varepsilon_{low},1+\varepsilon_{high})\hat A\big),\ \ \text{s.t. } 0<\#\text{correct}<G$$

    | 技术 | 设定 | 消融 |
    |---|---|---|
    | 超长过滤 | 截断样本 mask | 30 → 36 |
    | Clip-Higher | ε_low 0.2 / ε_high 0.28 | → 38 |
    | 软超长惩罚 | L_max 20480，L_cache 4096 | → 41 |
    | token 级 loss | 除以 Σ\|o_i\| | → 42 |
    | 动态采样 | 0 < 答对数 < G | → 50 |

    \`\`\`python
    keep = (k > 0) & (k < G)                       # k = 每组答对数
    m = mask * keep[:, None]
    surr = np.minimum(ratio * A, np.clip(ratio, 1 - 0.2, 1 + 0.28) * A)
    loss = -(surr * m).sum() / m.sum()             # token-mean，无 KL
    r_len = np.where(L <= Lm - Lc, 0, np.where(L <= Lm, (Lm - Lc - L) / Lc, -1))
    \`\`\`
    - 奖励 ±1；verl；lr 1e-6 恒定 + 20 步 warm-up；512 题 × 16 回答，每 rollout 16 次更新
    - Qwen2.5-32B base：AIME24 50（R1-Zero-Qwen-32B 47，步数 50%）
  `,
})
