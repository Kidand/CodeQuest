/**
 * PPO 与 RLHF：四个模型与显存、逐 token KL 奖励塑形、GAE、clipped surrogate、value clip。
 * 事实来源：InstructGPT（arXiv 2203.02155 §3.5 与附录 C.4）、PPO（Schulman et al. 2017, 1707.06347）、
 * GAE（Schulman et al. 2015, 1506.02438）、Llama 2（2307.09288 §3.2.3）、Tülu 3（2411.15124 Table 21）、
 * Xu et al. 2024（2404.10719）、The 37 Implementation Details of PPO（ICLR Blog Track 2022，引用 Engstrom et al. 2020）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { RlhfMemoryViz } from '../viz/RlhfMemoryViz'
import { TokenRewardViz } from '../viz/TokenRewardViz'
import { GaeViz } from '../viz/GaeViz'
import { ClipPlaneViz } from '../viz/ClipPlaneViz'

const PY_REF_GAE = py`
  import numpy as np

  def _ref_gae_1d(rewards, values, gamma, lam):
      T = len(rewards)
      adv = np.zeros(T)
      last = 0.0
      for t in reversed(range(T)):
          nv = values[t + 1] if t + 1 < T else 0.0
          delta = rewards[t] + gamma * nv - values[t]
          last = delta + gamma * lam * last
          adv[t] = last
      return adv, adv + np.asarray(values, dtype=float)

  def _ref_rlhf_gae(logp, ref_logp, values, rm_score, mask, beta, gamma, lam):
      m = np.asarray(mask, dtype=float)
      B, T = m.shape
      rewards = np.zeros((B, T)); adv = np.zeros((B, T)); ret = np.zeros((B, T))
      for b in range(B):
          L = int(m[b].sum())
          r = -beta * (logp[b, :L] - ref_logp[b, :L])
          r[L - 1] += rm_score[b]
          a, R = _ref_gae_1d(r, values[b, :L], gamma, lam)
          rewards[b, :L], adv[b, :L], ret[b, :L] = r, a, R
      return rewards, adv, ret

  def _make_rollout(seed, B=4, T=9, garbage=True):
      rng = np.random.default_rng(seed)
      lens = rng.integers(1, T + 1, size=B)
      mask = (np.arange(T)[None, :] < lens[:, None]).astype(np.int64)
      logp = -rng.exponential(1.0, size=(B, T))
      ref_logp = logp + rng.normal(0, 0.3, size=(B, T))
      values = rng.normal(0.5, 0.5, size=(B, T))
      rm = rng.normal(0, 1.5, size=B)
      if garbage:
          pad = mask == 0
          logp[pad] = rng.uniform(-5, 5, size=pad.sum())
          ref_logp[pad] = rng.uniform(-5, 5, size=pad.sum())
          values[pad] = rng.uniform(-5, 5, size=pad.sum())
      return logp, ref_logp, values, rm, mask
`

export default defineLevel({
  learn: [
    {
      id: 'overview',
      title: 'RLHF 全景：四个模型各司其职',
      body: md`
        InstructGPT 的第三阶段把语言模型当作一个 **bandit 环境**里的策略：给一个 prompt，生成完整回复，RM 打分，回合结束。PPO 需要四个模型：

        | 模型 | 来源 | 是否训练 | 作用 |
        |---|---|---|---|
        | Actor $\pi_\theta$ | SFT 模型 | ✅ | rollout 生成回复；按 clipped loss 更新 |
        | Critic $V_\psi$ | 常从 RM 初始化 | ✅ | 估计每个 token 位置的价值 $V(s_t)$，供 GAE 计算优势 |
        | Reference $\pi_{\mathrm{ref}}$ | SFT 模型（冻结） | ❌ | 提供 $\log\pi_{\mathrm{ref}}$，计算逐 token KL 惩罚 |
        | Reward Model $r_\phi$ | 偏好数据训练（冻结） | ❌ | 给完整回复打一个标量分 |

        显存按“模型状态”粗算（混合精度 Adam）：可训练模型每参数 **16 字节**（bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + Adam 一二阶矩 8），冻结模型 bf16 推理每参数 **2 字节**。点击节点看各自角色，拖动规模看显存。
      `,
      viz: RlhfMemoryViz,
      after: md`
        > [!NOTE] 工程上的进一步开销
        > 以上还没算激活和 rollout 时的 KV cache。现代框架（OpenRLHF、verl 等）用 vLLM / SGLang 做 rollout，actor 权重还要同步一份给推理引擎；四个模型常常按角色分配到不同 GPU 组，或在同一组 GPU 上分时复用（colocate + offload）。
        > InstructGPT 的选择是：所有尺寸的策略都共用 **6B 的 RM 和 6B 的 value function**（value 由 RM 初始化），论文称 175B 的 RM 训练不稳定、不适合作 value function。

        与之对比，DPO 只需要 policy（训练）+ reference（推理，甚至可离线预计算），这是它在工业界流行的主要原因之一。
      `,
      checkpoint: {
        id: 'cp-overview',
        type: 'numeric',
        unit: 'GB',
        prompt: md`PPO-RLHF 中 actor、critic、reference、RM 都是 7B。按“可训练模型 16 字节/参数、冻结模型 2 字节/参数”估算，四个模型的模型状态合计多少 GB？（1 GB = $10^9$ 字节）`,
        answer: 252,
        tolerance: 0.001,
        explanation: md`actor 与 critic：$2\times7\times10^9\times16=224$ GB；ref 与 RM：$2\times7\times10^9\times2=28$ GB；合计 252 GB，超过 3 张 80GB 卡——还没算激活、KV cache 与 rollout 引擎的权重副本。`,
      },
    },
    {
      id: 'reward',
      title: '奖励塑形：逐 token KL 惩罚，RM 分数只在末 token',
      body: md`
        把生成过程看作 token 级 MDP：状态 $s_t=(x,y_{<t})$，动作是下一个 token。InstructGPT 的目标（Eq.2）是

        $$
        \mathbb E_{(x,y)\sim\pi_\phi^{\mathrm{RL}}}\Big[r_\theta(x,y)-\beta\log\frac{\pi_\phi^{\mathrm{RL}}(y|x)}{\pi^{\mathrm{SFT}}(y|x)}\Big]+\gamma\,\mathbb E_{x\sim D_{\mathrm{pretrain}}}\big[\log\pi_\phi^{\mathrm{RL}}(x)\big]
        $$

        序列级 KL 等于逐 token log-ratio 之和，于是实现上把奖励摊到每个 token：

        $$
        r_t=-\beta\big(\log\pi_\theta(y_t|s_t)-\log\pi_{\mathrm{ref}}(y_t|s_t)\big)+\mathbb 1[t=T]\cdot r_\phi(x,y)
        $$

        即**每个 token 都被扣 KL，RM 分数只加在最后一个 token（EOS）上**。最后一项 $\gamma$ 是 PPO-ptx 的预训练损失（与 GAE 的折扣 γ 不是一回事），用来缓解在公开 NLP 任务上的性能回退（alignment tax）。
      `,
      viz: TokenRewardViz,
      after: md`
        > [!TIP] 真实超参
        > - InstructGPT：KL 系数 β = 0.02，预训练损失系数 γ = 27.8（附录 C.4）；RM 先加一个偏置，使人类示范的平均分为 0。
        > - Llama 2：7B/13B 用 β = 0.01，34B/70B 用 β = 0.005；安全与有用两个 RM 组合后的分数先用 logit 函数反变换回线性尺度，再 **whiten**，以便与 KL 项平衡。
        > - Tülu 3：没有生成 EOS 的回复直接给 **−10** 的惩罚分。

        **为什么放在奖励里而不是 loss 里？** KL 作为奖励的一部分会进入回报和优势，被 critic 一起估计，能“提前”影响前面的 token；GRPO 则不用 critic，直接把 k3 形式的 KL 加进 loss——两种放法的差异在「推理强化学习」章节详述。
      `,
      checkpoint: {
        id: 'cp-reward',
        type: 'single',
        prompt: md`在 PPO-RLHF 的奖励塑形中，一条长 50 token 的回复，RM 分数加在哪里？`,
        options: ['只加在最后一个 token（EOS）的奖励上', '平均分摊到 50 个 token', '加在第一个 token 上', '不进入奖励，只用于计算优势的基线'],
        answer: 0,
        explanation: md`RM 只对完整回复打分，因此是稀疏的终局奖励；前面 token 的贡献由 critic 的价值估计和 GAE 向前传播。平均分摊是一种启发式，并非标准做法；B 选项会改变问题定义（每个 token 都“看到”终局分数）。`,
      },
    },
    {
      id: 'gae',
      title: 'GAE：用 λ 在偏差与方差之间折中',
      body: md`
        优势 $A_t$ 回答“这个 token 比平均水平好多少”。一步 TD 误差 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ 依赖 critic、方差小但有偏；
        蒙特卡洛 $G_t-V(s_t)$ 无偏但方差大。**GAE**（Schulman et al. 2015）对各步 TD 误差做指数加权：

        $$
        \hat A_t^{\mathrm{GAE}(\gamma,\lambda)}=\sum_{l=0}^{T-t}(\gamma\lambda)^l\,\delta_{t+l}\quad\Longleftrightarrow\quad \hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}
        $$

        - $\lambda=0$：$\hat A_t=\delta_t$，完全信任 critic；
        - $\lambda=1$：$\hat A_t=\sum_l\gamma^l r_{t+l}-V(s_t)$，即回报减基线；
        - $\gamma$ 是折扣。RLHF 的回合很短且只关心整条回复，通常取 **γ = 1**（InstructGPT 写明 GAE 不加折扣；Tülu 3 用 γ = 1、λ = 0.95）。

        critic 的回归目标 $\hat R_t=\hat A_t+V(s_t)$（TD(λ) 回报）。从后往前逐步算一遍：
      `,
      viz: GaeViz,
      after: md`
        > [!WARN] 实现细节
        > - 递推必须**倒序**；最后一个有效 token 之后的价值视为 0（回合结束）。padding 位置的 reward、value 要先置 0，否则垃圾值会通过 $V(s_{t+1})$ 漏进最后一个 token 的优势。
        > - 优势常做**白化**（减均值除标准差，统计只在有效 token 上做）。Xu et al. 2024 把 advantage normalization、大 batch、reference 的 EMA 更新列为 PPO 在 LLM 上效果好的关键因素。
        > - critic 一开始很不准（RM 头与 value 头的含义不同），一些实现会先单独 warmup critic。
      `,
      checkpoint: {
        id: 'cp-gae',
        type: 'numeric',
        prompt: md`两个 token 的回复：$r=[0,\ 1]$，$V=[0.5,\ 0.8]$，$\gamma=1$，$\lambda=0.9$，回合在第 2 个 token 后结束。求 $\hat A_0$。`,
        answer: 0.48,
        tolerance: 0.001,
        explanation: md`$\delta_1=1+0-0.8=0.2$，$\hat A_1=0.2$；$\delta_0=0+0.8-0.5=0.3$，$\hat A_0=0.3+1\times0.9\times0.2=0.48$。`,
      },
    },
    {
      id: 'clip',
      title: 'Clipped surrogate：把每次更新限制在信任域内',
      body: md`
        同一批 rollout 要反复用好几个 mini-batch 更新，旧数据来自 $\pi_{\mathrm{old}}$，用重要性比 $r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{old}}(a_t|s_t)}$ 修正。PPO 的目标（Schulman et al. 2017）：

        $$
        L^{\mathrm{CLIP}}(\theta)=\mathbb E_t\Big[\min\big(r_t\hat A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big]
        $$

        取 min 是“悲观下界”：
        - $\hat A_t>0$：好动作，$r_t$ 超过 $1+\epsilon$ 后目标不再增加，**梯度为 0**；
        - $\hat A_t<0$：坏动作，$r_t$ 低于 $1-\epsilon$ 后**梯度为 0**；
        - 但往“错误方向”走（好动作概率变小、坏动作概率变大）时，min 总是选未裁剪项，梯度照常——裁剪只限制“改过头”，不阻止“纠正”。

        下图每个点是一个 token，落在红色区域的不贡献梯度，占比就是常监控的 **clipfrac**。ε 常取 0.2（InstructGPT、Tülu 3）。
      `,
      viz: ClipPlaneViz,
      after: md`
        > [!INTERVIEW] 高频追问：A < 0 且 r 很大会怎样？
        > min 选未裁剪项 $r_t\hat A_t$，它随 $r_t$ 线性趋向 $-\infty$——单个 token 就能主导梯度。**dual-clip PPO**（Ye et al. 2020，腾讯《王者荣耀》AI）对 $\hat A_t<0$ 再取 $\max(\cdot,\ c\hat A_t)$，原文取 $c=3$（$\epsilon=0.2$）。
        > 另一个常问点：clip 只是近似信任域，ratio 实际可能远超 $1\pm\epsilon$（多个 epoch 后），所以还会监控 approx KL 并早停。

        token 级 ratio、序列级 ratio（GSPO）、上下界解耦（DAPO 的 clip-higher）等改进见「推理强化学习」章节。
      `,
      checkpoint: {
        id: 'cp-clip',
        type: 'single',
        prompt: md`ε = 0.2。某 token 的 $\hat A=-1$，$r=1.5$。它的梯度情况是：`,
        options: [
          '有梯度：min 选中未裁剪项 r·A = −1.5，会推动降低该 token 的概率',
          '梯度为 0：r 已超出 [0.8, 1.2]',
          '梯度为 0：A < 0 时 PPO 永远不更新',
          '有梯度，但方向是提高该 token 的概率',
        ],
        answer: 0,
        explanation: md`$r\hat A=-1.5$，$\mathrm{clip}(r)\hat A=1.2\times(-1)=-1.2$，min 取 $-1.5$（未裁剪项），梯度为 $\hat A=-1$，梯度上升会降低 $r$。只有 $\hat A<0$ 且 $r<1-\epsilon$ 时才裁剪。这正是 dual-clip 要处理的“坏动作被大幅提高概率”的情形。`,
      },
    },
    {
      id: 'value',
      title: 'Value loss、value clip 与完整的 PPO 迭代',
      body: md`
        critic 用 MSE 回归 $\hat R_t$。PPO 的常见实现（OpenAI baselines PPO2、TRL、OpenRLHF）还对 value 做裁剪：

        $$
        V^{\mathrm{clip}}_t=V_{\mathrm{old}}(s_t)+\mathrm{clip}\big(V_\psi(s_t)-V_{\mathrm{old}}(s_t),-\epsilon_v,\epsilon_v\big),\quad
        L^V=\tfrac12\,\mathbb E_t\Big[\max\big((V_\psi-\hat R_t)^2,\ (V^{\mathrm{clip}}_t-\hat R_t)^2\big)\Big]
        $$

        注意这里取的是 **max**（悲观上界），与 policy 的 min 对应。但它的价值有争议：Engstrom et al. 2020 没有发现 value clipping 带来提升的证据，Andrychowicz et al. 2021 认为它甚至可能有害。

        **一次 PPO 迭代**：
        1. rollout：actor 对一批 prompt 采样回复，记录 $\log\pi_{\mathrm{old}}$；
        2. 打分：ref 给 $\log\pi_{\mathrm{ref}}$，RM 给分数，critic 给 $V_{\mathrm{old}}$；
        3. 奖励塑形 → GAE 得到 $\hat A_t,\hat R_t$（优势白化）；
        4. 把这批数据切成若干 mini-batch，训练 $K$ 个 epoch：$L=-L^{\mathrm{CLIP}}+c_1L^V$（LLM 场景一般不加熵奖励）。

        InstructGPT：batch 512、mini-batch 64、只训 1 个 inner epoch；Tülu 3：value 系数 $c_1=0.1$、梯度裁剪 1.0。
      `,
      after: md`
        > [!NOTE] 为什么后来大家想绕开 PPO
        > critic 与 actor 同规模，显存翻倍；critic 在稀疏终局奖励下难学，优势估计噪声大；超参多（β、γ、λ、ε、ε_v、c₁、K、mini-batch……）。
        > 这催生了两条路线：**去掉 RL**（DPO 系，下一关）与**去掉 critic**（RLOO、GRPO 等用多次采样的组内基线，见「推理强化学习」）。
      `,
      checkpoint: {
        id: 'cp-value',
        type: 'judge',
        prompt: md`PPO 的 value clipping 已被大量实验证明能稳定提升效果，是必须保留的技巧。`,
        answer: false,
        explanation: md`《The 37 Implementation Details of PPO》引用 Engstrom et al. 2020：没有发现 value loss clipping 有助于性能的证据；Andrychowicz et al. 2021 甚至认为它可能有害。它被保留在许多实现里主要是为了复现原版行为。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-trainable',
      type: 'multi',
      difficulty: 1,
      prompt: md`PPO-RLHF 的四个模型中，哪些在 RL 阶段会被更新参数？`,
      options: ['Actor（策略）', 'Critic（价值模型）', 'Reference model', 'Reward model'],
      answer: [0, 1],
      explanation: md`actor 与 critic 训练；reference 与 RM 冻结，只做前向。若 RM 也在训练中被更新，策略优化的目标会漂移（在线更新 RM 是另一类“iterative RLHF”方法，不是标准 PPO 阶段）。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把 PPO-RLHF 中的量与产生它的模型配对：`,
      pairs: [
        [md`$\log\pi_{\mathrm{old}}(y_t|s_t)$`, 'Actor（rollout 时）'],
        [md`$\log\pi_{\mathrm{ref}}(y_t|s_t)$`, 'Reference model'],
        [md`$r_\phi(x,y)$`, 'Reward model'],
        [md`$V(s_t)$`, 'Critic'],
      ],
      explanation: md`四个量分别用于：重要性比、KL 惩罚、终局奖励、GAE 的基线。`,
    },
    {
      id: 'q-gae-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`GAE 中，λ = 0 时 $\hat A_t$ 退化为 ____(1)____；γ = λ = 1 时 $\hat A_t$ 等于从 t 开始的回报之和减去 ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['δ_t', 'δt', 'delta_t', 'TD误差', '一步TD误差', 'r_t+γV(s_{t+1})-V(s_t)', 'delta'] },
        { label: '(2)', answers: ['V(s_t)', 'V(st)', 'V_t', 'Vt', '价值', 'V'] },
      ],
      explanation: md`$\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$。λ = 0 只剩 $\delta_t$；γ = λ = 1 时 TD 误差望远镜相消，得 $\sum_l r_{t+l}-V(s_t)$（终止后价值为 0）。`,
    },
    {
      id: 'q-pick-clip',
      type: 'pick',
      difficulty: 2,
      prompt: md`ε = 0.2，使用标准 PPO-clip（无 dual-clip）。点出**梯度为 0** 的全部 token：`,
      items: ['r=1.3, A=+1', 'r=1.3, A=−1', 'r=0.7, A=+1', 'r=0.7, A=−1', 'r=1.1, A=+1', 'r=0.9, A=−1', 'r=1.25, A=+0.5', 'r=0.75, A=−0.5'],
      cols: 2,
      answer: [0, 3, 6, 7],
      explanation: md`只有“改过头”的情形被裁剪：$A>0$ 且 $r>1.2$（第 1、7 个），$A<0$ 且 $r<0.8$（第 4、8 个）。$r=1.3, A=-1$ 是坏动作概率变大，min 选未裁剪项 $-1.3$，有梯度；$r=0.7, A=+1$ 同理。`,
    },
    {
      id: 'q-kl',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 PPO-RLHF 中的 KL 惩罚，下列说法正确的有：`,
      options: [
        '它约束策略不要偏离 SFT/reference 太远，缓解 reward hacking',
        '常见实现把 −β·(log π_θ − log π_ref) 作为每个 token 的奖励',
        'k1 估计 log(π_θ/π_ref) 在单个 token 上可能为负',
        'β 越大，策略越容易钻 RM 的漏洞',
        'InstructGPT 的 KL 系数 β = 0.02',
      ],
      answer: [0, 1, 2, 4],
      explanation: md`D 反了：β 越大约束越强，越不容易 reward hacking，但奖励提升也越慢。C：k1 是无偏的序列级 KL 估计，但逐 token 可以为负；k3 逐项非负。E 见 InstructGPT 附录 C.4。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把一次 PPO-RLHF 迭代的步骤按顺序排列：`,
      items: [
        'actor 对一批 prompt 采样回复，记录 log π_old',
        'reference、RM、critic 前向，得到 log π_ref、分数与 V_old',
        '计算逐 token 奖励（KL + 末 token RM 分数）',
        '倒序递推 GAE 得到优势与回报，并做优势白化',
        '切 mini-batch，用 clipped policy loss + value loss 训练若干 epoch',
      ],
      explanation: md`rollout → 打分 → 奖励塑形 → GAE → 多轮 mini-batch 更新。注意 GAE 需要完整的奖励序列，必须在所有前向结束后才能计算。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 GAE 实现（单条回复，回合在最后一个 token 后结束）有 **2 处** bug，点出来：`,
      code: py`
        def compute_gae(rewards, values, gamma=1.0, lam=0.95):
            T = len(rewards)
            adv = [0.0] * T
            last_gae = 0.0
            for t in range(T):
                next_v = values[t + 1] if t + 1 < T else 0.0
                delta = rewards[t] + gamma * next_v - values[t]
                last_gae = delta + gamma * last_gae
                adv[t] = last_gae
            returns = [a + v for a, v in zip(adv, values)]
            return adv, returns
      `,
      bugLines: [5, 8],
      explanation: md`
        第 5 行：必须倒序 \`for t in reversed(range(T))\`，因为 $\hat A_t$ 依赖 $\hat A_{t+1}$。
        第 8 行：少了 λ，应为 \`delta + gamma * lam * last_gae\`；现在等价于 λ = 1。
        其余正确：终止后价值取 0；returns = adv + values。
      `,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 token 级 PPO clipped policy loss（\`mask\` 为有效 token，最小化 loss）。有 2 行干扰项。`,
      lines: [
        'def ppo_policy_loss(logp, old_logp, adv, mask, eps=0.2):',
        '    ratio = np.exp(logp - old_logp)',
        '    pg1 = -adv * ratio',
        '    pg2 = -adv * np.clip(ratio, 1 - eps, 1 + eps)',
        '    pg = np.maximum(pg1, pg2)',
        '    return (pg * mask).sum() / mask.sum()',
      ],
      distractors: ['    pg = np.minimum(pg1, pg2)', '    ratio = logp / old_logp'],
      explanation: md`目标是 $\min(rA,\mathrm{clip}(r)A)$，取负号变成 loss 后 min 变 max。ratio 是概率之比，用 logp 之差取 exp，不是 logp 相除。`,
    },
    {
      id: 'q-valueclip',
      type: 'single',
      difficulty: 2,
      prompt: md`PPO 的 value loss 写作 $\tfrac12\max\big((V-\hat R)^2,(V^{\mathrm{clip}}-\hat R)^2\big)$。为什么这里是 max 而 policy 部分是 min？`,
      options: [
        '两者都是取“悲观”的一边：policy 要最大化目标所以取下界 min，value 要最小化误差所以取上界 max',
        'max 让 value loss 更小，训练更快',
        '这是笔误，两者都应该取 min',
        'max 等价于对 value 做梯度裁剪',
      ],
      answer: 0,
      explanation: md`取 max 意味着：当裁剪后的预测误差更大时，用它计算损失，从而阻止 value 一步走太远；与 policy 的 min 一样是保守估计。B 恰恰相反；D 不对——裁剪的是预测值，不是梯度范数。`,
    },
    {
      id: 'q-dual',
      type: 'single',
      difficulty: 3,
      prompt: md`标准 PPO-clip 中，若某 token 的 $\hat A=-2$ 且 ratio 因多轮更新涨到 $r=8$，单个 token 的目标值是多少？dual-clip（$c=3$）下又是多少？`,
      options: ['标准：−16；dual-clip：−6', '标准：−2.4；dual-clip：−2.4', '标准：−16；dual-clip：−16', '标准：0；dual-clip：0'],
      answer: 0,
      explanation: md`标准：$\min(8\times(-2),\ 1.2\times(-2))=\min(-16,-2.4)=-16$，且梯度不为 0，单个 token 可以主导更新。dual-clip：$\max(-16,\ 3\times(-2))=-6$，此时取常数 $c\hat A$，梯度为 0。`,
    },
    {
      id: 'q-gamma',
      type: 'judge',
      difficulty: 2,
      prompt: md`InstructGPT 计算 GAE 时使用 γ = 1，即不对未来奖励做折扣。`,
      answer: true,
      explanation: md`附录 C.4 原文：“No discount is applied when estimating the generalized advantage”。RLHF 的回合只有一条回复、奖励在末尾，折扣会让前面 token 的回报系统性变小，没有必要；Tülu 3 也用 γ = 1、λ = 0.95。`,
    },
    {
      id: 'q-mem2',
      type: 'numeric',
      difficulty: 3,
      unit: 'GB',
      prompt: md`InstructGPT 式配置：actor 与 reference 为 13B，critic 与 RM 都是 6B。按可训练 16 字节/参数、冻结 2 字节/参数估算模型状态总量（GB，1 GB = $10^9$ 字节）。`,
      answer: 342,
      tolerance: 0.001,
      explanation: md`actor $13\times16=208$；critic $6\times16=96$；ref $13\times2=26$；RM $6\times2=12$；合计 342 GB。critic 用小模型能显著省显存——InstructGPT 正是对所有策略尺寸都用 6B 的 RM/value。`,
    },
  ],

  guided: [
    {
      id: 'g-gae',
      title: '一步步写出 GAE',
      intro: md`
        实现单条回复的 \`compute_gae(rewards, values, gamma, lam)\`，返回 \`(advantages, returns)\`。
        约定：\`rewards[t]\`、\`values[t]\` 长度均为 $T$，回合在最后一个 token 之后结束（终止状态价值为 0）。
      `,
      template: py`
        import numpy as np

        def compute_gae(rewards, values, gamma=1.0, lam=0.95):
            T = len(rewards)
            adv = np.zeros(T)
            last_gae = 0.0
            for t in {{1}}:
                next_v = {{2}}
                delta = {{3}}
                last_gae = {{4}}
                adv[t] = last_gae
            returns = {{5}}
            return adv, returns
      `,
      steps: [
        {
          blank: 1,
          instruction: md`倒序遍历时间步。`,
          answer: 'reversed(range(T))',
          accept: ['range(T - 1, -1, -1)', 'range(T-1, -1, -1)'],
          explain: md`$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$ 依赖后一步，必须从后往前。`,
        },
        {
          blank: 2,
          instruction: md`下一个状态的价值；最后一步之后回合结束。`,
          answer: 'values[t + 1] if t + 1 < T else 0.0',
          accept: ['values[t+1] if t < T - 1 else 0.0', 'values[t + 1] if t < T - 1 else 0'],
          explain: md`终止状态价值为 0。如果回复是被 max_len 截断而不是正常结束，严格来说应 bootstrap 用 $V(s_T)$，但 RLHF 实现通常也按终止处理（并可能给未输出 EOS 的回复惩罚分）。`,
        },
        {
          blank: 3,
          instruction: md`TD 误差 $\delta_t$。`,
          answer: 'rewards[t] + gamma * next_v - values[t]',
          explain: md`$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$：critic 认为“走完这一步后比走之前好多少”。`,
        },
        {
          blank: 4,
          instruction: md`GAE 递推。`,
          answer: 'delta + gamma * lam * last_gae',
          accept: ['delta + lam * gamma * last_gae'],
          explain: md`把后面的优势按 $\gamma\lambda$ 衰减后加进来；λ 控制“看多远”。`,
        },
        {
          blank: 5,
          instruction: md`critic 的回归目标。`,
          answer: 'adv + np.asarray(values)',
          accept: ['adv + values', 'adv + np.array(values)'],
          explain: md`$\hat R_t=\hat A_t+V(s_t)$ 即 TD(λ) 回报。注意它用的是**旧**的 values，计算完就固定，不随 critic 更新而变。`,
        },
      ],
      setup: PY_REF_GAE,
      tests: [
        {
          name: '手算示例',
          code: py`
            adv, ret = compute_gae([0.0, 1.0], [0.5, 0.8], gamma=1.0, lam=0.9)
            assert np.allclose(adv, [0.48, 0.2]), f"期望 A = [0.48, 0.2]，得到 {adv}"
            assert np.allclose(ret, [0.98, 1.0]), f"期望 R = [0.98, 1.0]，得到 {ret}"
          `,
        },
        {
          name: 'λ = 0 时等于 TD 误差；γ = λ = 1 时等于回报减基线',
          code: py`
            r = [0.1, -0.2, 0.0, 1.5]; v = [0.3, 0.4, 0.9, 1.2]
            a0, _ = compute_gae(r, v, gamma=1.0, lam=0.0)
            d = [r[t] + (v[t + 1] if t + 1 < 4 else 0.0) - v[t] for t in range(4)]
            assert np.allclose(a0, d), f"λ=0：期望 {np.round(d, 4)}，得到 {np.round(a0, 4)}"
            a1, _ = compute_gae(r, v, gamma=1.0, lam=1.0)
            g = np.cumsum(r[::-1])[::-1] - np.array(v)
            assert np.allclose(a1, g), f"λ=1：期望 {np.round(g, 4)}，得到 {np.round(a1, 4)}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(0)
            for _ in range(30):
                T = int(rng.integers(1, 12))
                r, v = rng.normal(size=T), rng.normal(size=T)
                g, l = float(rng.uniform(0.9, 1.0)), float(rng.uniform(0, 1))
                a, R = compute_gae(r, v, g, l)
                ea, eR = _ref_gae_1d(r, v, g, l)
                assert np.allclose(a, ea) and np.allclose(R, eR), f"T={T}, γ={g:.3f}, λ={l:.3f}"
          `,
        },
      ],
      outro: md`
        批量版只需把标量换成形状 \`(B,)\` 的向量、按列倒序循环；PyTorch 里常写成 \`torch.stack(advs_reversed[::-1], dim=1)\`。
        也可以用矩阵形式 $\hat A=\delta\,M$（$M_{ij}=(\gamma\lambda)^{j-i}$，$j\ge i$）一次算完，但 $O(T^2)$，长回复下不如循环划算。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-rlhf-gae',
      title: 'RLHF 奖励塑形 + 批量 GAE（含 padding）',
      difficulty: 'medium',
      source: '大模型岗手撕 · PPO',
      timeLimitMin: 25,
      description: md`
        给定一批 rollout（右侧 padding），计算逐 token 奖励、优势与回报：

        \`\`\`python
        def rlhf_gae(logp, ref_logp, values, rm_score, mask,
                     beta=0.05, gamma=1.0, lam=0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
        \`\`\`

        - \`logp\`、\`ref_logp\`、\`values\`：\`(B, T)\`，分别为 actor（rollout 时）与 reference 对**采样 token** 的 log-prob、critic 的 $V(s_t)$；
        - \`rm_score\`：\`(B,)\`，RM 对每条完整回复的分数；
        - \`mask\`：\`(B, T)\` 0/1，右侧 padding，每行至少 1 个有效 token；**padding 位置的 logp / values 是任意垃圾值**；
        - 奖励：$r_t=-\beta(\log\pi-\log\pi_{\mathrm{ref}})$，并在每行**最后一个有效 token** 上加 \`rm_score\`；
        - GAE：倒序递推，最后一个有效 token 之后价值为 0；
        - 返回 \`(rewards, advantages, returns)\`，均为 \`(B, T)\`，**padding 位置必须为 0**；returns = advantages + values（有效位置）。

        要求：沿时间维循环可以，但不要对 batch 维写 Python 循环。
      `,
      starter: py`
        import numpy as np

        def rlhf_gae(logp, ref_logp, values, rm_score, mask, beta=0.05, gamma=1.0, lam=0.95):
            # 1) 逐 token KL 奖励 + 末 token 加 RM 分数
            # 2) 把 padding 位置的 values / rewards 清零
            # 3) 沿 t 倒序做 GAE
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def rlhf_gae(logp, ref_logp, values, rm_score, mask, beta=0.05, gamma=1.0, lam=0.95):
            m = np.asarray(mask, dtype=np.float64)
            B, T = m.shape
            rewards = np.where(m > 0, -beta * (logp - ref_logp), 0.0)     # 逐 token KL 惩罚（k1）
            last = m.sum(axis=1).astype(int) - 1                          # 每行最后一个有效 token
            rewards[np.arange(B), last] += rm_score                       # RM 分数只给末 token
            v = np.where(m > 0, values, 0.0)                              # padding 的价值视为 0（终止）

            adv = np.zeros((B, T))
            last_gae = np.zeros(B)
            for t in reversed(range(T)):
                next_v = v[:, t + 1] if t + 1 < T else np.zeros(B)
                delta = rewards[:, t] + gamma * next_v - v[:, t]
                last_gae = delta + gamma * lam * last_gae
                adv[:, t] = last_gae
            adv *= m                                                      # 右侧 padding 处本来就是 0，这里保险
            returns = (adv + v) * m
            return rewards, adv, returns
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            @torch.no_grad()
            def rlhf_gae(logp, ref_logp, values, rm_score, mask, beta=0.05, gamma=1.0, lam=0.95):
                mask = mask.float()
                B, T = mask.shape
                rewards = -beta * (logp - ref_logp) * mask
                last = mask.sum(1).long() - 1
                rewards[torch.arange(B), last] += rm_score
                values = values * mask
                advs, last_gae = [], torch.zeros(B, device=values.device)
                for t in reversed(range(T)):
                    next_v = values[:, t + 1] if t + 1 < T else torch.zeros_like(last_gae)
                    delta = rewards[:, t] + gamma * next_v - values[:, t]
                    last_gae = delta + gamma * lam * last_gae
                    advs.append(last_gae)
                adv = torch.stack(advs[::-1], dim=1) * mask
                returns = (adv + values) * mask
                return rewards, adv, returns
          `,
        },
      ],
      explanation: md`
        **关键点**：
        1. **先清零 padding 的 values**：否则最后一个有效 token 的 $\delta$ 会读到 \`values[:, L]\` 的垃圾值作为 $V(s_{t+1})$；
        2. **RM 分数放在 \`mask.sum(1) - 1\` 位置**（右 padding）；左 padding 时位置是 \`T - 1\`，面试时要先问清楚；
        3. 倒序递推中 padding 区段 reward、value 都是 0，优势自然为 0，递推跨过它不会污染有效区；
        4. 白化（advantage whitening）通常在这之后、只用有效 token 统计；本题不做。

        **复杂度**：$O(B\cdot T)$，时间维必须串行（可用矩阵形式 $O(BT^2)$ 并行）。
      `,
      complexity: { time: 'O(B·T)', space: 'O(B·T)' },
      hints: [
        md`\`rewards = np.where(mask > 0, -beta * (logp - ref_logp), 0.0)\`，再 \`rewards[np.arange(B), lens - 1] += rm_score\`。`,
        md`\`v = np.where(mask > 0, values, 0.0)\`：终止之后价值为 0。`,
        md`时间维倒序循环，变量 \`last_gae\` 是形状 \`(B,)\` 的向量。`,
      ],
      setup: PY_REF_GAE,
      tests: [
        {
          name: '单条回复手算',
          code: py`
            logp = np.array([[-1.0, -0.5]]); ref = np.array([[-1.2, -0.5]])
            vals = np.array([[0.5, 0.8]]); mask = np.array([[1, 1]])
            r, a, R = rlhf_gae(logp, ref, vals, np.array([1.0]), mask, beta=0.1, gamma=1.0, lam=0.9)
            assert np.allclose(r, [[-0.02, 1.0]]), f"rewards 期望 [[-0.02, 1.0]]，得到 {r}"
            # δ1 = 1 - 0.8 = 0.2；δ0 = -0.02 + 0.8 - 0.5 = 0.28；A0 = 0.28 + 0.9*0.2 = 0.46
            assert np.allclose(a, [[0.46, 0.2]]), f"advantages 期望 [[0.46, 0.2]]，得到 {a}"
            assert np.allclose(R, [[0.96, 1.0]]), f"returns 期望 [[0.96, 1.0]]，得到 {R}"
          `,
        },
        {
          name: '形状与 padding 位置为 0',
          code: py`
            logp, ref, vals, rm, mask = _make_rollout(1)
            out = rlhf_gae(logp, ref, vals, rm, mask)
            for name, x in zip(("rewards", "advantages", "returns"), out):
                assert x.shape == mask.shape, f"{name} 形状应为 {mask.shape}，得到 {x.shape}"
                assert np.allclose(x[mask == 0], 0), f"{name} 在 padding 位置应为 0"
          `,
        },
        {
          name: '与逐条参考实现一致',
          code: py`
            logp, ref, vals, rm, mask = _make_rollout(2)
            got = rlhf_gae(logp, ref, vals, rm, mask, 0.05, 1.0, 0.95)
            exp = _ref_rlhf_gae(logp, ref, vals, rm, mask, 0.05, 1.0, 0.95)
            for name, g, e in zip(("rewards", "advantages", "returns"), got, exp):
                assert np.allclose(g, e, atol=1e-9), f"{name} 最大误差 {np.abs(g - e).max():.3e}"
          `,
        },
        {
          name: '随机对拍（不同 β / γ / λ、长度 1 的回复）',
          hidden: true,
          code: py`
            for s, (b, g, l) in enumerate([(0.0, 1.0, 1.0), (0.1, 0.99, 0.0), (0.02, 1.0, 0.95), (0.3, 0.9, 0.5)]):
                logp, ref, vals, rm, mask = _make_rollout(10 + s, B=6, T=11)
                mask[0] = 0; mask[0, 0] = 1                     # 只有 1 个 token 的回复
                got = rlhf_gae(logp, ref, vals, rm, mask, b, g, l)
                exp = _ref_rlhf_gae(logp, ref, vals, rm, mask, b, g, l)
                for name, x, e in zip(("rewards", "advantages", "returns"), got, exp):
                    assert np.allclose(x, e, atol=1e-9), f"β={b}, γ={g}, λ={l}：{name} 最大误差 {np.abs(x - e).max():.3e}"
          `,
        },
        {
          name: 'padding 处的垃圾 value 不能漏进最后一个有效 token',
          hidden: true,
          code: py`
            logp, ref, vals, rm, mask = _make_rollout(30, B=3, T=6)
            vals2 = vals.copy(); vals2[mask == 0] += 100.0
            a1 = rlhf_gae(logp, ref, vals, rm, mask)[1]
            a2 = rlhf_gae(logp, ref, vals2, rm, mask)[1]
            assert np.allclose(a1, a2), "改变 padding 位置的 values 后优势变了——终止后的价值没有清零"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-left-pad',
          q: md`如果 rollout 是左侧 padding（生成时常见），哪些地方要改？`,
          a: md`
            响应部分一般仍从 prompt 之后开始；若把 prompt + response 拼在一起左 padding，response 的最后一个有效 token 固定在 \`T-1\`（遇到 EOS 后的位置要再 mask）。通常做法是只截取 response 段并转成右 padding 的 \`response_mask\`，再套用本题逻辑；RM 分数加在 \`response_mask\` 的最后一个 1 处（需要处理“没有 EOS 被截断”的情况，Tülu 3 直接给 −10 分）。
          `,
        },
        {
          id: 'f-whiten',
          q: md`优势白化具体怎么做？为什么 returns 用白化前的优势？`,
          a: md`
            \`adv = (adv - mean) / (std + 1e-8)\`，mean/std 只在有效 token 上统计（masked_whiten）。returns 必须用白化**前**的优势加 values，否则 critic 的回归目标被改变了尺度和偏移，学不到真实价值。
          `,
        },
      ],
    },
    {
      id: 'c-ppo-loss',
      title: '手写 PPO loss：clipped policy + clipped value + 监控指标',
      difficulty: 'medium',
      source: '大模型岗手撕 · PPO',
      timeLimitMin: 25,
      description: md`
        \`\`\`python
        def ppo_loss(logp, old_logp, advantages, values, old_values, returns, mask,
                     clip_eps=0.2, clip_vf=0.2, vf_coef=0.1) -> dict: ...
        \`\`\`

        所有数组形状 \`(B, T)\`；\`mask\` 为 0/1，padding 位置的其它数组是任意有限值。所有“平均”都是对**全部有效 token** 的平均（token 级）。返回 dict：

        | key | 定义 |
        |---|---|
        | \`pg_loss\` | ratio = exp(logp − old_logp)；平均 $\max(-A\cdot r,\ -A\cdot\mathrm{clip}(r,1-\epsilon,1+\epsilon))$ |
        | \`vf_loss\` | $V^{\mathrm{clip}}=V_{\mathrm{old}}+\mathrm{clip}(V-V_{\mathrm{old}},-\epsilon_v,\epsilon_v)$；$\tfrac12\cdot$平均 $\max\big((V-R)^2,(V^{\mathrm{clip}}-R)^2\big)$ |
        | \`loss\` | \`pg_loss + vf_coef * vf_loss\` |
        | \`clipfrac\` | 有效 token 中 $-A\cdot\mathrm{clip}(r) > -A\cdot r$（裁剪项被选中）的比例 |
        | \`approx_kl\` | 平均 $(r-1)-\log r$（k3 估计，恒 ≥ 0） |

        全部返回 Python float。
      `,
      starter: py`
        import numpy as np

        def ppo_loss(logp, old_logp, advantages, values, old_values, returns, mask,
                     clip_eps=0.2, clip_vf=0.2, vf_coef=0.1):
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def ppo_loss(logp, old_logp, advantages, values, old_values, returns, mask,
                     clip_eps=0.2, clip_vf=0.2, vf_coef=0.1):
            m = np.asarray(mask, dtype=np.float64)
            n = m.sum()
            mmean = lambda x: float(np.where(m > 0, x, 0.0).sum() / n)   # 用 where 而不是乘法，防 inf*0

            log_ratio = np.where(m > 0, logp - old_logp, 0.0)
            ratio = np.exp(log_ratio)
            pg1 = -advantages * ratio
            pg2 = -advantages * np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            pg_loss = mmean(np.maximum(pg1, pg2))

            v_clip = old_values + np.clip(values - old_values, -clip_vf, clip_vf)
            vf1 = (values - returns) ** 2
            vf2 = (v_clip - returns) ** 2
            vf_loss = 0.5 * mmean(np.maximum(vf1, vf2))

            return {
                "pg_loss": pg_loss,
                "vf_loss": vf_loss,
                "loss": pg_loss + vf_coef * vf_loss,
                "clipfrac": mmean((pg2 > pg1).astype(np.float64)),
                "approx_kl": mmean((ratio - 1.0) - log_ratio),
            }
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def masked_mean(x, mask):
                return (x * mask).sum() / mask.sum()

            def ppo_loss(logp, old_logp, adv, values, old_values, returns, mask,
                         clip_eps=0.2, clip_vf=0.2, vf_coef=0.1):
                mask = mask.float()
                log_ratio = (logp - old_logp) * mask
                ratio = log_ratio.exp()
                pg1 = -adv * ratio
                pg2 = -adv * ratio.clamp(1 - clip_eps, 1 + clip_eps)
                pg_loss = masked_mean(torch.max(pg1, pg2), mask)
                v_clip = old_values + (values - old_values).clamp(-clip_vf, clip_vf)
                vf_loss = 0.5 * masked_mean(torch.max((values - returns) ** 2, (v_clip - returns) ** 2), mask)
                with torch.no_grad():
                    clipfrac = masked_mean((pg2 > pg1).float(), mask)
                    approx_kl = masked_mean((ratio - 1) - log_ratio, mask)
                return pg_loss + vf_coef * vf_loss, dict(pg_loss=pg_loss, vf_loss=vf_loss,
                                                        clipfrac=clipfrac, approx_kl=approx_kl)
          `,
        },
      ],
      explanation: md`
        **要点**：
        - policy：目标 $\min(rA,\mathrm{clip}(r)A)$ 取负号，变成 $\max(-rA,-\mathrm{clip}(r)A)$；
        - value：裁剪的是**预测值相对旧值的变化量**，再与未裁剪的平方误差取 max；
        - clipfrac 按“裁剪项被选中”统计（TRL 的定义），而不是 \`|r - 1| > ε\`——后者会把 $A>0, r<1-\epsilon$ 这种“未被裁剪”的 token 也算进去；
        - approx_kl 用 k3：$(r-1)-\log r\ge0$，比 $-\log r$ 方差小，且不会为负；
        - masked mean 用 \`np.where\` 而不是乘 mask，避免 padding 处 ratio 溢出成 inf 后 $\infty\times0=\mathrm{nan}$。

        **复杂度**：$O(B\cdot T)$。
      `,
      complexity: { time: 'O(B·T)', space: 'O(B·T)' },
      hints: [
        md`写一个 \`mmean(x) = where(mask, x, 0).sum() / mask.sum()\`。`,
        md`\`pg_loss = mmean(np.maximum(-A * r, -A * np.clip(r, 1 - eps, 1 + eps)))\`。`,
        md`\`v_clip = old_values + np.clip(values - old_values, -clip_vf, clip_vf)\`，value loss 前面有 0.5。`,
      ],
      setup: py`
        import numpy as np

        def _ref_ppo(logp, old_logp, A, V, Vold, R, mask, ce=0.2, cv=0.2, vc=0.1):
            out = {k: [] for k in ("pg", "vf", "cf", "kl")}
            for b in range(mask.shape[0]):
                for t in range(mask.shape[1]):
                    if mask[b, t] == 0:
                        continue
                    lr = logp[b, t] - old_logp[b, t]; r = np.exp(lr)
                    p1 = -A[b, t] * r; p2 = -A[b, t] * min(max(r, 1 - ce), 1 + ce)
                    vcl = Vold[b, t] + min(max(V[b, t] - Vold[b, t], -cv), cv)
                    out["pg"].append(max(p1, p2))
                    out["vf"].append(max((V[b, t] - R[b, t]) ** 2, (vcl - R[b, t]) ** 2))
                    out["cf"].append(1.0 if p2 > p1 else 0.0)
                    out["kl"].append((r - 1) - lr)
            pg, vf = np.mean(out["pg"]), 0.5 * np.mean(out["vf"])
            return {"pg_loss": pg, "vf_loss": vf, "loss": pg + vc * vf, "clipfrac": np.mean(out["cf"]), "approx_kl": np.mean(out["kl"])}

        def _make_ppo(seed, B=4, T=8):
            rng = np.random.default_rng(seed)
            lens = rng.integers(1, T + 1, size=B)
            mask = (np.arange(T)[None, :] < lens[:, None]).astype(np.int64)
            old = -rng.exponential(1.0, size=(B, T))
            new = old + rng.normal(0, 0.3, size=(B, T))
            A = rng.normal(0, 1, size=(B, T))
            Vold = rng.normal(0, 1, size=(B, T)); V = Vold + rng.normal(0, 0.4, size=(B, T)); R = Vold + rng.normal(0, 1, size=(B, T))
            pad = mask == 0
            for x in (old, new, A, V, Vold, R):
                x[pad] = rng.uniform(-5, 5, size=pad.sum())
            return new, old, A, V, Vold, R, mask

        def _cmp(got, exp, tag=""):
            assert isinstance(got, dict), "应返回 dict"
            for k in ("pg_loss", "vf_loss", "loss", "clipfrac", "approx_kl"):
                assert k in got, f"缺少键 {k}"
                assert abs(float(got[k]) - exp[k]) < 1e-9, f"{tag}{k}：期望 {exp[k]:.6f}，得到 {float(got[k]):.6f}"
      `,
      tests: [
        {
          name: 'ratio = 1：pg_loss = −mean(A)，clipfrac = approx_kl = 0',
          code: py`
            lp = np.log(np.full((2, 3), 0.5)); A = np.array([[1.0, -2.0, 0.5], [0.3, 9.0, 9.0]])
            mask = np.array([[1, 1, 1], [1, 0, 0]])
            V = np.zeros((2, 3)); R = np.ones((2, 3))
            out = ppo_loss(lp, lp.copy(), A, V, V.copy(), R, mask)
            assert abs(out["pg_loss"] - (-(1 - 2 + 0.5 + 0.3) / 4)) < 1e-12, f"pg_loss 得到 {out['pg_loss']}"
            assert out["clipfrac"] == 0 and abs(out["approx_kl"]) < 1e-12, f"clipfrac / approx_kl 应为 0，得到 {out['clipfrac']}, {out['approx_kl']}"
            assert abs(out["vf_loss"] - 0.5) < 1e-12, f"V=0、R=1 时 vf_loss 应为 0.5，得到 {out['vf_loss']}"
          `,
        },
        {
          name: '裁剪与 value clip 的手算例子',
          code: py`
            lp = np.log(np.array([[1.5, 0.5]])); old = np.zeros((1, 2))      # ratio = [1.5, 0.5]
            A = np.array([[1.0, 1.0]]); mask = np.ones((1, 2))
            Vold = np.array([[0.0, 0.0]]); V = np.array([[1.0, 0.1]]); R = np.array([[1.0, 0.0]])
            out = ppo_loss(lp, old, A, V, Vold, R, mask)
            # token0：A>0, r=1.5 → 取 -1.2（裁剪项被选中）；token1：A>0, r=0.5 → 取 -0.5
            assert abs(out["pg_loss"] - (-(1.2 + 0.5) / 2)) < 1e-12, f"pg_loss 期望 -0.85，得到 {out['pg_loss']}"
            assert abs(out["clipfrac"] - 0.5) < 1e-12, f"clipfrac 期望 0.5（只有 token0 被裁剪），得到 {out['clipfrac']}"
            # token0：V=1 但 Vclip=0.2，max(0, 0.64)=0.64；token1：max(0.01, 0.01)=0.01
            assert abs(out["vf_loss"] - 0.5 * (0.64 + 0.01) / 2) < 1e-12, f"vf_loss 期望 0.1625，得到 {out['vf_loss']}"
          `,
        },
        {
          name: '随机对拍（含 padding 垃圾值）',
          code: py`
            args = _make_ppo(0)
            _cmp(ppo_loss(*args), _ref_ppo(*args))
          `,
        },
        {
          name: '随机对拍（不同 ε、vf_coef）',
          hidden: true,
          code: py`
            for s, (ce, cv, vc) in enumerate([(0.1, 0.5, 1.0), (0.3, 0.05, 0.5), (0.2, 10.0, 0.1)]):
                args = _make_ppo(10 + s, B=5, T=9)
                _cmp(ppo_loss(*args, clip_eps=ce, clip_vf=cv, vf_coef=vc), _ref_ppo(*args, ce, cv, vc), f"seed={10+s}: ")
          `,
        },
        {
          name: 'clipfrac 定义：A>0 且 r<1−ε 不算裁剪',
          hidden: true,
          code: py`
            lp = np.log(np.array([[0.5, 0.5, 1.5, 1.5]])); old = np.zeros((1, 4))
            A = np.array([[1.0, -1.0, 1.0, -1.0]]); mask = np.ones((1, 4)); Z = np.zeros((1, 4))
            out = ppo_loss(lp, old, A, Z, Z, Z, mask)
            # 被裁剪的只有 (r=0.5, A<0) 与 (r=1.5, A>0)
            assert abs(out["clipfrac"] - 0.5) < 1e-12, f"clipfrac 期望 0.5，得到 {out['clipfrac']}——是不是用了 |r-1|>ε？"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-early-stop',
          q: md`训练中 approx_kl 突然变大、clipfrac 很高，说明什么？怎么处理？`,
          a: md`
            说明新策略已经明显偏离采样时的策略，重要性采样的方差变大、clip 近似失效。常见处理：降低学习率或 PPO epoch 数、增大 mini-batch、按 approx_kl 早停（超过阈值就停止本轮剩余的 mini-batch 更新）；也要检查 rollout 引擎（vLLM）与训练前向的 logp 是否一致（训推不一致会制造虚假的 ratio 偏离，见「推理强化学习」的 TIS 一节）。
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-four',
      freq: 3,
      q: md`讲一下 PPO-RLHF 的整体流程，四个模型分别做什么？`,
      a: md`
        **结论**：actor 采样、reference 给 KL、RM 给终局分数、critic 估计价值；actor 和 critic 训练，另外两个冻结。

        **流程**：① actor 对 prompt 生成回复并记录 $\log\pi_{\mathrm{old}}$；② ref、RM、critic 前向；③ 奖励 $r_t=-\beta\,\mathrm{KL}_t$，末 token 再加 RM 分数；④ 倒序 GAE 得到 $\hat A_t$ 与 $\hat R_t$，优势白化；⑤ 切 mini-batch 训练若干 epoch：clipped policy loss + value loss。

        **延伸**：InstructGPT 用 6B RM/value、β = 0.02、ε = 0.2、γ = 1；PPO-ptx 加预训练损失（γ = 27.8）缓解 alignment tax。显存上 actor/critic 需要完整优化器状态，四模型共存是 PPO 工程上最大的负担，也是 DPO、GRPO 流行的原因。
      `,
    },
    {
      id: 'qa-gae',
      freq: 3,
      q: md`GAE 是什么？γ 和 λ 分别起什么作用？RLHF 里通常怎么取？`,
      a: md`
        **结论**：GAE 是对多步 TD 误差的指数加权平均，$\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$，递推式 $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$。

        **原因**：γ 是 MDP 的折扣，决定未来奖励的权重；λ 是估计器的超参，在 TD(0)（λ = 0，低方差高偏差，完全依赖 critic）与蒙特卡洛（λ = 1，无偏高方差）之间插值。

        **延伸**：RLHF 奖励稀疏在末尾、回合短，通常 γ = 1（InstructGPT 明确不加折扣），λ 取 0.95 左右（Tülu 3）；critic 的目标 $\hat R_t=\hat A_t+V$；实现要倒序，padding 处 value 置 0。
      `,
    },
    {
      id: 'qa-clip',
      freq: 3,
      q: md`PPO 为什么要 clip？clip 之后哪些 token 没有梯度？`,
      a: md`
        **结论**：clip 是信任域的廉价近似，防止同一批旧数据被反复利用时策略一步走太远。

        **原因**：重要性比 $r=\pi_\theta/\pi_{\mathrm{old}}$ 偏离 1 越多，用旧数据估计的目标越不可信。$\min(rA,\mathrm{clip}(r)A)$ 是悲观下界：$A>0$ 且 $r>1+\epsilon$、或 $A<0$ 且 $r<1-\epsilon$ 时取常数项，梯度为 0；反方向（变差）时保留梯度，允许纠正。

        **延伸**：$A<0$ 且 $r\gg1$ 时损失无下界，dual-clip 用 $\max(\cdot, cA)$ 截断；监控 clipfrac 与 approx_kl；DAPO 的 clip-higher 把上界放宽以缓解熵坍缩。
      `,
    },
    {
      id: 'qa-kl-where',
      freq: 2,
      q: md`KL 惩罚放在 reward 里和放在 loss 里有什么区别？`,
      a: md`
        **结论**：PPO-RLHF 把 $-\beta\log(\pi/\pi_{\mathrm{ref}})$ 作为逐 token 奖励；GRPO 把 k3 形式的 KL 直接加进 loss。

        **原因**：放在 reward 里，KL 会进入回报，被 critic 估计并通过 GAE 影响前面所有 token 的优势，梯度经过 $\nabla\log\pi\cdot A$ 传递（是 REINFORCE 式的估计）；放在 loss 里则是对 KL 的直接解析梯度，只作用于当前 token，不经过优势，方差更小、实现更简单，但不再严格对应“最大化 $\mathbb E[r]-\beta\mathrm{KL}$”这一目标。

        **延伸**：k1 无偏但逐 token 可负，k3 = $(r-1)-\log r$ 非负、低方差；作为 loss 时用 k3 的梯度是否无偏是近年的讨论点（详见「推理强化学习」）。
      `,
    },
    {
      id: 'qa-critic',
      freq: 2,
      q: md`PPO-RLHF 中 critic 难训在哪里？有哪些改进？`,
      a: md`
        **结论**：奖励稀疏、critic 与 actor 同规模、初始时 critic 不准，导致优势估计噪声大且训练开销翻倍。

        **原因**：RM 分数只在末尾，critic 要从一个标量回推到每个 token 的价值；用 RM 初始化的 critic 在含义上与 V(s_t) 并不相同；actor 改变后 critic 的目标也在漂移。

        **延伸**：① critic 预热、value clip、更大的 batch；② 用较小的 critic（InstructGPT 6B）；③ 干脆去掉 critic：RLOO、ReMax、GRPO 用多次采样的均值作 baseline；④ 去掉 RL：DPO 系。
      `,
    },
  ],

  cheatsheet: md`
    **四个模型**：Actor（训）· Critic（训，常由 RM 初始化）· Ref（冻）· RM（冻）；模型状态 ≈ 16P + 16P_c + 2P + 2P_rm 字节

    $$r_t=-\beta\log\tfrac{\pi_\theta(y_t|s_t)}{\pi_{\mathrm{ref}}(y_t|s_t)}+\mathbb 1[t=T]\,r_\phi,\quad \delta_t=r_t+\gamma V_{t+1}-V_t,\quad \hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$$

    $$L^{\mathrm{CLIP}}=\mathbb E\big[\min(rA,\mathrm{clip}(r,1\pm\epsilon)A)\big],\quad L^V=\tfrac12\mathbb E\big[\max((V-\hat R)^2,(V^{\mathrm{clip}}-\hat R)^2)\big]$$

    \`\`\`python
    for t in reversed(range(T)):
        next_v = v[:, t+1] if t+1 < T else 0.0        # padding 处 v 先置 0
        delta = r[:, t] + gamma * next_v - v[:, t]
        last = delta + gamma * lam * last; adv[:, t] = last
    returns = adv + v
    pg = np.maximum(-A * ratio, -A * np.clip(ratio, 1-eps, 1+eps))
    \`\`\`

    - InstructGPT：β = 0.02，ptx γ = 27.8，ε = 0.2，GAE 不折扣，6B RM/value，batch 512 / mini-batch 64
    - Llama 2：β = 0.01（7B/13B）/ 0.005（34B/70B）；Tülu 3：γ = 1、λ = 0.95、c₁ = 0.1、无 EOS 罚 −10
    - 零梯度：A>0 且 r>1+ε；A<0 且 r<1−ε；dual-clip 再截 A<0 且 r>c
    - value clip 是否有用存疑（Engstrom 2020）；clipfrac / approx_kl 是核心监控
  `,
})
