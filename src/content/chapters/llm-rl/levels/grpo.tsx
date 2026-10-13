/**
 * llm-rl-grpo：RLVR 与 GRPO（DeepSeekMath arXiv 2402.03300 原式逐项核对）
 */
import { defineLevel, md, py } from '@/content/helpers'
import { GroupAdvantageViz } from '../viz/GroupAdvantageViz'
import { ClipRegionViz } from '../viz/ClipRegionViz'
import { KLEstimatorViz } from '../viz/KLEstimatorViz'
import { PY_MAKE_BATCH, PY_REF_GRPO } from '../shared'

const SETUP = PY_MAKE_BATCH + '\n\n' + PY_REF_GRPO

export default defineLevel({
  learn: [
    {
      id: 'rlvr',
      title: '从 RLHF 到 RLVR：奖励来自验证器，而不是奖励模型',
      body: md`
        经典 RLHF（InstructGPT 式）用人类偏好训练一个**奖励模型**，再用带 critic 的 PPO 优化策略。
        推理模型时代的主角是 **RLVR**（Reinforcement Learning with Verifiable Rewards）：奖励由一段**程序**判定，而不是另一个神经网络。

        | 任务 | 验证器 | 典型坑 |
        |---|---|---|
        | 数学 | 抽取 \`\boxed{...}\` 里的答案，与标准答案做等价判断（如 Math-Verify：$\tfrac12$ 与 $0.5$ 等价） | 抽取规则过松 → 写多个答案也能命中 |
        | 代码 | 沙箱里跑单元测试，全过记 1 | 测试覆盖不足 → 硬编码样例输出 |
        | 格式 | 是否把推理包在 \`<think>...</think>\` 里、答案是否在指定位置 | 只有格式分 → 模型学会“空想” |

        DeepSeek-R1-Zero 的奖励只有两项且**等权相加**：$\text{Reward}_{\text{rule}} = \text{Reward}_{\text{acc}} + \text{Reward}_{\text{format}}$；
        R1 后续阶段又加了**语言一致性奖励**（CoT 中目标语言词的比例），用来抑制中英混杂。
        R1 论文明确说：推理任务**不用**神经奖励模型（无论 ORM 还是 PRM），因为大规模 RL 中它们容易被 **reward hacking**，重训 RM 又贵又复杂。

        可验证奖励的代价是**稀疏**：一整条几千 token 的回答只拿到一个 0/1。如何把这个序列级信号分配给每个 token，就是 GRPO 要回答的问题。
      `,
      after: md`
        > [!NOTE] 奖励取值的约定不统一
        > DeepSeek-R1-Zero：准确率 + 格式；Dr.GRPO 论文：答对 1、否则 0；DAPO：答对 $+1$、否则 $-1$。
        > 用组内归一化后，奖励的**平移**不影响优势（减均值抵消了），**缩放**在除以 std 时也被抵消——但不除 std 的变体（Dr.GRPO）对缩放敏感。
      `,
      checkpoint: {
        id: 'cp-rlvr',
        type: 'single',
        prompt: md`DeepSeek-R1-Zero 在推理任务上**没有**使用神经奖励模型，论文给出的主要理由是：`,
        options: ['神经奖励模型推理太慢', '大规模 RL 中神经奖励模型容易被 reward hacking，且重训成本高', '规则奖励能提供逐 token 的稠密信号', '神经奖励模型无法处理数学公式'],
        answer: 1,
        explanation: md`论文原话的要点是：neural reward model 在大规模 RL 中容易被 reward hacking，重训它需要额外算力并让流程复杂化。规则奖励恰恰是**稀疏**的（每条回答一个分数），C 说反了；A、D 不是论文给出的理由。`,
      },
    },
    {
      id: 'advantage',
      title: '去掉 critic：用同组 G 个回答的平均奖励当 baseline',
      body: md`
        PPO 需要一个和策略同等规模的**价值模型** $V_\psi$ 来算 GAE 优势：显存翻倍，而且 LLM 场景里奖励只落在最后一个 token 上，
        训练一个“对每个 token 都准确”的价值函数本身就很难。

        **GRPO**（Group Relative Policy Optimization，DeepSeekMath 提出）的做法：对每个问题 $q$，用旧策略 $\pi_{\theta_{old}}$ 采样一组 $G$ 个回答
        $\{o_1,\dots,o_G\}$，打分得到 $\mathbf r = \{r_1,\dots,r_G\}$，然后**组内标准化**：

        $$
        \hat A_{i,t} = \tilde r_i = \frac{r_i - \mathrm{mean}(\mathbf r)}{\mathrm{std}(\mathbf r)}
        $$

        这叫 **outcome supervision**：回答 $o_i$ 的**每个 token** 共享同一个优势。
        论文还给了 **process supervision** 版本：PRM 给每个推理步骤打分，步骤奖励同样做标准化，token $t$ 的优势是它**之后所有步骤**标准化奖励之和
        $\hat A_{i,t} = \sum_{\text{index}(j) \ge t} \tilde r_i^{\,\text{index}(j)}$。

        点击下面每个回答切换对错，观察优势如何变化——尤其是把 8 个都设成同一结果时。
      `,
      viz: GroupAdvantageViz,
      after: md`
        > [!WARN] 实现细节（面试常被追问）
        > - **std 用无偏还是有偏？** 论文没写。TRL / verl 都直接调 \`torch.std\`，默认无偏（ddof=1）；NumPy 的 \`np.std\` 默认 ddof=0。本章统一用 ddof=1。
        > - **分母加 eps**：全组同分时 std=0，需要 \`(std + 1e-6)\`（verl 的写法）或 \`1e-4\`（TRL），否则 0/0。
        > - **全对 / 全错的组优势全 0**：不考虑 KL 项时这道题**不产生梯度**，白白浪费了 G 次采样——这是 DAPO 动态采样的动机。
      `,
      checkpoint: {
        id: 'cp-adv',
        type: 'numeric',
        prompt: md`一组 $G=4$ 个回答的奖励为 $[1, 0, 0, 0]$。按 $\hat A = (r-\text{mean})/\text{std}$（std 用 **ddof=1**，忽略 eps），答对那条回答的优势是多少？`,
        answer: 1.5,
        tolerance: 0.01,
        explanation: md`mean $=0.25$；偏差 $[0.75,-0.25,-0.25,-0.25]$；方差 $=(0.5625+3\times0.0625)/3=0.25$，std $=0.5$；优势 $=0.75/0.5=1.5$。若误用 ddof=0，std $=\sqrt{0.1875}\approx0.433$，得到 $1.732$。`,
      },
    },
    {
      id: 'objective',
      title: 'token 级 clipped 目标与 1/|oᵢ| 长度归一化',
      body: md`
        DeepSeekMath 中 GRPO 的完整目标（式 3）：

        $$
        \mathcal J_{GRPO}(\theta)=\mathbb E\Big[\frac1G\sum_{i=1}^{G}\frac1{|o_i|}\sum_{t=1}^{|o_i|}\Big\{\min\big[r_{i,t}\hat A_{i,t},\ \mathrm{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big]-\beta\,\mathbb D_{KL}[\pi_\theta\Vert\pi_{ref}]\Big\}\Big]
        $$

        其中 $r_{i,t}=\dfrac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})}$ 是**逐 token** 的重要性比。逐项拆开：

        | 项 | 作用 | 来源 |
        |---|---|---|
        | $r_{i,t}\hat A_{i,t}$ | 重要性加权的策略梯度：样本来自 $\pi_{old}$，在 $\pi_\theta$ 上更新 | PPO |
        | $\min(\cdot,\ \mathrm{clip}(\cdot))$ | 取悲观下界：$r$ 越界后目标变平，**梯度为 0** | PPO |
        | $\frac1{\lvert o_i\rvert }\sum_t$ | 先在**每条回答内**对 token 求平均 | GRPO 原式（Dr.GRPO 指出它有偏） |
        | $\frac1G\sum_i$ | 再对组内回答求平均 | GRPO |
        | $-\beta\,\mathbb D_{KL}$ | 直接放在 loss 里的 KL 正则 | GRPO（下一节） |

        **什么时候 $r\neq1$？** 采样完一大批 rollout 后要切成多个 mini-batch 依次更新：第一个 mini-batch 时 $\pi_\theta=\pi_{old}$，$r\equiv1$，clip 不起作用；
        之后参数已经变了，$r$ 开始偏离 1，clip 才生效。DeepSeekMath 的设置是“每次探索后只更新一次”（纯 on-policy），
        R1-Zero 则把每次 rollout 的 8,192 条输出切成 16 个 mini-batch、各训一次。
      `,
      viz: ClipRegionViz,
      after: md`
        > [!INTERVIEW] 一句话讲清 clip
        > “A>0 时只允许把概率提到 $(1+\varepsilon)\pi_{old}$，再往上 min 会选中常数项，梯度断掉；A<0 时只允许压到 $(1-\varepsilon)\pi_{old}$。
        > 它不是硬约束，而是**让越界的 token 不再提供梯度**，近似一个可信域。”

        注意 $r$ 在可信域**内侧**但方向“不利”时（A>0 且 $r<1-\varepsilon$）仍有梯度——min 选的是未裁剪项，目的是把它拉回来。
      `,
      checkpoint: {
        id: 'cp-obj',
        type: 'judge',
        prompt: md`若每个 rollout batch 只做**一次**梯度更新（不切 mini-batch），则 GRPO 中的 clip 操作不会影响梯度。`,
        answer: true,
        explanation: md`唯一一次更新时 $\pi_\theta=\pi_{old}$，所有 $r_{i,t}=1$ 都在 $[1-\varepsilon,1+\varepsilon]$ 内，min 两项相等，梯度就是普通的 $\hat A\,\nabla\log\pi_\theta$。clip 只在同一批数据上做多步更新（off-policy）时才生效。`,
      },
    },
    {
      id: 'kl',
      title: 'KL 正则：k3 估计，直接加进 loss',
      body: md`
        PPO-RLHF 的 KL 是**逐 token 加进奖励**的：$r_t = r_\varphi(q,o_{\le t}) - \beta\log\frac{\pi_\theta(o_t)}{\pi_{ref}(o_t)}$，再经过 GAE 进入优势。
        GRPO 改为**直接加在 loss 里**，“避免把 $\hat A_{i,t}$ 的计算搞复杂”，并采用 Schulman（2020）的无偏估计：

        $$
        \mathbb D_{KL}[\pi_\theta\Vert\pi_{ref}]=\frac{\pi_{ref}(o_{i,t}\mid\cdot)}{\pi_\theta(o_{i,t}\mid\cdot)}-\log\frac{\pi_{ref}(o_{i,t}\mid\cdot)}{\pi_\theta(o_{i,t}\mid\cdot)}-1
        $$

        记 $x=\pi_{ref}/\pi_\theta$，这就是 **k3** $=x-1-\log x$。因为 $\mathbb E_{\pi_\theta}[x]=1$，它与 k1 $=-\log x$ 期望相同（无偏）；又因 $\log x\le x-1$，每个样本都 $\ge0$。

        **对它求导**（样本固定，只对 $\theta$ 求导）：$\nabla_\theta\,k_3=(1-x)\nabla_\theta\log\pi_\theta$。
        这正是论文附录里 GRPO 的梯度系数 $\hat A_{i,t}+\beta\big(\frac{\pi_{ref}}{\pi_\theta}-1\big)$ 中的 KL 部分：$\pi_\theta<\pi_{ref}$ 的 token 被往上拉，反之往下压。

        R1 论文还补充了一个工程理由：PPO 的逐 token KL 进入**累计回报**，会隐式惩罚长回答，阻碍长 CoT；并且长时间训练中策略会远离初始模型，所以他们**每 400 步把参考模型重置为当前策略**。
      `,
      viz: KLEstimatorViz,
      after: md`
        > [!INTERVIEW] 压轴追问：k3 当 loss 求导，优化的到底是哪个 KL？
        > 在 $o\sim\pi_\theta$ 的单步 on-policy 设定下：
        > $\mathbb E_{\pi_\theta}[(1-\tfrac{\pi_{ref}}{\pi_\theta})\nabla\log\pi_\theta]=\sum_o\nabla\pi_\theta-\sum_o\pi_{ref}\nabla\log\pi_\theta=-\mathbb E_{\pi_{ref}}[\nabla\log\pi_\theta]=\nabla_\theta\,\mathrm{KL}(\pi_{ref}\Vert\pi_\theta)$，
        > 即**正向 KL** 的梯度，而不是我们写在目标里的反向 KL。对比：k1 当 loss 的期望梯度是 $\mathbb E[\nabla\log\pi_\theta]=0$（完全没约束）；
        > k2 $=\tfrac12(\log x)^2$ 当 loss 的期望梯度恰好是反向 KL 的梯度——REINFORCE++（带 baseline 版）因此改用 k2。

        > [!NEW] 2025 之后的走向
        > DAPO、CISPO 等直接**去掉 KL**（β=0）；DeepSeek-V3.2 保留 KL，但给 k3 乘上 $\pi_\theta/\pi_{old}$ 做重要性修正，使其梯度无偏，
        > 并指出原始 k3 在 $\pi_\theta\ll\pi_{ref}$ 时梯度权重无界、会让训练不稳；同时说数学等领域用很弱的 KL 甚至不用效果更好。
      `,
      checkpoint: {
        id: 'cp-kl',
        type: 'numeric',
        prompt: md`某 token 上 $\pi_{ref}/\pi_\theta = 2$，k3 估计值是多少？（自然对数，保留 3 位小数）`,
        answer: 0.307,
        tolerance: 0.01,
        explanation: md`$k_3 = x-1-\ln x = 2-1-\ln2 = 1-0.6931 = 0.3069$。注意 k1 $=-\ln 2=-0.693$ 是负数——单样本 k1 可以为负，k3 不会。`,
      },
    },
    {
      id: 'recipe',
      title: '训练循环与真实超参',
      body: md`
        \`\`\`text
        for step in range(M):
            π_old ← π_θ                                  # 同步采样权重
            对 batch 中每个 q 用 π_old 采样 G 个回答；验证器打分得到 r
            Â ← 组内 (r − mean) / std                     # 每条回答一个数，广播到它的所有 token
            for 每个 mini-batch:                          # 第 2 个起 π_θ ≠ π_old，clip 才生效
                计算 r_{i,t}、clipped 目标、k3 KL → 掩码后先回答内平均、再回答间平均 → 反向传播
        （每隔若干步把 π_ref 重置为 π_θ；DeepSeekMath 的 iterative 版本还会用新数据续训奖励模型）
        \`\`\`

        | 超参 | DeepSeekMath-RL 7B | DeepSeek-R1-Zero |
        |---|---|---|
        | 学习率 | 1e-6 | 3e-6 |
        | KL 系数 β | 0.04 | 0.001 |
        | 每题采样 G | 64 | 16 |
        | 最大长度 | 1,024 | 32,768（8.2k 步后提到 65,536） |
        | batch | 1,024 | 32 题 × 16 = 512 |
        | 更新方式 | 每次探索后只更新一次 | 每次 rollout 8,192 条，切 16 个 mini-batch，单个 inner epoch |
        | 参考模型 | iterative 版本每轮重置 | 每 400 步重置为当前策略 |

        R1（非 Zero）第一阶段 RL 的 clip 比 $\varepsilon$ 设为 **10**，论文说太小会截断大量 token 的梯度、太大又不稳——可见 $\varepsilon=0.2$ 并不是金科玉律。
        DeepSeekMath 还发现 RL 提升了 **Maj@K 却没提升 Pass@K**，认为 RL 主要是让输出分布更稳健，而非提升基础能力——这是后来 “RL 是否扩展能力边界” 之争的源头（见「训练动力学」一关）。
      `,
      checkpoint: {
        id: 'cp-recipe',
        type: 'fill',
        prompt: md`DeepSeekMath 的 GRPO 实验中，每个问题采样 ____(1)____ 个输出，KL 系数 β = ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['64'], numeric: true, tolerance: 0 },
          { label: '(2)', answers: ['0.04'], numeric: true, tolerance: 0.001 },
        ],
        explanation: md`原文：policy 学习率 1e-6，KL 系数 0.04，每题采样 64 个输出，最大长度 1024，训练 batch 1024，每次探索只更新一次。R1-Zero 则是 G=16、β=0.001。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-baseline',
      type: 'single',
      difficulty: 1,
      prompt: md`GRPO 用什么替代了 PPO 中价值模型（critic）提供的 baseline？`,
      options: ['参考模型 π_ref 的对数概率', '同一问题 G 个采样回答的平均奖励', '整个 batch 所有回答的平均奖励', '奖励模型对 prompt 的打分'],
      answer: 1,
      explanation: md`GRPO 的 baseline 是**同组**回答的平均奖励（再除以组内 std）。C 是 REINFORCE++ 一类“全局归一化”的做法；A 是 KL 正则用的；D 不存在。`,
    },
    {
      id: 'q-adv-num',
      type: 'numeric',
      difficulty: 2,
      prompt: md`一组 $G=8$ 个回答的 0/1 奖励为 $[1,0,0,1,1,0,0,0]$。std 用 ddof=1、忽略 eps，答对回答的优势是多少？（保留 3 位小数）`,
      answer: 1.208,
      tolerance: 0.005,
      explanation: md`mean $=3/8=0.375$；方差 $=\frac{3\times0.625^2+5\times0.375^2}{7}=\frac{1.875}{7}=0.2679$，std $=0.5175$；优势 $=0.625/0.5175=1.208$，答错的为 $-0.375/0.5175=-0.725$。`,
    },
    {
      id: 'q-allsame',
      type: 'judge',
      difficulty: 1,
      prompt: md`不考虑 KL 项时，若某题的 G 个回答全部答对，这道题对本次策略更新的梯度贡献为 0。`,
      answer: true,
      explanation: md`全组同分 → 每个 $r_i-\text{mean}=0$ → 优势全 0 → 策略梯度项为 0。只有 $\beta>0$ 时 KL 项还会产生梯度，所以题干限定了“不考虑 KL”。`,
    },
    {
      id: 'q-k3',
      type: 'multi',
      difficulty: 3,
      prompt: md`关于 GRPO 使用的 KL 估计 $k_3 = \frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$，下列说法正确的是：（多选）`,
      options: [
        '对每个采样 token，k3 的值都 ≥ 0',
        '样本来自 π_θ 时，它是 KL(π_θ‖π_ref) 的无偏估计',
        'GRPO 把它逐 token 加进奖励，再计算组内优势',
        '直接把 k3 当 loss 项求导，期望梯度就是反向 KL(π_θ‖π_ref) 的梯度',
        'π_θ ≪ π_ref 的 token 上，该项的梯度权重会非常大',
      ],
      answer: [0, 1, 4],
      explanation: md`A：$\log x\le x-1$ 保证非负。B：$\mathbb E_{\pi_\theta}[x]=1$，故与 k1 同期望，无偏。C 错：GRPO 恰恰是**加在 loss 里**，PPO-RLHF 才加进奖励。D 错：期望梯度是 $\nabla\mathrm{KL}(\pi_{ref}\Vert\pi_\theta)$（正向 KL）。E：梯度系数 $1-\pi_{ref}/\pi_\theta$ 在 $\pi_\theta\ll\pi_{ref}$ 时绝对值无界，DeepSeek-V3.2 正是因此做了修正。`,
    },
    {
      id: 'q-pick-clip',
      type: 'pick',
      difficulty: 2,
      prompt: md`$\varepsilon=0.2$。下列 token 中，哪些被 clip **截断了梯度**（∂L/∂r = 0）？`,
      items: ['A>0, r=1.35', 'A>0, r=0.70', 'A<0, r=0.70', 'A<0, r=1.35', 'A>0, r=1.10', 'A<0, r=0.95'],
      answer: [0, 2],
      explanation: md`A>0 时只有 $r>1.2$ 被截断（①）；A>0、$r=0.7$ 时 min 选未裁剪项 $0.7A<0.8A$，仍有梯度（②）。A<0 时只有 $r<0.8$ 被截断（③）；A<0、$r=1.35$ 时 $1.35A<1.2A$，min 选未裁剪项，梯度照常——**坏回答的概率被意外抬高时，clip 不会放过它**。`,
    },
    {
      id: 'q-onpolicy-loss',
      type: 'single',
      difficulty: 3,
      prompt: md`on-policy 第一次更新（$\log\pi_\theta=\log\pi_{old}$）、$\beta=0$、各组大小相同。GRPO loss（seq-mean-token-mean 聚合）的**数值**是：`,
      options: ['0，且梯度也为 0', '0，但梯度一般不为 0', '等于 −mean(Â)，非零', '取决于各回答长度，无法确定'],
      answer: 1,
      explanation: md`$r\equiv1$ 时每条回答的 token 平均值就是 $-\hat A_i$，再对回答求平均得 $-\frac1B\sum_i\hat A_i$；每组优势之和为 0，所以 loss **恰为 0**。但梯度是 $-\frac1B\sum_i\frac{\hat A_i}{|o_i|}\sum_t\nabla\log\pi_\theta(o_{i,t})$，一般不为 0。**loss 曲线贴着 0 不代表没学**——这是复现 GRPO 时的常见困惑。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 GRPO 一个训练 step 的流程按顺序排列：`,
      items: [
        'π_old ← π_θ，同步采样权重',
        '对每个问题用 π_old 采样 G 个回答',
        '规则验证器给每个回答打分',
        '组内 (r − mean)/std 得到每条回答的优势',
        '逐 token 计算 ratio、clipped 目标与 k3 KL，掩码后聚合成 loss',
        '对各 mini-batch 反向传播并更新 π_θ',
      ],
      explanation: md`先同步权重再采样（保证 rollout 来自 $\pi_{old}$）；奖励 → 优势只依赖整条回答；最后才进入 token 级的 loss 与多次 mini-batch 更新。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把 PPO-RLHF 与 GRPO 的组件配对：`,
      pairs: [
        ['PPO 的优势', 'GAE + 价值模型'],
        ['GRPO 的优势', '同组奖励标准化'],
        ['PPO 的 KL', '逐 token 并入奖励'],
        ['GRPO 的 KL', 'k3 估计直接进 loss'],
      ],
      explanation: md`GRPO 去掉价值模型，用组内相对奖励做优势；KL 也从“奖励的一部分”挪到“loss 的一项”，因此优势计算与 KL 解耦。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出组内优势（rewards 形状 (B,)，每 G 个连续样本为一组；有 2 行干扰项）：`,
      lines: [
        'def group_advantage(rewards, G, eps=1e-6):',
        '    r = rewards.reshape(-1, G)',
        '    mean = r.mean(axis=1, keepdims=True)',
        '    std = r.std(axis=1, ddof=1, keepdims=True)',
        '    adv = (r - mean) / (std + eps)',
        '    return adv.reshape(-1)',
      ],
      distractors: ['    mean = r.mean()', '    std = r.std(axis=0, ddof=1, keepdims=True)'],
      explanation: md`统计量必须沿**组内**轴（axis=1）算并 keepdims 以便广播。\`r.mean()\` 是全 batch 均值（那是另一种算法）；axis=0 是跨组同位置统计，没有意义。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 GRPO loss 有 **2 处** bug，点出来：`,
      code: py`
        def grpo_loss(logp_new, logp_old, logp_ref, adv, mask, eps=0.2, beta=0.04):
            # adv: (B,) 组内归一化后的优势；其余 (B, T)
            A = adv[:, None]
            ratio = np.exp(logp_new - logp_ref)
            surr1 = ratio * A
            surr2 = np.clip(ratio, 1 - eps, 1 + eps) * A
            pg = -np.maximum(surr1, surr2)
            d = logp_ref - logp_new
            kl = np.exp(d) - d - 1
            tok = (pg + beta * kl) * mask
            per_seq = tok.sum(axis=1) / mask.sum(axis=1)
            return per_seq.mean()
      `,
      bugLines: [4, 7],
      explanation: md`第 4 行：重要性比的分母是**采样策略** $\pi_{old}$，应为 \`np.exp(logp_new - logp_old)\`；用 \`logp_ref\` 会把 KL 正则和 IS 混为一谈。第 7 行：目标取 **min**（悲观下界），loss 取负后应为 \`-np.minimum(surr1, surr2)\`；写成 max 会让越界 token 反而走未裁剪项，clip 失效。`,
    },
    {
      id: 'q-r1-reward',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 DeepSeek-R1 / R1-Zero 的奖励设计，正确的是：（多选）`,
      options: [
        'R1-Zero 的规则奖励 = 准确率奖励 + 格式奖励，两者等权',
        '格式奖励要求把推理过程放在 <think> 与 </think> 之间',
        '推理任务上使用了过程奖励模型（PRM）',
        'R1 加入的语言一致性奖励 = CoT 中目标语言词的比例',
      ],
      answer: [0, 1, 3],
      explanation: md`A、B、D 均为论文原文设定。C 错：论文明确不在推理任务上使用神经 ORM/PRM，理由是 reward hacking 与重训成本。`,
    },
    {
      id: 'q-k3-val',
      type: 'numeric',
      difficulty: 2,
      prompt: md`某 token 上 $\pi_\theta = 0.5$，$\pi_{ref} = 0.2$。求 k3 估计值（自然对数，保留 3 位小数）。`,
      answer: 0.316,
      tolerance: 0.01,
      explanation: md`$x=\pi_{ref}/\pi_\theta=0.4$，$k_3=0.4-1-\ln0.4=-0.6+0.9163=0.3163$。注意 $x$ 是 **ref 比 θ**，倒过来算会得到 $2.5-1-\ln2.5=0.584$。`,
    },
  ],

  guided: [
    {
      id: 'g-grpo',
      title: '一步步写出 GRPO loss（连续分组版）',
      intro: md`
        输入：\`logp_new / logp_old / logp_ref\` 形状 \`(B, T)\`，是**采样 token** 在三个策略下的对数概率；\`rewards\` 形状 \`(B,)\`；
        \`mask\` 为 \`(B, T)\` 的 0/1 回答掩码；每 \`G\` 个连续样本属于同一道题。返回标量 loss（**最小化**它）。
        约定：std 用 ddof=1，分母加 1e-6；聚合方式为 GRPO 原式（先回答内 token 平均，再回答间平均）。
      `,
      template: py`
        import numpy as np

        def grpo_loss_simple(logp_new, logp_old, logp_ref, rewards, mask, G, clip_eps=0.2, beta=0.04):
            # 1) 组内优势：(B,) -> (B/G, G) 上做标准化
            r = rewards.reshape(-1, G)
            adv = {{1}}
            A = adv.reshape(-1, 1)                    # (B, 1)：同一回答的所有 token 共享优势
            # 2) 逐 token 重要性比
            ratio = {{2}}
            # 3) clipped surrogate（悲观下界）
            surr = {{3}}
            # 4) k3 KL 估计
            kl = {{4}}
            # 5) 逐 token loss：最大化目标 → 取负
            per_tok = {{5}}
            # 6) 先在每条回答内对有效 token 平均，再对回答平均
            return {{6}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`沿组内轴（axis=1）减均值、除以 std（ddof=1）加 1e-6。记得 keepdims，才能和 \`r\` 广播。`,
          answer: '(r - r.mean(axis=1, keepdims=True)) / (r.std(axis=1, ddof=1, keepdims=True) + 1e-6)',
          accept: ['(r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-6)'],
          hint: 'np.std 默认 ddof=0，要显式写 ddof=1。',
          explain: md`这一步把“绝对奖励”变成“组内相对好坏”。ddof 差一点数值，面试时说清楚“和 torch.std 对齐用无偏估计”即可。`,
        },
        {
          blank: 2,
          instruction: md`重要性比 $r_{i,t}=\pi_\theta/\pi_{old}$，在对数空间相减再取 exp。`,
          answer: 'np.exp(logp_new - logp_old)',
          explain: md`分母是**采样策略** $\pi_{old}$，不是 $\pi_{ref}$。在 log 空间相减避免概率下溢。`,
        },
        {
          blank: 3,
          instruction: md`未裁剪项 \`ratio * A\` 与裁剪项 \`clip(ratio, 1-ε, 1+ε) * A\` 取逐元素最小值。`,
          answer: 'np.minimum(ratio * A, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)',
          explain: md`取 min 才是“悲观下界”：越界后选中常数项，梯度为 0。`,
        },
        {
          blank: 4,
          instruction: md`$k_3=x-\log x-1$，$x=\pi_{ref}/\pi_\theta$；用 \`d = logp_ref - logp_new\` 表示 $\log x$。`,
          answer: 'np.exp(logp_ref - logp_new) - (logp_ref - logp_new) - 1',
          accept: ['np.exp(logp_ref - logp_new) - logp_ref + logp_new - 1'],
          explain: md`方向别反：是 **ref 比 new**。它对每个 token 非负，期望等于 KL(π_θ‖π_ref)。`,
        },
        {
          blank: 5,
          instruction: md`目标是 $\text{surr}-\beta\,\text{kl}$，loss 取负。`,
          answer: '-(surr - beta * kl)',
          accept: ['-surr + beta * kl', 'beta * kl - surr'],
          explain: md`KL 以 **+β·kl** 的形式出现在 loss 中——惩罚偏离参考模型。`,
        },
        {
          blank: 6,
          instruction: md`掩码后按行求和、除以每行有效长度得到每条回答的平均，再对所有回答取均值。`,
          answer: '((per_tok * mask).sum(axis=1) / mask.sum(axis=1)).mean()',
          accept: ['((per_tok * mask).sum(1) / mask.sum(1)).mean()', 'np.mean((per_tok * mask).sum(axis=1) / mask.sum(axis=1))'],
          explain: md`这就是 $\frac1G\sum_i\frac1{|o_i|}\sum_t$。下一关会看到：$\frac1{|o_i|}$ 让“又长又错”的回答每个 token 受罚更轻——Dr.GRPO 说它有偏。`,
        },
      ],
      setup: SETUP,
      tests: [
        {
          name: 'on-policy 第一步：loss 应为 0',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(0, garbage=False)
            got = float(grpo_loss_simple(lo, lo, lo, r, m, 4))
            assert abs(got) < 1e-9, f"logp_new == logp_old == logp_ref 时 loss 应为 0（组内优势和为 0），得到 {got}"
          `,
        },
        {
          name: '与参考实现对拍',
          code: py`
            for seed in range(8):
                ln, lo, lr, r, gid, m = _make_batch(seed, garbage=False)
                got = float(grpo_loss_simple(ln, lo, lr, r, m, 4))
                exp, _ = _ref_grpo_loss(ln, lo, lr, r, gid, m)
                assert abs(got - exp) < 1e-8, f"seed={seed}: 期望 {exp:.8f}，得到 {got:.8f}"
          `,
        },
        {
          name: 'β 与 ε 参数生效',
          hidden: true,
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(11, garbage=False)
            for eps, beta in [(0.1, 0.0), (0.3, 0.5)]:
                got = float(grpo_loss_simple(ln, lo, lr, r, m, 4, clip_eps=eps, beta=beta))
                exp, _ = _ref_grpo_loss(ln, lo, lr, r, gid, m, clip_eps=eps, beta=beta)
                assert abs(got - exp) < 1e-8, f"eps={eps}, beta={beta}: 期望 {exp:.8f}，得到 {got:.8f}"
          `,
        },
      ],
      outro: md`
        6 行就是 GRPO 的全部。实战题会把它推广到**乱序的 group_ids**、带 padding 垃圾值的 batch，并要求返回 clip 比例与 KL 统计量——
        这些正是训练时监控面板上的曲线。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-grpo-loss',
      title: '手写 GRPO loss（含 mask、乱序分组与监控统计）',
      difficulty: 'hard',
      source: '大模型岗高频手撕 · GRPO',
      timeLimitMin: 25,
      description: md`
        实现 DeepSeekMath 原式的 GRPO loss（**最小化**），并返回训练监控用的统计量。

        \`\`\`python
        def grpo_loss(logp_new, logp_old, logp_ref, rewards, group_ids, mask,
                      clip_eps=0.2, beta=0.04, adv_eps=1e-6) -> tuple[float, dict]: ...
        \`\`\`

        **输入**（float64 NumPy 数组）
        - \`logp_new / logp_old / logp_ref\`：\`(B, T)\`，采样 token 在当前 / 采样时 / 参考策略下的 log 概率；
        - \`rewards\`：\`(B,)\`；\`group_ids\`：\`(B,)\` 整数，**相同 id 的行属于同一道题，不保证连续、id 不保证从 0 开始**（每组 ≥ 2 条）；
        - \`mask\`：\`(B, T)\` 的 0/1，每行至少 1 个有效 token。**padding 位置的 logp 是任意垃圾值**，不能影响结果。

        **计算**
        1. 组内优势 $\hat A_i=(r_i-\text{mean}_g)/(\text{std}_g+\texttt{adv\_eps})$，std 用 **ddof=1**；
        2. $r_{i,t}=\exp(\log\pi_{new}-\log\pi_{old})$，token loss $=-\min(r\hat A,\ \mathrm{clip}(r,1\pm\varepsilon)\hat A)+\beta\,k_3$，
           $k_3=e^{d}-d-1$，$d=\log\pi_{ref}-\log\pi_{new}$；
        3. 聚合：每行对有效 token 求平均，再对 **B 行**求平均。

        **返回** \`(loss, stats)\`：\`loss\` 为 Python float；\`stats\` 含
        - \`"adv"\`：\`(B,)\` 优势；
        - \`"kl"\`：所有有效 token 上 $k_3$ 的平均（token 级平均）；
        - \`"clip_frac"\`：有效 token 中“裁剪项被 min 选中且严格更小”的比例，即 \`surr2 < surr1\`。

        **示例**：\`logp_new == logp_old == logp_ref\` 时 loss = 0、kl = 0、clip_frac = 0。
      `,
      starter: py`
        import numpy as np

        def grpo_loss(logp_new, logp_old, logp_ref, rewards, group_ids, mask,
                      clip_eps=0.2, beta=0.04, adv_eps=1e-6):
            # TODO 1: 按 group_ids 计算组内 mean / std(ddof=1) → 优势 adv (B,)
            # TODO 2: ratio、clipped surrogate、k3 KL（注意 padding 位置的垃圾值）
            # TODO 3: 先行内平均、再行间平均；统计 kl 与 clip_frac
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def grpo_loss(logp_new, logp_old, logp_ref, rewards, group_ids, mask,
                      clip_eps=0.2, beta=0.04, adv_eps=1e-6):
            m = np.asarray(mask, dtype=np.float64)
            r = np.asarray(rewards, dtype=np.float64)

            # 1) 组内优势：np.unique 把任意 id 映射到 0..n-1，再用 bincount 做分组统计
            _, inv = np.unique(group_ids, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv).astype(np.float64)
            mean = np.bincount(inv, weights=r) / cnt
            dev = r - mean[inv]
            std = np.sqrt(np.bincount(inv, weights=dev ** 2) / (cnt - 1))   # ddof=1
            adv = dev / (std[inv] + adv_eps)
            A = adv[:, None]

            # 2) token 级目标；先把 padding 位置的差值置 0，防止 exp 溢出污染
            log_ratio = np.where(m > 0, logp_new - logp_old, 0.0)
            ratio = np.exp(log_ratio)
            surr1 = ratio * A
            surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
            d = np.where(m > 0, logp_ref - logp_new, 0.0)
            kl = np.exp(d) - d - 1                       # k3 >= 0
            tok = (-np.minimum(surr1, surr2) + beta * kl) * m

            # 3) 聚合：1/|o_i| Σ_t，再对 B 条回答平均
            lens = m.sum(axis=1)
            loss = float(np.mean(tok.sum(axis=1) / lens))

            n_tok = m.sum()
            stats = {
                "adv": adv,
                "kl": float((kl * m).sum() / n_tok),
                "clip_frac": float(((surr2 < surr1) * m).sum() / n_tok),
            }
            return loss, stats
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def grpo_loss_torch(logp_new, logp_old, logp_ref, rewards, group_ids, mask,
                                clip_eps=0.2, beta=0.04, adv_eps=1e-6):
                """logp_new 带梯度 (B, T)；logp_old / logp_ref / rewards 不需要梯度。"""
                mask = mask.float()
                _, inv = torch.unique(group_ids, return_inverse=True)
                n = int(inv.max()) + 1
                z = torch.zeros(n, dtype=rewards.dtype, device=rewards.device)
                cnt = z.index_add(0, inv, torch.ones_like(rewards))
                mean = z.index_add(0, inv, rewards) / cnt
                dev = rewards - mean[inv]
                std = (z.index_add(0, inv, dev ** 2) / (cnt - 1)).sqrt()      # 与 torch.std 一样无偏
                adv = (dev / (std[inv] + adv_eps)).detach()
                A = adv.unsqueeze(-1)

                ratio = torch.exp((logp_new - logp_old.detach()) * mask)
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
                d = (logp_ref.detach() - logp_new) * mask
                kl = torch.exp(d) - d - 1
                tok = (-torch.min(surr1, surr2) + beta * kl) * mask
                loss = (tok.sum(-1) / mask.sum(-1)).mean()

                with torch.no_grad():
                    n_tok = mask.sum()
                    stats = {"adv": adv,
                             "kl": (kl * mask).sum() / n_tok,
                             "clip_frac": ((surr2 < surr1).float() * mask).sum() / n_tok}
                return loss, stats
          `,
        },
      ],
      explanation: md`
        **思路**：三段式——组内优势 → token 级 clipped 目标 + k3 → 掩码聚合。

        **关键点**
        - 分组不连续时不能 reshape；\`np.unique(..., return_inverse=True)\` + \`np.bincount\` 是 $O(B)$ 的通用分组统计（PyTorch 用 \`index_add\` / \`scatter_add\`）。
        - padding 位置的垃圾值必须在 \`exp\` **之前**处理（\`np.where\` 置 0），否则可能出现 \`inf * 0 = nan\`。
        - 聚合顺序决定了算法：先行内平均再行间平均是 GRPO 原式；换成全部 token 一起平均就是 DAPO 的 token-level loss（下一关）。
        - \`clip_frac\` 用 \`surr2 < surr1\`（等价于 verl 的 \`pg_losses2 > pg_losses\`），是判断“更新步子是否过大”的首要监控指标。

        **复杂度**：$O(BT)$ 时间，$O(BT)$ 空间。

        **常见错误**：std 用了 ddof=0；ratio 分母写成 \`logp_ref\`；KL 方向写反；忘了 mask 或用 \`mean(axis=1)\` 把 padding 也算进长度。
      `,
      complexity: { time: 'O(B·T)', space: 'O(B·T)' },
      hints: [
        md`先写优势：把 \`group_ids\` 映射成 \`0..n-1\` 的下标 \`inv\`，用 \`np.bincount(inv, weights=...)\` 求每组的和与平方和。`,
        md`ratio 与 k3 都要 \`exp\`：先用 \`np.where(mask > 0, diff, 0.0)\` 把 padding 位置清零。`,
        md`loss：\`(tok * m).sum(1) / m.sum(1)\` 再 \`.mean()\`；\`clip_frac\` 是 \`((surr2 < surr1) * m).sum() / m.sum()\`。`,
      ],
      setup: SETUP,
      tests: [
        {
          name: '返回值类型与 stats 字段',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(0)
            loss, st = grpo_loss(ln, lo, lr, r, gid, m)
            assert isinstance(loss, (float, np.floating)), f"loss 应为 float，得到 {type(loss)}"
            for k in ("adv", "kl", "clip_frac"):
                assert k in st, f"stats 缺少字段 {k!r}"
            assert np.shape(st["adv"]) == (len(r),), f"adv 形状应为 {(len(r),)}，得到 {np.shape(st['adv'])}"
          `,
        },
        {
          name: 'on-policy 第一步：loss = kl = clip_frac = 0',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(1, garbage=False)
            loss, st = grpo_loss(lo, lo, lo, r, gid, m)
            assert abs(loss) < 1e-9 and abs(st["kl"]) < 1e-12 and st["clip_frac"] == 0, f"得到 loss={loss}, kl={st['kl']}, clip_frac={st['clip_frac']}"
          `,
        },
        {
          name: '全对组优势为 0；只有一个对的组优势符合 ddof=1',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(2)
            _, st = grpo_loss(ln, lo, lr, r, gid, m)
            adv = np.asarray(st["adv"])
            assert np.allclose(adv[:4], 0), f"第 0 组奖励全为 1，优势应全 0，得到 {adv[:4]}"
            assert np.allclose(adv[4:8], [1.5, -0.5, -0.5, -0.5], atol=1e-5), f"奖励 [1,0,0,0] 的优势应为 [1.5,-0.5,-0.5,-0.5]（ddof=1），得到 {adv[4:8]}"
          `,
        },
        {
          name: '随机对拍（含 padding 垃圾值）',
          hidden: true,
          code: py`
            for seed in range(30):
                ln, lo, lr, r, gid, m = _make_batch(100 + seed, n_groups=4, G=5, T=9)
                loss, st = grpo_loss(ln, lo, lr, r, gid, m)
                el, es = _ref_grpo_loss(ln, lo, lr, r, gid, m)
                assert abs(loss - el) < 1e-8, f"seed={seed}: loss 期望 {el:.8f}，得到 {loss:.8f}"
                assert abs(st["kl"] - es["kl"]) < 1e-8, f"seed={seed}: kl 期望 {es['kl']:.8f}，得到 {st['kl']:.8f}"
                assert abs(st["clip_frac"] - es["clip_frac"]) < 1e-9, f"seed={seed}: clip_frac 期望 {es['clip_frac']}，得到 {st['clip_frac']}"
          `,
        },
        {
          name: '乱序 group_ids 与任意 id',
          hidden: true,
          code: py`
            rng = np.random.default_rng(7)
            ln, lo, lr, r, gid, m = _make_batch(7, n_groups=4, G=4, T=8)
            perm = rng.permutation(len(r))
            ids = np.array([42, 7, 1000, 3])[gid]
            args = (ln[perm], lo[perm], lr[perm], r[perm], ids[perm], m[perm])
            loss, st = grpo_loss(*args)
            el, es = _ref_grpo_loss(*args)
            assert abs(loss - el) < 1e-8, f"乱序分组：loss 期望 {el:.8f}，得到 {loss:.8f}"
            assert np.allclose(st["adv"], es["adv"], atol=1e-8), "乱序分组：优势不一致，检查是否假设了连续分组"
          `,
        },
        {
          name: '聚合方式是 seq-mean-token-mean',
          hidden: true,
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(21, n_groups=3, G=4, T=10)
            loss, _ = grpo_loss(ln, lo, lr, r, gid, m, beta=0.0)
            el, _ = _ref_grpo_loss(ln, lo, lr, r, gid, m, beta=0.0)
            adv = _ref_group_adv(r, gid)[:, None]
            ratio = np.exp(np.where(m > 0, ln - lo, 0))
            tok = -np.minimum(ratio * adv, np.clip(ratio, 0.8, 1.2) * adv) * m
            token_mean = tok.sum() / m.sum()
            assert abs(loss - token_mean) > 1e-6 or abs(el - token_mean) < 1e-6, "你的结果等于 token-mean（DAPO 式）聚合；GRPO 原式是先行内平均再行间平均"
            assert abs(loss - el) < 1e-8, f"期望 {el:.8f}，得到 {loss:.8f}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-std0',
          q: md`如果组大小 $G=1$ 会怎样？verl 是怎么处理的？`,
          a: md`ddof=1 的 std 无定义（除以 0）。verl 的 GRPO 优势函数对单样本组把 mean 设为 0、std 设为 1，即优势退化为原始奖励——此时已不是“组内相对”，等价于无 baseline 的 REINFORCE，方差很大。实际训练应保证 $G\ge2$（常用 8–64）。`,
        },
        {
          id: 'f-process',
          q: md`换成 process supervision（PRM 给每步打分），代码要改哪里？`,
          a: md`优势从 \`(B,)\` 变成 \`(B, T)\`：先把所有步骤奖励（跨组内全部回答）做标准化，再对每个 token 取“它所在步骤及之后所有步骤”的标准化奖励之和——对步骤奖励做**后缀和**后按 token 所属步骤展开。loss 部分不变。`,
        },
      ],
    },
    {
      id: 'c-grpo-grad',
      title: '推导并实现 GRPO 对 log π 的逐 token 梯度',
      difficulty: 'hard',
      source: '面试追问 · clip 后哪些 token 没有梯度',
      timeLimitMin: 20,
      description: md`
        面试官：“你说 clip 会让 token 没梯度，那写出 loss 对每个 \`logp_new[i, t]\` 的梯度。”

        沿用上一题的 loss 定义（同样的参数与约定），实现

        \`\`\`python
        def grpo_grad(logp_new, logp_old, logp_ref, rewards, group_ids, mask,
                      clip_eps=0.2, beta=0.04, adv_eps=1e-6) -> np.ndarray:  # (B, T)
        \`\`\`

        返回 $\partial\,\text{loss}/\partial\log\pi_{new}(o_{i,t})$，padding 位置为 0。提示：$\partial r/\partial\log\pi_{new}=r$；
        $\partial k_3/\partial\log\pi_{new}=1-\pi_{ref}/\pi_{new}$；min 选中**裁剪项**（\`surr2 < surr1\`）时策略项梯度为 0。
        测试会用中心差分数值梯度对拍（随机数据不会恰好落在裁剪边界上）。
      `,
      starter: py`
        import numpy as np

        def grpo_grad(logp_new, logp_old, logp_ref, rewards, group_ids, mask,
                      clip_eps=0.2, beta=0.04, adv_eps=1e-6):
            # 梯度 = 聚合权重 × (策略项梯度 + β × KL 项梯度)
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def grpo_grad(logp_new, logp_old, logp_ref, rewards, group_ids, mask,
                      clip_eps=0.2, beta=0.04, adv_eps=1e-6):
            m = np.asarray(mask, dtype=np.float64)
            r = np.asarray(rewards, dtype=np.float64)
            _, inv = np.unique(group_ids, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv).astype(np.float64)
            dev = r - (np.bincount(inv, weights=r) / cnt)[inv]
            std = np.sqrt(np.bincount(inv, weights=dev ** 2) / (cnt - 1))
            A = (dev / (std[inv] + adv_eps))[:, None]

            ratio = np.exp(np.where(m > 0, logp_new - logp_old, 0.0))
            surr1 = ratio * A
            surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A
            # 策略项：loss_pg = -min(surr1, surr2)；选中未裁剪项时 d/dlogp = -A * r，选中裁剪项（常数）时为 0
            g_pg = np.where(surr2 < surr1, 0.0, -A * ratio)
            # KL 项：k3 = e^d - d - 1，d = logp_ref - logp_new → d k3 / d logp_new = 1 - e^d
            d = np.where(m > 0, logp_ref - logp_new, 0.0)
            g_kl = beta * (1.0 - np.exp(d))
            # 聚合权重：1 / (B * |o_i|)
            w = m / (m.sum(axis=1, keepdims=True) * m.shape[0])
            return (g_pg + g_kl) * w
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：用 autograd 验证手推梯度）',
          code: py`
            import torch

            def check_grad(logp_new, *args):
                x = logp_new.clone().requires_grad_(True)
                loss, _ = grpo_loss_torch(x, *args)      # 上一题的 PyTorch 实现
                loss.backward()
                return x.grad                            # 应与手推的 grpo_grad 一致
          `,
        },
      ],
      explanation: md`
        对有效 token $(i,t)$，记 $w_i=\frac1{B\,|o_i|}$：

        $$
        \frac{\partial\,\text{loss}}{\partial\log\pi_\theta(o_{i,t})}=w_i\Big[-\hat A_i\,r_{i,t}\cdot\mathbb 1[\text{未裁剪项被选中}]+\beta\Big(1-\frac{\pi_{ref}}{\pi_\theta}\Big)\Big]
        $$

        - 在 $r=1$ 处它就是 DeepSeekMath 附录式 (21) 的梯度系数 $\hat A_{i,t}+\beta(\frac{\pi_{ref}}{\pi_\theta}-1)$ 取负、乘聚合权重。
        - **被裁剪的 token 只剩 KL 梯度**；β=0 时彻底没有梯度——这就是 CISPO 批评的“丢 token”。
        - $w_i\propto1/|o_i|$：长回答的每个 token 权重更小，下一关 Dr.GRPO 的出发点。
      `,
      hints: [
        md`链式法则：$\partial r/\partial\log\pi_{new}=r$，所以未裁剪项 $-rA$ 的导数是 $-rA$。`,
        md`裁剪项 $-\mathrm{clip}(r)A$ 在越界时是常数，导数 0；用 \`np.where(surr2 < surr1, 0, -A * ratio)\`。`,
        md`别忘了乘聚合权重 \`mask / (len_i * B)\`，以及 KL 项 \`beta * (1 - exp(logp_ref - logp_new))\`。`,
      ],
      setup: SETUP,
      tests: [
        {
          name: '形状与 padding 为 0',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(3)
            g = np.asarray(grpo_grad(ln, lo, lr, r, gid, m))
            assert g.shape == ln.shape, f"形状应为 {ln.shape}，得到 {g.shape}"
            assert np.all(g[m == 0] == 0), "padding 位置的梯度必须为 0"
          `,
        },
        {
          name: 'on-policy、β=0：梯度 = −Â_i / (B·|o_i|)',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(4, garbage=False)
            g = np.asarray(grpo_grad(lo, lo, lo, r, gid, m, beta=0.0))
            adv = _ref_group_adv(r, gid)
            exp = -adv[:, None] * m / (m.sum(1, keepdims=True) * len(r))
            assert np.allclose(g, exp, atol=1e-12), f"最大误差 {np.abs(g - exp).max():.3e}"
          `,
        },
        {
          name: '数值梯度对拍',
          hidden: true,
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(5, n_groups=2, G=4, T=6)
            g = np.asarray(grpo_grad(ln, lo, lr, r, gid, m))
            h = 1e-6
            num = np.zeros_like(ln)
            for i, t in zip(*np.nonzero(m)):
                p = ln.copy(); p[i, t] += h
                q = ln.copy(); q[i, t] -= h
                num[i, t] = (_ref_grpo_loss(p, lo, lr, r, gid, m)[0] - _ref_grpo_loss(q, lo, lr, r, gid, m)[0]) / (2 * h)
            err = np.abs(num - g).max()
            assert err < 1e-6, f"与数值梯度最大误差 {err:.3e}"
          `,
        },
        {
          name: '随机对拍（含裁剪 token）',
          hidden: true,
          code: py`
            for seed in range(20):
                ln, lo, lr, r, gid, m = _make_batch(300 + seed, n_groups=3, G=4, T=8)
                ln = lo + 2.5 * (ln - lo)          # 放大偏移，制造大量越界 token
                g = np.asarray(grpo_grad(ln, lo, lr, r, gid, m, beta=0.1))
                e = _ref_grpo_grad(ln, lo, lr, r, gid, m, beta=0.1)
                assert np.allclose(g, e, atol=1e-10), f"seed={seed}: 最大误差 {np.abs(g - e).max():.3e}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-clip-neg',
          q: md`A<0 且 $r$ 很大（坏回答的 token 概率被意外抬高很多）时，这个 token 有梯度吗？有什么风险？`,
          a: md`有：min 选中未裁剪项 $rA$，梯度 $-Ar$ 随 $r$ 线性放大、**没有上界**。极端 off-policy 样本会产生巨大梯度，这正是 Dual-clip PPO（对 A<0 再加下界 $c\cdot A$）以及 DeepSeek-V3.2“只屏蔽负优势的高偏离序列”的动机。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-grpo-vs-ppo',
      freq: 3,
      q: md`GRPO 相比 PPO 改了什么？为什么特别适合 LLM 推理任务？`,
      a: md`
        **结论**：去掉价值模型，用同一问题 G 个回答的组内标准化奖励当优势；KL 从奖励挪进 loss。

        **原因**：① 价值模型与策略同规模，显存与算力翻倍；② 推理任务奖励只在序列末尾、且是 0/1，逐 token 价值函数难学且偏差大；
        ③ 组内比较与“同一道题哪个解法更好”的结构天然吻合，可验证奖励也让组内比较没有 RM 噪声。

        **延伸**：代价是每题要采 G 个样本（rollout 成本上升）、全对/全错组没有信号；组内 std 归一化与 $1/|o_i|$ 引入了偏差（Dr.GRPO）；token 级 IS 在长序列和 MoE 上不稳（GSPO）。R1 论文也报告：PPO 把 GAE 的 λ 调到 1.0 后能接近 GRPO，但调参成本高。
      `,
    },
    {
      id: 'qa-kl',
      freq: 3,
      q: md`GRPO 的 KL 项为什么用 k3？放在 loss 里和放在 reward 里有什么区别？现在还要不要 KL？`,
      a: md`
        **结论**：k3 $=x-1-\log x$（$x=\pi_{ref}/\pi_\theta$）对每个样本非负、且无偏；放进 loss 让优势计算与 KL 解耦。

        **原因**：放 reward 里（PPO）KL 会进入累计回报，越长的回答累计惩罚越多，抑制长 CoT；放 loss 里只是逐 token 的正则。

        **延伸**：严格说，k3 当 loss 求导的期望梯度是**正向** KL(π_ref‖π_θ) 的梯度；k2 当 loss 才对应反向 KL 的梯度（REINFORCE++ 的选择）。
        DeepSeek-V3.2 给 k3 乘 $\pi_\theta/\pi_{old}$ 做 IS 修正。长 CoT 训练中策略本就该远离初始模型，所以 DAPO 等直接 β=0，R1 则周期性把 π_ref 重置为当前策略。
      `,
    },
    {
      id: 'qa-zero',
      freq: 2,
      q: md`训练中发现 loss 一直在 0 附近、但 reward 在涨，正常吗？`,
      a: md`
        **结论**：正常。

        **原因**：组内优势和为 0；on-policy 时 $r=1$，按 GRPO 原式聚合 loss 恰好是 $-\frac1B\sum_i\hat A_i=0$（多 mini-batch 时也只是在 0 附近波动）。
        梯度却不为 0：每条回答按 $\hat A_i/|o_i|$ 加权推/压自己的 token。

        **延伸**：该看的监控是 reward 均值、回答长度、熵、clip_frac、KL、全对/全错组占比，而不是 loss 数值本身。
      `,
    },
    {
      id: 'qa-rlvr',
      freq: 2,
      q: md`RLVR 的可验证奖励有哪些？格式奖励有什么用、有什么风险？`,
      a: md`
        **结论**：数学答案等价判断、代码单测、格式检查（以及逻辑谜题的规则判定）。

        **原因**：程序判定没有 RM 的系统性偏差，难以被“讨好”；格式奖励让模型学会把推理放进 \`<think>\`、答案放在可抽取的位置，否则准确率奖励都抽不到答案。

        **延伸**：风险是**验证器漏洞即 reward hacking**：抽取过松（多个 \\boxed）、测试覆盖不足（硬编码输出）、格式分权重过高（学会空壳）。
        R1-Zero 用准确率 + 格式等权；R1 加语言一致性奖励，代价是性能略降但可读性更好。
      `,
    },
  ],

  cheatsheet: md`
    **GRPO（DeepSeekMath 式 3）**
    $$\mathcal J=\tfrac1G\sum_i\tfrac1{|o_i|}\sum_t\big\{\min[r_{i,t}\hat A_i,\mathrm{clip}(r_{i,t},1\pm\varepsilon)\hat A_i]-\beta\,k_3\big\}$$
    - $\hat A_i=(r_i-\text{mean})/\text{std}$（组内；本章 std 用 ddof=1，+1e-6）
    - $k_3=\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1\ \ge0$，直接进 loss
    - 梯度系数（r=1）：$\hat A_{i,t}+\beta(\pi_{ref}/\pi_\theta-1)$

    \`\`\`python
    A = (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-6)
    ratio = np.exp(lp_new - lp_old)
    surr = np.minimum(ratio * A, np.clip(ratio, 1 - eps, 1 + eps) * A)
    d = lp_ref - lp_new;  kl = np.exp(d) - d - 1
    tok = -(surr - beta * kl)
    loss = ((tok * mask).sum(1) / mask.sum(1)).mean()
    \`\`\`

    | | DeepSeekMath | R1-Zero |
    |---|---|---|
    | lr / β / G | 1e-6 / 0.04 / 64 | 3e-6 / 0.001 / 16 |
    | 长度 | 1024 | 32K → 65K |

    - 全对/全错组优势 = 0 → 无梯度；on-policy 时 loss 数值 ≈ 0 但梯度不为 0
    - clip 只在同批数据多步更新时生效；越界 token 梯度为 0
    - k3 当 loss 的期望梯度 = ∇KL(π_ref‖π_θ)（正向）；k2 才对应反向 KL
  `,
})
