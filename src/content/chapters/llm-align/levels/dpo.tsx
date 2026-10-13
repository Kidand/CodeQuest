/**
 * DPO：从 KL 约束奖励最大化的闭式解推导到 DPO loss；隐式奖励、β、序列 log-prob 的 mask 实现。
 * 事实来源：Rafailov et al. 2023（arXiv 2305.18290，Eq.3–7 与附录 B）、Llama 3（2407.21783 §4.1.4）、
 * Tülu 3（2411.15124 §5、Table 20）、Xu et al. 2024（2404.10719）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { ClosedFormViz } from '../viz/ClosedFormViz'
import { DerivationViz } from '../viz/DerivationViz'
import { DpoLossViz } from '../viz/DpoLossViz'
import { SeqLogpViz } from '../viz/SeqLogpViz'
import { PY_MAKE_PAIRS, PY_REF_LOGPS } from '../shared'

export default defineLevel({
  learn: [
    {
      id: 'objective',
      title: '出发点：带 KL 约束的奖励最大化',
      body: md`
        经典 RLHF 分三步：SFT → 训练奖励模型 $r_\phi$ → 用 PPO 优化下面这个目标（DPO 论文 Eq.3）：

        $$
        \max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal D,\ y\sim\pi_\theta(\cdot|x)}\big[r_\phi(x,y)\big]\;-\;\beta\,\mathbb{D}_{\mathrm{KL}}\big[\pi_\theta(y|x)\,\|\,\pi_{\mathrm{ref}}(y|x)\big]
        $$

        KL 项不是装饰：奖励模型只在 SFT 分布附近可信，策略一旦跑远就会找到 RM 的漏洞（reward hacking），
        还会丢掉语言流畅性和多样性。PPO 这一步要同时放 actor、critic、RM、reference **四个模型**，还要在线采样，又贵又不稳。

        DPO 的出发点是一个观察：**这个目标对任意奖励函数都有闭式最优解**

        $$
        \pi^*(y|x)=\frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y|x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big),\qquad Z(x)=\sum_y \pi_{\mathrm{ref}}(y|x)\exp\!\Big(\frac{r(x,y)}{\beta}\Big)
        $$

        也就是把 $\pi_{\mathrm{ref}}$ 按 $\exp(r/\beta)$ **重新加权**。下面用 6 个候选回答感受 β 的作用。
      `,
      viz: ClosedFormViz,
      after: md`
        > [!TIP] 两个可以背下来的结论
        > 1. 最优目标值恰好等于 $\beta\log Z(x)$（右下角的卡片始终与 J 相等）：把目标改写成 $-\beta\,\mathbb{D}_{\mathrm{KL}}[\pi\|\pi^*]+\beta\log Z(x)$ 就能看出来。
        > 2. β→0 时 $\pi^*$ 退化成“只选奖励最高的回答”；β→∞ 时 $\pi^*\to\pi_{\mathrm{ref}}$。β 越大，越不敢离开参考模型。

        这个闭式解不能直接用：$Z(x)$ 要对**所有可能的回答序列**求和，词表 10 万、长度 1000 的序列空间根本数不完。DPO 的巧妙之处就是让 $Z(x)$ 自己消失。
      `,
      checkpoint: {
        id: 'cp-closed',
        type: 'single',
        prompt: md`在闭式解 $\pi^*\propto\pi_{\mathrm{ref}}\exp(r/\beta)$ 中，若某个回答在 $\pi_{\mathrm{ref}}$ 下概率为 0，那么无论它的奖励多高，它在 $\pi^*$ 中的概率是：`,
        options: ['仍为 0', '与奖励成正比', '当 β 足够小时变为 1', '取决于 Z(x)'],
        answer: 0,
        explanation: md`$\pi^*$ 是 $\pi_{\mathrm{ref}}$ 乘一个有限的正因子再归一化，$0\times$ 任何有限数仍为 0。这也说明 KL 约束的含义：**最优策略的支撑集不会超出参考模型**——DPO/RLHF 只能在 SFT 模型“会说的话”里重新分配概率，无法凭空创造新能力。选项 C 混淆了“集中到奖励最高者”与“支撑集外的回答”。`,
      },
    },
    {
      id: 'derive',
      title: '推导：Z(x) 为什么能消掉',
      body: md`
        核心是**换元**：不再“先有奖励、再求策略”，而是把闭式解取对数反解出奖励（DPO 论文 Eq.5）：

        $$
        r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)}+\beta\log Z(x)
        $$

        人类偏好用 **Bradley-Terry** 模型描述：$p(y_w\succ y_l|x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$，它**只依赖奖励之差**。
        把上式代入，两条回答共享同一个 prompt $x$，$\beta\log Z(x)$ 一加一减正好抵消。再把 $\pi^*$ 换成待训练的 $\pi_\theta$，对偏好数据做最大似然，就得到 DPO 损失（Eq.7）：

        $$
        \mathcal L_{\mathrm{DPO}}=-\,\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Big)\Big]
        $$

        逐帧看一遍完整推导，注意第 ⑥ 步。
      `,
      viz: DerivationViz,
      after: md`
        > [!WARN] 消掉 Z(x) 的前提
        > - **同一个 prompt**：$Z(x)$ 只依赖 $x$。两条回答若来自不同 prompt，$\log Z(x_1)-\log Z(x_2)$ 消不掉——所以 DPO 必须用**成对**数据；只有“好/坏”单条标注时要用 KTO 一类方法（见「DPO 变体」）。
        > - **Bradley-Terry 假设**：偏好只由奖励差决定。BT 换成 Plackett-Luce（K 个回答的完整排序）推导同样成立；IPO 则干脆不假设 BT。

        论文标题 *Your Language Model is Secretly a Reward Model* 就来自这一步：$\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$ 被称为**隐式奖励**，
        DPO 本质上是在用这个参数化训练一个 BT 奖励模型，而对应的最优策略就是 $\pi_\theta$ 本身。
      `,
      checkpoint: {
        id: 'cp-derive',
        type: 'fill',
        prompt: md`DPO 推导中，配分函数 $Z(x)$ 能在损失中消掉，是因为 Bradley-Terry 模型只依赖两条回答的奖励之 ____(1)____，且 $Z$ 只依赖于 ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['差', '差值', 'difference'] },
          { label: '(2)', answers: ['x', 'prompt', '提示', '输入', '提示词', 'prompt x'] },
        ],
        explanation: md`$r(x,y_w)-r(x,y_l)$ 中两个 $\beta\log Z(x)$ 相同而相减为 0。若偏好模型依赖奖励的绝对值，或两条回答对应不同 prompt，$Z$ 就消不掉。`,
      },
    },
    {
      id: 'gradient',
      title: '梯度：σ(−βΔ) 是“纠错权重”',
      body: md`
        记 log-ratio margin $\Delta=\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}$，单条损失 $\ell=-\log\sigma(\beta\Delta)$。求导（DPO 论文 §4）：

        $$
        \nabla_\theta\mathcal L_{\mathrm{DPO}}=-\beta\,\mathbb E\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{=\ \sigma(-\beta\Delta)}\big(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\big)\Big]
        $$

        - 方向：拉高 $y_w$ 的似然、压低 $y_l$ 的似然；
        - 权重：**隐式奖励模型排错得越离谱，权重越接近 1**；已经排对的样本权重趋于 0（但永不为 0）。论文指出去掉这个权重的朴素版本会让模型退化。

        **β 的双重身份**：在 RLHF 目标里是 KL 系数；在 DPO loss 里是隐式奖励的尺度。拖动下图的 β 和 Δ 观察 loss 与权重。
      `,
      viz: DpoLossViz,
      after: md`
        > [!INTERVIEW] 一个常被追问的 sanity check
        > 训练第一步 $\pi_\theta=\pi_{\mathrm{ref}}$，所有样本 $\Delta=0$，**loss 恰好等于 $\ln 2\approx0.693$，与 β 无关**，两条隐式奖励都为 0（TRL 的 rewards/accuracies 用严格大于统计，此时记为 0）。
        > 如果你的第一步 loss 不是 0.693，多半是 reference 没加载对、dropout 没关（TRL 默认 \`disable_dropout=True\`），或者 policy 与 reference 的 padding/截断不一致。

        β 小：loss 曲线更平，要把 Δ 推得很大（离 π_ref 很远）才能把 loss 降下来，约束弱；β 大：小幅改动 Δ 就能让 loss 饱和、权重归零，模型被牢牢拴在 π_ref 附近。
      `,
      checkpoint: {
        id: 'cp-grad',
        type: 'numeric',
        prompt: md`DPO 训练开始时 policy 与 reference 完全相同（β = 0.1）。此时一个 batch 的平均 DPO loss 是多少？（保留 3 位小数）`,
        answer: 0.693,
        tolerance: 0.005,
        explanation: md`$\Delta=0\Rightarrow\ell=-\log\sigma(0)=-\log\tfrac12=\ln2\approx0.693$，与 β 无关。这是检查 DPO 实现的第一个断言。`,
      },
    },
    {
      id: 'logp',
      title: '实现：用 mask 求序列 log-prob',
      body: md`
        $\log\pi_\theta(y|x)=\sum_{t\in\text{回复}}\log\pi_\theta(y_t|x,y_{<t})$，实现上和 SFT 共用一套“labels + -100”约定（TRL 的 \`concatenated_forward\`）：

        1. chosen 与 rejected 拼成一个 $2B$ 的 batch 做一次前向（第 $b$ 行与第 $B+b$ 行是一对）；
        2. **shift**：\`logits[:, t]\` 预测的是第 $t+1$ 个 token，所以用 \`logits[:, :-1]\` 对齐 \`labels[:, 1:]\`；
        3. prompt 与 padding 的 label 为 -100，\`log_softmax\` 后按 label gather，乘 mask **求和**（原版 DPO 是求和，不是平均）；
        4. reference 在 \`no_grad\` 下算同样的量，也可以训练前一次性离线算好（TRL 的 \`precompute_ref_log_probs\`），训练时显存里只剩 policy。

        打开下图的“错误开关”，看看各种漏 mask 的后果。
      `,
      viz: SeqLogpViz,
      after: md`
        > [!WARN] 两个容易混淆的结论
        > - **prompt 算进去**：因果模型下 chosen/rejected 的 prompt 前缀逐 token 相同，求和后在 Δ 中**精确抵消**，loss 不变；但 rewards/chosen、rewards/rejected 等监控指标会被整体平移，且一旦改成长度归一化（SimPO、Tülu 3 的 length-normalized DPO），就不再抵消。前提是两边 prompt 的 tokenization 完全一致。
        > - **padding 算进去**：两条回答长度不同、pad 数不同，垃圾 logp 直接污染 Δ——这是真 bug。

        > [!NOTE] 显存大户是 logits
        > $2B\times T\times V$ 的 fp32 logits 很大：16 条 × 2048 token × 128,256 词表（Llama 3）≈ 16.8 GB。工程上会分块计算 log-softmax、只 gather 需要的位置，或用融合 kernel（如 Liger Kernel）避免物化完整 logits。
      `,
      checkpoint: {
        id: 'cp-logp',
        type: 'judge',
        prompt: md`在原版 DPO（序列 logp 求和）中，若误把 prompt token 的 log-prob 也累加进序列 logp，且 chosen 与 rejected 的 prompt tokenization 完全相同，则 DPO loss 的数值会改变。`,
        answer: false,
        explanation: md`错误。prompt 部分对 chosen 和 rejected 贡献相同的 $\sum\log\pi_\theta-\sum\log\pi_{\mathrm{ref}}$，在 $\Delta$ 中相减为 0，loss 与梯度都不变；变化的只是单条隐式奖励（监控指标）。长度归一化的变体则会受影响。`,
      },
    },
    {
      id: 'practice',
      title: '工程实践：β、chosen 概率下降与真实配方',
      body: md`
        | 场景 | 配置（均来自原文） |
        |---|---|
        | DPO 原论文 | 默认 β = 0.1、batch 64、RMSprop、lr 1e-6（150 步线性 warmup）；TL;DR 摘要用 β = 0.5 |
        | Llama 3 | β = 0.1、lr 1e-5；**mask 掉格式 token**（header、终止符）；chosen 上加系数 **0.2** 的 NLL 损失；六轮迭代，每轮用上一轮最好模型采样的新偏好数据 |
        | Tülu 3 | 消融后选用 **length-normalized DPO**（logp 除以长度），β = 5；lr 5e-7（8B）/ 2e-7（70B），batch 128，1 epoch |

        **chosen 的 logp 往往也在下降。** DPO 只约束差值 Δ，常见现象是 chosen 与 rejected 的 logp 一起降、rejected 降得更快——概率被挪到了数据之外的回答上（文献称 likelihood displacement）。
        Llama 3 加 NLL 项，以及它引用的 Smaug（DPOP，Pal et al. 2024）、IRPO（Pang et al. 2024），都是在对抗它。Llama 3 给出的屏蔽格式 token 的理由是：两边都有的公共 token 会被同时要求升高和降低，产生冲突，导致结尾重复或突然输出终止符。

        **DPO 是离线算法。** 数据来自别的模型（或旧版本自己），训练中不采样；Xu et al. 2024（*Is DPO Superior to PPO for LLM Alignment?*）证明 PPO 能达到的策略集合是 DPO 的真子集——DPO 可能收敛到利用分布外回答的有偏解；实验上调好的 PPO 在从对话到代码竞赛的所有测试中都超过了其它对齐方法。工业界的折中是 **iterative / online DPO**：每轮用当前模型采样、打分、再训 DPO（见「DPO 变体」）。
      `,
      after: md`
        > [!NEW] 推理模型时代
        > 2025 年后，数学 / 代码等可验证任务的后训练主线转向 RLVR + GRPO 系算法（见「推理强化学习」章节）；DPO 及其变体仍广泛用于通用对话的偏好对齐、风格与安全性调优，以及作为 RL 之前的廉价热身。
      `,
      checkpoint: {
        id: 'cp-practice',
        type: 'multi',
        prompt: md`Llama 3 技术报告中对 DPO 做了哪些修改？（多选）`,
        options: ['在 chosen 上加系数 0.2 的 NLL 损失', '从 chosen/rejected 的 loss 中 mask 掉 header 与终止符等格式 token', '去掉 reference model', '把序列 logp 改成长度平均', '多轮迭代，每轮使用最新模型生成的偏好数据'],
        answer: [0, 1, 4],
        explanation: md`Llama 3（§4.1.4、§4.1.6）：β = 0.1、lr 1e-5、mask 格式 token、0.2 × NLL 正则，并沿用 Llama 2 的多轮迭代（共六轮）。去掉 reference 是 SimPO / ORPO 的做法；长度平均是 Tülu 3 选用的 length-normalized DPO。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-models',
      type: 'single',
      difficulty: 1,
      prompt: md`标准 DPO 训练时，需要哪些模型参与计算？`,
      options: [
        'policy 与冻结的 reference（reference 的 logp 也可预先离线算好）',
        'policy、reference、reward model 三个',
        'actor、critic、reward model、reference 四个',
        '只需要 policy',
      ],
      answer: 0,
      explanation: md`DPO 用隐式奖励 $\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ 取代显式 RM，也不需要 critic；但 reference 必须有（$\pi_{\mathrm{ref}}$ 出现在 loss 里），只是可以预计算。C 是 PPO-RLHF；D 是 SimPO / ORPO 这类无 reference 的变体。`,
    },
    {
      id: 'q-nodata',
      type: 'judge',
      difficulty: 1,
      prompt: md`DPO 不需要训练奖励模型，因此也不需要人工偏好数据。`,
      answer: false,
      explanation: md`DPO 省掉的是**显式 RM 的训练和 RL 采样**，而不是偏好数据本身。它直接在 $(x, y_w, y_l)$ 三元组上做 BT 最大似然——偏好数据是它唯一的监督信号。`,
    },
    {
      id: 'q-fill-z',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        KL 约束奖励最大化的闭式解为 $\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\exp(r(x,y)/\beta)$。
        反解得 $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)}+$ ____(1)____ 。
        最优目标值 $\max_\pi J(\pi)=$ ____(2)____ 。
      `,
      blanks: [
        { label: '(1)', answers: ['βlogZ(x)', 'betalogZ(x)', 'β*logZ(x)', 'beta*log(Z(x))', 'βlog(Z(x))', 'β·logZ(x)', 'beta*logZ(x)'] },
        { label: '(2)', answers: ['βlogZ(x)', 'betalogZ(x)', 'β*logZ(x)', 'beta*log(Z(x))', 'βlog(Z(x))', 'β·logZ(x)', 'beta*logZ(x)'] },
      ],
      explanation: md`(1) 对闭式解取对数：$\log\pi^*=\log\pi_{\mathrm{ref}}+r/\beta-\log Z$，移项即得。(2) 目标可改写为 $-\beta\,\mathbb D_{\mathrm{KL}}[\pi\|\pi^*]+\beta\log Z(x)$，KL 在 $\pi=\pi^*$ 时为 0，所以最优值就是 $\beta\log Z(x)$。`,
    },
    {
      id: 'q-loss',
      type: 'numeric',
      difficulty: 2,
      prompt: md`
        一对偏好样本（β = 0.1）：policy 下 $\log\pi_\theta(y_w)=-12$、$\log\pi_\theta(y_l)=-15$；
        reference 下 $\log\pi_{\mathrm{ref}}(y_w)=-13$、$\log\pi_{\mathrm{ref}}(y_l)=-14$。求这对样本的 DPO loss（自然对数，保留 3 位小数）。
      `,
      answer: 0.598,
      tolerance: 0.005,
      explanation: md`
        $\Delta=(-12-(-15))-(-13-(-14))=3-1=2$，$\beta\Delta=0.2$。
        $\ell=-\log\sigma(0.2)=\log(1+e^{-0.2})=\log 1.8187\approx0.598$。
        隐式奖励：chosen $=0.1\times(-12+13)=0.1$，rejected $=0.1\times(-15+14)=-0.1$，margin 0.2。
        常见错误：把 reference 的差加上而不是减去，得到 $\Delta=4$。
      `,
    },
    {
      id: 'q-beta',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 DPO 中的 β，下列说法正确的有：`,
      options: [
        'β 越大，对应的最优策略越接近 π_ref',
        '固定 Δ > 0 时，β 越大 loss 越小',
        'β 只缩放 loss 的数值，不影响训练结果',
        '训练刚开始（π_θ = π_ref）时 loss 都是 ln2，与 β 无关',
        'β 越小，Δ = 0 处的梯度系数 β·σ(−βΔ) 越大',
      ],
      answer: [0, 1, 3],
      explanation: md`
        A：$\pi^*\propto\pi_{\mathrm{ref}}e^{r/\beta}$，β 大则重加权弱。B：$-\log\sigma(\beta\Delta)$ 对 $\beta\Delta$ 单调递减。D：$\Delta=0\Rightarrow\ln2$。
        C 错：β 改变 loss 的形状（饱和速度），等价于改变 KL 约束强度。E 错：$\Delta=0$ 处梯度系数为 $\beta/2$，β 越小越小。
      `,
    },
    {
      id: 'q-prompt',
      type: 'single',
      difficulty: 3,
      prompt: md`某实现误把 prompt token 的 logp 也累加进序列 logp（prompt 在 chosen/rejected 中 tokenization 完全相同）。对**原版 DPO** 与 **length-normalized DPO**（logp 除以计入的 token 数）分别有什么影响？`,
      options: [
        '原版：loss 不变，只有 rewards/chosen 等指标被平移；length-normalized：loss 会改变',
        '两者的 loss 都不变',
        '两者的 loss 都会改变',
        '原版：loss 改变；length-normalized：不变',
      ],
      answer: 0,
      explanation: md`
        原版 DPO：prompt 部分给 chosen、rejected 各加同一个常数 $c=\sum_{\text{prompt}}(\log\pi_\theta-\log\pi_{\mathrm{ref}})$，在 Δ 中抵消，loss、梯度都不变。
        length-normalized：chosen 的奖励变为 $(S_w+c)/(n_w+p)$，rejected 为 $(S_l+c)/(n_l+p)$，两者分母不同（$n_w\ne n_l$ 时），$c$ 不再抵消，且长回答被 prompt 长度稀释——loss 和梯度都会变。
      `,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 DPO 的推导步骤按顺序排列：`,
      items: [
        '写出 KL 约束的奖励最大化目标',
        '改写目标并归一化，得到闭式解 π* ∝ π_ref·exp(r/β)',
        '对闭式解取对数，用策略反解奖励 r = β·log(π*/π_ref) + β·log Z(x)',
        '代入 Bradley-Terry 模型，同一 prompt 下 β·log Z(x) 相消',
        '用 π_θ 替换 π*，对偏好数据做最大似然得到 DPO loss',
      ],
      explanation: md`闭式解 → 反解奖励 → 代入只依赖奖励差的偏好模型 → Z 相消 → 参数化做 MLE。顺序错在第 2、3 步的人最多：必须先有闭式解，才能“用策略表示奖励”。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把 DPO 公式中的量与其含义配对：`,
      pairs: [
        [md`$\pi_{\mathrm{ref}}$`, '冻结的 SFT 模型，KL 约束的锚点'],
        [md`$\beta$`, 'KL 强度，同时是隐式奖励的尺度'],
        [md`$Z(x)$`, '配分函数，只依赖 prompt，在 loss 中相消'],
        [md`$\sigma(\hat r_\theta(y_l)-\hat r_\theta(y_w))$`, '梯度权重，排错越多权重越大'],
      ],
      explanation: md`四个量分别对应：参考策略、温度/KL 系数、归一化常数、梯度中的纠错权重。$\hat r_\theta=\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ 即隐式奖励。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`
        下面的 DPO 实现有 **2 处** bug。约定：batch 前 B 行是 chosen、后 B 行是 rejected；\`labels\` 与 input_ids 对齐（未移位），\`-100\` 不计入；\`log_softmax\` 已正确实现。点出有问题的行。
      `,
      code: py`
        def dpo_loss(pi_logits, ref_logits, labels, beta=0.1):
            B = labels.shape[0] // 2
            def seq_logp(logits):
                lp = log_softmax(logits[:, :-1], axis=-1)
                lab = labels[:, :-1]
                mask = lab != -100
                tok = np.take_along_axis(lp, np.where(mask, lab, 0)[..., None], -1)[..., 0]
                return (tok * mask).sum(-1)
            pi, ref = seq_logp(pi_logits), seq_logp(ref_logits)
            pi_lr = pi[:B] - pi[B:]
            ref_lr = ref[:B] - ref[B:]
            z = beta * (pi_lr + ref_lr)
            return np.logaddexp(0.0, -z).mean()
      `,
      bugLines: [5, 12],
      explanation: md`
        第 5 行：\`logits[:, t]\` 预测第 $t+1$ 个 token，labels 应取 \`labels[:, 1:]\`，写成 \`[:, :-1]\` 就是拿位置 $t$ 的预测去对齐第 $t$ 个 token，整个 logp 都错位了。
        第 12 行：应为 \`pi_lr - ref_lr\`——reference 的 log-ratio 是被**减去**的基线。
        第 13 行 \`np.logaddexp(0, -z)\` 正是 $-\log\sigma(z)$ 的稳定写法，没有问题。
      `,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`由四个序列 logp 拼出数值稳定的 DPO loss（含隐式奖励）。有 2 行干扰项。`,
      lines: [
        'def dpo_from_logps(pi_c, pi_r, ref_c, ref_r, beta=0.1):',
        '    pi_logratios = pi_c - pi_r',
        '    ref_logratios = ref_c - ref_r',
        '    logits = beta * (pi_logratios - ref_logratios)',
        '    losses = np.logaddexp(0.0, -logits)   # = -log σ(logits)',
        '    chosen_rewards = beta * (pi_c - ref_c)',
        '    rejected_rewards = beta * (pi_r - ref_r)',
        '    return losses.mean(), chosen_rewards, rejected_rewards',
      ],
      distractors: ['    logits = beta * (pi_logratios + ref_logratios)', '    losses = np.logaddexp(0.0, logits)'],
      explanation: md`$-\log\sigma(z)=\log(1+e^{-z})=$ \`logaddexp(0, -z)\`，对很大的 $|z|$ 也不会溢出。干扰项一个把 reference 的符号写反，一个把 $-z$ 写成 $z$（变成了 $-\log\sigma(-z)$，会鼓励 rejected）。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 1,
      prompt: md`一条右侧 padding 的 chosen 序列如下。计算 $\log\pi_\theta(y_w|x)$ 时，哪些 token 的 log-prob 应被累加？（点出全部）`,
      items: ['<user>', '1+1=?', '<asst>', '答案', '是', '2', '<eos>', '<pad>', '<pad>'],
      answer: [3, 4, 5, 6],
      explanation: md`只累加回复部分，**包括 \`<eos>\`**（模型必须学会何时结束）。prompt 和 padding 的 label 都是 -100。注意 shift：“答案”这个 token 的 logp 来自 \`<asst>\` 位置的 logits。`,
    },
    {
      id: 'q-displace',
      type: 'judge',
      difficulty: 2,
      prompt: md`DPO 训练过程中如果 rewards/chosen（chosen 的隐式奖励）持续下降，说明实现一定有 bug。`,
      answer: false,
      explanation: md`DPO 只优化差值 Δ，chosen 与 rejected 的 logp 同时下降、rejected 降得更快是很常见的现象（likelihood displacement）。它不一定是 bug，但可能有害（概率流向数据外的回答），所以 Llama 3 在 chosen 上加了 0.2 × NLL 正则来抑制。真正该报警的是 margin 不涨、accuracy 不涨，或第一步 loss ≠ ln2。`,
    },
    {
      id: 'q-mem',
      type: 'numeric',
      difficulty: 3,
      unit: 'GB',
      prompt: md`
        DPO 一个 micro-batch 有 8 对偏好（chosen + rejected 共 16 条序列），每条长度 2048，词表 128,256（Llama 3）。
        若把完整 logits 以 fp32 物化，**仅 policy 的 logits 张量**占多少 GB？（1 GB = $10^9$ 字节）
      `,
      answer: 16.81,
      tolerance: 0.01,
      explanation: md`
        $16\times2048\times128256\times4\ \text{B}=16{,}810{,}770{,}432\ \text{B}\approx16.8\ \text{GB}$。
        这还没算 log_softmax 的同尺寸中间结果和 reference 的 logits。工程上用分块 log-softmax、只 gather label 位置，或 Liger 这类融合 kernel 避免物化。
      `,
    },
    {
      id: 'q-offline',
      type: 'single',
      difficulty: 2,
      prompt: md`与 PPO-RLHF 相比，标准 DPO 的主要局限是：`,
      options: [
        '训练数据是固定的离线偏好对，模型在训练中不采样，对分布外回答缺少约束',
        '需要训练一个比 policy 更大的奖励模型',
        '无法使用 LoRA',
        'loss 不可导，只能用强化学习优化',
      ],
      answer: 0,
      explanation: md`DPO 是离线的：数据分布与当前策略不一致时，隐式奖励只在数据覆盖到的回答上被约束（Xu et al. 2024 的核心论点）。解决办法是 iterative / online DPO：每轮用当前模型采样、打分、构造新偏好对。B、D 与 DPO 无关；DPO 完全兼容 LoRA（此时 reference 可以直接用关闭 adapter 的基座）。`,
    },
  ],

  guided: [
    {
      id: 'g-seq-logps',
      title: '一步步写出带 mask 的序列 log-prob',
      intro: md`
        实现 \`sequence_logps(logits, labels)\`：返回每条序列**回复部分**的 $\sum_t\log\pi(y_t|\cdot)$，形状 \`(N,)\`。

        - \`logits\`：\`(N, T, V)\`，\`logits[:, t]\` 是读完前 $t+1$ 个 token 后对**下一个** token 的预测；
        - \`labels\`：\`(N, T)\`，与 input_ids 对齐（**未移位**），prompt 与 padding 处为 \`-100\`。

        这是 DPO、SimPO、PPO 求 logp 的公共积木。
      `,
      template: py`
        import numpy as np

        def sequence_logps(logits, labels, ignore_index=-100):
            logits = {{1}}
            labels = {{2}}
            mask = {{3}}
            safe = np.where(mask, labels, 0)          # -100 不能当下标
            m = logits.max(axis=-1, keepdims=True)
            logp = {{4}}
            tok = np.take_along_axis(logp, safe[..., None], axis=-1)[..., 0]
            return {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`shift：最后一个位置的 logits 预测的是序列之外的 token，丢掉。`,
          answer: 'logits[:, :-1, :]',
          accept: ['logits[:, :-1]'],
          explain: md`位置 $t$ 的 logits 预测 token $t+1$，所以有效的预测只有 $T-1$ 个。`,
        },
        {
          blank: 2,
          instruction: md`labels 去掉第一个（它没有前文，不被任何位置预测）。`,
          answer: 'labels[:, 1:]',
          explain: md`对齐后 \`logits[:, t]\` ↔ \`labels[:, t+1]\`。HF 的 \`ForCausalLMLoss\` 内部做的就是这一步，所以喂给模型的 labels 不需要你提前移位。`,
        },
        {
          blank: 3,
          instruction: md`计入的位置：label 不等于 ignore_index。`,
          answer: 'labels != ignore_index',
          accept: ['labels != -100', '(labels != ignore_index)'],
          explain: md`mask 定义在**被预测的 token** 上：label 为 -100 的 token 不贡献 logp。`,
        },
        {
          blank: 4,
          instruction: md`数值稳定的 log_softmax（已减去行最大值 m）。`,
          answer: 'logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))',
          accept: [
            '(logits - m) - np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))',
            'logits - m - np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))',
          ],
          explain: md`$\log\mathrm{softmax}(z)_i=z_i-m-\log\sum_j e^{z_j-m}$。不减 $m$ 时 logits 达到几百就会 \`exp\` 溢出成 inf。`,
        },
        {
          blank: 5,
          instruction: md`只累加 mask 内的 token，按序列求和。`,
          answer: '(tok * mask).sum(axis=-1)',
          accept: ['(tok * mask).sum(-1)', 'np.where(mask, tok, 0.0).sum(axis=-1)', 'np.where(mask, tok, 0).sum(-1)'],
          explain: md`被 mask 的位置 gather 到的是下标 0 的垃圾 logp，必须乘 mask 清零。原版 DPO 用**求和**；SimPO / length-normalized DPO 再除以 \`mask.sum(-1)\`。`,
        },
      ],
      setup: PY_REF_LOGPS + '\n' + PY_MAKE_PAIRS,
      tests: [
        {
          name: '手算示例',
          code: py`
            lg = np.log(np.array([[[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]]]))   # (1, 3, 2)
            lab = np.array([[-100, 1, 0]])
            got = sequence_logps(lg, lab)
            exp = np.log(0.5) + np.log(0.2)        # 位置0预测token1=1 → 0.5；位置1预测token2=0 → 0.2
            assert got.shape == (1,), f"形状应为 (1,)，得到 {got.shape}"
            assert np.allclose(got, exp), f"期望 {exp:.6f}，得到 {got}"
          `,
        },
        {
          name: 'padding / prompt 处的垃圾 logits 不影响结果',
          code: py`
            pol, ref, labels = _make_pairs(0)
            got = sequence_logps(pol, labels)
            exp = _ref_sequence_logps(pol, labels)
            assert np.allclose(got, exp, atol=1e-8), f"最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: '随机对拍 + 大 logits 数值稳定',
          hidden: true,
          code: py`
            for s in range(1, 6):
                pol, ref, labels = _make_pairs(s, B=4, T=11, V=9)
                pol = pol * 300.0
                got = sequence_logps(pol, labels)
                assert np.all(np.isfinite(got)), "出现 inf/nan：log_softmax 不稳定？"
                exp = _ref_sequence_logps(pol, labels)
                assert np.allclose(got, exp, atol=1e-6), f"seed={s} 最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
      ],
      outro: md`
        把第 5 空换成 \`(tok * mask).sum(-1) / mask.sum(-1)\` 就是平均 logp（SimPO 的奖励、ORPO 的 $P_\theta$ 都用它）。
        PyTorch 里常写成 \`torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2)).squeeze(2)\`。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-dpo-loss',
      title: '由 logits 计算 DPO loss 与隐式奖励',
      difficulty: 'medium',
      source: '大模型岗手撕 · DPO',
      timeLimitMin: 25,
      description: md`
        实现 TRL 风格的 DPO 损失。chosen 与 rejected 已经拼成一个 batch（前 $B$ 行 chosen、后 $B$ 行 rejected，第 $b$ 行与第 $B+b$ 行是一对）。

        \`\`\`python
        def dpo_loss(policy_logits: np.ndarray, ref_logits: np.ndarray,
                     labels: np.ndarray, beta: float = 0.1
                     ) -> tuple[float, np.ndarray, np.ndarray]: ...
        \`\`\`

        - \`policy_logits\`、\`ref_logits\`：\`(2B, T, V)\` float64，\`logits[:, t]\` 预测第 $t+1$ 个 token；
        - \`labels\`：\`(2B, T)\` int，与 input_ids 对齐（**未移位**），prompt 和 padding 为 \`-100\`；
        - 序列 logp = 回复部分逐 token log-prob 之**和**；
        - 返回 \`(loss, chosen_rewards, rejected_rewards)\`：
          - \`loss\`：$B$ 对样本的 $-\log\sigma\big(\beta[(\log\pi_\theta(y_w)-\log\pi_\theta(y_l))-(\log\pi_{\mathrm{ref}}(y_w)-\log\pi_{\mathrm{ref}}(y_l))]\big)$ 的平均（Python float）；
          - \`chosen_rewards\` / \`rejected_rewards\`：形状 \`(B,)\`，$\beta(\log\pi_\theta-\log\pi_{\mathrm{ref}})$。

        **要求**：数值稳定——logits 可能达到 $\pm10^4$，margin 可能让 $\beta\Delta$ 达到 $\pm10^3$，结果不能出现 inf / nan。

        **示例**：policy 与 reference 完全相同时，loss = $\ln2\approx0.6931$，两个 rewards 全为 0。
      `,
      starter: py`
        import numpy as np

        def dpo_loss(policy_logits, ref_logits, labels, beta=0.1):
            """返回 (loss: float, chosen_rewards: (B,), rejected_rewards: (B,))"""
            # TODO 1: 写一个 sequence_logps(logits, labels)：shift + 稳定 log_softmax + gather + mask 求和
            # TODO 2: 拆出 chosen（前 B 行）与 rejected（后 B 行）
            # TODO 3: 稳定地计算 -log σ(β·Δ) 并取平均；计算隐式奖励
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def _log_softmax(x):
            m = x.max(axis=-1, keepdims=True)
            return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

        def sequence_logps(logits, labels, ignore_index=-100):
            logits, labels = logits[:, :-1, :], labels[:, 1:]       # shift：位置 t 预测 token t+1
            mask = labels != ignore_index
            safe = np.where(mask, labels, 0)
            tok = np.take_along_axis(_log_softmax(logits), safe[..., None], axis=-1)[..., 0]
            return (tok * mask).sum(axis=-1)                         # (2B,)

        def dpo_loss(policy_logits, ref_logits, labels, beta=0.1):
            B = labels.shape[0] // 2
            pi = sequence_logps(policy_logits, labels)
            ref = sequence_logps(ref_logits, labels)                 # PyTorch 里放在 no_grad 下
            pi_c, pi_r, ref_c, ref_r = pi[:B], pi[B:], ref[:B], ref[B:]
            margin = (pi_c - pi_r) - (ref_c - ref_r)
            losses = np.logaddexp(0.0, -beta * margin)               # -log σ(z) = log(1 + e^{-z})
            chosen_rewards = beta * (pi_c - ref_c)
            rejected_rewards = beta * (pi_r - ref_r)
            return float(losses.mean()), chosen_rewards, rejected_rewards
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch
            import torch.nn.functional as F

            def sequence_logps(logits, labels, ignore_index=-100):
                logits, labels = logits[:, :-1, :], labels[:, 1:].clone()
                mask = labels != ignore_index
                labels[~mask] = 0
                tok = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2)).squeeze(2)
                return (tok * mask).sum(-1)

            def dpo_loss(policy_model, ref_model, input_ids, attention_mask, labels, beta=0.1):
                B = input_ids.shape[0] // 2
                pi = sequence_logps(policy_model(input_ids, attention_mask=attention_mask).logits.float(), labels)
                with torch.no_grad():
                    ref = sequence_logps(ref_model(input_ids, attention_mask=attention_mask).logits.float(), labels)
                margin = (pi[:B] - pi[B:]) - (ref[:B] - ref[B:])
                loss = -F.logsigmoid(beta * margin).mean()
                chosen_rewards = beta * (pi[:B] - ref[:B]).detach()
                rejected_rewards = beta * (pi[B:] - ref[B:]).detach()
                return loss, chosen_rewards, rejected_rewards
          `,
        },
      ],
      explanation: md`
        **思路**：一次前向拿到 $2B$ 条序列的 logp，再按行切成 chosen / rejected。

        1. **shift**：\`logits[:, :-1]\` 对齐 \`labels[:, 1:]\`；
        2. **稳定 log-softmax**：减去最大值；
        3. **gather + mask**：-100 的位置先换成 0 再 gather，结果乘 mask；
        4. **稳定 logsigmoid**：$-\log\sigma(z)=\log(1+e^{-z})$ 用 \`np.logaddexp(0, -z)\`；直接 \`-np.log(1/(1+np.exp(-z)))\` 在 $z=-1000$ 时 \`exp\` 溢出；
        5. 隐式奖励在 PyTorch 里要 \`detach()\`，只用于监控（rewards/chosen、rewards/margins、rewards/accuracies）。

        **复杂度**：$O(N\cdot T\cdot V)$，瓶颈在 log-softmax；显存瓶颈是完整 logits（见图解）。

        **常见错误**：不做 shift；把 \`-100\` 直接当下标（NumPy 的负下标会静默取到倒数第 100 个元素，或越界）；reference 的 log-ratio 符号写反；mean 与 sum 混用。
      `,
      complexity: { time: 'O(N·T·V)', space: 'O(N·T·V)' },
      hints: [
        md`先写辅助函数 \`sequence_logps\`：它和跟练里的完全一样。`,
        md`\`B = labels.shape[0] // 2\`，\`pi[:B]\` 是 chosen，\`pi[B:]\` 是 rejected。`,
        md`\`margin = (pi_c - pi_r) - (ref_c - ref_r)\`，loss 用 \`np.logaddexp(0.0, -beta * margin)\`。`,
      ],
      setup: PY_REF_LOGPS + '\n' + PY_MAKE_PAIRS + '\n' + py`
        def _ref_dpo(pol, ref, labels, beta):
            B = labels.shape[0] // 2
            pi = _ref_sequence_logps(pol, labels)
            rf = _ref_sequence_logps(ref, labels)
            m = (pi[:B] - pi[B:]) - (rf[:B] - rf[B:])
            return float(np.mean(-_ref_log_sigmoid(beta * m))), beta * (pi[:B] - rf[:B]), beta * (pi[B:] - rf[B:])

        def _check(got, exp, tag=""):
            assert isinstance(got, tuple) and len(got) == 3, "应返回 (loss, chosen_rewards, rejected_rewards) 三元组"
            l, c, r = got
            el, ec, er = exp
            c = np.asarray(c); r = np.asarray(r)
            assert c.shape == ec.shape and r.shape == er.shape, f"{tag} rewards 形状应为 {ec.shape}，得到 {c.shape} / {r.shape}"
            assert np.isfinite(l), f"{tag} loss 不是有限数：{l}"
            assert abs(float(l) - el) < 1e-6 * max(1.0, abs(el)), f"{tag} loss 期望 {el:.6f}，得到 {float(l):.6f}"
            assert np.allclose(c, ec, rtol=1e-6, atol=1e-6), f"{tag} chosen_rewards 期望 {np.round(ec, 4)}，得到 {np.round(c, 4)}"
            assert np.allclose(r, er, rtol=1e-6, atol=1e-6), f"{tag} rejected_rewards 期望 {np.round(er, 4)}，得到 {np.round(r, 4)}"
      `,
      tests: [
        {
          name: 'policy == reference → ln2',
          code: py`
            pol, _, labels = _make_pairs(1)
            l, c, r = dpo_loss(pol, pol.copy(), labels, beta=0.1)
            assert abs(l - np.log(2)) < 1e-9, f"期望 ln2 = 0.693147，得到 {l}"
            assert np.allclose(c, 0) and np.allclose(r, 0), f"rewards 应全为 0，得到 {c}, {r}"
          `,
        },
        {
          name: '小例子（B=2）',
          code: py`
            pol, ref, labels = _make_pairs(2, B=2, T=6, V=5)
            _check(dpo_loss(pol, ref, labels, beta=0.1), _ref_dpo(pol, ref, labels, 0.1))
          `,
        },
        {
          name: '方向检查：提高 chosen 的概率应降低 loss',
          code: py`
            pol, ref, labels = _make_pairs(3, B=2)
            base, _, _ = dpo_loss(pol, ref, labels)
            better = pol.copy()
            B = 2
            lab = labels[:B, 1:]
            for b in range(B):
                for t in range(lab.shape[1]):
                    if lab[b, t] != -100:
                        better[b, t, lab[b, t]] += 2.0     # 只抬高 chosen 回复 token 的 logit
            l2, _, _ = dpo_loss(better, ref, labels)
            assert l2 < base, f"抬高 chosen 概率后 loss 应下降：{base:.4f} → {l2:.4f}（reference 符号或 shift 写反了？）"
          `,
        },
        {
          name: '随机对拍（含 padding 垃圾值、不同 β）',
          hidden: true,
          code: py`
            for s, beta in [(10, 0.1), (11, 0.5), (12, 0.05), (13, 1.0)]:
                pol, ref, labels = _make_pairs(s, B=4, T=10, V=8)
                _check(dpo_loss(pol, ref, labels, beta=beta), _ref_dpo(pol, ref, labels, beta), f"seed={s}:")
          `,
        },
        {
          name: '数值稳定：logits ±1e4、βΔ ±1e3',
          hidden: true,
          code: py`
            pol, _, labels = _make_pairs(21, B=3, T=8, V=6, scale=3000.0)
            ref = np.random.default_rng(99).normal(0, 3000.0, size=pol.shape)   # 与 policy 无关 → |βΔ| ~ 1e3
            got = dpo_loss(pol, ref, labels, beta=0.5)
            assert np.isfinite(got[0]), f"loss = {got[0]}：log_softmax 或 logsigmoid 溢出了"
            _check(got, _ref_dpo(pol, ref, labels, 0.5), "大数值：")
          `,
        },
        {
          name: 'B = 1 与全 mask 的边界',
          hidden: true,
          code: py`
            pol, ref, labels = _make_pairs(30, B=1, T=5, V=4)
            _check(dpo_loss(pol, ref, labels), _ref_dpo(pol, ref, labels, 0.1), "B=1:")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-ref-free',
          q: md`如果显存放不下 reference model，有哪些办法？`,
          a: md`
            1. **预计算**：训练前用 reference 把所有样本的序列 logp 算好存盘（TRL \`precompute_ref_log_probs=True\`），训练时只保留 policy；
            2. **LoRA**：policy = 基座 + adapter，关闭 adapter 的前向就是 reference，不需要第二份权重；
            3. **去掉 reference**：改用 SimPO / ORPO 这类 reference-free 目标，但失去了 KL 锚点，需要靠 margin、SFT 项或早停控制漂移。
          `,
        },
        {
          id: 'f-label-smooth',
          q: md`偏好标签有噪声（约 10% 被标反）时，DPO loss 可以怎么改？`,
          a: md`
            把标签当成“以概率 $\epsilon$ 翻转”：conservative DPO 用 $-(1-\epsilon)\log\sigma(\beta\Delta)-\epsilon\log\sigma(-\beta\Delta)$（label smoothing），
            Robust DPO（Chowdhury et al. 2024）进一步做了无偏修正，TRL 中对应 \`loss_type="robust"\` 与 \`label_smoothing\`。另一个思路是 IPO 的平方损失，它对确定性偏好不会无限推大 margin。
          `,
        },
      ],
    },
    {
      id: 'c-dpo-grad',
      title: '推导并实现 DPO 对 logits 的梯度',
      difficulty: 'hard',
      source: '大模型岗手撕 · DPO 反向',
      timeLimitMin: 30,
      description: md`
        沿用上一题的约定（前 $B$ 行 chosen、后 $B$ 行 rejected，labels 未移位、-100 不计入），
        损失 $\mathcal L=\frac1B\sum_{b}-\log\sigma(\beta\Delta_b)$。实现它对 **policy logits** 的解析梯度：

        \`\`\`python
        def dpo_logits_grad(policy_logits: np.ndarray, ref_logits: np.ndarray,
                            labels: np.ndarray, beta: float = 0.1) -> np.ndarray: ...
        \`\`\`

        返回与 \`policy_logits\` 同形状 \`(2B, T, V)\` 的数组。reference 不参与求导。

        **提示性推导**（请自己补全）：
        - $\partial\mathcal L/\partial\Delta_b=?$
        - 序列 logp $S_i=\sum_t m_{i,t}\log\mathrm{softmax}(z_{i,t})_{y_{i,t+1}}$，$\partial S_i/\partial z_{i,t}=?$
        - chosen 行与 rejected 行的符号有什么区别？最后一个位置的梯度是多少？

        测试会用中心差分数值梯度检查你的结果。
      `,
      starter: py`
        import numpy as np

        def dpo_logits_grad(policy_logits, ref_logits, labels, beta=0.1):
            """返回 dL/d policy_logits，形状 (2B, T, V)"""
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def _log_softmax(x):
            m = x.max(axis=-1, keepdims=True)
            return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

        def _seq_logps(logits, labels):
            lab = labels[:, 1:]
            mask = lab != -100
            safe = np.where(mask, lab, 0)
            tok = np.take_along_axis(_log_softmax(logits[:, :-1]), safe[..., None], axis=-1)[..., 0]
            return (tok * mask).sum(-1)

        def dpo_logits_grad(policy_logits, ref_logits, labels, beta=0.1):
            N, T, V = policy_logits.shape
            B = N // 2
            pi, ref = _seq_logps(policy_logits, labels), _seq_logps(ref_logits, labels)
            delta = (pi[:B] - pi[B:]) - (ref[:B] - ref[B:])
            # dL/dΔ_b = -(β/B)·σ(-βΔ_b)，σ(-x) = exp(-logaddexp(0, x)) 数值稳定
            coef = -(beta / B) * np.exp(-np.logaddexp(0.0, beta * delta))
            sign = np.concatenate([coef, -coef])                   # Δ 对 chosen 是 +S，对 rejected 是 -S

            lab = labels[:, 1:]
            mask = (lab != -100).astype(policy_logits.dtype)
            safe = np.where(lab != -100, lab, 0)
            p = np.exp(_log_softmax(policy_logits[:, :-1]))        # (N, T-1, V)
            onehot = np.zeros_like(p)
            np.put_along_axis(onehot, safe[..., None], 1.0, axis=-1)
            dS = (onehot - p) * mask[..., None]                    # dS_i / dz_{i,t}

            grad = np.zeros_like(policy_logits)
            grad[:, :-1, :] = sign[:, None, None] * dS             # 最后一个位置不参与任何预测
            return grad
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：用 autograd 验证手推公式）',
          code: py`
            import torch
            import torch.nn.functional as F

            def dpo_logits_grad_manual(pol, ref, labels, beta=0.1):
                N, T, V = pol.shape
                B = N // 2
                lab = labels[:, 1:]
                mask = (lab != -100).float()
                safe = lab.clamp(min=0)
                lp = pol[:, :-1].log_softmax(-1)
                S = (lp.gather(2, safe.unsqueeze(2)).squeeze(2) * mask).sum(-1)
                with torch.no_grad():
                    R = (ref[:, :-1].log_softmax(-1).gather(2, safe.unsqueeze(2)).squeeze(2) * mask).sum(-1)
                delta = (S[:B] - S[B:]) - (R[:B] - R[B:])
                coef = -(beta / B) * torch.sigmoid(-beta * delta)
                sign = torch.cat([coef, -coef])
                dS = (F.one_hot(safe, V).float() - lp.exp()) * mask.unsqueeze(-1)
                grad = torch.zeros_like(pol)
                grad[:, :-1] = sign[:, None, None] * dS
                return grad

            # 校验：pol.requires_grad_(); loss.backward(); torch.allclose(pol.grad, manual)
          `,
        },
      ],
      explanation: md`
        **推导**：
        1. $\ell_b=-\log\sigma(\beta\Delta_b)\Rightarrow\frac{\partial\mathcal L}{\partial\Delta_b}=-\frac{\beta}{B}\sigma(-\beta\Delta_b)$——正是论文里的“纠错权重”；
        2. $\Delta_b=S_b-S_{B+b}-(\text{ref 常数})$，所以 chosen 行系数为 $c_b$、rejected 行为 $-c_b$；
        3. softmax 交叉熵的经典结果：$\frac{\partial\log\mathrm{softmax}(z)_y}{\partial z}=\mathbf e_y-\mathrm{softmax}(z)$，乘上 mask；
        4. 位置 $T-1$ 的 logits 不预测任何 token（被 shift 丢掉），梯度为 0；被 mask 的位置梯度也为 0。

        $$
        \frac{\partial\mathcal L}{\partial z_{i,t}}=\pm\Big(-\frac{\beta}{B}\sigma(-\beta\Delta_b)\Big)\,m_{i,t}\,\big(\mathbf e_{y_{i,t+1}}-\mathrm{softmax}(z_{i,t})\big)
        $$

        **可以从中读出的性质**：每个位置的梯度在词表维度上求和为 0；chosen 的 label token 被推高（梯度为负，梯度下降后 logit 变大），rejected 的 label token 被压低；样本被排对后 $\sigma(-\beta\Delta)\to0$，整对样本的梯度一起消失。
      `,
      complexity: { time: 'O(N·T·V)', space: 'O(N·T·V)' },
      hints: [
        md`先算出每对的 $\Delta_b$，再算 $c_b=-\frac{\beta}{B}\sigma(-\beta\Delta_b)$。`,
        md`对 rejected 行，$\Delta$ 中是 $-S$，所以系数取 $-c_b$；可以 \`np.concatenate([c, -c])\`。`,
        md`\`dS = (onehot - softmax) * mask[..., None]\`，放进 \`grad[:, :-1, :]\`，最后一个位置保持 0。`,
      ],
      setup: PY_REF_LOGPS + '\n' + PY_MAKE_PAIRS + '\n' + py`
        def _ref_loss(pol, ref, labels, beta):
            B = labels.shape[0] // 2
            pi = _ref_sequence_logps(pol, labels)
            rf = _ref_sequence_logps(ref, labels)
            m = (pi[:B] - pi[B:]) - (rf[:B] - rf[B:])
            return float(np.mean(-_ref_log_sigmoid(beta * m)))

        def _num_grad(pol, ref, labels, beta, eps=1e-6):
            g = np.zeros_like(pol)
            it = np.nditer(pol, flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                a = pol.copy(); a[i] += eps
                b = pol.copy(); b[i] -= eps
                g[i] = (_ref_loss(a, ref, labels, beta) - _ref_loss(b, ref, labels, beta)) / (2 * eps)
            return g
      `,
      tests: [
        {
          name: '形状与最后位置为 0',
          code: py`
            pol, ref, labels = _make_pairs(5, B=2, T=6, V=5)
            g = dpo_logits_grad(pol, ref, labels, 0.1)
            assert g.shape == pol.shape, f"形状应为 {pol.shape}，得到 {g.shape}"
            assert np.allclose(g[:, -1, :], 0), "最后一个位置的 logits 不预测任何 token，梯度应为 0"
          `,
        },
        {
          name: '数值梯度检查（小规模）',
          code: py`
            pol, ref, labels = _make_pairs(6, B=1, T=4, V=3, garbage=False)
            g = dpo_logits_grad(pol, ref, labels, 0.3)
            ng = _num_grad(pol, ref, labels, 0.3)
            err = np.abs(g - ng).max()
            assert err < 1e-6, f"与数值梯度的最大误差 {err:.2e}"
          `,
        },
        {
          name: '每个位置在词表维度上梯度和为 0',
          code: py`
            pol, ref, labels = _make_pairs(7, B=3, T=7, V=6)
            g = dpo_logits_grad(pol, ref, labels, 0.1)
            s = np.abs(g.sum(-1)).max()
            assert s < 1e-10, f"softmax 的梯度在 V 维上应和为 0，最大偏差 {s:.2e}"
          `,
        },
        {
          name: '数值梯度检查（含 padding、B=2）',
          hidden: true,
          code: py`
            pol, ref, labels = _make_pairs(8, B=2, T=5, V=3, scale=1.5, garbage=False)
            for beta in (0.1, 1.0):
                g = dpo_logits_grad(pol, ref, labels, beta)
                ng = _num_grad(pol, ref, labels, beta)
                err = np.abs(g - ng).max()
                assert err < 1e-6, f"beta={beta} 与数值梯度的最大误差 {err:.2e}"
          `,
        },
        {
          name: '被 mask 的位置梯度为 0、极端 margin 不出 nan',
          hidden: true,
          code: py`
            pol, ref, labels = _make_pairs(9, B=3, T=8, V=5, scale=500.0)
            g = dpo_logits_grad(pol, ref, labels, 1.0)
            assert np.all(np.isfinite(g)), "梯度出现 inf/nan"
            masked = labels[:, 1:] == -100
            assert np.allclose(g[:, :-1][masked], 0), "label 为 -100 的位置不应有梯度"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-grad-unweighted',
          q: md`如果去掉权重 $\sigma(-\beta\Delta)$，直接最大化 $\log\pi(y_w)-\log\pi(y_l)$，会发生什么？`,
          a: md`
            这相当于“对 chosen 做 SFT、对 rejected 做 unlikelihood”，没有饱和机制：已经排对的样本仍以全力推开，rejected 的 logp 可以被无限压低，模型很快退化（DPO 论文 §4 明确指出朴素版本会让模型退化）。
            σ 权重让排对的样本自动退出训练，等价于一个“自适应早停”。
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-derive',
      freq: 3,
      q: md`请推导 DPO 的损失函数。`,
      a: md`
        **结论**：$\mathcal L=-\mathbb E\log\sigma\big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\big)$。

        **推导**：① RLHF 目标 $\max\mathbb E[r]-\beta\mathbb D_{\mathrm{KL}}[\pi\|\pi_{\mathrm{ref}}]$ 可改写为 $-\beta\mathbb D_{\mathrm{KL}}[\pi\|\pi^*]+\beta\log Z(x)$，于是最优解 $\pi^*=\pi_{\mathrm{ref}}e^{r/\beta}/Z(x)$；② 取对数反解 $r=\beta\log\frac{\pi^*}{\pi_{\mathrm{ref}}}+\beta\log Z(x)$；③ 代入只依赖奖励差的 Bradley-Terry 模型，同一 prompt 下 $\beta\log Z(x)$ 相消；④ 用 $\pi_\theta$ 参数化，对偏好数据做负对数似然。

        **延伸**：Z(x) 消掉的前提是成对、同 prompt；对 K 个回答的排序可用 Plackett-Luce 推出同样形式；梯度中的 $\sigma(-\beta\Delta)$ 是纠错权重。
      `,
    },
    {
      id: 'qa-vs-ppo',
      freq: 3,
      q: md`DPO 和 PPO-RLHF 各有什么优缺点？工业界怎么选？`,
      a: md`
        **结论**：DPO 简单、便宜、稳定；PPO 上限更高但工程复杂。

        **原因**：DPO 只需 policy + reference（还能预计算），是一个分类损失，无需 RM、critic 和在线采样，超参少。代价是**离线**：数据不来自当前策略，隐式奖励在分布外缺少约束，容易出现 chosen 概率下降、长度变长、过拟合确定性偏好。PPO 在线采样、显式 RM 可以泛化到新回答，Xu et al. 2024 显示调好的 PPO 在对话和代码竞赛上都超过 DPO；但要同时维护四个模型、GAE、value 训练，调参困难。

        **延伸**：折中方案是 iterative / online DPO（Llama 3 共六轮，每轮用新模型采样构造偏好）；可验证任务（数学、代码）现在主流用 GRPO 系 RLVR。
      `,
    },
    {
      id: 'qa-beta',
      freq: 2,
      q: md`DPO 里的 β 有什么作用？调大调小分别会怎样？`,
      a: md`
        **结论**：β 是 KL 约束强度，也是隐式奖励 $\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ 的尺度。

        **原因**：最优解 $\pi^*\propto\pi_{\mathrm{ref}}e^{r/\beta}$，β 大则离 reference 近；在 loss 上，β 大时较小的 log-ratio margin 就能让 $\sigma(\beta\Delta)$ 饱和、梯度消失，模型被拴得更紧，β 小则需要把 Δ 推得很大，偏离 reference 更多、更容易过拟合与胡言乱语。

        **延伸**：原论文默认 0.1（TL;DR 用 0.5）；Llama 3 用 0.1；对 logp 做长度平均后数值尺度变小，β 要相应变大（Tülu 3 的 length-normalized DPO 用 β = 5，SimPO 推荐 2.0–2.5）。
      `,
    },
    {
      id: 'qa-chosen-drop',
      freq: 2,
      q: md`为什么 DPO 训练时 chosen 的 log-prob 也会下降？怎么缓解？`,
      a: md`
        **结论**：因为 DPO 只约束 chosen 与 rejected 的**相对**差值，没有约束绝对值。

        **原因**：chosen 与 rejected 往往高度相似（共享大量 token），压低 rejected 的梯度会连带压低 chosen；只要 rejected 降得更快，loss 就在下降。概率质量被挪到了数据之外的回答上，可能导致输出变长、格式错乱、甚至能力下降。

        **延伸**：① 在 chosen 上加 NLL 项（Llama 3 用系数 0.2，IRPO 同理）；② DPOP（Smaug）对 chosen 概率低于 reference 的情况加惩罚；③ mask 掉两边共有的格式 token（Llama 3）；④ 用 on-policy 数据做 iterative DPO。
      `,
    },
    {
      id: 'qa-implicit',
      freq: 2,
      q: md`什么是 DPO 的“隐式奖励”？训练完的 DPO 模型能当奖励模型用吗？`,
      a: md`
        **结论**：$\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$，在同一 prompt 下它与真实奖励相差一个只依赖 $x$ 的常数 $\beta\log Z(x)$。

        **原因**：DPO 本质是用这个参数化拟合 BT 奖励模型，所以同 prompt 内的**排序与差值**有意义，可以拿来给回答排序或做 best-of-n；但跨 prompt 的绝对值不可比（$Z(x)$ 不同），也不保证在分布外校准。

        **延伸**：RewardBench 等评测里常把 DPO 模型当作隐式 RM 打分；训练监控的 rewards/chosen、rewards/margins、rewards/accuracies 都是这个量。
      `,
    },
  ],

  cheatsheet: md`
    **推导四步**：闭式解 $\pi^*\propto\pi_{\mathrm{ref}}e^{r/\beta}$ → 反解 $r=\beta\log\frac{\pi^*}{\pi_{\mathrm{ref}}}+\beta\log Z$ → 代入 BT，同 prompt 下 Z 相消 → $\pi_\theta$ 参数化做 MLE

    $$\mathcal L_{\mathrm{DPO}}=-\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\log\tfrac{\pi_\theta(y_l)}{\pi_{\mathrm{ref}}(y_l)}\big]\Big),\quad \nabla\propto-\beta\,\sigma(-\beta\Delta)\,[\nabla\log\pi(y_w)-\nabla\log\pi(y_l)]$$

    \`\`\`python
    lg, lab = logits[:, :-1], labels[:, 1:]          # shift
    mask = lab != -100
    tok = take_along_axis(log_softmax(lg), where(mask, lab, 0)[..., None], -1)[..., 0]
    S = (tok * mask).sum(-1)                          # 前 B 行 chosen，后 B 行 rejected
    delta = (S[:B] - S[B:]) - (S_ref[:B] - S_ref[B:])
    loss = logaddexp(0, -beta * delta).mean()         # 稳定的 -log σ
    rewards = beta * (S - S_ref)                      # 隐式奖励（detach）
    \`\`\`

    - 第一步 loss 必为 $\ln2=0.693$；原论文 β = 0.1（TL;DR 0.5）、lr 1e-6
    - Llama 3：β = 0.1，mask 格式 token，chosen 上 0.2 × NLL，六轮迭代
    - Tülu 3：length-normalized DPO，β = 5
    - prompt 误算入：原版 loss 不变（相消），长度归一化版会变；padding 误算入：真 bug
    - chosen logp 下降 ≠ bug；离线 → 分布偏移，iterative DPO 缓解
  `,
})
