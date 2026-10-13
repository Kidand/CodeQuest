/**
 * DPO 变体：IPO、SimPO、KTO、ORPO、online / iterative DPO。
 * 公式均按原论文核对：IPO（Azar et al., arXiv 2310.12036, Algorithm 1）、SimPO（Meng et al., 2405.14734, Eq.4/6 与 Table 8）、
 * KTO（Ethayarajh et al., 2402.01306, §4.1）、ORPO（Hong et al., 2403.07691, §4）、Self-Rewarding LM（2401.10020）、
 * OAIF（2402.04792）、Llama 3（2407.21783 §4.1.6）、Tülu 3（2411.15124）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { IpoViz } from '../viz/IpoViz'
import { SimpoViz } from '../viz/SimpoViz'
import { KtoViz } from '../viz/KtoViz'

const PY_TOKEN_BATCH = py`
  import numpy as np

  def _make_tok(seed, B=4, T=8, lo=-3.0):
      """逐 token logp（已 shift / gather）与 mask，右侧 padding，padding 处填垃圾值"""
      rng = np.random.default_rng(seed)
      out = []
      for _ in range(2):
          lens = rng.integers(1, T + 1, size=B)
          mask = (np.arange(T)[None, :] < lens[:, None]).astype(np.float64)
          lp = rng.uniform(lo, -0.01, size=(B, T))
          lp[mask == 0] = rng.uniform(-50, 50, size=int((mask == 0).sum()))
          out += [lp, mask]
      return out   # logp_c, mask_c, logp_r, mask_r

  def _avg(lp, m):
      return np.where(m > 0, lp, 0.0).sum(-1) / m.sum(-1)
`

export default defineLevel({
  learn: [
    {
      id: 'map',
      title: 'DPO 的软肋与变体地图',
      body: md`
        DPO 把 RLHF 变成了一个分类损失，但留下了几个问题，每个变体各补一处：

        | 问题 | 代表方法 | 改法 | 需要 ref？ | 数据 |
        |---|---|---|---|---|
        | 确定性偏好下过拟合，KL 约束失效 | **IPO** | 用平方损失把 margin 拉向固定目标 $\frac{1}{2\tau}$ | 是 | 成对 |
        | 奖励与解码时的似然口径不一致、长度偏置、reference 的开销 | **SimPO** | 奖励改为 $\frac{\beta}{\lvert y\rvert}\log\pi_\theta(y\mid x)$，加目标 margin γ | 否 | 成对 |
        | 只有“好 / 坏”单条标注，没有成对比较 | **KTO** | 前景理论价值函数，参考点为 KL | 是 | 非成对 |
        | SFT 与对齐两阶段、reference 开销 | **ORPO** | SFT 损失 + λ·odds ratio 损失，一步完成 | 否 | 成对 |
        | 离线数据与当前策略分布不一致 | **online / iterative DPO** | 每轮用当前模型采样、打分、再训 | 视基础算法 | 在线生成 |

        这些方法在 TRL 等框架中多已作为 DPO 的 \`loss_type\` 或独立 Trainer 提供。下面逐个拆开，重点是**公式与原论文一致**、**解决了什么、代价是什么**。
      `,
      after: md`
        > [!WARN] 别把“变体更新”当成“变体更好”
        > 这些方法的实验多在 UltraFeedback、AlpacaEval 2、Arena-Hard、MT-Bench 等基准上，结论高度依赖基座、数据与超参。Tülu 3 的消融里，SimPO 与原版 DPO 都没跑赢 length-normalized DPO；
        > 工业界报告（Llama 3、Tülu 3）里最常见的仍是 DPO 本身加若干修补。面试时讲清每个方法**针对的具体问题**，比背排行榜有用。
      `,
      checkpoint: {
        id: 'cp-map',
        type: 'multi',
        prompt: md`下列方法中，训练时**不需要** reference model 的有：`,
        options: ['DPO', 'IPO', 'SimPO', 'KTO', 'ORPO'],
        answer: [2, 4],
        explanation: md`SimPO 的奖励是 $\frac{\beta}{|y|}\log\pi_\theta$，ORPO 用 $\pi_\theta$ 自身的 odds，都不含 $\pi_{\mathrm{ref}}$。DPO、IPO 的 $h$ 中有 $\pi_{\mathrm{ref}}$；KTO 的隐式奖励 $r_\theta=\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$ 与参考点 $z_0$ 都依赖 reference。`,
      },
    },
    {
      id: 'ipo',
      title: 'IPO：给 margin 一个有限的目标',
      body: md`
        Azar et al.（*A General Theoretical Paradigm to Understand Learning from Human Preferences*）提出 **ΨPO** 统一框架：最大化 $\mathbb E[\Psi(p^*(y\succ y'|x))]-\tau\,\mathbb D_{\mathrm{KL}}[\pi\|\pi_{\mathrm{ref}}]$。RLHF 与 DPO 对应 $\Psi(q)=\log\frac{q}{1-q}$（BT 下的 logit）。

        **问题**：若数据中 $y$ 总是优于 $y'$（$p^*=1$），BT 要求 $r(y)-r(y')\to+\infty$，代入最优策略得 $\pi^*(y')=0$，**无论 KL 系数 τ 取多少**——偏好越确定，KL 正则越弱。有限数据中这很常见：一对样本只被标注一次，就是一个“确定性偏好”。

        **IPO**（Ψ 取恒等映射）的经验损失（Algorithm 1）：

        $$
        h_\pi(y_w,y_l,x)=\log\frac{\pi(y_w|x)\,\pi_{\mathrm{ref}}(y_l|x)}{\pi(y_l|x)\,\pi_{\mathrm{ref}}(y_w|x)},\qquad
        \mathcal L_{\mathrm{IPO}}=\mathbb E\Big[\Big(h_\pi(y_w,y_l,x)-\frac{\tau^{-1}}{2}\Big)^2\Big]
        $$

        $h$ 就是 DPO 里的 log-ratio margin Δ。平方损失在 $h=\frac{1}{2\tau}$ 处有最小值，到了就停；DPO 的 $-\log\sigma(\beta h)$ 则永远想更大。
      `,
      viz: IpoViz,
      after: md`
        > [!NOTE] 代价与实践
        > - IPO 的目标 margin 对所有样本相同，不区分“强偏好”与“弱偏好”；τ 越小目标越大，约束越弱。
        > - TRL 中 \`loss_type="ipo"\` 时 \`beta\` 参数即论文的 τ。
        > - DPO 的过拟合在实践中也常用早停、label smoothing（cDPO）、更大的 β 来缓解；IPO 是从目标函数层面给出的解释与修正。
      `,
      checkpoint: {
        id: 'cp-ipo',
        type: 'numeric',
        prompt: md`IPO 中 τ = 0.1。某对样本当前 $h=3$，它的 IPO loss 是多少？`,
        answer: 4,
        tolerance: 0.001,
        explanation: md`目标 $\frac{1}{2\tau}=5$，loss $=(3-5)^2=4$。注意若 $h=7$，loss 同样是 4——IPO 会把“推过头”的 margin 往回拉，DPO 不会。`,
      },
    },
    {
      id: 'simpo',
      title: 'SimPO：长度归一化奖励 + 目标 margin，无 reference',
      body: md`
        Meng et al. 2024 指出 DPO 的隐式奖励 $\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$ 与指导生成的似然指标——**平均对数似然**（beam search 等按它比较候选）——不一致：DPO 奖励更高的回答，平均 logp 未必更高。SimPO（Eq.4、Eq.6）直接把后者当奖励：

        $$
        r_{\mathrm{SimPO}}(x,y)=\frac{\beta}{|y|}\log\pi_\theta(y|x)=\frac{\beta}{|y|}\sum_{t}\log\pi_\theta(y_t|x,y_{<t})
        $$

        $$
        \mathcal L_{\mathrm{SimPO}}=-\,\mathbb E\Big[\log\sigma\Big(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l|x)-\gamma\Big)\Big]
        $$

        - **长度归一化**：去掉“回复越长、logp 之和越负”的系统偏差，也抑制了靠拉长回复刷奖励；
        - **目标 margin γ**：要求 chosen 的奖励至少高出 γ（BT 模型加 margin，类似 Llama 2 的 RM）；
        - **无 reference**：省掉一个模型的显存与前向，但也失去了 KL 锚点。

        论文建议 β 取 2.0–2.5、γ 取 0.5–1.5；附录给出的具体设置如 Llama-3-Instruct β = 2.5、γ = 1.4，Mistral-Base β = 2.0、γ = 1.6（后续 v0.2 设置用到 β = 10、γ = 3）。
      `,
      viz: SimpoViz,
      after: md`
        > [!WARN] 没有 reference 的代价
        > 没有 KL 约束，模型可以把概率压到任何地方，只要平均 logp 的相对差达标。SimPO 依赖小学习率、少 epoch 与 margin 控制漂移；论文附录也报告了加 SFT 项的变体在 AlpacaEval 2 上反而变差、在 GSM8K 上有帮助——效果随任务而变。
        > 与之相对，Tülu 3 选用的 **length-normalized DPO** 保留了 reference，只把两边的 log-ratio 除以长度（β = 5）。
      `,
      checkpoint: {
        id: 'cp-simpo',
        type: 'numeric',
        prompt: md`SimPO 中 β = 2、γ = 1。chosen 长 40 token、$\log\pi_\theta(y_w)=-20$；rejected 长 10 token、$\log\pi_\theta(y_l)=-10$。求这对样本的 SimPO loss（自然对数，保留 3 位小数）。`,
        answer: 0.693,
        tolerance: 0.005,
        explanation: md`平均 logp：chosen $-0.5$、rejected $-1.0$。$z=2\times(-0.5)-2\times(-1.0)-1=0$，loss $=-\log\sigma(0)=\ln2\approx0.693$。若按序列和比较，chosen（−20）反而比 rejected（−10）“差”，这正是长度归一化要解决的问题。`,
      },
    },
    {
      id: 'kto',
      title: 'KTO：只有“好 / 坏”标签时怎么对齐',
      body: md`
        很多真实反馈是单条的（点赞 / 点踩），没有同 prompt 的成对比较。Ethayarajh et al. 2024 借用 Kahneman-Tversky 的**前景理论**：人对结果的感受相对于一个参考点，收益区风险厌恶、损失区更敏感。KTO（§4.1）：

        $$
        r_\theta(x,y)=\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)},\qquad z_0=\mathrm{KL}\big(\pi_\theta(y'|x)\,\|\,\pi_{\mathrm{ref}}(y'|x)\big)
        $$

        $$
        v(x,y)=\begin{cases}\lambda_D\,\sigma\big(\beta(r_\theta(x,y)-z_0)\big) & y\ \text{desirable}\\ \lambda_U\,\sigma\big(\beta(z_0-r_\theta(x,y))\big) & y\ \text{undesirable}\end{cases},\qquad
        \mathcal L_{\mathrm{KTO}}=\mathbb E\big[\lambda_y-v(x,y)\big]
        $$

        - **参考点 $z_0$ 的估计**：在 micro-batch 内把 prompt 与**别的样本**的回复错位配对（$x_i$ 配 $y_{i+1}$），估计 $\hat z_0=\max\big(0,\frac1m\sum\log\frac{\pi_\theta(y_j|x_i)}{\pi_{\mathrm{ref}}(y_j|x_i)}\big)$，**不回传梯度**；
        - **λ_D、λ_U**：论文建议 $\frac{\lambda_D n_D}{\lambda_U n_U}\in[1,\frac32]$，默认 β = 0.1、λ 都为 1；
        - 论文报告即使丢掉 90% 的 desirable 数据（此时取 λ_D = 13.33、λ_U = 1）仍优于 DPO。
      `,
      viz: KtoViz,
      after: md`
        > [!TIP] 为什么用 logistic 而不是 KT 的幂函数
        > 原始 KT 价值函数 $v(z)=z^\alpha$（收益）/ $-\lambda(-z)^\alpha$（损失），中位参数 α = 0.88、λ = 2.25。KTO 论文认为幂函数数值不稳定，改用 σ 保留“凹 / 凸 + 饱和”的形状；
        > 饱和带来一个副作用：离参考点很远的样本几乎没有梯度，相当于自动忽略了噪声大或过于简单的样本。

        **代价**：每个 micro-batch 要额外前向错位样本来估计 $z_0$（policy 和 reference 各一次）；micro-batch 至少要 2 条。
      `,
      checkpoint: {
        id: 'cp-kto',
        type: 'single',
        prompt: md`KTO 的参考点 $z_0$ 在实现中是如何处理的？`,
        options: [
          '用 batch 内错位配对的 (x_i, y_j) 估计 log-ratio 的均值，截断到 ≥ 0，且不回传梯度',
          '固定为 0',
          '用 chosen 与 rejected 的 log-ratio 之差',
          '作为可学习参数与模型一起训练',
        ],
        answer: 0,
        explanation: md`论文用错位配对估计 $\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})$（匹配的样本往往被刻意挑选成极好或极差，不具代表性），截断 $\max(0,\cdot)$ 以降低方差（代价是向上的偏差），并且不对它求导——它只用来控制饱和位置。C 是 DPO 的 margin，KTO 根本没有成对数据。`,
      },
    },
    {
      id: 'orpo',
      title: 'ORPO：把对齐塞进 SFT',
      body: md`
        Hong et al. 2024 观察到：SFT 在提高 chosen 似然的同时，也会提高风格相近的 rejected 的似然。ORPO 在 SFT 损失上加一个 **odds ratio** 惩罚，一个阶段、不要 reference：

        $$
        P_\theta(y|x)=\exp\Big(\frac{1}{|y|}\sum_t\log P_\theta(y_t|x,y_{<t})\Big),\qquad
        \mathrm{odds}_\theta(y|x)=\frac{P_\theta(y|x)}{1-P_\theta(y|x)}
        $$

        $$
        \mathcal L_{\mathrm{OR}}=-\log\sigma\Big(\log\frac{\mathrm{odds}_\theta(y_w|x)}{\mathrm{odds}_\theta(y_l|x)}\Big),\qquad
        \mathcal L_{\mathrm{ORPO}}=\mathbb E\big[\mathcal L_{\mathrm{SFT}}+\lambda\,\mathcal L_{\mathrm{OR}}\big]
        $$

        - $P_\theta$ 是**长度归一化**的（几何平均每 token 概率），因此 $0<P<1$，odds 有定义；$\mathcal L_{\mathrm{SFT}}$ 是 chosen 上的标准 NLL；
        - 论文取 λ = 0.1（Mistral-ORPO-α 7B）、0.2（Llama-2 7B）、0.25（Phi-2）；
        - 梯度分析：$\mathcal L_{\mathrm{OR}}$ 的梯度带一个因子 $\big[1+\frac{\mathrm{odds}(y_w)}{\mathrm{odds}(y_l)}\big]^{-1}$，当模型更倾向 rejected 时惩罚更大。

        **实现要点**：$\log\mathrm{odds}=\bar\ell-\log(1-e^{\bar\ell})$，$\bar\ell$ 为平均 logp（<0）。当 $\bar\ell\to0^-$ 时 $1-e^{\bar\ell}$ 发生灾难性抵消，要用 \`log(-expm1(ℓ̄))\`。
      `,
      after: md`
        > [!NOTE] 定位
        > ORPO 适合“只有偏好数据、想一步从 base 得到对齐模型”的场景；没有 reference 意味着没有显式 KL 约束，λ 太大时会伤害 SFT 本身的效果。和 SimPO 一样，它是“去掉 reference”路线的代表。
      `,
      checkpoint: {
        id: 'cp-orpo',
        type: 'fill',
        prompt: md`ORPO 中 chosen 的几何平均概率 $P_w=0.8$，rejected 的 $P_l=0.5$。则 $\mathrm{odds}_w$ = ____(1)____，$\mathcal L_{\mathrm{OR}}$ = ____(2)____（保留 3 位小数）。`,
        blanks: [
          { label: '(1)', answers: ['4'], numeric: true, tolerance: 0.001 },
          { label: '(2)', answers: ['0.223'], numeric: true, tolerance: 0.01 },
        ],
        explanation: md`$\mathrm{odds}_w=0.8/0.2=4$，$\mathrm{odds}_l=0.5/0.5=1$，$\log\mathrm{OR}=\ln4\approx1.386$，$\mathcal L_{\mathrm{OR}}=-\log\sigma(\ln4)=\log(1+\tfrac14)\approx0.223$。`,
      },
    },
    {
      id: 'online',
      title: 'Online / iterative DPO：把数据换成当前策略的样本',
      body: md`
        标准 DPO 的偏好对来自别的模型或旧版本自己，是 **off-policy、offline** 的；训练越久，策略越偏离数据分布，隐式奖励在“当前模型真正会说的话”上缺乏约束。补救办法是让数据跟着策略走：

        1. 用当前模型对一批 prompt 各采样多个回复；
        2. 用 RM、LLM-as-a-Judge 或可验证规则打分，取最好 / 最差构成偏好对；
        3. 训练一轮 DPO（或 IPO 等），reference 可更新为上一轮模型；
        4. 重复。

        | 工作 | 做法 |
        |---|---|
        | Llama 3 | 沿用 Llama 2 的多轮流程共 **六轮**；每轮用上一轮最好的模型采样、收集新偏好，做拒绝采样 SFT + DPO |
        | Self-Rewarding LM（Yuan et al. 2024） | Llama 2 70B 用 LLM-as-a-Judge **给自己打分**构造偏好对，迭代 DPO 三轮 |
        | OAIF（Guo et al. 2024） | 每步从当前策略采两个回复，由 LLM 标注者在线选优，用于 DPO 等直接对齐方法 |

        这把 DPO 拉回到“近似 on-policy RL”：数据分布问题缓解了，但又引入了采样与打分的成本，打分器的偏差也会被迭代放大（见「奖励模型」关卡的 LLM-as-a-Judge 偏差）。
      `,
      after: md`
        > [!NEW] 2025–2026
        > 可验证任务上，“在线采样 + 规则打分”已经直接演变成 RLVR（GRPO 系），不再需要构造偏好对；对话、风格、安全等不可验证任务上，iterative DPO 与 RM + PPO/GRPO 并存。详见「推理强化学习」章节。
      `,
      checkpoint: {
        id: 'cp-online',
        type: 'order',
        prompt: md`把 iterative DPO 的一轮按顺序排列：`,
        items: ['当前策略对 prompt 采样多个回复', '用 RM / LLM-as-a-Judge / 规则给回复打分', '选出最好与最差的回复组成偏好对', '在新偏好对上训练 DPO，得到下一轮策略'],
        explanation: md`采样 → 打分 → 构造偏好对 → 训练。与 PPO 相比，“优化”这一步仍是离线的分类损失，但数据每轮都来自最新策略。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把方法与它的核心改动配对：`,
      pairs: [
        ['IPO', '平方损失，把 margin 拉向 1/(2τ)'],
        ['SimPO', '平均 logp 作奖励 + 目标 margin γ，无 reference'],
        ['KTO', '单条好/坏标注，前景理论价值函数'],
        ['ORPO', 'SFT 损失 + odds ratio 惩罚，无 reference'],
      ],
      explanation: md`四种方法分别针对：确定性偏好过拟合、奖励口径与长度、数据形态、两阶段与 reference 开销。`,
    },
    {
      id: 'q-paired',
      type: 'judge',
      difficulty: 1,
      prompt: md`KTO 和 DPO 一样，需要同一 prompt 下的 chosen / rejected 成对数据。`,
      answer: false,
      explanation: md`KTO 只需要每条 (x, y) 是否 desirable 的二元标签，好坏样本可以来自不同 prompt、数量也可以不平衡（用 λ_D、λ_U 调节）。这是它相对 DPO 的主要卖点。`,
    },
    {
      id: 'q-ipo-why',
      type: 'single',
      difficulty: 2,
      prompt: md`IPO 论文认为 DPO 在“确定性偏好”下会过拟合，核心原因是：`,
      options: [
        'BT 模型下 p* = 1 要求奖励差趋于无穷，最优策略令 π(y_l) = 0，与 KL 系数无关',
        'DPO 的学习率太大',
        'DPO 没有使用 reference model',
        'DPO 使用了平方损失',
      ],
      answer: 0,
      explanation: md`$\Psi=\mathrm{logit}$ 把概率 1 映射到 $+\infty$，所以“永远更好”的偏好对应无穷大的奖励差，KL 正则无力约束。IPO 取 $\Psi$ 为恒等，目标有界。C、D 都把 DPO 与其它方法搞反了。`,
    },
    {
      id: 'q-ipo-num',
      type: 'numeric',
      difficulty: 2,
      prompt: md`
        IPO（τ = 0.5）：$\log\pi_\theta(y_w)=-10$、$\log\pi_\theta(y_l)=-12$、$\log\pi_{\mathrm{ref}}(y_w)=-11$、$\log\pi_{\mathrm{ref}}(y_l)=-11.5$。求 IPO loss。
      `,
      answer: 0.25,
      tolerance: 0.001,
      explanation: md`$h=(-10+12)-(-11+11.5)=2-0.5=1.5$；目标 $\frac{1}{2\tau}=1$；loss $=(1.5-1)^2=0.25$。`,
    },
    {
      id: 'q-simpo-len',
      type: 'single',
      difficulty: 2,
      prompt: md`SimPO 把奖励定义为 $\frac{\beta}{|y|}\log\pi_\theta(y|x)$ 而不是 $\beta\log\pi_\theta(y|x)$，论文给出的主要理由是：`,
      options: [
        '让训练奖励与生成时使用的平均对数似然一致，并消除长度偏差',
        '为了减少计算量',
        '为了与 reference model 的 logp 对齐',
        '为了让 loss 有限最小值点',
      ],
      answer: 0,
      explanation: md`beam search（带长度归一化）比较候选、多选题评测打分时用的都是平均对数似然；求和形式下长回复天然得分更低，模型可能通过调整长度“刷”奖励。B 无关；C 错——SimPO 根本没有 reference；D 是 IPO 的动机。`,
    },
    {
      id: 'q-orpo-num',
      type: 'numeric',
      difficulty: 3,
      prompt: md`
        ORPO 中某对样本：chosen 平均 token logp $\bar\ell_w=\ln0.9$，rejected $\bar\ell_l=\ln0.6$。求 $\mathcal L_{\mathrm{OR}}$（保留 3 位小数）。
      `,
      answer: 0.154,
      tolerance: 0.01,
      explanation: md`$P_w=0.9\Rightarrow\mathrm{odds}_w=9$；$P_l=0.6\Rightarrow\mathrm{odds}_l=1.5$；$\log\mathrm{OR}=\ln6\approx1.792$；$\mathcal L_{\mathrm{OR}}=-\log\sigma(\ln6)=\log(1+e^{-\ln6})=\log\frac76\approx0.154$。常见错误：直接用概率比 $0.9/0.6=1.5$ 代替 odds 比，得到 $\log(1+\frac{1}{1.5})\approx0.511$。`,
    },
    {
      id: 'q-pick-finite',
      type: 'pick',
      difficulty: 3,
      prompt: md`把每种损失看作 margin $h$ 的函数（其它量固定）。哪些损失在**有限的 $h$** 处就达到最小值、之后不再推动 margin 继续增大？`,
      items: ['DPO：−log σ(βh)', 'IPO：(h − 1/(2τ))²', 'SLiC / hinge：max(0, 1 − βh)', 'SimPO：−log σ(βh − γ)', 'ORPO 的 L_OR：−log σ(h)'],
      answer: [1, 2],
      explanation: md`IPO 在 $h=\frac1{2\tau}$ 处取最小（之后反而被拉回）；hinge 在 $h\ge\frac1\beta$ 后 loss 为 0、梯度为 0。DPO、SimPO、ORPO 都是 logistic 型，loss 随 $h$ 单调递减且永不为 0，只是梯度越来越小。`,
    },
    {
      id: 'q-kto-lambda',
      type: 'single',
      difficulty: 2,
      prompt: md`用 KTO 训练时，desirable 样本只有 undesirable 的 1/10。按论文建议 $\frac{\lambda_D n_D}{\lambda_U n_U}\in[1,1.5]$，若 λ_U = 1，λ_D 可取：`,
      options: ['13.33', '1', '0.1', '1.5'],
      answer: 0,
      explanation: md`$\lambda_D\times0.1/1\in[1,1.5]\Rightarrow\lambda_D\in[10,15]$。论文在丢弃 90% desirable 数据的实验中正是取 λ_D = 13.33、λ_U = 1。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`
        下面的 KTO loss 有 **2 处** bug。\`pi\`、\`ref\` 是每条样本回复的序列 logp，\`desirable\` 是 bool 数组，\`kl_pi\`、\`kl_ref\` 是错位配对样本的序列 logp。点出有问题的行。
      `,
      code: py`
        def kto_loss(pi, ref, desirable, kl_pi, kl_ref, beta=0.1, lam_d=1.0, lam_u=1.0):
            r = pi - ref                                   # 隐式奖励 log(π/π_ref)
            z0 = np.mean(kl_pi - kl_ref)                   # 参考点：KL 估计
            v_d = lam_d * sigmoid(beta * (r - z0))
            v_u = lam_u * sigmoid(beta * (r - z0))
            v = np.where(desirable, v_d, v_u)
            lam = np.where(desirable, lam_d, lam_u)
            return np.mean(lam - v)
      `,
      bugLines: [3, 5],
      explanation: md`
        第 3 行：论文的估计是 $\max(0,\cdot)$，应写 \`max(0.0, np.mean(kl_pi - kl_ref))\`（PyTorch 里还要 \`detach\` / \`clamp(min=0)\`）。
        第 5 行：undesirable 的价值是 $\lambda_U\sigma(\beta(z_0-r))$，符号写反了——现在等于鼓励坏样本的奖励升高。
      `,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 IPO loss（输入为四个序列 logp，形状 (B,)）。有 2 行干扰项。`,
      lines: [
        'def ipo_loss(pi_c, pi_r, ref_c, ref_r, tau=0.1):',
        '    h = (pi_c - pi_r) - (ref_c - ref_r)',
        '    target = 1.0 / (2.0 * tau)',
        '    return np.mean((h - target) ** 2)',
      ],
      distractors: ['    target = 2.0 * tau', '    return np.mean(np.logaddexp(0.0, -tau * h))'],
      explanation: md`目标是 $\tau^{-1}/2$，不是 $2\tau$；最后一行干扰项是 DPO 的 logistic 损失。`,
    },
    {
      id: 'q-online-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 online / iterative DPO，下列说法正确的有：`,
      options: [
        '每轮的偏好对由当前（或上一轮）策略采样得到，缓解离线数据的分布偏移',
        'Llama 3 的后训练共做了六轮，每轮收集新偏好数据',
        'Self-Rewarding LM 用模型自己作为 LLM-as-a-Judge 给回复打分',
        '迭代越多轮效果一定越好，打分器的偏差不会累积',
      ],
      answer: [0, 1, 2],
      explanation: md`D 错：打分器（RM 或 LLM judge）的偏差，例如偏好更长的回答，会在迭代中被反复强化（reward hacking 的离线版本）。`,
    },
    {
      id: 'q-norm',
      type: 'judge',
      difficulty: 2,
      prompt: md`SimPO 与 Tülu 3 使用的 length-normalized DPO 本质上相同，都是把奖励改成平均 logp、都不需要 reference。`,
      answer: false,
      explanation: md`length-normalized DPO 的奖励是 $\frac{\beta}{|y|}\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$，**保留 reference**，也没有 margin γ；SimPO 是 $\frac{\beta}{|y|}\log\pi_\theta(y|x)$ 再减 γ，不含 reference。`,
    },
  ],

  guided: [
    {
      id: 'g-simpo',
      title: '一步步写出 SimPO loss',
      intro: md`
        输入是已经 shift + gather 好的**逐 token logp** 与 mask（右侧 padding，padding 处是垃圾值）：
        \`logp_c, mask_c\`（chosen）与 \`logp_r, mask_r\`（rejected），形状均为 \`(B, T)\`。
        返回 \`(loss, chosen_rewards, rejected_rewards)\`，奖励为 $\frac{\beta}{|y|}\log\pi_\theta(y|x)$。
      `,
      template: py`
        import numpy as np

        def simpo_loss(logp_c, mask_c, logp_r, mask_r, beta=2.0, gamma=1.0):
            avg_c = {{1}}
            avg_r = {{2}}
            z = {{3}}
            losses = {{4}}
            return float(losses.mean()), beta * avg_c, beta * avg_r
      `,
      steps: [
        {
          blank: 1,
          instruction: md`chosen 的平均每 token logp（只在 mask 内求和，除以有效长度）。`,
          answer: 'np.where(mask_c > 0, logp_c, 0.0).sum(-1) / mask_c.sum(-1)',
          accept: ['(logp_c * mask_c).sum(-1) / mask_c.sum(-1)', '(logp_c * mask_c).sum(axis=-1) / mask_c.sum(axis=-1)'],
          explain: md`除以的是**每条样本自己的**有效长度 $|y|$。用乘 mask 也行，但若 padding 处是 inf 会得到 nan，\`np.where\` 更稳妥。`,
        },
        {
          blank: 2,
          instruction: md`rejected 同理。`,
          answer: 'np.where(mask_r > 0, logp_r, 0.0).sum(-1) / mask_r.sum(-1)',
          accept: ['(logp_r * mask_r).sum(-1) / mask_r.sum(-1)', '(logp_r * mask_r).sum(axis=-1) / mask_r.sum(axis=-1)'],
        },
        {
          blank: 3,
          instruction: md`带目标 margin 的 logit。`,
          answer: 'beta * (avg_c - avg_r) - gamma',
          accept: ['beta * avg_c - beta * avg_r - gamma'],
          explain: md`γ 在 σ 里面：只有当奖励差超过 γ，loss 才会低于 $\ln2$。`,
        },
        {
          blank: 4,
          instruction: md`数值稳定的 $-\log\sigma(z)$。`,
          answer: 'np.logaddexp(0.0, -z)',
          accept: ['np.logaddexp(0, -z)'],
          explain: md`$-\log\sigma(z)=\log(1+e^{-z})$。β 取 2.5、平均 logp 差几十时 $z$ 可达上百，直接 \`exp\` 会溢出。`,
        },
      ],
      setup: PY_TOKEN_BATCH + '\n' + py`
        def _ref_simpo(lc, mc, lr, mr, beta, gamma):
            ac, ar = _avg(lc, mc), _avg(lr, mr)
            z = beta * (ac - ar) - gamma
            return float(np.mean(np.logaddexp(0.0, -z))), beta * ac, beta * ar
      `,
      tests: [
        {
          name: '手算示例（loss = ln2）',
          code: py`
            lc = np.array([[-0.5, -0.5, -0.5, -0.5]]); mc = np.ones((1, 4))
            lr = np.array([[-1.0, -1.0, 99.0, 99.0]]); mr = np.array([[1.0, 1.0, 0.0, 0.0]])
            loss, rc, rr = simpo_loss(lc, mc, lr, mr, beta=2.0, gamma=1.0)
            assert abs(loss - np.log(2)) < 1e-9, f"期望 ln2，得到 {loss}"
            assert np.allclose(rc, [-1.0]) and np.allclose(rr, [-2.0]), f"rewards 期望 [-1] / [-2]，得到 {rc} / {rr}"
          `,
        },
        {
          name: '随机对拍（padding 垃圾值、大 β）',
          code: py`
            for s in range(5):
                lc, mc, lr, mr = _make_tok(s)
                for beta, gamma in [(2.0, 1.0), (2.5, 1.4), (50.0, 3.0)]:
                    got = simpo_loss(lc, mc, lr, mr, beta, gamma)
                    exp = _ref_simpo(lc, mc, lr, mr, beta, gamma)
                    assert np.isfinite(got[0]) and abs(got[0] - exp[0]) < 1e-9, f"seed={s} β={beta}：期望 {exp[0]:.6f}，得到 {got[0]}"
          `,
        },
      ],
      outro: md`
        把 \`avg_c\`、\`avg_r\` 换成 $\frac1{|y|}\sum(\log\pi_\theta-\log\pi_{\mathrm{ref}})$ 并去掉 γ，就是 Tülu 3 的 length-normalized DPO；
        把 sum 换回不除长度、去掉 γ、加上 reference，就回到原版 DPO。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-ipo-orpo',
      title: '手写 IPO 与 ORPO loss',
      difficulty: 'medium',
      source: '大模型岗手撕 · 偏好优化',
      timeLimitMin: 25,
      description: md`
        实现两个函数（均返回 Python float / tuple of float）：

        \`\`\`python
        def ipo_loss(pi_c, pi_r, ref_c, ref_r, tau=0.1) -> float: ...
        def orpo_loss(logp_c, mask_c, logp_r, mask_r, lam=0.1) -> tuple[float, float, float]: ...
        \`\`\`

        **IPO**：\`pi_c\` 等是形状 \`(B,)\` 的**序列** logp（回复部分之和）。$h=(\pi_c-\pi_r)-(\mathrm{ref}_c-\mathrm{ref}_r)$，返回 batch 平均的 $\big(h-\frac{1}{2\tau}\big)^2$。

        **ORPO**：输入为逐 token logp 与 mask，形状 \`(B, T)\`，右侧 padding（padding 处是垃圾值）。
        - $\bar\ell=\frac{1}{|y|}\sum_t\log P(y_t)$（平均 token logp，$\bar\ell<0$）；
        - $\log\mathrm{odds}=\bar\ell-\log(1-e^{\bar\ell})$；
        - $\mathcal L_{\mathrm{OR}}=-\log\sigma(\log\mathrm{odds}_w-\log\mathrm{odds}_l)$；
        - $\mathcal L_{\mathrm{SFT}}=-\bar\ell_w$（chosen 的平均 token NLL）；
        - 返回 \`(total, sft, or_loss)\`，其中 \`sft\`、\`or_loss\` 是 batch 平均，\`total = sft + lam * or_loss\`。

        **数值要求**：$\bar\ell$ 可能非常接近 0（如 $-10^{-12}$，模型对回复极度自信），此时 $1-e^{\bar\ell}$ 会灾难性抵消；$\log\mathrm{odds}$ 之差也可能很大。
      `,
      starter: py`
        import numpy as np

        def ipo_loss(pi_c, pi_r, ref_c, ref_r, tau=0.1):
            raise NotImplementedError

        def orpo_loss(logp_c, mask_c, logp_r, mask_r, lam=0.1):
            # 平均 logp → log odds（注意 ℓ̄ → 0⁻ 时的数值问题）→ L_OR；L_SFT = -平均 logp(chosen)
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def ipo_loss(pi_c, pi_r, ref_c, ref_r, tau=0.1):
            h = (np.asarray(pi_c) - pi_r) - (np.asarray(ref_c) - ref_r)
            return float(np.mean((h - 1.0 / (2.0 * tau)) ** 2))

        def _avg_logp(lp, mask):
            return np.where(mask > 0, lp, 0.0).sum(-1) / mask.sum(-1)

        def _log_odds(avg):
            # log(P / (1 - P)) = ℓ̄ - log(1 - e^ℓ̄)；log(1 - e^ℓ̄) = log(-expm1(ℓ̄)) 在 ℓ̄→0⁻ 时也精确
            return avg - np.log(-np.expm1(avg))

        def orpo_loss(logp_c, mask_c, logp_r, mask_r, lam=0.1):
            avg_c, avg_r = _avg_logp(logp_c, mask_c), _avg_logp(logp_r, mask_r)
            log_or = _log_odds(avg_c) - _log_odds(avg_r)
            or_loss = float(np.mean(np.logaddexp(0.0, -log_or)))          # -log σ(log OR)
            sft = float(np.mean(-avg_c))
            return sft + lam * or_loss, sft, or_loss
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，与 TRL ORPOTrainer 的写法一致）',
          code: py`
            import torch
            import torch.nn.functional as F

            def ipo_loss(pi_c, pi_r, ref_c, ref_r, tau=0.1):
                h = (pi_c - pi_r) - (ref_c - ref_r)
                return ((h - 1 / (2 * tau)) ** 2).mean()

            def orpo_loss(avg_c, avg_r, lam=0.1):
                # avg_*: 平均 token logp（由 logits + labels 求得，形状 (B,)）
                log_odds = (avg_c - avg_r) - (torch.log1p(-torch.exp(avg_c)) - torch.log1p(-torch.exp(avg_r)))
                or_loss = -F.logsigmoid(log_odds).mean()
                sft = -avg_c.mean()          # 实际实现中常直接用 chosen 的 token 级交叉熵
                return sft + lam * or_loss, sft, or_loss
            # 注：log1p(-exp(x)) 在 x→0⁻ 时精度差，更稳妥的是 torch.log(-torch.expm1(x))
          `,
        },
      ],
      explanation: md`
        **IPO**：一行公式，关键是目标 $\frac{1}{2\tau}$ 与“序列 logp 求和”的口径（与 DPO 相同）。

        **ORPO**：
        1. $P$ 是几何平均概率，所以先求平均 logp $\bar\ell$；
        2. $\log\frac{P}{1-P}=\bar\ell-\log(1-e^{\bar\ell})$。$\bar\ell=-10^{-9}$ 时 \`1 - np.exp(ℓ̄)\` 的相对误差约 $10^{-7}$，而 \`-np.expm1(ℓ̄)\` 精确；\`log1p(-exp(ℓ̄))\` 也有同样的问题；
        3. $-\log\sigma(x)$ 用 \`logaddexp(0, -x)\`；
        4. TRL 的 ORPOTrainer 中 \`beta\` 参数对应论文的 λ。

        **复杂度**：$O(B\cdot T)$。
      `,
      complexity: { time: 'O(B·T)', space: 'O(B·T)' },
      hints: [
        md`IPO：\`h = (pi_c - pi_r) - (ref_c - ref_r)\`，\`np.mean((h - 1/(2*tau))**2)\`。`,
        md`ORPO：先 \`avg = where(mask, lp, 0).sum(-1) / mask.sum(-1)\`。`,
        md`\`log_odds = avg - np.log(-np.expm1(avg))\`；\`or_loss = mean(logaddexp(0, -(lo_c - lo_r)))\`。`,
      ],
      setup: PY_TOKEN_BATCH + '\n' + py`
        from decimal import Decimal, getcontext
        getcontext().prec = 50

        def _ref_ipo(pc, pr, rc, rr, tau):
            h = (pc - pr) - (rc - rr)
            return float(np.mean((h - 1 / (2 * tau)) ** 2))

        def _hp_log_odds(a):
            # 高精度参考：log(P/(1-P))，P = exp(a)
            P = Decimal(float(a)).exp()
            return float((P / (1 - P)).ln())

        def _ref_orpo(lc, mc, lr, mr, lam):
            ac, ar = _avg(lc, mc), _avg(lr, mr)
            lo = np.array([_hp_log_odds(a) - _hp_log_odds(b) for a, b in zip(ac, ar)])
            orl = float(np.mean(np.logaddexp(0.0, -lo)))
            sft = float(np.mean(-ac))
            return sft + lam * orl, sft, orl

        def _cmp3(got, exp, tag="", rtol=1e-7):
            assert isinstance(got, tuple) and len(got) == 3, "orpo_loss 应返回 (total, sft, or_loss)"
            for name, g, e in zip(("total", "sft", "or_loss"), got, exp):
                assert np.isfinite(g), f"{tag}{name} 不是有限数：{g}"
                assert abs(g - e) <= rtol * max(1.0, abs(e)), f"{tag}{name}：期望 {e:.9f}，得到 {g:.9f}"
      `,
      tests: [
        {
          name: 'IPO 手算',
          code: py`
            got = ipo_loss(np.array([-10.0]), np.array([-12.0]), np.array([-11.0]), np.array([-11.5]), tau=0.5)
            assert abs(got - 0.25) < 1e-12, f"h=1.5、目标 1 → 0.25，得到 {got}"
          `,
        },
        {
          name: 'ORPO 手算（P_w=0.8, P_l=0.5）',
          code: py`
            lc = np.log(np.array([[0.8, 0.8, 0.8]])); mc = np.ones((1, 3))
            lr = np.log(np.array([[0.5, 0.5, 7.0]])); mr = np.array([[1.0, 1.0, 0.0]])
            total, sft, orl = orpo_loss(lc, mc, lr, mr, lam=0.1)
            assert abs(orl - np.log(1.25)) < 1e-9, f"L_OR 期望 log(1.25)=0.2231，得到 {orl}"
            assert abs(sft + np.log(0.8)) < 1e-9, f"L_SFT 期望 -ln0.8=0.2231，得到 {sft}"
            assert abs(total - (sft + 0.1 * orl)) < 1e-12, "total 应为 sft + lam * or_loss"
          `,
        },
        {
          name: '随机对拍（padding 垃圾值）',
          code: py`
            rng = np.random.default_rng(0)
            pc, pr, rc, rr = (rng.normal(-20, 5, size=6) for _ in range(4))
            for tau in (0.05, 0.1, 1.0):
                assert abs(ipo_loss(pc, pr, rc, rr, tau) - _ref_ipo(pc, pr, rc, rr, tau)) < 1e-9, f"IPO τ={tau}"
            for s in range(4):
                lc, mc, lr, mr = _make_tok(s)
                _cmp3(orpo_loss(lc, mc, lr, mr, 0.2), _ref_orpo(lc, mc, lr, mr, 0.2), f"seed={s}: ")
          `,
        },
        {
          name: '数值稳定：平均 logp ≈ −1e-12',
          hidden: true,
          code: py`
            # chosen 与 rejected 都几乎是确定输出（平均 logp ∈ [-1e-12, -1e-15]）：log odds 都很大，二者之差决定 L_OR
            rng = np.random.default_rng(5)
            a = -10 ** rng.uniform(-15, -12, size=6); b = -10 ** rng.uniform(-15, -12, size=6)
            lc = np.repeat(a[:, None], 3, axis=1); lr = np.repeat(b[:, None], 3, axis=1); m = np.ones((6, 3))
            _cmp3(orpo_loss(lc, m, lr, m, 0.1), _ref_orpo(lc, m, lr, m, 0.1), "平均 logp 极接近 0（1 - exp 灾难性抵消？用 expm1）：", rtol=1e-6)
          `,
        },
        {
          name: '数值稳定：log OR 很大或很负',
          hidden: true,
          code: py`
            lc = np.full((2, 4), -1e-12); mc = np.ones((2, 4))
            lr = np.full((2, 4), -40.0); mr = np.ones((2, 4))
            got = orpo_loss(lc, mc, lr, mr, 0.1)
            _cmp3(got, _ref_orpo(lc, mc, lr, mr, 0.1), "chosen 远好于 rejected：", rtol=1e-6)
            got2 = orpo_loss(lr, mr, lc, mc, 0.1)
            _cmp3(got2, _ref_orpo(lr, mr, lc, mc, 0.1), "chosen 远差于 rejected：", rtol=1e-6)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-orpo-vs-dpo',
          q: md`ORPO 为什么能不要 reference？它靠什么防止模型跑偏？`,
          a: md`
            reference 在 DPO 中的作用是 KL 锚点与 $Z(x)$ 的消去；ORPO 不是从 KL 约束 RL 推出来的，而是“SFT + 对比惩罚”的启发式组合：$\mathcal L_{\mathrm{SFT}}$ 持续把 chosen 的似然往上拉，相当于用数据分布本身当锚，λ 控制对比项的强度。代价是没有显式的“别离初始模型太远”的约束，λ 过大或训练过久会损伤生成质量。
          `,
        },
      ],
    },
    {
      id: 'c-kto',
      title: '手写 KTO loss（含参考点 z₀ 的估计）',
      difficulty: 'medium',
      source: '大模型岗手撕 · KTO',
      timeLimitMin: 20,
      description: md`
        \`\`\`python
        def kto_loss(pi, ref, desirable, kl_pi, kl_ref, beta=0.1, lam_d=1.0, lam_u=1.0) -> tuple[float, float]: ...
        \`\`\`

        - \`pi\`、\`ref\`：形状 \`(B,)\`，每条 (x, y) 回复部分的序列 logp（policy / reference）；
        - \`desirable\`：\`(B,)\` bool，样本是否为“好”样本；
        - \`kl_pi\`、\`kl_ref\`：\`(m,)\`，**错位配对** $(x_i, y_{i+1})$ 的序列 logp（policy / reference），用于估计参考点；
        - $\hat z_0=\max\big(0,\ \mathrm{mean}(\mathrm{kl\_pi}-\mathrm{kl\_ref})\big)$；
        - $r=\mathrm{pi}-\mathrm{ref}$；好样本 $v=\lambda_D\sigma(\beta(r-\hat z_0))$，坏样本 $v=\lambda_U\sigma(\beta(\hat z_0-r))$；
        - 返回 \`(loss, z0)\`，\`loss\` 为 batch 内 $\lambda_y-v$ 的平均。

        要求数值稳定（$\beta|r-z_0|$ 可能上千）。
      `,
      starter: py`
        import numpy as np

        def kto_loss(pi, ref, desirable, kl_pi, kl_ref, beta=0.1, lam_d=1.0, lam_u=1.0):
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def _sigmoid(x):
            return np.exp(-np.logaddexp(0.0, -x))          # 稳定的 σ(x)

        def kto_loss(pi, ref, desirable, kl_pi, kl_ref, beta=0.1, lam_d=1.0, lam_u=1.0):
            z0 = max(0.0, float(np.mean(np.asarray(kl_pi) - kl_ref)))   # 截断到 ≥ 0；PyTorch 中还要 detach
            r = np.asarray(pi) - ref
            d = np.asarray(desirable, dtype=bool)
            v = np.where(d, lam_d * _sigmoid(beta * (r - z0)), lam_u * _sigmoid(beta * (z0 - r)))
            lam = np.where(d, lam_d, lam_u)
            return float(np.mean(lam - v)), z0
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，与 TRL KTOTrainer 的写法一致）',
          code: py`
            import torch

            def kto_loss(pi, ref, desirable, kl_pi, kl_ref, beta=0.1, lam_d=1.0, lam_u=1.0):
                kl = (kl_pi - kl_ref).mean().clamp(min=0).detach()      # 多卡时先 all_gather 再取均值
                r = pi - ref
                chosen = 1 - torch.sigmoid(beta * (r[desirable] - kl))
                rejected = 1 - torch.sigmoid(beta * (kl - r[~desirable]))
                losses = torch.cat([lam_d * chosen, lam_u * rejected])
                return losses.mean(), kl
          `,
        },
      ],
      explanation: md`
        **思路**：$\lambda_y-v=\lambda_D\big(1-\sigma(\beta(r-z_0))\big)$（好样本）或 $\lambda_U\big(1-\sigma(\beta(z_0-r))\big)$（坏样本），与 TRL 的实现完全一致。

        **易错点**：
        - $z_0$ 要截断到 $\ge0$ 且不回传梯度；
        - 坏样本是 $\sigma(\beta(z_0-r))$，符号与好样本相反；
        - 错位配对的目的是得到“与当前 prompt 无关的典型回复”的 log-ratio，从而估计 $\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})$；需要额外前向，micro-batch 至少 2 条；
        - 稳定的 σ：\`exp(-logaddexp(0, -x))\`。

        **复杂度**：$O(B+m)$（不含前向）。
      `,
      complexity: { time: 'O(B + m)', space: 'O(B)' },
      hints: [
        md`\`z0 = max(0.0, mean(kl_pi - kl_ref))\`。`,
        md`好样本 \`v = lam_d * σ(β(r - z0))\`，坏样本 \`v = lam_u * σ(β(z0 - r))\`。`,
        md`\`loss = mean(lam_y - v)\`，σ 用 \`exp(-logaddexp(0, -x))\` 避免溢出。`,
      ],
      setup: py`
        import numpy as np

        def _ref_kto(pi, ref, d, kp, kr, beta, ld, lu):
            z0 = max(0.0, float(np.mean(kp - kr)))
            out = []
            for p, q, good in zip(pi, ref, d):
                r = p - q
                x = beta * (r - z0) if good else beta * (z0 - r)
                s = 1.0 / (1.0 + np.exp(-x)) if x >= 0 else np.exp(x) / (1.0 + np.exp(x))
                out.append((ld if good else lu) * (1 - s))
            return float(np.mean(out)), z0

        def _make_kto(seed, B=8, m=8, scale=5.0):
            rng = np.random.default_rng(seed)
            pi = rng.normal(-30, scale, size=B); ref = pi + rng.normal(0, 2, size=B)
            d = rng.random(B) < 0.5
            kp = rng.normal(-35, scale, size=m); kr = kp + rng.normal(-0.3, 1, size=m)
            return pi, ref, d, kp, kr
      `,
      tests: [
        {
          name: '手算：r = z0 时 loss = λ/2',
          code: py`
            pi = np.array([-10.0, -12.0]); ref = np.array([-11.0, -13.0])     # r = 1
            kp = np.array([-20.0, -30.0]); kr = np.array([-21.0, -31.0])      # z0 = 1
            loss, z0 = kto_loss(pi, ref, np.array([True, False]), kp, kr, beta=0.1, lam_d=1.0, lam_u=3.0)
            assert abs(z0 - 1.0) < 1e-12, f"z0 期望 1，得到 {z0}"
            assert abs(loss - (0.5 + 1.5) / 2) < 1e-12, f"loss 期望 1.0，得到 {loss}"
          `,
        },
        {
          name: 'z0 截断到 0',
          code: py`
            kp = np.array([-20.0, -30.0]); kr = np.array([-18.0, -29.0])      # 均值为负
            loss, z0 = kto_loss(np.array([-5.0]), np.array([-5.0]), np.array([True]), kp, kr)
            assert z0 == 0.0, f"KL 估计为负时应截断为 0，得到 {z0}"
            assert abs(loss - 0.5) < 1e-12, f"r = z0 = 0 时好样本 loss = 0.5，得到 {loss}"
          `,
        },
        {
          name: '坏样本的方向：奖励越低 loss 越小',
          code: py`
            kp = np.zeros(2); kr = np.zeros(2)
            hi, _ = kto_loss(np.array([5.0]), np.array([0.0]), np.array([False]), kp, kr)
            lo, _ = kto_loss(np.array([-5.0]), np.array([0.0]), np.array([False]), kp, kr)
            assert lo < hi, f"坏样本 r=-5 的 loss（{lo:.4f}）应小于 r=+5（{hi:.4f}）——符号写反了？"
          `,
        },
        {
          name: '随机对拍（不同 λ、β）',
          hidden: true,
          code: py`
            for s, (beta, ld, lu) in enumerate([(0.1, 1.0, 1.0), (0.5, 13.33, 1.0), (0.05, 1.0, 1.5)]):
                args = _make_kto(s)
                got = kto_loss(*args, beta=beta, lam_d=ld, lam_u=lu)
                exp = _ref_kto(*args, beta, ld, lu)
                assert abs(got[0] - exp[0]) < 1e-9 and abs(got[1] - exp[1]) < 1e-9, f"seed={s}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '数值稳定：β|r − z0| 上千、全是同一类样本',
          hidden: true,
          code: py`
            pi = np.array([-1.0, -3000.0, 2000.0]); ref = np.zeros(3)
            for d in (np.array([True] * 3), np.array([False] * 3)):
                got = kto_loss(pi, ref, d, np.zeros(2), np.zeros(2), beta=1.0)
                exp = _ref_kto(pi, ref, d, np.zeros(2), np.zeros(2), 1.0, 1.0, 1.0)
                assert np.isfinite(got[0]) and abs(got[0] - exp[0]) < 1e-9, f"期望 {exp[0]}，得到 {got[0]}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-kto-when',
          q: md`什么时候选 KTO 而不是 DPO？`,
          a: md`
            ① 数据天然是单条反馈（点赞/点踩、客服满意度、规则判定对错），构造成对需要丢弃大量数据或人为配对；② 好坏样本极不平衡；③ 标注噪声大——KTO 的饱和使极端样本的梯度变小。若已有高质量成对数据，DPO 的对比信号更直接。KTO 仍需要 reference model，并且每步要额外前向错位样本估计 $z_0$。
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-ipo',
      freq: 2,
      q: md`IPO 相比 DPO 解决了什么问题？`,
      a: md`
        **结论**：DPO 在偏好接近确定（数据中某对总是 $y_w\succ y_l$）时会把 margin 推向无穷、令 $\pi(y_l)\to0$，KL 约束形同虚设；IPO 用平方损失 $(h-\frac1{2\tau})^2$ 给 margin 一个有限目标。

        **原因**：DPO/RLHF 等价于 ΨPO 中 Ψ = logit，概率 1 映射到无穷大奖励差；有限数据里每对只标一次，经验偏好就是 0/1。IPO 取 Ψ 为恒等映射，避免了这个映射，τ 真正控制与 reference 的距离。

        **延伸**：IPO 对所有样本用同一个目标 margin，不区分偏好强弱；实践中 DPO 的过拟合也常靠早停、β、label smoothing 缓解。
      `,
    },
    {
      id: 'qa-simpo',
      freq: 2,
      q: md`SimPO 为什么可以不要 reference model？有什么风险？`,
      a: md`
        **结论**：SimPO 直接用长度归一化的 $\frac{\beta}{|y|}\log\pi_\theta(y|x)$ 当奖励，配合目标 margin γ，不再需要 $\pi_{\mathrm{ref}}$。

        **原因**：作者认为 DPO 的隐式奖励与生成时的平均似然不一致，而平均似然正是解码时决定输出的量；长度归一化去掉了长度偏置。省掉 reference 带来显存与前向的节省。

        **风险**：没有 KL 锚点，模型可能偏离初始分布（遗忘、生成退化）；对学习率、β、γ 更敏感（论文建议 β 2.0–2.5、γ 0.5–1.5）。如果想保留 reference 又要长度归一化，可以用 length-normalized DPO（Tülu 3，β = 5）。
      `,
    },
    {
      id: 'qa-kto',
      freq: 2,
      q: md`KTO 的核心思想是什么？参考点 z₀ 怎么算？`,
      a: md`
        **结论**：用前景理论的价值函数直接最大化“生成结果的效用”，只需要单条好/坏标签。

        **原因**：人对收益与损失的感受相对参考点、且损失厌恶。KTO 令 $r_\theta=\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}$，好样本 $v=\lambda_D\sigma(\beta(r_\theta-z_0))$，坏样本 $v=\lambda_U\sigma(\beta(z_0-r_\theta))$，最小化 $\lambda_y-v$。$z_0$ 估计 $\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})$：batch 内错位配对 $(x_i,y_{i+1})$ 的 log-ratio 均值，截断到 $\ge0$，不回传梯度。

        **延伸**：建议 $\frac{\lambda_Dn_D}{\lambda_Un_U}\in[1,1.5]$；论文显示丢掉 90% 好样本（λ_D = 13.33）仍优于 DPO。
      `,
    },
    {
      id: 'qa-online',
      freq: 2,
      q: md`为什么要做 online / iterative DPO？它和 PPO 还差什么？`,
      a: md`
        **结论**：离线 DPO 的数据分布与当前策略不一致；每轮用当前策略采样、打分、构造新偏好对，可以缓解分布偏移，通常显著优于单轮离线 DPO。

        **原因**：DPO 的隐式奖励只在数据覆盖的回答上被约束；策略移动后，它真正会生成的回答可能从未出现在训练对里。iterative 版本让数据跟着策略走（Llama 3 共六轮；Self-Rewarding LM 用模型自评迭代三轮）。

        **差距**：每轮内部仍是离线分类损失，没有逐 token 的信用分配，也不能像 PPO 那样在一轮里多次利用新样本；打分器的偏差会被迭代放大。可验证任务上，这条路线已经演化为 RLVR。
      `,
    },
  ],

  cheatsheet: md`
    | 方法 | 损失 | ref | 数据 |
    |---|---|---|---|
    | DPO | $-\log\sigma(\beta h)$，$h=\log\frac{\pi(y_w)}{\pi_{\mathrm{ref}}(y_w)}-\log\frac{\pi(y_l)}{\pi_{\mathrm{ref}}(y_l)}$ | ✅ | 成对 |
    | IPO | $(h-\frac{1}{2\tau})^2$ | ✅ | 成对 |
    | SimPO | $-\log\sigma\big(\frac{\beta}{\lvert y_w\rvert}\log\pi(y_w)-\frac{\beta}{\lvert y_l\rvert}\log\pi(y_l)-\gamma\big)$ | ❌ | 成对 |
    | KTO | $\lambda_y-v$，$v=\lambda_D\sigma(\beta(r-z_0))$ / $\lambda_U\sigma(\beta(z_0-r))$ | ✅ | 单条 |
    | ORPO | $\mathcal L_{\mathrm{SFT}}+\lambda\cdot(-\log\sigma(\log\frac{\mathrm{odds}_w}{\mathrm{odds}_l}))$，$P=e^{\bar\ell}$ | ❌ | 成对 |

    \`\`\`python
    h = (pi_c - pi_r) - (ref_c - ref_r); ipo = ((h - 1/(2*tau))**2).mean()
    z = beta * (avg_c - avg_r) - gamma; simpo = logaddexp(0, -z).mean()
    log_odds = avg - log(-expm1(avg))            # ORPO，avg→0⁻ 也稳定
    z0 = max(0, mean(kl_pi - kl_ref))            # KTO 参考点，错位配对，不回传梯度
    \`\`\`

    - SimPO：β 2.0–2.5、γ 0.5–1.5（Llama-3-Instruct 2.5 / 1.4）；Tülu 3 length-normalized DPO 保留 ref，β = 5
    - KTO：β = 0.1，$\frac{\lambda_Dn_D}{\lambda_Un_U}\in[1,1.5]$；KT 价值函数 α = 0.88、λ = 2.25
    - ORPO：λ = 0.1（Mistral）/ 0.2（Llama-2）/ 0.25（Phi-2）
    - online/iterative DPO：Llama 3 六轮；Self-Rewarding（Llama 2 70B，三轮）；OAIF 在线 AI 标注
  `,
})
