/**
 * llm-rl-gspo：GSPO（arXiv 2507.18071）、CISPO（MiniMax-M1 arXiv 2506.13585）、训推不一致与 TIS、REINFORCE++
 */
import { defineLevel, md, py } from '@/content/helpers'
import { SeqRatioViz } from '../viz/SeqRatioViz'
import { CispoViz } from '../viz/CispoViz'
import { MismatchViz } from '../viz/MismatchViz'
import { PY_MAKE_BATCH, PY_REF_ADV } from '../shared'

const PY_REF_GSPO = py`
  def _ref_gspo_loss(logp_new, logp_old, rewards, group_ids, mask, eps_low=3e-4, eps_high=4e-4, adv_eps=1e-6):
      m = np.asarray(mask, dtype=np.float64)
      A = _ref_group_adv(rewards, group_ids, "group", adv_eps)
      lr = np.where(m > 0, logp_new - logp_old, 0.0)
      s = np.exp(lr.sum(axis=1) / m.sum(axis=1))
      surr1 = s * A
      surr2 = np.clip(s, 1 - eps_low, 1 + eps_high) * A
      loss = float(np.mean(-np.minimum(surr1, surr2)))
      return loss, {"seq_ratio": s, "clip_frac": float(np.mean(surr2 < surr1))}

  def _ref_cispo(logp_new, logp_old, adv, mask, eps_high=0.2, eps_low=None):
      m = np.asarray(mask, dtype=np.float64)
      ratio = np.exp(np.where(m > 0, logp_new - logp_old, 0.0))
      lo = -np.inf if eps_low is None else 1 - eps_low
      w = np.clip(ratio, lo, 1 + eps_high)
      A = np.asarray(adv, dtype=np.float64)[:, None]
      n = m.sum()
      lp = np.where(m > 0, logp_new, 0.0)
      loss = float(-(w * A * lp * m).sum() / n)
      grad = -(w * A) * m / n
      return loss, grad

  def _ref_tis_loss(logp_new, logp_old_train, logp_infer, adv, mask, clip_eps=0.2, C=2.0):
      m = np.asarray(mask, dtype=np.float64)
      A = np.asarray(adv, dtype=np.float64)[:, None]
      ratio = np.exp(np.where(m > 0, logp_new - logp_old_train, 0.0))
      tis = np.minimum(np.exp(np.where(m > 0, logp_old_train - logp_infer, 0.0)), C)
      surr = np.minimum(ratio * A, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)
      return float(-(tis * surr * m).sum() / m.sum())
`

const SETUP = PY_MAKE_BATCH + '\n\n' + PY_REF_ADV + '\n\n' + PY_REF_GSPO

export default defineLevel({
  learn: [
    {
      id: 'illposed',
      title: 'GSPO 的批评：token 级重要性权重是“病态”的',
      body: md`
        Qwen 团队的 GSPO（Group Sequence Policy Optimization，2025-07）从重要性采样（IS）的原理出发质疑 GRPO：

        $$
        \mathbb E_{z\sim\pi_{tar}}[f(z)]=\mathbb E_{z\sim\pi_{beh}}\Big[\frac{\pi_{tar}(z)}{\pi_{beh}(z)}f(z)\Big]
        $$

        IS 能纠正分布差异，前提是对**很多样本**取平均。GRPO 在每个位置 $t$ 用的权重 $\frac{\pi_\theta(y_{i,t}\mid\cdot)}{\pi_{\theta_{old}}(y_{i,t}\mid\cdot)}$
        只基于该位置分布下的**一个样本**，起不到分布校正的作用，反而给梯度注入高方差噪声；噪声沿长序列累积，又被 clip 放大。
        论文报告这会导致**往往不可逆的模型崩溃**：回退 checkpoint、调 clip 范围、加长生成长度、换数据都救不回来。

        由此得出一条原则：**优化的单位应与奖励的单位一致**。奖励给的是整条序列，那么 off-policy 校正与裁剪也应该在序列级做。

        | | GRPO | GSPO |
        |---|---|---|
        | 重要性比 | 逐 token $w_{i,t}=\pi_\theta/\pi_{old}$ | 序列级 $s_i=(\pi_\theta(y_i)/\pi_{old}(y_i))^{1/\lvert y_i\rvert }$ |
        | 裁剪单位 | token | 整条回答 |
        | token 梯度权重 | 各不相同，A>0 时在 $(0,1+\varepsilon]$，A<0 时在 $[1-\varepsilon,\infty)$ | 同一回答内全部相同 |
      `,
      checkpoint: {
        id: 'cp-ill',
        type: 'single',
        prompt: md`GSPO 认为 GRPO 的 token 级重要性权重“病态”的核心原因是：`,
        options: [
          '它的计算需要额外的价值模型',
          '每个位置只有一个样本，权重起不到分布校正作用，反而引入随长度累积的高方差噪声',
          '它无法处理 padding',
          '它与 KL 正则冲突',
        ],
        answer: 1,
        explanation: md`IS 校正依赖对大量样本取平均；GRPO 在每个位置只用一个采样 token 的比值，等于给梯度乘了一个随机噪声，长序列上累积并被 clip 放大。其它选项与论文论证无关。`,
      },
    },
    {
      id: 'gspo',
      title: 'GSPO：序列级重要性比与序列级裁剪',
      body: md`
        GSPO 的目标（式 5–7）：

        $$
        \mathcal J_{GSPO}=\mathbb E\Big[\frac1G\sum_{i=1}^{G}\min\big(s_i(\theta)\hat A_i,\ \mathrm{clip}(s_i(\theta),1-\varepsilon,1+\varepsilon)\hat A_i\big)\Big],\quad
        s_i(\theta)=\Big(\frac{\pi_\theta(y_i\mid x)}{\pi_{\theta_{old}}(y_i\mid x)}\Big)^{\frac1{|y_i|}}=\exp\Big(\frac1{|y_i|}\sum_{t}\log\frac{\pi_\theta(y_{i,t}\mid\cdot)}{\pi_{\theta_{old}}(y_{i,t}\mid\cdot)}\Big)
        $$

        优势仍是组内 $(r-\text{mean})/\text{std}$。$1/|y_i|$ 的**长度归一化**（几何平均）是为了降方差、并让不同长度的回答共用一个裁剪范围；
        否则几个 token 的变化就会让整条序列的比值剧烈波动。因为定义不同，GSPO 的裁剪范围与 GRPO 差几个数量级：
        论文实验用左右 **3e-4 / 4e-4**，对比的 GRPO 基线用 0.2 / 0.27。

        **梯度**（略去 clip）：$\nabla\mathcal J=\mathbb E\big[\frac1G\sum_i s_i\hat A_i\cdot\frac1{|y_i|}\sum_t\nabla\log\pi_\theta(y_{i,t})\big]$——
        同一回答里**所有 token 等权**，消除了 GRPO 中逐 token 权重不均带来的不稳定。

        **反直觉的观察**：GSPO 被裁剪的 token 比例比 GRPO **高两个数量级**（论文图 2：约 0.15 vs 0.0013），用来估计梯度的 token 更少，训练效率却更高——
        说明 GRPO 的 token 级梯度本身噪声很大。
      `,
      viz: SeqRatioViz,
      checkpoint: {
        id: 'cp-gspo',
        type: 'numeric',
        prompt: md`一条 4 个 token 的回答，逐 token 的 $\log(\pi_\theta/\pi_{old})$ 为 $[0.1,-0.1,0.3,0.1]$。GSPO 的 $s_i$ 是多少？`,
        answer: 1.10517,
        tolerance: 0.002,
        explanation: md`$s_i=\exp(\frac{0.1-0.1+0.3+0.1}{4})=\exp(0.1)=1.105$。若忘了 $1/|y_i|$，得到 $\exp(0.4)=1.49$——序列越长偏离越夸张，这就是要做长度归一化的原因。`,
      },
    },
    {
      id: 'moe',
      title: '为什么对 MoE 更稳：专家激活的波动',
      body: md`
        MoE 的稀疏激活带来 RL 特有的问题：一次梯度更新之后，**同一条回答**在新旧策略下激活的专家可能明显不同。
        论文的数据：48 层的 Qwen3-30B-A3B-Base，每次 RL 梯度更新后，同一 rollout 样本约有 **10%** 的激活专家发生变化（越深的 MoE 越明显）。
        token 级比值 $w_{i,t}$ 因此剧烈波动、失去意义，GRPO 无法正常收敛。

        - **Routing Replay**（Qwen 之前的做法）：缓存 $\pi_{old}$ 激活的专家，计算 $\pi_\theta$ 时“回放”同样的路由，保证分子分母走同一个子网络。
          能收敛，但有额外显存与通信开销，还限制了模型的有效容量。
        - **GSPO**：只依赖序列似然 $\pi_\theta(y_i\mid x)$，而 MoE 模型的语言建模能力整体是稳定的，序列似然不会剧烈波动——**不再需要 Routing Replay**。

        **基础设施上的好处**：训练引擎（如 Megatron）与推理引擎（SGLang、vLLM）有精度差异，通常要用训练引擎重算 $\pi_{old}$。
        序列级似然对精度差异更宽容，GSPO 使直接使用推理引擎返回的似然成为可能，这在 partial rollout、多轮 RL、训推分离的框架里尤其有用。

        **GSPO-token**：多轮场景想要逐 token 调优势时，用 $s_{i,t}=\mathrm{sg}[s_i]\cdot\frac{\pi_\theta(y_{i,t})}{\mathrm{sg}[\pi_\theta(y_{i,t})]}$。
        它的**数值**恒等于 $s_i$，梯度方向是 $\nabla\log\pi_\theta(y_{i,t})$；所有 token 优势相同时与 GSPO 完全等价。
      `,
      after: md`
        > [!NEW] DeepSeek 的做法：Keep Routing
        > DeepSeek-V3.2 的 RL 选择了另一条路：把推理框架采样时的**专家路由路径保存下来，训练时强制使用同样的路由**，保证优化的是同一组专家参数。
        > 论文称这对 MoE 的 RL 稳定性“至关重要”，并且从 DeepSeek-V3-0324 起就在用。思路与 Routing Replay 相同，但覆盖了“训推引擎不一致”这个来源。
      `,
      checkpoint: {
        id: 'cp-moe',
        type: 'multi',
        prompt: md`关于 MoE 上的 RL 稳定性，正确的是：（多选）`,
        options: [
          'Qwen3-30B-A3B 每次梯度更新后，同一样本约 10% 的激活专家会变化',
          'Routing Replay 让 π_θ 回放 π_old 的路由，使 token 级 ratio 重新稳定',
          'GSPO 仍然必须配合 Routing Replay 才能收敛',
          'GSPO-token 的 s_{i,t} 在数值上等于 s_i',
        ],
        answer: [0, 1, 3],
        explanation: md`C 错：GSPO 的卖点正是去掉了对 Routing Replay 的依赖。D 对：$\pi_\theta/\mathrm{sg}[\pi_\theta]$ 的数值恒为 1，只是保留了梯度通路。`,
      },
    },
    {
      id: 'cispo',
      title: 'CISPO：裁剪 IS 权重并 stop-gradient，保留所有 token 的梯度',
      body: md`
        MiniMax-M1 在混合注意力架构上做 zero-RL 时发现 GRPO 学不出长 CoT。消融后定位到 **clip**：
        “However / Recheck / Wait / Aha” 这类反思 token 是推理路径上的“分叉点”，在 base 模型里概率很低，更新时 $r_{i,t}$ 很大，
        **第一次 on-policy 更新后就被裁剪掉**，在后续 off-policy 更新中再也不贡献梯度（他们每个生成 batch 做 16 轮 off-policy 更新）。
        DAPO 的 Clip-Higher 在他们的设定里不够用。

        CISPO（Clipped IS-weight Policy Optimization）从带 IS 校正的 REINFORCE 出发，**裁剪权重而不是裁剪更新**：

        $$
        \mathcal J_{CISPO}=\mathbb E\Big[\frac{1}{\sum_i|o_i|}\sum_{i}\sum_{t}\mathrm{sg}\big(\hat r_{i,t}(\theta)\big)\,\hat A_{i,t}\,\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\Big],\quad
        \hat r_{i,t}=\mathrm{clip}\big(r_{i,t},\,1-\varepsilon^{IS}_{low},\,1+\varepsilon^{IS}_{high}\big)
        $$

        - 优势用 GRPO 的组内相对优势，聚合用 token 级（同 DAPO）；**不加 KL**；同时沿用 DAPO 的动态采样与长度惩罚；
        - 实验中**不设下界**（$\varepsilon^{IS}_{low}$ 取很大），只调 $\varepsilon^{IS}_{high}$；
        - 不裁剪权重时就是标准策略梯度；裁剪使梯度略有偏，但保留了所有 token 的贡献、降低方差。

        论文还给了统一形式：在 CISPO 上乘一个 token 掩码 $M_{i,t}$，当 $\hat A>0$ 且 $r>1+\varepsilon_{high}$、或 $\hat A<0$ 且 $r<1-\varepsilon_{low}$ 时 $M=0$——
        这正是 PPO 可信域隐含的掩码，于是 PPO-clip 与 CISPO 可在一个框架里切换。
      `,
      viz: CispoViz,
      after: md`
        **效果**：Qwen2.5-32B base 上的 zero-RL 对比（数学数据集同 DAPO，AIME 2024），CISPO 在相同步数下优于 DAPO 与 GRPO，
        **用 50% 的步数达到 DAPO 的性能**。整个 MiniMax-M1 的 RL 在 512 张 H800 上用了三周，论文报告租用成本约 53.47 万美元。
      `,
      checkpoint: {
        id: 'cp-cispo',
        type: 'judge',
        prompt: md`在 CISPO 中，$r_{i,t}$ 超过 $1+\varepsilon^{IS}_{high}$ 的 token 梯度为 0。`,
        answer: false,
        explanation: md`CISPO 只把**权重**截到 $1+\varepsilon^{IS}_{high}$ 并 stop-gradient，梯度 $\mathrm{sg}(\hat r)\hat A\nabla\log\pi_\theta$ 仍然存在。梯度为 0 是 PPO-clip 的行为（对应统一形式中 $M=0$ 的情形）。`,
      },
    },
    {
      id: 'mismatch',
      title: '训推不一致与截断重要性采样（TIS）',
      body: md`
        现代 RL 框架用推理引擎（vLLM / SGLang）做 rollout，用训练框架（FSDP / Megatron）算梯度。**同一份权重**在两边算出的 token 概率并不相同，
        这让名义上 on-policy 的训练**悄悄变成了 off-policy**。Yao、Liu 等（OPT 2025 workshop）的观测：某些 token 甚至出现 $\pi_{vllm}=1$ 而 $\pi_{fsdp}=0$；
        把 vLLM 的 lm_head 改成 fp32、暴露真实采样概率之后，不一致**依然存在**；用 INT8 / FP8 量化 rollout 时更严重。

        **修正**：把梯度 $\mathbb E_{a\sim\pi_{vllm}}[R\,\nabla\log\pi_{fsdp}]$ 改为截断重要性采样

        $$
        \mathbb E_{a\sim\pi_{vllm}(\theta_{old})}\Big[\min\Big(\frac{\pi_{fsdp}(a,\theta_{old})}{\pi_{vllm}(a,\theta_{old})},\,C\Big)\cdot\nabla_\theta\min\Big(\frac{\pi_{fsdp}(a,\theta)}{\pi_{fsdp}(a,\theta_{old})}\hat A,\ \mathrm{clip}(\cdot)\hat A\Big)\Big]
        $$

        即 PPO 的 ratio 仍用训练引擎重算的 $\pi_{old}$，外面再乘一个**截断的训推比值**（$C$ 是超参，各框架默认值不同）。论文在 DAPO-32B 设置上报告 TIS 显著提升效果，并让 INT8 量化 rollout 接近 BF16 的效果。

        其它一手做法：
        - **MiniMax-M1**：逐层排查发现 LM head 的高幅值激活是误差主要来源，把 **LM head 提到 FP32**，训推概率相关系数从约 0.9x 提升到 0.99x；
        - **DeepSeek-V3.2 Off-Policy Sequence Masking**：若 $\hat A_i<0$ 且 $\frac{1}{|o_i|}\sum_t\log\frac{\pi_{old}}{\pi_\theta}>\delta$（$\pi_{old}$ 取推理框架返回的采样概率），整条序列 mask 掉——只屏蔽**高偏离的负样本**；
        - **DeepSeek-V3.2 Keep Sampling Mask**：top-p / top-k 截断让 $\pi_{old}$ 与 $\pi_\theta$ 的动作空间不一致，于是保存采样时的截断掩码、训练时同样施加；
        - **FP16**（Qi 等，2025-10）：认为根因是 BF16 的舍入误差，训练与推理都换成 FP16 即可基本消除不一致。

        下图是一个**简化模拟**：只对推理端的 logits 做 BF16 / FP16 舍入，看采样 token 的概率差与 TIS 权重。
      `,
      viz: MismatchViz,
      checkpoint: {
        id: 'cp-tis',
        type: 'numeric',
        prompt: md`某采样 token：训练引擎重算 $\pi_{fsdp}(a,\theta_{old})=0.30$，推理引擎 $\pi_{vllm}(a,\theta_{old})=0.05$，$C=2$。TIS 权重是多少？`,
        answer: 2,
        tolerance: 0,
        explanation: md`比值 $0.30/0.05=6$，超过上限，截断为 $C=2$。截断引入少量偏差，换来有界的方差——这是 TIS 与“完整 IS”的取舍。`,
      },
    },
    {
      id: 'reinforcepp',
      title: 'REINFORCE++：不分组，做全局优势归一化',
      body: md`
        REINFORCE++（Hu 等，arXiv 2501.03262）同样去掉 critic，但批评 GRPO 的**组内**归一化：分子分母不独立，组小时 std 接近 0 会让优势爆炸，
        还会“只奖励比同组好，而不是绝对好”，在简单题上过拟合。两个变体：

        | 变体 | 优势 | KL |
        |---|---|---|
        | REINFORCE++（$k=1$，每题一个样本） | $A_{q,o_t}=r(o_{1:T},q)-\beta\sum_{i=t}^{T}\mathrm{KL}(i)$，然后在**全局 batch** 上 $(A-\text{mean})/(\text{std}+\epsilon)$ | k1 形式，逐 token 并入奖励 |
        | REINFORCE++ w/ Baseline（$k>1$） | 先减**组均值**，再用**全局 batch** 的统计量归一化 | 单独的 loss 项，用 **k2** |

        选 k2 的理由在附录中有推导：样本来自 $\pi_\theta$ 时，k2 当 loss 的梯度等于反向 KL 的梯度；k1 当 loss 梯度不含 $\pi_{ref}$、没有约束作用；k3 是不稳定的近似（与 GRPO 一关的推导一致）。
        它等价于“去掉 critic、GAE 取 $\lambda=\gamma=1$、用两步全局归一化当 baseline 的 PPO”。
      `,
      checkpoint: {
        id: 'cp-rpp',
        type: 'single',
        prompt: md`REINFORCE++ w/ Baseline 的优势计算是：`,
        options: ['组内减均值、组内除 std', '组内减均值、全局 batch 除 std', '全局减均值、组内除 std', '只减全局均值'],
        answer: 1,
        explanation: md`两步：先减组均值（对奖励尺度鲁棒），再用全局 batch 的 mean/std 归一化（稳定、无组内 std 的难度偏差）。A 是 GRPO。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-def',
      type: 'single',
      difficulty: 1,
      prompt: md`GSPO 的序列级重要性比 $s_i$ 定义为：`,
      options: [
        md`$\frac{\pi_\theta(y_i\mid x)}{\pi_{old}(y_i\mid x)}$`,
        md`$\big(\frac{\pi_\theta(y_i\mid x)}{\pi_{old}(y_i\mid x)}\big)^{1/|y_i|}$`,
        md`$\frac1{|y_i|}\sum_t\frac{\pi_\theta(y_{i,t})}{\pi_{old}(y_{i,t})}$`,
        md`$\max_t\frac{\pi_\theta(y_{i,t})}{\pi_{old}(y_{i,t})}$`,
      ],
      answer: 1,
      explanation: md`是序列似然比的几何平均（log 空间的算术平均）。A 不做长度归一化，方差随长度爆炸；C 是 token 比值的算术平均，不是论文定义。`,
    },
    {
      id: 'q-clipfrac',
      type: 'judge',
      difficulty: 2,
      prompt: md`GSPO 训练中被裁剪的 token 比例比 GRPO 低得多，这是它更高效的原因。`,
      answer: false,
      explanation: md`相反：GSPO 被裁剪的 token 比例高两个数量级（约 0.15 vs 0.0013），却更高效。论文据此认为 GRPO 的 token 级梯度估计噪声大、利用样本低效。`,
    },
    {
      id: 'q-range',
      type: 'fill',
      difficulty: 2,
      prompt: md`GSPO 论文实验中，序列级裁剪的左、右范围分别为 ____(1)____ 与 ____(2)____（填小数或科学计数法）。`,
      blanks: [
        { label: '(1)', answers: ['3e-4'], numeric: true, tolerance: 0.001 },
        { label: '(2)', answers: ['4e-4'], numeric: true, tolerance: 0.001 },
      ],
      explanation: md`3e-4 与 4e-4；对比的 GRPO 基线为 0.2 与 0.27。由于 $s_i$ 是几何平均，正常更新下它离 1 极近，裁剪范围自然小几个数量级。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把算法与它施加在梯度上的“权重”配对：`,
      pairs: [
        ['GRPO / PPO', '逐 token 的 r，越界则梯度为 0'],
        ['GSPO', '同一回答共用的几何平均比值 s_i'],
        ['CISPO', 'sg(clip(r))，越界也保留梯度'],
        ['TIS', 'min(π_train/π_infer, C)'],
      ],
      explanation: md`四者分别处理：新旧策略的 token 级差异、序列级差异、裁剪造成的梯度丢失、训推引擎之间的差异。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`$\varepsilon=0.2$。下列 token 中，哪些在 PPO-clip 下**梯度为 0**，但在 CISPO（只设上界 $1+\varepsilon^{IS}_{high}=1.2$）下**仍有梯度**？`,
      items: ['A>0, r=1.5', 'A>0, r=0.6', 'A<0, r=0.6', 'A<0, r=1.5', 'A>0, r=1.1'],
      answer: [0, 2],
      explanation: md`PPO 中被裁掉的是 ① (A>0, r>1.2) 与 ③ (A<0, r<0.8)；CISPO 对它们的权重分别为 1.2 与 0.6（不设下界），梯度保留。②④⑤ 在 PPO 中本来就有梯度。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 GSPO 实现有 **2 处** bug，点出来：`,
      code: py`
        def gspo_loss(logp_new, logp_old, adv, mask, eps_low=3e-4, eps_high=4e-4):
            # logp_*: (B, T)，adv: (B,)
            log_ratio = (logp_new - logp_old) * mask
            s = np.exp(log_ratio.sum(axis=1))
            surr1 = s * adv
            surr2 = np.clip(s, 1 - eps_low, 1 + eps_high) * adv
            per_seq = -np.minimum(surr1, surr2)
            return per_seq.sum() / mask.sum()
      `,
      bugLines: [4, 8],
      explanation: md`第 4 行缺长度归一化：应为 \`np.exp(log_ratio.sum(axis=1) / mask.sum(axis=1))\`。第 8 行：GSPO 是对**回答**平均（$\frac1G\sum_i$），应为 \`per_seq.mean()\`；除以 token 总数会把长度偏差又引入回来。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 CISPO 的 loss（\`adv\` 为 \`(B, 1)\`；有 2 行干扰项）：`,
      lines: [
        'def cispo_loss(logp_new, logp_old, adv, mask, eps_high):',
        '    ratio = np.exp(logp_new - logp_old)',
        '    w = np.minimum(ratio, 1 + eps_high)        # 只截上界，视为常数（sg）',
        '    obj = w * adv * logp_new',
        '    return -(obj * mask).sum() / mask.sum()',
      ],
      distractors: ['    obj = np.minimum(ratio * adv, w * adv)', '    return -(obj * mask).sum(axis=1).mean()'],
      explanation: md`CISPO 的目标是 $\mathrm{sg}(\hat r)\hat A\log\pi_\theta$，不是 PPO 式的 min；聚合用 token 级（除以 token 总数）。在 PyTorch 里 \`w\` 需要 \`.detach()\`。`,
    },
    {
      id: 'q-tis',
      type: 'single',
      difficulty: 3,
      prompt: md`TIS 的截断权重 $\min(\cdot,C)$ 中，比值的分子与分母分别是：`,
      options: [
        'π_fsdp(a, θ) / π_fsdp(a, θ_old)',
        'π_fsdp(a, θ_old) / π_vllm(a, θ_old)',
        'π_vllm(a, θ_old) / π_fsdp(a, θ_old)',
        'π_θ(a) / π_ref(a)',
      ],
      answer: 1,
      explanation: md`TIS 校正的是“采样分布（推理引擎）→ 训练引擎下的旧策略”这一层差异，所以是 $\pi_{fsdp}(\theta_{old})/\pi_{vllm}(\theta_{old})$。A 是 PPO 自身的 ratio（仍在 clip 里）；D 是 KL 相关的比值。`,
    },
    {
      id: 'q-m1',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 MiniMax-M1 的 RL，正确的是：（多选）`,
      options: [
        'CISPO 在 Qwen2.5-32B 上用约 50% 的步数达到 DAPO 的效果',
        '把 LM head 提到 FP32，显著提高了训推概率的相关性',
        'CISPO 带有 KL 惩罚项',
        '实验中 CISPO 只调 ε_high，不设下界',
      ],
      answer: [0, 1, 3],
      explanation: md`C 错：论文明确 CISPO 没有 KL 项（与 DAPO 等一致）。其余均为论文原文结论。`,
    },
    {
      id: 'q-tis-num',
      type: 'numeric',
      difficulty: 2,
      prompt: md`某 token：$\log\pi_{fsdp}(\theta_{old})=-1.2$，$\log\pi_{vllm}(\theta_{old})=-1.5$，$C=2$。TIS 权重是多少？（保留 3 位小数）`,
      answer: 1.35,
      tolerance: 0.005,
      explanation: md`$\exp(-1.2-(-1.5))=e^{0.3}=1.3499<2$，未被截断，权重 1.350。`,
    },
    {
      id: 'q-rpp',
      type: 'judge',
      difficulty: 2,
      prompt: md`REINFORCE++ 认为组内 std 归一化在组很小时可能让优势爆炸，因此改用全局 batch 统计量归一化。`,
      answer: true,
      explanation: md`组小（如 4、8）且奖励相近时组内 std 接近 0，优势被放大；全局 batch（上千样本）的 mean/std 接近常数，更稳定。`,
    },
  ],

  guided: [
    {
      id: 'g-tis',
      title: '给 PPO-clip 加上截断重要性采样（TIS）',
      intro: md`
        rollout 来自推理引擎，日志概率为 \`logp_infer\`；训练引擎在**同一份旧权重**下重算得到 \`logp_old_train\`；当前策略为 \`logp_new\`。
        实现带 TIS 的 PPO-clip loss（token 级聚合）：PPO 的 ratio 用训练引擎的新旧比值，外面乘截断权重 $\min(\pi_{train}^{old}/\pi_{infer},\,C)$。
        \`adv\` 为 \`(B,)\`，其余 \`(B, T)\`，padding 位置无垃圾值。
      `,
      template: py`
        import numpy as np

        def tis_ppo_loss(logp_new, logp_old_train, logp_infer, adv, mask, clip_eps=0.2, C=2.0):
            A = adv[:, None]
            ratio = {{1}}                          # PPO ratio：训练引擎的新 / 旧
            tis = {{2}}                            # 截断的训推比值（常数，不求梯度）
            surr = {{3}}
            return {{4}}                           # token 级聚合，取负
      `,
      steps: [
        {
          blank: 1,
          instruction: md`PPO 的比值只在训练引擎内部比较：$\pi_{fsdp}(\theta)/\pi_{fsdp}(\theta_{old})$。`,
          answer: 'np.exp(logp_new - logp_old_train)',
          explain: md`分母用**训练引擎重算**的旧策略，而不是推理引擎的概率——否则训推差异会混进 PPO 的可信域判断（论文中的 “PPO-IS” 变体效果更差）。`,
        },
        {
          blank: 2,
          instruction: md`TIS 权重 $\min(\pi_{train}^{old}/\pi_{infer},\ C)$。`,
          answer: 'np.minimum(np.exp(logp_old_train - logp_infer), C)',
          accept: ['np.clip(np.exp(logp_old_train - logp_infer), None, C)', 'np.minimum(C, np.exp(logp_old_train - logp_infer))'],
          explain: md`只截上界：比值很大的 token（推理引擎严重低估）不会主导梯度。截断带来少量偏差，换来有界方差。`,
        },
        {
          blank: 3,
          instruction: md`标准 clipped surrogate。`,
          answer: 'np.minimum(ratio * A, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)',
          explain: md`TIS 乘在 clip 之外，不改变可信域的判断。`,
        },
        {
          blank: 4,
          instruction: md`乘上 TIS 权重和 mask，除以有效 token 数，取负。`,
          answer: '-(tis * surr * mask).sum() / mask.sum()',
          accept: ['-((tis * surr) * mask).sum() / mask.sum()', '-np.sum(tis * surr * mask) / np.sum(mask)'],
          explain: md`在 PyTorch 中 \`tis\` 由不带梯度的张量算出（或显式 \`.detach()\`），只作为权重。`,
        },
      ],
      setup: SETUP,
      tests: [
        {
          name: '训推一致时退化为普通 PPO',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(0, garbage=False)
            adv = _ref_group_adv(r, gid)
            got = float(tis_ppo_loss(ln, lo, lo, adv, m))
            exp = _ref_tis_loss(ln, lo, lo, adv, m)
            assert abs(got - exp) < 1e-10, f"期望 {exp:.10f}，得到 {got:.10f}"
          `,
        },
        {
          name: '有训推差异时对拍（含截断）',
          code: py`
            for seed in range(6):
                ln, lo, lr, r, gid, m = _make_batch(seed, garbage=False)
                adv = _ref_group_adv(r, gid)
                linf = lo + np.random.default_rng(seed).normal(0, 0.8, size=lo.shape)
                for C in (1.5, 3.0):
                    got = float(tis_ppo_loss(ln, lo, linf, adv, m, C=C))
                    exp = _ref_tis_loss(ln, lo, linf, adv, m, C=C)
                    assert abs(got - exp) < 1e-10, f"seed={seed}, C={C}: 期望 {exp:.10f}，得到 {got:.10f}"
          `,
        },
      ],
      outro: md`
        这就是 verl / TRL 中 “vLLM importance sampling correction” 一类选项的核心。更激进的做法是 **MIS**（超出范围直接 mask 掉）或序列级的比值（DeepSeek-V3.2 的 off-policy sequence masking 也属于这一思路）。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-gspo',
      title: '手写 GSPO loss',
      difficulty: 'medium',
      source: 'Qwen GSPO · 大模型岗手撕',
      timeLimitMin: 18,
      description: md`
        \`\`\`python
        def gspo_loss(logp_new, logp_old, rewards, group_ids, mask,
                      eps_low=3e-4, eps_high=4e-4, adv_eps=1e-6) -> tuple[float, dict]: ...
        \`\`\`

        - \`logp_new / logp_old / mask\`：\`(B, T)\`，padding 位置是垃圾值；\`rewards / group_ids\`：\`(B,)\`（可乱序，每组 ≥ 2）。
        - 优势：组内 $(r-\text{mean})/(\text{std}_{ddof=1}+\texttt{adv\_eps})$。
        - $s_i=\exp\big(\frac1{|y_i|}\sum_{t\in\text{valid}}(\log\pi_{new}-\log\pi_{old})\big)$。
        - loss $=-\frac1B\sum_i\min\big(s_i\hat A_i,\ \mathrm{clip}(s_i,1-\varepsilon_{low},1+\varepsilon_{high})\hat A_i\big)$。

        返回 \`(loss, stats)\`，\`stats\` 含 \`"seq_ratio"\`（\`(B,)\`）与 \`"clip_frac"\`（被裁剪的**序列**比例，即 \`surr2 < surr1\` 的行占比）。
      `,
      starter: py`
        import numpy as np

        def gspo_loss(logp_new, logp_old, rewards, group_ids, mask,
                      eps_low=3e-4, eps_high=4e-4, adv_eps=1e-6):
            # TODO: 组内优势 → 序列级几何平均比值 → 序列级 clip → 对 B 条回答平均
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def gspo_loss(logp_new, logp_old, rewards, group_ids, mask,
                      eps_low=3e-4, eps_high=4e-4, adv_eps=1e-6):
            m = np.asarray(mask, dtype=np.float64)
            r = np.asarray(rewards, dtype=np.float64)
            _, inv = np.unique(group_ids, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv).astype(np.float64)
            dev = r - (np.bincount(inv, weights=r) / cnt)[inv]
            std = np.sqrt(np.bincount(inv, weights=dev ** 2) / (cnt - 1))
            A = dev / (std[inv] + adv_eps)

            # 序列级比值：log 空间按有效长度取平均（几何平均），padding 先清零
            log_ratio = np.where(m > 0, logp_new - logp_old, 0.0)
            s = np.exp(log_ratio.sum(axis=1) / m.sum(axis=1))

            surr1 = s * A
            surr2 = np.clip(s, 1 - eps_low, 1 + eps_high) * A
            loss = float(np.mean(-np.minimum(surr1, surr2)))       # 1/G Σ_i，再对组平均
            return loss, {"seq_ratio": s, "clip_frac": float(np.mean(surr2 < surr1))}
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，连续分组）',
          code: py`
            import torch

            def gspo_loss_torch(logp_new, logp_old, rewards, mask, G, eps_low=3e-4, eps_high=4e-4):
                mask = mask.float()
                r = rewards.view(-1, G)
                A = ((r - r.mean(-1, keepdim=True)) / (r.std(-1, keepdim=True) + 1e-6)).reshape(-1).detach()
                log_ratio = (logp_new - logp_old.detach()) * mask
                s = torch.exp(log_ratio.sum(-1) / mask.sum(-1))              # (B,)
                surr = torch.min(s * A, torch.clamp(s, 1 - eps_low, 1 + eps_high) * A)
                return -surr.mean()

            # GSPO-token：s_it = sg(s_i) * π_θ(y_t) / sg(π_θ(y_t))
            # s_it = s.detach()[:, None] * torch.exp(logp_new - logp_new.detach())
          `,
        },
      ],
      explanation: md`
        与 GRPO 的差异：ratio 先在序列内做 log 平均再 exp；clip 与 min 在**序列**上做；聚合是对回答求平均（不再有 token 维）。

        **易错点**：忘了除以 $|y_i|$（长序列的 $s$ 会离 1 很远、几乎全被裁）；用 \`mean(axis=1)\` 把 padding 也算进长度；把 GRPO 的 0.2 当成 GSPO 的裁剪范围。

        复杂度 $O(BT)$。
      `,
      hints: [
        md`\`log_ratio = np.where(mask > 0, logp_new - logp_old, 0)\`，\`s = exp(log_ratio.sum(1) / mask.sum(1))\`。`,
        md`\`surr1 = s * A\`、\`surr2 = clip(s, 1 - eps_low, 1 + eps_high) * A\`，loss 对 B 行取平均。`,
      ],
      setup: SETUP,
      tests: [
        {
          name: 'on-policy：s ≡ 1、loss = 0',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(0)
            loss, st = gspo_loss(lo, lo, r, gid, m)
            assert np.allclose(st["seq_ratio"], 1.0), f"logp_new == logp_old 时 s 应全为 1，得到 {st['seq_ratio']}"
            assert abs(loss) < 1e-9 and st["clip_frac"] == 0, f"得到 loss={loss}, clip_frac={st['clip_frac']}"
          `,
        },
        {
          name: '几何平均（含长度归一化）',
          code: py`
            lo = np.zeros((2, 4)); ln = np.array([[0.1, -0.1, 0.3, 0.1], [0.2, 0.2, 9.0, 9.0]])
            m = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
            _, st = gspo_loss(ln, lo, np.array([1.0, 0.0]), np.array([0, 0]), m)
            exp = np.exp([0.1, 0.2])
            assert np.allclose(st["seq_ratio"], exp), f"期望 {exp}，得到 {st['seq_ratio']}（检查长度归一化与 padding）"
          `,
        },
        {
          name: '随机对拍（含 padding 垃圾值、小偏移）',
          hidden: true,
          code: py`
            for seed in range(30):
                ln, lo, lr, r, gid, m = _make_batch(200 + seed, n_groups=4, G=4, T=9)
                ln = lo + 0.002 * (ln - lo)              # 让 s 落在 1±1e-3 附近，裁剪有进有出
                loss, st = gspo_loss(ln, lo, r, gid, m)
                el, es = _ref_gspo_loss(ln, lo, r, gid, m)
                assert abs(loss - el) < 1e-10, f"seed={seed}: 期望 {el:.10f}，得到 {loss:.10f}"
                assert abs(st["clip_frac"] - es["clip_frac"]) < 1e-12, f"seed={seed}: clip_frac 期望 {es['clip_frac']}，得到 {st['clip_frac']}"
          `,
        },
        {
          name: '乱序分组与自定义裁剪范围',
          hidden: true,
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(77, n_groups=3, G=5, T=8)
            ln = lo + 0.01 * (ln - lo)
            perm = np.random.default_rng(0).permutation(len(r))
            ids = np.array([9, 4, 100])[gid]
            args = (ln[perm], lo[perm], r[perm], ids[perm], m[perm])
            for el_, eh_ in [(3e-4, 4e-4), (0.01, 0.02)]:
                loss, _ = gspo_loss(*args, eps_low=el_, eps_high=eh_)
                el, _ = _ref_gspo_loss(*args, eps_low=el_, eps_high=eh_)
                assert abs(loss - el) < 1e-10, f"eps=({el_},{eh_}): 期望 {el:.10f}，得到 {loss:.10f}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-gspo-token',
          q: md`多轮 agent 场景想给不同 token 不同优势，GSPO 怎么改？`,
          a: md`用 GSPO-token：$s_{i,t}=\mathrm{sg}[s_i]\cdot\pi_\theta(y_{i,t})/\mathrm{sg}[\pi_\theta(y_{i,t})]$，目标对 token 平均 $\frac1{|y_i|}\sum_t\min(s_{i,t}\hat A_{i,t},\mathrm{clip}(s_{i,t})\hat A_{i,t})$。数值上 $s_{i,t}=s_i$，裁剪条件仍按序列判断，但每个 token 可以有自己的优势。`,
        },
      ],
    },
    {
      id: 'c-cispo',
      title: 'CISPO：loss 与对 log π 的梯度',
      difficulty: 'medium',
      source: 'MiniMax-M1 · CISPO',
      timeLimitMin: 15,
      description: md`
        \`\`\`python
        def cispo_loss_and_grad(logp_new, logp_old, adv, mask, eps_high=0.2, eps_low=None) -> tuple[float, np.ndarray]: ...
        \`\`\`

        - \`logp_new / logp_old / mask\`：\`(B, T)\`（padding 是垃圾值）；\`adv\`：\`(B,)\`，同一回答的 token 共享。
        - $\hat r=\mathrm{clip}(r,\,1-\varepsilon_{low},\,1+\varepsilon_{high})$；\`eps_low=None\` 表示**不设下界**（MiniMax-M1 的做法）。
        - loss $=-\frac{1}{N}\sum_{i,t}\mathrm{sg}(\hat r_{i,t})\,\hat A_i\,\log\pi_{new}(o_{i,t})$，$N$ 为有效 token 总数。
        - 同时返回 $\partial\,\text{loss}/\partial\log\pi_{new}$，形状 \`(B, T)\`，padding 为 0。**注意 sg**：$\hat r$ 视为常数。
      `,
      starter: py`
        import numpy as np

        def cispo_loss_and_grad(logp_new, logp_old, adv, mask, eps_high=0.2, eps_low=None):
            # TODO: 截断 IS 权重（视为常数）→ token 级 REINFORCE 目标 → 梯度
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def cispo_loss_and_grad(logp_new, logp_old, adv, mask, eps_high=0.2, eps_low=None):
            m = np.asarray(mask, dtype=np.float64)
            ratio = np.exp(np.where(m > 0, logp_new - logp_old, 0.0))
            lo = -np.inf if eps_low is None else 1.0 - eps_low
            w = np.clip(ratio, lo, 1.0 + eps_high)                 # sg(clipped IS weight)
            A = np.asarray(adv, dtype=np.float64)[:, None]
            n = m.sum()
            lp = np.where(m > 0, logp_new, 0.0)
            loss = float(-(w * A * lp * m).sum() / n)
            grad = -(w * A) * m / n                                  # w 是常数：d/dlogp (w·A·logp) = w·A
            return loss, grad
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def cispo_loss_torch(logp_new, logp_old, adv, mask, eps_high=0.2):
                mask = mask.float()
                ratio = torch.exp(logp_new - logp_old)
                w = torch.clamp(ratio, max=1 + eps_high).detach()       # 关键：stop-gradient
                obj = w * adv.unsqueeze(-1) * logp_new
                return -(obj * mask).sum() / mask.sum()
          `,
        },
      ],
      explanation: md`
        CISPO 的梯度是 $-\frac1N\mathrm{sg}(\hat r)\hat A\,\nabla\log\pi$：**每个有效 token 都有梯度**，只是权重被截断。
        若忘了 stop-gradient、直接对 $\hat r\cdot\hat A\cdot\log\pi$ 求导，会多出 $\hat A\log\pi\cdot\partial\hat r/\partial\log\pi$ 一项（未截断区域等于 $r\hat A\log\pi$），梯度方向就错了。

        **对比**：PPO-clip 对 A>0 且 $r>1+\varepsilon$ 的 token 梯度为 0；CISPO 给它 $(1+\varepsilon_{high})\hat A$ 的权重。
      `,
      hints: [md`\`w = np.clip(ratio, lo, 1 + eps_high)\`，\`eps_low is None\` 时 \`lo = -np.inf\`。`, md`梯度：\`-(w * A) * mask / N\`，不要对 \`w\` 求导。`],
      setup: SETUP,
      tests: [
        {
          name: '越界 token 仍有梯度',
          code: py`
            lo = np.zeros((1, 3)); ln = np.log(np.array([[1.0, 2.0, 0.5]]))
            m = np.ones((1, 3))
            loss, g = cispo_loss_and_grad(ln, lo, np.array([1.0]), m, eps_high=0.2)
            exp_g = -np.array([[1.0, 1.2, 0.5]]) / 3
            assert np.allclose(g, exp_g), f"期望梯度 {exp_g}，得到 {g}"
            exp_l = -(1.0 * ln[0, 0] + 1.2 * ln[0, 1] + 0.5 * ln[0, 2]) / 3
            assert abs(loss - exp_l) < 1e-12, f"期望 loss {exp_l}，得到 {loss}"
          `,
        },
        {
          name: '设置下界',
          code: py`
            lo = np.zeros((1, 2)); ln = np.log(np.array([[0.5, 1.5]]))
            _, g = cispo_loss_and_grad(ln, lo, np.array([-1.0]), np.ones((1, 2)), eps_high=0.2, eps_low=0.2)
            assert np.allclose(g, np.array([[0.8, 1.2]]) / 2), f"得到 {g}"
          `,
        },
        {
          name: '随机对拍（含 padding）',
          hidden: true,
          code: py`
            for seed in range(20):
                ln, lo, lr, r, gid, m = _make_batch(400 + seed, n_groups=3, G=4, T=8)
                ln = lo + 3 * (ln - lo)
                adv = _ref_group_adv(r, gid)
                for eh, el in [(0.2, None), (1.0, None), (0.28, 0.2)]:
                    loss, g = cispo_loss_and_grad(ln, lo, adv, m, eh, el)
                    e_loss, e_g = _ref_cispo(ln, lo, adv, m, eh, el)
                    assert abs(loss - e_loss) < 1e-10, f"seed={seed}: loss 期望 {e_loss}，得到 {loss}"
                    assert np.allclose(g, e_g, atol=1e-12), f"seed={seed}: 梯度最大误差 {np.abs(g - e_g).max():.3e}"
          `,
        },
        {
          name: 'stop-gradient：梯度与数值微分（固定权重）一致',
          hidden: true,
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(9, n_groups=2, G=3, T=5)
            adv = _ref_group_adv(r, gid)
            _, g = cispo_loss_and_grad(ln, lo, adv, m, 0.2)
            assert np.all(g[m == 0] == 0), "padding 位置梯度应为 0"
            full_ratio_grad = -(np.exp(ln - lo) * adv[:, None]) * (1 + ln) * m / m.sum()
            assert not np.allclose(g, full_ratio_grad), "看起来你对 IS 权重也求了导——CISPO 要 stop-gradient"
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-gspo',
      freq: 3,
      q: md`GSPO 和 GRPO 的区别是什么？为什么对 MoE 更稳？`,
      a: md`
        **结论**：GSPO 把重要性比、裁剪和优化都放到序列级：$s_i$ 是序列似然比的几何平均，同一回答所有 token 等权。

        **原因**：GRPO 的 token 级比值只基于单个样本，起不到 IS 校正作用，噪声随长度累积并被 clip 放大，可能导致不可逆崩溃。
        MoE 每次更新后同一样本约 10% 的激活专家会变（Qwen3-30B-A3B），token 级比值剧烈波动，GRPO 需要 Routing Replay 才能收敛；序列似然整体稳定，GSPO 不需要。

        **延伸**：裁剪范围 3e-4 / 4e-4；被裁剪 token 比例高两个数量级但效率更高；对训推精度差异更宽容；多轮场景用 GSPO-token。DeepSeek-V3.2 则用 Keep Routing（训练强制使用采样时的路由）。
      `,
    },
    {
      id: 'qa-cispo',
      freq: 2,
      q: md`CISPO 解决了什么问题？与 PPO-clip、DAPO 的 Clip-Higher 有何不同？`,
      a: md`
        **结论**：PPO-clip 会丢掉越界 token 的梯度，而低概率的反思 token（Wait、However）正是最容易越界的；CISPO 改为截断 IS 权重并 stop-gradient，保留全部 token 的梯度。

        **原因**：MiniMax-M1 每个生成 batch 做 16 轮 off-policy 更新，这些 token 第一次更新后就被裁掉；Clip-Higher 只是放宽上界，在他们的设定里不够。

        **延伸**：目标是 $\mathrm{sg}(\hat r)\hat A\log\pi$、token 级聚合、无 KL、只调 $\varepsilon_{high}$；Qwen2.5-32B 上 50% 步数追平 DAPO。统一形式里加掩码 $M_{i,t}$ 即可退回 PPO 的行为。
      `,
    },
    {
      id: 'qa-mismatch',
      freq: 3,
      q: md`什么是训推不一致？怎么缓解？`,
      a: md`
        **结论**：rollout 用推理引擎、梯度用训练引擎，同一权重下两边的 token 概率不同，on-policy 训练实际变成了 off-policy。

        **原因**：kernel 实现、累加顺序、精度（BF16 舍入、FP8/INT8 量化 rollout）、MoE 路由差异、top-p 截断都会导致差异；LM head 的高幅值激活是常见误差源。

        **延伸**：① 算法侧：TIS $\min(\pi_{train}/\pi_{infer},C)$、MIS（超出范围直接 mask）、序列级比值 / 掩码（GSPO、DeepSeek-V3.2 只屏蔽高偏离的负样本）；② 系统侧：LM head 用 FP32（MiniMax-M1）、全程 FP16（Qi 等 2025）、Keep Routing、Keep Sampling Mask；③ 监控训推 logprob 的差异或相关系数。
      `,
    },
    {
      id: 'qa-rpp',
      freq: 2,
      q: md`REINFORCE++ 的核心思想是什么？`,
      a: md`
        **结论**：去掉 critic，用**全局 batch** 的优势归一化代替 GRPO 的组内归一化。

        **原因**：组内 std 在组小时不稳、存在难度偏差，且只奖励“比同组好”，容易在简单题上过拟合；全局 batch 统计量接近常数，偏差随 batch 增大而消失。

        **延伸**：$k=1$ 版本把 k1 形式的 KL 逐 token 并入奖励；w/ Baseline 版本先减组均值再全局归一化，KL 作为单独 loss 用 k2（其梯度对应反向 KL）。等价于 GAE 取 $\lambda=\gamma=1$、无 critic 的 PPO。
      `,
    },
  ],

  cheatsheet: md`
    | 算法 | 权重 / 比值 | 裁剪 | 聚合 |
    |---|---|---|---|
    | GRPO | token $r_{i,t}$ | token 级，越界梯度 0 | 行内平均 |
    | GSPO | $s_i=\exp(\frac1{\lvert y_i\rvert }\sum_t\log r_{i,t})$ | 序列级，3e-4 / 4e-4 | 回答平均 |
    | CISPO | $\mathrm{sg}(\mathrm{clip}(r,\cdot,1+\varepsilon_{high}))$ | 只截权重，不丢 token | token 级 |
    | TIS | $\min(\pi_{train}^{old}/\pi_{infer},C)$ | 乘在 PPO 外面 | — |

    \`\`\`python
    s = np.exp((np.where(m > 0, lp_new - lp_old, 0)).sum(1) / m.sum(1))       # GSPO
    loss_gspo = -np.minimum(s * A, np.clip(s, 1 - 3e-4, 1 + 4e-4) * A).mean()
    w = np.minimum(np.exp(lp_new - lp_old), 1 + eps_high)                       # CISPO（PyTorch 要 detach）
    loss_cispo = -(w * A[:, None] * lp_new * m).sum() / m.sum()
    \`\`\`
    - MoE：Qwen3-30B-A3B 每步约 10% 激活专家变化 → GRPO 需 Routing Replay；GSPO 不需要；DeepSeek-V3.2 用 Keep Routing
    - MiniMax-M1：LM head FP32；CISPO 50% 步数追平 DAPO；512×H800 三周
    - REINFORCE++：全局优势归一化；w/ Baseline 用 k2 KL loss
  `,
})
