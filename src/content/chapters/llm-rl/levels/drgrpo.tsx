/**
 * llm-rl-drgrpo：Understanding R1-Zero-Like Training（Dr.GRPO，arXiv 2503.20783）——长度偏差、难度偏差与 loss 聚合
 */
import { defineLevel, md, py } from '@/content/helpers'
import { AggregationViz } from '../viz/AggregationViz'
import { StdBiasViz } from '../viz/StdBiasViz'
import { GroupAdvantageNoStdViz } from '../viz/GroupAdvantageViz'
import { PY_MAKE_BATCH, PY_REF_ADV } from '../shared'

const PY_REF_DR = py`
  def _ref_agg(tok, mask, mode, max_len=None):
      m = np.asarray(mask, dtype=np.float64)
      x = np.asarray(tok, dtype=np.float64) * m
      x = np.where(m > 0, x, 0.0)
      if mode == "token-mean":
          return float(x.sum() / m.sum())
      if mode == "seq-mean-token-mean":
          return float(np.mean(x.sum(axis=1) / m.sum(axis=1)))
      if mode == "seq-mean-token-sum":
          return float(np.mean(x.sum(axis=1)))
      if mode == "seq-mean-token-sum-norm":
          L = m.shape[1] if max_len is None else max_len
          return float(np.mean(x.sum(axis=1)) / L)
      raise ValueError(mode)

  def _ref_adv_all(rewards, group_ids, mode, eps=1e-6):
      if mode in ("group", "none", "batch"):
          return _ref_group_adv(rewards, group_ids, mode, eps)
      r = np.asarray(rewards, dtype=np.float64)
      _, inv = np.unique(np.asarray(group_ids), return_inverse=True)
      inv = inv.reshape(-1)
      cnt = np.bincount(inv).astype(np.float64)
      s = np.bincount(inv, weights=r)
      return r - (s[inv] - r) / (cnt[inv] - 1)        # rloo

  def _ref_drgrpo_loss(logp_new, logp_old, rewards, group_ids, mask, max_tokens, clip_eps=0.2):
      m = np.asarray(mask, dtype=np.float64)
      A = _ref_group_adv(rewards, group_ids, "none")[:, None]
      ratio = np.exp(np.where(m > 0, logp_new - logp_old, 0.0))
      tok = -np.minimum(ratio * A, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)
      return _ref_agg(tok, m, "seq-mean-token-sum-norm", max_tokens)
`

const SETUP = PY_MAKE_BATCH + '\n\n' + PY_REF_ADV + '\n\n' + PY_REF_DR

export default defineLevel({
  learn: [
    {
      id: 'recap',
      title: '把 GRPO 的“有效优势”写出来，偏差就藏在里面',
      body: md`
        「Understanding R1-Zero-Like Training: A Critical Perspective」（Sea AI Lab 等，COLM 2025）把 GRPO 目标里每个 token 实际受到的权重写成
        一个“有效优势”。记无偏的中心化优势 $\tilde A_i = R(q,o_i)-\mathrm{mean}(R)$，GRPO 中 token $(i,t)$ 的权重是：

        $$
        a_{i,t}=\frac{1}{G}\cdot\frac{\tilde A_i}{\mathrm{std}(R)\cdot|o_i|}
        $$

        相比 PPO 目标（每个 token 的权重就是 $\tilde A_i$，配合一个常数归一化），多出来的两个因子各引入一种偏差：

        | 因子 | 偏差 | 后果 |
        |---|---|---|
        | $1/\lvert o_i\rvert $ | **回答级长度偏差** | 答对时短回答更新更大（偏好简短的正确答案）；答错时长回答**罚得更轻**（偏好冗长的错误答案） |
        | $1/\mathrm{std}(R)$ | **题目级难度偏差** | std 小的题（几乎全对或全错）权重被放大 |

        论文的修法极其简单：**两个都删掉**，叫 Dr. GRPO（GRPO Done Right）。下面先回顾不除 std 时优势长什么样（切换上方 Segmented 即可对比 GRPO）。
      `,
      viz: GroupAdvantageNoStdViz,
      after: md`
        > [!NOTE] 与奖励缩放的关系
        > 去掉 std 后优势随奖励**尺度**线性变化（1/0 与 ±1 差 2 倍），相当于改了学习率；GRPO 的 std 归一化对尺度不敏感。这是 Dr.GRPO 唯一的“代价”，用学习率吸收即可。
      `,
      checkpoint: {
        id: 'cp-recap',
        type: 'multi',
        prompt: md`Dr. GRPO 相对 GRPO 删掉了哪些项？（多选）`,
        options: ['优势中的 1/std(R)', '目标中的 1/|o_i|', 'clip 操作', '组内均值 baseline'],
        answer: [0, 1],
        explanation: md`只删两个归一化因子。clip 保留（仍是 PPO 式目标）；组内均值 baseline 保留——它保证了优势无偏且方差更小（等价于 RLOO 的缩放）。`,
      },
    },
    {
      id: 'length-bias',
      title: '长度偏差：错误回答“变长”就能稀释惩罚',
      body: md`
        $\frac1{|o_i|}\sum_t$ 让**每条回答的总推力相同**，于是每个 token 的推力与长度成反比：

        - $\tilde A_i>0$（答对）：越短的回答每个 token 被推得越多 → 偏好简短的正确答案；
        - $\tilde A_i<0$（答错）：越长的回答每个 token 被罚得越少 → **偏好冗长的错误答案**。

        论文图 5 的证据：GRPO 与 Dr.GRPO 的回答长度都随奖励上涨（与 R1-Zero 的趋势一致），但奖励增长放缓后 GRPO 的长度仍在持续增长，
        而且增长主要来自**错误回答**；Dr.GRPO 显著压低了错误回答的长度、提升了 token 效率，准确率不降。
        论文据此提醒：常被解读为“长 CoT 涌现”的长度增长，**一部分可能只是优化偏差**。

        拖动下图中错误长回答 o4 的长度，在三种聚合方式之间切换。
      `,
      viz: AggregationViz,
      after: md`
        > [!WARN] 这个偏差在 PPO 实现里也有
        > 论文检查了 trl、OpenRLHF、verl、SimpleRL-Zero、Open-Reasoner-Zero 的 PPO loss，**全部**按回答长度做了归一化——与 PPO 目标本身不符。
        > 他们推测来源于预训练：packing 成定长上下文后 \`loss.mean(-1)\` 除以的是常数；到 RL 阶段回答长度不再恒定，同样的写法就引入了偏差。
      `,
      checkpoint: {
        id: 'cp-len',
        type: 'numeric',
        prompt: md`GRPO 原式下，同一组里两条**错误**回答优势相同、长度分别为 500 和 2000。短回答每个 token 受到的惩罚是长回答的多少倍？`,
        answer: 4,
        tolerance: 0,
        explanation: md`每个 token 的权重 $\propto\tilde A_i/|o_i|$，优势相同 → 比值 $=2000/500=4$。在 token-mean 或 Dr.GRPO 下这个比值为 1。`,
      },
    },
    {
      id: 'difficulty-bias',
      title: '难度偏差：组内 std 把“一边倒”的题放大',
      body: md`
        0/1 奖励下，一道题 $G$ 个回答中答对 $k$ 个，$p=k/G$。不除 std 时这道题的总权重
        $\sum_i|\tilde A_i| = \frac{2k(G-k)}{G}$，在 $p=0.5$ 最大、向两端平滑下降——这符合直觉：有对有错的题信息量最大。

        除以组内 std（ddof=1 时 $\mathrm{std}=\sqrt{\tfrac{G}{G-1}p(1-p)}$）后，$p$ 接近 0 或 1 的题 std 很小、被除得更多，**相对权重被放大**。
        论文原话的要点：优势归一化本是 RL 的常见技巧，但通常在**整个 batch** 上做；按**题目**做会让不同题在目标里的权重不同，形成难度偏差。

        动态采样（DAPO）只扔掉 $p\in\{0,1\}$ 的端点；难度偏差说的是端点**附近**的题被放大。REINFORCE++ 的思路与此一致：组内减均值、**全局** batch 上除 std。
      `,
      viz: StdBiasViz,
      checkpoint: {
        id: 'cp-diff',
        type: 'pick',
        prompt: md`$G=8$、0/1 奖励。相对 $k=4$ 的题，GRPO（除组内 std，ddof=1）对哪些 $k$ 的题施加了**超过 1.5 倍**的额外加权？`,
        items: ['k=1', 'k=2', 'k=3', 'k=4', 'k=5', 'k=6', 'k=7'],
        answer: [0, 6],
        explanation: md`额外加权 $=\mathrm{std}_4/\mathrm{std}_k=\sqrt{0.25/(p(1-p))}$：$k=1,7$ 时 $p(1-p)=0.1094$，约 $1.51\times$；$k=2,6$ 约 $1.15\times$；$k=3,5$ 约 $1.03\times$。`,
      },
    },
    {
      id: 'fix',
      title: 'Dr.GRPO：除以常数，回到 PPO 目标',
      body: md`
        去掉两项后，Dr.GRPO 的目标为：

        $$
        \mathcal J=\mathbb E\Big[\frac1G\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\big(r_{i,t}\tilde A_{i},\ \mathrm{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)\tilde A_{i}\big)\Big],\qquad \tilde A_i=R(q,o_i)-\mathrm{mean}(R)
        $$

        实现上只改一行：把 \`masked_mean\` 的分母从 \`mask.sum(axis=dim)\` 换成全局常数 \`MAX_TOKENS\`（生成预算），即 **token 求和 / 常数**。
        论文说其它常数也行，只是梯度范数不同（等价于换学习率）。

        **与 RLOO 的关系**：$\frac{G}{G-1}\tilde A_i=R_i-\frac1{G-1}\sum_{j\ne i}R_j=\hat A_i^{RLOO}$——只差一个常数缩放，可并入学习率，所以 $\tilde A$ 是无偏 baseline 的 MC 回报。

        各框架的聚合选项（以 verl 的 \`loss_agg_mode\` 命名为例）：

        | 模式 | 计算 | 对应 |
        |---|---|---|
        | \`seq-mean-token-mean\` | 行内 token 平均 → 行间平均 | GRPO 原式 |
        | \`token-mean\` | 全部 token 求和 / token 总数 | DAPO（verl 默认） |
        | \`seq-mean-token-sum\` | 行内 token 求和 → 行间平均 | 无归一化的 REINFORCE |
        | \`seq-mean-token-sum-norm\` | 上一行再除以常数（默认序列维长度） | Dr.GRPO |

        verl 的 GRPO 优势还有开关 \`norm_adv_by_std_in_grpo\`（关掉即 Dr.GRPO 的不除 std）。
      `,
      after: md`
        > [!INTERVIEW] token-mean 就完全无偏了吗？
        > 没有。token-mean 的分母是**这个 batch 的 token 总数**，每个 batch 不同：同一个 token 出现在“长回答多的 batch”里梯度就更小。
        > Dr.GRPO 的 Listing 1 把这种 per-batch 归一化（trl、verl 的写法）也标为有偏，只有除以**全局常数**才真正与回答长度无关。
      `,
      checkpoint: {
        id: 'cp-rloo',
        type: 'numeric',
        prompt: md`$G=8$ 时，Dr.GRPO 的优势 $\tilde A_i$ 乘以多少就等于 RLOO 的优势？`,
        answer: 1.142857,
        tolerance: 0.005,
        explanation: md`$G/(G-1)=8/7\approx1.1429$。推导：$R_i-\frac1{G-1}\sum_{j\ne i}R_j=R_i-\frac{G\,\bar R-R_i}{G-1}=\frac{G}{G-1}(R_i-\bar R)$。`,
      },
    },
    {
      id: 'findings',
      title: '论文的其它发现：base 模型与模板',
      body: md`
        Dr.GRPO 只是这篇论文的一半，另一半是对 **base 模型**的考察：

        - **DeepSeek-V3-Base 本身就已经有“Aha moment”**（自我反思），不完全是 RL 涌现的；
        - **Qwen2.5 base 模型不加任何模板**也有很强的推理能力，作者推测其预训练数据里可能已包含问答格式的推理数据；
        - 模板与问题集的匹配对 RL 效果影响很大；对 Llama 这类数学预训练较弱的模型，GRPO 出现了“准确率与长度双增”，但长度增长可能来自上面的优化偏差。

        **极简 R1-Zero 配方**：Qwen2.5-Math-7B + Dr.GRPO + MATH level 3–5 题目 + Qwen-Math 模板，8×A100 训练 27 小时，
        得到 **Oat-Zero-7B，AIME 2024 43.3%**（论文发表时 7B 的 SOTA）。实现基于他们的 Oat 框架，奖励用 Math-Verify 判定、答对 1 否则 0。

        > [!NEW] 这些结论的影响
        > 不除 std、按常数归一化后来被各框架做成开关（verl 的 \`norm_adv_by_std_in_grpo\`、TRL 的 \`loss_type="dr_grpo"\`）；
        > DeepSeek-V3.2 的 GRPO 优势也写成 $\hat A_{i,t}=R_i-\mathrm{mean}(\mathbf R)$，不再除 std。
      `,
      checkpoint: {
        id: 'cp-find',
        type: 'judge',
        prompt: md`Dr.GRPO 论文发现，R1-Zero 式训练中观察到的“回答越来越长”，可以完全归因于模型推理能力的增强。`,
        answer: false,
        explanation: md`论文的观点正相反：长度增长**部分**来自 GRPO 的长度偏差（尤其是错误回答变长）；用 Dr.GRPO 后错误回答的长度显著下降而准确率不降。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-two',
      type: 'single',
      difficulty: 1,
      prompt: md`GRPO 的“回答级长度偏差”来自目标中的哪一项？`,
      options: ['clip(r, 1−ε, 1+ε)', '1/|o_i|', '1/std(R)', 'β·KL'],
      answer: 1,
      explanation: md`$1/|o_i|$ 让每条回答总推力相同、每个 token 的推力与长度成反比。$1/\mathrm{std}$ 导致的是**题目级**难度偏差。`,
    },
    {
      id: 'q-dir',
      type: 'judge',
      difficulty: 2,
      prompt: md`GRPO 的长度偏差会让**正确**回答越来越长。`,
      answer: false,
      explanation: md`方向相反：对正确回答，短的更新更大 → 偏好简短；对错误回答，长的罚得更轻 → 偏好冗长。论文观察到长度增长主要来自错误回答。`,
    },
    {
      id: 'q-impl',
      type: 'multi',
      difficulty: 2,
      prompt: md`根据 Dr.GRPO 论文的检查，哪些开源实现的 PPO loss 存在长度偏差？（多选）`,
      options: ['trl', 'OpenRLHF', 'verl', 'SimpleRL-Zero', 'Open-Reasoner-Zero'],
      answer: [0, 1, 2, 3, 4],
      explanation: md`论文表 2：所检查的实现**全部**按回答长度归一化 loss（OpenRLHF 是逐回答平均；trl / verl 是逐 batch 的 token 平均），与 PPO 目标不一致。`,
    },
    {
      id: 'q-rloo',
      type: 'fill',
      difficulty: 2,
      prompt: md`RLOO 的优势 $\hat A_i^{RLOO}=R_i-\frac{1}{G-1}\sum_{j\neq i}R_j$。若组内奖励为 $[1,0,0,0]$：答对回答的 RLOO 优势 = ____(1)____，Dr.GRPO 优势 = ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['1'], numeric: true, tolerance: 0.001 },
        { label: '(2)', answers: ['0.75'], numeric: true, tolerance: 0.001 },
      ],
      explanation: md`RLOO：$1-0=1$；Dr.GRPO：$1-0.25=0.75$；比值 $4/3=G/(G-1)$。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把 verl 的 \`loss_agg_mode\` 与它对应的算法配对：`,
      pairs: [
        ['seq-mean-token-mean', 'GRPO 原式'],
        ['token-mean', 'DAPO'],
        ['seq-mean-token-sum-norm', 'Dr.GRPO'],
        ['seq-mean-token-sum', '逐回答 token 求和、不归一化'],
      ],
      explanation: md`四种聚合只差“分母是谁”：回答长度、batch token 总数、全局常数、或什么都不除。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`$G=4$ 的一组回答：o1 对（200 tok）、o2 对（800）、o3 错（300）、o4 错（1200）。用 GRPO **原式**聚合时，哪些回答的**每个 token** 受到的梯度权重绝对值最小？（点出所有并列最小的）`,
      items: ['o1', 'o2', 'o3', 'o4'],
      answer: [3],
      explanation: md`四条回答优势绝对值相同（2 对 2 错），每 token 权重 $\propto1/|o_i|$，所以最长的 o4（1200）最小。这正是“错误回答越长越罚不动”。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面想实现 Dr.GRPO 的 loss（连续分组，\`max_tokens\` 为生成预算），有 **2 处**仍保留了 GRPO 的偏差，点出来：`,
      code: py`
        def drgrpo_loss(logp_new, logp_old, rewards, mask, G, max_tokens, eps=0.2):
            r = rewards.reshape(-1, G)
            adv = r - r.mean(axis=1, keepdims=True)
            adv = adv / (r.std(axis=1, ddof=1, keepdims=True) + 1e-6)
            A = adv.reshape(-1, 1)
            ratio = np.exp(logp_new - logp_old)
            surr = np.minimum(ratio * A, np.clip(ratio, 1 - eps, 1 + eps) * A)
            per_seq = (surr * mask).sum(axis=1) / mask.sum(axis=1)
            return -per_seq.mean() / max_tokens
      `,
      bugLines: [4, 8],
      explanation: md`第 4 行又除了 std（难度偏差）——删掉。第 8 行除以了每行有效长度（长度偏差）——应为 \`(surr * mask).sum(axis=1)\`，常数 \`max_tokens\` 已在第 9 行除过。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出一个支持三种聚合的函数（\`x\` 已乘过 mask；有 2 行干扰项）：`,
      lines: [
        'def agg(x, mask, mode, max_len):',
        '    if mode == "token-mean":',
        '        return x.sum() / mask.sum()',
        '    if mode == "seq-mean-token-mean":',
        '        return (x.sum(axis=1) / mask.sum(axis=1)).mean()',
        '    return x.sum(axis=1).mean() / max_len',
      ],
      distractors: ['        return x.mean()', '    return x.sum(axis=1).mean() / mask.sum()'],
      explanation: md`\`x.mean()\` 会把 padding 也算进分母；最后一行若除以 \`mask.sum()\` 就退化成与 batch token 数相关的归一化，不再是 Dr.GRPO 的常数归一化。`,
    },
    {
      id: 'q-tokmean',
      type: 'single',
      difficulty: 3,
      prompt: md`关于 DAPO 的 token-mean 聚合，Dr.GRPO 论文的观点最接近：`,
      options: [
        '完全无偏，与 Dr.GRPO 等价',
        '消除了回答间的长度偏差，但分母随 batch 的 token 总数变化，仍被视为有偏；应除以全局常数',
        '比 GRPO 原式偏差更大',
        '只在 G=1 时有偏',
      ],
      answer: 1,
      explanation: md`同一 batch 内所有 token 等权，回答间不再因长度区别对待；但分母是 batch 的 token 总数，长回答多的 batch 每个 token 的梯度更小。Listing 1 中 per-batch 归一化（trl / verl）同样被标为有偏。`,
    },
    {
      id: 'q-aha',
      type: 'judge',
      difficulty: 1,
      prompt: md`Dr.GRPO 论文发现 DeepSeek-V3-Base 在 RL 之前就已经表现出 “Aha moment”。`,
      answer: true,
      explanation: md`论文摘要原意：DeepSeek-V3-Base 已经展现 “Aha moment”；Qwen2.5 base 不用模板也有很强的推理能力，暗示预训练中可能已有相关数据。`,
    },
    {
      id: 'q-weight',
      type: 'numeric',
      difficulty: 3,
      prompt: md`$G=16$、0/1 奖励，相对答对 8 个的题，答对 1 个的题在 GRPO（ddof=1 std）下的**额外加权**（即 $\mathrm{std}_8/\mathrm{std}_1$）是多少？`,
      answer: 2.066,
      tolerance: 0.01,
      explanation: md`$\mathrm{std}=\sqrt{\frac{G}{G-1}p(1-p)}$：$k=8$ 时 $\sqrt{\frac{16}{15}\times0.25}=0.5164$；$k=1$ 时 $\sqrt{\frac{16}{15}\times\frac{15}{256}}=0.25$；比值 $2.066$。`,
    },
  ],

  guided: [
    {
      id: 'g-drgrpo',
      title: '把 GRPO 改成 Dr.GRPO（连续分组版）',
      intro: md`
        \`logp_new / logp_old / mask\`：\`(B, T)\`；\`rewards\`：\`(B,)\`，每 \`G\` 个连续样本一组；\`max_tokens\` 是生成预算（常数）。
        实现 Dr.GRPO loss：优势只减组内均值；clipped 目标；**行内 token 求和 → 行间平均 → 除以常数**。
      `,
      template: py`
        import numpy as np

        def drgrpo_loss(logp_new, logp_old, rewards, mask, G, max_tokens, clip_eps=0.2):
            r = rewards.reshape(-1, G)
            A = ({{1}}).reshape(-1, 1)          # 只减均值，不除 std
            ratio = np.exp(logp_new - logp_old)
            surr = {{2}}
            per_seq = {{3}}                     # 行内求和，不除长度
            return {{4}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`组内中心化：$\tilde A=R-\mathrm{mean}(R)$。`,
          answer: 'r - r.mean(axis=1, keepdims=True)',
          accept: ['r - r.mean(1, keepdims=True)'],
          explain: md`不除 std，消除难度偏差；它与 RLOO 优势只差常数 $G/(G-1)$。`,
        },
        {
          blank: 2,
          instruction: md`PPO 式 clipped 目标。`,
          answer: 'np.minimum(ratio * A, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)',
          explain: md`clip 不变——Dr.GRPO 只改归一化。`,
        },
        {
          blank: 3,
          instruction: md`每行对有效 token **求和**（乘 mask），不要除以行长度。`,
          answer: '(surr * mask).sum(axis=1)',
          accept: ['(surr * mask).sum(1)', 'np.sum(surr * mask, axis=1)'],
          explain: md`这一步就是删掉 $1/|o_i|$。`,
        },
        {
          blank: 4,
          instruction: md`对回答取平均，再除以全局常数 \`max_tokens\`，最后取负得到 loss。`,
          answer: '-per_seq.mean() / max_tokens',
          accept: ['-np.mean(per_seq) / max_tokens', '-(per_seq.mean() / max_tokens)'],
          explain: md`常数只影响梯度尺度（等价于学习率），不随回答长度变化，所以没有长度偏差。`,
        },
      ],
      setup: SETUP,
      tests: [
        {
          name: '与参考实现对拍',
          code: py`
            for seed in range(6):
                ln, lo, lr, r, gid, m = _make_batch(seed, garbage=False)
                got = float(drgrpo_loss(ln, lo, r, m, 4, 512))
                exp = _ref_drgrpo_loss(ln, lo, r, gid, m, 512)
                assert abs(got - exp) < 1e-10, f"seed={seed}: 期望 {exp:.10f}，得到 {got:.10f}"
          `,
        },
        {
          name: '缩放奖励 → loss 等比缩放（不除 std）',
          code: py`
            ln, lo, lr, r, gid, m = _make_batch(3, garbage=False)
            a = float(drgrpo_loss(ln, lo, r, m, 4, 512))
            b = float(drgrpo_loss(ln, lo, 2 * r, m, 4, 512))
            assert abs(b - 2 * a) < 1e-10, f"奖励 ×2 时 Dr.GRPO loss 应 ×2：{a} → {b}（是不是还除了 std？）"
          `,
        },
      ],
      outro: md`
        对比 GRPO 只改了两处：去掉 \`/ std\`、把 \`/ mask.sum(1)\` 换成 \`/ max_tokens\`。面试时能说出“这两处分别对应难度偏差和长度偏差”，就答到点上了。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-agg',
      title: '实现四种 loss 聚合方式',
      difficulty: 'medium',
      source: 'verl loss_agg_mode / Dr.GRPO',
      timeLimitMin: 12,
      description: md`
        \`\`\`python
        def agg_loss(token_loss, mask, mode: str, max_len: int | None = None) -> float: ...
        \`\`\`

        \`token_loss\`、\`mask\` 形状 \`(B, T)\`；**padding 位置的 token_loss 可能是 nan / inf 等垃圾值**，不能影响结果；每行至少一个有效 token。

        | mode | 计算 |
        |---|---|
        | \`"token-mean"\` | 所有有效 token 求和 / 有效 token 总数 |
        | \`"seq-mean-token-mean"\` | 每行有效 token 平均，再对 B 行平均 |
        | \`"seq-mean-token-sum"\` | 每行有效 token 求和，再对 B 行平均 |
        | \`"seq-mean-token-sum-norm"\` | 上一行的结果再除以 \`max_len\`；\`max_len\` 为 None 时用 \`T\` |

        其它 mode 抛出 \`ValueError\`。返回 Python float。
      `,
      starter: py`
        import numpy as np

        def agg_loss(token_loss, mask, mode, max_len=None):
            # TODO: 先把 padding 位置清零（注意 nan * 0 = nan），再按 mode 聚合
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def agg_loss(token_loss, mask, mode, max_len=None):
            m = np.asarray(mask, dtype=np.float64)
            x = np.where(m > 0, np.asarray(token_loss, dtype=np.float64), 0.0)   # 用 where 而不是乘法：nan*0 仍是 nan
            if mode == "token-mean":
                return float(x.sum() / m.sum())
            if mode == "seq-mean-token-mean":
                return float(np.mean(x.sum(axis=1) / m.sum(axis=1)))
            if mode == "seq-mean-token-sum":
                return float(np.mean(x.sum(axis=1)))
            if mode == "seq-mean-token-sum-norm":
                L = m.shape[1] if max_len is None else max_len
                return float(np.mean(x.sum(axis=1)) / L)
            raise ValueError(f"unknown mode: {mode}")
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，对应 verl 的 agg_loss 单卡简化）',
          code: py`
            import torch

            def agg_loss_torch(loss_mat, mask, mode, max_len=None):
                mask = mask.float()
                x = torch.where(mask > 0, loss_mat, torch.zeros_like(loss_mat))
                if mode == "token-mean":
                    return x.sum() / mask.sum()
                seq = x.sum(-1)
                if mode == "seq-mean-token-mean":
                    return (seq / mask.sum(-1)).mean()
                if mode == "seq-mean-token-sum":
                    return seq.mean()
                if mode == "seq-mean-token-sum-norm":
                    return seq.mean() / (max_len or mask.shape[-1])
                raise ValueError(mode)
          `,
        },
      ],
      explanation: md`
        四种方式只差分母。真实框架（如 verl）还要处理**数据并行**：token-mean 需要全局 token 数（各 rank 的 token 数不同，直接对每个 rank 的均值再平均会再引入偏差），
        所以 verl 的 token-mean 用 \`batch_num_tokens\` 与 \`dp_size\` 做全局归一化。

        **易错点**：用乘法做掩码时 padding 处的 \`nan * 0 = nan\`，必须用 \`np.where\`。
      `,
      hints: [md`先 \`x = np.where(mask > 0, token_loss, 0.0)\`。`, md`行级：\`x.sum(axis=1)\`；行长：\`mask.sum(axis=1)\`；token 总数：\`mask.sum()\`。`],
      setup: SETUP,
      tests: [
        {
          name: '手算示例',
          code: py`
            tl = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]])
            mk = np.array([[1, 1, 1], [1, 0, 0]])
            cases = {"token-mean": 10 / 4, "seq-mean-token-mean": (2 + 4) / 2, "seq-mean-token-sum": (6 + 4) / 2}
            for mode, exp in cases.items():
                got = agg_loss(tl, mk, mode)
                assert abs(got - exp) < 1e-12, f"{mode}: 期望 {exp}，得到 {got}"
            got = agg_loss(tl, mk, "seq-mean-token-sum-norm", max_len=10)
            assert abs(got - 0.5) < 1e-12, f"seq-mean-token-sum-norm(max_len=10): 期望 0.5，得到 {got}"
            got = agg_loss(tl, mk, "seq-mean-token-sum-norm")
            assert abs(got - 5 / 3) < 1e-12, f"max_len=None 时应除以 T=3：期望 {5 / 3}，得到 {got}"
          `,
        },
        {
          name: '未知 mode 抛 ValueError',
          code: py`
            try:
                agg_loss(np.ones((1, 2)), np.ones((1, 2)), "sample-mean")
            except ValueError:
                pass
            else:
                raise AssertionError("未知 mode 应抛出 ValueError")
          `,
        },
        {
          name: 'padding 含 nan / inf',
          hidden: true,
          code: py`
            tl = np.array([[1.0, np.nan, np.inf], [2.0, 3.0, -np.inf]])
            mk = np.array([[1, 0, 0], [1, 1, 0]])
            for mode in ["token-mean", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"]:
                got = agg_loss(tl, mk, mode, 8)
                exp = _ref_agg(tl, mk, mode, 8)
                assert np.isfinite(got) and abs(got - exp) < 1e-12, f"{mode}: 期望 {exp}，得到 {got}（padding 的 nan 泄漏了？）"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(0)
            for _ in range(50):
                B, T = rng.integers(1, 6), rng.integers(1, 9)
                lens = rng.integers(1, T + 1, size=B)
                mk = (np.arange(T)[None, :] < lens[:, None]).astype(float)
                tl = rng.normal(size=(B, T))
                for mode in ["token-mean", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"]:
                    ml = None if rng.random() < 0.5 else 64
                    got = agg_loss(tl, mk, mode, ml)
                    exp = _ref_agg(tl, mk, mode, ml)
                    assert abs(got - exp) < 1e-10, f"{mode}, max_len={ml}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
    },
    {
      id: 'c-adv-modes',
      title: '四种优势估计：GRPO / Dr.GRPO / batch-std / RLOO',
      difficulty: 'medium',
      source: 'Dr.GRPO · REINFORCE++ · RLOO',
      timeLimitMin: 15,
      description: md`
        \`\`\`python
        def compute_advantage(rewards, group_ids, mode: str, eps: float = 1e-6) -> np.ndarray: ...
        \`\`\`

        \`rewards\`：\`(B,)\`；\`group_ids\`：\`(B,)\`，同 id 为同一题，**可乱序**，每组至少 2 个。返回 \`(B,)\`：

        | mode | 公式 | 出处 |
        |---|---|---|
        | \`"group"\` | $(r_i-\bar r_g)/(\mathrm{std}_g+\epsilon)$，std 用 ddof=1 | GRPO |
        | \`"none"\` | $r_i-\bar r_g$ | Dr.GRPO |
        | \`"batch"\` | 先 $d_i=r_i-\bar r_g$，再除以**整个 batch** 上 $d$ 的 std（ddof=1）$+\epsilon$ | REINFORCE++ w/ baseline |
        | \`"rloo"\` | $r_i-\frac{1}{n_g-1}\sum_{j\in g,\,j\ne i}r_j$ | RLOO |

        要求 $O(B)$ 的向量化实现（不要对每个样本再遍历一遍组）。
      `,
      starter: py`
        import numpy as np

        def compute_advantage(rewards, group_ids, mode, eps=1e-6):
            # TODO: np.unique(return_inverse=True) + np.bincount 做分组统计
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def compute_advantage(rewards, group_ids, mode, eps=1e-6):
            r = np.asarray(rewards, dtype=np.float64)
            _, inv = np.unique(group_ids, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv).astype(np.float64)
            s = np.bincount(inv, weights=r)
            dev = r - (s / cnt)[inv]                          # 组内中心化
            if mode == "none":
                return dev
            if mode == "group":
                std = np.sqrt(np.bincount(inv, weights=dev ** 2) / (cnt - 1))
                return dev / (std[inv] + eps)
            if mode == "batch":
                return dev / (dev.std(ddof=1) + eps)           # 全局 std：不再按题目区别对待
            if mode == "rloo":
                return r - (s[inv] - r) / (cnt[inv] - 1)      # 留一均值
            raise ValueError(mode)
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，连续分组）',
          code: py`
            import torch

            def compute_advantage_torch(rewards, G, mode, eps=1e-6):
                r = rewards.view(-1, G)
                dev = r - r.mean(-1, keepdim=True)
                if mode == "none":
                    out = dev
                elif mode == "group":
                    out = dev / (r.std(-1, keepdim=True) + eps)     # torch.std 默认无偏
                elif mode == "batch":
                    out = dev / (dev.std() + eps)
                elif mode == "rloo":
                    out = r - (r.sum(-1, keepdim=True) - r) / (G - 1)
                else:
                    raise ValueError(mode)
                return out.reshape(-1)
          `,
        },
      ],
      explanation: md`
        - \`rloo\` 与 \`none\` 只差常数 $n_g/(n_g-1)$（组大小相同时）——Dr.GRPO 附录的结论，可用测试验证。
        - \`batch\` 保留了 std 归一化的“尺度不变”好处，但统计量来自整个 batch，不再给几乎全对 / 全错的题额外加权（REINFORCE++ 的论点）。
        - 复杂度 $O(B)$。
      `,
      hints: [
        md`\`_, inv = np.unique(group_ids, return_inverse=True)\`；\`s = np.bincount(inv, weights=r)\`、\`cnt = np.bincount(inv)\`。`,
        md`RLOO：其余样本之和 = \`s[inv] - r\`，个数 = \`cnt[inv] - 1\`。`,
      ],
      setup: SETUP,
      tests: [
        {
          name: '示例',
          code: py`
            r = np.array([1.0, 0, 0, 0, 1, 1, 0, 1])
            g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            got = compute_advantage(r, g, "none")
            assert np.allclose(got, [0.75, -0.25, -0.25, -0.25, 0.25, 0.25, -0.75, 0.25]), f"none: 得到 {got}"
            got = compute_advantage(r, g, "rloo")
            assert np.allclose(got, [1, -1 / 3, -1 / 3, -1 / 3, 1 / 3, 1 / 3, -1, 1 / 3]), f"rloo: 得到 {got}"
            got = compute_advantage(r, g, "group")
            assert np.allclose(got[:4], [1.5, -0.5, -0.5, -0.5], atol=1e-5), f"group（ddof=1）: 得到 {got[:4]}"
          `,
        },
        {
          name: 'RLOO = G/(G−1) × Dr.GRPO',
          code: py`
            rng = np.random.default_rng(1)
            r = rng.normal(size=24)
            g = np.repeat(np.arange(4), 6)
            a, b = compute_advantage(r, g, "rloo"), compute_advantage(r, g, "none")
            assert np.allclose(a, 6 / 5 * b), f"最大误差 {np.abs(a - 6 / 5 * b).max():.3e}"
          `,
        },
        {
          name: '乱序分组随机对拍（四种 mode）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(2)
            for _ in range(30):
                n, G = rng.integers(1, 6), rng.integers(2, 7)
                g = rng.permutation(np.repeat(rng.choice(1000, size=n, replace=False), G))
                r = rng.integers(0, 2, size=len(g)).astype(float) if rng.random() < 0.5 else rng.normal(size=len(g))
                for mode in ["group", "none", "batch", "rloo"]:
                    got = compute_advantage(r, g, mode)
                    exp = _ref_adv_all(r, g, mode)
                    assert np.allclose(got, exp, atol=1e-9), f"mode={mode}: 最大误差 {np.abs(got - exp).max():.3e}"
          `,
        },
        {
          name: '规模：B = 200000 应在 0.5s 内',
          hidden: true,
          code: py`
            import time
            rng = np.random.default_rng(3)
            g = rng.permutation(np.repeat(np.arange(25000), 8))
            r = rng.integers(0, 2, size=len(g)).astype(float)
            t0 = time.perf_counter()
            for mode in ["group", "rloo"]:
                compute_advantage(r, g, mode)
            dt = time.perf_counter() - t0
            assert dt < 0.5, f"用时 {dt:.2f}s，请用 bincount 向量化，不要逐样本循环"
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-bias',
      freq: 3,
      q: md`GRPO 有哪两个偏差？Dr.GRPO 怎么修？`,
      a: md`
        **结论**：回答级长度偏差（$1/|o_i|$）和题目级难度偏差（$1/\mathrm{std}$）；Dr.GRPO 把两项都删掉，token 求和后除以全局常数（生成预算），优势只减组内均值。

        **原因**：$1/|o_i|$ 让每条回答总推力相同，错误回答越长每个 token 罚得越轻，长度虚涨；组内 std 让几乎全对 / 全错的题权重更大。

        **延伸**：修改后的目标就是 PPO 目标配 MC 回报 + 无偏 baseline，优势与 RLOO 只差 $G/(G-1)$。Qwen2.5-Math-7B 上用该配方训练得到 Oat-Zero-7B（AIME24 43.3%，8×A100、27 小时）。
      `,
    },
    {
      id: 'qa-agg',
      freq: 3,
      q: md`RL loss 有哪几种聚合方式？各自的取舍是什么？`,
      a: md`
        **结论**：seq-mean-token-mean（GRPO 原式）、token-mean（DAPO）、seq-mean-token-sum（不归一化）、seq-mean-token-sum-norm（Dr.GRPO，除常数）。

        **原因**：它们决定了每个 token 的梯度权重。行内平均有长度偏差；token-mean 让 batch 内 token 等权，但分母随 batch 变化；token-sum 无偏但梯度尺度随长度剧烈变化，需要除一个常数稳定尺度。

        **延伸**：分布式训练时 token-mean 还要用**全局** token 数归一化，否则各 rank token 数不同又会引入偏差（verl 用 \`batch_num_tokens\` 和 \`dp_size\` 处理）。
      `,
    },
    {
      id: 'qa-length',
      freq: 2,
      q: md`RL 训练中回答越来越长，能说明模型学会了更深的推理吗？`,
      a: md`
        **结论**：不能直接这么下结论。

        **原因**：Dr.GRPO 发现 GRPO 的长度偏差会让错误回答越来越长，奖励增长放缓后长度仍在涨；去偏后错误回答长度明显下降、准确率不变。DAPO 也指出样本级 loss 会导致长度与熵“不健康”地增长。

        **延伸**：要把长度拆成“正确回答长度”和“错误回答长度”分别看，结合截断率、重复率、熵与 pass@k 一起判断；R1-Zero 长度跳变还与最大长度从 32K 提到 65K 有关。
      `,
    },
    {
      id: 'qa-std',
      freq: 2,
      q: md`优势要不要除以 std？在组内还是 batch 内除？`,
      a: md`
        **结论**：看需求：除 std 带来尺度不变性，但组内 std 有难度偏差；折中是组内减均值、batch 内除 std（REINFORCE++ w/ baseline、TRL 的 \`scale_rewards="batch"\`）。

        **原因**：组内 std 在 G 很小时噪声大、几乎全对的题会被放大；batch std 由上千个样本估计，接近常数，相当于自适应学习率。

        **延伸**：DeepSeek-V3.2 的 GRPO 只减均值（$\hat A=R_i-\mathrm{mean}(R)$），不除 std。
      `,
    },
  ],

  cheatsheet: md`
    **GRPO 的有效权重**：$a_{i,t}=\dfrac{\tilde A_i}{G\cdot\mathrm{std}(R)\cdot|o_i|}$，$\tilde A_i=R_i-\mathrm{mean}(R)$

    | 因子 | 偏差 | Dr.GRPO |
    |---|---|---|
    | $1/\lvert o_i\rvert $ | 错误长回答罚得轻 → 长度虚涨 | 改为 /MAX_TOKENS |
    | $1/\mathrm{std}$ | 几乎全对/全错的题被放大 | 删除 |

    | verl loss_agg_mode | 分母 |
    |---|---|
    | seq-mean-token-mean | 每行长度，再行平均（GRPO） |
    | token-mean | batch token 总数（DAPO） |
    | seq-mean-token-sum-norm | 常数（Dr.GRPO） |

    \`\`\`python
    A = r - r.mean(1, keepdims=True)                    # 不除 std
    loss = -(surr * mask).sum(1).mean() / MAX_TOKENS    # 不除长度
    rloo = r - (r.sum(1, keepdims=True) - r) / (G - 1)  # = G/(G-1) * A
    \`\`\`
    - token-mean 仍随 batch token 数变化，Dr.GRPO 视为有偏
    - Oat-Zero-7B：Qwen2.5-Math-7B + MATH L3–5 + Qwen-Math 模板，AIME24 43.3%，8×A100 27h
  `,
})
