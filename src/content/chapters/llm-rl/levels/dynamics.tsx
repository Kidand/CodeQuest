/**
 * llm-rl-dynamics：RL 训练动力学——熵坍缩、分叉 token、reward hacking、pass@k 之争、长度与难度
 */
import { defineLevel, md, py } from '@/content/helpers'
import { EntropyDynamicsViz } from '../viz/EntropyDynamicsViz'
import { ForkingViz } from '../viz/ForkingViz'
import { PassKViz } from '../viz/PassKViz'

const SETUP = py`
  import numpy as np, inspect, time
  from fractions import Fraction
  from math import comb as _comb

  def _ref_passk_one(n, c, k):
      if n - c < k:
          return 1.0
      return float(1 - Fraction(_comb(n - c, k), _comb(n, k)))

  def _ref_log_softmax(x):
      x = np.asarray(x, dtype=np.float64)
      m = np.max(np.where(np.isfinite(x), x, -np.inf), axis=-1, keepdims=True)
      z = x - m
      return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

  def _ref_entropy(logits, mask):
      lp = _ref_log_softmax(logits)
      p = np.exp(lp)
      H = -np.where(p > 0, p * lp, 0.0).sum(axis=-1)
      m = np.asarray(mask, dtype=np.float64)
      H = H * m
      return H, float(H.sum() / m.sum())

  def _ref_fork_mask(logits, mask, rho):
      H, _ = _ref_entropy(logits, mask)
      m = np.asarray(mask) > 0
      idx = np.flatnonzero(m.reshape(-1))
      k = max(1, int(round(rho * len(idx))))
      order = np.argsort(-H.reshape(-1)[idx], kind="stable")
      out = np.zeros(m.size, dtype=bool)
      out[idx[order[:k]]] = True
      return out.reshape(m.shape)
`

export default defineLevel({
  learn: [
    {
      id: 'entropy',
      title: '熵坍缩：性能是用熵“换”来的',
      body: md`
        token 级策略熵 $H_t=-\sum_v\pi_\theta(v\mid\cdot)\log\pi_\theta(v\mid\cdot)$，对所有生成 token 取平均就是训练曲线上的 “entropy”。
        不加干预的 RLVR 中，熵在训练**早期急剧下降**，模型变得过度自信、探索消失，性能随之饱和——即**熵坍缩**。

        Cui 等（「The Entropy Mechanism of RL for Reasoning LMs」，2025-05）给出两个结论：

        1. **经验公式** $R=-a\,e^{H}+b$：下游性能 $R$ 与熵 $H$ 呈指数关系，性能是拿熵换的；熵耗尽（$H=0$）时上限可预测为 $-a+b$。
        2. **机制**：对 softmax 策略做策略梯度，熵的变化由“动作对数概率”与“logit 变化”的**协方差**驱动，一阶近似为
           $\Delta H\approx-\eta\,\mathrm{Cov}_{a\sim\pi}\big(\log\pi(a),\ \pi(a)A(a)\big)$。
           高概率且高优势的动作让熵下降，低概率但高优势的动作让熵上升；训练中这个协方差**大多为正**，所以熵单调下降。

        下面是一个 6 个 token 的单步策略，用精确的策略梯度迭代。对比“正确答案是模型最自信的 token”与“正确答案是低概率 token”两种情形。
      `,
      viz: EntropyDynamicsViz,
      after: md`
        **缓解手段**（各有代价）：

        | 手段 | 做法 | 代价 / 注意 |
        |---|---|---|
        | Clip-Higher（DAPO） | 放宽 $\varepsilon_{high}$，让低概率 token 能涨 | 太大会不稳 |
        | Clip-Cov / KL-Cov（Cui 等） | 对**协方差最高**的少量 token 截断梯度 / 加 KL | 需要算协方差 |
        | 熵正则 $-\alpha H$ | 直接奖励熵 | 系数极敏感，容易熵爆炸、输出乱码 |
        | CISPO | 不丢弃任何 token 的梯度 | 梯度略有偏 |
        | 提高采样温度、保留 top-p 掩码 | 从采样端维持多样性 | 训推一致性要处理好 |
      `,
      checkpoint: {
        id: 'cp-ent',
        type: 'single',
        prompt: md`按协方差机制，下列哪种更新最会**降低**策略熵？`,
        options: ['高概率 token 获得正优势', '低概率 token 获得正优势', '高概率 token 获得负优势', '所有 token 优势为 0'],
        answer: 0,
        explanation: md`$\log\pi$ 大且 $\pi A$ 大 → 协方差为正 → $\Delta H<0$。B 会提高熵（探索）；C 让高概率 token 下降，协方差为负，熵上升；D 无更新。`,
      },
    },
    {
      id: 'forking',
      title: '少数高熵 token 才是“分叉点”',
      body: md`
        Qwen 与清华 LeapLab 的「Beyond the 80/20 Rule」（2025-06）统计 CoT 的 token 熵分布：**只有少数 token 熵高**，它们像路口一样决定推理走向
        （例如 “wait”“however”“suppose”），其余绝大多数 token 熵很低、几乎确定。

        他们把策略梯度**只施加在熵最高的约 20% token 上**：在 Qwen3-8B 上与全 token 训练效果相当，在 Qwen3-14B / 32B 上反而更好；
        保留比例降到 10% 会丢掉有用的 token，提高到 50% 或 100% 又会把低熵 token 加回来、削弱探索。

        这与 MiniMax-M1 的观察一致：低概率的反思 token 是关键的分叉点，PPO-clip 却最容易把它们裁掉。
      `,
      viz: ForkingViz,
      checkpoint: {
        id: 'cp-fork',
        type: 'judge',
        prompt: md`「80/20」论文发现只用熵最高的约 20% token 计算策略梯度，在 Qwen3-32B 上效果不如使用全部 token。`,
        answer: false,
        explanation: md`相反：在 8B 上持平，在 14B、32B 上更好，且模型越大收益越明显。`,
      },
    },
    {
      id: 'hacking',
      title: 'reward hacking 与格式奖励',
      body: md`
        **reward hacking**：模型利用奖励函数的漏洞拿高分，而不是真正完成任务。RLVR 用程序判分，漏洞少了，但没有消失：

        | 形态 | 例子 | 防御 |
        |---|---|---|
        | 答案抽取过松 | 输出多个 \`\boxed{}\`、枚举所有选项 | 只取最后一个答案；多答案判错 |
        | 测试覆盖不足 | 针对样例硬编码输出、修改 / 跳过测试 | 隐藏测试、沙箱禁止改测试文件 |
        | 格式分权重过高 | 只学会输出 \`<think></think>\` 空壳 | 格式只作门槛或小权重 |
        | 长度 / 重复 | 写到截断或陷入循环 | 超长整形（DAPO）、重复检测提前截断（MiniMax-M1） |
        | 神经 RM 偏差 | 迎合奖励模型的风格偏好 | 推理任务不用神经 RM（R1）；RM 与规则混合 |

        R1 论文的附录记录了一个直观例子：使用 helpful 奖励模型时，**奖励持续上升，而推理任务上的表现下降**——这正是它在推理任务上坚持规则奖励的原因。
      `,
      checkpoint: {
        id: 'cp-hack',
        type: 'multi',
        prompt: md`下列哪些是 RLVR 中**真实存在**的 reward hacking 风险？（多选）`,
        options: ['验证器取任意一个 \\boxed，模型就输出多个候选答案', '代码任务只跑公开样例，模型硬编码样例输出', '规则验证器本身无法被利用，所以 RLVR 不存在 reward hacking', '格式奖励权重过大，模型只学会输出合规但空洞的格式'],
        answer: [0, 1, 3],
        explanation: md`C 错：验证器只是比神经 RM 难利用，漏洞（抽取规则、测试覆盖、格式分）依然存在。`,
      },
    },
    {
      id: 'passk',
      title: 'pass@k vs pass@1：RL 是否扩展了能力边界？',
      body: md`
        **pass@k**：采 $k$ 个样本至少一个正确的概率。Codex 论文给出的**无偏估计**：每题采 $n\ge k$ 个、其中 $c$ 个正确，

        $$
        \text{pass@}k=\mathbb E_{\text{题目}}\Big[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\Big]=1-\prod_{j=n-c+1}^{n}\Big(1-\frac{k}{j}\Big)\quad(n-c\ge k)
        $$

        直接算组合数会溢出，Codex 的实现逐项连乘；$n-c<k$ 时恒为 1。用 $1-(1-\hat p)^k$（$\hat p=c/n$）是**有偏**的，会系统性低估。

        **争论的时间线**：
        - DeepSeekMath（2024）：RL 提升了 Maj@K **但没有提升 Pass@K**，认为 RL 让输出分布更稳健，而非提升基础能力；
        - Yue 等（「Does RL Really Incentivize Reasoning Capacity…」，2025-04）：多个模型家族、六种 RLVR 算法上，**小 k 时 RL 模型更好、大 k 时 base 更好**，且随训练推进能力边界常常**收窄**；RL 模型的推理路径已在 base 的采样分布中；反而**蒸馏**能引入新的推理模式；
        - ProRL（NVIDIA，NeurIPS 2025）：带 KL 控制、周期性重置参考策略、任务多样的**长时间** RL 后，RL 模型在各种 k 上都优于 base，包括 base 怎么采都做不对的题；收益与 base 的 pass@128 呈负相关（base 越弱的任务涨得越多）。结论基于 1.5B 模型。

        下图是一个玩具模型：RL 只把“已经有一定把握”的题推高、把“偶尔做对”的题压低。
      `,
      viz: PassKViz,
      after: md`
        > [!INTERVIEW] 怎么回答“RL 有没有扩展能力边界”
        > 先给定义：用大 k 的 pass@k 衡量边界。再给证据：短程 RLVR 主要是**锐化**（提升采样效率，pass@1 涨、大 k 可能跌）；
        > 长程、带探索与多样任务的 RL（ProRL）有扩展边界的证据，但规模与任务有限；蒸馏可以引入 teacher 的新模式。
        > 最后给工程含义：评估要同时报 pass@1 和大 k 的 pass@k，并关注熵与多样性。
      `,
      checkpoint: {
        id: 'cp-passk',
        type: 'numeric',
        prompt: md`某题采样 $n=10$ 次，$c=2$ 次正确。pass@5 的无偏估计是多少？（保留 4 位小数）`,
        answer: 0.7778,
        tolerance: 0.002,
        explanation: md`$1-\binom{8}{5}/\binom{10}{5}=1-56/252=0.7778$。连乘形式：$1-(1-\frac59)(1-\frac5{10})=1-\frac49\cdot\frac12=0.7778$。有偏估计 $1-(0.8)^5=0.672$ 明显偏低。`,
      },
    },
    {
      id: 'curriculum',
      title: '长度增长、难度筛选与课程',
      body: md`
        **长度增长**：R1-Zero 的平均回答长度随训练持续上涨，被视为“学会了花更多时间思考”。但要拆开看：
        R1-Zero 在第 8.2k 步把最大长度从 32,768 提到 65,536，性能与长度都出现跳变；Dr.GRPO 指出一部分增长来自优化偏差（错误回答变长）；
        DAPO 指出样本级 loss 会导致不健康的长度与熵增长。

        **难度筛选**：0/1 奖励下，一道题对梯度的信号强度（不除 std 时）$\propto p(1-p)$，在通过率 $p=0.5$ 最大，在 $p\in\{0,1\}$ 为 0。
        常见做法：

        - 训练前用当前模型（或 SFT 模型）对每题采样估计通过率，剔除恒对 / 恒错的题；
        - 训练中动态采样（DAPO）或周期性重估通过率，按难度分桶、由易到难做**课程**；
        - 长度维度的课程：逐步放开最大生成长度（R1-Zero 的 32K → 65K）。

        **推理强度档位**：DeepSeek-V4 为不同推理档位训练专家时，使用**不同的长度惩罚与上下文窗口**，再通过 on-policy 蒸馏合并（见下一关）。
      `,
      checkpoint: {
        id: 'cp-diff',
        type: 'numeric',
        prompt: md`按信号强度 $\propto p(1-p)$，通过率 $p=0.9$ 的题的信号是 $p=0.5$ 的题的多少倍？`,
        answer: 0.36,
        tolerance: 0.01,
        explanation: md`$0.9\times0.1/(0.5\times0.5)=0.09/0.25=0.36$。所以过易的题虽然有梯度，但信号弱；GRPO 的 std 归一化会把它重新放大（难度偏差）。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-collapse',
      type: 'single',
      difficulty: 1,
      prompt: md`“熵坍缩”最直接的后果是：`,
      options: ['显存占用上升', '同组回答趋同、探索消失，性能提前饱和', '奖励模型被 hacking', '训练 loss 变成 NaN'],
      answer: 1,
      explanation: md`熵低 → 采样几乎确定 → 同一题的 G 个回答一样 → 组内优势为 0、学不到新东西。其它选项不是熵坍缩的直接表现。`,
    },
    {
      id: 'q-bonus',
      type: 'judge',
      difficulty: 2,
      prompt: md`在 loss 里加一个熵奖励项 $-\alpha H$，就能稳定可靠地解决熵坍缩。`,
      answer: false,
      explanation: md`熵正则对系数极其敏感：小了没用，大了熵爆炸、输出退化。这也是 Clip-Higher、Clip-Cov / KL-Cov 等“定点”方法被提出的原因。`,
    },
    {
      id: 'q-passk',
      type: 'numeric',
      difficulty: 2,
      prompt: md`$n=20$ 个样本中 $c=1$ 个正确，pass@10 的无偏估计是多少？`,
      answer: 0.5,
      tolerance: 0.001,
      explanation: md`$1-\binom{19}{10}/\binom{20}{10}=1-\frac{10}{20}=0.5$。直觉：从 20 个里无放回取 10 个，取到那 1 个正确样本的概率是 $10/20$。`,
    },
    {
      id: 'q-biased',
      type: 'fill',
      difficulty: 2,
      prompt: md`$n=10,\ c=2,\ k=5$。有偏估计 $1-(1-c/n)^k$ = ____(1)____（保留 3 位小数），它相对无偏估计 0.778 是 ____(2)____（高估 / 低估）。`,
      blanks: [
        { label: '(1)', answers: ['0.672'], numeric: true, tolerance: 0.002 },
        { label: '(2)', answers: ['低估', '偏低', 'underestimate'] },
      ],
      explanation: md`$1-0.8^5=0.672<0.778$。有偏估计相当于“有放回抽样”，Codex 论文附录指出它会一致地低估，且样本数大于 $5k$ 时差距仍不完全消失。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把结论与出处配对：`,
      pairs: [
        ['RL 提升 Maj@K 但不提升 Pass@K', 'DeepSeekMath'],
        ['大 k 时 base 的 pass@k 高于 RL 模型', 'Yue 等（Does RL Really…）'],
        ['长时间 RL 能解出 base 怎么采都做不对的题', 'ProRL'],
        ['性能与熵满足 R = −a·e^H + b', 'Cui 等（熵机制）'],
      ],
      explanation: md`这四篇构成了“RL 到底学到了什么”的主要证据链，面试时按时间顺序讲最清楚。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`一段 CoT 的 token 熵如下（括号内）。取熵最高的 **25%**（向最近整数取整）作为分叉 token，应选哪些？`,
      items: ['Let (0.21)', 'x (0.05)', 'Wait (1.42)', ', (0.08)', 'maybe (1.10)', 'try (0.64)', 'then (0.33)', '= (0.02)'],
      answer: [2, 4],
      explanation: md`8 个 token 的 25% 是 2 个：熵最高的是 Wait(1.42) 与 maybe(1.10)。try(0.64) 排第三，不选。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 pass@k 实现有 **2 处** bug，点出来：`,
      code: py`
        import numpy as np

        def pass_at_k(n, c, k):
            """n: 总样本数，c: 正确数，k: k in pass@k"""
            if n - c <= k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c, n + 1))
      `,
      bugLines: [5, 7],
      explanation: md`第 5 行应为 \`n - c < k\`：$n-c=k$ 时 $\binom{n-c}{k}=1$，结果是 $1-1/\binom nk$ 而不是 1。第 7 行区间应从 $n-c+1$ 开始：\`np.arange(n - c + 1, n + 1)\`；多乘的一项 $(1-\frac{k}{n-c})$ 会让结果偏大。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出数值稳定的 pass@k（Codex 论文写法；有 2 行干扰项）：`,
      lines: [
        'def pass_at_k(n, c, k):',
        '    if n - c < k:',
        '        return 1.0',
        '    j = np.arange(n - c + 1, n + 1)',
        '    return 1.0 - np.prod(1.0 - k / j)',
      ],
      distractors: ['    return 1.0 - comb(n - c, k) / comb(n, k)', '    return 1.0 - (1.0 - c / n) ** k'],
      explanation: md`直接算组合数在浮点下会溢出（$\binom{2000}{1000}$ 远超 float64 上限）；\`(1 - c/n) ** k\` 是有偏估计。`,
    },
    {
      id: 'q-hack',
      type: 'single',
      difficulty: 2,
      prompt: md`R1 论文附录 B.5 描述的 reward hacking 现象是：`,
      options: [
        '格式奖励导致模型输出空的 think 标签',
        '使用 helpful 奖励模型时，奖励持续上升而推理任务表现下降',
        '代码任务中模型修改了单元测试',
        '语言一致性奖励导致准确率下降',
      ],
      answer: 1,
      explanation: md`附录 B.5：RM 存在系统性偏差时，模型学会生成被 RM 高评分但偏离真实偏好的回答，推理表现下降。D 是语言一致性奖励的消融结论（轻微下降），不属于 hacking。`,
    },
    {
      id: 'q-cov',
      type: 'multi',
      difficulty: 3,
      prompt: md`关于熵的协方差机制，正确的是：（多选）`,
      options: [
        '熵变化与 Cov(log π(a), π(a)·A(a)) 成负相关（vanilla PG 下）',
        '训练中该协方差大多为正，因此熵倾向单调下降',
        'Clip-Cov 对协方差最高的一小部分 token 截断梯度',
        '只要优势均值为 0，熵就不会变化',
      ],
      answer: [0, 1, 2],
      explanation: md`D 错：优势均值为 0 不代表协方差为 0——正优势落在高概率 token 上时熵照样下降。`,
    },
    {
      id: 'q-signal',
      type: 'numeric',
      difficulty: 3,
      prompt: md`0/1 奖励、$G=8$、不除 std（Dr.GRPO 式优势）。一道答对 2 个的题，组内 $\sum_i|\tilde A_i|$ 是多少？`,
      answer: 3,
      tolerance: 0.001,
      explanation: md`mean $=0.25$：2 个正确各 $0.75$，6 个错误各 $0.25$，合计 $1.5+1.5=3$。公式 $2k(G-k)/G=2\times2\times6/8=3$。`,
    },
  ],

  guided: [
    {
      id: 'g-entropy',
      title: '从 logits 稳定地计算 token 熵',
      intro: md`
        \`logits\`：\`(B, T, V)\`，\`mask\`：\`(B, T)\`。返回每个 token 的熵 \`H\`（\`(B, T)\`，padding 处为 0）以及有效 token 上的平均熵。
        logits 可能很大（如 1000），直接 \`exp\` 会溢出——用 log-softmax。
      `,
      template: py`
        import numpy as np

        def token_entropy(logits, mask):
            mx = logits.max(axis=-1, keepdims=True)
            logp = {{1}}                       # 稳定的 log-softmax
            p = np.exp(logp)
            H = {{2}}                          # H = -Σ p·log p
            H = H * mask
            return H, {{3}}                    # 有效 token 上的平均熵
      `,
      steps: [
        {
          blank: 1,
          instruction: md`$\log\text{softmax}(x)=x-m-\log\sum e^{x-m}$，$m$ 为最大值。`,
          answer: 'logits - mx - np.log(np.exp(logits - mx).sum(axis=-1, keepdims=True))',
          accept: ['(logits - mx) - np.log(np.sum(np.exp(logits - mx), axis=-1, keepdims=True))'],
          explain: md`减去最大值后 $e^{x-m}\le1$，不会溢出；最大项为 $e^0=1$，求和 $\ge1$，log 不会遇到 0。`,
        },
        {
          blank: 2,
          instruction: md`沿词表维度求 $-\sum p\log p$。`,
          answer: '-(p * logp).sum(axis=-1)',
          accept: ['-np.sum(p * logp, axis=-1)'],
          explain: md`若 logits 里有 \`-inf\`（被 top-k / top-p 截断的词），$p=0$、$\log p=-\infty$，乘积是 nan——实战题会处理这种情况。`,
        },
        {
          blank: 3,
          instruction: md`对有效 token 求平均。`,
          answer: 'H.sum() / mask.sum()',
          accept: ['np.sum(H) / np.sum(mask)'],
          explain: md`这就是训练面板上的 “entropy” 曲线（按 token 平均）。`,
        },
      ],
      setup: SETUP,
      tests: [
        {
          name: '均匀分布与确定分布',
          code: py`
            logits = np.zeros((1, 2, 4)); logits[0, 1] = [100.0, 0, 0, 0]
            H, mean = token_entropy(logits, np.ones((1, 2)))
            assert abs(H[0, 0] - np.log(4)) < 1e-9, f"均匀分布熵应为 log 4，得到 {H[0, 0]}"
            assert H[0, 1] < 1e-30 + 1e-12, f"近似确定分布熵应接近 0，得到 {H[0, 1]}"
          `,
        },
        {
          name: '大 logits 不溢出、与参考一致',
          code: py`
            rng = np.random.default_rng(0)
            logits = rng.normal(0, 3, size=(2, 5, 7)) + 1000.0
            mask = (np.arange(5)[None, :] < np.array([[3], [5]])).astype(float)
            H, mean = token_entropy(logits, mask)
            eH, emean = _ref_entropy(logits, mask)
            assert np.all(np.isfinite(H)), "出现了 inf / nan：是否减去了最大值？"
            assert np.allclose(H, eH) and abs(mean - emean) < 1e-10, f"期望平均熵 {emean}，得到 {mean}"
          `,
        },
      ],
      outro: md`
        监控熵时还常看**分位数**（高熵 token 占比）而不只是均值——均值稳定时，高熵分叉 token 可能已经在消失。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-passk',
      title: 'pass@k 无偏估计（数值稳定、向量化）',
      difficulty: 'medium',
      source: 'Codex（Chen et al. 2021）',
      timeLimitMin: 15,
      description: md`
        \`\`\`python
        def pass_at_k(n, c, k: int) -> np.ndarray: ...
        \`\`\`

        - \`n\`：每题样本数，可以是 int（所有题相同）或 \`(P,)\` 整数数组；\`c\`：\`(P,)\` 每题正确数；\`k\`：满足 $1\le k\le\min(n)$；
        - 返回 \`(P,)\` float64：每题的 $1-\binom{n-c}{k}/\binom{n}{k}$；$n-c<k$ 时为 1.0。
        - **不得**使用 \`comb\` / \`factorial\`（会检查源码）；$n$ 可达数千，结果必须精确到 1e-9。
      `,
      starter: py`
        import numpy as np

        def pass_at_k(n, c, k):
            # TODO: 逐项连乘 1 - k/j，j 从 n-c+1 到 n；n-c < k 时为 1
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def pass_at_k(n, c, k):
            c = np.asarray(c, dtype=np.int64)
            n = np.broadcast_to(np.asarray(n, dtype=np.int64), c.shape)
            out = np.ones(c.shape, dtype=np.float64)
            for i in range(c.size):
                ni, ci = int(n[i]), int(c[i])
                if ni - ci >= k:                       # 否则任取 k 个必含正确样本，保持 1.0
                    j = np.arange(ni - ci + 1, ni + 1, dtype=np.float64)
                    out[i] = 1.0 - np.prod(1.0 - k / j)
            return out
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：在 log 空间向量化，n 相同时用前缀和）',
          code: py`
            import torch

            def pass_at_k_torch(n: int, c: torch.Tensor, k: int) -> torch.Tensor:
                # S[m] = Σ_{j=k+1}^{m} log(1 - k/j)；所需乘积 = exp(S[n] - S[n-c])（仅当 n-c >= k）
                j = torch.arange(k + 1, n + 1, dtype=torch.float64)
                S = torch.cat([torch.zeros(k + 1, dtype=torch.float64), torch.cumsum(torch.log1p(-k / j), 0)])
                nc = (n - c).clamp(min=k)
                est = 1 - torch.exp(S[n] - S[nc])
                return torch.where(n - c < k, torch.ones_like(est), est)
          `,
        },
      ],
      explanation: md`
        **推导**：$\frac{\binom{n-c}{k}}{\binom nk}=\prod_{j=n-c+1}^{n}\frac{j-k}{j}=\prod_{j=n-c+1}^{n}\Big(1-\frac kj\Big)$（展开阶乘后相邻项约掉）。
        连乘的每一项都在 $(0,1]$，不会溢出；只有 $c$ 项。

        **向量化**：所有题 $n$ 相同时，令 $S(m)=\sum_{j=k+1}^{m}\log(1-k/j)$，则乘积 $=\exp(S(n)-S(n-c))$，一次 cumsum 即可（见 PyTorch 版）。
      `,
      hints: [md`先处理 $n-c<k$ → 1.0。`, md`其余情况：\`1 - np.prod(1 - k / np.arange(n - c + 1, n + 1))\`。`],
      setup: SETUP,
      tests: [
        {
          name: '示例',
          code: py`
            got = pass_at_k(10, np.array([2, 0, 10, 5]), 5)
            exp = np.array([1 - 56 / 252, 0.0, 1.0, 1 - 1 / 252])
            assert np.allclose(got, exp, atol=1e-12), f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: 'k = 1 等于 c / n',
          code: py`
            c = np.arange(0, 21)
            got = pass_at_k(20, c, 1)
            assert np.allclose(got, c / 20), f"pass@1 应为 c/n，得到 {got}"
          `,
        },
        {
          name: '不使用 comb / factorial',
          code: py`
            src = inspect.getsource(pass_at_k)
            for banned in ("comb", "factorial"):
                assert banned not in src, f"请用连乘形式，不要使用 {banned}"
          `,
        },
        {
          name: '随机对拍（精确分数）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(0)
            for _ in range(200):
                n = int(rng.integers(1, 60)); k = int(rng.integers(1, n + 1)); c = int(rng.integers(0, n + 1))
                got = float(pass_at_k(n, np.array([c]), k)[0])
                exp = _ref_passk_one(n, c, k)
                assert abs(got - exp) < 1e-9, f"n={n}, c={c}, k={k}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '大 n 不溢出、数组形式的 n',
          hidden: true,
          code: py`
            got = pass_at_k(np.array([2000, 3000]), np.array([500, 7]), 1000)
            exp = np.array([_ref_passk_one(2000, 500, 1000), _ref_passk_one(3000, 7, 1000)])
            assert np.all(np.isfinite(got)) and np.allclose(got, exp, atol=1e-9), f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：5000 题 × n=200',
          hidden: true,
          code: py`
            rng = np.random.default_rng(1)
            c = rng.integers(0, 201, size=5000)
            t0 = time.perf_counter()
            got = pass_at_k(200, c, 100)
            dt = time.perf_counter() - t0
            assert got.shape == (5000,), f"形状应为 (5000,)，得到 {got.shape}"
            assert dt < 1.5, f"用时 {dt:.2f}s，太慢"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-majk',
          q: md`Maj@k 和 pass@k 有什么区别？RL 后为什么常见“Maj@k 涨、pass@k 不涨”？`,
          a: md`Maj@k 是 k 个样本多数投票的答案是否正确，衡量分布的“众数”是否正确；pass@k 衡量 k 个样本的覆盖。RL 锐化分布：把已有的正确模式概率推高 → 众数更常正确（Maj@k↑），但不增加覆盖、甚至减少多样性（大 k 的 pass@k 不变或下降）。`,
        },
      ],
    },
    {
      id: 'c-fork',
      title: '高熵分叉 token 掩码（含 −inf logits）',
      difficulty: 'medium',
      source: 'Beyond the 80/20 Rule（2025）',
      timeLimitMin: 15,
      description: md`
        只在熵最高的 $\rho$ 比例 token 上计算策略梯度。实现

        \`\`\`python
        def high_entropy_mask(logits, mask, rho=0.2) -> np.ndarray:  # (B, T) bool
        \`\`\`

        - \`logits\`：\`(B, T, V)\`，**可能含 \`-inf\`**（被 top-k / top-p 截断的词，对应概率 0），也可能数值很大；\`mask\`：\`(B, T)\`。
        - 先算每个有效 token 的熵（$0\cdot\log0$ 按 0 处理），在**整个 batch 的有效 token** 中选熵最高的 $k=\max(1,\ \mathrm{round}(\rho N))$ 个（$N$ 为有效 token 数），
          熵相同时按展平后的下标从小到大优先。返回布尔掩码，padding 处为 False。
      `,
      starter: py`
        import numpy as np

        def high_entropy_mask(logits, mask, rho=0.2):
            # TODO 1: 稳定的 log-softmax（注意 -inf）
            # TODO 2: 熵（0·log0 = 0）
            # TODO 3: 在有效 token 中选 top-k
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def high_entropy_mask(logits, mask, rho=0.2):
            x = np.asarray(logits, dtype=np.float64)
            mx = x.max(axis=-1, keepdims=True)                 # 每行至少一个有限值，max 有限
            z = x - mx                                          # -inf 保持 -inf，exp 后为 0
            logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
            p = np.exp(logp)
            H = -np.where(p > 0, p * logp, 0.0).sum(axis=-1)    # 0·(-inf) 会得 nan，必须先 where

            valid = np.asarray(mask) > 0
            idx = np.flatnonzero(valid.reshape(-1))
            k = max(1, int(round(rho * len(idx))))
            order = np.argsort(-H.reshape(-1)[idx], kind="stable")   # 稳定排序：同熵按下标优先
            out = np.zeros(valid.size, dtype=bool)
            out[idx[order[:k]]] = True
            return out.reshape(valid.shape)
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch

            def high_entropy_mask_torch(logits, mask, rho=0.2):
                logp = torch.log_softmax(logits.float(), dim=-1)
                p = logp.exp()
                H = -torch.where(p > 0, p * logp, torch.zeros_like(p)).sum(-1)
                H = H.masked_fill(mask == 0, float("-inf"))
                k = max(1, round(rho * int(mask.sum())))
                thr = torch.topk(H.flatten(), k).values[-1]
                return (H >= thr) & (mask > 0)          # 有并列时可能多选；需严格 k 个时用 topk 的 indices
          `,
        },
      ],
      explanation: md`
        两个数值坑：① 大 logits 必须减最大值；② \`-inf\` logits 让 $p=0$、$\log p=-\infty$，\`p * logp\` 是 nan，必须用 \`np.where(p > 0, ...)\`。
        选择时只在有效 token 中排序，稳定排序保证并列时的确定性。之后把这个掩码乘进 loss mask 即可。
      `,
      hints: [md`\`H = -np.where(p > 0, p * logp, 0).sum(-1)\`。`, md`\`idx = np.flatnonzero(mask.reshape(-1) > 0)\`，对 \`-H.reshape(-1)[idx]\` 做 \`argsort(kind="stable")\`。`],
      setup: SETUP,
      tests: [
        {
          name: '选中最不确定的位置',
          code: py`
            logits = np.zeros((1, 4, 3))
            logits[0, 0] = [10, 0, 0]; logits[0, 1] = [0, 0, 0]; logits[0, 2] = [5, 0, 0]; logits[0, 3] = [0.1, 0, 0]
            got = high_entropy_mask(logits, np.ones((1, 4)), rho=0.5)
            assert got.tolist() == [[False, True, False, True]], f"得到 {got.tolist()}"
          `,
        },
        {
          name: 'padding 不参与、至少选 1 个',
          code: py`
            rng = np.random.default_rng(2)
            logits = rng.normal(size=(2, 3, 5))
            mask = np.array([[1, 0, 0], [0, 0, 0]])
            got = high_entropy_mask(logits, mask, rho=0.01)
            assert got.sum() == 1 and got[0, 0], f"得到 {got.tolist()}"
          `,
        },
        {
          name: '-inf 与大 logits',
          hidden: true,
          code: py`
            rng = np.random.default_rng(3)
            logits = rng.normal(0, 2, size=(3, 6, 8)) + 500.0
            logits[rng.random(logits.shape) < 0.4] = -np.inf
            logits[..., 0] = 500.0                      # 保证每行至少一个有限值
            mask = (np.arange(6)[None, :] < np.array([[6], [4], [2]])).astype(float)
            got = high_entropy_mask(logits, mask, 0.3)
            exp = _ref_fork_mask(logits, mask, 0.3)
            assert got.tolist() == exp.tolist(), "与参考不一致（检查 0·log0 与 -inf 的处理）"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rng = np.random.default_rng(4)
            for _ in range(30):
                B, T, V = rng.integers(1, 4), rng.integers(1, 9), rng.integers(2, 10)
                logits = rng.normal(0, 3, size=(B, T, V))
                mask = (rng.random((B, T)) < 0.7).astype(float); mask.reshape(-1)[0] = 1
                rho = float(rng.choice([0.1, 0.2, 0.35, 0.6, 1.0]))
                got = high_entropy_mask(logits, mask, rho)
                exp = _ref_fork_mask(logits, mask, rho)
                assert got.tolist() == exp.tolist(), f"rho={rho}: 选择不一致"
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-entropy',
      freq: 3,
      q: md`什么是熵坍缩？怎么监控、怎么缓解？`,
      a: md`
        **结论**：RL 训练早期策略熵急剧下降、模型过度自信、探索停止，性能随之饱和。

        **原因**：softmax 策略下熵的变化 $\approx-\eta\,\mathrm{Cov}(\log\pi,\pi A)$；RLVR 中正确答案多是模型本来就较自信的路径，协方差为正，熵单调下降；性能与熵近似满足 $R=-ae^{H}+b$。

        **延伸**：监控平均熵、高熵 token 占比、同组回答多样性、clip_frac。缓解：Clip-Higher、Clip-Cov / KL-Cov、只在高熵分叉 token 上更新、CISPO 保留梯度、调温度；熵正则要慎用。
      `,
    },
    {
      id: 'qa-boundary',
      freq: 3,
      q: md`RLVR 能扩展模型的推理能力边界吗？`,
      a: md`
        **结论**：目前证据倾向于“短程 RLVR 主要是锐化，长程、多样任务的 RL 可能扩展边界”，尚无定论。

        **原因**：DeepSeekMath 发现 RL 提升 Maj@K 不提升 Pass@K；Yue 等发现大 k 时 base 的 pass@k 更高、RL 路径已在 base 分布内，而蒸馏能引入新模式；ProRL 用 KL 控制 + 参考策略重置 + 多样任务的长时间 RL，在 1.5B 模型上报告了 base 完全做不出的题被解出。

        **延伸**：评估要报大 k 的 pass@k（用无偏估计）与多样性指标；想扩展边界需要更好的探索、数据与 on-policy 蒸馏等手段配合。
      `,
    },
    {
      id: 'qa-hack',
      freq: 2,
      q: md`RLVR 中常见的 reward hacking 有哪些？怎么防？`,
      a: md`
        **结论**：答案抽取漏洞、测试覆盖不足、格式分过高、靠长度 / 重复拿分、神经 RM 的风格偏好。

        **原因**：模型优化的是奖励函数而不是任务本身，任何可被利用的规则都会被找到。

        **延伸**：只取最后答案、多答案判错；隐藏测试与沙箱只读；格式奖励只作门槛；超长整形与重复检测截断；推理任务不用或少用神经 RM，并持续抽查高奖励样本。
      `,
    },
    {
      id: 'qa-data',
      freq: 2,
      q: md`RL 训练题目怎么选？要不要做课程？`,
      a: md`
        **结论**：优先通过率在 (0, 1) 之间、接近 0.5 的题；随模型变强动态更新难度分布。

        **原因**：0/1 奖励下信号强度 $\propto p(1-p)$，恒对 / 恒错的题没有梯度（DAPO 动态采样）；过易的题信号弱。

        **延伸**：离线预估通过率筛题、难度分桶由易到难、逐步放开最大长度（R1-Zero 32K → 65K）；多领域时为不同领域 / 推理强度训练专家再合并（DeepSeek-V4）。
      `,
    },
  ],

  cheatsheet: md`
    **熵**：$H=-\sum p\log p$；$\Delta H\approx-\eta\,\mathrm{Cov}(\log\pi,\pi A)$；$R=-ae^{H}+b$

    **pass@k 无偏估计（Codex）**
    \`\`\`python
    def pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    \`\`\`
    - $1-(1-c/n)^k$ 有偏（低估）
    - DeepSeekMath：Maj@K↑ Pass@K 不变；Yue 等：大 k 时 base 更好；ProRL：长程 RL 可扩展边界（1.5B）

    | 现象 | 手段 |
    |---|---|
    | 熵坍缩 | Clip-Higher、Clip-Cov/KL-Cov、高熵 20% token、CISPO |
    | 无信号题 | 动态采样、通过率筛题 |
    | 长度虚涨 | token 级 / 常数归一化、超长整形 |
    | reward hacking | 严格抽取、隐藏测试、格式作门槛 |

    - 稳定熵：\`H = -np.where(p > 0, p * logp, 0).sum(-1)\`（防 -inf）
  `,
})
