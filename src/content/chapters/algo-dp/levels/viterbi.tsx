/**
 * algo-dp-viterbi：解码 DP —— HMM Viterbi、log 域、NumPy 向量化、与 beam search 的关系、Unigram 分词
 */
import { defineLevel, md, py } from '@/content/helpers'
import { TrellisViz, BroadcastViz, BeamVsViterbiViz, UnigramLatticeViz } from '../viz/ViterbiViz'

export default defineLevel({
  learn: [
    {
      id: 'trellis',
      title: 'HMM 解码：网格图上的最优路径',
      body: md`
        **隐马尔可夫模型（HMM）**：隐状态序列 $z_1 \dots z_T$ 是一条马尔可夫链，每个状态发射一个观测 $x_t$。参数为初始分布 $\pi$、转移矩阵 $A_{ij} = P(z_t = j \mid z_{t-1} = i)$、发射矩阵 $B_j(o) = P(x_t = o \mid z_t = j)$。

        **解码问题**：给定观测，求最可能的隐状态序列
        $$
        z^* = \arg\max_{z} P(z \mid x) = \arg\max_{z}\ \pi_{z_1} B_{z_1}(x_1) \prod_{t=2}^{T} A_{z_{t-1} z_t} B_{z_t}(x_t)
        $$
        （$P(x)$ 与 $z$ 无关，所以最大化后验等价于最大化联合概率。）暴力枚举有 $N^T$ 条路径。

        **Viterbi = 在网格图（trellis）上做 DP**：
        - 状态：$\delta_t(j)$ = 在时刻 $t$ 处于状态 $j$ 的**所有**路径中最大的（log）概率
        - 转移：$\delta_t(j) = \max_i \big[\delta_{t-1}(i) + \log A_{ij}\big] + \log B_j(x_t)$，同时记录回溯指针 $\psi_t(j) = \arg\max_i(\cdot)$
        - 初始化：$\delta_1(j) = \log \pi_j + \log B_j(x_1)$
        - 答案：$\max_j \delta_T(j)$，再沿 $\psi$ 倒推出整条路径

        时间 $O(TN^2)$，空间 $O(TN)$（回溯指针）。能这样做的根本原因是**马尔可夫性**：到达 $(t, j)$ 之后的未来只取决于“当前在 $j$”，所以以 $j$ 结尾的众多路径里只需保留最优的一条。
      `,
      viz: TrellisViz,
      after: md`
        观察 $t=1$ 的三个格子：贪心地看，$t=0$ 时“阴”最好（−1.90），但最终最优路径是 **阴 → 雨 → 晴 → 阴 → 雨**，第二步走的是当时看起来并不突出的“雨”。Viterbi 不在中途做决定，而是为**每个状态**保留一条最优路径，直到最后才回溯。

        > [!NOTE] 和前面学过的 DP 是同一回事
        > 网格图是一个分层 DAG，$\delta$ 就是“从源点到每个节点的最长路”。LIS、编辑距离、Unigram 分词同样是 DAG 上的最优路径问题。
      `,
      checkpoint: {
        id: 'cp-trellis',
        type: 'single',
        prompt: md`在上面的模型中，计算 $\delta_1(\text{晴})$ 时（$t=1$ 观测为“散步”），最优前驱是哪个状态？（$\delta_0 = [-2.81, -1.90, -2.12]$，对应 晴、阴、雨；$\log A_{\cdot,\text{晴}} = [\log 0.2, \log 0.4, \log 0.7]$）`,
        options: ['晴', '阴', '雨', '三者相同'],
        answer: 2,
        explanation: md`三个候选：晴 $-2.81 + \log 0.2 = -4.42$；阴 $-1.90 + \log 0.4 = -2.81$；雨 $-2.12 + \log 0.7 = -2.48$。雨最大。虽然 $\delta_0$ 中“阴”更高，但“阴→晴”的转移概率只有 0.4，而“雨→晴”是 0.7。`,
      },
    },
    {
      id: 'log',
      title: 'log 域与 Forward 算法：max 与 sum 的区别',
      body: md`
        **为什么用 log**：路径概率是 $T$ 个小于 1 的数连乘。float64 能表示的最小正数约 $4.9 \times 10^{-324}$，每步概率 0.01 时 $T = 200$ 就是 $10^{-400}$，直接下溢成 0，所有路径都“相等”。取 log 后连乘变连加，$\log$ 单调递增，$\arg\max$ 不变：
        $$
        \max \prod_t p_t \;\Longleftrightarrow\; \max \sum_t \log p_t
        $$
        零概率对应 $\log 0 = -\infty$，NumPy 中 \`np.log(0)\` 得到 \`-inf\`（伴随一个 RuntimeWarning），加法和 \`max\` 都能正确处理。

        **Viterbi vs Forward**：两者是同一张网格图上的同一个递推，只把“合并”的运算换掉：

        | | Viterbi | Forward |
        |---|---|---|
        | 合并前驱 | $\max_i$ | $\sum_i$（log 域用 $\operatorname{logsumexp}$） |
        | 结果 | 最优路径及其概率 $\max_z P(x, z)$ | 观测概率 $P(x) = \sum_z P(x, z)$ |
        | 用途 | 解码（序列标注、分词） | 似然计算、训练（Baum-Welch）、CTC 损失 |

        这种“把 $(+, \times)$ 换成 $(\max, +)$”的视角叫**半环（semiring）**：同一套动态规划代码，换个半环就从求和变成求最优。CTC 的前向算法、CRF 的配分函数都是 sum 版本，对应的 Viterbi 都是 max 版本。
      `,
      checkpoint: {
        id: 'cp-log',
        type: 'judge',
        prompt: md`Viterbi 算法输出的最优路径概率 $\max_z P(x, z)$ 就等于观测序列的概率 $P(x)$。`,
        answer: false,
        explanation: md`错误。$P(x) = \sum_z P(x, z)$ 是所有路径之和，需要 Forward 算法。在本关的玩具模型里，最优路径概率约 $3.5 \times 10^{-4}$，而 $P(x) \approx 3.6 \times 10^{-3}$——最优路径只占约 10%。`,
      },
    },
    {
      id: 'numpy',
      title: 'NumPy 实现：一步 = 广播 + 按列 max',
      body: md`
        把内层对 $i, j$ 的双重循环向量化：\`delta\` 形状 \`(N,)\`，\`log_A\` 形状 \`(N, N)\`，

        \`\`\`python
        scores = delta[:, None] + log_A        # (N, 1) + (N, N) → (N, N)，scores[i, j] = δ(i) + log A[i, j]
        back_t = scores.argmax(axis=0)         # (N,)：对每个 j（列），最优的前驱 i
        delta  = scores.max(axis=0) + log_B[:, o_t]
        \`\`\`

        **轴别搞反**：行是前驱 $i$、列是当前 $j$，所以对 **axis=0**（沿行方向压缩）取 max。若写成 \`delta[None, :] + log_A.T\` 也可以，但要保持一致。

        回溯：\`path[T-1] = delta.argmax()\`，然后 \`path[t-1] = back[t][path[t]]\` 倒推。时间仍是 $O(TN^2)$，但 $N^2$ 的部分交给 NumPy 的 C 循环，Python 层只剩长度为 $T$ 的循环。
      `,
      viz: BroadcastViz,
      after: md`
        > [!TIP] 批量解码
        > 若有一批 $B$ 条序列，\`delta\` 形状为 \`(B, N)\`，一步写成 \`scores = delta[:, :, None] + log_A[None]\`（\`(B, N, N)\`），沿 \`axis=1\` 取 max。PyTorch 中对应 \`(delta.unsqueeze(-1) + log_A).max(dim=1)\`，返回值同时带 \`values\` 与 \`indices\`。
      `,
      checkpoint: {
        id: 'cp-numpy',
        type: 'fill',
        prompt: md`
          \`delta\` 形状为 \`(N,)\`，\`log_A\` 形状为 \`(N, N)\`。\`scores = delta[:, None] + log_A\` 的形状是 ____(1)____；
          为了对每个当前状态 $j$ 求最优前驱，应调用 \`scores.argmax(axis=\`____(2)____\`)\`。
        `,
        blanks: [
          { label: '(1)', answers: ['(N, N)', '(N,N)', 'N,N', 'NxN', 'N×N'] },
          { label: '(2)', answers: ['0'], numeric: true },
        ],
        explanation: md`\`(N, 1)\` 与 \`(N, N)\` 广播为 \`(N, N)\`，第 0 维是前驱 $i$，第 1 维是当前 $j$。对每一列（固定 $j$）在 $i$ 上取最优，即 \`axis=0\`。`,
      },
    },
    {
      id: 'beam',
      title: '与 beam search 的关系：精确 DP vs 剪枝近似',
      body: md`
        **beam search** 在每一步只保留得分最高的 $B$ 条**路径前缀**，扩展后再截断到 $B$ 条。在 HMM 上它与 Viterbi 的区别是：

        | | Viterbi | Beam search |
        |---|---|---|
        | 每步保留 | 每个状态各一条（共 $N$ 条） | 全局前 $B$ 条，可能多条结尾相同 |
        | 合并 | 结尾状态相同的路径**合并**，只留最优 | 不合并 |
        | 结果 | 精确最优 | 近似；$B=1$ 即贪心 |
        | 代价 | $O(TN^2)$ | $O(T \cdot B \cdot N \log(BN))$ |

        **为什么 HMM 上能精确、LLM 上不能**：Viterbi 的合并依赖马尔可夫性——未来只取决于当前状态，状态只有 $N$ 个。语言模型的“状态”是**整个前缀**，两条不同前缀之后的分布不同，无法合并；精确搜索需要遍历 $V^T$ 条序列（$V \approx 10^5$），只能用 beam 这种剪枝近似。

        拖动下面的 beam 宽度：$B = 1$（贪心）与 $B = 2$ 都丢掉了最优路径——最优路径在 $t=1$ 时的前缀“阴→雨”只排第 3。
      `,
      viz: BeamVsViterbiViz,
      after: md`
        > [!WARN] 常见误解
        > “beam 宽度 $B \ge N$ 就等价于 Viterbi”是错的：不做合并时，前 $B$ 条里可能有两条结尾状态相同，把另一个状态的最优前缀挤掉。只有在每步按结尾状态合并（recombination）后，$B = N$ 才精确——那就是 Viterbi 本身。

        > [!INTERVIEW] 延伸
        > 在 LLM 解码中，beam search 倾向于生成偏短、重复的文本（概率最大 ≠ 质量最好），因此开放式生成多用采样；beam 仍常用于翻译、ASR 等有明确“正确答案”的任务。语音识别中的 WFST 解码、CTC prefix beam search 都是“Viterbi + 剪枝”的结合。
      `,
      checkpoint: {
        id: 'cp-beam',
        type: 'single',
        prompt: md`为什么 HMM 可以用 Viterbi 精确解码，而 Transformer 语言模型的解码通常只能用 beam search 近似？`,
        options: [
          'HMM 满足马尔可夫性，结尾状态相同的路径可以合并；LM 的条件依赖整个前缀，不同前缀无法合并，状态空间是 V^T',
          'Transformer 的 softmax 不能取 log',
          'Viterbi 只能处理离散观测',
          'Transformer 的词表太小，beam search 就足够精确',
        ],
        answer: 0,
        explanation: md`DP 的前提是“只看状态就能决定未来”。LM 的 $p(y_t \mid y_{<t})$ 依赖完整前缀，状态数随长度指数增长，没有可合并的子问题。B、C 与问题无关；D 恰恰相反，词表很大（$10^5$ 量级）。`,
      },
    },
    {
      id: 'unigram',
      title: '大模型岗场景：Unigram 分词 = 位置格上的 Viterbi',
      body: md`
        **Unigram 语言模型分词**（Kudo, 2018, *Subword Regularization*）假设一段文本被切成若干 piece 后，各 piece **独立**出现：
        $$
        P(\text{x}_1, \dots, \text{x}_m) = \prod_{k} p(\text{x}_k), \qquad \text{切分}^* = \arg\max \sum_k \log p(\text{x}_k)
        $$
        编码（推理时分词）就是在“位置格”上做 Viterbi：
        - 状态：\`best[i]\` = 前 $i$ 个字符的最优切分的 log 概率，\`best[0] = 0\`
        - 转移：$best[i] = \max_{j < i,\ s[j:i] \in V} best[j] + \log p(s[j:i])$
        - 只需枚举长度不超过最长 piece $L$ 的 $s[j:i]$，复杂度 $O(nL)$

        SentencePiece 库同时支持 BPE 与 Unigram；按 Hugging Face 文档，**T5、BigBird、Pegasus** 等模型使用 Unigram，而 GPT-2、Llama、Qwen 等主流 LLM 使用（字节级）BPE。Unigram 训练时用 EM 估计 $p$ 并迭代删除对似然贡献最小的 piece；训练中还可以按概率**采样**不同切分（subword regularization）做数据增强。
      `,
      viz: UnigramLatticeViz,
      after: md`
        > [!TIP] 和 WordPiece 贪心最长匹配的区别
        > WordPiece（BERT）编码时从左到右每次取**最长**的匹配 piece，不回头；Unigram 用 Viterbi 求**全局**最大概率。上例中贪心先吃掉“研究生”，导致后面只能切出低频的“命”。
      `,
      checkpoint: {
        id: 'cp-unigram',
        type: 'numeric',
        prompt: md`
          词表（括号内为概率）：\`a\`(0.2)、\`b\`(0.2)、\`ab\`(0.1)、\`abc\`(0.01)、\`c\`(0.1)、\`bc\`(0.05)。
          对 \`"abc"\`，Unigram 最大概率切分的**概率**是多少？（直接填小数）
        `,
        answer: 0.01,
        tolerance: 0.001,
        explanation: md`
          全部切分：\`a/b/c\` = $0.2 \times 0.2 \times 0.1 = 0.004$；\`ab/c\` = $0.1 \times 0.1 = 0.01$；\`a/bc\` = $0.2 \times 0.05 = 0.01$；\`abc\` = 0.01。
          最大概率是 0.01（三种切分并列）。实际实现中并列时取哪一个取决于比较顺序，这也是测试题通常用随机浮点概率避免平局的原因。
        `,
      },
    },
  ],

  quiz: [
    {
      id: 'q-cplx',
      type: 'single',
      difficulty: 1,
      prompt: md`$N$ 个隐状态、长度 $T$ 的观测序列，Viterbi 算法的时间复杂度是：`,
      options: ['$O(TN)$', '$O(TN^2)$', '$O(N^T)$', '$O(T^2 N)$'],
      answer: 1,
      explanation: md`每个时间步、每个状态 $j$ 要在 $N$ 个前驱中取 max，共 $T \cdot N \cdot N$。$O(N^T)$ 是暴力枚举所有路径。`,
    },
    {
      id: 'q-log',
      type: 'judge',
      difficulty: 1,
      prompt: md`在 log 域做 Viterbi 时，连乘变成连加，而 $\max$ 与 $\arg\max$ 的结果不变。`,
      answer: true,
      explanation: md`$\log$ 严格单调递增，所以 $\max_z \prod p = \exp(\max_z \sum \log p)$，最优路径完全相同。log 域的目的是避免下溢。`,
    },
    {
      id: 'q-flops',
      type: 'numeric',
      difficulty: 1,
      prompt: md`$T = 100$、$N = 1000$ 的 HMM，按 $T \cdot N^2$ 估算，Viterbi 大约需要多少次“加法 + 比较”？`,
      answer: 1e8,
      tolerance: 0.01,
      explanation: md`$100 \times 1000^2 = 10^8$。这也是大状态空间（如语音识别的大词表）需要剪枝（beam）的原因。`,
    },
    {
      id: 'q-fwd',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 Viterbi 与 Forward 算法，下列说法正确的是：`,
      options: [
        'Forward 在合并前驱时用求和（log 域为 logsumexp），Viterbi 用 max',
        '两者的时间复杂度都是 $O(TN^2)$',
        'Forward 算法也需要保存回溯指针',
        'Viterbi 求的是 $\\arg\\max_z P(z \\mid x)$',
        'Forward 算法的结果是最优路径的概率',
      ],
      answer: [0, 1, 3],
      explanation: md`C 错：Forward 只求总概率，不需要回溯。E 错：Forward 得到的是 $P(x)$（所有路径之和）。D 正确：$P(z \mid x) = P(x, z)/P(x)$，分母与 $z$ 无关。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 NumPy 版 Viterbi 的执行步骤按顺序排列：`,
      items: [
        '`delta = log_pi + log_B[:, obs[0]]`',
        '对每个后续观测：`scores = delta[:, None] + log_A`',
        '`back.append(scores.argmax(axis=0))`，`delta = scores.max(axis=0) + log_B[:, o]`',
        '`path = [delta.argmax()]`，沿 back 倒序回溯',
        '反转 path，返回路径与 `delta.max()`',
      ],
      explanation: md`初始化 → 逐步“广播 + 列 max”并记录回溯指针 → 从最后一列的最大值出发回溯 → 反转。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把算法与描述配对：`,
      pairs: [
        ['Viterbi', '按结尾状态合并路径，精确求最优序列'],
        ['Beam search', '每步保留得分前 B 的前缀，剪枝近似'],
        ['贪心解码', 'B = 1 的 beam search'],
        ['Forward 算法', '对所有路径求和，得到观测概率'],
      ],
      explanation: md`四者都在同一张网格图上工作：Viterbi 与 Forward 是精确 DP（max / sum 半环），beam 与贪心是对“最优路径”问题的剪枝近似。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 Unigram 最大概率分词（\`logp\` 为 piece → log 概率的字典，\`L\` 为最长 piece 长度），有 2 行干扰项：`,
      lines: [
        'def segment(s, logp, L):',
        "    best = [0.0] + [float('-inf')] * len(s)",
        '    prev = [0] * (len(s) + 1)',
        '    for i in range(1, len(s) + 1):',
        '        for j in range(max(0, i - L), i):',
        '            w = s[j:i]',
        '            if w in logp and best[j] + logp[w] > best[i]:',
        '                best[i], prev[i] = best[j] + logp[w], j',
        '    out, i = [], len(s)',
        '    while i > 0:',
        '        out.append(s[prev[i]:i])',
        '        i = prev[i]',
        '    return out[::-1]',
      ],
      distractors: ['            if w in logp and best[j] * logp[w] > best[i]:', "    best = [0.0] * (len(s) + 1)"],
      explanation: md`log 域里概率相乘变成 log 相加，所以是 \`best[j] + logp[w]\`；干扰项的乘法是概率域思维的残留。\`best\` 必须初始化为 \`-inf\`（不可达），全 0 会让“没有合法切分的位置”看起来比真实切分更好（log 概率都是负数）。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 NumPy Viterbi 返回的路径经常是错的。点出有 bug 的那一行：`,
      code: py`
        import numpy as np

        def viterbi(log_pi, log_A, log_B, obs):
            delta = log_pi + log_B[:, obs[0]]
            back = []
            for o in obs[1:]:
                scores = delta[:, None] + log_A
                back.append(scores.argmax(axis=1))
                delta = scores.max(axis=0) + log_B[:, o]
            path = [int(delta.argmax())]
            for bp in reversed(back):
                path.append(int(bp[path[-1]]))
            return path[::-1], float(delta.max())
      `,
      bugLines: [8],
      explanation: md`第 8 行应为 \`argmax(axis=0)\`。\`scores[i, j]\` 的行是前驱 $i$、列是当前 $j$；\`axis=1\` 求的是“从每个 $i$ 出发最优的下一个 $j$”，与第 9 行的 \`max(axis=0)\` 不一致，回溯指针指向了错误的前驱（分数仍然正确，路径却错了——这种 bug 只看分数的测试抓不出来）。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 3,
      prompt: md`
        本关玩具模型、beam 宽度 $B = 2$。$t=0$ 后 beam 中是 \`[阴]\`（概率 0.15）与 \`[雨]\`（0.12）。$t=1$ 观测“散步”，各扩展的概率为
        前缀概率 × $A$ × $B(\cdot, \text{散步})$。哪两条会被保留？
      `,
      items: ['阴→晴', '阴→阴', '阴→雨', '雨→晴', '雨→阴', '雨→雨'],
      answer: [0, 3],
      explanation: md`
        阴→晴 $0.15 \times 0.4 \times 0.6 = 0.036$；阴→阴 $0.0045$；阴→雨 $0.15 \times 0.5 \times 0.3 = 0.0225$；雨→晴 $0.12 \times 0.7 \times 0.6 = 0.0504$；雨→阴 $0.0036$；雨→雨 $0.0072$。
        保留 雨→晴、阴→晴。而 Viterbi 最优路径的前缀正是 **阴→雨**（第 3 名），在这一步就被剪掉了——之后再也找不回来。
      `,
    },
    {
      id: 'q-models',
      type: 'single',
      difficulty: 2,
      prompt: md`根据 Hugging Face 文档，下列哪个模型的分词器使用 Unigram 算法（通过 SentencePiece）？`,
      options: ['T5', 'GPT-2', 'BERT', 'Llama 3'],
      answer: 0,
      explanation: md`T5 使用 SentencePiece Unigram。GPT-2 是字节级 BPE；BERT 是 WordPiece（编码时贪心最长匹配）；Llama 3 使用基于 tiktoken 的 BPE（128K 词表）。`,
    },
    {
      id: 'q-beam-n',
      type: 'judge',
      difficulty: 3,
      prompt: md`在 HMM 上，只要 beam 宽度 $B \ge N$（状态数），不合并相同结尾状态的 beam search 就一定能找到 Viterbi 最优路径。`,
      answer: false,
      explanation: md`错误。前 $B$ 条路径中可能有多条结尾状态相同（例如两条都以“晴”结尾），它们挤掉了以“雨”结尾的最优前缀。随机生成的 3 状态模型中，这种失败并不罕见。只有每步**按结尾状态合并**（每个状态只留最优）时，保留 $N$ 条才是精确的——那正是 Viterbi。`,
    },
  ],

  guided: [
    {
      id: 'g-viterbi-np',
      title: 'NumPy 实现 log 域 Viterbi',
      intro: md`
        实现 \`viterbi(log_pi, log_A, log_B, obs)\`：
        - \`log_pi\`：\`(N,)\`；\`log_A\`：\`(N, N)\`，\`log_A[i, j]\` = log P(j | i)；\`log_B\`：\`(N, M)\`，\`log_B[j, o]\` = log P(o | j)
        - \`obs\`：长度 $T \ge 1$ 的观测下标列表
        - 返回 \`(path, best)\`：最优状态序列（int 列表）与其 log 概率
      `,
      template: py`
        import numpy as np

        def viterbi(log_pi, log_A, log_B, obs):
            delta = {{1}}
            back = []
            for o in obs[1:]:
                scores = {{2}}
                back.append({{3}})
                delta = {{4}}
            path = [int(delta.argmax())]
            for bp in reversed(back):
                path.append({{5}})
            return path[::-1], float(delta.max())
      `,
      steps: [
        {
          blank: 1,
          instruction: md`初始化 $\delta_0(j) = \log \pi_j + \log B_j(x_0)$，形状 \`(N,)\`。`,
          answer: 'log_pi + log_B[:, obs[0]]',
          accept: ['log_B[:, obs[0]] + log_pi'],
        },
        {
          blank: 2,
          instruction: md`广播出 \`(N, N)\` 的得分矩阵：行是前驱 $i$，列是当前 $j$。`,
          answer: 'delta[:, None] + log_A',
          accept: ['delta.reshape(-1, 1) + log_A', 'log_A + delta[:, None]', 'delta[:, np.newaxis] + log_A'],
          explain: md`\`delta[:, None]\` 是列向量 \`(N, 1)\`，加到 \`log_A\` 的每一列上：\`scores[i, j] = δ(i) + log A[i, j]\`。`,
        },
        {
          blank: 3,
          instruction: md`记录每个当前状态 $j$ 的最优前驱。`,
          answer: 'scores.argmax(axis=0)',
          accept: ['np.argmax(scores, axis=0)'],
          explain: md`沿行方向（axis=0）对每一列取 argmax。写成 axis=1 分数不变、路径却错——测试会专门检查路径。`,
        },
        {
          blank: 4,
          instruction: md`新的 $\delta$：每列最大值再加上当前观测的发射 log 概率。`,
          answer: 'scores.max(axis=0) + log_B[:, o]',
          accept: ['np.max(scores, axis=0) + log_B[:, o]', 'log_B[:, o] + scores.max(axis=0)'],
        },
        {
          blank: 5,
          instruction: md`回溯：当前路径最后一个状态是 \`path[-1]\`，它在这一步的最优前驱是什么？`,
          answer: 'int(bp[path[-1]])',
          accept: ['bp[path[-1]]', 'int(bp[path[-1]].item())'],
          explain: md`\`back\` 中第 $t$ 个元素存的是“到达 $t$ 时各状态的最优前驱”，从最后一步往前查即可。`,
        },
      ],
      setup: py`
        import itertools
        import numpy as np

        def _ref_brute(log_pi, log_A, log_B, obs):
            N = len(log_pi)
            best, arg = -np.inf, None
            for z in itertools.product(range(N), repeat=len(obs)):
                s = log_pi[z[0]] + log_B[z[0], obs[0]]
                for t in range(1, len(obs)):
                    s += log_A[z[t - 1], z[t]] + log_B[z[t], obs[t]]
                if s > best:
                    best, arg = s, list(z)
            return arg, best

        def _rand_hmm(rng, N, M):
            norm = lambda x: x / x.sum(axis=-1, keepdims=True)
            return (np.log(norm(rng.random(N) + 0.05)), np.log(norm(rng.random((N, N)) + 0.05)),
                    np.log(norm(rng.random((N, M)) + 0.05)))
      `,
      tests: [
        {
          name: '本关玩具模型',
          code: py`
            pi = np.log([0.1, 0.5, 0.4])
            A = np.log([[0.2, 0.5, 0.3], [0.4, 0.1, 0.5], [0.7, 0.1, 0.2]])
            B = np.log([[0.6, 0.1, 0.3], [0.3, 0.6, 0.1], [0.3, 0.2, 0.5]])
            path, best = viterbi(pi, A, B, [0, 0, 2, 1, 2])
            assert [int(z) for z in path] == [1, 2, 0, 1, 2], f"路径应为 [1, 2, 0, 1, 2]，得到 {path}"
            assert abs(best - (-7.945154883482258)) < 1e-6, f"log 概率应为 -7.9452，得到 {best}"
          `,
        },
        {
          name: '随机对拍（暴力枚举）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(42)
            for _ in range(40):
                N, M, T = int(rng.integers(2, 5)), int(rng.integers(2, 4)), int(rng.integers(1, 6))
                lp, la, lb = _rand_hmm(rng, N, M)
                obs = [int(x) for x in rng.integers(0, M, T)]
                path, best = viterbi(lp, la, lb, obs)
                ep, eb = _ref_brute(lp, la, lb, obs)
                assert abs(best - eb) < 1e-9, f"N={N}, T={T}: log 概率期望 {eb:.6f}，得到 {best:.6f}"
                assert [int(z) for z in path] == ep, f"N={N}, T={T}: 路径期望 {ep}，得到 {list(path)}"
          `,
        },
      ],
      outro: md`
        这 10 行就是面试中最常被要求手写的 Viterbi。把 \`max / argmax\` 换成 \`logsumexp\` 并去掉回溯，就是 Forward 算法。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-viterbi',
      title: 'HMM 解码：NumPy Viterbi（含零概率）',
      difficulty: 'medium',
      source: '大模型岗手撕 · 序列解码',
      timeLimitMin: 20,
      description: md`
        实现 log 域 Viterbi 解码：

        \`\`\`python
        def viterbi(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray, obs: list[int]) -> tuple[list[int], float]: ...
        \`\`\`

        - \`log_pi\`：\`(N,)\`；\`log_A\`：\`(N, N)\`，\`log_A[i, j] = log P(z_t=j | z_{t-1}=i)\`；\`log_B\`：\`(N, M)\`，\`log_B[j, o] = log P(x_t=o | z_t=j)\`
        - 概率可能为 0，对应的 log 值为 \`-inf\`；保证至少存在一条概率大于 0 的路径
        - 返回 \`(path, best)\`：长度为 $T$ 的 **Python int 列表**，以及该路径的联合 log 概率（float）
        - 要求时间 $O(TN^2)$，**对状态维度向量化**（隐藏测试 $N = 64$、$T = 400$）

        **示例**（本关玩具模型）：\`obs = [0, 0, 2, 1, 2]\` → \`([1, 2, 0, 1, 2], -7.9452)\`
      `,
      starter: py`
        import numpy as np

        def viterbi(log_pi, log_A, log_B, obs):
            # TODO: delta 初始化 → 每步广播 + 按列 max/argmax → 回溯
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def viterbi(log_pi, log_A, log_B, obs):
            log_pi, log_A, log_B = (np.asarray(x, dtype=np.float64) for x in (log_pi, log_A, log_B))
            T = len(obs)
            N = log_pi.shape[0]
            delta = log_pi + log_B[:, obs[0]]                 # (N,)
            back = np.zeros((T, N), dtype=np.int64)           # back[t, j]: t 时刻处于 j 的最优前驱
            for t in range(1, T):
                scores = delta[:, None] + log_A               # (N, N)：scores[i, j] = δ(i) + log A[i, j]
                back[t] = scores.argmax(axis=0)
                delta = scores.max(axis=0) + log_B[:, obs[t]]
            path = [int(delta.argmax())]
            for t in range(T - 1, 0, -1):
                path.append(int(back[t, path[-1]]))
            return path[::-1], float(delta.max())
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，支持 batch）',
          code: py`
            import torch

            def viterbi_batch(log_pi, log_A, log_B, obs):
                """log_pi: (N,), log_A: (N, N), log_B: (N, M), obs: (Bsz, T) LongTensor"""
                Bsz, T = obs.shape
                delta = log_pi[None, :] + log_B[:, obs[:, 0]].T          # (Bsz, N)
                backs = []
                for t in range(1, T):
                    scores = delta.unsqueeze(-1) + log_A[None]           # (Bsz, N, N)
                    best, arg = scores.max(dim=1)                        # 沿前驱维取 max
                    backs.append(arg)                                    # (Bsz, N)
                    delta = best + log_B[:, obs[:, t]].T
                best_score, last = delta.max(dim=-1)                     # (Bsz,)
                path = [last]
                for arg in reversed(backs):
                    last = arg.gather(1, last.unsqueeze(1)).squeeze(1)
                    path.append(last)
                return torch.stack(path[::-1], dim=1), best_score        # (Bsz, T), (Bsz,)
          `,
        },
      ],
      explanation: md`
        **递推**：$\delta_t(j) = \max_i[\delta_{t-1}(i) + \log A_{ij}] + \log B_j(x_t)$，用广播一次算出 \`(N, N)\` 的得分矩阵，\`max/argmax(axis=0)\` 得到新 δ 与回溯指针。

        **零概率**：\`-inf\` 与有限数相加仍为 \`-inf\`，\`max\` / \`argmax\` 正常工作；只要不出现 \`-inf - (-inf)\`（例如做了归一化减法）就不会产生 NaN。

        **复杂度**：时间 $O(TN^2)$，空间 $O(TN)$。Python 层只有长度 $T$ 的循环。

        **常见错误**：\`argmax\` 轴写反（分数对、路径错）；回溯少走一步或多走一步；返回 numpy 整数数组而不是 int 列表（本题测试做了兼容，但面试时要注意类型）；在概率域连乘导致长序列下溢。
      `,
      complexity: { time: 'O(T·N²)', space: 'O(T·N)' },
      hints: [
        md`\`delta\` 形状 \`(N,)\`；一步的得分矩阵是 \`delta[:, None] + log_A\`，形状 \`(N, N)\`。`,
        md`对每一列（当前状态 $j$）在行（前驱 $i$）上取 max 与 argmax：\`axis=0\`。`,
        md`回溯：\`path = [delta.argmax()]\`，然后 \`for t in range(T-1, 0, -1): path.append(back[t, path[-1]])\`，最后反转。`,
      ],
      setup: py`
        import itertools, time
        import numpy as np

        def _ref_brute(log_pi, log_A, log_B, obs):
            N = len(log_pi)
            best, arg = -np.inf, None
            for z in itertools.product(range(N), repeat=len(obs)):
                s = log_pi[z[0]] + log_B[z[0], obs[0]]
                for t in range(1, len(obs)):
                    s += log_A[z[t - 1], z[t]] + log_B[z[t], obs[t]]
                if s > best:
                    best, arg = s, list(z)
            return arg, best

        def _ref_fast(log_pi, log_A, log_B, obs):
            delta = log_pi + log_B[:, obs[0]]
            back = []
            for o in obs[1:]:
                sc = delta[:, None] + log_A
                back.append(sc.argmax(0))
                delta = sc.max(0) + log_B[:, o]
            p = [int(delta.argmax())]
            for bp in reversed(back):
                p.append(int(bp[p[-1]]))
            return p[::-1], float(delta.max())

        def _path_score(log_pi, log_A, log_B, obs, z):
            s = log_pi[z[0]] + log_B[z[0], obs[0]]
            for t in range(1, len(obs)):
                s += log_A[z[t - 1], z[t]] + log_B[z[t], obs[t]]
            return float(s)

        def _rand_hmm(rng, N, M, zero_frac=0.0):
            def norm(x):
                return x / x.sum(axis=-1, keepdims=True)
            pi, A, B = rng.random(N) + 0.05, rng.random((N, N)) + 0.05, rng.random((N, M)) + 0.05
            if zero_frac:
                A = A * (rng.random((N, N)) > zero_frac)
                A[np.arange(N), rng.integers(0, N, N)] += 0.5        # 每行至少一个非零
            with np.errstate(divide="ignore"):
                return np.log(norm(pi)), np.log(norm(A)), np.log(norm(B))

        def _check(lp, la, lb, obs, exp_score):
            path, best = viterbi(lp, la, lb, obs)
            path = [int(z) for z in path]
            assert len(path) == len(obs), f"路径长度应为 {len(obs)}，得到 {len(path)}"
            assert abs(float(best) - exp_score) < 1e-6, f"log 概率期望 {exp_score:.6f}，得到 {best}"
            ps = _path_score(lp, la, lb, obs, path)
            assert abs(ps - exp_score) < 1e-6, f"返回的路径实际得分 {ps:.6f} ≠ 最优 {exp_score:.6f}（argmax 的轴写反了？）"
      `,
      tests: [
        {
          name: '示例：玩具模型',
          code: py`
            lp = np.log([0.1, 0.5, 0.4])
            la = np.log([[0.2, 0.5, 0.3], [0.4, 0.1, 0.5], [0.7, 0.1, 0.2]])
            lb = np.log([[0.6, 0.1, 0.3], [0.3, 0.6, 0.1], [0.3, 0.2, 0.5]])
            path, best = viterbi(lp, la, lb, [0, 0, 2, 1, 2])
            assert [int(z) for z in path] == [1, 2, 0, 1, 2], f"路径应为 [1, 2, 0, 1, 2]，得到 {path}"
            assert abs(best + 7.945154883482258) < 1e-6, f"log 概率应为 -7.9452，得到 {best}"
          `,
        },
        {
          name: 'T = 1',
          code: py`
            lp = np.log([0.3, 0.7])
            lb = np.log([[0.9, 0.1], [0.2, 0.8]])
            path, best = viterbi(lp, np.log([[0.5, 0.5], [0.5, 0.5]]), lb, [0])
            assert [int(z) for z in path] == [0], f"0.3*0.9 > 0.7*0.2，应返回 [0]，得到 {path}"
          `,
        },
        {
          name: '随机对拍（暴力枚举）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(7)
            for _ in range(40):
                N, M, T = int(rng.integers(2, 5)), int(rng.integers(2, 4)), int(rng.integers(1, 6))
                lp, la, lb = _rand_hmm(rng, N, M)
                obs = [int(x) for x in rng.integers(0, M, T)]
                ep, eb = _ref_brute(lp, la, lb, obs)
                _check(lp, la, lb, obs, eb)
          `,
        },
        {
          name: '含零概率转移（-inf）',
          hidden: true,
          code: py`
            rng = np.random.default_rng(11)
            for _ in range(30):
                N, M, T = 4, 3, int(rng.integers(2, 6))
                lp, la, lb = _rand_hmm(rng, N, M, zero_frac=0.5)
                obs = [int(x) for x in rng.integers(0, M, T)]
                ep, eb = _ref_brute(lp, la, lb, obs)
                if not np.isfinite(eb):
                    continue
                _check(lp, la, lb, obs, eb)
          `,
        },
        {
          name: '规模：N = 64，T = 400',
          hidden: true,
          code: py`
            rng = np.random.default_rng(64)
            lp, la, lb = _rand_hmm(rng, 64, 20)
            obs = [int(x) for x in rng.integers(0, 20, 400)]
            _, eb = _ref_fast(lp, la, lb, obs)
            t0 = time.perf_counter()
            _check(lp, la, lb, obs, eb)
            dt = time.perf_counter() - t0
            assert dt < 4, f"用时 {dt:.2f}s，请对状态维度向量化（广播 + max/argmax）"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-kbest',
          q: md`如果要输出前 $k$ 条最优路径（k-best Viterbi）怎么做？`,
          a: md`每个格子保留前 $k$ 个（分数, 前驱, 前驱的第几名）而不是 1 个，合并时从 $N \cdot k$ 个候选中取前 $k$（堆），复杂度 $O(TN^2 k \log k)$；回溯时按名次追踪。这也是语音识别输出 N-best 列表的基础。`,
        },
        {
          id: 'f-crf',
          q: md`BiLSTM-CRF / BERT-CRF 做 NER 时，解码和这里有什么不同？`,
          a: md`结构完全相同，只是把 $\log A$ 换成学到的转移分数矩阵，把 $\log B_j(x_t)$ 换成网络输出的发射分数（未归一化的 logits），在线性链上做同样的 max-plus 递推。训练时需要 Forward 算法（logsumexp）计算配分函数。`,
        },
      ],
    },
    {
      id: 'c-unigram',
      title: 'Unigram 最大概率分词',
      difficulty: 'medium',
      source: '大模型岗手撕 · 分词',
      timeLimitMin: 20,
      description: md`
        给定文本 \`text\` 与词表 \`vocab\`（piece → **log 概率**，均为负数），返回使各 piece log 概率之和最大的切分（piece 列表，按顺序拼接等于 \`text\`）。

        \`\`\`python
        def unigram_segment(text: str, vocab: dict[str, float]) -> list[str]: ...
        \`\`\`

        - 保证 \`text\` 的每个**单字符**都在 \`vocab\` 中（因此切分总是存在）；\`text\` 可以为空（返回 \`[]\`）
        - 测试中的概率为随机浮点数，不会出现并列最优
        - 设最长 piece 长度为 $L$，要求时间 $O(nL)$（隐藏测试 $n = 3000$）

        **示例**
        \`\`\`python
        vocab = {"研究": log(0.03), "研究生": log(0.01), "生命": log(0.02), "起源": log(0.015),
                 "研": log(0.002), "究": log(0.001), "生": log(0.01), "命": log(0.004), "起": log(0.005), "源": log(0.003)}
        unigram_segment("研究生命起源", vocab)   # ["研究", "生命", "起源"]
        \`\`\`
      `,
      starter: py`
        def unigram_segment(text: str, vocab: dict[str, float]) -> list[str]:
            # TODO: best[i] = 前 i 个字符的最大 log 概率；只枚举长度 ≤ L 的 piece；回溯
            raise NotImplementedError
      `,
      solution: py`
        def unigram_segment(text: str, vocab: dict[str, float]) -> list[str]:
            n = len(text)
            L = max((len(p) for p in vocab), default=1)     # 最长 piece
            NEG = float('-inf')
            best = [0.0] + [NEG] * n                        # best[i]: text[:i] 的最优 log 概率
            prev = [-1] * (n + 1)                           # prev[i]: 最后一个 piece 的起点
            for i in range(1, n + 1):
                for j in range(max(0, i - L), i):
                    if best[j] == NEG:
                        continue
                    lp = vocab.get(text[j:i])
                    if lp is not None and best[j] + lp > best[i]:
                        best[i] = best[j] + lp
                        prev[i] = j
            out = []
            i = n
            while i > 0:                                    # 沿 prev 回溯
                out.append(text[prev[i]:i])
                i = prev[i]
            return out[::-1]
      `,
      explanation: md`
        **建模**：位置 $0..n$ 是节点，词表中每个出现的 piece \`text[j:i]\` 是一条 $j \to i$ 的边，权重为 $\log p$。最大概率切分 = 从 0 到 $n$ 的最长路，DAG 上按位置递增做 DP：
        $$best[i] = \max_{i - L \le j < i,\ text[j:i] \in V} best[j] + \log p(text[j:i])$$

        **复杂度**：每个 $i$ 只看 $L$ 个起点，切片与哈希查询 $O(L)$，总 $O(nL^2)$（严格来说切片也要 $O(L)$），通常记作 $O(nL)$。用 Trie 从每个 $j$ 向右走可以去掉切片开销。

        **常见错误**：枚举所有 $j < i$ 导致 $O(n^2)$；\`best\` 初始化为 0（log 概率都是负数，0 会被误认为“更好”）；log 概率相乘而不是相加；忘了回溯后反转。
      `,
      complexity: { time: 'O(n·L)', space: 'O(n)' },
      hints: [
        md`把它看成 DAG 最长路：节点是字符间的位置 $0..n$，词表里的 piece 是边。`,
        md`\`best = [0] + [-inf] * n\`，\`best[i] = max(best[j] + vocab[text[j:i]])\`，$j$ 只需从 $i - L$ 枚举到 $i - 1$。`,
        md`同时记录 \`prev[i] = j\`，最后从 $n$ 沿 \`prev\` 回溯到 0，再反转。`,
      ],
      setup: py`
        import math, random

        def _ref_brute(text, vocab):
            best = (-math.inf, None)
            def go(i, acc, parts):
                nonlocal best
                if i == len(text):
                    if acc > best[0]:
                        best = (acc, list(parts))
                    return
                for j in range(i + 1, len(text) + 1):
                    w = text[i:j]
                    if w in vocab:
                        parts.append(w)
                        go(j, acc + vocab[w], parts)
                        parts.pop()
            go(0, 0.0, [])
            return best[1]

        def _ref_dp(text, vocab):
            L = max(map(len, vocab))
            best = [0.0] + [-math.inf] * len(text)
            prev = [0] * (len(text) + 1)
            for i in range(1, len(text) + 1):
                for j in range(max(0, i - L), i):
                    w = text[j:i]
                    if w in vocab and best[j] + vocab[w] > best[i]:
                        best[i], prev[i] = best[j] + vocab[w], j
            out, i = [], len(text)
            while i > 0:
                out.append(text[prev[i]:i])
                i = prev[i]
            return out[::-1]

        def _rand_vocab(rnd, alphabet, n_multi, max_len):
            v = {c: math.log(rnd.uniform(0.001, 0.05)) for c in alphabet}
            while len(v) < len(alphabet) + n_multi:
                w = ''.join(rnd.choice(alphabet) for _ in range(rnd.randint(2, max_len)))
                v[w] = math.log(rnd.uniform(0.001, 0.08))
            return v
      `,
      tests: [
        {
          name: '示例：研究生命起源',
          code: py`
            v = {"研究": math.log(0.03), "研究生": math.log(0.01), "生命": math.log(0.02), "起源": math.log(0.015),
                 "研": math.log(0.002), "究": math.log(0.001), "生": math.log(0.01), "命": math.log(0.004),
                 "起": math.log(0.005), "源": math.log(0.003)}
            got = unigram_segment("研究生命起源", v)
            assert got == ["研究", "生命", "起源"], f"期望 ['研究', '生命', '起源']，得到 {got}"
          `,
        },
        {
          name: '空串与单字',
          code: py`
            assert unigram_segment("", {"a": -1.0}) == []
            assert unigram_segment("a", {"a": -1.0}) == ["a"]
            v = {"a": -1.0, "b": -1.0, "ab": -3.0}
            got = unigram_segment("ab", v)
            assert got == ["a", "b"], f"-1 + -1 > -3，应拆开，得到 {got}"
          `,
        },
        {
          name: '随机对拍（暴力枚举）',
          hidden: true,
          code: py`
            rnd = random.Random(2018)
            for _ in range(150):
                v = _rand_vocab(rnd, 'abc', rnd.randint(1, 8), 4)
                text = ''.join(rnd.choice('abc') for _ in range(rnd.randint(1, 10)))
                got, exp = unigram_segment(text, v), _ref_brute(text, v)
                assert ''.join(got) == text, f"切分拼接后应等于原文 {text!r}，得到 {got}"
                assert got == exp, f"text={text!r}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：n = 3000，L = 6',
          hidden: true,
          code: py`
            rnd = random.Random(3000)
            v = _rand_vocab(rnd, 'abcdefgh', 400, 6)
            text = ''.join(rnd.choice('abcdefgh') for _ in range(3000))
            got = unigram_segment(text, v)
            assert ''.join(got) == text, "切分拼接后不等于原文"
            exp = _ref_dp(text, v)
            gs, es = sum(v[w] for w in got), sum(v[w] for w in exp)
            assert abs(gs - es) < 1e-9, f"总 log 概率期望 {es:.4f}，得到 {gs:.4f}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-sample',
          q: md`SentencePiece 的 subword regularization 如何“按概率采样”一种切分，而不是取最优？`,
          a: md`把 max 换成 sum 做 Forward：$\alpha[i] = \operatorname{logsumexp}_j(\alpha[j] + \log p(s[j:i]))$，得到所有切分的总概率；然后**从后往前采样**：在位置 $i$ 以正比于 $\exp(\alpha[j] + \log p(s[j:i]))$ 的概率选择上一个切点 $j$（forward-filtering backward-sampling）。温度参数 $\alpha$ 可以先作用在 $\log p$ 上控制随机程度。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-viterbi',
      freq: 2,
      q: md`手写 Viterbi 的关键点有哪些？复杂度多少？`,
      a: md`
        **结论**：log 域、$\delta$ 递推 + 回溯指针、向量化，$O(TN^2)$ 时间、$O(TN)$ 空间。

        **要点**：
        1. $\delta_t(j) = \max_i[\delta_{t-1}(i) + \log A_{ij}] + \log B_j(x_t)$；
        2. NumPy 一步：\`scores = delta[:, None] + log_A\`，\`max/argmax(axis=0)\`——轴不能反；
        3. 回溯从 \`delta.argmax()\` 开始倒推 $T-1$ 步再反转；
        4. 零概率用 \`-inf\`，避免 \`-inf - (-inf)\` 产生 NaN。

        **延伸**：换成 logsumexp 就是 Forward 算法；CRF 解码结构相同，只是分数来自网络。
      `,
    },
    {
      id: 'qa-beam',
      freq: 3,
      q: md`Viterbi 和 beam search 是什么关系？为什么 LLM 解码不用 Viterbi？`,
      a: md`
        **结论**：beam search 是“只保留前 B 条路径、不合并”的剪枝近似；Viterbi 是“按状态合并”的精确 DP。

        **原因**：Viterbi 的合并依赖马尔可夫性——未来只取决于当前状态，以同一状态结尾的路径只需保留最优的一条，状态数 $N$ 有限。LLM 的下一个 token 分布依赖完整前缀，“状态”就是整个前缀，没有可以合并的子问题，精确搜索是 $V^T$，只能剪枝。

        **延伸**：$B=1$ 是贪心；即使 $B \ge N$，不合并的 beam 在 HMM 上也可能错过最优解。开放式生成中 beam 容易产生重复、偏短的文本，所以更常用采样；翻译、ASR 等任务仍常用 beam（加长度惩罚）。
      `,
    },
    {
      id: 'qa-tokenizer',
      freq: 2,
      q: md`BPE、WordPiece、Unigram 三种分词在编码时有什么区别？`,
      a: md`
        **结论**：BPE 按训练时学到的合并规则顺序合并；WordPiece（BERT）从左到右贪心最长匹配；Unigram 用 Viterbi 求全局最大概率切分。

        **细节**：
        - BPE（GPT-2、Llama、Qwen 等，多为字节级）：确定性，结果由合并顺序决定；
        - WordPiece：贪心不回头，可能得到次优切分，找不到匹配时输出 \`[UNK]\`；
        - Unigram（T5、BigBird、Pegasus 等，经 SentencePiece）：每个 piece 有概率，编码是位置格上的 Viterbi，$O(nL)$；训练用 EM + 剪枝词表，并支持采样切分做正则化。

        **延伸**：SentencePiece 是一个库，同时支持 BPE 与 Unigram，并把空格编码为 \`▁\`，因此可以直接处理不分词的语言。
      `,
    },
    {
      id: 'qa-semiring',
      freq: 1,
      q: md`为什么说 Viterbi、Forward、最短路是“同一个算法”？`,
      a: md`
        **结论**：它们都是 DAG 上的同一个递推，只是换了半环。

        **解释**：路径的“权重”沿路径用 $\otimes$ 组合，多条路径用 $\oplus$ 合并。Forward 用 $(+, \times)$（log 域 $(\operatorname{logsumexp}, +)$）；Viterbi 用 $(\max, +)$；最短路用 $(\min, +)$；计数用 $(+, \times)$ 且权重为 1。只要满足分配律，DP 的正确性就成立。

        **延伸**：WFST（加权有限状态转换器）工具包就是按半环抽象实现的，同一套代码既能算概率又能求最优路径。
      `,
    },
  ],

  cheatsheet: md`
    \`\`\`python
    delta = log_pi + log_B[:, obs[0]]
    for o in obs[1:]:
        sc = delta[:, None] + log_A          # (N, N)：行 i 前驱，列 j 当前
        back.append(sc.argmax(axis=0))
        delta = sc.max(axis=0) + log_B[:, o]
    path = [delta.argmax()]; 逆序 path.append(bp[path[-1]])；反转
    # Unigram：best[i] = max_{i-L<=j<i} best[j] + logp[s[j:i]]
    \`\`\`

    | | 合并 | 结果 |
    |---|---|---|
    | Viterbi | max | 最优路径（精确） |
    | Forward | logsumexp | $P(x)$ |
    | Beam | 前 B 条，不合并 | 近似，B=1 即贪心 |

    - $O(TN^2)$；log 域防下溢；\`-inf\` 表示零概率
    - LLM 不能 Viterbi：状态 = 整个前缀，不可合并
    - Unigram（T5 等）编码 = 位置格 Viterbi；WordPiece = 贪心最长匹配
  `,
})
