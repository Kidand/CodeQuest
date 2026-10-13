/**
 * algo-dp-twoseq：LCS 与编辑距离（双序列 DP）+ 大模型场景 WER/CER
 */
import { defineLevel, md, py } from '@/content/helpers'
import { LcsViz, EditDistanceViz, WerViz } from '../viz/TwoSeqViz'

export default defineLevel({
  learn: [
    {
      id: 'lcs',
      title: 'LCS：双序列 DP 的标准形态',
      body: md`
        **最长公共子序列**（LC 1143）：两个串的公共子序列（可以不连续）的最大长度。

        双序列问题的标准状态：**\`dp[i][j]\` = \`a\` 的前 $i$ 个字符与 \`b\` 的前 $j$ 个字符的答案**。看两个前缀的**末尾字符**：

        $$
        dp[i][j] = \begin{cases} dp[i-1][j-1] + 1 & a_{i-1} = b_{j-1} \\ \max(dp[i-1][j],\ dp[i][j-1]) & a_{i-1} \ne b_{j-1} \end{cases}
        $$

        - **相等**：两个末尾配对一定不亏（交换论证：若最优解没用这对，把它最后一个匹配换成这对，长度不变），于是 $+1$ 后缩到 $(i-1, j-1)$。
        - **不等**：两个末尾不可能互相配对，所以**至少有一个**不在 LCS 里——要么丢 $a$ 的末尾（上），要么丢 $b$ 的末尾（左），取较大者。
        - **初始化**：第 0 行、第 0 列表示空串，全为 0。表的大小是 $(m+1) \times (n+1)$，字符下标统一用 \`i-1\`、\`j-1\`。
        - **遍历顺序**：依赖左上、上、左 → 行优先。时间、空间都是 $O(mn)$。
      `,
      viz: LcsViz,
      after: md`
        回溯得到的路径就是一种**对齐**方式：对角线上的配对格就是 LCS 中的字符。很多题是 LCS 的直接推论：

        | 题 | 公式 |
        |---|---|
        | LC 583 两个字符串的删除操作（只能删） | $m + n - 2\cdot LCS$ |
        | LC 1092 最短公共超序列长度 | $m + n - LCS$ |
        | LC 516 最长回文子序列 | $LCS(s,\ s[::-1])$ |

        > [!WARN] 子序列 ≠ 子串
        > **最长公共子串**要求连续：相等时 $dp[i][j] = dp[i-1][j-1] + 1$，不等时直接**清零**，答案是全表最大值。
      `,
      checkpoint: {
        id: 'cp-lcs',
        type: 'numeric',
        prompt: md`LC 583：每步可以删除任意一个字符串中的一个字符，使 \`"sea"\` 与 \`"eat"\` 相同，最少需要几步？`,
        answer: 2,
        tolerance: 0,
        explanation: md`$LCS(\text{sea}, \text{eat}) = 2$（\`"ea"\`），两串都只保留 LCS：删除步数 $= 3 + 3 - 2 \times 2 = 2$（删掉 \`s\` 和 \`t\`）。`,
      },
    },
    {
      id: 'edit',
      title: '编辑距离：三个方向 = 三种操作',
      body: md`
        **编辑距离**（LC 72，Levenshtein 距离）：把 \`word1\` 变成 \`word2\` 的最少操作次数，操作为插入、删除、替换一个字符。

        **状态**：\`dp[i][j]\` = 把 \`word1[:i]\` 变成 \`word2[:j]\` 的最少操作数。按**最后一步操作**分类：

        $$
        dp[i][j] = \begin{cases} dp[i-1][j-1] & w_1[i-1] = w_2[j-1] \\ 1 + \min\big(\underbrace{dp[i-1][j-1]}_{\text{替换}},\ \underbrace{dp[i-1][j]}_{\text{删除 } w_1[i-1]},\ \underbrace{dp[i][j-1]}_{\text{插入 } w_2[j-1]}\big) & \text{否则} \end{cases}
        $$

        - **删除**：先删掉 \`word1\` 的末尾，剩下 \`word1[:i-1]\` → \`word2[:j]\`，即 $dp[i-1][j] + 1$；
        - **插入**：先把 \`word1[:i]\` 变成 \`word2[:j-1]\`，再在末尾插入 \`word2[j-1]\`，即 $dp[i][j-1] + 1$；
        - **初始化**：$dp[i][0] = i$（删光），$dp[0][j] = j$（全插入）——这一步最容易漏。

        在表上看：**对角线 = 保留 / 替换，向下 = 删除，向右 = 插入**。填完表后从右下角回溯，就能还原出一组最优操作序列。
      `,
      viz: EditDistanceViz,
      after: md`
        > [!TIP] 回放模式
        > 「操作回放」沿回溯路径从 $(0,0)$ 走到 $(m,n)$，每走一步就对字符串执行一次操作。替换和删除都会**消耗** \`word1\` 的一个字符，插入不消耗。
        > 最优操作序列往往不唯一（例如 \`"ab" → "ba"\` 可以替换两次，也可以删一个再插一个），回溯时的优先级决定了得到哪一条。

        > [!INTERVIEW] 常见追问
        > - 只允许插入和删除？→ $m + n - 2\cdot LCS$（LC 583）。
        > - 操作代价不同（如替换代价 2）？→ 把转移里的 $+1$ 换成对应权重即可，结构不变。
      `,
      checkpoint: {
        id: 'cp-edit',
        type: 'numeric',
        prompt: md`\`"sunday"\` → \`"saturday"\` 的编辑距离是多少？`,
        answer: 3,
        tolerance: 0,
        explanation: md`插入 \`a\`、插入 \`t\`、把 \`n\` 替换为 \`r\`：s(a)(t)u**n→r**day，共 3 次。长度差为 2，至少要 2 次插入；而 \`sunday\` 与 \`saturday\` 的 LCS 是 \`"suday"\`（5），只靠插入删除需要 $6 + 8 - 10 = 4$ 次，用一次替换代替“删 + 插”节省 1 次。`,
      },
    },
    {
      id: 'space',
      title: '初始化细节与空间优化',
      body: md`
        双序列 DP 的三个高频细节：

        1. **多开一行一列**：\`dp\` 的形状是 $(m+1)\times(n+1)$，第 0 行 / 列表示空前缀，字符下标写 \`a[i-1]\`。这样边界统一，不用对 $i=0$ 特判。
        2. **Python 二维数组**：必须写 \`[[0] * (n + 1) for _ in range(m + 1)]\`。写成 \`[[0] * (n + 1)] * (m + 1)\` 会让每一行都是**同一个列表对象**，改一行全变。
        3. **滚动到一维**：第 $i$ 行只依赖第 $i-1$ 行，可以只留一行 \`dp[j]\`。但转移要用到**左上角** $dp[i-1][j-1]$，而它在算 \`dp[j-1]\` 时已被覆盖——需要一个变量 \`prev\` 提前保存：

        \`\`\`python
        dp = list(range(n + 1))          # 第 0 行：dp[0][j] = j
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i       # prev = 左上角 dp[i-1][0]
            for j in range(1, n + 1):
                tmp = dp[j]              # 旧值 = 上方 dp[i-1][j]，留给下一格当左上角
                if a[i-1] == b[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = tmp
        \`\`\`

        空间 $O(\min(m, n))$（让短串做列）。代价是**无法回溯操作序列**；如果既要省空间又要路径，可以用 Hirschberg 分治算法（$O(mn)$ 时间、线性空间）。
      `,
      checkpoint: {
        id: 'cp-space',
        type: 'single',
        prompt: md`编辑距离压成一维 \`dp[j]\` 后，内层循环里（更新 \`dp[j]\` 之前）\`dp[j]\` 和 \`dp[j-1]\` 分别对应二维表中的：`,
        options: [
          '`dp[i-1][j]`（上方）和 `dp[i][j-1]`（左方）',
          '`dp[i-1][j]`（上方）和 `dp[i-1][j-1]`（左上）',
          '`dp[i][j]`（自己）和 `dp[i][j-1]`（左方）',
          '`dp[i-1][j-1]`（左上）和 `dp[i][j-1]`（左方）',
        ],
        answer: 0,
        explanation: md`$j$ 正序：\`dp[j]\` 还没被本行覆盖，是上一行的值（上方）；\`dp[j-1]\` 刚被本行更新（左方）。左上角 $dp[i-1][j-1]$ 已经丢失，所以需要 \`prev\` 变量。`,
      },
    },
    {
      id: 'wer',
      title: '大模型岗场景：WER / CER 就是词级编辑距离',
      body: md`
        语音识别（ASR）、OCR、语音大模型的核心评测指标 **WER（Word Error Rate）** 就是**以词为单位**的编辑距离：

        $$
        \text{WER} = \frac{S + D + I}{N},\qquad N = S + D + C\ (\text{参考文本的词数})
        $$

        $S, D, I, C$ 分别是对齐后的替换、删除、插入、正确数。中文没有空格分词，通常按**字**算，叫 **CER**。

        几个面试必知的细节：
        - **分母是参考文本长度**，插入次数不受限制，所以 WER **可以超过 100%**（识别结果“幻觉”出大量多余词时）。
        - **语料级 WER = 总错误数 / 总参考词数**，不是逐句 WER 的平均。常用库 \`jiwer\` 就是先把各句的 $S, D, I, C$ 累加再相除；对句平均会让短句的权重被放大。
        - 编辑距离总数是唯一的，但 $S/D/I$ 的**拆分**取决于回溯时的优先级（一次替换 vs 一删一插），不同工具可能略有差异。
        - 计算前要做**文本归一化**（大小写、标点、数字写法等）。Whisper 论文专门设计了英文文本归一化器，否则 WER 会被格式差异严重高估。
      `,
      viz: WerViz,
      after: md`
        除语音外，编辑距离在大模型工作里也常见：代码补全评测里的 **Edit Similarity**、OCR / 文档解析的字符错误率、数据清洗时的近似去重（短文本）等，底层都是这张 DP 表。

        > [!INTERVIEW] 加分回答
        > “WER 的分子是 Levenshtein 距离，我会先做归一化，再在语料级累加 $S/D/I$ 后统一除以总参考词数；中文用 CER。WER 能超过 100%，因为插入错误不受参考长度限制。”
      `,
      checkpoint: {
        id: 'cp-wer',
        type: 'judge',
        prompt: md`WER 的分母是**识别结果**（hypothesis）的词数，因此 WER 不可能超过 100%。`,
        answer: false,
        explanation: md`错误。分母是**参考文本**（reference）的词数 $N$。插入错误 $I$ 没有上限，例如参考 \`"hello world"\`、识别 \`"oh hello there big world"\`：$I = 3$，WER = $3/2 = 150\%$。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-def',
      type: 'single',
      difficulty: 1,
      prompt: md`编辑距离中 \`dp[i][j]\` 的含义是：`,
      options: [
        '把 `word1[:i]` 变成 `word2[:j]` 的最少操作次数',
        '把 `word1[i:]` 变成 `word2[j:]` 的最少操作次数',
        '`word1[i]` 与 `word2[j]` 是否相等',
        '`word1[:i]` 与 `word2[:j]` 的最长公共子序列长度',
      ],
      answer: 0,
      explanation: md`标准定义是“前缀到前缀”。B 是后缀定义（也能做，但遍历顺序要倒过来）；D 是 LCS 的定义，二者相关但不同。`,
    },
    {
      id: 'q-subseq',
      type: 'judge',
      difficulty: 1,
      prompt: md`最长公共子序列要求公共部分在两个原串中都是**连续**的。`,
      answer: false,
      explanation: md`子序列只要求相对顺序一致，不要求连续；要求连续的是**最长公共子串**，它的转移在字符不等时清零：\`dp[i][j] = 0\`。`,
    },
    {
      id: 'q-lcs-num',
      type: 'numeric',
      difficulty: 1,
      prompt: md`\`text1 = "abcde"\`，\`text2 = "ace"\`，LCS 长度是多少？`,
      answer: 3,
      tolerance: 0,
      explanation: md`公共子序列 \`"ace"\`，长度 3。`,
    },
    {
      id: 'q-init',
      type: 'fill',
      difficulty: 2,
      prompt: md`编辑距离的边界初始化：\`dp[i][0] = \`____(1)____，\`dp[0][j] = \`____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['i'] },
        { label: '(2)', answers: ['j'] },
      ],
      explanation: md`\`word1[:i]\` 变成空串只能删 $i$ 次；空串变成 \`word2[:j]\` 只能插 $j$ 次。LCS 的边界是 0，编辑距离不是——混用是常见错误。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`编辑距离的转移来源与它代表的操作配对：`,
      pairs: [
        ['`dp[i-1][j-1]`，且末尾字符相同', '不操作（保留）'],
        ['`dp[i-1][j-1] + 1`，末尾字符不同', '替换 word1[i-1] 为 word2[j-1]'],
        ['`dp[i-1][j] + 1`', '删除 word1[i-1]'],
        ['`dp[i][j-1] + 1`', '插入 word2[j-1]'],
      ],
      explanation: md`“上”方向消耗了 \`word1\` 的一个字符却没产出 \`word2\` 的字符 → 删除；“左”方向产出了 \`word2\` 的字符却没消耗 \`word1\` → 插入；对角线两者各消耗一个 → 保留或替换。`,
    },
    {
      id: 'q-derive',
      type: 'multi',
      difficulty: 2,
      prompt: md`设 $L = LCS(a, b)$，$|a| = m$，$|b| = n$。下列结论正确的是：`,
      options: [
        '只允许删除时，使两串相同的最少步数是 $m + n - 2L$',
        '最短公共超序列的长度是 $m + n - L$',
        '编辑距离（插入 / 删除 / 替换）等于 $m + n - 2L$',
        '$s$ 的最长回文子序列长度 = $LCS(s, s[::-1])$',
        '最长公共子串的长度等于 $L$',
      ],
      answer: [0, 1, 3],
      explanation: md`A、B、D 都是经典推论。C 错：允许替换后，一次替换可以顶替“删 + 插”两步，所以编辑距离 $\le m+n-2L$，例如 \`"a" → "b"\` 编辑距离 1，而 $m+n-2L = 2$。E 错：子串要求连续，一般 $\le L$。`,
    },
    {
      id: 'q-wer-order',
      type: 'order',
      difficulty: 2,
      prompt: md`在一个测试集上计算语料级 WER 的正确步骤：`,
      items: [
        '对参考文本和识别结果做相同的文本归一化（大小写、标点等）',
        '按空格切成词序列（中文 CER 则按字切分）',
        '逐句做词级编辑距离对齐，得到每句的 S / D / I',
        '把所有句子的 S、D、I 与参考词数 N 分别累加',
        'WER = (ΣS + ΣD + ΣI) / ΣN',
      ],
      explanation: md`归一化必须在对齐之前，否则大小写、标点差异都会被算作错误；语料级 WER 先累加再相除，而不是对每句 WER 求平均。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出**一维滚动数组**版本的 LCS（有 2 行干扰项）：`,
      lines: [
        'def lcs(a, b):',
        '    dp = [0] * (len(b) + 1)',
        '    for ca in a:',
        '        prev = 0',
        '        for j in range(1, len(b) + 1):',
        '            tmp = dp[j]',
        '            if ca == b[j - 1]:',
        '                dp[j] = prev + 1',
        '            else:',
        '                dp[j] = max(dp[j], dp[j - 1])',
        '            prev = tmp',
        '    return dp[-1]',
      ],
      distractors: ['            prev = dp[j]', '        for j in range(len(b), 0, -1):'],
      explanation: md`\`prev\` 要保存**更新前**的 \`dp[j]\`（即上一行的 $j$，下一格的左上角），所以先 \`tmp = dp[j]\`、算完再 \`prev = tmp\`；干扰项 \`prev = dp[j]\` 拿到的是新值。$j$ 必须正序，因为“左方” \`dp[j-1]\` 需要是本行的新值。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的编辑距离实现有一处 bug（像是从 LCS 抄过来的），点出那一行：`,
      code: py`
        def min_distance(w1, w2):
            m, n = len(w1), len(w2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if w1[i - 1] == w2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]
      `,
      bugLines: [11],
      explanation: md`第 11 行应为 \`dp[i][j] = dp[i - 1][j - 1]\`：末尾字符相同**不需要**任何操作，代价不加 1。“相等时 +1”是 LCS 的写法（LCS 求的是最大匹配数）。按错误写法，\`min_distance("a", "a")\` 会返回 1。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`参考 \`"the cat sat"\`，识别结果 \`"a cat sat down"\`。按最小编辑距离对齐后，识别结果中哪些词属于错误（替换或插入）？`,
      items: ['a', 'cat', 'sat', 'down'],
      answer: [0, 3],
      explanation: md`对齐：\`the→a\`（S）、\`cat\`（C）、\`sat\`（C）、\`down\`（I）。WER = $(1 + 0 + 1)/3 \approx 66.7\%$。`,
    },
    {
      id: 'q-corpus',
      type: 'numeric',
      difficulty: 3,
      unit: '%',
      prompt: md`
        测试集有两句：
        - 句 1：参考 \`"hello"\`，识别 \`"hello world big"\`（2 次插入）
        - 句 2：参考 9 个词，识别完全正确

        语料级 WER 是多少（百分数，填数字即可）？
      `,
      answer: 20,
      tolerance: 0.01,
      explanation: md`语料级：$(0+0+2) / (1+9) = 20\%$。如果对逐句 WER 求平均会得到 $(200\% + 0\%)/2 = 100\%$——短句被严重放大，这正是要按语料级累加的原因。`,
    },
    {
      id: 'q-unique',
      type: 'judge',
      difficulty: 3,
      prompt: md`两个字符串之间编辑距离的值是唯一的，对应的最优编辑操作序列也一定唯一。`,
      answer: false,
      explanation: md`距离唯一，但操作序列不唯一。例如 \`"ab" → "ba"\`：替换两次、删 a 后在末尾插 a、先插 b 再删末尾 b，代价都是 2。这也是不同 WER 工具在 $S/D/I$ 拆分上可能不同、但总错误数一致的原因。`,
    },
  ],

  guided: [
    {
      id: 'g-lcs',
      title: '一步步写出 LCS（LC 1143）',
      intro: md`
        实现 \`longest_common_subsequence(a, b)\`，返回两个字符串的最长公共子序列长度。使用 $(m+1)\times(n+1)$ 的二维表，第 0 行 / 列表示空串。
      `,
      template: py`
        def longest_common_subsequence(a, b):
            m, n = len(a), len(b)
            dp = {{1}}
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if {{2}}:
                        dp[i][j] = {{3}}
                    else:
                        dp[i][j] = {{4}}
            return dp[m][n]
      `,
      steps: [
        {
          blank: 1,
          instruction: md`创建 $(m+1)$ 行、$(n+1)$ 列、全 0 的二维表（第 0 行 / 列即“空串”边界）。`,
          answer: '[[0] * (n + 1) for _ in range(m + 1)]',
          accept: ['[[0]*(n+1) for _ in range(m+1)]', '[[0 for _ in range(n + 1)] for _ in range(m + 1)]'],
          explain: md`千万不要写 \`[[0] * (n + 1)] * (m + 1)\`：外层乘法复制的是**同一个**列表的引用，改 \`dp[1][j]\` 时 \`dp[0][j]\` 也跟着变，隐藏测试会抓出来。`,
        },
        {
          blank: 2,
          instruction: md`判断两个前缀的末尾字符是否相同（注意表下标比字符串下标大 1）。`,
          answer: 'a[i - 1] == b[j - 1]',
          accept: ['a[i-1] == b[j-1]', 'b[j - 1] == a[i - 1]'],
        },
        {
          blank: 3,
          instruction: md`末尾相同：两者配对，长度在左上角的基础上 +1。`,
          answer: 'dp[i - 1][j - 1] + 1',
          accept: ['dp[i-1][j-1] + 1', '1 + dp[i - 1][j - 1]'],
          explain: md`交换论证保证“相同就配对”不会更差，所以不必再和上、左比较。`,
        },
        {
          blank: 4,
          instruction: md`末尾不同：至少丢掉一个末尾，取上方与左方的较大值。`,
          answer: 'max(dp[i - 1][j], dp[i][j - 1])',
          accept: ['max(dp[i-1][j], dp[i][j-1])', 'max(dp[i][j - 1], dp[i - 1][j])'],
          explain: md`不需要再考虑左上角：\`dp[i-1][j-1]\` 不会大于上方或左方（前缀更短，答案只会更小或相等）。`,
        },
      ],
      setup: py`
        import random
        from functools import lru_cache
        def _ref_lcs(a, b):
            @lru_cache(None)
            def go(i, j):
                if i == len(a) or j == len(b):
                    return 0
                if a[i] == b[j]:
                    return 1 + go(i + 1, j + 1)
                return max(go(i + 1, j), go(i, j + 1))
            return go(0, 0)
      `,
      tests: [
        { name: 'LC 示例', code: py`
          assert longest_common_subsequence("abcde", "ace") == 3
          assert longest_common_subsequence("abc", "abc") == 3
          assert longest_common_subsequence("abc", "def") == 0
        ` },
        { name: '空串', code: py`
          assert longest_common_subsequence("", "abc") == 0
          assert longest_common_subsequence("abc", "") == 0
        ` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(1143)
            for _ in range(300):
                a = ''.join(rnd.choice('abc') for _ in range(rnd.randint(0, 9)))
                b = ''.join(rnd.choice('abc') for _ in range(rnd.randint(0, 9)))
                got, exp = longest_common_subsequence(a, b), _ref_lcs(a, b)
                assert got == exp, f"a={a!r}, b={b!r}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        把“相等时 +1、不等取 max”换成“相等时不变、不等时 1 + min(三个方向)”，并把边界改成 $i$ / $j$，就得到了编辑距离——两道题是同一张表的两种读法。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-edit-script',
      title: '编辑距离：返回距离与一组最优操作序列',
      difficulty: 'hard',
      source: 'LeetCode 72 进阶',
      timeLimitMin: 25,
      description: md`
        返回把 \`word1\` 变成 \`word2\` 的最少操作数，以及**一组**达到该最小值的操作序列。

        \`\`\`python
        def edit_script(word1: str, word2: str) -> tuple[int, list[tuple[str, str, str]]]: ...
        \`\`\`

        操作序列按**从左到右**扫描 \`word1\` 的顺序给出，每个操作是一个三元组：

        | 操作 | 三元组 | 含义 |
        |---|---|---|
        | 保留 | \`('keep', c, c)\` | \`word1\` 当前字符 \`c\` 原样输出 |
        | 替换 | \`('sub', c, d)\` | 当前字符 \`c\` 换成 \`d\`（\`c != d\`） |
        | 删除 | \`('del', c, '')\` | 删掉当前字符 \`c\` |
        | 插入 | \`('ins', '', d)\` | 输出字符 \`d\`，不消耗 \`word1\` |

        测试会把你的操作序列**真正执行一遍**，检查结果等于 \`word2\`，且非 \`keep\` 操作的个数等于最小距离。最优序列不唯一，任意一组都算对。

        **示例**：\`edit_script("horse", "ros")\` → \`(3, [('sub','h','r'), ('keep','o','o'), ('del','r',''), ('keep','s','s'), ('del','e','')])\`

        **约束**：$0 \le |word1|, |word2| \le 300$。
      `,
      starter: py`
        def edit_script(word1: str, word2: str) -> tuple[int, list[tuple[str, str, str]]]:
            # 1) 填 (m+1)x(n+1) 的编辑距离表  2) 从 (m, n) 回溯出操作  3) 反转
            raise NotImplementedError
      `,
      solution: py`
        def edit_script(word1: str, word2: str) -> tuple[int, list[tuple[str, str, str]]]:
            m, n = len(word1), len(word2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if word1[i - 1] == word2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

            # 回溯：每一步都选一个「能解释 dp[i][j] 取值」的来源
            ops = []
            i, j = m, n
            while i > 0 or j > 0:
                if i > 0 and j > 0 and word1[i - 1] == word2[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
                    ops.append(('keep', word1[i - 1], word2[j - 1]))
                    i, j = i - 1, j - 1
                elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                    ops.append(('sub', word1[i - 1], word2[j - 1]))
                    i, j = i - 1, j - 1
                elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                    ops.append(('del', word1[i - 1], ''))
                    i -= 1
                else:                                   # 只剩插入一种可能
                    ops.append(('ins', '', word2[j - 1]))
                    j -= 1
            ops.reverse()
            return dp[m][n], ops
      `,
      explanation: md`
        **填表**：标准编辑距离，注意边界 $dp[i][0] = i$、$dp[0][j] = j$。

        **回溯**：从 $(m, n)$ 出发，每一步找一个满足等式的来源：
        - 末尾相同且 $dp[i][j] = dp[i-1][j-1]$ → keep，走对角；
        - $dp[i][j] = dp[i-1][j-1] + 1$ → sub，走对角；
        - $dp[i][j] = dp[i-1][j] + 1$ → del，向上；
        - 否则必然是 $dp[i][j] = dp[i][j-1] + 1$ → ins，向左。

        走到第 0 行时只能向左（插入），走到第 0 列时只能向上（删除），上面的条件写法天然覆盖了这两种边界。收集的操作是逆序的，最后反转。

        **复杂度**：时间、空间 $O(mn)$（回溯需要完整的表，不能滚动）。

        **常见错误**：回溯时直接比较三个方向的 dp 值取最小，而忘了检查“字符相同”与“替换 +1”的区别，导致把替换当成保留；边界上 $j = 0$ 时访问 \`dp[i][-1]\`（Python 不报错，悄悄读错列）。
      `,
      complexity: { time: 'O(mn)', space: 'O(mn)' },
      hints: [
        md`先写出只返回距离的标准 DP，保留完整的二维表。`,
        md`从右下角往回走，每一步问：\`dp[i][j]\` 是由哪个方向转移来的？满足等式的方向就是一个合法的“上一步”。`,
        md`对角线分两种：字符相同且值不变是 keep，值 +1 是 sub；向上是 del，向左是 ins。最后 \`ops.reverse()\`。`,
      ],
      setup: py`
        import random

        def _ref_dist(a, b):
            prev_row = list(range(len(b) + 1))
            for i in range(1, len(a) + 1):
                cur = [i] + [0] * len(b)
                for j in range(1, len(b) + 1):
                    cur[j] = prev_row[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev_row[j - 1], prev_row[j], cur[j - 1])
                prev_row = cur
            return prev_row[-1]

        def _apply(w1, ops):
            assert isinstance(ops, list), f"操作序列应为 list，得到 {type(ops).__name__}"
            i, out = 0, []
            for k, op in enumerate(ops):
                assert isinstance(op, tuple) and len(op) == 3, f"第 {k} 个操作应为三元组，得到 {op!r}"
                kind, x, y = op
                if kind == 'keep':
                    assert i < len(w1) and w1[i] == x == y, f"第 {k} 个操作 {op!r} 与 word1[{i}] 不符"
                    out.append(y); i += 1
                elif kind == 'sub':
                    assert i < len(w1) and w1[i] == x and len(y) == 1 and x != y, f"第 {k} 个操作 {op!r} 不合法"
                    out.append(y); i += 1
                elif kind == 'del':
                    assert i < len(w1) and w1[i] == x and y == '', f"第 {k} 个操作 {op!r} 不合法"
                    i += 1
                elif kind == 'ins':
                    assert x == '' and len(y) == 1, f"第 {k} 个操作 {op!r} 不合法"
                    out.append(y)
                else:
                    raise AssertionError(f"未知操作类型 {kind!r}")
            assert i == len(w1), f"操作结束时只消耗了 word1 的 {i}/{len(w1)} 个字符"
            return ''.join(out)

        def _check(w1, w2):
            res = edit_script(w1, w2)
            assert isinstance(res, tuple) and len(res) == 2, f"应返回 (距离, 操作列表)，得到 {res!r}"
            d, ops = res
            exp = _ref_dist(w1, w2)
            assert d == exp, f"{w1!r} → {w2!r}：距离应为 {exp}，得到 {d}"
            out = _apply(w1, ops)
            assert out == w2, f"{w1!r} 执行你的操作后得到 {out!r}，应为 {w2!r}"
            cost = sum(1 for op in ops if op[0] != 'keep')
            assert cost == exp, f"{w1!r} → {w2!r}：操作序列含 {cost} 次编辑，不是最优的 {exp} 次"
      `,
      tests: [
        { name: '示例 horse → ros', code: py`_check("horse", "ros")` },
        { name: 'intention → execution', code: py`_check("intention", "execution")` },
        { name: '空串', code: py`
          _check("", "abc")
          _check("abc", "")
          assert edit_script("", "") == (0, []), "两个空串应返回 (0, [])"
        ` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(72)
            for _ in range(300):
                a = ''.join(rnd.choice('abcd') for _ in range(rnd.randint(0, 8)))
                b = ''.join(rnd.choice('abcd') for _ in range(rnd.randint(0, 8)))
                _check(a, b)
          `,
        },
        {
          name: '规模：长度 300',
          hidden: true,
          code: py`
            rnd = random.Random(300)
            a = ''.join(rnd.choice('acgt') for _ in range(300))
            b = ''.join(rnd.choice('acgt') for _ in range(280))
            _check(a, b)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-hirschberg',
          q: md`如果两个串都很长（例如 $10^5$），$O(mn)$ 的表存不下，但仍要输出对齐，怎么办？`,
          a: md`Hirschberg 算法：把 \`word1\` 从中间切开，分别对前半做正向、对后半做反向的一维滚动 DP，找到 \`word2\` 上的最佳切分点，然后递归两半。时间仍是 $O(mn)$，空间降到 $O(m+n)$。若只关心距离是否 $\le k$，还可以只算对角线附近宽为 $2k+1$ 的带状区域，时间 $O(k \cdot n)$。`,
        },
      ],
    },
    {
      id: 'c-corpus-wer',
      title: '语料级 WER 计算',
      difficulty: 'medium',
      source: '大模型岗手撕 · 评测',
      timeLimitMin: 20,
      description: md`
        实现语音识别评测中的**语料级词错误率**：

        \`\`\`python
        def corpus_wer(refs: list[str], hyps: list[str]) -> float: ...
        \`\`\`

        - \`refs[k]\`、\`hyps[k]\` 是第 $k$ 句的参考文本与识别结果，用 \`str.split()\` 按空白切词，**不做**其它归一化；
        - 每句做**词级**编辑距离（插入 / 删除 / 替换代价都为 1）；
        - 返回 $\dfrac{\sum_k \text{dist}_k}{\sum_k N_k}$，其中 $N_k$ 是第 $k$ 句**参考**的词数（**不是**逐句 WER 的平均）；
        - 保证 $\sum_k N_k > 0$；某一句的参考可以为空（此时识别出的词全部算插入）。

        **示例**
        \`\`\`python
        corpus_wer(["the cat sat on the mat"], ["the cat sit on mat"])   # 2/6 ≈ 0.3333
        corpus_wer(["hello", "a b c d e f g h i"], ["hello world big", "a b c d e f g h i"])   # 2/10 = 0.2
        \`\`\`

        **约束**：句子数 $\le 200$，每句 $\le 60$ 个词。
      `,
      starter: py`
        def corpus_wer(refs: list[str], hyps: list[str]) -> float:
            # TODO: 逐句词级编辑距离，累加错误数与参考词数后再相除
            raise NotImplementedError
      `,
      solution: py`
        def corpus_wer(refs: list[str], hyps: list[str]) -> float:
            def word_edit_distance(r: list[str], h: list[str]) -> int:
                # 一维滚动的编辑距离，prev 保存左上角
                dp = list(range(len(h) + 1))
                for i in range(1, len(r) + 1):
                    prev, dp[0] = dp[0], i
                    for j in range(1, len(h) + 1):
                        tmp = dp[j]
                        if r[i - 1] == h[j - 1]:
                            dp[j] = prev
                        else:
                            dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                        prev = tmp
                return dp[-1]

            errors = total = 0
            for ref, hyp in zip(refs, hyps):
                r, h = ref.split(), hyp.split()
                errors += word_edit_distance(r, h)
                total += len(r)                     # 分母：参考文本的词数
            return errors / total
      `,
      solutionAlt: [
        {
          label: '用 jiwer 库（仅展示，浏览器中不可用）',
          code: py`
            import jiwer
            # jiwer 对列表输入时，先累加各句的 S/D/I/H 再计算，即语料级 WER
            wer = jiwer.wer(refs, hyps)
          `,
        },
      ],
      explanation: md`
        **思路**：WER 的分子就是词级 Levenshtein 距离。逐句计算距离并累加，分母累加各句参考词数，最后相除。

        **为什么不能对逐句 WER 取平均**：短句的一个错误会贡献很大的比例，平均会让短句的权重被放大（示例 2 中，平均会得到 100%，语料级是 20%）。工业界的 \`jiwer\`、Kaldi / ESPnet 的评分脚本都是语料级累加。

        **参考为空的句子**：$N_k = 0$，距离等于识别词数（全是插入），只计入分子。

        **复杂度**：每句 $O(|r|\cdot|h|)$ 时间、$O(|h|)$ 空间。

        **常见错误**：按字符而不是按词计算；除以识别结果的词数；逐句求 WER 再平均；空参考句导致除零。
      `,
      complexity: { time: 'O(Σ |r_k|·|h_k|)', space: 'O(max |h_k|)' },
      hints: [
        md`把句子 \`split()\` 成词列表，编辑距离的“字符”换成“词”，代码完全不变。`,
        md`不要逐句相除：分别累加 \`errors\` 和 \`total\`（参考词数），最后做一次除法。`,
        md`参考为空的句子：距离 = 识别词数，\`total\` 加 0，循环里不需要特判。`,
      ],
      setup: py`
        import random, math

        def _ref_ed(r, h):
            dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
            for i in range(len(r) + 1):
                dp[i][0] = i
            for j in range(len(h) + 1):
                dp[0][j] = j
            for i in range(1, len(r) + 1):
                for j in range(1, len(h) + 1):
                    dp[i][j] = dp[i - 1][j - 1] if r[i - 1] == h[j - 1] else 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
            return dp[-1][-1]

        def _ref_wer(refs, hyps):
            e = sum(_ref_ed(r.split(), h.split()) for r, h in zip(refs, hyps))
            return e / sum(len(r.split()) for r in refs)

        def _close(got, exp, ctx=""):
            assert isinstance(got, (int, float)), f"应返回 float，得到 {type(got).__name__}"
            assert math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-12), f"{ctx}期望 {exp:.6f}，得到 {got}"
      `,
      tests: [
        { name: '示例 1：单句', code: py`_close(corpus_wer(["the cat sat on the mat"], ["the cat sit on mat"]), 2 / 6)` },
        {
          name: '示例 2：语料级而非句平均',
          code: py`
            got = corpus_wer(["hello", "a b c d e f g h i"], ["hello world big", "a b c d e f g h i"])
            assert not math.isclose(got, 1.0), "得到 1.0：你是不是对逐句 WER 求了平均？"
            _close(got, 0.2)
          `,
        },
        {
          name: 'WER 可以超过 1、识别为空',
          code: py`
            _close(corpus_wer(["hello world"], ["oh hello there big world"]), 1.5, "全插入：")
            _close(corpus_wer(["a b c"], [""]), 1.0, "识别为空（全删除）：")
          `,
        },
        {
          name: '按词而不是按字符',
          hidden: true,
          code: py`
            _close(corpus_wer(["speech recognition"], ["speech recognitions"]), 0.5, "一个词拼错只算 1 个替换：")
          `,
        },
        {
          name: '参考为空的句子',
          hidden: true,
          code: py`
            _close(corpus_wer(["", "x y"], ["a b", "x y"]), 1.0, "空参考句的 2 个插入只计入分子：")
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(2024)
            vocab = ["the", "a", "cat", "dog", "sat", "on", "mat", "big"]
            for _ in range(60):
                k = rnd.randint(1, 6)
                refs = [" ".join(rnd.choice(vocab) for _ in range(rnd.randint(1, 10))) for _ in range(k)]
                hyps = [" ".join(rnd.choice(vocab) for _ in range(rnd.randint(0, 10))) for _ in range(k)]
                _close(corpus_wer(refs, hyps), _ref_wer(refs, hyps), f"refs={refs}, hyps={hyps}：")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-sdi',
          q: md`如果还要分别报告 $S$、$D$、$I$ 的数量，怎么做？不同工具的结果为什么可能不同？`,
          a: md`保留二维表并回溯，按路径方向统计（对角且值 +1 为 S，向上为 D，向左为 I）。总错误数是唯一的，但当“一次替换”与“一删一插”等多种路径代价相同时，回溯优先级不同会得到不同的 $S/D/I$ 拆分，所以不同工具的细分数字可能略有差异，WER 总值一致。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-edit',
      freq: 3,
      q: md`讲一下编辑距离的状态定义和转移，三个方向分别代表什么？`,
      a: md`
        **结论**：\`dp[i][j]\` = \`word1[:i]\` 变为 \`word2[:j]\` 的最少操作数；末尾相同时等于左上角，否则 $1 + \min(\text{左上}, \text{上}, \text{左})$。

        **原因**：按最后一步操作分类——替换（两串都消耗一个字符，左上角）、删除 \`word1\` 末尾（上方）、插入 \`word2\` 末尾（左方）；末尾相同时不操作。边界是 $dp[i][0] = i$、$dp[0][j] = j$。

        **延伸**：空间可滚动到 $O(\min(m,n))$，需要额外变量保存左上角；要输出操作序列则保留整表回溯，或用 Hirschberg 做到线性空间。
      `,
    },
    {
      id: 'qa-substr',
      freq: 2,
      q: md`最长公共子序列和最长公共子串的 DP 有什么区别？`,
      a: md`
        **结论**：只差在字符不等时的处理。

        **原因**：子序列可以跳过字符，不等时继承 $\max(\text{上}, \text{左})$，答案在 \`dp[m][n]\`；子串必须连续，状态要定义为“**以** $a_{i-1}$、$b_{j-1}$ **结尾**的公共子串长度”，不等时清零，答案是全表最大值。

        **延伸**：最长公共子串还可以用后缀自动机 / 后缀数组做到线性或 $O(n \log n)$；二分长度 + 滚动哈希也是常见解法。
      `,
    },
    {
      id: 'qa-wer',
      freq: 2,
      q: md`ASR 模型的 WER 怎么算？有哪些坑？`,
      a: md`
        **结论**：WER = $(S + D + I) / N$，$N$ 是参考词数，分子是词级编辑距离。

        **坑**：
        1. **语料级**要先累加 $S, D, I, N$ 再相除，不能平均逐句 WER；
        2. **可以超过 100%**（大量插入，常见于模型幻觉或重复输出）；
        3. 必须先做**文本归一化**（大小写、标点、数字、缩写），Whisper 论文为此专门设计了 normalizer；
        4. 中文等无空格语言用 **CER**；
        5. $S/D/I$ 的拆分依赖回溯优先级，只有总数是确定的。

        **延伸**：大模型语音系统（ASR + LLM）也常同时看语义层面的指标，因为 WER 对同义改写过于严苛。
      `,
    },
    {
      id: 'qa-variants',
      freq: 1,
      q: md`编辑距离有哪些常见变体？各自怎么改？`,
      a: md`
        - **只允许插入 / 删除**：$m + n - 2\cdot LCS$（LC 583）；
        - **带权编辑距离**（如替换代价 2、删除代价 $c_d$）：把转移中的 $+1$ 换成对应权重，边界也乘以权重（如 $dp[i][0] = i\cdot c_d$）；
        - **ASCII 删除和**（LC 712）：代价换成被删字符的 ASCII 值；
        - **相邻交换也算一次操作**（Damerau–Levenshtein）：多一个来源 $dp[i-2][j-2] + 1$，条件是 $a_{i-1} = b_{j-2}$ 且 $a_{i-2} = b_{j-1}$；
        - **判断距离是否 $\le k$**：只算对角线附近 $2k+1$ 宽的带状区域，$O(kn)$。
      `,
    },
  ],

  cheatsheet: md`
    \`\`\`python
    # LCS：相等 +1 走对角；不等取 max(上, 左)；边界 0
    # 编辑距离：相等取左上；不等 1 + min(左上替换, 上删除, 左插入)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if a[i-1] == b[j-1] else \
                       1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    \`\`\`

    | 推论 | 公式 |
    |---|---|
    | 只删除（583） | $m + n - 2L$ |
    | 最短公共超序列 | $m + n - L$ |
    | 最长回文子序列 | $LCS(s, s^{R})$ |
    | WER / CER | $(S + D + I) / N_{ref}$，语料级累加 |

    - 二维表用列表推导，别用 \`[[0]*n]*m\`
    - 滚动一维需 \`prev\` 保存左上角，滚动后不能回溯
    - WER 可以 > 100%；分母是参考长度
  `,
})
