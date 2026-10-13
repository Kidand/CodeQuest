/**
 * algo-dp-interval：区间 DP 与回文
 */
import { defineLevel, md, py } from '@/content/helpers'
import { PalinTableViz, CenterExpandViz, BalloonViz } from '../viz/IntervalViz'

export default defineLevel({
  learn: [
    {
      id: 'table',
      title: '区间 DP：状态是一段区间，按长度从短到长',
      body: md`
        区间 DP 的状态是 **\`dp[i][j]\` = 子区间 \`s[i..j]\` 上的答案**，转移把区间缩短（去掉一端或两端），或在区间内枚举一个分割点。

        **回文子串判定**（LC 5 / 647 的 DP 解法）：
        $$
        P[i][j] = \big(s_i = s_j\big) \wedge \big(j - i < 2 \ \vee\ P[i+1][j-1]\big)
        $$
        两端相等，并且去掉两端后的内部也是回文（长度 ≤ 2 时内部为空或单字符，自动成立）。

        **最长回文子序列**（LC 516）：
        $$
        dp[i][j] = \begin{cases} dp[i+1][j-1] + 2 & s_i = s_j \\ \max(dp[i+1][j],\ dp[i][j-1]) & s_i \ne s_j \end{cases},\qquad dp[i][i] = 1
        $$
        和 LCS 的结构完全一样——它本来就等于 $LCS(s, s^R)$。

        **遍历顺序是区间 DP 的第一难点**：$dp[i][j]$ 依赖 $dp[i+1][\cdot]$（**下一行**）。所以要么**按区间长度递增**，要么 **$i$ 倒序、$j$ 正序**。写成 $i$ 正序会读到还没算的下一行。
      `,
      viz: PalinTableViz,
      after: md`
        在表上看：依赖都在当前格的**左下方**（左、下、左下），所以填表方向必须从对角线往右上推进。按长度填是“一条条斜线”，$i$ 倒序是“从最后一行往上一行行填”，两者都保证左下先算好。

        > [!TIP] 用“i 倒序”还是“按长度”？
        > 两者等价。**i 倒序**代码更短、缓存更友好；**按长度**更直观，且在“枚举分割点”类问题（戳气球、合并石子）里更不容易写错。
      `,
      checkpoint: {
        id: 'cp-table',
        type: 'numeric',
        prompt: md`\`s = "abcbaad"\` 共有多少个回文子串？（位置不同的相同子串分别计数，单个字符也算。）`,
        answer: 10,
        tolerance: 0,
        explanation: md`7 个单字符 + \`"aa"\`（下标 4–5）+ \`"bcb"\` + \`"abcba"\` = 10，即回文表中 T 的个数。`,
      },
    },
    {
      id: 'center',
      title: '中心扩展：同样 O(n²)，但只要 O(1) 空间',
      body: md`
        每个回文都有一个**中心**：奇数长度的中心是一个字符，偶数长度的中心是两个字符之间。长度为 $n$ 的串共有 $2n - 1$ 个中心。

        从每个中心出发向两侧扩展，直到两端字符不等或越界：

        \`\`\`python
        def expand(s, l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return l + 1, r - 1        # 最后一次合法的区间
        \`\`\`

        对每个中心调用 \`expand(s, i, i)\` 与 \`expand(s, i, i + 1)\`，取最长者。时间 $O(n^2)$（最坏如 \`"aaaa…"\`），空间 $O(1)$。
        扩展过程中每成功一次就是一个回文子串，所以 **LC 647 计数** 也能直接用它。
      `,
      viz: CenterExpandViz,
      after: md`
        | 方法 | 时间 | 空间 | 适用 |
        |---|---|---|---|
        | 区间 DP | $O(n^2)$ | $O(n^2)$ | 需要任意 \`s[i..j]\` 是否回文的查询（如 LC 131 回文分割） |
        | 中心扩展 | $O(n^2)$ | $O(1)$ | LC 5、647 首选 |
        | Manacher | $O(n)$ | $O(n)$ | 进阶追问：利用已知回文的对称性跳过重复扩展 |

        > [!WARN] 最常见的漏洞
        > 只从字符出发扩展，漏掉偶数长度回文（如 \`"abba"\`）；返回子串时下标差一（扩展停止时 \`l, r\` 已经越过了回文边界，要用 \`l + 1, r - 1\`）。
      `,
      checkpoint: {
        id: 'cp-center',
        type: 'single',
        prompt: md`对 \`s = "abba"\`，只用 \`expand(s, i, i)\`（字符中心）而不用两字符间的中心，返回的最长回文子串长度是：`,
        options: ['1', '2', '3', '4'],
        answer: 0,
        explanation: md`字符中心只能找到奇数长度的回文。\`"abba"\` 中没有长度 3 的回文（\`"abb"\`、\`"bba"\` 都不是），所以只能得到长度 1；真正的答案 4 需要以两个 \`b\` 之间为中心。`,
      },
    },
    {
      id: 'burst',
      title: '戳气球：逆向思维——枚举「最后一个」',
      body: md`
        **LC 312**：戳破第 $k$ 个气球得到 $p_{left} \cdot p_k \cdot p_{right}$（左右是**当前**相邻的气球），求最大总得分。

        如果枚举“**第一个**戳哪个”，戳完后左右两部分变成相邻，子问题互相影响，没法分治。
        **反过来想：区间里最后一个被戳的是 $k$。** 在它被戳之前，$(i, k)$ 和 $(k, j)$ 两边的气球已经全部戳完，而且 $k$ 一直挡在中间——两边**互不影响**；最后戳 $k$ 时，它的邻居恰好是区间外的 $p_i$ 与 $p_j$。

        两端补虚拟气球 1，定义 **\`dp[i][j]\` = 戳光开区间 $(i, j)$ 内所有气球的最大得分**：
        $$
        dp[i][j] = \max_{i < k < j}\ dp[i][k] + dp[k][j] + p_i \cdot p_k \cdot p_j
        $$
        相邻时 $dp[i][i+1] = 0$。按区间长度递增计算，答案 $dp[0][n+1]$，时间 $O(n^3)$。
      `,
      viz: BalloonViz,
      after: md`
        > [!INTERVIEW] 「枚举分割点」类区间 DP 的共同模板
        > $dp[i][j] = \min / \max_{k} \big(dp[i][k] + dp[k][j] + \text{cost}(i, k, j)\big)$，例如矩阵链乘法、合并石子、多边形三角剖分（LC 1039）、切棍子（LC 1547）。
        > 关键都是找到“使左右两边独立”的那个 $k$ 的含义——戳气球里是“最后戳”，矩阵链乘法里是“最后一次乘法的位置”。
      `,
      checkpoint: {
        id: 'cp-burst',
        type: 'numeric',
        prompt: md`\`nums = [2, 3, 4]\`，戳气球的最大得分是多少？`,
        answer: 36,
        tolerance: 0,
        explanation: md`
          $p = [1, 2, 3, 4, 1]$。最后戳 4：先在 $(0, 3)$ 内戳 2、3，最优是先戳 3（$2 \cdot 3 \cdot 4 = 24$）再戳 2（$1 \cdot 2 \cdot 4 = 8$），得 32；最后戳 4 得 $1 \cdot 4 \cdot 1 = 4$，合计 36。
          若最后戳 3：$dp[0][2] = 6$，$dp[2][4] = 12$，再加 $1\cdot3\cdot1$ 得 21；最后戳 2 更差。
        `,
      },
    },
  ],

  quiz: [
    {
      id: 'q-order',
      type: 'single',
      difficulty: 1,
      prompt: md`区间 DP 中 $dp[i][j]$ 依赖 $dp[i+1][j-1]$、$dp[i+1][j]$、$dp[i][j-1]$。下列哪种遍历顺序是**错误**的？`,
      options: ['区间长度从 1 到 n 递增', '`i` 从 n-1 到 0 倒序，`j` 从 i 到 n-1 正序', '`i` 从 0 到 n-1 正序，`j` 从 i 到 n-1 正序', '`j` 从 0 到 n-1 正序，`i` 从 j 到 0 倒序'],
      answer: 2,
      explanation: md`$i$ 正序时，计算第 $i$ 行需要的第 $i+1$ 行还没算。D 按列推进：处理第 $j$ 列时，左边的列（$j-1$）都已算完，而同一列中 $i$ 倒序保证 $dp[i+1][j]$ 先算好，所以也正确。`,
    },
    {
      id: 'q-even',
      type: 'judge',
      difficulty: 1,
      prompt: md`中心扩展法中，长度为 $n$ 的字符串一共有 $n$ 个中心需要尝试。`,
      answer: false,
      explanation: md`共 $2n - 1$ 个：$n$ 个字符中心（奇数长度回文）+ $n - 1$ 个字符间隙（偶数长度回文）。`,
    },
    {
      id: 'q-count',
      type: 'numeric',
      difficulty: 1,
      prompt: md`LC 647：\`s = "aaa"\` 有多少个回文子串？`,
      answer: 6,
      tolerance: 0,
      explanation: md`3 个 \`"a"\`、2 个 \`"aa"\`、1 个 \`"aaa"\`，共 6 个。位置不同就分别计数。`,
    },
    {
      id: 'q-lps',
      type: 'fill',
      difficulty: 2,
      prompt: md`LC 516：\`"bbbab"\` 的最长回文子序列长度为 ____(1)____；\`"cbbd"\` 的为 ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['4'], numeric: true },
        { label: '(2)', answers: ['2'], numeric: true },
      ],
      explanation: md`\`"bbbb"\` 长度 4（子序列可以跳过 \`a\`）；\`"cbbd"\` 中最长是 \`"bb"\`。注意子序列与子串的区别：\`"bbbab"\` 的最长回文**子串**是 \`"bbb"\` 或 \`"bab"\`，长度 3。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把问题与它的转移方程配对：`,
      pairs: [
        ['LC 5 / 647 回文判定', '`P[i][j] = s[i]==s[j] and (j-i<2 or P[i+1][j-1])`'],
        ['LC 516 最长回文子序列', '相等 `dp[i+1][j-1]+2`，否则 `max(dp[i+1][j], dp[i][j-1])`'],
        ['LC 312 戳气球', '`max_k dp[i][k] + dp[k][j] + p[i]*p[k]*p[j]`'],
        ['LC 1312 让字符串成为回文串的最少插入次数', '`n - LPS(s)`'],
      ],
      explanation: md`前两个“缩短区间”，戳气球“枚举分割点”。最少插入次数 = 不在最长回文子序列里的字符个数，每个都需要插入一个镜像字符来配对。`,
    },
    {
      id: 'q-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 LC 312 戳气球的 DP，下列说法正确的是：`,
      options: [
        '`dp[i][j]` 表示戳光开区间 (i, j) 内气球的最大得分，i、j 本身不戳',
        '枚举的 k 表示区间内**第一个**被戳的气球',
        '两端补 1 是为了统一边界：越界的邻居按 1 计算',
        '时间复杂度 O(n³)',
        '可以按 i 正序、j 正序直接遍历',
      ],
      answer: [0, 2, 3],
      explanation: md`B 错：是**最后一个**被戳的，只有这样左右子区间才独立。E 错：$dp[i][j]$ 依赖 $dp[k][j]$（$k > i$，下面的行），需要按长度递增或 $i$ 倒序。`,
    },
    {
      id: 'q-steps',
      type: 'order',
      difficulty: 2,
      prompt: md`把“中心扩展求最长回文子串”的步骤按顺序排列：`,
      items: [
        '初始化最优区间 best = (0, 0)',
        '对每个下标 i，分别以 (i, i) 和 (i, i+1) 为起点扩展',
        '两端字符相等且未越界时 l -= 1、r += 1',
        '扩展结束时合法区间为 (l + 1, r - 1)，与 best 比较',
        '返回 s[best_l : best_r + 1]',
      ],
      explanation: md`注意第 4 步：循环退出时 \`l, r\` 已经指向不匹配（或越界）的位置，真正的回文是 \`s[l+1 .. r-1]\`。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`用中心扩展统计回文子串个数（LC 647），有 2 行干扰项：`,
      lines: [
        'def count_substrings(s):',
        '    n, cnt = len(s), 0',
        '    for c in range(2 * n - 1):',
        '        l, r = c // 2, c // 2 + c % 2',
        '        while l >= 0 and r < n and s[l] == s[r]:',
        '            cnt += 1',
        '            l, r = l - 1, r + 1',
        '    return cnt',
      ],
      distractors: ['    for c in range(n):', '        while l > 0 and r < n and s[l] == s[r]:'],
      explanation: md`用一个变量 \`c\` 编码 $2n-1$ 个中心：偶数 \`c\` 是字符中心 \`(c/2, c/2)\`，奇数是间隙 \`(c//2, c//2 + 1)\`。干扰项 \`range(n)\` 漏掉一半中心；\`l > 0\` 会漏掉以下标 0 为左端的回文。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面求最长回文子串的区间 DP 有一处 bug，点出那一行：`,
      code: py`
        def longest_palindrome(s):
            n = len(s)
            dp = [[False] * n for _ in range(n)]
            best = (0, 0)
            for i in range(n):
                for j in range(i, n):
                    if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                        dp[i][j] = True
                        if j - i > best[1] - best[0]:
                            best = (i, j)
            return s[best[0]:best[1] + 1]
      `,
      bugLines: [5],
      explanation: md`第 5 行 \`i\` 正序：算 \`dp[i][j]\` 时 \`dp[i + 1][j - 1]\` 还没算（仍为 False），长度 ≥ 3 的回文全部漏掉，例如 \`"aba"\` 只会返回 \`"a"\`。应改为 \`for i in range(n - 1, -1, -1)\`。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 3,
      prompt: md`回文判定表 \`P\`（\`s = "abcbaad"\`），点出值为 True 且**长度 ≥ 2** 的全部区间。`,
      items: ['[0,4] abcba', '[1,3] bcb', '[3,5] baa', '[4,5] aa', '[0,5] abcbaa', '[2,4] cba', '[5,6] ad'],
      answer: [0, 1, 3],
      explanation: md`\`"abcba"\`、\`"bcb"\`、\`"aa"\` 是回文；\`"baa"\` 两端 b≠a；\`"abcbaa"\` 两端 a=a 但内部 \`"bcba"\` 不是回文；\`"cba"\`、\`"ad"\` 两端不等。`,
    },
    {
      id: 'q-why-last',
      type: 'single',
      difficulty: 3,
      prompt: md`戳气球为什么不能把 $k$ 定义成区间内**第一个**被戳的气球？`,
      options: [
        '戳掉 k 之后，k 左右两侧的气球变成相邻，左右两个子问题的得分互相依赖，无法独立求解',
        '因为那样时间复杂度会变成 O(n⁴)',
        '因为第一个戳的气球得分总是最小的',
        '其实可以，只是需要从右往左遍历',
      ],
      answer: 0,
      explanation: md`DP 要求子问题互相独立（最优子结构）。“先戳 k”后，左半最右边的气球的右邻居变成了右半的某个气球，两边耦合；“最后戳 k”则 k 一直把两边隔开，各自的邻居边界确定为 $p_i, p_k$ 和 $p_k, p_j$。`,
    },
  ],

  guided: [
    {
      id: 'g-lps',
      title: '最长回文子序列（LC 516）：i 倒序的区间 DP',
      intro: md`
        实现 \`longest_palindrome_subseq(s)\`。状态 \`dp[i][j]\` = \`s[i..j]\` 的最长回文子序列长度，用“$i$ 倒序、$j$ 正序”的顺序填表。
      `,
      template: py`
        def longest_palindrome_subseq(s):
            n = len(s)
            dp = [[0] * n for _ in range(n)]
            for i in {{1}}:
                dp[i][i] = {{2}}
                for j in range(i + 1, n):
                    if s[i] == s[j]:
                        dp[i][j] = {{3}}
                    else:
                        dp[i][j] = {{4}}
            return dp[0][n - 1]
      `,
      steps: [
        {
          blank: 1,
          instruction: md`$dp[i][\cdot]$ 依赖下一行 $dp[i+1][\cdot]$，所以 $i$ 要从大到小。`,
          answer: 'range(n - 1, -1, -1)',
          accept: ['range(n-1, -1, -1)', 'reversed(range(n))'],
          explain: md`$i$ 正序会读到还没算的下一行（全 0），长度 ≥ 3 的结果都会偏小。`,
        },
        {
          blank: 2,
          instruction: md`单个字符本身就是长度为 1 的回文子序列。`,
          answer: '1',
        },
        {
          blank: 3,
          instruction: md`两端相等：两端都放进回文，加上内部的最长回文子序列。`,
          answer: 'dp[i + 1][j - 1] + 2',
          accept: ['dp[i+1][j-1] + 2', '2 + dp[i + 1][j - 1]'],
          explain: md`当 $j = i + 1$ 时 \`dp[i+1][j-1] = dp[i+1][i]\` 是左下三角，值为 0，结果为 2，恰好正确——这就是不必特判长度 2 的原因。`,
        },
        {
          blank: 4,
          instruction: md`两端不等：至少有一端不在回文里，去掉左端或右端取较大者。`,
          answer: 'max(dp[i + 1][j], dp[i][j - 1])',
          accept: ['max(dp[i+1][j], dp[i][j-1])', 'max(dp[i][j - 1], dp[i + 1][j])'],
        },
      ],
      setup: py`
        import random
        def _ref_lps(s):
            t = s[::-1]
            n = len(s)
            dp = [[0] * (n + 1) for _ in range(n + 1)]
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    dp[i][j] = dp[i-1][j-1] + 1 if s[i-1] == t[j-1] else max(dp[i-1][j], dp[i][j-1])
            return dp[n][n]
      `,
      tests: [
        { name: 'LC 示例', code: py`
          assert longest_palindrome_subseq("bbbab") == 4
          assert longest_palindrome_subseq("cbbd") == 2
        ` },
        { name: '单字符与全同', code: py`
          assert longest_palindrome_subseq("a") == 1
          assert longest_palindrome_subseq("aaaa") == 4
        ` },
        {
          name: '随机对拍（LCS(s, s[::-1])）',
          hidden: true,
          code: py`
            rnd = random.Random(516)
            for _ in range(300):
                s = ''.join(rnd.choice('abc') for _ in range(rnd.randint(1, 12)))
                got, exp = longest_palindrome_subseq(s), _ref_lps(s)
                assert got == exp, f"s={s!r}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        同样的表结构还能直接回答“最少插入几个字符使其成为回文”（LC 1312）：答案是 \`n - dp[0][n-1]\`。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-longest-pal',
      title: '最长回文子串',
      difficulty: 'medium',
      source: 'LeetCode 5',
      timeLimitMin: 20,
      description: md`
        给定字符串 \`s\`，返回其中**最长的回文子串**（连续）。若有多个长度相同的答案，返回任意一个。

        \`\`\`python
        def longest_palindrome(s: str) -> str: ...
        \`\`\`

        **示例**
        - \`"babad"\` → \`"bab"\`（\`"aba"\` 也正确）
        - \`"cbbd"\` → \`"bb"\`
        - \`"a"\` → \`"a"\`

        **约束**：$1 \le |s| \le 1000$，只含字母与数字。要求 $O(n^2)$ 时间（隐藏测试 $n = 1000$；$O(n^3)$ 的暴力枚举会超时）。
      `,
      starter: py`
        def longest_palindrome(s: str) -> str:
            # TODO: 中心扩展（2n-1 个中心）或区间 DP
            return s[:1]
      `,
      solution: py`
        def longest_palindrome(s: str) -> str:
            n = len(s)
            best_l, best_r = 0, 0

            def expand(l: int, r: int) -> tuple[int, int]:
                while l >= 0 and r < n and s[l] == s[r]:
                    l -= 1
                    r += 1
                return l + 1, r - 1             # 退出时已越过边界，回退一步

            for i in range(n):
                for l, r in (expand(i, i), expand(i, i + 1)):   # 奇数中心、偶数中心
                    if r - l > best_r - best_l:
                        best_l, best_r = l, r
            return s[best_l:best_r + 1]
      `,
      explanation: md`
        **中心扩展**：枚举 $2n-1$ 个中心（字符与字符间隙），向两边扩展到不能再扩为止，记录最长区间。时间 $O(n^2)$，空间 $O(1)$。

        **区间 DP 写法**：$P[i][j] = (s_i = s_j) \wedge (j - i < 2 \vee P[i+1][j-1])$，$i$ 倒序、$j$ 正序，同时记录最长的 True 区间。时间、空间都是 $O(n^2)$；在 Python 中二维表常数较大，$n = 1000$ 时明显慢于中心扩展。

        **Manacher**：利用已知回文的对称性复用半径，$O(n)$，面试中能讲清思路即可。

        **常见错误**：漏掉偶数中心；返回区间差一；DP 写成 $i$ 正序。
      `,
      complexity: { time: 'O(n²)', space: 'O(1)' },
      hints: [
        md`每个回文都有中心。中心可能是一个字符，也可能是两个字符之间——一共 $2n - 1$ 个。`,
        md`写一个 \`expand(l, r)\`：两端相等就向外扩，返回最后一次合法的 \`(l + 1, r - 1)\`。`,
        md`对每个 $i$ 调用 \`expand(i, i)\` 和 \`expand(i, i + 1)\`，保留最长区间。`,
      ],
      setup: py`
        import random, time
        def _ref_len(s):
            n, best = len(s), 1
            for c in range(2 * n - 1):
                l, r = c // 2, c // 2 + c % 2
                while l >= 0 and r < n and s[l] == s[r]:
                    l, r = l - 1, r + 1
                best = max(best, r - l - 1)
            return best
        def _check(s):
            got = longest_palindrome(s)
            assert isinstance(got, str), f"应返回 str，得到 {type(got).__name__}"
            assert got == got[::-1], f"s={s[:40]!r}…：返回的 {got[:40]!r} 不是回文"
            assert got in s, f"返回的 {got[:40]!r} 不是 s 的子串"
            exp = _ref_len(s)
            assert len(got) == exp, f"s={s[:40]!r}：最长回文长度应为 {exp}，得到 {len(got)}"
      `,
      tests: [
        { name: '示例', code: py`
          for s in ("babad", "cbbd", "a"):
              _check(s)
        ` },
        { name: '偶数长度与整串回文', code: py`
          assert longest_palindrome("abba") == "abba"
          assert longest_palindrome("forgeeksskeegfor") == "geeksskeeg"
          _check("ac")
        ` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(5)
            for _ in range(400):
                _check(''.join(rnd.choice('ab') for _ in range(rnd.randint(1, 16))))
          `,
        },
        {
          name: '规模：n = 1000',
          hidden: true,
          code: py`
            rnd = random.Random(1000)
            s1 = ''.join(rnd.choice('abc') for _ in range(1000))
            s2 = 'a' * 1000
            t0 = time.perf_counter()
            _check(s1)
            _check(s2)
            dt = time.perf_counter() - t0
            assert dt < 6, f"两组 n=1000 用时 {dt:.2f}s，过慢——是不是 O(n^3)？"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-manacher',
          q: md`能做到 $O(n)$ 吗？简述 Manacher 的思路。`,
          a: md`先在字符间插入分隔符把所有回文统一成奇数长度。维护当前右边界最远的回文 $(C, R)$。对位置 $i < R$，其关于 $C$ 的镜像 $j = 2C - i$ 的半径已知，$i$ 的半径至少为 $\min(r_j, R - i)$，在此基础上再暴力扩展；每次扩展都会推动 $R$ 右移，$R$ 至多移动 $n$ 次，所以总复杂度 $O(n)$。`,
        },
      ],
    },
    {
      id: 'c-burst',
      title: '戳气球',
      difficulty: 'hard',
      source: 'LeetCode 312',
      timeLimitMin: 25,
      description: md`
        \`n\` 个气球排成一排，编号 0 到 n-1，每个气球上有数字 \`nums[i]\`。戳破第 \`i\` 个气球可以获得 \`nums[left] * nums[i] * nums[right]\` 枚硬币，\`left\`、\`right\` 是**当前**与 \`i\` 相邻的气球；越界时视为数字 1。求能获得的最大硬币数。

        \`\`\`python
        def max_coins(nums: list[int]) -> int: ...
        \`\`\`

        **示例**
        - \`[3, 1, 5, 8]\` → \`167\`（戳的顺序 1, 5, 3, 8：$3\cdot1\cdot5 + 3\cdot5\cdot8 + 1\cdot3\cdot8 + 1\cdot8\cdot1 = 167$）
        - \`[1, 5]\` → \`10\`

        **约束**：$1 \le n \le 60$（隐藏测试 $n = 60$，枚举戳的顺序会超时），$0 \le nums[i] \le 100$。
      `,
      starter: py`
        def max_coins(nums: list[int]) -> int:
            # TODO: 两端补 1；dp[i][j] = 戳光开区间 (i, j) 的最大得分；枚举最后戳的 k
            raise NotImplementedError
      `,
      solution: py`
        def max_coins(nums: list[int]) -> int:
            p = [1] + nums + [1]               # 虚拟边界气球
            m = len(p)
            dp = [[0] * m for _ in range(m)]   # dp[i][j]: 戳光开区间 (i, j) 内气球的最大得分
            for length in range(2, m):         # j - i = length，从短到长
                for i in range(m - length):
                    j = i + length
                    best = 0
                    for k in range(i + 1, j):  # k 是 (i, j) 中最后一个被戳的
                        v = dp[i][k] + dp[k][j] + p[i] * p[k] * p[j]
                        if v > best:
                            best = v
                    dp[i][j] = best
            return dp[0][m - 1]
      `,
      explanation: md`
        **逆向思维**：枚举区间 $(i, j)$ 内**最后**被戳的气球 $k$。在它之前，$(i, k)$ 与 $(k, j)$ 内的气球已被戳光，且 $k$ 把两边隔开、互不影响；最后戳 $k$ 时邻居恰好是 $p_i$、$p_j$。

        $$dp[i][j] = \max_{i<k<j} dp[i][k] + dp[k][j] + p_i p_k p_j$$

        **顺序**：$dp[i][j]$ 依赖更短的区间，按长度递增（或 $i$ 倒序、$j$ 正序）。

        **复杂度**：$O(n^3)$ 时间，$O(n^2)$ 空间。

        **常见错误**：枚举“第一个戳的”导致子问题不独立；用闭区间定义却把 $p_i, p_j$ 当成邻居（闭区间写法里邻居是 $p_{i-1}$、$p_{j+1}$，需要统一）；按 $i$ 正序遍历。
      `,
      complexity: { time: 'O(n³)', space: 'O(n²)' },
      hints: [
        md`“先戳谁”会让左右两部分变得相邻、互相影响。换个角度：一个区间里**最后**被戳的是谁？`,
        md`两端补 1，定义 \`dp[i][j]\` 为戳光**开区间** $(i, j)$ 的最大得分。最后戳 $k$ 时，它的邻居是谁？`,
        md`$dp[i][j] = \max_k dp[i][k] + dp[k][j] + p_i p_k p_j$，按区间长度从 2 开始递增。`,
      ],
      setup: py`
        import random
        from functools import lru_cache
        def _ref_brute(nums):
            @lru_cache(None)
            def go(t):
                if not t:
                    return 0
                best = 0
                for i in range(len(t)):
                    l = t[i - 1] if i > 0 else 1
                    r = t[i + 1] if i + 1 < len(t) else 1
                    best = max(best, l * t[i] * r + go(t[:i] + t[i + 1:]))
                return best
            return go(tuple(nums))
        def _ref_dp(nums):
            p = [1] + nums + [1]
            m = len(p)
            dp = [[0] * m for _ in range(m)]
            for L in range(2, m):
                for i in range(m - L):
                    j = i + L
                    dp[i][j] = max(dp[i][k] + dp[k][j] + p[i] * p[k] * p[j] for k in range(i + 1, j))
            return dp[0][m - 1]
      `,
      tests: [
        { name: '示例', code: py`
          for a, exp in [([3, 1, 5, 8], 167), ([1, 5], 10)]:
              got = max_coins(a)
              assert got == exp, f"nums={a}：期望 {exp}，得到 {got}"
        ` },
        { name: '单个气球与含 0', code: py`
          for a, exp in [([7], 7), ([0], 0), ([2, 3, 4], 36)]:
              got = max_coins(a)
              assert got == exp, f"nums={a}：期望 {exp}，得到 {got}"
        ` },
        {
          name: '随机对拍（暴力枚举顺序）',
          hidden: true,
          code: py`
            rnd = random.Random(312)
            for _ in range(60):
                a = [rnd.randint(0, 9) for _ in range(rnd.randint(1, 7))]
                got, exp = max_coins(a), _ref_brute(a)
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：n = 60',
          hidden: true,
          code: py`
            rnd = random.Random(60)
            a = [rnd.randint(0, 100) for _ in range(60)]
            got, exp = max_coins(a), _ref_dp(a)
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-stones',
          q: md`合并石子（相邻两堆合并的代价是两堆之和，求合并成一堆的最小代价）怎么做？`,
          a: md`$dp[i][j] = \min_{i \le k < j} dp[i][k] + dp[k+1][j] + \text{sum}(i..j)$，用前缀和 $O(1)$ 求区间和，按长度递增，$O(n^3)$。若石子围成环，把数组复制一遍成长度 $2n$，答案取所有长度为 $n$ 的区间的最小值。（相邻合并的版本可以用四边形不等式优化到 $O(n^2)$。）`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-order',
      freq: 3,
      q: md`区间 DP 的遍历顺序怎么确定？为什么不能 i 正序？`,
      a: md`
        **结论**：按区间长度递增，或者 $i$ 倒序、$j$ 正序。

        **原因**：$dp[i][j]$ 依赖更短的区间（$dp[i+1][\cdot]$、$dp[\cdot][j-1]$、$dp[i][k]$、$dp[k][j]$），在表上都位于当前格的左下方。$i$ 正序时第 $i+1$ 行还没算，读到的是初始值。

        **延伸**：如果写成记忆化搜索就不用管顺序——区间 DP 是“记忆化更不容易写错”的典型场景，代价是递归开销和深度限制。
      `,
    },
    {
      id: 'qa-pal',
      freq: 3,
      q: md`最长回文子串有哪些解法？复杂度分别是多少？`,
      a: md`
        **结论**：暴力 $O(n^3)$；区间 DP $O(n^2)$ 时间 $O(n^2)$ 空间；中心扩展 $O(n^2)$ 时间 $O(1)$ 空间（首选）；Manacher $O(n)$。

        **要点**：中心扩展要枚举 $2n-1$ 个中心（含字符间隙）；DP 要 $i$ 倒序；Manacher 先插分隔符统一奇偶，再利用当前最右回文的对称性复用半径，右边界单调右移保证线性。

        **延伸**：回文**子序列**（516）不能中心扩展，只能 DP，且等于 $LCS(s, s^R)$。
      `,
    },
    {
      id: 'qa-burst',
      freq: 2,
      q: md`戳气球的状态和转移是怎么想出来的？`,
      a: md`
        **结论**：枚举区间内**最后一个**被戳的气球，并用开区间 + 虚拟边界 1 定义状态。

        **推导**：DP 要求子问题独立。如果枚举第一个戳的，戳完后左右两部分相邻，得分互相依赖；最后戳 $k$ 时，$k$ 在整个过程中都隔开左右两部分，两边的邻居边界分别固定为 $(p_i, p_k)$ 与 $(p_k, p_j)$，于是 $dp[i][j] = \max_k dp[i][k] + dp[k][j] + p_i p_k p_j$。

        **延伸**：这是“枚举分割点”类区间 DP 的通用思路，矩阵链乘法、多边形三角剖分、合并石子都一样。
      `,
    },
  ],

  cheatsheet: md`
    \`\`\`python
    # 中心扩展（LC 5 / 647）
    for c in range(2 * n - 1):
        l, r = c // 2, c // 2 + c % 2
        while l >= 0 and r < n and s[l] == s[r]:
            l, r = l - 1, r + 1          # 回文为 s[l+1 : r]
    # 区间 DP：i 倒序 j 正序
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n): ...
    # 戳气球：p = [1]+nums+[1]，按长度递增
    dp[i][j] = max(dp[i][k] + dp[k][j] + p[i]*p[k]*p[j] for k in range(i+1, j))
    \`\`\`

    | 题 | 转移 | 复杂度 |
    |---|---|---|
    | 5 / 647 | \`s[i]==s[j] and P[i+1][j-1]\` | $O(n^2)$ |
    | 516 | 相等 +2，否则 max(下, 左) | $O(n^2)$ |
    | 312 | 枚举最后戳的 k | $O(n^3)$ |

    - 依赖在左下方 → 不能 i 正序
    - 中心有 $2n-1$ 个；Manacher $O(n)$
    - 516 = LCS(s, reversed s)；1312 最少插入 = n − LPS
  `,
})
