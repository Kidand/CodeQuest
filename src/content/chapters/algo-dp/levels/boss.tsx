/**
 * algo-dp-boss：动态规划综合 BOSS 关
 */
import { defineLevel, md, py } from '@/content/helpers'

export default defineLevel({
  learn: [
    {
      id: 'recap',
      title: '复盘：一张表收拢本章所有 DP',
      body: md`
        | 类型 | 代表题 | 状态 | 转移要点 | 遍历顺序 |
        |---|---|---|---|---|
        | 线性 | 198 / 91 | \`dp[i]\` 前 i 个 | 选 / 不选；按最后一段分类 | i 递增 |
        | 以 i 结尾 | 300 / 53 / 152 | 以 \`nums[i]\` 结尾 | 接在前面 / 重新开始；答案取 max | i 递增 |
        | 网格 | 62 / 64 | \`dp[i][j]\` 到达 (i, j) | 上、左 | 行优先 |
        | 背包 | 416 / 322 / 518 | \`dp[j]\` 容量 j | 0/1 倒序、完全正序；组合外物品 | 见左 |
        | 双序列 | 1143 / 72 | 前缀 \`a[:i], b[:j]\` | 末尾配对 / 丢一端；三方向 = 三操作 | 行优先 |
        | 区间 | 5 / 516 / 312 | \`s[i..j]\` | 缩两端 / 枚举分割点（最后一个） | 长度递增或 i 倒序 |
        | 状态机 | 121–714 | 当天结束时的处境 | 买 / 卖 / 等待；同时赋值 | 按天 |
        | 解码 | Viterbi / Unigram | $\delta_t(j)$、\`best[i]\` | max + 回溯指针；log 域 | 按时间 / 位置 |

        **面试答题流程**（写不出来时按这个顺序自救）：
        1. 先写出**暴力递归**，明确“每一步在做什么选择”；
        2. 找出递归参数——那就是**状态**；加 \`@cache\` 得到记忆化；
        3. 由依赖方向推出**遍历顺序**，改写成递推；
        4. 看依赖跨度，做**滚动数组**；
        5. 报复杂度：状态数 × 转移代价；说清楚能否回溯出方案。

        本关的两道实战：**正则表达式匹配**（双序列 + 分类讨论，LC 10，Hard）与 **一和零**（二维费用的 0/1 背包，LC 474）。
      `,
    },
  ],

  quiz: [
    {
      id: 'b-01dir',
      type: 'single',
      difficulty: 1,
      prompt: md`一维滚动数组写 0/1 背包，容量 $j$ 的遍历方向应为：`,
      options: ['正序', '倒序', '正序倒序都可以', '取决于物品的价值'],
      answer: 1,
      explanation: md`倒序保证 \`dp[j - w]\` 还是上一轮（未考虑当前物品）的值；正序会让同一物品被重复使用，变成完全背包。`,
    },
    {
      id: 'b-tails',
      type: 'judge',
      difficulty: 1,
      prompt: md`LIS 的 tails 算法结束时，\`tails\` 数组本身就是一条最长递增子序列。`,
      answer: false,
      explanation: md`\`tails[k]\` 是“长度 k+1 的递增子序列的最小结尾”，各项可能来自不同时刻，例如 \`[3, 4, 1]\` 结束时 tails 为 \`[1, 4]\`，而 1 在 4 后面。只有长度可信。`,
    },
    {
      id: 'b-memo',
      type: 'judge',
      difficulty: 1,
      prompt: md`记忆化搜索不需要人为确定遍历顺序，递归会自动先求解依赖的子问题。`,
      answer: true,
      explanation: md`正确。这正是区间 DP、树形 DP 等“顺序难推”的题常用记忆化的原因。代价是递归开销、Python 递归深度限制，以及难以做滚动数组。`,
    },
    {
      id: 'b-edit',
      type: 'numeric',
      difficulty: 2,
      prompt: md`\`"kitten"\` → \`"sitting"\` 的编辑距离是多少？`,
      answer: 3,
      tolerance: 0,
      explanation: md`k→s 替换、e→i 替换、末尾插入 g，共 3 次。`,
    },
    {
      id: 'b-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        填写初始化 / 返回值：计数类背包（LC 518）的 \`dp[0]\` = ____(1)____；LIS 的 $O(n^2)$ 解法应返回 ____(2)____（填表达式）；
        编辑距离的 \`dp[i][0]\` = ____(3)____。
      `,
      blanks: [
        { label: '(1)', answers: ['1'], numeric: true },
        { label: '(2)', answers: ['max(dp)'] },
        { label: '(3)', answers: ['i'] },
      ],
      explanation: md`空方案计数为 1；LIS 可能在任意位置结束，取 \`max(dp)\`；把 \`word1\` 的前 $i$ 个字符变成空串需要删除 $i$ 次。`,
    },
    {
      id: 'b-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与它所属的 DP 类型配对：`,
      pairs: [
        ['LC 72 编辑距离', '双序列 DP'],
        ['LC 312 戳气球', '区间 DP（枚举最后一个）'],
        ['LC 309 冷冻期股票', '状态机 DP'],
        ['LC 518 零钱兑换 II', '完全背包计数（外物品内容量）'],
        ['HMM 解码', '网格图上的最优路径（Viterbi）'],
      ],
      explanation: md`识别类型是答题的第一步：类型决定了状态的形状（前缀、区间、容量、状态机）与遍历顺序。`,
    },
    {
      id: 'b-reverse',
      type: 'multi',
      difficulty: 2,
      prompt: md`下列写法中，哪些循环**必须倒序**（从大到小）？`,
      options: [
        '0/1 背包一维写法的容量 j',
        '区间 DP 用「i 外层、j 内层」写法时的 i',
        '完全背包一维写法的容量 j',
        'LCS 一维滚动写法中的列 j',
        'LC 416 分割等和子集的目标和 j',
      ],
      answer: [0, 1, 4],
      explanation: md`A、E 是 0/1 背包（要读旧值）；B 因为 $dp[i][\cdot]$ 依赖 $dp[i+1][\cdot]$。C 完全背包要读新值，正序；D 的左方 \`dp[j-1]\` 要是本行新值，正序，并用 \`prev\` 保存左上角。`,
    },
    {
      id: 'b-flow',
      type: 'order',
      difficulty: 2,
      prompt: md`面试中推导一道陌生 DP 题的推荐流程：`,
      items: ['写出暴力递归，明确每步的选择', '把递归参数作为状态，加缓存得到记忆化搜索', '根据依赖方向确定遍历顺序，改写成递推', '分析依赖跨度，用滚动数组压缩空间', '报告时间 / 空间复杂度，说明能否回溯方案'],
      explanation: md`从正确的暴力出发，逐步优化。直接写递推容易在状态定义或遍历顺序上出错，而记忆化版本可以先验证正确性。`,
    },
    {
      id: 'b-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LC 518 零钱兑换 II（组合数），有 2 行干扰项：`,
      lines: [
        'def change(amount, coins):',
        '    dp = [1] + [0] * amount',
        '    for c in coins:',
        '        for j in range(c, amount + 1):',
        '            dp[j] += dp[j - c]',
        '    return dp[amount]',
      ],
      distractors: ['        for j in range(amount, c - 1, -1):', '    dp = [0] * (amount + 1)'],
      explanation: md`硬币可重复使用 → 金额正序；计数问题 \`dp[0] = 1\`。外层遍历硬币保证数的是组合。倒序会变成“每种硬币最多一枚”。`,
    },
    {
      id: 'b-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 LIS（$O(n^2)$）实现在 \`[1, 3, 5, 2]\` 上返回 2，点出 bug 所在行：`,
      code: py`
        def length_of_lis(nums):
            n = len(nums)
            dp = [1] * n
            for i in range(n):
                for j in range(i):
                    if nums[j] < nums[i]:
                        dp[i] = max(dp[i], dp[j] + 1)
            return dp[-1]
      `,
      bugLines: [8],
      explanation: md`第 8 行应为 \`return max(dp)\`。\`dp[i]\` 定义为“以 \`nums[i]\` 结尾”，LIS \`[1, 3, 5]\` 以 5 结尾，\`dp[2] = 3\`，而 \`dp[-1] = dp[3] = 2\`。`,
    },
    {
      id: 'b-pick',
      type: 'pick',
      difficulty: 3,
      prompt: md`下列问题中，哪些的标准 DP 可以把**额外空间**压缩到 $O(1)$（与输入规模无关）？`,
      items: ['198 打家劫舍', '121 买卖股票', '309 冷冻期股票', '72 编辑距离', '1143 LCS', '53 最大子数组和', '312 戳气球'],
      answer: [0, 1, 2, 5],
      explanation: md`198、121、309、53 都只依赖前一（两）步的常数个状态。72、1143 依赖上一整行，最少 $O(\min(m, n))$；312 依赖任意更短区间，需要 $O(n^2)$。`,
    },
    {
      id: 'b-combo',
      type: 'numeric',
      difficulty: 3,
      prompt: md`硬币 \`[2, 3, 5]\`（每种无限枚），凑出金额 8 的**组合数**是多少？`,
      answer: 3,
      tolerance: 0,
      explanation: md`\`2+2+2+2\`、\`2+3+3\`、\`3+5\`，共 3 种。若求排列数（顺序不同算不同），\`2+3+3\` 有 3 种排列、\`3+5\` 有 2 种，合计 $1 + 3 + 2 = 6$。`,
    },
    {
      id: 'b-viterbi',
      type: 'single',
      difficulty: 3,
      prompt: md`关于 Viterbi 与 beam search，下列说法**正确**的是：`,
      options: [
        'beam 宽度等于状态数 N 时，不合并路径的 beam search 在 HMM 上一定得到最优解',
        'Viterbi 精确的前提是马尔可夫性：结尾状态相同的路径可以合并，只保留最优',
        'Viterbi 求出的最优路径概率等于观测序列的总概率 P(x)',
        'Transformer 语言模型也可以用 Viterbi 在 O(TV²) 内精确解码',
      ],
      answer: 1,
      explanation: md`A 错：不合并时前 N 条可能有多条同结尾，挤掉其它状态的最优前缀。C 错：$P(x)$ 是所有路径之和（Forward）。D 错：LM 的条件依赖整个前缀，不满足一阶马尔可夫性，没有可合并的状态。`,
    },
  ],

  guided: [],

  challenges: [
    {
      id: 'c-regex',
      title: '正则表达式匹配',
      difficulty: 'hard',
      source: 'LeetCode 10',
      timeLimitMin: 30,
      description: md`
        实现支持 \`'.'\` 和 \`'*'\` 的正则匹配，要求**完整匹配**整个字符串 \`s\`：
        - \`'.'\` 匹配任意单个字符
        - \`'*'\` 匹配零个或多个**前面那一个**元素

        \`\`\`python
        def is_match(s: str, p: str) -> bool: ...
        \`\`\`

        **示例**
        - \`is_match("aa", "a")\` → \`False\`
        - \`is_match("aa", "a*")\` → \`True\`
        - \`is_match("ab", ".*")\` → \`True\`
        - \`is_match("aab", "c*a*b")\` → \`True\`
        - \`is_match("mississippi", "mis*is*p*.")\` → \`False\`

        **约束**：$1 \le |s| \le 20$，$1 \le |p| \le 20$（隐藏测试更长）；\`s\` 只含小写字母；\`p\` 只含小写字母、\`.\`、\`*\`，且每个 \`*\` 前面都有一个合法字符。**不得使用 \`re\` 模块**（会检查源码）。要求 $O(|s| \cdot |p|)$。
      `,
      starter: py`
        def is_match(s: str, p: str) -> bool:
            # TODO: dp[i][j] = s[:i] 能否被 p[:j] 完整匹配
            raise NotImplementedError
      `,
      solution: py`
        def is_match(s: str, p: str) -> bool:
            m, n = len(s), len(p)
            # dp[i][j]: s[:i] 是否能被 p[:j] 完整匹配
            dp = [[False] * (n + 1) for _ in range(m + 1)]
            dp[0][0] = True
            for j in range(2, n + 1):                 # 空串只能被 "x*x*..." 匹配
                if p[j - 1] == '*':
                    dp[0][j] = dp[0][j - 2]

            def same(a: str, b: str) -> bool:
                return b == '.' or a == b

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if p[j - 1] == '*':
                        # 情况 1：「x*」匹配 0 次 → 丢掉这两个字符
                        # 情况 2：「x*」再多匹配一个 s[i-1] → s 退一格，p 不动
                        dp[i][j] = dp[i][j - 2] or (same(s[i - 1], p[j - 2]) and dp[i - 1][j])
                    else:
                        dp[i][j] = same(s[i - 1], p[j - 1]) and dp[i - 1][j - 1]
            return dp[m][n]
      `,
      explanation: md`
        **状态**：\`dp[i][j]\` = \`s[:i]\` 能否被 \`p[:j]\` 完整匹配（双序列前缀 DP）。

        **转移**（看 \`p\` 的最后一个字符）：
        - 普通字符或 \`.\`：末尾必须匹配，\`dp[i][j] = match(s[i-1], p[j-1]) and dp[i-1][j-1]\`；
        - \`*\`（与前一个字符 \`x = p[j-2]\` 组成一个单元）：
          - 匹配 0 次：\`dp[i][j-2]\`；
          - 匹配 ≥ 1 次：\`s[i-1]\` 要能被 \`x\` 匹配，然后**s 退一格、p 不动**——\`dp[i-1][j]\`，因为 \`x*\` 还可以继续吃掉更多字符。

        **初始化**：\`dp[0][0] = True\`；空串只能被 \`a*b*c*\` 这类模式匹配：\`dp[0][j] = dp[0][j-2]\`（当 \`p[j-1] == '*'\`）。

        **复杂度**：$O(mn)$ 时间与空间。朴素递归（不加缓存）在 \`"aaa…ab"\` 配 \`"a*a*a*…c"\` 时会指数爆炸，隐藏测试专门卡这一点。

        **常见错误**：“匹配多次”写成 \`dp[i-1][j-2]\`（只能多匹配恰好一次）；漏掉 \`dp[0][j]\` 的初始化；把 \`*\` 当成独立通配符。
      `,
      complexity: { time: 'O(mn)', space: 'O(mn)' },
      hints: [
        md`状态：\`dp[i][j]\` 表示 \`s\` 的前 $i$ 个字符能否被 \`p\` 的前 $j$ 个字符完整匹配。`,
        md`看 \`p[j-1]\`：不是 \`*\` 时，末尾两个字符必须匹配；是 \`*\` 时，把 \`p[j-2:j]\` 当成一个整体 \`x*\`。`,
        md`\`x*\` 要么匹配 0 次（\`dp[i][j-2]\`），要么在 \`s[i-1]\` 与 \`x\` 匹配的前提下继续使用（\`dp[i-1][j]\`）。别忘了初始化 \`dp[0][j]\`。`,
      ],
      setup: py`
        import inspect, random, re as _re, time

        def _ref(s, p):
            return _re.fullmatch(p, s) is not None

        def _rand_pattern(rnd, n):
            out = []
            while len(out) < n:
                out.append(rnd.choice('ab.'))
                if rnd.random() < 0.4:
                    out.append('*')
            return ''.join(out)
      `,
      tests: [
        {
          name: 'LC 示例',
          code: py`
            for s, p, exp in [("aa", "a", False), ("aa", "a*", True), ("ab", ".*", True), ("aab", "c*a*b", True), ("mississippi", "mis*is*p*.", False)]:
                got = is_match(s, p)
                assert got == exp, f"s={s!r}, p={p!r}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '不使用 re 模块',
          code: py`
            src = inspect.getsource(is_match)
            banned = _re.search(r"\bre\.\w+\(|\bimport\s+re\b|\bfullmatch\b", src)
            assert banned is None, f"请自己实现 DP，不要调用正则库（发现 {banned.group(0) if banned else ''}）"
          `,
        },
        {
          name: '空匹配与边界',
          code: py`
            for s, p, exp in [("a", "ab*", True), ("ab", ".*c", False), ("aaa", "a*a", True), ("aaa", "ab*a*c*a", True), ("a", ".*..a*", False)]:
                got = is_match(s, p)
                assert got == exp, f"s={s!r}, p={p!r}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '随机对拍（与标准正则比较）',
          hidden: true,
          code: py`
            rnd = random.Random(10)
            for _ in range(600):
                s = ''.join(rnd.choice('ab') for _ in range(rnd.randint(1, 8)))
                p = _rand_pattern(rnd, rnd.randint(1, 8))
                got, exp = is_match(s, p), _ref(s, p)
                assert got == exp, f"s={s!r}, p={p!r}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：卡指数级回溯',
          hidden: true,
          code: py`
            s = "a" * 40 + "b"
            p = "a*" * 25 + "c"
            t0 = time.perf_counter()
            got = is_match(s, p)
            dt = time.perf_counter() - t0
            assert got is False, f"期望 False，得到 {got}"
            assert dt < 2, f"用时 {dt:.2f}s：朴素递归会指数爆炸，请用 DP 或记忆化"
            assert is_match("a" * 60, "a*" * 30 + ".*a") is True
          `,
        },
      ],
      followUps: [
        {
          id: 'f-wildcard',
          q: md`通配符匹配（LC 44，\`'?'\` 匹配单个字符、\`'*'\` 匹配任意字符串）与本题有什么区别？`,
          a: md`LC 44 的 \`*\` 是**独立**的通配符，不依附前一个字符：\`dp[i][j] = dp[i][j-1]\`（\`*\` 匹配空）\`or dp[i-1][j]\`（\`*\` 多吃一个字符）。初始化时 \`dp[0][j]\` 只有前缀全是 \`*\` 才为 True。还可以用双指针 + 回溯到上一个 \`*\` 的贪心做到 $O(1)$ 空间。`,
        },
        {
          id: 'f-memo',
          q: md`用记忆化递归写这题会更自然吗？`,
          a: md`会：\`go(i, j)\` 判断 \`s[i:]\` 与 \`p[j:]\` 是否匹配，若 \`p[j+1] == '*'\` 则 \`go(i, j+2) or (first_match and go(i+1, j))\`，否则 \`first_match and go(i+1, j+1)\`。加 \`@cache\` 后同样是 $O(mn)$，但要注意递归深度（本题长度很小，不成问题）。`,
        },
      ],
    },
    {
      id: 'c-ones-zeroes',
      title: '一和零（二维费用背包）',
      difficulty: 'medium',
      source: 'LeetCode 474',
      timeLimitMin: 20,
      description: md`
        给定二进制字符串数组 \`strs\` 和两个整数 \`m\`、\`n\`，找出 \`strs\` 的**最大子集**的大小，使子集中所有字符串合计**最多**有 \`m\` 个 \`0\` 和 \`n\` 个 \`1\`。

        \`\`\`python
        def find_max_form(strs: list[str], m: int, n: int) -> int: ...
        \`\`\`

        **示例**
        - \`find_max_form(["10", "0001", "111001", "1", "0"], 5, 3)\` → \`4\`（\`{"10", "0001", "1", "0"}\`）
        - \`find_max_form(["10", "0", "1"], 1, 1)\` → \`2\`

        **约束**：$1 \le |strs| \le 200$，每个串长度 $\le 100$，$1 \le m, n \le 60$。要求 $O(|strs| \cdot m \cdot n)$ 时间、$O(mn)$ 空间。
      `,
      starter: py`
        def find_max_form(strs: list[str], m: int, n: int) -> int:
            # TODO: 每个字符串是一个物品，有两种“重量”（0 的个数、1 的个数），价值为 1
            raise NotImplementedError
      `,
      solution: py`
        def find_max_form(strs: list[str], m: int, n: int) -> int:
            # dp[i][j]: 最多用 i 个 0、j 个 1 时能选出的最多字符串数
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for s in strs:
                z, o = s.count('0'), s.count('1')
                # 0/1 背包：两个容量维度都要倒序，保证每个字符串只用一次
                for i in range(m, z - 1, -1):
                    for j in range(n, o - 1, -1):
                        if dp[i - z][j - o] + 1 > dp[i][j]:
                            dp[i][j] = dp[i - z][j - o] + 1
            return dp[m][n]
      `,
      explanation: md`
        **建模**：每个字符串是一个物品，“重量”有两维（0 的个数 $z$、1 的个数 $o$），价值为 1，求容量 $(m, n)$ 下的最大价值——**二维费用的 0/1 背包**。

        **状态与转移**：原本是三维 \`dp[k][i][j]\`（前 $k$ 个串），滚动掉物品维后 \`dp[i][j] = max(dp[i][j], dp[i-z][j-o] + 1)\`。

        **遍历方向**：每个字符串只能用一次，**两个容量维都倒序**，保证 \`dp[i-z][j-o]\` 是上一个物品处理后的旧值。

        **复杂度**：$O(|strs| \cdot m \cdot n)$ 时间，$O(mn)$ 空间。

        **常见错误**：容量正序（字符串被重复选）；用 \`[[0]*(n+1)]*(m+1)\` 造成行别名；把“最多 m 个 0”理解成“恰好”。
      `,
      complexity: { time: 'O(L·m·n)', space: 'O(m·n)' },
      hints: [
        md`每个字符串选或不选，代价是“0 的个数”和“1 的个数”两种资源——这是什么模型？`,
        md`二维费用的 0/1 背包：\`dp[i][j]\` = 最多 $i$ 个 0、$j$ 个 1 时的最大子集大小。`,
        md`对每个字符串，\`i\` 从 \`m\` 倒序到 \`z\`、\`j\` 从 \`n\` 倒序到 \`o\`：\`dp[i][j] = max(dp[i][j], dp[i-z][j-o] + 1)\`。`,
      ],
      setup: py`
        import random, time
        def _ref_brute(strs, m, n):
            best = 0
            k = len(strs)
            cnt = [(s.count('0'), s.count('1')) for s in strs]
            for mask in range(1 << k):
                z = o = c = 0
                for t in range(k):
                    if mask >> t & 1:
                        z += cnt[t][0]; o += cnt[t][1]; c += 1
                if z <= m and o <= n:
                    best = max(best, c)
            return best
        def _ref_dp(strs, m, n):
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for s in strs:
                z, o = s.count('0'), s.count('1')
                for i in range(m, z - 1, -1):
                    for j in range(n, o - 1, -1):
                        dp[i][j] = max(dp[i][j], dp[i - z][j - o] + 1)
            return dp[m][n]
      `,
      tests: [
        {
          name: 'LC 示例',
          code: py`
            for strs, m, n, exp in [(["10", "0001", "111001", "1", "0"], 5, 3, 4), (["10", "0", "1"], 1, 1, 2)]:
                got = find_max_form(strs, m, n)
                assert got == exp, f"strs={strs}, m={m}, n={n}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '每个字符串只能用一次',
          code: py`
            got = find_max_form(["0"], 5, 5)
            assert got == 1, f"只有一个字符串，期望 1，得到 {got}（是不是容量正序了？）"
            assert find_max_form(["111", "1000", "1000", "1000"], 9, 3) == 3
          `,
        },
        {
          name: '随机对拍（暴力枚举子集）',
          hidden: true,
          code: py`
            rnd = random.Random(474)
            for _ in range(200):
                strs = [''.join(rnd.choice('01') for _ in range(rnd.randint(1, 4))) for _ in range(rnd.randint(1, 9))]
                m, n = rnd.randint(0, 6), rnd.randint(0, 6)
                got, exp = find_max_form(strs, m, n), _ref_brute(strs, m, n)
                assert got == exp, f"strs={strs}, m={m}, n={n}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：200 个串，m = n = 60',
          hidden: true,
          code: py`
            rnd = random.Random(200)
            strs = [''.join(rnd.choice('01') for _ in range(rnd.randint(1, 8))) for _ in range(200)]
            t0 = time.perf_counter()
            got = find_max_form(strs, 60, 60)
            dt = time.perf_counter() - t0
            exp = _ref_dp(strs, 60, 60)
            assert got == exp, f"期望 {exp}，得到 {got}"
            assert dt < 5, f"用时 {dt:.2f}s，过慢"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-dims',
          q: md`如果有 $k$ 种资源约束（$k$ 维费用背包），复杂度会怎样？还能怎么做？`,
          a: md`DP 状态变成 $k$ 维，时间 $O(L \cdot \prod_i C_i)$，随维度指数增长（维度灾难）。$k$ 较大时一般改用整数规划 / 分支定界，或拉格朗日松弛把部分约束放进目标函数，再对乘子二分 / 次梯度优化。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'b-qa-identify',
      freq: 3,
      q: md`拿到一道题，怎么判断它能不能用 DP？`,
      a: md`
        **结论**：看三个信号——求最值 / 计数 / 可行性；决策可以分步进行；子问题会重复出现且“只依赖子问题的结果、不依赖其过程”。

        **方法**：先写暴力递归，若递归参数（状态）的取值空间是多项式级的、递归树中有大量重复参数，就可以记忆化；若需要“历史的具体路径”才能决定未来（不满足无后效性），尝试把必要的历史信息加入状态（如股票的“是否持有、第几笔”），状态空间仍可控就能 DP，否则考虑搜索或其它方法。

        **反例**：最长简单路径（需要记录访问过的点集，状态指数级）、一般图的 TSP（只能状压到 $O(2^n n^2)$）。
      `,
    },
    {
      id: 'b-qa-regex',
      freq: 2,
      q: md`正则匹配中 \`*\` 的转移为什么是 \`dp[i-1][j]\` 而不是 \`dp[i-1][j-2]\`？`,
      a: md`
        **结论**：\`dp[i-1][j]\` 表示 \`x*\` 吃掉 \`s[i-1]\` 后**仍然保留**，可以继续匹配更多字符；\`dp[i-1][j-2]\` 只能表示“恰好匹配一次”。

        **原因**：“匹配 $k \ge 1$ 次”可以拆成“先匹配一个字符，剩下的交给同一个 \`x*\` 继续匹配 $k-1$ 次”，这就是 \`dp[i-1][j]\` 的递归含义；配合“匹配 0 次”的 \`dp[i][j-2]\`，覆盖了所有次数。这与完全背包的“选了还能再选”（来源在本行）是同一个结构。
      `,
    },
  ],

  cheatsheet: md`
    **DP 答题五步**：暴力递归 → 状态 + 记忆化 → 推遍历顺序改递推 → 滚动数组 → 报复杂度与能否回溯

    \`\`\`python
    # LC 10：x* 匹配 0 次 or (末尾匹配 and 继续用 x*)
    dp[i][j] = dp[i][j-2] or (same(s[i-1], p[j-2]) and dp[i-1][j])
    # LC 474：二维费用 0/1 背包，两维都倒序
    for i in range(m, z-1, -1):
        for j in range(n, o-1, -1):
            dp[i][j] = max(dp[i][j], dp[i-z][j-o] + 1)
    \`\`\`

    | 倒序 | 正序 |
    |---|---|
    | 0/1 背包容量、区间 DP 的 i | 完全背包容量、LCS 滚动的 j |
  `,
})
