/**
 * algo-dp-stock：状态机 DP（股票系列）
 */
import { defineLevel, md, py } from '@/content/helpers'
import { StockStateViz, StockKViz } from '../viz/StockViz'

export default defineLevel({
  learn: [
    {
      id: 'machine',
      title: '状态机建模：持有 / 不持有',
      body: md`
        股票系列（LC 121 / 122 / 123 / 188 / 309 / 714）是同一个模型的不同约束。关键是把“今天结束时手里的情况”做成状态：

        - \`hold[i]\` = 第 $i$ 天结束时**持有**一股，此时的最大现金（买入花掉的钱记为负）
        - \`cash[i]\` = 第 $i$ 天结束时**不持有**，此时的最大现金

        每天只有三种动作：买、卖、什么都不做。画成状态机，转移就是图上的边：

        $$
        \begin{aligned}
        hold[i] &= \max(hold[i-1],\ cash[i-1] - p_i) && \text{保持 / 今天买入} \\
        cash[i] &= \max(cash[i-1],\ hold[i-1] + p_i) && \text{保持 / 今天卖出}
        \end{aligned}
        $$

        初始化 $hold[0] = -p_0$、$cash[0] = 0$，答案 $cash[n-1]$（最后不能还拿着股票）。这就是 **LC 122（不限交易次数）**，$O(n)$ 时间、$O(1)$ 空间。

        **LC 121（只能交易一次）** 是特例：买入只能从“从未交易过”的 0 出发，所以 $hold[i] = \max(hold[i-1], -p_i)$。等价于维护“历史最低价”，答案 $\max_i (p_i - \min_{j \le i} p_j)$。
      `,
      viz: StockStateViz,
      after: md`
        > [!TIP] 为什么用“现金”而不是“利润”
        > 把持有状态记成“扣掉买入价后的现金”，买入就是 $-p$、卖出就是 $+p$，所有转移都是加减法，不需要额外记录买入价。

        > [!NOTE] 同一天先买后卖？
        > 代码里用元组同时赋值，保证两个状态都从“昨天”转移。若写成先更新 \`hold\` 再用新 \`hold\` 更新 \`cash\`，对 122 结果不变（同一天买卖净收益为 0）；但在 309 冷冻期里，某些原地更新顺序会让“昨天刚卖出”在今天就转移到买入——**习惯性地用昨天的值**最安全。
      `,
      checkpoint: {
        id: 'cp-machine',
        type: 'numeric',
        prompt: md`LC 121（只能买卖一次）：\`prices = [7, 1, 5, 3, 6, 4]\`，最大利润是多少？`,
        answer: 5,
        tolerance: 0,
        explanation: md`第 1 天以 1 买入，第 4 天以 6 卖出，利润 5。不限次数（122）时可以 1→5、3→6，利润 7。`,
      },
    },
    {
      id: 'cooldown-fee',
      title: '加约束 = 加状态或改边：冷冻期与手续费',
      body: md`
        **LC 309 冷冻期**：卖出后的第二天不能买入。只靠 hold / cash 不够——“不持有”还得区分“是不是刚卖”。拆成三个状态：

        | 状态 | 含义 | 转移 |
        |---|---|---|
        | \`hold\` | 持有 | $\max(hold,\ rest - p)$：只能从**冷却结束 / 空仓**买入 |
        | \`sold\` | 今天刚卖出 | $hold + p$ |
        | \`rest\` | 不持有且不在冷冻 | $\max(rest,\ sold)$：昨天卖的今天冷冻 |

        答案 $\max(sold, rest)$。另一种等价写法是两状态 + “买入从 $cash[i-2]$ 转移”：$hold[i] = \max(hold[i-1],\ cash[i-2] - p_i)$。

        **LC 714 手续费**：只改一条边——卖出时 $+p - fee$（或买入时 $-p - fee$，二选一，不能两边都扣）。

        在上一节的图里切换到 309 / 714 模式，看状态图和表格怎么变。
      `,
      checkpoint: {
        id: 'cp-cool',
        type: 'single',
        prompt: md`LC 309 两状态写法中，买入转移为什么是 \`cash[i-2] - p\` 而不是 \`cash[i-1] - p\`？`,
        options: [
          '如果第 i-1 天刚卖出，第 i 天不能买；用 i-2 天的不持有状态，能保证中间至少隔了一天冷冻',
          '因为价格要隔一天才能更新',
          '因为 cash[i-1] 一定小于 cash[i-2]',
          '为了把空间复杂度降到 O(1)',
        ],
        answer: 0,
        explanation: md`$cash[i-1]$ 可能来自“第 $i-1$ 天卖出”，此时第 $i$ 天处于冷冻期。$cash[i-2]$ 的任何来源（包括第 $i-2$ 天卖出）到第 $i$ 天都已过了冷冻期。若第 $i-1$ 天没有卖出，$cash[i-1] = cash[i-2]$，用 $i-2$ 也不吃亏。`,
      },
    },
    {
      id: 'k',
      title: '至多 k 笔交易：交易次数也是状态',
      body: md`
        **LC 123（k = 2）/ LC 188（任意 k）**：交易次数受限，就把“已经进行到第几笔”放进状态：

        - \`buy[j]\` = 进行第 $j$ 笔买入后（持有）的最大现金
        - \`sell[j]\` = 完成第 $j$ 笔卖出后（不持有）的最大现金，\`sell[0] = 0\`

        $$
        buy[j] = \max(buy[j],\ sell[j-1] - p), \qquad sell[j] = \max(sell[j],\ buy[j] + p)
        $$

        “一笔交易”按**买入**计数：第 $j$ 次买入只能接在第 $j-1$ 次卖出之后。时间 $O(nk)$，空间 $O(k)$，答案 \`sell[k]\`。

        **k 很大时的退化**：一笔有效交易至少占两天，最多 $\lfloor n/2 \rfloor$ 笔。所以 $k \ge n/2$ 时等价于不限次数，直接用 122 的贪心（累加所有上涨差价）$O(n)$ 解决——否则 $k = 10^9$ 时 $O(nk)$ 会超时甚至开不出数组。
      `,
      viz: StockKViz,
      after: md`
        > [!INTERVIEW] 面试官常问
        > - \`buy[j]\` 用的是**今天刚更新**的 \`sell[j-1]\` 还是昨天的？内层 $j$ 从小到大更新时用的是今天的，相当于允许“同一天卖出第 $j-1$ 笔再买入第 $j$ 笔”，这笔操作净收益为 0、不会产生更优解，所以结果正确；按昨天的值写同样正确。
        > - 为什么 \`sell[k]\` 就是答案而不用取 max？因为 \`sell[j]\` 关于 $j$ 单调不减（多一次机会不会更差），\`sell[k]\` 已经包含“少做几笔”的情况。
      `,
      checkpoint: {
        id: 'cp-k',
        type: 'fill',
        prompt: md`\`prices = [3, 3, 5, 0, 0, 3, 1, 4]\`：$k = 1$ 时最大利润 ____(1)____，$k = 2$ 时 ____(2)____，$k = 3$ 时 ____(3)____。`,
        blanks: [
          { label: '(1)', answers: ['4'], numeric: true },
          { label: '(2)', answers: ['6'], numeric: true },
          { label: '(3)', answers: ['8'], numeric: true },
        ],
        explanation: md`$k=1$：0 买 4 卖，利润 4。$k=2$：0 买 3 卖（+3）、1 买 4 卖（+3），共 6。$k=3$：再加上 3 买 5 卖（+2），共 8；这也是不限次数的答案（所有上涨段之和）。`,
      },
    },
    {
      id: 'unify',
      title: '统一框架：六道题一张表',
      body: md`
        | 题 | 约束 | 状态 | 关键转移 |
        |---|---|---|---|
        | 121 | 交易 1 次 | hold, cash | $hold = \max(hold, -p)$ |
        | 122 | 不限次数 | hold, cash | $hold = \max(hold, cash - p)$ |
        | 123 | 至多 2 次 | buy1, sell1, buy2, sell2 | $buy_2 = \max(buy_2, sell_1 - p)$ |
        | 188 | 至多 k 次 | buy[j], sell[j] | $k \ge n/2$ 退化为 122 |
        | 309 | 冷冻期 1 天 | hold, sold, rest | 买入只能从 rest |
        | 714 | 手续费 | hold, cash | 卖出 $+p - fee$ |

        **建模套路**：
        1. 列出“今天结束时”可能的处境（持有？第几笔？冷冻中？）作为状态；
        2. 画出状态之间的边（买、卖、等待），每条边标上现金变化；
        3. 每个状态取所有入边的最大值；初始化第 0 天，答案取所有“不持有”状态的最大值。

        这套方法不只用于股票——任何“带有限个模式切换”的序列决策问题（如 LC 376 摆动序列、LC 552 学生出勤记录）都可以这样建模。
      `,
      checkpoint: {
        id: 'cp-unify',
        type: 'single',
        prompt: md`LC 188 中 $n = 1000$ 天，$k = 10^9$。最合适的做法是：`,
        options: [
          '直接开 k+1 大小的 buy/sell 数组做 O(nk) 的 DP',
          '因为 k ≥ n/2，退化为不限次数，累加所有相邻上涨差价，O(n)',
          '对 k 二分答案',
          '用堆维护前 k 大的上涨段，O(n log n)',
        ],
        answer: 1,
        explanation: md`$n$ 天最多完成 $\lfloor n/2 \rfloor = 500$ 笔有效交易，$k = 10^9$ 的限制等于没有限制。A 会超时且内存爆炸；D 不正确——前 k 大的“相邻上涨段”不能直接合并或拆分成最优交易（需要更复杂的栈 + 堆处理）。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-121',
      type: 'numeric',
      difficulty: 1,
      prompt: md`LC 121：\`prices = [7, 6, 4, 3, 1]\`，只能买卖一次，最大利润是多少？`,
      answer: 0,
      tolerance: 0,
      explanation: md`价格一路下跌，任何买卖都会亏钱，不交易利润为 0。初始化 \`cash = 0\` 天然覆盖了“不交易”。`,
    },
    {
      id: 'q-122-greedy',
      type: 'judge',
      difficulty: 1,
      prompt: md`LC 122（不限交易次数）的答案等于所有相邻两天正差价之和：$\sum_i \max(0, p_i - p_{i-1})$。`,
      answer: true,
      explanation: md`正确。任何一段上涨 $p_a \to p_b$ 的利润都等于其中相邻正差之和；不限次数时把每一段上涨都吃下即可。这是 122 的贪心解，与状态机 DP 结果相同。`,
    },
    {
      id: 'q-init',
      type: 'single',
      difficulty: 1,
      prompt: md`状态机写法中，第 0 天的初始化应为：`,
      options: ['`hold = -prices[0]`, `cash = 0`', '`hold = 0`, `cash = 0`', '`hold = prices[0]`, `cash = 0`', '`hold = -inf`, `cash = -inf`'],
      answer: 0,
      explanation: md`第 0 天持有只能是当天买入，现金为 $-p_0$；不持有就是什么都没做，现金 0。把 hold 初始化为 0 等于“免费拿到一股”，会高估利润。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与需要的状态设计配对：`,
      pairs: [
        ['LC 122 不限次数', '持有 / 不持有 两个状态'],
        ['LC 309 冷冻期', '持有 / 刚卖出 / 冷却空仓 三个状态'],
        ['LC 123 至多两次', 'buy1 / sell1 / buy2 / sell2 四个状态'],
        ['LC 714 手续费', '两个状态，卖出边扣 fee'],
      ],
      explanation: md`约束越多，状态越细：冷冻期要区分“刚卖”，交易次数要记录“第几笔”，手续费只改变边上的现金变化、不需要新状态。`,
    },
    {
      id: 'q-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 LC 188（至多 k 笔交易），下列说法正确的是：`,
      options: [
        '一笔交易可以在买入时计数，也可以在卖出时计数，但全程要一致',
        '当 k ≥ n/2 时可以退化为不限次数的贪心',
        '`sell[j]` 关于 j 单调不减，所以答案是 `sell[k]`',
        '空间可以压缩到 O(k)',
        '时间复杂度是 O(n log k)',
      ],
      answer: [0, 1, 2, 3],
      explanation: md`E 错：标准 DP 是 $O(nk)$。$O(n \log n)$ 级别的解法存在（如基于“反悔”的堆 / 栈方法），但不是这个 DP。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`用状态机方法设计一道新的股票题，步骤顺序是：`,
      items: [
        '列出“当天结束时”所有可能的处境作为状态',
        '画出状态间的边（买 / 卖 / 等待）并标注现金变化',
        '每个状态取所有入边（含自环“保持”）的最大值作为转移',
        '初始化第 0 天各状态，不可能的状态设为 -inf',
        '答案取所有“不持有”状态在最后一天的最大值',
      ],
      explanation: md`状态 → 转移 → 初始化 → 答案，就是 DP 四步法在状态机上的具体化。遍历顺序天然是按天从前往后。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LC 309（冷冻期）的三状态解法（有 2 行干扰项）：`,
      lines: [
        'def max_profit(prices):',
        '    hold, sold, rest = -prices[0], 0, 0',
        '    for p in prices[1:]:',
        '        hold, sold, rest = max(hold, rest - p), hold + p, max(rest, sold)',
        '    return max(sold, rest)',
      ],
      distractors: ['        hold, sold, rest = max(hold, sold - p), hold + p, max(rest, sold)', '    return hold'],
      explanation: md`买入只能从 \`rest\`（冷却结束或一直空仓）转移；干扰项从 \`sold\` 买入等于卖出第二天就买，违反冷冻期。最后一天不能持有，答案在 \`sold\` 与 \`rest\` 中取。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 LC 714（手续费）实现有一处 bug，点出那一行：`,
      code: py`
        def max_profit(prices, fee):
            hold, cash = -prices[0] - fee, 0
            for p in prices[1:]:
                new_hold = max(hold, cash - p - fee)
                new_cash = max(cash, hold + p - fee)
                hold, cash = new_hold, new_cash
            return cash
      `,
      bugLines: [5],
      explanation: md`买入时已经扣了手续费（第 2、4 行），第 5 行卖出时又扣了一次，每笔交易被收了两次费。手续费只能在买入或卖出其中一处扣：把第 5 行改为 \`hold + p\`（或把买入处的 \`- fee\` 去掉、保留卖出处的）。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`LC 122：\`prices = [1, 3, 2, 5, 4, 6]\`。贪心解在哪几天“卖出”（即 $p_i > p_{i-1}$ 的那些天 $i$）？`,
      items: ['第 0 天 (1)', '第 1 天 (3)', '第 2 天 (2)', '第 3 天 (5)', '第 4 天 (4)', '第 5 天 (6)'],
      answer: [1, 3, 5],
      explanation: md`上涨发生在 1→3、2→5、4→6，总利润 $2 + 3 + 2 = 7$。`,
    },
    {
      id: 'q-cool-num',
      type: 'numeric',
      difficulty: 3,
      prompt: md`LC 309（冷冻期 1 天）：\`prices = [6, 1, 3, 2, 4, 7]\`，最大利润是多少？`,
      answer: 6,
      tolerance: 0,
      explanation: md`若 1 买 3 卖（+2），第 3 天冷冻，第 4 天才能以 4 买入、7 卖出（+3），共 5；直接 1 买 7 卖 = 6 更优。没有冷冻期时可以 1→3、2→7 得 7。`,
    },
    {
      id: 'q-same-day',
      type: 'judge',
      difficulty: 3,
      prompt: md`LC 309 中，如果不用元组同时赋值，而是按「先 \`rest = max(rest, sold)\`，再 \`hold = max(hold, rest - p)\`，最后 \`sold = hold + p\`」的顺序逐个原地更新，结果仍然正确。`,
      answer: false,
      explanation: md`错误。\`hold\` 读到的是**今天刚更新**的 \`rest\`，而它可能来自昨天的 \`sold\`——等于卖出的第二天就买入，违反冷冻期。反例 \`[1, 2, 0, 3]\`：正确答案 3（1 买 3 卖，或 0 买 3 卖），这种写法得到 4（1 买 2 卖、第二天 0 买 3 卖）。有趣的是「hold → rest → sold」的顺序恰好是对的，但这种依赖顺序的写法极易出错，**用元组同时赋值**最稳妥。`,
    },
  ],

  guided: [
    {
      id: 'g-fee',
      title: '含手续费的股票（LC 714）：两状态模板',
      intro: md`
        实现 \`max_profit(prices, fee)\`：不限交易次数，每笔交易（买入 + 卖出）需付一次手续费 \`fee\`，求最大利润。用 hold / cash 两个状态。
      `,
      template: py`
        def max_profit(prices, fee):
            hold, cash = {{1}}
            for p in prices[1:]:
                hold, cash = {{2}}, {{3}}
            return {{4}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`第 0 天：持有只能是今天买入；不持有就是什么都不做。`,
          answer: '-prices[0], 0',
          explain: md`手续费统一在卖出时扣，所以买入这里不扣。`,
        },
        {
          blank: 2,
          instruction: md`今天结束时持有：要么昨天就持有，要么昨天不持有、今天买入。`,
          answer: 'max(hold, cash - p)',
          accept: ['max(cash - p, hold)'],
        },
        {
          blank: 3,
          instruction: md`今天结束时不持有：要么昨天就不持有，要么昨天持有、今天卖出（扣手续费）。`,
          answer: 'max(cash, hold + p - fee)',
          accept: ['max(hold + p - fee, cash)', 'max(cash, hold - fee + p)'],
          explain: md`元组同时赋值，右边的 \`hold\`、\`cash\` 都是昨天的值。手续费只在一处扣。`,
        },
        {
          blank: 4,
          instruction: md`返回答案：最后一天不能还持有股票。`,
          answer: 'cash',
        },
      ],
      setup: py`
        import random
        from functools import lru_cache
        def _ref(prices, fee):
            @lru_cache(None)
            def go(i, holding):
                if i == len(prices):
                    return 0
                best = go(i + 1, holding)
                if holding:
                    best = max(best, prices[i] - fee + go(i + 1, False))
                else:
                    best = max(best, -prices[i] + go(i + 1, True))
                return best
            return go(0, False)
      `,
      tests: [
        { name: 'LC 示例', code: py`
          assert max_profit([1, 3, 2, 8, 4, 9], 2) == 8
          assert max_profit([1, 3, 7, 5, 10, 3], 3) == 6
        ` },
        { name: '单日与下跌', code: py`
          assert max_profit([5], 1) == 0
          assert max_profit([9, 7, 4, 1], 1) == 0
        ` },
        {
          name: '随机对拍（记忆化搜索）',
          hidden: true,
          code: py`
            rnd = random.Random(714)
            for _ in range(200):
                prices = [rnd.randint(1, 15) for _ in range(rnd.randint(1, 12))]
                fee = rnd.randint(0, 4)
                got, exp = max_profit(prices, fee), _ref(prices, fee)
                assert got == exp, f"prices={prices}, fee={fee}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        把 \`- fee\` 去掉就是 LC 122；把买入改成 \`max(hold, -p)\` 就是 LC 121。同一个模板，改一条边就是一道新题。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-k-trans',
      title: '买卖股票的最佳时机 IV（至多 k 笔）',
      difficulty: 'hard',
      source: 'LeetCode 188',
      timeLimitMin: 25,
      description: md`
        给定整数 \`k\` 和价格数组 \`prices\`，最多完成 \`k\` 笔交易（买入并卖出算一笔），同一时间最多持有一股（必须先卖再买），求最大利润。

        \`\`\`python
        def max_profit(k: int, prices: list[int]) -> int: ...
        \`\`\`

        **示例**
        - \`max_profit(2, [2, 4, 1])\` → \`2\`
        - \`max_profit(2, [3, 2, 6, 5, 0, 3])\` → \`7\`

        **约束**：$0 \le n \le 10^4$，$0 \le prices[i] \le 1000$，$0 \le k \le 10^9$（**注意 k 可能远大于 n**，隐藏测试会卡 $O(nk)$）。
      `,
      starter: py`
        def max_profit(k: int, prices: list[int]) -> int:
            # TODO: buy[j] / sell[j] 状态机；k 很大时退化为不限次数
            raise NotImplementedError
      `,
      solution: py`
        def max_profit(k: int, prices: list[int]) -> int:
            n = len(prices)
            if n < 2 or k == 0:
                return 0
            if k >= n // 2:                       # 交易次数不构成限制 → LC 122 贪心
                return sum(max(0, prices[i] - prices[i - 1]) for i in range(1, n))
            buy = [-prices[0]] * (k + 1)          # buy[j]: 第 j 笔买入后的最大现金
            sell = [0] * (k + 1)                  # sell[j]: 第 j 笔卖出后的最大现金
            for p in prices[1:]:
                for j in range(1, k + 1):
                    buy[j] = max(buy[j], sell[j - 1] - p)
                    sell[j] = max(sell[j], buy[j] + p)
            return sell[k]
      `,
      solutionAlt: [
        {
          label: '二维 dp[i][j][0/1] 写法（便于理解，空间更大）',
          code: py`
            def max_profit(k, prices):
                n = len(prices)
                if n < 2 or k == 0:
                    return 0
                k = min(k, n // 2)
                NEG = float('-inf')
                # dp[j][0]: 已完成 j 笔且不持有；dp[j][1]: 正在进行第 j 笔（持有）
                dp = [[0, NEG] for _ in range(k + 1)]
                for p in prices:
                    for j in range(k, 0, -1):          # 倒序：用昨天的 dp[j-1]
                        dp[j][0] = max(dp[j][0], dp[j][1] + p)
                        dp[j][1] = max(dp[j][1], dp[j - 1][0] - p)
                return max(dp[j][0] for j in range(k + 1))
          `,
        },
      ],
      explanation: md`
        **状态**：\`buy[j]\` / \`sell[j]\` 为进行到第 $j$ 笔买入 / 卖出后的最大现金，\`sell[0] = 0\`。

        **转移**：$buy[j] = \max(buy[j], sell[j-1] - p)$，$sell[j] = \max(sell[j], buy[j] + p)$。按买入计数，第 $j$ 次买入只能接在第 $j-1$ 次卖出后。

        **k 的退化**：$n$ 天最多 $\lfloor n/2 \rfloor$ 笔有效交易，$k \ge n/2$ 时直接累加所有上涨差价。这一步不是优化细节，而是正确性的一部分：$k = 10^9$ 时 $O(nk)$ 必然超时、\`[0] * (k+1)\` 会耗尽内存。

        **复杂度**：$O(n \cdot \min(k, n/2))$ 时间，$O(\min(k, n))$ 空间。

        **常见错误**：忽略大 $k$；\`buy\` 初始化为 0（相当于免费持股）；按卖出计数时边界写错导致多算一笔。
      `,
      complexity: { time: 'O(n·min(k, n/2))', space: 'O(min(k, n))' },
      hints: [
        md`把“已经是第几笔”放进状态：\`buy[j]\`、\`sell[j]\`，$j = 1..k$。`,
        md`\`buy[j] = max(buy[j], sell[j-1] - p)\`，\`sell[j] = max(sell[j], buy[j] + p)\`，初始化 \`buy = -prices[0]\`、\`sell = 0\`。`,
        md`$n$ 天最多能完成几笔有意义的交易？当 $k$ 超过它时，问题退化成哪道题？`,
      ],
      setup: py`
        import random, time
        from functools import lru_cache
        def _ref_brute(k, prices):
            @lru_cache(None)
            def go(i, left, holding):
                if i == len(prices):
                    return 0
                best = go(i + 1, left, holding)
                if holding:
                    best = max(best, prices[i] + go(i + 1, left, False))
                elif left > 0:
                    best = max(best, -prices[i] + go(i + 1, left - 1, True))
                return best
            return go(0, k, False)
      `,
      tests: [
        { name: '示例', code: py`
          for k, p, exp in [(2, [2, 4, 1], 2), (2, [3, 2, 6, 5, 0, 3], 7)]:
              got = max_profit(k, p)
              assert got == exp, f"k={k}, prices={p}：期望 {exp}，得到 {got}"
        ` },
        { name: '边界：k=0、空数组、单日', code: py`
          assert max_profit(0, [1, 5]) == 0, "k=0 不能交易"
          assert max_profit(3, []) == 0
          assert max_profit(3, [4]) == 0
          assert max_profit(1, [3, 3, 5, 0, 0, 3, 1, 4]) == 4
        ` },
        {
          name: '随机对拍（记忆化搜索）',
          hidden: true,
          code: py`
            rnd = random.Random(188)
            for _ in range(200):
                p = [rnd.randint(0, 12) for _ in range(rnd.randint(0, 11))]
                k = rnd.randint(0, 5)
                got, exp = max_profit(k, p), _ref_brute(k, p)
                assert got == exp, f"k={k}, prices={p}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '超大 k：k = 10^9，n = 10^4',
          hidden: true,
          code: py`
            rnd = random.Random(10 ** 9)
            p = [rnd.randint(0, 1000) for _ in range(10_000)]
            exp = sum(max(0, p[i] - p[i - 1]) for i in range(1, len(p)))
            t0 = time.perf_counter()
            got = max_profit(10 ** 9, p)
            dt = time.perf_counter() - t0
            assert got == exp, f"期望 {exp}，得到 {got}"
            assert dt < 2, f"用时 {dt:.2f}s：k 远大于 n/2 时应退化为 O(n) 贪心"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-123',
          q: md`LC 123（$k = 2$）能只用 4 个变量写吗？`,
          a: md`可以：\`b1 = max(b1, -p); s1 = max(s1, b1 + p); b2 = max(b2, s1 - p); s2 = max(s2, b2 + p)\`，按这个顺序原地更新，答案 \`s2\`。原地更新相当于允许同一天卖出再买入，净收益为 0，不影响结果。`,
        },
      ],
    },
    {
      id: 'c-cooldown',
      title: '最佳买卖股票时机含冷冻期',
      difficulty: 'medium',
      source: 'LeetCode 309',
      timeLimitMin: 20,
      description: md`
        不限交易次数，但卖出股票后，**第二天**不能买入（冷冻期 1 天）。同一时间最多持有一股。求最大利润。

        \`\`\`python
        def max_profit_cooldown(prices: list[int]) -> int: ...
        \`\`\`

        **示例**
        - \`[1, 2, 3, 0, 2]\` → \`3\`（买、卖、冷冻、买、卖）
        - \`[1]\` → \`0\`

        **约束**：$0 \le n \le 5000$，$0 \le prices[i] \le 1000$。要求 $O(n)$。
      `,
      starter: py`
        def max_profit_cooldown(prices: list[int]) -> int:
            # TODO: hold / sold / rest 三个状态
            return 0
      `,
      solution: py`
        def max_profit_cooldown(prices: list[int]) -> int:
            if not prices:
                return 0
            hold, sold, rest = -prices[0], 0, 0
            for p in prices[1:]:
                # 同时赋值：右侧全部是昨天的状态
                hold, sold, rest = (
                    max(hold, rest - p),   # 继续持有 / 冷却结束后买入
                    hold + p,              # 今天卖出
                    max(rest, sold),       # 继续空仓 / 昨天卖出今天冷冻
                )
            return max(sold, rest)
      `,
      explanation: md`
        **状态**：\`hold\`（持有）、\`sold\`（今天刚卖出）、\`rest\`（不持有且可以买）。

        **转移**：
        - \`hold = max(hold, rest - p)\`：只能从 \`rest\` 买入——昨天刚卖（\`sold\`）的今天处于冷冻；
        - \`sold = hold + p\`；
        - \`rest = max(rest, sold)\`：昨天卖出的，今天进入冷冻（变成 rest，明天才能买）。

        **答案**：\`max(sold, rest)\`。**复杂度** $O(n)$ 时间、$O(1)$ 空间。

        **等价写法**：两状态 + 买入看前天：$hold_i = \max(hold_{i-1},\ cash_{i-2} - p_i)$。

        **常见错误**：从 \`sold\` 直接买入；逐个原地更新导致用到今天的新值；空数组未处理。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`“不持有”要区分两种：刚卖出（明天不能买）和可以买。`,
        md`三个状态 hold / sold / rest，画出它们之间的边：rest → hold（买）、hold → sold（卖）、sold → rest（冷冻）。`,
        md`用元组同时赋值，保证右侧都是昨天的值；答案 \`max(sold, rest)\`。`,
      ],
      setup: py`
        import random
        from functools import lru_cache
        def _ref(prices):
            @lru_cache(None)
            def go(i, holding):
                if i >= len(prices):
                    return 0
                best = go(i + 1, holding)
                if holding:
                    best = max(best, prices[i] + go(i + 2, False))   # 卖出后跳过冷冻日
                else:
                    best = max(best, -prices[i] + go(i + 1, True))
                return best
            return go(0, False)
      `,
      tests: [
        { name: '示例', code: py`
          assert max_profit_cooldown([1, 2, 3, 0, 2]) == 3
          assert max_profit_cooldown([1]) == 0
        ` },
        { name: '冷冻期生效', code: py`
          got = max_profit_cooldown([6, 1, 3, 2, 4, 7])
          assert got == 6, f"[6,1,3,2,4,7]：期望 6，得到 {got}（没有冷冻期才是 7）"
          assert max_profit_cooldown([]) == 0
        ` },
        {
          name: '随机对拍（记忆化搜索）',
          hidden: true,
          code: py`
            rnd = random.Random(309)
            for _ in range(300):
                p = [rnd.randint(0, 10) for _ in range(rnd.randint(0, 12))]
                got, exp = max_profit_cooldown(p), _ref(p)
                assert got == exp, f"prices={p}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：n = 5000',
          hidden: true,
          code: py`
            import sys
            rnd = random.Random(5000)
            p = [rnd.randint(0, 1000) for _ in range(5000)]
            # 参考：两状态 + 前天写法
            h, c_prev, c = -p[0], 0, 0
            for x in p[1:]:
                h, c_prev, c = max(h, c_prev - x), c, max(c, h + x)
            got = max_profit_cooldown(p)
            assert got == c, f"期望 {c}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-k-cool',
          q: md`如果冷冻期是 $d$ 天，怎么改？`,
          a: md`用两状态写法：$hold_i = \max(hold_{i-1},\ cash_{i-d-1} - p_i)$，即买入要从 $d+1$ 天前的“不持有”状态转移；保留最近 $d+1$ 个 \`cash\` 值（环形缓冲区），空间 $O(d)$。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-framework',
      freq: 3,
      q: md`股票系列问题有统一的解法吗？讲讲你的建模思路。`,
      a: md`
        **结论**：都是状态机 DP——状态是“当天结束时的处境”，转移是买 / 卖 / 等待三类边。

        **原因**：基础状态是持有 / 不持有；有交易次数限制就加一维“第几笔”（123/188），有冷冻期就把“不持有”拆成“刚卖出 / 可买入”（309），有手续费就修改卖出边的收益（714）。每个状态取所有入边的最大值，初始化第 0 天，答案取不持有状态的最大值。

        **延伸**：$O(n)$ 或 $O(nk)$ 时间，空间都能滚动到 $O(1)$ / $O(k)$；$k \ge n/2$ 时 188 退化为 122 的贪心。
      `,
    },
    {
      id: 'qa-k',
      freq: 2,
      q: md`LC 188 中 $k$ 很大会出什么问题？怎么处理？`,
      a: md`
        **结论**：$k \ge n/2$ 时交易次数不构成约束，直接用不限次数的贪心 $O(n)$。

        **原因**：每笔有效交易至少占两天（买、卖不同天），$n$ 天最多 $\lfloor n/2 \rfloor$ 笔。不做这个判断，$k = 10^9$ 时 $O(nk)$ 会超时，开 $k+1$ 大小的数组还会内存溢出。

        **延伸**：存在 $O(n \log n)$ 的解法：先把价格拆成若干“谷—峰”对，用栈合并可以合并的区间、用堆取最大的 $k$ 段收益（类似 wqs 二分 / 反悔贪心的思路），面试能说出思路即可。
      `,
    },
    {
      id: 'qa-update',
      freq: 2,
      q: md`状态机 DP 中，同一天的几个状态应该“同时更新”还是“依次更新”？`,
      a: md`
        **结论**：默认同时更新（所有右侧都用昨天的值），最安全。

        **原因**：依次更新会让后更新的状态读到今天的新值，相当于允许同一天连续发生多个动作。对 122、123、188，“同一天卖了再买 / 买了再卖”净收益为 0，不影响答案；但对冷冻期这类有时间间隔约束的问题，会让“昨天卖出”的状态在今天就转移到买入，违反约束。

        **写法**：Python 用元组同时赋值，C++/Java 先算出所有新值再统一赋值。
      `,
    },
  ],

  cheatsheet: md`
    \`\`\`python
    hold, cash = -prices[0], 0                  # 122
    for p in prices[1:]:
        hold, cash = max(hold, cash - p), max(cash, hold + p)
    # 121：hold = max(hold, -p)          714：卖出 hold + p - fee
    # 309：hold, sold, rest = max(hold, rest - p), hold + p, max(rest, sold)
    # 188：k >= n//2 → 贪心；否则
    for p in prices[1:]:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j-1] - p)
            sell[j] = max(sell[j], buy[j] + p)
    \`\`\`

    - 状态 = 当天结束时的处境；转移 = 买 / 卖 / 等待
    - 初始化 hold = −p₀；答案取“不持有”
    - 用元组同时赋值；手续费只扣一次
  `,
})
