/**
 * algo-dp-knapsack：背包问题（0/1、完全、可行性、计数）
 */
import { defineLevel, md, py } from '@/content/helpers'
import { Knapsack2DViz, RollingViz, ComboPermViz } from '../viz/KnapsackViz'

export default defineLevel({
  learn: [
    {
      id: 'zero-one',
      title: '0/1 背包：二维表与「选 / 不选」',
      body: md`
        $n$ 个物品，第 $i$ 个重量 $w_i$、价值 $v_i$，每个**最多选一次**，背包容量 $C$，求最大总价值。

        - **状态**：\`dp[i][j]\` = 只考虑前 $i$ 个物品、容量**不超过** $j$ 时的最大价值
        - **转移**：对第 $i$ 个物品二选一
          $$
          dp[i][j] = \max\big(\underbrace{dp[i-1][j]}_{\text{不选}},\ \underbrace{dp[i-1][j-w_i] + v_i}_{\text{选（需 } j \ge w_i)}\big)
          $$
        - **初始化**：$dp[0][j] = 0$（不选任何物品）
        - **遍历顺序**：$i$ 从小到大；第 $i$ 行只依赖第 $i-1$ 行，所以 $j$ 的顺序在二维写法里无所谓
        - 复杂度 $O(nC)$ 时间、$O(nC)$ 空间

        关键观察：**“选”的来源 $dp[i-1][j-w_i]$ 在上一行**——上一行还没考虑物品 $i$，所以物品 $i$ 最多被加入一次。
      `,
      viz: Knapsack2DViz,
      after: md`
        > [!NOTE] “不超过” vs “恰好装满”
        > 上面的初始化（第 0 行全 0）求的是**容量不超过** $j$ 的最大价值。如果题目要求**恰好装满**，就把 $dp[0][0] = 0$、$dp[0][j>0] = -\infty$，不可达的状态会一直是 $-\infty$。
        > 求最小值（如零钱兑换的最少硬币数）时对应地初始化为 $+\infty$。

        > [!WARN] 复杂度是「伪多项式」
        > $O(nC)$ 依赖容量 $C$ 的**数值**大小，而不是输入长度。0/1 背包（判定版）是 NP 完全问题，当 $C$ 达到 $10^9$ 时这个 DP 不可用，需要换成按价值 DP、折半枚举（meet in the middle）等方法。
      `,
      checkpoint: {
        id: 'cp-01',
        type: 'numeric',
        prompt: md`物品 (w, v) = (1, 1), (3, 4), (4, 5), (5, 7)，容量 $C = 7$，0/1 背包的最大价值是多少？`,
        answer: 9,
        tolerance: 0,
        explanation: md`选 (3, 4) 与 (4, 5)：重量 7，价值 9。其它组合如 (1,1)+(5,7) 只有 8。最后一行 $dp[4][7] = 9$。`,
      },
    },
    {
      id: 'rolling',
      title: '一维滚动：0/1 背包为什么必须倒序',
      body: md`
        第 $i$ 行只依赖第 $i-1$ 行，可以压成一维数组 \`dp[j]\`，原地覆盖：

        \`\`\`python
        for w, v in zip(weights, values):
            for j in range(C, w - 1, -1):     # 倒序！
                dp[j] = max(dp[j], dp[j - w] + v)
        \`\`\`

        覆盖之前，\`dp[j]\` 是上一行的值（不选）；而 \`dp[j - w]\` 必须也是**上一行**的值（选之前的状态）。
        - **倒序**：\`j\` 从大到小，\`dp[j - w]\`（更小的下标）在本轮还**没被更新**，读到的是上一行——正确；
        - **正序**：\`dp[j - w]\` 在本轮**已经被更新**过，它可能已经包含了当前物品，再加一次 $v$ 就等于同一个物品放了两次。

        正序恰好就是**完全背包**（每种物品无限个）的转移：
        $$
        dp[i][j] = \max(dp[i-1][j],\ dp[\,\boldsymbol{i}\,][j - w_i] + v_i)
        $$
        “选”的来源在**本行**，表示选了一个物品 $i$ 之后还可以继续选物品 $i$。
      `,
      viz: RollingViz,
      after: md`
        > [!TIP] 一句话记忆
        > **0/1 倒序读旧值，完全正序读新值。** 判断滚动方向的通用方法：想清楚转移需要的是“旧值（上一行）”还是“新值（本行）”，再让遍历顺序保证这一点。
        > 内层循环的下界写 \`w - 1\`（即 $j \ge w$），既省掉 \`if\`，又避免负下标在 Python 里悄悄读到数组末尾。
      `,
      checkpoint: {
        id: 'cp-roll',
        type: 'single',
        prompt: md`只有一个物品 (w=2, v=3)，容量 8。若把 0/1 背包的一维写法错写成**正序**，最终 \`dp[8]\` 是多少？`,
        options: ['3', '6', '9', '12'],
        answer: 3,
        explanation: md`正序时 \`dp[2]=3\`，\`dp[4]=dp[2]+3=6\`，\`dp[6]=9\`，\`dp[8]=12\`——同一个物品被放了 4 次。正确的倒序结果是 3。`,
      },
    },
    {
      id: 'complete',
      title: '完全背包与零钱兑换：最值型',
      body: md`
        **零钱兑换**（LC 322）：硬币面值 \`coins\`（每种无限枚），凑出 \`amount\` 的**最少**硬币数，凑不出返回 -1。

        这是完全背包的“最小值 + 恰好装满”版本：
        - **状态**：\`dp[j]\` = 恰好凑出金额 $j$ 的最少硬币数
        - **转移**：$dp[j] = \min_{c \in coins,\ c \le j} dp[j - c] + 1$
        - **初始化**：$dp[0] = 0$，其余为 $+\infty$（**恰好装满**，不可达就是 $\infty$）
        - **顺序**：硬币可重复 → 完全背包 → 容量**正序**；对最值问题，内外层循环谁在外都可以
        - **返回**：$dp[amount]$ 仍为 $\infty$ 时返回 -1

        为什么不能贪心（每次拿最大面值）？\`coins = [1, 3, 4]\`、\`amount = 6\`：贪心 $4+1+1$ 用 3 枚，最优 $3+3$ 只要 2 枚。人民币、美元这类面值体系恰好让贪心成立，但任意面值不行。

        **分割等和子集**（LC 416）则是 0/1 背包的**可行性**版本：能否选出若干个数，和恰为 $\text{sum}/2$？\`dp[j]\` 是布尔值，转移 \`dp[j] = dp[j] or dp[j - x]\`，倒序遍历。总和为奇数可以直接返回 False。
      `,
      checkpoint: {
        id: 'cp-coin',
        type: 'numeric',
        prompt: md`\`coins = [1, 2, 5, 10]\`，凑出 27 最少需要几枚硬币？`,
        answer: 4,
        tolerance: 0,
        explanation: md`$10 + 10 + 5 + 2 = 27$，4 枚。这组面值下贪心恰好正确；DP 的结果与贪心一致。`,
      },
    },
    {
      id: 'combo-perm',
      title: '计数型：组合数 vs 排列数，循环顺序说了算',
      body: md`
        **零钱兑换 II**（LC 518）：凑出 \`amount\` 的**组合数**（\`1+2\` 与 \`2+1\` 算同一种）。
        **组合总和 IV**（LC 377）：凑出 \`target\` 的**排列数**（顺序不同算不同）。

        两题的转移都是 $dp[j] \mathrel{+}= dp[j - c]$，$dp[0] = 1$，**唯一区别是两层循环的顺序**：

        | | 外层 | 内层 | 计数的是 |
        |---|---|---|---|
        | LC 518 | 物品 \`for c in coins\` | 容量 \`for j in range(c, amount+1)\` | 组合 |
        | LC 377 | 容量 \`for j in range(1, target+1)\` | 物品 \`for c in nums\` | 排列 |

        **为什么**：外层是物品时，处理到硬币 $c$ 时只用过 $c$ 之前的硬币，方案里的硬币天然**按面值顺序**加入，每个多重集合只被数一次；
        外层是容量时，每个 $dp[j]$ 都在枚举“**最后一枚**是哪种硬币”，最后一枚不同的序列被分别计数，于是顺序不同的序列都被数到。
      `,
      viz: ComboPermViz,
      after: md`
        > [!WARN] 计数题的易错点
        > - $dp[0] = 1$，不是 0（空方案是计数的起点）；
        > - LC 377 中间值可能非常大（题目保证答案在 32 位内，但中间的 $dp[j]$ 可能溢出 C++/Java 的 int）；
        > - 最值问题（322）里内外层顺序无所谓，**只有计数问题**才区分组合与排列。
      `,
      checkpoint: {
        id: 'cp-cp',
        type: 'fill',
        prompt: md`硬币 \`[1, 2]\`，凑出金额 5：组合数（LC 518）= ____(1)____，排列数（LC 377）= ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['3'], numeric: true },
          { label: '(2)', answers: ['8'], numeric: true },
        ],
        explanation: md`组合：\`1×5\`、\`1×3+2\`、\`1+2×2\`，共 3 种。排列：按 2 的个数分，0 个 2 有 1 种，1 个 2（共 4 个位置）有 $\binom{4}{1}=4$ 种，2 个 2（共 3 个位置）有 $\binom{3}{2}=3$ 种，合计 8——也就是斐波那契数，和爬楼梯完全相同。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-reverse',
      type: 'single',
      difficulty: 1,
      prompt: md`0/1 背包一维滚动写法中，容量 $j$ 必须倒序遍历的原因是：`,
      options: [
        '保证 `dp[j - w]` 读到的是上一个物品处理完后的旧值，避免同一物品被重复选',
        '倒序遍历的常数更小，速度更快',
        '保证 `dp[j]` 读到的是本轮更新后的新值',
        '防止下标 `j - w` 越界',
      ],
      answer: 0,
      explanation: md`倒序时更小的下标还没被本轮更新，\`dp[j-w]\` 仍是“没考虑当前物品”的状态。正序会读到本轮新值，等价于完全背包。越界问题靠循环下界 $j \ge w$ 解决，与方向无关。`,
    },
    {
      id: 'q-complete-judge',
      type: 'judge',
      difficulty: 1,
      prompt: md`完全背包（每种物品无限个）的一维写法中，容量应当正序遍历。`,
      answer: true,
      explanation: md`正确。正序时 \`dp[j - w]\` 可能已经包含当前物品，再选一次正是“无限个”的语义，对应二维转移 $dp[i][j-w] + v$（来源在本行）。`,
    },
    {
      id: 'q-coin',
      type: 'numeric',
      difficulty: 1,
      prompt: md`LC 322：\`coins = [1, 2, 5]\`，\`amount = 11\`，最少需要几枚硬币？`,
      answer: 3,
      tolerance: 0,
      explanation: md`$5 + 5 + 1 = 11$，3 枚。`,
    },
    {
      id: 'q-init',
      type: 'match',
      difficulty: 2,
      prompt: md`把问题与正确的 dp 初始化配对：`,
      pairs: [
        ['0/1 背包，容量不超过 C 的最大价值', '`dp = [0] * (C + 1)`'],
        ['背包恰好装满时的最大价值', '`dp[0] = 0`，其余 `-inf`'],
        ['零钱兑换：最少硬币数', '`dp[0] = 0`，其余 `inf`'],
        ['零钱兑换 II：组合数', '`dp[0] = 1`，其余 0'],
      ],
      explanation: md`“不超过”时任意容量都可以什么都不放，初值 0；“恰好装满”要用 $\pm\infty$ 标记不可达；计数问题 $dp[0]=1$ 表示空方案。初始化错误是背包题第二常见的 bug（第一是遍历方向）。`,
    },
    {
      id: 'q-combo',
      type: 'multi',
      difficulty: 2,
      prompt: md`下列关于 LC 518（组合数）与 LC 377（排列数）的说法，正确的是：`,
      options: [
        '518 外层遍历硬币、内层遍历金额',
        '377 外层遍历目标值、内层遍历元素',
        '两题的内层遍历方向都必须是倒序',
        '对 `coins = [1, 2]`、金额 3，518 的答案是 2，377 的答案是 3',
        '两层循环交换顺序不影响 LC 322（最少硬币数）的结果',
      ],
      answer: [0, 1, 3, 4],
      explanation: md`C 错：两题元素都可以重复使用（完全背包），金额正序。D：组合 \`{1,1,1}, {1,2}\` 共 2 种；排列 \`111, 12, 21\` 共 3 种。E：最值问题只关心最优值，不区分方案的顺序，交换循环不影响结果。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 LC 416「分割等和子集」的解题步骤按顺序排列：`,
      items: [
        '求总和 s，若 s 为奇数直接返回 False',
        '目标 target = s // 2，建立布尔数组 dp，dp[0] = True',
        '外层遍历每个数 x',
        '内层 j 从 target 倒序到 x：dp[j] = dp[j] or dp[j - x]',
        '返回 dp[target]',
      ],
      explanation: md`把“分成两个等和子集”转化为“能否选出和为 $s/2$ 的子集”，即 0/1 背包的可行性版本，每个数只能用一次，所以倒序。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LC 322 零钱兑换（有 2 行干扰项）：`,
      lines: [
        'def coin_change(coins, amount):',
        "    INF = float('inf')",
        '    dp = [0] + [INF] * amount',
        '    for c in coins:',
        '        for j in range(c, amount + 1):',
        '            dp[j] = min(dp[j], dp[j - c] + 1)',
        '    return dp[amount] if dp[amount] != INF else -1',
      ],
      distractors: ['    dp = [0] * (amount + 1)', '        for j in range(amount, c - 1, -1):'],
      explanation: md`“恰好凑出”要把不可达初始化为 $\infty$；全 0 初始化会让所有 \`dp[j]\` 都是 0。硬币可以重复使用，所以金额**正序**；倒序就变成了每种硬币最多用一枚。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 LC 416 实现在 \`[1, 5, 11, 5]\` 上正确，但在 \`[2, 2, 3, 5]\`（总和 12）上错误地返回 True。点出有 bug 的行：`,
      code: py`
        def can_partition(nums):
            s = sum(nums)
            if s % 2:
                return False
            target = s // 2
            dp = [False] * (target + 1)
            dp[0] = True
            for x in nums:
                for j in range(x, target + 1):
                    dp[j] = dp[j] or dp[j - x]
            return dp[target]
      `,
      bugLines: [9],
      explanation: md`第 9 行应倒序：\`for j in range(target, x - 1, -1)\`。正序让每个数可以重复使用：\`[2, 2, 3, 5]\` 的目标是 6，正序时 $3 + 3 = 6$ 被判为可行，但 3 只有一个。正确答案是 False（和为 6 的子集不存在：2+2=4，2+3=5，5+2=7…）。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`0/1 背包一维写法，处理物品 (w=3, v=4) 时，若 **正序** 遍历 $j = 3 \dots 9$，哪些 $j$ 的更新会读到**本轮已被更新过**的 \`dp[j - 3]\`？`,
      items: ['j=3', 'j=4', 'j=5', 'j=6', 'j=7', 'j=8', 'j=9'],
      answer: [3, 4, 5, 6],
      explanation: md`本轮更新的是 $j \ge 3$ 的格子。读 \`dp[j-3]\` 时，若 $j - 3 \ge 3$，即 $j \ge 6$，该格本轮已更新——$j = 6, 7, 8, 9$ 都会把物品重复放入。$j = 3, 4, 5$ 读的是 $dp[0..2]$，本轮没被动过。`,
    },
    {
      id: 'q-perm-num',
      type: 'numeric',
      difficulty: 3,
      prompt: md`LC 377：\`nums = [1, 2, 3]\`，\`target = 4\`，排列数是多少？`,
      answer: 7,
      tolerance: 0,
      explanation: md`$dp = [1, 1, 2, 4, 7]$：$dp[4] = dp[3] + dp[2] + dp[1] = 4 + 2 + 1 = 7$。分别是 1111、112、121、211、22、13、31。同样的硬币求组合数（LC 518）只有 4 种。`,
    },
    {
      id: 'q-np',
      type: 'judge',
      difficulty: 3,
      prompt: md`0/1 背包的 DP 复杂度是 $O(nC)$，所以 0/1 背包问题属于多项式时间可解的问题。`,
      answer: false,
      explanation: md`错误。$O(nC)$ 是**伪多项式**：$C$ 的数值可以是输入长度（约 $\log C$ 位）的指数级。0/1 背包的判定版本是 NP 完全问题。$C$ 很大而物品价值小时可以改为“按价值 DP”（\`dp[v]\` = 达到价值 $v$ 的最小重量），$n \le 40$ 时可用折半枚举。`,
    },
  ],

  guided: [
    {
      id: 'g-01-1d',
      title: '一维滚动的 0/1 背包模板',
      intro: md`
        实现 \`knapsack_01(weights, values, C)\`：每个物品最多选一次，返回容量不超过 $C$ 时的最大总价值。使用一维 \`dp\` 数组。
      `,
      template: py`
        def knapsack_01(weights, values, C):
            dp = {{1}}
            for w, v in zip(weights, values):
                for j in {{2}}:
                    dp[j] = {{3}}
            return {{4}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`初始化：容量不超过 $j$、一个物品都不选时价值为 0。数组长度是多少？`,
          answer: '[0] * (C + 1)',
          accept: ['[0]*(C+1)', '[0 for _ in range(C + 1)]'],
          explain: md`下标 $0 \dots C$ 共 $C+1$ 个状态。“不超过”的语义下全部初始化为 0。`,
        },
        {
          blank: 2,
          instruction: md`容量从大到小遍历，只遍历放得下当前物品的 $j$。`,
          answer: 'range(C, w - 1, -1)',
          accept: ['range(C, w-1, -1)', 'reversed(range(w, C + 1))', 'reversed(range(w, C+1))'],
          explain: md`倒序保证 \`dp[j - w]\` 还是上一个物品处理后的旧值。写成正序就变成了完全背包，隐藏测试会抓出来。`,
        },
        {
          blank: 3,
          instruction: md`选 / 不选取较大值。`,
          answer: 'max(dp[j], dp[j - w] + v)',
          accept: ['max(dp[j], dp[j-w] + v)', 'max(dp[j - w] + v, dp[j])'],
        },
        {
          blank: 4,
          instruction: md`返回答案。`,
          answer: 'dp[C]',
          accept: ['dp[-1]', 'max(dp)'],
          explain: md`“不超过”语义下 $dp$ 单调不减，\`dp[C]\` 就是最大值。`,
        },
      ],
      setup: py`
        import random
        from itertools import combinations
        def _ref_brute(w, v, C):
            best = 0
            n = len(w)
            for mask in range(1 << n):
                tw = sum(w[i] for i in range(n) if mask >> i & 1)
                if tw <= C:
                    best = max(best, sum(v[i] for i in range(n) if mask >> i & 1))
            return best
      `,
      tests: [
        { name: '示例', code: py`
          got = knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7)
          assert got == 9, f"期望 9，得到 {got}"
        ` },
        { name: '单个物品不能重复选', code: py`
          got = knapsack_01([2], [3], 8)
          assert got == 3, f"只有一个物品，期望 3，得到 {got}（是不是写成了正序？）"
        ` },
        {
          name: '随机对拍（暴力枚举）',
          hidden: true,
          code: py`
            rnd = random.Random(416)
            for _ in range(200):
                n = rnd.randint(0, 8)
                w = [rnd.randint(1, 6) for _ in range(n)]
                v = [rnd.randint(0, 10) for _ in range(n)]
                C = rnd.randint(0, 15)
                got, exp = knapsack_01(w, v, C), _ref_brute(w, v, C)
                assert got == exp, f"w={w}, v={v}, C={C}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        把第 2 空换成 \`range(w, C + 1)\` 就是完全背包；把 \`max\` 换成 \`or\`、初始化改为 \`[True] + [False] * C\` 就是可行性背包（LC 416）。
        **一个模板，三种变体，区别只在遍历方向、初始化与合并运算。**
      `,
    },
  ],

  challenges: [
    {
      id: 'c-partition',
      title: '分割等和子集',
      difficulty: 'medium',
      source: 'LeetCode 416',
      timeLimitMin: 15,
      description: md`
        给定只含正整数的非空数组 \`nums\`，判断能否把它分成两个**元素和相等**的子集。

        \`\`\`python
        def can_partition(nums: list[int]) -> bool: ...
        \`\`\`

        **示例**
        - \`[1, 5, 11, 5]\` → \`True\`（\`[1, 5, 5]\` 与 \`[11]\`）
        - \`[1, 2, 3, 5]\` → \`False\`

        **约束**：$1 \le n \le 200$，$1 \le nums[i] \le 100$。要求时间 $O(n \cdot \text{sum})$（隐藏测试 $n = 200$，暴力枚举子集会超时）。
      `,
      starter: py`
        def can_partition(nums: list[int]) -> bool:
            # TODO: 转化为「能否选出和为 sum/2 的子集」——0/1 背包可行性
            return False
      `,
      solution: py`
        def can_partition(nums: list[int]) -> bool:
            s = sum(nums)
            if s % 2:                       # 奇数和不可能平分
                return False
            target = s // 2
            dp = [False] * (target + 1)     # dp[j]: 能否选出和恰为 j 的子集
            dp[0] = True
            for x in nums:
                for j in range(target, x - 1, -1):   # 0/1：倒序
                    if dp[j - x]:
                        dp[j] = True
                if dp[target]:              # 提前结束
                    return True
            return dp[target]
      `,
      solutionAlt: [
        {
          label: '位运算版（Python 大整数当 bitset）',
          code: py`
            def can_partition(nums):
                s = sum(nums)
                if s % 2:
                    return False
                bits = 1                       # 第 j 位为 1 表示和 j 可达
                for x in nums:
                    bits |= bits << x          # 整体左移 = 所有已达和都加上 x
                return bool(bits >> (s // 2) & 1)
          `,
        },
      ],
      explanation: md`
        **转化**：两个子集和相等 ⇔ 存在子集和为 $s/2$。$s$ 为奇数直接 False。

        **0/1 背包可行性**：\`dp[j]\` 表示能否凑出 $j$，\`dp[0] = True\`；每个数只能用一次，容量倒序：\`dp[j] |= dp[j - x]\`。

        **复杂度**：时间 $O(n \cdot s/2)$，空间 $O(s)$。本题最大 $200 \times 10^4 = 2 \times 10^6$ 次操作。

        **位运算优化**：用一个大整数的二进制位表示所有可达的和，\`bits |= bits << x\` 一次完成整行更新，常数小几十倍（C++ 中对应 \`std::bitset\`）。

        **常见错误**：正序遍历（数字被重复使用）；忘记判奇数；用 \`set\` 存所有可达和时没有剪枝到 \`target\` 以内，内存暴涨。
      `,
      complexity: { time: 'O(n · sum)', space: 'O(sum)' },
      hints: [
        md`两个子集和相等，等价于什么？总和是奇数时能分吗？`,
        md`问题变成：能否从 \`nums\` 中选出若干个数，和恰为 \`sum // 2\`？每个数最多用一次。`,
        md`\`dp = [True] + [False] * target\`，对每个 \`x\`：\`for j in range(target, x - 1, -1): dp[j] = dp[j] or dp[j - x]\`。`,
      ],
      setup: py`
        import random
        def _ref(nums):
            s = sum(nums)
            if s % 2:
                return False
            bits = 1
            for x in nums:
                bits |= bits << x
            return bool(bits >> (s // 2) & 1)
      `,
      tests: [
        { name: '示例', code: py`
          assert can_partition([1, 5, 11, 5]) is True, "[1,5,11,5] 应为 True"
          assert can_partition([1, 2, 3, 5]) is False, "[1,2,3,5] 应为 False"
        ` },
        { name: '每个数只能用一次', code: py`
          got = can_partition([2, 2, 3, 5])
          assert got is False, f"[2,2,3,5] 应为 False（目标 6 不能用两个 3 凑），得到 {got}"
          assert can_partition([1]) is False
          assert can_partition([3, 3]) is True
        ` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(416)
            for _ in range(300):
                a = [rnd.randint(1, 12) for _ in range(rnd.randint(1, 12))]
                got, exp = can_partition(a), _ref(a)
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：n = 200（暴力枚举会超时）',
          hidden: true,
          code: py`
            rnd = random.Random(200)
            for _ in range(2):
                a = [rnd.randint(1, 100) for _ in range(200)]
                if sum(a) % 2:
                    a[0] += 1
                got, exp = can_partition(a), _ref(a)
                assert got == exp, f"期望 {exp}，得到 {got}"
            a = [100] * 199 + [2]          # 和 19902 为偶数，但凑不出 9951
            got = can_partition(a)
            assert got == _ref(a), f"[100]*199+[2]：期望 {_ref(a)}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-min-diff',
          q: md`如果要求两个子集和的**差最小**（LC 1049 最后一块石头的重量 II），怎么改？`,
          a: md`同样求出所有 $\le s/2$ 的可达和，取最大的可达 $j$，答案是 $s - 2j$。DP 完全相同，只是返回值不同。`,
        },
        {
          id: 'f-k',
          q: md`如果要分成 $k$ 个等和子集（LC 698）呢？`,
          a: md`背包不再适用（需要同时追踪多个桶），通常用回溯 + 剪枝（排序后从大到小放、相同桶状态去重），或状态压缩 DP：\`dp[mask]\` 表示已用元素集合 mask 时当前桶的余量，$O(n 2^n)$。`,
        },
      ],
    },
    {
      id: 'c-coin-change',
      title: '零钱兑换',
      difficulty: 'medium',
      source: 'LeetCode 322',
      timeLimitMin: 15,
      description: md`
        给定不同面值的硬币 \`coins\`（每种数量无限）和总金额 \`amount\`，返回凑成总金额所需的**最少**硬币个数；无法凑成返回 \`-1\`。

        \`\`\`python
        def coin_change(coins: list[int], amount: int) -> int: ...
        \`\`\`

        **示例**
        - \`coin_change([1, 2, 5], 11)\` → \`3\`
        - \`coin_change([2], 3)\` → \`-1\`
        - \`coin_change([1], 0)\` → \`0\`

        **约束**：$1 \le |coins| \le 12$，$1 \le coins[i] \le 2^{31}-1$，$0 \le amount \le 10^4$。注意：贪心不正确。
      `,
      starter: py`
        def coin_change(coins: list[int], amount: int) -> int:
            # TODO: dp[j] = 恰好凑出 j 的最少硬币数
            raise NotImplementedError
      `,
      solution: py`
        def coin_change(coins: list[int], amount: int) -> int:
            INF = amount + 1                  # 任何可行解都不超过 amount 枚，用它当「无穷大」
            dp = [0] + [INF] * amount         # 恰好凑出：不可达为 INF
            for c in coins:
                for j in range(c, amount + 1):    # 完全背包：正序
                    if dp[j - c] + 1 < dp[j]:
                        dp[j] = dp[j - c] + 1
            return dp[amount] if dp[amount] <= amount else -1
      `,
      explanation: md`
        **建模**：完全背包 + 恰好装满 + 求最小值。\`dp[j]\` = 恰好凑出 $j$ 的最少硬币数，$dp[0] = 0$，其它初始化为“无穷大”。

        **转移**：$dp[j] = \min(dp[j],\ dp[j - c] + 1)$，硬币可重复使用，金额正序。

        **无穷大的取值**：最少硬币数不会超过 \`amount\`（全用面值 1 也只要 \`amount\` 枚），所以用 \`amount + 1\` 当无穷大，避免 \`float('inf')\` 与整数混用；也可以直接用 \`float('inf')\`。

        **复杂度**：时间 $O(|coins| \cdot amount)$，空间 $O(amount)$。面值大于 \`amount\` 的硬币内层循环不执行，无需特判。

        **常见错误**：贪心；\`dp\` 初始化为 0；返回前没把“无穷大”转成 -1；BFS 也可以做（每层 +1 枚硬币），但常数更大。
      `,
      complexity: { time: 'O(|coins| · amount)', space: 'O(amount)' },
      hints: [
        md`贪心反例：\`coins = [1, 3, 4]\`，\`amount = 6\`。需要枚举“最后一枚硬币是哪种”。`,
        md`\`dp[j]\` = 恰好凑出 $j$ 的最少硬币数，$dp[0] = 0$，其它是无穷大。`,
        md`\`for c in coins: for j in range(c, amount + 1): dp[j] = min(dp[j], dp[j - c] + 1)\`，最后无穷大返回 -1。`,
      ],
      setup: py`
        import random
        from collections import deque
        def _ref_bfs(coins, amount):
            if amount == 0:
                return 0
            seen = {0}
            q = deque([(0, 0)])
            while q:
                s, d = q.popleft()
                for c in coins:
                    t = s + c
                    if t == amount:
                        return d + 1
                    if t < amount and t not in seen:
                        seen.add(t)
                        q.append((t, d + 1))
            return -1
      `,
      tests: [
        { name: '示例', code: py`
          for coins, amt, exp in [([1, 2, 5], 11, 3), ([2], 3, -1), ([1], 0, 0)]:
              got = coin_change(coins, amt)
              assert got == exp, f"coins={coins}, amount={amt}：期望 {exp}，得到 {got}"
        ` },
        { name: '贪心反例', code: py`
          got = coin_change([1, 3, 4], 6)
          assert got == 2, f"coins=[1,3,4], amount=6：期望 2（3+3），得到 {got}"
        ` },
        {
          name: '无解与超大面值',
          hidden: true,
          code: py`
            assert coin_change([3, 7], 11) == -1, "3 和 7 凑不出 11"
            assert coin_change([2 ** 31 - 1], 5) == -1
            assert coin_change([5, 2 ** 31 - 1], 10) == 2
          `,
        },
        {
          name: '随机对拍（BFS）',
          hidden: true,
          code: py`
            rnd = random.Random(322)
            for _ in range(200):
                coins = rnd.sample(range(1, 30), rnd.randint(1, 5))
                amt = rnd.randint(0, 120)
                got, exp = coin_change(coins, amt), _ref_bfs(coins, amt)
                assert got == exp, f"coins={coins}, amount={amt}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：amount = 10^4',
          hidden: true,
          code: py`
            coins = [186, 419, 83, 408]
            got = coin_change(coins, 6249)
            assert got == 20, f"LC 经典用例：期望 20，得到 {got}"
            got = coin_change([7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47], 10_000)
            assert got == _ref_bfs([7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47], 10_000), "amount=10^4 结果错误"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-ways',
          q: md`如果改成求凑出金额的**组合数**（LC 518），代码要改哪些地方？`,
          a: md`初始化 \`dp = [1] + [0] * amount\`，转移改为 \`dp[j] += dp[j - c]\`，并且**必须**外层遍历硬币、内层遍历金额（否则数的是排列）。返回 \`dp[amount]\`，无需 -1。`,
        },
        {
          id: 'f-limited',
          q: md`如果每种硬币数量有限（第 $i$ 种最多 $k_i$ 枚，多重背包）呢？`,
          a: md`朴素做法把每种硬币拆成 $k_i$ 个 0/1 物品，$O(amount \cdot \sum k_i)$。常用**二进制拆分**：拆成 $1, 2, 4, \dots$ 和余数几组，每组当一个 0/1 物品，复杂度降到 $O(amount \cdot \sum \log k_i)$；再进一步可用单调队列优化到 $O(amount \cdot |coins|)$。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-dir',
      freq: 3,
      q: md`0/1 背包和完全背包的一维写法有什么区别？为什么？`,
      a: md`
        **结论**：0/1 背包容量**倒序**，完全背包容量**正序**，其它完全相同。

        **原因**：一维数组原地覆盖，\`dp[j]\` 依赖更小下标 \`dp[j - w]\`。倒序时 \`dp[j - w]\` 还没被本轮更新，是“没考虑当前物品”的旧值，对应二维的 $dp[i-1][j-w]$，物品只用一次；正序时它已被本轮更新，可能已经包含当前物品，对应 $dp[i][j-w]$，正是完全背包“选了还能再选”的转移。

        **延伸**：多重背包用二进制拆分转成 0/1 背包；二维费用背包（LC 474 一和零）两个容量维都要倒序。
      `,
    },
    {
      id: 'qa-cp',
      freq: 3,
      q: md`零钱兑换 II（组合数）和组合总和 IV（排列数）为什么只是循环顺序不同？`,
      a: md`
        **结论**：外层物品、内层容量 → 组合；外层容量、内层物品 → 排列。

        **原因**：外层遍历物品时，所有方案中的物品都按物品编号顺序被“加入”，同一个多重集只对应一种加入顺序，只计一次；外层遍历容量时，\`dp[j]\` 枚举“最后一个放的是谁”，最后一个不同的序列分开计数，所有排列都被计入。

        **延伸**：最值问题（LC 322）两种顺序结果相同，因为最优值与顺序无关；排列数问题本质就是“爬楼梯”的推广。
      `,
    },
    {
      id: 'qa-init',
      freq: 2,
      q: md`背包问题中“容量不超过 C”与“恰好装满 C”在实现上有什么区别？`,
      a: md`
        **结论**：只差初始化。

        **原因**：“不超过”时 $dp[0..C]$ 全部初始化为 0——任何容量都可以什么都不装；“恰好装满”时只有 $dp[0] = 0$ 合法，其余为 $-\infty$（求最大）或 $+\infty$（求最小），不可达状态在转移中自动被淘汰，最后若 $dp[C]$ 仍是无穷则无解。

        **延伸**：计数问题 $dp[0] = 1$；可行性问题 $dp[0] = $ True。初始化本质上是在定义“空方案”的值。
      `,
    },
    {
      id: 'qa-np',
      freq: 1,
      q: md`背包 DP 的复杂度 $O(nC)$ 是多项式的吗？$C$ 非常大时怎么办？`,
      a: md`
        **结论**：不是，是伪多项式；0/1 背包判定版是 NP 完全问题。

        **原因**：输入中 $C$ 只占 $O(\log C)$ 位，$O(nC)$ 关于输入长度是指数级。

        **应对**：价值总和小 → 按价值 DP（\`dp[v]\` = 达到价值 $v$ 的最小重量）；$n \le 40$ → 折半枚举 $O(2^{n/2} \cdot n)$；工程上要近似解时用贪心（按性价比）或 FPTAS（对价值缩放取整，$(1-\varepsilon)$ 近似）。
      `,
    },
  ],

  cheatsheet: md`
    \`\`\`python
    # 0/1 背包：倒序           # 完全背包：正序
    for w, v in items:         for w, v in items:
        for j in range(C, w-1, -1):  for j in range(w, C+1):
            dp[j] = max(dp[j], dp[j-w] + v)
    # 518 组合数：外 coins 内 j；377 排列数：外 j 内 coins；dp[0] = 1
    # 416：bits |= bits << x
    \`\`\`

    | 目标 | 初始化 | 合并 |
    |---|---|---|
    | 不超过 C 的最大值 | 全 0 | max |
    | 恰好装满最大值 | dp[0]=0，其余 -inf | max |
    | 最少个数（322） | dp[0]=0，其余 inf | min(+1) |
    | 方案数（518/377） | dp[0]=1，其余 0 | += |
    | 可行性（416） | dp[0]=True | or |

    - 0/1 倒序读旧值，完全正序读新值
    - 计数：外物品 = 组合，外容量 = 排列
    - $O(nC)$ 是伪多项式
  `,
})
