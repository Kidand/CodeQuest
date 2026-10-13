/**
 * algo-dp-intro：DP 四步法（状态 → 转移 → 初始化 → 遍历顺序）
 */
import { defineLevel, md, py } from '@/content/helpers'
import { RecursionTreeViz } from '../viz/RecursionTreeViz'
import { RobberViz } from '../viz/RobberViz'
import { GridPathViz } from '../viz/GridPathViz'
import { DecodeWaysViz } from '../viz/DecodeWaysViz'

export default defineLevel({
  learn: [
    {
      id: 'why',
      title: '从递归到 DP：重叠子问题让递归树指数爆炸',
      body: md`
        爬楼梯（LC 70）：每次爬 1 或 2 阶，爬到第 $n$ 阶有几种走法？最后一步要么从 $n-1$ 跨 1 阶上来，要么从 $n-2$ 跨 2 阶上来，两类互不重叠，所以

        $$
        f(n) = f(n-1) + f(n-2),\qquad f(0) = f(1) = 1
        $$

        直接写成递归是对的，但**慢得离谱**：$f(n-1)$ 内部又会算一遍 $f(n-2)$，同一个子问题在递归树里出现了很多次。
        调用次数 $C(n) = C(n-1) + C(n-2) + 1$，与斐波那契数同阶，是 $O(\varphi^n)$（$\varphi \approx 1.618$）。

        DP 能派上用场，需要两个条件：

        | 条件 | 含义 | 爬楼梯中的体现 |
        |---|---|---|
        | **重叠子问题** | 不同分支会反复求解同一个子问题 | $f(3)$ 在 $f(5)$ 的递归树里出现 2 次 |
        | **最优子结构 / 无后效性** | 大问题的解只依赖子问题的**解**，不依赖子问题**是怎么得到的** | $f(n)$ 只关心 $f(n-1)$ 的数值，不关心那些走法具体长什么样 |

        把每个子问题的结果**存起来**（记忆化），每个 $f(k)$ 只算一次，复杂度立刻降为 $O(n)$。下面切换两种模式，看递归树怎样从一棵大树变成一条链。
      `,
      viz: RecursionTreeViz,
      after: md`
        > [!TIP] 读图的关键
        > 朴素递归里**同色节点**就是被重复计算的同一个子问题；记忆化后，绿色的「命中」节点直接返回，整棵子树被剪掉。
        > 记忆化的总调用次数是 $2n-1$：每个 $f(k)$ 真正计算一次，外加至多一次命中。

        > [!NOTE] 没有重叠子问题时
        > 归并排序、快速排序的递归子问题互不重叠（左右两半是不同的数组段），这时加缓存毫无收益——它们属于**分治**，不是 DP。
      `,
      checkpoint: {
        id: 'cp-why',
        type: 'numeric',
        prompt: md`朴素递归 \`f(n) = f(n-1) + f(n-2)\`，边界 \`n <= 1\` 时直接返回。计算 $f(6)$ 时，函数 \`f\` 一共被调用多少次（含最外层这一次）？`,
        answer: 25,
        tolerance: 0,
        explanation: md`$C(0)=C(1)=1$，$C(n) = C(n-1)+C(n-2)+1$：$C(2)=3, C(3)=5, C(4)=9, C(5)=15, C(6)=25$。记忆化版本只需 $2\cdot 6-1 = 11$ 次调用。`,
      },
    },
    {
      id: 'four-steps',
      title: 'DP 四步法：状态 → 转移 → 初始化 → 遍历顺序',
      body: md`
        以**打家劫舍**（LC 198）为例：一排房子，相邻两间不能同时偷，求最大金额。所有 DP 题都按同一套流程拆解：

        1. **状态定义**：\`dp[i]\` = 只考虑**前 $i$ 间**房子时能偷到的最大金额。状态定义要能“装下”答案，并且能从更小的状态推出来。
        2. **转移方程**：对第 $i$ 间房做“偷 / 不偷”二选一——
           $$
           dp[i] = \max\big(\underbrace{dp[i-1]}_{\text{不偷第 } i \text{ 间}},\ \underbrace{dp[i-2] + nums[i-1]}_{\text{偷第 } i \text{ 间，则第 } i-1 \text{ 间不能偷}}\big)
           $$
        3. **初始化 / 边界**：\`dp[0] = 0\`（没有房子），\`dp[1] = nums[0]\`。
        4. **遍历顺序**：\`dp[i]\` 依赖 \`dp[i-1]\` 与 \`dp[i-2]\`，所以 $i$ **从小到大**。
        5. **返回值**：\`dp[n]\`。

        多开一格（长度 $n+1$、\`dp[0]\` 表示空前缀）是最常用的技巧：边界变得统一，不用对 $i-2<0$ 特判。
      `,
      viz: RobberViz,
      after: md`
        切到「两个变量」模式：因为 \`dp[i]\` 只依赖前两格，变暗的格子算完就再也不会被读——这就是**滚动数组 / 状态压缩**，空间从 $O(n)$ 降到 $O(1)$。

        > [!WARN] 滚动之后就不能回溯了
        > 最后一帧的回溯（判断 \`dp[i] != dp[i-1]\`）需要完整的 dp 表。面试官要求“输出偷了哪些房子”时，要么保留整张表，要么额外记录选择。

        > [!INTERVIEW] 环形变体（LC 213，本关实战）
        > 房子围成一圈，首尾相邻。任何合法方案都**不会同时**包含首尾两间，所以答案 = \`max(rob(nums[1:]), rob(nums[:-1]))\`，注意 $n=1$ 时要特判。
      `,
      checkpoint: {
        id: 'cp-four',
        type: 'single',
        prompt: md`打家劫舍中，如果把状态定义改为「\`dp[i]\` = **一定偷**第 $i$ 间时，前 $i$ 间的最大金额」，最终答案应该返回：`,
        options: ['`dp[n]`', '`max(dp[n], dp[n-1])`', '`max(dp)`（整张表的最大值）', '`dp[n] + dp[n-1]`'],
        answer: 1,
        explanation: md`“一定偷第 $i$ 间”的定义下，最优方案的最后一间被偷的房子只能是第 $n$ 或第 $n-1$ 间（若最后偷的是第 $n-2$ 间及之前，还可以再偷第 $n$ 间，金额不会更少）。所以返回 \`max(dp[n], dp[n-1])\`；\`max(dp)\` 也对但多扫一遍，而 A 漏掉了不偷最后一间的情况。**状态定义变了，返回值也要跟着变**——这正是面试中最容易出错的地方。`,
      },
    },
    {
      id: 'grid',
      title: '二维 DP 与遍历顺序：依赖必须先算好',
      body: md`
        **最小路径和**（LC 64）：$m \times n$ 网格，只能向右或向下，求左上到右下的最小路径和。

        - 状态：\`dp[i][j]\` = 从起点走到 $(i, j)$ 的最小路径和
        - 转移：最后一步来自上方或左方，$dp[i][j] = \min(dp[i-1][j],\ dp[i][j-1]) + grid[i][j]$
        - 初始化：第一行只能从左边来，第一列只能从上边来（**边界单独处理**）
        - 遍历顺序：依赖的是“上”和“左”，所以**行从上到下、列从左到右**

        **不同路径**（LC 62）结构完全相同，只是把 $\min(\cdot)+grid$ 换成加法：$dp[i][j] = dp[i-1][j] + dp[i][j-1]$，边界全为 1。

        遍历顺序不是背出来的，而是由**依赖方向**推出来的。下面把顺序换成「自下而上」，看看会发生什么。
      `,
      viz: GridPathViz,
      after: md`
        自下而上时，计算最后一行要读“上一行”，但上一行还没算，读到的是初始值 0，错误会一路传染（红色）。

        > [!TIP] 顺序的通用判据
        > 把每个状态看成有向图的节点、依赖关系看成边——**遍历顺序就是这张 DAG 的一个拓扑序**。行优先恰好满足“上、左先于自己”。
        > 如果你把状态定义成「从 $(i,j)$ **到终点**的最小和」，依赖就变成“下、右”，此时**倒序**遍历才是对的。

        一维滚动版：\`dp[j] = min(dp[j], dp[j-1]) + grid[i][j]\`——更新前的 \`dp[j]\` 存的是**上一行**（上方），\`dp[j-1]\` 已经是**本行**（左方），所以 $j$ 必须正序。
      `,
      checkpoint: {
        id: 'cp-grid',
        type: 'judge',
        prompt: md`对 LC 64，如果定义 \`dp[i][j]\` 为「从 $(i,j)$ 走到右下角的最小路径和」，那么必须从右下角开始、行和列都**倒序**遍历，返回 \`dp[0][0]\`。`,
        answer: true,
        explanation: md`正确。此时 $dp[i][j] = \min(dp[i+1][j],\ dp[i][j+1]) + grid[i][j]$，依赖“下”和“右”，只有倒序才能保证依赖先算好。顺序没有绝对的对错，**只看是否与依赖方向一致**。`,
      },
    },
    {
      id: 'memo-vs-tab',
      title: '记忆化搜索 vs 自底向上递推',
      body: md`
        同一个转移方程有两种实现方式：

        | | 记忆化搜索（自顶向下） | 递推填表（自底向上） |
        |---|---|---|
        | 写法 | 递归 + 缓存（\`@functools.cache\`） | 循环按拓扑序填表 |
        | 计算的状态 | **只算用得到的**状态 | 通常把整张表都填满 |
        | 遍历顺序 | 递归自动保证，不用想 | 必须自己推导 |
        | 空间优化 | 难（缓存要一直留着） | 容易做滚动数组 |
        | 风险 | Python 递归深度默认 1000，$n$ 大时 \`RecursionError\` | 无 |
        | 适合 | 状态空间稀疏、顺序不好推（如树形、区间、数位 DP） | 状态稠密、要压缩空间 |

        **复杂度估算公式**：时间 = 状态数 × 每个状态的转移代价；空间 = 需要同时保存的状态数。
        打家劫舍：$n$ 个状态 × $O(1)$ 转移 = $O(n)$；LIS 的 $O(n^2)$ 解：$n$ 个状态 × $O(n)$ 转移。

        > [!INTERVIEW] 面试推荐的说法
        > “我先写记忆化验证状态和转移是对的，再改写成递推，最后看依赖范围能不能滚动压缩。”——体现从正确到高效的完整思路。
      `,
      checkpoint: {
        id: 'cp-memo',
        type: 'fill',
        prompt: md`
          一个 DP 有 $n \times m$ 个状态，每个状态要枚举 $k$ 个候选转移。时间复杂度是 ____(1)____；若 \`dp[i][*]\` 只依赖 \`dp[i-1][*]\`，滚动后空间复杂度是 ____(2)____。（用 \`O(...)\` 形式作答）
        `,
        blanks: [
          { label: '(1)', answers: ['O(nmk)', 'O(n*m*k)', 'O(nkm)', 'O(mnk)', 'O(k*n*m)', 'O(n m k)'] },
          { label: '(2)', answers: ['O(m)', 'O(2m)'] },
        ],
        explanation: md`时间 = 状态数 $nm$ × 转移代价 $k$ = $O(nmk)$；只保留上一行（或两行）时，空间为 $O(m)$。`,
      },
    },
    {
      id: 'decode',
      title: '带条件的转移：解码方法',
      body: md`
        **解码方法**（LC 91）：\`'A'→"1"\`、…、\`'Z'→"26"\`，给一个数字串，问有多少种解码方式。

        - 状态：\`dp[i]\` = 前 $i$ 个字符的解码方式数
        - 转移：看**最后一段**是 1 个字符还是 2 个字符——
          $$
          dp[i] = \underbrace{[s_{i-1} \ne 0]\cdot dp[i-1]}_{\text{最后一段 1 位}} + \underbrace{[10 \le \overline{s_{i-2}s_{i-1}} \le 26]\cdot dp[i-2]}_{\text{最后一段 2 位}}
          $$
        - 初始化：\`dp[0] = 1\`（空前缀 1 种），它是**乘法单位元**，不是“空串的答案”
        - 计数类 DP 的核心是“**按最后一步分类，不重不漏**”：最后一段长度不同，两类方案必然不同。

        \`'0'\` 是全部难点：它不能单独解码，只能作为 \`"10"\` 或 \`"20"\` 的后半；\`"06"\` 这种前导 0 也不合法。
      `,
      viz: DecodeWaysViz,
      after: md`
        试试 \`"2101"\`：第 3 位的 \`'0'\` 必须和前面的 \`'1'\` 组成 \`"10"\`，于是第 2 位的 \`'1'\` 被“锁死”，答案只有 1。
        \`"06"\` 则在第 1 步就得到 \`dp[1] = 0\`，之后再也恢复不了。

        > [!WARN] 常见错误
        > - 用 \`int(s[i-2:i]) <= 26\` 判断两位数，漏掉了 \`"06"\` 这种前导 0（\`int("06") = 6\`）；
        > - 把 \`dp[0]\` 设成 0，导致所有结果都是 0；
        > - 两个 \`if\` 写成 \`if / elif\`——两类方案要**相加**，不是二选一。
      `,
      checkpoint: {
        id: 'cp-decode',
        type: 'numeric',
        prompt: md`\`num_decodings("10011")\` 的结果是多少？`,
        answer: 0,
        tolerance: 0,
        explanation: md`$dp_0=1$；\`'1'\` → $dp_1=1$；\`'0'\` 不能单独解码，\`"10"\` 合法 → $dp_2 = dp_0 = 1$；第 3 位 \`'0'\` 不能单独解码，\`"00"\` 也不合法 → $dp_3 = 0$，之后全为 0。连续两个 0 的串一定无法解码。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-greedy',
      type: 'fill',
      difficulty: 1,
      prompt: md`硬币面值 \`[1, 3, 4]\`，凑出金额 6。贪心（每次拿不超过剩余金额的最大面值）用 ____(1)____ 枚硬币；最优解只需 ____(2)____ 枚。`,
      blanks: [
        { label: '(1)', answers: ['3'], numeric: true },
        { label: '(2)', answers: ['2'], numeric: true },
      ],
      explanation: md`贪心：$4 + 1 + 1$，3 枚；最优：$3 + 3$，2 枚。贪心做了“局部最优”的选择后无法反悔，而 DP 会枚举“最后一枚是 1 / 3 / 4”的全部情况再取最小。这是“贪心不一定对、需要 DP”的经典反例（下一关的零钱兑换）。`,
    },
    {
      id: 'q-memo-cplx',
      type: 'judge',
      difficulty: 1,
      prompt: md`对同一个转移方程，记忆化搜索与自底向上递推的时间复杂度在渐近意义上相同（都等于“被计算的状态数 × 每个状态的转移代价”）。`,
      answer: true,
      explanation: md`正确。两者计算的是同一组状态（记忆化可能少算一些用不到的状态，只会更少），每个状态只计算一次。区别在常数（递归调用开销）、递归深度限制、能否滚动压缩空间。`,
    },
    {
      id: 'q-stairs',
      type: 'numeric',
      difficulty: 1,
      prompt: md`爬楼梯：每次可以爬 1 或 2 阶，爬到第 10 阶共有多少种方法？（$f(0)=f(1)=1$）`,
      answer: 89,
      tolerance: 0,
      explanation: md`$f$ 依次为 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89，$f(10) = 89$。注意 LC 70 的 $f(1)=1, f(2)=2$ 与这里的定义一致。`,
    },
    {
      id: 'q-paths',
      type: 'single',
      difficulty: 1,
      prompt: md`LC 62：$m = 3$ 行、$n = 7$ 列的网格，从左上走到右下（只能向右或向下），共有多少条不同路径？`,
      options: ['21', '28', '36', '84'],
      answer: 1,
      explanation: md`共走 $(m-1)+(n-1) = 8$ 步，其中选 2 步向下：$\binom{8}{2} = 28$。DP 表的结果相同：第一行、第一列全为 1，$dp[i][j] = dp[i-1][j] + dp[i][j-1]$。21 是 $\binom{7}{2}$（少算了一步），84 是 $\binom{9}{3}$。`,
    },
    {
      id: 'q-rolling',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于滚动数组（状态压缩），下列说法正确的是：`,
      options: [
        '打家劫舍可以只用两个变量 `prev2, prev1`',
        'LC 64 可以压成一维：`dp[j] = min(dp[j], dp[j-1]) + grid[i][j]`',
        '滚动压缩后，仍然可以直接回溯出最优路径',
        '在上述一维写法里，`dp[j]` 更新前保存的是上一行同一列的值',
        '任何 DP 都可以滚动到 O(1) 空间',
      ],
      answer: [0, 1, 3],
      explanation: md`A、B、D 正确：一维写法中更新前的 \`dp[j]\` 是“上方”，\`dp[j-1]\` 已是本行的“左方”。C 错：被覆盖的历史值没了，回溯需要整张表或额外记录选择。E 错：能压到多小取决于依赖跨度，例如编辑距离依赖上一整行，最少要 $O(n)$。`,
    },
    {
      id: 'q-steps',
      type: 'order',
      difficulty: 2,
      prompt: md`把设计一个 DP 解法的思考步骤按正确顺序排列：`,
      items: [
        '定义状态：`dp[i]` 表示什么（要能装下答案）',
        '推导转移方程：按「最后一步」分类讨论',
        '确定初始化与边界（空前缀、第一行 / 第一列）',
        '根据依赖方向确定遍历顺序',
        '确定返回值，再考虑滚动数组优化空间',
      ],
      explanation: md`状态定义决定一切：转移依赖于定义，初始化依赖于转移中会读到的“最小状态”，遍历顺序依赖于转移的方向，返回值依赖于定义。顺序颠倒（比如先想循环怎么写）是 DP 写不出来的最常见原因。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与其转移方程配对：`,
      pairs: [
        ['LC 70 爬楼梯', '`f[i] = f[i-1] + f[i-2]`'],
        ['LC 198 打家劫舍', '`dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])`'],
        ['LC 62 不同路径', '`dp[i][j] = dp[i-1][j] + dp[i][j-1]`'],
        ['LC 64 最小路径和', '`dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]`'],
      ],
      explanation: md`计数类问题用加法（按最后一步分类，方案数相加）；最优化问题用 max / min。爬楼梯和不同路径都是计数，打家劫舍和最小路径和都是最优化。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`用**一维滚动数组**写出 LC 64 最小路径和（有 2 行干扰项）：`,
      lines: [
        'def min_path_sum(grid):',
        '    n = len(grid[0])',
        "    dp = [float('inf')] * n",
        '    dp[0] = 0',
        '    for row in grid:',
        '        dp[0] += row[0]',
        '        for j in range(1, n):',
        '            dp[j] = min(dp[j], dp[j - 1]) + row[j]',
        '    return dp[-1]',
      ],
      distractors: ['        for j in range(n - 1, 0, -1):', '    dp = [0] * n'],
      explanation: md`初始化为 \`inf\`（\`dp[0] = 0\`）使得第一行自然退化为“只能从左边来”。$j$ 必须**正序**：\`dp[j-1]\` 要先更新成本行的“左方”。倒序会让 \`dp[j-1]\` 仍是上一行的值，相当于从“左上角”转移，错误。全 0 初始化会让第一行取到 \`min(0, …)\`，同样错误。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面是 LC 213 打家劫舍 II（环形）的实现，有一处 bug，点出那一行：`,
      code: py`
        def rob(nums):
            def line(arr):
                prev2, prev1 = 0, 0
                for x in arr:
                    prev2, prev1 = prev1, max(prev1, prev2 + x)
                return prev1

            if len(nums) == 1:
                return nums[0]
            return max(line(nums[1:]), line(nums[:-2]))
      `,
      bugLines: [10],
      explanation: md`第 10 行应为 \`line(nums[:-1])\`（去掉最后一间）。\`nums[:-2]\` 把倒数第二间也错误地排除了，例如 \`[1, 2, 3, 1]\`：正确答案 4（偷 1 和 3），而 \`max(line([2,3,1]), line([1,2])) = 3\`。第 5 行的同时赋值是正确的滚动写法。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`打家劫舍 \`nums = [5, 1, 1, 5, 1, 6]\`，点出最优方案中被偷的全部房屋。`,
      items: ['5', '1', '1', '5', '1', '6'],
      answer: [0, 3, 5],
      explanation: md`dp（前 $i$ 间）依次为 0, 5, 5, 6, 10, 10, 16。回溯：$dp_6 \ne dp_5$ → 偷第 6 间（6）；跳到 $i=4$，$dp_4 \ne dp_3$ → 偷第 4 间（5）；跳到 $i=2$，$dp_2 = dp_1$ → 不偷；$i=1$ → 偷第 1 间（5）。合计 16。“隔一间偷一间”（5+1+1=7 或 1+5+6=12）都不是最优。`,
    },
    {
      id: 'q-dp0',
      type: 'single',
      difficulty: 3,
      prompt: md`在 LC 91 解码方法的标准写法中，如果把初始化 \`dp[0] = 1\` 改成 \`dp[0] = 0\`，\`num_decodings("12")\` 会返回：`,
      options: ['0', '1', '2', '抛出 IndexError'],
      answer: 0,
      explanation: md`$dp_1 = dp_0 = 0$，$dp_2 = dp_1 + dp_0 = 0$。所有方案都是从“空前缀”出发累乘（累加）过来的，\`dp[0] = 1\` 是计数的起点（乘法单位元）。这与“空串本身算不算一种解码”的语义争论无关——它只是让转移公式成立的边界值。`,
    },
    {
      id: 'q-circle',
      type: 'judge',
      difficulty: 3,
      prompt: md`环形打家劫舍中，\`line(nums[:-1])\` 计算的是「**一定偷**第一间房」时的最优金额。`,
      answer: false,
      explanation: md`错误。\`nums[:-1]\` 只是“**不偷最后一间**”，第一间可偷可不偷。正确性来自：任何合法方案不会同时偷首尾，所以它一定属于“不偷最后一间”或“不偷第一间”之一，两个子问题的并集覆盖了全部方案（有重叠也不影响取 max）。`,
    },
    {
      id: 'q-stairs3',
      type: 'numeric',
      difficulty: 3,
      prompt: md`爬楼梯变体：每次可以爬 1、2 或 3 阶，但**不能连续两次爬 2 阶**。爬到第 5 阶有多少种方法？（提示：需要给状态加一维“上一步是不是 2”。）`,
      answer: 11,
      tolerance: 0,
      explanation: md`
        设 $a_i$ = 到第 $i$ 阶且最后一步不是 2 的方法数，$b_i$ = 最后一步是 2 的方法数。
        $a_i = (a_{i-1}+b_{i-1}) + (a_{i-3}+b_{i-3})$，$b_i = a_{i-2}$（上一步不能是 2）。
        $a_0=1,b_0=0$；$a_1=1,b_1=0$；$a_2=1,b_2=1$；$a_3=(1+1)+1=3,b_3=1$；$a_4=(3+1)+1=5,b_4=1$；$a_5=(5+1)+(1+1)=8,b_5=a_3=3$。合计 $8+3=11$。
        没有限制时是 13，少掉的 2 种是 2+2+1 与 1+2+2。“**状态不够用就加一维**”是 DP 设计的通用手法（股票问题会大量使用）。
      `,
    },
  ],

  guided: [
    {
      id: 'g-min-path',
      title: '一步步写出 LC 64 最小路径和',
      intro: md`
        实现 \`min_path_sum(grid)\`：$m \times n$ 非负整数网格，从左上走到右下，每次只能向右或向下，返回路径上数字之和的最小值。
        按“初始化 → 边界 → 转移 → 返回”的顺序逐空填写。
      `,
      template: py`
        def min_path_sum(grid):
            m, n = len(grid), len(grid[0])
            dp = [[0] * n for _ in range(m)]
            dp[0][0] = {{1}}
            for j in range(1, n):            # 第一行
                dp[0][j] = {{2}}
            for i in range(1, m):            # 第一列
                dp[i][0] = {{3}}
            for i in range(1, m):
                for j in range(1, n):
                    dp[i][j] = {{4}}
            return {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`起点的最小路径和就是它自己的值。`,
          answer: 'grid[0][0]',
          explain: md`起点没有任何来源，直接等于 \`grid[0][0]\`。`,
        },
        {
          blank: 2,
          instruction: md`第一行只能从**左边**走过来。`,
          answer: 'dp[0][j - 1] + grid[0][j]',
          accept: ['dp[0][j-1] + grid[0][j]', 'grid[0][j] + dp[0][j - 1]', 'grid[0][j] + dp[0][j-1]'],
          explain: md`边界单独处理，是为了避免在通用转移里访问 \`dp[-1][j]\`——Python 的负下标**不会报错**，而是悄悄读到最后一行，这是很隐蔽的 bug。`,
        },
        {
          blank: 3,
          instruction: md`第一列只能从**上边**走过来。`,
          answer: 'dp[i - 1][0] + grid[i][0]',
          accept: ['dp[i-1][0] + grid[i][0]', 'grid[i][0] + dp[i - 1][0]', 'grid[i][0] + dp[i-1][0]'],
        },
        {
          blank: 4,
          instruction: md`通用转移：最后一步来自上方或左方，取较小者再加上当前格。`,
          answer: 'min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]',
          accept: ['min(dp[i-1][j], dp[i][j-1]) + grid[i][j]', 'grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])', 'grid[i][j] + min(dp[i-1][j], dp[i][j-1])'],
          explain: md`行优先遍历保证了 \`dp[i-1][j]\`（上一行）与 \`dp[i][j-1]\`（本行左侧）都已算好。`,
        },
        {
          blank: 5,
          instruction: md`返回终点的值。`,
          answer: 'dp[m - 1][n - 1]',
          accept: ['dp[m-1][n-1]', 'dp[-1][-1]'],
        },
      ],
      setup: py`
        import random
        def _ref_mps(grid):
            m, n = len(grid), len(grid[0])
            dp = [float('inf')] * n
            dp[0] = 0
            for row in grid:
                dp[0] += row[0]
                for j in range(1, n):
                    dp[j] = min(dp[j], dp[j - 1]) + row[j]
            return dp[-1]
      `,
      tests: [
        { name: 'LC 示例 1', code: py`assert min_path_sum([[1,3,1],[1,5,1],[4,2,1]]) == 7` },
        { name: '单行 / 单列', code: py`
          assert min_path_sum([[1, 2, 3]]) == 6
          assert min_path_sum([[1], [2], [3]]) == 6
          assert min_path_sum([[5]]) == 5
        ` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(64)
            for _ in range(200):
                m, n = rnd.randint(1, 6), rnd.randint(1, 6)
                g = [[rnd.randint(0, 9) for _ in range(n)] for _ in range(m)]
                got, exp = min_path_sum(g), _ref_mps(g)
                assert got == exp, f"grid={g}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        进一步优化：观察到每行只依赖上一行与本行左侧，可以压成一维 \`dp[j] = min(dp[j], dp[j-1]) + grid[i][j]\`（$j$ 正序），空间 $O(n)$。
        如果面试官要求输出路径，就保留二维表，从终点沿较小的来源倒推。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-rob-circle',
      title: '打家劫舍 II（环形）',
      difficulty: 'medium',
      source: 'LeetCode 213',
      timeLimitMin: 15,
      description: md`
        房屋**围成一圈**（第一间与最后一间相邻），\`nums[i]\` 是第 $i$ 间房的金额。相邻的两间不能同时偷，求能偷到的最大金额。

        \`\`\`python
        def rob(nums: list[int]) -> int: ...
        \`\`\`

        **示例**
        - \`rob([2, 3, 2])\` → \`3\`（不能同时偷首尾的 2）
        - \`rob([1, 2, 3, 1])\` → \`4\`
        - \`rob([1, 2, 3])\` → \`3\`

        **约束**：$1 \le n \le 10^4$（隐藏测试更大），$0 \le nums[i] \le 1000$。要求时间 $O(n)$。
      `,
      starter: py`
        def rob(nums: list[int]) -> int:
            # TODO: 环形 → 拆成两个线性问题
            return 0
      `,
      solution: py`
        def rob(nums: list[int]) -> int:
            def rob_line(lo: int, hi: int) -> int:
                """线性打家劫舍，只看 nums[lo:hi]，O(1) 空间"""
                prev2, prev1 = 0, 0          # dp[i-2], dp[i-1]
                for i in range(lo, hi):
                    prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
                return prev1

            n = len(nums)
            if n == 1:
                return nums[0]
            # 首尾不能同时偷：要么不偷最后一间，要么不偷第一间
            return max(rob_line(0, n - 1), rob_line(1, n))
      `,
      explanation: md`
        **思路**：任何合法方案都不会同时偷第一间和最后一间，因此它一定属于以下两类之一：
        1. 不偷最后一间 → 在 \`nums[0 : n-1]\` 上做线性打家劫舍；
        2. 不偷第一间 → 在 \`nums[1 : n]\` 上做线性打家劫舍。

        两类的并集覆盖全部合法方案（有交集不影响取 max），答案取两者较大值。

        **边界**：$n = 1$ 时两个区间都为空，会返回 0，必须特判返回 \`nums[0]\`。

        **复杂度**：时间 $O(n)$，空间 $O(1)$（传下标代替切片，避免复制）。

        **常见错误**：切片写成 \`nums[:-2]\`；忘记 $n=1$；以为要分“偷第一间 / 不偷第一间”两种情况并强制偷第一间，写得很复杂。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`环的麻烦只在于首尾相邻。能不能把它拆成两个**不成环**的问题？`,
        md`合法方案不会同时包含首尾——分别考虑“去掉最后一间”和“去掉第一间”。`,
        md`线性版本用两个变量滚动：\`prev2, prev1 = prev1, max(prev1, prev2 + x)\`；别忘了 $n = 1$ 的特判。`,
      ],
      setup: py`
        import random

        def _ref_brute(nums):
            n = len(nums)
            best = 0
            for mask in range(1 << n):
                ok = True
                for i in range(n):
                    if mask >> i & 1 and mask >> ((i + 1) % n) & 1 and n > 1:
                        ok = False
                        break
                if ok:
                    best = max(best, sum(nums[i] for i in range(n) if mask >> i & 1))
            return best

        def _ref_rob(nums):
            def line(a):
                p2 = p1 = 0
                for x in a:
                    p2, p1 = p1, max(p1, p2 + x)
                return p1
            return nums[0] if len(nums) == 1 else max(line(nums[:-1]), line(nums[1:]))
      `,
      tests: [
        { name: '示例', code: py`
          assert rob([2, 3, 2]) == 3, f"[2,3,2]：期望 3，得到 {rob([2, 3, 2])}"
          assert rob([1, 2, 3, 1]) == 4, f"[1,2,3,1]：期望 4，得到 {rob([1, 2, 3, 1])}"
          assert rob([1, 2, 3]) == 3, f"[1,2,3]：期望 3，得到 {rob([1, 2, 3])}"
        ` },
        { name: '单间与两间', code: py`
          assert rob([7]) == 7, f"[7]：期望 7，得到 {rob([7])}"
          assert rob([1, 9]) == 9, f"[1,9]：期望 9，得到 {rob([1, 9])}"
          assert rob([0]) == 0
        ` },
        {
          name: '随机对拍（暴力枚举）',
          hidden: true,
          code: py`
            rnd = random.Random(213)
            for _ in range(300):
                a = [rnd.randint(0, 20) for _ in range(rnd.randint(1, 11))]
                got, exp = rob(a), _ref_brute(a)
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：n = 10^5',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            a = [rnd.randint(0, 1000) for _ in range(100_000)]
            got, exp = rob(a), _ref_rob(a)
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-tree',
          q: md`如果房子组成一棵**二叉树**（LC 337），相连的父子不能同时偷，怎么做？`,
          a: md`树形 DP：后序遍历，每个节点返回二元组 \`(偷它, 不偷它)\`。\`偷 = val + 左.不偷 + 右.不偷\`，\`不偷 = max(左) + max(右)\`。时间 $O(n)$，这是“向上返回多个状态”的典型模板。`,
        },
        {
          id: 'f-path',
          q: md`如何同时输出偷了哪些房子？`,
          a: md`保留完整 dp 数组（不滚动），从后往前：若 \`dp[i] != dp[i-1]\` 则第 $i$ 间被偷、跳到 $i-2$，否则跳到 $i-1$。环形版本对两个子问题各做一次，取较优者的方案。`,
        },
      ],
    },
    {
      id: 'c-decode',
      title: '解码方法',
      difficulty: 'medium',
      source: 'LeetCode 91',
      timeLimitMin: 20,
      description: md`
        字母按 \`'A' → "1"\`、\`'B' → "2"\`、…、\`'Z' → "26"\` 编码。给定只含数字的非空字符串 \`s\`，返回解码方法的总数。
        \`"06"\` 不能解码为 \`'F'\`（前导 0 不合法）。

        \`\`\`python
        def num_decodings(s: str) -> int: ...
        \`\`\`

        **示例**
        - \`num_decodings("12")\` → \`2\`（"AB" 或 "L"）
        - \`num_decodings("226")\` → \`3\`（"BZ"、"VF"、"BBF"）
        - \`num_decodings("06")\` → \`0\`

        **约束**：$1 \le |s| \le 10^4$（隐藏测试会给很长的串，答案可能非常大——Python 整数不会溢出，直接返回即可）。要求时间 $O(n)$。
      `,
      starter: py`
        def num_decodings(s: str) -> int:
            # TODO: dp[i] = 前 i 个字符的解码方式数
            raise NotImplementedError
      `,
      solution: py`
        def num_decodings(s: str) -> int:
            # prev2 = dp[i-2], prev1 = dp[i-1]；dp[0] = 1 是空前缀
            prev2, prev1 = 0, 1
            for i in range(1, len(s) + 1):
                cur = 0
                if s[i - 1] != '0':                          # 最后一段 1 位
                    cur += prev1
                if i >= 2 and '10' <= s[i - 2:i] <= '26':    # 最后一段 2 位（字符串比较自动排除 "0x"）
                    cur += prev2
                prev2, prev1 = prev1, cur
            return prev1
      `,
      explanation: md`
        **状态**：\`dp[i]\` = 前 $i$ 个字符的解码方式数，\`dp[0] = 1\`。

        **转移**（按最后一段的长度分类）：
        - \`s[i-1] != '0'\` 时，最后一段是 1 位，加上 \`dp[i-1]\`；
        - \`"10" <= s[i-2:i] <= "26"\` 时，最后一段是 2 位，加上 \`dp[i-2]\`。

        两位数用**字符串比较**：长度相同的数字串，字典序就是数值序，而且 \`"06" < "10"\` 自然排除了前导 0。用 \`int()\` 的话要额外判断 \`s[i-2] != '0'\`。

        **复杂度**：时间 $O(n)$；只依赖前两项，空间 $O(1)$。

        **常见错误**：\`dp[0] = 0\`；两个 \`if\` 写成 \`elif\`；朴素递归不加缓存（隐藏测试会超时）。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`按**最后一段**解码的长度分类：1 位还是 2 位？两类方案互不重叠，数量相加。`,
        md`1 位合法的条件：不是 \`'0'\`；2 位合法的条件：在 \`"10"\` 到 \`"26"\` 之间（注意 \`"06"\` 不合法）。`,
        md`\`dp[0] = 1\`，\`dp[i] = [s[i-1]≠'0']·dp[i-1] + [10≤s[i-2:i]≤26]·dp[i-2]\`，只需要两个滚动变量。`,
      ],
      setup: py`
        import random
        from functools import lru_cache

        def _ref_brute(s):
            @lru_cache(None)
            def go(i):
                if i == len(s):
                    return 1
                if s[i] == '0':
                    return 0
                r = go(i + 1)
                if i + 1 < len(s) and 10 <= int(s[i:i + 2]) <= 26:
                    r += go(i + 2)
                return r
            return go(0)

        def _ref_iter(s):
            p2, p1 = 0, 1
            for i in range(1, len(s) + 1):
                c = (p1 if s[i - 1] != '0' else 0) + (p2 if i >= 2 and '10' <= s[i - 2:i] <= '26' else 0)
                p2, p1 = p1, c
            return p1
      `,
      tests: [
        { name: '示例', code: py`
          for s, exp in [("12", 2), ("226", 3), ("06", 0)]:
              got = num_decodings(s)
              assert got == exp, f"s={s!r}：期望 {exp}，得到 {got}"
        ` },
        { name: '与 0 有关的边界', code: py`
          for s, exp in [("0", 0), ("10", 1), ("100", 0), ("2101", 1), ("27", 1), ("1010", 1), ("30", 0)]:
              got = num_decodings(s)
              assert got == exp, f"s={s!r}：期望 {exp}，得到 {got}"
        ` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(91)
            for _ in range(500):
                s = ''.join(rnd.choice('0112223456789') for _ in range(rnd.randint(1, 14)))
                got, exp = num_decodings(s), _ref_brute(s)
                assert got == exp, f"s={s!r}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：长度 10^4（朴素递归会超时）',
          hidden: true,
          code: py`
            rnd = random.Random(2026)
            s = ''.join(rnd.choice('1122') for _ in range(10_000))
            got, exp = num_decodings(s), _ref_iter(s)
            assert got == exp, f"长串结果不一致（期望值有 {len(str(exp))} 位数）"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-star',
          q: md`如果字符串里还有 \`'*'\`，可以代表 1–9 中任意一个数字（LC 639），转移怎么改？`,
          a: md`转移结构不变，只是系数变了：单字符 \`'*'\` 贡献 $9 \cdot dp[i-1]$；两字符要分类，例如 \`"1*"\` 有 9 种（11–19）、\`"2*"\` 有 6 种（21–26）、\`"**"\` 有 15 种、\`"*d"\` 在 $d \le 6$ 时 2 种否则 1 种。结果对 $10^9+7$ 取模。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-when',
      freq: 3,
      q: md`什么样的问题适合用动态规划？它和贪心、分治有什么区别？`,
      a: md`
        **结论**：满足“最优子结构 + 重叠子问题”的问题适合 DP；典型信号是“求最值 / 方案数 / 可行性”，且决策是一步步做出的。

        **原因**：
        - 最优子结构（无后效性）保证了大问题的最优解能由子问题的最优**值**组合而来，不需要知道子问题的具体方案；
        - 重叠子问题让“存表复用”有收益，否则就是普通的分治（归并排序）。

        **对比**：分治的子问题互不重叠；贪心每一步只做局部最优选择、**不回头**，需要额外证明（交换论证）才能保证正确——它可以看成 DP 的特例：每个状态只考虑一个转移。硬币 \`[1,3,4]\` 凑 6 就是贪心失败、DP 正确的反例。
      `,
    },
    {
      id: 'qa-memo',
      freq: 2,
      q: md`记忆化搜索和递推（填表）怎么选？各有什么坑？`,
      a: md`
        **结论**：复杂度相同；状态稀疏或顺序难推时用记忆化，需要压缩空间或 $n$ 很大时用递推。

        **原因**：
        - 记忆化只计算真正用到的状态，顺序由递归自动保证，适合区间 DP、树形 DP、数位 DP；但 Python 默认递归深度 1000，$n = 10^4$ 时会 \`RecursionError\`（可以 \`sys.setrecursionlimit\`，但栈仍可能溢出），而且缓存无法滚动释放。
        - 递推要自己推导拓扑序，但能做滚动数组，常数也更小。

        **延伸**：\`@functools.cache\` 的参数必须可哈希（list 要转 tuple）；多组测试之间要 \`cache_clear()\`，否则缓存会串数据。
      `,
    },
    {
      id: 'qa-order',
      freq: 2,
      q: md`如何确定 DP 的遍历顺序？什么时候可以用滚动数组？`,
      a: md`
        **结论**：遍历顺序 = 状态依赖图的拓扑序；当 \`dp[i]\` 只依赖有限的前几层时可以滚动。

        **原因**：把状态看作节点、“A 依赖 B”看作 B→A 的边，任何拓扑序都能保证依赖先算好。二维网格依赖“上、左”就行优先；区间 DP 依赖更短的区间就按长度递增（或 $i$ 倒序、$j$ 正序）。

        **滚动的关键**是想清楚“覆盖前读的是旧值还是新值”：LC 64 压成一维时 \`dp[j-1]\` 必须是新值（本行的左边），所以 $j$ 正序；0/1 背包要读上一行的 \`dp[j-w]\`（旧值），所以容量倒序。编辑距离还需要额外保存左上角 \`prev\`。
      `,
    },
    {
      id: 'qa-state',
      freq: 2,
      q: md`DP 的状态定义不出来时怎么办？有哪些常见的状态设计套路？`,
      a: md`
        **结论**：从“答案依赖哪些信息”出发设计状态，信息不够就加维度。

        **常见套路**：
        - 线性：\`dp[i]\` = 前 $i$ 个 / 以第 $i$ 个结尾（LIS、最大子数组和）；
        - 双序列：\`dp[i][j]\` = $a$ 的前 $i$ 个与 $b$ 的前 $j$ 个（LCS、编辑距离）；
        - 区间：\`dp[i][j]\` = 区间 $[i, j]$ 上的答案（回文、戳气球）；
        - 背包：\`dp[i][c]\` = 前 $i$ 个物品、容量 $c$；
        - 状态机：附加一维表示“持有 / 不持有 / 冷冻期 / 剩余交易次数”（股票）。

        **判断状态是否够用**：转移时如果发现“还需要知道之前某个选择的细节”（如上一步是不是 2 阶），就把那个细节加进状态里。
      `,
    },
  ],

  cheatsheet: md`
    **四步法**：状态定义 → 转移方程（按最后一步分类）→ 初始化 / 边界 → 遍历顺序（依赖的拓扑序）→ 返回值

    \`\`\`python
    # 打家劫舍：O(n) 时间，O(1) 空间
    prev2, prev1 = 0, 0
    for x in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    # 解码方法：dp[0] = 1，两段来源相加
    cur = (prev1 if s[i-1] != '0' else 0) + (prev2 if '10' <= s[i-2:i] <= '26' else 0)
    \`\`\`

    | 题 | 转移 |
    |---|---|
    | 70 爬楼梯 | \`f[i] = f[i-1] + f[i-2]\` |
    | 62 不同路径 | \`dp[i][j] = dp[i-1][j] + dp[i][j-1]\` |
    | 64 最小路径和 | \`dp[i][j] = min(上, 左) + grid[i][j]\` |
    | 213 环形打家劫舍 | \`max(rob(nums[:-1]), rob(nums[1:]))\`，n=1 特判 |

    - 复杂度 = 状态数 × 转移代价；滚动后不能回溯路径
    - 多开一格 \`dp[0]\` 表示空前缀，计数类 \`dp[0] = 1\`
    - Python 递归深度默认 1000；\`@cache\` 参数须可哈希
  `,
})
