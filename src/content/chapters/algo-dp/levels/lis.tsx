/**
 * algo-dp-lis：LIS 与子数组（以 i 结尾的状态定义）
 */
import { defineLevel, md, py } from '@/content/helpers'
import { LisN2Viz, TailsViz } from '../viz/LisViz'
import { KadaneViz } from '../viz/KadaneViz'

export default defineLevel({
  learn: [
    {
      id: 'n2',
      title: '状态定义：为什么是「以 nums[i] 结尾」',
      body: md`
        **最长递增子序列**（LC 300）：子序列可以不连续，要求严格递增，求最长长度。

        如果定义 \`dp[i]\` = “前 $i$ 个元素的 LIS 长度”，转移时就卡住了：想知道 \`nums[i]\` 能不能接上去，必须知道前面那条 LIS 的**结尾是多少**，而这个信息不在状态里。

        把定义改成 **\`dp[i]\` = 以 \`nums[i]\` 结尾的 LIS 长度**，结尾被固定了，转移就自然了：

        $$
        dp[i] = 1 + \max_{\,j < i,\ nums[j] < nums[i]} dp[j] \qquad(\text{没有合法的 } j \text{ 时 } dp[i] = 1)
        $$

        - 初始化：每个元素自身就是长度 1 的序列，\`dp = [1] * n\`
        - 遍历顺序：$i$ 从左到右，内层枚举所有 $j < i$
        - 返回值：\`max(dp)\`——LIS 可以在任何位置结束
        - 复杂度：$n$ 个状态 × $O(n)$ 转移 = $O(n^2)$
      `,
      viz: LisN2Viz,
      after: md`
        > [!TIP] 「以 i 结尾」是子序列 / 子数组问题的通用状态
        > 最大子数组和、乘积最大子数组、最长有效括号、最长等差数列……只要转移需要知道“上一个元素是谁”，就把它固定为结尾。
        > 代价是答案要对所有结尾取 \`max\`。

        想输出具体序列，就额外记录每格的“前驱” \`prev[i]\`（取到最大值的那个 $j$），从 \`argmax(dp)\` 倒着走回去。
      `,
      checkpoint: {
        id: 'cp-n2',
        type: 'single',
        prompt: md`对 \`nums = [4, 10, 4, 3, 8, 9]\` 用 $O(n^2)$ 解法，\`dp\` 数组是：`,
        options: ['`[1, 2, 1, 1, 2, 3]`', '`[1, 2, 2, 1, 2, 3]`', '`[1, 2, 1, 1, 3, 4]`', '`[1, 2, 2, 2, 3, 4]`'],
        answer: 0,
        explanation: md`$dp_0=1$；10 可接 4 → 2；第二个 4 前面没有**严格**更小的数 → 1（B 把相等也算进来了）；3 → 1；8 可接 4、4、3 → 2；9 可接 4、4、3、8 → $dp_4+1 = 3$。最终 LIS = 3（如 4, 8, 9）。`,
      },
    },
    {
      id: 'tails',
      title: 'O(n log n)：tails 数组 = 贪心 + 二分',
      body: md`
        换一个角度：维护数组 **\`tails[k]\` = 所有长度为 $k+1$ 的递增子序列中，最小的结尾**。

        **贪心直觉**：同样长度的序列，结尾越小越好——后面的数更容易接上。

        **关键性质：\`tails\` 严格递增。** 若长度 $k+2$ 的某条序列结尾为 $t$，去掉它的最后一个元素得到长度 $k+1$ 的序列，结尾 $< t$，所以 \`tails[k] < tails[k+1]\`。

        于是对每个新元素 $x$，在 \`tails\` 上 **lower_bound**（第一个 $\ge x$ 的位置 $k$）：
        - $k$ 越界：$x$ 比所有结尾都大，能接在最长的序列后面 → **追加**，LIS 长度 +1；
        - 否则：$x$ 可以接在长度 $k$ 的序列后（\`tails[k-1] < x\`），得到长度 $k+1$、结尾为 $x \le$ \`tails[k]\` 的序列 → **替换** \`tails[k] = x\`。

        每个元素一次二分，总复杂度 $O(n \log n)$。这也叫**耐心排序（patience sorting）**：\`tails\` 就是每一叠牌的堆顶。
      `,
      viz: TailsViz,
      after: md`
        > [!WARN] 两个易错点
        > 1. **tails 不是 LIS 本身**。上例最终 \`tails = [1, 3, 4, 5, 8]\`，但 1 出现在 3 之后，它不是合法子序列；只有**长度**是对的。要输出序列，需要同时记录 \`tails\` 中每个位置对应的下标和每个元素的前驱。
        > 2. **严格 vs 非严格**：严格递增用 \`bisect_left\`（第一个 $\ge x$），非降序列用 \`bisect_right\`（第一个 $> x$）。\`[7, 7, 7]\` 的严格 LIS 是 1，非降是 3。

        > [!INTERVIEW] 面试官常追问
        > “为什么替换不会出错？”——替换不改变 \`tails\` 的长度（答案不变），只是让某个长度的结尾变小，对后续元素只会更有利；而所有长度的“可行性”都被保留。
      `,
      checkpoint: {
        id: 'cp-tails',
        type: 'fill',
        prompt: md`对 \`nums = [3, 5, 1, 2, 4]\` 执行严格递增的 tails 算法。处理完 \`1\` 之后 \`tails\` = ____(1)____（写成 \`[a, b]\` 形式）；最终 LIS 长度 = ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['[1,5]', '[1, 5]', '1,5'] },
          { label: '(2)', answers: ['3'], numeric: true },
        ],
        explanation: md`3 → \`[3]\`；5 → \`[3, 5]\`；1 的 lower_bound 是 0，替换 → \`[1, 5]\`；2 替换 5 → \`[1, 2]\`；4 追加 → \`[1, 2, 4]\`，长度 3（例如 1, 2, 4）。`,
      },
    },
    {
      id: 'kadane',
      title: '子数组：Kadane 与乘积最大子数组',
      body: md`
        子**数组**要求连续，状态同样定义为“以 $i$ 结尾”，但转移只有两种：**接在 $i-1$ 后面**，或**从 $i$ 重新开始**。

        **LC 53 最大子数组和（Kadane）**：

        $$
        cur_i = \max(nums_i,\ cur_{i-1} + nums_i), \qquad ans = \max_i cur_i
        $$

        直觉：如果前面的和 $cur_{i-1} < 0$，它只会拖后腿，不如扔掉。$O(n)$ 时间、$O(1)$ 空间。

        **LC 152 乘积最大子数组**：乘以负数会让大小颠倒——之前的**最小**积（很负）乘上一个负数，可能一跃成为**最大**积。
        所以必须同时维护以 $i$ 结尾的最大积 $hi_i$ 与最小积 $lo_i$：

        $$
        hi_i = \max(x,\ hi_{i-1} x,\ lo_{i-1} x), \quad lo_i = \min(x,\ hi_{i-1} x,\ lo_{i-1} x)
        $$

        遇到 0 时三个候选里的 $x=0$ 会把两者都“重置”，天然处理了被 0 隔开的情况。
      `,
      viz: KadaneViz,
      after: md`
        > [!WARN] 同时更新
        > \`hi = max(...)\` 之后再算 \`lo = min(x, hi * x, lo * x)\`，用到的就是**新的** \`hi\`——经典 bug。用元组同时赋值，或先把候选存进临时变量。

        > [!NOTE] 初始化
        > Kadane 要用 \`cur = best = nums[0]\` 初始化。如果写成 \`best = 0\`，全负数组（如 \`[-3, -1, -2]\`）会错误地返回 0，而正确答案是 -1（子数组不能为空）。
      `,
      checkpoint: {
        id: 'cp-kadane',
        type: 'numeric',
        prompt: md`\`nums = [3, -4, 5, -1, 2, -6, 4]\` 的最大子数组和是多少？`,
        answer: 6,
        tolerance: 0,
        explanation: md`$cur$ 依次为 3, -1, 5, 4, 6, 0, 4。在 $i=2$ 处 $-1+5 < 5$ 重新开始，最大值 6 对应子数组 \`[5, -1, 2]\`。`,
      },
    },
    {
      id: 'envelope',
      title: '二维 LIS：俄罗斯套娃信封',
      body: md`
        **LC 354**：信封 $(w, h)$，只有宽和高都**严格**更大才能套进去，求最多套几层。

        二维偏序的标准降维手法：**按 $w$ 升序排序，同宽时按 $h$ 降序**，然后对 $h$ 序列求**严格递增**的 LIS（用 tails + 二分，$O(n \log n)$）。

        - $w$ 升序后，LIS 中后面的信封宽度一定 $\ge$ 前面的；
        - 为什么同宽要 $h$ **降序**？同宽的信封不能互相套。降序排列后，同宽的那一组 $h$ 是递减的，**严格递增**子序列里最多只能选其中一个，自动排除了“同宽相套”。
        - 若同宽按 $h$ 升序，\`[(3,3), (3,4)]\` 会被误判为可以套 2 层。

        > [!INTERVIEW] 一句话讲清
        > “排序消掉一维，同宽降序防止同宽相套，剩下一维就是 LIS。”
      `,
      checkpoint: {
        id: 'cp-env',
        type: 'single',
        prompt: md`信封 \`[[5,4],[6,4],[6,7],[2,3]]\` 按「宽升序、同宽高降序」排序后，用于求 LIS 的高度序列是：`,
        options: ['`[3, 4, 7, 4]`', '`[3, 4, 4, 7]`', '`[4, 4, 7, 3]`', '`[3, 7, 4, 4]`'],
        answer: 0,
        explanation: md`排序后为 \`(2,3), (5,4), (6,7), (6,4)\`，高度序列 \`[3, 4, 7, 4]\`，严格 LIS = 3（3, 4, 7），即 \`(2,3) → (5,4) → (6,7)\`。B 是同宽按高**升序**排的结果，会让 \`(6,4)\` 与 \`(6,7)\` 同时被选中，错误。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-basic',
      type: 'single',
      difficulty: 1,
      prompt: md`\`nums = [10, 9, 2, 5, 3, 7, 101, 18]\` 的最长严格递增子序列长度是：`,
      options: ['3', '4', '5', '6'],
      answer: 1,
      explanation: md`例如 2, 3, 7, 18 或 2, 5, 7, 101，长度 4。不存在长度 5 的递增子序列：前 3 个元素 10, 9, 2 两两不能同时出现在递增序列中。`,
    },
    {
      id: 'q-ret',
      type: 'judge',
      difficulty: 1,
      prompt: md`在 $O(n^2)$ 的 LIS 解法中（\`dp[i]\` = 以 \`nums[i]\` 结尾的 LIS 长度），直接返回 \`dp[n-1]\` 就是答案。`,
      answer: false,
      explanation: md`错误。LIS 不一定以最后一个元素结尾，例如 \`[1, 2, 3, 0]\` 的 \`dp[3] = 1\`，而答案是 3。应返回 \`max(dp)\`。这是“以 $i$ 结尾”类状态的通病：答案要对所有结尾取最值。`,
    },
    {
      id: 'q-tails-props',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 tails 数组算法，下列说法正确的是：`,
      options: [
        '任意时刻 tails 都是严格递增的（严格 LIS 版本）',
        '算法结束时 tails 本身就是一条最长递增子序列',
        '处理每个元素时，tails 至多有一个位置发生变化',
        '求最长非降子序列，只需把 bisect_left 换成 bisect_right',
        '处理过程中 tails 的长度可能减少',
      ],
      answer: [0, 2, 3],
      explanation: md`A：由“去掉最后一个元素”的论证可得；C：要么追加一个、要么替换一个；D：非降时相等的元素可以接上，要找第一个 $> x$ 的位置。B 错：tails 各项来自不同时刻，例如 \`[3, 4, 1]\` 结束时 tails = \`[1, 4]\`，1 在 4 之后。E 错：只有追加和替换，长度单调不减。`,
    },
    {
      id: 'q-kadane',
      type: 'numeric',
      difficulty: 1,
      prompt: md`用 Kadane 求 \`[-2, 1, -3, 4, -1, 2, 1, -5, 4]\` 的最大子数组和。`,
      answer: 6,
      tolerance: 0,
      explanation: md`$cur$ 依次为 -2, 1, -2, 4, 3, 5, 6, 1, 5，最大值 6，对应 \`[4, -1, 2, 1]\`。在 $i=3$ 处前缀和 $-2$ 为负，从 4 重新开始。`,
    },
    {
      id: 'q-order-env',
      type: 'order',
      difficulty: 2,
      prompt: md`把 LC 354 俄罗斯套娃信封的 $O(n \log n)$ 解法步骤按顺序排列：`,
      items: [
        '按宽度升序排序，宽度相同时按高度降序',
        '取出排序后的高度序列',
        '对高度序列用 tails + bisect_left 求严格递增 LIS',
        '返回 tails 的长度',
      ],
      explanation: md`先排序降掉一维；同宽降序保证同宽的信封在严格 LIS 中至多出现一个；然后就是标准 LIS。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与最合适的状态定义配对：`,
      pairs: [
        ['LC 300 最长递增子序列', '`dp[i]`：以 nums[i] 结尾的 LIS 长度'],
        ['LC 53 最大子数组和', '`cur`：以 i 结尾的子数组最大和'],
        ['LC 152 乘积最大子数组', '`hi, lo`：以 i 结尾的最大积与最小积'],
        ['LC 354 俄罗斯套娃信封', '排序后对高度序列求 LIS'],
      ],
      explanation: md`前三题都是“以 $i$ 结尾”；152 因为负数会颠倒大小关系，需要同时记最大和最小；354 用排序把二维偏序降成一维。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LC 152 乘积最大子数组（有 2 行干扰项）：`,
      lines: [
        'def max_product(nums):',
        '    hi = lo = best = nums[0]',
        '    for x in nums[1:]:',
        '        cands = (x, hi * x, lo * x)',
        '        hi, lo = max(cands), min(cands)',
        '        best = max(best, hi)',
        '    return best',
      ],
      distractors: ['        hi = max(cands)', '        lo = min(x, hi * x, lo * x)'],
      explanation: md`干扰项是“先更新 hi、再用新 hi 算 lo”的经典错误：\`lo\` 的候选里应该是**旧的** \`hi * x\`。把候选先存进 \`cands\` 再同时赋值即可避免。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面求**严格**递增 LIS 长度的代码有一处 bug（\`[7, 7, 7]\` 会返回 3），点出那一行：`,
      code: py`
        from bisect import bisect_left, bisect_right

        def length_of_lis(nums):
            tails = []
            for x in nums:
                k = bisect_right(tails, x)
                if k == len(tails):
                    tails.append(x)
                else:
                    tails[k] = x
            return len(tails)
      `,
      bugLines: [6],
      explanation: md`第 6 行应为 \`bisect_left(tails, x)\`。\`bisect_right\` 找第一个 $> x$ 的位置，遇到相等元素时会把 $x$ 追加到后面，相当于允许“相等也算递增”，求的是非降子序列。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`对 \`nums = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9]\` 执行 tails 算法（严格递增），哪些元素被处理时触发了**追加**（tails 变长）？`,
      items: ['0', '8', '4', '12', '2', '10', '6', '14', '1', '9'],
      answer: [0, 1, 3, 7],
      explanation: md`tails 变化：\`[0]\` → \`[0,8]\` → \`[0,4]\` → \`[0,4,12]\` → \`[0,2,12]\` → \`[0,2,10]\` → \`[0,2,6]\` → \`[0,2,6,14]\` → \`[0,1,6,14]\` → \`[0,1,6,9]\`。追加发生在 0、8、12、14，最终 LIS 长度 4。`,
    },
    {
      id: 'q-env-why',
      type: 'single',
      difficulty: 3,
      prompt: md`LC 354 中如果把排序改为「宽升序、同宽**高升序**」，然后对高度求严格 LIS，哪组输入会得到**错误**答案？`,
      options: ['`[[1,1],[2,2],[3,3]]`', '`[[3,3],[3,4]]`', '`[[2,5],[3,4]]`', '`[[1,9],[2,8],[3,7]]`'],
      answer: 1,
      explanation: md`\`[(3,3), (3,4)]\` 的高度序列 \`[3, 4]\` 严格递增，LIS = 2，但两者宽度相同不能相套，正确答案是 1。同宽按高**降序**后序列为 \`[4, 3]\`，LIS = 1。其余选项没有同宽信封，升降序无影响。`,
    },
    {
      id: 'q-prod',
      type: 'numeric',
      difficulty: 3,
      prompt: md`\`nums = [-1, -2, -3, 0, 2, -2, -3]\` 的乘积最大子数组的乘积是多少？`,
      answer: 12,
      tolerance: 0,
      explanation: md`被 0 分成两段：\`[-1,-2,-3]\` 的最大积是 $(-2)(-3) = 6$；\`[2,-2,-3]\` 的最大积是 $2 \cdot (-2) \cdot (-3) = 12$。在第二段，$i=5$ 时 $lo = 2 \cdot (-2) = -4$，$i=6$ 时 $lo \cdot (-3) = 12$ 翻身成最大值。`,
    },
    {
      id: 'q-init',
      type: 'judge',
      difficulty: 2,
      prompt: md`Kadane 算法若初始化为 \`cur = best = 0\` 再遍历全部元素，在 \`[-3, -1, -2]\` 上会返回错误答案 0。`,
      answer: true,
      explanation: md`正确。子数组至少包含一个元素，正确答案是 -1。\`best = 0\` 隐含了“可以选空子数组”。应初始化为 \`nums[0]\` 并从第二个元素开始遍历（或初始化为 \`-inf\`）。`,
    },
  ],

  guided: [
    {
      id: 'g-tails',
      title: 'tails + 手写二分，O(n log n) 求 LIS 长度',
      intro: md`
        实现 \`length_of_lis(nums)\`，返回最长**严格**递增子序列的长度。用 tails 数组，二分自己写（左闭右开的 lower_bound）。
      `,
      template: py`
        def length_of_lis(nums):
            tails = []
            for x in nums:
                lo, hi = {{1}}
                while lo < hi:
                    mid = (lo + hi) // 2
                    if {{2}}:
                        lo = mid + 1
                    else:
                        hi = mid
                if {{3}}:
                    tails.append(x)
                else:
                    {{4}}
            return {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`在 \`tails\` 上找第一个 $\ge x$ 的位置，答案可能是 $0 \dots$ \`len(tails)\`，用左闭右开区间。`,
          answer: '0, len(tails)',
          explain: md`\`hi = len(tails)\` 让“比所有结尾都大”的情况自然落在越界位置，无需特判。`,
        },
        {
          blank: 2,
          instruction: md`谓词为 False 的条件：\`tails[mid]\` 还**严格小于** $x$，答案在右边。`,
          answer: 'tails[mid] < x',
          accept: ['x > tails[mid]'],
          explain: md`严格递增要找第一个 $\ge x$；若写成 \`<=\` 就变成找第一个 $> x$，求的是非降子序列，\`[7,7,7]\` 会返回 3。`,
        },
        {
          blank: 3,
          instruction: md`什么情况下 $x$ 能让 LIS 变长？`,
          answer: 'lo == len(tails)',
          accept: ['lo >= len(tails)'],
          explain: md`越界说明所有结尾都 $< x$，$x$ 可以接在当前最长的序列后面。`,
        },
        {
          blank: 4,
          instruction: md`否则用 $x$ 替换掉第一个 $\ge x$ 的结尾。`,
          answer: 'tails[lo] = x',
          explain: md`替换后长度不变，但长度 \`lo+1\` 的序列结尾变小（或不变），对后续元素只会更有利。`,
        },
        {
          blank: 5,
          instruction: md`返回答案。`,
          answer: 'len(tails)',
        },
      ],
      setup: py`
        import random
        def _ref_lis(a):
            dp = [1] * len(a)
            for i in range(len(a)):
                for j in range(i):
                    if a[j] < a[i]:
                        dp[i] = max(dp[i], dp[j] + 1)
            return max(dp, default=0)
      `,
      tests: [
        { name: '示例', code: py`
          assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
          assert length_of_lis([0, 1, 0, 3, 2, 3]) == 4
        ` },
        { name: '严格递增与空数组', code: py`
          got = length_of_lis([7, 7, 7, 7])
          assert got == 1, f"[7,7,7,7] 的严格 LIS 是 1，得到 {got}"
          assert length_of_lis([]) == 0
        ` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(300)
            for _ in range(300):
                a = [rnd.randint(-5, 10) for _ in range(rnd.randint(0, 15))]
                got, exp = length_of_lis(a), _ref_lis(a)
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        Python 里可以直接 \`k = bisect.bisect_left(tails, x)\`，但面试官经常要求手写二分——这正好复用「查找」章节的 lower_bound 模板。
        下一步挑战：不仅返回长度，还要输出一条具体的 LIS。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-lis-path',
      title: '最长递增子序列：输出一条具体序列（O(n log n)）',
      difficulty: 'hard',
      source: 'LeetCode 300 进阶',
      timeLimitMin: 25,
      description: md`
        给定整数数组 \`nums\`，返回它的**一条**最长严格递增子序列（元素值组成的列表）。若有多条，返回任意一条即可；空数组返回 \`[]\`。

        \`\`\`python
        def longest_increasing_subsequence(nums: list[int]) -> list[int]: ...
        \`\`\`

        **示例**
        - \`[10, 9, 2, 5, 3, 7, 101, 18]\` → \`[2, 3, 7, 18]\`（或 \`[2, 5, 7, 101]\` 等）
        - \`[0, 1, 0, 3, 2, 3]\` → \`[0, 1, 2, 3]\`
        - \`[7, 7, 7]\` → \`[7]\`

        **约束与要求**：$0 \le n \le 10^5$，**时间复杂度 $O(n \log n)$**。隐藏测试包含 $n = 10^5$ 的数据，并会统计你读取 \`nums[i]\` 的次数——$O(n^2)$ 的双重循环会超出上限或超时。
      `,
      starter: py`
        def longest_increasing_subsequence(nums: list[int]) -> list[int]:
            # TODO: tails 记录「下标」，并为每个元素记录前驱 prev[i]
            return []
      `,
      solution: py`
        from bisect import bisect_left

        def longest_increasing_subsequence(nums: list[int]) -> list[int]:
            n = len(nums)
            tail_val: list[int] = []   # tail_val[k]: 长度 k+1 的递增子序列的最小结尾（值，用于二分）
            tail_idx: list[int] = []   # tail_idx[k]: 该结尾在 nums 中的下标
            prev = [-1] * n            # prev[i]: 以 i 结尾的最优序列中，i 的前一个元素下标
            for i in range(n):
                x = nums[i]
                k = bisect_left(tail_val, x)
                if k > 0:
                    prev[i] = tail_idx[k - 1]      # 接在长度 k 的序列后面
                if k == len(tail_val):
                    tail_val.append(x)
                    tail_idx.append(i)
                else:
                    tail_val[k] = x
                    tail_idx[k] = i
            # 从最长序列的结尾沿 prev 回溯
            out = []
            i = tail_idx[-1] if tail_idx else -1
            while i != -1:
                out.append(nums[i])
                i = prev[i]
            return out[::-1]
      `,
      explanation: md`
        **思路**：在 tails 算法的基础上多记两样东西——
        1. \`tail_idx[k]\`：长度 $k+1$ 的最小结尾在原数组里的**下标**；
        2. \`prev[i]\`：处理 \`nums[i]\` 时它落在位置 $k$，说明它接在“长度 $k$ 的最小结尾”后面，即 \`prev[i] = tail_idx[k-1]\`。

        结束时 \`tail_idx[-1]\` 是某条最长序列的结尾，沿 \`prev\` 回溯再反转即可。

        **为什么回溯出的序列合法**：\`prev[i]\` 记录的是**处理 $i$ 那一刻**的 \`tail_idx[k-1]\`，它的下标 $< i$、值 $<$ \`nums[i]\`，并且以它结尾的序列长度恰为 $k$——这些在之后 tails 被替换时都不会失效，因为 \`prev\` 是快照。

        **复杂度**：时间 $O(n \log n)$，空间 $O(n)$。

        **常见错误**：直接返回 \`tail_val\`（它不是合法子序列）；\`prev\` 记成 \`tail_idx[k]\`；用 \`bisect_right\` 导致相等元素被接上。
      `,
      complexity: { time: 'O(n log n)', space: 'O(n)' },
      hints: [
        md`先写出只求长度的 tails 算法。tails 里存“值”用来二分，再开一个平行数组存对应的**下标**。`,
        md`元素 \`nums[i]\` 落在位置 $k$ 时，它的前驱就是此刻 \`tail_idx[k-1]\`（$k=0$ 时没有前驱）。`,
        md`从 \`tail_idx[-1]\` 开始沿 \`prev\` 往回走，收集值后反转。`,
      ],
      setup: py`
        import random
        from bisect import bisect_left as _bl
        from collections.abc import Sequence

        def _ref_len(a):
            t = []
            for x in a:
                k = _bl(t, x)
                if k == len(t):
                    t.append(x)
                else:
                    t[k] = x
            return len(t)

        def _check(nums, got):
            exp = _ref_len(nums)
            assert isinstance(got, list), f"应返回 list，得到 {type(got).__name__}"
            assert len(got) == exp, f"长度应为 {exp}，得到 {len(got)}：{got[:20]}{'...' if len(got) > 20 else ''}"
            assert all(got[i] < got[i + 1] for i in range(len(got) - 1)), f"返回的序列不是严格递增：{got[:20]}"
            j = 0
            for x in nums:
                if j < len(got) and x == got[j]:
                    j += 1
            assert j == len(got), "返回的序列不是 nums 的子序列（顺序或元素不对）"

        class _Probe(Sequence):
            """只读序列：统计元素读取次数，超过上限立即中止（防止 O(n^2) 跑到超时）"""
            def __init__(self, data, budget):
                self._d = list(data)
                self.reads = 0
                self.budget = budget
            def __len__(self):
                return len(self._d)
            def __getitem__(self, i):
                if isinstance(i, slice):
                    r = self._d[i]
                    self.reads += len(r)
                else:
                    r = self._d[i]
                    self.reads += 1
                if self.reads > self.budget:
                    raise AssertionError(f"读取 nums 已超过 {self.budget} 次（n={len(self._d)}），疑似 O(n^2)：请改用 tails + 二分")
                return r
      `,
      tests: [
        { name: '示例 1', code: py`
          a = [10, 9, 2, 5, 3, 7, 101, 18]
          _check(a, longest_increasing_subsequence(a))
        ` },
        { name: '示例 2、3', code: py`
          for a in ([0, 1, 0, 3, 2, 3], [7, 7, 7]):
              _check(a, longest_increasing_subsequence(a))
        ` },
        { name: '空数组与单元素', code: py`
          assert longest_increasing_subsequence([]) == [], "空数组应返回 []"
          assert longest_increasing_subsequence([5]) == [5]
        ` },
        {
          name: '随机对拍（含重复值）',
          hidden: true,
          code: py`
            rnd = random.Random(354)
            for _ in range(300):
                a = [rnd.randint(-6, 6) for _ in range(rnd.randint(1, 20))]
                _check(a, longest_increasing_subsequence(a))
          `,
        },
        {
          name: '规模：n = 10^5 随机（统计读取次数）',
          hidden: true,
          code: py`
            rnd = random.Random(10 ** 5)
            data = [rnd.randint(0, 10 ** 9) for _ in range(100_000)]
            p = _Probe(data, budget=20 * len(data))
            got = longest_increasing_subsequence(p)
            _check(data, got)
          `,
        },
        {
          name: '规模：n = 10^5 递增 + 递减',
          hidden: true,
          code: py`
            data = list(range(50_000)) + list(range(50_000, 0, -1))
            p = _Probe(data, budget=20 * len(data))
            _check(data, longest_increasing_subsequence(p))
          `,
        },
      ],
      followUps: [
        {
          id: 'f-count',
          q: md`如果要求**最长递增子序列的个数**（LC 673）呢？`,
          a: md`$O(n^2)$：同时维护 \`length[i]\` 与 \`count[i]\`。枚举 $j<i$ 且 \`nums[j] < nums[i]\`：若 \`length[j]+1 > length[i]\` 则更新长度并令 \`count[i] = count[j]\`；若相等则 \`count[i] += count[j]\`。答案是所有 \`length[i]\` 等于最大值的 \`count[i]\` 之和。$O(n \log n)$ 需要树状数组 / 线段树按值维护 (最大长度, 计数)。`,
        },
        {
          id: 'f-min-cover',
          q: md`把数组划分成若干个**非递增**子序列，最少需要几个？`,
          a: md`等于最长严格递增子序列的长度（Dilworth 定理的对偶形式 / Mirsky 定理）。耐心排序直观地证明了它：每一叠牌是一个非递增序列，堆数恰好是 tails 的长度。`,
        },
      ],
    },
    {
      id: 'c-max-product',
      title: '乘积最大子数组',
      difficulty: 'medium',
      source: 'LeetCode 152',
      timeLimitMin: 15,
      description: md`
        给定整数数组 \`nums\`（非空），找出乘积最大的非空**连续**子数组，返回该乘积。

        \`\`\`python
        def max_product(nums: list[int]) -> int: ...
        \`\`\`

        **示例**
        - \`[2, 3, -2, 4]\` → \`6\`（子数组 \`[2, 3]\`）
        - \`[-2, 0, -1]\` → \`0\`
        - \`[-2, 3, -4]\` → \`24\`

        **约束**：$1 \le n \le 2 \times 10^4$，$-10 \le nums[i] \le 10$。要求时间 $O(n)$、空间 $O(1)$。
      `,
      starter: py`
        def max_product(nums: list[int]) -> int:
            # TODO: 同时维护以 i 结尾的最大积与最小积
            raise NotImplementedError
      `,
      solution: py`
        def max_product(nums: list[int]) -> int:
            hi = lo = best = nums[0]          # 以当前元素结尾的最大积 / 最小积
            for x in nums[1:]:
                cands = (x, hi * x, lo * x)   # 重新开始 / 接在最大积后 / 接在最小积后
                hi, lo = max(cands), min(cands)
                best = max(best, hi)
            return best
      `,
      explanation: md`
        **状态**：\`hi\`、\`lo\` 分别是以当前元素结尾的子数组的最大积与最小积。

        **转移**：新的最大积只可能来自三个候选——$x$ 本身（重新开始）、$hi \cdot x$、$lo \cdot x$。当 $x < 0$ 时，最小积（很负）乘以 $x$ 会变成最大积，所以必须记 \`lo\`。0 会让候选中出现 0，自然把两者重置。

        **复杂度**：$O(n)$ 时间，$O(1)$ 空间。

        **常见错误**：只维护最大积；分两行更新 \`hi\`、\`lo\` 导致 \`lo\` 用到了新的 \`hi\`；\`best\` 初始化为 0（\`[-2]\` 应返回 -2）。

        **另一种思路**：最大积子数组一定是某个前缀积或后缀积（在被 0 分隔的段内），正反各扫一遍取最大，遇到 0 重置为 1。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`和 Kadane 一样用“以 $i$ 结尾”的状态。但乘以负数会让最大变最小、最小变最大。`,
        md`同时维护最大积 \`hi\` 与最小积 \`lo\`；新值的候选是 \`x, hi*x, lo*x\`。`,
        md`用元组同时赋值 \`hi, lo = max(c), min(c)\`，避免 \`lo\` 用到新的 \`hi\`。`,
      ],
      setup: py`
        import random
        def _ref_brute(a):
            best = a[0]
            for i in range(len(a)):
                p = 1
                for j in range(i, len(a)):
                    p *= a[j]
                    best = max(best, p)
            return best
        def _ref_fast(a):
            hi = lo = best = a[0]
            for x in a[1:]:
                c = (x, hi * x, lo * x)
                hi, lo = max(c), min(c)
                best = max(best, hi)
            return best
      `,
      tests: [
        { name: '示例', code: py`
          for a, exp in [([2, 3, -2, 4], 6), ([-2, 0, -1], 0), ([-2, 3, -4], 24)]:
              got = max_product(a)
              assert got == exp, f"nums={a}：期望 {exp}，得到 {got}"
        ` },
        { name: '单元素与全负', code: py`
          for a, exp in [([-2], -2), ([0], 0), ([-3, -1, -2], 3), ([-1, -1, -1], 1)]:
              got = max_product(a)
              assert got == exp, f"nums={a}：期望 {exp}，得到 {got}"
        ` },
        {
          name: '随机对拍（含 0 与负数）',
          hidden: true,
          code: py`
            rnd = random.Random(152)
            for _ in range(400):
                a = [rnd.randint(-4, 4) for _ in range(rnd.randint(1, 12))]
                got, exp = max_product(a), _ref_brute(a)
                assert got == exp, f"nums={a}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：n = 2×10^4',
          hidden: true,
          code: py`
            rnd = random.Random(2)
            a = [rnd.choice([-2, -1, 1, 2, 0]) if k % 25 else 0 for k in range(20_000)]
            got, exp = max_product(a), _ref_fast(a)
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-range',
          q: md`如何同时返回最大积子数组的起止下标？`,
          a: md`为 \`hi\` 和 \`lo\` 各记录一个起点：选 $x$ 本身时起点为 $i$；选 \`hi*x\` 或 \`lo*x\` 时继承对应的起点。更新 \`best\` 时记下 \`(起点, i)\`。注意同时更新两个起点。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-tails',
      freq: 3,
      q: md`讲一下 LIS 的 $O(n \log n)$ 解法，为什么是对的？`,
      a: md`
        **结论**：维护 \`tails[k]\` = 长度 $k+1$ 的递增子序列的最小结尾，对每个元素做 lower_bound 后追加或替换，最终 \`len(tails)\` 就是答案。

        **原因**：
        1. \`tails\` 严格递增（长度 $k+2$ 的序列去掉末尾得到长度 $k+1$ 的、结尾更小的序列），所以可以二分；
        2. $x$ 落在位置 $k$ 意味着它能接在长度 $k$ 的序列后面，形成长度 $k+1$、结尾更小的序列——替换只会让未来更容易接上，追加说明找到了更长的序列；
        3. 每个长度的“存在性”都被保留，所以长度始终正确。

        **延伸**：\`tails\` 不是 LIS 本身；要输出序列需记录下标和前驱；非降版本改用 \`bisect_right\`；它等价于耐心排序的堆数。
      `,
    },
    {
      id: 'qa-end',
      freq: 2,
      q: md`为什么 LIS、最大子数组和都把状态定义为“以 $i$ 结尾”，而不是“前 $i$ 个”？`,
      a: md`
        **结论**：因为转移需要知道“上一个被选元素是谁”。

        **原因**：LIS 要判断 \`nums[i]\` 能否接上，需要前一个元素的值；子数组要求连续，必须知道前一段是否恰好在 $i-1$ 结束。“前 $i$ 个的最优值”丢失了这些信息，无法转移；“以 $i$ 结尾”把这个信息固定在状态里。

        **代价**：答案是所有状态的最大值，而不是最后一个。若题目要求“前 $i$ 个”的答案，可以再维护一个前缀最大值。
      `,
    },
    {
      id: 'qa-kadane',
      freq: 3,
      q: md`Kadane 算法是贪心还是 DP？全是负数时怎么办？如何返回子数组位置？`,
      a: md`
        **结论**：它是 DP（状态“以 $i$ 结尾的最大和”）+ 滚动优化，“负前缀丢弃”是转移方程 $\max(x, cur+x)$ 的直观解释。

        **全负数**：初始化 \`cur = best = nums[0]\`，返回最大的那个负数；不能用 0 初始化（那隐含允许空子数组）。

        **位置**：记录当前段起点 \`start\`，每次“重新开始”时 \`start = i\`；更新 \`best\` 时保存 \`(start, i)\`。

        **延伸**：环形最大子数组和（LC 918）= $\max(\text{普通最大和},\ \text{总和} - \text{最小子数组和})$，全负时取前者。
      `,
    },
    {
      id: 'qa-2d',
      freq: 1,
      q: md`俄罗斯套娃信封中为什么同宽要按高度降序？如果是三维（箱子堆叠）还能这么做吗？`,
      a: md`
        **结论**：同宽降序保证同宽的信封在严格递增 LIS 中最多选一个；三维不能简单降维。

        **原因**：排序后宽度非降，剩下要保证“宽严格大”——同宽一组的高度降序排列，严格递增子序列至多取其一，于是同宽相套被自动排除。

        **延伸**：三维偏序（如 LC 1691 堆叠长方体）一般排序后用 $O(n^2)$ DP，或 CDQ 分治 / 树状数组做到 $O(n \log^2 n)$。
      `,
    },
  ],

  cheatsheet: md`
    \`\`\`python
    from bisect import bisect_left
    def lis(nums):                 # 严格递增；非降用 bisect_right
        tails = []
        for x in nums:
            k = bisect_left(tails, x)
            if k == len(tails): tails.append(x)
            else: tails[k] = x
        return len(tails)
    # Kadane：cur = max(x, cur + x)；best = max(best, cur)
    # 152：c = (x, hi*x, lo*x); hi, lo = max(c), min(c)
    \`\`\`

    | 题 | 状态 | 复杂度 |
    |---|---|---|
    | 300 LIS | 以 i 结尾 / tails | $O(n^2)$ / $O(n\log n)$ |
    | 53 最大和 | 以 i 结尾的最大和 | $O(n)$ |
    | 152 最大积 | 以 i 结尾的 max 与 min | $O(n)$ |
    | 354 信封 | 宽升、同宽高降 + LIS | $O(n\log n)$ |

    - tails 严格递增，但**不是** LIS 本身；输出序列要记下标 + 前驱
    - 答案是 \`max(dp)\` 不是 \`dp[-1]\`；Kadane 用 \`nums[0]\` 初始化
  `,
})
