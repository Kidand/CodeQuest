/**
 * 滑动窗口（algo-search-window）
 * LC3 / 76 / 438 / 209；大模型岗场景：滑动窗口注意力的感受野、n-gram 重复检测。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { MinSubarrayViz } from '../viz/MinSubarrayViz'
import { LongestSubstrViz } from '../viz/LongestSubstrViz'
import { MinWindowViz } from '../viz/MinWindowViz'
import { SwaViz } from '../viz/SwaViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'template',
      title: '模板：右扩找可行，左缩找最优',
      body: md`
        滑动窗口 = **同向双指针** + 窗口内的状态（和、计数、哈希表）。枚举右端点 \`r\`，左端点 \`l\` 只增不减：

        \`\`\`python
        for r in range(n):
            加入 nums[r]，更新窗口状态
            while 窗口“满足 / 违反”条件:
                (求最短) 更新答案
                移出 nums[l]，l += 1
            (求最长) 更新答案
        \`\`\`

        每个元素最多进窗一次、出窗一次，总复杂度 $O(n)$（不是 $O(n^2)$，尽管有两层循环）。

        **什么时候能用？** 关键是**单调性**：对固定的 \`r\`，若 \`[l, r]\` 满足条件，则更长的 \`[l-1, r]\` 也满足（或反过来）。
        这样“\`l\` 只右移”才不会漏解。LC209（正整数，和 ≥ target 的最短子数组）满足：元素都为正，窗口变长和一定变大。
      `,
      viz: MinSubarrayViz,
      after: md`
        > [!WARN] 负数打破单调性
        > 打开“含负数”开关：\`[1, 2, -5, 4, 3, 1]\`、target = 7，滑窗返回 0，正确答案却是 2（\`[4, 3]\`）。负数让“窗口变长和变大”不再成立，窗口在错误的时机停止收缩/扩张。
        > 含负数时改用**前缀和**：和为 K 的子数组（LC560）用前缀和 + 哈希表；和 ≥ K 的最短子数组（LC862）用前缀和 + **单调队列**。
      `,
      checkpoint: {
        id: 'cp-template',
        type: 'single',
        prompt: md`LC209 的滑动窗口解法中有两层循环（\`for r\` 套 \`while\`），它的时间复杂度是：`,
        options: ['$O(n)$', '$O(n \\log n)$', '$O(n^2)$', '取决于 target'],
        answer: 0,
        explanation: md`\`l\` 和 \`r\` 都只增不减，各自最多移动 $n$ 次，\`while\` 循环体在整个算法中总共最多执行 $n$ 次。这是**均摊分析**：不能简单地把两层循环的次数相乘。`,
      },
    },
    {
      id: 'longest',
      title: '求最长：无重复字符的最长子串（LC3）',
      body: md`
        窗口条件：窗口内每个字符最多出现一次。它满足单调性——无重复的窗口，缩短后仍无重复。

        **写法一：计数 + while 收缩**。\`cnt[c] += 1\` 后若 \`cnt[c] > 1\`，不断移出 \`s[l]\` 直到 \`cnt[c] == 1\`。
        **求最长时在收缩之后更新答案**——此时窗口合法。

        **写法二：下标跳跃**。记录每个字符最近出现的下标 \`last[c]\`。遇到重复时 \`l\` 直接跳到 \`last[c] + 1\`，省掉逐个收缩。
        但必须加条件 \`last[c] >= l\`：那个重复字符可能**早已不在窗口里**，此时不能让 \`l\` 往回跳（经典反例 \`"abba"\`）。
      `,
      viz: LongestSubstrViz,
      after: md`
        > [!TIP] “求最长”与“求最短”的位置差异
        > | 目标 | while 条件 | 在哪里更新答案 |
        > |---|---|---|
        > | 最长（LC3） | 窗口**不合法**时收缩 | while **之后** |
        > | 最短（LC209、LC76） | 窗口**合法**时收缩 | while **之内** |
        >
        > 想清楚“while 循环结束时窗口是什么状态”，就不会放错位置。
      `,
      checkpoint: {
        id: 'cp-longest',
        type: 'numeric',
        prompt: md`\`s = "dvdfxdv"\` 的无重复字符最长子串长度是多少？`,
        answer: 4,
        tolerance: 0,
        explanation: md`窗口依次为 \`"dv"\` → 遇到第二个 d，l 跳到 1 得 \`"vd"\` → \`"vdf"\` → \`"vdfx"\`（长度 4）→ 遇到 d（下标 2）跳到 3 得 \`"fxd"\` → \`"fxdv"\`（v 上次在下标 1 < l，不跳），长度 4。`,
      },
    },
    {
      id: 'cover',
      title: '求最短：最小覆盖子串（LC76）与计数技巧',
      body: md`
        **LC76**：在 \`s\` 中找包含 \`t\` 全部字符（含重复）的最短子串。窗口条件“覆盖 t”同样单调：覆盖了 t 的窗口，加长后仍覆盖。

        朴素做法每次比较两个计数表，代价 $O(|\Sigma|)$。更好的做法维护两个量：

        - \`need[c]\`：还需要多少个 \`c\`。进窗 \`need[c] -= 1\`，出窗 \`need[c] += 1\`；**可以为负**（表示窗口里多余）；
        - \`missing\`：还缺多少个字符。只有当 \`need[c] > 0\` 时进窗才让 \`missing -= 1\`（这是一个“有用”的字符）；出窗后若 \`need[c] > 0\` 则 \`missing += 1\`。

        \`missing == 0\` 即覆盖，判断变成 $O(1)$，总复杂度 $O(|s| + |t|)$。

        **固定长度窗口**（LC438 找所有字母异位词）是更简单的特例：窗口长度恒为 \`len(p)\`，每步进一个、出一个，比较计数即可。
      `,
      viz: MinWindowViz,
      after: md`
        > [!INTERVIEW] 讲清楚 need 为负的含义
        > “\`need[c] < 0\` 表示窗口里 c 多了几个，移出它们不会破坏覆盖；只有把 \`need[c]\` 从 0 加到 1 的那次移出才会让 missing 加一。”这说明你理解了计数器的不变量，而不是在背代码。
      `,
      checkpoint: {
        id: 'cp-cover',
        type: 'judge',
        prompt: md`LC76 中，窗口每进入一个字符 \`c\` 都应该执行 \`missing -= 1\`。`,
        answer: false,
        explanation: md`错误。只有 \`need[c] > 0\`（c 确实还缺）时才减。若 c 不在 t 中或已经够了，它是多余字符，不能减少 missing，否则会在没有真正覆盖时误判为覆盖。`,
      },
    },
    {
      id: 'llm',
      title: '大模型岗场景：滑动窗口注意力与 n-gram 重复检测',
      body: md`
        **滑动窗口注意力（SWA）**：每个 token 只关注最近 $W$ 个 token，注意力计算从 $O(n^2)$ 降到 $O(nW)$。Mistral 7B 论文中 $W = 4096$、32 层：

        - 单层看 $W$ 个，但第 $k$ 层的隐状态已经汇聚了上一层窗口内的信息，**感受野随层数线性增长**，约 $k \times W$；论文称最后一层的理论注意力跨度约 131K token；
        - **Rolling buffer cache**：KV cache 大小固定为 $W$，位置 $i$ 写入槽 $i \bmod W$。论文报告 32K 序列下 cache 内存降为 1/8（$32768 / 4096$）。

        之后的模型多采用**局部 + 全局交替**：Gemma 2 隔层交替（局部窗口 4096），Gemma 3 改为每 5 层局部（窗口 1024）配 1 层全局。
        纯 SWA 只靠层数传递远距离信息，长程检索能力有限，混合全局层是对这个代价的补偿。

        **n-gram 重复检测**：生成时 \`no_repeat_ngram_size = n\`（HF transformers 的生成参数）禁止任何 n-gram 出现两次。
        做法就是一个长度为 $n-1$ 的滑动窗口：取已生成序列的最后 $n-1$ 个 token 作为前缀，查哈希表中该前缀之后出现过哪些 token，把它们的 logit 置为 $-\infty$。
      `,
      viz: SwaViz,
      after: md`
        > [!NOTE] 两个“窗口”不要混
        > SWA 的窗口是**注意力掩码**（每层每个 query 能看哪些 key），算法题的窗口是**区间状态**。共同点是“只维护最近 W 个元素的信息”，所以 rolling buffer 和环形队列是同一个数据结构。

        > [!WARN] 感受野 ≠ 有效上下文
        > 理论感受野 $k \times W$ 是信息**可能**传播到的范围；信息每经过一层都会被稀释，实际能有效利用的距离远小于此。长上下文检索任务上，纯 SWA 明显弱于全局注意力。
      `,
      checkpoint: {
        id: 'cp-llm',
        type: 'single',
        prompt: md`Mistral 风格的 rolling buffer cache，$W = 4096$。位置 $i = 10000$ 的 KV 写入哪个槽？`,
        options: ['1808', '4096', '10000', '2048'],
        answer: 0,
        explanation: md`$10000 \bmod 4096 = 10000 - 2 \times 4096 = 1808$。它会覆盖位置 $10000 - 4096 = 5904$ 的旧 KV——那个位置已经超出窗口，不再被任何 query 需要。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-when',
      type: 'single',
      difficulty: 1,
      prompt: md`滑动窗口能够保证 $O(n)$ 且不漏解的核心前提是：`,
      options: [
        '窗口条件具有单调性：固定右端点时，左端点右移只会让窗口从“合法”变“不合法”（或反之），不会来回反复',
        '数组必须有序',
        '答案一定是连续子数组',
        '数组元素互不相同',
      ],
      answer: 0,
      explanation: md`单调性保证 \`l\` 只需右移、不必回退。B 不需要（LC3 的字符串无序）；C 是必要条件但不充分（LC560 也求连续子数组，但有负数时不能滑窗）；D 无关。`,
    },
    {
      id: 'q-neg',
      type: 'judge',
      difficulty: 1,
      prompt: md`数组含负数时，“和 ≥ target 的最短子数组”仍然可以用 LC209 的滑动窗口模板求解。`,
      answer: false,
      explanation: md`错误。负数破坏了“窗口变长和变大”的单调性，例如 \`[1, 2, -5, 4, 3, 1]\`、target = 7 时滑窗返回 0，正确答案是 2。应使用前缀和 + 单调队列（LC862），$O(n)$。`,
    },
    {
      id: 'q-which',
      type: 'multi',
      difficulty: 2,
      prompt: md`以下哪些题目适合用滑动窗口（同向双指针）在 $O(n)$ 内解决？（多选）`,
      options: [
        'LC3 无重复字符的最长子串',
        'LC438 找到字符串中所有字母异位词',
        'LC209 长度最小的子数组（元素为正整数）',
        'LC560 和为 K 的子数组（元素可为负）',
        'LC53 最大子数组和',
      ],
      answer: [0, 1, 2],
      explanation: md`A、B、C 的窗口条件都单调（B 是固定长度窗口）。D 有负数，“和”不单调，应用前缀和 + 哈希表统计 \`prefix - K\` 出现次数；E 用 Kadane（DP），窗口的左端点何时右移没有单调判据。`,
    },
    {
      id: 'q-lc3',
      type: 'numeric',
      difficulty: 2,
      prompt: md`\`s = "abcbdefbgh"\` 的无重复字符最长子串长度是多少？`,
      answer: 6,
      tolerance: 0,
      explanation: md`下标 3 的 b 与下标 1 重复，l 跳到 2：\`"cbdef"\`（长度 5）；下标 7 的 b 与下标 3 重复，l 跳到 4：\`"defbgh"\`（下标 4–9），长度 6。`,
    },
    {
      id: 'q-lc76',
      type: 'fill',
      difficulty: 2,
      prompt: md`LC76：\`s = "ADOBECODEBANC"\`、\`t = "ABC"\`，最小覆盖子串是 ____(1)____，长度为 ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['BANC', '"BANC"', "'BANC'"] },
        { label: '(2)', answers: ['4'], numeric: true },
      ],
      explanation: md`第一个覆盖窗口是 \`"ADOBEC"\`（长 6），之后右扩到末尾、左缩掉多余字符，最短的是末尾的 \`"BANC"\`。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把“求最短可行子数组”的滑动窗口一轮循环内的操作按顺序排列：`,
      items: [
        '右端点 r 前进，把 nums[r] 加入窗口状态',
        '检查窗口是否满足条件',
        '满足时用 r - l + 1 更新最短答案',
        '把 nums[l] 移出窗口状态，l += 1，回到条件检查',
      ],
      explanation: md`求最短时在 while（满足条件）内部先记录答案，再收缩；求最长时则相反——在 while（不满足）收缩完成**之后**记录。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把窗口类型与“更新答案的时机”配对：`,
      pairs: [
        ['求最长合法窗口（LC3）', '违反条件时 while 收缩，收缩之后更新'],
        ['求最短合法窗口（LC209 / 76）', '满足条件时 while 收缩，循环内部更新'],
        ['固定长度 k（LC438）', '窗口长度达到 k 后每步进一出一并检查'],
        ['统计合法子数组个数（LC713）', '每个 r 贡献 r - l + 1 个以 r 结尾的子数组'],
      ],
      explanation: md`第四种是常见变体：窗口 \`[l, r]\` 合法且条件单调时，\`[l..r, r]\` 这 \`r - l + 1\` 个以 r 结尾的子数组都合法（LC713 乘积小于 K 的子数组）。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LC3 的“下标跳跃”写法（有 2 行干扰项）：`,
      lines: [
        'def length_of_longest(s):',
        '    last = {}',
        '    best = l = 0',
        '    for r, c in enumerate(s):',
        '        if c in last and last[c] >= l:',
        '            l = last[c] + 1',
        '        last[c] = r',
        '        best = max(best, r - l + 1)',
        '    return best',
      ],
      distractors: ['        if c in last:', '            l = last[c]'],
      explanation: md`缺少 \`last[c] >= l\` 时，已经离开窗口的旧位置会把 l 往回拉（\`"abba"\` 会得到 3）；\`l = last[c]\` 会把重复字符本身留在窗口里。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 LC3 实现在 \`"abba"\` 上返回 3（正确答案是 2）。点出有问题的行：`,
      code: py`
        def length_of_longest(s):
            last = {}
            best = l = 0
            for r, c in enumerate(s):
                if c in last:
                    l = last[c] + 1
                last[c] = r
                best = max(best, r - l + 1)
            return best
      `,
      bugLines: [5],
      explanation: md`第 5 行应为 \`if c in last and last[c] >= l:\`。\`"abba"\`：r=2 时 b 重复，l 跳到 2；r=3 时 a 上次在下标 0，已不在窗口内，但代码仍执行 \`l = 1\`，l 往回跳，窗口 \`"bba"\` 被误认为无重复。也可以写成 \`l = max(l, last[c] + 1)\`。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 3,
      prompt: md`滑动窗口注意力，窗口 $W = 3$（含自身），共 2 层。token \`t7\` 在第 2 层的输出**能间接利用**哪些位置的信息？（点出全部）`,
      items: ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7'],
      answer: [3, 4, 5, 6, 7],
      explanation: md`第 2 层的 t7 直接看第 1 层的 t5、t6、t7；而第 1 层的 t5 又看了原始输入的 t3、t4、t5。所以感受野是 \`[7 - 2×(3-1), 7] = [3, 7]\`，共 $k(W-1)+1 = 5$ 个 token。`,
    },
    {
      id: 'q-mistral',
      type: 'numeric',
      difficulty: 3,
      prompt: md`Mistral 7B 使用滑动窗口 $W = 4096$，共 32 层。按“感受野 ≈ 层数 × 窗口”估算，最后一层的理论注意力跨度约为多少个 token？`,
      answer: 131072,
      tolerance: 0.01,
      unit: 'token',
      explanation: md`$32 \times 4096 = 131072 \approx 131\text{K}$，与论文表述一致。精确地说若窗口含自身，跨度为 $k(W-1)+1 = 131041$，两者相差不到 0.1%。`,
    },
    {
      id: 'q-ngram',
      type: 'single',
      difficulty: 2,
      prompt: md`
        生成时设置 \`no_repeat_ngram_size = 3\`，已生成序列为 \`[5, 8, 2, 7, 5, 8]\`。下一步必须禁止的 token 是：
      `,
      options: ['`2`', '`7`', '`5` 和 `8`', '不禁止任何 token'],
      answer: 0,
      explanation: md`取最后 $n-1 = 2$ 个 token \`(5, 8)\` 作为前缀，在历史中查找：\`(5, 8)\` 之后出现过 \`2\`（三元组 \`(5,8,2)\`）。若下一个生成 2，就会重复 \`(5, 8, 2)\`，所以禁止 2。实现上用哈希表 \`前缀 → 后继集合\` 增量维护，每步 $O(n)$。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-anagrams',
      title: '固定长度窗口：找到字符串中所有字母异位词',
      intro: md`
        LC438：返回 \`s\` 中所有是 \`p\` 的字母异位词的子串的**起始下标**（升序）。
        窗口长度恒为 \`k = len(p)\`：每一步右端进一个字符、左端出一个字符，再比较窗口计数与 \`p\` 的计数。
        （Python 3.10+ 中 \`Counter\` 比较相等时把缺失的键视为 0，所以计数减到 0 不必删除。）
      `,
      template: py`
        from collections import Counter

        def find_anagrams(s, p):
            k = len(p)
            need, win = Counter(p), Counter()
            res = []
            for r, c in enumerate(s):
                {{1}}                          # 右端 s[r] 进窗
                if r >= k:
                    {{2}}                      # 左端 s[r-k] 出窗，保持长度为 k
                if {{3}}:                      # 窗口是 p 的异位词
                    res.append({{4}})          # 记录窗口起点
            return res
      `,
      steps: [
        {
          blank: 1,
          instruction: md`把右端字符 \`c\` 计入窗口。`,
          answer: 'win[c] += 1',
          accept: ['win[s[r]] += 1'],
        },
        {
          blank: 2,
          instruction: md`窗口长度超过 k 了：移出窗口最左边的字符 \`s[r - k]\`。`,
          answer: 'win[s[r - k]] -= 1',
          explain: md`当 \`r >= k\` 时，加入 \`s[r]\` 后窗口是 \`[r-k, r]\`，长度 k+1，需要移出 \`s[r-k]\`，窗口变为 \`[r-k+1, r]\`。`,
        },
        {
          blank: 3,
          instruction: md`判断窗口计数是否与 \`need\` 相同。`,
          answer: 'win == need',
          accept: ['r >= k - 1 and win == need', 'need == win'],
          explain: md`窗口不足 k 个字符时总数不等，\`win == need\` 自然为假，因此不需要额外判断长度。比较两个 Counter 是 $O(|\Sigma|)$；进阶写法维护“匹配的字符种类数”可以做到每步 $O(1)$。`,
        },
        {
          blank: 4,
          instruction: md`窗口 \`[r-k+1, r]\` 的起点下标。`,
          answer: 'r - k + 1',
        },
      ],
      setup: py`
        import random
        from collections import Counter as _C
        def _ref_anagrams(s, p):
            k = len(p)
            return [i for i in range(len(s) - k + 1) if _C(s[i:i + k]) == _C(p)]
      `,
      tests: [
        { name: '示例 1', code: py`assert find_anagrams("cbaebabacd", "abc") == [0, 6]` },
        { name: '示例 2', code: py`assert find_anagrams("abab", "ab") == [0, 1, 2]` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(438)
            for _ in range(300):
                s = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 14)))
                p = "".join(rnd.choice("abc") for _ in range(rnd.randint(1, 4)))
                exp, got = _ref_anagrams(s, p), find_anagrams(s, p)
                assert got == exp, f"s={s!r}, p={p!r}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        固定长度窗口没有 while 收缩，结构更简单。把“比较两个计数表”优化成维护一个 \`valid\`（计数恰好相等的字符种类数），
        就和 LC76 的 \`missing\` 是同一个思想：用一个整数概括整张计数表的状态。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-lc3',
      title: '无重复字符的最长子串',
      difficulty: 'medium',
      source: 'LeetCode 3',
      timeLimitMin: 15,
      description: md`
        给定字符串 \`s\`，返回不含重复字符的**最长子串**的长度。

        \`\`\`python
        def length_of_longest_substring(s: str) -> int: ...
        \`\`\`

        **示例**
        - \`"abcabcbb"\` → \`3\`（\`"abc"\`）
        - \`"bbbbb"\` → \`1\`
        - \`"pwwkew"\` → \`3\`（\`"wke"\`；注意 \`"pwke"\` 是子序列不是子串）
        - \`""\` → \`0\`

        **约束**：$0 \le n \le 5 \times 10^4$，字符可以是任意 Unicode（字母、数字、空格、中文…）。要求 $O(n)$。
      `,
      starter: py`
        def length_of_longest_substring(s: str) -> int:
            # TODO: 滑动窗口；窗口内每个字符最多出现一次
            raise NotImplementedError
      `,
      solution: py`
        def length_of_longest_substring(s: str) -> int:
            last = {}                  # 字符 -> 最近一次出现的下标
            best = l = 0
            for r, c in enumerate(s):
                if c in last and last[c] >= l:
                    l = last[c] + 1    # 重复字符在窗口内：l 跳到它之后
                last[c] = r
                best = max(best, r - l + 1)
            return best
      `,
      solutionAlt: [
        {
          label: '计数 + while 收缩写法',
          code: py`
            from collections import defaultdict

            def length_of_longest_substring(s):
                cnt = defaultdict(int)
                best = l = 0
                for r, c in enumerate(s):
                    cnt[c] += 1
                    while cnt[c] > 1:          # 违反条件 → 收缩
                        cnt[s[l]] -= 1
                        l += 1
                    best = max(best, r - l + 1)   # 求最长：收缩后更新
                return best
          `,
        },
      ],
      explanation: md`
        **思路**：窗口 \`[l, r]\` 维护“无重复”。右端每前进一步，若新字符已在窗口内出现，就把 \`l\` 移到它上次出现位置之后。

        **两种写法**：计数 + while 收缩（通用模板，易推广）；下标跳跃（每步 $O(1)$，但必须判断 \`last[c] >= l\`，或写成 \`l = max(l, last[c] + 1)\`）。

        **复杂度**：时间 $O(n)$，空间 $O(\min(n, |\Sigma|))$。

        **常见错误**：漏掉 \`last[c] >= l\` 导致 \`l\` 回退（\`"abba"\` 返回 3）；在 while 收缩之前更新答案。
      `,
      complexity: { time: 'O(n)', space: 'O(min(n, |Σ|))' },
      hints: [
        md`用 \`[l, r]\` 表示当前无重复的窗口。右端加入字符 c 时，什么情况下窗口会变得不合法？`,
        md`记录每个字符最近一次出现的下标。若 c 上次出现的位置在窗口内，\`l\` 应该移到哪里？`,
        md`\`if c in last and last[c] >= l: l = last[c] + 1\`，然后更新 \`last[c] = r\` 和答案。`,
      ],
      setup: py`
        import random, time

        def _ref_lls(s):
            best = 0
            for i in range(len(s)):
                seen = set()
                for j in range(i, len(s)):
                    if s[j] in seen:
                        break
                    seen.add(s[j])
                best = max(best, len(seen))
            return best
      `,
      tests: [
        {
          name: '示例',
          code: py`
            for s, exp in [("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0)]:
                got = length_of_longest_substring(s)
                assert got == exp, f"s={s!r}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '回跳陷阱 "abba"',
          code: py`
            got = length_of_longest_substring("abba")
            assert got == 2, f"期望 2，得到 {got}——是不是让 l 往回跳了？"
          `,
        },
        {
          name: '空格与 Unicode',
          code: py`
            assert length_of_longest_substring(" ") == 1
            assert length_of_longest_substring("a b c a") == 3
            assert length_of_longest_substring("大模型大模型面试") == 5
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(3)
            for _ in range(400):
                s = "".join(rnd.choice("abcde") for _ in range(rnd.randint(0, 16)))
                exp, got = _ref_lls(s), length_of_longest_substring(s)
                assert got == exp, f"s={s!r}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：3000 个互不相同的字符',
          hidden: true,
          code: py`
            s = "".join(chr(0x4E00 + i) for i in range(3000)) * 2
            t0 = time.perf_counter()
            got = length_of_longest_substring(s)
            dt = time.perf_counter() - t0
            assert got == 3000, f"期望 3000，得到 {got}"
            assert dt < 0.5, f"用时 {dt:.2f}s，疑似 O(n^2)"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-k',
          q: md`“至多包含 K 个不同字符的最长子串”（LC340）怎么改？`,
          a: md`窗口条件改为“不同字符数 ≤ K”：维护 \`cnt\` 与不同字符数 \`kinds\`；进窗时若 \`cnt[c]\` 从 0 变 1 则 \`kinds += 1\`；\`while kinds > K\` 收缩，出窗时计数归零则 \`kinds -= 1\`。收缩后更新最长答案。`,
        },
      ],
    },
    {
      id: 'c-lc76',
      title: '最小覆盖子串',
      difficulty: 'hard',
      source: 'LeetCode 76',
      timeLimitMin: 25,
      description: md`
        给定字符串 \`s\` 和 \`t\`，返回 \`s\` 中涵盖 \`t\` 所有字符（**包括重复次数**）的最短子串；不存在返回 \`""\`。
        区分大小写。若有多个最短答案，返回任意一个即可（测试只检查长度与是否覆盖）。

        \`\`\`python
        def min_window(s: str, t: str) -> str: ...
        \`\`\`

        **示例**
        - \`min_window("ADOBECODEBANC", "ABC")\` → \`"BANC"\`
        - \`min_window("a", "a")\` → \`"a"\`
        - \`min_window("a", "aa")\` → \`""\`（需要两个 a）

        **约束**：$1 \le |s|, |t| \le 10^5$。要求 $O(|s| + |t|)$：每一步判断“是否覆盖”应为 $O(1)$。
      `,
      starter: py`
        def min_window(s: str, t: str) -> str:
            # TODO: need 计数 + missing；右扩直到覆盖，再左缩到不覆盖
            raise NotImplementedError
      `,
      solution: py`
        from collections import Counter

        def min_window(s: str, t: str) -> str:
            need = Counter(t)                 # 还需要的数量，可以为负（表示多余）
            missing = len(t)                  # 还缺多少个字符
            l, best_l, best_len = 0, 0, float('inf')
            for r, c in enumerate(s):
                if need[c] > 0:
                    missing -= 1              # c 是一个有用的字符
                need[c] -= 1
                while missing == 0:           # 窗口 [l, r] 覆盖 t → 左缩
                    if r - l + 1 < best_len:
                        best_l, best_len = l, r - l + 1
                    need[s[l]] += 1
                    if need[s[l]] > 0:        # 移出了一个必需字符
                        missing += 1
                    l += 1
            return "" if best_len == float('inf') else s[best_l:best_l + best_len]
      `,
      explanation: md`
        **思路**：\`need[c]\` 记录还需要多少个 c（可为负），\`missing\` 记录还缺的字符总数。右端进窗：\`need[c] > 0\` 时 \`missing -= 1\`；
        \`missing == 0\` 时窗口覆盖 t，记录答案并左缩；左端出窗后若 \`need[c] > 0\`，说明移出了必需字符，\`missing += 1\`，停止收缩。

        **复杂度**：每个字符进出窗口各一次，$O(|s| + |t|)$；空间 $O(|\Sigma|)$。

        **常见错误**
        - 每进一个字符都 \`missing -= 1\`（应只在 \`need[c] > 0\` 时）；
        - 用 \`len(need)\`（字符种类数）代替 \`len(t)\` 却又按个数递减，混淆了两种计数口径；
        - 每步比较两个 Counter，退化为 $O(|s| \cdot |\Sigma|)$——功能正确但面试官会追问。
      `,
      complexity: { time: 'O(|s| + |t|)', space: 'O(|Σ|)' },
      hints: [
        md`窗口“覆盖 t”这个条件单调吗？覆盖了 t 的窗口变长后还覆盖吗？`,
        md`怎样 $O(1)$ 判断覆盖？维护 \`need = Counter(t)\` 和一个整数 \`missing = len(t)\`。`,
        md`进窗：\`if need[c] > 0: missing -= 1\`；\`need[c] -= 1\`。\`while missing == 0\`：更新答案，出窗 \`need[s[l]] += 1\`，若变为正则 \`missing += 1\`，\`l += 1\`。`,
      ],
      setup: py`
        import random, time
        from collections import Counter as _C

        def _covers(w, t):
            cw = _C(w)
            return all(cw[ch] >= k for ch, k in _C(t).items())

        def _ref_len(s, t):
            best = 0
            for i in range(len(s)):
                for j in range(i, len(s)):
                    if _covers(s[i:j + 1], t):
                        L = j - i + 1
                        best = L if best == 0 else min(best, L)
                        break
            return best

        def _check(s, t, got):
            exp_len = _ref_len(s, t)
            if exp_len == 0:
                assert got == "", f"s={s!r}, t={t!r}: 不存在覆盖子串，应返回 \"\"，得到 {got!r}"
            else:
                assert isinstance(got, str) and got in s, f"s={s!r}, t={t!r}: {got!r} 不是 s 的子串"
                assert _covers(got, t), f"s={s!r}, t={t!r}: {got!r} 没有覆盖 t"
                assert len(got) == exp_len, f"s={s!r}, t={t!r}: 期望长度 {exp_len}，得到 {got!r}（长度 {len(got)}）"
      `,
      tests: [
        { name: '示例 1', code: py`assert min_window("ADOBECODEBANC", "ABC") == "BANC"` },
        { name: '示例 2 / 3', code: py`assert min_window("a", "a") == "a" and min_window("a", "aa") == ""` },
        {
          name: 't 含重复字符 / 大小写',
          code: py`
            got = min_window("aaflslflsldkalskaaa", "aaa")
            assert got == "aaa", f"期望 'aaa'，得到 {got!r}"
            assert min_window("aA", "A") == "A", "区分大小写"
            assert min_window("abc", "d") == ""
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(76)
            for _ in range(250):
                s = "".join(rnd.choice("abcd") for _ in range(rnd.randint(1, 14)))
                t = "".join(rnd.choice("abc") for _ in range(rnd.randint(1, 3)))
                _check(s, t, min_window(s, t))
          `,
        },
        {
          name: '复杂度：|s| = 10^5',
          hidden: true,
          code: py`
            rnd = random.Random(1)
            s = "".join(rnd.choice("abcdefghijklmnopqrstuvwxy") for _ in range(100_000)) + "z"
            t = "zyxa"
            t0 = time.perf_counter()
            got = min_window(s, t)
            dt = time.perf_counter() - t0
            assert got.endswith("z") and _covers(got, t), f"结果 {got[:30]!r}... 不正确"
            assert dt < 1.5, f"用时 {dt:.2f}s，覆盖判断是不是每步都比较了整张计数表？"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-subseq',
          q: md`如果要求的是“最短的包含 t 作为**子序列**的子串”（LC727），还能用这个滑窗吗？`,
          a: md`不能直接用计数，因为子序列有顺序要求。常见解法是双指针“正向匹配 + 反向收缩”：从 s 的某处正向匹配完 t，再从结束位置反向匹配得到最靠右的起点，记录窗口后从起点 + 1 继续，最坏 $O(|s| \cdot |t|)$；或用 DP / 预处理“下一个字符位置”表。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-when',
      freq: 3,
      q: md`怎么判断一道题能不能用滑动窗口？有负数时怎么办？`,
      a: md`
        **结论**：看窗口条件是否单调——固定右端点时，左端点右移只会让窗口单向地从“合法”变“不合法”（或反之）。

        **原因**：单调性保证 \`l\` 永不回退，每个元素进出窗口各一次，$O(n)$。正整数数组的“和 ≥ target”、字符串的“无重复”“覆盖 t”都满足；含负数的“和”不满足。

        **延伸**：有负数时换成前缀和：“和恰为 K 的子数组个数”（LC560）用前缀和 + 哈希表；“和 ≥ K 的最短子数组”（LC862）用前缀和 + 单调递增队列，仍是 $O(n)$。
      `,
    },
    {
      id: 'qa-longest-shortest',
      freq: 2,
      q: md`滑动窗口“求最长”和“求最短”的模板有什么区别？`,
      a: md`
        **结论**：区别在 while 的条件和更新答案的位置。

        **原因**：求最长时窗口应保持合法，while 在**不合法**时收缩，收缩结束后窗口合法，此时更新；求最短时 while 在**合法**时收缩，每次收缩前窗口都合法，所以在循环内更新。

        **延伸**：统计“合法子数组个数”时，若条件单调，每个 r 贡献 \`r - l + 1\` 个；“恰好 K 个”的计数可以转成“至多 K 个 − 至多 K−1 个”（LC992）。
      `,
    },
    {
      id: 'qa-missing',
      freq: 2,
      q: md`最小覆盖子串里，为什么要单独维护 missing，而不是直接比较两个计数表？`,
      a: md`
        **结论**：missing 把“窗口是否覆盖 t”的判断从 $O(|\Sigma|)$ 降到 $O(1)$。

        **原因**：need 允许为负，只有当 \`need[c]\` 在 \`>0\` 与 \`≤0\` 之间跨越时 missing 才变化：进窗时 \`need[c] > 0\` 才减，出窗后 \`need[c] > 0\` 才加。这样 missing 始终等于“还缺的字符总数”，\`missing == 0\` 即覆盖。

        **延伸**：这是“用一个整数维护整张表的摘要”的通用技巧，LC438 的 \`valid\` 种类数、LC567 字符串排列都可以这样优化。
      `,
    },
    {
      id: 'qa-swa',
      freq: 2,
      q: md`滑动窗口注意力（SWA）的复杂度、感受野和 KV cache 各是多少？有什么代价？`,
      a: md`
        **结论**：每层计算 $O(nW)$，堆叠 $k$ 层后理论感受野约 $kW$，KV cache 每层固定 $W$ 个（rolling buffer）。

        **原因**：Mistral 7B 用 $W = 4096$、32 层，论文称理论跨度约 131K token；rolling buffer 把位置 $i$ 写入槽 $i \bmod W$，32K 序列 cache 降为 1/8。远距离信息需要逐层传递，越远越被稀释。

        **延伸**：所以主流做法是局部 + 全局混合（Gemma 2 隔层交替，Gemma 3 为 5:1 且窗口 1024）。另外 SWA 下 PagedAttention 等 KV 管理需要能回收窗口外的 block；attention sink（保留开头几个 token）是 StreamingLLM 对纯窗口注意力的另一种补救。
      `,
    },
  ],

  cheatsheet: md`
    **可变窗口模板**
    \`\`\`python
    l = 0
    for r, x in enumerate(a):
        加入 x
        while 需要收缩:            # 最短：合法时收缩；最长：不合法时收缩
            (最短) 更新答案
            移出 a[l]; l += 1
        (最长) 更新答案
    \`\`\`
    | 题 | 状态 | 关键点 |
    |---|---|---|
    | 209 最短子数组 | 窗口和 | 仅正数可用 |
    | 3 最长无重复 | \`last[c]\` | 跳跃需 \`last[c] >= l\` |
    | 76 最小覆盖 | \`need\` + \`missing\` | need 可为负，O(1) 判覆盖 |
    | 438 异位词 | 固定长度 k | 进一个出一个 |

    - 负数：前缀和 + 哈希（560）/ 单调队列（862）
    - SWA：每层 $O(nW)$，感受野 ≈ $kW$（Mistral 4096×32 ≈ 131K），cache 槽 = $i \bmod W$
    - n-gram 去重：最后 $n-1$ 个 token 为键查后继集合
  `,
})
