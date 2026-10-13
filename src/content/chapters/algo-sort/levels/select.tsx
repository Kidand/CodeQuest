/**
 * 快速选择与线性排序（algo-sort-select）
 * LC215 / LC164：quickselect 期望 O(n)、比较排序下界、计数 / 基数 / 桶排序、复杂度总表；大模型岗场景：argpartition。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { QuickSelectViz } from '../viz/QuickSelectViz'
import { RadixViz } from '../viz/RadixViz'
import { ArgpartitionViz } from '../viz/ArgpartitionViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'quickselect',
      title: '快速选择：快排只递归一侧',
      body: md`
        找第 k 大（LC215）不需要把整个数组排好序。快排的一次划分后 pivot 落在最终位置 \`p\`：

        - \`p\` 恰好是目标下标 $n-k$ → 找到；
        - \`p < n-k\` → 目标在右侧，**左侧整体丢弃**；否则只进左侧。

        与快排的区别只是“两侧都递归”变成“只进一侧”。若每次 pivot 大致居中，工作量是几何级数：

        $$
        n + \frac{n}{2} + \frac{n}{4} + \cdots < 2n = O(n)
        $$

        随机 pivot 下的严格结论：找中位数的期望比较次数约 $2(1 + \ln 2)\,n \approx 3.39n$，找最大/最小附近的元素约 $2n$。
        最坏情况与快排一样是 $O(n^2)$（有序输入 + 固定 pivot，或大量重复值 + 二路划分）。
      `,
      viz: QuickSelectViz,
      after: md`
        > [!TIP] 写成循环，而不是递归
        > 快速选择每次只进一侧，天然是尾递归，直接用 \`while True\` 维护 \`lo, hi\` 即可，空间 $O(1)$，也不会撞上 Python 的递归上限。
      `,
      checkpoint: {
        id: 'cp-quickselect',
        type: 'numeric',
        prompt: md`若每次 pivot 恰好把区间对半分、且只进入一侧，对 $n = 1024$ 个元素做快速选择，所有轮次扫描的区间长度之和最多是多少？`,
        answer: 2047,
        tolerance: 0,
        explanation: md`$1024 + 512 + 256 + \dots + 1 = 2^{11} - 1 = 2047 < 2n$。对比全排序约 $n \log_2 n = 10240$ 次。`,
      },
    },
    {
      id: 'worstcase',
      title: '最坏情况与工程实现：随机化、BFPRT、introselect',
      body: md`
        | 方法 | 期望 | 最坏 | 说明 |
        |---|---|---|---|
        | 固定 pivot | $O(n)$（随机输入） | $O(n^2)$ | 有序输入每轮只少一个元素 |
        | 随机 pivot | $O(n)$ | $O(n^2)$（概率极小） | 面试标准写法 |
        | 随机 pivot + 三路划分 | $O(n)$ | 同上 | 大量重复值也稳定 |
        | **BFPRT**（中位数的中位数） | $O(n)$ | $O(n)$ | 每 5 个一组取中位数，再递归求这些中位数的中位数作 pivot；常数大 |
        | **introselect** | $O(n)$ | $O(n\log n)$ 或 $O(n)$ | 快速选择为主，递归过深时切换到保证最坏界的算法；NumPy \`partition\` 的默认算法 |

        BFPRT（Blum–Floyd–Pratt–Rivest–Tarjan，1973）的关键：选出的 pivot 至少大于约 $3n/10$ 个元素、也至少小于约 $3n/10$ 个，
        于是 $T(n) \le T(n/5) + T(7n/10) + O(n)$，因为 $1/5 + 7/10 < 1$，解得 $T(n) = O(n)$。面试中能讲清这个递推即可，很少要求手写。

        **堆 vs 快速选择**：小顶堆 $O(n\log k)$、可流式、不改原数组；快速选择期望 $O(n)$、需随机访问且会打乱数组。数据量巨大或流式到达时选堆。
      `,
      checkpoint: {
        id: 'cp-worstcase',
        type: 'multi',
        prompt: md`下面哪些情况会让“末尾元素作 pivot + Lomuto 二路划分”的快速选择退化到 $O(n^2)$？（多选）`,
        options: ['输入已升序，求中位数', '所有元素都相同', '输入随机排列', '输入已降序，求中位数'],
        answer: [0, 1, 3],
        explanation: md`有序 / 逆序输入时 pivot 总是区间极值，每轮只排除一个元素；全相同时二路划分把所有元素放到一侧。随机输入期望 $O(n)$。修复：随机 pivot 解决有序输入，三路划分解决重复值。`,
      },
    },
    {
      id: 'linear',
      title: '突破 Ω(n log n)：计数、基数、桶排序',
      body: md`
        **比较排序的下界**：任何只靠比较的排序算法都对应一棵决策树，叶子至少有 $n!$ 个（每种排列一个），树高（最坏比较次数）至少

        $$
        \log_2 n! = n\log_2 n - n\log_2 e + O(\log n) = \Omega(n \log n)
        $$

        想更快，就得利用**值本身的结构**，而不是只做比较：

        | 算法 | 时间 | 空间 | 稳定 | 前提 |
        |---|---|---|---|---|
        | 计数排序 | $O(n + V)$ | $O(V)$ | ✅（反向填充） | 整数、值域 $V$ 不大 |
        | 基数排序（LSD） | $O(d(n + b))$ | $O(n + b)$ | ✅ | 定长整数 / 字符串，每一位用稳定排序 |
        | 桶排序 | 平均 $O(n)$ | $O(n)$ | 取决于桶内排序 | 数据大致均匀分布 |

        LSD 基数排序从最低位开始，每一位做一次**稳定**的分桶。为什么必须稳定？按十位分桶时，十位相同的元素之间要保持“上一轮按个位排好”的顺序，否则低位的排序成果被打乱。

        **LC164 最大间距**是桶思想的经典应用：$n$ 个数的最大相邻差（排序后）至少是平均间距 $\frac{\max - \min}{n-1}$。
        把桶宽设为不超过这个平均值，则**同一个桶内的差一定小于答案**，只需比较“前一个非空桶的最大值”与“后一个非空桶的最小值”，$O(n)$。
      `,
      viz: RadixViz,
      after: md`
        > [!NOTE] 这些“线性”排序的代价
        > 计数排序在值域 $V \gg n$ 时反而更慢、更费内存；基数排序的 $d$ 对 64 位整数（按字节）是 8，常数不小。NumPy 对布尔与窄整数类型的稳定排序使用基数排序，其它类型的稳定排序用 timsort。
      `,
      checkpoint: {
        id: 'cp-linear',
        type: 'fill',
        prompt: md`对 \`[170, 45, 75, 90, 802, 24, 2, 66]\` 做 LSD 基数排序，第一轮（按个位）收集后的序列是 ____(1)____（形如 \`[a,b,c]\`）。`,
        blanks: [{ label: '(1)', answers: ['[170,90,802,2,24,45,75,66]', '170,90,802,2,24,45,75,66'] }],
        explanation: md`个位为 0：170、90；为 2：802、2；为 4：24；为 5：45、75；为 6：66。同一个桶内保持原相对顺序（170 在 90 前、802 在 2 前），得到 \`[170, 90, 802, 2, 24, 45, 75, 66]\`。`,
      },
    },
    {
      id: 'llm',
      title: '复杂度总表 + 大模型岗场景：argpartition 取 top-k',
      body: md`
        | 算法 | 平均 | 最坏 | 额外空间 | 稳定 |
        |---|---|---|---|---|
        | 冒泡 / 插入 | $O(n^2)$ | $O(n^2)$ | $O(1)$ | ✅ |
        | 选择 | $O(n^2)$ | $O(n^2)$ | $O(1)$ | ❌ |
        | 快速排序 | $O(n\log n)$ | $O(n^2)$ | $O(\log n)$ 栈 | ❌ |
        | 归并排序 | $O(n\log n)$ | $O(n\log n)$ | $O(n)$（链表 $O(1)$） | ✅ |
        | 堆排序 | $O(n\log n)$ | $O(n\log n)$ | $O(1)$ | ❌ |
        | Timsort | $O(n\log n)$ | $O(n\log n)$ | $O(n)$ | ✅（有序输入 $O(n)$） |
        | 计数 / 基数 | $O(n+V)$ / $O(d(n+b))$ | 同左 | $O(n+V)$ | ✅ |
        | 快速选择 | $O(n)$ | $O(n^2)$ | $O(1)$ | — |

        **大模型岗场景**：top-k 采样、评测时取 top-k 预测、检索时取相似度最高的 k 个文档，都只需要“哪 k 个最大”，不需要全排序。
        NumPy 中用 \`np.argpartition\`（默认 introselect）：

        \`\`\`python
        idx = np.argpartition(-logits, k - 1)[:k]     # top-k 下标（组内无序），期望 O(V)
        idx = idx[np.argsort(-logits[idx])]           # 需要有序时再排这 k 个：O(k log k)
        \`\`\`

        词表 $V$ 约 15 万、$k = 50$ 时，\`argsort\` 要做 $O(V\log V)$ 的全排序，而 argpartition + 小排序是 $O(V + k\log k)$。
        PyTorch 中对应 \`torch.topk\`（返回值默认已按大小排序，\`sorted=False\` 可跳过排序）。
      `,
      viz: ArgpartitionViz,
      after: md`
        > [!WARN] argpartition 的保证只有一条
        > 第 kth 个位置上的元素就是排好序后该位置的元素，它之前的都不大于它、之后的都不小于它；**两侧内部的顺序未定义**。直接把 \`argpartition(...)[:k]\` 当作“按分数排好序的 top-k”是常见 bug。
      `,
      checkpoint: {
        id: 'cp-llm',
        type: 'single',
        prompt: md`\`logits\` 长度为 $V$，下面哪段代码能以 $O(V + k\log k)$ 得到**按分数降序**的 top-k 下标？`,
        options: [
          '`i = np.argpartition(-logits, k-1)[:k]; i = i[np.argsort(-logits[i])]`',
          '`np.argpartition(-logits, k-1)[:k]`',
          '`np.argsort(-logits)[:k]`',
          '`np.argpartition(logits, k)[:k]`',
        ],
        answer: 0,
        explanation: md`A 先期望 $O(V)$ 选出 top-k，再对 k 个排序。B 集合正确但无序；C 正确但是 $O(V \log V)$；D 选出的是**最小的** k 个（且 kth=k 时取的是前 k 个 ≤ 第 k+1 小的元素）。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-complexity',
      type: 'single',
      difficulty: 1,
      prompt: md`随机 pivot 的快速选择，期望时间与最坏时间分别是：`,
      options: ['$O(n)$ 与 $O(n^2)$', '$O(n\\log n)$ 与 $O(n^2)$', '$O(n)$ 与 $O(n)$', '$O(\\log n)$ 与 $O(n)$'],
      answer: 0,
      explanation: md`只进入一侧使期望工作量为几何级数 $O(n)$；随机化不改变最坏界 $O(n^2)$。最坏 $O(n)$ 需要 BFPRT（中位数的中位数）。`,
    },
    {
      id: 'q-lower-bound',
      type: 'judge',
      difficulty: 1,
      prompt: md`$\Omega(n \log n)$ 的排序下界同样适用于计数排序和基数排序。`,
      answer: false,
      explanation: md`错误。这个下界只针对**基于比较**的排序（决策树模型）。计数 / 基数排序利用了键的数值结构（作为数组下标、按位分桶），不受此限制，可以做到 $O(n + V)$ 或 $O(d(n + b))$。`,
    },
    {
      id: 'q-stable',
      type: 'multi',
      difficulty: 2,
      prompt: md`以下哪些排序算法（按标准实现）是**稳定**的？（多选）`,
      options: ['归并排序', '插入排序', '堆排序', 'LSD 基数排序', '快速排序', '选择排序'],
      answer: [0, 1, 3],
      explanation: md`归并（合并时相等取左边）、插入（只交换严格逆序的相邻元素）、LSD 基数（每一位用稳定的分配 / 收集）是稳定的。堆排序、快速排序、选择排序都有远距离交换，不稳定——例如选择排序 \`[2a, 2b, 1]\` 第一步把 1 与 2a 交换，得到 \`[1, 2b, 2a]\`。`,
    },
    {
      id: 'q-log-fact',
      type: 'numeric',
      difficulty: 2,
      prompt: md`对 8 个互不相同的元素做比较排序，最坏情况下至少需要多少次比较？（信息论下界 $\lceil \log_2 8! \rceil$）`,
      answer: 16,
      tolerance: 0,
      explanation: md`$8! = 40320$，$\log_2 40320 \approx 15.3$，向上取整得 16。这个下界对 $n = 8$ 是可以达到的（Ford–Johnson 归并插入排序）。`,
    },
    {
      id: 'q-radix',
      type: 'fill',
      difficulty: 2,
      prompt: md`对 \`[329, 457, 657, 839, 436, 720, 355]\` 做 LSD 基数排序。第一轮（个位）后第 1 个元素是 ____(1)____；第二轮（十位）后第 1 个元素是 ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['720'], numeric: true },
        { label: '(2)', answers: ['720'], numeric: true },
      ],
      explanation: md`
        个位：720(0)、355(5)、436(6)、457/657(7)、329/839(9) → \`[720, 355, 436, 457, 657, 329, 839]\`。
        十位：720/329(2)、436/839(3)、355/457/657(5) → \`[720, 329, 436, 839, 355, 457, 657]\`，第 1 个仍是 720（十位 2 的桶中它先进入）。第三轮后才完全有序。
      `,
    },
    {
      id: 'q-qs-steps',
      type: 'order',
      difficulty: 2,
      prompt: md`把“快速选择求第 k 大”一轮循环内的步骤按顺序排列：`,
      items: [
        '计算目标下标 target = n - k（只在开始时做一次）',
        '在 [lo, hi] 中随机选 pivot 并划分，得到 pivot 的最终位置 p',
        '若 p == target，返回 nums[p]',
        '若 p < target，令 lo = p + 1；否则令 hi = p - 1',
      ],
      explanation: md`第 k 大对应升序下标 \`n - k\`（例如 k = 1 是最大值，下标 n-1）。每轮划分后根据 p 与 target 的关系只保留一侧。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把算法与它的时间复杂度配对：`,
      pairs: [
        ['计数排序（值域 V）', '$O(n + V)$'],
        ['LSD 基数排序（d 位，基数 b）', '$O(d(n + b))$'],
        ['堆排序', '最坏 $O(n \\log n)$'],
        ['随机快速选择', '期望 $O(n)$'],
        ['用大小为 k 的堆求 top-k', '$O(n \\log k)$'],
      ],
      explanation: md`注意区分“期望”与“最坏”：快速选择期望 $O(n)$、最坏 $O(n^2)$；堆排序最坏也是 $O(n\log n)$。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出循环版快速选择（第 k 大，Lomuto 划分；有 2 行干扰项）：`,
      lines: [
        'def kth_largest(nums, k):',
        '    target, lo, hi = len(nums) - k, 0, len(nums) - 1',
        '    while True:',
        '        r = random.randint(lo, hi)',
        '        nums[r], nums[hi] = nums[hi], nums[r]',
        '        p = lo',
        '        for j in range(lo, hi):',
        '            if nums[j] < nums[hi]:',
        '                nums[p], nums[j] = nums[j], nums[p]',
        '                p += 1',
        '        nums[p], nums[hi] = nums[hi], nums[p]',
        '        if p == target:',
        '            return nums[p]',
        '        lo, hi = (p + 1, hi) if p < target else (lo, p - 1)',
      ],
      distractors: ['    target = k - 1', '        lo, hi = (lo, p - 1) if p < target else (p + 1, hi)'],
      explanation: md`第 k 大在升序数组中的下标是 \`n - k\`（\`k - 1\` 是第 k 小）。\`p < target\` 时目标在右侧，应保留 \`[p+1, hi]\`；第二个干扰项方向反了。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的“第 k 小”快速选择在很多输入上返回错误结果或死循环。点出有问题的行：`,
      code: py`
        import random

        def kth_smallest(nums, k):
            target, lo, hi = k - 1, 0, len(nums) - 1
            while lo <= hi:
                r = random.randint(lo, hi)
                nums[r], nums[hi] = nums[hi], nums[r]
                p = lo
                for j in range(lo, hi):
                    if nums[j] < nums[hi]:
                        nums[p], nums[j] = nums[j], nums[p]
                        p += 1
                nums[p], nums[hi] = nums[hi], nums[p]
                if p == target:
                    return nums[p]
                elif p < target:
                    hi = p - 1
                else:
                    lo = p + 1
      `,
      bugLines: [17, 19],
      explanation: md`第 17、19 行方向写反了：\`p < target\` 说明目标在 pivot **右侧**，应 \`lo = p + 1\`；\`p > target\` 时应 \`hi = p - 1\`。写反后目标被丢弃，最终 \`lo > hi\` 退出循环返回 None。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`\`logits = [0.3, 2.5, -1.0, 1.8, 2.2, 0.9, 3.1, -0.4]\`，\`np.argpartition(-logits, 2)[:3]\` 返回的下标集合包含哪些？（点出全部）`,
      items: ['0: 0.3', '1: 2.5', '2: -1.0', '3: 1.8', '4: 2.2', '5: 0.9', '6: 3.1', '7: -0.4'],
      answer: [1, 4, 6],
      explanation: md`kth = 2 表示 \`-logits\` 中第 3 小（即 logits 第 3 大）的元素放到位置 2，前面的都不大于它（对 \`-logits\` 而言），所以前 3 个是 logits 最大的 3 个：3.1（下标 6）、2.5（下标 1）、2.2（下标 4）。返回顺序未定义。`,
    },
    {
      id: 'q-heap-vs-qs',
      type: 'single',
      difficulty: 3,
      prompt: md`服务端日志中有 $10^9$ 条请求延迟（无法全部装入内存），需要求 P99.9 延迟附近最慢的 1000 条。最合适的做法是：`,
      options: [
        '一遍扫描，维护大小为 1000 的小顶堆',
        '全部读入后用快速选择',
        '全部读入后用基数排序',
        '对每条记录做二分插入到有序数组',
      ],
      answer: 0,
      explanation: md`数据无法放进内存时，快速选择和基数排序都不可行；大小为 k 的堆只需 $O(k)$ 内存、$O(n\log k)$ 时间、单遍流式。D 每次插入 $O(k)$ 的移动，总 $O(nk)$。若只要求近似分位数，还可以用 t-digest 等草图结构。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-quickselect',
      title: '一步步写出快速选择（第 k 大）',
      intro: md`
        实现 \`kth_largest(nums, k)\`：返回数组中第 k 大的元素（$1 \le k \le n$，可以修改 \`nums\`）。随机 pivot + Lomuto 划分，循环实现。
      `,
      template: py`
        import random

        def kth_largest(nums, k):
            target = {{1}}                      # 第 k 大在升序数组中的下标
            lo, hi = 0, len(nums) - 1
            while True:
                r = random.randint(lo, hi)
                nums[r], nums[hi] = nums[hi], nums[r]
                pivot, i = nums[hi], lo
                for j in range(lo, hi):
                    if nums[j] < pivot:
                        nums[i], nums[j] = nums[j], nums[i]
                        i += 1
                nums[i], nums[hi] = nums[hi], nums[i]
                if {{2}}:
                    return nums[i]
                elif {{3}}:
                    {{4}}
                else:
                    {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`第 k 大在升序排列中的下标。`,
          answer: 'len(nums) - k',
          explain: md`k = 1（最大值）对应下标 n-1；k = n（最小值）对应下标 0。`,
        },
        {
          blank: 2,
          instruction: md`pivot 恰好落在目标位置。`,
          answer: 'i == target',
          accept: ['target == i'],
        },
        {
          blank: 3,
          instruction: md`pivot 落在目标左边，目标在哪一侧？`,
          answer: 'i < target',
          accept: ['target > i'],
        },
        {
          blank: 4,
          instruction: md`只保留右侧。`,
          answer: 'lo = i + 1',
          explain: md`左侧 \`[lo, i]\` 的元素都 ≤ pivot，不可能是目标，整体丢弃。`,
        },
        {
          blank: 5,
          instruction: md`只保留左侧。`,
          answer: 'hi = i - 1',
        },
      ],
      setup: py`
        import random as _r
      `,
      tests: [
        {
          name: '示例',
          code: py`
            assert kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
            assert kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = _r.Random(215)
            for _ in range(300):
                a = [rnd.randint(-20, 20) for _ in range(rnd.randint(1, 25))]
                k = rnd.randint(1, len(a))
                exp = sorted(a, reverse=True)[k - 1]
                got = kth_largest(list(a), k)
                assert got == exp, f"nums={a}, k={k}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        这是二路划分版本。实战题的隐藏测试包含“全相同”数组，二路划分会退化到 $O(n^2)$——请升级成三路划分：
        \`target\` 落在 \`[lt, gt]\` 中即可直接返回。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-kth-largest',
      title: '数组中的第 K 个最大元素（快速选择）',
      difficulty: 'medium',
      source: 'LeetCode 215',
      timeLimitMin: 20,
      description: md`
        返回数组 \`nums\` 中第 \`k\` 大的元素（排序后的第 k 大，不是第 k 个不同的值）。

        \`\`\`python
        def find_kth_largest(nums: list, k: int): ...
        \`\`\`

        **要求**
        - 期望 $O(n)$：用快速选择；不能使用 \`sorted\`、\`.sort\`、\`heapq\`（会检查源码）；
        - 隐藏测试用“计数比较次数”的元素包装输入（$n = 16384$），包括**已排序**数组与**全相同**数组，比较次数超过 $12n$ 立即失败——
          固定 pivot 或二路划分会被卡住；
        - 元素只支持比较运算；可以修改 \`nums\`；返回 \`nums\` 中的元素本身。

        **示例**
        - \`find_kth_largest([3,2,1,5,6,4], 2)\` → \`5\`
        - \`find_kth_largest([3,2,3,1,2,4,5,5,6], 4)\` → \`4\`
      `,
      starter: py`
        import random

        def find_kth_largest(nums: list, k: int):
            # TODO: 随机 pivot + 三路划分的快速选择，循环实现
            raise NotImplementedError
      `,
      solution: py`
        import random

        def find_kth_largest(nums: list, k: int):
            target = len(nums) - k                    # 升序下的目标下标
            lo, hi = 0, len(nums) - 1
            while True:
                p = nums[random.randint(lo, hi)]      # 随机 pivot（取值）
                lt, i, gt = lo, lo, hi                # 三路划分
                while i <= gt:
                    if nums[i] < p:
                        nums[lt], nums[i] = nums[i], nums[lt]
                        lt += 1
                        i += 1
                    elif nums[i] > p:
                        nums[i], nums[gt] = nums[gt], nums[i]
                        gt -= 1
                    else:
                        i += 1
                # [lo, lt) < p，[lt, gt] == p，(gt, hi] > p
                if target < lt:
                    hi = lt - 1
                elif target > gt:
                    lo = gt + 1
                else:
                    return nums[target]               # 落在等于 p 的段内
      `,
      solutionAlt: [
        {
          label: '堆解法（O(n log k)，面试中先说这个也可以）',
          code: py`
            import heapq

            def find_kth_largest(nums, k):
                h = nums[:k]
                heapq.heapify(h)                      # 大小为 k 的小顶堆
                for x in nums[k:]:
                    if x > h[0]:
                        heapq.heapreplace(h, x)
                return h[0]
          `,
        },
        {
          label: 'NumPy / PyTorch（仅展示）',
          code: py`
            import numpy as np
            # 第 k 大：partition 后位置 n-k 上就是答案，期望 O(n)
            def kth_np(a, k):
                a = np.asarray(a)
                return np.partition(a, a.size - k)[a.size - k]

            import torch
            def kth_torch(a: torch.Tensor, k: int):
                return torch.topk(a, k).values[-1]    # 或 torch.kthvalue(a, a.numel() - k + 1)
          `,
        },
      ],
      explanation: md`
        **思路**：快排的划分 + 只进入目标所在一侧。三路划分后 \`[lt, gt]\` 都等于 pivot：目标下标落在其中就直接返回，否则只保留一侧。

        **为什么要随机 pivot + 三路划分**：已排序输入上固定 pivot 每轮只排除一个元素；全相同输入上二路划分同样每轮只排除一个。两者都是 $O(n^2)$。

        **复杂度**：期望 $O(n)$（三路划分的比较次数约为二路的 1.5～2 倍，仍是线性），额外空间 $O(1)$。

        **常见错误**：把第 k 大当成下标 \`k - 1\`；递归实现在大输入上爆栈；三路划分中与 gt 交换后 i 前进。
      `,
      complexity: { time: '期望 O(n)', space: 'O(1)' },
      hints: [
        md`第 k 大在升序数组中的下标是 \`n - k\`。快排划分后 pivot 的位置是确定的——目标在它的哪一边？`,
        md`用 \`while True\` 维护 \`[lo, hi]\`；每轮随机选 pivot。`,
        md`三路划分得到 \`[lt, gt]\`：\`target < lt\` 则 \`hi = lt - 1\`，\`target > gt\` 则 \`lo = gt + 1\`，否则返回。`,
      ],
      setup: py`
        import linecache, random as _rnd

        class _K:
            """计数比较次数的包装类型：超过预算立即失败"""
            __slots__ = ("v",)
            n = 0
            budget = 10**12
            def __init__(self, v):
                self.v = v
            @staticmethod
            def _tick():
                _K.n += 1
                if _K.n > _K.budget:
                    raise AssertionError(f"比较次数超过 {_K.budget}：不是期望 O(n)（固定 pivot？二路划分遇到重复值？全排序？）")
            def __lt__(self, o): _K._tick(); return self.v < o.v
            def __le__(self, o): _K._tick(); return self.v <= o.v
            def __gt__(self, o): _K._tick(); return self.v > o.v
            def __ge__(self, o): _K._tick(); return self.v >= o.v
            def __eq__(self, o): _K._tick(); return self.v == o.v
            def __ne__(self, o): _K._tick(); return self.v != o.v
            __hash__ = None

        def _run_budget(vals, k, factor=12):
            exp = sorted(vals, reverse=True)[k - 1]
            a = [_K(v) for v in vals]
            _K.n, _K.budget = 0, factor * len(a)
            try:
                got = find_kth_largest(a, k)
            finally:
                _K.budget = 10**12
            got = getattr(got, "v", got)
            assert got == exp, f"期望 {exp}，得到 {got}"
      `,
      tests: [
        { name: '示例 1', code: py`assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5` },
        { name: '示例 2', code: py`assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4` },
        {
          name: '边界',
          code: py`
            assert find_kth_largest([1], 1) == 1
            assert find_kth_largest([2, 1], 1) == 2 and find_kth_largest([2, 1], 2) == 1
            assert find_kth_largest([-1, -1, -1], 2) == -1
          `,
        },
        {
          name: '不使用库排序',
          code: py`
            src = "".join(linecache.getlines("solution.py"))
            for banned in ("sorted(", ".sort(", "heapq", "nlargest"):
                assert banned not in src, f"请用快速选择，不要使用 {banned}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = _rnd.Random(215)
            for _ in range(300):
                a = [rnd.randint(-30, 30) for _ in range(rnd.randint(1, 40))]
                k = rnd.randint(1, len(a))
                exp = sorted(a, reverse=True)[k - 1]
                got = find_kth_largest(list(a), k)
                assert got == exp, f"nums={a}, k={k}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '已排序 n = 16384，求中位数（卡固定 pivot）',
          hidden: true,
          code: py`_run_budget(list(range(16384)), 8192)`,
        },
        {
          name: '全相同 n = 16384（卡二路划分）',
          hidden: true,
          code: py`_run_budget([5] * 16384, 100)`,
        },
      ],
      followUps: [
        {
          id: 'f-bfprt',
          q: md`如何把快速选择的最坏复杂度降到 $O(n)$？`,
          a: md`BFPRT：5 个一组取中位数，递归求这些中位数的中位数作为 pivot。它保证 pivot 两侧各至少约 $3n/10$ 个元素，于是 $T(n) \le T(n/5) + T(7n/10) + O(n) = O(n)$。常数较大，工程上用 introselect：先随机快速选择，递归过深再切换。`,
        },
        {
          id: 'f-stream',
          q: md`如果数据是流式的、不能全部存下呢？`,
          a: md`用大小为 k 的小顶堆，$O(n\log k)$ 时间、$O(k)$ 空间。若只需近似分位数，可用 t-digest、GK 草图等，空间与 n 无关。`,
        },
      ],
    },
    {
      id: 'c-max-gap',
      title: '最大间距（线性时间）',
      difficulty: 'hard',
      source: 'LeetCode 164',
      timeLimitMin: 25,
      description: md`
        给定非负整数数组 \`nums\`，返回其排序后相邻元素之间的**最大差值**；元素少于 2 个时返回 0。

        **要求线性时间与线性空间**：不能使用 \`sorted\` / \`.sort\`（会检查源码），请用桶（鸽笼原理）或基数排序。

        \`\`\`python
        def maximum_gap(nums: list[int]) -> int: ...
        \`\`\`

        **示例**
        - \`maximum_gap([3, 6, 9, 1])\` → \`3\`（排序后 \`[1,3,6,9]\`）
        - \`maximum_gap([10])\` → \`0\`

        **约束**：$1 \le n \le 10^5$，$0 \le \text{nums}[i] \le 10^9$。
      `,
      starter: py`
        def maximum_gap(nums: list[int]) -> int:
            # TODO: 桶宽 = max(1, (max - min) // (n - 1))；只比较相邻非空桶的 (min, max)
            raise NotImplementedError
      `,
      solution: py`
        def maximum_gap(nums: list[int]) -> int:
            n = len(nums)
            if n < 2:
                return 0
            lo, hi = min(nums), max(nums)
            if lo == hi:
                return 0
            # 桶宽不超过平均间距 ⇒ 桶内差 < 答案，最大间距只出现在相邻非空桶之间
            size = max(1, (hi - lo) // (n - 1))
            cnt = (hi - lo) // size + 1
            bmin = [None] * cnt
            bmax = [None] * cnt
            for x in nums:
                b = (x - lo) // size
                if bmin[b] is None:
                    bmin[b] = bmax[b] = x
                else:
                    if x < bmin[b]:
                        bmin[b] = x
                    if x > bmax[b]:
                        bmax[b] = x
            ans, prev = 0, lo
            for b in range(cnt):
                if bmin[b] is None:
                    continue
                ans = max(ans, bmin[b] - prev)    # 当前桶最小值 - 前一个非空桶最大值
                prev = bmax[b]
            return ans
      `,
      explanation: md`
        **关键不等式**：$n$ 个数、极差 $R = \max - \min$，相邻差共 $n - 1$ 个，其平均为 $R/(n-1)$，所以最大间距 $\ge \lceil R/(n-1) \rceil$。

        **桶的设计**：桶宽 $s = \max(1, \lfloor R/(n-1) \rfloor) \le$ 平均间距 $\le$ 答案。同一个桶内两个数的差至多 $s - 1 <$ 答案，所以答案一定出现在**相邻非空桶**之间：
        后一个桶的最小值 − 前一个非空桶的最大值。每个桶只需记录 min 和 max。

        **复杂度**：$O(n)$ 时间，桶数 $\le n$，$O(n)$ 空间。

        **另解**：基数排序（基数 $2^{16}$ 或 10）后线性扫描，$O(d \cdot n)$。

        **常见错误**：桶宽为 0（所有数相同或 $R < n-1$ 时要取 \`max(1, ...)\`）；只比较桶内差；用浮点除法计算桶号产生边界误差。
      `,
      complexity: { time: 'O(n)', space: 'O(n)' },
      hints: [
        md`排序是 $O(n\log n)$。想想 $n$ 个数的最大相邻差至少是多少？（平均间距）`,
        md`把值域切成宽度不超过平均间距的桶，同一个桶内的差还可能是答案吗？`,
        md`每个桶只记 min、max；遍历非空桶，答案 = max(当前桶 min − 上一个非空桶 max)。`,
      ],
      setup: py`
        import linecache, random, time

        def _ref_gap(a):
            s = sorted(a)
            return max((s[i + 1] - s[i] for i in range(len(s) - 1)), default=0)
      `,
      tests: [
        { name: '示例', code: py`assert maximum_gap([3, 6, 9, 1]) == 3 and maximum_gap([10]) == 0` },
        {
          name: '边界：相同元素 / 两个元素 / 0',
          code: py`
            assert maximum_gap([5, 5, 5]) == 0
            assert maximum_gap([1, 10000000]) == 9999999
            assert maximum_gap([0, 0, 1, 1]) == 1
            assert maximum_gap([1, 3, 100]) == 97
          `,
        },
        {
          name: '不使用库排序',
          code: py`
            src = "".join(linecache.getlines("solution.py"))
            for banned in ("sorted(", ".sort("):
                assert banned not in src, f"要求线性时间，不要使用 {banned}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(164)
            for _ in range(400):
                a = [rnd.randint(0, rnd.choice([5, 50, 10**9])) for _ in range(rnd.randint(1, 20))]
                exp, got = _ref_gap(a), maximum_gap(list(a))
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：n = 10^5',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            a = [rnd.randint(0, 10**9) for _ in range(100_000)]
            exp = _ref_gap(a)
            t0 = time.perf_counter()
            got = maximum_gap(a)
            dt = time.perf_counter() - t0
            assert got == exp, f"期望 {exp}，得到 {got}"
            assert dt < 2.0, f"用时 {dt:.2f}s"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-why-not-count',
          q: md`为什么这里不用计数排序？`,
          a: md`值域高达 $10^9$，计数数组 $O(V)$ 的时间和内存都不可接受。桶的数量只取决于 $n$（不超过 $n$ 个），这是“按数量分桶”而非“按值域分桶”的关键区别。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-qs',
      freq: 3,
      q: md`快速选择为什么期望是 O(n)？和快排有什么区别？`,
      a: md`
        **结论**：快排两侧都递归，快速选择只进入目标所在的一侧，期望总工作量是几何级数，$O(n)$。

        **原因**：随机 pivot 有一半概率落在中间一半位置，此时区间至少缩小到 $3/4$；期望下每轮扫描量形如 $n + cn + c^2 n + \dots$（$c < 1$），和为 $O(n)$。严格计算求中位数的期望比较次数约 $3.39n$。

        **延伸**：最坏仍是 $O(n^2)$；BFPRT 用“中位数的中位数”保证最坏 $O(n)$；工程上的 introselect（NumPy \`partition\` / \`argpartition\` 的默认算法）在递归过深时切换。全相同元素要配合三路划分。
      `,
    },
    {
      id: 'qa-lower',
      freq: 2,
      q: md`比较排序的下界是什么？计数排序为什么能突破它？`,
      a: md`
        **结论**：比较排序最坏需要 $\lceil \log_2 n! \rceil = \Omega(n\log n)$ 次比较；计数 / 基数排序不靠比较，因此不受限。

        **原因**：比较排序的执行过程是一棵二叉决策树，每个叶子对应一种输出排列，至少 $n!$ 个叶子，树高 $\ge \log_2 n!$。计数排序把键直接当数组下标，一次 $O(1)$ 操作获得的信息远多于一次比较的 1 bit。

        **延伸**：代价是对键有要求（整数、值域小、或可按位分解），并需要 $O(V)$ 或 $O(n + b)$ 额外空间。基数排序每一位必须用稳定排序。
      `,
    },
    {
      id: 'qa-stable-table',
      freq: 2,
      q: md`常见排序算法中哪些稳定？稳定性在什么场景下重要？`,
      a: md`
        **结论**：稳定：插入、冒泡、归并、计数、LSD 基数、Timsort；不稳定：选择、快排、堆排、希尔。

        **原因**：不稳定的算法都存在“跨越中间元素的远距离交换”，会改变相等元素的相对次序。

        **延伸**：多关键字排序依赖稳定性（先按次关键字排，再按主关键字稳定排）；LSD 基数排序本身就依赖每一位的稳定性。Python 的 \`sorted\` 稳定，所以 \`sorted(sorted(xs, key=b), key=a)\` 等价于按 \`(a, b)\` 排序。
      `,
    },
    {
      id: 'qa-argpartition',
      freq: 2,
      q: md`从 15 万维的 logits 中取 top-50 的下标，NumPy 和 PyTorch 分别怎么写？复杂度是多少？`,
      a: md`
        **结论**：NumPy 用 \`np.argpartition(-logits, k-1)[:k]\` 再对这 k 个排序，期望 $O(V + k\log k)$；PyTorch 用 \`torch.topk(logits, k)\`。

        **原因**：\`argsort\` 全排序是 $O(V\log V)$，而我们只关心最大的 k 个；argpartition 默认 introselect，只保证第 kth 位置正确、两侧分组正确，组内顺序未定义，所以需要有序时再排 k 个。\`torch.topk\` 在 GPU 上用并行选择算法实现，\`sorted=True\`（默认）时结果按值降序。

        **延伸**：批量处理时对 \`axis=-1\` 操作即可；top-p（nucleus）采样需要按概率排序后求前缀和，常先做 top-k 截断再在小集合上排序，降低排序代价。
      `,
    },
  ],

  cheatsheet: md`
    **快速选择（第 k 大，三路划分）**
    \`\`\`python
    target, lo, hi = len(a) - k, 0, len(a) - 1
    while True:
        p = a[random.randint(lo, hi)]; lt, i, gt = lo, lo, hi
        while i <= gt:                       # 三路划分
            if a[i] < p:   a[lt], a[i] = a[i], a[lt]; lt += 1; i += 1
            elif a[i] > p: a[i], a[gt] = a[gt], a[i]; gt -= 1
            else:          i += 1
        if target < lt: hi = lt - 1
        elif target > gt: lo = gt + 1
        else: return a[target]
    \`\`\`
    - 期望 $O(n)$（中位数 ≈ 3.39n 次比较），最坏 $O(n^2)$；BFPRT 最坏 $O(n)$
    - 比较排序下界 $\lceil\log_2 n!\rceil = \Omega(n\log n)$
    - 计数 $O(n+V)$ · LSD 基数 $O(d(n+b))$（每位必须稳定）· 桶平均 $O(n)$
    - 最大间距：桶宽 $\max(1, \lfloor R/(n-1) \rfloor)$，只看相邻非空桶
    - top-k：\`np.argpartition(-x, k-1)[:k]\`（组内无序）· \`torch.topk\`
  `,
})
