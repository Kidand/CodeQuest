/**
 * BOSS：查找综合（algo-search-boss）
 * 复盘全章 + LC4 两个正序数组的中位数（分割线二分）+ LC378 有序矩阵第 K 小（值域二分 + 阶梯计数）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { PartitionViz } from '../viz/PartitionViz'

export default defineLevel({
  learn: [
    {
      id: 'recap',
      title: '复盘：看到什么信号，用什么技巧',
      body: md`
        | 题目信号 | 技巧 | 核心不变量 / 前提 | 代表题 |
        |---|---|---|---|
        | 有序数组找位置、找边界 | 二分（lower / upper_bound） | 区间外的都已排除；谓词 F…FT…T | 704 / 34 / 35 |
        | 部分有序、山峰 | 二分 | 每步能判断答案在哪一半 | 33 / 153 / 162 |
        | “最小的最大值”“能否在 X 内完成” | 二分答案 | check(x) 单调，上界必可行 | 875 / 1011 / 410 |
        | 有序数组上的数对 | 对撞指针 | 每步排除一行或一列 | 167 / 15 / 11 / 42 |
        | 原地过滤、去重 | 同向读写指针 | \`[0, slow)\` 是结果 | 26 / 27 / 283 |
        | 连续子数组 / 子串 + 单调条件 | 滑动窗口 | l 只增不减 | 3 / 76 / 209 / 438 |

        **BOSS 题 LC4**：两个正序数组（长度 $m \le n$）的中位数，要求 $O(\log(m+n))$。

        不要想着“二分查找某个值”，而是**二分分割线**：在 A 中取前 $i$ 个、B 中取前 $j = \text{half} - i$ 个组成左半，$\text{half} = \lfloor (m+n+1)/2 \rfloor$。
        左半恰好是合并后最小的 half 个元素，当且仅当

        $$
        A[i-1] \le B[j] \quad\text{且}\quad B[j-1] \le A[i]
        $$

        （越界视为 $\pm\infty$）。若 $A[i-1] > B[j]$，A 给多了，\`i\` 要变小；若 $B[j-1] > A[i]$，A 给少了，\`i\` 要变大——单调，可以二分。
        在**较短**的数组上二分，保证 $j \in [0, n]$ 不越界，复杂度 $O(\log \min(m, n))$。
      `,
      viz: PartitionViz,
      after: md`
        > [!TIP] 另一种思路：找第 k 小
        > 把中位数转为“两个有序数组的第 k 小”：比较 \`A[k/2-1]\` 与 \`B[k/2-1]\`，较小者及其左边的 k/2 个元素一定不是第 k 小，整体丢掉，k 减半。$O(\log(m+n))$，思路更直观，但边界处理（某个数组不足 k/2 个）更繁琐。

        > [!INTERVIEW] 讲 LC4 的顺序
        > 先说 $O(m+n)$ 的归并做法（面试官默认你会），再说“中位数 = 把两个数组切成等长的两半”，最后给出分割线条件与二分方向。条件推导比代码更重要。
      `,
    },
  ],

  quiz: [
    {
      id: 'q-last-le',
      type: 'single',
      difficulty: 1,
      prompt: md`在升序数组中找“最后一个 ≤ t 的位置”，用“找第一个 True”模板时，谓词和返回值应该是：`,
      options: ['谓词 `nums[i] > t`，返回 `lo - 1`', '谓词 `nums[i] >= t`，返回 `lo - 1`', '谓词 `nums[i] <= t`，返回 `lo`', '谓词 `nums[i] > t`，返回 `lo`'],
      answer: 0,
      explanation: md`“最后一个 ≤ t” = “第一个 > t” − 1，即 upper_bound − 1。B 得到的是最后一个 < t；C 的谓词呈 T…TF…F，不符合“找第一个 True”模板；D 是 upper_bound 本身。`,
    },
    {
      id: 'q-lc4-facts',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 LC4 的分割线二分解法，哪些说法是正确的？（多选）`,
      options: [
        '应在较短的数组上二分，以保证 j = half - i 不会越界',
        '时间复杂度为 $O(\\log \\min(m, n))$',
        '`half = (m + n + 1) // 2` 时，总数为奇数则中位数是左半的最大值',
        '只需检查 `A[i-1] <= A[i]` 和 `B[j-1] <= B[j]` 两个条件',
        '若 `A[i-1] > B[j]`，应增大 i',
      ],
      answer: [0, 1, 2],
      explanation: md`D 检查的是数组内部的有序性，本来就成立，毫无意义——要检查的是**交叉**条件 \`A[i-1] <= B[j]\` 与 \`B[j-1] <= A[i]\`。E 方向反了：A 左边的元素太大，说明 A 给多了，应减小 i。`,
    },
    {
      id: 'q-median',
      type: 'numeric',
      difficulty: 1,
      prompt: md`\`nums1 = [1, 2]\`，\`nums2 = [3, 4]\`，中位数是多少？`,
      answer: 2.5,
      tolerance: 0,
      explanation: md`合并后 \`[1, 2, 3, 4]\`，偶数个，中位数 $(2 + 3)/2 = 2.5$。分割线：\`i = 2, j = 0\`，左半 \`[1, 2]\`、右半 \`[3, 4]\`，\`(max(2, -∞) + min(+∞, 3)) / 2 = 2.5\`。`,
    },
    {
      id: 'q-378-exists',
      type: 'judge',
      difficulty: 2,
      prompt: md`LC378 用“值域二分”求有序矩阵第 k 小：找最小的 x 使得“矩阵中 ≤ x 的元素个数 ≥ k”。这个 x 一定是矩阵中的某个元素。`,
      answer: true,
      explanation: md`正确。设第 k 小的元素为 v，则 count(v) ≥ k；对任意 x < v，count(x) < k（否则第 k 小会 ≤ x）。所以满足条件的最小 x 恰好是 v，必然在矩阵中——这也是为什么值域二分不需要额外验证“x 是否存在”。`,
    },
    {
      id: 'q-378-example',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        \`matrix = [[1, 5, 9], [10, 11, 13], [12, 13, 15]]\`。矩阵中 ≤ 12 的元素有 ____(1)____ 个；第 8 小的元素是 ____(2)____。
      `,
      blanks: [
        { label: '(1)', answers: ['6'], numeric: true },
        { label: '(2)', answers: ['13'], numeric: true },
      ],
      explanation: md`≤ 12 的有 1, 5, 9, 10, 11, 12，共 6 个；排序后为 1, 5, 9, 10, 11, 12, 13, 13, 15，第 8 个是 13（count(12) = 6 < 8，count(13) = 8 ≥ 8）。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与最合适的技巧配对：`,
      pairs: [
        ['LC4 两个正序数组的中位数', '在较短数组上二分分割线'],
        ['LC378 有序矩阵第 K 小', '值域二分 + 阶梯计数'],
        ['LC42 接雨水', '对撞指针，结算较矮一侧'],
        ['LC76 最小覆盖子串', '滑动窗口 + missing 计数'],
        ['LC1011 D 天内送达包裹', '二分答案 + 贪心 check'],
      ],
      explanation: md`这五道题覆盖了本章的全部主干技巧：二分（分割线 / 值域 / 答案空间）、对撞指针、滑动窗口。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LC378 中“统计矩阵里 ≤ x 的元素个数”的阶梯计数函数（从左下角出发，$O(n)$；有 2 行干扰项）：`,
      lines: [
        'def count_le(matrix, x):',
        '    n = len(matrix)',
        '    i, j, cnt = n - 1, 0, 0',
        '    while i >= 0 and j < n:',
        '        if matrix[i][j] <= x:',
        '            cnt += i + 1',
        '            j += 1',
        '        else:',
        '            i -= 1',
        '    return cnt',
      ],
      distractors: ['            cnt += 1', '    i, j, cnt = 0, 0, 0'],
      explanation: md`从左下角出发：若 \`matrix[i][j] <= x\`，则第 j 列从第 0 行到第 i 行全部 ≤ x，一次计入 \`i + 1\` 个并右移；否则上移。每步排除一行或一列——与对撞指针的“搜索空间矩阵”是同一个论证。\`cnt += 1\` 只计了一个；从左上角出发无法判断该往哪走。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 LC4 实现有一处逻辑错误（方向写反）。点出有问题的行：`,
      code: py`
        def find_median(A, B):
            if len(A) > len(B):
                A, B = B, A
            m, n = len(A), len(B)
            half = (m + n + 1) // 2
            lo, hi = 0, m
            while lo <= hi:
                i = (lo + hi) // 2
                j = half - i
                Al = A[i - 1] if i > 0 else float('-inf')
                Ar = A[i] if i < m else float('inf')
                Bl = B[j - 1] if j > 0 else float('-inf')
                Br = B[j] if j < n else float('inf')
                if Al > Br:
                    lo = i + 1
                elif Bl > Ar:
                    lo = i + 1
                else:
                    left = max(Al, Bl)
                    return left if (m + n) % 2 else (left + min(Ar, Br)) / 2
      `,
      bugLines: [15],
      explanation: md`第 15 行应为 \`hi = i - 1\`：\`A[i-1] > B[j]\` 说明 A 的左半含有过大的元素，A 贡献得太多，i 应减小。写成 \`lo = i + 1\` 会把搜索推向错误方向，最终 \`lo > hi\` 退出循环、函数返回 None。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`矩阵 \`[[1, 5, 9], [10, 11, 13], [12, 13, 15]]\` 中，阶梯计数 \`count_le(matrix, 12)\` 统计到的元素是哪些？（点出全部）`,
      items: ['1', '5', '9', '10', '11', '13', '12', '13', '15'],
      cols: 3,
      rowLabels: ['行 0', '行 1', '行 2'],
      colLabels: ['列 0', '列 1', '列 2'],
      answer: [0, 1, 2, 3, 4, 6],
      explanation: md`≤ 12 的元素构成左上方的一块“阶梯”：第 0 列 3 个（1, 10, 12）、第 1 列 2 个（5, 11）、第 2 列 1 个（9），共 6 个。阶梯形状正是“行列都有序”带来的单调性。`,
    },
    {
      id: 'q-rounds',
      type: 'numeric',
      difficulty: 2,
      prompt: md`LC875 中 \`max(piles) = 10^9\`，在 \`[1, 10^9]\` 上二分答案，最多调用多少次 check？`,
      answer: 30,
      tolerance: 0,
      explanation: md`区间长度 $10^9$，每轮减半，需要 $\lceil \log_2 10^9 \rceil = 30$ 轮（$2^{29} \approx 5.4 \times 10^8 < 10^9 < 2^{30} \approx 1.07 \times 10^9$）。每次 check 是 $O(n)$，总复杂度 $O(30n)$。`,
    },
    {
      id: 'q-neg',
      type: 'single',
      difficulty: 2,
      prompt: md`“统计和恰好为 K 的连续子数组个数，元素可正可负”（LC560），最合适的做法是：`,
      options: ['前缀和 + 哈希表计数', '滑动窗口', '排序后对撞指针', '二分答案'],
      answer: 0,
      explanation: md`有负数时窗口和不单调，滑窗失效；排序会破坏“连续”；也没有可二分的答案空间。前缀和 \`P[j] - P[i] = K\` ⇔ \`P[i] = P[j] - K\`，遍历时用哈希表统计之前各前缀和出现的次数，$O(n)$。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 3,
      prompt: md`把 LC4 分割线二分的推理步骤按逻辑顺序排列：`,
      items: [
        '中位数等价于把合并数组切成左右两半，左半元素都 ≤ 右半',
        '左半由 A 的前 i 个与 B 的前 j 个组成，j = half - i',
        '合法条件：A[i-1] ≤ B[j] 且 B[j-1] ≤ A[i]',
        '不合法时根据哪个条件被违反决定 i 增大还是减小（单调）',
        '找到合法的 i 后，由 max(左半) 与 min(右半) 计算中位数',
      ],
      explanation: md`先把“中位数”转化为“分割”，再用一个自由变量 i 参数化分割，写出合法条件，证明违反条件的方向单调，最后由边界元素求答案。`,
    },
  ],

  guided: [],

  challenges: [
    {
      id: 'c-median',
      title: '寻找两个正序数组的中位数',
      difficulty: 'hard',
      source: 'LeetCode 4',
      timeLimitMin: 30,
      description: md`
        给定两个升序数组 \`nums1\`（长 $m$）和 \`nums2\`（长 $n$），返回合并后的中位数（\`float\`）。

        **要求时间复杂度 $O(\log(m+n))$**：隐藏测试会统计你访问数组元素的次数，归并的 $O(m+n)$ 做法无法通过。

        \`\`\`python
        def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float: ...
        \`\`\`

        **示例**
        - \`([1, 3], [2])\` → \`2.0\`
        - \`([1, 2], [3, 4])\` → \`2.5\`
        - \`([], [1])\` → \`1.0\`

        **约束**：$0 \le m, n \le 10^5$，$m + n \ge 1$，元素范围 $[-10^6, 10^6]$。
      `,
      starter: py`
        def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
            # TODO: 在较短的数组上二分分割线 i，令 j = (m + n + 1) // 2 - i
            raise NotImplementedError
      `,
      solution: py`
        def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:
            A, B = nums1, nums2
            if len(A) > len(B):
                A, B = B, A                            # 在较短的数组上二分
            m, n = len(A), len(B)
            half = (m + n + 1) // 2                    # 左半元素个数（奇数时左半多一个）
            INF = float('inf')
            lo, hi = 0, m
            while lo <= hi:
                i = (lo + hi) // 2                     # A 贡献 i 个
                j = half - i                           # B 贡献 j 个
                Al = A[i - 1] if i > 0 else -INF
                Ar = A[i] if i < m else INF
                Bl = B[j - 1] if j > 0 else -INF
                Br = B[j] if j < n else INF
                if Al > Br:
                    hi = i - 1                         # A 给多了
                elif Bl > Ar:
                    lo = i + 1                         # A 给少了
                else:
                    left = max(Al, Bl)
                    if (m + n) % 2 == 1:
                        return float(left)
                    return (left + min(Ar, Br)) / 2
            raise ValueError("输入数组不是有序的")
      `,
      solutionAlt: [
        {
          label: '第 k 小思路（每次丢掉 k/2 个）',
          code: py`
            def find_median_sorted_arrays(A, B):
                def kth(a0, b0, k):                    # A[a0:] 与 B[b0:] 中第 k 小（k 从 1 起）
                    while True:
                        if a0 == len(A):
                            return B[b0 + k - 1]
                        if b0 == len(B):
                            return A[a0 + k - 1]
                        if k == 1:
                            return min(A[a0], B[b0])
                        h = k // 2
                        ia = min(a0 + h, len(A)) - 1
                        ib = min(b0 + h, len(B)) - 1
                        if A[ia] <= B[ib]:
                            k -= ia - a0 + 1           # 丢掉 A[a0..ia]
                            a0 = ia + 1
                        else:
                            k -= ib - b0 + 1
                            b0 = ib + 1
                total = len(A) + len(B)
                if total % 2:
                    return float(kth(0, 0, total // 2 + 1))
                return (kth(0, 0, total // 2) + kth(0, 0, total // 2 + 1)) / 2
          `,
        },
      ],
      explanation: md`
        **思路**：中位数把合并数组分成等长（奇数时左边多一个）的两半。用 A 的前 i 个和 B 的前 $j = \text{half} - i$ 个组成左半，
        合法条件是 \`A[i-1] <= B[j]\` 且 \`B[j-1] <= A[i]\`（越界取 ±∞）。
        - \`A[i-1] > B[j]\`：A 给多了，\`hi = i - 1\`；
        - \`B[j-1] > A[i]\`：A 给少了，\`lo = i + 1\`；
        - 否则找到：奇数返回 \`max(Al, Bl)\`，偶数返回 \`(max(Al, Bl) + min(Ar, Br)) / 2\`。

        **为什么在短数组上二分**：$i \in [0, m]$，则 $j = \text{half} - i \in [\text{half} - m, \text{half}]$。$m \le n$ 时 $\text{half} - m \ge 0$ 且 $\text{half} \le n$，j 永不越界。

        **复杂度**：$O(\log \min(m, n))$，空间 $O(1)$。

        **常见错误**：不交换数组导致 j 为负；偶数情况忘记取右半最小；用 \`(m + n) // 2\` 作为 half 却仍返回左半最大值（口径不一致）；整数除法丢失 .5。
      `,
      complexity: { time: 'O(log min(m, n))', space: 'O(1)' },
      hints: [
        md`中位数 = 把两个数组一起切成左右两半，左半所有元素 ≤ 右半所有元素。左半应该有多少个元素？`,
        md`设 A 贡献 i 个、B 贡献 j = half - i 个。“左半 ≤ 右半”只需检查哪两个交叉不等式？`,
        md`\`A[i-1] > B[j]\` ⇒ i 太大；\`B[j-1] > A[i]\` ⇒ i 太小。在较短的数组上对 i 二分，越界用 ±∞。`,
      ],
      setup: py`
        import math, random
        from collections.abc import Sequence

        def _ref_median(a, b):
            c = sorted(a + b)
            k = len(c)
            return float(c[k // 2]) if k % 2 else (c[k // 2 - 1] + c[k // 2]) / 2

        class _Probe(Sequence):
            """只读序列：统计元素访问次数（切片按长度计）"""
            def __init__(self, data):
                self._d = list(data)
                self.reads = 0
            def __len__(self):
                return len(self._d)
            def __getitem__(self, i):
                if isinstance(i, slice):
                    r = self._d[i]
                    self.reads += len(r)
                    return r
                self.reads += 1
                return self._d[i]
      `,
      tests: [
        {
          name: '示例',
          code: py`
            for a, b, exp in [([1, 3], [2], 2.0), ([1, 2], [3, 4], 2.5), ([], [1], 1.0), ([2], [], 2.0)]:
                got = find_median_sorted_arrays(a, b)
                assert abs(got - exp) < 1e-9, f"nums1={a}, nums2={b}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '边界：一侧全小 / 全大 / 有重复',
          code: py`
            cases = [([1, 2, 3], [4, 5, 6, 7], 4.0), ([10, 20, 30], [1, 2, 3, 4, 5], 4.5),
                     ([1, 1, 1], [1, 1], 1.0), ([-5, 3], [-2, -1, 8], -1.0), ([0, 0], [0, 0], 0.0)]
            for a, b, exp in cases:
                got = find_median_sorted_arrays(a, b)
                assert abs(got - exp) < 1e-9, f"nums1={a}, nums2={b}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '返回值类型',
          code: py`
            got = find_median_sorted_arrays([1, 3], [2])
            assert isinstance(got, float), f"应返回 float，得到 {type(got).__name__}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(4)
            for _ in range(500):
                a = sorted(rnd.randint(-20, 20) for _ in range(rnd.randint(0, 9)))
                b = sorted(rnd.randint(-20, 20) for _ in range(rnd.randint(0 if a else 1, 9)))
                exp, got = _ref_median(a, b), find_median_sorted_arrays(a, b)
                assert abs(got - exp) < 1e-9, f"nums1={a}, nums2={b}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：m = n = 10^5',
          hidden: true,
          code: py`
            rnd = random.Random(9)
            a = sorted(rnd.randint(-10**6, 10**6) for _ in range(100_000))
            b = sorted(rnd.randint(-10**6, 10**6) for _ in range(100_000))
            exp = _ref_median(a, b)
            pa, pb = _Probe(a), _Probe(b)
            got = find_median_sorted_arrays(pa, pb)
            assert abs(got - exp) < 1e-9, f"期望 {exp}，得到 {got}"
            reads = pa.reads + pb.reads
            limit = 8 * math.ceil(math.log2(100_001)) + 20
            assert reads <= limit, f"访问了 {reads} 次元素，超过 O(log n) 上限 {limit}——是不是归并了？"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-kth',
          q: md`如何求两个有序数组合并后的第 k 小？`,
          a: md`比较 \`A[k/2-1]\` 与 \`B[k/2-1]\`（不足 k/2 个时取末尾），较小的那一侧前 k/2 个元素都不可能是第 k 小，整体丢掉，k 减去丢掉的个数；直到某个数组为空或 k == 1。$O(\log k)$。分割线方法同样可以推广：令 \`half = k\`。`,
        },
        {
          id: 'f-many',
          q: md`如果是 K 个有序数组的中位数呢？`,
          a: md`可以在**值域**上二分：对候选值 x，用 K 次 bisect 统计 ≤ x 的元素总数，$O(K \log n)$；找最小的 x 使计数 ≥ 目标名次。总复杂度 $O(K \log n \log V)$。也可以用堆做 K 路归并到中间位置，$O(N/2 \cdot \log K)$。`,
        },
      ],
    },
    {
      id: 'c-kth-matrix',
      title: '有序矩阵中第 K 小的元素',
      difficulty: 'medium',
      source: 'LeetCode 378',
      timeLimitMin: 25,
      description: md`
        $n \times n$ 矩阵 \`matrix\` 的每行、每列都按**非递减**排序，返回所有元素排序后的第 \`k\` 小（$1 \le k \le n^2$，重复元素分别计数）。

        **要求**：时间 $O(n \log(\max - \min))$，额外空间 $O(1)$。隐藏测试会统计你读取矩阵元素的次数：
        把矩阵展平排序（读取 $n^2$ 次）或用堆弹出 k 次（$k \approx n^2/2$ 时读取同样很多）都无法通过。

        \`\`\`python
        def kth_smallest(matrix: list[list[int]], k: int) -> int: ...
        \`\`\`

        **示例**
        - \`kth_smallest([[1,5,9],[10,11,13],[12,13,15]], 8)\` → \`13\`
        - \`kth_smallest([[-5]], 1)\` → \`-5\`

        **约束**：$1 \le n \le 300$，$-10^9 \le$ 元素 $\le 10^9$。
      `,
      starter: py`
        def kth_smallest(matrix: list[list[int]], k: int) -> int:
            # TODO: 在值域 [matrix[0][0], matrix[-1][-1]] 上二分；
            #       用“从左下角出发的阶梯”在 O(n) 内统计 <= mid 的元素个数
            raise NotImplementedError
      `,
      solution: py`
        def kth_smallest(matrix: list[list[int]], k: int) -> int:
            n = len(matrix)

            def count_le(x: int) -> int:
                # 从左下角出发：<= x 则整列上方都 <= x，计入 i + 1 并右移；否则上移
                i, j, cnt = n - 1, 0, 0
                while i >= 0 and j < n:
                    if matrix[i][j] <= x:
                        cnt += i + 1
                        j += 1
                    else:
                        i -= 1
                return cnt

            lo, hi = matrix[0][0], matrix[n - 1][n - 1]
            while lo < hi:                     # 找最小的 x 使 count_le(x) >= k
                mid = (lo + hi) // 2           # Python 的 // 向下取整，负数也安全
                if count_le(mid) >= k:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
      `,
      solutionAlt: [
        {
          label: '堆：K 路归并（O(k log n)，k 小时更快）',
          code: py`
            import heapq

            def kth_smallest(matrix, k):
                n = len(matrix)
                heap = [(matrix[i][0], i, 0) for i in range(min(n, k))]   # 每行的首元素
                heapq.heapify(heap)
                for _ in range(k - 1):
                    _, i, j = heapq.heappop(heap)
                    if j + 1 < n:
                        heapq.heappush(heap, (matrix[i][j + 1], i, j + 1))
                return heap[0][0]
          `,
        },
      ],
      explanation: md`
        **思路**：在**值域**上二分答案。谓词 $P(x) = [\text{count}(\le x) \ge k]$ 关于 x 单调（F…F T…T），最小的满足 P 的 x 就是第 k 小，且一定是矩阵中的元素。

        **阶梯计数**：从左下角 \`(n-1, 0)\` 出发，若 \`matrix[i][j] <= x\`，第 j 列的前 i+1 个都 ≤ x，计入并右移；否则上移。每步排除一行或一列，$O(n)$。

        **复杂度**：$O(n \log(\max - \min))$，空间 $O(1)$。

        **常见错误**
        - mid 用 \`(lo + hi) / 2\`（浮点）或 \`int((lo+hi)/2)\`（负数向零取整，可能死循环）；Python 的 \`//\` 向下取整是正确的；
        - 谓词写成 \`count == k\` 并直接返回 mid——mid 可能不在矩阵中；
        - 阶梯从左上角出发，无法确定移动方向。
      `,
      complexity: { time: 'O(n log(max − min))', space: 'O(1)' },
      hints: [
        md`不在下标上二分，而在**值**上二分：给定 x，矩阵中 ≤ x 的元素有多少个？这个数随 x 单调吗？`,
        md`统计 ≤ x 的个数可以 $O(n)$：从左下角出发，≤ x 就把这一列上面的 i+1 个全部计入并右移，否则上移。`,
        md`找最小的 x 使 \`count_le(x) >= k\`：\`lo, hi = matrix[0][0], matrix[-1][-1]\`，满足则 \`hi = mid\`，否则 \`lo = mid + 1\`。`,
      ],
      setup: py`
        import random

        def _make(n, rnd, step=3, base=0):
            m = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    up = m[i - 1][j] if i else base
                    left = m[i][j - 1] if j else base
                    m[i][j] = max(up, left) + rnd.randint(0, step)
            return m

        class _Reads:
            def __init__(self, budget):
                self.n = 0
                self.budget = budget

        class _Row:
            """只读行：统计元素读取次数，超过预算立即失败"""
            def __init__(self, data, ctr):
                self._d, self._c = list(data), ctr
            def __len__(self):
                return len(self._d)
            def __getitem__(self, j):
                r = self._d[j]
                self._c.n += len(r) if isinstance(j, slice) else 1
                if self._c.n > self._c.budget:
                    raise AssertionError(f"读取矩阵元素超过 {self._c.budget} 次，复杂度不是 O(n log V)")
                return r
            def __iter__(self):
                for j in range(len(self._d)):
                    yield self[j]
      `,
      tests: [
        { name: '示例 1', code: py`assert kth_smallest([[1, 5, 9], [10, 11, 13], [12, 13, 15]], 8) == 13` },
        { name: '示例 2', code: py`assert kth_smallest([[-5]], 1) == -5` },
        {
          name: '重复与负数',
          code: py`
            m = [[-3, -3, 0], [-3, 0, 2], [0, 2, 2]]
            flat = sorted(x for row in m for x in row)
            for k in range(1, 10):
                got = kth_smallest(m, k)
                assert got == flat[k - 1], f"k={k}: 期望 {flat[k - 1]}，得到 {got}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(378)
            for _ in range(150):
                n = rnd.randint(1, 6)
                m = _make(n, rnd, step=rnd.randint(0, 4), base=rnd.randint(-10, 10))
                flat = sorted(x for row in m for x in row)
                k = rnd.randint(1, n * n)
                got = kth_smallest([row[:] for row in m], k)
                assert got == flat[k - 1], f"matrix={m}, k={k}: 期望 {flat[k - 1]}，得到 {got}"
          `,
        },
        {
          name: '复杂度：n = 300，k ≈ n²/2',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            n = 300
            m = _make(n, rnd, step=5, base=-10**6)
            flat = sorted(x for row in m for x in row)
            k = n * n // 2 + 17
            ctr = _Reads(budget=n * n // 3)
            got = kth_smallest([_Row(r, ctr) for r in m], k)
            assert got == flat[k - 1], f"期望 {flat[k - 1]}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-719',
          q: md`LC719「找出第 K 小的数对距离」和本题有什么共同点？`,
          a: md`同样是“值域二分 + 双指针计数”：排序后对距离 d 二分，用同向双指针在 $O(n)$ 内统计距离 ≤ d 的数对个数，找最小的 d 使计数 ≥ k。总复杂度 $O(n \log n + n \log W)$。“第 K 小”类问题只要能高效计数“≤ x 的有多少个”，就可以值域二分。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-lc4',
      freq: 3,
      q: md`两个正序数组的中位数，为什么一定要在较短的数组上二分？复杂度是多少？`,
      a: md`
        **结论**：为了保证 $j = \text{half} - i$ 始终落在 $[0, n]$ 内，复杂度 $O(\log \min(m, n))$。

        **原因**：$i \in [0, m]$，$\text{half} = \lfloor (m+n+1)/2 \rfloor$。当 $m \le n$ 时 $\text{half} - m \ge 0$、$\text{half} \le n$，所以 j 不会越界；若在长数组上二分，i 取较大值时 j 可能为负。

        **延伸**：二分的是“分割线”而不是“值”，合法条件是两个交叉不等式，违反哪个就往哪个方向调——这一套“参数化 + 单调条件”的思路同样适用于第 k 小和多数组问题。
      `,
    },
    {
      id: 'qa-value-bs',
      freq: 2,
      q: md`什么时候在“值域”上二分，而不是在下标上二分？`,
      a: md`
        **结论**：当数据不是一维有序、但能高效统计“≤ x 的元素有多少个”时。

        **原因**：第 k 小 = 最小的 x 使 count(≤ x) ≥ k，谓词关于 x 单调。行列有序矩阵（LC378）用阶梯 $O(n)$ 计数；数对距离（LC719）用排序 + 双指针计数；乘法表第 k 小（LC668）每行 $\min(x // i, n)$ 计数。

        **延伸**：值域二分的复杂度带 $\log V$，元素范围很大或为浮点时要注意迭代次数；整数时用 \`//\` 向下取整避免负数死循环。
      `,
    },
  ],

  cheatsheet: md`
    **LC4 分割线二分**（A 为较短数组）
    \`\`\`python
    half = (m + n + 1) // 2; lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2; j = half - i
        if A[i-1] > B[j]:   hi = i - 1      # 越界取 ±inf
        elif B[j-1] > A[i]: lo = i + 1
        else: 奇数 max(Al, Bl)；偶数 (max(Al,Bl) + min(Ar,Br)) / 2
    \`\`\`
    **LC378 值域二分**：最小 x 使 \`count_le(x) >= k\`；阶梯从**左下角**出发，\`<= x\` 时 \`cnt += i + 1; j += 1\`，否则 \`i -= 1\`

    | 信号 | 技巧 |
    |---|---|
    | 有序 / 部分有序 | 二分（边界、旋转、峰值） |
    | 最小可行值 | 二分答案 |
    | 第 k 小 + 可计数 | 值域二分 |
    | 有序数对 | 对撞指针 |
    | 连续区间 + 单调条件 | 滑动窗口 |
  `,
})
