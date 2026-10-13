/**
 * 快速排序（algo-sort-quick）
 * LC912 / LC75：Lomuto / Hoare / 三路划分、随机 pivot、最坏 O(n²) 的成因、稳定性与工程实现。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { PartitionViz } from '../viz/PartitionViz'
import { RecursionShapeViz } from '../viz/RecursionShapeViz'
import { ThreeWayViz } from '../viz/ThreeWayViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'partition',
      title: '一次划分：快排的全部秘密',
      body: md`
        快排 = **划分** + 递归。划分（partition）选一个 pivot，把区间重排成“左边 ≤ pivot、右边 ≥ pivot”，然后对两边递归。
        合并步骤为空——这和归并排序（划分为空、合并做全部工作）恰好相反。

        **Lomuto 划分**（最容易写对）：pivot 取 \`nums[hi]\`，维护循环不变量

        $$
        \underbrace{[lo, i)}_{< \text{pivot}}\quad \underbrace{[i, j)}_{\ge \text{pivot}}\quad \underbrace{[j, hi)}_{\text{未处理}}\quad \text{pivot}
        $$

        \`nums[j] < pivot\` 时把它换到 \`i\` 处并 \`i += 1\`。扫描结束后把 pivot 换到 \`i\`——**pivot 落在最终位置**，递归 \`[lo, i-1]\` 与 \`[i+1, hi]\`。

        **Hoare 划分**：两个指针从两端相向扫描，左边找 ≥ pivot 的、右边找 ≤ pivot 的，交换后继续。交换次数平均约为 Lomuto 的 1/3，
        且遇到大量相等元素时两侧仍会均匀分配；代价是 pivot 不一定归位，递归区间是 \`[lo, j]\` 与 \`[j+1, hi]\`，边界更容易写错。
      `,
      viz: PartitionViz,
      after: md`
        > [!WARN] Hoare 的两个坑
        > 1. 返回的是 \`j\` 而不是 pivot 的位置，递归必须用 \`[lo, j]\` 和 \`[j+1, hi]\`；若写成 \`[lo, j-1]\` 会漏元素。
        > 2. pivot 取 \`nums[hi]\` 配合 \`[lo, j]\` / \`[j+1, hi]\` 会在两元素有序数组上死循环——Hoare 要取中点（向下取整）或 \`nums[lo]\`。
      `,
      checkpoint: {
        id: 'cp-partition',
        type: 'fill',
        prompt: md`对 \`[3, 8, 2, 5, 1, 4]\` 做一次 Lomuto 划分（pivot = 末尾的 4）。返回的下标是 ____(1)____，划分后的数组是 ____(2)____（形如 \`[a,b,c]\`）。`,
        blanks: [
          { label: '(1)', answers: ['3'], numeric: true },
          { label: '(2)', answers: ['[3,2,1,4,8,5]', '3,2,1,4,8,5'] },
        ],
        explanation: md`3 < 4 换到下标 0（与自己交换），i=1；8 跳过；2 < 4 与 8 交换 → \`[3,2,8,5,1,4]\`，i=2；5 跳过；1 < 4 与 8 交换 → \`[3,2,1,5,8,4]\`，i=3；最后 pivot 与 5 交换 → \`[3,2,1,4,8,5]\`，返回 3。`,
      },
    },
    {
      id: 'complexity',
      title: '复杂度：递归树的形状说了算',
      body: md`
        每一层递归的划分总共扫描 $O(n)$ 个元素，所以总代价 = $O(n) \times$ 递归深度：

        | 情形 | 划分比例 | 深度 | 时间 |
        |---|---|---|---|
        | 最好 / 平均 | 大致均分 | $O(\log n)$ | $O(n \log n)$ |
        | 最坏 | 每次只切掉一个 | $O(n)$ | $O(n^2)$ |

        **最坏情况是怎么来的**：固定取末尾为 pivot 时，**已排序或逆序**的输入每次都选到最大/最小值；
        **大量重复值**时，二路划分把所有等于 pivot 的元素分到同一侧——即使随机选 pivot 也无济于事。

        **对策**
        - **随机 pivot**：任何输入上的期望比较次数都是 $2n\ln n \approx 1.39\, n\log_2 n$，最坏情况只在概率上极小；
        - **三数取中**：取 \`lo, mid, hi\` 的中位数，对有序输入有效，但可被特意构造的输入击败；
        - **三路划分**：等于 pivot 的元素一次归位，全相同输入变成 $O(n)$；
        - **introsort**：递归深度超过约 $2\log_2 n$ 时改用堆排序，保证最坏 $O(n \log n)$（libstdc++ 的 \`std::sort\` 就是这样）。
      `,
      viz: RecursionShapeViz,
      after: md`
        > [!TIP] 空间复杂度与尾递归
        > 递归栈深度最坏 $O(n)$。工程写法是**只递归较短的一侧，较长的一侧用循环处理**，栈深度最坏 $O(\log n)$（每次递归的区间长度至少减半）。在 Python 中这还能避免 1000 层的默认递归上限。
      `,
      checkpoint: {
        id: 'cp-complexity',
        type: 'numeric',
        prompt: md`对已排序的 \`[1, 2, …, 10]\` 用“末尾元素为 pivot 的 Lomuto 快排”，总共要做多少次元素与 pivot 的比较？`,
        answer: 45,
        tolerance: 0,
        explanation: md`每次 pivot 都是最大值，区间长度依次为 10, 9, …, 2，每次比较 (长度 − 1) 次：$9 + 8 + \dots + 1 = 45 = n(n-1)/2$。`,
      },
    },
    {
      id: 'threeway',
      title: '三路划分与荷兰国旗（LC75）',
      body: md`
        三路划分把区间分成 \`< p\`、\`== p\`、\`> p\` 三段，用三个指针维护四个区域：

        $$
        \underbrace{[0, lt)}_{< p}\ \underbrace{[lt, i)}_{= p}\ \underbrace{[i, gt]}_{\text{未处理}}\ \underbrace{(gt, n)}_{> p}
        $$

        - \`nums[i] < p\`：与 \`nums[lt]\` 交换，\`lt += 1; i += 1\`（换过来的一定是 \`== p\` 的元素或它自己，已检查过）；
        - \`nums[i] > p\`：与 \`nums[gt]\` 交换，\`gt -= 1\`，**i 不动**（换过来的元素还没检查）；
        - \`nums[i] == p\`：\`i += 1\`。

        **LC75 颜色分类**正是 $p = 1$ 的三路划分：一趟扫描、常数空间。快排中使用三路划分后，只需递归 \`[lo, lt-1]\` 和 \`[gt+1, hi]\`。
      `,
      viz: ThreeWayViz,
      after: md`
        > [!INTERVIEW] 为什么交换 gt 后 i 不动，而交换 lt 后 i 要动？
        > \`lt ≤ i\` 且 \`[lt, i)\` 全是 \`== p\`，所以从 \`lt\` 换来的要么是一个已检查过的 \`== p\` 元素，要么（\`lt == i\` 时）就是它自己；而 \`gt\` 处的元素属于未处理区，必须留在 \`i\` 处再检查一次。
      `,
      checkpoint: {
        id: 'cp-threeway',
        type: 'judge',
        prompt: md`三路划分中，\`nums[i] > p\` 与 \`nums[gt]\` 交换后，应同时执行 \`gt -= 1\` 和 \`i += 1\`。`,
        answer: false,
        explanation: md`错误。从 \`gt\` 换来的元素尚未检查，可能是 0、1 或 2；若 \`i += 1\` 就跳过了它。例如 \`[2, 0]\`：交换后为 \`[0, 2]\`，若 i 前进，0 留在中间段，结果错误。`,
      },
    },
    {
      id: 'engineering',
      title: '稳定性与工程实现：真实的库在用什么',
      body: md`
        **快排不稳定**：划分中的长距离交换会打乱相等元素的相对顺序。例如按分数排序 \`[(2,"a"), (2,"b"), (1,"c")]\`，Lomuto 以 \`(1,"c")\` 为 pivot 时，第一次交换就会把 \`(2,"a")\` 换到 \`(2,"b")\` 之后。

        | 库 | 默认算法 | 稳定 |
        |---|---|---|
        | Python \`list.sort\` / \`sorted\` | Timsort（3.11 起合并策略改为 powersort） | ✅ |
        | C++ \`std::sort\`（libstdc++） | introsort（快排 + 堆排兜底 + 小区间插入排序） | ❌ |
        | C++ \`std::stable_sort\` | 归并排序 | ✅ |
        | Java \`Arrays.sort(int[])\` | 双轴快排（Dual-Pivot Quicksort） | ❌（基本类型无所谓） |
        | Java \`Arrays.sort(Object[])\` | TimSort | ✅ |
        | NumPy \`np.sort\` 默认 | introsort（\`kind="stable"\` 用 timsort / 整数用 radix） | ❌ |

        为什么工程上快排常常最快？原地、顺序访问内存（缓存友好）、内层循环极简；小区间（十几个元素）切换到插入排序进一步减少常数。
        为什么 Python / Java 对象排序选 Timsort？对象比较代价高，Timsort 在部分有序的真实数据上比较次数少，并且**稳定**——多关键字排序依赖稳定性。
      `,
      after: md`
        > [!NOTE] 稳定性什么时候重要
        > “先按次要关键字排，再按主要关键字**稳定**排序”等价于按（主，次）排序。数据库 ORDER BY 多列、LLM 训练数据按长度分桶后保持原顺序，都依赖稳定排序。
      `,
      checkpoint: {
        id: 'cp-engineering',
        type: 'single',
        prompt: md`要按“分数降序、分数相同时保持输入顺序”排序一个 Python 对象列表，最合适的做法是：`,
        options: [
          '`sorted(items, key=lambda x: -x.score)`——Python 排序是稳定的',
          '手写随机化快排，以 score 为键',
          '`heapq.nlargest(len(items), items, key=...)`',
          '先 `random.shuffle` 再排序',
        ],
        answer: 0,
        explanation: md`Timsort 稳定，相同分数的元素保持输入顺序（\`reverse=True\` 同样保持稳定性）。B 快排不稳定；C 的 \`nlargest\` 文档上等价于 \`sorted(..., reverse=True)[:n]\`，但用它做全排序是绕远路；D 破坏了输入顺序。`,
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
      prompt: md`普通（随机 pivot）快速排序的平均时间复杂度、最坏时间复杂度和额外空间分别是：`,
      options: [
        '$O(n\\log n)$、$O(n^2)$、平均 $O(\\log n)$ 栈空间',
        '$O(n\\log n)$、$O(n\\log n)$、$O(1)$',
        '$O(n)$、$O(n^2)$、$O(n)$',
        '$O(n\\log n)$、$O(n^2)$、$O(n)$ 辅助数组',
      ],
      answer: 0,
      explanation: md`随机化只改变“遇到最坏情况的概率”，最坏仍是 $O(n^2)$。快排原地，额外空间来自递归栈：平均深度 $O(\log n)$（最坏 $O(n)$，只递归短边可压到 $O(\log n)$）。D 描述的是归并排序的辅助数组。`,
    },
    {
      id: 'q-stable',
      type: 'judge',
      difficulty: 1,
      prompt: md`快速排序是稳定的排序算法。`,
      answer: false,
      explanation: md`不稳定。划分中的远距离交换会改变相等元素的相对顺序。稳定的比较排序有归并排序、插入排序、Timsort。`,
    },
    {
      id: 'q-worst-inputs',
      type: 'multi',
      difficulty: 2,
      prompt: md`使用“末尾元素作 pivot + Lomuto 二路划分”的快排，在哪些输入上会退化到 $O(n^2)$？（多选）`,
      options: ['已升序排列', '已降序排列', '所有元素都相同', '随机排列的互不相同元素'],
      answer: [0, 1, 2],
      explanation: md`A、B 每次选到最大/最小值，划分为 $n-1$ 和 0。C 中 \`nums[j] < pivot\` 永远为假，所有元素落在右侧，同样每次只少一个。D 期望 $O(n \log n)$。注意 C 即使换成随机 pivot 也依旧是 $O(n^2)$——需要三路划分或 Hoare 划分。`,
    },
    {
      id: 'q-depth',
      type: 'numeric',
      difficulty: 2,
      prompt: md`对 $n = 2^{20}$（约 100 万）个元素，若每次划分都恰好均分，快排的递归深度约为多少层？`,
      answer: 20,
      tolerance: 0,
      explanation: md`每层区间长度减半，$\log_2 2^{20} = 20$ 层。若退化成每次只切一个元素，深度将是 $2^{20}$——Python 默认递归上限只有 1000，这就是“有序大数组让递归快排 RecursionError”的原因。`,
    },
    {
      id: 'q-lomuto-steps',
      type: 'order',
      difficulty: 2,
      prompt: md`把 Lomuto 划分的步骤按执行顺序排列：`,
      items: [
        '（随机化时）随机选一个下标，与 hi 交换',
        '令 pivot = nums[hi]，i = lo',
        'j 从 lo 扫到 hi - 1：若 nums[j] < pivot，交换 nums[i]、nums[j] 并 i += 1',
        '交换 nums[i] 与 nums[hi]，pivot 归位',
        '递归 [lo, i - 1] 与 [i + 1, hi]',
      ],
      explanation: md`随机化只是在划分前把随机元素换到末尾，划分本身不变。pivot 归位后它在最终位置，所以递归区间不包含 i。`,
    },
    {
      id: 'q-libs',
      type: 'match',
      difficulty: 2,
      prompt: md`把排序接口与其默认算法配对：`,
      pairs: [
        ['Python `list.sort`', 'Timsort（稳定）'],
        ['C++ `std::sort`（libstdc++）', 'introsort'],
        ['Java `Arrays.sort(int[])`', '双轴快排'],
        ['C++ `std::stable_sort`', '归并排序'],
      ],
      explanation: md`introsort = 快排 + 深度超限时堆排序兜底 + 小区间插入排序；Java 对基本类型用双轴快排（稳定性对基本类型无意义），对对象用 TimSort；\`std::stable_sort\` 通常是归并排序（有额外内存时 $O(n\log n)$）。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出三路划分版的颜色分类（LC75，pivot = 1；有 2 行干扰项）：`,
      lines: [
        'def sort_colors(nums):',
        '    lt, i, gt = 0, 0, len(nums) - 1',
        '    while i <= gt:',
        '        if nums[i] == 0:',
        '            nums[lt], nums[i] = nums[i], nums[lt]',
        '            lt, i = lt + 1, i + 1',
        '        elif nums[i] == 2:',
        '            nums[i], nums[gt] = nums[gt], nums[i]',
        '            gt -= 1',
        '        else:',
        '            i += 1',
      ],
      distractors: ['    while i < gt:', '            gt -= 1; i += 1'],
      explanation: md`\`i == gt\` 时该位置仍未处理，所以是 \`i <= gt\`；与 gt 交换后 i 不能前进。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的快排在输入 \`[1, 2]\` 上会无限递归（RecursionError）。点出有问题的行：`,
      code: py`
        def quick_sort(nums, lo, hi):
            if lo >= hi:
                return
            pivot, i = nums[hi], lo
            for j in range(lo, hi):
                if nums[j] < pivot:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
            nums[i], nums[hi] = nums[hi], nums[i]
            quick_sort(nums, lo, i)
            quick_sort(nums, i + 1, hi)
      `,
      bugLines: [10],
      explanation: md`第 10 行应为 \`quick_sort(nums, lo, i - 1)\`。pivot 已经在最终位置 i，不应再参与递归；当 pivot 是区间最大值时 \`i == hi\`，\`quick_sort(nums, lo, i)\` 等于原样再调用一次，永不收缩。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 3,
      prompt: md`
        对 \`[5, 1, 4, 2, 8, 3]\` 做一次 Lomuto 划分（pivot = 3）后数组变为 \`[1, 2, 3, 5, 8, 4]\`。
        此时哪些位置上的元素已经与**最终排序结果**一致？（点出全部）
      `,
      items: ['1', '2', '3', '5', '8', '4'],
      answer: [0, 1, 2],
      explanation: md`最终结果是 \`[1, 2, 3, 4, 5, 8]\`，下标 0、1、2 一致。但要注意：**只有 pivot（下标 2）是划分保证归位的**，下标 0、1 只是碰巧（左侧恰好有序）；右侧 \`[5, 8, 4]\` 还需要递归。`,
    },
    {
      id: 'q-stack',
      type: 'single',
      difficulty: 3,
      prompt: md`为了让快排在最坏情况下的递归栈深度也只有 $O(\log n)$，正确的做法是：`,
      options: [
        '只对较短的一侧递归，较长的一侧在当前函数中用循环继续处理',
        '只对较长的一侧递归，较短的一侧用循环',
        '使用随机 pivot',
        '把递归改成多线程并行',
      ],
      answer: 0,
      explanation: md`每次递归的区间不超过当前区间的一半，所以递归深度 ≤ $\log_2 n$；较长一侧通过循环（尾调用消除）处理，不增加栈深。B 恰好相反，最坏深度 $O(n)$；C 只降低最坏情况的概率，不改变最坏深度。`,
    },
    {
      id: 'q-hoare',
      type: 'single',
      difficulty: 2,
      prompt: md`关于 Hoare 划分与 Lomuto 划分，正确的是：`,
      options: [
        'Hoare 平均交换次数更少，且对大量重复元素也能均匀划分；但划分后 pivot 不一定在最终位置',
        'Lomuto 在全相同元素上是 $O(n \\log n)$',
        'Hoare 划分返回 pivot 的最终下标',
        '两者的交换次数相同，只是写法不同',
      ],
      answer: 0,
      explanation: md`Hoare 在等于 pivot 的元素上两个指针都会停下并交换，所以全相同时划分在中间，$O(n \log n)$；Lomuto 则全部落在一侧，$O(n^2)$。Hoare 返回的是分界下标 j，递归区间为 \`[lo, j]\` 与 \`[j+1, hi]\`。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-quick',
      title: '一步步写出随机化快排（Lomuto）',
      intro: md`
        实现原地的 \`quick_sort(nums, lo=0, hi=None)\`。先把随机选中的 pivot 换到末尾，再做 Lomuto 划分，最后递归两侧。
      `,
      template: py`
        import random

        def quick_sort(nums, lo=0, hi=None):
            if hi is None:
                hi = len(nums) - 1
            if lo >= hi:
                return
            k = random.randint(lo, hi)
            {{1}}                                   # 随机 pivot 换到末尾
            pivot, i = nums[hi], lo
            for j in range(lo, hi):
                if {{2}}:
                    nums[i], nums[j] = nums[j], nums[i]
                    {{3}}
            {{4}}                                   # pivot 归位到 i
            quick_sort(nums, lo, i - 1)
            {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`把随机下标 \`k\` 处的元素与末尾交换，后面的代码就能统一用 \`nums[hi]\` 作 pivot。`,
          answer: 'nums[k], nums[hi] = nums[hi], nums[k]',
          accept: ['nums[hi], nums[k] = nums[k], nums[hi]'],
          explain: md`随机化让“对手”无法构造必然退化的输入，期望比较次数 $\approx 1.39\, n \log_2 n$。`,
        },
        {
          blank: 2,
          instruction: md`什么样的元素应该放进左边的“小于区” \`[lo, i)\`？`,
          answer: 'nums[j] < pivot',
          accept: ['pivot > nums[j]', 'nums[j] <= pivot'],
          explain: md`写 \`<=\` 也正确，只是等于 pivot 的元素会去左边。无论哪种，二路划分遇到大量重复值都会退化，这正是三路划分存在的原因。`,
        },
        {
          blank: 3,
          instruction: md`小于区扩大一格。`,
          answer: 'i += 1',
        },
        {
          blank: 4,
          instruction: md`扫描结束：\`[lo, i)\` < pivot，\`[i, hi)\` ≥ pivot。把 pivot 放到分界处。`,
          answer: 'nums[i], nums[hi] = nums[hi], nums[i]',
          accept: ['nums[hi], nums[i] = nums[i], nums[hi]'],
          explain: md`此后 pivot 在最终位置 i，不再参与递归。`,
        },
        {
          blank: 5,
          instruction: md`递归右侧。`,
          answer: 'quick_sort(nums, i + 1, hi)',
          explain: md`如果担心递归深度，可以改成“递归较短的一侧、循环处理较长的一侧”。`,
        },
      ],
      setup: py`
        import random as _r
      `,
      tests: [
        {
          name: '示例',
          code: py`
            a = [5, 2, 3, 1]
            quick_sort(a)
            assert a == [1, 2, 3, 5], f"得到 {a}"
          `,
        },
        {
          name: '随机对拍',
          code: py`
            rnd = _r.Random(912)
            for _ in range(300):
                a = [rnd.randint(-9, 9) for _ in range(rnd.randint(0, 15))]
                exp = sorted(a)
                quick_sort(a)
                assert a == exp, f"期望 {exp}，得到 {a}"
          `,
        },
      ],
      outro: md`
        这就是面试中最常被要求“现场手写”的快排。实战题里隐藏测试会用**全相同元素**卡住二路划分——你需要升级成三路划分（或 Hoare 划分）。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-quicksort',
      title: '手撕原地快速排序',
      difficulty: 'medium',
      source: 'LeetCode 912（快排版）',
      timeLimitMin: 20,
      description: md`
        实现**原地**快速排序：把 \`nums\` 就地排成升序（函数返回值会被忽略）。

        \`\`\`python
        def quick_sort(nums: list) -> None: ...
        \`\`\`

        **要求**
        - 不能使用 \`sorted\`、\`list.sort\`、\`heapq\`（会检查源码）；
        - 期望比较次数 $O(n \log n)$。隐藏测试包含**已排序**数组、**全相同**数组、**只有 3 种取值**的数组（$n$ 为 2000～3000），
          会用一个“计数比较次数”的包装类型替换元素，比较次数超过 $8\,n\log_2 n$ 立即判失败；
        - 元素只支持比较运算（\`<\`、\`<=\`、\`>\`、\`>=\`、\`==\`），不要对元素做加减。

        **示例**：\`nums = [5, 2, 3, 1]\` → 原地变为 \`[1, 2, 3, 5]\`。
      `,
      starter: py`
        import random

        def quick_sort(nums: list) -> None:
            # TODO: 随机 pivot + 三路划分（或 Hoare 划分），原地排序
            raise NotImplementedError
      `,
      solution: py`
        import random

        def quick_sort(nums: list) -> None:
            def sort(lo: int, hi: int) -> None:
                while lo < hi:
                    p = nums[random.randint(lo, hi)]        # 随机 pivot（取值）
                    lt, i, gt = lo, lo, hi                  # 三路划分
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
                    # 递归较短的一侧，较长的一侧继续循环：栈深 O(log n)
                    if lt - lo < hi - gt:
                        sort(lo, lt - 1)
                        lo = gt + 1
                    else:
                        sort(gt + 1, hi)
                        hi = lt - 1

            sort(0, len(nums) - 1)
      `,
      solutionAlt: [
        {
          label: 'Hoare 划分版（同样能扛住大量重复值）',
          code: py`
            import random

            def quick_sort(nums):
                def sort(lo, hi):
                    if lo >= hi:
                        return
                    k = random.randint(lo, hi)
                    nums[k], nums[(lo + hi) // 2] = nums[(lo + hi) // 2], nums[k]
                    p = nums[(lo + hi) // 2]
                    i, j = lo - 1, hi + 1
                    while True:
                        i += 1
                        while nums[i] < p:
                            i += 1
                        j -= 1
                        while nums[j] > p:
                            j -= 1
                        if i >= j:
                            break
                        nums[i], nums[j] = nums[j], nums[i]
                    sort(lo, j)
                    sort(j + 1, hi)
                sort(0, len(nums) - 1)
          `,
        },
      ],
      explanation: md`
        **为什么要三路划分（或 Hoare）**：全相同或取值很少的数组上，Lomuto 二路划分把等于 pivot 的元素全部分到一侧，每次只减少一个元素，$O(n^2)$；随机 pivot 也无济于事。
        三路划分让 \`== p\` 的整段直接归位；Hoare 划分在相等元素上两侧指针都会停下交换，分界落在中间。

        **为什么要随机 pivot**：已排序输入上固定 pivot 每次选到极值，$O(n^2)$。

        **栈深度**：参考解“递归短边、循环长边”，最坏栈深 $O(\log n)$，避免 Python 的 1000 层递归上限。

        **复杂度**：期望 $O(n \log n)$；重复值越多越快（全相同为 $O(n)$）；额外空间 $O(\log n)$。
      `,
      complexity: { time: '期望 O(n log n)', space: 'O(log n)' },
      hints: [
        md`先写出普通的随机化 Lomuto 快排，再想：数组全是同一个数时会发生什么？`,
        md`把划分改成三段：\`< p\`、\`== p\`、\`> p\`（荷兰国旗）。三个指针 \`lt, i, gt\`，与 gt 交换后 i 不前进。`,
        md`递归只处理 \`[lo, lt-1]\` 与 \`[gt+1, hi]\`。为了不爆栈，可以递归较短的一侧、用 while 循环处理较长的一侧。`,
      ],
      setup: py`
        import linecache, math, random as _rnd

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
                    raise AssertionError(f"比较次数超过 {_K.budget}：复杂度退化到 O(n^2)（固定 pivot？二路划分遇到大量重复值？）")
            def __lt__(self, o): _K._tick(); return self.v < o.v
            def __le__(self, o): _K._tick(); return self.v <= o.v
            def __gt__(self, o): _K._tick(); return self.v > o.v
            def __ge__(self, o): _K._tick(); return self.v >= o.v
            def __eq__(self, o): _K._tick(); return self.v == o.v
            def __ne__(self, o): _K._tick(); return self.v != o.v
            __hash__ = None
            def __repr__(self):
                return repr(self.v)

        def _run_budget(vals, factor=8):
            a = [_K(v) for v in vals]
            n = len(a)
            _K.n, _K.budget = 0, int(factor * n * math.log2(max(n, 2))) + 50
            try:
                quick_sort(a)
            finally:
                _K.budget = 10**12
            got = [x.v for x in a]
            assert got == sorted(vals), "排序结果不正确（注意要原地修改 nums）"
            return _K.n
      `,
      tests: [
        {
          name: '示例',
          code: py`
            a = [5, 2, 3, 1]
            quick_sort(a)
            assert a == [1, 2, 3, 5], f"期望 [1, 2, 3, 5]，得到 {a}（需要原地排序）"
          `,
        },
        {
          name: '边界',
          code: py`
            for a in ([], [1], [2, 1], [1, 1], [3, -1, 3, -1, 0]):
                b = list(a)
                quick_sort(b)
                assert b == sorted(a), f"输入 {a}：期望 {sorted(a)}，得到 {b}"
          `,
        },
        {
          name: '不使用库排序',
          code: py`
            src = "".join(linecache.getlines("solution.py"))
            for banned in ("sorted(", ".sort(", "heapq", "heapify"):
                assert banned not in src, f"请手写快排，不要使用 {banned}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = _rnd.Random(912)
            for _ in range(300):
                a = [rnd.randint(-50, 50) for _ in range(rnd.randint(0, 40))]
                b = list(a)
                quick_sort(b)
                assert b == sorted(a), f"输入 {a}：期望 {sorted(a)}，得到 {b}"
          `,
        },
        {
          name: '已排序 n = 2000（卡固定 pivot）',
          hidden: true,
          code: py`_run_budget(list(range(2000)))`,
        },
        {
          name: '全相同 n = 2000（卡二路划分）',
          hidden: true,
          code: py`_run_budget([7] * 2000)`,
        },
        {
          name: '只有 0/1/2，n = 3000',
          hidden: true,
          code: py`
            rnd = _rnd.Random(75)
            _run_budget([rnd.randint(0, 2) for _ in range(3000)])
          `,
        },
      ],
      followUps: [
        {
          id: 'f-introsort',
          q: md`怎样保证快排**最坏**也是 $O(n \log n)$？`,
          a: md`introsort：记录递归深度，超过约 $2\log_2 n$ 时对当前区间改用堆排序；小区间改用插入排序。libstdc++ 的 \`std::sort\` 即如此。另一种是用 median-of-medians（BFPRT）选 pivot，理论最坏 $O(n\log n)$，但常数大，工程上很少用。`,
        },
        {
          id: 'f-why-fast',
          q: md`同样是 $O(n \log n)$，为什么快排在实践中常比堆排序快？`,
          a: md`快排的内层循环是顺序扫描，缓存命中率高、分支简单；堆排序的 sift down 在数组上跳跃访问（下标 $i \to 2i+1$），缓存不友好，且每层要两次比较。快排的比较次数常数（约 $1.39 n\log_2 n$）也低于堆排序（约 $2 n \log_2 n$）。`,
        },
      ],
    },
    {
      id: 'c-sort-colors',
      title: '颜色分类（荷兰国旗）',
      difficulty: 'medium',
      source: 'LeetCode 75',
      timeLimitMin: 10,
      description: md`
        数组 \`nums\` 只包含 0、1、2（红、白、蓝），**原地**排序使相同颜色相邻，顺序为 0、1、2。

        \`\`\`python
        def sort_colors(nums: list[int]) -> None: ...
        \`\`\`

        **要求**：不能使用库排序（会检查源码）；目标是**一趟扫描**、$O(1)$ 额外空间。

        **示例**：\`[2,0,2,1,1,0]\` → \`[0,0,1,1,2,2]\`；\`[2,0,1]\` → \`[0,1,2]\`。
      `,
      starter: py`
        def sort_colors(nums: list[int]) -> None:
            # TODO: 三指针 lt / i / gt，一趟完成
            raise NotImplementedError
      `,
      solution: py`
        def sort_colors(nums: list[int]) -> None:
            lt, i, gt = 0, 0, len(nums) - 1     # [0,lt)=0，[lt,i)=1，[i,gt] 未处理，(gt,n)=2
            while i <= gt:
                if nums[i] == 0:
                    nums[lt], nums[i] = nums[i], nums[lt]
                    lt += 1
                    i += 1
                elif nums[i] == 2:
                    nums[i], nums[gt] = nums[gt], nums[i]
                    gt -= 1                     # 换来的元素未检查，i 不动
                else:
                    i += 1
      `,
      explanation: md`
        **思路**：pivot = 1 的三路划分。维护 \`[0, lt)\` 全 0、\`[lt, i)\` 全 1、\`(gt, n)\` 全 2，\`[i, gt]\` 未处理。
        遇 0 与 lt 交换并双双前进；遇 2 与 gt 交换且 i 不动；遇 1 直接前进。

        **复杂度**：一趟，$O(n)$ 时间、$O(1)$ 空间。

        **另解**：计数排序（先数 0/1/2 的个数再回填）两趟，也是 $O(n)$，但题目进阶要求一趟。

        **常见错误**：\`while i < gt\`（漏掉 \`i == gt\`）；与 gt 交换后 \`i += 1\`。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`把数组分成四段：0 区、1 区、未处理区、2 区。需要几个指针？`,
        md`\`lt, i, gt = 0, 0, n - 1\`，循环条件 \`i <= gt\`。`,
        md`遇 2 时与 \`nums[gt]\` 交换、\`gt -= 1\`，**不要**移动 i。`,
      ],
      setup: py`
        import linecache, random
      `,
      tests: [
        {
          name: '示例',
          code: py`
            for a, exp in (([2, 0, 2, 1, 1, 0], [0, 0, 1, 1, 2, 2]), ([2, 0, 1], [0, 1, 2])):
                b = list(a)
                sort_colors(b)
                assert b == exp, f"输入 {a}：期望 {exp}，得到 {b}"
          `,
        },
        {
          name: '边界',
          code: py`
            for a in ([], [0], [2], [2, 0], [1, 1, 1], [2, 2, 0, 0]):
                b = list(a)
                sort_colors(b)
                assert b == sorted(a), f"输入 {a}：期望 {sorted(a)}，得到 {b}"
          `,
        },
        {
          name: '不使用库排序',
          code: py`
            src = "".join(linecache.getlines("solution.py"))
            for banned in ("sorted(", ".sort("):
                assert banned not in src, f"请不要使用 {banned}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(75)
            for _ in range(400):
                a = [rnd.randint(0, 2) for _ in range(rnd.randint(0, 20))]
                b = list(a)
                sort_colors(b)
                assert b == sorted(a), f"输入 {a}：期望 {sorted(a)}，得到 {b}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-k-colors',
          q: md`如果有 k 种颜色（LC 的“颜色分类 II”），怎么做？`,
          a: md`计数排序 $O(n + k)$ 最简单；若要求 $O(1)$ 额外空间，可以对颜色区间做“彩虹排序”：取中间颜色为 pivot 做划分，递归两侧颜色区间，$O(n \log k)$。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-worst',
      freq: 3,
      q: md`快排最坏情况是什么？怎么避免？`,
      a: md`
        **结论**：每次划分极不均衡时（一侧 $n-1$、一侧 0）退化为 $O(n^2)$。

        **原因**：固定取首/尾为 pivot 时，有序或逆序输入每次都选到极值；大量重复值时，二路划分把等于 pivot 的元素都放到同一侧，随机 pivot 也无效。

        **延伸**：对策分别是随机 pivot / 三数取中（解决有序输入）、三路划分或 Hoare 划分（解决重复值）、introsort（深度超过 $2\log_2 n$ 切换堆排序，保证最坏 $O(n\log n)$）。再加“递归短边、循环长边”把栈深压到 $O(\log n)$。
      `,
    },
    {
      id: 'qa-compare',
      freq: 3,
      q: md`快排、归并、堆排序怎么选？`,
      a: md`
        **结论**：内存中的基本类型数组首选快排（introsort）；需要稳定或排序链表 / 外部数据用归并；需要最坏保证且 $O(1)$ 空间用堆排序。

        **原因**：快排原地、顺序访存、常数小，平均最快；归并稳定、最坏 $O(n\log n)$，但数组上需要 $O(n)$ 辅助空间，链表上则天然 $O(1)$；堆排序最坏 $O(n\log n)$ 且原地，但访存跳跃、比较次数多，实际较慢。

        **延伸**：Python 选 Timsort（归并 + 插入，利用现实数据中的有序片段，稳定）；Java 对基本类型用双轴快排、对对象用 TimSort——对象比较贵且需要稳定性。
      `,
    },
    {
      id: 'qa-lomuto-hoare',
      freq: 2,
      q: md`Lomuto 和 Hoare 划分有什么区别？`,
      a: md`
        **结论**：Lomuto 单向扫描、好写、pivot 归位；Hoare 双向扫描、交换少、对重复值稳健，但 pivot 不一定归位。

        **原因**：Lomuto 维护 \`[lo,i) < p\`，每遇到一个小元素就交换一次；Hoare 只交换“站错边”的一对，平均交换次数约为 Lomuto 的 1/3。等于 pivot 的元素在 Hoare 中两个指针都会停下，所以全相同时分界在中间。

        **延伸**：Hoare 的递归区间必须是 \`[lo, j]\` 与 \`[j+1, hi]\`，pivot 取中点或首元素；取尾元素会在两元素区间上死循环。面试手写时用 Lomuto + 三路划分更稳妥。
      `,
    },
    {
      id: 'qa-stack',
      freq: 2,
      q: md`快排的空间复杂度是多少？怎么优化？`,
      a: md`
        **结论**：额外空间来自递归栈，平均 $O(\log n)$、最坏 $O(n)$；递归短边、循环长边后最坏 $O(\log n)$。

        **原因**：较短一侧的长度不超过当前区间的一半，所以递归链上每一层区间长度至少减半，深度 ≤ $\log_2 n$；较长一侧通过修改 \`lo\`/\`hi\` 在同一栈帧中继续处理。

        **延伸**：Python 默认递归上限 1000，对 $10^5$ 级的有序输入，未优化的递归快排会直接 RecursionError——这也是面试时主动提出该优化的好理由。
      `,
    },
  ],

  cheatsheet: md`
    **随机化三路快排（面试版）**
    \`\`\`python
    def sort(lo, hi):
        while lo < hi:
            p = nums[random.randint(lo, hi)]
            lt, i, gt = lo, lo, hi
            while i <= gt:
                if nums[i] < p:   nums[lt], nums[i] = nums[i], nums[lt]; lt += 1; i += 1
                elif nums[i] > p: nums[i], nums[gt] = nums[gt], nums[i]; gt -= 1
                else:             i += 1
            if lt - lo < hi - gt: sort(lo, lt - 1); lo = gt + 1   # 递归短边
            else:                 sort(gt + 1, hi); hi = lt - 1
    \`\`\`
    - 平均 $O(n\log n)$（≈ $1.39\,n\log_2 n$ 次比较），最坏 $O(n^2)$，不稳定，栈 $O(\log n)$
    - 退化：有序 + 固定 pivot；大量重复 + 二路划分
    - Lomuto：\`[lo,i) < p\`，pivot 归位；Hoare：返回 j，递归 \`[lo,j]\`、\`[j+1,hi]\`
    - 三路：与 gt 交换后 **i 不动**
    - 库：Python Timsort（稳定）· C++ introsort · Java 基本类型双轴快排
  `,
})
