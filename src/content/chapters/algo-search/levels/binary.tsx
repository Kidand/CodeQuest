/**
 * 【样板关卡 / 黄金标准】二分查找模板
 * 其它关卡请对照本文件的结构、密度与写法（docs/AUTHORING.md）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { BinarySearchViz, BinarySearchHalfOpenViz } from '../viz/BinarySearchViz'
import { BoundaryViz } from '../viz/BoundaryViz'
import { CdfSamplingViz } from '../viz/CdfSamplingViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'why',
      title: '二分的前提：单调性，而不只是“有序”',
      body: md`
        在长度为 $n$ 的有序数组里找一个数，逐个比较要 $O(n)$；二分每轮看中点 **mid**，
        根据 \`nums[mid]\` 与 \`target\` 的大小关系**一次排除一半**，因此只需
        $\lceil \log_2(n+1) \rceil$ 轮——100 万个数最多 20 轮。

        二分能成立的真正前提是：**能根据 mid 处的信息，确定答案在哪一半**。“数组有序”只是最常见的一种情况，
        旋转数组、山峰数组、“答案空间单调”的问题（下一关的二分答案）同样满足。

        下面用**闭区间**模板 \`[lo, hi]\` 走一遍。拖动 target、切换模板，观察 \`lo / mid / hi\` 怎么移动，
        以及代码当前执行到哪一行。
      `,
      viz: BinarySearchViz,
      after: md`
        > [!TIP] 读图的关键
        > 变暗的格子是**已经被排除**的区间。每一轮结束后，答案若存在，一定还在亮着的区间里——这就是**循环不变量**。
      `,
      checkpoint: {
        id: 'cp-why',
        type: 'single',
        prompt: md`有序数组长度为 $1023$，用二分查找一个**不存在**的数，最多需要多少轮循环？`,
        options: ['9', '10', '11', '1023'],
        answer: 1,
        explanation: md`每轮把区间长度从 $L$ 变为最多 $\lfloor L/2 \rfloor$：$1023 \to 511 \to 255 \to \dots \to 1 \to 0$，共 10 轮。一般地为 $\lceil \log_2(n+1) \rceil = \lceil \log_2 1024 \rceil = 10$。`,
      },
    },
    {
      id: 'invariant',
      title: '循环不变量：闭区间 vs 左闭右开',
      body: md`
        二分写错几乎都源于**没想清楚区间的含义**。两种主流写法只要“定义—条件—更新”三者一致，都不会错：

        | | 闭区间 \`[lo, hi]\` | 左闭右开 \`[lo, hi)\` |
        |---|---|---|
        | 初始化 | \`hi = n - 1\` | \`hi = n\` |
        | 循环条件（区间非空） | \`lo <= hi\` | \`lo < hi\` |
        | 去右半 | \`lo = mid + 1\` | \`lo = mid + 1\` |
        | 去左半 | \`hi = mid - 1\` | \`hi = mid\` |

        记忆方法：**hi 指向的位置在不在区间里？** 在（闭区间）→ mid 已检查，所以 \`hi = mid - 1\`；
        不在（开区间）→ \`hi = mid\` 就已经把 mid 排除了。

        切换到左闭右开再走一遍，注意 \`hi\` 指针可以指向数组**外面**（下标 $n$）。
      `,
      viz: BinarySearchHalfOpenViz,
      after: md`
        > [!WARN] 三个经典 bug
        > 1. \`hi = len(nums)\` 却写 \`while lo <= hi\` → \`mid\` 可能等于 $n$，越界；
        > 2. \`lo = mid\`（没有 +1）→ 当 \`hi = lo + 1\` 时 \`mid == lo\`，**死循环**；
        > 3. C++/Java 中 \`(lo + hi) / 2\` 可能整数溢出，应写 \`lo + (hi - lo) / 2\`（Python 整数无溢出，但面试时主动提到是加分项）。
      `,
      checkpoint: {
        id: 'cp-inv',
        type: 'judge',
        prompt: md`左闭右开写法中，当 \`nums[mid] > target\` 时写 \`hi = mid - 1\` 也是正确的，只是多排除了一个元素。`,
        answer: false,
        explanation: md`错误。左闭右开时区间是 \`[lo, hi)\`，\`hi = mid - 1\` 会把下标 \`mid - 1\` 也排除掉，而它**没有被检查过**，可能恰好就是答案。`,
      },
    },
    {
      id: 'boundary',
      title: '边界二分：lower_bound 与 upper_bound',
      body: md`
        面试里更常见的不是“找到任意一个”，而是“**找第一个 / 最后一个**”。统一的思路是把问题转成**单调谓词**：

        $$
        P(i) = [\,\text{nums}[i] \ge \text{target}\,] \quad\Rightarrow\quad \underbrace{F\,F\,\cdots\,F}_{< \text{target}}\ \underbrace{T\,T\,\cdots\,T}_{\ge \text{target}}
        $$

        **lower_bound** = 第一个 $P(i)=T$ 的位置；把谓词换成 $>$ 就是 **upper_bound**。
        于是 \`target\` 的出现次数 = \`upper_bound - lower_bound\`，最后一次出现的位置 = \`upper_bound - 1\`。

        Python 标准库里它们叫 \`bisect.bisect_left\` / \`bisect.bisect_right\`。
      `,
      viz: BoundaryViz,
      after: md`
        注意这个模板里**没有** \`==\` 分支：遇到 True 时不急着返回，而是 \`hi = mid\` 继续向左收缩，
        直到 \`lo == hi\` 收敛到第一个 True。这是“找边界”类二分（包括下一关的二分答案）的通用骨架。

        > [!INTERVIEW] 面试官想听到的
        > “我把它抽象成在 F…FT…T 上找第一个 T，区间用左闭右开 \`[0, n]\`，结果天然落在插入位置上，不需要特判找不到的情况。”
      `,
      checkpoint: {
        id: 'cp-bound',
        type: 'fill',
        prompt: md`对 \`nums = [1, 2, 2, 2, 3, 5]\`，\`lower_bound(nums, 2)\` = ____(1)____，\`upper_bound(nums, 2)\` = ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['1'], numeric: true },
          { label: '(2)', answers: ['4'], numeric: true },
        ],
        explanation: md`第一个 $\ge 2$ 的下标是 1；第一个 $> 2$ 的下标是 4。二者相减得 2 出现了 3 次。`,
      },
    },
    {
      id: 'llm',
      title: '大模型岗场景：采样下一个 token 就是一次二分',
      body: md`
        语言模型每一步输出词表上的概率分布 $p$。**按概率采样**最直接的实现是逆 CDF 法：

        1. 求前缀和 $c_i = \sum_{j \le i} p_j$（单调不减）；
        2. 取 $u \sim U[0, 1)$；
        3. 返回**第一个** $c_i > u$ 的下标——正是 upper_bound！

        词表通常有 10 万 ~ 25 万个 token，二分只需 17~18 次比较。NumPy / PyTorch 对应的函数是
        \`np.searchsorted\` / \`torch.searchsorted\`（\`side='right'\` 即“第一个 > u”）。
      `,
      viz: CdfSamplingViz,
      after: md`
        把温度调低，分布变尖，绝大多数 $u$ 都会落到概率最大的 token 上；调高则更“随机”。
        温度、top-k、top-p 的完整讲解在「大模型 · 分词与解码」章节。

        > [!NOTE] 为什么用 $c_i > u$ 而不是 $\ge$
        > 若某 token 概率为 0，它的 $c_i = c_{i-1}$。用严格大于可以保证零概率 token **永远不会**被采到（第一个使 $c_i > u$ 的 $i$ 必有 $p_i > 0$）。
      `,
    },
  ],

  /* ================================================================ */
  /* 随堂练习（覆盖全部 10 种题型）                                       */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-cond',
      type: 'single',
      difficulty: 1,
      prompt: md`闭区间写法 \`lo, hi = 0, len(nums) - 1\` 下，循环条件应当是：`,
      options: ['`while lo < hi:`', '`while lo <= hi:`', '`while lo + 1 < hi:`', '`while hi - lo > 1:`'],
      answer: 1,
      explanation: md`闭区间 \`[lo, hi]\` 在 \`lo == hi\` 时仍包含一个元素，必须继续检查，所以是 \`lo <= hi\`。写成 \`lo < hi\` 会漏掉最后一个元素；后两个是“保留两端点”的另一种模板，需要配合循环后的额外判断。`,
    },
    {
      id: 'q-overflow',
      type: 'judge',
      difficulty: 1,
      prompt: md`在 Python 中 \`mid = (lo + hi) // 2\` 可能因为整数溢出而出错。`,
      answer: false,
      explanation: md`Python 的 \`int\` 是任意精度的，不会溢出。但在 C++/Java 中 \`lo + hi\` 可能超过 \`INT_MAX\`，应写成 \`lo + (hi - lo) / 2\`。注意 **NumPy 的 int32/int64 数组元素**是定长整数，相加仍可能溢出。`,
    },
    {
      id: 'q-applicable',
      type: 'multi',
      difficulty: 2,
      prompt: md`以下哪些问题可以用 $O(\log n)$ 的二分解决？（多选）`,
      options: [
        '在有序数组中找 target 的插入位置',
        '在旋转过的有序数组（如 [4,5,6,7,0,1,2]）中找 target',
        '在无序数组中找最大值',
        '求最小的整数 $x$，使单调递增函数 $f(x) \\ge k$',
        '在有序**单链表**中查找 target',
      ],
      answer: [0, 1, 3],
      explanation: md`A 是 lower_bound；B 每次至少有一半是有序的，可以判断 target 在哪半（下一关）；D 是“二分答案”，答案空间上谓词单调。C 无序没有单调性，至少要 $O(n)$；E 链表不支持随机访问，取中点就要 $O(n)$。`,
    },
    {
      id: 'q-rounds',
      type: 'numeric',
      difficulty: 2,
      prompt: md`对长度为 $1{,}000{,}000$ 的有序数组做二分查找，**最坏情况**下 while 循环最多执行多少轮？`,
      answer: 20,
      tolerance: 0,
      explanation: md`最坏轮数为 $\lceil \log_2(n+1) \rceil = \lceil \log_2 1000001 \rceil = 20$（$2^{19} = 524288 < 10^6 < 2^{20} = 1048576$）。`,
    },
    {
      id: 'q-bisect',
      type: 'single',
      difficulty: 2,
      prompt: md`
        下面代码输出什么？
        \`\`\`python
        from bisect import bisect_left, bisect_right
        a = [1, 3, 3, 3, 5, 8]
        print(bisect_right(a, 3) - bisect_left(a, 3), bisect_left(a, 4))
        \`\`\`
      `,
      options: ['`3 4`', '`3 3`', '`2 4`', '`4 4`'],
      answer: 0,
      explanation: md`\`bisect_left(a,3)=1\`，\`bisect_right(a,3)=4\`，差为 3 即 3 的出现次数。4 不存在，第一个 $\ge 4$ 的位置是下标 4（元素 5），也就是 4 的插入位置。`,
    },
    {
      id: 'q-steps',
      type: 'order',
      difficulty: 2,
      prompt: md`把用二分解决“边界类问题”的思考步骤按正确顺序排列：`,
      items: [
        '确定答案可能的取值范围（左闭右开 `[lo, hi)`）',
        '定义单调谓词 P(i)，确认它呈 F…F T…T 形态',
        '循环：P(mid) 为 False 则 `lo = mid + 1`，否则 `hi = mid`',
        '循环结束后 lo 即第一个 True；按需处理“全为 False”（lo == n）',
      ],
      explanation: md`先定范围、再定谓词、再写循环、最后处理边界——这个顺序能保证每一步都有明确的不变量，而不是凭感觉调 \`±1\`。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把左侧写法与右侧的正确描述配对：`,
      pairs: [
        ['`hi = len(nums) - 1`', '与 `while lo <= hi` 配套'],
        ['`hi = len(nums)`', '与 `while lo < hi` 配套'],
        ['`bisect_left(a, x)`', '第一个 ≥ x 的位置'],
        ['`bisect_right(a, x)`', '第一个 > x 的位置'],
      ],
      explanation: md`前两行是两种区间定义各自的初始化与循环条件；后两行是标准库里 lower_bound / upper_bound 的语义。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`用左闭右开模板拼出 \`lower_bound\`（有 2 行干扰项，不应使用）：`,
      lines: [
        'def lower_bound(nums, target):',
        '    lo, hi = 0, len(nums)',
        '    while lo < hi:',
        '        mid = (lo + hi) // 2',
        '        if nums[mid] < target:',
        '            lo = mid + 1',
        '        else:',
        '            hi = mid',
        '    return lo',
      ],
      distractors: ['    while lo <= hi:', '            hi = mid - 1'],
      explanation: md`\`hi = len(nums)\` 决定了区间左闭右开，于是条件用 \`lo < hi\`、收缩用 \`hi = mid\`；干扰项属于闭区间写法，混用会越界或漏解。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的闭区间二分在某些输入上会**死循环**。点出有问题的那一行：`,
      code: py`
        def search(nums, target):
            lo, hi = 0, len(nums) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] == target:
                    return mid
                elif nums[mid] < target:
                    lo = mid
                else:
                    hi = mid - 1
            return -1
      `,
      bugLines: [8],
      explanation: md`第 8 行应为 \`lo = mid + 1\`。例如 \`nums=[1,3]\`、\`target=3\`：\`lo=0, hi=1, mid=0\`，\`nums[0]<3\` 后 \`lo\` 仍为 0，区间不缩小，永远循环。**每一轮都必须让区间严格变小**。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`对 \`nums = [1, 3, 5, 7, 9, 11, 13, 15, 17]\` 用**闭区间**模板查找 \`13\`，依次被选为 \`mid\` 的有哪些元素？（点出全部）`,
      items: ['1', '3', '5', '7', '9', '11', '13', '15', '17'],
      answer: [4, 6],
      explanation: md`\`lo=0, hi=8 → mid=4\`（值 9 < 13）→ \`lo=5\`；\`mid=(5+8)//2=6\`（值 13）命中。只探查了下标 4 和 6。`,
    },
    {
      id: 'q-llm',
      type: 'single',
      difficulty: 3,
      prompt: md`逆 CDF 采样中，前缀和 \`cdf = [0.5, 0.5, 0.8, 1.0]\`（第 2 个 token 概率为 0）。为保证零概率 token 永远不被采到，对 $u\in[0,1)$ 应返回：`,
      options: ['第一个 `cdf[i] >= u` 的 i', '第一个 `cdf[i] > u` 的 i', '最后一个 `cdf[i] <= u` 的 i', '`round(u * len(cdf))`'],
      answer: 1,
      explanation: md`取 $u = 0.5$：用 $\ge$ 会返回下标 0（还好），但若 $u$ 恰等于某段的右端点，$\ge$ 可能返回零概率区段的下标；用 $>$（upper_bound / \`searchsorted(side="right")\`）时，满足 $c_i > u$ 的第一个 $i$ 必有 $c_i > c_{i-1}$，即 $p_i > 0$。C 选项在 $u < c_0$ 时无解；D 完全忽略了概率。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-lower-bound',
      title: '一步步写出 lower_bound',
      intro: md`
        实现 \`lower_bound(nums, target)\`：返回**第一个** $\ge$ \`target\` 的下标；若所有元素都小于 \`target\`，返回 \`len(nums)\`。
        我们用左闭右开 \`[lo, hi)\`，每填对一空就会解释它为什么这样写。
      `,
      template: py`
        def lower_bound(nums, target):
            lo, hi = {{1}}
            while {{2}}:
                mid = {{3}}
                if {{4}}:
                    lo = mid + 1
                else:
                    {{5}}
            return lo
      `,
      steps: [
        {
          blank: 1,
          instruction: md`初始化答案区间。答案可能是 $0 \dots n$ 中的任意一个（$n$ 表示“都小于 target”），用左闭右开表示。`,
          answer: '0, len(nums)',
          hint: '左闭右开时，hi 指向“最后一个可能答案的下一个位置”。',
          explain: md`答案的取值是 $0 \dots n$ 共 $n+1$ 种（$n$ 表示“全都小于 target”）。\`hi = len(nums)\` 时，若所有元素都小于 target，\`lo\` 会一路右移直到等于 $n$，循环结束正好返回 $n$，无需特判。`,
        },
        {
          blank: 2,
          instruction: md`写循环条件：区间 \`[lo, hi)\` 非空。`,
          answer: 'lo < hi',
          explain: md`左闭右开区间在 \`lo == hi\` 时为空，此时 \`lo\` 就是答案。`,
        },
        {
          blank: 3,
          instruction: md`取中点（向下取整）。`,
          answer: '(lo + hi) // 2',
          accept: ['lo + (hi - lo) // 2', '(lo+hi)>>1', 'lo + ((hi - lo) >> 1)'],
          explain: md`向下取整保证 \`mid < hi\`，因此 \`hi = mid\` 一定让区间变小；\`lo = mid + 1\` 同理。`,
        },
        {
          blank: 4,
          instruction: md`写出“谓词为 False”的条件——此时第一个 $\ge$ target 的位置一定在 mid **右边**。`,
          answer: 'nums[mid] < target',
          accept: ['target > nums[mid]', 'not nums[mid] >= target'],
          explain: md`\`nums[mid] < target\` 说明 mid 及其左边都不满足 $\ge$ target，可以整体丢弃。`,
        },
        {
          blank: 5,
          instruction: md`谓词为 True：mid 本身可能就是答案，**不能丢**，只收缩右端点。`,
          answer: 'hi = mid',
          explain: md`这是与“找任意一个”模板最大的不同：不返回，继续向左逼近第一个 True。`,
        },
      ],
      setup: py`
        import bisect, random
        def _ref_lower_bound(nums, target):
            return bisect.bisect_left(nums, target)
      `,
      tests: [
        { name: '示例', code: py`assert lower_bound([1, 2, 2, 2, 3, 5], 2) == 1` },
        { name: '大于所有元素', code: py`assert lower_bound([1, 2, 3], 9) == 3` },
        { name: '空数组', code: py`assert lower_bound([], 1) == 0` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(42)
            for _ in range(300):
                a = sorted(rnd.randint(0, 20) for _ in range(rnd.randint(0, 15)))
                t = rnd.randint(-2, 22)
                assert lower_bound(a, t) == _ref_lower_bound(a, t), f"nums={a}, target={t}"
          `,
        },
      ],
      outro: md`
        把第 4 空改成 \`nums[mid] <= target\`，同一段代码就变成了 **upper_bound**。
        记住这个骨架，下一关的“二分答案”只是把 \`nums[mid] < target\` 换成 \`not check(mid)\`。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-search-range',
      title: '在排序数组中查找元素的第一个和最后一个位置',
      difficulty: 'medium',
      source: 'LeetCode 34',
      timeLimitMin: 15,
      description: md`
        给定按**非递减**顺序排列的整数数组 \`nums\` 和目标值 \`target\`，返回 \`target\` 在数组中出现的
        \`[开始位置, 结束位置]\`；若不存在返回 \`[-1, -1]\`。

        **要求时间复杂度 $O(\log n)$**（隐藏测试会统计你访问 \`nums[i]\` 的次数）。

        \`\`\`python
        def search_range(nums: list[int], target: int) -> list[int]: ...
        \`\`\`

        **示例**
        - \`search_range([5,7,7,8,8,10], 8)\` → \`[3, 4]\`
        - \`search_range([5,7,7,8,8,10], 6)\` → \`[-1, -1]\`
        - \`search_range([], 0)\` → \`[-1, -1]\`

        **约束**：$0 \le n \le 10^5$。
      `,
      starter: py`
        def search_range(nums: list[int], target: int) -> list[int]:
            # TODO: 用两次边界二分在 O(log n) 内求出 [first, last]
            return [-1, -1]
      `,
      solution: py`
        def search_range(nums: list[int], target: int) -> list[int]:
            def lower_bound(t):
                lo, hi = 0, len(nums)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if nums[mid] < t:
                        lo = mid + 1
                    else:
                        hi = mid
                return lo

            first = lower_bound(target)
            if first == len(nums) or nums[first] != target:
                return [-1, -1]
            last = lower_bound(target + 1) - 1  # 整数数组：第一个 > target 的位置 - 1
            return [first, last]
      `,
      explanation: md`
        **思路**：两次 lower_bound。
        - \`first = lower_bound(target)\`；若越界或 \`nums[first] != target\`，说明不存在。
        - \`last = upper_bound(target) - 1\`。对整数数组，\`upper_bound(target) == lower_bound(target + 1)\`，可以复用同一个函数。

        **复杂度**：时间 $O(\log n)$，空间 $O(1)$。

        **常见错误**：找到一个 target 后向两边线性扩展——当数组全是 target 时退化为 $O(n)$，隐藏测试会卡掉。
      `,
      complexity: { time: 'O(log n)', space: 'O(1)' },
      hints: [
        md`把问题拆成两个边界：第一个 $\ge$ target 的位置，和第一个 $>$ target 的位置。`,
        md`写一个 \`lower_bound(t)\` 辅助函数，\`last\` 可以用 \`lower_bound(target + 1) - 1\` 得到。`,
        md`别忘了判断 \`first == len(nums)\` 或 \`nums[first] != target\` 的情况。`,
      ],
      setup: py`
        import bisect, math, random
        from collections.abc import Sequence

        def _ref_search_range(nums, target):
            l = bisect.bisect_left(nums, target)
            if l == len(nums) or nums[l] != target:
                return [-1, -1]
            return [l, bisect.bisect_right(nums, target) - 1]

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
        { name: '示例 1', code: py`assert search_range([5, 7, 7, 8, 8, 10], 8) == [3, 4]` },
        { name: '示例 2：不存在', code: py`assert search_range([5, 7, 7, 8, 8, 10], 6) == [-1, -1]` },
        { name: '示例 3：空数组', code: py`assert search_range([], 0) == [-1, -1]` },
        {
          name: '单元素与两端',
          code: py`
            assert search_range([1], 1) == [0, 0]
            assert search_range([1, 2, 3], 1) == [0, 0]
            assert search_range([1, 2, 3], 3) == [2, 2]
            assert search_range([1, 2, 3], 4) == [-1, -1]
            assert search_range([1, 2, 3], 0) == [-1, -1]
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(2026)
            for _ in range(400):
                a = sorted(rnd.randint(-5, 5) for _ in range(rnd.randint(0, 12)))
                t = rnd.randint(-6, 6)
                got = search_range(a, t)
                exp = _ref_search_range(a, t)
                assert got == exp, f"nums={a}, target={t}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：全部相同的 10^5 个元素',
          hidden: true,
          code: py`
            n = 100_000
            p = _Probe([7] * n)
            got = search_range(p, 7)
            assert got == [0, n - 1], f"期望 [0, {n - 1}]，得到 {got}"
            limit = 4 * math.ceil(math.log2(n + 1)) + 10
            assert p.reads <= limit, f"访问了 nums {p.reads} 次，超过 O(log n) 上限 {limit}——是不是找到后线性扩展了？"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-float',
          q: md`如果数组元素是浮点数，\`lower_bound(target + 1) - 1\` 还成立吗？`,
          a: md`不成立。浮点数之间不存在“下一个整数”的关系，应单独实现 upper_bound（谓词换成 \`nums[mid] <= target\`），或用 \`bisect_right\`。`,
        },
        {
          id: 'f-count',
          q: md`如果要对同一个数组做 $10^6$ 次“统计 target 出现次数”的查询，怎么优化？`,
          a: md`单次查询 $O(\log n)$ 已经最优；若值域有限可预处理哈希表 \`Counter\` 做到 $O(1)$，代价是 $O(n)$ 预处理与额外内存——需要和面试官确认数组是否会变化。`,
        },
      ],
    },
    {
      id: 'c-sample',
      title: '手写逆 CDF 采样（不用 searchsorted）',
      difficulty: 'medium',
      source: '大模型岗手撕 · 采样',
      timeLimitMin: 15,
      description: md`
        给定非负权重 \`weights\`（长度 $V$，未归一化，至少一个为正）和一组随机数 \`us\`（每个 $u \in [0,1)$），
        对每个 $u$ 返回按权重采样到的下标：

        1. 归一化 $p_i = w_i / \sum_j w_j$，求前缀和 $c_i$；
        2. 返回**第一个满足 $c_i > u$** 的 $i$。

        **要求**
        - 自己写二分，**不要**使用 \`np.searchsorted\` / \`bisect\`（会检查源码）；
        - 零权重的下标永远不能被返回；
        - 浮点误差可能导致 $c_{V-1}$ 略小于 1：若找不到满足条件的 $i$，返回**最后一个正权重**的下标。

        \`\`\`python
        def sample_indices(weights: list[float], us: list[float]) -> list[int]: ...
        \`\`\`

        **示例**：\`weights=[1, 0, 3]\`，\`cdf=[0.25, 0.25, 1.0]\`
        - $u=0.1 \to 0$，$u=0.25 \to 2$（$c_1 = 0.25$ 不大于 0.25），$u=0.9 \to 2$
      `,
      starter: py`
        def sample_indices(weights: list[float], us: list[float]) -> list[int]:
            # 1) 前缀和  2) 对每个 u 二分找第一个 cdf[i] > u  3) 处理浮点误差
            raise NotImplementedError
      `,
      solution: py`
        def sample_indices(weights: list[float], us: list[float]) -> list[int]:
            total = float(sum(weights))
            cdf, acc = [], 0.0
            for w in weights:
                acc += w / total
                cdf.append(acc)
            last_pos = max(i for i, w in enumerate(weights) if w > 0)

            out = []
            for u in us:
                lo, hi = 0, len(cdf)          # 在 [0, V) 中找第一个 cdf[i] > u
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cdf[mid] <= u:
                        lo = mid + 1
                    else:
                        hi = mid
                out.append(min(lo, last_pos))  # lo == V 时（浮点误差）退回最后一个正权重
            return out
      `,
      solutionAlt: [
        {
          label: 'NumPy 向量化版（面试中可作为补充）',
          code: py`
            import numpy as np

            def sample_indices_np(weights, us):
                w = np.asarray(weights, dtype=np.float64)
                cdf = np.cumsum(w / w.sum())
                idx = np.searchsorted(cdf, us, side="right")   # 第一个 cdf > u
                last_pos = np.flatnonzero(w > 0)[-1]
                return np.minimum(idx, last_pos).tolist()
          `,
        },
        {
          label: 'PyTorch 版',
          code: py`
            import torch

            def sample_indices_torch(weights: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
                cdf = torch.cumsum(weights / weights.sum(), dim=-1)
                idx = torch.searchsorted(cdf, us, right=True)
                last_pos = torch.nonzero(weights > 0).max()
                return torch.clamp(idx, max=last_pos)
          `,
        },
      ],
      explanation: md`
        **思路**：前缀和单调不减，“第一个 $c_i > u$”就是 upper_bound，谓词 $P(i) = [c_i > u]$ 呈 F…FT…T。

        **零权重为何安全**：若 $i$ 是第一个满足 $c_i > u$ 的位置，则 $c_{i-1} \le u < c_i$，所以 $p_i = c_i - c_{i-1} > 0$。

        **浮点误差**：累加 $V$ 个浮点数后 $c_{V-1}$ 可能是 $0.9999999999999998$，当 $u$ 更大时二分返回 $V$，需要钳制到最后一个正权重下标。

        **复杂度**：预处理 $O(V)$，每次采样 $O(\log V)$。实际推理框架在 GPU 上更常用 **Gumbel-Max**（$\arg\max_i(\log p_i + g_i)$）等并行友好的方式，这是很好的追问方向。
      `,
      complexity: { time: 'O(V + m log V)', space: 'O(V)' },
      hints: [
        md`先算 \`cdf\`，再对每个 $u$ 套用 upper_bound 模板：谓词为 \`cdf[mid] > u\`。`,
        md`谓词为 False（\`cdf[mid] <= u\`）时 \`lo = mid + 1\`，否则 \`hi = mid\`。`,
        md`最后 \`min(lo, last_pos)\`，其中 \`last_pos\` 是最后一个 \`w > 0\` 的下标。`,
      ],
      setup: py`
        import inspect, random
        from itertools import accumulate

        def _ref_sample(weights, us):
            total = float(sum(weights))
            cdf = list(accumulate(w / total for w in weights))
            last_pos = max(i for i, w in enumerate(weights) if w > 0)
            out = []
            for u in us:
                i = next((k for k, c in enumerate(cdf) if c > u), len(cdf))
                out.append(min(i, last_pos))
            return out
      `,
      tests: [
        { name: '示例', code: py`assert sample_indices([1, 0, 3], [0.1, 0.25, 0.9]) == [0, 2, 2]` },
        { name: 'u = 0', code: py`assert sample_indices([0, 0, 2, 1], [0.0]) == [2]` },
        {
          name: '零权重永不被采到',
          code: py`
            w = [0, 1, 0, 0, 2, 0]
            us = [i / 1000 for i in range(1000)]
            got = sample_indices(w, us)
            assert set(got) <= {1, 4}, f"采到了零权重下标：{sorted(set(got) - {1, 4})}"
          `,
        },
        {
          name: '不使用库函数',
          code: py`
            src = inspect.getsource(sample_indices)
            for banned in ("searchsorted", "bisect"):
                assert banned not in src, f"请自己实现二分，不要使用 {banned}"
          `,
        },
        {
          name: '浮点误差兜底',
          hidden: true,
          code: py`
            w = [0.1] * 10 + [0.0, 0.0]
            assert sample_indices(w, [0.9999999999999999]) == [9]
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            for _ in range(200):
                V = rnd.randint(1, 30)
                w = [rnd.choice([0, 0, rnd.random() * 5]) for _ in range(V)]
                if sum(w) == 0:
                    w[rnd.randrange(V)] = 1.0
                us = [rnd.random() for _ in range(20)]
                assert sample_indices(w, us) == _ref_sample(w, us), f"weights={w}"
          `,
        },
        {
          name: '分布正确（10^4 次采样）',
          hidden: true,
          code: py`
            rnd = random.Random(1)
            w = [5, 3, 1, 1]
            us = [rnd.random() for _ in range(10000)]
            got = sample_indices(w, us)
            for i, wi in enumerate(w):
                freq = got.count(i) / len(us)
                assert abs(freq - wi / 10) < 0.02, f"下标 {i} 频率 {freq:.3f}，期望 {wi / 10}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-gumbel',
          q: md`GPU 上批量采样时，为什么常用 Gumbel-Max 技巧而不是逆 CDF？`,
          a: md`逆 CDF 需要前缀和（scan）再二分，在 GPU 上是带依赖的操作；Gumbel-Max 只需对每个位置独立加噪声 $g_i = -\log(-\log U_i)$ 后取 $\arg\max$，完全并行，且可以直接在 logits 上做（无需先 softmax 归一化）。两者在数学上都精确服从 $p$。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-template',
      freq: 3,
      q: md`你平时用哪种二分模板？怎么保证不写错边界？`,
      a: md`
        **结论**：固定用左闭右开 \`[lo, hi)\` 的“找第一个 True”模板。

        **原因**：所有二分都可以转成单调谓词 F…FT…T 上找第一个 T；区间定义明确后，条件 \`lo < hi\`、收缩 \`lo = mid + 1\` / \`hi = mid\` 都是推出来的，不靠记忆；循环结束时 \`lo == hi\` 就是答案（含“不存在时的插入位置”）。

        **延伸**：要找“最后一个满足条件的位置”，就找“第一个不满足的位置”再减 1。
      `,
    },
    {
      id: 'qa-dead-loop',
      freq: 2,
      q: md`二分什么时候会死循环？给一个具体例子。`,
      a: md`
        当某一轮区间没有严格缩小时。典型是 \`mid\` 向下取整却写了 \`lo = mid\`：区间为 \`[3, 4]\` 时 \`mid = 3\`，\`lo\` 不变。
        修正方法：要么改成 \`lo = mid + 1\`，要么在需要 \`lo = mid\` 的模板里用向上取整 \`mid = (lo + hi + 1) // 2\`。
      `,
    },
    {
      id: 'qa-complexity',
      freq: 2,
      q: md`二分查找的时间复杂度是多少？在链表或磁盘上还适用吗？`,
      a: md`
        数组上 $O(\log n)$，前提是 $O(1)$ 随机访问。链表取中点就要 $O(n)$，整体退化到 $O(n)$——这正是跳表、平衡树存在的原因。
        磁盘/数据库索引用 **B+ 树**，本质是“多路二分”，用更大的扇出减少 IO 次数。
      `,
    },
    {
      id: 'qa-llm',
      freq: 2,
      q: md`大模型推理或训练里有哪些地方会用到二分思想？`,
      a: md`
        - **采样**：逆 CDF 采样就是在前缀和上做 upper_bound（\`torch.searchsorted\`）。
        - **二分答案类工程问题**：在显存预算下求可行的最大 batch size / 最大上下文长度，前提是“是否放得下”关于该参数单调。
        - **数据侧**：在排好序的样本长度上二分做分桶、在 token 偏移数组上二分定位样本（如多个序列拼接后的 \`cu_seqlens\`）。

        回答时说清楚“为什么满足单调性”比罗列场景更重要。
      `,
    },
  ],

  cheatsheet: md`
    **通用模板（左闭右开，找第一个 True）**
    \`\`\`python
    lo, hi = 0, n            # 答案 ∈ [0, n]
    while lo < hi:
        mid = (lo + hi) // 2
        if not P(mid): lo = mid + 1
        else:          hi = mid
    return lo                # 第一个 P 为真的位置；全假时为 n
    \`\`\`

    | 需求 | 谓词 P(i) | 结果 |
    |---|---|---|
    | lower_bound | \`nums[i] >= t\` | 第一个 ≥ t |
    | upper_bound | \`nums[i] > t\` | 第一个 > t |
    | 最后一个 ≤ t | \`nums[i] > t\` | 结果 − 1 |
    | 出现次数 | — | upper − lower |

    - 闭区间：\`hi = n-1\` · \`lo <= hi\` · \`hi = mid - 1\`
    - 每轮区间必须**严格缩小**；\`lo = mid\` 要配向上取整
    - C++/Java：\`mid = lo + (hi - lo) / 2\` 防溢出
    - 采样：\`searchsorted(cdf, u, side="right")\` = 第一个 cdf > u
  `,
})
