/**
 * 旋转数组与二分答案（algo-search-rotated）
 * LC33 / 153 / 162 / 69 / 875 / 1011；大模型岗场景：显存预算下二分最大 batch。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { RotatedSearchViz } from '../viz/RotatedSearchViz'
import { PredicateHalfViz } from '../viz/PredicateHalfViz'
import { AnswerSpaceViz } from '../viz/AnswerSpaceViz'
import { BatchProbeViz } from '../viz/BatchProbeViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'rotated',
      title: '旋转数组：切一刀，至少一半有序',
      body: md`
        把升序数组 \`[2,5,8,11,14,17,20,23,26,29,32]\` 的前 4 个挪到末尾，得到
        \`[14,17,20,23,26,29,32,2,5,8,11]\`——两段上坡，中间一个“断崖”。整体不有序，普通二分的
        “\`nums[mid] < target\` 就往右”不再成立。

        关键观察：**断崖只有一个**。任取 mid，把区间切成 \`[lo, mid]\` 和 \`[mid, hi]\`，断崖最多落在其中一边，
        所以**至少有一半是有序的**。判断方法只需一次比较：

        $$
        \begin{aligned}
        \text{nums}[lo] \le \text{nums}[mid] &\;\Rightarrow\; [lo, mid]\ \text{有序} \\
        \text{否则} &\;\Rightarrow\; [mid, hi]\ \text{有序}
        \end{aligned}
        $$

        对有序的那一半，target 在不在里面可以用**两端点**直接判断（$O(1)$）；在就进去，不在就去另一半。
        每轮依然丢掉一半，复杂度 $O(\log n)$。
      `,
      viz: RotatedSearchViz,
      after: md`
        > [!WARN] 两个最容易写错的地方
        > 1. 判断左半有序必须用 \`nums[lo] <= nums[mid]\`（带等号）。当 \`lo == mid\`（只剩两个元素）时左半只有一个元素，当然有序；写成 \`<\` 会误判成“右半有序”，在 \`[3, 1]\` 找 1 时返回 -1。
        > 2. 落在有序半段的判断是**半开**的：左半用 \`nums[lo] <= target < nums[mid]\`，右半用 \`nums[mid] < target <= nums[hi]\`——\`nums[mid]\` 已经比较过了，不能再算进去。
      `,
      checkpoint: {
        id: 'cp-rotated',
        type: 'single',
        prompt: md`在 \`nums = [6, 7, 9, 1, 3, 4]\` 中用上面的模板找 \`target = 3\`。第一轮 \`mid = 2\` 时，程序判断出的有序半段与下一步操作是：`,
        options: [
          '左半 `[0, 2]` 有序，target 不在其中 → `lo = 3`',
          '右半 `[2, 5]` 有序，target 在其中 → `lo = 3`',
          '左半 `[0, 2]` 有序，target 在其中 → `hi = 1`',
          '两半都无序，只能线性扫描',
        ],
        answer: 0,
        explanation: md`\`nums[0]=6 <= nums[2]=9\`，左半 \`[6,7,9]\` 有序；\`6 <= 3 < 9\` 不成立，所以去右半：\`lo = mid + 1 = 3\`。B 错在 \`[2,5]\` = \`[9,1,3,4]\` 含断崖，并不有序；D 错在“至少一半有序”永远成立。`,
      },
    },
    {
      id: 'predicate',
      title: '不找 target 的二分：最小值（153）与峰值（162）',
      body: md`
        很多二分题根本没有 target，套用上一关的“找第一个 True”骨架即可：\`lo < hi\`、\`lo = mid + 1\`、\`hi = mid\`。

        **LC153 旋转数组最小值**：定义 $P(i) = [\,\text{nums}[i] \le \text{nums}[n-1]\,]$。左段的值都大于 \`nums[-1]\`，右段都小于等于它，
        于是 $P$ 呈 F…F T…T，**最小值就是第一个 T**。代码里和 \`nums[hi]\` 比较（\`hi\` 始终停在右段内，二者等价）。

        为什么不和 \`nums[lo]\` 比？对**没有旋转**的 \`[1,2,3,4,5]\`，\`nums[mid] > nums[lo]\` 成立，但最小值在左边；
        和 \`nums[lo]\` 比较无法区分“整体有序”与“mid 在左段”，和 \`nums[hi]\` 比则没有这个歧义。

        **LC162 寻找峰值**（相邻元素不等，越界视为 $-\infty$）：数组**完全不单调**，但依然能二分。
        靠的是一个**不变量**：

        $$
        \begin{gathered}
        \text{nums}[lo-1] < \text{nums}[lo] \\
        \text{nums}[hi] > \text{nums}[hi+1]
        \end{gathered}
        $$

        两式同时成立，即区间两端都“往里走是上坡”，所以 \`[lo, hi]\` 内必有峰。看 \`nums[mid]\` 与 \`nums[mid+1]\`：上坡则 \`lo = mid + 1\`，下坡则 \`hi = mid\`，
        两种更新都保持不变量。
      `,
      viz: PredicateHalfViz,
      after: md`
        > [!INTERVIEW] 二分的本质
        > “二分不要求单调，只要求**每一步能确定答案在哪一半**并且保持一个不变量。153 靠单调谓词，162 靠‘区间内必有峰’，33 靠‘至少一半有序’。”这句话能明显拉开和背模板的候选人的差距。

        > [!WARN] 有重复元素时（LC81 / LC154）
        > \`[1, 1, 1, 1, 2, 1, 1]\` 中 \`nums[lo] == nums[mid] == nums[hi]\`，无法判断哪边有序，只能 \`hi -= 1\` 缩一步，**最坏退化到 $O(n)$**。这是信息论意义上的下界，不是写法问题。
      `,
      checkpoint: {
        id: 'cp-predicate',
        type: 'judge',
        prompt: md`LC162 中，若 \`nums[mid] > nums[mid + 1]\`，则峰值一定在 \`[lo, mid]\` 中，\`[mid+1, hi]\` 里一定没有峰。`,
        answer: false,
        explanation: md`错误。\`[mid+1, hi]\` 里也可能有峰（比如 \`[1, 5, 2, 7, 3]\`）。二分只保证**被保留的一半一定有峰**，并不保证被丢掉的一半没有——题目只要求返回任意一个峰，所以丢掉另一半是安全的。`,
      },
    },
    {
      id: 'answer',
      title: '二分答案：把“求最小可行值”变成 check(x) 的边界',
      body: md`
        **LC875 爱吃香蕉的珂珂**：\`piles = [3, 6, 7, 11]\`，每小时只能吃一堆中的至多 $k$ 根，问在 $h$ 小时内吃完的最小速度 $k$。

        直接求 $k$ 没有公式，但**验证**一个 $k$ 很容易：

        $$
        \begin{aligned}
        \text{hours}(k) &= \textstyle\sum_i \lceil p_i / k \rceil \\
        \text{check}(k) &= [\,\text{hours}(k) \le h\,]
        \end{aligned}
        $$

        $k$ 越大，$\text{hours}(k)$ 不增，所以 check 呈 F…F T…T——**在答案空间上二分**，找第一个 True。

        二分答案的三要素：

        | 要素 | 875 珂珂 | 1011 运包裹 | 69 平方根 |
        |---|---|---|---|
        | 答案范围 | \`[1, max(p)]\` | \`[max(w), sum(w)]\` | \`[0, x]\` |
        | check(x) | $\sum\lceil p_i/x\rceil \le h$ | 贪心装箱天数 $\le D$ | $x^2 > \text{target}$ |
        | 找什么 | 第一个 T | 第一个 T | 第一个 T，再减 1 |

        复杂度 = $O(\log(\text{范围}) \times \text{check 的代价})$，875 是 $O(n \log \max p)$。
      `,
      viz: AnswerSpaceViz,
      after: md`
        > [!TIP] 上下界怎么定
        > 下界取“一定不可行或刚好可能可行”的最小值，上界取“**一定可行**”的值（875 中 \`k = max(piles)\` 每堆 1 小时，只要 $h \ge n$ 就可行）。这样循环结束的 \`lo\` 必然是合法答案，不用特判。

        > [!WARN] 整数向上取整
        > 写 \`(p + k - 1) // k\` 而不是 \`math.ceil(p / k)\`：前者纯整数运算，不受浮点误差影响；面试中这是一个小加分点。
      `,
      checkpoint: {
        id: 'cp-answer',
        type: 'numeric',
        prompt: md`\`piles = [3, 6, 7, 11]\`、\`h = 6\`，最小速度 $k$ 是多少？`,
        answer: 6,
        tolerance: 0,
        explanation: md`$\text{hours}(5) = 1+2+2+3 = 8 > 6$，$\text{hours}(6) = 1+1+2+2 = 6 \le 6$，所以第一个 True 是 $k=6$。`,
      },
    },
    {
      id: 'llm',
      title: '大模型岗场景：显存预算下二分最大 batch',
      body: md`
        训练或推理前常要回答：“这张卡上 **batch / 上下文长度最多开多大**？”显存占用 $\text{mem}(b)$ 关于 batch $b$ 单调递增，
        所以 $\text{fits}(b) = [\,\text{mem}(b) \le \text{预算}\,]$ 呈 T…T F…F——标准的二分答案。

        以 **Llama-3-8B（BF16）推理**为例：
        - 权重：$8.03\text{B} \times 2\ \text{字节} \approx 16.1\ \text{GB} \approx 15\ \text{GiB}$；
        - 每个 token 的 KV cache：$2\,(K,V) \times 32\ \text{层} \times 8\ \text{KV 头} \times 128\ \text{维} \times 2\ \text{字节} = 128\ \text{KiB}$；
        - 上下文 8K 时一条序列的 KV 恰好 1 GiB。

        **上界未知怎么办？** 先倍增 $1, 2, 4, \dots$ 直到第一次 OOM，再在 $[b, 2b]$ 内二分。总试探次数约 $2\log_2 b^*$。
        PyTorch Lightning 的 \`BatchSizeFinder(mode="binsearch")\` 就是这个策略：先翻倍，遇到 OOM 后在最后一次成功与失败之间二分。
      `,
      viz: BatchProbeViz,
      after: md`
        > [!NOTE] 能算就别试
        > 上图的显存模型是闭式的，直接除法就能得到答案（图里也给出了对照）。二分 + 真实试跑适用于 **check 是黑盒**的情况：激活峰值、显存碎片、通信 buffer 都难以精确建模。
        > 推理引擎走的是另一条路：vLLM 启动时先用假数据做一次 profile 前向，测出峰值显存，再把 \`gpu_memory_utilization × 总显存 − 峰值\` 除以每个 KV block 的字节数，得到可分配的 block 数——一次测量 + 一次除法，无需二分。

        同理还可以二分“最大上下文长度”“满足延迟 SLO 的最大并发数”，前提永远是**先论证单调性**。
      `,
      checkpoint: {
        id: 'cp-llm',
        type: 'numeric',
        prompt: md`沿用上文：80 GiB 显存、预算取 90%（72 GiB），权重按 15 GiB 计，上下文 8K（每条序列 KV = 1 GiB），最多能放多少条序列？（忽略激活）`,
        answer: 57,
        tolerance: 0,
        explanation: md`$\lfloor (72 - 15) / 1 \rfloor = 57$。用倍增 + 二分需要试探：1、2、4、…、32 成功，64 失败，再在 $[33, 64]$ 内二分 5 次，共 12 次。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-sorted-half',
      type: 'single',
      difficulty: 1,
      prompt: md`LC33 中，为什么“旋转数组任取 mid，至少有一半有序”？`,
      options: [
        '旋转数组只有一个“断崖”（下降点），它最多落在两半中的一半',
        '因为 mid 总是落在较长的那段上坡里',
        '因为 nums[lo] 总是小于 nums[hi]',
        '只有当旋转点恰好在中间时才成立',
      ],
      answer: 0,
      explanation: md`升序数组旋转后恰好有一个下降点（未旋转时没有）。把 \`[lo, hi]\` 切成两半，下降点最多在其中一半，另一半必然有序。B 不成立（mid 可能在较短的段）；C 对旋转数组恰恰相反（通常 \`nums[lo] > nums[hi]\`）；D 是错误的直觉。`,
    },
    {
      id: 'q-dup',
      type: 'judge',
      difficulty: 1,
      prompt: md`允许重复元素的旋转数组（LC81），只要写法正确，仍能保证最坏 $O(\log n)$。`,
      answer: false,
      explanation: md`错误。例如 \`[1,1,1,1,1,2,1,1,1]\`，\`nums[lo] == nums[mid] == nums[hi]\` 时无法判断 2 在哪一半，只能 \`lo += 1\` 或 \`hi -= 1\` 一步步缩，最坏 $O(n)$。平均情况仍然很快，但最坏界无法改善。`,
    },
    {
      id: 'q-applicable',
      type: 'multi',
      difficulty: 2,
      prompt: md`以下哪些问题可以用“二分答案 + check(x)”求解？（多选）`,
      options: [
        'LC875：h 小时内吃完香蕉的最小速度',
        'LC1011：D 天内运完包裹的最低运载能力',
        'LC69：求 $\\lfloor\\sqrt{x}\\rfloor$',
        '在无序数组中找出现次数最多的元素',
        '求无序数组中两数之和等于 target 的下标',
      ],
      answer: [0, 1, 2],
      explanation: md`A、B、C 的可行性都关于答案单调：速度越快越能吃完、载重越大天数越少、$k^2 \le x$ 在 $k$ 上是 T…T F…F。D 的“众数”与任何参数都没有单调关系；E 是哈希表 / 排序 + 双指针问题，没有可二分的答案空间。`,
    },
    {
      id: 'q-koko',
      type: 'numeric',
      difficulty: 2,
      prompt: md`LC875：\`piles = [30, 11, 23, 4, 20]\`，\`h = 6\`，最小吃香蕉速度 $k$ 是多少？`,
      answer: 23,
      tolerance: 0,
      explanation: md`$\text{hours}(23) = 2+1+1+1+1 = 6 \le 6$；$\text{hours}(22) = 2+1+2+1+1 = 7 > 6$。所以第一个可行速度是 23。注意 $k$ 从 22 到 23 时，只有 23 那一堆从 2 小时降到 1 小时——hours 是**阶梯状**下降的，所以必须二分到精确边界，不能用公式估算。`,
    },
    {
      id: 'q-trace153',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        用 LC153 模板（\`lo, hi = 0, n-1\`，\`while lo < hi\`，与 \`nums[hi]\` 比较）在 \`[4, 5, 6, 7, 0, 1, 2]\` 上找最小值。
        第一轮 \`mid\` = ____(1)____；循环结束时 \`lo\` = ____(2)____。
      `,
      blanks: [
        { label: '(1)', answers: ['3'], numeric: true },
        { label: '(2)', answers: ['4'], numeric: true },
      ],
      explanation: md`第一轮 \`mid = 3\`（值 7 > \`nums[6]=2\`）→ \`lo = 4\`；第二轮 \`mid = 5\`（1 ≤ 2）→ \`hi = 5\`；第三轮 \`mid = 4\`（0 ≤ 1）→ \`hi = 4\`；\`lo == hi == 4\`，最小值 \`nums[4] = 0\`。`,
    },
    {
      id: 'q-steps',
      type: 'order',
      difficulty: 2,
      prompt: md`把“二分答案”的解题步骤按正确顺序排列：`,
      items: [
        '识别问题：求“满足条件的最小/最大值”，且直接求解困难',
        '写出 check(x)，并证明它关于 x 单调（F…F T…T）',
        '确定答案范围：下界与一定可行的上界',
        '用“找第一个 True”模板二分，每轮调用 check(mid)',
        '分析复杂度：O(log 范围 × check 代价)',
      ],
      explanation: md`先确认问题形态，再定义并证明 check 的单调性（这一步决定能不能二分），然后确定范围（保证答案在区间内）、套模板，最后给出复杂度。面试中跳过“证明单调性”是最常见的扣分点。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与它二分时使用的“判断依据”配对：`,
      pairs: [
        ['LC33 搜索旋转数组', '`nums[lo] <= nums[mid]` 判断哪半有序'],
        ['LC153 旋转数组最小值', '`nums[mid] > nums[hi]` 则最小值在右侧'],
        ['LC162 寻找峰值', '`nums[mid] < nums[mid+1]` 则右侧必有峰'],
        ['LC875 珂珂吃香蕉', '`hours(mid) <= h` 则可以更慢'],
      ],
      explanation: md`四道题都是“每轮能判定答案在哪一半”，但依据不同：33 看有序半段、153 看单调谓词、162 看坡向（不变量）、875 看答案空间上的 check。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LC153「寻找旋转排序数组中的最小值」（有 2 行干扰项）：`,
      lines: [
        'def find_min(nums):',
        '    lo, hi = 0, len(nums) - 1',
        '    while lo < hi:',
        '        mid = (lo + hi) // 2',
        '        if nums[mid] > nums[hi]:',
        '            lo = mid + 1',
        '        else:',
        '            hi = mid',
        '    return nums[lo]',
      ],
      distractors: ['        if nums[mid] > nums[lo]:', '            hi = mid - 1'],
      explanation: md`必须与 \`nums[hi]\` 比较：对未旋转的 \`[1,2,3]\`，\`nums[mid] > nums[lo]\` 会把 \`lo\` 推到右边，错过最小值 1。\`hi = mid - 1\` 会丢掉 mid——而 \`nums[mid] <= nums[hi]\` 时 mid 本身可能就是最小值。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 LC33 实现在 \`nums = [3, 1], target = 1\` 时返回 -1。点出有问题的行：`,
      code: py`
        def search(nums, target):
            lo, hi = 0, len(nums) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] == target:
                    return mid
                if nums[lo] < nums[mid]:
                    if nums[lo] <= target < nums[mid]:
                        hi = mid - 1
                    else:
                        lo = mid + 1
                else:
                    if nums[mid] < target <= nums[hi]:
                        lo = mid + 1
                    else:
                        hi = mid - 1
            return -1
      `,
      bugLines: [7],
      explanation: md`第 7 行应为 \`nums[lo] <= nums[mid]\`。\`[3,1]\` 中 \`lo = mid = 0\`，左半只有一个元素 \`[3]\`，是有序的；写成 \`<\` 时 \`3 < 3\` 为假，程序误以为右半 \`[3, 1]\` 有序，判断 \`3 < 1 <= 1\` 失败后执行 \`hi = -1\`，直接退出。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 3,
      prompt: md`用 LC33 闭区间模板在 \`[15, 18, 21, 2, 5, 8, 11, 13]\` 中查找 \`5\`，依次被选为 \`mid\` 的是哪些位置？（点出全部）`,
      items: ['15', '18', '21', '2', '5', '8', '11', '13'],
      answer: [3, 4, 5],
      explanation: md`
        1. \`lo=0, hi=7, mid=3\`（值 2）：\`nums[0]=15 > 2\` ⇒ 右半 \`[3,7]\` 有序，\`2 < 5 <= 13\` ⇒ \`lo=4\`；
        2. \`lo=4, hi=7, mid=5\`（值 8）：\`5 <= 8\` ⇒ 左半 \`[4,5]\` 有序，\`5 <= 5 < 8\` ⇒ \`hi=4\`；
        3. \`mid=4\`，命中。下标 3、5、4 被探查过。
      `,
    },
    {
      id: 'q-why-hi',
      type: 'multi',
      difficulty: 3,
      prompt: md`LC153 若改成“与 \`nums[lo]\` 比较”：\`if nums[mid] >= nums[lo]: lo = mid + 1 else: hi = mid\`（其余不变，返回 \`nums[lo]\`）。它在哪些输入上会返回**错误**结果？（多选）`,
      options: ['`[1, 2, 3, 4, 5]`（未旋转）', '`[3, 4, 5, 1, 2]`', '`[2, 1]`', '`[5, 1, 2, 3, 4]`'],
      answer: [0, 1],
      explanation: md`
        - A：未旋转时 \`nums[mid] >= nums[lo]\` 恒成立，\`lo\` 一路右移，返回 5。
        - B：mid=2（5 ≥ 3）→ lo=3；此时 \`lo == mid == 3\`，\`1 >= 1\` 恒真 → lo=4，返回 2。**一旦 lo 落到最小值上，与自己比较永远为真，就会被推过头**。
        - C：mid=0（2 ≥ 2）→ lo=1，返回 1，碰巧正确。
        - D：mid=2（2 < 5）→ hi=2；mid=1（1 < 5）→ hi=1；mid=0（5 ≥ 5）→ lo=1，返回 1，正确。

        与 \`nums[hi]\` 比较时 \`hi\` 始终在右段（最小值所在段），不会出现这种歧义。
      `,
    },
    {
      id: 'q-kv',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Llama-3-8B：32 层、8 个 KV 头、head_dim = 128，KV cache 用 BF16（2 字节）。**每个 token** 的 KV cache 是多少 KiB？（1 KiB = 1024 字节）`,
      answer: 128,
      tolerance: 0,
      unit: 'KiB',
      explanation: md`$2\,(K \text{ 和 } V) \times 32 \times 8 \times 128 \times 2 = 131072$ 字节 $= 128$ KiB。常见错误：用 32 个 Q 头代替 8 个 KV 头（GQA），会得到 512 KiB；或漏掉 K、V 两份。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-rotated',
      title: '一步步写出“搜索旋转排序数组”',
      intro: md`
        实现 \`search(nums, target)\`：\`nums\` 是**元素互不相同**的升序数组旋转后的结果，返回 target 的下标，不存在返回 -1。
        骨架用闭区间 \`[lo, hi]\`，核心是每轮先判断哪半有序。
      `,
      template: py`
        def search(nums, target):
            lo, hi = 0, len(nums) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] == target:
                    return mid
                if {{1}}:                          # 左半 [lo, mid] 有序
                    if {{2}}:                      # target 落在有序的左半
                        hi = mid - 1
                    else:
                        lo = mid + 1
                else:                              # 右半 [mid, hi] 有序
                    if {{3}}:                      # target 落在有序的右半
                        {{4}}
                    else:
                        {{5}}
            return -1
      `,
      steps: [
        {
          blank: 1,
          instruction: md`判断左半 \`[lo, mid]\` 是否有序。`,
          answer: 'nums[lo] <= nums[mid]',
          accept: ['nums[mid] >= nums[lo]'],
          hint: '注意 lo == mid 的情况。',
          explain: md`必须带等号：只剩两个元素时 \`lo == mid\`，单元素区间当然有序。`,
        },
        {
          blank: 2,
          instruction: md`左半有序时，target 是否落在 \`[nums[lo], nums[mid])\` 中？`,
          answer: 'nums[lo] <= target < nums[mid]',
          accept: ['nums[lo] <= target and target < nums[mid]'],
          explain: md`右端用 \`<\`：\`nums[mid] == target\` 已在上面判断过。左端用 \`<=\`：target 可能恰好是 \`nums[lo]\`。`,
        },
        {
          blank: 3,
          instruction: md`右半有序时，target 是否落在 \`(nums[mid], nums[hi]]\` 中？`,
          answer: 'nums[mid] < target <= nums[hi]',
          accept: ['nums[mid] < target and target <= nums[hi]'],
          explain: md`与上一空对称：mid 端开、hi 端闭。`,
        },
        {
          blank: 4,
          instruction: md`target 在有序的右半，收缩哪一端？`,
          answer: 'lo = mid + 1',
          explain: md`丢掉左半（含 mid）。`,
        },
        {
          blank: 5,
          instruction: md`target 不在有序的右半，它只可能在左半。`,
          answer: 'hi = mid - 1',
          explain: md`注意这里“左半”可能是无序的——没关系，下一轮会重新判断哪半有序。`,
        },
      ],
      setup: py`
        import random
        def _ref_rot(n, k, rnd):
            base = sorted(rnd.sample(range(-50, 50), n))
            return base[k:] + base[:k]
      `,
      tests: [
        {
          name: '示例',
          code: py`
            assert search([4, 5, 6, 7, 0, 1, 2], 0) == 4
            assert search([4, 5, 6, 7, 0, 1, 2], 3) == -1
            assert search([3, 1], 1) == 1, "两元素：lo == mid 的情况"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(33)
            for _ in range(300):
                n = rnd.randint(1, 12)
                a = _ref_rot(n, rnd.randrange(n), rnd)
                t = rnd.choice(a + [99, -99, 0])
                exp = a.index(t) if t in a else -1
                got = search(a, t)
                assert got == exp, f"nums={a}, target={t}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        这个骨架的精髓是“**先定有序半段，再用端点判断归属**”。如果允许重复（LC81），只需在最前面加一个分支：
        \`nums[lo] == nums[mid] == nums[hi]\` 时 \`lo += 1; hi -= 1\`。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-search-rotated',
      title: '搜索旋转排序数组',
      difficulty: 'medium',
      source: 'LeetCode 33',
      timeLimitMin: 15,
      description: md`
        整数数组 \`nums\` 原本按升序排列、**元素互不相同**，在某个未知下标处被旋转（例如 \`[0,1,2,4,5,6,7]\` → \`[4,5,6,7,0,1,2]\`）。
        给定 \`target\`，返回它的下标；不存在返回 \`-1\`。

        **要求时间复杂度 $O(\log n)$**（隐藏测试会统计你访问 \`nums[i]\` 的次数）。

        \`\`\`python
        def search(nums: list[int], target: int) -> int: ...
        \`\`\`

        **示例**
        - \`search([4,5,6,7,0,1,2], 0)\` → \`4\`
        - \`search([4,5,6,7,0,1,2], 3)\` → \`-1\`
        - \`search([1], 0)\` → \`-1\`

        **约束**：$1 \le n \le 10^5$；数组可能没有旋转。
      `,
      starter: py`
        def search(nums: list[int], target: int) -> int:
            # TODO: 每轮判断哪一半有序，再看 target 是否在有序的那一半
            raise NotImplementedError
      `,
      solution: py`
        def search(nums: list[int], target: int) -> int:
            lo, hi = 0, len(nums) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] == target:
                    return mid
                if nums[lo] <= nums[mid]:              # 左半 [lo, mid] 有序
                    if nums[lo] <= target < nums[mid]:
                        hi = mid - 1
                    else:
                        lo = mid + 1
                else:                                  # 右半 [mid, hi] 有序
                    if nums[mid] < target <= nums[hi]:
                        lo = mid + 1
                    else:
                        hi = mid - 1
            return -1
      `,
      solutionAlt: [
        {
          label: '另一种思路：先找旋转点，再做普通二分',
          code: py`
            from bisect import bisect_left

            def search(nums, target):
                n = len(nums)
                lo, hi = 0, n - 1                      # 1) LC153：找最小值下标 k
                while lo < hi:
                    mid = (lo + hi) // 2
                    if nums[mid] > nums[hi]:
                        lo = mid + 1
                    else:
                        hi = mid
                k = lo
                # 2) target 落在哪一段由 nums[-1] 决定，在该段上做标准二分
                l, r = (k, n) if target <= nums[-1] else (0, k)
                i = bisect_left(nums, target, l, r)
                return i if i < r and nums[i] == target else -1
          `,
        },
      ],
      explanation: md`
        **思路**：旋转数组只有一个下降点，任取 mid，\`[lo, mid]\` 与 \`[mid, hi]\` 至少一个有序。
        用 \`nums[lo] <= nums[mid]\` 判断左半是否有序；对有序的半段用两个端点判断 target 是否在其中，在则进入，不在则去另一半。

        **复杂度**：时间 $O(\log n)$，空间 $O(1)$。

        **常见错误**
        - \`nums[lo] < nums[mid]\` 少了等号，两元素时出错；
        - 有序半段的判断写成闭区间 \`nums[lo] <= target <= nums[mid]\`——功能上碰巧也对（mid 已排除），但右半写成 \`nums[mid] <= target\` 时概念不清；
        - 先线性找旋转点，退化为 $O(n)$。

        **另解**：先用 LC153 找旋转点 $k$，再根据 \`target <= nums[-1]\` 决定在 \`[k, n)\` 还是 \`[0, k)\` 上做标准二分，两次 $O(\log n)$。
      `,
      complexity: { time: 'O(log n)', space: 'O(1)' },
      hints: [
        md`任取 mid，左右两半至少有一半是有序的。怎么用一次比较判断左半是否有序？`,
        md`\`nums[lo] <= nums[mid]\` ⇒ 左半有序。此时 target 在左半当且仅当 \`nums[lo] <= target < nums[mid]\`。`,
        md`右半有序时对称：\`nums[mid] < target <= nums[hi]\` 则 \`lo = mid + 1\`，否则 \`hi = mid - 1\`。`,
      ],
      setup: py`
        import math, random
        from collections.abc import Sequence

        def _ref_rot(n, k, rnd):
            base = sorted(rnd.sample(range(-1000, 1000), n))
            return base[k:] + base[:k]

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
        { name: '示例 1', code: py`assert search([4, 5, 6, 7, 0, 1, 2], 0) == 4` },
        { name: '示例 2：不存在', code: py`assert search([4, 5, 6, 7, 0, 1, 2], 3) == -1` },
        { name: '示例 3：单元素', code: py`assert search([1], 0) == -1 and search([1], 1) == 0` },
        {
          name: '未旋转 / 两元素 / 两端',
          code: py`
            assert search([1, 2, 3, 4, 5], 1) == 0
            assert search([1, 2, 3, 4, 5], 5) == 4
            assert search([3, 1], 1) == 1, "[3,1] 找 1：注意 nums[lo] <= nums[mid] 的等号"
            assert search([3, 1], 3) == 0
            assert search([5, 1, 3], 5) == 0
            assert search([5, 1, 3], 3) == 2
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(2033)
            for _ in range(400):
                n = rnd.randint(1, 15)
                a = _ref_rot(n, rnd.randrange(n), rnd)
                t = rnd.choice(a + [1001, -1001, 0])
                exp = a.index(t) if t in a else -1
                got = search(a, t)
                assert got == exp, f"nums={a}, target={t}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：n = 10^5',
          hidden: true,
          code: py`
            n = 100_000
            base = list(range(0, 2 * n, 2))
            k = 61_803
            data = base[k:] + base[:k]
            limit = 5 * math.ceil(math.log2(n + 1)) + 10
            for t in (data[0], data[-1], data[n // 2], 7, base[k]):
                p = _Probe(data)
                got = search(p, t)
                exp = data.index(t) if t % 2 == 0 else -1
                assert got == exp, f"target={t}: 期望 {exp}，得到 {got}"
                assert p.reads <= limit, f"target={t} 访问了 nums {p.reads} 次，超过 O(log n) 上限 {limit}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-dup',
          q: md`如果数组中允许有重复元素（LC81），代码怎么改？复杂度如何？`,
          a: md`在判断有序半段之前加一个分支：\`nums[lo] == nums[mid] == nums[hi]\` 时无法判断，只能 \`lo += 1; hi -= 1\`。最坏情况（几乎全相同）退化为 $O(n)$，且无法改进——任何算法都必须看遍那些相同的元素才能找到唯一不同的那个。`,
        },
        {
          id: 'f-rotate-k',
          q: md`如何求数组被旋转了多少位？`,
          a: md`旋转位数 = 最小值的下标，即 LC153 的结果：与 \`nums[hi]\` 比较的“找第一个 True”二分，$O(\log n)$。`,
        },
      ],
    },
    {
      id: 'c-koko',
      title: '爱吃香蕉的珂珂',
      difficulty: 'medium',
      source: 'LeetCode 875',
      timeLimitMin: 15,
      description: md`
        有 $n$ 堆香蕉 \`piles\`，守卫 \`h\` 小时后回来（保证 $h \ge n$）。珂珂每小时选一堆，吃掉其中至多 $k$ 根；
        若该堆不足 $k$ 根，吃完后这一小时不再吃别的。求能在 \`h\` 小时内吃完所有香蕉的**最小整数速度** $k$。

        \`\`\`python
        def min_eating_speed(piles: list[int], h: int) -> int: ...
        \`\`\`

        **示例**
        - \`min_eating_speed([3,6,7,11], 8)\` → \`4\`
        - \`min_eating_speed([30,11,23,4,20], 5)\` → \`30\`
        - \`min_eating_speed([30,11,23,4,20], 6)\` → \`23\`

        **约束**：$1 \le n \le 10^4$，$n \le h \le 10^9$，$1 \le p_i \le 10^9$。
        要求 $O(n \log \max p)$：隐藏测试会统计你遍历 \`piles\` 的次数，逐个尝试 $k$ 的写法无法通过。
      `,
      starter: py`
        def min_eating_speed(piles: list[int], h: int) -> int:
            # TODO: 1) 写 check(k)  2) 在 [1, max(piles)] 上找第一个 check 为 True 的 k
            raise NotImplementedError
      `,
      solution: py`
        def min_eating_speed(piles: list[int], h: int) -> int:
            def check(k: int) -> bool:
                # 速度 k 下需要的小时数；(p + k - 1) // k 是整数向上取整
                return sum((p + k - 1) // k for p in piles) <= h

            lo, hi = 1, max(piles)          # k = max(piles) 一定可行（h >= n）
            while lo < hi:
                mid = (lo + hi) // 2
                if check(mid):
                    hi = mid                # mid 可行，答案在 [lo, mid]
                else:
                    lo = mid + 1            # mid 太慢
            return lo
      `,
      explanation: md`
        **思路**：$\text{hours}(k) = \sum \lceil p_i / k \rceil$ 随 $k$ 增大而不增，故 $\text{check}(k) = [\text{hours}(k) \le h]$ 呈 F…F T…T，
        在 $[1, \max p]$ 上找第一个 True。上界 $\max p$ 时每堆恰好 1 小时，共 $n \le h$ 小时，一定可行。

        **复杂度**：$O(n \log \max p)$，$\max p = 10^9$ 时约 30 轮 check。

        **常见错误**
        - 上界取 \`sum(piles)\` 也对但多几轮；下界取 0 会除零；
        - \`math.ceil(p / k)\` 在本题范围内恰好安全，但整数写法 \`(p + k - 1) // k\` 更稳；
        - check 写成 \`< h\`（应为 \`<= h\`）。
      `,
      complexity: { time: 'O(n log max(p))', space: 'O(1)' },
      hints: [
        md`给定速度 $k$，你能在 $O(n)$ 内算出需要多少小时吗？这个小时数随 $k$ 怎么变化？`,
        md`check(k) = 小时数 ≤ h，它关于 $k$ 单调。答案范围是 \`[1, max(piles)]\`。`,
        md`套用“找第一个 True”：\`check(mid)\` 为真则 \`hi = mid\`，否则 \`lo = mid + 1\`。`,
      ],
      setup: py`
        import random
        from collections.abc import Sequence

        def _ref_koko(piles, h):
            lo, hi = 1, max(piles)
            while lo < hi:
                mid = (lo + hi) // 2
                if sum((p + mid - 1) // mid for p in piles) <= h:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        class _Budget(Sequence):
            """只读序列：访问元素超过预算时立即失败（防止逐个尝试 k 的写法跑很久）"""
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
                    raise AssertionError(f"遍历 piles 的元素次数超过 {self.budget}，复杂度不是 O(n log max p)")
                return r
      `,
      tests: [
        { name: '示例 1', code: py`assert min_eating_speed([3, 6, 7, 11], 8) == 4` },
        { name: '示例 2', code: py`assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30` },
        { name: '示例 3', code: py`assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23` },
        {
          name: '边界',
          code: py`
            assert min_eating_speed([1], 1) == 1
            assert min_eating_speed([10**9], 2) == 500_000_000
            assert min_eating_speed([5, 5, 5], 10**9) == 1, "时间极其充裕时速度为 1"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(875)
            for _ in range(300):
                piles = [rnd.randint(1, 60) for _ in range(rnd.randint(1, 8))]
                h = rnd.randint(len(piles), sum(piles) + 3)
                exp, got = _ref_koko(piles, h), min_eating_speed(piles, h)
                assert got == exp, f"piles={piles}, h={h}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：n = 2000，p 到 1e9',
          hidden: true,
          code: py`
            rnd = random.Random(1)
            data = [rnd.randint(10**8, 10**9) for _ in range(2000)]
            h = 2500
            exp = _ref_koko(data, h)
            p = _Budget(data, 70 * len(data))
            got = min_eating_speed(p, h)
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-1011',
          q: md`LC1011「D 天内送达包裹」和本题有什么异同？`,
          a: md`同样是二分答案，答案是运载能力 $c$，范围 \`[max(w), sum(w)]\`（下界保证最重的包裹能装上）。check 变成**贪心**：按顺序装，超了就开新的一天，数天数是否 $\le D$。区别在于包裹必须按顺序且不可拆分，所以 check 是贪心模拟而不是求和公式。`,
        },
        {
          id: 'f-float',
          q: md`如果答案是实数（如“最小化最大距离”保留 6 位小数），二分怎么写？`,
          a: md`用 \`while hi - lo > eps\` 或固定迭代 100 次（每次区间减半，100 次后精度远超 double），更新时 \`lo = mid\` / \`hi = mid\`（实数二分没有 ±1）。固定次数的写法更稳，不会因 eps 选得太小而死循环。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-why',
      freq: 3,
      q: md`旋转数组为什么还能用二分？有重复元素时呢？`,
      a: md`
        **结论**：能。因为每一轮都能确定 target 在哪一半，而不是因为数组有序。

        **原因**：升序数组旋转后只有一个下降点，任取 mid 后 \`[lo, mid]\` 与 \`[mid, hi]\` 至少一个有序（\`nums[lo] <= nums[mid]\` 即左半有序）。对有序半段用端点 $O(1)$ 判断 target 是否在内，从而每轮丢一半。

        **延伸**：有重复时（LC81 / 154），\`nums[lo] == nums[mid] == nums[hi]\` 无法区分，只能两端各缩一步，最坏 $O(n)$——这是下界，不是写法问题。
      `,
    },
    {
      id: 'qa-answer',
      freq: 3,
      q: md`什么是“二分答案”？怎么判断一道题能不能用？`,
      a: md`
        **结论**：在**答案的取值空间**上二分，前提是存在单调的可行性判定 check(x)。

        **原因**：很多“最小化最大值 / 最大化最小值 / 最小可行值”问题直接求解难，但给定候选答案验证容易（贪心或求和，通常 $O(n)$）。若 x 可行蕴含所有更大的 x 都可行，check 就呈 F…F T…T，可以用 $O(\log R)$ 次 check 找边界。

        **延伸**：判断信号是题目里出现“最小的最大值”“至少”“能否在 X 内完成”。务必说明单调性的理由（例如“速度越快用时越少”），并保证上界一定可行。总复杂度 $O(\log R \cdot C_{\text{check}})$。
      `,
    },
    {
      id: 'qa-peak',
      freq: 2,
      q: md`寻找峰值（LC162）的数组并不单调，为什么能二分？`,
      a: md`
        **结论**：二分依赖的是**不变量**而非单调性。

        **原因**：维护“\`nums[lo-1] < nums[lo]\` 且 \`nums[hi] > nums[hi+1]\`（越界为 −∞）”，于是 \`[lo, hi]\` 内必有峰。若 \`nums[mid] < nums[mid+1]\`，令 \`lo = mid+1\` 后左侧条件仍成立；否则令 \`hi = mid\` 后右侧条件仍成立。区间缩到一点时它比两个邻居都大。

        **延伸**：这说明二分的本质是“每步排除一部分且答案仍在剩余部分”。同类思想还有三分法求单峰函数极值。
      `,
    },
    {
      id: 'qa-llm',
      freq: 2,
      q: md`在显存受限时，如何确定训练 / 推理能用的最大 batch size 或上下文长度？`,
      a: md`
        **结论**：能建模就直接算；建模不准就“倍增 + 二分”试跑。

        **原因**：显存占用随 batch 单调递增，fits(b) 呈 T…T F…F。推理时 KV cache 可以闭式计算（例如 Llama-3-8B 每 token 128 KiB），vLLM 就是先 profile 一次峰值显存，再用剩余显存除以 block 大小得到 KV block 数。训练时激活、碎片、通信 buffer 难以精确建模，就先翻倍找到第一次 OOM，再在 $[b, 2b]$ 内二分，约 $2\log_2 b^*$ 次试跑（PyTorch Lightning 的 BatchSizeFinder binsearch 模式即如此）。

        **延伸**：试跑要用最长的序列（最坏情况），并留余量应对碎片；OOM 后要 \`torch.cuda.empty_cache()\` 并确认没有残留引用。
      `,
    },
  ],

  cheatsheet: md`
    **LC33 旋转数组搜索**
    \`\`\`python
    if nums[lo] <= nums[mid]:                    # 左半有序（带等号！）
        if nums[lo] <= target < nums[mid]: hi = mid - 1
        else:                              lo = mid + 1
    else:                                        # 右半有序
        if nums[mid] < target <= nums[hi]: lo = mid + 1
        else:                              hi = mid - 1
    \`\`\`

    | 题 | 谓词 / 依据 | 更新 |
    |---|---|---|
    | 153 最小值 | \`nums[mid] > nums[hi]\` | 真 → \`lo=mid+1\`，否则 \`hi=mid\` |
    | 162 峰值 | \`nums[mid] < nums[mid+1]\` | 真 → \`lo=mid+1\`，否则 \`hi=mid\` |
    | 二分答案 | \`check(mid)\` 单调 | 真 → \`hi=mid\`，否则 \`lo=mid+1\` |

    - 二分答案三要素：**范围（上界必可行）· check 单调性证明 · 找第一个 T**
    - 向上取整：\`(p + k - 1) // k\`
    - 有重复（81/154）：三者相等时 \`hi -= 1\`，最坏 $O(n)$
    - 上界未知：先倍增再二分，约 $2\log_2 b^*$ 次
    - Llama-3-8B KV：$2 \times 32 \times 8 \times 128 \times 2$ B = 128 KiB / token
  `,
})
