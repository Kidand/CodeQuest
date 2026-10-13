/**
 * 双指针（algo-search-twopointers）
 * LC167 / 15 / 11 / 42 / 283：对撞指针的正确性、三数之和去重、接雨水、同向读写指针。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { TwoSumSpaceViz } from '../viz/TwoSumSpaceViz'
import { ThreeSumViz } from '../viz/ThreeSumViz'
import { WaterViz } from '../viz/WaterViz'
import { MoveZeroesViz } from '../viz/MoveZeroesViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'collide',
      title: '对撞指针：为什么可以放心丢掉一侧',
      body: md`
        **LC167**：升序数组中找两个数，和为 target。暴力枚举 $\binom{n}{2}$ 对是 $O(n^2)$；对撞指针 \`l = 0, r = n - 1\` 只要 $O(n)$。

        很多人会写，但说不清**为什么不会漏解**。把所有数对 $(i, j),\ i<j$ 排成一个上三角矩阵，格子里是 $\text{nums}[i] + \text{nums}[j]$：
        同一行从左到右递增，同一列从下到上递增。指针 $(l, r)$ 从右上角出发：

        - $s < \text{target}$：$\text{nums}[l]$ 配**最大的** $\text{nums}[r]$ 都不够，配第 $l$ 行其它任何数更不够 ⇒ **整行排除**，\`l += 1\`；
        - $s > \text{target}$：$\text{nums}[r]$ 配**最小的** $\text{nums}[l]$ 都太大 ⇒ **整列排除**，\`r -= 1\`。

        每一步排除一整行或一整列，最多 $n-1$ 步就排除完全部 $O(n^2)$ 个格子——这就是“剪枝”的严格含义。
      `,
      viz: TwoSumSpaceViz,
      after: md`
        > [!INTERVIEW] 正确性怎么讲
        > “每次移动指针时，被跳过的数对都已经被**证明**不可能是解：\`s < target\` 时，\`nums[l]\` 与 \`[l+1, r]\` 中任何数之和都 ≤ s。所以我们只是在排除不可能的候选，不会漏解。”画出这个矩阵是最有说服力的回答。

        > [!WARN] 前提是有序
        > 无序数组上对撞指针没有依据。LC1（无序、返回原下标）应该用哈希表 $O(n)$；若先排序再对撞，要记得保存原下标。
      `,
      checkpoint: {
        id: 'cp-collide',
        type: 'single',
        prompt: md`对撞指针中 \`nums[l] + nums[r] < target\` 时执行 \`l += 1\`。被“丢掉”的候选数对是：`,
        options: [
          '`(l, j)`，其中 `l < j ≤ r`——它们的和都 ≤ 当前和',
          '`(i, r)`，其中 `l ≤ i < r`',
          '只有 `(l, r)` 这一对',
          '`(l, j)`，其中 `j > r`',
        ],
        answer: 0,
        explanation: md`\`nums[l] + nums[j] ≤ nums[l] + nums[r] < target\`（因为 \`nums[j] ≤ nums[r]\`），所以第 l 行剩余的所有格子都不可能是解。B 是“整列排除”的情形（对应 \`s > target\`）；D 中 \`j > r\` 的格子早已在之前的步骤中被排除。`,
      },
    },
    {
      id: 'threesum',
      title: '三数之和：排序 + 固定一个 + 对撞，去重是关键',
      body: md`
        **LC15**：找出所有和为 0 且**不重复**的三元组。思路：

        1. 排序（$O(n \log n)$），使相同值相邻、使对撞指针可用；
        2. 枚举最小的数 \`nums[i]\`，在 \`[i+1, n-1]\` 上用对撞指针找两数之和 $= -\text{nums}[i]$；
        3. 两处去重：
           - **外层**：\`i > 0 and nums[i] == nums[i-1]\` 时跳过——以这个值为最小数的三元组上一轮已全部找到；
           - **内层**：命中后，把 \`l\` 跳过所有相同的值，再 \`l += 1, r -= 1\`。

        剪枝：\`nums[i] > 0\` 时直接 \`break\`（三个数都 > 0）。总复杂度 $O(n^2)$。
      `,
      viz: ThreeSumViz,
      after: md`
        > [!WARN] 外层去重的方向
        > 必须和**前一个**比较（\`nums[i] == nums[i-1]\`），不能和后一个比较（\`nums[i] == nums[i+1]\`）。后者在 \`[-1, -1, 2]\` 上会跳过 \`i = 0\`，于是 \`i = 1\` 时 \`l = r = 2\`，漏掉 \`[-1, -1, 2]\`——“同值的第一个”必须被处理，因为它的右边可能还要用到另一个同值元素。

        > [!TIP] 推广到 k 数之和
        > 四数之和（LC18）= 两层枚举 + 对撞，$O(n^3)$；一般 k 数之和是 $O(n^{k-1})$，可以写成递归，最底层用对撞指针。
      `,
      checkpoint: {
        id: 'cp-threesum',
        type: 'judge',
        prompt: md`三数之和中，命中解后**只**执行 \`l += 1; r -= 1\`、不跳过重复值，最终结果可能包含重复的三元组。`,
        answer: true,
        explanation: md`正确。例如 \`[-2, 1, 1, 1, 1]\`：命中 \`[-2,1,1]\` 后 \`l, r\` 各走一步，新的 \`nums[l], nums[r]\` 仍是 1 和 1，再次命中同一个三元组。必须至少跳过一侧的重复值（通常跳 \`l\`）。`,
      },
    },
    {
      id: 'water',
      title: '接雨水与盛水容器：短板决定一切',
      body: md`
        **LC42 接雨水**：第 $i$ 列的水量 $= \min(\text{leftMax}_i, \text{rightMax}_i) - h_i$，其中两个最大值都**包含**第 $i$ 列本身。

        直接做法是预处理前缀最大、后缀最大两个数组，$O(n)$ 时间、$O(n)$ 空间。双指针把空间降到 $O(1)$，关键论证：

        > 维护 \`lmax = max(h[0..l])\`、\`rmax = max(h[r..n-1])\`。若 \`lmax < rmax\`，则第 $l$ 列真正的右侧最大值 $\ge$ \`rmax\` $>$ \`lmax\`，
        > 所以 $\min(\text{leftMax}_l, \text{rightMax}_l) = $ \`lmax\`——**第 l 列的水量此刻就能确定**，不需要知道右边的全部信息。

        **LC11 盛最多水的容器**：面积 $= \min(h_l, h_r) \times (r - l)$。每次移动**短板**：以短板为一条边的其它容器，宽度更小、高度不超过短板，面积不可能更大，可以整体丢弃（和对撞指针的“整行排除”同一个逻辑）。
      `,
      viz: WaterViz,
      after: md`
        > [!NOTE] 接雨水的四种解法
        > | 解法 | 时间 | 空间 | 思路 |
        > |---|---|---|---|
        > | 暴力 | $O(n^2)$ | $O(1)$ | 每列向两边扫最大值 |
        > | 前后缀最大值 | $O(n)$ | $O(n)$ | 两个数组 |
        > | 单调栈 | $O(n)$ | $O(n)$ | 按“层”横向计算 |
        > | 双指针 | $O(n)$ | $O(1)$ | 较矮一侧的水位已确定 |
        >
        > 面试时先说前后缀数组（最好想），再优化到双指针，展示思考过程。
      `,
      checkpoint: {
        id: 'cp-water',
        type: 'numeric',
        prompt: md`\`height = [4, 2, 0, 3, 2, 5]\` 能接多少单位雨水？`,
        answer: 9,
        tolerance: 0,
        explanation: md`各列水位 = min(左最大, 右最大)：第 1–4 列都是 \`min(4, 5) = 4\`，水量分别为 \`4-2, 4-0, 4-3, 4-2\` = 2 + 4 + 1 + 2 = 9。两端的柱子不存水。`,
      },
    },
    {
      id: 'samedir',
      title: '同向双指针：读写指针原地改数组',
      body: md`
        对撞指针从两端向中间走；**同向双指针**两个指针都从左往右，速度不同：\`fast\` 负责读、\`slow\` 负责写。

        经典不变量：**\`[0, slow)\` 是已经处理好的结果**。每读到一个“要保留”的元素，就写到 \`slow\` 处并 \`slow += 1\`。

        | 题目 | 保留条件 | 写法 |
        |---|---|---|
        | LC283 移动零 | \`nums[fast] != 0\` | 交换 \`nums[slow], nums[fast]\` |
        | LC26 删除有序数组重复项 | \`nums[fast] != nums[slow-1]\` | 赋值 \`nums[slow] = nums[fast]\` |
        | LC27 移除元素 | \`nums[fast] != val\` | 赋值 |

        LC283 用交换而不是赋值，是为了让 0 自然地被换到后面，省掉最后补 0 的循环。
        链表中的快慢指针（判环、找中点、倒数第 k 个）是同一思想，会在数据结构章节详细讲。
      `,
      viz: MoveZeroesViz,
      after: md`
        > [!TIP] 写同向双指针前先说不变量
        > “\`[0, slow)\` 是结果、\`[slow, fast)\` 是被丢弃的元素、\`[fast, n)\` 未处理。”说清楚这三段，代码几乎不会写错，也方便面试官跟上你的思路。
      `,
      checkpoint: {
        id: 'cp-samedir',
        type: 'fill',
        prompt: md`用读写指针对有序数组 \`[0, 0, 1, 1, 1, 2, 3, 3]\` 去重（LC26），结束时 \`slow\` = ____(1)____，数组前 \`slow\` 个元素之和 = ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['4'], numeric: true },
          { label: '(2)', answers: ['6'], numeric: true },
        ],
        explanation: md`去重后前缀为 \`[0, 1, 2, 3]\`，\`slow = 4\`，和为 6。\`slow\` 之后的内容是什么无关紧要（题目只看前 \`slow\` 个）。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-why-l',
      type: 'single',
      difficulty: 1,
      prompt: md`升序数组上做两数之和，\`nums[l] + nums[r] > target\` 时应该：`,
      options: ['`r -= 1`', '`l += 1`', '`l += 1` 和 `r -= 1` 都可以', '重新从 `l = 0` 开始'],
      answer: 0,
      explanation: md`和太大，需要变小。\`nums[r]\` 配最小的 \`nums[l]\` 都太大，第 r 列可以整列排除，所以 \`r -= 1\`。\`l += 1\` 会让和更大，且可能跳过真正的解（例如解恰好是 \`(l, r-1)\`）。`,
    },
    {
      id: 'q-unsorted',
      type: 'judge',
      difficulty: 1,
      prompt: md`对撞指针可以直接用于**无序**数组的两数之和（LC1），只是可能需要多走几步。`,
      answer: false,
      explanation: md`错误。对撞指针的“整行/整列排除”依赖单调性，无序数组上 \`s < target\` 并不能推出第 l 行都太小，会漏解。LC1 应用哈希表：遍历时查 \`target - x\` 是否出现过，$O(n)$ 时间。`,
    },
    {
      id: 'q-dedup',
      type: 'multi',
      difficulty: 2,
      prompt: md`三数之和（已排序）中，以下哪些去重做法能得到**正确**结果？（多选）`,
      options: [
        '外层：`if i > 0 and nums[i] == nums[i - 1]: continue`',
        '外层：`if i + 1 < n and nums[i] == nums[i + 1]: continue`',
        '命中后：`while l < r and nums[l] == nums[l + 1]: l += 1`，再 `l += 1; r -= 1`',
        '不做任何去重，最后把结果转成 `set(tuple(t) for t in res)`',
      ],
      answer: [0, 2, 3],
      explanation: md`A、C 是标准写法。D 也正确，但需要额外的集合空间，且产生重复结果的过程本身会浪费时间（面试官通常会追问如何避免）。B 错误：在 \`[-1, -1, 2]\` 上会跳过 \`i = 0\`，而 \`i = 1\` 时右边只剩一个元素，漏掉 \`[-1, -1, 2]\`。`,
    },
    {
      id: 'q-trap',
      type: 'numeric',
      difficulty: 2,
      prompt: md`\`height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]\` 能接多少单位雨水？`,
      answer: 6,
      tolerance: 0,
      explanation: md`逐列 \`min(左最大, 右最大) - h\`：下标 2 得 1，下标 4 得 1，下标 5 得 2，下标 6 得 1，下标 9 得 1，合计 6。可以在图解中的可视化里逐步验证。`,
    },
    {
      id: 'q-container',
      type: 'fill',
      difficulty: 2,
      prompt: md`LC11：\`height = [1, 8, 6, 2, 5, 4, 8, 3, 7]\`。第一步计算的面积是 ____(1)____；最大面积是 ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['8'], numeric: true },
        { label: '(2)', answers: ['49'], numeric: true },
      ],
      explanation: md`第一步 \`min(1, 7) × 8 = 8\`，然后丢掉短板 1（\`l += 1\`）。最大面积由下标 1（高 8）与下标 8（高 7）构成：\`min(8, 7) × 7 = 49\`。`,
    },
    {
      id: 'q-steps',
      type: 'order',
      difficulty: 2,
      prompt: md`把三数之和的算法步骤按执行顺序排列：`,
      items: [
        '对数组排序',
        '枚举 i；若 `nums[i] > 0` 则 break',
        '若 `nums[i]` 与前一个相同则 continue',
        '令 `l = i + 1, r = n - 1`，按和与 0 的大小关系移动指针',
        '命中后记录结果，跳过 l 的重复值，再同时收缩 l、r',
      ],
      explanation: md`排序是一切的前提（去重和对撞都依赖它）；剪枝与外层去重都在进入内层之前；内层命中后先去重再收缩。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与它的指针移动规则配对：`,
      pairs: [
        ['LC167 两数之和 II', '比较 s 与 target，太小移 l，太大移 r'],
        ['LC11 盛最多水的容器', '移动较矮的一端'],
        ['LC42 接雨水（双指针）', 'lmax < rmax 时结算并移动 l'],
        ['LC283 移动零', '同向：遇到非零元素就与 slow 交换'],
      ],
      explanation: md`前三个是对撞指针，区别在于“丢掉哪一侧”的依据不同；LC283 是同向读写指针，维护 \`[0, slow)\` 为结果。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 LC167「两数之和 II」（返回 1 起始的下标；有 2 行干扰项）：`,
      lines: [
        'def two_sum(numbers, target):',
        '    l, r = 0, len(numbers) - 1',
        '    while l < r:',
        '        s = numbers[l] + numbers[r]',
        '        if s == target:',
        '            return [l + 1, r + 1]',
        '        elif s < target:',
        '            l += 1',
        '        else:',
        '            r -= 1',
      ],
      distractors: ['    while l <= r:', '            r += 1'],
      explanation: md`\`l < r\` 保证取到的是两个不同元素（\`l <= r\` 会允许同一个元素用两次）；和太大时只能 \`r -= 1\`，\`r += 1\` 会越界且方向错误。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 LC11 实现在 \`[1, 8, 6, 2, 5, 4, 8, 3, 7]\` 上返回 8 而不是 49。点出有问题的行：`,
      code: py`
        def max_area(height):
            l, r = 0, len(height) - 1
            best = 0
            while l < r:
                best = max(best, min(height[l], height[r]) * (r - l))
                if height[l] > height[r]:
                    l += 1
                else:
                    r -= 1
            return best
      `,
      bugLines: [6],
      explanation: md`第 6 行条件写反了，应为 \`height[l] < height[r]\`（移动短板）。这里移动的是**高板**：第一步 \`1 > 7\` 为假，执行 \`r -= 1\`，把高 7 的板丢掉而保留了高 1 的短板；之后短板 1 一直留在左端，所有容器高度都被限制为 1，只能得到 8。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 3,
      prompt: md`\`nums = [1, 2, 4, 6, 9, 11]\`、\`target = 14\`，对撞指针**依次计算过**哪些数对的和？（点出全部）`,
      items: ['1+11', '2+11', '4+11', '6+11', '1+9', '2+9', '4+9', '6+9', '4+6', '9+11'],
      answer: [0, 1, 2, 6, 7],
      explanation: md`
        \`1+11=12\` < 14 → l；\`2+11=13\` → l；\`4+11=15\` > 14 → r；\`4+9=13\` → l；\`6+9=15\` → r；此时 \`l == r\`，结束，无解。
        共计算 5 对，其余 10 对都被“整行/整列排除”掉了。
      `,
    },
    {
      id: 'q-trap-proof',
      type: 'single',
      difficulty: 3,
      prompt: md`接雨水双指针中，\`lmax < rmax\` 时可以立即确定第 \`l\` 列的水量为 \`lmax - height[l]\`，最准确的理由是：`,
      options: [
        '第 l 列真正的右侧最大值 ≥ rmax > lmax，所以 min(左最大, 右最大) = lmax',
        'lmax 是整个数组的最大值',
        '右指针以后只会遇到更矮的柱子',
        '第 l 列右边的柱子都比 lmax 矮',
      ],
      answer: 0,
      explanation: md`rmax 是 \`[r, n)\` 中的最大值，而 \`[r, n)\` ⊂ \`(l, n)\`，所以第 l 列的右侧最大值至少是 rmax。既然 rmax > lmax，水位就由 lmax 决定。B、C、D 都不成立：我们并不知道中间那些柱子的高度，也不需要知道。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-threesum',
      title: '一步步写出三数之和（含去重）',
      intro: md`
        实现 \`three_sum(nums)\`：返回所有和为 0 的**不重复**三元组（每个三元组内部升序，整体顺序不限）。
        骨架已经给出排序与循环结构，你来填写去重与指针移动。
      `,
      template: py`
        def three_sum(nums):
            nums.sort()
            n, res = len(nums), []
            for i in range(n - 2):
                if nums[i] > 0:
                    break
                if {{1}}:                          # 外层去重
                    continue
                l, r = i + 1, n - 1
                while l < r:
                    s = nums[i] + nums[l] + nums[r]
                    if s < 0:
                        {{2}}
                    elif s > 0:
                        {{3}}
                    else:
                        res.append([nums[i], nums[l], nums[r]])
                        while {{4}}:               # 跳过与 nums[l] 相同的值
                            l += 1
                        l += 1
                        r -= 1
            return res
      `,
      steps: [
        {
          blank: 1,
          instruction: md`外层去重：当前的 \`nums[i]\` 与**前一个**相同则跳过。注意 \`i = 0\` 时没有前一个。`,
          answer: 'i > 0 and nums[i] == nums[i - 1]',
          accept: ['i and nums[i] == nums[i - 1]', 'i >= 1 and nums[i] == nums[i - 1]'],
          explain: md`和前一个比较，保证“同值的第一个”一定被处理——它的右边可能还需要用到同值元素（如 \`[-1, -1, 2]\`）。`,
        },
        {
          blank: 2,
          instruction: md`三数之和小于 0：需要更大的数。`,
          answer: 'l += 1',
          explain: md`\`nums[l]\` 配当前最大的 \`nums[r]\` 都不够，它不可能再参与任何解。`,
        },
        {
          blank: 3,
          instruction: md`三数之和大于 0：需要更小的数。`,
          answer: 'r -= 1',
        },
        {
          blank: 4,
          instruction: md`命中后，让 \`l\` 停在一串相同值的**最后一个**上（随后的 \`l += 1\` 会跨过它）。别忘了防止越界。`,
          answer: 'l < r and nums[l] == nums[l + 1]',
          accept: ['l + 1 < r and nums[l] == nums[l + 1]'],
          explain: md`\`l < r\` 保证 \`l + 1\` 不越界。只跳 \`l\` 就够了：\`nums[l]\` 变大后，若 \`nums[r]\` 不变，和一定 > 0，\`r\` 会在后续步骤中自然左移。`,
        },
      ],
      setup: py`
        import random
        from itertools import combinations
        def _ref_3sum(nums):
            return sorted({tuple(sorted(c)) for c in combinations(nums, 3) if sum(c) == 0})
        def _norm(res):
            return sorted(tuple(t) for t in res)
      `,
      tests: [
        {
          name: '示例',
          code: py`
            got = _norm(three_sum([-1, 0, 1, 2, -1, -4]))
            assert got == [(-1, -1, 2), (-1, 0, 1)], f"得到 {got}"
          `,
        },
        {
          name: '重复值',
          code: py`
            assert _norm(three_sum([0, 0, 0, 0])) == [(0, 0, 0)]
            assert _norm(three_sum([-2, 0, 0, 2, 2])) == [(-2, 0, 2)]
            assert _norm(three_sum([-1, -1, 2])) == [(-1, -1, 2)]
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(15)
            for _ in range(200):
                a = [rnd.randint(-6, 6) for _ in range(rnd.randint(0, 12))]
                exp = _ref_3sum(a)
                got = _norm(three_sum(list(a)))
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        这个“排序 + 枚举 + 对撞 + 去重”的骨架可以直接推广：最接近的三数之和（LC16）把“命中”换成“更新最优差值”；
        四数之和（LC18）再套一层枚举，注意两层都要去重。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-3sum',
      title: '三数之和',
      difficulty: 'medium',
      source: 'LeetCode 15',
      timeLimitMin: 20,
      description: md`
        给定整数数组 \`nums\`，返回所有满足 \`nums[i] + nums[j] + nums[k] == 0\`（\`i, j, k\` 互不相同）的三元组，**不能包含重复的三元组**。
        每个三元组内部按升序排列，三元组之间的顺序不限。

        \`\`\`python
        def three_sum(nums: list[int]) -> list[list[int]]: ...
        \`\`\`

        **示例**
        - \`three_sum([-1,0,1,2,-1,-4])\` → \`[[-1,-1,2],[-1,0,1]]\`
        - \`three_sum([0,1,1])\` → \`[]\`
        - \`three_sum([0,0,0])\` → \`[[0,0,0]]\`

        **约束**：$0 \le n \le 3000$，$-10^5 \le \text{nums}[i] \le 10^5$。要求 $O(n^2)$；隐藏测试含 $n = 400$ 的计时用例，$O(n^3)$ 枚举会超时。
      `,
      starter: py`
        def three_sum(nums: list[int]) -> list[list[int]]:
            # TODO: 排序 → 枚举最小数 i → 对撞指针找另外两个数；注意两处去重
            raise NotImplementedError
      `,
      solution: py`
        def three_sum(nums: list[int]) -> list[list[int]]:
            nums = sorted(nums)
            n, res = len(nums), []
            for i in range(n - 2):
                if nums[i] > 0:                         # 剪枝：最小数已 > 0
                    break
                if i > 0 and nums[i] == nums[i - 1]:    # 外层去重
                    continue
                l, r = i + 1, n - 1
                while l < r:
                    s = nums[i] + nums[l] + nums[r]
                    if s < 0:
                        l += 1
                    elif s > 0:
                        r -= 1
                    else:
                        res.append([nums[i], nums[l], nums[r]])
                        while l < r and nums[l] == nums[l + 1]:   # 内层去重
                            l += 1
                        l += 1
                        r -= 1
            return res
      `,
      explanation: md`
        **思路**：排序后枚举三元组中最小的数 \`nums[i]\`，问题化为在 \`[i+1, n)\` 上找两数之和 $= -\text{nums}[i]$，用对撞指针 $O(n)$ 完成。

        **去重**
        - 外层与**前一个**比较：同值的第一个已经找到了以它为最小数的全部三元组；
        - 内层命中后跳过 \`l\` 的重复值，再同时收缩。

        **剪枝**：\`nums[i] > 0\` 时直接结束；还可以加 \`nums[i] + nums[i+1] + nums[i+2] > 0\` 时 break、\`nums[i] + nums[n-2] + nums[n-1] < 0\` 时 continue。

        **复杂度**：时间 $O(n^2)$，除输出外空间 $O(\log n)$～$O(n)$（排序）。

        **常见错误**：外层与后一个比较导致漏解；内层不去重导致重复；用 \`set\` 去重虽然正确但面试官会追问能否不用。
      `,
      complexity: { time: 'O(n²)', space: 'O(n)（排序副本）' },
      hints: [
        md`先排序。三元组中最小的那个数枚举一遍，剩下的问题是什么？`,
        md`剩下的是“有序数组中两数之和 = -nums[i]”，对撞指针 $O(n)$。`,
        md`去重：外层 \`i > 0 and nums[i] == nums[i-1]\` 跳过；内层命中后 \`while l < r and nums[l] == nums[l+1]: l += 1\`。`,
      ],
      setup: py`
        import random, time
        from itertools import combinations

        def _ref_3sum(nums):
            return sorted({tuple(sorted(c)) for c in combinations(nums, 3) if sum(c) == 0})

        def _norm(res):
            return sorted(tuple(t) for t in res)
      `,
      tests: [
        {
          name: '示例 1',
          code: py`
            got = _norm(three_sum([-1, 0, 1, 2, -1, -4]))
            assert got == [(-1, -1, 2), (-1, 0, 1)], f"期望 [(-1,-1,2), (-1,0,1)]，得到 {got}"
          `,
        },
        { name: '示例 2 / 3', code: py`assert _norm(three_sum([0, 1, 1])) == [] and _norm(three_sum([0, 0, 0])) == [(0, 0, 0)]` },
        {
          name: '边界与重复',
          code: py`
            assert three_sum([]) == [] and three_sum([0]) == [] and three_sum([1, -1]) == []
            assert _norm(three_sum([0, 0, 0, 0, 0])) == [(0, 0, 0)], "全 0：只能输出一次"
            got = _norm(three_sum([-2, 0, 0, 2, 2]))
            assert got == [(-2, 0, 2)], f"期望 [(-2,0,2)]，得到 {got}"
            got = _norm(three_sum([-1, -1, 2]))
            assert got == [(-1, -1, 2)], f"外层去重方向错了？期望 [(-1,-1,2)]，得到 {got}"
          `,
        },
        {
          name: '三元组内部升序',
          code: py`
            for t in three_sum([3, -2, 1, 0, -1, 2, -3]):
                assert list(t) == sorted(t), f"三元组 {t} 内部应升序"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(2015)
            for _ in range(250):
                a = [rnd.randint(-8, 8) for _ in range(rnd.randint(0, 14))]
                exp = _ref_3sum(a)
                got = _norm(three_sum(list(a)))
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：n = 400',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            a = [rnd.randint(-300, 300) for _ in range(400)]
            t0 = time.perf_counter()
            got = three_sum(list(a))
            dt = time.perf_counter() - t0
            assert len(got) == len(set(map(tuple, got))), "结果中有重复三元组"
            assert all(sum(t) == 0 for t in got), "结果中有和不为 0 的三元组"
            assert dt < 1.0, f"用时 {dt:.2f}s，疑似 O(n^3)"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-closest',
          q: md`LC16「最接近的三数之和」怎么改？`,
          a: md`骨架不变，内层每算一个 \`s\` 就用 \`abs(s - target)\` 更新最优值；\`s < target\` 移 l、\`s > target\` 移 r、相等直接返回。不需要去重（只要一个数值答案）。`,
        },
        {
          id: 'f-4sum',
          q: md`四数之和、k 数之和的复杂度和写法？`,
          a: md`四数之和两层枚举 + 对撞，$O(n^3)$；k 数之和递归到 2 数时用对撞，$O(n^{k-1})$。每层枚举都要做“与前一个相同则跳过”的去重，并可以用“最小的 k 个之和 > target 则 break”剪枝。`,
        },
      ],
    },
    {
      id: 'c-trap',
      title: '接雨水',
      difficulty: 'hard',
      source: 'LeetCode 42',
      timeLimitMin: 20,
      description: md`
        给定 $n$ 个非负整数表示宽度为 1 的柱子高度 \`height\`，计算下雨后能接多少雨水。

        \`\`\`python
        def trap(height: list[int]) -> int: ...
        \`\`\`

        **示例**
        - \`trap([0,1,0,2,1,0,1,3,2,1,2,1])\` → \`6\`
        - \`trap([4,2,0,3,2,5])\` → \`9\`

        **要求**：时间 $O(n)$、**额外空间 $O(1)$**（双指针）。隐藏测试会统计你读取 \`height\` 的次数，并用 \`tracemalloc\` 检查额外内存——前后缀数组解法能通过正确性测试，但过不了空间测试。

        **约束**：$0 \le n \le 2 \times 10^4$，$0 \le h_i \le 10^5$。
      `,
      starter: py`
        def trap(height: list[int]) -> int:
            # TODO: 双指针 l, r 与 lmax, rmax；每步结算较矮一侧的一列
            raise NotImplementedError
      `,
      solution: py`
        def trap(height: list[int]) -> int:
            l, r = 0, len(height) - 1
            lmax = rmax = ans = 0
            while l < r:
                lmax = max(lmax, height[l])
                rmax = max(rmax, height[r])
                if lmax < rmax:
                    # 第 l 列右侧最大值 >= rmax > lmax，水位由 lmax 决定
                    ans += lmax - height[l]
                    l += 1
                else:
                    # 对称：第 r 列左侧最大值 >= lmax >= rmax，水位由 rmax 决定
                    ans += rmax - height[r]
                    r -= 1
            return ans
      `,
      solutionAlt: [
        {
          label: '前后缀最大值版（O(n) 空间，最容易想到）',
          code: py`
            def trap(height):
                n = len(height)
                if n == 0:
                    return 0
                left, right = [0] * n, [0] * n
                left[0], right[-1] = height[0], height[-1]
                for i in range(1, n):
                    left[i] = max(left[i - 1], height[i])
                for i in range(n - 2, -1, -1):
                    right[i] = max(right[i + 1], height[i])
                return sum(min(left[i], right[i]) - height[i] for i in range(n))
          `,
        },
        {
          label: '单调栈版（按层横向计算）',
          code: py`
            def trap(height):
                st, ans = [], 0                      # 栈中存下标，对应高度单调递减
                for i, h in enumerate(height):
                    while st and height[st[-1]] < h:
                        bottom = st.pop()
                        if not st:
                            break
                        left = st[-1]
                        w = i - left - 1
                        ans += w * (min(height[left], h) - height[bottom])
                    st.append(i)
                return ans
          `,
        },
      ],
      explanation: md`
        **公式**：第 $i$ 列水量 $= \min(\text{leftMax}_i, \text{rightMax}_i) - h_i$（两个最大值都包含第 $i$ 列）。

        **双指针**：维护 \`lmax = max(h[0..l])\`、\`rmax = max(h[r..n-1])\`。若 \`lmax < rmax\`，第 l 列的右侧最大值 ≥ rmax > lmax，
        所以它的水量就是 \`lmax - h[l]\`，结算后 \`l += 1\`；否则对称地结算第 r 列。每步结算一列，$O(n)$ 时间、$O(1)$ 空间。

        **为什么 \`lmax - h[l]\` 不会是负数**：\`lmax\` 在结算前已经和 \`h[l]\` 取过 max。

        **常见错误**：先结算再更新 \`lmax\`（会出现负水量）；比较 \`height[l] < height[r]\` 的写法也正确，但要保证 lmax/rmax 的更新顺序与之配合。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`先写出单列水量公式：$\min(\text{左侧最高}, \text{右侧最高}) - h_i$。`,
        md`前后缀最大值数组可以 $O(n)$ 解决。能不能不存数组？什么时候只知道一侧的最大值就够了？`,
        md`两端各一个指针，维护 \`lmax\`、\`rmax\`。若 \`lmax < rmax\`，第 l 列的水位已经确定为 \`lmax\`。`,
      ],
      setup: py`
        import random, tracemalloc
        from collections.abc import Sequence

        def _ref_trap(h):
            n = len(h)
            if n == 0:
                return 0
            L, R = h[:], h[:]
            for i in range(1, n):
                L[i] = max(L[i - 1], h[i])
            for i in range(n - 2, -1, -1):
                R[i] = max(R[i + 1], h[i])
            return sum(min(L[i], R[i]) - h[i] for i in range(n))

        class _Budget(Sequence):
            """只读序列：读取次数超过预算立即失败"""
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
                    raise AssertionError(f"读取 height 超过 {self.budget} 次，复杂度不是 O(n)")
                return r
      `,
      tests: [
        { name: '示例 1', code: py`assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6` },
        { name: '示例 2', code: py`assert trap([4, 2, 0, 3, 2, 5]) == 9` },
        {
          name: '边界',
          code: py`
            assert trap([]) == 0 and trap([5]) == 0 and trap([5, 3]) == 0
            assert trap([1, 2, 3, 4]) == 0, "单调递增不存水"
            assert trap([4, 3, 2, 1]) == 0, "单调递减不存水"
            assert trap([3, 0, 3]) == 3
            assert trap([2, 2, 2]) == 0
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(42)
            for _ in range(300):
                h = [rnd.randint(0, 9) for _ in range(rnd.randint(0, 20))]
                exp, got = _ref_trap(h), trap(list(h))
                assert got == exp, f"height={h}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '时间：n = 20000，读取次数 ≤ 4n',
          hidden: true,
          code: py`
            rnd = random.Random(3)
            data = [rnd.randint(0, 10**5) for _ in range(20000)]
            exp = _ref_trap(data)
            got = trap(_Budget(data, 4 * len(data)))
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '空间：额外内存 O(1)',
          hidden: true,
          code: py`
            rnd = random.Random(5)
            data = [rnd.randint(0, 10**5) for _ in range(4000)]
            tracemalloc.start()
            trap(data)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert peak < 24 * 1024, f"峰值额外内存 {peak / 1024:.0f} KiB，应为 O(1)（不要开前后缀数组）"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-2d',
          q: md`二维接雨水（LC407）怎么做？双指针还适用吗？`,
          a: md`不适用。二维时“短板”是整个外围边界的最低点，用**最小堆**从外围向内做类似 Dijkstra 的扩展：每次弹出最低的边界格子，访问其未访问的邻居，邻居水量为 \`max(0, 当前水位 - h)\`，并以 \`max(当前水位, h)\` 入堆。复杂度 $O(mn \log(mn))$。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-correct',
      freq: 3,
      q: md`对撞指针为什么不会漏解？请严格说明。`,
      a: md`
        **结论**：每次移动指针时，被跳过的所有候选都已被证明不可能是解。

        **原因**：把数对 $(i, j)$ 排成上三角矩阵，行内递增、列内递增。当 $\text{nums}[l] + \text{nums}[r] < \text{target}$ 时，第 $l$ 行剩余格子 $(l, j),\ j \le r$ 的和都 $\le$ 当前和，整行排除；$>$ 时整列排除。每步排除一行或一列，$n-1$ 步覆盖全部格子。

        **延伸**：同一论证适用于 LC11（以短板为边的容器整体排除）、LC240 搜索二维矩阵（从右上角出发每步排除一行或一列）。
      `,
    },
    {
      id: 'qa-3sum',
      freq: 3,
      q: md`三数之和如何去重？能否不用 set？`,
      a: md`
        **结论**：排序后两处去重即可，不需要 set。

        **原因**：外层 \`i > 0 and nums[i] == nums[i-1]\` 时跳过，因为以该值为最小数的三元组上一轮已全部枚举；内层命中后把 \`l\` 跳过所有相同值，再同时收缩 \`l\`、\`r\`。注意外层必须和**前一个**比较，和后一个比较会漏掉 \`[-1,-1,2]\` 这类解。

        **延伸**：用 set 去重正确但浪费空间，且仍会产生重复计算。k 数之和递归时每层都做同样的去重，复杂度 $O(n^{k-1})$。
      `,
    },
    {
      id: 'qa-trap',
      freq: 3,
      q: md`接雨水有哪些解法？各自的复杂度？`,
      a: md`
        **结论**：前后缀最大值 $O(n)/O(n)$、单调栈 $O(n)/O(n)$、双指针 $O(n)/O(1)$。

        **原因**：核心公式是第 $i$ 列水量 $= \min(\text{leftMax}_i, \text{rightMax}_i) - h_i$。前后缀数组直接预处理两个最大值；单调栈按“凹槽”横向一层层结算；双指针利用“较矮一侧的水位已经确定”省掉数组。

        **延伸**：二维版本（LC407）用最小堆从边界向内扩展；“盛最多水的容器”（LC11）虽然名字相似，但求的是两根柱子围成的最大矩形，用移动短板的双指针。
      `,
    },
    {
      id: 'qa-container',
      freq: 2,
      q: md`盛最多水的容器为什么要移动较矮的那根？`,
      a: md`
        **结论**：以短板为一条边的所有其它容器都不可能更优，可以整体丢弃。

        **原因**：设 $h_l < h_r$。任何以 $l$ 为左边、$j < r$ 为右边的容器，宽度 $j - l < r - l$，高度 $\min(h_l, h_j) \le h_l$，面积必然小于当前面积。所以 $l$ 已经“用完”，移动它不会漏掉最优解；反之移动高板可能丢掉最优解。

        **延伸**：两板等高时移动任意一侧都可以（甚至可以同时移动）：以其中任何一根为边、宽度更小的容器都不会超过当前面积。
      `,
    },
  ],

  cheatsheet: md`
    **对撞指针**（有序 / 可剪枝）
    \`\`\`python
    l, r = 0, n - 1
    while l < r:
        s = nums[l] + nums[r]
        if s == t: return [l, r]
        if s < t: l += 1        # 整行排除
        else:     r -= 1        # 整列排除
    \`\`\`
    **同向读写指针**：\`[0, slow)\` 是结果；保留元素时 \`nums[slow] = x; slow += 1\`

    | 题 | 要点 |
    |---|---|
    | 15 三数之和 | 排序 + 固定 i + 对撞；外层与**前一个**比较去重，命中后跳 l |
    | 11 盛水容器 | 面积 $\min(h_l,h_r)(r-l)$，移动**短板** |
    | 42 接雨水 | $\min(L_i, R_i) - h_i$；\`lmax < rmax\` 结算 l |
    | 283 移动零 | 非零时与 slow 交换 |

    - k 数之和 $O(n^{k-1})$；无序两数之和用哈希
    - 接雨水：前后缀 $O(n)$ 空间 → 双指针 $O(1)$
  `,
})
