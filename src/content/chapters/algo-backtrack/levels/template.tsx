/**
 * 回溯模板（algo-backtrack-template）
 * LC78 / 90 / 46 / 47 / 39 / 40：选 / 不选与枚举 start 两种子集写法、used[] 排列、排序 + 同层去重、组合总和的复用与剪枝。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { COUNTED_NUM, RESULT_HELPERS } from '../shared'
import { BacktrackTreeViz } from '../viz/BacktrackTreeViz'
import { DedupViz } from '../viz/DedupViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'tree',
      title: '回溯就是在决策树上做 DFS',
      body: md`
        回溯问题（子集、组合、排列、切割、棋盘）都可以画成一棵**决策树**：每个节点是“已经做出的选择”（\`path\`），
        每条边是一次选择，叶子（或每个节点）对应一个候选答案。回溯 = 对这棵树做 DFS，写代码前先回答三个问题：

        1. **路径** \`path\`：已经做了哪些选择；
        2. **选择列表**：当前节点还能选什么（由 \`start\` 下标或 \`used[]\` 数组决定）；
        3. **结束条件**：什么时候收集答案（到叶子 / 每个节点都收集）。

        \`\`\`python
        def dfs(状态):
            if 满足结束条件: res.append(path[:]); return
            for 选择 in 选择列表:
                做选择（path.append）
                dfs(新状态)
                撤销选择（path.pop）
        \`\`\`

        子集有两种建树方式：**选 / 不选**（每层决定一个元素，二叉树，答案在叶子）和**枚举下一个选谁**
        （\`for i in range(start, n)\`，每个节点都是答案）。排列没有“只往后选”的限制，用 \`used[]\` 标记已在路径上的元素。
      `,
      viz: BacktrackTreeViz,
      after: md`
        > [!WARN] 两个最常见的 bug
        > 1. \`res.append(path)\` 而不是 \`res.append(path[:])\`：存进去的是同一个列表对象，回溯结束时它被 pop 成空，结果全是 \`[]\`；
        > 2. 子集 / 组合递归时传 \`dfs(start + 1)\` 而不是 \`dfs(i + 1)\`：会出现 [2,2] 这类重复使用元素、或漏掉 [1,3] 之类的组合。
      `,
      checkpoint: {
        id: 'cp-tree',
        type: 'single',
        prompt: md`下面的代码对 \`nums = [1, 2]\` 返回什么？
\`\`\`python
res, path = [], []
def dfs(start):
    res.append(path)
    for i in range(start, len(nums)):
        path.append(nums[i]); dfs(i + 1); path.pop()
dfs(0)
\`\`\``,
        options: ['`[[], [1], [1, 2], [2]]`', '`[[], [], [], []]`', '`[[1, 2]]`', '`[[], [1], [2]]`'],
        answer: 1,
        explanation: md`\`res\` 里存了 4 次**同一个** \`path\` 对象；DFS 结束时 path 已被全部 pop，所以打印出 4 个空列表。必须 \`res.append(path[:])\`（或 \`list(path)\`）拷贝当前快照。`,
      },
    },
    {
      id: 'dedup',
      title: '有重复元素：排序 + 同层去重',
      body: md`
        输入含重复值时（LC90 子集 II、LC47 全排列 II、LC40 组合总和 II），重复答案从哪来？
        在**同一层**（同一个 for 循环）里，选了两个**值相同的不同元素**，它们展开的子树完全一样。

        解法：先**排序**让相同值相邻，然后在同一层只让第一个展开：

        | 题型 | 去重条件（满足就 \`continue\`） |
        |---|---|
        | 子集 / 组合（start 写法） | \`i > start and nums[i] == nums[i-1]\` |
        | 排列（used 写法） | \`i > 0 and nums[i] == nums[i-1] and not used[i-1]\` |

        关键是区分**同层**与**同枝**：\`i > start\` 保证比较的是同一层里的前一个兄弟；
        而同一条路径上先后选两个相同的值（如 [1,2,2]）是合法的，不能剪。排列里 \`not used[i-1]\` 表示前一个相同值
        刚刚在**本层**被撤销，现在再选就和刚才那棵子树重复了。
      `,
      viz: DedupViz,
      after: md`
        > [!INTERVIEW] 面试官想听到的
        > “先排序，把相同值放在一起；同一层里相同值只展开第一个。这样不需要用 set 对结果去重——set 去重要先把重复的子树全部搜完，复杂度没降下来。”
      `,
      checkpoint: {
        id: 'cp-dedup',
        type: 'judge',
        prompt: md`子集 II 中，把去重条件写成 \`i > 0 and nums[i] == nums[i-1]\`（而不是 \`i > start\`），结果仍然正确。`,
        answer: false,
        explanation: md`\`i > 0\` 会把**同一条路径上**的合法选择也剪掉：在节点 [1,2] 处（start = 2），i = 2 时 \`nums[2] == nums[1]\`，[1,2,2] 被错误跳过。只有 \`i > start\` 才保证比较的是同层兄弟。`,
      },
    },
    {
      id: 'combsum',
      title: '组合总和：可重复选、只用一次与排序剪枝',
      body: md`
        组合总和类题目在模板上只差**递归时传什么下标**：

        | 题目 | 元素能否重复使用 | 递归参数 | 去重 |
        |---|---|---|---|
        | LC39 组合总和（无重复元素） | 可以无限次 | \`dfs(i, ...)\` | 不需要 |
        | LC40 组合总和 II（有重复元素） | 每个元素一次 | \`dfs(i + 1, ...)\` | 排序 + 同层去重 |
        | LC216 组合总和 III | 1~9 各一次，恰好 k 个 | \`dfs(i + 1, ...)\` | 不需要 |
        | LC77 组合 | 每个一次，恰好 k 个 | \`dfs(i + 1)\` | 不需要 |

        传 \`i\` 表示“下一层还可以选我自己”，传 \`i + 1\` 表示“只能选后面的”。两者都不会回头选前面的元素，
        所以 [2,3] 和 [3,2] 不会同时出现——**组合靠 start 保证无序，排列靠 used 允许任意顺序**。

        **排序剪枝**：候选数排好序后，一旦 \`candidates[i] > remain\`，后面的数更大，可以直接 \`break\`（而不是 \`continue\`）。
        这一行常常能把搜索量减少一个数量级。LC77 也有经典剪枝：剩余元素不够凑满 k 个时（\`n - i + 1 < k - len(path)\`）提前结束循环。
      `,
      checkpoint: {
        id: 'cp-combsum',
        type: 'fill',
        prompt: md`LC39（每个数可以无限次使用）中，递归调用应写成 \`dfs(____(1)____, remain - candidates[i])\`；LC40（每个数最多用一次）应写成 \`dfs(____(2)____, remain - candidates[i])\`。`,
        blanks: [
          { label: '(1)', answers: ['i'] },
          { label: '(2)', answers: ['i + 1', 'i+1'] },
        ],
        explanation: md`传 \`i\` 允许下一层继续选当前元素（重复使用）；传 \`i + 1\` 则只能选后面的元素。写成 \`start + 1\` 是常见错误：会跳过一些组合，或让当前元素以外的元素被重复使用。`,
      },
    },
    {
      id: 'complexity',
      title: '复杂度：决策树有多大',
      body: md`
        回溯的时间复杂度 ≈ **节点数 × 每个节点的工作量**（拷贝答案是 $O(n)$）：

        | 问题 | 叶子 / 答案数 | 时间复杂度 |
        |---|---|---|
        | 子集（LC78） | $2^n$ | $O(n \cdot 2^n)$ |
        | 组合（LC77，选 k 个） | $\binom{n}{k}$ | $O(k \cdot \binom{n}{k})$ |
        | 排列（LC46） | $n!$ | $O(n \cdot n!)$ |

        这些是**输出规模**决定的下界：答案本身就有这么多，任何算法都必须至少把它们写出来。
        回溯能优化的是“不产生答案的节点”——剪枝（下一关）与去重都是在砍这部分。

        **空间**：递归深度 $O(n)$ + path $O(n)$（不计输出）。$n \le 20$ 左右的子集枚举、$n \le 10$ 的排列是回溯的舒适区。
        如果题目只要**数量**而不要具体方案（如“和为 target 的组合有几种”），往往可以换成 DP 计数，复杂度从指数降到多项式。
      `,
      checkpoint: {
        id: 'cp-complexity',
        type: 'numeric',
        prompt: md`用“枚举 start”写法求 \`nums = [1, 2, 3, 4]\` 的全部子集，\`dfs\` 一共被调用多少次（含最初的 \`dfs(0)\`）？`,
        answer: 16,
        tolerance: 0,
        explanation: md`这种写法的递归树里**每个节点恰好对应一个子集**，节点数 = 子集数 = $2^4 = 16$。“选 / 不选”写法则是满二叉树，调用 $2^{5} - 1 = 31$ 次，答案只在 16 个叶子上。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                          */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-three',
      type: 'match',
      difficulty: 1,
      prompt: md`把回溯模板的三要素与它在“全排列”中的具体形式配对：`,
      pairs: [
        ['路径', '`path`：已经排好的前几个数'],
        ['选择列表', '`used[i] == False` 的所有 i'],
        ['结束条件', '`len(path) == len(nums)`'],
        ['撤销选择', '`used[i] = False; path.pop()`'],
      ],
      explanation: md`排列的每一层都可以从全部元素中选，只是要跳过已经在路径上的（\`used\`）；路径长度达到 n 时收集答案。`,
    },
    {
      id: 'q-copy',
      type: 'judge',
      difficulty: 1,
      prompt: md`收集答案时写 \`res.append(path)\` 与 \`res.append(path[:])\` 效果相同，只是后者多一次拷贝。`,
      answer: false,
      explanation: md`前者存的是同一个列表对象的引用，之后 path 被修改（最终被 pop 空），res 里的所有元素都跟着变。必须存快照。`,
    },
    {
      id: 'q-count-perm',
      type: 'numeric',
      difficulty: 2,
      prompt: md`\`nums = [1, 1, 2, 2]\` 的**不重复**全排列共有多少个？`,
      answer: 6,
      tolerance: 0,
      explanation: md`多重集排列数 $\frac{4!}{2!\,2!} = 6$：1122、1212、1221、2112、2121、2211。不去重的回溯会生成 $4! = 24$ 个。`,
    },
    {
      id: 'q-start-vs-i',
      type: 'single',
      difficulty: 2,
      prompt: md`“枚举 start”的子集代码里，把递归调用 \`dfs(i + 1)\` 误写成 \`dfs(start + 1)\`，对 \`nums = [1, 2, 3]\` 会怎样？`,
      options: ['结果完全正确', '出现重复子集，例如 [2] 出现多次、以及 [2,2] 这类重复使用元素的结果', '只返回空集', '无限递归'],
      answer: 1,
      explanation: md`在根节点（start = 0）选了 3 之后递归的是 \`dfs(1)\`，又可以选 2、3，于是出现 [3,2]、[3,3] 等错误结果，同一子集也会重复出现。不会无限递归：start 每层至少 +1。`,
    },
    {
      id: 'q-dedup-rule',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于“排序 + 同层去重”，哪些说法正确？（多选）`,
      options: [
        '必须先排序，让相同的值相邻',
        '子集 II 的条件是 `i > start and nums[i] == nums[i-1]`',
        '去重后，结果中不可能出现两个相同的值（如 [2,2]）',
        '在每一层用一个局部 `seen = set()` 记录已展开的值，也能做到同层去重，而且不要求排序',
      ],
      answer: [0, 1, 3],
      explanation: md`A、B 是标准做法。C 错：同一路径（不同层）选两个 2 是合法的，[1,2,2] 就是答案之一。D 正确：层内 set 也能保证同层同值只展开一次，不需要排序（但组合题仍常需要排序来做 break 剪枝），额外开销是每层一个 set。`,
    },
    {
      id: 'q-used-variant',
      type: 'judge',
      difficulty: 3,
      prompt: md`全排列 II 中，把去重条件里的 \`not used[i-1]\` 改成 \`used[i-1]\`，仍然能得到正确且不重复的结果，只是效率更低。`,
      answer: true,
      explanation: md`
        改成 \`used[i-1]\` 等价于强制相同值的元素按**下标从大到小**的顺序出现在路径里（前一个相同值已在路径上时，不许再选后一个）。
        每个不同排列仍恰好对应一种合法顺序，所以结果正确不重复；但它要等到更深的层才发现死路（树枝去重），访问的节点远多于 \`not used[i-1]\`（树层去重）。
        面试时写 \`not used[i-1]\`，并能解释两者区别，是加分项。
      `,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`LC40 组合总和 II 的循环体（\`for i in range(start, len(c))\`）中，以下语句的正确顺序是：`,
      items: [
        '`if i > start and c[i] == c[i-1]: continue`',
        '`if c[i] > remain: break`',
        '`path.append(c[i])`',
        '`dfs(i + 1, remain - c[i])`',
        '`path.pop()`',
      ],
      explanation: md`先同层去重、再判断超额（排序后可直接 break），然后做选择 → 递归 → 撤销。前两步交换也不影响正确性（相同的值必然同时超额或同时不超额），但“做选择”必须在两次检查之后。`,
    },
    {
      id: 'q-parsons-perm',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出全排列（LC46）的回溯函数（有 2 行干扰项）：`,
      lines: [
        'def dfs():',
        '    if len(path) == len(nums):',
        '        res.append(path[:])',
        '        return',
        '    for i in range(len(nums)):',
        '        if used[i]:',
        '            continue',
        '        used[i] = True',
        '        path.append(nums[i])',
        '        dfs()',
        '        path.pop()',
        '        used[i] = False',
      ],
      distractors: ['    for i in range(start, len(nums)):', '        res.append(path)'],
      explanation: md`排列每层都从 0 开始枚举，用 used 跳过已选元素；\`range(start, ...)\` 是组合 / 子集的写法，会漏掉 [2,1] 这类逆序排列。\`res.append(path)\` 没有拷贝。撤销时 \`path.pop()\` 与 \`used[i] = False\` 的顺序无所谓。`,
    },
    {
      id: 'q-bug-comb',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 LC39（组合总和，元素可重复使用）会输出 [2,3,2]、[3,2,2] 这样的重复组合。点出错误的行：`,
      code: py`
        def combination_sum(candidates, target):
            res, path = [], []
            candidates.sort()
            def dfs(start, remain):
                if remain == 0:
                    res.append(path[:])
                    return
                for i in range(0, len(candidates)):
                    if candidates[i] > remain:
                        break
                    path.append(candidates[i])
                    dfs(i, remain - candidates[i])
                    path.pop()
            dfs(0, target)
            return res
      `,
      bugLines: [8],
      explanation: md`第 8 行应为 \`range(start, len(candidates))\`。从 0 开始枚举意味着可以回头选更小的数，于是同一个组合的不同排列都被当成答案。递归传 \`i\`（允许重复使用自己）是对的。`,
    },
    {
      id: 'q-pick-prune',
      type: 'pick',
      difficulty: 3,
      prompt: md`
        对 \`nums = [1, 2, 2, 3]\` 用“枚举 start + 同层去重”求子集 II。在**根节点**这一层（start = 0）依次考察 i = 0..3，以及在节点 **[2]**（由 i = 1 进入，start = 2）这一层考察 i = 2..3。
        下面哪些分支会被 \`continue\` 剪掉？
      `,
      items: ['根层 i=0（1）', '根层 i=1（2）', '根层 i=2（第二个 2）', '根层 i=3（3）', '[2] 层 i=2（第二个 2）', '[2] 层 i=3（3）'],
      answer: [2],
      explanation: md`根层 i = 2 时 \`i > start = 0\` 且 \`nums[2] == nums[1]\`，剪掉。节点 [2] 这层 start = 2，i = 2 时 \`i > start\` 不成立，不剪——[2,2] 是合法子集。其余分支值都不同。`,
    },
    {
      id: 'q-complexity',
      type: 'single',
      difficulty: 2,
      prompt: md`全排列（LC46，元素互不相同）的时间复杂度最接近？`,
      options: ['$O(n^2)$', '$O(2^n)$', '$O(n \\cdot n!)$', '$O(n^n)$'],
      answer: 2,
      explanation: md`有 $n!$ 个答案，每个拷贝 $O(n)$；递归树的内部节点总数为 $\sum_{k} n!/(n-k)! \le e \cdot n!$，同阶。$O(n^n)$ 是不用 used 剪枝、每层都枚举 n 个再判重的暴力上界。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-subsets-dup',
      title: '子集 II：排序 + 同层去重（LC90）',
      intro: md`
        给定可能含重复元素的整数数组 \`nums\`，返回所有**不重复**的子集（顺序任意）。用“枚举 start”模板：每个节点都是一个子集。
      `,
      template: py`
        def subsets_with_dup(nums):
            nums = {{1}}
            res, path = [], []
            def dfs(start):
                {{2}}
                for i in range(start, len(nums)):
                    if {{3}}:
                        continue
                    path.append(nums[i])
                    {{4}}
                    path.pop()
            dfs(0)
            return res
      `,
      steps: [
        {
          blank: 1,
          instruction: md`先排序，让相同的值相邻（不要原地修改调用方的列表）。`,
          answer: 'sorted(nums)',
          explain: md`同层去重依赖“相同值相邻”。用 \`sorted\` 返回新列表，避免副作用。`,
        },
        {
          blank: 2,
          instruction: md`每个节点都是一个子集：收集当前路径的**快照**。`,
          answer: 'res.append(path[:])',
          accept: ['res.append(list(path))', 'res.append(path.copy())'],
          explain: md`不拷贝的话 res 里全是同一个列表对象，最后都变成空列表。`,
        },
        {
          blank: 3,
          instruction: md`同层去重：同一层里，值与前一个兄弟相同就跳过。`,
          answer: 'i > start and nums[i] == nums[i - 1]',
          accept: ['i > start and nums[i] == nums[i-1]', 'i != start and nums[i] == nums[i - 1]'],
          explain: md`\`i > start\` 是“同层”的判定：i == start 时它是这一层第一个候选，即使与前一个元素相同（那个元素在路径上），也必须展开。`,
        },
        {
          blank: 4,
          instruction: md`递归：下一层只能选 i 之后的元素。`,
          answer: 'dfs(i + 1)',
          explain: md`传 \`i + 1\` 保证组合无序、元素不重复使用。`,
        },
      ],
      setup: py`
        import random
        from itertools import combinations

        def _ref_sub2(nums):
            s = set()
            for k in range(len(nums) + 1):
                for c in combinations(sorted(nums), k):
                    s.add(c)
            return sorted(s)

        def _check_sub2(nums):
            got = subsets_with_dup(list(nums))
            g = sorted(tuple(sorted(x)) for x in got)
            assert len(g) == len(set(g)), f"nums={nums}：结果有重复子集"
            exp = _ref_sub2(nums)
            assert g == exp, f"nums={nums}：期望 {len(exp)} 个子集 {exp}，得到 {len(g)} 个 {g}"
      `,
      tests: [
        { name: '示例 [1,2,2]', code: py`_check_sub2([1, 2, 2])` },
        { name: '单元素 / 全相同', code: py`_check_sub2([0]); _check_sub2([3, 3, 3])` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(90)
            for _ in range(150):
                _check_sub2([rnd.randint(-2, 3) for _ in range(rnd.randint(0, 8))])
          `,
        },
      ],
      outro: md`
        同一个骨架换两处就是 LC40：递归时带上 \`remain\`，在 \`remain == 0\` 时收集；循环里加一行排序剪枝 \`if nums[i] > remain: break\`。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-perm2',
      title: '全排列 II',
      difficulty: 'medium',
      source: 'LeetCode 47',
      timeLimitMin: 20,
      description: md`
        给定一个**可能包含重复数字**的序列 \`nums\`，按任意顺序返回所有**不重复**的全排列。

        \`\`\`python
        def permute_unique(nums: list) -> list[list]: ...
        \`\`\`

        **要求**：在搜索过程中去重（排序 + 同层去重，或每层一个局部 set），**不要**先生成全部 $n!$ 个排列再用 set 去重。
        隐藏测试的元素是自定义数值对象，会统计比较与运算次数：\`[1,1,1,1,1,1,1,2]\` 只有 8 个不同排列，先生成 $8! = 40320$ 个再去重的写法会超出预算。

        **示例**
        - \`[1, 1, 2]\` → \`[[1,1,2], [1,2,1], [2,1,1]]\`
        - \`[1, 2, 3]\` → 6 个排列

        **约束**：$1 \le n \le 8$。
      `,
      starter: py`
        def permute_unique(nums):
            # TODO: 排序 + used[]；同层去重：i > 0 and nums[i] == nums[i-1] and not used[i-1]
            raise NotImplementedError
      `,
      solution: py`
        def permute_unique(nums):
            nums = sorted(nums)
            n = len(nums)
            res, path, used = [], [], [False] * n

            def dfs():
                if len(path) == n:
                    res.append(path[:])
                    return
                for i in range(n):
                    if used[i]:
                        continue
                    # 同层去重：前一个相同值刚在本层被撤销（未使用），再选就会重复
                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                        continue
                    used[i] = True
                    path.append(nums[i])
                    dfs()
                    path.pop()
                    used[i] = False

            dfs()
            return res
      `,
      solutionAlt: [
        {
          label: '每层一个局部 set（不依赖排序）',
          code: py`
            def permute_unique(nums):
                n, res, path, used = len(nums), [], [], [False] * len(nums)

                def dfs():
                    if len(path) == n:
                        res.append(path[:])
                        return
                    seen = set()                      # 本层已经展开过的值
                    for i in range(n):
                        if used[i] or nums[i] in seen:
                            continue
                        seen.add(nums[i])
                        used[i] = True
                        path.append(nums[i])
                        dfs()
                        path.pop()
                        used[i] = False

                dfs()
                return res
          `,
        },
      ],
      explanation: md`
        **思路**：排序后用 \`used[]\` 枚举排列；同一层里，若 \`nums[i] == nums[i-1]\` 且 \`nums[i-1]\` 未被使用，说明它刚在本层展开过同值子树，跳过。

        **为什么是 \`not used[i-1]\`**：前一个相同值若在路径上（used），说明我们处在它下面的某一层，选 \`nums[i]\` 是“同枝”的合法选择；
        若不在路径上，它只可能是在本层被撤销了，再选 \`nums[i]\` 会产生完全相同的子树。

        **复杂度**：答案数为多重集排列数 $\frac{n!}{\prod c_v!}$，时间 $O(n \cdot \text{答案数} \cdot n)$ 量级（每个节点循环 n 次）；空间 $O(n)$（不计输出）。

        **常见错误**：忘记排序；用 \`set(map(tuple, ...))\` 事后去重（复杂度是 $O(n \cdot n!)$）；去重条件漏掉 \`not used[i-1]\`。
      `,
      complexity: { time: 'O(n · n!) 最坏', space: 'O(n)' },
      hints: [
        md`先写出不去重的全排列（used 数组）。重复从哪里来？——同一层选了两个值相同的元素。`,
        md`排序让相同值相邻；在同一层，相同值只展开第一个未使用的。`,
        md`条件：\`i > 0 and nums[i] == nums[i-1] and not used[i-1]\` 时 \`continue\`。`,
      ],
      setup: COUNTED_NUM + '\n' + RESULT_HELPERS + '\n' + py`
        import random
        from itertools import permutations

        def _ref_perm2(nums):
            return [list(p) for p in sorted(set(permutations(nums)))]

        def _check_perm2(nums):
            got = permute_unique(list(nums))
            _assert_same(got, _ref_perm2(nums), False, f"nums={nums}")
      `,
      tests: [
        { name: '示例 [1,1,2]', code: py`_check_perm2([1, 1, 2])` },
        { name: '示例 [1,2,3]', code: py`_check_perm2([1, 2, 3])` },
        { name: '单元素 / 全相同', code: py`_check_perm2([5]); _check_perm2([2, 2, 2])` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(47)
            for _ in range(80):
                _check_perm2([rnd.randint(-1, 2) for _ in range(rnd.randint(1, 6))])
          `,
        },
        {
          name: '搜索中去重：[1×7, 2]',
          hidden: true,
          code: py`
            vals = [1] * 7 + [2]
            nums = [_Num(v) for v in vals]
            got, ops = _with_budget(6000, "比较 / 运算次数超出预算：是不是先生成全部 n! 个排列再去重？", permute_unique, nums)
            _assert_same(got, _ref_perm2(vals), False, "nums=[1]*7+[2]")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-next-perm',
          q: md`如果只要求“下一个排列”（LC31），怎么 $O(n)$ 做？能用它枚举全部不重复排列吗？`,
          a: md`从右往左找第一个 \`a[i] < a[i+1]\`，再从右找第一个大于 \`a[i]\` 的 \`a[j]\`，交换后把 \`a[i+1:]\` 反转。从升序开始反复调用，可以按字典序枚举全部**不重复**排列（天然处理重复元素），且只需 $O(1)$ 额外空间。`,
        },
      ],
    },
    {
      id: 'c-combsum2',
      title: '组合总和 II',
      difficulty: 'medium',
      source: 'LeetCode 40',
      timeLimitMin: 20,
      description: md`
        给定候选数集合 \`candidates\`（正整数，**可能有重复**）和目标数 \`target\`，找出所有和为 \`target\` 的组合。
        每个数字在每个组合中**只能使用一次**，解集不能包含重复的组合（组合内顺序、组合之间顺序都任意）。

        \`\`\`python
        def combination_sum2(candidates: list, target: int) -> list[list]: ...
        \`\`\`

        **要求**：搜索中去重 + 排序剪枝。隐藏测试中 \`candidates = [1]*25, target = 12\` 只有 1 个答案，
        不做同层去重的写法会枚举 $\binom{25}{12} \approx 5.2 \times 10^6$ 种选法，超出运算预算。

        **示例**
        - \`[10,1,2,7,6,1,5]\`，\`target = 8\` → \`[[1,1,6], [1,2,5], [1,7], [2,6]]\`
        - \`[2,5,2,1,2]\`，\`target = 5\` → \`[[1,2,2], [5]]\`

        **约束**：$1 \le n \le 100$，$1 \le$ candidates[i] $\le 50$，$1 \le$ target $\le 30$。
      `,
      starter: py`
        def combination_sum2(candidates, target):
            # TODO: 排序；dfs(start, remain)；同层去重 + 超额 break；递归传 i + 1
            raise NotImplementedError
      `,
      solution: py`
        def combination_sum2(candidates, target):
            c = sorted(candidates)
            res, path = [], []

            def dfs(start, remain):
                if remain == 0:
                    res.append(path[:])
                    return
                for i in range(start, len(c)):
                    if i > start and c[i] == c[i - 1]:   # 同层去重
                        continue
                    if c[i] > remain:                    # 已排序：后面只会更大
                        break
                    path.append(c[i])
                    dfs(i + 1, remain - c[i])            # 每个数只用一次
                    path.pop()

            dfs(0, target)
            return res
      `,
      explanation: md`
        **思路**：排序后套“枚举 start”模板。三处关键：
        1. \`i > start and c[i] == c[i-1]\` → 同层去重，避免重复组合；
        2. \`c[i] > remain\` → \`break\`（排序后的剪枝）；
        3. 递归传 \`i + 1\` → 每个数只用一次。

        **复杂度**：最坏指数级（答案本身可能指数多），剪枝和去重只砍掉无效分支；空间 $O(\text{target})$ 的递归深度（每层至少减 1）。

        **常见错误**：用 \`set\` 对结果去重（先搜重复子树，慢）；去重条件写成 \`i > 0\`（误剪 [1,1,6]）；\`break\` 写成 \`continue\`（正确但慢）；
        递归传 \`i\`（变成可重复使用的 LC39）。
      `,
      complexity: { time: '最坏 O(2^n · n)', space: 'O(target)' },
      hints: [
        md`和子集 II 几乎一样：排序 + \`dfs(start, remain)\`，\`remain == 0\` 时收集。`,
        md`同层去重 \`if i > start and c[i] == c[i-1]: continue\`；排序后 \`if c[i] > remain: break\`。`,
        md`每个数只能用一次，所以递归传 \`i + 1\`。`,
      ],
      setup: COUNTED_NUM + '\n' + RESULT_HELPERS + '\n' + py`
        import random
        from itertools import combinations

        def _ref_cs2(cands, target):
            s = set()
            n = len(cands)
            for k in range(1, n + 1):
                for comb in combinations(sorted(cands), k):
                    if sum(comb) == target:
                        s.add(comb)
            return [list(x) for x in sorted(s)]

        def _check_cs2(cands, target):
            got = combination_sum2(list(cands), target)
            _assert_same(got, _ref_cs2(cands, target), True, f"candidates={cands}, target={target}")
      `,
      tests: [
        { name: '示例 1', code: py`_check_cs2([10, 1, 2, 7, 6, 1, 5], 8)` },
        { name: '示例 2', code: py`_check_cs2([2, 5, 2, 1, 2], 5)` },
        { name: '无解 / 单元素', code: py`_check_cs2([3, 5], 1); _check_cs2([7], 7)` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(40)
            for _ in range(80):
                cands = [rnd.randint(1, 8) for _ in range(rnd.randint(1, 11))]
                _check_cs2(cands, rnd.randint(1, 20))
          `,
        },
        {
          name: '搜索中去重：[1]*25，target = 12',
          hidden: true,
          code: py`
            nums = [_Num(1) for _ in range(25)]
            got, ops = _with_budget(8000, "比较 / 运算次数超出预算：没有在搜索中做同层去重（或没有排序剪枝）", combination_sum2, nums, 12)
            _assert_same(got, [[1] * 12], True, "candidates=[1]*25, target=12")
          `,
        },
        {
          name: '排序剪枝：大数在前',
          hidden: true,
          code: py`
            vals = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41] * 3 + [1, 2, 3, 4, 5, 6]
            nums = [_Num(v) for v in vals]
            got, ops = _with_budget(20000, "运算次数超出预算：排序后遇到 c[i] > remain 应 break", combination_sum2, nums, 10)
            _assert_same(got, _ref_cs2([1, 2, 3, 4, 5, 6], 10), True, "大数在前")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-count',
          q: md`如果只问“有多少种组合”，还需要回溯吗？`,
          a: md`不需要。按值分组后做 0/1（或多重）背包计数：\`dp[s]\` 表示和为 s 的组合数，每个不同值 v（出现 c 次）可选 0..c 个，复杂度 $O(\text{distinct} \cdot \text{target} \cdot c)$。回溯只在需要输出具体方案时才必要。`,
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
      q: md`回溯的通用模板是什么？子集、组合、排列在模板上有什么区别？`,
      a: md`
        **结论**：回溯 = 决策树上的 DFS，三要素是路径、选择列表、结束条件，核心循环是“做选择 → 递归 → 撤销选择”。

        **原因**：三类题只是选择列表不同：子集 / 组合用 \`start\` 保证只往后选（结果无序、不重复），子集在每个节点收集，组合在满足条件（长度或和）时收集；
        排列每层都从头枚举，用 \`used[]\` 跳过已在路径上的元素，在叶子收集。可重复使用元素时递归传 \`i\`，否则传 \`i + 1\`。

        **延伸**：有重复元素时排序 + 同层去重；剪枝（排序后 break、可行性判断）决定实际速度；只求数量时考虑 DP。
      `,
    },
    {
      id: 'qa-dedup',
      freq: 3,
      q: md`全排列 II 的去重条件为什么是 \`not used[i-1]\`？写成 \`used[i-1]\` 行不行？`,
      a: md`
        **结论**：两者都能得到正确结果，\`not used[i-1]\` 是“树层去重”，效率高得多。

        **原因**：排序后相同值相邻。\`not used[i-1]\` 表示前一个相同值不在路径上——它只可能是在本层刚被撤销，再选当前值会展开一棵相同的子树，所以剪掉；
        \`used[i-1]\` 则强制相同值按下标逆序进入路径，错误的顺序要到更深层才被发现（树枝去重），产生大量死路。

        **延伸**：不想排序时，可以在每一层用一个局部 set 记录已展开的值；结果层面的 set 去重是最差的做法——重复子树已经搜过了。
      `,
    },
    {
      id: 'qa-complexity',
      freq: 2,
      q: md`子集、排列的时间复杂度分别是多少？为什么说这是“下界”？`,
      a: md`
        **结论**：子集 $O(n \cdot 2^n)$，排列 $O(n \cdot n!)$，组合 $O(k \binom{n}{k})$。

        **原因**：答案个数分别是 $2^n$、$n!$、$\binom{n}{k}$，每个答案要拷贝 $O(n)$ 输出——只要题目要求列出全部方案，任何算法都不可能比输出规模更快。
        回溯的优化空间在于“不产生答案的节点”：剪枝、去重、更好的搜索顺序。

        **延伸**：只要计数时，组合计数常可用 DP 或组合数公式做到多项式时间；$n$ 较大时考虑位运算枚举（子集用 \`for mask in range(1 << n)\`）。
      `,
    },
    {
      id: 'qa-vs-dp',
      freq: 2,
      q: md`回溯和动态规划有什么关系？什么时候回溯可以改成 DP？`,
      a: md`
        **结论**：DP 是“带记忆的搜索”，当子问题的结果只依赖少量状态、且不需要输出具体路径时，回溯可以加记忆化变成 DP。

        **原因**：回溯的递归函数若只依赖 \`(start, remain)\` 这类可哈希的小状态，而不依赖完整的 path，那么相同状态的子树结果相同，可以缓存（如组合计数、单词拆分 LC139）。
        需要枚举全部方案（输出规模指数级）时，记忆化帮不了太多。

        **延伸**：LC139 单词拆分用 DP 判可行，LC140 要输出全部拆分方案就回到回溯（可配合记忆化剪掉不可行的后缀）。
      `,
    },
  ],

  cheatsheet: md`
    **子集 / 组合（枚举 start）**
    \`\`\`python
    def dfs(start, remain):
        if remain == 0: res.append(path[:]); return
        for i in range(start, len(c)):
            if i > start and c[i] == c[i-1]: continue   # 同层去重（需排序）
            if c[i] > remain: break                      # 排序剪枝
            path.append(c[i]); dfs(i + 1, remain - c[i]); path.pop()
    \`\`\`
    **排列**：每层 \`for i in range(n)\` + \`used[]\`；去重 \`i>0 and a[i]==a[i-1] and not used[i-1]\`

    | 题型 | 递归参数 | 收集时机 |
    |---|---|---|
    | 子集 | \`i + 1\` | 每个节点 |
    | 组合（每个一次 / 可重复） | \`i + 1\` / \`i\` | 满足条件 |
    | 排列 | — (used) | 叶子 |

    - \`res.append(path[:])\` 必须拷贝；复杂度：子集 $n2^n$、排列 $n \cdot n!$
  `,
})
