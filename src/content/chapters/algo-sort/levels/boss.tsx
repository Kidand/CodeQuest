/**
 * BOSS：排序综合（algo-sort-boss）
 * 复盘全章 + LC23 合并 K 个升序链表（堆 / 分治）+ LC315 计算右侧小于当前元素的个数（归并 + 下标数组）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { SortRaceViz } from '../viz/SortRaceViz'

export default defineLevel({
  learn: [
    {
      id: 'recap',
      title: '复盘：怎么选排序 / 选择算法',
      body: md`
        | 场景 | 首选 | 理由 |
        |---|---|---|
        | 内存中的基本类型数组 | 快排（introsort） | 原地、缓存友好、常数小 |
        | 需要稳定 / 多关键字排序 | 归并、Timsort | 相等元素保持原顺序 |
        | 链表 | 归并 | 不需随机访问，合并只改指针 |
        | 数据大于内存 | 外部归并（分块内排 + K 路归并） | 只需顺序读写 |
        | 第 k 大、数组在内存 | 快速选择（三路 + 随机） | 期望 $O(n)$ |
        | top-k、流式 / 超大数据 | 大小为 k 的小顶堆 | $O(n\log k)$、$O(k)$ 内存 |
        | 整数、值域小 | 计数 / 基数排序 | 突破 $\Omega(n\log n)$ 比较下界 |
        | GPU 上的 top-k（采样、beam、MoE） | radix select / \`torch.topk\` | 并行友好 |

        **三个最常见的退化**：固定 pivot 遇到有序输入；二路划分遇到大量重复值；递归快排 / 递归合并链表撞上 Python 的 1000 层递归上限。

        本关的两道 BOSS 题各自融合了两种技巧：**LC23** = 归并思想 + 堆（K 路归并），**LC315** = 归并排序 + 下标数组（逆序对的“按元素分组”版本）。
      `,
      viz: SortRaceViz,
      after: md`
        > [!INTERVIEW] 回答“哪个排序最快”的正确姿势
        > 没有绝对最快：先问数据规模、是否在内存、是否需要稳定、键的类型与分布、是否部分有序。再给出选择与理由，最后补一句工程实现（Python Timsort、C++ introsort、Java 双轴快排 / TimSort）。
      `,
    },
  ],

  quiz: [
    {
      id: 'q-list',
      type: 'single',
      difficulty: 1,
      prompt: md`对一个有 $10^6$ 个节点的单链表排序，最合适的算法是：`,
      options: ['归并排序', '快速排序（随机 pivot）', '堆排序', '插入排序'],
      answer: 0,
      explanation: md`链表不支持随机访问：随机 pivot 要先遍历、堆排序依赖下标跳转、插入排序 $O(n^2)$。归并排序只需顺序访问，合并只改指针，自底向上可做到 $O(1)$ 额外空间。`,
    },
    {
      id: 'q-worst',
      type: 'multi',
      difficulty: 2,
      prompt: md`以下哪些算法的**最坏**时间复杂度是 $O(n\log n)$？（多选）`,
      options: ['归并排序', '堆排序', 'introsort（C++ std::sort）', '随机 pivot 快速排序', 'Timsort'],
      answer: [0, 1, 2, 4],
      explanation: md`随机化快排最坏仍是 $O(n^2)$（概率极小）；introsort 在递归过深时切换堆排序，最坏 $O(n\log n)$；Timsort 本质是归并，最坏 $O(n\log n)$、有序输入 $O(n)$。`,
    },
    {
      id: 'q-heapify',
      type: 'judge',
      difficulty: 1,
      prompt: md`自底向上建堆（heapify）的时间复杂度是 $O(n \log n)$。`,
      answer: false,
      explanation: md`错误，是 $O(n)$。下沉代价取决于节点高度，高度 h 的节点约 $n/2^{h+1}$ 个，$\sum_h h \cdot n/2^{h+1} = O(n)$。逐个 push 建堆才是 $O(n\log n)$。`,
    },
    {
      id: 'q-bound5',
      type: 'numeric',
      difficulty: 2,
      prompt: md`对 5 个互不相同的元素做比较排序，信息论下界 $\lceil \log_2 5! \rceil$ 是多少次比较？`,
      answer: 7,
      tolerance: 0,
      explanation: md`$5! = 120$，$\log_2 120 \approx 6.91$，向上取整为 7。存在恰好用 7 次比较排好 5 个元素的方法（归并插入排序），所以这个下界在 n = 5 时是紧的。`,
    },
    {
      id: 'q-inv',
      type: 'numeric',
      difficulty: 2,
      prompt: md`\`[1, 5, 4, 3, 2]\` 有多少个逆序对？`,
      answer: 6,
      tolerance: 0,
      explanation: md`1 不参与任何逆序对；\`[5, 4, 3, 2]\` 完全逆序，$\binom{4}{2} = 6$ 个。`,
    },
    {
      id: 'q-lomuto',
      type: 'fill',
      difficulty: 2,
      prompt: md`对 \`[4, 9, 1, 7, 3, 6]\` 做一次 Lomuto 划分（pivot = 末尾的 6），返回的下标是 ____(1)____；划分后下标 0 处的元素是 ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['3'], numeric: true },
        { label: '(2)', answers: ['4'], numeric: true },
      ],
      explanation: md`4 < 6 → 与自己交换，i=1；9 跳过；1 < 6 → 与 9 交换 → \`[4,1,9,7,3,6]\`，i=2；7 跳过；3 < 6 → 与 9 交换 → \`[4,1,3,7,9,6]\`，i=3；pivot 与 7 交换 → \`[4,1,3,6,9,7]\`，返回 3，下标 0 仍是 4。`,
    },
    {
      id: 'q-scenario',
      type: 'match',
      difficulty: 2,
      prompt: md`把场景与最合适的方法配对：`,
      pairs: [
        ['$10^7$ 个取值在 0～100 的整数排序', '计数排序'],
        ['日志流中实时维护延迟最高的 100 条', '大小为 100 的小顶堆'],
        ['内存中数组求中位数', '快速选择'],
        ['按（部门，入职时间）多关键字排序', '稳定排序分两次排（先次键后主键）'],
        ['100 GB 文件排序、4 GB 内存', '外部归并排序'],
      ],
      explanation: md`值域小用计数；流式 top-k 用堆；一次性选择用快速选择；多关键字依赖稳定性；超出内存用外部归并。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出用堆合并 K 个升序链表的核心代码（有 2 行干扰项）：`,
      lines: [
        'def merge_k_lists(lists):',
        '    h = [(node.val, i, node) for i, node in enumerate(lists) if node]',
        '    heapq.heapify(h)',
        '    dummy = tail = ListNode()',
        '    while h:',
        '        _, i, node = heapq.heappop(h)',
        '        tail.next = tail = node',
        '        if node.next:',
        '            heapq.heappush(h, (node.next.val, i, node.next))',
        '    return dummy.next',
      ],
      distractors: ['    h = [(node.val, node) for node in lists if node]', '        tail = tail.next = node'],
      explanation: md`
        元组里放下标 \`i\` 作为第二关键字：值相等时比较 i，避免去比较两个 ListNode（没有定义 \`<\` 会抛 TypeError）。
        \`tail.next = tail = node\` 的赋值顺序是先 \`tail.next = node\` 再 \`tail = node\`（Python 链式赋值从左到右）；干扰项 \`tail = tail.next = node\` 先把 tail 改成 node，再执行 \`node.next = node\`，造成自环。
      `,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 top-k（最大的 k 个）实现在 \`nums = [5, 1, 9, 7, 3], k = 2\` 上返回 \`[3, 9]\`。点出有问题的行：`,
      code: py`
        import heapq

        def top_k(nums, k):
            h = nums[:k]
            heapq.heapify(h)
            for x in nums[k:]:
                heapq.heapreplace(h, x)
            return sorted(h)
      `,
      bugLines: [7],
      explanation: md`第 7 行无条件 \`heapreplace\`：即使 x 比堆顶小也会挤掉堆顶。应改为 \`if x > h[0]: heapq.heapreplace(h, x)\`，或直接 \`heapq.heappushpop(h, x)\`（x 更小时直接被弹出，堆不变）。以本例为例：处理到 3 时堆为 \`[7, 9]\`，无条件替换把 7 挤掉，得到 \`[3, 9]\`；修正后结果为 \`[7, 9]\`。`,
    },
    {
      id: 'q-degenerate',
      type: 'pick',
      difficulty: 2,
      prompt: md`“随机 pivot + Lomuto 二路划分”的快排在哪些输入上会退化到 $O(n^2)$？（点出全部）`,
      items: ['随机排列', '已排序', '已逆序', '全部相同', '只有 0 和 1'],
      answer: [3, 4],
      explanation: md`随机 pivot 解决了有序 / 逆序输入；但等于 pivot 的元素都被分到同一侧，全相同时每轮只减少一个元素。只有 0/1 两种值时，递归很快进入“全是 0”或“全是 1”的子区间，同样退化。三路划分可以解决。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把外部排序（100 GB 数据、4 GB 内存）的步骤按顺序排列：`,
      items: [
        '按 4 GB 分块读入，每块在内存中排序',
        '把 25 个有序块分别写回磁盘',
        '为每个有序块分配读缓冲，取各自的首元素建小顶堆',
        '反复弹出堆顶写入输出文件，并从对应块补充下一个元素',
      ],
      explanation: md`第一阶段生成有序段（runs），第二阶段 K 路归并。若缓冲区不够同时打开 25 路，就分多轮归并。`,
    },
    {
      id: 'q-gpu',
      type: 'single',
      difficulty: 3,
      prompt: md`为什么推理框架在 GPU 上做 top-k 采样时不使用二叉堆？`,
      options: [
        '堆的 push / pop 存在顺序依赖，难以并行；radix select 等算法每轮可以对所有元素并行统计',
        'GPU 不支持整数比较',
        '堆只能找最小值，不能找最大值',
        'top-k 采样必须对整个词表完全排序',
      ],
      answer: 0,
      explanation: md`GPU 需要大规模数据并行。radix select 按位分轮、每轮并行构建直方图来定位第 k 大的阈值；堆的操作依赖上一步结果，几乎是串行的。C 取负即可；D 错，top-k 只需要选出 k 个，\`torch.topk\` 也不做完全排序。`,
    },
  ],

  guided: [],

  challenges: [
    {
      id: 'c-merge-k',
      title: '合并 K 个升序链表',
      difficulty: 'hard',
      source: 'LeetCode 23',
      timeLimitMin: 25,
      description: md`
        给定 K 个升序链表组成的列表 \`lists\`，合并为一个升序链表并返回头节点。

        \`\`\`python
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def merge_k_lists(lists: list[ListNode | None]) -> ListNode | None: ...
        \`\`\`

        **要求**
        - 时间 $O(N \log K)$（N 为节点总数）：用堆或两两分治合并。隐藏测试中节点的 \`val\` 会被替换为“计数比较次数”的对象，
          $K = 400$、$N = 2000$ 时比较次数须 ≤ $40N$——逐个依次合并（$O(NK)$）会被卡住；
        - 只修改 next 指针，不要新建值节点；\`val\` 只支持比较运算（\`<\`、\`<=\`、\`==\` 等）。
        - 如果你想把节点直接放进堆，可以在自己的 \`ListNode\` 里定义 \`__lt__\`（测试会沿用你定义的类）。

        **示例**：\`[[1,4,5],[1,3,4],[2,6]]\` → \`1→1→2→3→4→4→5→6\`；\`[]\` → \`None\`；\`[None]\` → \`None\`。
      `,
      starter: py`
        import heapq

        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def merge_k_lists(lists):
            # TODO: 小顶堆里放 (node.val, i, node)；或者两两分治合并
            raise NotImplementedError
      `,
      solution: py`
        import heapq

        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def merge_k_lists(lists):
            # (值, 链表编号, 节点)：编号作为第二关键字，避免值相等时去比较节点对象
            h = [(node.val, i, node) for i, node in enumerate(lists) if node]
            heapq.heapify(h)
            dummy = tail = ListNode()
            while h:
                _, i, node = heapq.heappop(h)
                tail.next = node
                tail = node
                if node.next:
                    heapq.heappush(h, (node.next.val, i, node.next))
            tail.next = None
            return dummy.next
      `,
      solutionAlt: [
        {
          label: '两两分治合并（O(N log K)，不需要堆）',
          code: py`
            def merge_two(a, b):
                dummy = tail = ListNode()
                while a and b:
                    if a.val <= b.val:
                        tail.next, a = a, a.next
                    else:
                        tail.next, b = b, b.next
                    tail = tail.next
                tail.next = a or b
                return dummy.next

            def merge_k_lists(lists):
                lists = [x for x in lists if x]
                if not lists:
                    return None
                while len(lists) > 1:                 # 每轮链表数减半，共 log K 轮
                    lists = [merge_two(lists[i], lists[i + 1] if i + 1 < len(lists) else None)
                             for i in range(0, len(lists), 2)]
                return lists[0]
          `,
        },
      ],
      explanation: md`
        **堆**：堆中始终保存每个链表的当前头节点（至多 K 个）。弹出最小的接到结果尾部，再把它的后继放入堆。每个节点进出堆各一次，$O(N\log K)$，堆空间 $O(K)$。
        元组 \`(val, i, node)\` 中的 \`i\` 是为了值相等时不去比较 ListNode。

        **分治**：像归并排序一样两两合并，每轮链表数减半，共 $\lceil\log_2 K\rceil$ 轮，每轮处理全部 N 个节点，$O(N\log K)$，不需要堆。

        **反例**：依次把每个链表合并进结果，第 i 次合并要遍历约 $iN/K$ 个节点，总共 $O(NK)$。

        **常见错误**：堆元素写成 \`(node.val, node)\`，值相等时比较节点抛 TypeError；忘记处理空链表；合并用递归导致爆栈。
      `,
      complexity: { time: 'O(N log K)', space: 'O(K)' },
      hints: [
        md`每一步只需要知道 K 个链表当前头节点中的最小者。什么数据结构能 $O(\log K)$ 取出最小值并插入新元素？`,
        md`堆中放 \`(node.val, i, node)\`：i 是链表编号，用于值相等时打破平局。`,
        md`或者不用堆：像归并排序那样两两合并，每轮链表数减半。`,
      ],
      setup: py`
        import random

        if "ListNode" not in globals():
            class ListNode:
                def __init__(self, val=0, next=None):
                    self.val = val
                    self.next = next

        class _K:
            """计数比较次数的值包装"""
            __slots__ = ("v",)
            n = 0
            budget = 10**12
            def __init__(self, v):
                self.v = v
            @staticmethod
            def _tick():
                _K.n += 1
                if _K.n > _K.budget:
                    raise AssertionError(f"比较次数超过 {_K.budget}：复杂度不是 O(N log K)（是不是逐个合并了？）")
            def __lt__(self, o): _K._tick(); return self.v < o.v
            def __le__(self, o): _K._tick(); return self.v <= o.v
            def __gt__(self, o): _K._tick(); return self.v > o.v
            def __ge__(self, o): _K._tick(); return self.v >= o.v
            def __eq__(self, o): _K._tick(); return self.v == o.v
            def __ne__(self, o): _K._tick(); return self.v != o.v
            __hash__ = None

        def _build(vals, wrap=False):
            dummy = ListNode()
            cur, nodes = dummy, []
            for v in vals:
                cur.next = ListNode(_K(v) if wrap else v)
                cur = cur.next
                nodes.append(cur)
            return dummy.next, nodes

        def _to_list(head, limit=10**6):
            out = []
            while head is not None:
                out.append(getattr(head.val, "v", head.val))
                head = head.next
                if len(out) > limit:
                    raise AssertionError("链表中出现了环（或长度异常）")
            return out

        def _free(nodes):
            # 断开 next，避免长链表递归析构（Pyodide 的栈较浅）
            for x in nodes:
                x.next = None
      `,
      tests: [
        {
          name: '示例',
          code: py`
            lists, alln = [], []
            for vals in ([1, 4, 5], [1, 3, 4], [2, 6]):
                h, ns = _build(vals)
                lists.append(h)
                alln += ns
            got = _to_list(merge_k_lists(lists))
            assert got == [1, 1, 2, 3, 4, 4, 5, 6], f"得到 {got}"
          `,
        },
        {
          name: '空输入',
          code: py`
            assert merge_k_lists([]) is None
            assert merge_k_lists([None]) is None
            assert merge_k_lists([None, None]) is None
            h, _ = _build([1])
            assert _to_list(merge_k_lists([None, h])) == [1]
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(23)
            for _ in range(150):
                groups = [sorted(rnd.randint(-9, 9) for _ in range(rnd.randint(0, 6))) for _ in range(rnd.randint(0, 8))]
                lists = [_build(g)[0] for g in groups]
                got = _to_list(merge_k_lists(lists))
                exp = sorted(x for g in groups for x in g)
                assert got == exp, f"lists={groups}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: 'K = 400，N = 2000：比较次数 ≤ 40N、复用原节点',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            groups = [sorted(rnd.randint(0, 10**6) for _ in range(5)) for _ in range(400)]
            lists, alln = [], []
            for g in groups:
                h, ns = _build(g, wrap=True)
                lists.append(h)
                alln += ns
            N = len(alln)
            try:
                _K.n, _K.budget = 0, 40 * N
                res = merge_k_lists(lists)
                _K.budget = 10**12
                seen, p = set(), res
                while p is not None and len(seen) <= N:
                    seen.add(id(p))
                    p = p.next
                assert seen == {id(x) for x in alln}, "结果中的节点与原节点不一致：请只修改 next 指针"
                got = _to_list(res)
                assert got == sorted(x for g in groups for x in g), "合并结果不正确"
            finally:
                _K.budget = 10**12
                _free(alln)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-k-arrays',
          q: md`如果 K 个有序数组都在磁盘上，各自很大，怎么合并？`,
          a: md`这就是外部排序的 K 路归并：每个文件分配一个读缓冲块，堆中放 (值, 文件编号)，弹出后从对应缓冲取下一个，缓冲读空再从磁盘读下一块。输出也用缓冲批量写。K 太大时分多轮归并，或用败者树减少比较次数。`,
        },
      ],
    },
    {
      id: 'c-count-smaller',
      title: '计算右侧小于当前元素的个数',
      difficulty: 'hard',
      source: 'LeetCode 315',
      timeLimitMin: 30,
      description: md`
        给定整数数组 \`nums\`，返回数组 \`counts\`，其中 \`counts[i]\` 是 \`nums[i]\` 右侧**严格小于** \`nums[i]\` 的元素个数。

        \`\`\`python
        def count_smaller(nums: list[int]) -> list[int]: ...
        \`\`\`

        **示例**：\`[5, 2, 6, 1]\` → \`[2, 1, 1, 0]\`；\`[-1, -1]\` → \`[0, 0]\`。

        **要求**：$O(n\log n)$。隐藏测试含 $n = 6000$ 的计时用例；不能使用 \`bisect.insort\`（向列表中间插入是 $O(n)$ 的，整体 $O(n^2)$，会检查源码）。
        推荐：对**下标数组**做归并排序；也可以离散化 + 树状数组。
      `,
      starter: py`
        def count_smaller(nums: list[int]) -> list[int]:
            # TODO: 对下标数组归并排序；取左段元素时，加上已从右段取走的个数
            raise NotImplementedError
      `,
      solution: py`
        def count_smaller(nums: list[int]) -> list[int]:
            n = len(nums)
            ans = [0] * n
            idx = list(range(n))          # 对下标排序（按 nums 值），这样能把计数记回原位置
            buf = [0] * n

            def sort(lo: int, hi: int) -> None:     # 处理 idx[lo:hi]
                if hi - lo <= 1:
                    return
                mid = (lo + hi) // 2
                sort(lo, mid)
                sort(mid, hi)
                i, j, k = lo, mid, lo
                while i < mid and j < hi:
                    if nums[idx[j]] < nums[idx[i]]:    # 右段元素严格更小：先取右
                        buf[k] = idx[j]; j += 1
                    else:                             # 取左：右段已取走的 j - mid 个都比它小
                        ans[idx[i]] += j - mid
                        buf[k] = idx[i]; i += 1
                    k += 1
                while i < mid:
                    ans[idx[i]] += j - mid
                    buf[k] = idx[i]; i += 1; k += 1
                while j < hi:
                    buf[k] = idx[j]; j += 1; k += 1
                idx[lo:hi] = buf[lo:hi]

            sort(0, n)
            return ans
      `,
      solutionAlt: [
        {
          label: '离散化 + 树状数组（从右往左扫）',
          code: py`
            def count_smaller(nums):
                rank = {v: i + 1 for i, v in enumerate(sorted(set(nums)))}
                m = len(rank)
                tree = [0] * (m + 1)
                out = []
                for x in reversed(nums):
                    r, s = rank[x] - 1, 0              # 查询已插入的、排名 < rank[x] 的个数
                    while r > 0:
                        s += tree[r]; r -= r & -r
                    out.append(s)
                    r = rank[x]
                    while r <= m:
                        tree[r] += 1; r += r & -r
                return out[::-1]
          `,
        },
      ],
      explanation: md`
        **与逆序对的关系**：\`counts[i]\` 就是“以 i 为左端点的逆序对个数”，逆序对总数等于 \`sum(counts)\`。

        **归并做法**：对下标数组 \`idx\` 按值归并。合并两段时，若取的是**左段**元素 \`idx[i]\`，右段中已经被取走的 \`j - mid\` 个元素都严格小于它、且原本都在它右边，于是 \`ans[idx[i]] += j - mid\`。
        相等时先取左（只有右段严格更小才先取右），保证只统计“严格小于”。

        **复杂度**：$O(n\log n)$ 时间，$O(n)$ 空间。

        **常见错误**：直接对值排序导致丢失原下标；相等元素先取右导致多计；左段剩余元素没有补加 \`j - mid\`。
      `,
      complexity: { time: 'O(n log n)', space: 'O(n)' },
      hints: [
        md`\`counts[i]\` 是以 i 为左端点的逆序对个数。逆序对可以在归并时统计——但现在要按元素分别记账，怎么知道每个值原来在哪个位置？`,
        md`对**下标**排序：比较 \`nums[idx[i]]\` 与 \`nums[idx[j]]\`，移动的是下标。`,
        md`合并时取左段元素 \`idx[i]\`，就把“已经取走的右段元素个数 \`j - mid\`”加到 \`ans[idx[i]]\` 上；左段剩余元素在最后也要补加。`,
      ],
      setup: py`
        import linecache, random, time

        def _brute(a):
            return [sum(1 for j in range(i + 1, len(a)) if a[j] < a[i]) for i in range(len(a))]

        def _ref_fast(a):
            rank = {v: i + 1 for i, v in enumerate(sorted(set(a)))}
            m, tree, out = len(rank), [0] * (len(rank) + 1), []
            for x in reversed(a):
                r, s = rank[x] - 1, 0
                while r > 0:
                    s += tree[r]
                    r -= r & -r
                out.append(s)
                r = rank[x]
                while r <= m:
                    tree[r] += 1
                    r += r & -r
            return out[::-1]
      `,
      tests: [
        { name: '示例', code: py`assert count_smaller([5, 2, 6, 1]) == [2, 1, 1, 0]` },
        {
          name: '边界与重复',
          code: py`
            assert count_smaller([]) == [] and count_smaller([-1]) == [0]
            assert count_smaller([-1, -1]) == [0, 0], "严格小于：相等不计"
            assert count_smaller([2, 0, 1]) == [2, 0, 0]
            assert count_smaller([3, 3, 1, 3]) == [1, 1, 0, 0]
          `,
        },
        {
          name: '不使用 insort',
          code: py`
            src = "".join(linecache.getlines("solution.py"))
            assert "insort" not in src, "bisect.insort 每次插入 O(n)，整体 O(n^2)，请用归并或树状数组"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(315)
            for _ in range(250):
                a = [rnd.randint(-8, 8) for _ in range(rnd.randint(0, 25))]
                exp, got = _brute(a), count_smaller(list(a))
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：n = 6000',
          hidden: true,
          code: py`
            rnd = random.Random(2)
            a = [rnd.randint(-10**4, 10**4) for _ in range(6000)]
            t0 = time.perf_counter()
            got = count_smaller(list(a))
            dt = time.perf_counter() - t0
            assert dt < 0.8, f"用时 {dt:.2f}s，疑似 O(n^2)"
            exp = _ref_fast(a)
            assert got == exp, "n = 6000 时结果不正确"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-493',
          q: md`LC493「翻转对」要求统计 \`nums[i] > 2 * nums[j]\`（i < j）的对数，直接套用本题的合并会出什么问题？`,
          a: md`合并时的取数顺序依据是 \`nums[i]\` 与 \`nums[j]\` 的大小，而计数条件是 \`nums[i] > 2·nums[j]\`，两者不一致，不能在同一次比较中完成。做法是：在合并之前，利用两段各自有序，用双指针单独统计一遍满足 \`L[i] > 2·R[j]\` 的对数（O(len)），然后再正常合并。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-which',
      freq: 3,
      q: md`如果让你实现一个通用的排序库函数，你会怎么设计？`,
      a: md`
        **结论**：参考 introsort 与 Timsort：基本类型用“随机 / 三数取中快排 + 三路划分 + 深度超限切堆排序 + 小区间插入排序”；对象与需要稳定的场景用 Timsort。

        **原因**：快排平均最快但有退化风险，introsort 用堆排序兜底保证最坏 $O(n\log n)$；插入排序在小区间常数最小；真实数据常部分有序，Timsort 识别天然有序段（run）并用归并合并，稳定且对有序输入接近 $O(n)$。

        **延伸**：还要考虑比较函数的代价（对象比较贵时减少比较次数更重要）、内存约束（归并需要 $O(n)$ 缓冲）、以及是否需要并行（GPU 上用基数排序 / 排序网络）。
      `,
    },
    {
      id: 'qa-merge-k',
      freq: 2,
      q: md`合并 K 个有序链表，堆和分治两种方法怎么选？`,
      a: md`
        **结论**：复杂度都是 $O(N\log K)$；堆适合流式（链表在不断产生）、分治不需要额外的堆空间且常数更小。

        **原因**：堆每次 pop / push 是 $O(\log K)$，每个节点各一次；分治像归并排序一样两两合并，共 $\log K$ 轮，每轮 $O(N)$。依次合并是 $O(NK)$，K 大时很慢。

        **延伸**：Python 中堆元素要用 \`(val, i, node)\` 避免比较节点；外部排序的 K 路归并还会用败者树，把每次调整的比较次数稳定在 $\log K$。
      `,
    },
  ],

  cheatsheet: md`
    **合并 K 个链表（堆）**
    \`\`\`python
    h = [(n.val, i, n) for i, n in enumerate(lists) if n]; heapq.heapify(h)
    dummy = tail = ListNode()
    while h:
        _, i, n = heapq.heappop(h); tail.next = n; tail = n
        if n.next: heapq.heappush(h, (n.next.val, i, n.next))
    \`\`\`
    **右侧更小的个数**：对下标归并；取左段元素时 \`ans[idx[i]] += j - mid\`；相等先取左

    | 场景 | 选择 |
    |---|---|
    | 数组（基本类型） | introsort |
    | 稳定 / 对象 | Timsort / 归并 |
    | 链表 | 归并（自底向上 $O(1)$ 空间） |
    | 第 k 大 | 快速选择（随机 + 三路） |
    | 流式 top-k | 大小 k 的小顶堆 |
    | 小值域整数 | 计数 / 基数 |
  `,
})
