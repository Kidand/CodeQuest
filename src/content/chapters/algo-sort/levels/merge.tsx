/**
 * 归并排序（algo-sort-merge）
 * LC912 / LC148 / 剑指 Offer 51（LCR 170）：自顶向下 / 自底向上、链表归并、逆序对计数、外部排序。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { MergeSortViz, MergeSortBottomUpViz } from '../viz/MergeSortViz'
import { MergeInvViz } from '../viz/MergeInvViz'

const LISTNODE_SETUP = py`
  class ListNode:
      def __init__(self, val=0, next=None):
          self.val = val
          self.next = next

  def _build(vals):
      dummy = ListNode()
      cur, nodes = dummy, []
      for v in vals:
          cur.next = ListNode(v)
          cur = cur.next
          nodes.append(cur)
      return dummy.next, nodes

  def _free(nodes):
      # 逐个断开 next，避免长链表在释放时递归析构（Pyodide 的栈较浅，几千层就会溢出）
      for x in nodes:
          x.next = None

  def _to_list(head, limit=10**6):
      out = []
      while head is not None:
          out.append(head.val)
          head = head.next
          if len(out) > limit:
              raise AssertionError("链表中出现了环（或长度异常）")
      return out
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'divide',
      title: '分治：拆到底，再合并上来',
      body: md`
        归并排序是最标准的**分治**：把数组对半拆开，分别排好序，再把两个有序段**合并**。
        与快排相反——快排的功夫在“拆”（划分），合并为空；归并的拆分是机械的对半，功夫全在“合”。

        $$
        T(n) = 2\,T(n/2) + O(n) \;\Rightarrow\; T(n) = O(n \log n)
        $$

        递归树有 $\lceil \log_2 n \rceil$ 层，每层的合并总共处理 $n$ 个元素。**与输入无关**——最好、平均、最坏都是 $O(n\log n)$，这是它相对快排最大的优势。

        代价：数组上合并需要 $O(n)$ 的辅助空间（不能原地高效合并）。递归栈深度 $O(\log n)$。
      `,
      viz: MergeSortViz,
      after: md`
        > [!TIP] 小区间用插入排序
        > 工程实现（如 Timsort、Java 的 TimSort）在段长较小时改用插入排序：小数组上插入排序常数更小、对缓存友好。Timsort 还会先找出数据中天然有序的“run”，对部分有序的真实数据比较次数远少于 $n \log n$。
      `,
      checkpoint: {
        id: 'cp-divide',
        type: 'numeric',
        prompt: md`对 $n = 1000$ 个元素做自顶向下归并排序，递归树（不含叶子的单元素调用）共有多少层合并？即 $\lceil \log_2 n \rceil$ 是多少？`,
        answer: 10,
        tolerance: 0,
        explanation: md`$2^9 = 512 < 1000 \le 1024 = 2^{10}$，所以是 10 层。每层合并处理 1000 个元素，总共约 $10^4$ 次元素移动。`,
      },
    },
    {
      id: 'merge',
      title: '合并：双指针、稳定性与逆序对',
      body: md`
        合并两个有序段 \`L\`、\`R\`：两个指针各指向段首，每次取较小者放入结果，一方耗尽后把另一方剩余部分整体接上。
        $|L| + |R|$ 个元素最多比较 $|L| + |R| - 1$ 次。

        **稳定性**：相等时取**左边**（\`L[i] <= R[j]\`），相等元素保持原相对顺序。写成 \`<\` 就不稳定了。

        **逆序对**（剑指 Offer 51）：数对 $(i, j)$ 满足 $i < j$ 且 $a_i > a_j$。总数 = 左段内部 + 右段内部 + 跨段。
        跨段部分可以在合并时顺手统计：当取右边的 \`R[j]\` 时，由于左段有序，\`L[i:]\` 中的每个元素都大于 \`R[j]\`，

        $$
        \text{inv} \mathrel{+}= |L| - i
        $$

        这样逆序对计数从 $O(n^2)$ 降到 $O(n \log n)$。逆序对数也是“冒泡排序需要的交换次数”，衡量数组的无序程度。
      `,
      viz: MergeInvViz,
      after: md`
        > [!WARN] 相等元素不是逆序对
        > 合并时 \`L[i] == R[j]\` 必须先取左边且**不计数**。若写成 \`L[i] < R[j]\` 才取左边，相等时会先取右边并错误地把相等的元素计入逆序对。
      `,
      checkpoint: {
        id: 'cp-merge',
        type: 'numeric',
        prompt: md`数组 \`[7, 5, 6, 4]\` 共有多少个逆序对？`,
        answer: 5,
        tolerance: 0,
        explanation: md`(7,5)、(7,6)、(7,4)、(5,4)、(6,4)，共 5 个。用归并：左段 \`[7,5]\` 内部 1 个、右段 \`[6,4]\` 内部 1 个；排序后合并 \`[5,7]\` 与 \`[4,6]\`：取 4 时 +2、取 6 时 +1，跨段 3 个，合计 5。`,
      },
    },
    {
      id: 'bottomup',
      title: '自底向上与链表归并（LC148）',
      body: md`
        **自底向上**：把每个元素看作长度 1 的有序段，按段长 1, 2, 4, … 逐轮合并相邻两段，$\lceil \log_2 n \rceil$ 轮。没有递归。

        **链表排序（LC148）首选归并**：
        - 链表不能随机访问，快排的划分与堆排序的下标跳跃都很别扭；
        - 链表合并只需改指针，**不需要辅助数组**——这正好抵消了归并在数组上的最大缺点；
        - 自顶向下：快慢指针找中点（\`slow\` 走一步、\`fast\` 走两步），断开后递归两半再合并，栈深 $O(\log n)$；
        - 自底向上：按段长倍增，每轮用“切下 k 个节点”与“合并后接到尾部”两个辅助函数，**额外空间 $O(1)$**，满足 LC148 的进阶要求。
      `,
      viz: MergeSortBottomUpViz,
      after: md`
        > [!WARN] 链表合并别写成递归
        > 递归版 \`merge(a, b)\`（\`a.next = merge(a.next, b)\`）的递归深度等于两条链表的总长度，链表有几千个节点就会撞上 Python 的 1000 层递归上限。合并一定用循环 + 哑节点（dummy）。
      `,
      checkpoint: {
        id: 'cp-bottomup',
        type: 'single',
        prompt: md`用快慢指针找链表中点时（\`slow = head, fast = head.next\`，每轮 \`slow\` 走一步、\`fast\` 走两步，直到 \`fast\` 或 \`fast.next\` 为空），对 \`1 → 2 → 3 → 4\`，\`slow\` 最终停在哪个节点？`,
        options: ['2', '3', '1', '4'],
        answer: 0,
        explanation: md`初始 slow=1、fast=2；fast.next=3 非空 → slow=2、fast=4；fast.next 为空，停止。slow=2，于是在 2 之后断开：\`1→2\` 与 \`3→4\`，两半等长。若初始 \`fast = head\`，偶数长度时 slow 会停在 3，两元素链表 \`1→2\` 会切成 \`1→2\` 与空链表，导致无限递归。`,
      },
    },
    {
      id: 'external',
      title: '外部排序：数据比内存大怎么办',
      body: md`
        经典面试题：**100 GB 的数据、4 GB 内存，如何排序？** 归并排序天然适合外部排序，因为它只需要**顺序读写**。

        1. **生成有序段（runs）**：每次读入约 4 GB，在内存中排序后写回磁盘，得到 25 个有序文件；
        2. **K 路归并**：同时打开 25 个文件，各读一个缓冲块，用**小顶堆**维护每个文件的当前最小元素，每次弹出最小者写出，并从它所在的文件补充下一个——$O(N \log K)$；
        3. 若 K 太大（文件句柄或缓冲区不够），就分多轮归并。

        总 I/O 约为“读写两遍数据”。合并 K 个有序链表（LC23）就是第 2 步的内存版。

        现实中的大规模数据处理（例如训练语料的去重、按长度分桶）通常交给 Spark / 数据库这类系统，它们的 sort-merge 正是这一思路的分布式版本。
      `,
      checkpoint: {
        id: 'cp-external',
        type: 'single',
        prompt: md`外部排序的 K 路归并阶段，为什么用小顶堆而不是每次线性扫描 K 个文件的当前元素？`,
        options: [
          '每次取最小值从 O(K) 降为 O(log K)，总代价 O(N log K) 而不是 O(N·K)',
          '堆可以减少磁盘读写的次数',
          '堆排序是稳定的',
          '堆可以让各个文件不必预先有序',
        ],
        answer: 0,
        explanation: md`磁盘 I/O 量由“读写各一遍”决定，与内存中用堆还是扫描无关（B 错）；但 CPU 侧每输出一个元素要找 K 个候选的最小值，堆把它降到 $O(\log K)$。C 错（堆排序不稳定，且这里也不是堆排序）；D 错，K 路归并的前提就是每个文件已有序。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-props',
      type: 'single',
      difficulty: 1,
      prompt: md`数组上的归并排序，时间复杂度（最好 / 最坏）、额外空间、稳定性分别是：`,
      options: [
        '$O(n\\log n)$ / $O(n\\log n)$、$O(n)$、稳定',
        '$O(n)$ / $O(n\\log n)$、$O(1)$、稳定',
        '$O(n\\log n)$ / $O(n^2)$、$O(\\log n)$、不稳定',
        '$O(n\\log n)$ / $O(n\\log n)$、$O(1)$、不稳定',
      ],
      answer: 0,
      explanation: md`标准归并对任何输入都做 $\lceil\log_2 n\rceil$ 层合并，复杂度与输入无关；合并需要辅助数组 $O(n)$；相等时取左边保证稳定。B 描述的是 Timsort 在已有序输入上的表现；C 是快排；D 接近堆排序。`,
    },
    {
      id: 'q-list-space',
      type: 'judge',
      difficulty: 1,
      prompt: md`对链表做归并排序，和数组一样需要 $O(n)$ 的辅助空间。`,
      answer: false,
      explanation: md`错误。链表合并只需修改 next 指针，不需要辅助数组。自顶向下递归版的额外空间是 $O(\log n)$ 的栈，自底向上迭代版是 $O(1)$。`,
    },
    {
      id: 'q-facts',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于归并排序，哪些说法正确？（多选）`,
      options: [
        '合并时相等元素取左边，才能保证稳定',
        '自底向上的实现不需要递归',
        '合并 $|L|$ 与 $|R|$ 两段最多需要 $|L| + |R| - 1$ 次比较',
        '归并排序在已排序的输入上会退化到 $O(n^2)$',
        '归并排序天然适合外部排序，因为它只需顺序读写',
      ],
      answer: [0, 1, 2, 4],
      explanation: md`D 错：归并排序的复杂度与输入无关，始终 $O(n\log n)$；退化到 $O(n^2)$ 的是固定 pivot 的快排。C：每次比较输出一个元素，最后一个元素无需比较。`,
    },
    {
      id: 'q-inv',
      type: 'numeric',
      difficulty: 2,
      prompt: md`合并有序段 \`L = [3, 7, 9, 12]\` 与 \`R = [1, 5, 8, 10]\` 时，按“取右边元素时加 \`len(L) - i\`”统计的**跨段**逆序对共有多少个？`,
      answer: 10,
      tolerance: 0,
      explanation: md`取 1 时 L 剩 4 个 → +4；取 5 时 L 剩 [7,9,12] → +3；取 8 时剩 [9,12] → +2；取 10 时剩 [12] → +1。合计 10。`,
    },
    {
      id: 'q-cmp',
      type: 'fill',
      difficulty: 2,
      prompt: md`合并 \`[1, 2, 3]\` 与 \`[4, 5, 6]\` 需要 ____(1)____ 次比较；合并 \`[1, 3, 5]\` 与 \`[2, 4, 6]\` 需要 ____(2)____ 次比较。`,
      blanks: [
        { label: '(1)', answers: ['3'], numeric: true },
        { label: '(2)', answers: ['5'], numeric: true },
      ],
      explanation: md`第一种情况左段 3 次比较后耗尽，右段整体接上；第二种两段交错，每次比较输出一个元素，直到剩最后一个：$3 + 3 - 1 = 5$ 次。合并的比较次数在 $\min(|L|,|R|)$ 与 $|L|+|R|-1$ 之间。`,
    },
    {
      id: 'q-148-steps',
      type: 'order',
      difficulty: 2,
      prompt: md`把自顶向下的链表归并排序（LC148）步骤按顺序排列：`,
      items: [
        '若链表为空或只有一个节点，直接返回',
        '用快慢指针找到中点 slow',
        '断开：right = slow.next；slow.next = None',
        '递归排序左右两半',
        '用哑节点循环合并两个有序链表并返回',
      ],
      explanation: md`断开链表是关键一步：不把 \`slow.next\` 置空，左半就仍连着右半，递归无法终止。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与它借助归并的关键点配对：`,
      pairs: [
        ['剑指 Offer 51 逆序对', '取右段元素时累加 len(L) - i'],
        ['LC148 排序链表', '快慢指针找中点 + 循环合并'],
        ['LC23 合并 K 个升序链表', '两两合并或用小顶堆 K 路归并'],
        ['LC315 计算右侧小于当前元素的个数', '对下标数组归并，取左段元素时累加已取的右段个数'],
      ],
      explanation: md`LC315 是逆序对的“按元素分组”版本：需要知道每个元素贡献了多少逆序对，所以对下标排序，并在**取左段元素**时加上右段已经取走的元素个数。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出合并两个有序链表的循环写法（有 2 行干扰项）：`,
      lines: [
        'def merge(a, b):',
        '    dummy = tail = ListNode()',
        '    while a and b:',
        '        if a.val <= b.val:',
        '            tail.next, a = a, a.next',
        '        else:',
        '            tail.next, b = b, b.next',
        '        tail = tail.next',
        '    tail.next = a or b',
        '    return dummy.next',
      ],
      distractors: ['    while a or b:', '    return dummy'],
      explanation: md`\`while a or b\` 会在一方为空时访问 \`None.val\`；剩余部分用 \`tail.next = a or b\` 一次接上。返回的是 \`dummy.next\` 而不是哑节点本身。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的归并排序在 \`[3, 1, 2]\` 上返回 \`[1, 2]\`，丢了元素。点出有问题的行：`,
      code: py`
        def merge_sort(a):
            if len(a) <= 1:
                return a
            mid = len(a) // 2
            L, R = merge_sort(a[:mid]), merge_sort(a[mid:])
            out, i, j = [], 0, 0
            while i < len(L) and j < len(R):
                if L[i] <= R[j]:
                    out.append(L[i]); i += 1
                else:
                    out.append(R[j]); j += 1
            return out + R[j:]
      `,
      bugLines: [12],
      explanation: md`第 12 行应为 \`return out + L[i:] + R[j:]\`。当右段先耗尽时，左段剩余元素被丢掉：\`[3]\` 与 \`[1, 2]\` 合并时，取完 1、2 后 3 还在 L 中，没有被接上。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`数组 \`[3, 1, 4, 2]\` 中，哪些数对是逆序对？（点出全部）`,
      items: ['(3,1)', '(3,4)', '(3,2)', '(1,4)', '(1,2)', '(4,2)'],
      answer: [0, 2, 5],
      explanation: md`逆序对要求前面的数更大：(3,1)、(3,2)、(4,2)，共 3 个。也就是冒泡排序把它排好需要 3 次相邻交换。`,
    },
    {
      id: 'q-external',
      type: 'single',
      difficulty: 3,
      prompt: md`100 GB 数据、4 GB 可用内存、按 4 GB 一块生成有序段后进行一次 K 路归并。K 是多少？归并阶段每输出一个元素的 CPU 代价是多少？`,
      options: ['K = 25，$O(\\log 25)$（小顶堆）', 'K = 25，$O(1)$', 'K = 2，$O(\\log n)$', 'K = 100，$O(25)$'],
      answer: 0,
      explanation: md`$100 / 4 = 25$ 个有序段，25 路归并用大小为 25 的小顶堆，每输出一个元素一次 pop + 一次 push，$O(\log K)$。若内存放不下 25 个输入缓冲区，就要分多轮（例如先 5 路归并成 5 个段，再 5 路归并）。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-inversions',
      title: '一步步用归并统计逆序对',
      intro: md`
        实现 \`count_inversions(a)\`：返回逆序对个数，同时把 \`a\` 排好序。\`sort(lo, hi)\` 处理左闭右开区间 \`a[lo:hi]\`。
      `,
      template: py`
        def count_inversions(a):
            def sort(lo, hi):                  # 排序 a[lo:hi]，返回其中的逆序对数
                if hi - lo <= 1:
                    return 0
                mid = (lo + hi) // 2
                cnt = {{1}}                    # 左半内部 + 右半内部
                L, R = a[lo:mid], a[mid:hi]
                i = j = 0
                k = lo
                while i < len(L) and j < len(R):
                    if {{2}}:
                        a[k] = L[i]; i += 1
                    else:
                        a[k] = R[j]; j += 1
                        cnt += {{3}}           # 跨段逆序对
                    k += 1
                a[k:hi] = {{4}}                # 接上剩余部分
                return cnt
            return sort(0, len(a))
      `,
      steps: [
        {
          blank: 1,
          instruction: md`先递归处理两半，得到它们各自内部的逆序对数之和。`,
          answer: 'sort(lo, mid) + sort(mid, hi)',
          accept: ['sort(mid, hi) + sort(lo, mid)'],
          explain: md`递归返回后，\`a[lo:mid]\` 与 \`a[mid:hi]\` 都已各自有序，后面才能线性合并。`,
        },
        {
          blank: 2,
          instruction: md`什么时候取左边？注意相等的情况。`,
          answer: 'L[i] <= R[j]',
          accept: ['R[j] >= L[i]', 'not R[j] < L[i]'],
          explain: md`相等时取左边：既保证稳定，也保证相等元素不会被计成逆序对。`,
        },
        {
          blank: 3,
          instruction: md`取了右边的 \`R[j]\`，左边还剩多少个元素比它大？`,
          answer: 'len(L) - i',
          explain: md`\`L[i:]\` 有序且 \`L[i] > R[j]\`，所以剩下的全部大于 \`R[j]\`。`,
        },
        {
          blank: 4,
          instruction: md`一方耗尽后，另一方剩余部分直接接上（两者至多一个非空）。`,
          answer: 'L[i:] + R[j:]',
          accept: ['L[i:]+R[j:]', 'R[j:] + L[i:]'],
          explain: md`剩余部分已经有序且都不小于已输出的元素。此时不再产生新的跨段逆序对：若剩的是 L，它们对应的逆序对已在取每个 R 元素时计过。`,
        },
      ],
      setup: py`
        import random as _r
        def _brute(a):
            return sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > a[j])
      `,
      tests: [
        {
          name: '示例与重复值',
          code: py`
            a = [7, 5, 6, 4]
            assert count_inversions(a) == 5 and a == [4, 5, 6, 7], f"得到 {a}"
            assert count_inversions([1, 1, 1]) == 0, "相等元素不是逆序对"
            assert count_inversions([2, 2, 1]) == 2
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = _r.Random(51)
            for _ in range(200):
                a = [rnd.randint(0, 9) for _ in range(rnd.randint(0, 20))]
                exp = _brute(a)
                b = list(a)
                got = count_inversions(b)
                assert got == exp and b == sorted(a), f"a={a}: 期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        同一个骨架换一种计数方式就能解决 LC315（右侧小于当前元素的个数）和 LC493（翻转对：\`nums[i] > 2·nums[j]\`，需要在合并前单独用双指针计数）。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-reverse-pairs',
      title: '数组中的逆序对',
      difficulty: 'hard',
      source: '剑指 Offer 51 / LCR 170',
      timeLimitMin: 20,
      description: md`
        在数组中，若前面一个数字大于后面的数字，则这两个数字组成一个逆序对。返回逆序对的总数。

        \`\`\`python
        def reverse_pairs(nums: list[int]) -> int: ...
        \`\`\`

        **示例**：\`reverse_pairs([7, 5, 6, 4])\` → \`5\`

        **约束**：$0 \le n \le 5 \times 10^4$。要求 $O(n \log n)$；隐藏测试含 $n = 6000$ 的计时用例，$O(n^2)$ 双重循环会超时。可以修改输入数组。
      `,
      starter: py`
        def reverse_pairs(nums: list[int]) -> int:
            # TODO: 归并排序；合并时取右段元素，累加左段剩余个数
            raise NotImplementedError
      `,
      solution: py`
        def reverse_pairs(nums: list[int]) -> int:
            a = list(nums)
            buf = [0] * len(a)

            def sort(lo: int, hi: int) -> int:          # 处理 a[lo:hi]
                if hi - lo <= 1:
                    return 0
                mid = (lo + hi) // 2
                cnt = sort(lo, mid) + sort(mid, hi)
                i, j, k = lo, mid, lo
                while i < mid and j < hi:
                    if a[i] <= a[j]:                     # 相等取左：不计数
                        buf[k] = a[i]; i += 1
                    else:
                        buf[k] = a[j]; j += 1
                        cnt += mid - i                   # a[i:mid] 全部 > a[j]
                    k += 1
                buf[k:hi] = a[i:mid] + a[j:hi]
                a[lo:hi] = buf[lo:hi]
                return cnt

            return sort(0, len(a))
      `,
      solutionAlt: [
        {
          label: '树状数组（离散化 + BIT），同为 O(n log n)',
          code: py`
            def reverse_pairs(nums):
                rank = {v: i + 1 for i, v in enumerate(sorted(set(nums)))}   # 离散化
                n = len(rank)
                tree = [0] * (n + 1)
                ans = 0
                for x in reversed(nums):                 # 从右往左：统计右侧比 x 小的个数
                    r = rank[x] - 1
                    while r > 0:
                        ans += tree[r]; r -= r & -r
                    r = rank[x]
                    while r <= n:
                        tree[r] += 1; r += r & -r
                return ans
          `,
        },
      ],
      explanation: md`
        **思路**：逆序对 = 左半内部 + 右半内部 + 跨两半。前两者递归求得；跨半部分在合并两个有序段时统计：
        取右段元素 \`a[j]\` 时，左段剩余的 \`a[i:mid]\` 都大于它，贡献 \`mid - i\` 个。

        **复杂度**：$O(n\log n)$ 时间，$O(n)$ 空间（复用一个辅助数组 \`buf\`，避免每层都新建列表）。

        **常见错误**：相等时取右边导致相等元素被计数；计数加在“取左边”的分支；答案可能达到 $n(n-1)/2 \approx 1.25 \times 10^9$，C++/Java 需用 64 位整数（Python 无此问题）。
      `,
      complexity: { time: 'O(n log n)', space: 'O(n)' },
      hints: [
        md`把数组分成两半，逆序对分为三类：都在左半、都在右半、跨两半。前两类递归处理。`,
        md`如果两半都已排好序，跨半的逆序对能否在合并的同时数出来？`,
        md`合并时若 \`a[i] > a[j]\`（取右边），则左段剩余 \`mid - i\` 个元素都与 \`a[j]\` 构成逆序对。`,
      ],
      setup: py`
        import random, time

        def _brute(a):
            return sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > a[j])

        def _ref_fast(a):
            import bisect
            seen, cnt = [], 0
            for x in reversed(a):                 # 右侧已见元素中比 x 小的个数
                cnt += bisect.bisect_left(seen, x)
                bisect.insort(seen, x)
            return cnt
      `,
      tests: [
        { name: '示例', code: py`assert reverse_pairs([7, 5, 6, 4]) == 5` },
        {
          name: '边界与重复',
          code: py`
            assert reverse_pairs([]) == 0 and reverse_pairs([1]) == 0
            assert reverse_pairs([1, 2, 3]) == 0 and reverse_pairs([3, 2, 1]) == 3
            assert reverse_pairs([1, 1, 1]) == 0, "相等元素不构成逆序对"
            assert reverse_pairs([2, 2, 1]) == 2
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(51)
            for _ in range(300):
                a = [rnd.randint(-10, 10) for _ in range(rnd.randint(0, 25))]
                exp, got = _brute(a), reverse_pairs(list(a))
                assert got == exp, f"nums={a}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：n = 6000',
          hidden: true,
          code: py`
            rnd = random.Random(9)
            a = [rnd.randint(0, 10**6) for _ in range(6000)]
            t0 = time.perf_counter()
            got = reverse_pairs(list(a))
            dt = time.perf_counter() - t0
            assert dt < 0.5, f"用时 {dt:.2f}s，疑似 O(n^2)"
            exp = _ref_fast(a)
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '完全逆序 n = 2000',
          hidden: true,
          code: py`
            n = 2000
            got = reverse_pairs(list(range(n, 0, -1)))
            assert got == n * (n - 1) // 2, f"期望 {n * (n - 1) // 2}，得到 {got}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-315',
          q: md`LC315 要求返回每个元素右侧比它小的元素个数，怎么改？`,
          a: md`对**下标**数组做归并（按值比较），合并时每当从左段取出元素 \`idx\`，就 \`ans[idx] += 已经从右段取出的元素个数\`——这些右段元素都比它小且在它右边。复杂度不变。也可以用离散化 + 树状数组从右往左扫。`,
        },
      ],
    },
    {
      id: 'c-sort-list',
      title: '排序链表',
      difficulty: 'medium',
      source: 'LeetCode 148',
      timeLimitMin: 25,
      description: md`
        给定链表头节点 \`head\`，将其按升序排列并返回排序后的头节点。

        节点类已给出（测试使用同一个定义；测试中会用辅助函数把 Python 列表构造成链表、再把结果转回列表比较）：

        \`\`\`python
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def sort_list(head: ListNode | None) -> ListNode | None: ...
        \`\`\`

        **要求**
        - $O(n\log n)$ 时间；**只修改 next 指针**，不要新建值节点、不要把值导出到数组再排序（不能使用 \`sorted\` / \`.sort\`）；
        - 隐藏测试包含 1500 个节点的链表：递归版的“合并两个链表”递归深度为 $O(n)$，超过 Python 默认的 1000 层上限会触发 RecursionError——合并请用循环；
        - 进阶：额外空间 $O(1)$（自底向上归并）。

        **示例**：\`4 → 2 → 1 → 3\` → \`1 → 2 → 3 → 4\`；空链表返回 \`None\`。
      `,
      starter: py`
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def sort_list(head):
            # TODO: 快慢指针找中点 → 断开 → 递归排序 → 循环合并（或自底向上）
            raise NotImplementedError
      `,
      solution: py`
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def split(head, k):
            """从 head 起切下 k 个节点，返回剩余部分的头"""
            for _ in range(k - 1):
                if head is None:
                    return None
                head = head.next
            if head is None:
                return None
            rest, head.next = head.next, None
            return rest

        def merge(a, b, tail):
            """合并有序链表 a、b，接在 tail 之后，返回新的尾节点"""
            while a and b:
                if a.val <= b.val:
                    tail.next, a = a, a.next
                else:
                    tail.next, b = b, b.next
                tail = tail.next
            tail.next = a or b
            while tail.next:
                tail = tail.next
            return tail

        def sort_list(head):
            n, p = 0, head
            while p:
                n, p = n + 1, p.next
            dummy = ListNode(0, head)
            step = 1
            while step < n:                     # 自底向上：段长 1, 2, 4, ...
                prev, cur = dummy, dummy.next
                while cur:
                    left = cur
                    right = split(left, step)
                    cur = split(right, step)
                    prev = merge(left, right, prev)
                step *= 2
            return dummy.next
      `,
      solutionAlt: [
        {
          label: '自顶向下版（快慢指针 + 递归拆分，栈深 O(log n)）',
          code: py`
            def sort_list(head):
                if head is None or head.next is None:
                    return head
                slow, fast = head, head.next          # 偶数长度时 slow 停在前半末尾
                while fast and fast.next:
                    slow, fast = slow.next, fast.next.next
                right, slow.next = slow.next, None    # 断开
                a, b = sort_list(head), sort_list(right)
                dummy = tail = ListNode()
                while a and b:                        # 循环合并，不要递归
                    if a.val <= b.val:
                        tail.next, a = a, a.next
                    else:
                        tail.next, b = b, b.next
                    tail = tail.next
                tail.next = a or b
                return dummy.next
          `,
        },
      ],
      explanation: md`
        **为什么用归并**：链表无法随机访问，快排划分、堆排序都不方便；归并只需顺序遍历，且合并只改指针，不需要辅助数组。

        **自顶向下**：快慢指针找中点（\`fast\` 从 \`head.next\` 出发，保证两节点链表能切成 1 + 1），断开后递归，循环合并。时间 $O(n\log n)$，栈 $O(\log n)$。

        **自底向上（参考解）**：段长 \`step\` 从 1 倍增；每轮用 \`split\` 切下两段、\`merge\` 合并后接到 \`prev\` 后面并返回新的尾。额外空间 $O(1)$。

        **常见错误**：找中点后忘记断开（\`slow.next = None\`）导致无限递归；\`fast = head\` 起步时两节点链表切不开；递归合并在长链表上爆栈。
      `,
      complexity: { time: 'O(n log n)', space: 'O(1)（自底向上）/ O(log n)（递归）' },
      hints: [
        md`链表上最合适的 $O(n\log n)$ 排序是归并。先写一个**循环版**的“合并两个有序链表”。`,
        md`自顶向下：\`slow, fast = head, head.next\`，循环结束后在 slow 之后断开，递归两半。`,
        md`想要 $O(1)$ 空间：段长 1, 2, 4… 自底向上，需要“切下 k 个节点”和“合并并返回尾节点”两个辅助函数。`,
      ],
      setup: LISTNODE_SETUP + '\n' + py`
        import linecache, random, time
      `,
      tests: [
        {
          name: '示例',
          code: py`
            head, _ = _build([4, 2, 1, 3])
            got = _to_list(sort_list(head))
            assert got == [1, 2, 3, 4], f"期望 [1, 2, 3, 4]，得到 {got}"
            head, _ = _build([-1, 5, 3, 4, 0])
            got = _to_list(sort_list(head))
            assert got == [-1, 0, 3, 4, 5], f"得到 {got}"
          `,
        },
        {
          name: '空链表 / 单节点 / 两节点',
          code: py`
            assert sort_list(None) is None
            for vals in ([1], [2, 1], [1, 2], [3, 3]):
                head, _ = _build(vals)
                got = _to_list(sort_list(head))
                assert got == sorted(vals), f"输入 {vals}：期望 {sorted(vals)}，得到 {got}"
          `,
        },
        {
          name: '不使用库排序',
          code: py`
            src = "".join(linecache.getlines("solution.py"))
            for banned in ("sorted(", ".sort("):
                assert banned not in src, f"请通过修改指针排序，不要使用 {banned}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(148)
            for _ in range(200):
                vals = [rnd.randint(-20, 20) for _ in range(rnd.randint(0, 30))]
                head, _ = _build(vals)
                got = _to_list(sort_list(head))
                assert got == sorted(vals), f"输入 {vals}：期望 {sorted(vals)}，得到 {got}"
          `,
        },
        {
          name: '1500 个节点：不爆栈、复用原节点',
          hidden: true,
          code: py`
            rnd = random.Random(5)
            vals = [rnd.randint(-10**6, 10**6) for _ in range(1500)]
            head, nodes = _build(vals)
            try:
                t0 = time.perf_counter()
                res = sort_list(head)
                dt = time.perf_counter() - t0
                seen, p = set(), res
                while p is not None and len(seen) <= len(nodes):
                    seen.add(id(p))
                    p = p.next
                assert seen == {id(x) for x in nodes}, "结果链表中的节点与原链表不一致：请只修改 next 指针，不要新建节点"
                got = _to_list(res)
                assert got == sorted(vals), "排序结果不正确"
                assert dt < 1.0, f"用时 {dt:.2f}s"
            finally:
                _free(nodes)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-quick-list',
          q: md`链表能用快排吗？为什么一般不用？`,
          a: md`能：取头节点为 pivot，遍历把节点分到“小于 / 等于 / 大于”三条链表，递归后拼接。但 pivot 只能取头部（随机选要先遍历），已排序链表会退化到 $O(n^2)$；且递归深度最坏 $O(n)$。归并在链表上最坏 $O(n\log n)$、且无需额外空间，是更好的选择。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-vs-quick',
      freq: 3,
      q: md`归并排序和快速排序各有什么优缺点？分别在什么场景使用？`,
      a: md`
        **结论**：归并稳定、最坏 $O(n\log n)$、适合链表与外部排序，但数组上需 $O(n)$ 辅助空间；快排原地、常数小、平均最快，但不稳定、最坏 $O(n^2)$。

        **原因**：归并的拆分固定对半，复杂度与输入无关，合并只需顺序读写；快排的性能取决于 pivot 质量，但划分原地进行、访存局部性好。

        **延伸**：工程上数组排序多用 introsort（快排为主）或 Timsort（归并 + 插入，稳定，利用天然有序段）；链表排序、多关键字稳定排序、超出内存的数据排序用归并。
      `,
    },
    {
      id: 'qa-inversions',
      freq: 2,
      q: md`为什么逆序对可以在归并的合并阶段统计？`,
      a: md`
        **结论**：两段各自有序时，跨段逆序对可以按右段元素“批量”统计，每次 $O(1)$。

        **原因**：合并时若 \`L[i] > R[j]\`，由于 L 有序，\`L[i:]\` 都大于 \`R[j]\`，一次加上 \`len(L) - i\`。左段内部、右段内部的逆序对由递归统计；排序本身不会改变跨段数对的“左右”关系，所以三部分之和就是答案。

        **延伸**：变体有 LC315（每个元素右侧更小的个数，对下标归并）、LC493（翻转对 \`nums[i] > 2·nums[j]\`，需在合并前用双指针单独计数，因为判定条件与排序依据不同）；也可以用离散化 + 树状数组求解。
      `,
    },
    {
      id: 'qa-external',
      freq: 2,
      q: md`内存 4 GB，怎么对 100 GB 的整数文件排序？`,
      a: md`
        **结论**：外部归并排序——分块内排生成有序段，再 K 路归并。

        **原因**：每次读入约 4 GB 排序后写回，得到 25 个有序段；然后为每个段分配一个读缓冲，用大小为 25 的小顶堆做 K 路归并，按顺序写出。磁盘访问全部是顺序读写，总 I/O 约两遍数据，CPU 代价 $O(N\log K)$。

        **延伸**：缓冲区放不下时做多轮归并；可以用“置换选择”生成平均长度为内存两倍的初始段，减少段数；若值域有限（如 32 位整数）也可以按高位分桶后各自排序。
      `,
    },
    {
      id: 'qa-list',
      freq: 2,
      q: md`为什么链表排序首选归并排序？怎么做到 O(1) 额外空间？`,
      a: md`
        **结论**：链表上归并的两个缺点（随机访问、辅助数组）都不存在；自底向上迭代可做到 $O(1)$ 额外空间。

        **原因**：归并只需顺序遍历找中点与合并，合并只改指针；快排的 pivot 只能取头部、堆排序依赖下标跳转。自顶向下递归需要 $O(\log n)$ 栈；自底向上按段长 1, 2, 4… 切段合并，只用常数个指针。

        **延伸**：合并必须用循环 + 哑节点；递归合并的深度是链表长度，Python 中几千个节点就会 RecursionError。
      `,
    },
  ],

  cheatsheet: md`
    **合并 + 逆序对**
    \`\`\`python
    while i < len(L) and j < len(R):
        if L[i] <= R[j]: out.append(L[i]); i += 1      # 相等取左：稳定、不计数
        else:            out.append(R[j]); j += 1; inv += len(L) - i
    out += L[i:] + R[j:]
    \`\`\`
    **链表**：\`slow, fast = head, head.next\` 找中点 → \`slow.next = None\` 断开 → 递归 → 循环合并（dummy）

    - $T(n) = 2T(n/2) + O(n) = O(n\log n)$，与输入无关；数组 $O(n)$ 辅助空间，链表 $O(1)$
    - 稳定的关键：相等时取左边
    - 自底向上：段长 1, 2, 4…，无递归
    - 外部排序：分块内排 → K 路归并（小顶堆，$O(N\log K)$）
    - 变体：LC315 对下标归并；LC493 合并前单独双指针计数
  `,
})
