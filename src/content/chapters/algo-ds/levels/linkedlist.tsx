/**
 * 链表基本功（algo-ds-linkedlist）
 * LC206 / 92 / 25 / 21 / 142 / 19：哑节点、三指针反转、头插法、K 个一组、快慢指针（环入口证明）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { LISTNODE_SETUP } from '../shared'
import { ReverseListViz } from '../viz/ReverseListViz'
import { KGroupViz } from '../viz/KGroupViz'
import { CycleViz } from '../viz/CycleViz'

const LISTNODE_DEF = py`
  class ListNode:
      def __init__(self, val=0, next=None):
          self.val = val
          self.next = next
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'reverse',
      title: '反转链表：先存后改，每轮只动一条边',
      body: md`
        链表题几乎不考“算法”，考的是**在不丢节点的前提下改指针**。单链表只能从前往后走，一旦把 \`cur.next\` 改掉，
        原来的后继就再也找不回来了——所以一切链表操作的第一原则是：**改一条边之前，先把它原来指向的节点存起来**。

        反转链表（LC206）的迭代写法用三个指针：

        | 指针 | 含义（循环不变量） |
        |---|---|
        | \`prev\` | 已反转部分的头（初始为 None） |
        | \`cur\` | 待处理部分的头 |
        | \`nxt\` | 临时保存 \`cur.next\`，防止断链 |

        每轮做四件事：存后继 → 改向 → prev 前进 → cur 前进。$n$ 轮之后 \`cur\` 为 None，\`prev\` 就是新头。
        时间 $O(n)$，额外空间 $O(1)$。

        递归写法更短，但思路完全不同：先把 \`head.next\` 之后的部分反转好，再让后继**指回自己**，
        最后断开旧边。它的空间是递归栈 $O(n)$——Python 默认递归上限 1000 层，长链表会直接 \`RecursionError\`。
      `,
      viz: ReverseListViz,
      after: md`
        > [!WARN] 两个最常见的错误
        > 1. 先写 \`cur.next = prev\` 再写 \`nxt = cur.next\`：nxt 读到的是已经改过的值（prev），链表后半段丢失；
        > 2. 递归版漏写 \`head.next = None\`：原头节点和第二个节点互指成环，打印结果时死循环。

        > [!INTERVIEW] 面试官想听到的
        > “迭代版 O(1) 空间，是首选；递归版代码短但栈深 O(n)，生产代码和 Python 长链表都不推荐。”
      `,
      checkpoint: {
        id: 'cp-reverse',
        type: 'single',
        prompt: md`迭代反转中，把循环体写成 \`cur.next = prev; nxt = cur.next; prev = cur; cur = nxt\`（先改向再保存）。对 \`1→2→3\` 运行，结果是？`,
        options: ['正确得到 `3→2→1`', '返回 `1`（只有一个节点），2 和 3 丢失', '死循环', '抛出 AttributeError'],
        answer: 1,
        explanation: md`第一轮：\`1.next = None\`，然后 \`nxt = 1.next = None\`，\`prev = 1\`，\`cur = None\`，循环结束，返回节点 1。2、3 再也找不回来了。这正是“先存后改”的原因。不会死循环也不会报错——**静默出错**才最危险。`,
      },
    },
    {
      id: 'kgroup',
      title: '哑节点 + 区间反转：K 个一组翻转',
      body: md`
        只要**头节点可能变化**（删除头、反转前缀、K 个一组…），就在前面挂一个**哑节点**
        \`dummy = ListNode(0, head)\`，最后返回 \`dummy.next\`。这样“第一个节点的前驱”永远存在，不需要任何特判。

        **K 个一组翻转（LC25）** 每组只需 4 个指针：

        1. \`pre\`：本组之前的节点（第一组是 dummy）；
        2. \`end\`：从 pre 向前走 $k$ 步得到本组尾；走到 None 说明不足 $k$ 个，**直接返回**；
        3. \`start = pre.next\`（本组头）、\`nxt = end.next\`（下一组头），断开 \`end.next = None\`；
        4. 反转本组后，\`pre.next = end\`、\`start.next = nxt\`，再令 \`pre = start\`。

        **区间反转（LC92）** 还有一种不需要断开的写法——**头插法**：\`pre\` 固定在区间之前，
        把区间里的后续节点一个个摘下来，插到 \`pre\` 后面。区间 \`[left, right]\` 只需 \`right - left\` 次摘插。
      `,
      viz: KGroupViz,
      after: md`
        > [!TIP] 读图的关键
        > 第 ② 步之后，原来的组头 \`start\` 变成了组尾，所以第 ④ 步 \`pre = start\` 恰好让 pre 停在下一组之前——循环不变量“pre 是待处理部分的前驱”始终成立。

        复杂度：每个节点被“找尾”扫描一次、反转一次，总 $O(n)$；只用常数个指针，额外空间 $O(1)$。
        如果写成“每组递归一次”，栈深是 $n/k$，在 $k=1$ 时退化成 $O(n)$。
      `,
      checkpoint: {
        id: 'cp-kgroup',
        type: 'single',
        prompt: md`对 \`1→2→3→4→5→6→7→8\` 做 K 个一组翻转（$k = 3$），结果是？`,
        options: ['`3→2→1→6→5→4→8→7`', '`3→2→1→6→5→4→7→8`', '`8→7→6→5→4→3→2→1`', '`3→2→1→4→5→6→7→8`'],
        answer: 1,
        explanation: md`前两组 \`1,2,3\` 与 \`4,5,6\` 各自翻转；剩下的 \`7,8\` 不足 3 个，**保持原样**（LC25 的规定）。A 错在把不足 k 的尾组也翻了，C 是整体反转，D 只翻了第一组。`,
      },
    },
    {
      id: 'fastslow',
      title: '快慢指针：中点、倒数第 k 个、环的入口',
      body: md`
        两个指针以不同速度或不同起点同时走，可以在一次遍历里得到“位置关系”：

        | 问题 | 做法 |
        |---|---|
        | 中点（LC876） | slow 走 1、fast 走 2；fast 到尾时 slow 在中点 |
        | 删除倒数第 $n$ 个（LC19） | fast 先走 $n$ 步，再同速走；从 **dummy** 出发，slow 停在待删节点的前驱 |
        | 判环（LC141） | 有环则 fast 必然追上 slow（进环后每步追近 1 格） |
        | 环入口（LC142） | 相遇后一个回 head，两者同速走，再次相遇处即入口 |

        **入口公式的推导**：设尾长 $a$（head 到入口的节点数）、环长 $c$、入口到相遇点距离 $b$。相遇时
        $$
        s_{\text{slow}} = a + b,\qquad s_{\text{fast}} = 2(a+b) = a + b + kc \;\Rightarrow\; a + b = kc \;\Rightarrow\; a = (k-1)\,c + (c-b)
        $$
        $c - b$ 正是相遇点**继续向前**走回入口的距离。所以一个指针从 head、一个从相遇点同速出发，
        走 $a$ 步后前者到入口，后者绕了 $k-1$ 整圈再走 $c-b$ 步，也到入口。
      `,
      viz: CycleViz,
      after: md`
        > [!NOTE] 为什么 slow 在第一圈内就会被追上
        > slow 进环时，fast 已在环内，两者沿环方向的距离 $< c$；之后每步缩小 1，最多 $c-1$ 步相遇。
        > 所以 slow 总共不超过 $a + c - 1$ 步，判环是 $O(n)$ 的。也因此相遇步数 $s$ 是“$\ge a$ 的最小的 $c$ 的倍数”（$a = 0$ 时为 $c$）。

        把尾长 $a$ 调成 0（head 就是入口）或把 $a$ 调得比 $c$ 大，观察 $k$ 的取值如何变化。
      `,
      checkpoint: {
        id: 'cp-meet',
        type: 'numeric',
        prompt: md`链表尾长 $a = 4$、环长 $c = 5$。slow 每次 1 步、fast 每次 2 步，同时从 head 出发，**第几步**首次相遇？`,
        answer: 5,
        tolerance: 0,
        explanation: md`相遇的充要条件是 fast 多走的 $s$ 步是环长的整数倍且两者都已进环：$s \ge a$ 且 $c \mid s$，最小为 $s = 5$。验证：第 5 步 slow 在环内偏移 $5-4 = 1$；fast 走了 10 步，环内偏移 $(10-4) \bmod 5 = 1$，重合。`,
      },
    },
    {
      id: 'merge',
      title: '合并、拆分与 Python 链表的坑',
      body: md`
        **合并两个有序链表（LC21）** 是归并排序链表、合并 K 个链表的基础：dummy + 尾指针 \`tail\`，
        每次把较小的节点接到 \`tail\` 后面；循环结束后把剩下的整段直接接上（\`tail.next = l1 or l2\`），不用逐个复制。

        链表题的“工具箱”可以归为四件：

        | 工具 | 何时用 | 代表题 |
        |---|---|---|
        | 哑节点 | 头可能被删除 / 改变 | 19、21、82、92、25 |
        | 三指针反转 | 改变链接方向 | 206、92、25、234（回文链表的后半段反转） |
        | 快慢指针 | 中点、环、倒数第 k | 876、141、142、19、148（找中点拆分） |
        | 拆分 + 合并 | 排序、重排 | 21、23、148、143（重排链表 = 中点 + 反转 + 交替合并） |

        LC143 重排链表就是“三件套”的组合：快慢指针找中点 → 反转后半段 → 两段交替合并，是很好的综合练习。

        **Python 特有的坑**：
        - 递归深度默认 1000：递归反转 / 递归合并在几千个节点上会 \`RecursionError\`，面试代码优先写迭代；
        - 判断“是不是同一个节点”用 \`is\`，不要用 \`==\`（节点可能重载了 \`__eq__\`，值相同的不同节点会被误判）；
        - 不要把值导出到数组处理再建新链表——多数题明确要求“只改指针、不改值”，面试官也会追问空间。
      `,
      checkpoint: {
        id: 'cp-merge',
        type: 'judge',
        prompt: md`合并两个有序链表时，循环结束后必须用 while 把剩余链表的节点**逐个**接到 tail 后面。`,
        answer: false,
        explanation: md`不需要。剩余部分本身就是一条有序链表，\`tail.next = l1 or l2\` 一步接上整段，$O(1)$。逐个接只会多走一遍。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                          */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-dummy',
      type: 'single',
      difficulty: 1,
      prompt: md`下列哪种情况**最需要**哑节点（dummy）？`,
      options: ['求链表长度', '删除链表中所有值等于 val 的节点（头节点也可能被删）', '判断链表是否有环', '求链表中点'],
      answer: 1,
      explanation: md`删除操作要找“前驱”，而头节点没有前驱；挂上 dummy 后所有节点都有前驱，最后返回 \`dummy.next\`。求长度、判环、找中点都不改变头，不需要 dummy。`,
    },
    {
      id: 'q-rec-space',
      type: 'judge',
      difficulty: 1,
      prompt: md`递归反转链表的额外空间复杂度是 $O(1)$，因为它没有新建任何节点。`,
      answer: false,
      explanation: md`递归调用本身占用栈帧，深度等于链表长度，空间 $O(n)$。Python 默认递归上限 1000，长链表会 \`RecursionError\`。只有迭代版才是 $O(1)$。`,
    },
    {
      id: 'q-fastslow',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于快慢指针（slow 每次 1 步、fast 每次 2 步，循环条件 \`while fast and fast.next\`），哪些说法正确？（多选）`,
      options: [
        '`slow = head, fast = head.next` 起步时，偶数长度链表 slow 停在**前半段的最后一个**节点',
        '`slow = fast = head` 起步时，偶数长度链表 slow 停在**后半段的第一个**节点',
        '有环时 fast 可能恰好“跳过” slow，导致永远不相遇',
        '若把 fast 改为每次走 3 步，有环时仍然一定会相遇',
      ],
      answer: [0, 1],
      explanation: md`以 \`1→2→3→4\` 验证：A 中 slow 停在 2，B 中停在 3（LC876 要求的“第二个中点”）。C 错：进环后两者距离每步缩小**恰好 1**，不可能跳过。D 错：距离每步缩小 2，若环长为偶数且初始距离为奇数，会一直错开、永不相遇。`,
    },
    {
      id: 'q-meet-steps',
      type: 'numeric',
      difficulty: 2,
      prompt: md`链表尾长 $a = 7$（head 到入口有 7 个节点）、环长 $c = 3$。slow 每次 1 步、fast 每次 2 步，从 head 同时出发，第几步首次相遇？`,
      answer: 9,
      tolerance: 0,
      explanation: md`相遇条件：两者都在环内（$s \ge a = 7$）且 fast 多走的 $s$ 步是环长整数倍（$3 \mid s$），最小 $s = 9$。验证：第 9 步 slow 环内偏移 $9 - 7 = 2$，fast 走 18 步，偏移 $(18-7) \bmod 3 = 2$。此时 $k = 3$，$b = 2$，$a = (k-1)c + (c-b) = 6 + 1 = 7$ ✓。`,
    },
    {
      id: 'q-bug-nth',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的“删除倒数第 n 个节点”在某些合法输入上会抛异常。点出**根源**所在的行：`,
      code: py`
        def remove_nth_from_end(head, n):
            fast = slow = head
            for _ in range(n):
                fast = fast.next
            while fast.next:
                fast = fast.next
                slow = slow.next
            slow.next = slow.next.next
            return head
      `,
      bugLines: [2],
      explanation: md`当 $n$ 等于链表长度（要删的是头节点）时，fast 先走 $n$ 步变成 None，第 5 行 \`fast.next\` 抛 AttributeError。根源是第 2 行没有用哑节点：改为 \`dummy = ListNode(0, head); fast = slow = dummy\`，最后返回 \`dummy.next\`。这样 slow 总能停在待删节点的**前驱**上，删头也不例外。`,
    },
    {
      id: 'q-order-kgroup',
      type: 'order',
      difficulty: 2,
      prompt: md`K 个一组翻转中，处理**一组**的步骤按正确顺序排列：`,
      items: [
        '从 pre 向前走 k 步得到 end（走到 None 则直接返回）',
        '保存 start = pre.next 与 nxt = end.next',
        '断开 end.next = None 并反转 [start, end]',
        'pre.next = end，start.next = nxt（接回两端）',
        'pre = start，进入下一组',
      ],
      explanation: md`必须先确认凑满 k 个（否则按题意不翻转），再**先保存** nxt 才能断开；反转后 end 是新头、start 是新尾，接回两端后 start 就是下一组的前驱。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把技巧与它最典型的应用配对：`,
      pairs: [
        ['哑节点', '删除倒数第 N 个节点（可能删头）'],
        ['快慢指针（步长 1 / 2）', '判环并求环的入口'],
        ['头插法', '反转链表的区间 [left, right]'],
        ['快慢指针找中点 + 归并', '排序链表 O(n log n)'],
      ],
      explanation: md`哑节点解决“头没有前驱”；步长差 1 保证不会跳过；头插法让 pre 固定、逐个摘插，无需断开；排序链表用中点拆分 + 合并（链表上归并最合适）。`,
    },
    {
      id: 'q-parsons-merge',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出“合并两个有序链表”（迭代 + 哑节点，有 2 行干扰项）：`,
      lines: [
        'def merge_two_lists(l1, l2):',
        '    dummy = tail = ListNode()',
        '    while l1 and l2:',
        '        if l1.val <= l2.val:',
        '            tail.next, l1 = l1, l1.next',
        '        else:',
        '            tail.next, l2 = l2, l2.next',
        '        tail = tail.next',
        '    tail.next = l1 or l2',
        '    return dummy.next',
      ],
      distractors: ['    tail.next = l1 and l2', '    return dummy'],
      explanation: md`\`l1 or l2\` 取非空的那段整体接上；\`l1 and l2\` 在一方为空时得到 None，会丢掉剩余节点。返回 \`dummy\` 会多出一个值为 0 的假节点，应返回 \`dummy.next\`。用 \`<=\` 保证稳定（相等时先取 l1）。`,
    },
    {
      id: 'q-fill-headinsert',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        头插法反转区间：\`pre\` 是区间前驱，\`cur = pre.next\` 始终是区间里**原来的第一个节点**。每轮把 cur 的后继摘下插到 pre 之后：
        \`\`\`python
        nxt = cur.next
        cur.next = ____(1)____
        nxt.next = ____(2)____
        pre.next = nxt
        \`\`\`
      `,
      blanks: [
        { label: '(1)', answers: ['nxt.next'] },
        { label: '(2)', answers: ['pre.next'] },
      ],
      explanation: md`(1) 先让 cur 跳过 nxt（摘下 nxt），(2) 再让 nxt 指向当前区间的头 \`pre.next\`，最后 \`pre.next = nxt\` 把它插到最前。cur 本身不动，它会一路被“推”到区间末尾。顺序不能换：若先执行 \`pre.next = nxt\`，(2) 就会变成 nxt 指向自己。`,
    },
    {
      id: 'q-pick-92',
      type: 'pick',
      difficulty: 2,
      prompt: md`对 \`1→2→3→4→5→6→7\` 执行 \`reverse_between(left=2, right=5)\`，得到 \`1→5→4→3→2→6→7\`。哪些节点的 **next 指针**被修改了？（点出全部）`,
      items: ['1', '2', '3', '4', '5', '6', '7'],
      answer: [0, 1, 2, 3, 4],
      explanation: md`新旧 next 对比：1：2→5 改；5：6→4 改；4：5→3 改；3：4→2 改；2：3→6 改；6→7、7→None 不变。区间前驱（1）和区间内所有节点都会改，区间之后的节点不受影响。`,
    },
    {
      id: 'q-recursion-limit',
      type: 'single',
      difficulty: 2,
      prompt: md`在 Python 里用递归反转一条 $10^4$ 个节点的链表，最可能发生什么？`,
      options: ['正常返回，只是慢一些', '抛出 RecursionError', '返回原链表（未反转）', '产生环导致死循环'],
      answer: 1,
      explanation: md`CPython 默认递归上限 1000（\`sys.getrecursionlimit()\`），一万层递归直接 RecursionError。调大上限可能撑爆 C 栈导致进程崩溃，正确做法是改成迭代。`,
    },
    {
      id: 'q-entry-proof',
      type: 'single',
      difficulty: 3,
      prompt: md`在环入口证明里，相遇时 slow 走了 $s = a + b$ 步。下列哪一条是“$s$ 是环长 $c$ 的整数倍”的**正确理由**？`,
      options: [
        'fast 走了 $2s$ 步，比 slow 多走的 $s$ 步全部在环内绕圈，且两者此刻在同一节点',
        'slow 一定恰好绕环一整圈后才被追上',
        '$a$ 一定小于 $c$，所以 $a + b < 2c$',
        '相遇点一定是入口，所以 $b = 0$',
      ],
      answer: 0,
      explanation: md`两者在同一节点，说明 fast 比 slow 多走的路程是整圈数：$2s - s = kc$。B 错：slow 在第一圈内（不到一整圈）就会被追上；C 错：$a$ 可以远大于 $c$；D 错：相遇点一般不是入口（否则就不需要第二阶段）。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-reverse-between',
      title: '头插法反转链表区间（LC92）',
      intro: md`
        实现 \`reverse_between(head, left, right)\`：把第 \`left\` 到第 \`right\` 个节点（1 起始）反转，返回新头。
        用**哑节点 + 头插法**：pre 固定在区间之前，把区间内的节点一个个摘下插到 pre 后面，一次遍历完成。
        \`ListNode\` 已在环境中定义（\`ListNode(val=0, next=None)\`）。
      `,
      template: py`
        def reverse_between(head, left, right):
            dummy = {{1}}
            pre = dummy
            for _ in range(left - 1):
                {{2}}
            cur = pre.next
            for _ in range(right - left):
                nxt = cur.next
                {{3}}
                {{4}}
                {{5}}
            return dummy.next
      `,
      steps: [
        {
          blank: 1,
          instruction: md`创建哑节点并接在 head 前面（left = 1 时头节点会变）。`,
          answer: 'ListNode(0, head)',
          accept: ['ListNode(next=head)', 'ListNode(-1, head)'],
          explain: md`有了 dummy，“区间前驱”在 left = 1 时就是 dummy 本身，不用特判；最后返回 \`dummy.next\`。`,
        },
        {
          blank: 2,
          instruction: md`让 pre 走到区间的前驱（第 left − 1 个节点）。`,
          answer: 'pre = pre.next',
          explain: md`从 dummy 出发走 left − 1 步，正好停在第 left − 1 个节点上。`,
        },
        {
          blank: 3,
          instruction: md`摘下 nxt：让 cur 跳过它。`,
          answer: 'cur.next = nxt.next',
          explain: md`cur 是区间原来的第一个节点，它始终不动，每轮被“推”后一位，最终成为区间的尾。`,
        },
        {
          blank: 4,
          instruction: md`让 nxt 指向区间当前的头（pre 的后继）。`,
          answer: 'nxt.next = pre.next',
          explain: md`注意这里不能写 \`nxt.next = cur\`：第二轮起区间头已经不是 cur 了。`,
        },
        {
          blank: 5,
          instruction: md`把 nxt 接到 pre 后面，成为区间的新头。`,
          answer: 'pre.next = nxt',
          explain: md`三行的顺序是“摘 → 指 → 插”。right − left 轮之后区间就完全反转了。`,
        },
      ],
      setup: LISTNODE_SETUP + '\n' + py`
        import random

        def _ref_rb(vals, l, r):
            return vals[: l - 1] + vals[l - 1 : r][::-1] + vals[r:]

        def _check_rb(vals, l, r):
            head, nodes = _build(vals)
            try:
                got = _to_list(reverse_between(head, l, r))
            finally:
                _free(nodes)
            exp = _ref_rb(vals, l, r)
            assert got == exp, f"vals={vals}, left={l}, right={r}：期望 {exp}，得到 {got}"
      `,
      tests: [
        { name: '示例', code: py`_check_rb([1, 2, 3, 4, 5], 2, 4)` },
        { name: 'left = 1（换头）', code: py`_check_rb([3, 5], 1, 2)` },
        { name: '单节点 / left == right', code: py`_check_rb([7], 1, 1); _check_rb([1, 2, 3], 2, 2)` },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(92)
            for _ in range(200):
                n = rnd.randint(1, 12)
                l = rnd.randint(1, n)
                r = rnd.randint(l, n)
                _check_rb([rnd.randint(0, 9) for _ in range(n)], l, r)
          `,
        },
      ],
      outro: md`
        头插法不需要断开再接回，适合“只反转一段”；K 个一组（LC25）可以对每组重复头插，也可以“断开 → 复用 206 → 接回”。
        两种写法都是 $O(n)$ 时间、$O(1)$ 空间，面试时选自己最不容易写错的那种。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-kgroup',
      title: 'K 个一组翻转链表',
      difficulty: 'hard',
      source: 'LeetCode 25',
      timeLimitMin: 25,
      description: md`
        给定链表头 \`head\` 和正整数 \`k\`，每 \`k\` 个节点一组进行翻转，返回修改后的链表。
        节点总数不是 \`k\` 的整数倍时，最后剩余的节点**保持原有顺序**。

        \`\`\`python
        class ListNode:
            def __init__(self, val=0, next=None):
                self.val = val
                self.next = next

        def reverse_k_group(head: ListNode | None, k: int) -> ListNode | None: ...
        \`\`\`

        **要求**
        - 只能修改 \`next\` 指针，不能改节点的值、不能新建值节点（测试会核对节点身份）；
        - **额外空间 $O(1)$**：隐藏测试用 \`tracemalloc\` 检查，把节点存进数组再重连的写法过不了；
        - 隐藏测试包含 1500 个节点、$k = 1$ 的链表：按组递归的写法栈深为 $n/k$，会超过 Python 递归上限，请用迭代。

        **示例**
        - \`1→2→3→4→5\`，\`k = 2\` → \`2→1→4→3→5\`
        - \`1→2→3→4→5\`，\`k = 3\` → \`3→2→1→4→5\`

        **约束**：$0 \le n \le 1500$，$1 \le k$。
      `,
      starter: LISTNODE_DEF + '\n' + py`
        def reverse_k_group(head, k):
            # TODO: dummy + pre；每组先找 end（不足 k 个直接返回），再断开、反转、接回
            raise NotImplementedError
      `,
      solution: LISTNODE_DEF + '\n' + py`
        def reverse_k_group(head, k):
            dummy = ListNode(0, head)
            pre = dummy
            while True:
                end = pre
                for _ in range(k):              # 找本组尾；不足 k 个保持原样
                    end = end.next
                    if end is None:
                        return dummy.next
                start, nxt = pre.next, end.next
                # 组内反转 [start, end]：prev 初值取 nxt，反转后 start 自动接上下一组
                prev, cur = nxt, start
                while cur is not nxt:
                    tmp = cur.next
                    cur.next = prev
                    prev = cur
                    cur = tmp
                pre.next = end                  # end 成为本组新头
                pre = start                     # start 成为本组新尾 = 下一组的前驱
      `,
      explanation: md`
        **思路**：哑节点 + 每组 4 个指针。\`end\` 从 \`pre\` 向前走 $k$ 步，走到 None 就说明不足一组，直接返回。

        **一个省事的技巧**：反转本组时把 \`prev\` 的初值设为 \`nxt\`（而不是 None），循环条件写 \`cur is not nxt\`。
        这样反转结束时 \`start.next\` 已经等于 \`nxt\`，不需要先断开再接回，只剩 \`pre.next = end\` 和 \`pre = start\` 两步。

        **复杂度**：每个节点被找尾扫描一次、反转一次，$O(n)$ 时间；常数个指针，$O(1)$ 空间。

        **常见错误**
        - 反转前没保存 \`nxt\`，或反转后忘了把 \`start\` 接回，导致后半段丢失；
        - 最后不足 $k$ 个也翻转了（题目要求保持原样）；
        - \`pre = end\`（应为 \`pre = start\`：反转后 start 才是组尾）。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`先写一个哑节点 \`dummy = ListNode(0, head)\`，\`pre = dummy\`。每组从 pre 出发走 k 步找到 end。`,
        md`保存 \`start = pre.next\`、\`nxt = end.next\`。反转 [start, end] 后：\`pre.next = end\`，\`start.next = nxt\`，\`pre = start\`。`,
        md`反转时令 \`prev = nxt, cur = start\`，循环到 \`cur is nxt\` 为止，就能省掉断开和接回。`,
      ],
      setup: LISTNODE_SETUP + '\n' + py`
        import random, tracemalloc

        def _ref_kgroup(seq, k):
            m = len(seq) - len(seq) % k
            out = []
            for i in range(0, m, k):
                out += list(seq[i : i + k])[::-1]
            return out + list(seq[m:])

        def _free_chain(head, limit=5000):
            c = 0
            while head is not None and c < limit:
                head.next, head = None, head.next
                c += 1

        def _check_kgroup(vals, k):
            head, nodes = _build(vals)
            res = None
            try:
                res = reverse_k_group(head, k)
                got = _to_list(res)
                exp = _ref_kgroup(vals, k)
                assert got == exp, f"vals={vals}, k={k}：期望 {exp}，得到 {got}"
                order, p = [], res
                while p is not None and len(order) <= len(nodes):
                    order.append(id(p))
                    p = p.next
                assert order == [id(x) for x in _ref_kgroup(nodes, k)], "节点身份不一致：请只修改 next 指针，不要改值或新建节点"
            finally:
                _free(nodes)
                _free_chain(res)
      `,
      tests: [
        { name: '示例 k = 2', code: py`_check_kgroup([1, 2, 3, 4, 5], 2)` },
        { name: '示例 k = 3', code: py`_check_kgroup([1, 2, 3, 4, 5], 3)` },
        {
          name: '边界：空、k = 1、k = n、k > n',
          code: py`
            assert reverse_k_group(None, 2) is None
            _check_kgroup([1, 2, 3], 1)
            _check_kgroup([1, 2, 3], 3)
            _check_kgroup([1, 2], 3)
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(25)
            for _ in range(300):
                n = rnd.randint(0, 20)
                _check_kgroup([rnd.randint(0, 9) for _ in range(n)], rnd.randint(1, 6))
          `,
        },
        {
          name: '1500 个节点，k = 1 与 k = 7',
          hidden: true,
          code: py`
            _check_kgroup(list(range(1500)), 1)
            _check_kgroup(list(range(1500)), 7)
          `,
        },
        {
          name: '额外空间 O(1)',
          hidden: true,
          code: py`
            head, nodes = _build(list(range(1500)))
            res = None
            try:
                tracemalloc.start()
                res = reverse_k_group(head, 4)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                assert peak < 6 * 1024, f"峰值额外内存 {peak / 1024:.1f} KiB，应为 O(1)（不要把节点存进数组）"
                assert _to_list(res) == _ref_kgroup(list(range(1500)), 4), "结果不正确"
            finally:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                _free(nodes)
                _free_chain(res)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-tail',
          q: md`如果要求“不足 k 个的尾组也要翻转”，代码怎么改？如果要求从链表**尾部**开始分组呢？`,
          a: md`前者：找 end 时走到 None 不再返回，而是把本组截到最后一个节点照常翻转。后者：先遍历求长度 $n$，第一组只取 $n \bmod k$ 个**不翻转**（或按题意处理），之后每组 k 个正常翻转——等价于把头部的余数组跳过。`,
        },
        {
          id: 'f-recursive',
          q: md`递归写法怎么写？它的空间复杂度是多少？`,
          a: md`先检查是否凑满 k 个，翻转前 k 个后让原组头 \`start.next = reverse_k_group(nxt, k)\`，返回 end。空间是递归栈 $O(n/k)$，$k$ 很小时接近 $O(n)$，不满足题目的进阶要求。`,
        },
      ],
    },
    {
      id: 'c-cycle',
      title: '环形链表 II：找环的入口',
      difficulty: 'medium',
      source: 'LeetCode 142',
      timeLimitMin: 15,
      description: md`
        给定链表头 \`head\`，返回链表开始入环的第一个节点；无环返回 \`None\`。**不允许修改链表**。

        \`\`\`python
        def detect_cycle(head: ListNode | None) -> ListNode | None: ...
        \`\`\`

        **要求**：$O(n)$ 时间、**$O(1)$ 额外空间**（隐藏测试用 \`tracemalloc\` 检查：用 \`set\` 记录访问过的节点能通过正确性测试，但过不了空间测试）。
        测试比较的是**节点身份**（\`is\`）。

        **示例**（\`pos\` 表示尾节点连到的下标，仅用于描述输入）
        - \`[3, 2, 0, -4]\`，\`pos = 1\` → 返回值为 2 的那个节点
        - \`[1, 2]\`，\`pos = 0\` → 返回头节点
        - \`[1]\`，\`pos = -1\` → \`None\`
      `,
      starter: LISTNODE_DEF + '\n' + py`
        def detect_cycle(head):
            # TODO: 阶段一 快慢指针找相遇点；阶段二 一个回 head，同速走到再次相遇
            raise NotImplementedError
      `,
      solution: LISTNODE_DEF + '\n' + py`
        def detect_cycle(head):
            slow = fast = head
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                if slow is fast:                # 相遇：a = (k-1)c + (c-b)
                    p = head
                    while p is not slow:        # 同速走 a 步，在入口相遇
                        p = p.next
                        slow = slow.next
                    return p
            return None
      `,
      explanation: md`
        **阶段一**：slow 走 1、fast 走 2。fast 走到 None 说明无环；相遇说明有环。

        **阶段二**：设尾长 $a$、环长 $c$、入口到相遇点 $b$。相遇时 $2(a+b) = a + b + kc$，得 $a = (k-1)c + (c-b)$。
        让 p 从 head、slow 从相遇点同速前进，走 $a$ 步后 p 到入口，slow 绕 $k-1$ 圈再走 $c-b$ 步也到入口。

        **复杂度**：阶段一不超过 $a + c$ 步，阶段二 $a$ 步，总 $O(n)$；空间 $O(1)$。

        **常见错误**
        - 判断相遇写成 \`slow == fast\` 且节点重载了 \`__eq__\`（本题用 \`is\` 最稳妥）；
        - 循环条件只写 \`while fast.next\`，空链表时 AttributeError；
        - 修改节点（如把 val 改成标记值）来记录访问——题目禁止修改链表。
      `,
      complexity: { time: 'O(n)', space: 'O(1)' },
      hints: [
        md`先用快慢指针判断有没有环：\`while fast and fast.next\`，每轮 slow 走 1、fast 走 2，\`slow is fast\` 时相遇。`,
        md`推导：相遇时 slow 走了 $a + b$，fast 走了 $2(a+b)$，多出来的是整圈数 $kc$。`,
        md`相遇后令 \`p = head\`，p 与 slow 每次各走 1 步，\`p is slow\` 时就是入口。`,
      ],
      setup: LISTNODE_SETUP + '\n' + py`
        import random, tracemalloc

        def _check_cycle(vals, pos):
            head, nodes = _build(vals, pos)
            try:
                links = [x.next for x in nodes]
                got = detect_cycle(head)
                exp = nodes[pos] if pos >= 0 else None
                assert all(x.next is l for x, l in zip(nodes, links)), "链表结构被修改了（题目不允许修改链表）"
                if exp is None:
                    assert got is None, f"vals={vals}, pos={pos}：无环应返回 None，得到 {got}"
                else:
                    gi = next((i for i, x in enumerate(nodes) if x is got), None)
                    assert got is exp, f"vals={vals}, pos={pos}：期望下标 {pos} 的节点，得到 {'下标 ' + str(gi) if gi is not None else got}"
            finally:
                _free(nodes)
      `,
      tests: [
        { name: '示例 1', code: py`_check_cycle([3, 2, 0, -4], 1)` },
        { name: '示例 2', code: py`_check_cycle([1, 2], 0)` },
        { name: '无环 / 空链表', code: py`_check_cycle([1], -1); _check_cycle([1, 2, 3], -1); assert detect_cycle(None) is None` },
        { name: '自环', code: py`_check_cycle([5], 0)` },
        {
          name: '随机对拍（值可重复）',
          hidden: true,
          code: py`
            rnd = random.Random(142)
            for _ in range(300):
                n = rnd.randint(1, 30)
                _check_cycle([rnd.randint(0, 3) for _ in range(n)], rnd.randint(-1, n - 1))
          `,
        },
        {
          name: '1500 个节点 · 额外空间 O(1)',
          hidden: true,
          code: py`
            head, nodes = _build(list(range(1500)), 700)
            try:
                tracemalloc.start()
                got = detect_cycle(head)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                assert got is nodes[700], "入口不正确"
                assert peak < 8 * 1024, f"峰值额外内存 {peak / 1024:.1f} KiB，应为 O(1)（不要用 set 记录访问过的节点）"
            finally:
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                _free(nodes)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-len',
          q: md`怎么顺便求出环的长度？`,
          a: md`相遇后让一个指针在环内继续走，回到相遇点所用的步数就是环长 $c$。或者阶段二找到入口后从入口绕一圈计数。`,
        },
        {
          id: 'f-intersect',
          q: md`两个无环链表求相交节点（LC160）怎么做到 $O(1)$ 空间？和本题有什么联系？`,
          a: md`双指针 pa、pb 分别从两个头出发，走到尾就跳到另一条链表的头。两者都走 $a + c + b$ 步后在交点相遇（不相交则同时为 None）。联系：把链表 A 的尾接到 B 的头，问题就变成了“找环入口”。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-reverse',
      freq: 3,
      q: md`手写反转链表，迭代和递归各写一遍，比较一下。`,
      a: md`
        **结论**：迭代用 prev / cur / nxt 三指针，$O(n)$ 时间 $O(1)$ 空间，是首选；递归 $O(n)$ 栈空间。

        **原因**：迭代的循环不变量是“prev 是已反转部分的头、cur 是未处理部分的头”，每轮先存后继再改向；
        递归先反转 \`head.next\` 之后的部分，再 \`head.next.next = head\`、\`head.next = None\`，必须断开旧边否则成环。

        **延伸**：反转区间（LC92）用哑节点 + 头插法；K 个一组（LC25）是“找尾 → 断开 → 反转 → 接回”；
        回文链表（LC234）用快慢指针找中点 + 反转后半段比较，做到 $O(1)$ 空间（比较完最好再反转回来，恢复原链表）。
      `,
    },
    {
      id: 'qa-cycle-proof',
      freq: 3,
      q: md`为什么快慢指针相遇后，一个回到 head 再同速走就能找到环入口？`,
      a: md`
        **结论**：因为 $a = (k-1)c + (c - b)$——从 head 到入口的距离，等于从相遇点向前走回入口的距离再加若干整圈。

        **原因**：设尾长 $a$、环长 $c$、入口到相遇点 $b$。相遇时 slow 走了 $a + b$，fast 走了 $2(a+b)$；
        二者在同一点，fast 多走的必是整圈：$a + b = kc$。所以从相遇点再走 $a$ 步，等于走 $(k-1)$ 圈加 $c - b$，正好到入口。

        **延伸**：slow 进环后最多 $c - 1$ 步就会被追上（距离每步缩小 1），所以总步数 $O(n)$。
        若 fast 步长为 3，距离每步缩小 2，可能永远错开——这就是固定用 1、2 的原因。
      `,
    },
    {
      id: 'qa-dummy',
      freq: 2,
      q: md`什么时候需要哑节点？它解决了什么问题？`,
      a: md`
        **结论**：头节点可能被删除或改变时用哑节点，最后返回 \`dummy.next\`。

        **原因**：链表的删除、插入都要操作“前驱”，而头节点没有前驱，不用 dummy 就要单独处理“删头”“换头”，分支多、易错。
        挂上 dummy 后每个真实节点都有前驱，代码统一。

        **延伸**：合并链表（LC21/23）、删除倒数第 N 个（LC19）、删除重复元素 II（LC82）、区间反转（LC92）、K 个一组（LC25）都用它。
        LRU 缓存的双向链表更进一步，用**头尾两个哨兵**，让插入、删除永远不碰 None。
      `,
    },
    {
      id: 'qa-array-vs-list',
      freq: 1,
      q: md`链表和数组各适合什么场景？为什么现代系统里“真正的链表”越来越少？`,
      a: md`
        **结论**：已知节点位置时，链表插入 / 删除是 $O(1)$；数组随机访问 $O(1)$、缓存友好。现代 CPU 上数组往往更快。

        **原因**：链表节点分散在堆上，遍历时每一步都可能 cache miss；数组连续存储，预取效率高。
        链表的优势只在“已经拿到节点指针”的场景才成立——这正是 LRU（哈希表直接给出节点）和内核里侵入式链表的用法。

        **延伸**：vLLM 的 KV 块空闲队列就是把前后指针直接放在块对象里的双向链表，从中间摘除一个块是 $O(1)$（见本章 LRU 关）。
      `,
    },
  ],

  cheatsheet: md`
    **三指针反转（O(1) 空间）**
    \`\`\`python
    prev, cur = None, head
    while cur:
        nxt = cur.next; cur.next = prev   # 先存后改
        prev, cur = cur, nxt
    return prev
    \`\`\`
    **K 个一组**：end 走 k 步（None 则返回）→ \`prev, cur = nxt, start\` 反转到 \`cur is nxt\` → \`pre.next = end; pre = start\`

    | 技巧 | 用途 |
    |---|---|
    | \`dummy = ListNode(0, head)\` | 头可能变：删除、反转、合并 |
    | 头插法 \`cur.next=nxt.next; nxt.next=pre.next; pre.next=nxt\` | 区间反转 LC92 |
    | slow 1 / fast 2 | 中点、判环；\`fast=head.next\` 起步得前中点 |
    | 相遇后 p 从 head 同速走 | 环入口：$a = (k-1)c + (c-b)$ |
    | fast 先走 n 步（从 dummy） | 删除倒数第 n 个 |

    - 节点身份用 \`is\` 判断；Python 递归上限 1000，链表题优先迭代
    - 合并收尾 \`tail.next = l1 or l2\`
  `,
})
