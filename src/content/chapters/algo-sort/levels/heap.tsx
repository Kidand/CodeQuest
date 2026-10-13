/**
 * 堆与 Top-K（algo-sort-heap）
 * LC215 / 347 / 23 / 295：sift up / down、O(n) 建堆、heapq 用法、Top-K；大模型岗场景：top-k 采样 / beam search / MoE 路由。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { HeapOpsViz } from '../viz/HeapOpsViz'
import { HeapifyViz } from '../viz/HeapifyViz'
import { TopKStreamViz } from '../viz/TopKStreamViz'

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'structure',
      title: '堆 = 用数组存的完全二叉树',
      body: md`
        **小顶堆**：每个节点都 ≤ 它的孩子，所以堆顶是最小值。它只保证“父子有序”，**兄弟之间、不同子树之间没有顺序**——堆不是有序数组。

        完全二叉树可以无空洞地按层存进数组（0 起下标）：

        $$
        \text{parent}(i) = \left\lfloor \tfrac{i-1}{2} \right\rfloor, \qquad \text{left}(i) = 2i+1, \qquad \text{right}(i) = 2i+2
        $$

        两个基本操作都只沿**一条根到叶的路径**移动，树高 $\lfloor \log_2 n \rfloor$，所以都是 $O(\log n)$：

        - **push**：放到末尾，与父节点比较，比父亲小就交换（**sift up / 上浮**）；
        - **pop**：取走堆顶，把**末尾元素**补到堆顶，与**较小的孩子**比较并交换（**sift down / 下沉**）。

        下沉时必须选较小的孩子：如果换上去的是较大的孩子，它会比另一个孩子大，新的父子关系立刻违反堆序。
      `,
      viz: HeapOpsViz,
      after: md`
        > [!TIP] Python 的 heapq
        > \`heapq\` 直接在 list 上维护**小顶堆**：\`heappush\`、\`heappop\`、\`heapify\`（原地 $O(n)$，返回 None）、
        > \`heappushpop(h, x)\`（先 push 再 pop，比分开调用高效）、\`heapreplace(h, x)\`（先 pop 再 push，返回值可能大于 x）、\`nlargest / nsmallest\`。
        > 大顶堆传统写法是存相反数 \`-x\`；**Python 3.14** 起提供了 \`heappush_max / heappop_max / heapify_max\` 等公开函数，但面试环境未必是 3.14，取负数的写法仍要会。
      `,
      checkpoint: {
        id: 'cp-structure',
        type: 'fill',
        prompt: md`往小顶堆 \`[2, 4, 3, 8, 5, 6, 7, 10, 9]\` 中 push 1。上浮过程共交换 ____(1)____ 次；完成后下标 1 处的元素是 ____(2)____。`,
        blanks: [
          { label: '(1)', answers: ['3'], numeric: true },
          { label: '(2)', answers: ['2'], numeric: true },
        ],
        explanation: md`1 放在下标 9，父亲是下标 4（值 5）→ 交换到 4；父亲下标 1（值 4）→ 交换到 1；父亲下标 0（值 2）→ 交换到 0。共 3 次，结果 \`[1, 2, 3, 8, 4, 6, 7, 10, 9, 5]\`，下标 1 处是 2。`,
      },
    },
    {
      id: 'heapify',
      title: '建堆为什么是 O(n)',
      body: md`
        把 $n$ 个元素逐个 push 建堆是 $O(n \log n)$。**自底向上建堆**更快：下标 $\ge \lfloor n/2 \rfloor$ 的都是叶子，天然是堆；
        从 $\lfloor n/2 \rfloor - 1$ 倒序到 0，对每个节点做一次 sift down（它的两棵子树此时已经是堆）。

        为什么是 $O(n)$？下沉的代价是节点的**高度**而不是深度。高度为 $h$ 的节点至多 $\lceil n / 2^{h+1} \rceil$ 个：

        $$
        \sum_{h=0}^{\lfloor \log_2 n \rfloor} \frac{n}{2^{h+1}} \cdot O(h) = O\!\left(n \sum_{h \ge 0} \frac{h}{2^{h+1}}\right) = O(n)
        $$

        因为 $\sum_{h \ge 0} h / 2^{h} = 2$。直觉：一半的节点是叶子（不动），四分之一的节点最多下沉 1 层……绝大多数节点都在底部，走不了几步。
        逐个 push 则相反：上浮的代价是**深度**，而大多数节点恰恰在最深的几层。
      `,
      viz: HeapifyViz,
      after: md`
        > [!INTERVIEW] 堆排序
        > 建堆 $O(n)$ 后反复把堆顶（最大值，用大顶堆）与末尾交换、堆大小减一、下沉，得到升序数组：$O(n \log n)$、原地、**不稳定**。
        > 它的最坏复杂度有保证，但访存跳跃（$i \to 2i+1$）、缓存不友好，实践中比快排慢，主要作为 introsort 的兜底。
      `,
      checkpoint: {
        id: 'cp-heapify',
        type: 'numeric',
        prompt: md`对 7 个元素（满二叉树，高度 2）自底向上建堆，**最坏情况下**总共最多交换多少次？`,
        answer: 4,
        tolerance: 0,
        explanation: md`高度 1 的节点 2 个，各最多下沉 1 次；高度 2 的根最多下沉 2 次：$2 \times 1 + 1 \times 2 = 4$。逐个 push 的最坏交换次数是 $0 + 1 + 1 + 2 + 2 + 2 + 2 = 10$。`,
      },
    },
    {
      id: 'topk',
      title: 'Top-K：大小为 k 的小顶堆',
      body: md`
        求 $n$ 个数中最大的 $k$ 个（LC215 第 K 大、LC347 前 K 个高频元素）：

        - 全排序：$O(n \log n)$；
        - **大小为 k 的小顶堆**：堆顶是“当前第 k 大”，即新元素进入 top-k 的**门槛**。新元素 > 堆顶就替换堆顶并下沉，否则直接丢弃。$O(n \log k)$ 时间、$O(k)$ 空间，而且可以**流式**处理（数据只需过一遍、不必全部装进内存）；
        - 快速选择：期望 $O(n)$，但需要全部数据在内存中且会修改数组（下一关）。

        为什么是**小**顶堆找**最大**的 k 个？我们需要随时知道 top-k 里最弱的那个，以决定新元素能否挤进来——最弱的就是最小的。

        同样的“K 路维护候选”模式还有：**LC23 合并 K 个有序链表**（堆里放每个链表当前的头，弹出最小的再放入它的后继，$O(N \log K)$）、
        **LC295 数据流中位数**（大顶堆存较小的一半、小顶堆存较大的一半，两堆大小差不超过 1）。
      `,
      viz: TopKStreamViz,
      after: md`
        > [!WARN] heapreplace 与 heappushpop 的区别
        > \`heapreplace(h, x)\`：先弹出堆顶再放入 x，**即使 x 比堆顶还小也会放进去**——Top-K 中必须先判断 \`x > h[0]\` 再调用。
        > \`heappushpop(h, x)\`：先放入再弹出最小值，x 比堆顶小时直接返回 x、堆不变，可以不加判断，写法更简洁。
      `,
      checkpoint: {
        id: 'cp-topk',
        type: 'single',
        prompt: md`在 $n = 10^8$ 个从磁盘流式读入的分数中找最大的 100 个，最合适的做法是：`,
        options: [
          '维护大小为 100 的小顶堆，O(n log 100) 时间、O(100) 内存',
          '全部读入内存后排序',
          '维护大小为 100 的大顶堆',
          '全部读入后快速选择',
        ],
        answer: 0,
        explanation: md`数据量大且流式到达时，大小为 k 的小顶堆只需 $O(k)$ 内存、一遍扫描。B、D 都需要把 $10^8$ 个数装进内存；C 的堆顶是最大值，无法判断新元素能否挤掉 top-100 中最小的那个。`,
      },
    },
    {
      id: 'llm',
      title: '大模型岗场景：top-k 采样、beam search、MoE 路由',
      body: md`
        大模型推理与训练中到处是“选出最大的 k 个”：

        | 场景 | 在哪里选 top-k | 规模 |
        |---|---|---|
        | **top-k 采样** | 每步从词表 logits 中保留最大的 k 个，其余置 $-\infty$ 后再 softmax 采样 | 词表 $V$ 约 3 万～25 万，k 常取 20～100 |
        | **beam search** | 每步把 B 个 beam 各自扩展 V 个 token，从 $B \times V$ 个候选里选分数最高的若干个 | HF transformers 会多取约 2B 个候选，以便部分序列结束（EOS）后仍能补足 B 个 beam |
        | **MoE 路由** | 每个 token 对所有专家打分，选 top-k 个专家 | Mixtral 8 选 2；DeepSeek-V3 256 个路由专家选 8（另有 1 个共享专家）；Kimi K3 896 选 16 |

        **GPU 上通常不用堆**：堆的操作是串行的，GPU 需要大规模并行。PyTorch 的 CUDA \`topk\` 历史上基于 **radix select**（按二进制位逐轮统计直方图来确定第 k 大的阈值），
        也有基于排序网络 / 分块的实现（如 FlashInfer 为大词表提供的 radix top-k 算子）。
        但在 CPU 端的数据处理、评测脚本、流式日志中，\`heapq.nlargest\` / 大小为 k 的堆依然是首选。

        NumPy 中最常用的是 \`np.argpartition\`（introselect，期望线性），见下一关。
      `,
      after: md`
        > [!NOTE] MoE 路由不只是 top-k
        > 真实系统在 top-k 外还有约束：DeepSeek-V3 为了控制跨节点通信，限制每个 token 最多发往 4 个节点，并用“无辅助损失”的偏置项做负载均衡；这些细节在「大模型 · MoE」章节展开。
      `,
      checkpoint: {
        id: 'cp-llm',
        type: 'numeric',
        prompt: md`beam search 中 beam 数 $B = 4$、词表 $V = 32000$。每一步要从多少个候选（beam × token）中挑选？`,
        answer: 128000,
        tolerance: 0,
        explanation: md`$B \times V = 4 \times 32000 = 128000$ 个候选，从中选出分数最高的若干个（例如 2B = 8 个）。这正是“大 n、小 k”的 Top-K 问题，GPU 上一次 \`topk\` 即可完成。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-index',
      type: 'single',
      difficulty: 1,
      prompt: md`用 0 起下标的数组存储完全二叉树，下标为 \`i\` 的节点的父节点与左孩子分别是：`,
      options: ['`(i - 1) // 2` 与 `2 * i + 1`', '`i // 2` 与 `2 * i`', '`(i + 1) // 2` 与 `2 * i + 2`', '`i - 1` 与 `i + 1`'],
      answer: 0,
      explanation: md`0 起下标：孩子 \`2i+1\`、\`2i+2\`，父亲 \`(i-1)//2\`。B 是 **1 起下标**的公式（根在下标 1），混用两套公式是手写堆最常见的错误。`,
    },
    {
      id: 'q-sorted',
      type: 'judge',
      difficulty: 1,
      prompt: md`小顶堆对应的数组一定是升序排列的。`,
      answer: false,
      explanation: md`错误。堆只保证父 ≤ 子，例如 \`[1, 5, 2]\` 是合法小顶堆但不是升序。反过来，升序数组一定是合法的小顶堆。`,
    },
    {
      id: 'q-heapq',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 Python \`heapq\`，哪些说法正确？（多选）`,
      options: [
        '`heapq.heappop(h)` 弹出并返回最小元素',
        '`heapq.heapify(h)` 原地建堆，时间 $O(n)$',
        '`heapq.heapreplace(h, x)` 返回的值可能大于 x',
        '`heapq.heapify(h)` 返回一个新的堆列表',
        '大顶堆的传统写法是把元素取相反数后存入',
      ],
      answer: [0, 1, 2, 4],
      explanation: md`D 错：\`heapify\` 原地修改并返回 None，写 \`h = heapq.heapify(h)\` 会得到 None。C 正确：heapreplace 先弹出旧堆顶（可能大于 x）再放入 x。E 是 3.14 之前唯一的方式（3.14 新增了 \`heappush_max\` 等函数）。`,
    },
    {
      id: 'q-heapify-swaps',
      type: 'numeric',
      difficulty: 2,
      prompt: md`对 $n = 15$ 个元素（满二叉树，高度 3）自底向上建堆，最坏情况下总共最多交换多少次？`,
      answer: 11,
      tolerance: 0,
      explanation: md`高度 1 的节点 4 个 × 1 + 高度 2 的 2 个 × 2 + 高度 3 的 1 个 × 3 = 4 + 4 + 3 = 11。一般地，满二叉树的最坏交换次数为 $n - \log_2(n+1) = 15 - 4 = 11$，严格小于 $n$。`,
    },
    {
      id: 'q-pop-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把小顶堆 \`heappop\` 的步骤按执行顺序排列：`,
      items: [
        '记下堆顶元素（最小值）作为返回值',
        '弹出数组最后一个元素',
        '把它放到堆顶（下标 0）',
        '与较小的孩子比较，若更大则交换，重复直到满足堆序或成为叶子',
        '返回之前记下的堆顶',
      ],
      explanation: md`先取末尾补位，才能保持“完全二叉树”的形状（数组无空洞），再通过下沉恢复堆序。只剩一个元素时直接返回即可。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与堆的用法配对：`,
      pairs: [
        ['LC215 数组中的第 K 个最大元素', '大小为 k 的小顶堆，答案是堆顶'],
        ['LC295 数据流的中位数', '大顶堆存较小一半 + 小顶堆存较大一半'],
        ['LC23 合并 K 个升序链表', '小顶堆存 K 个链表的当前头节点'],
        ['LC347 前 K 个高频元素', '先计数，再按频率维护大小为 k 的小顶堆'],
      ],
      explanation: md`四种都是“堆维护一个小规模候选集”。LC347 还可以用桶排序做到 $O(n)$（频率最多为 n，按频率分桶）。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出小顶堆的 \`sift_down(h, i)\`（有 2 行干扰项）：`,
      lines: [
        'def sift_down(h, i):',
        '    n = len(h)',
        '    while 2 * i + 1 < n:',
        '        c = 2 * i + 1',
        '        if c + 1 < n and h[c + 1] < h[c]:',
        '            c += 1',
        '        if h[i] <= h[c]:',
        '            break',
        '        h[i], h[c] = h[c], h[i]',
        '        i = c',
      ],
      distractors: ['        c = 2 * i', '        if h[c + 1] < h[c]:'],
      explanation: md`0 起下标的左孩子是 \`2i+1\`；判断右孩子前必须检查 \`c + 1 < n\`，否则在只有左孩子的节点上越界。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的小顶堆 \`sift_down\` 在 \`h = [9, 1, 2]\`、\`i = 0\` 上得到 \`[2, 1, 9]\`，不再是合法的堆。点出有问题的行：`,
      code: py`
        def sift_down(h, i):
            n = len(h)
            while 2 * i + 1 < n:
                c = 2 * i + 1
                if c + 1 < n and h[c + 1] > h[c]:
                    c += 1
                if h[i] <= h[c]:
                    break
                h[i], h[c] = h[c], h[i]
                i = c
      `,
      bugLines: [5],
      explanation: md`第 5 行应为 \`h[c + 1] < h[c]\`：要和**较小**的孩子交换。这里选了较大的孩子 2，交换后堆顶 2 大于左孩子 1，违反堆序。`,
    },
    {
      id: 'q-leaves',
      type: 'pick',
      difficulty: 2,
      prompt: md`长度 $n = 10$ 的堆数组中，哪些下标是**叶子**节点？（点出全部）自底向上建堆时，这些节点不需要 sift down。`,
      items: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
      answer: [5, 6, 7, 8, 9],
      explanation: md`节点 i 是叶子当且仅当 \`2i + 1 >= n\`，即 \`i >= n // 2 = 5\`。下标 4 的左孩子是 9，所以 4 不是叶子；建堆从 \`n // 2 - 1 = 4\` 开始倒序。`,
    },
    {
      id: 'q-why-min',
      type: 'single',
      difficulty: 3,
      prompt: md`用堆求“最大的 k 个数”时使用**小顶堆**，根本原因是：`,
      options: [
        '堆顶是当前 top-k 中最小的，也就是新元素要进入 top-k 必须超过的门槛，可以 O(1) 判断、O(log k) 替换',
        'Python 只有小顶堆',
        '小顶堆的 push 比大顶堆快',
        '小顶堆可以保持 top-k 有序',
      ],
      answer: 0,
      explanation: md`判断“新元素能否挤进 top-k”只需和 top-k 中最小者比较。若用大顶堆存全部元素再弹 k 次，复杂度是 $O(n + k \log n)$ 且需要 $O(n)$ 内存；用大小为 k 的大顶堆则无法知道该淘汰谁。B 不是根本原因（取负即可得大顶堆）；D 错，堆内部并不有序。`,
    },
    {
      id: 'q-moe',
      type: 'single',
      difficulty: 2,
      prompt: md`DeepSeek-V3 每个 MoE 层有 256 个路由专家、每个 token 激活其中 8 个（另有 1 个共享专家）。对一个 token 来说，路由这一步本质上是：`,
      options: [
        '在 256 个专家亲和度分数中做 top-8 选择',
        '对 256 个专家做全排序',
        '在 8 个专家中随机选 1 个',
        '对所有专家做 softmax 后全部加权',
      ],
      answer: 0,
      explanation: md`路由 = 对每个 token 的专家打分做 top-k（这里 k = 8），再对选中的专家输出加权求和。D 是稠密混合（soft MoE 一类），计算量与专家数成正比，失去了稀疏激活的意义；全排序是不必要的开销。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-heap-ops',
      title: '一步步写出 heappush / heappop',
      intro: md`
        在 Python list 上实现小顶堆的 \`heappush(h, x)\` 与 \`heappop(h)\`（与 \`heapq\` 行为一致：0 起下标、堆顶在 \`h[0]\`）。
      `,
      template: py`
        def heappush(h, x):
            h.append(x)
            i = len(h) - 1
            while i > 0:
                p = {{1}}
                if h[p] <= h[i]:
                    break
                h[p], h[i] = h[i], h[p]
                i = p

        def heappop(h):
            last = h.pop()
            if not h:
                return last
            top, h[0] = h[0], last
            i, n = 0, len(h)
            while {{2}}:                       # i 还有孩子
                c = 2 * i + 1
                if {{3}}:                      # 右孩子存在且更小
                    c += 1
                if {{4}}:                      # 堆序已满足
                    break
                h[i], h[c] = h[c], h[i]
                {{5}}
            return top
      `,
      steps: [
        {
          blank: 1,
          instruction: md`0 起下标中，\`i\` 的父节点下标。`,
          answer: '(i - 1) // 2',
          accept: ['(i - 1) >> 1', '(i-1)//2'],
          explain: md`根在 0：孩子是 \`2p+1\` 和 \`2p+2\`，反解得 \`p = (i-1)//2\`。`,
        },
        {
          blank: 2,
          instruction: md`循环条件：\`i\` 至少有左孩子。`,
          answer: '2 * i + 1 < n',
          accept: ['2*i+1 < n', 'i * 2 + 1 < n', 'i < n // 2'],
          explain: md`没有左孩子就一定是叶子，下沉结束。等价写法 \`i < n // 2\`。`,
        },
        {
          blank: 3,
          instruction: md`若右孩子存在且比左孩子小，改为指向右孩子。`,
          answer: 'c + 1 < n and h[c + 1] < h[c]',
          explain: md`必须先判断 \`c + 1 < n\`（短路求值保证不越界）。`,
        },
        {
          blank: 4,
          instruction: md`当前节点已经不大于较小的孩子，堆序满足。`,
          answer: 'h[i] <= h[c]',
          accept: ['h[c] >= h[i]', 'not h[c] < h[i]'],
        },
        {
          blank: 5,
          instruction: md`交换后继续向下。`,
          answer: 'i = c',
        },
      ],
      setup: py`
        import heapq as _hq, random as _r
      `,
      tests: [
        {
          name: '示例',
          code: py`
            h = []
            for x in [5, 1, 4, 2, 3]:
                heappush(h, x)
            got = [heappop(h) for _ in range(5)]
            assert got == [1, 2, 3, 4, 5], f"依次弹出应为 [1..5]，得到 {got}"
          `,
        },
        {
          name: '随机操作对拍 heapq',
          hidden: true,
          code: py`
            rnd = _r.Random(2026)
            for _ in range(100):
                a, b = [], []
                for _ in range(rnd.randint(1, 40)):
                    if b and rnd.random() < 0.4:
                        x, y = heappop(a), _hq.heappop(b)
                        assert x == y, f"heappop 得到 {x}，期望 {y}"
                    else:
                        v = rnd.randint(-20, 20)
                        heappush(a, v)
                        _hq.heappush(b, v)
                    assert sorted(a) == sorted(b) and (not a or a[0] == b[0]), "堆内容或堆顶不一致"
          `,
        },
      ],
      outro: md`
        这两个函数就是 \`heapq\` 的核心。实战里你要把它们封装成一个类，并加上 $O(n)$ 的建堆。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-minheap',
      title: '手写 MinHeap 类（含 O(n) 建堆）',
      difficulty: 'medium',
      source: '经典手撕',
      timeLimitMin: 20,
      description: md`
        不使用 \`heapq\`，实现小顶堆类：

        \`\`\`python
        class MinHeap:
            def __init__(self, items=None): ...   # items 可选；用 O(n) 自底向上建堆
            def push(self, x) -> None: ...
            def pop(self): ...                     # 弹出并返回最小值；空堆抛 IndexError
            def peek(self): ...                    # 返回最小值但不弹出；空堆抛 IndexError
            def __len__(self) -> int: ...
        \`\`\`

        **要求**
        - 不能 \`import heapq\`，不能用 \`sorted\` / \`.sort\`（会检查源码）；
        - \`__init__(items)\` 必须是 $O(n)$ 建堆：隐藏测试用“计数比较次数”的元素对 4096 个**降序**元素建堆，比较次数须 ≤ $3n$（逐个 push 约需 $n\log_2 n$ 次）；
        - 元素只支持比较运算；不要修改传入的 \`items\` 列表。

        **示例**
        \`\`\`python
        h = MinHeap([5, 3, 8])
        h.push(1); h.peek()   # 1
        h.pop(), h.pop()      # (1, 3)
        len(h)                # 2
        \`\`\`
      `,
      starter: py`
        class MinHeap:
            def __init__(self, items=None):
                # TODO: 复制 items，并从 len // 2 - 1 倒序 sift down
                self.h = []

            def push(self, x) -> None:
                raise NotImplementedError

            def pop(self):
                raise NotImplementedError

            def peek(self):
                raise NotImplementedError

            def __len__(self) -> int:
                return len(self.h)
      `,
      solution: py`
        class MinHeap:
            def __init__(self, items=None):
                self.h = list(items) if items is not None else []   # 复制，不改原列表
                for i in range(len(self.h) // 2 - 1, -1, -1):      # O(n) 建堆
                    self._down(i)

            def _up(self, i: int) -> None:
                h = self.h
                while i > 0:
                    p = (i - 1) // 2
                    if h[p] <= h[i]:
                        break
                    h[p], h[i] = h[i], h[p]
                    i = p

            def _down(self, i: int) -> None:
                h, n = self.h, len(self.h)
                while 2 * i + 1 < n:
                    c = 2 * i + 1
                    if c + 1 < n and h[c + 1] < h[c]:
                        c += 1
                    if h[i] <= h[c]:
                        break
                    h[i], h[c] = h[c], h[i]
                    i = c

            def push(self, x) -> None:
                self.h.append(x)
                self._up(len(self.h) - 1)

            def pop(self):
                if not self.h:
                    raise IndexError("pop from empty heap")
                last = self.h.pop()
                if not self.h:
                    return last
                top, self.h[0] = self.h[0], last
                self._down(0)
                return top

            def peek(self):
                if not self.h:
                    raise IndexError("peek from empty heap")
                return self.h[0]

            def __len__(self) -> int:
                return len(self.h)
      `,
      explanation: md`
        **结构**：数组 \`self.h\` 存完全二叉树，父 \`(i-1)//2\`、孩子 \`2i+1 / 2i+2\`。
        - \`push\`：末尾加入 + 上浮；\`pop\`：末尾补到堆顶 + 下沉（与较小孩子交换）；
        - \`__init__\`：从 \`n//2 - 1\` 倒序下沉。下沉代价是节点高度，$\sum_h \frac{n}{2^{h+1}} h = O(n)$；每层最多 2 次比较，总比较次数 ≤ $2n$。

        **复杂度**：push / pop $O(\log n)$，peek / len $O(1)$，建堆 $O(n)$。

        **常见错误**：用 1 起下标公式；下沉时选错孩子；只剩一个元素时 pop 写成“末尾补到堆顶”导致返回错误或越界；直接引用传入列表导致外部被修改。
      `,
      complexity: { time: 'push/pop O(log n)，建堆 O(n)', space: 'O(n)' },
      hints: [
        md`先写 \`_up(i)\` 和 \`_down(i)\` 两个私有方法，push / pop / 建堆都基于它们。`,
        md`pop：保存 \`h[0]\`，把 \`h.pop()\` 得到的末尾元素放到 \`h[0]\` 再下沉；注意弹出后堆为空的情况。`,
        md`建堆：\`for i in range(len(h) // 2 - 1, -1, -1): self._down(i)\`。`,
      ],
      setup: py`
        import heapq as _hq, linecache, random as _r

        class _K:
            """计数比较次数的元素"""
            __slots__ = ("v",)
            n = 0
            def __init__(self, v):
                self.v = v
            def __lt__(self, o): _K.n += 1; return self.v < o.v
            def __le__(self, o): _K.n += 1; return self.v <= o.v
            def __gt__(self, o): _K.n += 1; return self.v > o.v
            def __ge__(self, o): _K.n += 1; return self.v >= o.v
            def __eq__(self, o): _K.n += 1; return self.v == o.v
            __hash__ = None
      `,
      tests: [
        {
          name: '示例',
          code: py`
            h = MinHeap([5, 3, 8])
            h.push(1)
            assert h.peek() == 1, f"peek 期望 1，得到 {h.peek()}"
            a, b = h.pop(), h.pop()
            assert (a, b) == (1, 3), f"期望 (1, 3)，得到 {(a, b)}"
            assert len(h) == 2, f"len 期望 2，得到 {len(h)}"
          `,
        },
        {
          name: '空堆与单元素',
          code: py`
            h = MinHeap()
            assert len(h) == 0
            for fn in (h.pop, h.peek):
                try:
                    fn()
                    raise AssertionError(f"空堆调用 {fn.__name__} 应抛出 IndexError")
                except IndexError:
                    pass
            h.push(7)
            assert h.pop() == 7 and len(h) == 0
          `,
        },
        {
          name: '不修改传入列表 / 不使用库',
          code: py`
            src = [4, 1, 3]
            MinHeap(src)
            assert src == [4, 1, 3], "不要原地修改传入的 items"
            code = "".join(linecache.getlines("solution.py"))
            for banned in ("heapq", "sorted(", ".sort("):
                assert banned not in code, f"请手写，不要使用 {banned}"
          `,
        },
        {
          name: '随机操作对拍',
          hidden: true,
          code: py`
            rnd = _r.Random(23)
            for _ in range(80):
                init = [rnd.randint(-30, 30) for _ in range(rnd.randint(0, 25))]
                mine, ref = MinHeap(init), list(init)
                _hq.heapify(ref)
                for _ in range(60):
                    if ref and rnd.random() < 0.45:
                        x, y = mine.pop(), _hq.heappop(ref)
                        assert x == y, f"init={init}: pop 得到 {x}，期望 {y}"
                    else:
                        v = rnd.randint(-30, 30)
                        mine.push(v)
                        _hq.heappush(ref, v)
                    assert len(mine) == len(ref)
                    if ref:
                        assert mine.peek() == ref[0], f"peek 得到 {mine.peek()}，期望 {ref[0]}"
          `,
        },
        {
          name: '建堆 O(n)：4096 个降序元素',
          hidden: true,
          code: py`
            n = 4096
            items = [_K(v) for v in range(n, 0, -1)]
            _K.n = 0
            h = MinHeap(items)
            cnt = _K.n
            assert cnt <= 3 * n, f"建堆比较了 {cnt} 次，超过 3n = {3 * n}——是不是逐个 push 了？"
            _K.n = 0
            out = [h.pop().v for _ in range(n)]
            assert out == list(range(1, n + 1)), "建堆后依次 pop 的结果不正确"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-decrease-key',
          q: md`Dijkstra 需要“降低某个元素的键值”，heapq 不支持，怎么办？`,
          a: md`常用“懒删除”：直接 push 新的 (dist, node)，弹出时若 dist 大于已知最短距离就跳过，堆大小 $O(E)$、复杂度 $O(E\log E)$。若必须原地 decrease-key，需要额外维护“元素 → 堆下标”的映射，在每次交换时同步更新，然后对该位置上浮。`,
        },
      ],
    },
    {
      id: 'c-median-finder',
      title: '数据流的中位数',
      difficulty: 'hard',
      source: 'LeetCode 295',
      timeLimitMin: 20,
      description: md`
        设计数据结构，支持不断加入整数并随时查询当前所有数的中位数（偶数个时取中间两数的平均值）。

        \`\`\`python
        class MedianFinder:
            def __init__(self): ...
            def add_num(self, num: int) -> None: ...
            def find_median(self) -> float: ...
        \`\`\`

        **要求**：\`add_num\` $O(\log n)$、\`find_median\` $O(1)$。使用堆（可以用 \`heapq\`）；不能使用排序、\`bisect\` 或 \`statistics\`（会检查源码）。

        **示例**：\`add_num(1); add_num(2); find_median()\` → \`1.5\`；\`add_num(3); find_median()\` → \`2.0\`。
      `,
      starter: py`
        import heapq

        class MedianFinder:
            def __init__(self):
                # TODO: lo 为大顶堆（存相反数），保存较小的一半；hi 为小顶堆，保存较大的一半
                pass

            def add_num(self, num: int) -> None:
                raise NotImplementedError

            def find_median(self) -> float:
                raise NotImplementedError
      `,
      solution: py`
        import heapq

        class MedianFinder:
            def __init__(self):
                self.lo = []   # 大顶堆（存相反数）：较小的一半，允许比 hi 多 1 个
                self.hi = []   # 小顶堆：较大的一半

            def add_num(self, num: int) -> None:
                # 先进 lo，再把 lo 的最大值移到 hi，保证 lo 的所有元素 <= hi 的所有元素
                heapq.heappush(self.lo, -num)
                heapq.heappush(self.hi, -heapq.heappop(self.lo))
                # 再平衡大小：len(lo) == len(hi) 或 len(lo) == len(hi) + 1
                if len(self.hi) > len(self.lo):
                    heapq.heappush(self.lo, -heapq.heappop(self.hi))

            def find_median(self) -> float:
                if len(self.lo) > len(self.hi):
                    return float(-self.lo[0])
                return (-self.lo[0] + self.hi[0]) / 2
      `,
      explanation: md`
        **思路**：用大顶堆 \`lo\` 存较小的一半、小顶堆 \`hi\` 存较大的一半，维持两个不变量：
        1. \`lo\` 中所有元素 ≤ \`hi\` 中所有元素；
        2. \`len(lo) == len(hi)\` 或 \`len(lo) == len(hi) + 1\`。

        于是中位数要么是 \`lo\` 的堆顶，要么是两个堆顶的平均。加入时“先进 lo、把 lo 的最大值挪到 hi、若 hi 更大再挪回一个”，三次堆操作就同时维护了两个不变量。

        **复杂度**：\`add_num\` $O(\log n)$，\`find_median\` $O(1)$。

        **常见错误**：忘记取相反数导致 lo 变成小顶堆；只按大小平衡而不保证不变量 1（直接比较 num 与堆顶的写法容易漏分支）；整数除法丢掉 0.5。
      `,
      complexity: { time: 'add O(log n)，find O(1)', space: 'O(n)' },
      hints: [
        md`把数据分成“较小的一半”和“较大的一半”。中位数只和两半的边界元素有关——用什么数据结构能 $O(1)$ 拿到边界？`,
        md`较小一半用**大顶堆**（Python 存 \`-x\`），较大一半用小顶堆。规定较小一半最多比较大一半多 1 个。`,
        md`\`add_num\`：push 到 lo → 把 lo 的堆顶移到 hi → 若 \`len(hi) > len(lo)\`，再把 hi 的堆顶移回 lo。`,
      ],
      setup: py`
        import linecache, random as _r, time as _t

        def _banned():
            code = "".join(linecache.getlines("solution.py"))
            for b in ("sorted(", ".sort(", "insort", "bisect", "statistics"):
                if b in code:
                    return b
            return None

        def _ref_median(xs):
            s = sorted(xs)
            k = len(s)
            return float(s[k // 2]) if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2
      `,
      tests: [
        {
          name: '示例',
          code: py`
            m = MedianFinder()
            m.add_num(1); m.add_num(2)
            assert m.find_median() == 1.5, f"期望 1.5，得到 {m.find_median()}"
            m.add_num(3)
            assert m.find_median() == 2.0, f"期望 2.0，得到 {m.find_median()}"
          `,
        },
        {
          name: '负数、重复与返回类型',
          code: py`
            m = MedianFinder()
            seq = [-5, -5, 3, 0, -5, 10]
            for i, x in enumerate(seq, 1):
                m.add_num(x)
                got, exp = m.find_median(), _ref_median(seq[:i])
                assert got == exp and isinstance(got, float), f"加入 {seq[:i]} 后期望 {exp}（float），得到 {got!r}"
          `,
        },
        {
          name: '不使用排序 / 二分',
          code: py`
            b = _banned()
            assert b is None, f"请用两个堆实现，不要使用 {b}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = _r.Random(295)
            for _ in range(60):
                m, xs = MedianFinder(), []
                for _ in range(rnd.randint(1, 40)):
                    x = rnd.randint(-50, 50)
                    m.add_num(x)
                    xs.append(x)
                    got, exp = m.find_median(), _ref_median(xs)
                    assert abs(got - exp) < 1e-9, f"序列 {xs}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '复杂度：5000 次交替加入与查询',
          hidden: true,
          code: py`
            b = _banned()
            assert b is None, f"使用了 {b}，跳过计时"
            rnd = _r.Random(1)
            m = MedianFinder()
            t0 = _t.perf_counter()
            for _ in range(5000):
                m.add_num(rnd.randint(-10**6, 10**6))
                m.find_median()
            dt = _t.perf_counter() - t0
            assert dt < 0.5, f"用时 {dt:.2f}s，find_median 是不是每次都排序 / 遍历了？"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-window',
          q: md`如果要求“滑动窗口中位数”（LC480），两个堆怎么支持删除窗口外的元素？`,
          a: md`用**延迟删除**：维护一个哈希表记录“待删除元素 → 次数”，同时维护两个堆的“有效大小”；只有当待删除元素出现在堆顶时才真正弹出。平衡时按有效大小比较。也可以用有序容器（如平衡树 / SortedList）直接 $O(\log k)$ 删除。`,
        },
        {
          id: 'f-range',
          q: md`如果所有数都在 $[0, 100]$ 内，怎么优化？`,
          a: md`用长度 101 的计数数组，加入 $O(1)$，查询时累加计数找到中间位置 $O(100) = O(1)$。若 99% 的数在 $[0, 100]$ 内，可对范围外的数分别用两个堆或计数器兜底。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-heapify',
      freq: 3,
      q: md`为什么建堆是 O(n)，而不是 O(n log n)？`,
      a: md`
        **结论**：自底向上建堆的代价是所有节点**高度**之和，为 $O(n)$。

        **原因**：从最后一个非叶子节点倒序做 sift down，节点下沉的步数不超过它的高度。高度为 $h$ 的节点至多 $n/2^{h+1}$ 个，总代价 $\sum_h \frac{n}{2^{h+1}} h \le n$（因为 $\sum h/2^h = 2$）。一半节点是叶子不用动，越往上节点越少。

        **延伸**：逐个 push 的代价是节点**深度**之和，而大部分节点在最深层，所以是 $\Theta(n \log n)$（降序输入时取到）。堆排序整体仍是 $O(n\log n)$，因为 n 次 pop 每次 $O(\log n)$。
      `,
    },
    {
      id: 'qa-topk',
      freq: 3,
      q: md`求第 K 大 / Top-K 有哪些做法？各自适用于什么场景？`,
      a: md`
        **结论**：排序 $O(n\log n)$；大小为 k 的小顶堆 $O(n\log k)$；快速选择期望 $O(n)$；值域有限时计数 / 桶 $O(n + V)$。

        **原因**：堆只保留 k 个候选、一遍扫描，适合流式和数据量大于内存的情况；快速选择需要随机访问全部数据并会打乱数组，但平均最快；GPU 上用 radix select / 排序网络等并行算法（\`torch.topk\`）。

        **延伸**：LC347 前 K 个高频元素可以“计数 + 桶排序”做到 $O(n)$；k 接近 n 时找“最小的 n−k 个”更省。面试中先给堆解法（稳、好写），再提快速选择作为优化。
      `,
    },
    {
      id: 'qa-two-heaps',
      freq: 2,
      q: md`数据流中位数为什么用两个堆？能保证什么复杂度？`,
      a: md`
        **结论**：两个堆把数据分成较小 / 较大两半，中位数只取决于两个堆顶，加入 $O(\log n)$、查询 $O(1)$。

        **原因**：大顶堆存较小一半、小顶堆存较大一半，维护“lo 全部 ≤ hi 全部”和“大小差 ≤ 1”。每次加入最多 3 次堆操作即可恢复两个不变量。

        **延伸**：有序数组 + 二分插入查询 $O(1)$ 但插入 $O(n)$；平衡树或跳表也能 $O(\log n)$。若需支持删除（滑动窗口中位数），两个堆要配合延迟删除。
      `,
    },
    {
      id: 'qa-gpu-topk',
      freq: 2,
      q: md`大模型推理中的 top-k（采样、beam search、MoE 路由）在 GPU 上一般怎么实现？为什么不用堆？`,
      a: md`
        **结论**：用并行的选择算法（radix select、分块 + 排序网络），而不是堆。

        **原因**：堆的每次 push / pop 都依赖上一次的结果，天然串行；GPU 需要成千上万线程同时工作。radix select 按二进制位分轮，每轮对所有元素并行统计直方图，确定第 k 大所在的桶，几轮后得到阈值，再并行收集大于阈值的元素。PyTorch 的 CUDA \`topk\` 历史上即基于 radix select。

        **延伸**：词表 15 万量级、batch 很大时 top-k / top-p 会成为采样阶段的瓶颈之一，FlashInfer 等库提供了专门的融合采样算子。MoE 路由的 top-k 规模小（几十到几百个专家），开销主要在后续的 token 分发（all-to-all）而不是选择本身。
      `,
    },
  ],

  cheatsheet: md`
    **0 起下标**：父 \`(i-1)//2\` · 左 \`2i+1\` · 右 \`2i+2\` · 叶子 \`i >= n//2\`
    \`\`\`python
    def sift_down(h, i):
        n = len(h)
        while 2 * i + 1 < n:
            c = 2 * i + 1
            if c + 1 < n and h[c + 1] < h[c]: c += 1
            if h[i] <= h[c]: break
            h[i], h[c] = h[c], h[i]; i = c
    # 建堆：for i in range(n//2 - 1, -1, -1): sift_down(h, i)   # O(n)
    \`\`\`
    | 操作 | 复杂度 |
    |---|---|
    | push / pop | $O(\log n)$ |
    | peek | $O(1)$ |
    | heapify | $O(n)$ |
    | Top-K（大小 k 小顶堆） | $O(n \log k)$ |

    - heapq 是小顶堆；大顶堆存 \`-x\`（3.14+ 有 \`heappush_max\` 等）
    - \`heapreplace\`：先弹后压；\`heappushpop\`：先压后弹
    - 中位数：大顶堆（小一半）+ 小顶堆（大一半），\`len(lo) - len(hi) ∈ {0, 1}\`
    - MoE：DeepSeek-V3 256 选 8；Mixtral 8 选 2；Kimi K3 896 选 16
  `,
})
