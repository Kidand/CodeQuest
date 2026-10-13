/**
 * algo-graph-tree 二叉树遍历（必考）
 * 递归三序 → 统一显式栈 → 层序 BFS → 递归的两种思路
 */
import { defineLevel, md, py } from '@/content/helpers'
import { PY_TREE } from '../shared'
import { LevelListViz } from '../viz/LevelListViz'
import { TraversalViz } from '../viz/TraversalViz'
import { StackTraversalViz } from '../viz/StackTraversalViz'
import { LevelOrderViz } from '../viz/LevelOrderViz'

const REF = py`
  def _ref_inorder(root):
      res, st, cur = [], [], root
      while cur or st:
          while cur:
              st.append(cur)
              cur = cur.left
          cur = st.pop()
          res.append(cur.val)
          cur = cur.right
      return res

  def _ref_postorder(root):
      res, st = [], [(root, False)] if root else []
      while st:
          node, vis = st.pop()
          if vis:
              res.append(node.val)
              continue
          st.append((node, True))
          if node.right: st.append((node.right, False))
          if node.left: st.append((node.left, False))
      return res

  def _ref_zigzag(root):
      if root is None:
          return []
      res, q = [], deque([root])
      while q:
          level = []
          for _ in range(len(q)):
              x = q.popleft()
              level.append(x.val)
              if x.left: q.append(x.left)
              if x.right: q.append(x.right)
          res.append(level[::-1] if len(res) % 2 else level)
      return res
`

const STARTER_HEAD = py`
  # 运行环境已提供以下定义（无需自己写，直接使用）：
  # class TreeNode:
  #     def __init__(self, val=0, left=None, right=None):
  #         self.val, self.left, self.right = val, left, right
  # list_to_tree([1, None, 2, 3]) -> TreeNode   LeetCode 层序列表 → 树
  # tree_to_list(root) -> list                  树 → 层序列表（调试时可以 print 看看）
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'repr',
      title: '二叉树的表示：TreeNode 与层序列表',
      body: md`
        面试里的二叉树几乎都是“节点 + 指针”表示，和 LeetCode 的定义一模一样：

        \`\`\`python
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val, self.left, self.right = val, left, right
        \`\`\`

        题面里给出的 \`[3,9,20,null,null,15,7]\` 是**层序列表**：按层、从左到右，**只为存在的节点**写出它的左右孩子，
        空孩子写 \`null\`（Python 里是 \`None\`），末尾多余的 \`None\` 全部截掉。

        它**不是**堆那种“下标 $i$ 的孩子在 $2i+1, 2i+2$”的数组：空节点的孩子不会再占位。对一条右斜链，层序列表长 $2n-1$，
        堆式数组却要 $2^n - 1$ 个格子——这就是为什么只有**完全二叉树**（堆）才适合用数组存。

        本章所有二叉树实战题的运行环境都提供 \`TreeNode\`、\`list_to_tree(vals)\`（列表 → 树）和 \`tree_to_list(root)\`（树 → 列表），
        测试用例直接写成层序列表。
      `,
      viz: LevelListViz,
      after: md`
        > [!TIP] 手写 list_to_tree 的要点
        > 用队列：每弹出一个已存在的节点，就从列表里**连续读两个值**作为它的左右孩子；值为 \`None\` 时不建节点、也不入队。
        > 这正是上面“None 只给存在的节点占位”的另一面。面试时如果要求“自己写测试用例”，这个函数要能当场写出来。
      `,
      checkpoint: {
        id: 'cp-repr',
        type: 'fill',
        prompt: md`一棵树：根 1，左孩子 2（2 没有孩子），右孩子 3，3 只有左孩子 4。它的 LeetCode 层序列表是 ____(1)____（用 None 表示空，逗号分隔，带方括号）。`,
        blanks: [{ label: '(1)', answers: ['[1,2,3,None,None,4]', '[1,2,3,null,null,4]'] }],
        explanation: md`第 0 层 \`1\`；第 1 层 \`2, 3\`；然后为 2 写两个空孩子 \`None, None\`，为 3 写左孩子 \`4\`、右孩子 \`None\`；4 的两个空孩子与 3 的右空孩子都在末尾，截掉。结果 \`[1,2,3,None,None,4]\`。注意不是堆式的 \`[1,2,3,None,None,4,None]\`。`,
      },
    },
    {
      id: 'orders',
      title: '递归三序：同一条路线，三个输出时机',
      body: md`
        递归遍历时，函数会**三次经过**同一个节点：刚进入时、左子树返回时、右子树返回时。
        前序 / 中序 / 后序**走的路线完全相同**，区别只在于“在第几次经过时把它放进结果”：

        | 遍历 | 输出位置 | 顺序 | 典型用途 |
        |---|---|---|---|
        | 前序 | 第 1 次（进入） | 根 → 左 → 右 | 复制 / 序列化树、按路径向下传参数 |
        | 中序 | 第 2 次（左子树完成） | 左 → 根 → 右 | BST 得到**升序**序列 |
        | 后序 | 第 3 次（左右都完成） | 左 → 右 → 根 | 需要子树信息的问题：高度、直径、最大路径和 |

        复杂度：每个节点被处理常数次，时间 $O(n)$；空间是递归深度 $O(h)$——平衡树 $O(\log n)$，退化成链时 $O(n)$。

        拖动下面的进度条，切换三种遍历：节点下方的 ①②③ 表示已经经过了几次，**路线一模一样**，只有输出的时机变了。
      `,
      viz: TraversalViz,
      after: md`
        > [!INTERVIEW] 后序是“进阶题”的母题
        > 后序位置能拿到左右子树的返回值，所以“树高、直径、最大路径和、最近公共祖先”都写在后序位置——下一关「二叉树进阶」专门讲这个框架。
        > 前序位置只能拿到从父节点**传下来**的参数（如当前深度、路径和）。
      `,
      checkpoint: {
        id: 'cp-orders',
        type: 'single',
        prompt: md`某二叉树前序为 \`[1, 2, 4, 5, 3]\`，中序为 \`[4, 2, 5, 1, 3]\`。它的后序是：`,
        options: ['`[4, 5, 2, 3, 1]`', '`[4, 2, 5, 3, 1]`', '`[5, 4, 2, 3, 1]`', '`[4, 5, 3, 2, 1]`'],
        answer: 0,
        explanation: md`前序第一个 1 是根；中序中 1 左边 \`[4,2,5]\` 是左子树、右边 \`[3]\` 是右子树。左子树前序 \`[2,4,5]\`：根 2，左 4、右 5。后序 = 左子树后序 \`[4,5,2]\` + 右子树 \`[3]\` + 根 \`1\`。B 是中序换了位置，D 把 3 放到了 2 前面（右子树先于左子树完成，错误）。`,
      },
    },
    {
      id: 'stack',
      title: '迭代：统一的显式栈写法',
      body: md`
        面试官常要求“不用递归”：递归本质是系统帮你维护了一个调用栈，迭代就是**自己维护这个栈**。
        难点在于中序、后序时，一个节点第一次弹出时**还不能输出**（它的子树还没处理）。

        **统一写法（标记法）**：栈里放 \`(node, visited)\`。

        1. 弹出 \`(node, False)\`：第一次见到，**不输出**，而是“展开”——把它的左右孩子和 \`(node, True)\` 按**想要的访问顺序的逆序**压栈（栈后进先出）；
        2. 弹出 \`(node, True)\`：第二次见到，输出。

        \`\`\`python
        def traverse(root, order):
            res, stack = [], [(root, False)] if root else []
            while stack:
                node, visited = stack.pop()
                if visited:
                    res.append(node.val)
                    continue
                if order == "post": stack.append((node, True))      # 左右根 → 逆序压：根 右 左
                if node.right: stack.append((node.right, False))
                if order == "in":   stack.append((node, True))      # 左根右 → 逆序压：右 根 左
                if node.left:  stack.append((node.left, False))
                if order == "pre":  stack.append((node, True))      # 根左右 → 逆序压：右 左 根
            return res
        \`\`\`

        **三种遍历只差 \`(node, True)\` 这一行放在哪里**，不用再分别背三套模板。每个节点入栈、出栈各 2 次，时间 $O(n)$，栈长 $O(h)$。
      `,
      viz: StackTraversalViz,
      after: md`
        **面试官最常要的“经典中序”**（指针 + 栈，必须也会写）：一路向左把节点压栈，弹出即输出，然后转向右子树。

        \`\`\`python
        def inorder(root):
            res, stack, cur = [], [], root
            while cur or stack:
                while cur:                 # 一路向左，沿途压栈
                    stack.append(cur)
                    cur = cur.left
                cur = stack.pop()          # 最左的未访问节点
                res.append(cur.val)
                cur = cur.right            # 转向右子树
            return res
        \`\`\`

        > [!WARN] “根右左再反转”得到后序的陷阱
        > 前序改成“根 → 右 → 左”再把结果反转，输出确实是后序；但节点的**处理时机**仍是先根后子。
        > 如果题目要在后序位置**用到子树的计算结果**（如求高度），这个技巧不能用，标记法可以。
      `,
      checkpoint: {
        id: 'cp-stack',
        type: 'single',
        prompt: md`用统一的标记法做**后序**遍历。弹出 \`(x, False)\` 后，三条压栈语句的正确顺序是：`,
        options: [
          '`(x.left, False)` → `(x.right, False)` → `(x, True)`',
          '`(x, True)` → `(x.right, False)` → `(x.left, False)`',
          '`(x.right, False)` → `(x, True)` → `(x.left, False)`',
          '`(x.right, False)` → `(x.left, False)` → `(x, True)`',
        ],
        answer: 1,
        explanation: md`后序要“左 → 右 → 根”，栈后进先出，所以要按逆序“根 → 右 → 左”压栈。A 是把访问顺序原样压栈（弹出变成根右左）；C 是中序的压法；D 是前序的压法。`,
      },
    },
    {
      id: 'bfs',
      title: '层序遍历：用 len(q) 切出每一层',
      body: md`
        层序遍历就是从根出发的 **BFS**：用队列，先进先出，天然一层一层往外扩。
        LC 102 要求**按层分组**输出，关键一行是：

        \`\`\`python
        for _ in range(len(q)):   # 进入这一层时，队列里恰好是这一层的全部节点
        \`\`\`

        \`range(len(q))\` 在循环开始时**只求值一次**，即使循环中不断有下一层的孩子入队，本轮也只弹出当前层的节点。

        常见变体都只改一两行：

        | 题目 | 改动 |
        |---|---|
        | LC 103 锯齿层序 | 奇数层把 \`level\` 反转（或用 \`deque.appendleft\` 往左插） |
        | LC 199 右视图 | 每层只取最后一个 |
        | LC 104 最大深度 | 数一共有几层 |
        | LC 111 最小深度 | 第一次遇到叶子就返回当前层号（BFS 比 DFS 更早停） |

        空间：队列最长是**最宽一层**的节点数 $w$；完全二叉树的最底层约有 $n/2$ 个节点，BFS 的空间是 $O(n)$，而 DFS 只要 $O(\log n)$。
      `,
      viz: LevelOrderViz,
      after: md`
        > [!WARN] 别用 list.pop(0)
        > Python 的 \`list.pop(0)\` 要把后面所有元素前移，是 $O(n)$ 的，整个 BFS 退化成 $O(n^2)$。队列一律用 \`collections.deque\` 的 \`popleft()\`。
      `,
      checkpoint: {
        id: 'cp-bfs',
        type: 'single',
        prompt: md`把 LC 102 代码中的 \`for _ in range(len(q)):\` 改成 \`while q:\`（循环体不变），对一棵 3 层的满二叉树会得到什么？`,
        options: ['仍是正确的分层结果', '只有一层，包含全部 7 个节点（按层序排列）', '每层只有一个节点', '死循环'],
        answer: 1,
        explanation: md`内层 \`while q\` 会一直弹到队列为空，期间新入队的孩子也被弹出并放进同一个 \`level\`，于是所有节点都挤进第一层。\`range(len(q))\` 的意义正是“冻结”本层节点数。不会死循环，因为每个节点只入队一次。`,
      },
    },
    {
      id: 'recursion',
      title: '递归的两种思路：遍历 vs 分解',
      body: md`
        写树的递归时先想清楚用哪种思路：

        - **遍历**：用一个外部变量，在走遍整棵树的过程中更新它（前序位置向下传参数）。
        - **分解**：定义函数的返回值，用“左右子树的答案”推出“整棵树的答案”（后序位置）。

        以 LC 104 最大深度为例：

        \`\`\`python
        # 思路一：遍历。depth 向下传，到达节点时更新全局最大值
        def max_depth(root):
            best = 0
            def dfs(node, depth):
                nonlocal best
                if node is None:
                    return
                best = max(best, depth)
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
            dfs(root, 1)
            return best

        # 思路二：分解。树高 = 1 + max(左子树高, 右子树高)
        def max_depth(root):
            if root is None:
                return 0
            return 1 + max(max_depth(root.left), max_depth(root.right))
        \`\`\`

        LC 226 翻转二叉树：在**前序或后序**位置交换左右孩子都对；但在**中序**位置交换是经典陷阱——
        交换后再“递归右子树”，处理的其实是**原来的左子树**，原左子树被翻转两次、原右子树没被翻转。

        > [!WARN] Python 的递归深度
        > CPython 默认递归上限 1000（\`sys.getrecursionlimit()\`）。一棵退化成链的 $10^4$ 节点的树，递归遍历会抛 \`RecursionError\`。
        > 面试中被问到“树很深怎么办”：改迭代（显式栈），或 BFS。
      `,
      checkpoint: {
        id: 'cp-recursion',
        type: 'judge',
        prompt: md`LC 226 翻转二叉树：在中序位置交换 \`root.left, root.right = root.right, root.left\`，然后照常写 \`invert(root.right)\`，结果仍然正确。`,
        answer: false,
        explanation: md`中序位置交换时，左子树已经翻转过；交换后 \`root.right\` 指向的正是**刚翻转过的原左子树**，于是它被再翻转一次（复原），而原右子树（现在是 \`root.left\`）根本没被处理。若坚持中序，交换后应继续递归 \`root.left\`。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-bst-inorder',
      type: 'single',
      difficulty: 1,
      prompt: md`对一棵二叉搜索树（BST）做哪种遍历，可以得到升序序列？`,
      options: ['前序', '中序', '后序', '层序'],
      answer: 1,
      explanation: md`BST 满足“左子树 < 根 < 右子树”，中序的“左 → 根 → 右”恰好按从小到大访问。前序/后序/层序都不保证有序，例如 \`[2,1,3]\` 的前序是 \`[2,1,3]\`。`,
    },
    {
      id: 'q-orders-fill',
      type: 'fill',
      difficulty: 1,
      prompt: md`树的层序列表为 \`[1, 2, 3, None, 4, 5]\`。前序遍历是 ____(1)____，后序遍历是 ____(2)____（逗号分隔，不带括号）。`,
      blanks: [
        { label: '(1)', answers: ['1,2,4,3,5', '[1,2,4,3,5]'] },
        { label: '(2)', answers: ['4,2,5,3,1', '[4,2,5,3,1]'] },
      ],
      explanation: md`树形：1 的左孩子 2、右孩子 3；2 只有右孩子 4；3 只有左孩子 5。前序：1 → 2 → 4 → 3 → 5；后序：4 → 2 → 5 → 3 → 1。易错点是把 \`None\` 当成 2 的右孩子——列表中第 3 个位置（下标 3）是 2 的**左**孩子。`,
    },
    {
      id: 'q-pre-post',
      type: 'judge',
      difficulty: 2,
      prompt: md`节点值互不相同时，已知前序和后序遍历，可以唯一确定一棵二叉树。`,
      answer: false,
      explanation: md`反例：根 1 只有左孩子 2，与根 1 只有右孩子 2，前序都是 \`[1,2]\`、后序都是 \`[2,1]\`。前序+后序无法区分“只有一个孩子时它在左还是右”；只有当树是**满二叉树**（每个节点 0 或 2 个孩子）时才能唯一确定（LC 889 允许返回任意一个答案）。`,
    },
    {
      id: 'q-unique',
      type: 'multi',
      difficulty: 2,
      prompt: md`节点值互不相同。以下哪些遍历组合可以**唯一**确定一棵一般的二叉树？（多选）`,
      options: ['前序 + 中序', '后序 + 中序', '层序 + 中序', '前序 + 后序'],
      answer: [0, 1, 2],
      explanation: md`关键是中序：知道根之后，中序能把剩余节点**切成左右两半**。前序/后序/层序都能告诉你“谁是根”（前序第一个、后序最后一个、层序中最先出现的那个），配合中序就能递归切分。前序+后序缺少左右的划分信息，见上一题的反例。`,
    },
    {
      id: 'q-space',
      type: 'numeric',
      difficulty: 2,
      prompt: md`一棵 $n = 1023$ 个节点的**满二叉树**（10 层）。用标准 BFS（每弹出一个节点就把它的孩子入队）做层序遍历，队列长度的最大值是多少？`,
      answer: 512,
      tolerance: 0,
      explanation: md`处理第 8 层（256 个节点）时，弹出 $k$ 个、压入 $2k$ 个，队列长 $256 - k + 2k = 256 + k$，在 $k = 256$ 时达到 512，即整个最底层。对比：递归 DFS 的最大深度只有 10。BFS 空间是 $O(w)$（最宽层），DFS 是 $O(h)$（树高）。`,
    },
    {
      id: 'q-order-push',
      type: 'order',
      difficulty: 2,
      prompt: md`统一标记法做**中序**遍历，弹出 \`(x, False)\` 后，按**压栈的先后**排列下面三条语句：`,
      items: ['`stack.append((x.right, False))`', '`stack.append((x, True))`', '`stack.append((x.left, False))`'],
      explanation: md`中序要“左 → 根 → 右”，栈后进先出，压栈顺序要反过来：先右、再根（标记为 True）、最后左。这样下一次弹出的是左子树，左子树处理完弹出 \`(x, True)\` 输出 x，最后才轮到右子树。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把任务与最自然的遍历方式配对：`,
      pairs: [
        ['按层输出 / 求最小深度', '层序（BFS）'],
        ['BST 中第 k 小的元素', '中序'],
        ['计算每个节点的子树高度', '后序'],
        ['把树序列化成字符串再还原', '前序（带空标记）'],
      ],
      explanation: md`层序天然分层，最小深度遇到第一个叶子即可停；BST 中序是升序，数到第 k 个即可；子树高度依赖左右子树结果，只能在后序位置算；前序先写根再写左右，配合空标记 \`#\` 可以唯一还原（下一关 LC 297）。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出“指针 + 栈”的经典迭代中序遍历（有 2 行干扰项）：`,
      lines: [
        'def inorder(root):',
        '    res, stack, cur = [], [], root',
        '    while cur or stack:',
        '        while cur:',
        '            stack.append(cur)',
        '            cur = cur.left',
        '        cur = stack.pop()',
        '        res.append(cur.val)',
        '        cur = cur.right',
        '    return res',
      ],
      distractors: ['    while stack:', '        cur = stack[-1]'],
      explanation: md`外层条件必须是 \`cur or stack\`：一开始栈是空的但 \`cur = root\`，只写 \`while stack\` 会直接退出；转向右子树后栈也可能暂时为空。\`stack[-1]\` 只是查看栈顶而不弹出，同一个节点会被反复访问，死循环。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面是 LC 103 锯齿形层序遍历（第 0 层从左到右，第 1 层从右到左，交替）。有两行错误，点出它们：`,
      code: py`
        from collections import deque

        def zigzag(root):
            if root is None:
                return []
            res, q = [], deque([root])
            while q:
                level = []
                for _ in range(len(q)):
                    node = q.pop()
                    level.append(node.val)
                    if node.left: q.append(node.left)
                    if node.right: q.append(node.right)
                if len(res) % 2 == 0:
                    level.reverse()
                res.append(level)
            return res
      `,
      bugLines: [10, 14],
      explanation: md`第 10 行 \`q.pop()\` 从**右端**弹出，队列变成了栈，层的划分和顺序都会乱，应为 \`q.popleft()\`。第 14 行条件写反：\`len(res)\` 是当前层号，第 0 层不该反转、第 1 层才反转，应为 \`len(res) % 2 == 1\`。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`树的层序列表为 \`[1, 2, 3, 4, 5, None, 6, None, None, 7]\`（即图解中的示例树）。**后序**遍历中，在节点 5 **之前**输出的节点有哪些？`,
      items: ['1', '2', '3', '4', '5', '6', '7'],
      answer: [3, 6],
      explanation: md`树形：1 的孩子 2、3；2 的孩子 4、5；3 只有右孩子 6；5 只有左孩子 7。后序为 \`4, 7, 5, 2, 6, 3, 1\`，5 之前是 4 和 7。2 是 5 的父节点，后序中父节点一定在孩子之后。`,
    },
    {
      id: 'q-invert',
      type: 'single',
      difficulty: 3,
      prompt: md`
        下面的“翻转二叉树”作用在 \`[4, 2, 7, 1, 3, 6, 9]\` 上，返回的树的层序列表是？
        \`\`\`python
        def invert(root):
            if root is None:
                return None
            invert(root.left)
            root.left, root.right = root.right, root.left
            invert(root.left)
            return root
        \`\`\`
      `,
      options: ['`[4, 7, 2, 9, 6, 3, 1]`', '`[4, 7, 2, 6, 9, 1, 3]`', '`[4, 2, 7, 1, 3, 6, 9]`', '`[4, 7, 2, 6, 9, 3, 1]`'],
      answer: 0,
      explanation: md`这段代码在中序位置交换，但交换后递归的是 \`root.left\`（也就是原右子树），所以是**正确**的写法：原左子树在交换前已翻转，原右子树在交换后被翻转。结果是完全翻转的 \`[4,7,2,9,6,3,1]\`。若第二次递归写成 \`invert(root.right)\` 才会出错（原左子树翻转两次，得到 B 那样的半翻转结果）。本题考察的是“看清交换之后指针指向谁”。`,
    },
    {
      id: 'q-recursion-limit',
      type: 'single',
      difficulty: 3,
      prompt: md`在 CPython 默认设置下，对一棵 $10^4$ 个节点、退化成左斜链的树调用递归中序遍历，最可能的结果是：`,
      options: ['正常返回，只是慢一些', '抛出 `RecursionError`', 'Python 自动把尾递归优化成循环', '返回结果正确但栈内存 O(1)'],
      answer: 1,
      explanation: md`CPython 默认递归上限为 1000，链深 $10^4$ 必然超限抛 \`RecursionError\`。CPython **没有**尾递归优化（而且中序遍历本来也不是尾递归）。可以 \`sys.setrecursionlimit\` 调大，但过深仍可能撑爆 C 栈导致崩溃；稳妥做法是改用显式栈。`,
    },
    {
      id: 'q-deque',
      type: 'judge',
      difficulty: 1,
      prompt: md`BFS 中用 \`list.pop(0)\` 代替 \`deque.popleft()\`，时间复杂度不变，只是常数稍大。`,
      answer: false,
      explanation: md`\`list.pop(0)\` 需要把其余元素整体前移，单次 $O(n)$，整个 BFS 变成 $O(n^2)$；\`deque.popleft()\` 是 $O(1)$。这是 Python 手撕 BFS 最常见的性能问题。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-postorder',
      title: '用统一显式栈写后序遍历',
      intro: md`
        实现 \`postorder(root)\`，返回后序遍历的值列表，**不用递归**。

        栈中元素是 \`(node, visited)\`：\`visited=False\` 表示第一次见到、需要展开；\`visited=True\` 表示子树已安排好、轮到输出。
        运行环境提供 \`TreeNode\` 与 \`list_to_tree\`。
      `,
      template: py`
        def postorder(root):
            res, stack = [], [(root, False)] if root else []
            while stack:
                node, visited = {{1}}
                if {{2}}:
                    res.append(node.val)
                    continue
                # 后序 = 左 → 右 → 根，按逆序压栈
                {{3}}
                if node.right:
                    {{4}}
                if node.left:
                    {{5}}
            return res
      `,
      steps: [
        {
          blank: 1,
          instruction: md`从栈顶取出一个元素并解包。`,
          answer: 'stack.pop()',
          hint: 'list 的哪个方法从末尾取出元素？',
          explain: md`Python 的 \`list\` 当栈用：\`append\` 压栈、\`pop()\` 弹栈顶，都是 $O(1)$。`,
        },
        {
          blank: 2,
          instruction: md`什么情况下应该输出这个节点？`,
          answer: 'visited',
          accept: ['visited is True', 'visited == True'],
          explain: md`只有第二次见到（标记为 True）时，它的左右子树已经全部输出完毕，才轮到它自己。`,
        },
        {
          blank: 3,
          instruction: md`后序中根**最后**输出，所以它要**最先**压栈（压在最底下）。写出把当前节点标记为“待输出”的压栈语句。`,
          answer: 'stack.append((node, True))',
          accept: ['stack += [(node, True)]', 'stack.extend([(node, True)])'],
          explain: md`这一行的位置就是三种遍历的唯一区别：前序放在最后、中序放在左右之间、后序放在最前。`,
        },
        {
          blank: 4,
          instruction: md`压入右孩子（第一次见到，未展开）。`,
          answer: 'stack.append((node.right, False))',
          accept: ['stack += [(node.right, False)]'],
          explain: md`右孩子比左孩子先压栈，就会比左孩子后弹出——保证“左 → 右”。`,
        },
        {
          blank: 5,
          instruction: md`压入左孩子。`,
          answer: 'stack.append((node.left, False))',
          accept: ['stack += [(node.left, False)]'],
          explain: md`左孩子最后压栈、最先弹出。整个过程每个节点进栈 2 次、出栈 2 次，时间 $O(n)$，栈长 $O(h)$。`,
        },
      ],
      setup: PY_TREE + '\n' + REF,
      tests: [
        { name: '示例 [1,None,2,3]', code: py`assert postorder(list_to_tree([1, None, 2, 3])) == [3, 2, 1]` },
        {
          name: '示例树与空树',
          code: py`
            got = postorder(list_to_tree([1, 2, 3, 4, 5, None, 6, None, None, 7]))
            assert got == [4, 7, 5, 2, 6, 3, 1], f"期望 [4, 7, 5, 2, 6, 3, 1]，得到 {got}"
            assert postorder(None) == []
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(102)
            for _ in range(150):
                t = _rand_tree(rnd, rnd.randint(0, 40))
                exp, got = _ref_postorder(t), postorder(t)
                assert got == exp, f"树 {tree_to_list(t)}：期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        把第 3 空移到两个 \`if\` 之间就是中序，移到最后就是前序。下面的实战要求写**中序**——你可以用这个统一写法，也可以用“指针 + 栈”的经典写法。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-inorder',
      title: '二叉树的中序遍历（迭代）',
      difficulty: 'easy',
      source: 'LeetCode 94',
      timeLimitMin: 12,
      description: md`
        给定二叉树的根节点 \`root\`，返回它的**中序遍历**结果。**不允许使用递归**（隐藏测试会把可用的递归深度压到几十层，再用一棵深 300 的树检查）。

        \`\`\`python
        def inorder_traversal(root: TreeNode | None) -> list[int]: ...
        \`\`\`

        运行环境已提供 \`TreeNode\`（字段 \`val / left / right\`）、\`list_to_tree(vals)\`（LeetCode 层序列表 → 树）和 \`tree_to_list(root)\`，测试用例写成层序列表。

        **示例**
        - \`[1, None, 2, 3]\` → \`[1, 3, 2]\`
        - \`[]\` → \`[]\`
        - \`[1, 2, 3, 4, 5, None, 6, None, None, 7]\` → \`[4, 2, 7, 5, 1, 3, 6]\`

        **约束**：节点数 $0 \le n \le 1500$；不得修改树的结构（遍历结束时树必须与输入相同）。
      `,
      starter: STARTER_HEAD + '\n\n' + py`
        def inorder_traversal(root: TreeNode | None) -> list[int]:
            # TODO: 用显式栈实现中序遍历（左 → 根 → 右），不要递归
            return []
      `,
      solution: py`
        def inorder_traversal(root: TreeNode | None) -> list[int]:
            res, stack, cur = [], [], root
            while cur or stack:
                while cur:                  # 一路向左，沿途压栈
                    stack.append(cur)
                    cur = cur.left
                cur = stack.pop()           # 最左的未访问节点
                res.append(cur.val)
                cur = cur.right             # 转向右子树
            return res
      `,
      explanation: md`
        **思路一（经典）**：\`cur\` 指针一路向左并压栈；走到底后弹出栈顶输出，再转向它的右子树。栈里保存的是“左子树还没处理完、自己还没输出”的祖先。

        **思路二（统一标记法）**：栈中放 \`(node, visited)\`，第一次弹出时按“右 → 根(True) → 左”压栈，第二次弹出时输出。

        **复杂度**：每个节点入栈、出栈各一次（标记法各两次），时间 $O(n)$；栈长不超过树高，空间 $O(h)$。

        **常见错误**
        - 外层写成 \`while stack\`：开始时栈为空直接退出；
        - 用 \`stack[-1]\` 查看而不弹出，导致同一节点重复访问；
        - 递归写法在深树上 \`RecursionError\`（本题隐藏测试专门检查）。
      `,
      complexity: { time: 'O(n)', space: 'O(h)' },
      hints: [
        md`递归时系统帮你保存了“回来以后还要处理的祖先”。迭代时用一个栈手动保存它们。`,
        md`中序的第一个节点是最左下的节点：从根出发一路 \`cur = cur.left\`，沿途压栈。`,
        md`弹出栈顶 \`cur\` 并输出后，下一步去处理 \`cur.right\`；外层循环条件是 \`while cur or stack\`。`,
      ],
      setup: PY_TREE + '\n' + REF,
      tests: [
        { name: '示例 1', code: py`assert inorder_traversal(list_to_tree([1, None, 2, 3])) == [1, 3, 2]` },
        {
          name: '空树与单节点',
          code: py`
            assert inorder_traversal(None) == [], "空树应返回 []"
            assert inorder_traversal(list_to_tree([7])) == [7]
          `,
        },
        {
          name: '示例树',
          code: py`
            got = inorder_traversal(list_to_tree([1, 2, 3, 4, 5, None, 6, None, None, 7]))
            assert got == [4, 2, 7, 5, 1, 3, 6], f"期望 [4, 2, 7, 5, 1, 3, 6]，得到 {got}"
          `,
        },
        {
          name: '随机对拍（含不修改树）',
          hidden: true,
          code: py`
            rnd = random.Random(94)
            for _ in range(200):
                t = _rand_tree(rnd, rnd.randint(0, 60))
                before = tree_to_list(t)
                exp, got = _ref_inorder(t), inorder_traversal(t)
                assert got == exp, f"树 {before}：期望 {exp}，得到 {got}"
                assert tree_to_list(t) == before, "遍历后树的结构被修改了"
          `,
        },
        {
          name: '不能用递归：深 300 的左斜链',
          hidden: true,
          code: py`
            vals = list(range(300))
            t = _chain(vals, "left")
            try:
                try:
                    with _NoDeepRecursion(30):
                        got = inorder_traversal(t)
                except RecursionError:
                    raise AssertionError("检测到深度递归（RecursionError）：请用显式栈实现")
                assert got == vals[::-1], "左斜链的中序应为从叶到根"
            finally:
                _free_tree(t)
          `,
        },
        {
          name: '规模：1500 个节点',
          hidden: true,
          code: py`
            import time
            rnd = random.Random(7)
            t = _rand_tree(rnd, 1500, -10**6, 10**6)
            try:
                t0 = time.perf_counter()
                got = inorder_traversal(t)
                dt = time.perf_counter() - t0
                assert got == _ref_inorder(t), "1500 个节点的结果不正确"
                assert dt < 0.5, f"用时 {dt:.2f}s，应为 O(n)"
            finally:
                _free_tree(t)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-morris',
          q: md`能做到 $O(1)$ 额外空间吗？`,
          a: md`**Morris 遍历**：对当前节点 \`cur\`，找到它左子树的最右节点 \`pre\`（中序前驱）。若 \`pre.right\` 为空，就让它临时指向 \`cur\`（线索），然后进入左子树；若已经指向 \`cur\`，说明左子树处理完了，断开线索、输出 \`cur\`、转向右子树。每条边最多走 3 次，时间 $O(n)$、空间 $O(1)$；代价是遍历过程中**临时修改了树**，不适合并发读。`,
        },
        {
          id: 'f-bst-k',
          q: md`在这个迭代框架上，如何求 BST 中第 k 小的元素（LC 230）？`,
          a: md`中序遍历是升序，每弹出一个节点计数加一，数到 k 立刻返回，时间 $O(h + k)$，不必遍历整棵树——这正是迭代写法比递归方便的地方（可以随时中断）。如果频繁查询且树会修改，可以在节点上维护子树大小，做到 $O(h)$。`,
        },
      ],
    },
    {
      id: 'c-zigzag',
      title: '二叉树的锯齿形层序遍历',
      difficulty: 'medium',
      source: 'LeetCode 103',
      timeLimitMin: 15,
      description: md`
        给定二叉树的根节点 \`root\`，返回节点值的**锯齿形层序遍历**：第 0 层从左往右，第 1 层从右往左，以此交替。

        \`\`\`python
        def zigzag_level_order(root: TreeNode | None) -> list[list[int]]: ...
        \`\`\`

        运行环境已提供 \`TreeNode\`、\`list_to_tree\`、\`tree_to_list\`；\`collections.deque\` 可直接 import。

        **示例**
        - \`[3, 9, 20, None, None, 15, 7]\` → \`[[3], [20, 9], [15, 7]]\`
        - \`[1]\` → \`[[1]]\`
        - \`[]\` → \`[]\`

        **约束**：$0 \le n \le 1500$，要求 $O(n)$ 时间。
      `,
      starter: STARTER_HEAD + '\n\n' + py`
        from collections import deque

        def zigzag_level_order(root: TreeNode | None) -> list[list[int]]:
            # TODO: BFS + 按层切分；奇数层反转
            raise NotImplementedError
      `,
      solution: py`
        from collections import deque

        def zigzag_level_order(root: TreeNode | None) -> list[list[int]]:
            if root is None:
                return []
            res, q = [], deque([root])
            while q:
                level = deque()
                left_to_right = len(res) % 2 == 0
                for _ in range(len(q)):          # 只处理当前层
                    node = q.popleft()
                    if left_to_right:
                        level.append(node.val)
                    else:
                        level.appendleft(node.val)   # 奇数层往左插，等价于反转
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
                res.append(list(level))
            return res
      `,
      explanation: md`
        **思路**：标准层序遍历（\`for _ in range(len(q))\` 切层），只改“值怎么放进本层列表”：
        偶数层 \`append\`，奇数层 \`appendleft\`（或先 \`append\` 再整体 \`reverse\`）。**孩子入队顺序始终是先左后右**，不要试图改变入队顺序——那会把下一层的顺序也搞乱。

        **复杂度**：每个节点入队出队一次，\`appendleft\` 是 $O(1)$，总时间 $O(n)$；空间 $O(w)$（最宽层）。

        **常见错误**
        - 交替反转**入队顺序**：第 2 层的顺序会被第 1 层的反向入队连带打乱；
        - 用 \`list.insert(0, x)\` 往左插：单次 $O(w)$，最坏 $O(n^2)$；
        - 用 \`list.pop(0)\` 当队列。
      `,
      complexity: { time: 'O(n)', space: 'O(w)' },
      hints: [
        md`先写出 LC 102 的普通层序遍历：\`while q\` 里套 \`for _ in range(len(q))\`。`,
        md`当前层号就是 \`len(res)\`；奇数层的结果需要从右往左。`,
        md`不要改孩子的入队顺序；只在放进 \`level\` 时决定 \`append\` 还是 \`appendleft\`（或最后 \`level.reverse()\`）。`,
      ],
      setup: PY_TREE + '\n' + REF,
      tests: [
        { name: '示例 1', code: py`assert zigzag_level_order(list_to_tree([3, 9, 20, None, None, 15, 7])) == [[3], [20, 9], [15, 7]]` },
        {
          name: '单节点与空树',
          code: py`
            assert zigzag_level_order(list_to_tree([1])) == [[1]]
            assert zigzag_level_order(None) == [], "空树应返回 []"
          `,
        },
        {
          name: '四层示例树',
          code: py`
            got = zigzag_level_order(list_to_tree([1, 2, 3, 4, 5, None, 6, None, None, 7]))
            exp = [[1], [3, 2], [4, 5, 6], [7]]
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '满二叉树（第 2 层不能被打乱）',
          hidden: true,
          code: py`
            t = list_to_tree(list(range(1, 16)))
            exp = [[1], [3, 2], [4, 5, 6, 7], [15, 14, 13, 12, 11, 10, 9, 8]]
            got = zigzag_level_order(t)
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(103)
            for _ in range(200):
                t = _rand_tree(rnd, rnd.randint(0, 50))
                exp, got = _ref_zigzag(t), zigzag_level_order(t)
                assert got == exp, f"树 {tree_to_list(t)}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：1500 个节点',
          hidden: true,
          code: py`
            import time
            rnd = random.Random(11)
            t = _rand_tree(rnd, 1500)
            try:
                t0 = time.perf_counter()
                got = zigzag_level_order(t)
                dt = time.perf_counter() - t0
                assert got == _ref_zigzag(t), "1500 个节点的结果不正确"
                assert dt < 0.5, f"用时 {dt:.2f}s，应为 O(n)"
            finally:
                _free_tree(t)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-dfs',
          q: md`能用 DFS 做层序分组吗？`,
          a: md`可以：DFS 时带上深度 \`d\`，第一次到达深度 \`d\` 时 \`res.append([])\`，然后把值放进 \`res[d]\`（锯齿形则奇数层 \`appendleft\`，需要 deque）。前序遍历保证同一层内从左到右到达。时间同样 $O(n)$，空间变为 $O(h)$ 的递归栈加输出。`,
        },
        {
          id: 'f-right-view',
          q: md`如何改成 LC 199「右视图」？`,
          a: md`层序遍历时只记录每层**最后一个**弹出的节点；或者 DFS 时先右后左，每层第一次到达的节点就是右视图。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-iter',
      freq: 3,
      q: md`不用递归，怎么写二叉树的前序、中序、后序遍历？`,
      a: md`
        **结论**：用显式栈模拟调用栈，我习惯“标记法”统一三种遍历。

        **原因**：递归里一个节点会被经过三次，中序 / 后序需要等子树处理完才输出。栈里放 \`(node, visited)\`：第一次弹出时按访问顺序的**逆序**把左右孩子和 \`(node, True)\` 压栈；第二次弹出时输出。三种遍历只差 \`(node, True)\` 压栈的位置（前序最后压、中序夹在中间、后序最先压）。每个节点进出栈两次，时间 $O(n)$，空间 $O(h)$。

        **延伸**：中序也会写“指针一路向左压栈、弹出后转右”的经典写法；后序可以用“根右左再反转”，但要说明它只改变了输出顺序，不能在后序位置利用子树结果。要 $O(1)$ 空间则用 Morris 遍历。
      `,
    },
    {
      id: 'qa-bfs-dfs',
      freq: 2,
      q: md`BFS 和 DFS 遍历二叉树，空间复杂度各是多少？什么时候选 BFS？`,
      a: md`
        **结论**：DFS 是 $O(h)$（树高），BFS 是 $O(w)$（最宽层）。

        **原因**：DFS 的栈里只有当前路径上的节点；BFS 的队列里是一整层。平衡树 $h = O(\log n)$、$w = O(n)$（最底层约 $n/2$），所以 DFS 省空间；退化成链时反过来，DFS $O(n)$、BFS $O(1)$。

        **延伸**：需要“按层”信息（层序输出、右视图、最小深度——BFS 遇到第一个叶子就能停）时用 BFS；需要子树信息（高度、路径和、LCA）时用 DFS 后序。
      `,
    },
    {
      id: 'qa-recursion',
      freq: 2,
      q: md`递归遍历一棵很深的树，Python 里会出什么问题？怎么处理？`,
      a: md`
        **结论**：CPython 默认递归上限 1000，深树会抛 \`RecursionError\`；改成显式栈的迭代，或 BFS。

        **原因**：每层递归占一个 Python 栈帧，CPython 不做尾递归优化。\`sys.setrecursionlimit\` 可以调大，但真实的 C 栈 / 线程栈有限，过深可能直接段错误；在 Pyodide、多线程等栈较小的环境里更脆弱。

        **延伸**：面试中写完递归版本，主动补一句“如果树可能退化成链，我会改成迭代”，并能当场写出迭代版本，是很常见的加分点。
      `,
    },
    {
      id: 'qa-unique',
      freq: 2,
      q: md`哪些遍历序列组合能唯一确定一棵二叉树？为什么前序 + 后序不行？`,
      a: md`
        **结论**：节点值互不相同时，前序+中序、后序+中序、层序+中序都可以；前序+后序一般不行。

        **原因**：前序 / 后序 / 层序都能确定“谁是根”，但只有中序能把剩下的节点**切成左右子树**。前序+后序在某个节点只有一个孩子时，无法判断这个孩子在左还是右：\`1→左2\` 与 \`1→右2\` 的前序、后序完全相同。

        **延伸**：如果树是满二叉树（每个节点 0 或 2 个孩子），前序+后序也能唯一确定（LC 889）。有重复值时任何组合都可能不唯一。
      `,
    },
  ],

  cheatsheet: md`
    **统一显式栈（标记法）**
    \`\`\`python
    stack = [(root, False)] if root else []
    while stack:
        node, vis = stack.pop()
        if vis: res.append(node.val); continue
        if order == "post": stack.append((node, True))
        if node.right: stack.append((node.right, False))
        if order == "in":   stack.append((node, True))
        if node.left:  stack.append((node.left, False))
        if order == "pre":  stack.append((node, True))
    \`\`\`

    **层序分组**：\`while q:\` → \`for _ in range(len(q)):\` → \`q.popleft()\`

    | 遍历 | 顺序 | 用途 |
    |---|---|---|
    | 前序 | 根左右 | 序列化、向下传参 |
    | 中序 | 左根右 | BST 升序 |
    | 后序 | 左右根 | 子树信息（高度 / 直径 / LCA） |
    | 层序 | 按层 | 分层、最小深度、右视图 |

    - 时间 $O(n)$；DFS 空间 $O(h)$，BFS 空间 $O(w)$
    - 队列用 \`deque\`，别用 \`list.pop(0)\`
    - Python 递归上限 1000，深树改迭代
    - 唯一确定树：中序 + (前序 / 后序 / 层序)
  `,
})
