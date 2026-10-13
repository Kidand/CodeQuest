/**
 * algo-graph-search BFS/DFS 与拓扑排序（必考）
 * 网格洪水填充 → 多源 BFS → Kahn 拓扑排序与判环 → DFS 三色标记 → autograd 的反向拓扑序
 */
import { defineLevel, md, py } from '@/content/helpers'
import { FloodFillViz } from '../viz/FloodFillViz'
import { MultiSourceViz } from '../viz/MultiSourceViz'
import { KahnViz } from '../viz/KahnViz'
import { AutogradViz } from '../viz/AutogradViz'

const GRID_SETUP = py`
  import random, time
  from collections import deque

  def _ref_islands(grid):
      g = [row[:] for row in grid]
      m, n = len(g), len(g[0])
      cnt = 0
      for i in range(m):
          for j in range(n):
              if g[i][j] != "1":
                  continue
              cnt += 1
              g[i][j] = "0"
              q = deque([(i, j)])
              while q:
                  r, c = q.popleft()
                  for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                      if 0 <= nr < m and 0 <= nc < n and g[nr][nc] == "1":
                          g[nr][nc] = "0"
                          q.append((nr, nc))
      return cnt

  def _rand_grid(rnd, m, n, p):
      return [["1" if rnd.random() < p else "0" for _ in range(n)] for _ in range(m)]
`

const ORANGE_SETUP = py`
  import random
  from collections import deque

  def _ref_rot(grid):
      g = [row[:] for row in grid]
      m, n = len(g), len(g[0])
      q = deque((i, j) for i in range(m) for j in range(n) if g[i][j] == 2)
      fresh = sum(row.count(1) for row in g)
      t = 0
      while q and fresh:
          for _ in range(len(q)):
              r, c = q.popleft()
              for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                  if 0 <= nr < m and 0 <= nc < n and g[nr][nc] == 1:
                      g[nr][nc] = 2
                      fresh -= 1
                      q.append((nr, nc))
          t += 1
      return t if fresh == 0 else -1
`

const TOPO_SETUP = py`
  import random, time
  from collections import deque

  def _check_order(n, pre, order):
      """校验 order 是否为合法拓扑序；返回错误信息或 None。"""
      if sorted(order) != list(range(n)):
          return f"结果应是 0..{n - 1} 的一个排列，得到 {order[:20]}{'...' if len(order) > 20 else ''}"
      pos = {c: i for i, c in enumerate(order)}
      for a, b in pre:
          if pos[b] > pos[a]:
              return f"课程 {a} 依赖 {b}，但 {b} 排在了 {a} 后面"
      return None

  def _has_cycle(n, pre):
      g = [[] for _ in range(n)]
      indeg = [0] * n
      for a, b in pre:
          g[b].append(a)
          indeg[a] += 1
      q = deque(i for i in range(n) if indeg[i] == 0)
      seen = 0
      while q:
          u = q.popleft()
          seen += 1
          for v in g[u]:
              indeg[v] -= 1
              if indeg[v] == 0:
                  q.append(v)
      return seen < n

  def _layered_dag(rnd, n, layers, m):
      """分层 DAG：边只从低层指向高层，最长路径 ≤ layers（避免递归 DFS 过深）。"""
      layer = sorted(rnd.randrange(layers) for _ in range(n))
      perm = list(range(n))
      rnd.shuffle(perm)
      pre = set()
      while len(pre) < m:
          u, v = rnd.randrange(n), rnd.randrange(n)
          if layer[u] < layer[v]:
              pre.add((perm[v], perm[u]))       # 先修 u 才能修 v
      return [list(p) for p in pre], perm, layer

  class _CountedList(list):
      """只读包装：统计被完整遍历 / 下标访问的次数，超出预算立即失败（避免超时）。"""
      def __init__(self, data, max_iters=6):
          super().__init__(data)
          self.iters = 0
          self.gets = 0
          self.max_iters = max_iters
      def __iter__(self):
          self.iters += 1
          if self.iters > self.max_iters:
              raise AssertionError(f"prerequisites 被完整遍历了 {self.iters} 次——是不是每一轮都在重新扫描全部边？请先建邻接表，做到 O(V+E)")
          return super().__iter__()
      def __getitem__(self, i):
          self.gets += 1
          if self.gets > self.max_iters * max(1, len(self)):
              raise AssertionError("按下标访问 prerequisites 的次数过多，复杂度超过 O(V+E)")
          return super().__getitem__(i)
`

const AUTOGRAD_SETUP = py`
  import math, random

  def _nodes_of(root):
      seen, st, out = set(), [root], []
      while st:
          v = st.pop()
          if id(v) in seen:
              continue
          seen.add(id(v))
          out.append(v)
          st.extend(v._prev)
      return out

  def _free_graph(root):
      # 断开 _prev 与闭包（闭包引用了节点自身，形成环），避免大图在释放时递归析构
      for v in _nodes_of(root):
          v._prev = ()
          v._backward = None

  def _instrument(root):
      """给每个节点的 _backward 包一层计数：同一节点被执行第 2 次就立即失败。"""
      calls = {}
      for v in _nodes_of(root):
          f = v._backward
          def wrapped(f=f, v=v):
              calls[id(v)] = calls.get(id(v), 0) + 1
              if calls[id(v)] > 1:
                  raise AssertionError(
                      f"节点（data={v.data:.4g}, op={getattr(v, '_op', '?')!r}）的 _backward 被执行了第 2 次："
                      "每个节点只能反传一次，而且要等它的梯度收齐——请先求拓扑序")
              f()
          v._backward = wrapped
      return calls

  def _rand_program(rnd, n_in, n_ops):
      prog = []
      for t in range(n_ops):
          k = n_in + t
          op = rnd.choice(["+", "*", "tanh", "pow", "+c", "*c"])
          i = rnd.randrange(max(0, k - 3), k)      # 倾向于用最近的节点：链更深、共享更多
          j = rnd.randrange(k)
          prog.append((op, i, j, round(rnd.uniform(-2, 2), 3)))
      return prog

  def _eval(prog, xs, V=None):
      vals = [V(x) for x in xs] if V else list(xs)
      for op, i, j, cst in prog:
          a, b = vals[i], vals[j]
          if op == "+":
              r = a + b
          elif op == "*":
              r = a * b
          elif op == "tanh":
              r = a.tanh() if V else math.tanh(a)
          elif op == "pow":
              r = a ** 2
          elif op == "+c":
              r = a + cst
          else:
              r = a * cst
          vals.append(r)
      out = vals[-1] + vals[-2] + vals[-3]     # 多个分支都要拿到梯度
      return out, vals[:len(xs)]

  def _num_grad(prog, xs, h=1e-6):
      g = []
      for k in range(len(xs)):
          xp, xm = list(xs), list(xs)
          xp[k] += h
          xm[k] -= h
          g.append((_eval(prog, xp)[0] - _eval(prog, xm)[0]) / (2 * h))
      return g
`

const VALUE_CLASS_HEAD = py`
  import math

  class Value:
      """标量自动微分节点。
      data：前向值；grad：∂L/∂self；_prev：输入节点元组；
      _backward：闭包，把 self.grad 按链式法则“累加”到各个输入的 grad 上。"""

      def __init__(self, data, _prev=(), _op=""):
          self.data = float(data)
          self.grad = 0.0
          self._prev = tuple(_prev)
          self._op = _op
          self._backward = lambda: None

      def __add__(self, other):
          other = other if isinstance(other, Value) else Value(other)
          out = Value(self.data + other.data, (self, other), "+")
          def _backward():
              self.grad += out.grad          # 用 +=：同一节点可能被多处使用
              other.grad += out.grad
          out._backward = _backward
          return out

      def __pow__(self, k):
          assert isinstance(k, (int, float)), "只支持常数指数"
          out = Value(self.data ** k, (self,), f"**{k}")
          def _backward():
              self.grad += k * self.data ** (k - 1) * out.grad
          out._backward = _backward
          return out
`

const VALUE_CLASS_TAIL = py`
      # ---- 语法糖（已提供） ----
      def __radd__(self, other):
          return self + other

      def __rmul__(self, other):
          return self * other

      def __neg__(self):
          return self * -1

      def __sub__(self, other):
          return self + (-other)

      def __repr__(self):
          return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'grid',
      title: '网格图上的 BFS / DFS：岛屿数量',
      body: md`
        网格题是图论题最常见的伪装：每个格子是一个顶点，上下左右相邻的格子之间有边（**4 邻接**，对角线不算）。
        LC 200 岛屿数量就是求“陆地顶点构成的**连通分量**个数”：

        1. 逐格扫描，遇到**还没访问过的陆地**，计数 +1；
        2. 从它出发 BFS 或 DFS，把同一座岛的陆地全部标记为已访问（“淹掉”）；
        3. 每个格子最多被访问一次，时间 $O(mn)$。

        标记 visited 的时机是第一个易错点：**入队（入栈）时就标记**。如果等到出队时才标记，同一个格子可能在被处理前被多个邻居重复入队，
        队列膨胀，最坏情况下变慢好几倍（下面打开“出队时才标记”的开关数一数重复次数）。

        BFS 用队列、DFS 用栈，代码只差 \`popleft()\` 与 \`pop()\` 一处；对“数连通分量”来说两者等价，访问顺序不同而已。
      `,
      viz: FloodFillViz,
      after: md`
        > [!WARN] 递归 DFS 的深度
        > LeetCode 中网格最大 $300 \times 300$。全是陆地时递归 DFS 的深度可达 $9 \times 10^4$，远超 Python 默认的递归上限 1000。
        > 面试写 Python 时优先 BFS 或显式栈；写递归要主动说明这个风险。
      `,
      checkpoint: {
        id: 'cp-grid',
        type: 'numeric',
        prompt: md`网格 \`[["1","0","1"],["0","1","0"],["1","0","1"]]\`（4 邻接）中有几座岛屿？`,
        answer: 5,
        tolerance: 0,
        explanation: md`五个 1 两两只在对角线上相邻，而对角线不算相邻，所以每个 1 都是一座独立的岛，共 5 座。若按 8 邻接算则只有 1 座——读题时要先确认邻接方式。`,
      },
    },
    {
      id: 'multi',
      title: '多源 BFS：所有源点一起出发',
      body: md`
        BFS 按层扩展，第 $k$ 层恰好是“离源点距离为 $k$”的点，所以它能求**无权图的最短路**。

        LC 994 腐烂的橘子：每分钟，烂橘子把上下左右的新鲜橘子变烂，问多久全烂。有多个烂橘子同时扩散——
        把**所有源点一起放进第 0 层**就行了，相当于加一个虚拟的“超级源点”连向它们，再做一次普通 BFS：

        - 每一层 = 一分钟（又是 \`for _ in range(len(q))\` 切层）；
        - 答案 = 层数；BFS 结束后还有新鲜橘子 ⇒ 不连通，返回 $-1$。

        对每个烂橘子分别 BFS 再取最小，是 $O(k \cdot mn)$；多源 BFS 每个格子只入队一次，$O(mn)$。
        同一套路还能解 LC 542（每个格子到最近 0 的距离）、LC 1162（离陆地最远的海洋）。
      `,
      viz: MultiSourceViz,
      after: md`
        > [!TIP] 细节：循环条件写 \`while q and fresh\`
        > 最后一批橘子变烂后队列里还有它们，如果条件只写 \`while q\`，会再多跑一层空的扩展，分钟数多算 1。
        > 另一种写法是每层结束后检查本层是否有新腐烂的橘子。
      `,
      checkpoint: {
        id: 'cp-multi',
        type: 'numeric',
        prompt: md`网格 \`[[2,1,1],[1,1,0],[0,1,1]]\`（2 烂、1 新鲜、0 空）全部腐烂需要几分钟？`,
        answer: 4,
        tolerance: 0,
        explanation: md`唯一的源在左上角。第 1 分钟：(0,1)、(1,0)；第 2 分钟：(0,2)、(1,1)；第 3 分钟：(2,1)；第 4 分钟：(2,2)。答案是 4（LC 994 示例 1）。`,
      },
    },
    {
      id: 'topo',
      title: '拓扑排序：Kahn 算法与判环',
      body: md`
        有向无环图（DAG）的**拓扑序**：把所有点排成一列，使每条边 $u \to v$ 都满足 $u$ 在 $v$ 前面。课程表（LC 207 / 210）、
        任务调度、编译依赖、深度学习计算图的执行顺序，都是拓扑排序。

        **Kahn 算法**（BFS 版）：

        1. 统计每个点的入度；入度为 0 的点没有任何前置条件，入队；
        2. 弹出 $u$ 放进结果，把 $u$ 的出边“删掉”：每个后继 $v$ 的入度减 1，减到 0 就入队；
        3. 队列空时，若输出了全部 $n$ 个点，结果就是拓扑序；**否则图中有环**。

        为什么能判环：环上的每个点都至少有一条来自环上前驱的入边，而环上没有任何一个点能先被输出，
        所以它们的入度永远减不到 0。时间 $O(V + E)$。

        > [!WARN] 边的方向
        > LC 207 中 \`[a, b]\` 表示“学 a 之前必须先学 b”，边是 **b → a**。方向建反了，得到的就是逆拓扑序。
      `,
      viz: KahnViz,
      after: md`
        拓扑序**一般不唯一**：同一时刻队列里有多个点时，先弹谁都可以。拓扑序唯一 ⇔ 每一步队列里恰好只有一个点（等价于 DAG 中存在一条经过所有点的路径）。
        需要字典序最小的拓扑序时，把队列换成**最小堆**。
      `,
      checkpoint: {
        id: 'cp-topo',
        type: 'judge',
        prompt: md`Kahn 算法结束时，如果输出的点数小于 $n$，则剩下的每一个点都一定在某个环上。`,
        answer: false,
        explanation: md`剩下的点要么在环上，要么是环的**下游**（依赖环上的点）。例如 $1 \to 2 \to 1$ 加上 $2 \to 3$，点 3 不在环上，但它的入度也永远减不到 0。能确定的只是“图中存在环”。`,
      },
    },
    {
      id: 'dfs-topo',
      title: 'DFS 版拓扑排序：三色标记与逆后序',
      body: md`
        另一种做法是 DFS。给每个点三种颜色：

        | 颜色 | 含义 |
        |---|---|
        | 白 0 | 还没访问 |
        | 灰 1 | 正在访问：在当前递归栈上 |
        | 黑 2 | 已完成：它能到达的点都访问完了 |

        - **判环**：沿边走到一个**灰色**点，说明找到了回到当前路径的边（回边），有环。走到黑色点没问题——那只是另一条路径汇合过来。
        - **拓扑序**：按“完成（变黑）”的时间记录点，得到**后序**，再整体反转即为拓扑序。

        为什么逆后序正确：对任意边 $u \to v$，当 $u$ 完成时，$v$ 一定已经完成（$v$ 若是白色会在 $u$ 的递归里先完成；若是黑色本来就完成了；
        不可能是灰色，否则有环）。所以 $v$ 在后序中排在 $u$ 前面，反转后 $u$ 在 $v$ 前面。

        有向图判环**不能**只用 visited 两色：两条路径汇合到同一个已访问的点（菱形）并不是环。无向图判环则不同——用两色 + 记录父节点，或直接用并查集（下一关）。
      `,
      checkpoint: {
        id: 'cp-dfs-topo',
        type: 'single',
        prompt: md`用 DFS 求拓扑序，正确的做法是：`,
        options: ['按前序（第一次访问）记录', '按后序（完成）记录，再反转', '按后序记录，不反转', '只要访问过就跳过，按访问顺序记录'],
        answer: 1,
        explanation: md`后序中每个点都排在它能到达的所有点**之后**，正好与拓扑序相反，所以要反转。前序不行：对菱形 $a \to b \to d$、$a \to c \to d$，前序可能是 $a, b, d, c$，$d$ 排在了它的前驱 $c$ 之前。`,
      },
    },
    {
      id: 'llm',
      title: '大模型岗场景：autograd 按反向拓扑序传梯度',
      body: md`
        前向计算会构造一张**计算图**（DAG）：每个张量 / 标量是一个节点，记录它由哪些输入、经过什么运算得到。反向传播要求出 $\partial L / \partial x$：

        - 每个节点 $v$ 知道自己的**局部导数**，\`_backward\` 做的事是 $\text{input.grad} \mathrel{+}= \frac{\partial v}{\partial \text{input}} \cdot v.\text{grad}$（链式法则）；
        - 一个节点可能被**多处使用**（残差连接、权重共享、同一个激活被两层读取），它的梯度是所有使用处的贡献**之和**，所以必须用 \`+=\`；
        - 只有当 $v$ 的**所有消费者**都已经反传完，$v.\text{grad}$ 才收齐，才能继续往输入传——这正是“**反向拓扑序**”。

        Karpathy 的 micrograd 用 DFS 后序求拓扑序，再反向遍历调用 \`_backward\`；PyTorch 的 autograd engine 则先统计每个节点的依赖数
        （\`compute_dependencies\`），某个节点执行完就把下游节点的依赖数减一，减到 0 才放进 ready queue——本质上就是 **Kahn 算法**。
      `,
      viz: AutogradViz,
      after: md`
        > [!INTERVIEW] 为什么不能“收到梯度就立刻往下传”？
        > 两个原因。① **正确性**：若在梯度没收齐时就反传（上图 BFS 顺序），后到的那部分梯度再也传不下去；
        > ② **复杂度**：若每收到一份梯度就递归往下传一次，结果虽然对（线性可叠加），但调用次数等于**路径数**——
        > 对 \`y = y + y\` 重复 40 次的图，节点只有 41 个，路径却有 $2^{40}$ 条。按拓扑序，每个节点只反传一次，$O(V+E)$。

        > [!NOTE] 和训练代码的联系
        > PyTorch 的 \`.grad\` 也是“累加”语义：多次 \`backward()\` 的梯度会叠加，这就是每步训练前要 \`optimizer.zero_grad()\` 的原因（梯度累积正是利用了这一点）。
      `,
      checkpoint: {
        id: 'cp-llm',
        type: 'single',
        prompt: md`计算图 $c = a \cdot b,\ d = c + b,\ L = d \cdot c$，其中 $a = 2, b = -3$。$\partial L / \partial c$（把两条路径都算上）等于：`,
        options: ['-9', '-6', '-15', '54'],
        answer: 2,
        explanation: md`$c = -6,\ d = -9$。$L$ 直接对 $c$ 的偏导是 $d = -9$；经过 $d$ 的路径贡献 $\frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial c} = c \cdot 1 = -6$。总梯度 $-15$。只算一条路径就会得到 A 或 B。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                           */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-bfs-shortest',
      type: 'single',
      difficulty: 1,
      prompt: md`在迷宫（无权网格）里求从起点到终点的**最少步数**，最合适的算法是：`,
      options: ['DFS，第一次到达终点时的步数就是答案', 'BFS，第一次到达终点时的层数就是答案', 'Dijkstra（必须用堆）', '拓扑排序'],
      answer: 1,
      explanation: md`BFS 按距离从小到大逐层扩展，第一次到达终点时的层数就是最短距离。DFS 第一次到达终点的路径不一定最短；Dijkstra 也能做但多一个 $\log$，边权全为 1 时就退化成 BFS；拓扑排序只适用于 DAG，与最短步数无关。`,
    },
    {
      id: 'q-kahn-cycle',
      type: 'judge',
      difficulty: 1,
      prompt: md`Kahn 算法结束后，若输出的节点数小于 $n$，则图中一定存在环。`,
      answer: true,
      explanation: md`若图是 DAG，任何时刻未输出的点构成的子图仍是 DAG，必有入度为 0 的点，队列不会提前变空，最终一定能输出全部 $n$ 个点。反过来，输出不足 $n$ 就说明存在环。`,
    },
    {
      id: 'q-multi-source',
      type: 'multi',
      difficulty: 2,
      prompt: md`以下哪些题目适合用**多源 BFS**（初始时把多个点一起放进队列）？（多选）`,
      options: ['LC 994 腐烂的橘子', 'LC 542 01 矩阵（每格到最近 0 的距离）', 'LC 1162 地图分析（离陆地最远的海洋格子）', 'LC 200 岛屿数量'],
      answer: [0, 1, 2],
      explanation: md`前三题都是“到**最近的某一类点**的距离”：把这一类点全部作为第 0 层同时出发。LC 200 求的是连通分量个数，每座岛单独洪水填充即可，与多源最短距离无关。`,
    },
    {
      id: 'q-orange',
      type: 'numeric',
      difficulty: 2,
      prompt: md`网格 \`[[2,1,1,0],[0,1,0,1],[1,1,1,2]]\`（2 烂、1 新鲜、0 空）全部腐烂需要几分钟？无法全部腐烂请填 -1。`,
      answer: 3,
      tolerance: 0,
      explanation: md`两个源 (0,0) 与 (2,3) 同时出发。第 1 分钟：(0,1)、(1,3)、(2,2)；第 2 分钟：(0,2)、(1,1)、(2,1)；第 3 分钟：(2,0)。全部腐烂，答案 3。`,
    },
    {
      id: 'q-kahn-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        \`n = 5\`，\`prerequisites = [[0,4],[1,0],[2,4],[1,2],[3,1]]\`（\`[a, b]\` 表示先修 b 才能修 a）。
        按 Kahn 算法（初始队列按编号从小到大入队，邻接表按 prerequisites 的顺序建立），输出顺序是 ____(1)____（逗号分隔）。
      `,
      blanks: [{ label: '(1)', answers: ['4,0,2,1,3', '[4,0,2,1,3]'] }],
      explanation: md`边：4→0、0→1、4→2、2→1、1→3。入度：0:1、1:2、2:1、3:1、4:0。队列 [4] → 弹 4，0、2 入度归零入队 → 弹 0，1 的入度变 1 → 弹 2，1 归零入队 → 弹 1，3 入队 → 弹 3。结果 4,0,2,1,3。把边建反（a→b）会得到逆序。`,
    },
    {
      id: 'q-kahn-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 Kahn 拓扑排序的步骤按正确顺序排列：`,
      items: [
        '按 `[a, b]` 建边 b → a，并统计每个点的入度',
        '把所有入度为 0 的点放进队列',
        '弹出 u 放进结果，u 的每个后继入度减 1，减到 0 的入队',
        '队列为空时检查结果长度是否等于 n（不等则有环）',
      ],
      explanation: md`建图与入度统计必须在初始化队列之前完成；主循环每次只处理“前置条件全部满足”的点；最后用输出数量判环。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把题目与最直接的解法配对：`,
      pairs: [
        ['LC 200 岛屿数量', '逐格扫描 + 洪水填充'],
        ['LC 210 课程表 II', 'Kahn 拓扑排序'],
        ['LC 994 腐烂的橘子', '多源 BFS 按层计时'],
        ['LC 1091 二进制矩阵最短路径', '单源 BFS（8 邻接）'],
      ],
      explanation: md`岛屿是连通分量计数；课程表是 DAG 排序 + 判环；橘子是多个源同时扩散；1091 是普通的无权最短路，只是邻接方式变成 8 个方向。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`用 DFS 三色标记判断有向图是否有环，拼出代码（\`graph[u]\` 是 u 的后继列表；有 2 行干扰项）：`,
      lines: [
        'def has_cycle(n, graph):',
        '    color = [0] * n                 # 0 白 / 1 灰 / 2 黑',
        '    def dfs(u):',
        '        color[u] = 1                # 进入：灰',
        '        for v in graph[u]:',
        '            if color[v] == 1 or (color[v] == 0 and dfs(v)):',
        '                return True',
        '        color[u] = 2                # 离开：黑',
        '        return False',
        '    return any(color[u] == 0 and dfs(u) for u in range(n))',
      ],
      distractors: ['            if color[v] != 0 or dfs(v):', '        color[u] = 0                # 离开：恢复白色'],
      explanation: md`只有指向**灰色**（仍在当前递归栈上）的边才是回边；黑色点是另一条已完成的路径，不构成环——第一个干扰项把黑色也当成环，菱形 DAG 会被误判。离开时应标黑而不是恢复白色，否则同一个点会被反复搜索，复杂度可能变成指数级。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面是 micrograd 风格的标量 autograd 片段。有两行错误，会导致梯度算错，点出它们：`,
      code: py`
        class Value:
            def __add__(self, other):
                out = Value(self.data + other.data, (self, other))
                def _backward():
                    self.grad = out.grad
                    other.grad += out.grad
                out._backward = _backward
                return out
            def backward(self):
                topo, seen = [], set()
                def build(v):
                    if v not in seen:
                        seen.add(v)
                        for u in v._prev:
                            build(u)
                        topo.append(v)
                build(self)
                self.grad = 1.0
                for v in topo:
                    v._backward()
      `,
      bugLines: [5, 19],
      explanation: md`第 5 行应为 \`self.grad += out.grad\`：一个节点被多处使用时，后到的梯度会覆盖先到的（例如 \`x + x\` 的梯度应为 2，这里只得到 1）。第 19 行应为 \`for v in reversed(topo)\`：\`topo\` 是后序（输入在前），要从输出 L 开始反传。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`仍是网格 \`[[2,1,1,0],[0,1,0,1],[1,1,1,2]]\`。点出**恰好在第 2 分钟**变烂的格子：`,
      items: ['2', '1', '1', '0', '0', '1', '0', '1', '1', '1', '1', '2'],
      cols: 4,
      rowLabels: ['r0', 'r1', 'r2'],
      colLabels: ['c0', 'c1', 'c2', 'c3'],
      answer: [2, 5, 9],
      explanation: md`第 1 分钟变烂的是 (0,1)、(1,3)、(2,2)；它们在第 2 分钟分别感染 (0,2)、(1,1)（来自 (0,1)），以及 (2,1)（来自 (2,2)）；(1,3) 的邻居 (0,3) 是空格。(2,0) 要到第 3 分钟。`,
    },
    {
      id: 'q-exp',
      type: 'single',
      difficulty: 3,
      prompt: md`
        有人这样写反向传播：每个节点收到梯度后立刻递归调用输入节点的反传。
        \`\`\`python
        def naive_backward(v):
            v._backward()            # 把 v.grad 累加给输入
            for u in v._prev:
                naive_backward(u)
        \`\`\`
        对 \`x = Value(1.0); y = x\`，再执行 30 次 \`y = y + y\` 得到的图（假设 \`_backward\` 每次只传递“本次新增的”梯度），\`_backward\` 大约会被调用多少次？
      `,
      options: ['31 次', '约 $60$ 次', '约 $2^{31}$ 次', '约 $31^2$ 次'],
      answer: 2,
      explanation: md`图只有 31 个节点，但每个 \`y = y + y\` 节点有两条边指向同一个输入，从输出到 x 的路径数翻倍，共 $2^{30}$ 条；递归调用次数与路径数同阶（$2^{31}-1$ 次）。按反向拓扑序每个节点只调用一次，共 31 次。这就是“先拓扑排序再反传”的复杂度意义。`,
    },
    {
      id: 'q-black',
      type: 'judge',
      difficulty: 2,
      prompt: md`DFS 判断有向图是否有环时，沿边走到一个已经是**黑色**（已完成）的点，说明存在环。`,
      answer: false,
      explanation: md`黑色点的所有后继都已探索完，从当前点走到它只是两条路径汇合（如菱形 $a\to b\to d$、$a \to c \to d$），不是环。只有走到**灰色**点（还在当前递归栈上）才说明出现了回边。`,
    },
    {
      id: 'q-direction',
      type: 'single',
      difficulty: 1,
      prompt: md`LC 207 中 \`prerequisites[i] = [a, b]\` 表示“想学 a 必须先学 b”。用 Kahn 算法时应当建哪条边、给谁的入度加 1？`,
      options: ['边 a → b，`indeg[b] += 1`', '边 b → a，`indeg[a] += 1`', '边 b → a，`indeg[b] += 1`', '无向边，两者都加 1'],
      answer: 1,
      explanation: md`b 是 a 的前置，边从前置指向后续：b → a，a 多了一个前置条件，所以 \`indeg[a] += 1\`。A 把方向建反，求出的是逆拓扑序（对 207 判环不影响，但 210 的输出顺序会错）；C 方向和入度不一致。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-oranges',
      title: '多源 BFS：腐烂的橘子',
      intro: md`
        实现 \`oranges_rotting(grid)\`：\`grid[i][j]\` 为 0（空）、1（新鲜）、2（腐烂）。每分钟，腐烂橘子上下左右的新鲜橘子变烂。
        返回全部腐烂所需的最少分钟数；不可能则返回 $-1$。可以直接修改 \`grid\`。
      `,
      template: py`
        from collections import deque

        def oranges_rotting(grid):
            m, n = len(grid), len(grid[0])
            q = {{1}}
            fresh = sum(row.count(1) for row in grid)
            minutes = 0
            while q and fresh:
                for _ in {{2}}:
                    r, c = q.popleft()
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if {{3}}:
                            grid[nr][nc] = 2
                            {{4}}
                            q.append((nr, nc))
                {{5}}
            return minutes if fresh == 0 else -1
      `,
      steps: [
        {
          blank: 1,
          instruction: md`多源 BFS 的关键：把**所有**腐烂橘子的坐标一起放进初始队列。`,
          answer: 'deque((i, j) for i in range(m) for j in range(n) if grid[i][j] == 2)',
          accept: ['deque([(i, j) for i in range(m) for j in range(n) if grid[i][j] == 2])'],
          hint: '用生成器表达式遍历全部格子，筛出值为 2 的。',
          explain: md`所有源点同时位于第 0 层，等价于一个虚拟“超级源点”连向它们。之后的 BFS 和单源完全一样。`,
        },
        {
          blank: 2,
          instruction: md`一层就是一分钟：本轮只处理“当前层”的橘子。`,
          answer: 'range(len(q))',
          explain: md`\`range(len(q))\` 只在进入循环时求值一次，本轮新入队的橘子留到下一分钟处理。`,
        },
        {
          blank: 3,
          instruction: md`邻居在网格内，并且是**新鲜**橘子时才会被感染。`,
          answer: '0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1',
          accept: ['nr in range(m) and nc in range(n) and grid[nr][nc] == 1'],
          explain: md`先判断边界再访问 \`grid[nr][nc]\`，利用 \`and\` 的短路避免越界（Python 负下标不会报错，而是悄悄访问到另一端，更危险）。`,
        },
        {
          blank: 4,
          instruction: md`一个新鲜橘子变烂后，更新剩余的新鲜数量。`,
          answer: 'fresh -= 1',
          explain: md`入队时就把格子改成 2（相当于标记 visited），并同步减少 \`fresh\`，保证每个橘子只被感染一次。`,
        },
        {
          blank: 5,
          instruction: md`一层处理完，时间过去一分钟。`,
          answer: 'minutes += 1',
          explain: md`配合 \`while q and fresh\`：最后一批橘子变烂后 \`fresh == 0\`，循环立即结束，不会多算一分钟。`,
        },
      ],
      setup: ORANGE_SETUP,
      tests: [
        { name: '示例 1', code: py`assert oranges_rotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]) == 4` },
        { name: '示例 2：有孤立橘子', code: py`assert oranges_rotting([[2, 1, 1], [0, 1, 1], [1, 0, 1]]) == -1` },
        {
          name: '没有新鲜橘子 / 多个源',
          code: py`
            assert oranges_rotting([[0, 2]]) == 0
            assert oranges_rotting([[2, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 2]]) == 3
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(994)
            for _ in range(200):
                m, n = rnd.randint(1, 7), rnd.randint(1, 7)
                g = [[rnd.choice([0, 1, 1, 1, 2]) for _ in range(n)] for _ in range(m)]
                exp = _ref_rot(g)
                got = oranges_rotting([row[:] for row in g])
                assert got == exp, f"grid={g}：期望 {exp}，得到 {got}"
          `,
        },
      ],
      outro: md`
        这个“多源 + 按层计数”的骨架可以直接迁移到 LC 542 / 1162。如果只关心可达性、不关心分钟数，去掉 \`for _ in range(len(q))\` 这一层即可。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-islands',
      title: '岛屿数量',
      difficulty: 'medium',
      source: 'LeetCode 200',
      timeLimitMin: 15,
      description: md`
        给定由 \`"1"\`（陆地）和 \`"0"\`（水）组成的二维网格 \`grid\`，计算岛屿数量。岛屿由**水平或竖直**方向相邻的陆地连接而成，网格四周都是水。

        \`\`\`python
        def num_islands(grid: list[list[str]]) -> int: ...
        \`\`\`

        **示例**
        \`\`\`text
        [["1","1","0","0","0"],
         ["1","1","0","0","0"],
         ["0","0","1","0","0"],
         ["0","0","0","1","1"]]   → 3
        \`\`\`

        **约束**：$1 \le m, n \le 20$（本题测试规模；原题到 300，那时递归 DFS 会超过 Python 递归上限）。可以修改 \`grid\`。要求 $O(mn)$。
      `,
      starter: py`
        from collections import deque

        def num_islands(grid: list[list[str]]) -> int:
            # TODO: 逐格扫描；遇到未访问的陆地，计数 +1，并用 BFS/DFS 把整座岛标记掉
            return 0
      `,
      solution: py`
        from collections import deque

        def num_islands(grid: list[list[str]]) -> int:
            m, n = len(grid), len(grid[0])
            count = 0
            for i in range(m):
                for j in range(n):
                    if grid[i][j] != "1":
                        continue
                    count += 1                      # 发现一座新岛
                    grid[i][j] = "0"                # 入队即标记，避免重复入队
                    q = deque([(i, j)])
                    while q:
                        r, c = q.popleft()
                        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == "1":
                                grid[nr][nc] = "0"
                                q.append((nr, nc))
            return count
      `,
      explanation: md`
        **思路**：连通分量计数。外层逐格扫描，每遇到一块还没被访问的陆地，说明发现了一座新岛，计数加一，然后从它出发 BFS，
        把这座岛上的所有陆地改成 \`"0"\`（就地标记，省掉 visited 数组）。

        **复杂度**：每个格子最多入队一次、检查 4 个邻居，时间 $O(mn)$；BFS 队列最坏 $O(\min(m, n))$ 到 $O(mn)$。

        **常见错误**
        - 出队时才标记：同一格被多个邻居重复入队；
        - 把对角线也当相邻；
        - 递归 DFS 在大网格上 \`RecursionError\`；
        - 比较时写成整数 \`grid[i][j] == 1\`（本题元素是字符串 \`"1"\`）。
      `,
      complexity: { time: 'O(mn)', space: 'O(mn)' },
      hints: [
        md`岛屿数 = 陆地格子构成的连通分量个数。`,
        md`外层双重循环扫描；遇到 \`"1"\` 就计数并从它出发 BFS，把同一座岛全部改成 \`"0"\`。`,
        md`入队时就把格子改成 \`"0"\`；注意元素是字符串。`,
      ],
      setup: GRID_SETUP,
      tests: [
        {
          name: '示例 1：一座大岛',
          code: py`
            g = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
            got = num_islands(g)
            assert got == 1, f"期望 1，得到 {got}"
          `,
        },
        {
          name: '示例 2：三座岛',
          code: py`
            g = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
            got = num_islands(g)
            assert got == 3, f"期望 3，得到 {got}"
          `,
        },
        {
          name: '边界：单格 / 全水',
          code: py`
            assert num_islands([["1"]]) == 1
            assert num_islands([["0"]]) == 0
            assert num_islands([["0", "0"], ["0", "0"]]) == 0
          `,
        },
        {
          name: '对角线不算相邻',
          hidden: true,
          code: py`
            got = num_islands([["1", "0", "1"], ["0", "1", "0"], ["1", "0", "1"]])
            assert got == 5, f"对角线相邻的陆地属于不同岛屿：期望 5，得到 {got}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(200)
            for _ in range(300):
                m, n = rnd.randint(1, 8), rnd.randint(1, 8)
                g = _rand_grid(rnd, m, n, rnd.choice([0.3, 0.5, 0.7]))
                exp = _ref_islands(g)
                got = num_islands([row[:] for row in g])
                assert got == exp, f"grid={g}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：20×20 × 40 组',
          hidden: true,
          code: py`
            rnd = random.Random(1)
            t0 = time.perf_counter()
            for _ in range(40):
                g = _rand_grid(rnd, 20, 20, 0.6)
                exp = _ref_islands(g)
                got = num_islands([row[:] for row in g])
                assert got == exp, f"期望 {exp}，得到 {got}"
            dt = time.perf_counter() - t0
            assert dt < 1.5, f"用时 {dt:.2f}s，应为 O(mn)"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-uf',
          q: md`如果陆地是一块一块**动态加入**的，每加一块都要回答当前岛屿数（LC 305），怎么做？`,
          a: md`每次重新 BFS 是 $O(k \cdot mn)$。用**并查集**：新陆地先让计数 +1，再与四周已有陆地逐一 union，每次成功合并计数 -1。每次操作近似 $O(1)$（$\alpha(n)$），总 $O(k)$。这是下一关的内容。`,
        },
        {
          id: 'f-dfs',
          q: md`递归 DFS 写起来更短，Python 里有什么风险？怎么规避？`,
          a: md`全是陆地时递归深度可达 $mn$（原题 $9 \times 10^4$），远超默认上限 1000，会抛 \`RecursionError\`；调大 \`sys.setrecursionlimit\` 还可能撑爆 C 栈直接崩溃。改用 BFS 或显式栈 DFS（把 \`popleft\` 换成 \`pop\`），或者并查集。`,
        },
      ],
    },
    {
      id: 'c-course',
      title: '课程表 II（拓扑排序）',
      difficulty: 'medium',
      source: 'LeetCode 210',
      timeLimitMin: 20,
      description: md`
        共有 \`num_courses\` 门课，编号 \`0 .. num_courses-1\`。\`prerequisites[i] = [a, b]\` 表示学习 \`a\` 之前必须先学完 \`b\`。
        返回一个可以学完所有课程的顺序；若不可能（存在环）返回空列表 \`[]\`。**答案不唯一**，返回任意合法顺序即可。

        \`\`\`python
        def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]: ...
        \`\`\`

        **示例**
        - \`find_order(2, [[1, 0]])\` → \`[0, 1]\`
        - \`find_order(4, [[1,0],[2,0],[3,1],[3,2]])\` → \`[0,1,2,3]\` 或 \`[0,2,1,3]\`
        - \`find_order(2, [[1,0],[0,1]])\` → \`[]\`

        **约束**：$1 \le n \le 3000$，边数 $\le 10^4$，$a \ne b$，边不重复。要求 $O(V + E)$（隐藏测试会统计你遍历 \`prerequisites\` 的次数）。
      `,
      starter: py`
        from collections import deque

        def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
            # TODO: 建图（[a, b] 表示边 b → a）+ 入度 + Kahn；输出不足 n 个说明有环
            return []
      `,
      solution: py`
        from collections import deque

        def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
            graph = [[] for _ in range(num_courses)]
            indeg = [0] * num_courses
            for a, b in prerequisites:          # 先修 b 才能修 a：边 b → a
                graph[b].append(a)
                indeg[a] += 1
            q = deque(i for i in range(num_courses) if indeg[i] == 0)
            order = []
            while q:
                u = q.popleft()
                order.append(u)
                for v in graph[u]:              # “删掉” u 的出边
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)
            return order if len(order) == num_courses else []
      `,
      explanation: md`
        **思路**：Kahn 算法。建邻接表与入度表；入度为 0 的课程入队；每弹出一门课就把它的后续课程入度减一，减到 0 入队。
        最终输出的课程数等于 $n$ 则为合法顺序，否则有环返回 \`[]\`。

        **DFS 写法**：三色标记，后序记录、最后反转；遇到灰色点说明有环。注意递归深度等于最长依赖链长度。

        **复杂度**：建图 $O(E)$，每个点入队出队一次、每条边被“删除”一次，总 $O(V + E)$。

        **常见错误**：边方向建反（得到逆序）；每轮重新扫描所有课程找入度为 0 的点（$O(V^2)$）；忘记判环直接返回不完整的 \`order\`。
      `,
      complexity: { time: 'O(V + E)', space: 'O(V + E)' },
      hints: [
        md`\`[a, b]\` 是边 b → a：b 完成后 a 才可能“解锁”。`,
        md`维护入度数组，入度为 0 的点可以立刻学习，放进队列。`,
        md`弹出 u 后遍历 \`graph[u]\` 把入度减一；最后检查 \`len(order) == num_courses\`。`,
      ],
      setup: TOPO_SETUP,
      tests: [
        { name: '示例 1', code: py`assert find_order(2, [[1, 0]]) == [0, 1]` },
        {
          name: '示例 2（答案不唯一）',
          code: py`
            pre = [[1, 0], [2, 0], [3, 1], [3, 2]]
            got = find_order(4, pre)
            err = _check_order(4, pre, got)
            assert err is None, err
          `,
        },
        {
          name: '有环返回 []',
          code: py`
            got = find_order(2, [[1, 0], [0, 1]])
            assert got == [], f"存在环 0 ↔ 1，应返回 []，得到 {got}"
          `,
        },
        {
          name: '没有先修关系 / 单门课',
          code: py`
            got = find_order(3, [])
            assert sorted(got) == [0, 1, 2], f"应返回 0..2 的任意排列，得到 {got}"
            assert find_order(1, []) == [0]
          `,
        },
        {
          name: '随机 DAG 与随机有环图',
          hidden: true,
          code: py`
            rnd = random.Random(210)
            for _ in range(200):
                n = rnd.randint(1, 9)
                pairs = set()
                for _ in range(rnd.randint(0, n * 2)):
                    a, b = rnd.randrange(n), rnd.randrange(n)
                    if a != b:
                        pairs.add((a, b))
                pre = [list(p) for p in pairs]
                got = find_order(n, [p[:] for p in pre])
                if _has_cycle(n, pre):
                    assert got == [], f"n={n}, pre={pre} 有环，应返回 []，得到 {got}"
                else:
                    err = _check_order(n, pre, got)
                    assert err is None, f"n={n}, pre={pre}：{err}"
          `,
        },
        {
          name: '规模：3000 门课、9000 条边（O(V+E)）',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            pre, _, _ = _layered_dag(rnd, 3000, 40, 9000)
            wrapped = _CountedList(pre)
            t0 = time.perf_counter()
            got = find_order(3000, wrapped)
            dt = time.perf_counter() - t0
            err = _check_order(3000, pre, got)
            assert err is None, err
            assert dt < 1.0, f"用时 {dt:.2f}s，应为 O(V+E)"
          `,
        },
        {
          name: '规模：大图里藏一个环',
          hidden: true,
          code: py`
            rnd = random.Random(8)
            pre, perm, layer = _layered_dag(rnd, 3000, 40, 9000)
            # 找一条边 b → a，再加一条 a → b 的回边形成 2 元环
            a, b = pre[0]
            pre.append([b, a])
            got = find_order(3000, _CountedList(pre))
            assert got == [], f"图中有环 {a} ↔ {b}，应返回 []，得到长度 {len(got)} 的结果"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-lex',
          q: md`如果要求返回**字典序最小**的合法顺序，怎么改？`,
          a: md`把队列换成最小堆（\`heapq\`）：每次弹出当前可学课程中编号最小的。复杂度变为 $O((V + E)\log V)$。正确性：拓扑序的每一步都在“当前入度为 0 的集合”里选，贪心选最小即可得到字典序最小。`,
        },
        {
          id: 'f-parallel',
          q: md`每学期可以同时修任意多门课，最少需要几个学期（LC 1136）？`,
          a: md`按层做 Kahn：每层是一个学期，层数即答案（有环则 -1）。等价于 DAG 上的最长路径（按点数计）。若每学期最多修 k 门（LC 1494），问题变成 NP-hard 的调度，小规模用状压 DP。`,
        },
      ],
    },
    {
      id: 'c-autograd',
      title: '标量 mini-autograd：按反向拓扑序传播梯度',
      difficulty: 'hard',
      source: '大模型岗手撕 · autograd',
      timeLimitMin: 30,
      description: md`
        补全一个 micrograd 风格的标量自动微分类 \`Value\`。已提供：构造函数、加法 \`__add__\`、幂 \`__pow__\`（常数指数）以及若干语法糖。你需要实现：

        1. \`__mul__\` 的局部梯度：$\frac{\partial (xy)}{\partial x} = y$，$\frac{\partial (xy)}{\partial y} = x$；
        2. \`tanh\` 的局部梯度：$\frac{d}{dx}\tanh x = 1 - \tanh^2 x$；
        3. \`backward(self)\`：从 \`self\`（输出 $L$）出发求出计算图的**拓扑序**，令 \`self.grad = 1.0\`，然后按**反向拓扑序**对每个节点调用一次 \`_backward()\`。

        **接口约定**（测试依赖）
        - 节点字段：\`data\`、\`grad\`、\`_prev\`（输入节点元组）、\`_backward\`（无参闭包）；
        - 所有 \`_backward\` 都用 \`+=\` 累加梯度（同一节点可能被多处使用，例如 \`x * x\`、残差）；
        - \`backward()\` 中**每个节点的 \`_backward\` 恰好调用一次**（隐藏测试会包装计数，第二次调用立即失败），且必须在它的梯度收齐之后。

        **示例**
        \`\`\`python
        a, b = Value(2.0), Value(-3.0)
        c = a * b; d = c + b; L = d * c
        L.backward()
        a.grad, b.grad   # (45.0, -36.0)
        \`\`\`

        隐藏测试会把随机生成的表达式的梯度与**数值梯度**（中心差分）对拍，并包含一个只有 41 个节点、却有 $2^{40}$ 条路径的共享图。
      `,
      starter:
        VALUE_CLASS_HEAD +
        '\n' +
        py`
              def __mul__(self, other):
                  other = other if isinstance(other, Value) else Value(other)
                  out = Value(self.data * other.data, (self, other), "*")
                  def _backward():
                      # TODO: 乘法的局部梯度（记得用 +=）
                      raise NotImplementedError
                  out._backward = _backward
                  return out

              def tanh(self):
                  t = math.tanh(self.data)
                  out = Value(t, (self,), "tanh")
                  def _backward():
                      # TODO: d tanh(x)/dx = 1 - tanh(x)^2
                      raise NotImplementedError
                  out._backward = _backward
                  return out

              def backward(self):
                  # TODO 1) 求以 self 为终点的拓扑序（输入排在前面）
                  #      2) self.grad = 1.0
                  #      3) 按反向拓扑序，对每个节点调用一次 v._backward()
                  raise NotImplementedError
        `.replace(/^/gm, '    ') +
        '\n' +
        VALUE_CLASS_TAIL,
      solution:
        VALUE_CLASS_HEAD +
        '\n' +
        py`
              def __mul__(self, other):
                  other = other if isinstance(other, Value) else Value(other)
                  out = Value(self.data * other.data, (self, other), "*")
                  def _backward():
                      self.grad += other.data * out.grad
                      other.grad += self.data * out.grad
                  out._backward = _backward
                  return out

              def tanh(self):
                  t = math.tanh(self.data)
                  out = Value(t, (self,), "tanh")
                  def _backward():
                      self.grad += (1.0 - t * t) * out.grad
                  out._backward = _backward
                  return out

              def backward(self):
                  # 迭代 DFS 求后序（= 拓扑序：每个节点排在它的所有输入之后），避免深图递归爆栈
                  topo, seen = [], set()
                  stack = [(self, False)]
                  while stack:
                      v, expanded = stack.pop()
                      if expanded:
                          topo.append(v)
                          continue
                      if v in seen:
                          continue
                      seen.add(v)
                      stack.append((v, True))
                      for child in v._prev:
                          if child not in seen:
                              stack.append((child, False))
                  self.grad = 1.0
                  for v in reversed(topo):        # 反向拓扑序：v 的所有消费者都已反传
                      v._backward()
        `.replace(/^/gm, '    ') +
        '\n' +
        VALUE_CLASS_TAIL,
      solutionAlt: [
        {
          label: 'micrograd 原版：递归 DFS 求拓扑序',
          code: py`
            def backward(self):
                topo, visited = [], set()
                def build_topo(v):
                    if v not in visited:
                        visited.add(v)
                        for child in v._prev:
                            build_topo(child)
                        topo.append(v)
                build_topo(self)
                self.grad = 1
                for v in reversed(topo):
                    v._backward()
          `,
        },
        {
          label: 'PyTorch 版（仅展示）：用 autograd 验证同一个例子',
          code: py`
            import torch

            a = torch.tensor(2.0, requires_grad=True)
            b = torch.tensor(-3.0, requires_grad=True)
            c = a * b
            d = c + b
            L = d * c
            L.backward()
            print(a.grad, b.grad)        # tensor(45.) tensor(-36.)

            # 自定义算子时，backward 只负责“局部导数 × 上游梯度”，调度顺序由 engine 负责
            class Tanh(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    t = torch.tanh(x)
                    ctx.save_for_backward(t)
                    return t

                @staticmethod
                def backward(ctx, grad_out):
                    (t,) = ctx.saved_tensors
                    return grad_out * (1 - t * t)
          `,
        },
      ],
      explanation: md`
        **核心**：反向传播 = 在计算图上按**反向拓扑序**执行链式法则。

        1. **拓扑序**：DFS 后序（一个节点在它的所有输入都完成后才加入列表），输入排在前、输出 $L$ 排在最后；
        2. **反向遍历**：从 $L$ 开始。轮到节点 $v$ 时，所有使用 $v$ 的节点（消费者）都排在 $v$ 之后、已经先反传完毕，$v.\text{grad}$ 已收齐；
        3. **累加**：\`_backward\` 中一律 \`+=\`，例如 \`x * x\` 时 \`self is other\`，两行各加一次，正好得到 $2x$。

        **为什么不能直接递归往下传**：调用次数等于输出到各节点的路径数，在共享严重的图上是指数级（\`y = y + y\` 重复 40 次有 $2^{40}$ 条路径）。按拓扑序每个节点只处理一次，$O(V + E)$。

        **工程细节**：参考解用迭代 DFS 求拓扑序。micrograd 原版用递归，在几千层深的图（如长序列 RNN 展开）上会超过 Python 的递归上限；
        PyTorch 的 engine 是 C++ 实现，先统计依赖数再用 ready queue 调度，相当于 Kahn 算法，同时天然支持多线程 / 多设备。
      `,
      complexity: { time: 'O(V + E)', space: 'O(V)' },
      hints: [
        md`先写 \`__mul__\` 与 \`tanh\`：局部导数 × \`out.grad\`，用 \`+=\` 加到输入上。\`tanh\` 可以直接用前向算好的 \`t\`。`,
        md`拓扑序 = DFS 后序：访问完一个节点的全部 \`_prev\` 之后，再把它加入列表；用 \`seen\` 集合保证每个节点只加入一次。`,
        md`\`self.grad = 1.0\`，然后 \`for v in reversed(topo): v._backward()\`。`,
      ],
      setup: AUTOGRAD_SETUP,
      tests: [
        {
          name: '示例：c 与 b 都被两处使用',
          code: py`
            a, b = Value(2.0), Value(-3.0)
            c = a * b
            d = c + b
            L = d * c
            L.backward()
            assert abs(L.data - 54.0) < 1e-12, f"前向 L 应为 54，得到 {L.data}"
            assert abs(a.grad - 45.0) < 1e-9, f"a.grad 期望 45，得到 {a.grad}"
            assert abs(b.grad + 36.0) < 1e-9, f"b.grad 期望 -36，得到 {b.grad}"
            assert abs(c.grad + 15.0) < 1e-9, f"c.grad 期望 -15（两条路径之和），得到 {c.grad}"
          `,
        },
        {
          name: 'tanh 神经元（micrograd 经典例子）',
          code: py`
            x1, x2, w1, w2 = Value(2.0), Value(0.0), Value(-3.0), Value(1.0)
            bias = Value(6.8813735870195432)
            o = (x1 * w1 + x2 * w2 + bias).tanh()
            o.backward()
            exp = {"x1": -1.5, "w1": 1.0, "x2": 0.5, "w2": 0.0}
            got = {"x1": x1.grad, "w1": w1.grad, "x2": x2.grad, "w2": w2.grad}
            for k in exp:
                assert abs(got[k] - exp[k]) < 1e-6, f"{k}.grad 期望 {exp[k]}，得到 {got[k]:.6f}"
          `,
        },
        {
          name: '同一节点自乘：x * x + x',
          code: py`
            x = Value(3.0)
            y = x * x + x
            y.backward()
            assert abs(x.grad - 7.0) < 1e-9, f"d(x²+x)/dx = 2x+1 = 7，得到 {x.grad}（是不是用了 = 而不是 +=？）"
          `,
        },
        {
          name: '随机表达式 vs 数值梯度',
          hidden: true,
          code: py`
            rnd = random.Random(2026)
            checked = 0
            for _ in range(80):
                prog = _rand_program(rnd, 3, rnd.randint(3, 12))
                xs = [rnd.uniform(-1.5, 1.5) for _ in range(3)]
                if abs(_eval(prog, xs)[0]) > 1e4:
                    continue
                out, ins = _eval(prog, xs, Value)
                try:
                    out.backward()
                    num = _num_grad(prog, xs)
                    for k, (v, ng) in enumerate(zip(ins, num)):
                        err = abs(v.grad - ng)
                        assert err <= 1e-4 * max(1.0, abs(ng)), f"程序 {prog}，输入 {xs}：x{k} 的梯度 {v.grad:.6f}，数值梯度 {ng:.6f}"
                    checked += 1
                finally:
                    _free_graph(out)
            assert checked >= 50
          `,
        },
        {
          name: '每个节点只反传一次（2^40 条路径的共享图）',
          hidden: true,
          code: py`
            x = Value(1.5)
            y = x
            for _ in range(40):
                y = y + y
            calls = _instrument(y)
            try:
                y.backward()
                assert x.grad == 2.0 ** 40, f"x.grad 应为 2^40，得到 {x.grad}"
                assert len(calls) == 41 and all(c == 1 for c in calls.values()), "每个节点的 _backward 应恰好调用一次"
            finally:
                _free_graph(y)
          `,
        },
        {
          name: '深链：150 层 tanh',
          hidden: true,
          code: py`
            x = Value(0.3)
            y = x
            fx, deriv = 0.3, 1.0
            for _ in range(150):
                t = math.tanh(fx)
                deriv *= 1 - t * t
                fx = t + 0.1
                y = y.tanh() + 0.1
            calls = _instrument(y)
            try:
                y.backward()
                assert abs(y.data - fx) < 1e-9, "前向结果不对"
                assert abs(x.grad - deriv) < 1e-9 * max(1.0, abs(deriv)), f"x.grad 期望 {deriv:.6e}，得到 {x.grad:.6e}"
            finally:
                _free_graph(y)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-kahn',
          q: md`不用 DFS，能用 Kahn 算法实现 \`backward\` 吗？`,
          a: md`可以，而且这正是 PyTorch engine 的做法。先从 $L$ 出发遍历一次，统计每个节点被多少个消费者使用（“依赖数”，即反向图上的入度）；令 \`L.grad = 1\` 并把 $L$ 放进队列；弹出 $v$ 执行 \`_backward\`，再把它的每个输入的依赖数减一，减到 0 说明该输入的梯度已收齐，入队。复杂度同样 $O(V+E)$，还天然适合并行调度：同一时刻所有依赖数为 0 的节点可以并发执行。`,
        },
        {
          id: 'f-memory',
          q: md`反向传播为什么要保存前向的中间结果？显存不够时怎么办？`,
          a: md`局部导数往往依赖前向的值（如 \`x * y\` 的梯度需要 \`x\`、\`y\`，\`tanh\` 需要输出 \`t\`），所以计算图要保存这些激活，直到反传用完才释放——这是训练显存的大头。显存不够时用**激活重计算**（gradient checkpointing）：只保存部分节点，反传时重算中间结果，用约一次额外前向的计算换显存。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-bfs-shortest',
      freq: 3,
      q: md`为什么 BFS 能求无权图的最短路？带权图呢？`,
      a: md`
        **结论**：BFS 按距离非递减的顺序访问顶点，第一次到达某点时的层数就是最短距离；带权图要用 Dijkstra（非负权）或 Bellman-Ford（有负权）。

        **原因**：归纳可证——队列中始终只有距离为 $k$ 和 $k+1$ 的点，且距离 $k$ 的都在前面；一个点第一次被发现时，发现它的点距离最小，所以它的距离也最小，之后再被发现只会更远，因此入队即标记是安全的。边权不全相等时，“层数少”不等于“距离短”，这个性质失效。

        **延伸**：边权只有 0 和 1 时用 **0-1 BFS**（双端队列，权 0 的边放队头、权 1 放队尾），$O(V+E)$；Dijkstra 可以看作“用堆代替队列”的 BFS。
      `,
    },
    {
      id: 'qa-topo',
      freq: 3,
      q: md`拓扑排序有哪几种实现？怎么判环？`,
      a: md`
        **结论**：Kahn（BFS + 入度）和 DFS（三色标记 + 逆后序），都是 $O(V+E)$。

        **原因**：Kahn 每次取入度为 0 的点，输出数小于 $n$ 即有环，环上的点入度永远减不到 0。DFS 中灰色表示“在当前递归栈上”，走到灰色点就是回边、有环；一个点完成时它能到达的点都已完成，所以后序反转即拓扑序。

        **延伸**：Kahn 是迭代的，没有递归深度问题，还能自然得到“按层”的信息（最少学期数、并行调度）；需要字典序最小时把队列换成堆。DFS 写法在判环 + 找出环上的具体节点时更方便（沿灰色路径回溯）。
      `,
    },
    {
      id: 'qa-grid',
      freq: 2,
      q: md`岛屿数量用 DFS、BFS 还是并查集？各有什么取舍？`,
      a: md`
        **结论**：静态网格用 BFS（或显式栈 DFS）最直接；陆地动态加入时用并查集。

        **原因**：三者都是 $O(mn)$。递归 DFS 代码最短，但深度最坏 $O(mn)$，Python 下大网格会 \`RecursionError\`；BFS 无此问题。并查集需要额外的 parent 数组，但支持**增量**合并：每加一块陆地只与四邻 union，可以在线回答岛屿数（LC 305）。

        **延伸**：如果要统计每座岛的面积、周长、形状（LC 695 / 463 / 694），在洪水填充的过程中顺便累计即可；形状去重时记录相对坐标或路径签名。
      `,
    },
    {
      id: 'qa-autograd',
      freq: 2,
      q: md`autograd 的反向传播和拓扑排序有什么关系？PyTorch 是怎么调度反向计算的？`,
      a: md`
        **结论**：反向传播必须按计算图的**反向拓扑序**执行，保证每个节点的梯度收齐后再往输入传；PyTorch 用依赖计数 + ready queue 实现，本质是 Kahn 算法。

        **原因**：一个张量可能被多处使用，它的梯度是各处贡献之和（多元链式法则）。反向拓扑序保证所有消费者先反传完。若每收到一份梯度就往下传，调用次数等于路径数，在共享多的图上是指数级。PyTorch 的 engine 先统计每个节点的依赖数，节点执行完后把下游依赖数减一，减到 0 才放进 ready queue。

        **延伸**：\`.grad\` 是累加语义，所以训练循环里要 \`zero_grad()\`，梯度累积也依赖这一点；前向保存的激活是反传所需，显存不够时用 activation checkpointing 重算。
      `,
    },
  ],

  cheatsheet: md`
    **Kahn 拓扑排序**
    \`\`\`python
    for a, b in pre: g[b].append(a); indeg[a] += 1   # [a,b]: b → a
    q = deque(i for i in range(n) if indeg[i] == 0)
    while q:
        u = q.popleft(); order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    return order if len(order) == n else []          # 不足 n 个 ⇒ 有环
    \`\`\`

    | 问题 | 方法 | 复杂度 |
    |---|---|---|
    | 连通分量 / 岛屿 | 扫描 + BFS 洪水填充（入队即标记） | $O(mn)$ |
    | 无权最短路 | BFS，层数 = 距离 | $O(V+E)$ |
    | 多个源点同时扩散 | 多源 BFS（源点全部放进第 0 层） | $O(V+E)$ |
    | 有向图判环 | Kahn 输出数 < n，或 DFS 遇灰色点 | $O(V+E)$ |
    | DFS 拓扑序 | 后序记录再反转 | $O(V+E)$ |

    - autograd：拓扑序（DFS 后序）→ \`L.grad = 1\` → \`reversed(topo)\` 逐个 \`_backward()\`；梯度一律 \`+=\`
    - 网格大时别用递归 DFS（Python 递归上限 1000）
    - \`while q and fresh\`：避免多算一层
  `,
})
