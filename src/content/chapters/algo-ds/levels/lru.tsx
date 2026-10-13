/**
 * LRU 与 LFU 缓存（algo-ds-lru）
 * LC146 / LC460：哈希 + 双向链表 O(1)、哨兵节点、OrderedDict、LFU 频次桶 + min_freq；大模型场景：KV / 前缀缓存块淘汰。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { LruViz } from '../viz/LruViz'
import { LfuViz } from '../viz/LfuViz'
import { PrefixCacheViz } from '../viz/PrefixCacheViz'

/** LRU 实战 / 跟练共用的测试辅助：OrderedDict 参考实现 + 计数 key（list.remove 之类的 O(n) 操作会立即超预算） */
const LRU_SETUP = py`
  import linecache, random, time
  from collections import OrderedDict as _OD

  class _RefLRU:
      def __init__(self, cap):
          self.cap, self.d = cap, _OD()
      def get(self, k):
          if k not in self.d:
              return -1
          self.d.move_to_end(k)
          return self.d[k]
      def put(self, k, v):
          if k in self.d:
              self.d.move_to_end(k)
          self.d[k] = v
          if len(self.d) > self.cap:
              self.d.popitem(last=False)

  class _Key:
      """计数 key：__eq__ 每调用一次 +1（字典按身份命中不会调用；线性查找 / remove 会调用）"""
      __slots__ = ("v",)
      n = 0
      budget = 10**12
      def __init__(self, v):
          self.v = v
      def __hash__(self):
          return hash(self.v)
      def __eq__(self, o):
          _Key.n += 1
          if _Key.n > _Key.budget:
              raise AssertionError(f"key 比较次数超过 {_Key.budget}：存在 O(n) 的线性查找 / 删除（list.remove、deque.remove、index…），不是 O(1)")
          return isinstance(o, _Key) and self.v == o.v
      def __repr__(self):
          return f"K({self.v})"

  def _run_ops(cls, cap, ops):
      c, out = cls(cap), []
      for op in ops:
          if op[0] == "get":
              out.append(c.get(op[1]))
          else:
              c.put(op[1], op[2])
              out.append(None)
      return out

  def _rand_ops(rnd, n, kr):
      return [("get", rnd.randint(1, kr)) if rnd.random() < 0.45 else ("put", rnd.randint(1, kr), rnd.randint(0, 99)) for _ in range(n)]

  def _check_lru(cap, ops):
      got = _run_ops(LRUCache, cap, ops)
      exp = _run_ops(_RefLRU, cap, ops)
      if got != exp:
          i = next(i for i, (a, b) in enumerate(zip(got, exp)) if a != b)
          raise AssertionError(f"capacity={cap}，第 {i} 个操作 {ops[i]} 起不一致：期望 {exp[i]}，得到 {got[i]}（前序操作：{ops[:i]}）")
`

export default defineLevel({
  /* ================================================================ */
  /* 图解                                                              */
  /* ================================================================ */
  learn: [
    {
      id: 'design',
      title: '需求拆解：为什么是哈希表 + 双向链表',
      body: md`
        LRU（Least Recently Used）缓存要求 \`get\` 和 \`put\` 都是 $O(1)$，满了就淘汰**最久未使用**的 key。拆开看要同时满足三件事：

        1. 按 key **查找**：$O(1)$ → 哈希表；
        2. 维护**使用顺序**，访问一次就把它移到“最新”：$O(1)$ 删除 + $O(1)$ 插入 → 链表；
        3. 找到**最旧**的那个并删除：$O(1)$ → 链表的一端。

        | 方案 | 查找 | 移到最新 | 淘汰最旧 |
        |---|---|---|---|
        | 只有哈希表 | $O(1)$ | 无顺序 | $O(n)$ 扫描时间戳 |
        | 数组 / 列表存顺序 | $O(n)$ | $O(n)$（删除要搬移） | $O(n)$ 或 $O(1)$ |
        | 哈希表 + **单**链表 | $O(1)$ | $O(n)$（删除要找前驱） | $O(n)$ |
        | 哈希表 + **双向**链表 | $O(1)$ | $O(1)$ | $O(1)$ |

        关键在于**哈希表的 value 存的是链表节点本身**：拿到节点后，双向链表通过 \`node.prev\` / \`node.next\` 就能 $O(1)$ 把它摘下，
        单链表做不到（找前驱要从头走）。
      `,
      viz: LruViz,
      after: md`
        > [!TIP] 读图的关键
        > 每个操作都只碰哈希表的一项和链表的常数个指针。淘汰时链表告诉你“是谁”（\`tail.prev\`），但要从哈希表里删除它，
        > 还得知道它的 key——所以**节点里必须同时存 key 和 value**。
      `,
      checkpoint: {
        id: 'cp-design',
        type: 'single',
        prompt: md`为什么 LRU 要用**双向**链表，而不是单链表？`,
        options: [
          '单链表无法 O(1) 删除任意节点：删除需要前驱，而前驱只能从头遍历得到',
          '单链表无法找到表头',
          '双向链表占用内存更少',
          '单链表不能存储 key',
        ],
        answer: 0,
        explanation: md`哈希表给出的是“要删的节点本身”，删除它需要把前驱的 next 改掉。双向链表的 \`node.prev\` 直接给出前驱；单链表只能 $O(n)$ 遍历。双向链表每个节点多一个指针，内存反而**更多**。`,
      },
    },
    {
      id: 'impl',
      title: '实现细节：哨兵节点、两个原语、OrderedDict',
      body: md`
        手写时只需要两个链表原语，所有操作都由它们拼出来：

        \`\`\`python
        def _remove(self, node):            # 从链表摘下（4 个指针里改 2 个）
            node.prev.next = node.next
            node.next.prev = node.prev

        def _add_front(self, node):         # 插到 head 之后（最近使用）
            node.prev, node.next = self.head, self.head.next
            self.head.next.prev = node
            self.head.next = node
        \`\`\`

        **头尾哨兵** \`head\` / \`tail\` 是两个不存数据的假节点：链表永远非空，\`node.prev\` / \`node.next\` 永远不是 None，
        上面两个函数里一个 if 都不需要。约定 \`head\` 侧是最近使用，\`tail.prev\` 就是要淘汰的 LRU 节点。

        - \`get\`：不在哈希表 → \`-1\`；在 → \`_remove\` + \`_add_front\`，返回值；
        - \`put\`：已存在 → 改值 + 移到最前；不存在且已满 → 删 \`tail.prev\`（链表和哈希表都删），再插入新节点。

        **Python 捷径**：\`collections.OrderedDict\` 本身就是“哈希表 + 双向链表”，\`move_to_end(k)\` 与
        \`popitem(last=False)\` 都是 $O(1)$。面试时可以先说出它，但面试官**几乎一定会要求手写**双向链表——这才是考点。
        另外，Python 3.7+ 的普通 \`dict\` 也保持插入顺序：\`d[k] = d.pop(k)\` 可以“移到末尾”，\`next(iter(d))\` 取最旧的 key。
      `,
      after: md`
        > [!WARN] 手写 LRU 的四个高频 bug
        > 1. 节点不存 key，淘汰时无法删除哈希表项；
        > 2. \`put\` 已存在的 key 时只改值，**忘了移到最前**；
        > 3. \`_add_front\` 里先执行 \`self.head.next = node\`，再写 \`self.head.next.prev = node\`——此时 head.next 已是 node 自己，原第一个节点的 prev 没更新；
        > 4. 先插入再判断容量，导致刚插入的 key 被自己淘汰（容量为 1 时最明显）。
      `,
      checkpoint: {
        id: 'cp-impl',
        type: 'judge',
        prompt: md`使用头尾两个哨兵节点后，\`_remove(node)\` 只需两行 \`node.prev.next = node.next; node.next.prev = node.prev\`，不需要判断 node 是不是第一个或最后一个真实节点。`,
        answer: true,
        explanation: md`有哨兵时任何真实节点的 prev、next 都不为 None：第一个真实节点的 prev 是 head，最后一个的 next 是 tail。这正是哨兵的价值——把边界情况变成一般情况。`,
      },
    },
    {
      id: 'lfu',
      title: 'LFU：频次桶 + min_freq，全部 O(1)',
      body: md`
        LFU（Least Frequently Used）淘汰**访问次数最少**的 key；次数相同时淘汰其中**最久未使用**的（LC460 的规定）。

        朴素做法是每次淘汰时扫一遍找最小频次——$O(n)$；用堆按 (freq, time) 排序可以做到 $O(\log n)$。
        $O(1)$ 的做法是三个结构配合：

        | 结构 | 作用 |
        |---|---|
        | \`kv[key] = (value, freq)\` | 查值与频次 |
        | \`buckets[freq]\` = 一条小 LRU 链表（有序集合） | 同频次内按最近使用排序 |
        | \`min_freq\` | 当前最小频次，淘汰时直接去这个桶的最旧端取 |

        **为什么 min_freq 能 $O(1)$ 维护**：一次访问只会让某个 key 的频次从 $f$ 变成 $f+1$。
        如果桶 $f$ 因此变空、且 $f$ 正是 \`min_freq\`，那么新的最小值只能是 $f+1$（这个 key 自己就在那里）。
        插入新 key 时频次为 1，\`min_freq\` 直接重置为 1。淘汰之后不需要重新计算——紧接着插入的新 key 会把它设为 1。
      `,
      viz: LfuViz,
      after: md`
        > [!NOTE] LRU 与 LFU 怎么选
        > LRU 对“最近”敏感，一次大范围扫描（每个 key 只访问一次）就能把热点全部冲掉；LFU 能保护长期热点，
        > 但对访问模式的变化反应迟钝（曾经很热的 key 频次很高，迟迟不被淘汰）。工程上常见折中有 LRU-K、2Q、ARC、W-TinyLFU，
        > Redis 的 LFU 模式则用对数计数器并随时间衰减。
      `,
      checkpoint: {
        id: 'cp-lfu',
        type: 'single',
        prompt: md`LFU 中，某次 \`get(key)\` 把 key 从桶 $f$ 搬到桶 $f+1$ 后，桶 $f$ 变空了。若此前 \`min_freq == f\`，新的 \`min_freq\` 应为？`,
        options: ['$f + 1$', '重新扫描所有桶取最小值', '1', '不变，仍为 $f$'],
        answer: 0,
        explanation: md`频次为 $f$ 的 key 原本只剩它一个，它现在在桶 $f+1$；其它 key 的频次都 $\ge f$ 且不等于 $f$（桶 $f$ 已空），所以最小值就是 $f+1$。不需要扫描；重置为 1 只发生在插入新 key 时。`,
      },
    },
    {
      id: 'llm',
      title: '大模型岗场景：KV / 前缀缓存的块淘汰',
      body: md`
        推理引擎把 KV cache 切成固定大小的**块**（vLLM 默认 16 个 token 一块），用 block table 把逻辑位置映射到物理块（PagedAttention）。
        开启**前缀缓存**后，一个满块的 KV 由“它之前的全部 token + 块内 token”唯一决定，于是可以：

        - 用 **哈希表** \`hash(父块哈希, 块内 token, 额外信息) → 物理块\` 查找能否复用；
        - 用 **引用计数** 记录有几个正在运行的请求在用这个块（共享系统提示词时 ref > 1）；
        - ref 归零的块不立刻清空，而是进入 **空闲队列**——vLLM v1 直接把前后指针放在块对象里，组成一条**双向链表**；
          分配新块时从队头取，若它还挂着旧哈希，就把哈希删掉（**淘汰**）。被命中的空闲块要从队列中间摘除，双向链表保证 $O(1)$。

        这正是 LRU 的“哈希表 + 双向链表”，只是多了“正在使用的块不参与淘汰”这一条。另外，请求结束时它的块按**逆序**加入队尾：
        靠后的块对应更长的前缀，被其它请求复用的概率更低，应该先被淘汰。
      `,
      viz: PrefixCacheViz,
      after: md`
        > [!INTERVIEW] 面试官想听到的
        > “前缀缓存的淘汰本质是 LRU：哈希表做块内容到物理块的映射，引用计数决定能不能淘汰，空闲块用双向链表维护 LRU 顺序，命中时 O(1) 从中间摘除。”
        > SGLang 的 RadixAttention 用基数树（radix tree）组织前缀，同样采用 LRU，从**叶子**开始淘汰（见「Trie 与前缀匹配」一关）。
      `,
      checkpoint: {
        id: 'cp-llm',
        type: 'multi',
        prompt: md`关于 vLLM 风格的前缀缓存块管理，哪些说法正确？（多选）`,
        options: [
          'ref_count 为 0 的块进入空闲队列，但其 KV 内容与哈希仍保留，可被后续请求命中复用',
          '空闲队列队头的块被重新分配时，如果它还挂着旧哈希，就要把这条哈希记录删掉',
          'ref_count 大于 0 的块也会按 LRU 顺序被淘汰',
          '命中一个处在空闲队列中间的块时，需要 O(n) 遍历队列才能把它移除',
        ],
        answer: [0, 1],
        explanation: md`C 错：正在被请求使用的块不在空闲队列里，不可能被淘汰。D 错：空闲队列是双向链表，块对象自带前后指针，从中间摘除是 $O(1)$——和 LRU 的 \`_remove\` 一模一样。`,
      },
    },
  ],

  /* ================================================================ */
  /* 随堂练习                                                          */
  /* ================================================================ */
  quiz: [
    {
      id: 'q-store-key',
      type: 'judge',
      difficulty: 1,
      prompt: md`手写 LRU 时，链表节点只需要存 value，key 存在哈希表里就够了。`,
      answer: false,
      explanation: md`淘汰时我们从链表拿到的是节点 \`tail.prev\`，要从哈希表删除它就必须知道它的 key。节点不存 key，就只能 $O(n)$ 遍历哈希表反查。`,
    },
    {
      id: 'q-sentinel',
      type: 'single',
      difficulty: 1,
      prompt: md`LRU 双向链表里的 head / tail 哨兵节点主要解决什么问题？`,
      options: ['让插入、删除不必处理“链表为空 / 节点在两端”的特殊情况', '加快哈希查找', '记录缓存容量', '保证线程安全'],
      answer: 0,
      explanation: md`哨兵让每个真实节点都有非空的 prev 和 next，\`_remove\` / \`_add_front\` 写成固定的几行，不需要任何 if。它与查找速度、容量、线程安全都无关。`,
    },
    {
      id: 'q-pick-lc146',
      type: 'pick',
      difficulty: 2,
      prompt: md`
        容量为 2 的 LRU 依次执行：\`put(1,1) put(2,2) get(1) put(3,3) get(2) put(4,4) get(1) get(3) get(4)\`。
        下面按顺序列出了其中的 5 次 get，哪些返回 **-1**？
      `,
      items: ['第 1 次 get(1)', 'get(2)', '第 2 次 get(1)', 'get(3)', 'get(4)'],
      answer: [1, 2],
      explanation: md`
        - \`get(1)\` 命中 → 顺序（新→旧）\`[1, 2]\`；\`put(3,3)\` 淘汰最旧的 2 → \`[3, 1]\`；
        - \`get(2)\` → -1；\`put(4,4)\` 淘汰最旧的 1 → \`[4, 3]\`；
        - \`get(1)\` → -1；\`get(3)\` → 3；\`get(4)\` → 4。

        易错点：第一次 \`get(1)\` 让 1 变成最新，所以 put(3) 淘汰的是 2 而不是 1。
      `,
    },
    {
      id: 'q-hits',
      type: 'numeric',
      difficulty: 2,
      prompt: md`容量为 3 的 LRU，依次访问 key：\`1 2 3 1 4 2 5 1 2 3\`（未命中时插入）。一共**命中**几次？`,
      answer: 2,
      tolerance: 0,
      explanation: md`
        顺序记为“新→旧”：1 2 3 未命中 → \`[3,2,1]\`；1 命中 → \`[1,3,2]\`；4 淘汰 2 → \`[4,1,3]\`；2 淘汰 3 → \`[2,4,1]\`；
        5 淘汰 1 → \`[5,2,4]\`；1 淘汰 4 → \`[1,5,2]\`；2 命中 → \`[2,1,5]\`；3 淘汰 5。共命中 2 次（第 4 次和第 9 次访问）。
      `,
    },
    {
      id: 'q-order-put',
      type: 'order',
      difficulty: 2,
      prompt: md`对一个**不存在**的 key 调用 \`put\`，且缓存已满。把步骤按正确顺序排列：`,
      items: [
        '在哈希表中确认 key 不存在',
        '取 lru = tail.prev（最久未使用）',
        '从链表摘下 lru，并 del map[lru.key]',
        '新建节点，插到 head 之后',
        'map[key] = 新节点',
      ],
      explanation: md`必须**先淘汰再插入**：若先插入，容量为 1 时 \`tail.prev\` 就是刚插入的新节点，会把自己淘汰掉。最后两步顺序可以互换。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把组件与它在缓存设计中承担的职责配对：`,
      pairs: [
        ['哈希表 key → 节点', 'O(1) 定位到节点'],
        ['双向链表', 'O(1) 调整使用顺序、摘除任意节点'],
        ['头尾哨兵节点', '消除空指针与边界判断'],
        ['LFU 的 min_freq', 'O(1) 找到要淘汰的频次桶'],
      ],
      explanation: md`四者各司其职：哈希表负责“找”，链表负责“排”，哨兵负责“省掉 if”，min_freq 让 LFU 不必扫描所有桶。`,
    },
    {
      id: 'q-bug-put',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 \`put\` 在缓存已满、插入新 key 时会抛异常。点出错误的行：`,
      code: py`
        def put(self, key, value):
            if key in self.map:
                node = self.map[key]
                node.val = value
                self._remove(node)
                self._add_front(node)
                return
            if len(self.map) == self.cap:
                lru = self.tail.prev
                self._remove(lru)
                del self.map[key]
            node = Node(key, value)
            self._add_front(node)
            self.map[key] = node
      `,
      bugLines: [11],
      explanation: md`第 11 行删的是**新 key**，它根本不在哈希表里，抛 KeyError；就算不报错，被淘汰的 lru 也还留在哈希表里，之后 \`get(lru.key)\` 会拿到已脱链的节点。应为 \`del self.map[lru.key]\`。`,
    },
    {
      id: 'q-ordereddict',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于用 Python 内置结构实现 LRU，哪些说法正确？（多选）`,
      options: [
        '`OrderedDict.move_to_end(k)` 是 O(1)',
        '`OrderedDict.popitem(last=False)` 弹出最早插入（最久未移动）的项',
        'Python 3.7+ 的普通 dict 也可以用 `d[k] = d.pop(k)` 把 k 移到末尾',
        '`OrderedDict` 的 `d[k]` 读取操作会自动把 k 移到末尾',
      ],
      answer: [0, 1, 2],
      explanation: md`OrderedDict 内部就是哈希表 + 双向链表，A、B 都是 $O(1)$。C 正确：dict 保持插入顺序，删除后重新插入就到了末尾。D 错：读取不会改变顺序，必须显式调用 \`move_to_end\`——这是用 OrderedDict 写 LRU 最常漏的一步。`,
    },
    {
      id: 'q-fill-lfu',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        LFU 维护 \`min_freq\`：
        - 插入一个**新** key 后，\`min_freq = ____(1)____\`；
        - 某次访问把 key 从桶 \`f\` 搬到桶 \`f + 1\` 后桶 \`f\` 变空，且 \`f == min_freq\`，则 \`min_freq = ____(2)____\`（用 f 表示）。
      `,
      blanks: [
        { label: '(1)', answers: ['1'], numeric: true },
        { label: '(2)', answers: ['f + 1', 'f+1', 'min_freq + 1', 'min_freq+1', 'self.min_freq + 1'] },
      ],
      explanation: md`新 key 频次为 1，一定是最小的；搬桶后最小频次只可能变成 $f+1$。这两条规则让 min_freq 始终 $O(1)$ 维护。`,
    },
    {
      id: 'q-lfu-evict',
      type: 'single',
      difficulty: 3,
      prompt: md`容量为 2 的 LFU 依次执行 \`put(1,1) put(2,2) get(1) put(3,3) get(2) get(3) put(4,4)\`。最后一次 \`put(4,4)\` 淘汰的是哪个 key？`,
      options: ['1', '2', '3', '不淘汰'],
      answer: 0,
      explanation: md`
        \`get(1)\` 后 1 的频次为 2；\`put(3,3)\` 时淘汰频次最低的 2；\`get(2)\` 返回 -1；\`get(3)\` 后 3 的频次为 2。
        此时 1 和 3 频次都是 2，按 LRU 打破平局：1 最后一次使用更早，淘汰 1。选 3 的同学是把“最近使用”理解反了。
      `,
    },
    {
      id: 'q-scan',
      type: 'judge',
      difficulty: 2,
      prompt: md`对 LRU 缓存做一次大范围顺序扫描（每个 key 只访问一次）不会影响缓存中原有的热点数据。`,
      answer: false,
      explanation: md`每个被扫描的 key 都会成为“最近使用”，容量满后原有热点会被逐个挤出——这叫**缓存污染**。LFU、LRU-K、2Q、W-TinyLFU 等策略都是为了抵抗这种扫描。`,
    },
    {
      id: 'q-vllm-reverse',
      type: 'single',
      difficulty: 3,
      prompt: md`vLLM 前缀缓存中，请求结束时它的块按**逆序**加入空闲队列尾部（最后一块最先入队，因此最先被淘汰）。这样设计的原因是？`,
      options: [
        '越靠后的块对应越长的前缀，被其它请求复用的概率越低，应优先淘汰',
        '为了让物理块地址保持连续',
        '因为双向链表只能从头部插入',
        '为了优先淘汰系统提示词所在的块',
      ],
      answer: 0,
      explanation: md`块哈希包含整个前缀，第 1 块（如系统提示词）被大量请求共享，最后一块只有完全相同的长前缀才能命中。逆序入队让“复用价值低”的尾部块先被淘汰。B 与 PagedAttention 的设计相反（物理块本就不需要连续）；C 错，双向链表两端都能 $O(1)$ 插入；D 恰好说反了。`,
    },
  ],

  /* ================================================================ */
  /* 跟练                                                              */
  /* ================================================================ */
  guided: [
    {
      id: 'g-lru',
      title: '手写 LRU 的链表原语与淘汰',
      intro: md`
        补全一个手写双向链表的 LRU：头尾哨兵已建好，\`head\` 侧是最近使用。你需要写出 \`_remove\`、\`_add_front\` 中的关键指针操作，以及淘汰逻辑。
      `,
      template: py`
        class Node:
            __slots__ = ("key", "val", "prev", "next")
            def __init__(self, key=0, val=0):
                self.key, self.val = key, val
                self.prev = self.next = None

        class LRUCache:
            def __init__(self, capacity):
                self.cap = capacity
                self.map = {}
                self.head, self.tail = Node(), Node()
                self.head.next, self.tail.prev = self.tail, self.head

            def _remove(self, node):
                {{1}}
                {{2}}

            def _add_front(self, node):
                node.prev, node.next = self.head, self.head.next
                {{3}}
                self.head.next = node

            def get(self, key):
                if key not in self.map:
                    return -1
                node = self.map[key]
                self._remove(node)
                self._add_front(node)
                return node.val

            def put(self, key, value):
                if key in self.map:
                    node = self.map[key]
                    node.val = value
                    self._remove(node)
                    self._add_front(node)
                    return
                if len(self.map) == self.cap:
                    lru = {{4}}
                    self._remove(lru)
                    {{5}}
                node = Node(key, value)
                self._add_front(node)
                self.map[key] = node
      `,
      steps: [
        {
          blank: 1,
          instruction: md`摘下节点第一步：让**前驱**跳过 node。`,
          answer: 'node.prev.next = node.next',
          explain: md`有哨兵，\`node.prev\` 一定不是 None。`,
        },
        {
          blank: 2,
          instruction: md`第二步：让**后继**的 prev 跳过 node。`,
          answer: 'node.next.prev = node.prev',
          explain: md`两行顺序可以互换：它们读的都是 node 自己的指针，而 node 的指针没有被修改。`,
        },
        {
          blank: 3,
          instruction: md`插入到 head 之后：先让原来的第一个节点的 prev 指向 node（必须在 \`self.head.next = node\` 之前）。`,
          answer: 'self.head.next.prev = node',
          accept: ['node.next.prev = node'],
          explain: md`此时 \`node.next\` 已经等于原来的 \`head.next\`，所以写 \`node.next.prev = node\` 也对。若放在 \`self.head.next = node\` 之后，\`self.head.next.prev\` 就成了 node 自己的 prev，原第一个节点的 prev 永远指着 head——经典 bug。`,
        },
        {
          blank: 4,
          instruction: md`缓存已满：取出最久未使用的节点。`,
          answer: 'self.tail.prev',
          explain: md`约定 head 侧最近、tail 侧最旧，所以 LRU 节点是 \`tail.prev\`（tail 本身是哨兵，不能淘汰）。`,
        },
        {
          blank: 5,
          instruction: md`从哈希表中删除被淘汰的节点。`,
          answer: 'del self.map[lru.key]',
          accept: ['self.map.pop(lru.key)', 'self.map.pop(lru.key, None)'],
          explain: md`这里用的是节点里存的 \`lru.key\`，而不是参数 key——这就是节点必须存 key 的原因。`,
        },
      ],
      setup: LRU_SETUP,
      tests: [
        {
          name: 'LC146 示例',
          code: py`
            ops = [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2), ("put", 4, 4), ("get", 1), ("get", 3), ("get", 4)]
            got = _run_ops(LRUCache, 2, ops)
            assert got == [None, None, 1, None, -1, None, -1, 3, 4], f"得到 {got}"
          `,
        },
        { name: '容量为 1', code: py`_check_lru(1, [("put", 2, 1), ("get", 2), ("put", 3, 2), ("get", 2), ("get", 3)])` },
        {
          name: '随机操作对拍',
          hidden: true,
          code: py`
            rnd = random.Random(146)
            for _ in range(150):
                _check_lru(rnd.randint(1, 4), _rand_ops(rnd, 40, 6))
          `,
        },
      ],
      outro: md`
        整个 LRU 只有两个原语，每个原语只改常数个指针。面试时先写 \`Node\` 和两个原语，再写 \`get\` / \`put\`，结构清晰、不容易出错。
      `,
    },
  ],

  /* ================================================================ */
  /* 实战                                                              */
  /* ================================================================ */
  challenges: [
    {
      id: 'c-lru',
      title: 'LRU 缓存（手写双向链表）',
      difficulty: 'medium',
      source: 'LeetCode 146',
      timeLimitMin: 20,
      description: md`
        设计并实现满足 LRU 约束的数据结构：

        \`\`\`python
        class LRUCache:
            def __init__(self, capacity: int): ...
            def get(self, key) -> int: ...          # 不存在返回 -1；存在则返回值，并记为“最近使用”
            def put(self, key, value) -> None: ...  # 存在则更新值并记为最近使用；不存在则插入，超出容量时先淘汰最久未使用的 key
        \`\`\`

        **要求**
        - \`get\` 和 \`put\` 平均 $O(1)$；
        - **不能使用** \`collections.OrderedDict\`（测试会检查源码）——请手写双向链表；
        - key 只保证**可哈希、可判等**，不一定是整数（隐藏测试会用自定义 key 对象统计比较次数：\`list.remove\`、\`deque.remove\`、\`index\` 等线性操作会被判为超时）。

        **示例**：容量 2，\`put(1,1) put(2,2) get(1)→1 put(3,3) get(2)→-1 put(4,4) get(1)→-1 get(3)→3 get(4)→4\`

        **约束**：$1 \le$ capacity $\le 3000$，操作次数 $\le 2 \times 10^4$。
      `,
      starter: py`
        class LRUCache:
            def __init__(self, capacity: int):
                # TODO: 哈希表 key -> 节点；双向链表（头尾哨兵），head 侧为最近使用
                raise NotImplementedError

            def get(self, key) -> int:
                raise NotImplementedError

            def put(self, key, value) -> None:
                raise NotImplementedError
      `,
      solution: py`
        class Node:
            __slots__ = ("key", "val", "prev", "next")
            def __init__(self, key=None, val=None):
                self.key, self.val = key, val
                self.prev = self.next = None

        class LRUCache:
            def __init__(self, capacity: int):
                self.cap = capacity
                self.map = {}                              # key -> Node
                self.head, self.tail = Node(), Node()      # 哨兵：head 后最近使用，tail 前最久未用
                self.head.next, self.tail.prev = self.tail, self.head

            def _remove(self, node):
                node.prev.next = node.next
                node.next.prev = node.prev

            def _add_front(self, node):
                node.prev, node.next = self.head, self.head.next
                self.head.next.prev = node
                self.head.next = node

            def get(self, key) -> int:
                node = self.map.get(key)
                if node is None:
                    return -1
                self._remove(node)
                self._add_front(node)
                return node.val

            def put(self, key, value) -> None:
                node = self.map.get(key)
                if node is not None:
                    node.val = value
                    self._remove(node)
                    self._add_front(node)
                    return
                if len(self.map) == self.cap:              # 先淘汰再插入
                    lru = self.tail.prev
                    self._remove(lru)
                    del self.map[lru.key]
                node = Node(key, value)
                self._add_front(node)
                self.map[key] = node
      `,
      solutionAlt: [
        {
          label: 'OrderedDict 版（本题禁止，仅作对照：面试时可先口述）',
          code: py`
            from collections import OrderedDict

            class LRUCache:
                def __init__(self, capacity):
                    self.cap, self.d = capacity, OrderedDict()

                def get(self, key):
                    if key not in self.d:
                        return -1
                    self.d.move_to_end(key)          # 读取不会自动调整顺序，必须显式移动
                    return self.d[key]

                def put(self, key, value):
                    if key in self.d:
                        self.d.move_to_end(key)
                    self.d[key] = value
                    if len(self.d) > self.cap:
                        self.d.popitem(last=False)   # 弹出最久未使用
          `,
        },
      ],
      explanation: md`
        **结构**：\`map: key → Node\` + 带头尾哨兵的双向链表，head 侧为最近使用。

        **操作**：
        - \`get\`：\`map.get\` 未命中返回 -1；命中则 \`_remove\` + \`_add_front\`；
        - \`put\`：已存在 → 改值并移到最前；否则若已满，淘汰 \`tail.prev\`（链表与哈希表都删，用 \`lru.key\`），再插入新节点。

        **复杂度**：每个操作常数次哈希与指针修改，$O(1)$；空间 $O(\text{capacity})$。

        **为什么测试用自定义 key**：用 \`list\` / \`deque\` 维护顺序时，\`remove(key)\` 要逐个比较元素，是 $O(n)$——
        正确性测试能过，但 capacity 很大时会超时。计数 key 让这种写法在几百次操作内就快速失败。
      `,
      complexity: { time: 'O(1) / 操作', space: 'O(capacity)' },
      hints: [
        md`哈希表的 value 存**链表节点**，节点里存 key、val、prev、next。`,
        md`建两个哨兵 \`head\`、\`tail\` 并互连；写 \`_remove(node)\` 与 \`_add_front(node)\` 两个原语。`,
        md`\`put\` 新 key 时：已满则 \`lru = self.tail.prev\`，\`_remove(lru)\`，\`del self.map[lru.key]\`，然后再插入。`,
      ],
      setup: LRU_SETUP,
      tests: [
        {
          name: 'LC146 示例',
          code: py`
            ops = [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2), ("put", 4, 4), ("get", 1), ("get", 3), ("get", 4)]
            got = _run_ops(LRUCache, 2, ops)
            exp = [None, None, 1, None, -1, None, -1, 3, 4]
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '更新已存在的 key 也算使用',
          code: py`_check_lru(2, [("put", 1, 1), ("put", 2, 2), ("put", 1, 10), ("put", 3, 3), ("get", 1), ("get", 2), ("get", 3)])`,
        },
        { name: '容量为 1', code: py`_check_lru(1, [("put", 2, 1), ("get", 2), ("put", 3, 2), ("get", 2), ("get", 3), ("put", 3, 5), ("get", 3)])` },
        {
          name: '不使用 OrderedDict',
          code: py`
            src = "".join(linecache.getlines("solution.py"))
            assert "OrderedDict" not in src, "本题要求手写双向链表，不能使用 OrderedDict"
            c = LRUCache(2)
            for v in vars(c).values():
                assert not isinstance(v, _OD), "本题要求手写双向链表，不能使用 OrderedDict"
          `,
        },
        {
          name: '随机操作对拍',
          hidden: true,
          code: py`
            rnd = random.Random(2024)
            for _ in range(300):
                _check_lru(rnd.randint(1, 5), _rand_ops(rnd, 50, 8))
          `,
        },
        {
          name: '字符串 key、元组 value',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            keys = ["kv", "prefix", "block", "attn", "moe", "rope"]
            for _ in range(100):
                ops = [("get", rnd.choice(keys)) if rnd.random() < 0.4 else ("put", rnd.choice(keys), (rnd.random(), "x")) for _ in range(30)]
                _check_lru(rnd.randint(1, 4), ops)
          `,
        },
        {
          name: 'O(1)：容量 2000、2 万次操作',
          hidden: true,
          code: py`
            rnd = random.Random(11)
            pool = [_Key(i) for i in range(4000)]
            ops = [("get", rnd.choice(pool)) if rnd.random() < 0.5 else ("put", rnd.choice(pool), i) for i in range(20000)]
            _Key.n, _Key.budget = 0, 40000
            t0 = time.perf_counter()
            try:
                got = _run_ops(LRUCache, 2000, ops)
            finally:
                _Key.budget = 10**12
            dt = time.perf_counter() - t0
            exp = _run_ops(_RefLRU, 2000, ops)
            assert got == exp, "大规模操作结果与参考实现不一致"
            assert dt < 1.5, f"用时 {dt:.2f}s，应为 O(1) / 操作"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-thread',
          q: md`多线程环境下怎么让 LRU 线程安全？高并发时怎么减少锁竞争？`,
          a: md`最简单是整个 get / put 加一把互斥锁（注意 get 也会修改链表，读写锁的读锁不够）。高并发时按 key 哈希**分片**，每个分片一把锁、一个独立的 LRU（近似全局 LRU）；或者像 Caffeine 那样把“访问记录”先写入缓冲区，批量重放到链表上，读路径无锁。`,
        },
        {
          id: 'f-ttl',
          q: md`如果每个 key 还有过期时间（TTL），怎么扩展？`,
          a: md`惰性删除：get 时发现过期就当未命中并删除；再加定期清理——按过期时间维护一个小顶堆（或时间轮），周期性弹出已过期的 key。淘汰时可以优先淘汰已过期的，再按 LRU。`,
        },
      ],
    },
    {
      id: 'c-lfu',
      title: 'LFU 缓存',
      difficulty: 'hard',
      source: 'LeetCode 460',
      timeLimitMin: 30,
      description: md`
        实现 LFU 缓存：

        \`\`\`python
        class LFUCache:
            def __init__(self, capacity: int): ...
            def get(self, key) -> int: ...          # 不存在返回 -1；存在则返回值，使用次数 +1
            def put(self, key, value) -> None: ...  # 存在则更新值（使用次数 +1）；不存在则插入（使用次数为 1）
        \`\`\`

        容量已满时插入新 key，要先淘汰**使用次数最少**的 key；若有多个，淘汰其中**最久未使用**的。

        **要求**：\`get\` / \`put\` 平均 $O(1)$（本题允许使用 \`OrderedDict\` 作为频次桶内的有序集合）。
        隐藏测试包含容量 2000、2 万次操作的序列，每次淘汰都扫描全部 key 的 $O(n)$ 写法会超时（测试按块计时，超过 1.5 秒直接判失败）。

        **示例**（容量 2）：\`put(1,1) put(2,2) get(1)→1 put(3,3) get(2)→-1 get(3)→3 put(4,4) get(1)→-1 get(3)→3 get(4)→4\`

        **约束**：$1 \le$ capacity $\le 10^4$。
      `,
      starter: py`
        class LFUCache:
            def __init__(self, capacity: int):
                # TODO: kv[key] = (value, freq)；buckets[freq] = 有序集合；min_freq
                raise NotImplementedError

            def get(self, key) -> int:
                raise NotImplementedError

            def put(self, key, value) -> None:
                raise NotImplementedError
      `,
      solution: py`
        from collections import defaultdict, OrderedDict

        class LFUCache:
            def __init__(self, capacity: int):
                self.cap = capacity
                self.kv = {}                              # key -> [value, freq]
                self.buckets = defaultdict(OrderedDict)   # freq -> 有序集合（最久未用在前）
                self.min_freq = 0

            def _touch(self, key):
                """把 key 的频次 +1：从桶 f 搬到桶 f+1 的末尾（最近使用）"""
                f = self.kv[key][1]
                bucket = self.buckets[f]
                del bucket[key]
                if not bucket:
                    del self.buckets[f]
                    if self.min_freq == f:
                        self.min_freq = f + 1
                self.kv[key][1] = f + 1
                self.buckets[f + 1][key] = None

            def get(self, key) -> int:
                if key not in self.kv:
                    return -1
                self._touch(key)
                return self.kv[key][0]

            def put(self, key, value) -> None:
                if self.cap <= 0:
                    return
                if key in self.kv:
                    self.kv[key][0] = value
                    self._touch(key)
                    return
                if len(self.kv) == self.cap:
                    bucket = self.buckets[self.min_freq]
                    old, _ = bucket.popitem(last=False)    # 最小频次里最久未用的
                    if not bucket:
                        del self.buckets[self.min_freq]
                    del self.kv[old]
                self.kv[key] = [value, 1]
                self.buckets[1][key] = None
                self.min_freq = 1                          # 新 key 频次最小
      `,
      solutionAlt: [
        {
          label: '用普通 dict 作有序集合（Python 3.7+ 保持插入顺序）',
          code: py`
            from collections import defaultdict

            class LFUCache:
                def __init__(self, capacity):
                    self.cap, self.kv, self.min_freq = capacity, {}, 0
                    self.buckets = defaultdict(dict)        # freq -> {key: None}，插入顺序即使用顺序

                def _touch(self, key):
                    f = self.kv[key][1]
                    del self.buckets[f][key]
                    if not self.buckets[f]:
                        del self.buckets[f]
                        if self.min_freq == f:
                            self.min_freq = f + 1
                    self.kv[key][1] = f + 1
                    self.buckets[f + 1][key] = None

                def get(self, key):
                    if key not in self.kv:
                        return -1
                    self._touch(key)
                    return self.kv[key][0]

                def put(self, key, value):
                    if key in self.kv:
                        self.kv[key][0] = value
                        self._touch(key)
                        return
                    if len(self.kv) == self.cap:
                        b = self.buckets[self.min_freq]
                        old = next(iter(b))               # 最早插入 = 最久未用
                        del b[old]
                        if not b:
                            del self.buckets[self.min_freq]
                        del self.kv[old]
                    self.kv[key] = [value, 1]
                    self.buckets[1][key] = None
                    self.min_freq = 1
          `,
        },
      ],
      explanation: md`
        **结构**：\`kv[key] = [value, freq]\`；\`buckets[freq]\` 是按使用时间排序的有序集合（每个频次一条小 LRU 链）；\`min_freq\`。

        **\`_touch(key)\`**：从桶 $f$ 删掉、加到桶 $f+1$ 末尾；若桶 $f$ 空了且 \`min_freq == f\`，则 \`min_freq = f + 1\`。

        **淘汰**：去 \`buckets[min_freq]\` 弹出最旧的 key（\`popitem(last=False)\`）。新 key 插入后 \`min_freq = 1\`。

        **复杂度**：所有操作 $O(1)$ 平均；空间 $O(\text{capacity})$。

        **常见错误**
        - \`put\` 已存在的 key 时没有增加频次（LC460 规定更新也算一次使用）；
        - 淘汰后试图重新计算 min_freq（不需要，马上会被置为 1）；
        - 桶空了不删，长期运行时 buckets 越积越多（不影响正确性，但浪费内存）；
        - 同频次平局时淘汰了“最近使用”的那个。
      `,
      complexity: { time: 'O(1) / 操作', space: 'O(capacity)' },
      hints: [
        md`把问题拆成“每个频次一个 LRU”：\`buckets[freq]\` 用 OrderedDict（或 dict）当有序集合。`,
        md`写一个 \`_touch(key)\` 负责把频次 +1；get 命中和 put 已存在时都调用它。`,
        md`维护 \`min_freq\`：搬桶后若旧桶空且等于 min_freq 则 +1；插入新 key 后置 1。淘汰时 \`buckets[min_freq].popitem(last=False)\`。`,
      ],
      setup: py`
        import random, time

        class _RefLFU:
            """朴素参考实现：O(n) 淘汰（只用于小规模对拍）"""
            def __init__(self, cap):
                self.cap, self.d, self.t = cap, {}, 0      # key -> [value, freq, last_time]
            def get(self, k):
                if k not in self.d:
                    return -1
                self.t += 1
                e = self.d[k]
                e[1] += 1
                e[2] = self.t
                return e[0]
            def put(self, k, v):
                self.t += 1
                if k in self.d:
                    e = self.d[k]
                    e[0], e[1], e[2] = v, e[1] + 1, self.t
                    return
                if len(self.d) == self.cap:
                    victim = min(self.d, key=lambda x: (self.d[x][1], self.d[x][2]))
                    del self.d[victim]
                self.d[k] = [v, 1, self.t]

        def _run_ops(cls, cap, ops):
            c, out = cls(cap), []
            for op in ops:
                if op[0] == "get":
                    out.append(c.get(op[1]))
                else:
                    c.put(op[1], op[2])
                    out.append(None)
            return out

        def _check_lfu(cap, ops):
            got = _run_ops(LFUCache, cap, ops)
            exp = _run_ops(_RefLFU, cap, ops)
            if got != exp:
                i = next(i for i, (a, b) in enumerate(zip(got, exp)) if a != b)
                raise AssertionError(f"capacity={cap}，第 {i} 个操作 {ops[i]} 起不一致：期望 {exp[i]}，得到 {got[i]}（前序操作：{ops[:i]}）")
      `,
      tests: [
        {
          name: 'LC460 示例',
          code: py`
            ops = [("put", 1, 1), ("put", 2, 2), ("get", 1), ("put", 3, 3), ("get", 2), ("get", 3), ("put", 4, 4), ("get", 1), ("get", 3), ("get", 4)]
            got = _run_ops(LFUCache, 2, ops)
            exp = [None, None, 1, None, -1, 3, None, -1, 3, 4]
            assert got == exp, f"期望 {exp}，得到 {got}"
          `,
        },
        { name: '更新已存在的 key 计入频次', code: py`_check_lfu(2, [("put", 1, 1), ("put", 2, 2), ("put", 1, 5), ("put", 3, 3), ("get", 1), ("get", 2), ("get", 3)])` },
        { name: '容量为 1', code: py`_check_lfu(1, [("put", 1, 1), ("get", 1), ("put", 2, 2), ("get", 1), ("get", 2)])` },
        {
          name: '随机操作对拍',
          hidden: true,
          code: py`
            rnd = random.Random(460)
            for _ in range(300):
                cap = rnd.randint(1, 5)
                ops = [("get", rnd.randint(1, 8)) if rnd.random() < 0.5 else ("put", rnd.randint(1, 8), rnd.randint(0, 99)) for _ in range(50)]
                _check_lfu(cap, ops)
          `,
        },
        {
          name: 'O(1)：容量 2000、2 万次操作（分块计时）',
          hidden: true,
          code: py`
            rnd = random.Random(9)
            c = LFUCache(2000)
            t0 = time.perf_counter()
            hits = 0
            for blk in range(40):
                for i in range(500):
                    k = rnd.randint(0, 6000)
                    if rnd.random() < 0.3:
                        hits += c.get(k) != -1
                    else:
                        c.put(k, i)
                dt = time.perf_counter() - t0
                assert dt < 1.5, f"前 {(blk + 1) * 500} 次操作已用时 {dt:.2f}s：淘汰是不是在扫描全部 key（O(n)）？"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-heap',
          q: md`用堆实现 LFU 可以吗？复杂度是多少？`,
          a: md`可以：堆元素为 (freq, last_time, key)，频次变化时不删除旧元素，而是压入新元素并在弹出时检查是否过期（惰性删除）。单次操作 $O(\log n)$，堆中可能积累大量过期元素，需要定期重建。频次桶方案是严格 $O(1)$。`,
        },
        {
          id: 'f-decay',
          q: md`LFU 的“历史包袱”问题怎么缓解？`,
          a: md`对频次做衰减：定期把所有频次减半，或像 Redis 那样用对数计数器（访问越多，计数增长概率越低）并按空闲时间衰减；也可以用 W-TinyLFU：用 Count-Min Sketch 近似统计频次，配合一个小的 LRU 窗口接纳新 key。`,
        },
      ],
    },
  ],

  /* ================================================================ */
  /* 面试追问                                                           */
  /* ================================================================ */
  interview: [
    {
      id: 'qa-lru-why',
      freq: 3,
      q: md`手写 LRU 缓存，并解释为什么是哈希表 + 双向链表。`,
      a: md`
        **结论**：哈希表负责 $O(1)$ 定位节点，双向链表负责 $O(1)$ 调整顺序与删除，二者缺一不可。

        **原因**：只有哈希表无法知道谁最久未用；只有链表查找是 $O(n)$；单链表删除任意节点要找前驱，$O(n)$。
        哈希表的 value 存节点本身，节点存 key / val / prev / next；头尾哨兵消除边界判断。get：命中则移到表头；
        put：存在则改值并移到表头，否则满了先淘汰 \`tail.prev\`（同时用 \`node.key\` 删哈希项）再插入。

        **延伸**：Python 可以用 OrderedDict（\`move_to_end\`、\`popitem(last=False)\`）几行写完，但面试通常要求手写；
        多线程需要加锁或分片；LFU 是常见的追问（频次桶 + min_freq）。
      `,
    },
    {
      id: 'qa-lru-vs-lfu',
      freq: 2,
      q: md`LRU 和 LFU 各有什么优缺点？实际系统怎么选？`,
      a: md`
        **结论**：LRU 实现简单、对访问模式的变化反应快，但会被一次性扫描污染；LFU 能保护长期热点，但对“过气热点”淘汰太慢、实现更复杂。

        **原因**：LRU 只看最近一次访问时间；LFU 只看累计次数，历史上的高频 key 即使不再访问也很难被淘汰。

        **延伸**：工程上多用折中方案：LRU-K / 2Q（访问两次才进入主缓存）、ARC（自适应调节 LRU 与 LFU 部分的比例）、
        W-TinyLFU（Caffeine 默认，频次用 Count-Min Sketch 近似）。Redis 的淘汰策略本身就是**近似 LRU / LFU**（随机采样少量 key 比较，而不是维护全局链表）。
      `,
    },
    {
      id: 'qa-kv-evict',
      freq: 3,
      q: md`大模型推理引擎（如 vLLM）的前缀缓存是怎么做淘汰的？和 LRU 有什么关系？`,
      a: md`
        **结论**：本质是“哈希表 + 引用计数 + 双向链表空闲队列”的 LRU。

        **原因**：KV cache 按块管理，每个满块用（父块哈希、块内 token、LoRA / 多模态等额外信息）计算哈希，哈希表把哈希映射到物理块；
        引用计数表示有几个运行中的请求在用它，ref > 0 的块不能淘汰；ref 归零后块进入空闲队列但保留内容，仍可被命中。
        分配新块时从队头取（最久未用），若它挂着旧哈希则删除——即淘汰；命中空闲块时要从队列中间摘除，双向链表保证 $O(1)$。

        **延伸**：请求结束时块按逆序入队，让长前缀的尾块先被淘汰；SGLang 的 RadixAttention 用基数树组织前缀，同样按 LRU 从叶子开始淘汰。
        追问还可能涉及：多轮对话、共享系统提示词的命中率，以及块大小对命中率与碎片的权衡。
      `,
    },
    {
      id: 'qa-ordereddict',
      freq: 2,
      q: md`Python 的 OrderedDict 是怎么实现 O(1) 的 move_to_end 的？普通 dict 能不能代替？`,
      a: md`
        **结论**：OrderedDict 内部额外维护了一条双向链表记录顺序，\`move_to_end\` 和 \`popitem\` 都是 $O(1)$；普通 dict 在 3.7+ 保持插入顺序，也能凑出 LRU。

        **原因**：OrderedDict = 哈希表 + 双向链表（C 实现），和手写 LRU 完全同构。普通 dict 用“删除再插入”把 key 移到末尾，
        用 \`next(iter(d))\` 取最早的 key；但它的有序性来自紧凑数组，频繁删除会留下空洞，取首个 key 可能需要跳过空洞（扩容重排时才清理）。

        **延伸**：OrderedDict 还支持 \`move_to_end(k, last=False)\` 把 key 移到开头，以及顺序敏感的相等比较，这些普通 dict 都没有。
      `,
    },
  ],

  cheatsheet: md`
    **LRU = 哈希表（key → 节点）+ 双向链表（头尾哨兵）**
    \`\`\`python
    def _remove(self, n):  n.prev.next = n.next; n.next.prev = n.prev
    def _add_front(self, n):
        n.prev, n.next = self.head, self.head.next
        self.head.next.prev = n; self.head.next = n
    # get: 命中 → remove + add_front
    # put: 存在 → 改值 + 移前；满 → lru = tail.prev; remove; del map[lru.key]
    \`\`\`
    **LFU**：\`kv[key]=[val,freq]\` + \`buckets[freq]\`（有序集合）+ \`min_freq\`
    - 搬桶后旧桶空且等于 min_freq → \`min_freq += 1\`；新 key → \`min_freq = 1\`
    - 淘汰：\`buckets[min_freq].popitem(last=False)\`

    - 节点必须存 key；先淘汰再插入；put 已存在也算使用
    - OrderedDict：\`move_to_end\` / \`popitem(last=False)\`，读取不会自动移动
    - vLLM 前缀缓存：块哈希 → 物理块；ref=0 进空闲双向链表；队头淘汰；请求结束块逆序入队
  `,
})
