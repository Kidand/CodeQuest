/**
 * algo-ds 章节公共 Python 片段（测试 setup 用）。
 *  - LISTNODE_SETUP：ListNode 定义 + 构造 / 转换 / 断链辅助函数（Pyodide 链表 ≤ 1500 节点，finally 中 _free）
 *  - COUNTED_NUM：计数数值包装 _Num（比较 / 相等 / 算术都计数，超预算立即 AssertionError）——让超复杂度写法快速失败
 */
import { py } from '@/content/helpers'

export const LISTNODE_SETUP = py`
  class ListNode:
      def __init__(self, val=0, next=None):
          self.val = val
          self.next = next
      def __repr__(self):
          return f"ListNode({self.val})"

  def _build(vals, pos=-1):
      """由列表建链表；pos >= 0 时尾节点接回下标 pos 处的节点（造环）。返回 (head, nodes)"""
      nodes = [ListNode(v) for v in vals]
      for a, b in zip(nodes, nodes[1:]):
          a.next = b
      if pos >= 0 and nodes:
          nodes[-1].next = nodes[pos]
      return (nodes[0] if nodes else None), nodes

  def _free(nodes):
      # 逐个断开 next：长链表整体释放时会递归析构，Pyodide 的栈较浅，几千层就会崩溃
      for x in nodes:
          x.next = None

  def _to_list(head, limit=5000):
      out = []
      while head is not None:
          out.append(head.val)
          head = head.next
          if len(out) > limit:
              raise AssertionError("结果链表中出现了环（或长度异常）——是不是有 next 忘了断开 / 接回？")
      return out
`

export const COUNTED_NUM = py`
  class _Num:
      """计数的数值包装：比较、相等、算术每次 +1，超过预算立即失败（复杂度检查用）"""
      __slots__ = ("v",)
      n = 0
      budget = 10**12
      msg = "操作次数超出预算：复杂度不达标"
      def __init__(self, v):
          self.v = v
      @staticmethod
      def _t():
          _Num.n += 1
          if _Num.n > _Num.budget:
              raise AssertionError(f"{_Num.msg}（预算 {_Num.budget} 次）")
      @staticmethod
      def _u(o):
          return o.v if isinstance(o, _Num) else o
      def __lt__(s, o): _Num._t(); return s.v < _Num._u(o)
      def __le__(s, o): _Num._t(); return s.v <= _Num._u(o)
      def __gt__(s, o): _Num._t(); return s.v > _Num._u(o)
      def __ge__(s, o): _Num._t(); return s.v >= _Num._u(o)
      def __eq__(s, o): _Num._t(); return s.v == _Num._u(o)
      def __ne__(s, o): _Num._t(); return s.v != _Num._u(o)
      def __hash__(s): return hash(s.v)
      def __add__(s, o): _Num._t(); return s.v + _Num._u(o)
      def __radd__(s, o): _Num._t(); return _Num._u(o) + s.v
      def __sub__(s, o): _Num._t(); return s.v - _Num._u(o)
      def __rsub__(s, o): _Num._t(); return _Num._u(o) - s.v
      def __mul__(s, o): _Num._t(); return s.v * _Num._u(o)
      def __rmul__(s, o): _Num._t(); return _Num._u(o) * s.v
      def __floordiv__(s, o): _Num._t(); return s.v // _Num._u(o)
      def __truediv__(s, o): _Num._t(); return s.v / _Num._u(o)
      def __mod__(s, o): _Num._t(); return s.v % _Num._u(o)
      def __neg__(s): return -s.v
      def __abs__(s): return abs(s.v)
      def __int__(s): return int(s.v)
      def __index__(s): return int(s.v)
      def __float__(s): return float(s.v)
      def __bool__(s): return bool(s.v)
      def __repr__(s): return repr(s.v)

  def _unwrap(x):
      if isinstance(x, _Num):
          return x.v
      if isinstance(x, (list, tuple)):
          return type(x)(_unwrap(y) for y in x)
      return x

  def _with_budget(budget, msg, fn, *args):
      """在操作预算内调用 fn(*args)；返回 (结果, 实际操作次数)"""
      _Num.n, _Num.budget, _Num.msg = 0, budget, msg
      try:
          return fn(*args), _Num.n
      finally:
          _Num.budget = 10**12
`
