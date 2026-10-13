/**
 * algo-graph 章节公共 Python 片段。
 *
 * PY_TREE：二叉树题统一的运行环境（拼接到 setup 最前面）。
 *   - TreeNode(val, left, right)                与 LeetCode 定义一致
 *   - list_to_tree([3, 9, 20, None, None, 15, 7]) 层序列表 → 树（None 表示空孩子）
 *   - tree_to_list(root)                        树 → 层序列表（去掉末尾 None）
 *   以上三个名字对用户公开（题面里会说明）；其余辅助函数一律以 _ 开头。
 *
 * ⚠️ Pyodide 限制：自定义节点结构一次最多 1500 个节点，树深控制在几百以内，
 *    测试结束在 finally 里调用 _free_tree(root) 逐个断开引用。
 */
import { py } from '@/content/helpers'

export const PY_TREE = py`
  import sys, random
  from collections import deque

  class TreeNode:
      def __init__(self, val=0, left=None, right=None):
          self.val = val
          self.left = left
          self.right = right

      def __repr__(self):
          return f"TreeNode({self.val})"

  def list_to_tree(vals):
      """LeetCode 层序列表 → 二叉树。None 只给“存在的节点”的空孩子占位。"""
      if not vals or vals[0] is None:
          return None
      root = TreeNode(vals[0])
      q, i = deque([root]), 1
      while q and i < len(vals):
          node = q.popleft()
          if i < len(vals) and vals[i] is not None:
              node.left = TreeNode(vals[i])
              q.append(node.left)
          i += 1
          if i < len(vals) and vals[i] is not None:
              node.right = TreeNode(vals[i])
              q.append(node.right)
          i += 1
      return root

  def tree_to_list(root):
      """二叉树 → LeetCode 层序列表（去掉末尾多余的 None）。"""
      out, q = [], deque([root])
      while q:
          node = q.popleft()
          if node is None:
              out.append(None)
              continue
          out.append(node.val)
          q.append(node.left)
          q.append(node.right)
      while out and out[-1] is None:
          out.pop()
      return out

  def _nodes(root):
      """迭代收集全部节点（前序）。"""
      out, st = [], [root] if root else []
      while st:
          x = st.pop()
          out.append(x)
          if x.right: st.append(x.right)
          if x.left: st.append(x.left)
      return out

  def _free_tree(*roots):
      """逐个断开 left/right，避免大树在释放时递归析构撑爆 Pyodide 的栈。"""
      for r in roots:
          for x in _nodes(r):
              x.left = x.right = None

  def _height(root):
      h, q = 0, [root] if root else []
      while q:
          h += 1
          q = [c for x in q for c in (x.left, x.right) if c]
      return h

  def _rand_tree(rnd, n, lo=-50, hi=50):
      """随机形状的二叉树：n 个节点，深度约 O(log n)（随机划分左右子树大小）。"""
      if n <= 0:
          return None
      root = TreeNode(rnd.randint(lo, hi))
      st = [(root, n)]
      while st:
          node, size = st.pop()
          rest = size - 1
          k = rnd.randint(0, rest)
          if k:
              node.left = TreeNode(rnd.randint(lo, hi))
              st.append((node.left, k))
          if rest - k:
              node.right = TreeNode(rnd.randint(lo, hi))
              st.append((node.right, rest - k))
      return root

  def _chain(vals, side="left"):
      """退化成链的树（用来卡递归）。"""
      root = None
      for v in reversed(vals):
          node = TreeNode(v)
          if root is not None:
              setattr(node, side, root)
          root = node
      return root

  def _frame_depth():
      f, d = sys._getframe(), 0
      while f is not None:
          d += 1
          f = f.f_back
      return d

  class _NoDeepRecursion:
      """with 块内只允许再嵌套 extra 层 Python 调用：递归遍历深树会触发 RecursionError。"""
      def __init__(self, extra=40):
          self.extra = extra
      def __enter__(self):
          self.old = sys.getrecursionlimit()
          sys.setrecursionlimit(_frame_depth() + self.extra)
          return self
      def __exit__(self, *exc):
          sys.setrecursionlimit(self.old)
          return False
`
