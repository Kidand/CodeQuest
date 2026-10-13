import { defineLevel, md, py } from '@/content/helpers'
import { TrieWalkViz } from '../viz/AdvancedViz'
export default defineLevel({
 learn:[{id:'review',title:'先辨认状态与依赖，再选择遍历结构',viz:TrieWalkViz,body:md`二叉树后序把孩子摘要交给父亲，图搜索用 visited 避免重复扩展，拓扑排序处理依赖，并查集处理无向连通性，Dijkstra 处理非负权最短路，Trie 处理共享前缀。这些结构不能仅凭题目名称互换，先写清节点、边、状态与目标。

 BOSS 将棋盘和词典组成一个乘积状态：棋盘坐标决定下一步能走哪里，Trie 节点决定下一字符还可能匹配什么，当前路径的 visited 决定哪些格子不能再用。任一条件失败都可剪枝；找到词只代表一个答案结束，不代表更长词的搜索也应结束。`}],
 quiz:[
 {id:'q1',type:'single',difficulty:1,prompt:md`无权图最短路优先选择哪种遍历？`,options:['BFS','任意 DFS 首次命中','并查集'],answer:0,explanation:md`BFS 按边数分层；DFS 首次找到的路径未必最短，并查集不保存距离。`},
 {id:'q2',type:'judge',difficulty:1,prompt:md`拓扑排序只输出少于 V 个节点，说明原有向图中有环。`,answer:true,explanation:md`Kahn 算法无法继续删除零入度点时，剩余子图必有环；有向无环图一定能完整输出。`},
 {id:'q3',type:'numeric',difficulty:2,prompt:md`根值 1，左右孩子分别为 2、3。最大路径和是多少？`,answer:6,tolerance:0,explanation:md`完整路径 2→1→3 和为 6；返回给根的父亲的单边贡献则是 4。`},
 {id:'q4',type:'fill',difficulty:2,prompt:md`Dijkstra 要求边权 ____ 0。填写比较符。`,blanks:[{answers:['>=']}],explanation:md`非负性支撑贪心确定距离；零权边合法，负边即使没有负环也可能破坏证明。`},
 {id:'q5',type:'multi',difficulty:2,prompt:md`多词棋盘搜索中哪些状态不可省略？`,options:['当前棋盘坐标','当前 Trie 节点','当前路径已用格子','全部历史失败路径的永久格子黑名单'],answer:[0,1,2],explanation:md`前三者决定合法下一步；失败路径不构成格子的永久禁用理由，黑名单会误剪。`},
 {id:'q6',type:'match',difficulty:2,prompt:md`配对信息维护方式：`,pairs:[['重建树的形状','前序加空指针标记'],['动态合并无向组','并查集根代表'],['计算图依赖顺序','拓扑排序']],explanation:md`树形状需要空孩子信息，分组需要等价类，依赖执行需要有向无环关系；不能只用节点值列表解决全部问题。`},
 {id:'q7',type:'order',difficulty:2,prompt:md`Trie + 棋盘 DFS 的一次分支流程：`,items:['检查字符边存在且格子未使用','标记格子并进入对应 Trie 节点','若有词尾则记录答案，继续探索可能的更长词','撤销格子标记'],explanation:md`词尾并不一定是叶子，例如同时有 car 与 cars。记录 car 后仍需继续寻找 cars。`},
 {id:'q8',type:'single',difficulty:3,prompt:md`词典中重复出现同一个词，题目只要求唯一输出，最简单的输出去重方法是？`,options:['找到后清除当前 Trie 节点的词尾标记','永久禁止使用该词最后一个棋盘格','遇到重复词时删除整棵 Trie'],answer:0,explanation:md`词尾标记只影响这个词的重复输出，不影响更长词或其它路径。删除棋盘格会漏解其它词，整树更不合理。`},
 {id:'q9',type:'numeric',difficulty:3,prompt:md`图中 4 个节点，边为 0→1、1→2、2→1，节点 3 独立。Kahn 能输出几个节点？`,answer:2,tolerance:0,explanation:md`最初 0、3 入度为 0，处理 0 后 1 仍有来自 2 的入边；1、2 构成环，最终只输出 2 个。`},
 {id:'q10',type:'judge',difficulty:1,prompt:md`树高等于 n 的退化树，递归遍历栈空间可能达到 O(n)。`,answer:true,explanation:md`递归空间取决于树高，不保证 O(log n)。深树应考虑显式栈，且仍需计算其最坏空间。`},
 ],guided:[],
 challenges:[{id:'c-search',title:'BOSS：单词搜索 II',difficulty:'hard',source:'LC 212',timeLimitMin:40,description:md`实现 find_words(board: list[list[str]], words: list[str]) -> list[str]。棋盘为矩形小写字母，四邻移动，每条路径不能重复格子。返回能在棋盘拼出的词，输出去重且顺序不限；词表中的词非空但可能重复。棋盘≤6×6，词长≤10，词表≤500。不得修改输入。空棋盘或空词表返回 []。应共享词典前缀剪枝，不对每个词完整重复搜索。`,starter:py`
  def find_words(board: list[list[str]], words: list[str]) -> list[str]:
      raise NotImplementedError
 `,solution:py`
  def find_words(board: list[list[str]], words: list[str]) -> list[str]:
      if not board or not board[0] or not words:
          return []
      root = {}
      for word in words:
          node = root
          for ch in word:
              node = node.setdefault(ch,{})
          node[None] = word
      m,n = len(board),len(board[0])
      used,out = set(),[]
      def dfs(r,c,parent):
          if not (0<=r<m and 0<=c<n) or (r,c) in used:
              return
          ch=board[r][c]
          if ch not in parent:
              return
          node=parent[ch]
          word=node.pop(None,None)
          if word is not None:
              out.append(word)
          used.add((r,c))
          for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
              dfs(r+dr,c+dc,node)
          used.remove((r,c))
          if not node:
              del parent[ch]
      for r in range(m):
          for c in range(n):
              dfs(r,c,root)
      return out
 `,explanation:md`词典构建 Trie 后，一次棋盘搜索共享所有词的前缀。词尾保存完整词，找到后清除以去重；如果节点已无词尾与孩子，就从父节点删除这条边，减少以后无意义的探索。used 只属于当前路径，遍历结束恢复。建树 O(S)，搜索保守上界 O(mn·4^L)，实际由 Trie 前缀大幅剪枝；空间 O(S+L)。`,hints:[md`搜索状态同时包括棋盘坐标和 Trie 节点。`,md`找到一个词时不能停止，可能还有更长词共享该前缀。`,md`清除已找到的词尾；空分支可以剪掉。`],setup:py`
  import random
  def _ref_word(board,word):
      if not board or not board[0]: return False
      m,n=len(board),len(board[0])
      def visit(r,c,i,used):
          if not(0<=r<m and 0<=c<n) or (r,c) in used or board[r][c]!=word[i]: return False
          if i==len(word)-1: return True
          return any(visit(r+dr,c+dc,i+1,used|{(r,c)}) for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))
      return any(visit(r,c,0,set()) for r in range(m) for c in range(n))
  def _check_words(board,words):
      before=[row[:] for row in board]; old=words[:]; exp={w for w in words if _ref_word(board,w)}; got=find_words(board,words)
      assert len(got)==len(set(got)) and set(got)==exp, f"期望 {sorted(exp)} 且无重复，得到 {got}"
      assert board==before and words==old, '输入棋盘和词表不得修改'
 `,tests:[{name:'典型多词',code:py`_check_words([list('oaan'),list('etae'),list('ihkr'),list('iflv')],['oath','pea','eat','rain'])`},{name:'共享词尾与重复',code:py`_check_words([list('cars')],['car','cars','car','ar','ss'])`},{name:'空输入',hidden:true,code:py`_check_words([],['a']); _check_words([[]],['a']); _check_words([['a']],[])`},{name:'随机逐词暴力对拍',hidden:true,code:py`
  rnd=random.Random(212)
  for _ in range(60):
      b=[[rnd.choice('abc') for _ in range(3)] for _ in range(3)]
      words=[''.join(rnd.choice('abc') for _ in range(rnd.randrange(1,6))) for _ in range(20)]
      _check_words(b,words)
 `},{name:'同格不可复用',hidden:true,code:py`_check_words([['a','b']],['a','ab','aba','b','ba','bab'])`}]}],interview:[],cheatsheet:md`图题先定义状态、方向、权重与目标。树后序返回摘要；BFS 求无权距离；拓扑判依赖环；UF 求无向组；Dijkstra 需非负边；Trie 共享前缀。棋盘多词搜索 = 坐标 × Trie 节点 × 当前路径 visited。词尾可清除，格子要恢复。`,
})
