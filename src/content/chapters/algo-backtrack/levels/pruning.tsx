import { defineLevel, md, py } from '@/content/helpers'
import { QueensViz } from '../viz/ConstraintViz'
export default defineLevel({
 learn:[
  {id:'queens',title:'约束越早检查，搜索树越小',viz:QueensViz,body:md`N 皇后逐行放置，每行只有一个皇后，因此不必再检查行冲突。选择列 c 后，用三组标记分别记录 c、r-c、r+c。两个格子在同一主对角线上当且仅当行减列相同，在同一副对角线上当且仅当行加列相同。每次放置前 O(1) 检查标记，冲突就跳过整棵子树。

  放置时加入三组标记，递归结束必须逐个移除。撤销不是优化，而是保证兄弟分支从同一父状态出发的正确性条件。最坏搜索仍呈指数增长，剪枝减少实际访问节点，不能因此声称多项式复杂度；输出所有方案本身也有不可省略的成本。`},
  {id:'grid',title:'单词搜索的 visited 只属于当前路径',body:md`棋盘寻找单词时，状态包含当前位置、匹配到的字符下标和当前已用格子。先检查边界、字符和访问标记，再尝试四邻。一个格子不能在同一路径重复使用，但回溯之后可以被其它路径使用，所以 visited 必须撤销。直接把它变成全局永久标记会漏解。

  可以先比较字符频次：目标某字符需求超过棋盘供给时立即失败。还可从出现次数更少的端点开始搜索，因为四邻路径可反向，反转目标词不会改变是否存在。这些都是保持正确性的剪枝。若在原棋盘上暂时写入哨兵，即使提前找到答案也要恢复，避免调用者输入被污染。`},
  {id:'constraints',title:'括号与回文切割：剪枝来自不变量',body:md`生成 n 对括号时，任意前缀必须满足右括号数不超过左括号数，且左右总数不能超过 n。只在 left<n 时加左括号，只在 right<left 时加右括号；这种构造从源头排除无效前缀，比生成所有 2^(2n) 串再筛选有效得多。

  回文切割先预处理 pal[i][j]，再只对回文片段扩展。单纯缓存“位置 i 已访问”不合法，因为不同历史路径在 i 之后虽然拥有相同后缀选择，却应产生不同完整答案。要区分判定、计数、枚举三个目标；可缓存的状态与是否需要保留路径取决于输出要求。`},
 ],
 quiz:[
  {id:'q1',type:'single',difficulty:1,prompt:md`逐行放 N 皇后时，哪项不需要额外集合检查？`,options:['行','列','主对角线'],answer:0,explanation:md`递归深度已经保证每行只放一个。列和两种对角线仍可能冲突。`},
  {id:'q2',type:'numeric',difficulty:2,prompt:md`格子 (2,4) 的主对角线标识 r-c 为多少？`,answer:-2,tolerance:0,explanation:md`2-4=-2，可以是负数；若用数组应加 n-1 偏移，集合则直接存整数。`},
  {id:'q3',type:'judge',difficulty:1,prompt:md`单词搜索找到答案并提前返回时，可以不恢复被改写的棋盘。`,answer:false,explanation:md`函数约定不修改输入时，即使成功也必须恢复。否则后续调用会受到之前搜索状态影响。`},
  {id:'q4',type:'fill',difficulty:2,prompt:md`生成括号时，已用 left 个左括号、right 个右括号，允许追加右括号的条件是 ____。`,blanks:[{answers:['right < left','left > right']}],explanation:md`任意前缀的右括号不能超过左括号，所以追加前必须严格小于。只检查 right<n 不够。`},
  {id:'q5',type:'multi',difficulty:2,prompt:md`单词搜索中哪些剪枝保证安全？`,options:['目标字符频次超过棋盘供给则失败','每条路径不允许重复格子','发现某格子失败后永久禁止其它路径访问它','从更稀有的端点反向搜索'],answer:[0,1,3],explanation:md`失败依赖当前前缀和已用集合，不能跨状态永久禁止格子。频次和反向路径都是必要条件或对称变换。`},
  {id:'q6',type:'order',difficulty:2,prompt:md`回溯的一个合法分支按顺序执行：`,items:['检查当前选择是否违反约束','记录选择并更新标记','递归探索下一层','移除选择并恢复标记'],explanation:md`恢复状态后才能探索兄弟分支；否则先前路径的占用会错误影响下一条路径。`},
  {id:'q7',type:'single',difficulty:3,prompt:md`枚举回文切割时，只用 visited[i] 记忆化某个后缀是否被探索过是否正确？`,options:['不正确，不同前缀需要组合相同后缀生成不同答案','正确，后缀位置唯一决定完整答案','仅字符串无重复字符时正确'],answer:0,explanation:md`后缀的可行切割可以缓存，但必须与每个历史前缀组合；一个布尔已访问标记会直接漏掉这些组合。`},
  {id:'q8',type:'numeric',difficulty:3,prompt:md`N=4 的标准 N 皇后共有多少种不同棋盘解（旋转或镜像仍视为不同）？`,answer:2,tolerance:0,explanation:md`列序列 [1,3,0,2] 与 [2,0,3,1] 是两种解；若按对称等价类计数才会合并成一种。`},
 ],
 guided:[{id:'g-parentheses',title:'只生成合法括号前缀',intro:md`parentheses(n) 返回所有 n 对合法括号串，n=0 返回 ['']。`,template:py`
  def parentheses(n):
      out = []
      def dfs(text, left, right):
          if len(text) == 2*n:
              out.append(text)
              return
          if {{1}}:
              dfs(text+'(', left+1, right)
          if {{2}}:
              dfs(text+')', left, {{3}})
      dfs('', 0, 0)
      return out
 `,steps:[{blank:1,instruction:md`左括号的使用数量不能超过 n。`,answer:'left < n'},{blank:2,instruction:md`右括号不能让前缀失衡。`,answer:'right < left'},{blank:3,instruction:md`加入右括号后更新它的数量。`,answer:'right + 1'}],tests:[{name:'两对括号',code:py`got=sorted(parentheses(2)); assert got==['(())','()()'], f"得到 {got}"`},{name:'边界',code:py`assert parentheses(0)==['']; assert parentheses(1)==['()']`}]}],
 challenges:[
 {id:'c-queens',title:'N 皇后',difficulty:'hard',source:'LC 51',description:md`实现 solve_n_queens(n: int) -> list[list[str]]，返回所有合法棋盘，每行长 n，以 Q 表示皇后、. 表示空位。1≤n≤8，每行每列恰好一个皇后且对角线不冲突；解的顺序不限，不允许重复。例 n=1 → [['Q']]；n=2 → []。`,starter:py`
  def solve_n_queens(n: int) -> list[list[str]]:
      raise NotImplementedError
 `,solution:py`
  def solve_n_queens(n: int) -> list[list[str]]:
      cols, down, up = set(), set(), set()
      path, out = [], []
      def dfs(r):
          if r == n:
              out.append(['.'*c+'Q'+'.'*(n-c-1) for c in path])
              return
          for c in range(n):
              if c in cols or r-c in down or r+c in up:
                  continue
              cols.add(c); down.add(r-c); up.add(r+c); path.append(c)
              dfs(r+1)
              path.pop(); up.remove(r+c); down.remove(r-c); cols.remove(c)
      dfs(0)
      return out
 `,explanation:md`递归深度表示行，用列和两种对角线集合检查冲突。一个分支使用的标记必须严格成对加入和删除。忽略输出构造，搜索上界可写 O(n·n!)；构造 S 个棋盘另需 O(S n²)。辅助搜索空间 O(n)，输出单独计。`,hints:[md`行由递归层数保证，剩余约束是列与两条对角线。`,md`主对角线看 r-c，副对角线看 r+c。`],setup:py`
  def _check_queens(n, boards, expected):
      assert len(boards)==expected, f"n={n} 期望 {expected} 个解，得到 {len(boards)}"
      seen=set()
      for board in boards:
          assert len(board)==n and all(len(row)==n and set(row)<={'.','Q'} and row.count('Q')==1 for row in board), f"棋盘格式错误 {board}"
          cols=[row.index('Q') for row in board]
          assert len(set(cols))==n and len({r-c for r,c in enumerate(cols)})==n and len({r+c for r,c in enumerate(cols)})==n, f"皇后冲突 {board}"
          key=tuple(board); assert key not in seen, f"重复方案 {board}"; seen.add(key)
 `,tests:[{name:'单点',code:py`_check_queens(1,solve_n_queens(1),1)`},{name:'无解与四皇后',code:py`_check_queens(2,solve_n_queens(2),0); _check_queens(4,solve_n_queens(4),2)`},{name:'五至七皇后',hidden:true,code:py`
  for n,count in [(3,0),(5,10),(6,4),(7,40)]: _check_queens(n,solve_n_queens(n),count)
 `},{name:'八皇后及重复调用',hidden:true,code:py`_check_queens(8,solve_n_queens(8),92); _check_queens(4,solve_n_queens(4),2)`}]},
 {id:'c-word',title:'棋盘单词搜索',difficulty:'medium',source:'LC 79',description:md`实现 exist(board: list[list[str]], word: str) -> bool，四邻移动且同一路径不能重复使用格子。board 为矩形，可为空；word 为空时返回 True。大小≤6×6，word 长度≤12。不得修改输入。例 [['A','B'],['C','D']] 中 ABD 存在，ABA 不存在。`,starter:py`
  def exist(board: list[list[str]], word: str) -> bool:
      raise NotImplementedError
 `,solution:py`
  from collections import Counter
  def exist(board: list[list[str]], word: str) -> bool:
      if not word:
          return True
      if not board or not board[0]:
          return False
      counts = Counter(ch for row in board for ch in row)
      need = Counter(word)
      if any(need[ch] > counts[ch] for ch in need):
          return False
      if counts[word[0]] > counts[word[-1]]:
          word = word[::-1]
      m,n = len(board),len(board[0])
      used = set()
      def dfs(r,c,i):
          if not (0<=r<m and 0<=c<n) or (r,c) in used or board[r][c]!=word[i]:
              return False
          if i == len(word)-1:
              return True
          used.add((r,c))
          found = any(dfs(r+dr,c+dc,i+1) for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))
          used.remove((r,c))
          return found
      return any(dfs(r,c,0) for r in range(m) for c in range(n))
 `,explanation:md`先做字符频次必要条件检查，再从较稀有端点开始。used 只记录当前路径，any 即使短路成功，也会执行后面的 remove。使用集合不改棋盘，避免恢复遗漏。最坏时间 O(mn·4^L)，路径空间 O(L)，计数空间与字符种类有关。`,hints:[md`状态包含已用格子，不能只按坐标记忆化。`,md`先做必要条件剪枝，再保证每次搜索都恢复状态。`],setup:py`
  import random
  def _ref_exist(board,word):
      if not word: return True
      if not board or not board[0]: return False
      m,n=len(board),len(board[0])
      def visit(r,c,i,used):
          if not (0<=r<m and 0<=c<n) or (r,c) in used or board[r][c]!=word[i]: return False
          if i+1==len(word): return True
          return any(visit(r+dr,c+dc,i+1,used|{(r,c)}) for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))
      return any(visit(r,c,0,set()) for r in range(m) for c in range(n))
 `,tests:[{name:'示例与复用禁止',code:py`b=[['A','B'],['C','D']]; assert exist(b,'ABD') is True; assert exist(b,'ABA') is False; assert b==[['A','B'],['C','D']]`},{name:'空边界',code:py`assert exist([],'') is True; assert exist([],'A') is False; assert exist([[]],'A') is False`},{name:'随机路径对拍',hidden:true,code:py`
  rnd=random.Random(79)
  for _ in range(100):
      b=[[rnd.choice('ABC') for _ in range(3)] for _ in range(3)]; before=[row[:] for row in b]; word=''.join(rnd.choice('ABC') for _ in range(rnd.randrange(7)))
      exp=_ref_exist(b,word); got=exist(b,word); assert got==exp and b==before, f"词 {word}：期望 {exp} 且输入不变，得到 {got}"
 `},{name:'频次剪枝',hidden:true,code:py`assert exist([['A']*6 for _ in range(6)],'AAAAAAAAAAAB') is False`}]},
 ],
 interview:[
 {id:'i-prune',freq:3,q:md`剪枝和去重有什么区别？`,a:md`剪枝排除不可能得到合法答案或更好答案的分支，去重排除会生成重复结果的分支。N 皇后冲突属于可行性剪枝，含重复元素的组合跳过同层同值属于去重。两者都减少搜索量，但证明不同：前者证明没有解，后者证明另一个保留分支已覆盖完全相同结果。不能仅凭“看起来像”就删除分支。`},
 {id:'i-state',freq:3,q:md`回溯状态应该原地改还是复制？`,a:md`两种方式都能正确。复制状态容易隔离分支，但每层会有额外分配与拷贝；原地修改更省开销，却要求所有退出路径恢复，包括提前成功和异常情况。小规模面试题可用 set 或 path 的添加与删除保持清晰。输出列表必须复制当前路径，否则后续撤销会把已保存结果一并改变。`},
 {id:'i-memo',freq:2,q:md`为什么单词搜索不直接缓存 (r,c,i)？`,a:md`相同位置和字符下标可能由不同已用格子集合到达，未来可走路径不同，因此这三个量不是完整状态。简单缓存失败会把其它合法路径误剪。将 visited 集合也编码进状态可以恢复正确性，但状态数可能巨大，得不偿失。只有在额外约束下未来与历史无关时，才能安全省略历史信息。`},
 ],cheatsheet:md`N 皇后冲突：col、r-c、r+c；做选择与撤销成对。括号：left<n 才加左，right<left 才加右。网格：visited 是当前路径，不是全局。频次不足可剪枝，稀有端点可反向。记忆化状态必须决定全部未来可能。`,
})
