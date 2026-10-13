import { defineLevel, md, py } from '@/content/helpers'
import { PY_TREE } from '../shared'
import { TreeGainViz } from '../viz/AdvancedViz'
export default defineLevel({
 learn:[
  {id:'gain',title:'后序：向上返回什么，与全局答案分开',viz:TreeGainViz,body:md`最大路径和的路径不能分叉。经过当前节点的最佳完整路径可以连接左右两条非负贡献，候选是 value+left+right；但返回给父亲时只能保留其中一边，否则再连父亲就产生三叉。返回值因此是 value+max(left,right)。

  每个孩子返回后先与 0 取最大，表示负贡献可以不选。全局答案必须初始化为负无穷，不能初始化为 0，因为题目要求至少一个节点；全负树应选择最大的单个节点。这个“局部摘要返回、完整答案另更新”的框架也适用于直径、平衡性与树形动态规划。`},
  {id:'lca',title:'公共祖先与 BST：充分的信息才能证明正确',body:md`最近公共祖先递归中，如果左右子树分别找到目标，当前节点就是分叉处；如果仅一侧返回非空，继续向上传递。这个经典模板默认两个目标都在树中，目标可能缺失时必须再返回已找到的目标数，不能把唯一找到的节点当成答案。

  验证二叉搜索树不能只比较直接孩子：左子树的所有值都必须小于根。可以递归传递开区间上下界，或用中序序列严格递增来验证。重复值是否允许要事先约定。重建树时，前序给根、中序给左右划分；值唯一时预建下标表避免每层切片和线性搜索造成平方复杂度。`},
  {id:'codec',title:'序列化必须保存空孩子',body:md`只保存前序节点值无法区分“左孩子 2”和“右孩子 2”。给空指针加入特殊标记 # 后，前序文法变为“一个值，左子树，右子树”，即可唯一还原结构。解析器按顺序消费 token：遇到 # 返回空，遇到整数则递归解析两个孩子。

  含 n 个真实节点的二叉树恰有 n+1 个空指针，因此这种编码有 2n+1 个 token。编码与解码时间均为 O(n)，输出占 O(n)，递归栈占 O(h)。树很深时递归可能溢出，应改为显式栈；本关测试将深度限制在 200 以内，重点验证结构和负数、多位整数。`},
 ],
 quiz:[
  {id:'q1',type:'single',difficulty:1,prompt:md`最大路径和返回给父节点的量是什么？`,options:['以当前节点为起点的最佳单边路径和','以当前节点为中心连接双边的路径和','整棵子树所有节点之和'],answer:0,explanation:md`父节点还会延长这条路径，必须保持不分叉。双边候选只能用来更新完整答案；子树总和通常不是路径。`},
  {id:'q2',type:'numeric',difficulty:2,prompt:md`节点值 -10，左侧单边贡献 9，右侧单边贡献 35，经过当前点的候选路径和是多少？`,answer:34,tolerance:0,explanation:md`候选 -10+9+35=34，而向上返回 -10+35=25。这两个量不能混为一谈。`},
  {id:'q3',type:'judge',difficulty:1,prompt:md`全负树最大路径和应为 0，因为所有负贡献都可以舍弃。`,answer:false,explanation:md`可以舍弃左右边，但路径必须包含至少一个节点。答案应为最大的节点值。`},
  {id:'q4',type:'fill',difficulty:2,prompt:md`真实节点 n=5，完整前序空标记编码中，空标记数是 ____。`,blanks:[{answers:['6'],numeric:true}],explanation:md`2n 个孩子指针中 n-1 个连接真实边，其余 n+1=6 个为空。`},
  {id:'q5',type:'multi',difficulty:2,prompt:md`哪些说法正确？`,options:['BST 的约束必须传递到所有后代','标准 LCA 模板通常假设两个目标存在','前序值序列不加空标记总能唯一恢复树','负数节点需要保留符号'],answer:[0,1,3],explanation:md`直接父子比较不足以检查 BST；LCA 的存在性影响接口；单独前序值缺失结构，不能唯一恢复。`},
  {id:'q6',type:'order',difficulty:2,prompt:md`后序最大路径和在当前节点的计算顺序：`,items:['获取左右孩子单边贡献并截断负值','使用 value+left+right 更新答案','向父节点返回 value+max(left,right)'],explanation:md`孩子信息必须先准备好；全局候选与向上摘要承担不同用途。`},
  {id:'q7',type:'single',difficulty:3,prompt:md`树根 5，右孩子 7，7 的左孩子 4。这是 BST 吗？`,options:['不是，4 违反祖先 5 的下界','是，每个节点只需与父亲比较','取决于是否平衡'],answer:0,explanation:md`4 在根 5 的右子树中，必须大于 5；平衡性与搜索树有序性无关。`},
  {id:'q8',type:'numeric',difficulty:3,prompt:md`前序编码 1,#,2,#,# 含多少个真实节点？`,answer:2,tolerance:0,explanation:md`根为 1，左空，右孩子 2 且两个孩子为空；共两个真实节点、三个空标记。`},
 ],
 guided:[{id:'g-depth',title:'后序求二叉树直径',intro:md`diameter(root) 返回最长路径的边数；空树返回 0。环境提供 TreeNode。`,template:py`
  def diameter(root):
      best = 0
      def height(node):
          nonlocal best
          if node is None:
              return {{1}}
          left, right = height(node.left), height(node.right)
          best = max(best, {{2}})
          return {{3}}
      height(root)
      return best
 `,steps:[{blank:1,instruction:md`空树高度按节点数定义为 0。`,answer:'0'},{blank:2,instruction:md`通过当前节点连接两边，边数等于两侧子树节点高度之和。`,answer:'left + right'},{blank:3,instruction:md`返回单边高度。`,answer:'1 + max(left, right)'}],setup:PY_TREE,tests:[{name:'分叉树',code:py`
  root=list_to_tree([1,2,3,4,5])
  try:
      got=diameter(root); assert got==3, f"期望 3，得到 {got}"
  finally: _free_tree(root)
 `},{name:'空树与单点',code:py`assert diameter(None)==0; assert diameter(TreeNode(1))==0`}]}],
 challenges:[
 {id:'c-max',title:'二叉树最大路径和',difficulty:'hard',source:'LC 124',description:md`实现 max_path_sum(root) -> int。环境提供 TreeNode，非空树，节点值为整数，路径至少含一个节点且不能重复经过节点。n≤1500，测试树深≤200。返回最大路径和，要求 O(n) 时间、O(h) 栈空间。例：[-10,9,20,None,None,15,7] → 42。`,starter:py`
  def max_path_sum(root) -> int:
      raise NotImplementedError
 `,solution:py`
  def max_path_sum(root) -> int:
      best = float('-inf')
      def gain(node):
          nonlocal best
          if node is None:
              return 0
          left = max(0, gain(node.left))
          right = max(0, gain(node.right))
          best = max(best, node.val + left + right)
          return node.val + max(left, right)
      gain(root)
      return best
 `,explanation:md`后序返回以当前节点为起点的单边贡献，经过当前节点的完整路径可以连左右两边。负贡献截断为 0，但当前节点不能省略，所以全局答案从负无穷开始。每个节点访问一次，时间 O(n)，栈 O(h)。`,hints:[md`分别定义“返回父亲的量”和“完整答案”。`,md`父亲只能连接当前节点的一条向下路径。`],setup:PY_TREE+"\n\n"+py`
  def _ref_max(root):
      nodes=_nodes(root); adj={x:[] for x in nodes}
      for x in nodes:
          for y in (x.left,x.right):
              if y: adj[x].append(y); adj[y].append(x)
      best=float('-inf')
      for start in nodes:
          stack=[(start,None,0)]
          while stack:
              x,prev,total=stack.pop(); total+=x.val; best=max(best,total)
              stack.extend((y,x,total) for y in adj[x] if y is not prev)
      return best
 `,tests:[{name:'示例',code:py`
  r=list_to_tree([-10,9,20,None,None,15,7])
  try:
      got=max_path_sum(r); assert got==42, f"期望 42，得到 {got}"
  finally: _free_tree(r)
 `},{name:'全负',code:py`
  r=list_to_tree([-3,-2,-8])
  try:
      got=max_path_sum(r); assert got==-2, f"期望 -2，得到 {got}"
  finally: _free_tree(r)
 `},{name:'随机路径枚举对拍',hidden:true,code:py`
  rnd=random.Random(124)
  for _ in range(70):
      r=_rand_tree(rnd,rnd.randrange(1,22),-10,10)
      try:
          exp=_ref_max(r); got=max_path_sum(r); assert got==exp, f"期望 {exp}，得到 {got}"
      finally: _free_tree(r)
 `},{name:'单链',hidden:true,code:py`
  r=_chain([1]*150)
  try:
      got=max_path_sum(r); assert got==150, f"期望 150，得到 {got}"
  finally: _free_tree(r)
 `}]},
 {id:'c-codec',title:'序列化与反序列化二叉树',difficulty:'hard',source:'LC 297',description:md`实现 serialize(root) -> str 与 deserialize(data: str)。使用逗号分隔的前序格式，空指针记为 #，负数和多位整数按十进制表示。输入 data 保证合法；空树编码为 #。环境提供 TreeNode，n≤1500、深≤200。要求 O(n) 时间。例：根 1、右孩子 2 编码为 1,#,2,#,#。`,starter:py`
  def serialize(root) -> str:
      raise NotImplementedError
  def deserialize(data: str):
      raise NotImplementedError
 `,solution:py`
  def serialize(root) -> str:
      tokens = []
      def visit(node):
          if node is None:
              tokens.append('#')
              return
          tokens.append(str(node.val))
          visit(node.left)
          visit(node.right)
      visit(root)
      return ','.join(tokens)

  def deserialize(data: str):
      tokens = iter(data.split(','))
      def read():
          value = next(tokens)
          if value == '#':
              return None
          node = TreeNode(int(value))
          node.left = read()
          node.right = read()
          return node
      return read()
 `,explanation:md`前序加空指针标记是无歧义文法。解析器每次消耗一个 token，再递归消费两个子树。编码使用列表收集后 join，避免逐次拼接长字符串；解码使用迭代器，避免 pop(0) 引入平方搬移。`,hints:[md`空指针也是结构的一部分。`,md`每次递归只消费自己所代表的子树。`],setup:PY_TREE,tests:[{name:'规定编码',code:py`
  r=list_to_tree([1,None,2]); out=None
  try:
      got=serialize(r); assert got=='1,#,2,#,#', f"得到 {got}"
      out=deserialize(got); assert tree_to_list(out)==[1,None,2]
  finally: _free_tree(r,out)
 `},{name:'空树',code:py`assert serialize(None)=='#'; assert deserialize('#') is None`},{name:'负数重复值',hidden:true,code:py`
  r=list_to_tree([-12,5,5,None,-7]); out=None
  try:
      out=deserialize(serialize(r)); got=tree_to_list(out); exp=tree_to_list(r); assert got==exp, f"期望 {exp}，得到 {got}"
  finally: _free_tree(r,out)
 `},{name:'随机往返结构',hidden:true,code:py`
  rnd=random.Random(297)
  for _ in range(60):
      r=_rand_tree(rnd,rnd.randrange(70)); out=None
      try:
          s=serialize(r); out=deserialize(s); got=tree_to_list(out); exp=tree_to_list(r)
          assert got==exp and serialize(out)==s, f"期望 {exp}，得到 {got}"
      finally: _free_tree(r,out)
 `}]},
 ],
 interview:[
 {id:'i-post',freq:3,q:md`怎样设计树递归的返回值？`,a:md`先站在父节点角度，只返回它组合答案所需的信息。例如最大路径和需要孩子的一条向下路径，平衡检查需要高度与是否合法，BST 检查需要范围信息。全局完整答案未必适合直接返回。写出返回值定义、空树值和组合公式，再写递归，会比先套模板再修边界可靠。`},
 {id:'i-lca',freq:2,q:md`最近公共祖先中有一个目标不存在怎么办？`,a:md`经典非空结果传播模板会返回唯一找到的目标，因此不适合直接使用。可以先检查两个对象都存在，或一次后序返回“找到多少目标”和候选祖先。当总数不足二时返回空。还要比较节点身份而非节点值，否则重复值会让目标判断错误。目标相同的情况也应在接口约定中明确。`},
 {id:'i-rec',freq:2,q:md`树递归的空间复杂度为什么不是总是 O(log n)？`,a:md`空间由同时存在的递归调用深度决定，等于树高 h。平衡树高为 O(log n)，退化链高为 O(n)。显式栈可以避免语言递归深度限制，但不会自动把最坏辅助空间降为常数。只有 Morris 等改变指针的特殊遍历能用常数额外空间，还需要保证结构恢复。`},
 ],cheatsheet:md`最大路径和：L=max(0,gain(left))，R=max(0,gain(right))；答案更新 val+L+R，返回 val+max(L,R)。全负树初始化 -∞。直径按边数：左右高度之和。BST 传祖先上下界。前序空标记：n 节点对应 2n+1 token。递归栈空间 O(h)。`,
})
