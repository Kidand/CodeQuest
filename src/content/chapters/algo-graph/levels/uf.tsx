import { defineLevel, md, py } from '@/content/helpers'
import { DijkstraViz, UnionFindViz } from '../viz/AdvancedViz'
export default defineLevel({
 learn:[
  {id:'uf',title:'并查集：只维护等价类，不保存路径',viz:UnionFindViz,body:md`并查集用一片父指针森林表示集合，根节点代表集合。find 沿父指针找到根并压缩路径；union 先求两个根，只把小树根挂到大树根。若两个根相同，本来就连通，不能再次减少集合数。路径压缩和按大小合并结合后，操作序列的摊还复杂度为 O(α(n))。

  近重复文档分组可以把候选相似对看成无向边，合并连通分量。但相似性本身未必传递：A 近似 B、B 近似 C，不意味着 A 近似 C。并查集求的是传递闭包；是否整个分量只保留一个代表，需要由数据质量策略决定。并查集也不能直接回答两点之间的具体路径。`},
  {id:'shortest',title:'Dijkstra 的贪心成立依赖非负边',viz:DijkstraViz,body:md`Dijkstra 每次从最小堆取出暂定距离最小的节点，再松弛它的出边。所有边非负时，绕经其它尚未确定节点的路径不可能更短，因此此时距离可最终确定。不能在第一次发现节点时就标记完成：可能稍后经另一条边得到更短路径。

  Python heapq 不提供方便的原地 decrease-key，常用做法是每次改进都加入新条目。弹出时若记录距离不等于 dist[u]，说明是过期条目，跳过即可。稀疏图复杂度常写 O((V+E) log V)；允许平行边的懒堆实现也可更精确写为 O((V+E) log(E+1))。`},
  {id:'choose',title:'最短路算法由边权和查询范围决定',body:md`无权图或等权图用 BFS，层数就是最短距离；边权只有 0 和 1 可以用双端队列做 0-1 BFS。非负一般权重用 Dijkstra；有负边时使用 Bellman-Ford，最坏 O(VE)，还可检测源点可达的负环。存在可达负环时，某些点没有有限最短距离。

  Floyd 通过逐个允许中间点来更新所有点对距离，O(V³) 时间、O(V²) 空间，适合规模较小的稠密图。并查集与这些算法不互相替代：它回答连通性，忽略边权与路径方向。把有向图当成无向图合并，会丢失可达性信息。`},
 ],
 quiz:[
  {id:'q1',type:'single',difficulty:1,prompt:md`union(a,b) 应连接什么？`,options:['两个集合的根','直接把 a 指向 b，不做 find','所有成员两两相连'],answer:0,explanation:md`根连接维护森林和集合语义；随意改普通节点可能只移动部分子树或成环，两两连边没有必要。`},
  {id:'q2',type:'judge',difficulty:1,prompt:md`并查集合并近重复边后，同组任意两份文档都必然直接达到相似阈值。`,answer:false,explanation:md`连通分量只保证存在相似边构成的链，相似阈值关系一般不传递。`},
  {id:'q3',type:'numeric',difficulty:2,prompt:md`5 个孤立点依次 union(0,1)、union(1,2)、union(0,2)、union(3,4)，最后有几个集合？`,answer:2,tolerance:0,explanation:md`只有三次成功合并，第三次连接已连通的点，不改变集合数，5-3=2。`},
  {id:'q4',type:'fill',difficulty:2,prompt:md`懒堆 Dijkstra 弹出 (d,u) 后，若 d ____ dist[u] 就跳过。填写比较符。`,blanks:[{answers:['!=','>']}],explanation:md`有效条目等于当前最短估计，过期条目更大。没有负边时两种写法在该实现里等价。`},
  {id:'q5',type:'multi',difficulty:2,prompt:md`哪些条件适合使用标准 Dijkstra？`,options:['边权可以为 0','所有边权非负','存在负边但没有负环','图可以有不可达节点'],answer:[0,1,3],explanation:md`零边不会破坏非负前提；不可达点保留无穷距离。无负环仍不足以保证 Dijkstra 的贪心正确。`},
  {id:'q6',type:'match',difficulty:2,prompt:md`匹配任务与合适的算法：`,pairs:[['无权单源最短路','BFS'],['含负边单源最短路','Bellman-Ford'],['小规模所有点对最短路','Floyd']],explanation:md`边权性质和查询范围决定选择；不能只因为题面出现“图”就套同一个模板。`},
  {id:'q7',type:'numeric',difficulty:3,prompt:md`有向边 0→1 权重 4，0→2 权重 1，2→1 权重 2，1→3 权重 1。0 到 3 最短距离是多少？`,answer:4,tolerance:0,explanation:md`路径 0→2→1→3 长度 1+2+1=4；第一次发现节点 1 的距离 4 还不是最终答案。`},
  {id:'q8',type:'single',difficulty:3,prompt:md`并查集直接删除一条已合并的边，能 O(α(n)) 恢复正确连通分量吗？`,options:['不能，合并已丢失足够的连接历史','可以，把两个端点 parent 设为自己','可以，集合数加一即可'],answer:0,explanation:md`另一条路径可能仍保持连通；删除任意边是动态连通性问题，需要额外结构或离线回滚方案。`},
 ],
 guided:[{id:'g-find',title:'路径折半与按大小合并',intro:md`实现 groups(n,edges)，点编号为 0..n-1，返回无向连通分量数量。`,template:py`
  def groups(n, edges):
      parent, size = list(range(n)), [1]*n
      count = n
      def find(x):
          while parent[x] != x:
              parent[x] = {{1}}
              x = parent[x]
          return x
      for a, b in edges:
          a, b = find(a), find(b)
          if a == b:
              continue
          if size[a] < size[b]:
              a, b = b, a
          parent[b] = {{2}}
          size[a] += size[b]
          count -= {{3}}
      return count
 `,steps:[{blank:1,instruction:md`把 x 直接连到祖父，实现路径折半。`,answer:'parent[parent[x]]'},{blank:2,instruction:md`小树根 b 挂到大树根 a。`,answer:'a'},{blank:3,instruction:md`一次成功合并减少一个分量。`,answer:'1'}],tests:[{name:'重复边与自环',code:py`got=groups(4,[(0,1),(1,0),(2,2)]); assert got==3, f"期望 3，得到 {got}"`},{name:'空与链',code:py`assert groups(0,[])==0; assert groups(3,[(0,1),(1,2)])==1`}]}],
 challenges:[
 {id:'c-uf',title:'实现 UnionFind 与文档分组',difficulty:'medium',description:md`实现 UnionFind(n)，暴露 count（分量数）、find(x)（代表）、union(a,b)（成功合并返回 True，否则 False）。编号 0..n-1，n≤100000。路径压缩配合按大小合并，平均摊还近似常数时间。近重复边按 union 输入后，同根文档构成一组；代表编号不要求最小。`,starter:py`
  class UnionFind:
      def __init__(self, n: int):
          raise NotImplementedError
      def find(self, x: int) -> int:
          raise NotImplementedError
      def union(self, a: int, b: int) -> bool:
          raise NotImplementedError
 `,solution:py`
  class UnionFind:
      def __init__(self, n: int):
          self.parent = list(range(n))
          self.size = [1] * n
          self.count = n
      def find(self, x: int) -> int:
          while self.parent[x] != x:
              self.parent[x] = self.parent[self.parent[x]]
              x = self.parent[x]
          return x
      def union(self, a: int, b: int) -> bool:
          a, b = self.find(a), self.find(b)
          if a == b:
              return False
          if self.size[a] < self.size[b]:
              a, b = b, a
          self.parent[b] = a
          self.size[a] += self.size[b]
          self.count -= 1
          return True
 `,explanation:md`find 路径折半，union 只连接根并把小集合挂到大集合。重复合并和自环不减少 count。初始化 O(n)，后续摊还 O(α(n))，空间 O(n)。文档分组只是把相似边转成连通分量，不能推断任意同组文档直接相似。`,hints:[md`所有合并前先找代表。`,md`只有根不同才更新大小和分量数。`],setup:py`import random`,tests:[{name:'基本接口',code:py`u=UnionFind(4); assert u.union(0,1) is True; assert u.union(1,0) is False; assert u.find(0)==u.find(1); assert u.count==3`},{name:'空与自环',code:py`assert UnionFind(0).count==0; u=UnionFind(1); assert u.union(0,0) is False; assert u.count==1`},{name:'随机显式分组对拍',hidden:true,code:py`
  rnd=random.Random(547); n=35; u=UnionFind(n); labels=list(range(n))
  for _ in range(300):
      a,b=rnd.randrange(n),rnd.randrange(n); x,y=labels[a],labels[b]; exp=x!=y; got=u.union(a,b)
      if exp: labels=[x if v==y else v for v in labels]
      assert got is exp and u.count==len(set(labels)), f"期望合并 {exp}、分量 {len(set(labels))}，得到 {got}、{u.count}"
      assert all((u.find(i)==u.find(j))==(labels[i]==labels[j]) for i in range(n) for j in range(n))
 `},{name:'重复合并',hidden:true,code:py`u=UnionFind(2000); results=[u.union(i,i+1) for i in range(1999)]; assert all(results) and u.count==1; assert not u.union(0,1999)`}]},
 {id:'c-delay',title:'网络延迟：堆优化最短路',difficulty:'medium',source:'LC 743',description:md`实现 network_delay(times: list[list[int]], n: int, k: int) -> int。节点 1..n，有向边 [u,v,w]，w≥0，允许平行边。返回源点 k 到所有点最短距离的最大值；有不可达点返回 -1。1≤n≤10000，边数≤50000。例：[[2,1,1],[2,3,1],[3,4,1]],4,2 → 2。`,starter:py`
  def network_delay(times: list[list[int]], n: int, k: int) -> int:
      raise NotImplementedError
 `,solution:py`
  import heapq
  def network_delay(times: list[list[int]], n: int, k: int) -> int:
      graph = [[] for _ in range(n+1)]
      for u, v, w in times:
          graph[u].append((v,w))
      dist = [float('inf')] * (n+1)
      dist[k] = 0
      heap = [(0,k)]
      while heap:
          d,u = heapq.heappop(heap)
          if d != dist[u]:
              continue
          for v,w in graph[u]:
              nd = d+w
              if nd < dist[v]:
                  dist[v] = nd
                  heapq.heappush(heap,(nd,v))
      answer = max(dist[1:])
      return -1 if answer == float('inf') else answer
 `,explanation:md`邻接表保存出边，最小堆按照当前最短估计扩展。每次成功松弛加入新条目，旧条目靠距离比较失效。零权边合法；源点自己距离为 0。懒堆至多加入 O(E) 个条目，时间 O((V+E) log(E+1))，空间 O(V+E)。`,hints:[md`最早发现不等于最短距离确定。`,md`用过期检查替代堆内原地更新。`],setup:py`
  import random
  def _ref_delay(edges,n,k):
      d=[float('inf')]*(n+1); d[k]=0
      for _ in range(n-1):
          for u,v,w in edges: d[v]=min(d[v],d[u]+w)
      ans=max(d[1:]); return -1 if ans==float('inf') else ans
 `,tests:[{name:'示例',code:py`got=network_delay([[2,1,1],[2,3,1],[3,4,1]],4,2); assert got==2, f"期望 2，得到 {got}"`},{name:'单点与不可达',code:py`assert network_delay([],1,1)==0; assert network_delay([],2,1)==-1`},{name:'零边和平行边',hidden:true,code:py`got=network_delay([[1,2,9],[1,2,0],[2,3,2],[1,3,8]],3,1); assert got==2, f"期望 2，得到 {got}"`},{name:'随机 Bellman-Ford 对拍',hidden:true,code:py`
  rnd=random.Random(743)
  for _ in range(100):
      n=rnd.randrange(1,10); edges=[[rnd.randrange(1,n+1),rnd.randrange(1,n+1),rnd.randrange(8)] for _ in range(rnd.randrange(35))]; k=rnd.randrange(1,n+1)
      got=network_delay(edges,n,k); exp=_ref_delay(edges,n,k); assert got==exp, f"边 {edges}，期望 {exp}，得到 {got}"
 `}]},
 ],
 interview:[
 {id:'i-uf',freq:3,q:md`按秩合并与按大小合并有什么区别？`,a:md`两者都避免把大树挂到小树。按大小记录集合元素数，合并后可直接求和；按秩记录高度上界，只有秩相同时才增加。路径压缩后实际高度可能降低，秩不会同步下降，因此秩不再是精确高度。二者与路径压缩组合后都有反阿克曼级摊还保证，别把这个结论误说成每次操作严格常数时间。`},
 {id:'i-negative',freq:3,q:md`负边为什么破坏 Dijkstra？`,a:md`确定距离时的论证依赖后续路径只能增加总长度。有负边时，先到更远节点再走负边，可能反而得到更短距离，使之前已确定的节点失效。即使没有负环也可能发生。Bellman-Ford 用反复松弛代替这个贪心假设，最多 V-1 轮覆盖所有简单路径，再额外检查一轮判断可达负环。`},
 {id:'i-stale',freq:2,q:md`堆里出现同一节点多次会影响正确性吗？`,a:md`只要弹出时跳过旧距离就不会。dist 数组是权威状态，堆只是等待处理的候选记录；每次改进加入新记录，旧记录保留但不扩展。这样实现简单，代价是堆空间可能达到 O(E)。如果省略过期检查，结果仍可能正确，但会重复扩展出边，复杂度明显恶化。`},
 ],cheatsheet:md`UF：find 路径压缩；union 先找根、小挂大、成功才 count--。Dijkstra：非负边，堆弹最小，d!=dist[u] 跳过，nd<dist[v] 才更新。BFS 用于无权；Bellman-Ford 处理负边；Floyd 求小图全源。文档相似边的连通闭包不保证组内两两相似。`,
})
