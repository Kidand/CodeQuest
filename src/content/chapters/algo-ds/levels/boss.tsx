import { defineLevel, md, py } from '@/content/helpers'
import { PrefixSumViz } from '../viz/PrefixSumViz'
export default defineLevel({
 learn:[{id:'review',title:'从操作约束反推数据结构',viz:PrefixSumViz,body:md`本章结构的共同点是为某类操作维护恰好足够的信息：链表维护邻接关系，哈希表定位对象，LRU 维护最近访问顺序，单调结构排除不可能再胜出的候选，前缀计数把区间查询转成历史状态查找。

 BOSS 把两个时间维度叠加：LRU 的“最近使用”与 TTL 的“何时失效”。读取会改变 LRU 顺序，但不一定延长 TTL；更新一个键会产生新的过期记录，因此旧记录不能误删新值。先写出操作语义和失效规则，再用数据结构分别维护这些不变量。`}],
 quiz:[
 {id:'q1',type:'single',difficulty:1,prompt:md`LRU 的 get 成功后应该做什么？`,options:['将命中项移动到最近使用端','不改变顺序','删除最少访问次数项'],answer:0,explanation:md`LRU 按最近使用时间排序；次数属于 LFU 语义。get 也是使用，不能忽略。`},
 {id:'q2',type:'judge',difficulty:1,prompt:md`双向链表删除已知节点时，不需要从头寻找其前驱。`,answer:true,explanation:md`节点已有 prev、next，改两个邻接指针即可 O(1) 删除；单向链表通常需要额外前驱信息。`},
 {id:'q3',type:'numeric',difficulty:2,prompt:md`数组 [1,-1,1,-1] 中和为 0 的非空连续区间有多少个？`,answer:4,tolerance:0,explanation:md`前缀为 [0,1,0,1,0]，相同前缀选两次：3 选 2 加 2 选 2 等于 4。`},
 {id:'q4',type:'fill',difficulty:2,prompt:md`窗口长度 k，扫描到下标 i 时，队首下标 j 满足 j ____ i-k 就已过期。`,blanks:[{answers:['<=']}],explanation:md`当前窗口左端为 i-k+1，因此 j≤i-k 不在窗口内。`},
 {id:'q5',type:'multi',difficulty:2,prompt:md`手写 LRU 应维护哪些不变量？`,options:['哈希表和链表包含相同活跃键','头尾哨兵不作为实际缓存项','每次 get 都扫描整条链表','容量不足时删除最久未使用项'],answer:[0,1,3],explanation:md`哈希定位实现 O(1)，扫描破坏复杂度。哨兵简化边界但不计入容量。`},
 {id:'q6',type:'match',difficulty:2,prompt:md`任务与结构配对：`,pairs:[['窗口最大值','单调双端队列'],['目标区间和计数','前缀和频次哈希'],['最近使用淘汰','哈希加双向链表']],explanation:md`分别需要候选支配关系、历史前缀次数和可 O(1) 移动的使用顺序；三个结构的查询目标不同。`},
 {id:'q7',type:'numeric',difficulty:3,prompt:md`TTL=5，键 a 在时刻 2 写入、时刻 4 重写，旧过期记录在时刻 7 弹出。新值的失效时刻是多少？`,answer:9,tolerance:0,explanation:md`重写使用新的写入时间，4+5=9。旧记录必须按版本识别，不能在 7 删除新值。`},
 {id:'q8',type:'judge',difficulty:3,prompt:md`用最小堆保存 TTL 过期记录时，键被 LRU 淘汰后必须立即在线性时间删除堆中对应记录。`,answer:false,explanation:md`可以懒删除：到期弹出时检查键是否仍存在、版本是否匹配。为旧条目付出的清理成本可按写入次数摊还。`},
 {id:'q9',type:'order',difficulty:2,prompt:md`带 TTL 的缓存处理一次 get 的顺序：`,items:['清理所有到期且版本有效的记录','查询键是否仍在活跃缓存','命中则更新 LRU 顺序并返回值'],explanation:md`先查再清理可能把过期值错误返回。过期清理和使用顺序维护是两个独立步骤。`},
 {id:'q10',type:'single',difficulty:1,prompt:md`链表反转 curr.next 前为什么先保存 next？`,options:['避免丢失尚未处理的后继链','减少节点数量','保证原链仍然有序'],answer:0,explanation:md`改向会覆盖唯一的后继引用，必须先保存。反转并不依赖数值有序，也不应删除节点。`},
 ],guided:[],
 challenges:[{id:'c-ttl',title:'BOSS：带过期时间的 LRU',difficulty:'hard',timeLimitMin:40,description:md`实现 ttl_lru(capacity: int, ttl: int, operations: list[tuple]) -> list[int]。操作为 ('put',time,key,value) 或 ('get',time,key)，时间非递减，键和值为整数；只返回每次 get 的结果，未命中为 -1。

 put 写入或更新时令过期时刻为 time+ttl，并更新为最近使用；get 命中只改变 LRU，不续期。每次操作前，失效时刻≤当前 time 的键应清除。写入超容量时淘汰最久未使用项。capacity≥0，ttl≥1，最多 10000 个操作。允许 OrderedDict；要求总时间 O(m log m)、空间 O(m)，不能每次扫描全部活跃键。`,starter:py`
  def ttl_lru(capacity: int, ttl: int, operations: list[tuple]) -> list[int]:
      raise NotImplementedError
 `,solution:py`
  import heapq
  from collections import OrderedDict
  def ttl_lru(capacity: int, ttl: int, operations: list[tuple]) -> list[int]:
      active = OrderedDict()
      expirations, answer = [], []
      version = 0
      for op in operations:
          kind, now, key = op[:3]
          while expirations and expirations[0][0] <= now:
              expiry, old_version, old_key = heapq.heappop(expirations)
              if old_key in active and active[old_key][1] == old_version:
                  del active[old_key]
          if kind == 'get':
              if key not in active:
                  answer.append(-1)
              else:
                  answer.append(active[key][0])
                  active.move_to_end(key)
          elif capacity > 0:
              version += 1
              active[key] = (op[3], version)
              active.move_to_end(key)
              heapq.heappush(expirations, (now+ttl,version,key))
              if len(active) > capacity:
                  active.popitem(last=False)
      return answer
 `,explanation:md`OrderedDict 维护 LRU 顺序，最小堆维护失效顺序。每次写入递增版本号，堆中的旧记录仅在活跃键版本相同时才能删除该键；这同时处理重写、淘汰后重建两种情况。get 不修改版本和失效时刻。每条过期记录最多入堆出堆一次，因此总 O(m log m)，最坏堆含 O(m) 历史记录。`,hints:[md`过期顺序与使用顺序不同，需要两套索引。`,md`同一个键被多次写入时，给每次写入唯一版本。`,md`堆中旧条目不用立即删除，弹出时判断是否仍有效。`],setup:py`
  import random
  def _ref_ttl(capacity,ttl,operations):
      active={}; tick=0; out=[]
      for op in operations:
          kind,t,key=op[:3]; tick+=1
          active={k:v for k,v in active.items() if v[1]>t}
          if kind=='get':
              if key not in active: out.append(-1)
              else:
                  val,expiry,_=active[key]; out.append(val); active[key]=(val,expiry,tick)
          elif capacity:
              active[key]=(op[3],t+ttl,tick)
              if len(active)>capacity: del active[min(active,key=lambda k:active[k][2])]
      return out
 `,tests:[{name:'LRU 与不续期',code:py`
  ops=[('put',0,1,10),('put',1,2,20),('get',2,1),('put',3,3,30),('get',3,2),('get',5,1)]
  got=ttl_lru(2,5,ops); exp=[10,-1,-1]; assert got==exp, f"期望 {exp}，得到 {got}"
 `},{name:'旧记录不能删除新值',code:py`
  ops=[('put',2,1,10),('put',4,1,11),('get',7,1),('get',9,1)]
  got=ttl_lru(1,5,ops); assert got==[11,-1], f"期望 [11,-1]，得到 {got}"
 `},{name:'零容量与相同时刻',hidden:true,code:py`
  ops=[('put',0,1,7),('get',0,1),('put',0,2,8),('get',0,1)]
  assert ttl_lru(0,1,ops)==[-1,-1]; assert ttl_lru(1,1,ops)==[7,-1]
 `},{name:'随机独立模型对拍',hidden:true,code:py`
  rnd=random.Random(460)
  for _ in range(90):
      cap=rnd.randrange(6); ttl=rnd.randrange(1,8); ops=[]; t=0
      for _ in range(90):
          t+=rnd.randrange(3); k=rnd.randrange(8)
          ops.append(('put',t,k,rnd.randrange(100)) if rnd.random()<.6 else ('get',t,k))
      exp=_ref_ttl(cap,ttl,ops); got=ttl_lru(cap,ttl,ops); assert got==exp, f"容量 {cap}, TTL {ttl}：期望 {exp}，得到 {got}"
 `},{name:'淘汰后同键重新插入',hidden:true,code:py`
  ops=[('put',0,1,1),('put',1,2,2),('put',2,1,3),('get',5,1),('get',7,1)]
  got=ttl_lru(1,5,ops); assert got==[3,-1], f"期望 [3,-1]，得到 {got}"
 `}]}],interview:[],cheatsheet:md`结构设计先列操作与不变量：哈希定位，链表或 OrderedDict 维护使用顺序，堆按失效时间排序。旧记录用版本号懒删除。链表先存后改；单调结构一生进出一次；前缀计数先查后写。TTL 到期条件是 expiry≤now；get 是否续期必须明确。`,
})
