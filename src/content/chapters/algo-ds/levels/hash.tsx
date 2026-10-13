import { defineLevel, md, py } from '@/content/helpers'
import { COUNTED_NUM } from '../shared'
import { PrefixSumViz } from '../viz/PrefixSumViz'
export default defineLevel({
 learn:[
  {id:'prefix',title:'从区间和变成两个前缀的差',viz:PrefixSumViz,body:md`定义 P[0]=0，P[j] 是前 j 个数之和，则半开区间 [i,j) 的和为 P[j]-P[i]。要求区间和等于 k，就等价于查找历史前缀 P[j]-k。用哈希表记录每个前缀出现次数，每处理一个新前缀即可累计贡献，平均时间 O(n)。

  初始化 count[0]=1 代表空前缀，它让从数组起点开始的区间被自然统计。必须先查历史，再登记当前前缀；否则 k=0 时会把 [j,j) 这个空区间算进去。前缀可能为负，也可能重复，所以不能使用只对正数有效的滑动窗口，不应只记录“是否见过”。`},
  {id:'hashing',title:'哈希存什么，决定能回答什么',body:md`两数之和存数值到下标的映射；字母异位词分组存字符计数或排序后字符串；最长连续序列存去重集合。连续序列只从没有前驱 x-1 的 x 出发向右扩展，每个数至多被扩展访问一次；如果从每个数都向右走，有序长链会退化为平方复杂度。

  哈希查找的常数时间是平均意义上的，冲突与恶意输入可能影响表现。还要区分“次数”和“下标”：统计所有和为 k 的子数组需要次数，求最长长度时应保存某前缀最早出现的位置。相同数学变换，问题目标改变后表里的信息也随之改变。`},
  {id:'packing',title:'差分与 cu_seqlens：边界比元素更重要',body:md`区间加操作可以在差分数组的左端加 v、右端后一位减 v，最后做前缀和还原。长度数组 [3,0,2] 的累积边界为 [0,3,3,5]；样本 i 占据半开区间 [cu[i],cu[i+1])，零长度样本形成重复边界。

  给定合法全局 token 下标 t，样本下标是 upper_bound(cu,t)-1，而不是 lower_bound：在 t=3 时应跳过空样本，落到样本 2，局部偏移为 0。边界最后一项是总 token 数，不是合法 token 下标。本关把这种定位写成手撕题，连接变长序列存储与二分边界语义。`},
 ],
 quiz:[
  {id:'q1',type:'single',difficulty:1,prompt:md`统计和为 k 的子数组，哈希表应该保存什么？`,options:['前缀和出现次数','每个前缀是否出现过','原数组数值的最大位置'],answer:0,explanation:md`不同起点可能有相同前缀，每次出现都贡献一个区间；布尔标记会漏计，原值位置无法表示任意区间和。`},
  {id:'q2',type:'numeric',difficulty:2,prompt:md`nums=[0,0,0]，和为 0 的非空子数组有多少个？`,answer:6,tolerance:0,explanation:md`长度 1、2、3 分别有 3、2、1 个，总计 6；不是只有 3 个，因为重复前缀要按次数累计。`},
  {id:'q3',type:'judge',difficulty:1,prompt:md`存在负数时，可以直接用“和大就缩左端，和小就扩右端”的窗口统计目标和。`,answer:false,explanation:md`负数破坏窗口和的单调性。移除负数会让和变大，扩张加入负数会让和变小，指针策略不再有正确依据。`},
  {id:'q4',type:'fill',difficulty:2,prompt:md`前缀和 s 与目标 k 已知，应查询历史前缀 ____。`,blanks:[{answers:['s-k','s - k']}],explanation:md`由 s-old=k 得 old=s-k；查 s+k 会把差的方向颠倒。`},
  {id:'q5',type:'multi',difficulty:2,prompt:md`最长连续序列的线性写法需要哪些步骤？`,options:['先用集合去重','只从 x-1 不在集合的 x 开始扩展','从每个元素都扩展到结尾','沿 x+1 连续前进统计长度'],answer:[0,1,3],explanation:md`只扩展链头使每条链遍历一次。每个元素都出发会在连续数组上做平方级访问。`},
  {id:'q6',type:'order',difficulty:2,prompt:md`前缀计数扫描的正确顺序是：`,items:['累加当前元素得到 s','读取 count[s-k] 并累计答案','把 count[s] 加一'],explanation:md`先登记会在 k=0 时匹配当前前缀自身，错误统计空子数组。`},
  {id:'q7',type:'numeric',difficulty:3,prompt:md`样本长度 [0,3,0,2]，从 0 开始编号；全局 token 下标 3 属于哪个样本？`,answer:3,tolerance:0,explanation:md`边界是 [0,0,3,3,5]；第一个大于 3 的边界下标为 4，因此样本为 3，局部偏移为 0。`},
  {id:'q8',type:'single',difficulty:3,prompt:md`求和为 k 的最长子数组，某个前缀再次出现时应怎样维护最早位置？`,options:['保留第一次位置','用当前位置覆盖','将所有位置相加'],answer:0,explanation:md`当前结束点固定，起点越早区间越长。覆盖最早位置会缩短后续候选；次数适合计数，不足以计算最长长度。`},
 ],
 guided:[{id:'g-count',title:'先查后写的前缀计数',intro:md`实现 count_sum(nums,k)，允许负数和零。`,template:py`
  def count_sum(nums, k):
      counts = {{1}}
      prefix = answer = 0
      for x in nums:
          prefix += x
          answer += counts.get({{2}}, 0)
          counts[prefix] = {{3}}
      return answer
 `,steps:[{blank:1,instruction:md`把空前缀登记一次。`,answer:'{0: 1}'},{blank:2,instruction:md`求出需要的历史前缀。`,answer:'prefix - k'},{blank:3,instruction:md`查询结束后，把当前前缀次数加一。`,answer:'counts.get(prefix, 0) + 1'}],tests:[{name:'重复前缀',code:py`got=count_sum([1,-1,0],0); assert got==3, f"期望 3，得到 {got}"`},{name:'空与起点',code:py`assert count_sum([],0)==0; assert count_sum([3],3)==1; assert count_sum([0,0],0)==3`}]}],
 challenges:[
 {id:'c-sum',title:'和为 K 的子数组',difficulty:'medium',source:'LC 560',description:md`实现 subarray_sum(nums: list[int], k: int) -> int，返回和恰好等于 k 的非空连续子数组数量。允许负数，空数组返回 0。n≤100000，要求平均 O(n) 时间、O(n) 空间。例：[1,1,1],2 → 2。`,starter:py`
  def subarray_sum(nums: list[int], k: int) -> int:
      raise NotImplementedError
 `,solution:py`
  def subarray_sum(nums: list[int], k: int) -> int:
      counts = {0: 1}
      prefix = answer = 0
      for x in nums:
          prefix += x
          answer += counts.get(prefix-k, 0)
          counts[prefix] = counts.get(prefix, 0) + 1
      return answer
 `,explanation:md`用 P[j]-P[i]=k 将区间问题转成历史前缀查找。计数表记录所有合法起点，而不是只记录一个下标。初始化空前缀、先查后写同时处理起点区间与 k=0；时间平均 O(n)，空间 O(n)。`,hints:[md`两个前缀的差表示连续区间和。`,md`每个历史前缀出现多少次，就对应多少个起点。`],setup:COUNTED_NUM + "\n\n" + py`import random`,tests:[{name:'示例',code:py`got=subarray_sum([1,1,1],2); assert got==2, f"期望 2，得到 {got}"`},{name:'边界',code:py`assert subarray_sum([],0)==0; assert subarray_sum([0]*4,0)==10; assert subarray_sum([-1,-1,1],-1)==3`},{name:'随机暴力对拍',hidden:true,code:py`
  rnd=random.Random(560)
  for _ in range(150):
      a=[rnd.randrange(-3,4) for _ in range(rnd.randrange(18))]; k=rnd.randrange(-6,7)
      exp=sum(sum(a[i:j])==k for i in range(len(a)) for j in range(i+1,len(a)+1)); got=subarray_sum(a,k)
      assert got==exp, f"输入 {a}, k={k}：期望 {exp}，得到 {got}"
 `},{name:'计数不能溢出或去重',hidden:true,code:py`n=1500; got=subarray_sum([0]*n,0); assert got==n*(n+1)//2, f"期望 {n*(n+1)//2}，得到 {got}"`},{name:'线性访问预算',hidden:true,code:py`
  n=3000; _Num.n=0; _Num.budget=12*n
  got=subarray_sum([_Num(0) for _ in range(n)],0); exp=n*(n+1)//2
  assert got==exp, f"期望 {exp}，得到 {got}"
 `}]},
 {id:'c-longest',title:'最长连续序列',difficulty:'medium',source:'LC 128',description:md`实现 longest_consecutive(nums: list[int]) -> int。忽略顺序与重复，求最长连续整数序列长度。n≤100000，平均 O(n) 时间。例：[100,4,200,1,3,2,2] → 4；空数组 → 0。`,starter:py`
  def longest_consecutive(nums: list[int]) -> int:
      raise NotImplementedError
 `,solution:py`
  def longest_consecutive(nums: list[int]) -> int:
      values = set(nums)
      best = 0
      for x in values:
          if x-1 in values:
              continue
          end = x
          while end in values:
              end += 1
          best = max(best, end-x)
      return best
 `,explanation:md`先去重，只有没有前驱的元素才是链头。各条链互不相交，因此 while 的总访问次数为 O(n)，而不是 O(n²)。排序也能做，但时间 O(n log n)，不满足题目要求。`,hints:[md`怎样避免从链中间重复出发？`,md`前驱不在集合中才开始计数。`],setup:COUNTED_NUM + "\n\n" + py`import random`,tests:[{name:'示例',code:py`got=longest_consecutive([100,4,200,1,3,2,2]); assert got==4, f"期望 4，得到 {got}"`},{name:'边界',code:py`assert longest_consecutive([])==0; assert longest_consecutive([7,7])==1; assert longest_consecutive([-2,-1,0])==3`},{name:'随机对拍',hidden:true,code:py`
  rnd=random.Random(128)
  for _ in range(100):
      a=[rnd.randrange(-20,21) for _ in range(rnd.randrange(60))]; ordered=sorted(set(a)); best=run=0; prev=None
      for x in ordered:
          run=run+1 if prev is not None and x==prev+1 else 1; best=max(best,run); prev=x
      got=longest_consecutive(a); assert got==best, f"输入 {a}：期望 {best}，得到 {got}"
 `},{name:'两条长链',hidden:true,code:py`got=longest_consecutive(list(range(500))+list(range(1000,1800))); assert got==800, f"期望 800，得到 {got}"`}]},
 {id:'c-tokens',title:'变长序列 token 定位',difficulty:'medium',source:'大模型数据工程',description:md`实现 locate_tokens(lengths: list[int], tokens: list[int]) -> tuple[list[int], list[tuple[int,int]]]。lengths 为非负长度，tokens 中每个下标满足 0≤t<总长度；返回 cu_seqlens 和每个 token 对应的 (样本下标,样本内偏移)。零长度样本合法，输入均可为空。预处理 O(n)，每次查询 O(log n)。例：[3,0,2],[0,3,4] → ([0,3,3,5],[(0,0),(2,0),(2,1)])。`,starter:py`
  def locate_tokens(lengths: list[int], tokens: list[int]):
      raise NotImplementedError
 `,solution:py`
  from bisect import bisect_right
  def locate_tokens(lengths: list[int], tokens: list[int]):
      cu = [0]
      for length in lengths:
          cu.append(cu[-1] + length)
      answer = []
      for t in tokens:
          i = bisect_right(cu, t) - 1
          answer.append((i, t-cu[i]))
      return cu, answer
 `,explanation:md`每个样本占据半开区间。upper_bound 找到第一个严格大于 t 的右边界，减一即对应样本；这会自动跳过所有重复边界形成的空样本。总时间 O(n+m log n)，空间 O(n+m)。`,hints:[md`先构造包含初始 0 的累积长度。`,md`边界可能重复，使用 bisect_right 而非 bisect_left。`],setup:COUNTED_NUM + "\n\n" + py`import random`,tests:[{name:'示例',code:py`got=locate_tokens([3,0,2],[0,3,4]); exp=([0,3,3,5],[(0,0),(2,0),(2,1)]); assert got==exp, f"期望 {exp}，得到 {got}"`},{name:'空与全空',code:py`assert locate_tokens([],[])==([0],[]); assert locate_tokens([0,0],[])==([0,0,0],[])`},{name:'首尾空样本',hidden:true,code:py`got=locate_tokens([0,2,0],[0,1]); exp=([0,0,2,2],[(1,0),(1,1)]); assert got==exp, f"期望 {exp}，得到 {got}"`},{name:'随机映射',hidden:true,code:py`
  rnd=random.Random(7)
  for _ in range(100):
      lens=[rnd.randrange(5) for _ in range(rnd.randrange(20))]; exp=[(i,j) for i,l in enumerate(lens) for j in range(l)]
      cu,got=locate_tokens(lens,list(range(sum(lens))))
      assert got==exp and cu==[sum(lens[:i]) for i in range(len(lens)+1)], f"长度 {lens}：期望映射 {exp}，得到 {got}"
 `}]},
 ],
 interview:[
 {id:'i-prefix',freq:3,q:md`为什么 count[0]=1？`,a:md`它代表扫描开始前的空前缀。如果当前前缀恰好等于 k，那么需要匹配前缀 0，对应从第一个元素开始的合法区间。统一登记空前缀比给这类区间另写分支更可靠。初始化的是次数 1，而不是位置 0；最长区间版本通常改为前缀 0 的位置 -1，二者语义必须区分。`},
 {id:'i-diff',freq:2,q:md`前缀和与差分有什么关系？`,a:md`两者是离散的累积与相邻差分，互为逆过程。前缀和将静态区间求和变成两个边界相减，差分把区间统一加值变成两个端点更新。如果更新和查询交错发生，单次全量还原就不够高效，通常需要树状数组或线段树。先明确离线批量处理还是在线交错操作，才能选对结构。`},
 {id:'i-cu',freq:2,q:md`cu_seqlens 中为什么保留零长度序列的重复边界？`,a:md`样本编号仍有意义，删除边界会改变后续样本的索引。半开区间自然表示空样本：左右端相同，没有 token 属于它。使用 upper_bound 定位可以跳过相等边界，将 token 映射到真正包含它的样本。工程中也可预先过滤空样本，但必须同步保留原始索引映射，不能默默改变数据语义。`},
 ],cheatsheet:md`区间 [i,j) 的和 = P[j]-P[i]。计数：count={0:1}，先 answer+=count[s-k]，再 count[s]+=1。最长区间：保存最早前缀位置。连续序列：只从无前驱链头扩展。差分：[l,r] 加 v → d[l]+=v,d[r+1]-=v。token 定位：bisect_right(cu,t)-1。`,
})
