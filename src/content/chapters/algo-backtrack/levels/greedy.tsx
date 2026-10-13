import { defineLevel, md, py } from '@/content/helpers'
import { PackingViz } from '../viz/ConstraintViz'
export default defineLevel({
 learn:[
  {id:'proof',title:'贪心的核心是证明当前选择可被保留',body:md`贪心每步只保留一个选择，省去回溯时的分支，因此必须证明这个选择不会错过最优解。区间调度选择结束最早的区间，可以用交换论证：把任意最优解的第一个区间换成结束更早者，后续区间仍可放下，解的数量不减少，然后递归处理剩余问题。

  合并区间与区间调度目标不同。合并区间按起点排序并维护当前并集，只要下一段起点不大于当前终点就延长右端；选择最多互不重叠区间则按终点排序。排序键必须服务于证明，不能只凭“区间题通常排序”记住一个固定写法。`},
  {id:'jump',title:'跳跃游戏：维护可达边界，而非枚举路径',body:md`跳跃游戏 I 从左到右扫描，维护已经知道能到达的最远位置 far。只有 i≤far 的位置才能扩展 far=max(far,i+nums[i])；若 i>far，说明出现无法跨越的断层。局部更新保留了所有已知路径的可达范围，因此无需保存具体路径。

  最少跳跃次数可把每次跳跃看作 BFS 的一层：当前层边界 end 以内的所有点共同计算下一层最远位置 next_end，扫描到 end 才增加一步并切换边界。若 next_end 没有推进则不可达。不要每看到更远候选就增加次数，它们可能属于同一跳能到达的层。糖果分配需要左右两次约束，划分字母区间则用字符最后位置决定闭合边界。`},
  {id:'packing',title:'序列打包：FFD 是启发式，合法性先于箱数',viz:PackingViz,body:md`SFT 数据打包可把序列长度看成物品体积、最大 token 长度看成箱容量。First-Fit Decreasing 先按长度降序，再把每个物品放入第一个剩余容量足够的箱；若没有则开新箱。顺序必须确定，长度相等时按原下标排序，输出原下标才能保留样本身份。

  总体积除以容量向上取整只是箱数下界，未必能达到。FFD 不保证最优，不能把它的结果当最优标签。本关用确定性 FFD 作为参考箱数上限，同时校验每个样本出现恰好一次、每箱不超容量；更好的合法打包也应通过。实际训练还需样本边界、attention mask 与位置处理，单纯拼接 token 并不等价于相互隔离的样本训练。`},
 ],
 quiz:[
  {id:'q1',type:'single',difficulty:1,prompt:md`选择最多互不重叠区间，经典贪心的排序依据是什么？`,options:['结束时间升序','开始时间升序','长度降序'],answer:0,explanation:md`结束越早，为后续留下的空间越多，能用交换论证证明；先开始或选最长都可能占住多个短区间。`},
  {id:'q2',type:'judge',difficulty:1,prompt:md`合并区间与最多不重叠区间是同一个优化目标，应该使用完全相同的贪心。`,answer:false,explanation:md`合并是求区间并集，调度是最大化选择数量；排序与状态含义不同。`},
  {id:'q3',type:'numeric',difficulty:2,prompt:md`nums=[2,3,1,1,4]，从下标 0 到最后一格最少跳几次？`,answer:2,tolerance:0,explanation:md`0→1→4，两跳；第一层同时考虑可达的 1、2，下一层边界可扩展至 4。`},
  {id:'q4',type:'fill',difficulty:2,prompt:md`合并闭区间时，next_start ____ current_end 表示两个区间相交或接触，应合并。填写比较符。`,blanks:[{answers:['<=']}],explanation:md`闭区间包含端点，相等也有交集。半开区间的相邻是否合并需要另行约定。`},
  {id:'q5',type:'multi',difficulty:2,prompt:md`校验序列打包结果必须检查哪些性质？`,options:['每个原始样本下标恰好出现一次','每箱长度和不超过容量','所有箱子的长度和彼此相等','原样本下标都有效'],answer:[0,1,3],explanation:md`箱子无需装满或等长；重复、漏样本、越界下标和超容量才是合法性错误。`},
  {id:'q6',type:'order',difficulty:2,prompt:md`First-Fit Decreasing 的流程是：`,items:['按长度降序排列样本下标','对当前样本从第一个箱开始找空间','若找到可容纳箱则放入，否则新建箱','处理完所有样本后返回箱内原下标'],explanation:md`Decreasing 指先降序，First-Fit 指第一个可容纳箱，而不是最紧凑箱或最后一个箱。`},
  {id:'q7',type:'numeric',difficulty:3,prompt:md`容量 10，长度 [6,6,6] 的体积下界为 2，但最少需要几个箱？`,answer:3,tolerance:0,explanation:md`任意两个 6 都无法同箱，因此需要 3。体积下界忽略了物品不可拆分性。`},
  {id:'q8',type:'single',difficulty:3,prompt:md`“这个局部选择在几个样例中都很好”足以证明贪心正确吗？`,options:['不足，需要交换论证、不变量或其它完整证明','足够，只要随机测试通过','足够，排序后就是贪心正确'],answer:0,explanation:md`测试能发现反例，不能覆盖所有输入。尤其装箱属于困难的组合优化，启发式表现好不等于全局最优。`},
 ],
 guided:[{id:'g-jump',title:'按 BFS 层边界统计跳跃次数',intro:md`实现 min_jumps(nums)，输入非负整数数组；长度 0 或 1 返回 0，不可达返回 -1。`,template:py`
  def min_jumps(nums):
      jumps = end = far = 0
      for i in range(len(nums)-1):
          if i > far:
              return -1
          far = max(far, {{1}})
          if i == end:
              if far == end:
                  return -1
              jumps += {{2}}
              end = {{3}}
      return jumps
 `,steps:[{blank:1,instruction:md`当前位置能扩展到的最远下标。`,answer:'i + nums[i]'},{blank:2,instruction:md`一整层扫描完成才增加一次跳跃。`,answer:'1'},{blank:3,instruction:md`下一层的边界是本层能到达的最远点。`,answer:'far'}],tests:[{name:'可达',code:py`assert min_jumps([2,3,1,1,4])==2; assert min_jumps([1,1,1])==2`},{name:'不可达与边界',code:py`assert min_jumps([3,2,1,0,4])==-1; assert min_jumps([0])==0; assert min_jumps([])==0`}]}],
 challenges:[
 {id:'c-merge',title:'合并闭区间',difficulty:'medium',source:'LC 56',description:md`实现 merge_intervals(intervals: list[list[int]]) -> list[list[int]]。每段为 [start,end]，start≤end；按起点升序返回合并后的闭区间，相接端点也合并。输入不应被修改。n≤100000，O(n log n) 时间。例 [[1,3],[2,6],[8,10]] → [[1,6],[8,10]]。`,starter:py`
  def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
      raise NotImplementedError
 `,solution:py`
  def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
      out = []
      for start,end in sorted(intervals):
          if not out or start > out[-1][1]:
              out.append([start,end])
          else:
              out[-1][1] = max(out[-1][1],end)
      return out
 `,explanation:md`按起点排序后，新区间只能影响当前最后一个合并区间。相交时只需延长终点；不相交时此前区间以后不可能再被连接，故新开一段。重新建立输出小列表，避免修改输入中嵌套列表。排序 O(n log n)，扫描 O(n)，空间 O(n)。`,hints:[md`排序后只需关注最后一段输出。`,md`闭区间起点等于当前终点时仍要合并。`],setup:py`
  import random
  def _ref_merge(intervals):
      # 独立地反复寻找任意相交区间合并，用于小规模对拍。
      work=[x[:] for x in intervals]
      while True:
          changed=False
          for i in range(len(work)):
              for j in range(i+1,len(work)):
                  if max(work[i][0],work[j][0])<=min(work[i][1],work[j][1]):
                      work[i]=[min(work[i][0],work[j][0]),max(work[i][1],work[j][1])]; work.pop(j); changed=True; break
              if changed: break
          if not changed: return sorted(work)
 `,tests:[{name:'示例',code:py`got=merge_intervals([[1,3],[2,6],[8,10]]); assert got==[[1,6],[8,10]], f"得到 {got}"`},{name:'边界与相接',code:py`assert merge_intervals([])==[]; assert merge_intervals([[1,2],[2,2]])==[[1,2]]; assert merge_intervals([[1,10],[2,3]])==[[1,10]]`},{name:'随机对拍与输入保护',hidden:true,code:py`
  rnd=random.Random(56)
  for _ in range(120):
      a=[sorted([rnd.randrange(-8,9),rnd.randrange(-8,9)]) for _ in range(rnd.randrange(15))]; before=[x[:] for x in a]
      exp=_ref_merge(a); got=merge_intervals(a); assert got==exp and a==before, f"期望 {exp} 且不改输入，得到 {got}"
 `},{name:'互不相交',hidden:true,code:py`a=[[3*i,3*i+1] for i in range(500)]; got=merge_intervals(a[::-1]); assert got==a, f"期望 {len(a)} 个有序区间，得到 {len(got)}"`}]},
 {id:'c-pack',title:'SFT 序列打包',difficulty:'hard',source:'大模型数据工程',description:md`实现 pack_sequences(lengths: list[int], capacity: int) -> list[list[int]]，输出每箱包含的原始样本下标。容量为正，所有长度满足 1≤length≤capacity；n≤1000。每个样本必须出现恰好一次，每箱总长度不超容量，箱数不多于确定性 First-Fit Decreasing 的结果。空输入返回 []，不允许空箱。长度相等按原下标排序。参考实现 O(n²)，允许更优合法结果。`,starter:py`
  def pack_sequences(lengths: list[int], capacity: int) -> list[list[int]]:
      raise NotImplementedError
 `,solution:py`
  def pack_sequences(lengths: list[int], capacity: int) -> list[list[int]]:
      bins, used = [], []
      for i in sorted(range(len(lengths)),key=lambda i:(-lengths[i],i)):
          for b in range(len(bins)):
              if used[b]+lengths[i] <= capacity:
                  bins[b].append(i)
                  used[b] += lengths[i]
                  break
          else:
              bins.append([i])
              used.append(lengths[i])
      return bins
 `,explanation:md`先处理大物品，减少后续无法放入剩余碎片的风险，再按箱顺序寻找第一个可容纳位置。这是启发式而非精确最优算法。测试对答案做性质校验并与 FFD 箱数比较，因此不会拒绝不同排列或更好的合法结果。排序 O(n log n)，朴素放置最坏 O(n²)，额外空间 O(n)。`,hints:[md`输出下标而不是长度，重复长度也必须保留独立样本身份。`,md`只在全部已有箱都放不下时新增箱。`],setup:py`
  import random
  def _ref_pack_limit(lengths,capacity):
      remaining=[]
      for x in sorted(lengths,reverse=True):
          for i,left in enumerate(remaining):
              if x<=left: remaining[i]-=x; break
          else: remaining.append(capacity-x)
      return len(remaining)
  def _check_pack(lengths,capacity):
      before=lengths[:]; out=pack_sequences(lengths,capacity)
      assert isinstance(out,list) and all(isinstance(b,list) and b for b in out), f"应返回非空箱组成的列表，得到 {out}"
      flat=[i for b in out for i in b]
      assert all(type(i) is int for i in flat) and sorted(flat)==list(range(len(lengths))), f"样本必须恰好出现一次，得到 {flat}"
      for b in out:
          used=sum(lengths[i] for i in b); assert used<=capacity, f"箱 {b} 用量 {used} 超过 {capacity}"
      limit=_ref_pack_limit(lengths,capacity)
      assert len(out)<=limit, f"箱数上限 {limit}，得到 {len(out)}"
      assert lengths==before, '不得修改输入长度'
 `,tests:[{name:'示例',code:py`_check_pack([8,7,6,5,4,3,2],12)`},{name:'空与刚好装满',code:py`_check_pack([],10); _check_pack([10,10,5,5],10)`},{name:'体积下界不可达',hidden:true,code:py`_check_pack([6,6,6],10); _check_pack([1]*35,8)`},{name:'随机性质校验',hidden:true,code:py`
  rnd=random.Random(2026)
  for _ in range(120):
      cap=rnd.randrange(1,31); lengths=[rnd.randrange(1,cap+1) for _ in range(rnd.randrange(60))]; _check_pack(lengths,cap)
 `}]},
 ],
 interview:[
 {id:'i-exchange',freq:3,q:md`交换论证应该怎样讲完整？`,a:md`先取任意一个最优解，指出它在哪一步与贪心选择不同，再把该选择替换成贪心选择，证明替换后仍合法且目标值不变差。这样至少存在一个最优解包含当前贪心选择；删去这个选择形成同类子问题，继续归纳。不能只说“当前看起来最好”，必须证明不会妨碍未来。`},
 {id:'i-ffd',freq:3,q:md`FFD 为什么不是“排序后肯定最优”？`,a:md`排序只是启发式，局部把大物品放入第一个可用箱不考虑未来所有组合。装箱的不可分性使体积下界不总可达，也可能让局部剩余空间分布阻碍更好的组合。工程上通常接受较低成本的近似结果，但指标必须分别报告合法性、箱数和利用率，不能把参考启发式当全局最优证明。`},
 {id:'i-isolation',freq:2,q:md`把不同 SFT 样本拼在一起训练，需要注意什么？`,a:md`首先保存每个样本边界，再按训练目标决定是否允许跨样本注意力、怎样计算位置以及哪些 token 参与 loss。如果要求样本相互隔离，普通因果 mask 仍会让后一个样本看到前一个，需要分段注意力语义或对应内核支持。打包算法只解决空间利用率，不能替代这些训练语义约定。`},
 ],cheatsheet:md`贪心先证再写：交换论证、保持最优解或可达边界。合并区间按起点，调度最多区间按终点。跳跃次数按 BFS 层末计数。FFD：降序 + 第一个可容纳箱；不保证最优。打包检查身份完整、容量合法、无空箱，并处理样本边界。`,
})
