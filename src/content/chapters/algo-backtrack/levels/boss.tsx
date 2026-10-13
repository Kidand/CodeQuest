import { defineLevel, md, py } from '@/content/helpers'
import { PackingViz } from '../viz/ConstraintViz'
export default defineLevel({
 learn:[{id:'review',title:'用贪心给上界，用搜索证明最优',viz:PackingViz,body:md`贪心与回溯可以合作。装箱先用 FFD 得到一个合法解，它提供最优箱数的上界；体积下界提供“不可能更少”的证据。然后按降序放物品做精确搜索，只探索可能改善上界的分支。若上下界相遇就可以停止，因为已经证明最优。

 对称性也是剪枝来源：当前物品放进两个负载相同的箱，得到的未来容量配置相同，搜索一个代表即可。空箱之间同样对称，只开一个新箱分支。注意体积下界不考虑碎片，不能用“下界可达到”替代搜索；剪枝只能删除已证明不可能更优或与保留分支等价的状态。`}],
 quiz:[
 {id:'q1',type:'single',difficulty:1,prompt:md`回溯结束一个分支后，为什么必须撤销选择？`,options:['保证兄弟分支从相同父状态出发','使程序总是 O(n)','让答案自动排序'],answer:0,explanation:md`撤销保证正确的状态隔离，不会自动改变渐近复杂度或答案顺序。`},
 {id:'q2',type:'judge',difficulty:1,prompt:md`FFD 找到 4 个箱的合法方案，只能推出最优箱数不超过 4。`,answer:true,explanation:md`任何合法方案给出上界；要证明等于 4 还需要下界或完整搜索。`},
 {id:'q3',type:'numeric',difficulty:2,prompt:md`容量 10，物品 [6,6,6]，体积下界是多少？`,answer:2,tolerance:0,explanation:md`ceil(18/10)=2，但真实最优是 3。下界可能严格小于最优值。`},
 {id:'q4',type:'fill',difficulty:2,prompt:md`N 皇后副对角线的标识为行号 r ____ 列号 c。填写运算符。`,blanks:[{answers:['+']}],explanation:md`副对角线行列和相同，主对角线行列差相同。`},
 {id:'q5',type:'multi',difficulty:2,prompt:md`精确装箱搜索中哪些剪枝是安全的？`,options:['已有箱数不小于当前最佳值则停止寻找更优方案','同层只尝试一个相同负载的箱','只要体积下界为 3 就直接返回 3','先放大物品以便尽早发现容量冲突'],answer:[0,1,3],explanation:md`已开箱数不会减少，等负载箱对未来等价，降序只是搜索顺序优化；体积下界未必可达，不能直接当答案。`},
 {id:'q6',type:'order',difficulty:2,prompt:md`用 FFD 辅助分支定界的流程：`,items:['构造一个合法 FFD 方案得到上界','计算必要条件给出的下界','按降序搜索并剪掉不可能更优的分支','找到更优合法方案时收紧上界'],explanation:md`上界必须来自可行解，下界来自所有解必须满足的条件；两者方向不能颠倒。`},
 {id:'q7',type:'single',difficulty:3,prompt:md`两个当前箱的负载都是 6，处理同一个下一个物品时为何只需尝试一个？`,options:['箱没有身份要求，两种选择仅交换箱名','所有箱最终都必须一样满','每个物品的长度都相同'],answer:0,explanation:md`如果没有额外箱身份约束，等负载箱交换后未来可行性完全相同。若有设备兼容或样本类别约束，这个剪枝需重新证明。`},
 {id:'q8',type:'numeric',difficulty:3,prompt:md`容量 10，物品 [8,7,3,2]，最优箱数为多少？`,answer:2,tolerance:0,explanation:md`[8,2] 和 [7,3] 两箱可行，总体积 20 给下界 2，因此可行上界与必要下界相等，已证明最优。`},
 {id:'q9',type:'judge',difficulty:1,prompt:md`排序同层去重可以用来消除重复组合，但不能误删同一路径上合法的重复值。`,answer:true,explanation:md`例如两个不同的 2 可以同时出现在组合 [2,2] 中；去重应限制兄弟分支，不是禁止同值共存。`},
 {id:'q10',type:'match',difficulty:2,prompt:md`配对证明或状态要点：`,pairs:[['括号生成','任意前缀右括号不多于左括号'],['区间调度','结束最早选择的交换论证'],['棋盘单词搜索','当前路径的格子不能复用']],explanation:md`三种题目各有不同不变量；认识具体约束才能判断剪枝是否保留全部合法答案。`},
 ],guided:[],
 challenges:[{id:'c-exact',title:'BOSS：精确求最少序列箱数',difficulty:'hard',timeLimitMin:40,description:md`实现 min_bins(lengths: list[int], capacity: int) -> int。每个序列不可拆分，箱容量固定，求装下全部序列的最少箱数；空输入返回 0。1≤capacity≤30，1≤length≤capacity，n≤12。要求精确最优，FFD 只能作为初始上界。不得修改输入。例：[6,6,6],10 → 3；[8,7,3,2],10 → 2。`,starter:py`
  def min_bins(lengths: list[int], capacity: int) -> int:
      raise NotImplementedError
 `,solution:py`
  def min_bins(lengths: list[int], capacity: int) -> int:
      if not lengths:
          return 0
      items = sorted(lengths,reverse=True)
      greedy = []
      for x in items:
          for i,load in enumerate(greedy):
              if load+x <= capacity:
                  greedy[i] += x
                  break
          else:
              greedy.append(x)
      best = len(greedy)
      lower = (sum(items)+capacity-1)//capacity
      loads = []
      def dfs(i):
          nonlocal best
          if best == lower or len(loads) >= best:
              return
          if i == len(items):
              best = len(loads)
              return
          x = items[i]
          seen = set()
          for b in range(len(loads)):
              old = loads[b]
              if old in seen or old+x > capacity:
                  continue
              seen.add(old)
              loads[b] += x
              dfs(i+1)
              loads[b] -= x
          if len(loads)+1 < best:
              loads.append(x)
              dfs(i+1)
              loads.pop()
      dfs(0)
      return best
 `,explanation:md`FFD 提供可行上界，体积提供下界。按降序搜索，把当前物品放入每种不同负载的已有箱，最后尝试一个新箱；同负载箱仅交换名字所以可去重。已有箱数达到上界时不可能再改善，直接剪枝。若上界等于下界，最优性已经证明。搜索最坏仍是指数级，粗上界 O(n^n)，辅助空间 O(n²)（各层去重集合），适合本题的小规模。`,hints:[md`先求任意合法解，搜索只负责寻找更好的解。`,md`当前箱数已经不小于最佳值时，后续不可能减少已有箱。`,md`同层负载相同的箱只尝试一个，开新箱只需一个分支。`],setup:py`
  import random
  def _ref_min_bins(items,cap):
      # 子集 DP：状态为 (已开箱数, 当前最后一箱的负载)，按字典序取最优。
      n=len(items)
      if n==0: return 0
      dp=[(n+1,0)]*(1<<n); dp[0]=(1,0)
      for mask in range(1<<n):
          count,load=dp[mask]
          for i,x in enumerate(items):
              if mask>>i&1: continue
              state=(count,load+x) if load+x<=cap else (count+1,x)
              nxt=mask|(1<<i)
              dp[nxt]=min(dp[nxt],state)
      return dp[-1][0]
 `,tests:[{name:'下界不紧与上下界相等',code:py`assert min_bins([6,6,6],10)==3; assert min_bins([8,7,3,2],10)==2`},{name:'空与单箱',code:py`assert min_bins([],10)==0; assert min_bins([3,2,4],10)==1; assert min_bins([10],10)==1`},{name:'独立子集 DP 对拍',hidden:true,code:py`
  rnd=random.Random(6)
  for _ in range(100):
      cap=rnd.randrange(3,16); a=[rnd.randrange(1,cap+1) for _ in range(rnd.randrange(10))]; before=a[:]
      exp=_ref_min_bins(a,cap); got=min_bins(a,cap); assert got==exp and a==before, f"长度 {a}, 容量 {cap}：期望最优 {exp}，得到 {got}"
 `},{name:'十二件物品',hidden:true,code:py`
  a=[6,6,5,5,4,4,3,3,2,2,1,1]; exp=_ref_min_bins(a,10); got=min_bins(a,10)
  assert got==exp, f"期望 {exp}，得到 {got}"
 `},{name:'FFD 非最优反例',hidden:true,code:py`
  a=[6,5,3,2,2,2]; got=min_bins(a,10)
  assert got==2, f"可分成 [6,2,2] 和 [5,3,2]，期望 2，得到 {got}"
 `}]}],interview:[],cheatsheet:md`可行解给上界，必要条件给下界；相遇即证明最优。剪枝分为不可行、不能改善、对称重复。FFD 是上界，不是最优标签。回溯每步恢复状态；同负载箱只探一个代表；新箱仅开一个分支。`,
})
