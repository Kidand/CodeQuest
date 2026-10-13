import { defineLevel, md, py } from '@/content/helpers'
import { COUNTED_NUM } from '../shared'
import { MonotoneViz } from '../viz/MonotoneViz'

export default defineLevel({
  learn: [
    { id:'stack', title:'弹栈不是丢弃，而是确定答案', viz:MonotoneViz, body:md`
      对“下一个更大元素”，栈中存还没找到答案的**下标**，对应值从底到顶递减。遇到新值 x，就弹出所有小于 x 的候选：x 是它们右侧遇到的第一个更大值。未弹出的候选仍需等待。存下标可以直接计算每日温度的等待天数，也能在有重复值时区分元素。

      嵌套 while 不代表平方复杂度：每个下标只入栈一次、出栈一次，总操作数不超过 2n。这个**摊还分析**比“每轮都是 O(1)”准确；某一轮确实可能弹出整个栈。严格更大应使用小于号，相等时不结算；改成大于等于的题目必须同步调整比较符。
    ` },
    { id:'area', title:'柱状图：高度固定，寻找最宽区间', body:md`
      固定一根高为 h 的柱子，矩形最多延伸到左右第一根更矮柱之前。维护递增栈，遇到更矮的新柱 i 时弹出 j；此时右边界确定为 i，弹出后的栈顶是左边界 l，面积为 $h_j(i-l-1)$。左边没有更矮柱时 l=-1。

      末尾加虚拟高度 0，让从未遇到更矮柱的候选也被结算。使用严格大于弹栈时，相等柱会留在栈中；较靠左的等高柱最终能覆盖完整宽度，所以最大值仍正确。接雨水也能用栈，但每次结算的是两侧边界夹出的横向水层，不要机械套用柱状图面积公式。
    ` },
    { id:'deque', title:'单调队列：同时处理过期和支配', body:md`
      窗口最值多了一个条件：元素会从左侧过期。双端队列从队首删除下标小于窗口左端的候选，从队尾删除不大于当前值的候选。后者被当前元素**支配**：当前值更大或相等，且更晚过期，因此旧值以后也不会胜出。

      队首始终是当前窗口最大值。队列里存下标才能判断过期，单存值遇到重复元素很容易删错。总时间 O(n)，辅助空间 O(k)；用 Python list.pop(0) 会搬移元素，应使用 collections.deque。这个结构适合在线滑动统计，但不能直接用来维护任意区间的中位数。
    ` },
  ],
  quiz: [
    {id:'q1',type:'single',difficulty:1,prompt:md`每日温度需要输出等待天数，栈最好存什么？`,options:['温度下标','温度的去重集合','目前最高温度'],answer:0,explanation:md`下标同时提供温度和距离。集合丢失顺序、重复位置；最高温度不能给每个未解位置答案。`},
    {id:'q2',type:'judge',difficulty:1,prompt:md`单调栈某一轮可以弹出 n 个元素，但整个扫描仍是 O(n)。`,answer:true,explanation:md`每个元素只会被弹出一次，总弹出次数受 n 限制。不能把某轮最坏耗时乘到每一轮。`},
    {id:'q3',type:'numeric',difficulty:2,prompt:md`柱高 [2,1,5,6,2,3] 的最大矩形面积是多少？`,answer:10,tolerance:0,explanation:md`高度 5 覆盖下标 2、3，面积 5×2=10；高度 6 只有宽 1，整段高度 1 面积只有 6。`},
    {id:'q4',type:'fill',difficulty:2,prompt:md`右边界 i、弹栈后的左边界 left 都是不能纳入矩形的柱，下标间的宽度为 ____。`,blanks:[{answers:['i-left-1','i - left - 1']}],explanation:md`可用下标是 left+1 到 i-1，数量为 i-left-1；少减一个会错误包含边界。`},
    {id:'q5',type:'multi',difficulty:2,prompt:md`固定窗口最大值的单调队列，哪些操作正确？`,options:['队首删除过期下标','队尾删除不大于新值的候选','每次从队首删除小于新值的全部元素','完整窗口形成后读取队首'],answer:[0,1,3],explanation:md`队首管过期，队尾管支配；从队首按值删除会破坏队列中仍有用的候选。`},
    {id:'q6',type:'order',difficulty:2,prompt:md`一次窗口推进采用下面的实现顺序：`,items:['删除已离开窗口的队首','删除被新值支配的队尾','当前下标入队','若窗口完整，输出队首对应值'],explanation:md`在输出前保证下标有效且队列单调。先输出可能用到过期的最大值。`},
    {id:'q7',type:'single',difficulty:3,prompt:md`柱状图用 > 弹栈，输入 [2,2,2]，末尾用高度 0 清栈，会漏掉面积 6 吗？`,options:['不会，最左等高柱最后结算完整宽度','会，相等时必须弹栈','会，哨兵必须是负数'],answer:0,explanation:md`三个柱依次结算宽 1、2、3，最大值为 6。非负柱高用 0 哨兵即可；零高不弹也不影响面积。`},
    {id:'q8',type:'numeric',difficulty:3,prompt:md`数组 [4,4,1,5]，窗口 k=2，所有窗口最大值之和为多少？`,answer:13,tolerance:0,explanation:md`三个窗口的最大值分别为 4、4、5；重复的 4 可以保留更晚下标，保证过期时仍有正确代表。`},
  ],
  guided:[{id:'g-temp',title:'每日温度：把等待者逐个结算',intro:md`实现 daily(temperatures)，返回等到更高温度的天数，无更高温度则为 0。`,template:py`
    def daily(temperatures):
        answer = [0] * len(temperatures)
        stack = []
        for i, value in enumerate(temperatures):
            while stack and {{1}}:
                j = {{2}}
                answer[j] = {{3}}
            stack.append(i)
        return answer
  `,steps:[{blank:1,instruction:md`只结算温度严格低于当前温度的候选。`,answer:'temperatures[stack[-1]] < value',explain:md`相等不是升温，必须继续等待。`},{blank:2,instruction:md`取出最近的未解下标。`,answer:'stack.pop()'},{blank:3,instruction:md`记录等待距离。`,answer:'i - j'}],tests:[{name:'普通和重复',code:py`got = daily([73,74,75,71,69,72,76,73]); assert got == [1,1,4,2,1,1,0,0], f"得到 {got}"`},{name:'边界',code:py`assert daily([]) == []; assert daily([5,5,4]) == [0,0,0]`}] }],
  challenges:[
    {id:'c-area',title:'柱状图最大矩形',difficulty:'hard',source:'LC 84',timeLimitMin:25,description:md`实现 largest_rectangle(heights: list[int]) -> int。柱宽为 1，高度非负；返回连续柱能容纳的最大矩形面积。空数组返回 0，不修改输入。长度不超过 100000，要求 O(n) 时间、O(n) 空间。例：[2,1,5,6,2,3] → 10。`,starter:py`
      def largest_rectangle(heights: list[int]) -> int:
          raise NotImplementedError
    `,solution:py`
      def largest_rectangle(heights: list[int]) -> int:
          stack, best = [], 0
          for i in range(len(heights) + 1):
              h = heights[i] if i < len(heights) else 0
              while stack and heights[stack[-1]] > h:
                  j = stack.pop()
                  left = stack[-1] if stack else -1
                  best = max(best, heights[j] * (i - left - 1))
              stack.append(i)
          return best
    `,explanation:md`维护高度不降的下标栈。弹出的柱已经得到右边界，新的栈顶给左边界。虚拟末尾只负责结算，不读取不存在的柱。每个真实下标进出一次，因此 O(n) 时间、O(n) 空间。宽度少减 1、忘记末尾清算是两个常见错误。`,hints:[md`固定矩形最矮柱，寻找两侧更矮边界。`,md`右边界在弹栈时确定；左边界是弹栈后的栈顶。`],setup:COUNTED_NUM + "\n\n" + py`
      import random
      def _ref_area(a):
          best = 0
          for i in range(len(a)):
              h = a[i]
              for j in range(i, len(a)):
                  h = min(h, a[j])
                  best = max(best, h * (j-i+1))
          return best
    `,tests:[{name:'示例',code:py`got=largest_rectangle([2,1,5,6,2,3]); assert got==10, f"期望 10，得到 {got}"`},{name:'空与平台',code:py`assert largest_rectangle([])==0; assert largest_rectangle([3,3,3])==9; assert largest_rectangle([0])==0`},{name:'随机对拍',hidden:true,code:py`
      rnd=random.Random(84)
      for _ in range(180):
          a=[rnd.randrange(9) for _ in range(rnd.randrange(20))]
          before=a[:]; got=largest_rectangle(a); exp=_ref_area(before)
          assert got==exp and a==before, f"输入 {before}：期望 {exp} 且不改输入，得到 {got}"
    `},{name:'单调边界',hidden:true,code:py`
      a=list(range(1,501)); got=largest_rectangle(a); exp=max(h*(501-h) for h in a)
      assert got==exp, f"期望 {exp}，得到 {got}"
    `},{name:'线性复杂度预算',hidden:true,code:py`
      n=3000; _Num.n=0; _Num.budget=32*n
      a=[_Num(i+1) for i in range(n)]; got=largest_rectangle(a); exp=max(h*(n-h+1) for h in range(1,n+1))
      assert got==exp, f"期望 {exp}，得到 {got}"
    `}],followUps:[{id:'f-stream',q:md`数据流没结束时，可以立刻得到最终最大面积吗？`,a:md`只能得到当前前缀的答案；栈中柱的右边界可能继续延伸。查询前缀答案时可以临时结算副本，但频繁这样做会增加总复杂度。`}]},
    {id:'c-window',title:'滑动窗口最大值',difficulty:'hard',source:'LC 239',timeLimitMin:20,description:md`实现 max_window(nums: list[int], k: int) -> list[int]。返回每个连续 k 个元素窗口的最大值；若 nums 为空且 k=0，返回 []，其它情况 1≤k≤n≤100000。要求 O(n) 时间、O(k) 辅助空间。例：[1,3,-1,-3,5,3,6,7],3 → [3,3,5,5,6,7]。`,starter:py`
      def max_window(nums: list[int], k: int) -> list[int]:
          raise NotImplementedError
    `,solution:py`
      from collections import deque
      def max_window(nums: list[int], k: int) -> list[int]:
          q, answer = deque(), []
          for i, x in enumerate(nums):
              if q and q[0] <= i-k:
                  q.popleft()
              while q and nums[q[-1]] <= x:
                  q.pop()
              q.append(i)
              if i >= k-1:
                  answer.append(nums[q[0]])
          return answer
    `,explanation:md`过期候选从队首离开，被新值支配的候选从队尾离开。队列下标递增、值严格递减，队首就是有效最大值。使用 deque 避免 list 的头部删除搬移。每个下标最多入队、出队各一次。`,hints:[md`把“还有资格成为最大值”的下标存进双端队列。`,md`相等时保留较新的下标，因为它更晚过期。`],setup:COUNTED_NUM + "\n\n" + py`import random`,tests:[{name:'示例',code:py`got=max_window([1,3,-1,-3,5,3,6,7],3); assert got==[3,3,5,5,6,7], f"得到 {got}"`},{name:'边界',code:py`assert max_window([],0)==[]; assert max_window([-2,-2],1)==[-2,-2]; assert max_window([1,2,3],3)==[3]`},{name:'随机窗口',hidden:true,code:py`
      rnd=random.Random(239)
      for _ in range(150):
          a=[rnd.randrange(-9,10) for _ in range(rnd.randrange(1,35))]; k=rnd.randrange(1,len(a)+1)
          exp=[max(a[i:i+k]) for i in range(len(a)-k+1)]; got=max_window(a,k)
          assert got==exp, f"a={a}, k={k}，期望 {exp}，得到 {got}"
    `},{name:'大量重复过期',hidden:true,code:py`got=max_window([8]*500+[1]*500,500); exp=[8]*500+[1]; assert got==exp, f"期望末窗为 1，结果长度 {len(got)}"`},{name:'窗口复杂度预算',hidden:true,code:py`
      n=3000; _Num.n=0; _Num.budget=32*n
      got=max_window([_Num(n-i) for i in range(n)],1500); exp=list(range(n,1499,-1))
      assert got==exp, f"期望 {len(exp)} 个递减窗口最大值，得到 {len(got)} 个"
    `} ]},
  ],
  interview:[
    {id:'i-amortized',freq:3,q:md`为什么嵌套 while 仍然是 O(n)？`,a:md`结论是总操作数线性，而非每次外循环恒定。把一次入栈和未来的一次出栈都记在同一个下标上，每个下标只支付两次操作，最多 2n 次。某一轮弹出 n 个元素后，后续轮次不可能再弹出这些元素。类似记账法还适用于双指针、单调队列和动态数组扩容的摊还分析。`},
    {id:'i-equal',freq:3,q:md`相等元素应该弹出吗？`,a:md`先按题意定义不变量。下一个严格更大元素中，相等值不能结算；窗口最大值中，相等的旧值可删，因为新值更晚过期。柱状图使用严格或非严格比较都能求最大面积，但左右边界的含义不同，不能混用证明。面试时先说明重复值策略，再用全相等输入验证。`},
    {id:'i-range',freq:2,q:md`任意区间最值也用单调队列吗？`,a:md`单调队列依赖窗口左右端都单调前进，才能永久丢弃过期元素。任意区间查询可能回头，已删除元素又会被需要。静态数组可以使用稀疏表或线段树；有点更新时通常选线段树。数据结构选择取决于查询顺序与更新模式，不只是“求最大值”四个字。`},
  ],cheatsheet:md`**下一个更大**：递减栈，遇到更大值弹栈结算距离。**柱状图**：递增栈，弹栈面积 = height × (right-left-1)，末尾 0 清算。**窗口最大值**：队首去过期、队尾去支配、队首读答案。存下标；每个元素最多进出一次；重复值先定义严格性。`,
})
