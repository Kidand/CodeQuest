import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'

export function MonotoneViz() {
  const [mode, setMode] = useState<'hist' | 'window'>('hist')
  const frames = useMemo(() => {
    const a = mode === 'hist' ? [2, 1, 5, 6, 2, 3] : [1, 3, -1, -3, 5, 3, 6, 7]
    const q: number[] = [], out: number[] = []
    let best = 0
    const fs = [{ a, q: [...q], i: -1, best, out: [...out], note: '保存下标，才能同时得到数值和距离。', line: 1 }]
    for (let i = 0; i < a.length + (mode === 'hist' ? 1 : 0); i++) {
      if (mode === 'window' && q.length && q[0] <= i - 3) {
        const gone = q.shift()!
        fs.push({ a, q: [...q], i, best, out: [...out], note: `下标 ${gone} 离开长度 3 的窗口，从队首删除。`, line: 2 })
      }
      while (q.length && (mode === 'hist' ? (a[q[q.length - 1]] > (a[i] ?? 0)) : a[q[q.length - 1]] <= a[i])) {
        const j = q.pop()!
        const left = q[q.length - 1] ?? -1
        if (mode === 'hist') best = Math.max(best, a[j] * (i - left - 1))
        fs.push({ a, q: [...q], i, best, out: [...out], note: mode === 'hist' ? `弹出柱 ${j}：高 ${a[j]}，左边界 ${left}，右边界 ${i}，宽 ${i - left - 1}；最大面积 ${best}。` : `${a[i]} 比候选 ${a[j]} 更新且不小，旧候选永远不可能成为最大值。`, line: 3 })
      }
      q.push(i)
      if (mode === 'window' && i >= 2) out.push(a[q[0]])
      fs.push({ a, q: [...q], i, best, out: [...out], note: mode === 'hist' ? `下标 ${i} 入栈；递增栈等待右侧更矮柱。末尾虚拟高度 0 会结算剩余柱。` : `下标 ${i} 入队；队首 ${q[0]} 是当前有效窗口最大值。`, line: 4 })
    }
    return fs
  }, [mode])
  const st = useStepper(frames), f = st.frame
  return <VizFrame title="每个下标只进出一次" caption={f.note} controls={<><StepControls stepper={st}/><Segmented value={mode} onChange={setMode} options={[{value:'hist',label:'柱状图面积'},{value:'window',label:'窗口最大值'}]}/></>}>
    <div className="grid gap-4 @3xl:grid-cols-2">
      <div className="min-w-0 space-y-4"><ArrayView values={f.a} highlights={f.i >= 0 && f.i < f.a.length ? {[f.i]:'accent'} : {}}/><div className="text-sm text-muted">{mode === 'hist' ? '栈底 → 栈顶' : '队首 → 队尾'}（下标）</div><ArrayView values={f.q.length ? f.q : ['空']} /><p className="text-sm text-fg">{mode === 'hist' ? `最大面积 = ${f.best}` : `已输出 = [${f.out.join(', ')}]`}</p></div>
      <CodeTrace code={mode === 'hist' ? 'for i in range(n + 1):\n    h = heights[i] if i < n else 0\n    while stack and heights[stack[-1]] > h: pop_and_measure()\n    stack.append(i)' : 'for i, x in enumerate(nums):\n    if q and q[0] <= i-k: q.popleft()\n    while q and nums[q[-1]] <= x: q.pop()\n    q.append(i)  # 完整窗口输出 nums[q[0]]'} active={f.line}/>
    </div>
  </VizFrame>
}
