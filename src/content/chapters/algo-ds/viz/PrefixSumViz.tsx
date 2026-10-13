import { useMemo, useState } from 'react'
import { ArrayView, StepControls, VizFrame, useStepper } from '@/viz'
import { Slider } from '@/components/ui'

export function PrefixSumViz() {
  const [target, setTarget] = useState(2)
  const a = [1, 2, -1, 1, -1, 2, 0, -2]
  const frames = useMemo(() => {
    const counts = new Map<number, number>([[0, 1]])
    let sum = 0, answer = 0
    const fs = [{i:-1,sum,answer,need: -target,hits:0,counts:[...counts],note:'空前缀出现一次；这样从下标 0 开始的子数组也能被计数。'}]
    a.forEach((x,i) => {
      sum += x
      const need = sum-target, hits = counts.get(need) ?? 0
      answer += hits
      fs.push({i,sum,answer,need,hits,counts:[...counts],note:`前缀和 ${sum}，先查询历史前缀 ${need}：出现 ${hits} 次，贡献 ${hits} 个非空子数组。`})
      counts.set(sum,(counts.get(sum) ?? 0)+1)
      fs.push({i,sum,answer,need,hits,counts:[...counts],note:`查询完成才登记当前前缀 ${sum}，避免 target=0 时数到长度为 0 的子数组。`})
    })
    return fs
  // a 是固定演示数据；仅 target 改变时重建帧。
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[target])
  const st = useStepper(frames), f = st.frame
  return <VizFrame title="先查历史，再登记当前前缀" caption={f.note} controls={<><StepControls stepper={st}/><Slider label="目标和 k" value={target} min={-2} max={4} onChange={setTarget}/></>}>
    <div className="min-w-0 space-y-4"><ArrayView values={a} highlights={f.i>=0?{[f.i]:'accent'}:{}}/><p className="text-sm text-fg">prefix = {f.sum} · 查找 {f.need} · 命中 {f.hits} · 累计答案 {f.answer}</p><div className="flex flex-wrap gap-2">{f.counts.map(([s,c])=><div key={s} className={`rounded-lg border p-3 text-sm ${s===f.need?'border-accent bg-surface-2 text-accent':'border-border text-muted'}`}>前缀 {s}：{c} 次</div>)}</div></div>
  </VizFrame>
}
