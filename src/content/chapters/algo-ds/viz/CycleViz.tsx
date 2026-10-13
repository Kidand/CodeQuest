/**
 * 环形链表 II（LC142）：快慢指针相遇 → 一个回到 head，同速走 a 步后在入口相遇。
 * 洞见：相遇时 slow 走了 s = a + b，fast 走了 2s，多走的 s 是环长 c 的整数倍 ⇒ a = (k−1)c + (c − b)。
 */
import { useMemo, useState } from 'react'
import { GraphView, Legend, StepControls, VizFrame, useStepper, type GraphEdge, type GraphNode, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { C, Stat } from './common'

interface Frame {
  phase: 1 | 2 | 3
  x: number // slow 或 p
  y: number // fast 或 q
  t: number
  caption: React.ReactNode
}

function build(a: number, c: number) {
  const step = (p: number) => (p < a + c - 1 ? p + 1 : a)
  const frames: Frame[] = []
  let s = 0
  let f = 0
  let t = 0
  frames.push({ phase: 1, x: 0, y: 0, t: 0, caption: <>阶段一：<C>slow</C> 每次走 1 步，<C>fast</C> 每次走 2 步，都从 head 出发。有环则必相遇（进环后 fast 每步追近 1 格）。</> })
  do {
    s = step(s)
    f = step(step(f))
    t++
    frames.push({
      phase: 1,
      x: s,
      y: f,
      t,
      caption: s === f ? (
        <>
          第 {t} 步相遇于节点 <b>{s}</b>：slow 走了 <C>s = {t}</C>，fast 走了 <C>2s = {2 * t}</C>。fast 多走的 {t} 步全在环里绕圈，所以 <b>s 是环长 c = {c} 的整数倍</b>。
        </>
      ) : (
        <>第 {t} 步：slow → {s}，fast → {f}。</>
      ),
    })
  } while (s !== f)
  const meet = s
  const b = (meet - a + c) % c
  let p = 0
  let q = meet
  let t2 = 0
  frames.push({
    phase: 2,
    x: p,
    y: q,
    t: 0,
    caption: (
      <>
        阶段二：<C>p</C> 回到 head，<C>q</C> 留在相遇点，两者<b>都每次走 1 步</b>。因为 <C>a = k·c − b</C>（a = {a}、b = {b}、c = {c}），p 走 a 步到入口时，q 也恰好绕回入口。
      </>
    ),
  })
  while (p !== q) {
    p = step(p)
    q = step(q)
    t2++
    frames.push({
      phase: p === q ? 3 : 2,
      x: p,
      y: q,
      t: t2,
      caption: p === q ? <>走了 {t2} 步（= a = {a}）后 p、q 在节点 <b>{p}</b> 相遇——这就是环的入口。全程 O(n) 时间、O(1) 空间。</> : <>第 {t2} 步：p → {p}，q → {q}。</>,
    })
  }
  if (frames[frames.length - 1].phase === 2) {
    frames[frames.length - 1] = { ...frames[frames.length - 1], phase: 3, caption: <>a = 0：head 本身就是入口，p、q 一开始就在同一位置。</> }
  }
  return { frames, meet, b }
}

export function CycleViz() {
  const [a, setA] = useState(3)
  const [c, setC] = useState(6)
  const { frames, meet, b } = useMemo(() => build(a, c), [a, c])
  const st = useStepper(frames, { intervalMs: 900 })
  const fr = st.frame

  const gap = 46
  const R = Math.max(46, (c * gap) / (2 * Math.PI))
  const x0 = 24
  const cy = R + 22
  const xe = x0 + gap * a
  const W = xe + 2 * R + 26
  const H = 2 * R + 58
  const pos = (i: number) => {
    if (i < a) return { x: x0 + gap * i, y: cy }
    const j = i - a
    const th = Math.PI - (2 * Math.PI * j) / c
    return { x: xe + R + R * Math.cos(th), y: cy - R * Math.sin(th) }
  }
  const n = a + c
  const nodes: GraphNode[] = Array.from({ length: n }, (_, i) => {
    const { x, y } = pos(i)
    const isX = fr.x === i
    const isY = fr.y === i
    let tone: VizTone | undefined
    let sub: string | undefined
    const [nx, ny] = fr.phase === 1 ? ['slow', 'fast'] : ['p', 'q']
    if (isX && isY) {
      tone = fr.phase === 3 ? 'success' : 'accent'
      sub = fr.phase === 3 ? '入口' : `${nx},${ny}`
    } else if (isX) {
      tone = 'info'
      sub = nx
    } else if (isY) {
      tone = 'warning'
      sub = ny
    }
    return { id: i, label: String(i), x, y, tone, sub, r: 15 }
  })
  const edges: GraphEdge[] = Array.from({ length: n }, (_, i) => ({
    from: i,
    to: i < n - 1 ? i + 1 : a,
    directed: true,
    tone: i === n - 1 ? ('danger' as VizTone) : undefined,
  }))

  return (
    <VizFrame
      title="快慢指针：相遇后再走 a 步到入口"
      caption={fr.caption}
      legend={
        <Legend
          items={[
            { label: fr.phase === 1 ? 'slow' : 'p（从 head）', tone: 'info', variant: 'soft' },
            { label: fr.phase === 1 ? 'fast' : 'q（从相遇点）', tone: 'warning', variant: 'soft' },
            { label: '回到入口的边', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="w-40" label="尾长 a" value={a} min={0} max={5} onChange={setA} />
          <Slider className="w-40" label="环长 c" value={c} min={3} max={8} onChange={setC} />
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <GraphView nodes={nodes} edges={edges} width={W} height={H} nodeR={15} />
        <div className="flex flex-wrap gap-2">
          <Stat label="尾长 a" value={a} />
          <Stat label="环长 c" value={c} />
          <Stat label="相遇点" value={meet} tone="accent" />
          <Stat label="入口→相遇点 b" value={b} />
          <Stat label={fr.phase === 1 ? '已走步数' : '阶段二步数'} value={fr.t} />
        </div>
      </div>
    </VizFrame>
  )
}
