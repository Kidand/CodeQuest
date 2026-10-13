/**
 * 爬楼梯 f(n) = f(n-1) + f(n-2)：朴素递归树 vs 记忆化递归树（逐次调用展开）。
 * 洞见：同色节点 = 同一个子问题被重复计算（重叠子问题）；记忆化后递归树退化成一条链。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, TreeView, VizFrame, useStepper, type TreeNode, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { C, Stat, Sub } from './common'

type Mode = 'naive' | 'memo'

const CODE: Record<Mode, string> = {
  naive: `def f(n):
    if n <= 1:
        return 1
    return f(n - 1) + f(n - 2)`,
  memo: `memo = {}
def f(n):
    if n in memo:
        return memo[n]          # 命中：直接返回
    if n <= 1:
        res = 1
    else:
        res = f(n - 1) + f(n - 2)
    memo[n] = res               # 返回前写入缓存
    return res`,
}

interface RNode {
  id: number
  k: number
  parent?: number
  children: number[]
  hit?: boolean
}
interface Ev {
  type: 'enter' | 'exit'
  id: number
  value?: number
}

function simulate(n: number, mode: Mode) {
  const nodes: RNode[] = []
  const events: Ev[] = []
  const memo = new Map<number, number>()
  const f = (k: number, parent?: number): number => {
    const id = nodes.length
    nodes.push({ id, k, parent, children: [] })
    if (parent != null) nodes[parent].children.push(id)
    events.push({ type: 'enter', id })
    if (mode === 'memo' && memo.has(k)) {
      nodes[id].hit = true
      const v = memo.get(k)!
      events.push({ type: 'exit', id, value: v })
      return v
    }
    const v = k <= 1 ? 1 : f(k - 1, id) + f(k - 2, id)
    if (mode === 'memo') memo.set(k, v)
    events.push({ type: 'exit', id, value: v })
    return v
  }
  f(n)
  return { nodes, events }
}

interface Frame {
  visible: Set<number>
  values: Map<number, number>
  active: number
  ev: Ev
  memo: (number | null)[]
  calls: number
}

function buildFrames(n: number, mode: Mode) {
  const { nodes, events } = simulate(n, mode)
  const frames: Frame[] = []
  const visible = new Set<number>()
  const values = new Map<number, number>()
  const memo: (number | null)[] = Array(n + 1).fill(null)
  let calls = 0
  for (const ev of events) {
    if (ev.type === 'enter') {
      visible.add(ev.id)
      calls++
    } else {
      values.set(ev.id, ev.value!)
      if (mode === 'memo' && !nodes[ev.id].hit) memo[nodes[ev.id].k] = ev.value!
    }
    frames.push({ visible: new Set(visible), values: new Map(values), active: ev.id, ev, memo: [...memo], calls })
  }
  return { nodes, frames, total: calls }
}

/** 朴素递归调用次数：C(n) = C(n-1) + C(n-2) + 1 */
function naiveCalls(n: number): number {
  let a = 1
  let b = 1
  for (let i = 2; i <= n; i++) [a, b] = [b, a + b + 1]
  return n <= 1 ? 1 : b
}

export function RecursionTreeViz() {
  const [n, setN] = useState(5)
  const [mode, setMode] = useState<Mode>('naive')
  const { nodes, frames, total } = useMemo(() => buildFrames(n, mode), [n, mode])
  const st = useStepper(frames, { intervalMs: 650 })
  const f = st.frame
  const cur = nodes[f.active]

  const toTree = (id: number): TreeNode => {
    const nd = nodes[id]
    const done = f.values.has(id)
    const isActive = id === f.active
    let tone: VizTone = ((nd.k % 8) + 1) as VizTone
    if (nd.hit) tone = 'success'
    return {
      id,
      label: `f${nd.k}`,
      tone: isActive ? 'accent' : done ? tone : undefined,
      sub: nd.hit ? '命中' : done ? `=${f.values.get(id)}` : isActive ? '…' : undefined,
      children: nd.children.filter((c) => f.visible.has(c)).map(toTree),
      edgeTone: isActive ? 'accent' : undefined,
    }
  }

  const line = (() => {
    const ev = f.ev
    if (mode === 'naive') {
      if (ev.type === 'enter') return cur.k <= 1 ? 2 : 4
      return cur.k <= 1 ? 3 : 4
    }
    if (ev.type === 'enter') return cur.hit ? 4 : cur.k <= 1 ? 6 : 8
    return cur.hit ? 4 : 9
  })()

  const caption =
    f.ev.type === 'enter' ? (
      cur.hit ? (
        <>
          调用 <C>f({cur.k})</C>：memo 里已有结果 <C>{f.memo[cur.k]}</C>，<b>直接返回</b>，这棵子树不再展开。
        </>
      ) : (
        <>
          调用 <C>f({cur.k})</C>
          {cur.k <= 1 ? '：到达边界，返回 1。' : <>：需要先算 <C>f({cur.k - 1})</C> 和 <C>f({cur.k - 2})</C>。</>}
        </>
      )
    ) : (
      <>
        <C>f({cur.k})</C> 返回 <C>{f.values.get(f.active)}</C>
        {mode === 'memo' && !cur.hit ? <>，写入 <C>memo[{cur.k}]</C>。</> : '。'}
        {st.atEnd && (
          <>
            {' '}
            全部完成：共 <b>{total}</b> 次调用
            {mode === 'naive' ? <>，其中大量是重复的子问题（同色节点）。</> : <>，每个子问题只真正计算一次。</>}
          </>
        )}
      </>
    )

  return (
    <VizFrame
      title="重叠子问题：朴素递归指数爆炸，记忆化后只剩一条链"
      caption={caption}
      legend={
        <Legend
          items={[
            { label: '同色 = 同一个子问题 f(k)', tone: 3, variant: 'soft' },
            { label: '当前调用', tone: 'accent', variant: 'soft' },
            { label: '记忆化命中（剪掉整棵子树）', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'naive', label: '朴素递归' },
              { value: 'memo', label: '记忆化递归' },
            ]}
          />
          <Slider className="w-40" label="台阶数 n" value={n} min={3} max={6} onChange={setN} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Stat label="已调用" value={f.calls} tone="accent" />
          <Stat label="本模式总调用" value={total} />
          <Stat label={`朴素递归 n=${n}`} value={naiveCalls(n)} tone="danger" />
          <Stat label="记忆化" value={2 * n - 1} tone="success" />
        </div>
        <div className="overflow-x-auto py-1">
          <TreeView root={toTree(0)} nodeSize={34} levelGap={54} siblingGap={8} />
        </div>
        <div className="grid items-start gap-4 @2xl:grid-cols-[1fr_1.2fr]">
          <div>
            <Sub>{mode === 'memo' ? 'memo 表（返回时写入）' : '朴素递归没有缓存：同一 f(k) 每次都从头算'}</Sub>
            {mode === 'memo' ? (
              <ArrayView
                values={f.memo.map((v) => (v == null ? '·' : v))}
                cellSize={40}
                highlights={Object.fromEntries(f.memo.flatMap((v, i) => (v == null ? [] : [[i, i === cur.k ? 'accent' : 'success']])))}
              />
            ) : (
              <div className="text-[13px] leading-relaxed text-muted">
                n={n} 时 <C>f(2)</C> 被算了 {nodes.filter((x) => x.k === 2).length} 次、<C>f(1)</C> 被算了 {nodes.filter((x) => x.k === 1).length} 次；调用次数满足 <C>C(n)=C(n-1)+C(n-2)+1</C>，与斐波那契数同阶，即 <C>O(φⁿ)</C>，φ≈1.618。
              </div>
            )}
          </div>
          <CodeTrace code={CODE[mode]} active={line} vars={{ 调用: `f(${cur.k})`, 已调用: f.calls }} />
        </div>
      </div>
    </VizFrame>
  )
}
