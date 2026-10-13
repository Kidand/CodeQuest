/**
 * Kahn 拓扑排序：入度为 0 的点入队；弹出一个点就“删掉”它的出边，下游入度减到 0 再入队。
 * 打开“加一条回边”：环上的点入度永远减不到 0，输出数量 < n ⇒ 有环。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, GraphView, Legend, StepControls, VizFrame, useStepper, type GraphEdge, type GraphNode } from '@/viz'
import { Switch } from '@/components/ui'
import { C, OutRow, StackList } from './common'

const POS: [number, number][] = [
  [50, 60],
  [50, 190],
  [185, 60],
  [185, 190],
  [325, 95],
  [325, 215],
  [465, 150],
]
const BASE: [number, number][] = [
  [0, 2],
  [1, 2],
  [1, 3],
  [2, 4],
  [3, 4],
  [3, 5],
  [4, 6],
  [5, 6],
]
const BACK: [number, number] = [6, 3]

const CODE = `def find_order(n, prerequisites):
    graph = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in prerequisites:      # 先修 b 才能修 a：边 b → a
        graph[b].append(a)
        indeg[a] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:          # “删掉”u 的出边
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else []`

interface Frame {
  indeg: number[]
  q: number[]
  order: number[]
  cur?: number
  removed: Set<number>
  hot: Set<number>
  line: number | number[]
  caption: React.ReactNode
  end?: boolean
}

function buildFrames(cycle: boolean): { edges: [number, number][]; frames: Frame[] } {
  const edges = cycle ? [...BASE, BACK] : BASE
  const n = POS.length
  const indeg = Array(n).fill(0)
  const out: number[][] = Array.from({ length: n }, () => [])
  edges.forEach(([a, b], k) => {
    out[a].push(k)
    indeg[b]++
  })
  const q: number[] = []
  for (let i = 0; i < n; i++) if (indeg[i] === 0) q.push(i)
  const order: number[] = []
  const removed = new Set<number>()
  const frames: Frame[] = []
  frames.push({
    indeg: [...indeg], q: [...q], order: [], removed: new Set(), hot: new Set(), line: [2, 3, 4, 5, 6, 7],
    caption: <>建图并统计入度。入度为 0 的点 {q.join('、')} 没有任何先修课，先放进队列。</>,
  })
  while (q.length) {
    const u = q.shift()!
    order.push(u)
    const freed: number[] = []
    for (const k of out[u]) {
      const v = edges[k][1]
      indeg[v]--
      removed.add(k)
      if (indeg[v] === 0) {
        q.push(v)
        freed.push(v)
      }
    }
    frames.push({
      indeg: [...indeg], q: [...q], order: [...order], cur: u, removed: new Set(removed), hot: new Set(out[u]), line: [10, 11, 12, 13, 14, 15],
      caption: (
        <>
          弹出 <C>{u}</C> 放进结果，删掉它的 {out[u].length} 条出边
          {out[u].length ? <>（{out[u].map((k) => edges[k][1]).join('、')} 的入度各减 1）</> : ''}。
          {freed.length ? <>{freed.join('、')} 入度变为 0，入队。</> : out[u].length ? '没有点的入度降到 0。' : ''}
        </>
      ),
    })
  }
  const left = POS.map((_, i) => i).filter((i) => !order.includes(i))
  frames.push({
    indeg: [...indeg], q: [], order: [...order], removed, hot: new Set(), line: 16, end: true,
    caption:
      left.length === 0 ? (
        <>队列空，输出了全部 {n} 个点：<C>[{order.join(', ')}]</C> 是一个合法的拓扑序（同一时刻入队的点之间顺序可以互换，答案不唯一）。</>
      ) : (
        <>
          队列空了，但只输出了 {order.length} / {n} 个点。剩下的 {left.join('、')} 入度始终 &gt; 0：它们在环 3 → 4 → 6 → 3 / 3 → 5 → 6 → 3 上，互相等待对方先完成。返回 <C>[]</C>（LC 207 返回 False）。
        </>
      ),
  })
  return { edges, frames }
}

export function KahnViz() {
  const [cycle, setCycle] = useState(false)
  const { edges, frames } = useMemo(() => buildFrames(cycle), [cycle])
  const st = useStepper(frames, { intervalMs: 1200 })
  const f = st.frame
  const inQ = new Set(f.q)
  const done = new Set(f.order)
  const nodes: GraphNode[] = POS.map(([x, y], i) => ({
    id: i,
    label: i,
    x,
    y,
    sub: `入度 ${f.indeg[i]}`,
    tone: i === f.cur ? 'accent' : done.has(i) ? 'success' : inQ.has(i) ? 'info' : f.end ? 'danger' : undefined,
  }))
  const gEdges: GraphEdge[] = edges.map(([a, b], k) => ({
    from: a,
    to: b,
    directed: true,
    dim: f.removed.has(k) && !f.hot.has(k),
    tone: f.hot.has(k) ? 'accent' : k === BASE.length ? 'danger' : undefined,
    dashed: k === BASE.length,
    curve: k === BASE.length ? -0.32 : 0,
  }))
  return (
    <VizFrame
      title="Kahn 拓扑排序：入度减到 0 才入队"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '刚弹出', tone: 'accent', variant: 'soft' },
            { label: '在队列中', tone: 'info', variant: 'soft' },
            { label: '已输出', tone: 'success', variant: 'soft' },
            { label: '永远出不了队（环）', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Switch checked={cycle} onChange={setCycle} label="加一条回边 6 → 3（制造环）" />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <GraphView nodes={nodes} edges={gEdges} width={515} height={255} nodeR={19} />
          <StackList kind="queue" title="队列 q（左为队头）" items={f.q.map((x) => ({ key: x, text: x, tone: 'info' }))} />
          <OutRow label="order =" values={f.order} />
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ 'len(order)': f.order.length, n: POS.length }} />
      </div>
    </VizFrame>
  )
}
