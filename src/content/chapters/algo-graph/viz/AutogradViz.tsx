/**
 * autograd 的反向传播 = 在计算图上按“反向拓扑序”执行每个节点的 _backward。
 * 例子：a=2, b=-3, c=a·b, d=c+b, L=d·c。c 被 d 和 L 两处使用，b 被 c 和 d 两处使用。
 * 切到“BFS 顺序（错误）”：c 在 d 之前反传，此时 c.grad 还没收齐，a、b 的梯度算错。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, GraphView, Legend, StepControls, VizFrame, useStepper, type GraphEdge, type GraphNode } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './common'

type Name = 'a' | 'b' | 'c' | 'd' | 'L'
const VAL: Record<Name, number> = { a: 2, b: -3, c: -6, d: -9, L: 54 }
const POS: Record<Name, [number, number]> = { a: [55, 60], b: [55, 200], c: [205, 60], d: [330, 190], L: [465, 115] }
/** 每个节点的输入及局部导数 ∂out/∂in */
const INPUTS: Record<Name, [Name, number][]> = {
  a: [],
  b: [],
  c: [
    ['a', VAL.b],
    ['b', VAL.a],
  ],
  d: [
    ['c', 1],
    ['b', 1],
  ],
  L: [
    ['d', VAL.c],
    ['c', VAL.d],
  ],
}
const TRUE_GRAD: Record<Name, number> = { L: 1, d: -6, c: -15, a: 45, b: -36 }
const ORDERS = {
  topo: ['L', 'd', 'c', 'b', 'a'] as Name[],
  bfs: ['L', 'c', 'd', 'a', 'b'] as Name[],
}
type Mode = keyof typeof ORDERS

const CODE = `def backward(self):
    topo, seen = [], set()
    def build(v):                  # DFS 后序 = 拓扑序
        if v not in seen:
            seen.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)         # 输入都排在 v 前面
    build(self)
    self.grad = 1.0
    for v in reversed(topo):       # 反向拓扑序：消费者全部
        v._backward()              # 处理完，v.grad 才收齐`

interface Frame {
  grad: Record<Name, number>
  cur?: Name
  done: Set<Name>
  line: number | number[]
  caption: React.ReactNode
  wrongAt?: Name
}

function buildFrames(mode: Mode): Frame[] {
  const grad: Record<Name, number> = { a: 0, b: 0, c: 0, d: 0, L: 1 }
  const done = new Set<Name>()
  const frames: Frame[] = []
  frames.push({
    grad: { ...grad }, done: new Set(), line: mode === 'topo' ? [2, 3, 9, 10] : 10,
    caption:
      mode === 'topo' ? (
        <>前向已完成（节点下方是值 v 和梯度 g）。DFS 后序得到拓扑序 <C>[a, b, c, d, L]</C>，反过来就是反传顺序 <C>L → d → c → b → a</C>。令 <C>L.grad = 1</C>。</>
      ) : (
        <>换成从 L 出发的 BFS 顺序 <C>L → c → d → a → b</C>（c、d 都是 L 的输入，同一层），看看会发生什么。令 <C>L.grad = 1</C>。</>
      ),
  })
  for (const v of ORDERS[mode]) {
    const contrib = INPUTS[v].map(([u, local]) => {
      const add = local * grad[v]
      grad[u] += add
      return { u, local, add }
    })
    done.add(v)
    // 是否在梯度没收齐时就向下传了
    const premature = grad[v] !== TRUE_GRAD[v] || (mode === 'bfs' && v === 'c')
    frames.push({
      grad: { ...grad }, cur: v, done: new Set(done), line: [11, 12], wrongAt: premature && contrib.length ? v : undefined,
      caption: contrib.length ? (
        <>
          执行 <C>{v}._backward()</C>，此时 <C>{v}.grad = {grad[v]}</C>
          {mode === 'bfs' && v === 'c' ? <b>（还没收到来自 d 的 −6，不完整！）</b> : ''}：
          {contrib.map((x, i) => (
            <span key={i}>
              {' '}
              <C>
                {x.u}.grad += {x.local} × {grad[v]} = {x.add >= 0 ? '+' : ''}
                {x.add}
              </C>
            </span>
          ))}
          。
        </>
      ) : (
        <>
          <C>{v}</C> 是叶子（输入），<C>_backward</C> 什么也不做。<C>{v}.grad = {grad[v]}</C>
          {grad[v] === TRUE_GRAD[v] ? <>，与解析解一致 ✓</> : <>，但正确值是 {TRUE_GRAD[v]} ✗</>}。
        </>
      ),
    })
  }
  const ok = (['a', 'b'] as Name[]).every((x) => grad[x] === TRUE_GRAD[x])
  frames.push({
    grad: { ...grad }, done, line: [], caption: ok ? (
      <>
        结果：<C>∂L/∂a = 45</C>、<C>∂L/∂b = −36</C>。验证：<C>L = (ab + b)·ab</C>，<C>∂L/∂a = 2ab² + b² = 45</C>，<C>∂L/∂b = 2a²b + 2ab = −36</C> ✓。
        关键：c 被两处使用，它的梯度是两路之和 −9 + (−6) = −15，必须<b>两路都到齐</b>后再往下传。
      </>
    ) : (
      <>
        结果 <C>a.grad = {grad.a}</C>（应为 45）、<C>b.grad = {grad.b}</C>（应为 −36）。c 在只收到 −9 时就把梯度传给了 a、b，之后 d 再送来的 −6 永远传不下去了。
        这就是必须按<b>反向拓扑序</b>的原因：一个节点只有在它的所有“消费者”都反传完以后才能反传。
      </>
    ),
  })
  return frames
}

const fmt = (x: number) => (Number.isInteger(x) ? String(x) : x.toFixed(2))

export function AutogradViz() {
  const [mode, setMode] = useState<Mode>('topo')
  const frames = useMemo(() => buildFrames(mode), [mode])
  const st = useStepper(frames, { intervalMs: 1500 })
  const f = st.frame
  const nodes: GraphNode[] = (Object.keys(POS) as Name[]).map((k) => ({
    id: k,
    label: k,
    x: POS[k][0],
    y: POS[k][1],
    sub: `v=${fmt(VAL[k])} g=${fmt(f.grad[k])}`,
    tone: k === f.cur ? (f.wrongAt === k ? 'danger' : 'accent') : f.done.has(k) ? 'success' : undefined,
  }))
  const edges: GraphEdge[] = (Object.keys(INPUTS) as Name[]).flatMap((out) =>
    INPUTS[out].map(([inp, local]) => ({
      from: inp,
      to: out,
      directed: true,
      label: `∂=${local}`,
      tone: out === f.cur ? (f.wrongAt === out ? 'danger' : 'accent') : undefined,
    })),
  )
  return (
    <VizFrame
      title="反向传播 = 反向拓扑序"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '正在 _backward（边 = 局部导数）', tone: 'accent', variant: 'soft' },
            { label: '梯度未收齐就反传', tone: 'danger', variant: 'soft' },
            { label: '已反传', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented size="sm" value={mode} onChange={setMode} options={[{ value: 'topo', label: '反向拓扑序' }, { value: 'bfs', label: 'BFS 顺序（错误）' }]} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0">
          <GraphView nodes={nodes} edges={edges} width={520} height={250} nodeR={22} />
          <div className="mt-2 text-center font-mono text-[12.5px] text-muted">a = 2, b = −3, c = a·b, d = c + b, L = d·c</div>
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ order: ORDERS[mode].join(' → ') }} />
      </div>
    </VizFrame>
  )
}
