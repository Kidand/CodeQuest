/**
 * 标量计算图上的前向与反向：L = (w·x + b − y)²。
 * 前向按拓扑序计算并缓存中间值；反向按逆拓扑序，每个节点把“上游梯度 × 局部导数”传给输入。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, toneColor, toneSoft, useElementWidth, useStepper, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { C } from './plot'

const CODE = `# 前向（缓存中间值）
u = w * x
z = u + b
r = z - y
L = r ** 2
# 反向（逆拓扑序）
dL = 1.0
dr = dL * 2 * r       # d(r²)/dr = 2r
dz = dr * 1.0         # d(z−y)/dz = 1
du = dz * 1.0; db = dz * 1.0
dw = du * x; dx = du * w   # d(w·x)/dw = x`

type Key = 'x' | 'w' | 'b' | 'y' | 'u' | 'z' | 'r' | 'L'
interface Frame {
  line: number
  fwd: Key[]
  bwd: Key[]
  active?: Key
  text: string
}

/** 设计坐标（宽 300、高 330），绘制时按容器宽度横向缩放，字号保持真实像素 */
const POS: Record<Key, [number, number]> = {
  w: [50, 32],
  x: [150, 32],
  u: [100, 112],
  b: [215, 112],
  z: [158, 192],
  y: [262, 192],
  r: [210, 272],
  L: [90, 272],
}
const GH = 320
const NR = 21
const LABEL: Record<Key, string> = { x: 'x', w: 'w', b: 'b', y: 'y', u: 'u=w·x', z: 'z=u+b', r: 'r=z−y', L: 'L=r²' }
const EDGES: [Key, Key][] = [
  ['w', 'u'],
  ['x', 'u'],
  ['u', 'z'],
  ['b', 'z'],
  ['z', 'r'],
  ['y', 'r'],
  ['r', 'L'],
]

export function CompGraphViz() {
  const [w, setW] = useState(0.5)
  const x = 2
  const b = 0.3
  const y = 2
  const v = useMemo(() => {
    const u = w * x
    const z = u + b
    const r = z - y
    const L = r * r
    const dr = 2 * r
    return { val: { x, w, b, y, u, z, r, L } as Record<Key, number>, grad: { L: 1, r: dr, z: dr, u: dr, b: dr, w: dr * x, x: dr * w, y: -dr } as Record<Key, number> }
  }, [w])
  const f2 = (n: number) => n.toFixed(2)
  const frames: Frame[] = useMemo(
    () => [
      { line: 1, fwd: ['x', 'w', 'b', 'y'], bwd: [], text: `输入：x = ${x}，w = ${f2(w)}，b = ${b}，目标 y = ${y}。计算图是一个 DAG，前向按拓扑序逐个计算。` },
      { line: 2, fwd: ['x', 'w', 'b', 'y', 'u'], bwd: [], active: 'u', text: `u = w·x = ${f2(v.val.u)}（缓存 w、x，反向要用）` },
      { line: 3, fwd: ['x', 'w', 'b', 'y', 'u', 'z'], bwd: [], active: 'z', text: `z = u + b = ${f2(v.val.z)}` },
      { line: 4, fwd: ['x', 'w', 'b', 'y', 'u', 'z', 'r'], bwd: [], active: 'r', text: `r = z − y = ${f2(v.val.r)}（缓存 r）` },
      { line: 5, fwd: ['x', 'w', 'b', 'y', 'u', 'z', 'r', 'L'], bwd: [], active: 'L', text: `L = r² = ${v.val.L.toFixed(3)}。前向结束，开始反向。` },
      { line: 7, fwd: ['x', 'w', 'b', 'y', 'u', 'z', 'r', 'L'], bwd: ['L'], active: 'L', text: '反向起点：∂L/∂L = 1。' },
      { line: 8, fwd: ['x', 'w', 'b', 'y', 'u', 'z', 'r', 'L'], bwd: ['L', 'r'], active: 'r', text: `∂L/∂r = 1 × 2r = ${f2(v.grad.r)}（局部导数用到了前向缓存的 r）` },
      { line: 9, fwd: ['x', 'w', 'b', 'y', 'u', 'z', 'r', 'L'], bwd: ['L', 'r', 'z'], active: 'z', text: `∂L/∂z = ∂L/∂r × 1 = ${f2(v.grad.z)}（减法对被减数的局部导数为 1）` },
      { line: 10, fwd: ['x', 'w', 'b', 'y', 'u', 'z', 'r', 'L'], bwd: ['L', 'r', 'z', 'u', 'b'], active: 'u', text: `加法把梯度原样“分发”给两个输入：∂L/∂u = ∂L/∂b = ${f2(v.grad.u)}` },
      { line: 11, fwd: ['x', 'w', 'b', 'y', 'u', 'z', 'r', 'L'], bwd: ['L', 'r', 'z', 'u', 'b', 'w', 'x'], active: 'w', text: `乘法“交换”输入：∂L/∂w = ∂L/∂u × x = ${f2(v.grad.w)}，∂L/∂x = ∂L/∂u × w = ${f2(v.grad.x)}。梯度下降：w ← w − η·(${f2(v.grad.w)})。` },
    ],
    [v, w],
  )
  const st = useStepper(frames, { intervalMs: 1300 })
  const f = st.frame
  const [ref, cw] = useElementWidth()
  const W = Math.max(280, Math.min(520, cw || 320))
  const sx = (px: number) => 10 + ((px - 10) * (W - 20)) / 290
  const P = (k: Key) => [sx(POS[k][0]), POS[k][1]] as const
  const toneOf = (k: Key): VizTone => (f.active === k ? 'gold' : f.bwd.includes(k) ? 'danger' : f.fwd.includes(k) ? 'accent' : 'muted')
  return (
    <VizFrame
      title="反向传播 = 在计算图上按逆拓扑序做链式法则"
      caption={f.text}
      legend={
        <Legend
          items={[
            { label: '已前向计算', tone: 'accent', variant: 'soft' },
            { label: '已得到梯度', tone: 'danger', variant: 'soft' },
            { label: '当前节点', tone: 'gold', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[180px] flex-1" label="参数 w" value={w} min={-1} max={2} step={0.05} onChange={setW} format={(n) => n.toFixed(2)} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[1.2fr_1fr]">
        <div ref={ref} className="min-w-0">
          <svg viewBox={`0 0 ${W} ${GH}`} width={W} height={GH} className="mx-auto block overflow-visible">
            <defs>
              {(['accent', 'danger', 'muted'] as const).map((t) => (
                <marker key={t} id={`cg-arrow-${t}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M0,0 L10,5 L0,10 z" fill={toneColor(t)} />
                </marker>
              ))}
            </defs>
            {EDGES.map(([a, c]) => {
              const [x1, y1] = P(a)
              const [x2, y2] = P(c)
              const len = Math.hypot(x2 - x1, y2 - y1)
              const ux = (x2 - x1) / len
              const uy = (y2 - y1) / len
              const back = f.bwd.includes(c) && f.bwd.includes(a)
              const t = back ? 'danger' : f.fwd.includes(c) ? 'accent' : 'muted'
              return (
                <line
                  key={`${a}${c}`}
                  x1={x1 + ux * NR}
                  y1={y1 + uy * NR}
                  x2={x2 - ux * (NR + 3)}
                  y2={y2 - uy * (NR + 3)}
                  stroke={toneColor(t)}
                  strokeWidth={2}
                  strokeDasharray={back ? '5 4' : undefined}
                  markerEnd={`url(#cg-arrow-${t})`}
                />
              )
            })}
            {(Object.keys(POS) as Key[]).map((k) => {
              const [cx, cy] = P(k)
              const done = f.fwd.includes(k)
              const back = f.bwd.includes(k)
              const t = toneOf(k)
              return (
                <g key={k} opacity={done ? 1 : 0.45} style={{ transition: 'opacity 0.3s' }}>
                  <circle cx={cx} cy={cy} r={NR} fill={toneSoft(t, 28)} stroke={toneColor(t)} strokeWidth={2} />
                  <text x={cx} y={cy} textAnchor="middle" dominantBaseline="central" fontSize={13} fontWeight={700} fill="var(--fg)" className="font-mono">
                    {k}
                  </text>
                  <text x={cx} y={cy + NR + 13} textAnchor="middle" fontSize={11} fill="var(--muted)" className="font-mono">
                    {done ? `${f2(v.val[k])}` : LABEL[k]}
                  </text>
                  {back && (
                    <text x={cx} y={cy + NR + 26} textAnchor="middle" fontSize={11} fontWeight={600} fill={toneColor('danger')} className="font-mono">
                      ∂={f2(v.grad[k])}
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
          <div className="mt-1 text-center text-[12px] text-muted">
            节点下方：前向值；红色为 <C>∂L/∂·</C>。{' '}
            {(['u', 'z', 'r', 'L'] as Key[]).map((k) => LABEL[k]).join('，')}
          </div>
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ L: v.val.L.toFixed(3), 'dL/dw': f.bwd.includes('w') ? f2(v.grad.w) : '—' }} />
      </div>
    </VizFrame>
  )
}
