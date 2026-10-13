/**
 * Kimi K3 的 Quantile Balancing（按论文 §2.3.3 的描述实现；直方图近似等工程细节以原文为准）：
 *   α_i = token i 的第 (k+1) 大选择分数（cutoff）；margin_ij = s_ij − α_i；
 *   b̂_j = −(第 q+1 大的 margin_j)，q = m·k/n；b = b̂ − mean(b̂)；新 bias 只在下一步生效。
 * 「机制」：m=8、n=4、k=1（q=2）的手算例子，一步从 (4,3,0,1) 到 (2,2,2,2)。
 * 「收敛」：E=16、k=2、每步 512 token 的冷热不均数据流，对比 sign 规则。
 */
import { useMemo, useState } from 'react'
import { Legend, MatrixView, StepControls, Tex, VizFrame, useStepper, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { maxVio, normals } from './moeMath'
import { Axes, Line, LoadBars, Stat, Svg, type Axis } from './plot'

const S = [
  [0.56, 0.9, 0.32, 0.52],
  [0.8, 0.67, 0.34, 0.51],
  [0.52, 0.25, 0.24, 0.44],
  [0.6, 0.82, 0.53, 0.45],
  [0.9, 0.62, 0.59, 0.26],
  [0.7, 0.84, 0.72, 0.48],
  [0.76, 0.63, 0.73, 0.15],
  [0.67, 0.57, 0.47, 0.7],
]
const M_ = 8
const N_ = 4
const Q = 2 // m·k/n

const argmax = (r: number[]) => r.reduce((b, v, i) => (v > r[b] ? i : b), 0)
const r2 = (v: number) => Math.round(v * 1000) / 1000

function mech() {
  const route0 = S.map(argmax)
  const load0 = Array.from({ length: N_ }, (_, j) => route0.filter((e) => e === j).length)
  const alpha = S.map((r) => [...r].sort((a, b) => b - a)[1]) // 第 k+1 = 2 大
  const margin = S.map((r, i) => r.map((v) => r2(v - alpha[i])))
  const thrRow = Array.from({ length: N_ }, (_, j) => {
    const col = margin.map((r) => r[j])
    const order = col.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0])
    return { thr: order[Q][0], thrIdx: order[Q][1], top: order.slice(0, Q).map(([, i]) => i) }
  })
  const bhat = thrRow.map((t) => -t.thr)
  const mb = bhat.reduce((a, b) => a + b, 0) / N_
  const bias = bhat.map((v) => v - mb)
  const choice = S.map((r) => r.map((v, j) => v + bias[j]))
  const route1 = choice.map(argmax)
  const load1 = Array.from({ length: N_ }, (_, j) => route1.filter((e) => e === j).length)
  return { route0, load0, alpha, margin, thrRow, bhat, bias, choice, route1, load1 }
}

// ---------- 收敛对比 ----------
const E = 16
const K = 2
const T = 512
const STEPS = 30
const POP = Array.from({ length: E }, (_, e) => 0.9 - (1.8 * e) / (E - 1))
let STREAM: Float64Array[] | null = null
function stream() {
  if (!STREAM) {
    STREAM = Array.from({ length: STEPS }, (_, t) => {
      const z = normals(9000 + t, T * E)
      return Float64Array.from(z, (v, i) => 1 / (1 + Math.exp(-(POP[i % E] + v))))
    })
  }
  return STREAM
}
function topkIdx(row: number[], k: number) {
  return row.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i)
}
function converge(mode: 'qb' | 'sign', gamma: number) {
  const data = stream()
  let b = new Array(E).fill(0)
  const out: number[] = []
  for (let t = 0; t < STEPS; t++) {
    const s = data[t]
    const load = new Array(E).fill(0)
    const alpha = new Array(T)
    for (let i = 0; i < T; i++) {
      const row = Array.from({ length: E }, (_, e) => s[i * E + e] + b[e])
      const sorted = [...row].sort((x, y) => y - x)
      alpha[i] = sorted[K]
      topkIdx(row, K).forEach((e) => load[e]++)
    }
    out.push(maxVio(load))
    if (mode === 'sign') {
      const mean = (T * K) / E
      b = b.map((v, e) => v + gamma * Math.sign(mean - load[e]))
    } else {
      const q = (T * K) / E
      const bh = Array.from({ length: E }, (_, e) => {
        const col = Array.from({ length: T }, (_, i) => s[i * E + e] - alpha[i]).sort((x, y) => y - x)
        return -col[q]
      })
      const m = bh.reduce((a, v) => a + v, 0) / E
      b = bh.map((v) => v - m)
    }
  }
  return out
}

const FRAMES = ['① 普通 top-1 路由', '② 每个 token 的 cutoff α（第 k+1 大）', '③ margin = s − α，每列取第 q+1 大', '④ 新 bias = −阈值 − 均值，重新路由']

export function QuantileBalanceViz() {
  const [view, setView] = useState<'mech' | 'conv'>('mech')
  const [gamma, setGamma] = useState(0.01)
  const d = useMemo(mech, [])
  const st = useStepper(FRAMES, { intervalMs: 1800 })
  const f = st.i
  const qb = useMemo(() => (view === 'conv' ? converge('qb', 0) : []), [view])
  const sg = useMemo(() => (view === 'conv' ? converge('sign', gamma) : []), [view, gamma])

  const cols = ['E0', 'E1', 'E2', 'E3']
  const rows = S.map((_, i) => `t${i}`)
  let data: number[][] = S
  let hl: (r: number, c: number) => VizTone | undefined = (r, c) => (d.route0[r] === c ? 'accent' : undefined)
  let colLabels: string[] = cols
  let scale: 'none' | 'div' = 'none'
  if (f === 1) {
    data = S.map((r, i) => [...r, d.alpha[i]])
    colLabels = [...cols, 'α']
    hl = (r, c) => (c === 4 ? 'warning' : S[r][c] === d.alpha[r] ? 'warning' : d.route0[r] === c ? 'accent' : undefined)
  } else if (f === 2) {
    data = d.margin
    scale = 'div'
    hl = (r, c) => (d.thrRow[c].top.includes(r) ? 'success' : d.thrRow[c].thrIdx === r ? 'danger' : undefined)
  } else if (f === 3) {
    data = d.choice
    hl = (r, c) => (d.route1[r] === c ? 'success' : undefined)
  }

  const captions = [
    <>普通 top-1：负载 ({d.load0.join(', ')})，目标 q = m·k/n = {Q}。E0 过载、E2 一个都没有。</>,
    <>每个 token 的 cutoff α_i 是它第 k+1 = 2 大的选择分数（橙色）：一个专家的分数要超过它才能进入该 token 的 top-1。</>,
    <>
      margin = s − α。每列中 margin 最大的 q = {Q} 个（绿）应该留下，第 q+1 = 3 大的（红）就是阈值 [{d.thrRow.map((t) => t.thr.toFixed(2)).join(', ')}]：
      把 bias 设成它的相反数，恰好 q 个 margin 在阈值之上。
    </>,
    <>
      <Tex>{String.raw`\hat b`}</Tex> = [{d.bhat.map((v) => v.toFixed(2)).join(', ')}]，减去均值得 b = [{d.bias.map((v) => v.toFixed(4)).join(', ')}]（整体平移不改变 top-k）。
      用 s + b 重新路由：负载 ({d.load1.join(', ')})——一步到位，没有步长 γ。
    </>,
  ]

  return (
    <VizFrame
      title="Quantile Balancing：用分位数直接算出 bias"
      caption={
        view === 'mech' ? (
          captions[f]
        ) : (
          <>
            16 个专家、top-2、每步 512 个新 token。QB 第 1 步之后 MaxVio 就降到 <b className="font-mono">{qb[1]?.toFixed(2)}</b>（此后在采样噪声水平附近波动）；
            sign 规则（γ={gamma}）第 5 步为 <b className="font-mono">{sg[5]?.toFixed(2)}</b>、第 20 步为 <b className="font-mono">{sg[20]?.toFixed(2)}</b>。
            QB 没有步长超参，不存在「γ 小了慢、大了震荡」的取舍。
          </>
        )
      }
      legend={
        view === 'mech' ? (
          <Legend
            items={[
              { label: '当前路由', tone: 'accent', variant: 'soft' },
              { label: 'cutoff α', tone: 'warning', variant: 'soft' },
              { label: '留下 / 新路由', tone: 'success', variant: 'soft' },
              { label: '第 q+1 大 = 阈值', tone: 'danger', variant: 'soft' },
            ]}
          />
        ) : (
          <Legend
            items={[
              { label: 'Quantile Balancing', tone: 'accent', variant: 'line' },
              { label: `sign 规则 γ=${gamma}`, tone: 'warning', variant: 'line' },
            ]}
          />
        )
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={view}
            onChange={setView}
            options={[
              { value: 'mech', label: '机制（8×4 手算）' },
              { value: 'conv', label: '收敛（对比 sign）' },
            ]}
          />
          {view === 'mech' ? (
            <StepControls stepper={st} />
          ) : (
            <div className="flex items-center gap-2 text-[13px] text-muted">
              sign 的 γ
              <Segmented size="sm" value={gamma} onChange={setGamma} options={[0.003, 0.01, 0.03].map((g) => ({ value: g, label: String(g) }))} />
            </div>
          )}
        </>
      }
    >
      {view === 'mech' ? (
        <div className="grid items-start gap-4 @3xl:grid-cols-[3fr_2fr]">
          <div className="min-w-0 space-y-2">
            <div className="text-xs font-medium text-muted">{FRAMES[f]}</div>
            <MatrixView data={data} rowLabels={rows} colLabels={colLabels} scale={scale} domain={scale === 'div' ? [-0.6, 0.6] : undefined} cellSize={46} highlight={hl} />
          </div>
          <div className="min-w-0 space-y-3">
            <div className="text-xs font-medium text-muted">各专家负载（目标 q = {Q}）</div>
            <LoadBars values={f === 3 ? d.load1 : d.load0} labels={cols} tones={(f === 3 ? d.load1 : d.load0).map((v) => (v > Q ? 'danger' : v < Q ? 'info' : 'success'))} refValue={Q} refLabel="q = 2" max={4} />
            {f === 3 && (
              <div className="grid grid-cols-2 gap-2">
                <Stat label="更新前 MaxVio" value={maxVio(d.load0).toFixed(2)} tone="danger" />
                <Stat label="更新后 MaxVio" value={maxVio(d.load1).toFixed(2)} tone="success" />
              </div>
            )}
          </div>
        </div>
      ) : (
        <ConvPlot qb={qb} sg={sg} />
      )}
    </VizFrame>
  )
}

function ConvPlot({ qb, sg }: { qb: number[]; sg: number[] }) {
  const x: Axis = { min: 0, max: STEPS - 1, ticks: [0, 5, 10, 15, 20, 25], label: '训练步' }
  const y: Axis = { min: 0, max: 2, ticks: [0, 0.5, 1, 1.5, 2], label: 'MaxVio', fmt: (v) => v.toFixed(1) }
  return (
    <Svg>
      <Axes x={x} y={y} />
      <Line pts={sg.map((v, i) => [i, v])} x={x} y={y} tone="warning" />
      <Line pts={qb.map((v, i) => [i, v])} x={x} y={y} tone="accent" />
    </Svg>
  )
}
