/**
 * Viterbi 解码（log 域）：
 *  - TrellisViz：网格图上逐格计算 δ_t(j) 与回溯指针，最后回溯最优路径
 *  - BroadcastViz：一步 Viterbi = 广播加法 (N,1)+(N,N) + 按列取 max / argmax
 *  - BeamVsViterbiViz：beam search（不合并同状态路径）= 剪枝近似；宽度不够会丢掉最优路径
 *  - UnigramLatticeViz：Unigram 分词 = 位置格上的 Viterbi，对比贪心最长匹配
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, MatrixView, StepControls, VizFrame, toneColor, toneSoft, useElementWidth, useStepper, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { C, Stat, Sub } from './common'

/* ------------------------------------------------------------------ */
/* 玩具 HMM（3 个隐状态、3 种观测，T = 5）                              */
/* ------------------------------------------------------------------ */

export const STATES = ['晴', '阴', '雨']
export const SYMS = ['散步', '购物', '宅家']
export const PI = [0.1, 0.5, 0.4]
export const A = [
  [0.2, 0.5, 0.3],
  [0.4, 0.1, 0.5],
  [0.7, 0.1, 0.2],
]
export const B = [
  [0.6, 0.1, 0.3],
  [0.3, 0.6, 0.1],
  [0.3, 0.2, 0.5],
]
export const OBS = [0, 0, 2, 1, 2]
const N = 3
const T = OBS.length
const L = Math.log
const f2 = (x: number) => (Number.isFinite(x) ? x.toFixed(2) : '−∞')

function runViterbi() {
  const delta: number[][] = [PI.map((p, i) => L(p) + L(B[i][OBS[0]]))]
  const back: number[][] = [[-1, -1, -1]]
  for (let t = 1; t < T; t++) {
    const d: number[] = []
    const b: number[] = []
    for (let j = 0; j < N; j++) {
      let best = -Infinity
      let arg = 0
      for (let i = 0; i < N; i++) {
        const s = delta[t - 1][i] + L(A[i][j])
        if (s > best) {
          best = s
          arg = i
        }
      }
      d.push(best + L(B[j][OBS[t]]))
      b.push(arg)
    }
    delta.push(d)
    back.push(b)
  }
  let j = delta[T - 1].indexOf(Math.max(...delta[T - 1]))
  const path = [j]
  for (let t = T - 1; t > 0; t--) {
    j = back[t][j]
    path.push(j)
  }
  path.reverse()
  return { delta, back, path, best: Math.max(...delta[T - 1]) }
}
const VIT = runViterbi()

/* ------------------------------------------------------------------ */
/* 网格图（trellis）绘制                                                */
/* ------------------------------------------------------------------ */

interface TEdge {
  t: number // 指向第 t 列
  i: number
  j: number
  tone?: VizTone
  dashed?: boolean
  label?: string
}

function Trellis({
  values,
  edges,
  nodeTone,
  path,
  pathTone = 'success',
  format = f2,
}: {
  values: (number | null)[][] // [t][j]
  edges: TEdge[]
  nodeTone?: (t: number, j: number) => VizTone | undefined
  path?: number[]
  pathTone?: VizTone
  format?: (v: number) => string
}) {
  const [ref, avail] = useElementWidth()
  const W = Math.max(300, Math.min(avail || 620, 680))
  const left = 34
  const right = 30
  const top = 34
  const rowGap = 74
  const r = W < 420 ? 19 : 22
  const H = top + (N - 1) * rowGap + r + 22
  const x = (t: number) => left + r + ((W - left - right - 2 * r) * t) / (T - 1)
  const y = (j: number) => top + r / 2 + 8 + j * rowGap
  const onPath = (t: number, j: number) => !!path && path[t] === j
  const pathEdges: TEdge[] = path ? path.slice(1).map((j, k) => ({ t: k + 1, i: path[k], j, tone: pathTone })) : []
  const line = (e: TEdge, w: number) => {
    const x1 = x(e.t - 1) + r
    const y1 = y(e.i)
    const x2 = x(e.t) - r - 3
    const y2 = y(e.j)
    return (
      <g key={`${e.t}-${e.i}-${e.j}-${w}`}>
        <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={e.tone ? toneColor(e.tone) : 'var(--border-strong)'} strokeWidth={w} strokeDasharray={e.dashed ? '4 3' : undefined} markerEnd={e.tone ? `url(#vt-${String(e.tone)})` : undefined} />
        {e.label && (
          <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 4} textAnchor="middle" fontSize={10.5} fontWeight={600} fill={toneColor(e.tone ?? 'muted')} paintOrder="stroke" stroke="var(--surface)" strokeWidth={3} className="font-mono">
            {e.label}
          </text>
        )}
      </g>
    )
  }
  return (
    <div ref={ref} className="w-full min-w-0">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="mx-auto block overflow-visible select-none">
        <defs>
          {(['success', 'accent', 'warning', 'danger', 'info', 'muted'] as VizTone[]).map((t) => (
            <marker key={String(t)} id={`vt-${String(t)}`} viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill={toneColor(t)} />
            </marker>
          ))}
        </defs>
        {OBS.map((o, t) => (
          <text key={`o${t}`} x={x(t)} y={14} textAnchor="middle" fontSize={11.5} fill="var(--muted)">
            t={t}·{SYMS[o]}
          </text>
        ))}
        {STATES.map((s, j) => (
          <text key={`s${j}`} x={4} y={y(j)} dominantBaseline="central" fontSize={12} fontWeight={600} fill="var(--muted)">
            {s}
          </text>
        ))}
        {/* 背景边：全连接 */}
        {Array.from({ length: T - 1 }, (_, k) => k + 1).flatMap((t) =>
          [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => <line key={`bg${t}${i}${j}`} x1={x(t - 1) + r} y1={y(i)} x2={x(t) - r} y2={y(j)} stroke="var(--border)" strokeWidth={1} />)),
        )}
        {edges.map((e) => line(e, 2))}
        {pathEdges.map((e) => line(e, 3))}
        {values.map((col, t) =>
          col.map((v, j) => {
            const tone = onPath(t, j) ? pathTone : nodeTone?.(t, j)
            return (
              <g key={`n${t}${j}`}>
                <circle
                  cx={x(t)}
                  cy={y(j)}
                  r={r}
                  fill={tone ? toneSoft(tone, 30) : v == null ? 'var(--surface-2)' : 'var(--surface)'}
                  stroke={tone ? toneColor(tone) : 'var(--border-strong)'}
                  strokeWidth={tone ? 2.4 : 1.2}
                />
                <text x={x(t)} y={y(j)} textAnchor="middle" dominantBaseline="central" fontSize={r < 21 ? 10 : 11} fontWeight={600} fill="var(--fg)" className="font-mono tabular-nums">
                  {v == null ? '·' : format(v)}
                </text>
              </g>
            )
          }),
        )}
      </svg>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* 1) 逐格 Viterbi                                                      */
/* ------------------------------------------------------------------ */

interface VFrame {
  vals: (number | null)[][]
  t: number
  j: number
  cands: number[] // 各前驱 i 的 δ_{t-1}(i) + log A[i][j]
  arg: number
  caption: React.ReactNode
  path?: number[]
  line: number | number[]
}

function buildVit(): VFrame[] {
  const vals: (number | null)[][] = Array.from({ length: T }, () => Array(N).fill(null))
  const fr: VFrame[] = []
  vals[0] = [...VIT.delta[0]]
  fr.push({
    vals: vals.map((c) => [...c]),
    t: 0,
    j: -1,
    cands: [],
    arg: -1,
    line: 2,
    caption: (
      <>
        初始化：<C>δ₀(j) = log π_j + log B_j(散步)</C>，如 晴：log 0.1 + log 0.6 = <C>{f2(VIT.delta[0][0])}</C>。
      </>
    ),
  })
  for (let t = 1; t < T; t++)
    for (let j = 0; j < N; j++) {
      const cands = [0, 1, 2].map((i) => VIT.delta[t - 1][i] + L(A[i][j]))
      const arg = VIT.back[t][j]
      vals[t][j] = VIT.delta[t][j]
      fr.push({
        vals: vals.map((c) => [...c]),
        t,
        j,
        cands,
        arg,
        line: [5, 6, 7],
        caption: (
          <>
            δ<sub>{t}</sub>({STATES[j]})：三个前驱 {cands.map((c, i) => `${STATES[i]} ${f2(c)}`).join('、')}，最大来自 <b>{STATES[arg]}</b>；再加 log B({STATES[j]}, {SYMS[OBS[t]]}) = {f2(L(B[j][OBS[t]]))}，得 <C>{f2(VIT.delta[t][j])}</C>。
          </>
        ),
      })
    }
  fr.push({
    vals: vals.map((c) => [...c]),
    t: T - 1,
    j: -1,
    cands: [],
    arg: -1,
    path: VIT.path,
    line: [8, 9, 10],
    caption: (
      <>
        最后一列取最大 <C>{f2(VIT.best)}</C>（概率 ≈ {Math.exp(VIT.best).toExponential(2)}），沿回溯指针倒推：<b>{VIT.path.map((j) => STATES[j]).join(' → ')}</b>。
      </>
    ),
  })
  return fr
}

const VIT_CODE = `def viterbi(log_pi, log_A, log_B, obs):
    delta = log_pi + log_B[:, obs[0]]            # (N,)
    back = []
    for o in obs[1:]:
        scores = delta[:, None] + log_A          # (N, N)：i → j
        back.append(scores.argmax(axis=0))       # 每个 j 的最优前驱
        delta = scores.max(axis=0) + log_B[:, o]
    path = [int(delta.argmax())]
    for bp in reversed(back):                    # 回溯
        path.append(int(bp[path[-1]]))
    return path[::-1], float(delta.max())`

export function TrellisViz() {
  const frames = useMemo(buildVit, [])
  const st = useStepper(frames, { intervalMs: 1500 })
  const f = st.frame
  const edges: TEdge[] =
    f.j >= 0
      ? f.cands.map((c, i) => ({ t: f.t, i, j: f.j, tone: i === f.arg ? ('success' as const) : ('muted' as const), dashed: i !== f.arg, label: i === f.arg ? f2(c) : undefined }))
      : []
  return (
    <VizFrame
      title="Viterbi：每个格子只保留「到达它的最优路径」，最后沿回溯指针倒推"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前计算的 δ_t(j)', tone: 'accent', variant: 'soft' },
            { label: '最优前驱（回溯指针）', tone: 'success', variant: 'line' },
            { label: '落选的前驱', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="space-y-4">
        <div className="grid gap-3 text-[12px] @xl:grid-cols-3">
          <div>
            <Sub>初始 π</Sub>
            <MatrixView data={[PI]} colLabels={STATES} scale="none" format={(v) => v.toFixed(1)} cellSize={38} />
          </div>
          <div>
            <Sub>转移 A[i][j]（行 i → 列 j）</Sub>
            <MatrixView data={A} rowLabels={STATES} colLabels={STATES} scale="seq" domain={[0, 1]} format={(v) => v.toFixed(1)} cellSize={34} />
          </div>
          <div>
            <Sub>发射 B[j][o]</Sub>
            <MatrixView data={B} rowLabels={STATES} colLabels={SYMS} scale="seq" domain={[0, 1]} format={(v) => v.toFixed(1)} cellSize={34} />
          </div>
        </div>
        <div className="overflow-x-auto py-1">
          <Trellis values={f.vals} edges={edges} nodeTone={(t, j) => (t === f.t && j === f.j ? 'accent' : undefined)} path={f.path} />
        </div>
        <CodeTrace code={VIT_CODE} active={f.line} />
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 2) 一步 Viterbi 的广播视图                                           */
/* ------------------------------------------------------------------ */

export function BroadcastViz() {
  const [t, setT] = useState(2)
  const prev = VIT.delta[t - 1]
  const scores = prev.map((d, i) => A[i].map((a) => d + L(a)))
  const arg = [0, 1, 2].map((j) => VIT.back[t][j])
  const colMax = [0, 1, 2].map((j) => scores[arg[j]][j])
  const emit = [0, 1, 2].map((j) => L(B[j][OBS[t]]))
  return (
    <VizFrame
      title="一步 Viterbi = (N,1) + (N,N) 广播，再按列 max / argmax"
      caption={
        <>
          t = {t}（观测「{SYMS[OBS[t]]}」）：<C>scores = delta[:, None] + log_A</C>，第 j 列的最大值（绿框）就是到达 j 的最优路径分数，
          <C>argmax(axis=0)</C> = [{arg.map((i) => STATES[i]).join(', ')}] 存为回溯指针。复杂度 O(N²)/步，总 O(TN²)。
        </>
      }
      legend={<Legend items={[{ label: '每列最大值（max / argmax 的位置）', tone: 'success', variant: 'soft' }]} />}
      controls={<Slider className="w-48" label="时间步 t" value={t} min={1} max={T - 1} onChange={setT} />}
    >
      <div className="grid items-start gap-5 @xl:grid-cols-[auto_1fr]">
        <div>
          <Sub>delta[:, None]（上一步 δ）</Sub>
          <MatrixView data={prev.map((d) => [d])} rowLabels={STATES} colLabels={['δ']} scale="none" format={f2} cellSize={52} />
        </div>
        <div className="min-w-0 space-y-3">
          <div>
            <Sub>scores[i, j] = δ(i) + log A[i, j]（行 = 前驱 i，列 = 当前 j）</Sub>
            <MatrixView
              data={scores}
              rowLabels={STATES}
              colLabels={STATES}
              scale="none"
              format={f2}
              cellSize={56}
              highlight={(r, c) => (arg[c] === r ? 'success' : undefined)}
            />
          </div>
          <div>
            <Sub>max(axis=0) + log B[:, o_t] = 新的 δ</Sub>
            <MatrixView
              data={[colMax, emit, colMax.map((m, j) => m + emit[j])]}
              rowLabels={['列最大', `+log B(${SYMS[OBS[t]]})`, 'δ_t']}
              colLabels={STATES}
              scale="none"
              format={f2}
              cellSize={56}
              highlight={(r) => (r === 2 ? 'accent' : undefined)}
            />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 3) Beam search vs Viterbi                                            */
/* ------------------------------------------------------------------ */

interface Hyp {
  path: number[]
  score: number
}

function beamSearch(W: number) {
  let beam: Hyp[] = [0, 1, 2]
    .map((i) => ({ path: [i], score: L(PI[i]) + L(B[i][OBS[0]]) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, W)
  const steps: Hyp[][] = [beam]
  for (let t = 1; t < T; t++) {
    const cand: Hyp[] = []
    for (const h of beam) for (let j = 0; j < N; j++) cand.push({ path: [...h.path, j], score: h.score + L(A[h.path[h.path.length - 1]][j]) + L(B[j][OBS[t]]) })
    beam = cand.sort((a, b) => b.score - a.score).slice(0, W)
    steps.push(beam)
  }
  return steps
}

export function BeamVsViterbiViz() {
  const [W, setW] = useState(1)
  const steps = useMemo(() => beamSearch(W), [W])
  const [t, setT] = useState(T - 1)
  const beam = steps[t]
  const final = steps[T - 1][0]
  const optimal = final.path.join() === VIT.path.join()
  // 当前 t 时刻 beam 中各路径的节点
  const vals: (number | null)[][] = Array.from({ length: T }, () => Array(N).fill(null))
  const edges: TEdge[] = []
  beam.forEach((h, k) => {
    h.path.forEach((j, tt) => {
      if (tt <= t) vals[tt][j] = 0
      if (tt >= 1 && tt <= t) edges.push({ t: tt, i: h.path[tt - 1], j, tone: k === 0 ? 'accent' : 'info', dashed: k !== 0 })
    })
  })
  const viterbiAtT = VIT.path.slice(0, t + 1)
  const kept = beam.some((h) => h.path.join() === viterbiAtT.join())
  return (
    <VizFrame
      title="Beam search = 只保留前 B 条路径的剪枝近似；Viterbi 按状态合并，精确"
      caption={
        <>
          B = {W}，t = {t}：beam 中{kept ? '仍然' : <b className="text-danger">已经没有</b>}包含最优路径的前缀 <C>{viterbiAtT.map((j) => STATES[j]).join('→')}</C>。
          最终 beam 输出 <C>{final.path.map((j) => STATES[j]).join('→')}</C>（{f2(final.score)}），Viterbi 最优为 <C>{f2(VIT.best)}</C>
          {optimal ? '，找到了最优解。' : <b className="text-danger">，beam 丢掉了最优解！</b>}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'beam 中得分最高的路径', tone: 'accent', variant: 'line' },
            { label: 'beam 中其它路径', tone: 'info', variant: 'line' },
            { label: 'Viterbi 最优路径', tone: 'success', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="w-44" label="beam 宽度 B" value={W} min={1} max={3} onChange={setW} />
          <Slider className="w-44" label="时间步 t" value={t} min={0} max={T - 1} onChange={setT} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Stat label="beam 输出" value={f2(final.score)} tone={optimal ? 'success' : 'danger'} />
          <Stat label="Viterbi 最优" value={f2(VIT.best)} tone="success" />
          <Stat label="差距（nats）" value={f2(VIT.best - final.score)} />
        </div>
        <div className="overflow-x-auto py-1">
          <Trellis values={vals} edges={edges} path={VIT.path} pathTone="success" format={() => ''} nodeTone={(tt, j) => (vals[tt][j] != null ? 'accent' : undefined)} />
        </div>
        <div>
          <Sub>t = {t} 时 beam 里的候选（按分数排序）</Sub>
          <div className="flex flex-wrap gap-2">
            {beam.map((h, k) => (
              <span key={k} className="rounded-lg border border-border bg-surface px-2 py-1 font-mono text-[12px]" style={{ borderColor: k === 0 ? 'var(--accent)' : undefined }}>
                {h.path.map((j) => STATES[j]).join('→')} <span className="text-muted">{f2(h.score)}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 4) Unigram 分词：位置格上的 Viterbi                                  */
/* ------------------------------------------------------------------ */

const TEXT = '研究生命起源'
export const UNI_VOCAB: [string, number][] = [
  ['研究', 0.03],
  ['研究生', 0.01],
  ['生命', 0.02],
  ['起源', 0.015],
  ['研', 0.002],
  ['究', 0.001],
  ['生', 0.01],
  ['命', 0.004],
  ['起', 0.005],
  ['源', 0.003],
]

interface Arc {
  s: number
  e: number
  piece: string
  lp: number
}
const ARCS: Arc[] = []
for (let s = 0; s < TEXT.length; s++)
  for (let e = s + 1; e <= TEXT.length; e++) {
    const v = UNI_VOCAB.find(([p]) => p === TEXT.slice(s, e))
    if (v) ARCS.push({ s, e, piece: v[0], lp: L(v[1]) })
  }

interface UFrame {
  i: number
  best: (number | null)[]
  inArcs: Arc[]
  win?: Arc
  path?: Arc[]
  caption: React.ReactNode
}

function buildUni(): UFrame[] {
  const n = TEXT.length
  const best: (number | null)[] = Array(n + 1).fill(null)
  const from: (Arc | null)[] = Array(n + 1).fill(null)
  best[0] = 0
  const fr: UFrame[] = [{ i: 0, best: [...best], inArcs: [], caption: <>位置 0（空前缀）：<C>best[0] = 0</C>。每条弧是词表中的一个 piece，权重是 log p。</> }]
  for (let i = 1; i <= n; i++) {
    const inArcs = ARCS.filter((a) => a.e === i && best[a.s] != null)
    let win: Arc | undefined
    for (const a of inArcs) if (!win || best[a.s]! + a.lp > best[win.s]! + win.lp) win = a
    best[i] = best[win!.s]! + win!.lp
    from[i] = win!
    fr.push({
      i,
      best: [...best],
      inArcs,
      win,
      caption: (
        <>
          位置 {i}：以「{TEXT[i - 1]}」结尾的候选 {inArcs.map((a) => `best[${a.s}]+「${a.piece}」= ${f2(best[a.s]! + a.lp)}`).join('，')}；最优是「{win!.piece}」，
          <C>best[{i}] = {f2(best[i]!)}</C>。
        </>
      ),
    })
  }
  const path: Arc[] = []
  let i = n
  while (i > 0) {
    path.push(from[i]!)
    i = from[i]!.s
  }
  path.reverse()
  fr.push({
    i: n,
    best: [...best],
    inArcs: [],
    path,
    caption: (
      <>
        回溯得到最大概率切分 <b>{path.map((a) => a.piece).join(' / ')}</b>，log p = <C>{f2(best[n]!)}</C>。注意 best[3] 走的是「研究生」，但全局最优并不经过位置 3——局部最优前缀不等于全局最优。
      </>
    ),
  })
  return fr
}

function greedyLongest(): Arc[] {
  const out: Arc[] = []
  let s = 0
  while (s < TEXT.length) {
    const a = ARCS.filter((x) => x.s === s).sort((p, q) => q.e - p.e)[0]
    out.push(a)
    s = a.e
  }
  return out
}

export function UnigramLatticeViz() {
  const frames = useMemo(buildUni, [])
  const [mode, setMode] = useState<'viterbi' | 'greedy'>('viterbi')
  const st = useStepper(frames, { intervalMs: 1400 })
  const f = st.frame
  const greedy = useMemo(greedyLongest, [])
  const greedyScore = greedy.reduce((s, a) => s + a.lp, 0)
  const [ref, avail] = useElementWidth()
  const n = TEXT.length
  const Wd = Math.max(300, Math.min(avail || 600, 620))
  const pad = 22
  const xs = (k: number) => pad + ((Wd - 2 * pad) * k) / n
  const base = 112
  const H = base + 62
  const shownPath = mode === 'greedy' ? greedy : f.path
  const arcPath = (a: Arc) => {
    const x1 = xs(a.s)
    const x2 = xs(a.e)
    const len = a.e - a.s
    const h = len === 1 ? -26 : 26 + (len - 1) * 30 // 单字弧画在下方
    const my = base - h * 1.5
    return { d: `M${x1},${base} Q${(x1 + x2) / 2},${my} ${x2},${base}`, lx: (x1 + x2) / 2, ly: base - h * 0.75 + (len === 1 ? 12 : -4) }
  }
  const toneOf = (a: Arc): VizTone | undefined => {
    if (shownPath?.includes(a)) return mode === 'greedy' ? 'danger' : 'success'
    if (mode === 'viterbi' && f.win === a) return 'success'
    if (mode === 'viterbi' && f.inArcs.includes(a)) return 'warning'
    return undefined
  }
  return (
    <VizFrame
      title="Unigram 分词：在「位置格」上跑 Viterbi，求最大概率切分"
      caption={
        mode === 'greedy' ? (
          <>
            贪心最长匹配（WordPiece / 正向最大匹配的思路）：每步取最长的 piece → <b>{greedy.map((a) => a.piece).join(' / ')}</b>，log p = <C>{f2(greedyScore)}</C>，比 Viterbi 的{' '}
            <C>{f2(frames[frames.length - 1].best[n]!)}</C> 低——“研究生”抢走了“生”，后面只能拆出低概率的“命”。
          </>
        ) : (
          f.caption
        )
      }
      legend={
        <Legend
          items={[
            { label: '当前位置的候选弧', tone: 'warning', variant: 'line' },
            { label: '最优（Viterbi）', tone: 'success', variant: 'line' },
            { label: '贪心最长匹配', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          {mode === 'viterbi' && <StepControls stepper={st} />}
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'viterbi', label: 'Viterbi 最大概率' },
              { value: 'greedy', label: '贪心最长匹配' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-3">
        <div ref={ref} className="w-full min-w-0">
          <svg width={Wd} height={H} viewBox={`0 0 ${Wd} ${H}`} className="mx-auto block overflow-visible select-none">
            <line x1={xs(0)} y1={base} x2={xs(n)} y2={base} stroke="var(--border-strong)" strokeWidth={1.2} />
            {ARCS.map((a, k) => {
              const g = arcPath(a)
              const tone = toneOf(a)
              return (
                <g key={k}>
                  <path d={g.d} fill="none" stroke={tone ? toneColor(tone) : 'var(--border-strong)'} strokeWidth={tone ? 2.6 : 1.2} strokeDasharray={tone ? undefined : '3 3'} />
                  <text x={g.lx} y={g.ly} textAnchor="middle" fontSize={11} fontWeight={tone ? 700 : 500} fill={tone ? toneColor(tone) : 'var(--muted)'} paintOrder="stroke" stroke="var(--surface)" strokeWidth={3}>
                    {a.piece.length > 1 ? `${a.piece} ${f2(a.lp)}` : f2(a.lp)}
                  </text>
                </g>
              )
            })}
            {Array.from({ length: n + 1 }, (_, k) => (
              <g key={k}>
                <circle cx={xs(k)} cy={base} r={6} fill={mode === 'viterbi' && k === f.i ? 'var(--accent)' : 'var(--surface)'} stroke="var(--border-strong)" />
                <text x={xs(k)} y={base + 54} textAnchor="middle" fontSize={10.5} fill="var(--muted)" className="font-mono">
                  {mode === 'viterbi' && f.best[k] != null ? f2(f.best[k]!) : ''}
                </text>
              </g>
            ))}
            {TEXT.split('').map((ch, k) => (
              <text key={k} x={(xs(k) + xs(k + 1)) / 2} y={base - 6} textAnchor="middle" fontSize={15} fontWeight={600} fill="var(--fg)">
                {ch}
              </text>
            ))}
          </svg>
        </div>
        <div className="text-xs text-muted">节点下方数字是 best[i]（前 i 个字的最大 log 概率）；上方弧为多字 piece，下方弧为单字 piece。</div>
      </div>
    </VizFrame>
  )
}
