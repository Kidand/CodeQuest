/**
 * 股票问题 = 状态机 DP。
 *  - StockStateViz：状态转移图 + 按天填表（122 无限次 / 309 冷冻期 / 714 手续费）
 *  - StockKViz：至多 k 笔交易（123 / 188），2k 个状态逐天推进
 */
import { useMemo, useState } from 'react'
import { CodeTrace, GraphView, Legend, StepControls, VizFrame, useStepper, type GraphEdge, type GraphNode } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { DpGrid, type Cell, type DpArrow } from './DpGrid'
import { C, Stat, Sub } from './common'

type Mode = '122' | '309' | '714'

interface Cfg {
  prices: number[]
  states: string[]
  labels: string[]
  fee: number
  init: (p0: number) => number[]
  /** 返回新值，以及每个状态的来源（-1 = 保持自己，其它 = 来源状态下标）与使用的转移名 */
  step: (prev: number[], p: number) => { val: number[]; src: number[] }
  answer: (v: number[]) => number
  code: string
  nodes: GraphNode[]
  edges: (GraphEdge & { key: string })[]
  /** 状态 s 从来源 t 转移时对应的边 key */
  edgeOf: (s: number, t: number) => string | undefined
}

const pick = (a: [number, number][]) => a.reduce((x, y) => (y[0] > x[0] ? y : x))

const CFG: Record<Mode, Cfg> = {
  '122': {
    prices: [7, 1, 5, 3, 6, 4],
    states: ['hold', 'cash'],
    labels: ['持有', '不持有'],
    fee: 0,
    init: (p0) => [-p0, 0],
    step: ([h, c], p) => {
      const H = pick([[h, -1], [c - p, 1]])
      const Cc = pick([[c, -1], [h + p, 0]])
      return { val: [H[0], Cc[0]], src: [H[1], Cc[1]] }
    },
    answer: (v) => v[1],
    code: `def max_profit(prices):              # LC 122：不限交易次数
    hold, cash = -prices[0], 0
    for p in prices[1:]:
        hold, cash = (max(hold, cash - p),  # 继续持有 / 今天买入
                      max(cash, hold + p))  # 继续空仓 / 今天卖出
    return cash`,
    nodes: [
      { id: 0, label: '持有', x: 70, y: 60, r: 26 },
      { id: 1, label: '不持有', x: 270, y: 60, r: 26 },
    ],
    edges: [
      { key: 'buy', from: 1, to: 0, label: '买 −p', directed: true, curve: 0.25 },
      { key: 'sell', from: 0, to: 1, label: '卖 +p', directed: true, curve: 0.25 },
    ],
    edgeOf: (s, t) => (s === 0 && t === 1 ? 'buy' : s === 1 && t === 0 ? 'sell' : undefined),
  },
  '309': {
    prices: [1, 2, 3, 0, 2],
    states: ['hold', 'sold', 'rest'],
    labels: ['持有', '刚卖出', '冷却/空仓'],
    fee: 0,
    init: (p0) => [-p0, 0, 0],
    step: ([h, s, r], p) => {
      const H = pick([[h, -1], [r - p, 2]])
      const S: [number, number] = [h + p, 0]
      const R = pick([[r, -1], [s, 1]])
      return { val: [H[0], S[0], R[0]], src: [H[1], S[1], R[1]] }
    },
    answer: (v) => Math.max(v[1], v[2]),
    code: `def max_profit(prices):              # LC 309：卖出后冷冻一天
    hold, sold, rest = -prices[0], 0, 0
    for p in prices[1:]:
        hold, sold, rest = (max(hold, rest - p),   # 只能从「冷却/空仓」买入
                            hold + p,              # 今天卖出
                            max(rest, sold))       # 昨天卖的，今天冷冻
    return max(sold, rest)`,
    nodes: [
      { id: 0, label: '持有', x: 60, y: 50, r: 26 },
      { id: 1, label: '刚卖出', x: 280, y: 50, r: 26 },
      { id: 2, label: '冷却', x: 170, y: 165, r: 26 },
    ],
    edges: [
      { key: 'sell', from: 0, to: 1, label: '卖 +p', directed: true },
      { key: 'cool', from: 1, to: 2, label: '冷冻一天', directed: true },
      { key: 'buy', from: 2, to: 0, label: '买 −p', directed: true },
    ],
    edgeOf: (s, t) => (s === 1 && t === 0 ? 'sell' : s === 2 && t === 1 ? 'cool' : s === 0 && t === 2 ? 'buy' : undefined),
  },
  '714': {
    prices: [1, 3, 2, 8, 4, 9],
    states: ['hold', 'cash'],
    labels: ['持有', '不持有'],
    fee: 2,
    init: (p0) => [-p0, 0],
    step: ([h, c], p) => {
      const H = pick([[h, -1], [c - p, 1]])
      const Cc = pick([[c, -1], [h + p - 2, 0]])
      return { val: [H[0], Cc[0]], src: [H[1], Cc[1]] }
    },
    answer: (v) => v[1],
    code: `def max_profit(prices, fee):         # LC 714：每笔交易付手续费
    hold, cash = -prices[0], 0
    for p in prices[1:]:
        hold, cash = (max(hold, cash - p),
                      max(cash, hold + p - fee))   # 卖出时扣手续费
    return cash`,
    nodes: [
      { id: 0, label: '持有', x: 70, y: 60, r: 26 },
      { id: 1, label: '不持有', x: 270, y: 60, r: 26 },
    ],
    edges: [
      { key: 'buy', from: 1, to: 0, label: '买 −p', directed: true, curve: 0.25 },
      { key: 'sell', from: 0, to: 1, label: '卖 +p−fee', directed: true, curve: 0.25 },
    ],
    edgeOf: (s, t) => (s === 0 && t === 1 ? 'buy' : s === 1 && t === 0 ? 'sell' : undefined),
  },
}

interface SFrame {
  day: number
  cols: (number | null)[][] // [state][day]
  src: number[]
  caption: React.ReactNode
}

function buildState(mode: Mode): SFrame[] {
  const cfg = CFG[mode]
  const P = cfg.prices
  const ns = cfg.states.length
  const cols: (number | null)[][] = Array.from({ length: ns }, () => Array(P.length).fill(null))
  let cur = cfg.init(P[0])
  cur.forEach((v, s) => (cols[s][0] = v))
  const fr: SFrame[] = [
    {
      day: 0,
      cols: cols.map((r) => [...r]),
      src: [],
      caption: <>第 0 天（价格 {P[0]}）：持有 = <C>−{P[0]}</C>（买入），其余状态为 0。</>,
    },
  ]
  for (let d = 1; d < P.length; d++) {
    const { val, src } = cfg.step(cur, P[d])
    val.forEach((v, s) => (cols[s][d] = v))
    const desc = val.map((v, s) => {
      const how = src[s] === -1 ? '保持' : cfg.edges.find((e) => e.key === cfg.edgeOf(s, src[s]))?.label ?? ''
      return (
        <span key={s}>
          {cfg.labels[s]} <C>{v}</C>（{how}）{s < ns - 1 ? '；' : ''}
        </span>
      )
    })
    fr.push({ day: d, cols: cols.map((r) => [...r]), src, caption: <>第 {d} 天（价格 {P[d]}）：{desc}</> })
    cur = val
  }
  fr.push({
    day: P.length - 1,
    cols: cols.map((r) => [...r]),
    src: [],
    caption: <>最大利润 = <C>{cfg.answer(cur)}</C>（最后一天手里不能还持有股票）。</>,
  })
  return fr
}

export function StockStateViz() {
  const [mode, setMode] = useState<Mode>('122')
  const cfg = CFG[mode]
  const frames = useMemo(() => buildState(mode), [mode])
  const st = useStepper(frames, { intervalMs: 1300 })
  const f = st.frame
  const used = new Set(f.src.map((t, s) => (t >= 0 ? cfg.edgeOf(s, t) : undefined)).filter(Boolean) as string[])
  const nodes = cfg.nodes.map((n) => ({ ...n, tone: f.src.length ? ('accent' as const) : undefined, sub: `${f.cols[n.id as number][f.day] ?? ''}` }))
  const edges = cfg.edges.map((e) => ({ ...e, tone: used.has(e.key) ? ('success' as const) : undefined }))
  const arrows: DpArrow[] = f.src.length
    ? f.src.map((t, s) => ({ from: [t === -1 ? s : t, f.day - 1] as [number, number], to: [s, f.day] as [number, number], tone: t === -1 ? 'muted' : 'success' }))
    : []
  return (
    <VizFrame
      title="股票问题 = 状态机：每天每个状态只从昨天的状态转移"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '今天发生的买 / 卖 / 冷冻', tone: 'success', variant: 'line' },
            { label: '保持昨天的状态', tone: 'muted', variant: 'line' },
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
              { value: '122', label: '122 不限次数' },
              { value: '309', label: '309 冷冻期' },
              { value: '714', label: '714 手续费 2' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <div className="grid items-center gap-4 @xl:grid-cols-[1fr_1.4fr]">
          <div>
            <Sub>状态转移图（节点下方是今天的最大利润）</Sub>
            <GraphView nodes={nodes} edges={edges} width={340} height={mode === '309' ? 205 : 120} nodeR={26} />
          </div>
          <div>
            <Sub>prices = [{cfg.prices.join(', ')}]</Sub>
            <DpGrid
              data={f.cols as Cell[][]}
              rowLabels={cfg.labels}
              colLabels={cfg.prices.map((p, d) => `${d}:${p}`)}
              corner="天:价"
              cellSize={46}
              arrows={arrows}
              ring={(_r, c) => (c === f.day && f.src.length ? 'accent' : undefined)}
            />
          </div>
        </div>
        <CodeTrace code={cfg.code} active={f.src.length ? [4, 5, 6].slice(0, mode === '309' ? 3 : 2) : f.day === 0 && !st.atEnd ? 2 : mode === '309' ? 7 : 6} />
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 至多 k 笔交易                                                        */
/* ------------------------------------------------------------------ */

const KP = [3, 3, 5, 0, 0, 3, 1, 4]

interface KFrame {
  day: number
  vals: (number | null)[][] // [2k][day]
  src: number[] // 每行的来源行（-1 = 保持）
  caption: React.ReactNode
}

function buildK(k: number): KFrame[] {
  const rows = 2 * k
  const vals: (number | null)[][] = Array.from({ length: rows }, () => Array(KP.length).fill(null))
  let cur = Array.from({ length: rows }, (_, r) => (r % 2 === 0 ? -KP[0] : 0))
  cur.forEach((v, r) => (vals[r][0] = v))
  const fr: KFrame[] = [{ day: 0, vals: vals.map((r) => [...r]), src: [], caption: <>第 0 天：所有 buy_j = <C>−{KP[0]}</C>，所有 sell_j = 0。</> }]
  for (let d = 1; d < KP.length; d++) {
    const p = KP[d]
    const nxt: number[] = []
    const src: number[] = []
    for (let r = 0; r < rows; r++) {
      const j = r >> 1
      if (r % 2 === 0) {
        const fromSell = j === 0 ? -p : cur[r - 1] - p // buy_j ← sell_{j-1} − p
        if (fromSell > cur[r]) {
          nxt.push(fromSell)
          src.push(j === 0 ? -2 : r - 1)
        } else {
          nxt.push(cur[r])
          src.push(-1)
        }
      } else {
        const fromBuy = cur[r - 1] + p // sell_j ← buy_j + p
        if (fromBuy > cur[r]) {
          nxt.push(fromBuy)
          src.push(r - 1)
        } else {
          nxt.push(cur[r])
          src.push(-1)
        }
      }
    }
    nxt.forEach((v, r) => (vals[r][d] = v))
    const changed = src.map((s, r) => (s !== -1 ? r : -1)).filter((r) => r >= 0)
    fr.push({
      day: d,
      vals: vals.map((r) => [...r]),
      src,
      caption: (
        <>
          第 {d} 天（价格 {p}）：
          {changed.length ? changed.map((r) => `${r % 2 ? 'sell' : 'buy'}${(r >> 1) + 1}=${nxt[r]}`).join('，') + ' 被更新' : '所有状态保持不变'}。
        </>
      ),
    })
    cur = nxt
  }
  fr.push({
    day: KP.length - 1,
    vals: vals.map((r) => [...r]),
    src: [],
    caption: <>至多 {k} 笔交易的最大利润 = <C>sell{k} = {cur[rows - 1]}</C>（sell_j 关于 j 单调不减，最后一行就是答案）。</>,
  })
  return fr
}

export function StockKViz() {
  const [k, setK] = useState(2)
  const frames = useMemo(() => buildK(k), [k])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame
  const labels = Array.from({ length: 2 * k }, (_, r) => `${r % 2 ? 'sell' : 'buy'}${(r >> 1) + 1}`)
  const arrows: DpArrow[] = f.src.flatMap((s, r) =>
    s >= 0 ? [{ from: [s, f.day - 1] as [number, number], to: [r, f.day] as [number, number], tone: r % 2 ? ('success' as const) : ('warning' as const) }] : [],
  )
  return (
    <VizFrame
      title="至多 k 笔交易：buy_j ← sell_{j−1} − p，sell_j ← buy_j + p"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '买入：从上一笔的 sell 转移', tone: 'warning', variant: 'line' },
            { label: '卖出：从同一笔的 buy 转移', tone: 'success', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="w-40" label="交易次数上限 k" value={k} min={1} max={3} onChange={setK} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Stat label="prices" value={`[${KP.join(', ')}]`} />
          <Stat label={`k=${k} 的答案`} value={frames[frames.length - 1].vals[2 * k - 1][KP.length - 1] ?? '—'} tone="success" />
        </div>
        <DpGrid
          data={f.vals as Cell[][]}
          rowLabels={labels}
          colLabels={KP.map((p, d) => `${d}:${p}`)}
          corner="天:价"
          cellSize={44}
          arrows={arrows}
          ring={(_r, c) => (c === f.day && f.src.length ? 'accent' : undefined)}
        />
        <CodeTrace
          code={`def max_profit(k, prices):
    buy = [-prices[0]] * (k + 1)     # buy[j]: 第 j 笔买入后手里的钱
    sell = [0] * (k + 1)             # sell[j]: 第 j 笔卖出后手里的钱（sell[0] = 0）
    for p in prices[1:]:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j - 1] - p)
            sell[j] = max(sell[j], buy[j] + p)
    return sell[k]`}
          active={f.src.length ? [6, 7] : st.atEnd ? 8 : [2, 3]}
        />
      </div>
    </VizFrame>
  )
}
