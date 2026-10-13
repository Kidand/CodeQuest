/**
 * KV 显存碎片：同一批请求、同一块 96 槽位的显存，对比两种分配方式随时间的演化。
 *  - 连续预分配：每个请求进来时按 prompt + max_new_tokens 预留一段连续空间（first-fit）
 *  - 分页：按块（block_size 个 token）按需分配，物理块可以不连续
 * 每个槽位存 1 个 token 的 K/V。逐帧 = 逐个解码迭代。
 */
import { useMemo, useState } from 'react'
import { Legend, StepControls, VizFrame, catColor, toneColor, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { shuffle } from '@/lib/utils'

const COLS = 16
const POOL = 96

interface Req {
  id: string
  prompt: number
  maxNew: number
  out: number
  arrive: number
}
const REQS: Req[] = [
  { id: 'R1', prompt: 8, maxNew: 24, out: 10, arrive: 0 },
  { id: 'R2', prompt: 4, maxNew: 28, out: 20, arrive: 0 },
  { id: 'R3', prompt: 12, maxNew: 12, out: 6, arrive: 0 },
  { id: 'R4', prompt: 6, maxNew: 18, out: 14, arrive: 2 },
  { id: 'R5', prompt: 10, maxNew: 14, out: 8, arrive: 4 },
  { id: 'R6', prompt: 5, maxNew: 27, out: 12, arrive: 6 },
]

type Kind = 'used' | 'future' | 'never' | 'slack' | 'ext' | 'free'
interface Cell {
  kind: Kind
  req?: number
}
interface Snap {
  cells: Cell[]
  running: number[]
  waiting: number[]
  done: number[]
  counts: Record<Kind, number>
}

interface Run {
  r: number
  gen: number
  start?: number // 连续：起始槽位
  blocks?: number[] // 分页：物理块
}

export function simulate(mode: 'contig' | 'paged', bs: number, T: number): Snap[] {
  const snaps: Snap[] = []
  let running: Run[] = []
  const waiting: number[] = []
  const done: number[] = []
  // 分页模式的空闲块：用固定种子打乱，体现「物理不连续」
  const nb = POOL / bs
  const freeBlocks = shuffle(Array.from({ length: nb }, (_, i) => i), 11)
  const lenOf = (x: Run) => REQS[x.r].prompt + x.gen

  const contigHoles = () => {
    const occ = Array(POOL).fill(false)
    running.forEach((x) => {
      const need = REQS[x.r].prompt + REQS[x.r].maxNew
      for (let i = 0; i < need; i++) occ[x.start! + i] = true
    })
    const holes: [number, number][] = []
    let i = 0
    while (i < POOL) {
      if (occ[i]) {
        i++
        continue
      }
      let j = i
      while (j < POOL && !occ[j]) j++
      holes.push([i, j - i])
      i = j
    }
    return holes
  }

  for (let t = 0; t <= T; t++) {
    // 1) 已在运行的请求各生成 1 个 token；分页模式按需申请新块
    if (t > 0) {
      for (const x of running) {
        x.gen++
        if (mode === 'paged' && lenOf(x) > x.blocks!.length * bs && freeBlocks.length) x.blocks!.push(freeBlocks.shift()!)
      }
      // 2) 生成完的请求释放显存
      const fin = running.filter((x) => x.gen >= REQS[x.r].out)
      fin.forEach((x) => {
        done.push(x.r)
        if (mode === 'paged') freeBlocks.push(...x.blocks!)
      })
      running = running.filter((x) => x.gen < REQS[x.r].out)
    }
    // 3) 新到达的请求排队，FIFO 接纳（队头放不下就整体等待）
    REQS.forEach((q, i) => q.arrive === t && waiting.push(i))
    while (waiting.length) {
      const q = REQS[waiting[0]]
      if (mode === 'contig') {
        const need = q.prompt + q.maxNew
        const hole = contigHoles().find(([, len]) => len >= need)
        if (!hole) break
        running.push({ r: waiting.shift()!, gen: 0, start: hole[0] })
      } else {
        const need = Math.ceil(q.prompt / bs)
        if (freeBlocks.length < need) break
        running.push({ r: waiting.shift()!, gen: 0, blocks: freeBlocks.splice(0, need) })
      }
    }
    // 4) 画出当前显存
    const cells: Cell[] = Array.from({ length: POOL }, () => ({ kind: 'free' as Kind }))
    if (mode === 'contig') {
      for (const x of running) {
        const q = REQS[x.r]
        const len = lenOf(x)
        for (let i = 0; i < q.prompt + q.maxNew; i++)
          cells[x.start! + i] = { req: x.r, kind: i < len ? 'used' : i < q.prompt + q.out ? 'future' : 'never' }
      }
      if (waiting.length) {
        const need = REQS[waiting[0]].prompt + REQS[waiting[0]].maxNew
        contigHoles().forEach(([s, len]) => {
          if (len < need) for (let i = s; i < s + len; i++) cells[i] = { kind: 'ext' }
        })
      }
    } else {
      for (const x of running) {
        const len = lenOf(x)
        x.blocks!.forEach((b, li) => {
          for (let o = 0; o < bs; o++) cells[b * bs + o] = { req: x.r, kind: li * bs + o < len ? 'used' : 'slack' }
        })
      }
    }
    const counts = { used: 0, future: 0, never: 0, slack: 0, ext: 0, free: 0 } as Record<Kind, number>
    cells.forEach((c) => counts[c.kind]++)
    snaps.push({ cells, running: running.map((x) => x.r), waiting: [...waiting], done: [...done], counts })
  }
  return snaps
}

const T_END = 30

function hatch(color: string) {
  return `repeating-linear-gradient(135deg, transparent 0 3px, color-mix(in oklab, ${color} 55%, transparent) 3px 4.5px)`
}

function cellStyle(c: Cell): React.CSSProperties {
  const col = c.req != null ? catColor(c.req) : 'var(--subtle)'
  switch (c.kind) {
    case 'used':
      return { background: `color-mix(in oklab, ${col} 78%, var(--surface))`, borderColor: col }
    case 'future':
      return { background: `color-mix(in oklab, ${col} 38%, var(--surface))`, borderColor: `color-mix(in oklab, ${col} 70%, var(--surface))` }
    case 'never':
      return { background: `color-mix(in oklab, ${col} 8%, var(--surface))`, backgroundImage: hatch(col), borderColor: `color-mix(in oklab, ${col} 45%, var(--surface))` }
    case 'slack':
      return { background: `color-mix(in oklab, ${col} 10%, var(--surface))`, backgroundImage: hatch(col), borderColor: col, borderStyle: 'dashed' }
    case 'ext':
      return { background: 'var(--surface)', backgroundImage: hatch(toneColor('danger')), borderColor: toneColor('danger') }
    default:
      return { background: 'var(--surface-2)', borderColor: 'var(--border)' }
  }
}

function Pool({ snap, title, mode }: { snap: Snap; title: string; mode: 'contig' | 'paged' }) {
  const c = snap.counts
  const alloc = POOL - c.free - c.ext
  return (
    <div className="min-w-0 rounded-xl border border-border bg-surface p-3">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <div className="text-[13px] font-semibold text-fg">{title}</div>
        <div className="font-mono text-[11px] text-muted">
          运行 {snap.running.length} · 排队 {snap.waiting.length} · 完成 {snap.done.length}
        </div>
      </div>
      <div className="mx-auto grid max-w-[460px] gap-[2px]" style={{ gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))` }}>
        {snap.cells.map((cell, i) => (
          <div key={i} className="aspect-square rounded-[3px] border transition-colors duration-300" style={cellStyle(cell)} />
        ))}
      </div>
      <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 font-mono text-[11.5px] text-muted">
        <span>
          真正存 token：<b className="text-fg">{c.used}</b>
        </span>
        <span>
          已分配的利用率：<b className={alloc && c.used / alloc < 0.6 ? 'text-danger' : 'text-success'}>{alloc ? Math.round((c.used / alloc) * 100) : 0}%</b>
        </span>
        {mode === 'contig' ? (
          <>
            <span>预留（将来会用）：{c.future}</span>
            <span>内部碎片（永不用）：{c.never}</span>
            <span className={c.ext ? 'text-danger' : ''}>外部碎片：{c.ext}</span>
          </>
        ) : (
          <span>最后一块的空位：{c.slack}</span>
        )}
        <span>空闲：{c.free}</span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1 text-[11px]">
        {snap.waiting.map((w) => (
          <span key={w} className="rounded border border-dashed px-1.5 font-mono" style={{ borderColor: catColor(w), color: catColor(w) }}>
            {REQS[w].id} 等待
          </span>
        ))}
      </div>
    </div>
  )
}

export function FragViz() {
  const [bs, setBs] = useState(4)
  const frames = useMemo(() => {
    const a = simulate('contig', bs, T_END)
    const b = simulate('paged', bs, T_END)
    const last = Math.max(a.findIndex((s) => s.done.length === REQS.length), b.findIndex((s) => s.done.length === REQS.length))
    return a.slice(0, (last < 0 ? T_END : last) + 1).map((s, t) => ({ t, a: s, b: b[t] }))
  }, [bs])
  const st = useStepper(frames, { intervalMs: 800 })
  const f = st.frame
  const ca = f.a.counts
  return (
    <VizFrame
      title="连续预分配 vs 分页：显存碎片"
      caption={
        <>
          第 {f.t} 次迭代。连续预分配：已分配的 {POOL - ca.free - ca.ext} 个槽位里只有 {ca.used} 个真正存了 token
          {ca.never > 0 && <>，{ca.never} 个槽位因为请求提前结束而<b>永远用不到</b></>}
          {ca.ext > 0 && (
            <>
              ；还有 {ca.ext} 个空闲槽位因为<b>不连续</b>，放不下队头请求（外部碎片）
            </>
          )}
          。分页：同一时刻运行 {f.b.running.length} 个请求，每个请求最多浪费最后一块里的 {bs - 1} 个槽位。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '已存 token', tone: 1, variant: 'solid' },
            { label: '预留（将来会用）', tone: 1, variant: 'soft' },
            { label: '内部碎片 / 块内空位（斜线）', tone: 'muted', variant: 'soft' },
            { label: '外部碎片', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <div className="flex items-center gap-2 text-[13px] text-muted">
            block_size
            <Segmented size="sm" value={bs} onChange={setBs} options={[4, 8, 16].map((v) => ({ value: v, label: String(v) }))} />
          </div>
        </>
      }
    >
      <div className="grid gap-3 @3xl:grid-cols-2">
        <Pool snap={f.a} mode="contig" title="① 连续预分配（prompt + max_new_tokens）" />
        <Pool snap={f.b} mode="paged" title={`② 分页（block_size = ${bs}）`} />
      </div>
      <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] text-muted">
        <span>请求（prompt + max_new → 实际输出 @到达步）：</span>
        {REQS.map((q, i) => (
          <span key={q.id} style={{ color: catColor(i) }}>
            {q.id} {q.prompt}+{q.maxNew}→{q.out} @{q.arrive}
          </span>
        ))}
      </div>
    </VizFrame>
  )
}
