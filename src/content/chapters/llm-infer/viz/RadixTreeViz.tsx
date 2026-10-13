/**
 * SGLang RadixAttention：用 radix tree 管理跨请求共享的 KV。
 * 每个请求：① 最长前缀匹配（命中的 KV 直接复用，跳过这部分 prefill）② 把剩余 token 插入（必要时在边中间分裂）
 * ③ 超出容量时按 LRU 淘汰叶子（父节点变成叶子后才可能被淘汰；正在用的路径不淘汰）。
 */
import { useMemo, useState } from 'react'
import { Legend, StepControls, VizFrame, toneColor, useStepper, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { C } from './common'

const SYS = '你是助手。'.split('')
const REQS: string[][] = [
  [...SYS, ...'讲个笑话'.split('')],
  [...SYS, ...'写首诗'.split('')],
  [...SYS, ...'讲个故事'.split('')],
  [...'你是程序员。'.split(''), ...'写代码'.split('')],
  [...SYS, ...'讲个笑话'.split('')],
  [...SYS, ...'写首词'.split('')],
]

interface RNode {
  id: number
  key: string[]
  children: RNode[]
  parent: RNode | null
  last: number
}
type Status = 'hit' | 'new' | 'split' | 'evict'
interface Row {
  id: number
  depth: number
  key: string
  last: number
  status?: Status
}
interface Frame {
  rows: Row[]
  total: number
  caption: React.ReactNode
  req: number
  hits: number
  seen: number
}

function common(a: string[], b: string[], off: number) {
  let k = 0
  while (k < a.length && off + k < b.length && a[k] === b[off + k]) k++
  return k
}

function build(capacity: number): Frame[] {
  let nid = 1
  const root: RNode = { id: 0, key: [], children: [], parent: null, last: 0 }
  const frames: Frame[] = []
  let hitsTotal = 0
  let seen = 0
  const total = () => {
    let s = 0
    const walk = (n: RNode) => n.children.forEach((c) => ((s += c.key.length), walk(c)))
    walk(root)
    return s
  }
  const snap = (marks: Map<number, Status>, extra: Row[] = []): Row[] => {
    const rows: Row[] = []
    const walk = (n: RNode, d: number) =>
      n.children.forEach((c) => {
        rows.push({ id: c.id, depth: d, key: c.key.join(''), last: c.last, status: marks.get(c.id) })
        walk(c, d + 1)
      })
    walk(root, 0)
    return [...rows, ...extra]
  }

  frames.push({ rows: [], total: 0, req: -1, hits: 0, seen: 0, caption: <>空的 radix tree。依次到来 {REQS.length} 个请求，容量上限 {capacity} 个 token。</> })

  REQS.forEach((toks, r) => {
    const t = r + 1
    // ① 匹配
    const marks = new Map<number, Status>()
    let node = root
    let i = 0
    let partial: { child: RNode; k: number } | null = null
    while (i < toks.length) {
      const child = node.children.find((c) => c.key[0] === toks[i])
      if (!child) break
      const k = common(child.key, toks, i)
      marks.set(child.id, 'hit')
      if (k < child.key.length) {
        partial = { child, k }
        i += k
        break
      }
      i += k
      node = child
    }
    const hit = i
    hitsTotal += hit
    seen += toks.length
    frames.push({
      rows: snap(marks),
      total: total(),
      req: r,
      hits: hitsTotal,
      seen,
      caption: (
        <>
          请求 {t}「{toks.join('')}」：最长前缀匹配命中 <b>{hit}</b> / {toks.length} 个 token
          {partial ? <>（在边 <C>{partial.child.key.join('')}</C> 中间第 {partial.k} 个 token 处分叉）</> : ''}。命中部分的 KV 直接复用，只需 prefill 剩下的 {toks.length - hit} 个。
        </>
      ),
    })
    // ② 插入
    const ins = new Map<number, Status>()
    node = root
    i = 0
    while (i < toks.length) {
      const child = node.children.find((c) => c.key[0] === toks[i])
      if (!child) {
        const leaf: RNode = { id: nid++, key: toks.slice(i), children: [], parent: node, last: t }
        node.children.push(leaf)
        ins.set(leaf.id, 'new')
        i = toks.length
        break
      }
      const k = common(child.key, toks, i)
      if (k < child.key.length) {
        // 分裂：child 变成 key[:k] 的中间节点，原来的剩余部分成为它的孩子
        const tail: RNode = { id: nid++, key: child.key.slice(k), children: child.children, parent: child, last: child.last }
        tail.children.forEach((g) => (g.parent = tail))
        child.key = child.key.slice(0, k)
        child.children = [tail]
        ins.set(child.id, 'split')
      } else ins.set(child.id, 'hit')
      child.last = t
      i += k
      node = child
    }
    frames.push({
      rows: snap(ins),
      total: total(),
      req: r,
      hits: hitsTotal,
      seen,
      caption:
        toks.length - hit > 0 ? (
          <>
            插入剩余的 {toks.length - hit} 个 token：{[...ins.values()].includes('split') ? '先把被走到一半的边分裂成两段（金色），' : ''}再挂一个新叶子（蓝色）。路径上所有节点的最近使用时间更新为 {t}。
          </>
        ) : (
          <>完全命中，无需插入，只把路径上节点的最近使用时间更新为 {t}。</>
        ),
    })
    // ③ 淘汰
    const evicted: Row[] = []
    while (total() > capacity) {
      const leaves: RNode[] = []
      const walk = (n: RNode) => n.children.forEach((c) => (c.children.length ? walk(c) : leaves.push(c)))
      walk(root)
      const cand = leaves.filter((l) => l.last < t).sort((a, b) => a.last - b.last || a.id - b.id)
      if (!cand.length) break
      const v = cand[0]
      let depth = 0
      for (let p = v.parent; p && p !== root; p = p.parent) depth++
      evicted.push({ id: v.id, depth, key: v.key.join(''), last: v.last, status: 'evict' })
      v.parent!.children = v.parent!.children.filter((c) => c !== v)
    }
    if (evicted.length)
      frames.push({
        rows: snap(new Map(), evicted),
        total: total(),
        req: r,
        hits: hitsTotal,
        seen,
        caption: (
          <>
            超出容量 {capacity}：按 LRU 淘汰最久未用的<b>叶子</b> {evicted.map((e) => `「${e.key}」`).join('、')}（红色）。内部节点只有在孩子全被淘汰、自己变成叶子后才会被淘汰——共享前缀活得最久。
          </>
        ),
      })
  })
  return frames
}

const STATUS_TONE: Record<Status, VizTone> = { hit: 'success', new: 'info', split: 'gold', evict: 'danger' }

export function RadixTreeViz() {
  const [cap, setCap] = useState(24)
  const frames = useMemo(() => build(cap), [cap])
  const st = useStepper(frames, { intervalMs: 1700 })
  const f = st.frame
  return (
    <VizFrame
      title="RadixAttention：前缀树 + LRU"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '命中的路径', tone: 'success', variant: 'soft' },
            { label: '新插入', tone: 'info', variant: 'soft' },
            { label: '分裂出的节点', tone: 'gold', variant: 'soft' },
            { label: '被淘汰', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[200px] flex-1" label="缓存容量（token）" value={cap} min={14} max={40} onChange={setCap} />
        </>
      }
    >
      <div className="grid items-start gap-4 @2xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="min-h-[180px] min-w-0 rounded-xl border border-border bg-surface p-3 font-mono text-[13px]">
          <div className="mb-1 text-[11px] text-subtle">根（空前缀）</div>
          {f.rows.length === 0 && <div className="text-subtle">（空）</div>}
          {f.rows.map((r) => {
            const tone = r.status ? STATUS_TONE[r.status] : undefined
            return (
              <div key={`${r.id}-${r.status ?? ''}`} className="flex items-center gap-2 py-0.5" style={{ paddingLeft: r.depth * 18 }}>
                <span className="text-subtle">└</span>
                <span
                  className="rounded-md border px-1.5 py-0.5"
                  style={{
                    borderColor: tone ? toneColor(tone) : 'var(--border-strong)',
                    background: tone ? `color-mix(in oklab, ${toneColor(tone)} 18%, var(--surface))` : 'var(--surface)',
                    textDecoration: r.status === 'evict' ? 'line-through' : undefined,
                    color: 'var(--fg)',
                  }}
                >
                  {r.key}
                </span>
                <span className="text-[11px] text-subtle">
                  {r.key.length} tok · t={r.last}
                </span>
              </div>
            )
          })}
        </div>
        <div className="min-w-0 space-y-2 text-[13px]">
          <div className="rounded-xl border border-border bg-surface p-3">
            <div className="text-[11px] text-muted">缓存 token 数 / 容量</div>
            <div className="font-mono text-[15px] font-semibold text-fg">
              {f.total} / {cap}
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded bg-surface-2">
              <div className="h-full rounded transition-[width] duration-300" style={{ width: `${Math.min(100, (f.total / cap) * 100)}%`, background: toneColor('accent') }} />
            </div>
          </div>
          <div className="rounded-xl border border-border bg-surface p-3">
            <div className="text-[11px] text-muted">累计命中率（复用的 prefill token 占比）</div>
            <div className="font-mono text-[15px] font-semibold text-success">{f.seen ? ((f.hits / f.seen) * 100).toFixed(0) : 0}%</div>
            <div className="text-[11px] text-subtle">
              {f.hits} / {f.seen} token
            </div>
          </div>
          <div className="space-y-0.5 font-mono text-[11.5px] text-muted">
            {REQS.map((q, i) => (
              <div key={i} className={i === f.req ? 'font-semibold text-fg' : ''}>
                {i === f.req ? '▶' : ' '} 请求 {i + 1}：{q.join('')}
              </div>
            ))}
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
