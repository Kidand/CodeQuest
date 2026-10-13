/**
 * LFU 缓存（LC460）：频次桶 + min_freq 逐帧演示。
 * 洞见：每个频次一条“小 LRU 链表”；访问只会让 key 从桶 f 搬到桶 f+1，所以 min_freq 只可能 +1 或被新 key 重置为 1——全部 O(1)。
 */
import { useMemo } from 'react'
import { motion } from 'motion/react'
import { Legend, StepControls, VizFrame, catColor, toneColor, toneSoft, useStepper, type VizTone } from '@/viz'
import { C, Stat, Sub } from './common'

type Op = { kind: 'get'; key: number } | { kind: 'put'; key: number; val: number }

const OPS: Op[] = [
  { kind: 'put', key: 1, val: 1 },
  { kind: 'put', key: 2, val: 2 },
  { kind: 'get', key: 1 },
  { kind: 'put', key: 3, val: 3 },
  { kind: 'get', key: 2 },
  { kind: 'get', key: 3 },
  { kind: 'put', key: 4, val: 4 },
  { kind: 'get', key: 1 },
  { kind: 'get', key: 3 },
  { kind: 'get', key: 4 },
]
const CAP = 2

interface Frame {
  op: number
  buckets: [number, number[]][] // freq → keys（最近使用在前）
  vals: Record<number, number>
  minFreq: number
  hl?: { key: number; tone: VizTone }
  results: (number | null)[]
  caption: React.ReactNode
}

const opText = (o: Op) => (o.kind === 'get' ? `get(${o.key})` : `put(${o.key},${o.val})`)

function build(): Frame[] {
  const freq: Record<number, number> = {}
  const vals: Record<number, number> = {}
  const buckets = new Map<number, number[]>()
  let minFreq = 0
  const results: (number | null)[] = OPS.map(() => null)
  const frames: Frame[] = []
  const push = (op: number, caption: React.ReactNode, hl?: Frame['hl']) =>
    frames.push({
      op,
      buckets: [...buckets.entries()].filter(([, ks]) => ks.length).sort((a, b) => a[0] - b[0]).map(([f, ks]) => [f, [...ks]]),
      vals: { ...vals },
      minFreq,
      hl,
      results: [...results],
      caption,
    })
  const touch = (k: number) => {
    const f = freq[k]
    const b = buckets.get(f)!
    b.splice(b.indexOf(k), 1)
    let note = ''
    if (b.length === 0) {
      buckets.delete(f)
      if (minFreq === f) {
        minFreq = f + 1
        note = `桶 ${f} 空了且它正是 min_freq，于是 min_freq → ${f + 1}。`
      }
    }
    freq[k] = f + 1
    buckets.set(f + 1, [k, ...(buckets.get(f + 1) ?? [])])
    return note
  }
  push(-1, <>容量 {CAP}。每个频次一个桶，桶内按<b>最近使用</b>排序（左新右旧）；淘汰时去 <C>min_freq</C> 桶里取<b>最右</b>的 key。</>)
  OPS.forEach((o, i) => {
    const has = o.key in vals
    if (o.kind === 'get') {
      if (!has) {
        results[i] = -1
        push(i, <><C>get({o.key})</C>：不存在，返回 -1。</>)
        return
      }
      push(i, <><C>get({o.key})</C>：命中，当前频次 {freq[o.key]}。</>, { key: o.key, tone: 'accent' })
      const note = touch(o.key)
      results[i] = vals[o.key]
      push(i, <>key {o.key} 从桶 {freq[o.key] - 1} 搬到桶 {freq[o.key]} 的最左端。{note}</>, { key: o.key, tone: 'success' })
      return
    }
    if (has) {
      vals[o.key] = o.val
      const note = touch(o.key)
      push(i, <><C>put({o.key},{o.val})</C>：已存在，更新值并算一次使用，频次 → {freq[o.key]}。{note}</>, { key: o.key, tone: 'success' })
      return
    }
    if (Object.keys(vals).length === CAP) {
      const b = buckets.get(minFreq)!
      const victim = b[b.length - 1]
      push(
        i,
        <><C>put({o.key},{o.val})</C>：已满。去 <C>min_freq = {minFreq}</C> 的桶，取最久未用的 key {victim}{b.length > 1 ? '（同频次时按 LRU 打破平局）' : ''}。</>,
        { key: victim, tone: 'danger' },
      )
      b.pop()
      if (b.length === 0) buckets.delete(minFreq)
      delete vals[victim]
      delete freq[victim]
      push(i, <>淘汰 key {victim}。注意这里<b>不用</b>重新计算 min_freq——马上插入的新 key 会把它置为 1。</>)
    }
    vals[o.key] = o.val
    freq[o.key] = 1
    buckets.set(1, [o.key, ...(buckets.get(1) ?? [])])
    minFreq = 1
    push(i, <>新 key {o.key} 进入桶 1，<C>min_freq = 1</C>（新 key 的频次一定是最小的）。</>, { key: o.key, tone: 'success' })
  })
  return frames
}

export function LfuViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 1300 })
  const f = st.frame
  return (
    <VizFrame
      title="LFU：频次桶 + min_freq"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '正在访问', tone: 'accent', variant: 'soft' },
            { label: '搬桶 / 新插入', tone: 'success', variant: 'soft' },
            { label: '将被淘汰', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="flex min-w-0 flex-col gap-4">
        <div>
          <Sub>操作序列（LC460 示例，容量 2）</Sub>
          <div className="flex flex-wrap gap-1.5">
            {OPS.map((o, i) => (
              <span
                key={i}
                className="rounded-md border px-1.5 py-0.5 font-mono text-[11.5px]"
                style={{
                  borderColor: i === f.op ? 'var(--accent)' : 'var(--border)',
                  background: i === f.op ? toneSoft('accent', 18) : 'var(--surface)',
                  opacity: i > f.op ? 0.55 : 1,
                }}
              >
                {opText(o)}
                {f.results[i] != null && <b className={f.results[i] === -1 ? 'text-danger' : 'text-success'}> →{f.results[i]}</b>}
              </span>
            ))}
          </div>
        </div>
        <div className="flex flex-col gap-2">
          {f.buckets.length === 0 && <div className="text-xs text-subtle">（缓存为空）</div>}
          {f.buckets.map(([fq, ks]) => (
            <div key={fq} className="flex items-center gap-3">
              <div className="w-20 shrink-0 font-mono text-[12px]">
                <div className="text-fg">freq = {fq}</div>
                {fq === f.minFreq && <div className="text-[10.5px] font-semibold text-warning">◀ min_freq</div>}
              </div>
              <div className="flex min-h-[44px] flex-1 items-center gap-1.5 rounded-lg border border-dashed border-border px-2 py-1">
                {ks.map((k) => {
                  const tone = f.hl?.key === k ? f.hl.tone : undefined
                  return (
                    <motion.div
                      key={k}
                      layoutId={`lfu-${k}`}
                      transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                      className="flex h-9 w-11 flex-col items-center justify-center rounded-lg border-2 font-mono leading-none"
                      style={{ borderColor: tone ? toneColor(tone) : catColor(k - 1), background: tone ? toneSoft(tone, 26) : 'var(--surface)' }}
                    >
                      <span className="text-[12.5px] font-semibold text-fg">{k}</span>
                      <span className="mt-0.5 text-[10px] text-muted">={f.vals[k]}</span>
                    </motion.div>
                  )
                })}
                <span className="ml-auto text-[10.5px] text-subtle">← 新 · 旧 →</span>
              </div>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          <Stat label="min_freq" value={f.minFreq || '—'} tone="warning" />
          <Stat label="已用" value={`${Object.keys(f.vals).length}/${CAP}`} />
        </div>
      </div>
    </VizFrame>
  )
}
