/**
 * 大模型岗场景：显存预算下求最大 batch。
 * fits(b) 关于 b 单调（T…T F…F），上界未知 ⇒ 先倍增找上界，再二分（Lightning BatchSizeFinder 的 binsearch 模式同理）。
 * 显存模型：Llama-3-8B，BF16 权重 + KV cache（32 层 × 8 个 KV 头 × head_dim 128 × K/V 2 份 × 2 字节 = 128 KiB / token）。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { cn } from '@/lib/utils'
import { C } from './common'

const GiB = 1024 ** 3
const WEIGHTS = 8.03e9 * 2 // 字节，≈ 14.96 GiB
const KV_PER_TOKEN = 2 * 32 * 8 * 128 * 2 // 131072 B = 128 KiB
const UTIL = 0.9

const CODE = `def max_batch(fits):          # fits(b)：真实跑一步，看会不会 OOM
    if not fits(1):
        return 0
    b = 1
    while fits(b * 2):        # ① 倍增：找到第一个失败的上界
        b *= 2
    lo, hi = b + 1, b * 2     # 第一个失败点 ∈ [b+1, 2b]
    while lo < hi:            # ② 二分：找第一个 fits 为 False 的 b
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid + 1
        else:
            hi = mid
    return lo - 1`

interface Probe {
  b: number
  ok: boolean
  phase: 'grow' | 'bin'
}
interface Frame {
  probes: Probe[]
  cur?: Probe
  line: number
  caption: React.ReactNode
  answer?: number
}

function build(memGiB: number, seqLen: number): Frame[] {
  const budget = memGiB * GiB * UTIL
  const perSeq = seqLen * KV_PER_TOKEN
  const fits = (b: number) => WEIGHTS + b * perSeq <= budget
  const probes: Probe[] = []
  const fr: Frame[] = []
  const probe = (b: number, phase: Probe['phase']) => {
    const p = { b, ok: fits(b), phase }
    probes.push(p)
    return p
  }
  let p = probe(1, 'grow')
  fr.push({ probes: [...probes], cur: p, line: 2, caption: p.ok ? <>先试 <C>b = 1</C>：放得下。</> : <><C>b = 1</C> 都放不下——权重 + 一条序列的 KV cache 已超出预算。</> })
  if (!p.ok) {
    fr.push({ probes: [...probes], line: 3, answer: 0, caption: <>返回 0：需要更大的卡、更短的上下文，或量化 / 张量并行。</> })
    return fr
  }
  let b = 1
  for (;;) {
    if (b * 2 > 1 << 14) break
    p = probe(b * 2, 'grow')
    fr.push({ probes: [...probes], cur: p, line: 5, caption: p.ok ? <>倍增：<C>b = {b * 2}</C> 放得下，继续翻倍。</> : <>倍增：<C>b = {b * 2}</C> OOM！上界找到了，答案在 <C>[{b}, {b * 2})</C>。</> })
    if (!p.ok) break
    b *= 2
  }
  let lo = b + 1
  let hi = b * 2
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    p = probe(mid, 'bin')
    if (p.ok) {
      fr.push({ probes: [...probes], cur: p, line: 11, caption: <>二分：<C>fits({mid})</C> 为 True，第一个失败点在右边，<C>lo = {mid + 1}</C>。</> })
      lo = mid + 1
    } else {
      fr.push({ probes: [...probes], cur: p, line: 13, caption: <>二分：<C>fits({mid})</C> 为 False（OOM），<C>hi = {mid}</C>。</> })
      hi = mid
    }
  }
  const ans = lo - 1
  fr.push({
    probes: [...probes],
    line: 14,
    answer: ans,
    caption: (
      <>
        最大 batch = <C>{ans}</C>，共试跑 {probes.length} 次（≈ 2·log₂b*）。闭式解 <C>⌊(0.9·M − W) / (L · 128 KiB)⌋ = {Math.floor((budget - WEIGHTS) / perSeq)}</C>，一致。
      </>
    ),
  })
  return fr
}

const fmtG = (bytes: number) => `${(bytes / GiB).toFixed(1)} GiB`

export function BatchProbeViz() {
  const [mem, setMem] = useState(80)
  const [seqK, setSeqK] = useState(8)
  const seqLen = seqK * 1024
  const frames = useMemo(() => build(mem, seqLen), [mem, seqLen])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const total = mem * GiB
  const budget = total * UTIL
  const cur = f.cur ?? (f.answer ? { b: f.answer, ok: true, phase: 'bin' as const } : undefined)
  const kv = cur ? cur.b * seqLen * KV_PER_TOKEN : 0
  const scale = Math.max(total, WEIGHTS + kv) * 1.02
  const pct = (x: number) => `${(x / scale) * 100}%`
  const over = WEIGHTS + kv > budget

  return (
    <VizFrame
      title="显存预算下二分最大 batch：fits(b) 单调"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '权重（BF16 ≈ 15 GiB）', tone: 'info' },
            { label: 'KV cache = b × L × 128 KiB', tone: 'accent' },
            { label: '超出预算', tone: 'danger' },
            { label: `预算 = ${UTIL} × 显存`, tone: 'warning', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mem}
            onChange={setMem}
            options={[
              { value: 24, label: '24 GiB' },
              { value: 48, label: '48 GiB' },
              { value: 80, label: '80 GiB' },
            ]}
          />
          <Slider className="min-w-[180px] flex-1" label="上下文长度 L" value={seqK} min={1} max={32} onChange={setSeqK} format={(v) => `${v}K tokens`} />
        </>
      }
    >
      <div className="space-y-5">
        <div>
          <div className="mb-1.5 text-xs text-muted">
            Llama-3-8B · 每个 token 的 KV cache = 2 × 32 层 × 8 KV 头 × 128 维 × 2 字节 = <span className="font-mono">128 KiB</span>
            ；当前试探 <span className="font-mono text-fg">b = {cur?.b ?? '—'}</span>
          </div>
          <div className="relative h-9 w-full overflow-hidden rounded-lg border border-border bg-surface-2">
            <div className="absolute inset-y-0 left-0 bg-info/70 transition-[width] duration-300" style={{ width: pct(WEIGHTS) }} />
            <div
              className={cn('absolute inset-y-0 transition-[width,left] duration-300', over ? 'bg-danger/70' : 'bg-accent/70')}
              style={{ left: pct(WEIGHTS), width: pct(kv) }}
            />
            <div className="absolute inset-y-0 w-0.5 bg-warning" style={{ left: pct(budget) }} />
            <div className="absolute inset-y-0 w-0.5 bg-fg/60" style={{ left: pct(total) }} />
          </div>
          <div className="mt-1 flex flex-wrap justify-between gap-x-3 font-mono text-[11px] text-muted">
            <span>已用 {fmtG(WEIGHTS + kv)}</span>
            <span className="text-warning">预算 {fmtG(budget)}</span>
            <span>显存 {mem} GiB</span>
          </div>
        </div>
        <div className="grid items-start gap-4 @3xl:grid-cols-2">
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">试探记录（① 倍增 → ② 二分）</div>
            <div className="flex flex-wrap gap-1.5">
              {f.probes.map((p, i) => (
                <span
                  key={i}
                  className={cn(
                    'rounded-md border px-1.5 py-0.5 font-mono text-[12px]',
                    p.ok ? 'border-success/60 bg-success-soft text-fg' : 'border-danger/60 bg-danger-soft text-fg',
                    p.phase === 'bin' && 'border-dashed',
                    i === f.probes.length - 1 && f.cur && 'ring-2 ring-accent',
                  )}
                >
                  {p.b}
                  {p.ok ? ' ✓' : ' ✗'}
                </span>
              ))}
            </div>
            {f.answer != null && (
              <div className="mt-3 text-sm">
                答案 <span className="font-mono font-semibold text-success">b* = {f.answer}</span>
              </div>
            )}
          </div>
          <CodeTrace code={CODE} active={f.line} maxHeight={300} />
        </div>
      </div>
    </VizFrame>
  )
}
