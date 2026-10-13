/**
 * chunked prefill：一个长 prompt 在第 2 次迭代到达，此时有 4 个请求正在 decode。
 * 迭代耗时用 roofline 估计（Llama-3.1-8B @ H100，忽略注意力与 KV 读取）：t = max(读权重, token 数 × 每 token 计算)。
 *  - 不切块：整段 prefill 和 decode 挤在同一次迭代里，这次迭代很长 → 所有 decode 请求出现一次 TPOT 尖峰
 *  - 切块：每次迭代的 token 预算 = decode token + 一块 prefill，decode 的间隔被限制住
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Slider } from '@/components/ui'
import { Stat } from './common'
import { fmtMs } from './hw'

const N = 8.03e9
const TW = (N * 2) / 3.35e12 // 读一遍权重
const C_TOK = (2 * N) / 989e12 // 每个 token 的计算时间
const DEC = 4
const ARRIVE = 2
const iterTime = (tokens: number) => Math.max(TW, tokens * C_TOK)

interface It {
  dec: number
  pre: number
  t: number
}

function simulate(P: number, budget: number | null) {
  const its: It[] = []
  let left = P
  let done = false
  let ttft = 0
  let tStart = 0
  let k = 0
  let after = 0
  while (after < 3 && k < 400) {
    const dec = DEC + (done ? 1 : 0)
    let pre = 0
    if (k >= ARRIVE && left > 0) {
      pre = budget == null ? left : Math.max(0, Math.min(left, budget - dec))
      left -= pre
    }
    const t = iterTime(dec + pre)
    if (k === ARRIVE) tStart = its.reduce((s, x) => s + x.t, 0)
    its.push({ dec, pre, t })
    if (k >= ARRIVE && !done && left === 0) {
      done = true
      ttft = its.reduce((s, x) => s + x.t, 0) - tStart
    }
    if (done) after++
    k++
  }
  const maxItl = Math.max(...its.map((x) => x.t))
  const nChunks = its.filter((x) => x.pre > 0).length
  return { its, ttft, maxItl, nChunks, total: its.reduce((s, x) => s + x.t, 0), tStart }
}

function Timeline({ title, sim, scale }: { title: string; sim: ReturnType<typeof simulate>; scale: number }) {
  let acc = 0
  return (
    <div className="min-w-0">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[12.5px] font-semibold text-fg">{title}</span>
        <span className="font-mono text-[11px] text-muted">
          TTFT {fmtMs(sim.ttft)} · 最大 decode 间隔 {fmtMs(sim.maxItl)}
        </span>
      </div>
      <div className="relative h-9 w-full rounded-md bg-surface-2">
        {sim.its.map((x, i) => {
          const left = (acc / scale) * 100
          acc += x.t
          const share = x.pre / (x.pre + x.dec)
          const tone = x.pre > 0 ? 'warning' : 'accent'
          return (
            <div
              key={i}
              className="absolute inset-y-0.5 flex items-center justify-center overflow-hidden rounded-sm border font-mono text-[10px] text-fg"
              style={{
                left: `${left}%`,
                width: `${(x.t / scale) * 100}%`,
                borderColor: 'var(--surface)',
                background: `color-mix(in oklab, ${toneColor(tone)} ${x.pre > 0 ? 35 + 45 * share : 35}%, var(--surface))`,
              }}
              title={`迭代 ${i}：decode ${x.dec} + prefill ${x.pre} token，${fmtMs(x.t)}`}
            >
              {x.pre > 0 && x.t / scale > 0.07 ? `+${x.pre}` : ''}
            </div>
          )
        })}
        <div className="absolute -inset-y-1 w-0.5" style={{ left: `${(sim.tStart / scale) * 100}%`, background: toneColor('danger') }} />
        <div className="absolute -inset-y-1 w-0.5" style={{ left: `${((sim.tStart + sim.ttft) / scale) * 100}%`, background: toneColor('success') }} />
      </div>
    </div>
  )
}

export function ChunkedPrefillViz() {
  const [P, setP] = useState(4000)
  const [be, setBe] = useState(9) // budget = 2^be
  const budget = 2 ** be
  const a = useMemo(() => simulate(P, null), [P])
  const b = useMemo(() => simulate(P, budget), [P, budget])
  const scale = Math.max(a.total, b.total)
  const free = Math.floor(TW / C_TOK)

  return (
    <VizFrame
      title="chunked prefill：把长 prefill 切块"
      caption={
        <>
          不切块时，{P} 个 token 的 prefill 和 4 个 decode 挤在一次迭代里，耗时 {fmtMs(a.maxItl)}——正在 decode 的请求这一步的间隔是平时（{fmtMs(TW)}）的 {(a.maxItl / TW).toFixed(0)} 倍。
          按每次迭代 {budget} 个 token 的预算切成 {b.nChunks} 块后，decode 最大间隔降到 {fmtMs(b.maxItl)}，TTFT 从 {fmtMs(a.ttft)} 变为 {fmtMs(b.ttft)}。
          预算 ≤ 约 {free} 个 token 时迭代仍是访存瓶颈，搭进来的 prefill 完全「免费」。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '只有 decode 的迭代', tone: 'accent', variant: 'soft' },
            { label: '含 prefill 的迭代（数字 = prefill token 数）', tone: 'warning', variant: 'soft' },
            { label: '新请求到达', tone: 'danger', variant: 'line' },
            { label: '首 token 产出', tone: 'success', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[180px] flex-1" label="新请求 prompt 长度" value={P} min={1000} max={16000} step={500} onChange={setP} />
          <Slider className="min-w-[180px] flex-1" label="每次迭代 token 预算" value={be} min={7} max={12} onChange={setBe} format={(v) => String(2 ** v)} />
        </>
      }
    >
      <div className="space-y-4">
        <Timeline title="① 不切块：整段 prefill" sim={a} scale={scale} />
        <Timeline title={`② chunked prefill（预算 ${budget}）`} sim={b} scale={scale} />
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="decode 最大间隔（不切块）" value={fmtMs(a.maxItl)} tone="danger" />
          <Stat label="decode 最大间隔（切块）" value={fmtMs(b.maxItl)} tone="success" />
          <Stat label="TTFT（不切块 / 切块）" value={`${fmtMs(a.ttft)} / ${fmtMs(b.ttft)}`} />
          <Stat label="免费搭车的 token 数" value={`≈ ${free - DEC}`} sub="迭代仍受带宽限制" />
        </div>
        <div className="text-[11.5px] text-subtle">
          模型：Llama-3.1-8B BF16 @ H100，迭代耗时 = max(读权重 {fmtMs(TW)}, token 数 × {(C_TOK * 1e6).toFixed(1)} µs)；忽略注意力、KV 读取与调度开销，长 prompt 的切块还要反复读前面块的 KV，实际 TTFT 会更差一些。
        </div>
      </div>
    </VizFrame>
  )
}
