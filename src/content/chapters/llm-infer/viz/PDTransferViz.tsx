/**
 * PD 分离的代价：prefill 节点算完的 KV 要传给 decode 节点。
 * 比较「KV 传输时间」与「prefill 计算时间下界」（8 卡 H100 张量并行、100% 算力，均为下界）。
 */
import { useState } from 'react'
import { VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Stat } from './common'
import { MODELS, fmtMs, fmtNum, kvPerToken, prefillCost } from './hw'

const NETS = [
  { v: 100, label: '100 Gb/s' },
  { v: 400, label: '400 Gb/s' },
  { v: 3200, label: '8×400 Gb/s' },
]

export function PDTransferViz() {
  const [mid, setMid] = useState('70b')
  const [pe, setPe] = useState(13) // P = 2^pe
  const [net, setNet] = useState(400)
  const [kvb, setKvb] = useState(2)
  const m = MODELS.find((x) => x.id === mid)!
  const P = 2 ** pe
  const kvBytes = P * kvPerToken(m, kvb)
  const tx = kvBytes / ((net * 1e9) / 8)
  const pf = prefillCost(m, P).flops / (8 * 989e12)
  const ratio = tx / pf
  const w = Math.max(tx, pf)

  return (
    <VizFrame
      title="PD 分离：KV 搬运要多久"
      caption={
        <>
          {m.name}，prompt {fmtNum(P, 0)} token：KV 共 {fmtNum(kvBytes, 2)}B，经 {NETS.find((x) => x.v === net)!.label} 传输约 <b>{fmtMs(tx)}</b>；
          同一个 prompt 在 8×H100 上 prefill 至少要 <b>{fmtMs(pf)}</b>。传输 / 计算 = {ratio.toFixed(2)}。
          {ratio < 1
            ? ' 传输比计算快，按层流水（算完一层就发一层）基本能把传输藏在 prefill 后面。'
            : ' 传输比计算还慢：需要更快的网络、更小的 KV（GQA / MLA / FP8 KV），或把 prefill 与 decode 放在高带宽域内。'}
          prompt 越长，prefill 的 P² 注意力项让计算增长得比传输更快。
        </>
      }
      controls={
        <>
          <Segmented size="sm" value={mid} onChange={setMid} options={MODELS.map((x) => ({ value: x.id, label: x.name.replace('Llama-3.1-', '') }))} />
          <Segmented size="sm" value={net} onChange={setNet} options={NETS.map((x) => ({ value: x.v, label: x.label }))} />
          <Segmented size="sm" value={kvb} onChange={setKvb} options={[{ value: 2, label: 'BF16 KV' }, { value: 1, label: 'FP8 KV' }]} />
          <Slider className="min-w-[200px] flex-1" label="prompt 长度" value={pe} min={10} max={17} onChange={setPe} format={(v) => fmtNum(2 ** v, 0)} />
        </>
      }
    >
      <div className="space-y-3">
        {[
          { label: 'KV 传输', v: tx, tone: 'warning' as const },
          { label: 'prefill 计算（下界）', v: pf, tone: 'accent' as const },
        ].map((r) => (
          <div key={r.label} className="grid items-center gap-2" style={{ gridTemplateColumns: '7.5rem minmax(0,1fr)' }}>
            <span className="text-[12.5px] text-fg">{r.label}</span>
            <div className="relative h-7 rounded-md bg-surface-2">
              <div className="absolute inset-y-0 left-0 rounded-md transition-[width] duration-300" style={{ width: `${(r.v / w) * 100}%`, background: toneColor(r.tone) }} />
              <span className="absolute inset-y-0 right-2 flex items-center font-mono text-[11.5px] text-fg">{fmtMs(r.v)}</span>
            </div>
          </div>
        ))}
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="每 token KV" value={`${(kvPerToken(m, kvb) / 1024).toFixed(0)} KiB`} sub={`${m.layers} 层 · ${m.kvHeads} 个 KV 头`} />
          <Stat label="KV 总量" value={`${fmtNum(kvBytes, 2)}B`} />
          <Stat label="传输 / 计算" value={ratio.toFixed(2)} tone={ratio < 1 ? 'success' : 'danger'} />
          <Stat label="网络带宽" value={`${fmtNum((net * 1e9) / 8, 0)}B/s`} />
        </div>
        <div className="text-[11.5px] text-subtle">prefill 下界 = (2NP + 2Ln_hd_hP²) / (8 × 989 TFLOPS)，实际算力利用率远低于 100%，所以真实的「传输 / 计算」比这里更小。网络带宽为示意档位。</div>
      </div>
    </VizFrame>
  )
}
