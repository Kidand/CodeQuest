/**
 * decode 时注意力核心的「每个缓存 token、每层」读取字节与 FLOPs，以及算术强度 FLOPs/Byte（BF16）。
 * 吸收后的 MLA ≈ n_h 个 query 头共享一个 576 维 KV 头的 MQA，强度随 n_h 线性增长。
 */
import { useState } from 'react'
import { VizFrame, toneColor, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { fmtCount } from '@/lib/utils'

const DH = 128
const DC = 512
const DR = 64
const B = 2 // BF16
const RIDGE = 295 // H100 SXM：~989 TFLOPS(BF16 稠密) / 3.35 TB/s

export function MLAIntensityViz() {
  const [nh, setNh] = useState(128)
  const rows: { name: string; bytes: number; flops: number; tone: VizTone; note: string }[] = [
    { name: 'MHA', bytes: 2 * nh * DH * B, flops: 4 * nh * DH, tone: 'danger', note: '每个 KV 元素只服务 1 个头' },
    { name: 'GQA-8', bytes: 2 * Math.min(8, nh) * DH * B, flops: 4 * nh * DH, tone: 'warning', note: `一份 K/V 服务 ${nh / Math.min(8, nh)} 个头` },
    { name: 'MLA 展开解码', bytes: (DC + DR) * B, flops: 2 * DC * 2 * nh * DH + nh * (2 * (DH + DR) + 2 * DH), tone: 'muted', note: '每步把全部历史重新上投影：FLOPs 爆炸' },
    { name: 'MLA 吸收解码', bytes: (DC + DR) * B, flops: nh * 2 * (DC + DR + DC), tone: 'success', note: `≈ ${nh} 头共享一个 576 维 KV 头的 MQA` },
  ]
  const logMax = Math.log10(4e4)

  return (
    <VizFrame
      title="吸收后的 MLA 解码：算术强度随头数线性增长"
      caption={
        <>
          每个缓存 token、每层（BF16）：吸收形式只读 {(DC + DR) * B} 字节，却做 {fmtCount(rows[3].flops)} FLOPs，强度 ≈{' '}
          <b className="font-mono">{(rows[3].flops / rows[3].bytes).toFixed(0)}</b> FLOP/B；MHA 恒为 1。H100 SXM 的拐点约 {RIDGE} FLOP/B（约 989 TFLOPS BF16 稠密 ÷ 3.35 TB/s）：
          低于它受带宽限制。张量并行把头切到多卡后，每卡头数 n_h 变小，强度也随之下降。
        </>
      }
      controls={<Slider className="min-w-[220px] flex-1" label="每卡 query 头数 n_h" value={nh} min={8} max={128} step={8} onChange={setNh} />}
    >
      <div className="space-y-2.5">
        {rows.map((r) => {
          const I = r.flops / r.bytes
          const w = (Math.log10(Math.max(1, I)) / logMax) * 100
          return (
            <div key={r.name} className="grid grid-cols-[88px_1fr] items-center gap-2 @xl:grid-cols-[120px_1fr_210px]">
              <div className="text-[13px] font-medium text-fg">{r.name}</div>
              <div className="relative h-6 rounded-md bg-surface-2">
                <div className="absolute inset-y-0 left-0 rounded-md transition-[width] duration-300" style={{ width: `${w}%`, background: toneColor(r.tone) }} />
                <div className="absolute inset-y-0 w-0.5 bg-fg/70" style={{ left: `${(Math.log10(RIDGE) / logMax) * 100}%` }} title="H100 拐点" />
                <span className="absolute inset-y-0 right-2 flex items-center font-mono text-[11.5px] text-fg">{I < 10 ? I.toFixed(1) : I.toFixed(0)} FLOP/B</span>
              </div>
              <div className="col-span-2 font-mono text-[11px] text-muted @xl:col-span-1">
                读 {fmtCount(r.bytes)} B · 算 {fmtCount(r.flops)} FLOPs · {r.note}
              </div>
            </div>
          )
        })}
        <div className="text-[11px] text-subtle">横轴为对数刻度（1 ~ 4×10⁴）；竖线 = H100 拐点 ≈ {RIDGE} FLOP/B。只统计注意力核心（打分 + 加权求和），不含投影矩阵。</div>
      </div>
    </VizFrame>
  )
}
