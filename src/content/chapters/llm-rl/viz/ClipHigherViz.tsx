/**
 * Clip-Higher 的概率空间视角：A>0 时，一次更新最多把 π 推到 (1+ε_high)·π_old。
 * 洞见：上界是「乘性」的——高概率 token 几乎不受限，低概率的探索 token 只能涨一点点。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Slider } from '@/components/ui'

const TOKENS: { tok: string; p: number }[] = [
  { tok: 'Wait', p: 0.01 },
  { tok: 'However', p: 0.04 },
  { tok: 'Alternatively', p: 0.12 },
  { tok: 'So', p: 0.45 },
  { tok: 'Therefore', p: 0.9 },
]

export function ClipHigherViz() {
  const [epsHigh, setEpsHigh] = useState(0.28)
  const base = 0.2
  return (
    <VizFrame
      title="上界是乘性的：低概率的「探索 token」几乎涨不动"
      caption={
        <>
          ε = 0.2 时，π_old = 0.01 的 “Wait” 一轮最多涨到 <b className="font-mono">0.012</b>（+0.002），而 π_old = 0.9 的 token 上界是 1.08，等于不设限。
          把 ε_high 调到 <b className="font-mono">{epsHigh.toFixed(2)}</b>：低概率 token 的上界变为 <b className="font-mono">{(0.01 * (1 + epsHigh)).toFixed(4)}</b>。
          DAPO 只放宽上界、保持 ε_low = 0.2：放宽下界会把低概率 token 压到 0，采样空间坍缩。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'π_old', tone: 'muted' },
            { label: 'ε = 0.2 允许的增量', tone: 'warning' },
            { label: `ε_high 额外放宽`, tone: 'success' },
          ]}
        />
      }
      controls={<Slider className="min-w-[220px] flex-1" label="ε_high" value={epsHigh} min={0.2} max={0.5} step={0.01} onChange={setEpsHigh} format={(v) => v.toFixed(2)} />}
    >
      <div className="space-y-3">
        {TOKENS.map(({ tok, p }) => {
          const cap0 = Math.min(1, p * (1 + base))
          const cap1 = Math.min(1, p * (1 + epsHigh))
          // 对数横轴：低概率 token 也能看清
          const lx = (v: number) => ((Math.log10(Math.max(v, 0.005)) + 2.3) / 2.3) * 100
          return (
            <div key={tok} className="grid grid-cols-[84px_minmax(0,1fr)] items-center gap-2 @md:grid-cols-[110px_minmax(0,1fr)_150px]">
              <span className="truncate text-right font-mono text-[13px]">{tok}</span>
              <div className="relative h-6 overflow-hidden rounded-md bg-surface-2">
                <div className="absolute inset-y-0 left-0" style={{ width: `${lx(cap1)}%`, background: toneSoft('success', 55) }} />
                <div className="absolute inset-y-0 left-0" style={{ width: `${lx(cap0)}%`, background: toneSoft('warning', 70) }} />
                <div className="absolute inset-y-0 left-0" style={{ width: `${lx(p)}%`, background: toneColor('muted') }} />
              </div>
              <span className="col-span-2 text-right font-mono text-[12px] text-muted @md:col-span-1">
                {p} → {cap1 >= 1 ? '1（不受限）' : cap1.toFixed(4)}
                <span style={{ color: toneColor('success') }}> +{(cap1 - p).toFixed(4)}</span>
              </span>
            </div>
          )
        })}
        <div className="pl-[92px] font-mono text-[11px] text-subtle @md:pl-[118px]">横轴为对数刻度：0.005 · 0.05 · 0.5</div>
      </div>
    </VizFrame>
  )
}
