/**
 * DAPO 的 Soft Overlong Punishment：长度超过 L_max − L_cache 后线性扣分，到 L_max 扣满 1。
 * 拖动回答长度与两个阈值，看「正确性奖励 + 长度惩罚」如何变化。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Axes, BOX, Line, Svg, Stat, useScales, type Axis } from './plot'

export function rLength(len: number, Lmax: number, Lcache: number) {
  if (len <= Lmax - Lcache) return 0
  if (len <= Lmax) return (Lmax - Lcache - len) / Lcache
  return -1
}

export function OverlongViz() {
  const [Lmax, setLmax] = useState(20480)
  const [Lcache, setLcache] = useState(4096)
  const [len, setLen] = useState(18432)
  const [correct, setCorrect] = useState<'yes' | 'no'>('yes')
  const rl = rLength(len, Lmax, Lcache)
  const base = correct === 'yes' ? 1 : -1
  const x: Axis = { min: 0, max: 24576, ticks: [0, 4096, 8192, 12288, 16384, 20480, 24576], label: '回答长度 |y|（token）', fmt: (v) => `${v / 1024}K` }
  const y: Axis = { min: -2.1, max: 1.2, ticks: [-2, -1, 0, 1], label: '奖励' }
  const { sx, sy } = useScales(x, y)
  const curve = (b: number) => {
    const pts: [number, number][] = []
    for (let L = 0; L <= 24576; L += 64) pts.push([L, b + rLength(L, Lmax, Lcache)])
    return pts
  }

  return (
    <VizFrame
      title="超长奖励整形：在截断之前给一段「缓冲区」线性扣分"
      caption={
        <>
          |y| = <b className="font-mono">{len}</b>：R_length = <b className="font-mono">{rl.toFixed(3)}</b>，
          总奖励 = {base > 0 ? '+1' : '−1'} {rl < 0 ? `− ${(-rl).toFixed(3)}` : '+ 0'} = <b className="font-mono">{(base + rl).toFixed(3)}</b>。
          {len > Lmax - Lcache && len <= Lmax && ' 处在缓冲区内：越长扣得越多，但正确回答仍然比错误回答好。'}
          {len > Lmax && ' 已超过 L_max（被截断）：扣满 1 分。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '答对：+1 + R_length', tone: 'success', variant: 'line' },
            { label: '答错：−1 + R_length', tone: 'danger', variant: 'line' },
            { label: '缓冲区 [L_max − L_cache, L_max]', tone: 'warning', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={correct}
            onChange={setCorrect}
            options={[
              { value: 'yes', label: '答对' },
              { value: 'no', label: '答错' },
            ]}
          />
          <Slider className="w-48" label="回答长度 |y|" value={len} min={1024} max={24576} step={256} onChange={setLen} />
          <Slider className="w-40" label="L_max" value={Lmax} min={12288} max={24576} step={1024} onChange={(v) => setLmax(Math.max(v, Lcache + 1024))} />
          <Slider className="w-40" label="L_cache" value={Lcache} min={1024} max={8192} step={512} onChange={(v) => setLcache(Math.min(v, Lmax - 1024))} />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <Svg>
          <rect
            x={sx(Lmax - Lcache)}
            y={BOX.pad.t}
            width={sx(Lmax) - sx(Lmax - Lcache)}
            height={BOX.H - BOX.pad.b - BOX.pad.t}
            fill={toneColor('warning')}
            opacity={0.12}
          />
          <Axes x={x} y={y} />
          <Line pts={curve(1)} x={x} y={y} tone="success" />
          <Line pts={curve(-1)} x={x} y={y} tone="danger" />
          <line x1={sx(len)} x2={sx(len)} y1={BOX.pad.t} y2={BOX.H - BOX.pad.b} stroke="var(--fg)" strokeDasharray="3 3" opacity={0.6} />
          <circle cx={sx(len)} cy={sy(base + rl)} r={6} fill={toneColor(base > 0 ? 'success' : 'danger')} stroke="var(--surface)" strokeWidth={2} />
        </Svg>
        <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
          <Stat label="期望最大长度 L_max − L_cache" value={String(Lmax - Lcache)} />
          <Stat label="R_length" value={rl.toFixed(3)} tone={rl < 0 ? 'danger' : 'success'} />
          <Stat label="总奖励" value={(base + rl).toFixed(3)} tone="accent" />
        </div>
      </div>
    </VizFrame>
  )
}
