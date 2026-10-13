/**
 * PPO/GRPO 的 clipped surrogate：L(r) = min(r·A, clip(r, 1−ε_low, 1+ε_high)·A)。
 * 拖动 ratio，看当前 token 处于「有梯度」还是「被裁剪、梯度为 0」的区域。
 * dual=true 时 ε_low / ε_high 分开调（DAPO 的 Clip-Higher）。
 */
import { useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Axes, BOX, Line, Svg, Stat, f3, sample, useScales, type Axis } from './plot'

export function ClipRegionViz({ dual = false }: { dual?: boolean }) {
  const [sign, setSign] = useState<'pos' | 'neg'>('pos')
  const [epsLow, setEpsLow] = useState(0.2)
  const [epsHigh, setEpsHigh] = useState(dual ? 0.28 : 0.2)
  const [r, setR] = useState(1.32)
  const eh = dual ? epsHigh : epsLow
  const A = sign === 'pos' ? 1 : -1
  const lo = 1 - epsLow
  const hi = 1 + eh
  const clip = (x: number) => Math.min(hi, Math.max(lo, x))
  const L = (x: number) => Math.min(x * A, clip(x) * A)
  // 当前点：哪一项被 min 选中
  const unclipped = r * A <= clip(r) * A + 1e-12
  const grad = unclipped ? A : 0

  const x: Axis = { min: 0.4, max: 1.8, ticks: [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8], label: 'ratio r = π_θ / π_old', fmt: (v) => v.toFixed(1) }
  const y: Axis = { min: -1.9, max: 1.9, ticks: [-1.5, -1, -0.5, 0, 0.5, 1, 1.5], label: 'L(r)', fmt: (v) => v.toFixed(1) }
  const { sx, sy } = useScales(x, y)

  return (
    <VizFrame
      title={dual ? 'Clip-Higher：上下界解耦后，可信域不再对称' : 'clipped 目标：越界的 token 梯度被直接截断为 0'}
      caption={
        unclipped ? (
          <>
            r = <b className="font-mono">{r.toFixed(2)}</b>，A {A > 0 ? '> 0' : '< 0'}：min 选中未裁剪项 <span className="font-mono">r·A</span>，
            该 token 的梯度系数 ∂L/∂r = <b className="font-mono">{A}</b>（照常更新）。
          </>
        ) : (
          <>
            r = <b className="font-mono">{r.toFixed(2)}</b>，A {A > 0 ? `> 0 且 r > 1+ε${dual ? '_high' : ''}` : `< 0 且 r < 1−ε${dual ? '_low' : ''}`}：min 选中常数 {A > 0 ? '(1+ε)A' : '(1−ε)A'}，
            <b>梯度为 0</b>——这个 token 已经「改够了」，本轮不再推它。
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: 'min(r·A, clip(r)·A)', tone: 'accent', variant: 'line' },
            { label: '未裁剪 r·A', tone: 'muted', variant: 'line' },
            { label: '梯度为 0 的区域', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={sign}
            onChange={setSign}
            options={[
              { value: 'pos', label: 'A > 0（好回答）' },
              { value: 'neg', label: 'A < 0（坏回答）' },
            ]}
          />
          <Slider className="w-40" label="当前 ratio r" value={r} min={0.45} max={1.75} step={0.01} onChange={setR} format={(v) => v.toFixed(2)} />
          <Slider className="w-36" label={dual ? 'ε_low' : 'ε'} value={epsLow} min={0.05} max={0.4} step={0.01} onChange={setEpsLow} format={(v) => v.toFixed(2)} />
          {dual && <Slider className="w-36" label="ε_high" value={epsHigh} min={0.05} max={0.5} step={0.01} onChange={setEpsHigh} format={(v) => v.toFixed(2)} />}
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <Svg>
          {/* 梯度为 0 的区域 */}
          {A > 0 ? (
            <rect x={sx(hi)} y={BOX.pad.t} width={sx(x.max) - sx(hi)} height={BOX.H - BOX.pad.b - BOX.pad.t} fill={toneColor('danger')} opacity={0.1} />
          ) : (
            <rect x={sx(x.min)} y={BOX.pad.t} width={sx(lo) - sx(x.min)} height={BOX.H - BOX.pad.b - BOX.pad.t} fill={toneColor('danger')} opacity={0.1} />
          )}
          <Axes x={x} y={y} />
          <line x1={sx(lo)} x2={sx(lo)} y1={BOX.pad.t} y2={BOX.H - BOX.pad.b} stroke={toneColor('warning')} strokeDasharray="3 3" />
          <line x1={sx(hi)} x2={sx(hi)} y1={BOX.pad.t} y2={BOX.H - BOX.pad.b} stroke={toneColor('warning')} strokeDasharray="3 3" />
          <text x={sx(lo)} y={BOX.pad.t + 10} fontSize={10} textAnchor="end" fill={toneColor('warning')}>
            1−ε{dual ? '_low' : ''}
          </text>
          <text x={sx(hi) + 3} y={BOX.pad.t + 10} fontSize={10} fill={toneColor('warning')}>
            1+ε{dual ? '_high' : ''}
          </text>
          <Line pts={sample((v) => v * A, x)} x={x} y={y} tone="muted" dashed width={1.4} />
          <Line pts={sample(L, x, 280)} x={x} y={y} tone="accent" />
          <circle cx={sx(r)} cy={sy(L(r))} r={6} fill={unclipped ? toneColor('success') : toneColor('danger')} stroke="var(--surface)" strokeWidth={2} />
        </Svg>
        <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
          <Stat label="L(r)" value={f3(L(r))} />
          <Stat label="∂L/∂r" value={String(grad)} tone={grad === 0 ? 'danger' : 'success'} />
          <Stat label="可信域" value={`[${lo.toFixed(2)}, ${hi.toFixed(2)}]`} tone="warning" />
        </div>
      </div>
    </VizFrame>
  )
}

export const ClipRegionDualViz = () => <ClipRegionViz dual />
