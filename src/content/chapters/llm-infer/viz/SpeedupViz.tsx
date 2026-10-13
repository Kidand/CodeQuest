/**
 * 期望接受长度与加速比（Leviathan et al. 2023 的 i.i.d. 假设）：
 *   E[每轮产出] = (1 - α^(γ+1)) / (1 - α)
 *   加速比      = E / (γ·c + 1)
 * 滑块调 α 与 c，柱状图给出每个 γ 的理论加速比，金色为最优 γ。
 */
import { useState } from 'react'
import { VizFrame, toneColor, toneSoft } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { PlotFrame, Stat, type Axis } from './common'

const G_MAX = 12
export const expectedTokens = (a: number, g: number) => (a >= 1 ? g + 1 : (1 - a ** (g + 1)) / (1 - a))
export const speedup = (a: number, g: number, c: number) => expectedTokens(a, g) / (g * c + 1)

const PRESETS = [
  { v: 'custom', label: '自定义', a: 0, c: 0 },
  { v: 't5s', label: 'T5-small', a: 0.75, c: 0.05 },
  { v: 't5l', label: 'T5-large', a: 0.82, c: 0.2 },
  { v: 'mtp', label: 'MTP', a: 0.87, c: 0.05 },
] as const

export function SpeedupViz() {
  const [alpha, setAlpha] = useState(0.8)
  const [c, setC] = useState(0.1)
  const [preset, setPreset] = useState<(typeof PRESETS)[number]['v']>('custom')
  const vals = Array.from({ length: G_MAX + 1 }, (_, g) => speedup(alpha, g, c))
  let best = 0
  vals.forEach((v, g) => {
    if (v > vals[best] + 1e-12) best = g
  })
  const yMax = Math.max(2, Math.ceil(Math.max(...vals) + 0.5))
  const x: Axis = { min: -0.6, max: G_MAX + 0.6, ticks: Array.from({ length: G_MAX + 1 }, (_, g) => g), label: '草稿长度 γ' }
  const y: Axis = { min: 0, max: yMax, ticks: Array.from({ length: yMax + 1 }, (_, i) => i), fmt: (v) => `${v}×`, label: '理论加速比' }

  return (
    <VizFrame
      title="加速比随 γ 先升后降"
      caption={
        <>
          α = {alpha.toFixed(2)}、c = {c.toFixed(2)} 时，最优 γ = <b>{best}</b>：每轮期望产出 {expectedTokens(alpha, best).toFixed(2)} 个 token，代价是 1 次 target 前向 + {best} 次草稿前向，理论加速{' '}
          <b>{vals[best].toFixed(2)}×</b>。γ 再大，第 γ 个草稿被用上的概率 α^γ 越来越小，草稿成本 γc 却线性增长。
          {alpha <= c && ' 现在 α ≤ c：任何 γ 都不划算（Leviathan 推论 3.9：α > c 才可能加速）。'}
        </>
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={preset}
            onChange={(v) => {
              setPreset(v)
              const p = PRESETS.find((x) => x.v === v)!
              if (v !== 'custom') {
                setAlpha(p.a)
                setC(p.c)
              }
            }}
            options={PRESETS.map((p) => ({ value: p.v, label: p.label }))}
          />
          <Slider className="min-w-[160px] flex-1" label="接受率 α" value={alpha} min={0.1} max={0.99} step={0.01} onChange={(v) => (setAlpha(v), setPreset('custom'))} format={(v) => v.toFixed(2)} />
          <Slider className="min-w-[160px] flex-1" label="成本比 c（草稿/目标）" value={c} min={0} max={0.6} step={0.01} onChange={(v) => (setC(v), setPreset('custom'))} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <PlotFrame x={x} y={y} height={240} pad={{ l: 40, r: 10, t: 22, b: 38 }}>
          {(s) => {
            const bw = Math.max(6, ((s.sx(1) - s.sx(0)) * 0.62) | 0)
            return (
              <g>
                <line x1={s.l} x2={s.W - s.r} y1={s.sy(1)} y2={s.sy(1)} stroke={toneColor('danger')} strokeDasharray="5 4" strokeWidth={1.4} />
                <text x={s.l + 4} y={s.sy(1) - 5} fontSize={11} fill={toneColor('danger')} paintOrder="stroke" stroke="var(--surface)" strokeWidth={3}>
                  1×
                </text>
                {vals.map((v, g) => {
                  const tone = g === best ? 'gold' : v < 1 ? 'danger' : 'accent'
                  return (
                    <g key={g}>
                      <rect
                        x={s.sx(g) - bw / 2}
                        y={s.sy(Math.min(v, yMax))}
                        width={bw}
                        height={Math.max(0, s.sy(0) - s.sy(Math.min(v, yMax)))}
                        rx={3}
                        fill={toneSoft(tone, g === best ? 80 : 45)}
                        stroke={toneColor(tone)}
                        strokeWidth={g === best ? 2 : 1}
                        style={{ transition: 'all 0.25s' }}
                      />
                      {g === best && (
                        <text x={s.sx(g)} y={s.sy(v) - 6} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="var(--fg)">
                          {v.toFixed(2)}×
                        </text>
                      )}
                    </g>
                  )
                })}
              </g>
            )
          }}
        </PlotFrame>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="最优 γ" value={best} tone="gold" />
          <Stat label="最优加速比" value={`${vals[best].toFixed(2)}×`} tone="gold" />
          <Stat label="E[每轮产出]（γ*）" value={expectedTokens(alpha, best).toFixed(2)} />
          <Stat label="上限 1/(1−α)（γ→∞）" value={alpha >= 1 ? '∞' : (1 / (1 - alpha)).toFixed(2)} />
          <div className="col-span-2 text-[11.5px] leading-relaxed text-subtle">
            预设的 α 来自论文（T5：Leviathan 等表 3，EnDe、t=0；MTP：DeepSeek-V3 报告第 2 个 token 接受率 85–90%，取中值），c 为示意值。
            理想化假设：验证 γ+1 个 token 与生成 1 个 token 耗时相同、各位置接受独立同分布。
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
