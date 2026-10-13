/**
 * PPO clipped surrogate 在 (ratio, advantage) 平面上的“零梯度区”。
 *   L = min(r·A, clip(r, 1−ε, 1+ε)·A)；dual-clip（A<0 时再与 c·A 取 max）。
 * 散点是一批 token：落在红色区域的 token 本轮不产生梯度，占比即 clipfrac。点击平面放置探针。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Slider, Switch } from '@/components/ui'
import { rng } from '@/lib/utils'
import { Axes, Stat, Svg, f3, scales, type Axis, type PlotBox } from './plot'

const BOXP: PlotBox = { W: 420, H: 270, pad: { l: 40, r: 12, t: 16, b: 38 } }

function tokens() {
  const r = rng(11)
  const gauss = () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r())
  return Array.from({ length: 36 }, () => ({ ratio: Math.exp(0.22 * gauss()), adv: 0.8 * gauss() }))
}
const TOKS = tokens()

export function ClipPlaneViz() {
  const [eps, setEps] = useState(0.2)
  const [dual, setDual] = useState(false)
  const [probe, setProbe] = useState({ ratio: 1.45, adv: 0.9 })
  const C = 3
  const xmax = dual ? 3.6 : 2
  const x: Axis = { min: 0, max: xmax, ticks: dual ? [0, 0.8, 1.2, 2, 3] : [0, 0.5, 0.8, 1, 1.2, 1.5, 2], fmt: (v) => String(v), label: 'ratio r = π_θ / π_old' }
  const y: Axis = { min: -2, max: 2, ticks: [-2, -1, 0, 1, 2], label: 'A' }
  const { sx, sy } = scales(x, y, BOXP)

  const clipped = (ratio: number, adv: number) =>
    (adv > 0 && ratio > 1 + eps) || (adv < 0 && ratio < 1 - eps) || (dual && adv < 0 && ratio > C)
  const L = (ratio: number, adv: number) => {
    const c = Math.min(Math.max(ratio, 1 - eps), 1 + eps)
    const base = Math.min(ratio * adv, c * adv)
    return dual && adv < 0 ? Math.max(base, C * adv) : base
  }
  const frac = useMemo(() => TOKS.filter((t) => clipped(t.ratio, t.adv)).length / TOKS.length, [eps, dual]) // eslint-disable-line react-hooks/exhaustive-deps

  const onClick = (e: React.MouseEvent<SVGRectElement>) => {
    const svg = e.currentTarget.ownerSVGElement
    if (!svg) return
    const rect = svg.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * BOXP.W
    const py = ((e.clientY - rect.top) / rect.height) * BOXP.H
    const ratio = x.min + ((px - BOXP.pad.l) / (BOXP.W - BOXP.pad.l - BOXP.pad.r)) * (x.max - x.min)
    const adv = y.max - ((py - BOXP.pad.t) / (BOXP.H - BOXP.pad.t - BOXP.pad.b)) * (y.max - y.min)
    if (ratio >= x.min && ratio <= x.max && adv >= y.min && adv <= y.max) setProbe({ ratio, adv })
  }

  const pc = clipped(probe.ratio, probe.adv)
  const top = sy(y.max)
  const mid = sy(0)
  const bot = sy(y.min)
  return (
    <VizFrame
      title="Clip 区域：哪些 token 没有梯度"
      caption={
        <>
          探针 r = <b className="font-mono">{probe.ratio.toFixed(2)}</b>、A = <b className="font-mono">{probe.adv.toFixed(2)}</b>：L = {f3(L(probe.ratio, probe.adv))}，
          {pc ? (
            <>
              <b style={{ color: toneColor('danger') }}>梯度为 0</b>——
              {probe.adv > 0 && probe.ratio > 1 + eps
                ? '好动作的概率已经提高超过 1+ε，不再继续推高。'
                : probe.adv < 0 && probe.ratio < 1 - eps
                  ? '坏动作的概率已经降到 1−ε 以下，不再继续压。'
                  : 'dual-clip：坏动作的 ratio 已经大到离谱，损失被截在 c·A，防止单个 token 的梯度爆炸。'}
            </>
          ) : (
            <>
              梯度 ∂L/∂r = <b className="font-mono">{probe.adv.toFixed(2)}</b>，照常更新。
              {probe.adv < 0 && probe.ratio > 1 + eps && !dual && ' 注意：A < 0 且 r 很大时 min 选未裁剪项，损失无下界——这正是 dual-clip 要补的洞。'}
            </>
          )}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '梯度为 0（被裁剪）', tone: 'danger', variant: 'soft' },
            { label: '信任域 [1−ε, 1+ε]', tone: 'accent', variant: 'soft' },
            { label: 'token（红 = 被裁剪）', tone: 'muted' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[160px] flex-1" label="ε" value={eps} min={0.05} max={0.5} step={0.01} onChange={setEps} format={(v) => v.toFixed(2)} />
          <Switch checked={dual} onChange={setDual} label={`dual-clip（c = ${C}）`} />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <Svg box={BOXP}>
          <rect x={sx(1 - eps)} y={top} width={sx(1 + eps) - sx(1 - eps)} height={bot - top} fill={toneSoft('accent', 30)} />
          <rect x={sx(1 + eps)} y={top} width={sx(x.max) - sx(1 + eps)} height={mid - top} fill={toneColor('danger')} opacity={0.14} />
          <rect x={sx(0)} y={mid} width={sx(1 - eps) - sx(0)} height={bot - mid} fill={toneColor('danger')} opacity={0.14} />
          {dual && <rect x={sx(C)} y={mid} width={sx(x.max) - sx(C)} height={bot - mid} fill={toneColor('danger')} opacity={0.14} />}
          <Axes x={x} y={y} box={BOXP} />
          {TOKS.map((t, i) => {
            const cx = sx(Math.min(t.ratio, x.max))
            const cy = sy(Math.max(y.min, Math.min(y.max, t.adv)))
            return <circle key={i} cx={cx} cy={cy} r={4} fill={clipped(t.ratio, t.adv) ? toneColor('danger') : toneColor('muted')} opacity={0.85} />
          })}
          <g>
            <circle cx={sx(probe.ratio)} cy={sy(probe.adv)} r={7} fill="none" stroke={toneColor('accent')} strokeWidth={2.5} />
            <circle cx={sx(probe.ratio)} cy={sy(probe.adv)} r={2.5} fill={toneColor('accent')} />
          </g>
          {/* 透明点击层 */}
          <rect
            x={BOXP.pad.l}
            y={BOXP.pad.t}
            width={BOXP.W - BOXP.pad.l - BOXP.pad.r}
            height={BOXP.H - BOXP.pad.t - BOXP.pad.b}
            fill="transparent"
            style={{ cursor: 'crosshair' }}
            onClick={onClick}
          />
        </Svg>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="clipfrac" value={`${(frac * 100).toFixed(0)}%`} tone="danger" sub={`${TOKS.length} 个 token 中被裁剪的比例`} />
          <Stat label="探针梯度 ∂L/∂r" value={pc ? '0' : probe.adv.toFixed(2)} tone={pc ? 'danger' : 'accent'} />
        </div>
      </div>
    </VizFrame>
  )
}
