/**
 * pass@k 曲线：把 RL 建模为在 log-odds 空间围绕阈值 τ=0.1 的“锐化”：ℓ' = γ(ℓ − ℓ_τ) + ℓ_τ。
 * 成功率高于 τ 的题被推高，低于 τ 的题被压向 0。
 * 洞见：锐化提高 pass@1，但大 k 时 pass@k 反而低于 base（Yue et al. 观察到的现象）。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { Axes, Line, Svg, Stat, useScales, type Axis } from './plot'

const N = 60
const TAU = 0.1
const KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

function baseProbs() {
  const r = rng(11)
  // 难度分布：不少题成功率很低（但不为 0），少数题已经很高
  return Array.from({ length: N }, () => {
    const u = r()
    return Math.min(0.97, Math.max(0.002, u ** 2.2))
  })
}

const passk = (ps: number[], k: number) => ps.reduce((s, p) => s + 1 - (1 - p) ** k, 0) / ps.length

export function PassKViz() {
  const [gamma, setGamma] = useState(2.5)
  const base = useMemo(baseProbs, [])
  const rl = useMemo(() => {
    const lt = Math.log(TAU / (1 - TAU))
    return base.map((p) => 1 / (1 + Math.exp(-(gamma * (Math.log(p / (1 - p)) - lt) + lt))))
  }, [base, gamma])
  const cb = KS.map((k) => passk(base, k))
  const cr = KS.map((k) => passk(rl, k))
  const cross = KS.find((k, i) => cb[i] > cr[i])
  const x: Axis = { min: 1, max: 1024, log: true, ticks: [1, 4, 16, 64, 256, 1024], label: 'k（对数轴）' }
  const y: Axis = { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], label: 'pass@k', fmt: (v) => v.toFixed(2) }
  const { sx, sy } = useScales(x, y)

  return (
    <VizFrame
      title="RL 提升 pass@1，却可能降低大 k 的 pass@k"
      caption={
        <>
          锐化强度 γ = <b className="font-mono">{gamma.toFixed(1)}</b>：pass@1 从 <b className="font-mono">{cb[0].toFixed(3)}</b> 变为{' '}
          <b className="font-mono">{cr[0].toFixed(3)}</b>；pass@1024 从 <b className="font-mono">{cb.at(-1)!.toFixed(3)}</b> 变为{' '}
          <b className="font-mono">{cr.at(-1)!.toFixed(3)}</b>。
          {cross ? ` 两条曲线在 k ≈ ${cross} 附近交叉：之后 base 更好——RL 把“偶尔能做对”的题推向了 0。` : ' 当前设置下 RL 在所有 k 上都不差于 base。'}
          （玩具模型：RL 只是把成功率高于 0.1 的题推高、低于 0.1 的题压低，不创造新解法。）
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'base 模型', tone: 'info', variant: 'line' },
            { label: 'RL 后（锐化）', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={<Slider className="min-w-[220px] flex-1" label="锐化强度 γ（1 = 不变）" value={gamma} min={1} max={5} step={0.1} onChange={setGamma} format={(v) => v.toFixed(1)} />}
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <Svg>
          <Axes x={x} y={y} />
          <Line pts={KS.map((k, i) => [k, cb[i]])} x={x} y={y} tone="info" />
          <Line pts={KS.map((k, i) => [k, cr[i]])} x={x} y={y} tone="danger" />
          {KS.map((k, i) => (
            <g key={k}>
              <circle cx={sx(k)} cy={sy(cb[i])} r={3} fill={toneColor('info')} />
              <circle cx={sx(k)} cy={sy(cr[i])} r={3} fill={toneColor('danger')} />
            </g>
          ))}
        </Svg>
        <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
          <Stat label="pass@1 base → RL" value={`${cb[0].toFixed(2)} → ${cr[0].toFixed(2)}`} tone="success" />
          <Stat label="pass@256 base → RL" value={`${cb[8].toFixed(2)} → ${cr[8].toFixed(2)}`} tone="danger" />
          <Stat label="交叉点 k" value={cross ? `≈ ${cross}` : '无'} tone="warning" />
        </div>
      </div>
    </VizFrame>
  )
}
