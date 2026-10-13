/**
 * Newton-Schulz 迭代把奇异值推向 1。
 * X ← aX + b(XXᵀ)X + c(XXᵀ)²X 对 X = UΣVᵀ 只改变奇异值：σ ← φ(σ) = aσ + bσ³ + cσ⁵（U、V 不变）。
 * 先按 Frobenius 范数归一化保证 σ ≤ 1。比较四种系数方案：
 *   经典三次 (1.5, −0.5, 0)：σ = 1 是稳定不动点，但小奇异值增长太慢（每步 ×1.5）
 *   (2, −1.5, 0.5)：φ(1) = 1 且 φ′(1) = 0，靠近 1 时收敛极快，小值每步 ×2
 *   Keller Jordan 五次 (3.4445, −4.7750, 2.0315)：φ′(0) = 3.4445，小值增长最快，但并不收敛到 1，而在 ≈[0.68, 1.13] 内振荡
 *   DeepSeek-V4 混合：前 8 步用五次系数快速拉近，后 2 步用 (2, −1.5, 0.5) 精确落到 1
 * 奇异值初值取 [1e-3, 1] 上对数均匀的 9 个点（归一化后真实矩阵的奇异值通常就在这个范围）。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, catColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Axes, HBand, Line, Plot, Stat, VLine, sample, scales, type Axis } from './plot'

type Scheme = 'cubic' | 'two' | 'quintic' | 'hybrid'
const COEF: Record<'cubic' | 'two' | 'quintic', [number, number, number]> = {
  cubic: [1.5, -0.5, 0],
  two: [2, -1.5, 0.5],
  quintic: [3.4445, -4.775, 2.0315],
}
const S0 = Array.from({ length: 9 }, (_, i) => 10 ** (-3 + (3 * i) / 8))
const phi = ([a, b, c]: [number, number, number]) => (x: number) => a * x + b * x ** 3 + c * x ** 5

function coeffAt(s: Scheme, k: number): [number, number, number] {
  if (s === 'hybrid') return k < 8 ? COEF.quintic : COEF.two
  return COEF[s]
}

export function NSIterViz() {
  const [scheme, setScheme] = useState<Scheme>('quintic')
  const [steps, setSteps] = useState(5)
  const N = scheme === 'hybrid' ? 10 : steps
  const traj = useMemo(() => {
    return S0.map((s0) => {
      const out: [number, number][] = [[0, s0]]
      let s = s0
      for (let k = 0; k < N; k++) {
        s = phi(coeffAt(scheme, k))(s)
        out.push([k + 1, s])
      }
      return out
    })
  }, [scheme, N])
  const finals = traj.map((t) => t[t.length - 1][1])
  const lo = Math.min(...finals)
  const hi = Math.max(...finals)
  const shown = scheme === 'hybrid' ? COEF.quintic : COEF[scheme]
  const xi: Axis = { min: 0, max: Math.max(N, 1), ticks: Array.from({ length: Math.max(N, 1) + 1 }, (_, i) => i).filter((i) => N <= 10 || i % 2 === 0), label: '迭代步 k' }
  const yi: Axis = { min: 1e-3, max: 2, log: true, ticks: [1e-3, 1e-2, 1e-1, 1], label: '奇异值（对数）', fmt: (v) => (v >= 1 ? '1' : `1e${Math.round(Math.log10(v))}`) }
  const xp: Axis = { min: 0, max: 1.3, ticks: [0, 0.25, 0.5, 0.75, 1, 1.25], label: '输入奇异值 σ', fmt: (v) => String(v) }
  const yp: Axis = { min: 0, max: 1.4, ticks: [0, 0.5, 1], label: 'φ(σ)', fmt: (v) => v.toFixed(1) }
  const label =
    scheme === 'cubic'
      ? '经典三次 (1.5, −0.5)'
      : scheme === 'two'
        ? '(2, −1.5, 0.5)'
        : scheme === 'quintic'
          ? 'Muon 五次 (3.4445, −4.7750, 2.0315)'
          : 'DeepSeek-V4 混合：8 步五次 + 2 步 (2, −1.5, 0.5)'

  return (
    <VizFrame
      title="Newton-Schulz：只动奇异值，把它们全部推到 1 附近"
      caption={
        <>
          {label}，{N} 步后 9 个奇异值落在 <b className="font-mono">[{lo.toFixed(3)}, {hi.toFixed(3)}]</b>。
          {scheme === 'cubic' && ' 三次迭代对小奇异值每步只放大 1.5 倍：1e-3 要十几步才能“醒过来”。'}
          {scheme === 'two' && ' φ′(1) = 0，接近 1 时收敛极快；但小值每步只放大 2 倍。'}
          {scheme === 'quintic' && ' 斜率 φ′(0) = 3.4445，小奇异值增长最快；代价是 φ(1) = 0.701，不会精确收敛到 1，而在约 [0.68, 1.13] 之间来回振荡——Muon 认为这“足够正交”。'}
          {scheme === 'quintic' && lo < 0.6 && ` 最小的那个（初值 1e-3）${N} 步后只到 ${lo.toFixed(2)}：初值不低于约 0.01 的奇异值 5 步就能进入区间，更小的需要更多步。`}
          {scheme === 'hybrid' && ' 前 8 步用五次系数快速拉近，最后 2 步换成 φ′(1) = 0 的系数“精修”，所有奇异值都落在 1 附近（误差约 1e-3 以内）。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'Muon 设计目标带 [0.7, 1.3]', tone: 'success', variant: 'soft' },
            { label: 'φ(σ)', tone: 'accent', variant: 'line' },
            { label: 'y = σ / y = 1', tone: 'muted', variant: 'line' },
            ...(scheme === 'hybrid' ? [{ label: '精修系数 (2, −1.5, 0.5)', tone: 'warning' as const, variant: 'line' as const }] : []),
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={scheme}
            onChange={setScheme}
            options={[
              { value: 'cubic', label: '三次' },
              { value: 'two', label: '(2,−1.5,0.5)' },
              { value: 'quintic', label: 'Muon 五次' },
              { value: 'hybrid', label: 'V4 混合' },
            ]}
          />
          {scheme !== 'hybrid' && <Slider className="min-w-[160px] flex-1" label="迭代步数" value={steps} min={1} max={12} step={1} onChange={setSteps} />}
        </>
      }
    >
      <div className="grid gap-5 @3xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium text-muted">① 每个奇异值随迭代的变化</div>
          <Plot height={250}>
            {(box) => (
              <>
                <HBand x={xi} y={yi} box={box} from={0.7} to={1.3} tone="success" />
                <Axes x={xi} y={yi} box={box} />
                {scheme === 'hybrid' && <VLine x={xi} y={yi} box={box} at={8} tone="warning" label="切换系数" />}
                {traj.map((t, i) => (
                  <Line key={i} pts={t} x={xi} y={yi} box={box} tone={((i % 8) + 1) as 1} width={1.8} />
                ))}
                {traj.map((t, i) => {
                  const { sx, sy } = scales(xi, yi, box)
                  const [k, v] = t[t.length - 1]
                  return <circle key={`d${i}`} cx={sx(k)} cy={sy(Math.max(1e-3, Math.min(2, v)))} r={3.5} fill={catColor(i)} stroke="var(--surface)" strokeWidth={1.2} />
                })}
              </>
            )}
          </Plot>
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium text-muted">② 一步迭代对奇异值的作用 φ(σ) = aσ + bσ³ + cσ⁵</div>
          <Plot height={250}>
            {(box) => (
              <>
                <HBand x={xp} y={yp} box={box} from={0.7} to={1.3} tone="success" />
                <Axes x={xp} y={yp} box={box} />
                <Line pts={[[0, 0], [1.3, 1.3]]} x={xp} y={yp} box={box} tone="muted" dashed width={1.4} />
                <Line pts={[[0, 1], [1.3, 1]]} x={xp} y={yp} box={box} tone="muted" width={1.2} />
                <Line pts={sample(phi(shown), xp, 200)} x={xp} y={yp} box={box} tone="accent" />
                {scheme === 'hybrid' && <Line pts={sample(phi(COEF.two), xp, 200)} x={xp} y={yp} box={box} tone="warning" dashed />}
              </>
            )}
          </Plot>
          <div className="mt-2 grid grid-cols-3 gap-2">
            <Stat label="φ′(0) = a" value={shown[0].toFixed(4)} />
            <Stat label="φ(1)" value={phi(shown)(1).toFixed(3)} />
            <Stat label={`${N} 步后最小值`} value={lo.toFixed(3)} tone={lo > 0.6 ? 'success' : 'danger'} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
