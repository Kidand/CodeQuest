/**
 * 正向 KL vs 反向 KL：用单峰高斯（学生）去拟合双峰分布（教师）。
 * forward KL(p_t‖q) 覆盖所有模式（mode-covering）；reverse KL(q‖p_t) 只抓一个模式（mode-seeking）。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Button, Slider } from '@/components/ui'
import { Axes, Line, Svg, Stat, useScales, type Axis } from './plot'

const XS = Array.from({ length: 81 }, (_, i) => -5 + i * 0.125)
const gauss = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / s
const norm = (v: number[]) => {
  const s = v.reduce((a, b) => a + b, 0)
  return v.map((x) => Math.max(x / s, 1e-12))
}
const TEACHER = norm(XS.map((x) => 0.6 * gauss(x, -2, 0.6) + 0.4 * gauss(x, 2.2, 0.6)))
const student = (m: number, s: number) => norm(XS.map((x) => gauss(x, m, s)))
const kl = (p: number[], q: number[]) => p.reduce((a, pi, i) => a + pi * Math.log(pi / q[i]), 0)

function argmin(f: (m: number, s: number) => number) {
  let best = { m: 0, s: 1, v: Infinity }
  for (let m = -4; m <= 4.001; m += 0.1)
    for (let s = 0.3; s <= 3.001; s += 0.05) {
      const v = f(m, s)
      if (v < best.v) best = { m, s, v }
    }
  return best
}

export function KLModeViz() {
  const [mu, setMu] = useState(0)
  const [sigma, setSigma] = useState(1.5)
  const q = useMemo(() => student(mu, sigma), [mu, sigma])
  const fwd = kl(TEACHER, q)
  const rev = kl(q, TEACHER)
  const opt = useMemo(
    () => ({
      fwd: argmin((m, s) => kl(TEACHER, student(m, s))),
      rev: argmin((m, s) => kl(student(m, s), TEACHER)),
    }),
    [],
  )
  const ymax = Math.max(...TEACHER, ...q) * 1.1
  const x: Axis = { min: -5, max: 5, ticks: [-4, -2, 0, 2, 4], label: '输出空间（示意）' }
  const y: Axis = { min: 0, max: ymax, ticks: [0], label: '概率' }
  const { sx, sy } = useScales(x, y)

  return (
    <VizFrame
      title="反向 KL 只追一个峰，正向 KL 想盖住所有峰"
      caption={
        <>
          当前学生 N(μ={mu.toFixed(1)}, σ={sigma.toFixed(2)})：正向 KL(p_t‖q) = <b className="font-mono">{fwd.toFixed(3)}</b>，
          反向 KL(q‖p_t) = <b className="font-mono">{rev.toFixed(3)}</b>。最小化正向 KL 的解是 μ≈{opt.fwd.m.toFixed(1)}、σ≈{opt.fwd.s.toFixed(2)}
          （横跨两峰、把概率放在教师认为不可能的谷底）；最小化反向 KL 的解是 μ≈{opt.rev.m.toFixed(1)}、σ≈{opt.rev.s.toFixed(2)}（锁定一个峰）。
          on-policy 蒸馏用反向 KL：学生只需在自己会走的地方学得像老师，不会被迫“平均”出教师从不说的话。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '教师 p_t（双峰）', tone: 'info', variant: 'line' },
            { label: '学生 q（单峰高斯）', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="w-36" label="μ" value={mu} min={-4} max={4} step={0.1} onChange={setMu} format={(v) => v.toFixed(1)} />
          <Slider className="w-36" label="σ" value={sigma} min={0.3} max={3} step={0.05} onChange={setSigma} format={(v) => v.toFixed(2)} />
          <Button size="sm" variant="secondary" onClick={() => (setMu(opt.fwd.m), setSigma(opt.fwd.s))}>
            最小化正向 KL
          </Button>
          <Button size="sm" variant="secondary" onClick={() => (setMu(opt.rev.m), setSigma(opt.rev.s))}>
            最小化反向 KL
          </Button>
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <Svg>
          <Axes x={x} y={y} />
          <path
            d={`M${XS.map((v, i) => `${sx(v)},${sy(TEACHER[i])}`).join(' L')} L${sx(5)},${sy(0)} L${sx(-5)},${sy(0)} Z`}
            fill={toneColor('info')}
            opacity={0.15}
          />
          <Line pts={XS.map((v, i) => [v, TEACHER[i]])} x={x} y={y} tone="info" />
          <Line pts={XS.map((v, i) => [v, q[i]])} x={x} y={y} tone="danger" />
        </Svg>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="正向 KL(p_t‖q)" value={fwd.toFixed(3)} tone="info" />
          <Stat label="反向 KL(q‖p_t)" value={rev.toFixed(3)} tone="danger" />
        </div>
      </div>
    </VizFrame>
  )
}
