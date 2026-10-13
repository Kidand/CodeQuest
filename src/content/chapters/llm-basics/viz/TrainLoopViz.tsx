/**
 * BOSS 复盘：把本章的零件拼成一个真实的训练循环，在浏览器里实时训练一个 2→16→2 的 MLP（XOR 型四簇数据，固定种子）。
 * 数据为四簇高斯（σ = 0.6，簇间有重叠），前向（Linear → ReLU → Linear → 稳定 softmax-CE）→ 反向（p − y、Linear/ReLU 的 VJP）→ 全局范数裁剪 → 优化器 → 学习率调度。
 * 对照组（灰色虚线）固定为 AdamW + warmup/cosine + 裁剪、lr = 0.03。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Segmented, Slider, Switch } from '@/components/ui'
import { rng } from '@/lib/utils'
import { Axes, Line, Plot, Stat, type Axis } from './plot'

const N = 80
const H = 16
const STEPS = 200
type Opt = 'sgd' | 'adaml2' | 'adamw'

function gauss(r: () => number) {
  return Math.sqrt(-2 * Math.log(Math.max(1e-12, r()))) * Math.cos(2 * Math.PI * r())
}
const DATA = (() => {
  const r = rng(5)
  const X: number[][] = []
  const y: number[] = []
  const centers = [
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 1],
    [-1, 1, 1],
  ]
  for (let i = 0; i < N; i++) {
    const [cx, cy, lab] = centers[i % 4]
    X.push([cx + 0.6 * gauss(r), cy + 0.6 * gauss(r)])
    y.push(lab)
  }
  return { X, y }
})()
const INIT = (() => {
  const r = rng(17)
  const W1 = Array.from({ length: 2 * H }, () => gauss(r) * Math.sqrt(2 / 2))
  const W2 = Array.from({ length: H * 2 }, () => gauss(r) * Math.sqrt(2 / H))
  return [W1, new Array(H).fill(0), W2, [0, 0]]
})()

/** 前向 + 反向，返回 loss、准确率与梯度（与 params 同结构） */
function lossGrad(P: number[][]) {
  const [W1, b1, W2, b2] = P
  const g = P.map((p) => new Array(p.length).fill(0))
  let loss = 0
  let correct = 0
  for (let n = 0; n < N; n++) {
    const x = DATA.X[n]
    const z1 = new Array(H)
    const h = new Array(H)
    for (let j = 0; j < H; j++) {
      z1[j] = x[0] * W1[j] + x[1] * W1[H + j] + b1[j]
      h[j] = Math.max(0, z1[j])
    }
    const s = [b2[0], b2[1]]
    for (let j = 0; j < H; j++) {
      s[0] += h[j] * W2[j * 2]
      s[1] += h[j] * W2[j * 2 + 1]
    }
    const m = Math.max(s[0], s[1])
    const lse = m + Math.log(Math.exp(s[0] - m) + Math.exp(s[1] - m))
    const yy = DATA.y[n]
    loss += (lse - s[yy]) / N
    if ((s[1] > s[0] ? 1 : 0) === yy) correct++
    const ds = [Math.exp(s[0] - lse) / N, Math.exp(s[1] - lse) / N]
    ds[yy] -= 1 / N
    g[3][0] += ds[0]
    g[3][1] += ds[1]
    for (let j = 0; j < H; j++) {
      g[2][j * 2] += h[j] * ds[0]
      g[2][j * 2 + 1] += h[j] * ds[1]
      const dh = ds[0] * W2[j * 2] + ds[1] * W2[j * 2 + 1]
      const dz = z1[j] > 0 ? dh : 0
      g[1][j] += dz
      g[0][j] += x[0] * dz
      g[0][H + j] += x[1] * dz
    }
  }
  return { loss, acc: correct / N, g }
}

function train(opt: Opt, maxLr: number, sched: boolean, clip: boolean) {
  const P = INIT.map((p) => p.slice())
  const m = P.map((p) => new Array(p.length).fill(0))
  const v = P.map((p) => new Array(p.length).fill(0))
  const losses: [number, number][] = []
  let gnMax = 0
  const warm = 15
  const wd = 0.01
  for (let t = 0; t < STEPS; t++) {
    const r = lossGrad(P)
    losses.push([t, r.loss])
    // 全局范数裁剪
    const gn = Math.sqrt(r.g.reduce((s, gg) => s + gg.reduce((a, b) => a + b * b, 0), 0))
    gnMax = Math.max(gnMax, gn)
    const coef = clip ? Math.min(1, 1 / (gn + 1e-6)) : 1
    // warmup + cosine（到 10%）或常数
    let lr = maxLr
    if (sched) {
      if (t < warm) lr = (maxLr * (t + 1)) / warm
      else lr = 0.1 * maxLr + 0.5 * 0.9 * maxLr * (1 + Math.cos((Math.PI * (t - warm)) / (STEPS - warm)))
    }
    for (let k = 0; k < 4; k++) {
      const decay = k === 0 || k === 2 // 只对矩阵做衰减
      for (let i = 0; i < P[k].length; i++) {
        let g = r.g[k][i] * coef
        if (opt === 'sgd') {
          m[k][i] = 0.9 * m[k][i] + g
          P[k][i] -= lr * m[k][i]
          continue
        }
        if (opt === 'adaml2' && decay) g += wd * P[k][i]
        if (opt === 'adamw' && decay) P[k][i] *= 1 - lr * wd
        m[k][i] = 0.9 * m[k][i] + 0.1 * g
        v[k][i] = 0.95 * v[k][i] + 0.05 * g * g
        const mh = m[k][i] / (1 - 0.9 ** (t + 1))
        const vh = v[k][i] / (1 - 0.95 ** (t + 1))
        P[k][i] -= (lr * mh) / (Math.sqrt(vh) + 1e-8)
      }
    }
  }
  const fin = lossGrad(P)
  return { losses, final: fin.loss, acc: fin.acc, gnMax }
}

export function TrainLoopViz() {
  const [opt, setOpt] = useState<Opt>('adamw')
  const [logLr, setLogLr] = useState(Math.log10(0.03))
  const [sched, setSched] = useState(true)
  const [clip, setClip] = useState(true)
  const lr = 10 ** logLr
  const cur = useMemo(() => train(opt, lr, sched, clip), [opt, lr, sched, clip])
  const base = useMemo(() => train('adamw', 0.03, true, true), [])
  const x: Axis = { min: 0, max: STEPS, ticks: [0, 50, 100, 150, 200], label: '训练步' }
  const y: Axis = { min: 1e-3, max: 3, log: true, ticks: [1e-3, 1e-2, 1e-1, 1], label: '交叉熵 loss（对数）', fmt: (v) => (v >= 1 ? '1' : String(v)) }
  const tail = cur.losses.slice(100).map((p) => p[1])
  const osc = Math.max(...tail) - Math.min(...tail)
  const diverged = !Number.isFinite(cur.final)

  return (
    <VizFrame
      title="把本章拼起来：一个真实运行的训练循环"
      caption={
        <>
          {opt === 'sgd' ? 'SGD + 动量 0.9' : opt === 'adaml2' ? 'Adam + L2' : 'AdamW'}，峰值 lr = {lr.toPrecision(2)}，{sched ? 'warmup 15 步 + cosine' : '常数学习率'}，{clip ? '全局范数裁剪 1.0' : '不裁剪'}：
          {diverged ? ' 训练发散（loss 变成 NaN/inf）。' : ` 200 步后 loss = ${cur.final.toFixed(3)}，训练准确率 ${(cur.acc * 100).toFixed(0)}%。`}
          {' '}初始 loss = {cur.losses[0][1].toFixed(3)}（输出完全均匀时为 ln 2 ≈ 0.693）。
          {opt === 'sgd' && lr > 0.3 && ' SGD 对学习率很敏感：同样的数值对 Adam 只是“每步约 lr”，对 SGD 则要乘以梯度大小。'}
          {osc > 0.2 && ` 后 100 步 loss 的波动幅度达 ${osc.toFixed(2)}：学习率太大又没有衰减（或没有裁剪），参数在最优点附近来回跳。`}
          {opt === 'adaml2' && ' Adam + L2 的衰减项被 √v̂ 放大到与梯度同量级，最终 loss 明显高于 AdamW。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '当前配置', tone: 'accent', variant: 'line' },
            { label: '对照：AdamW、lr 0.03、warmup+cosine、裁剪', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={opt}
            onChange={setOpt}
            options={[
              { value: 'sgd', label: 'SGD 动量' },
              { value: 'adaml2', label: 'Adam + L2' },
              { value: 'adamw', label: 'AdamW' },
            ]}
          />
          <Slider className="min-w-[160px] flex-1" label="峰值学习率" value={logLr} min={-3} max={0} step={0.05} onChange={setLogLr} format={() => lr.toPrecision(2)} />
          <Switch checked={sched} onChange={setSched} label="warmup + cosine" />
          <Switch checked={clip} onChange={setClip} label="梯度裁剪" />
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[1fr_200px]">
        <Plot height={250}>
          {(box) => (
            <>
              <Axes x={x} y={y} box={box} />
              <Line pts={base.losses} x={x} y={y} box={box} tone="muted" dashed width={1.8} />
              <Line pts={cur.losses} x={x} y={y} box={box} tone="accent" />
            </>
          )}
        </Plot>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label="最终 loss" value={diverged ? 'NaN' : cur.final.toFixed(3)} tone={diverged ? 'danger' : 'accent'} />
          <Stat label="训练准确率" value={diverged ? '—' : `${(cur.acc * 100).toFixed(0)}%`} />
          <Stat label="最大梯度范数" value={cur.gnMax.toFixed(2)} sub={clip ? '超过 1 的部分被裁剪' : '未裁剪'} />
          <Stat label="对照最终 loss" value={base.final.toFixed(3)} />
        </div>
      </div>
    </VizFrame>
  )
}
