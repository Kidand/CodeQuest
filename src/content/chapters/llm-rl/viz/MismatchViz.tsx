/**
 * 训推不一致的简化模拟：训练端用 fp64 精确计算 softmax；推理端把 logits 舍入到 BF16 / FP16 再算。
 * 只模拟了「logits 舍入」这一种误差来源（真实系统还有 kernel 累加顺序、算子融合等差异）。
 * 洞见：logit 幅度越大、尾数位越少，同一 token 的概率差越大；TIS 用 min(p_train/p_infer, C) 修正。
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Button, Segmented, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { Axes, Svg, Stat, useScales, type Axis } from './plot'

const V = 48
const POS = 60

/** 把 float32 的尾数舍入到 m 位（round-to-nearest-even），忽略指数范围差异 */
function roundMantissa(x: number, m: number) {
  const buf = new DataView(new ArrayBuffer(4))
  buf.setFloat32(0, x)
  const bits = buf.getUint32(0)
  const drop = 23 - m
  const half = 1 << (drop - 1)
  const lsb = (bits >>> drop) & 1
  const rounded = ((bits + half - 1 + lsb) >>> drop) << drop
  buf.setUint32(0, rounded >>> 0)
  return buf.getFloat32(0)
}

function softmax(z: number[]) {
  const mx = Math.max(...z)
  const e = z.map((v) => Math.exp(v - mx))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

function gauss(r: () => number) {
  const u = Math.max(1e-12, r())
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r())
}

export function MismatchViz() {
  const [fmt, setFmt] = useState<'bf16' | 'fp16'>('bf16')
  const [scale, setScale] = useState(24)
  const [C, setC] = useState(1.05)
  const [seed, setSeed] = useState(1)
  const pts = useMemo(() => {
    const r = rng(seed)
    const mBits = fmt === 'bf16' ? 7 : 10
    const out: { pt: number; pi: number }[] = []
    for (let k = 0; k < POS; k++) {
      // 分布形状固定（logit 差异的尺度约为 2），「幅度」只是给所有 logits 加一个大的公共偏移：
      // softmax 在精确算术下对偏移不变，但舍入误差与数值大小成正比
      const z = Array.from({ length: V }, () => scale + 2 * gauss(r))
      const pTrain = softmax(z)
      const pInfer = softmax(z.map((v) => roundMantissa(v, mBits)))
      // 按推理端分布采样一个 token（rollout 来自推理引擎）
      let u = r()
      let tok = V - 1
      for (let v = 0; v < V; v++) {
        u -= pInfer[v]
        if (u <= 0) {
          tok = v
          break
        }
      }
      out.push({ pt: pTrain[tok], pi: pInfer[tok] })
    }
    return out
  }, [fmt, scale, seed])

  const lr = pts.map((p) => Math.log(p.pt / p.pi))
  const maxAbs = Math.max(...lr.map(Math.abs))
  const nTrunc = pts.filter((p) => p.pt / p.pi > C).length
  const x: Axis = { min: 1e-4, max: 1, log: true, ticks: [1e-4, 1e-3, 1e-2, 0.1, 1], label: 'p_infer（推理引擎）', fmt: (v) => (v >= 0.01 ? String(v) : v.toExponential(0)) }
  const y: Axis = { min: 1e-4, max: 1, log: true, ticks: [1e-4, 1e-3, 1e-2, 0.1, 1], label: 'p_train', fmt: (v) => (v >= 0.01 ? String(v) : v.toExponential(0)) }
  const { sx, sy } = useScales(x, y)
  const cl = (v: number) => Math.max(1e-4, Math.min(1, v))

  return (
    <VizFrame
      title="训推不一致：同一权重，两个引擎算出的概率不同"
      caption={
        <>
          {POS} 个采样 token 中，|log(p_train/p_infer)| 最大为 <b className="font-mono">{maxAbs.toFixed(3)}</b>（比值约 {Math.exp(maxAbs).toFixed(2)}×）。
          把 logit 偏移调大或切到 BF16（7 位尾数）误差变大，FP16（10 位尾数）小一个数量级。
          TIS 权重 min(p_train/p_infer, C=<b>{C}</b>) 截断了 <b>{nTrunc}</b> 个 token。（简化模拟：只对 logits 舍入，真实差异还来自 kernel 实现。）
        </>
      }
      legend={
        <Legend
          items={[
            { label: '采样 token（虚线 y = x 表示完全一致）', tone: 'accent' },
            { label: 'p_train/p_infer > C（TIS 截断）', tone: 'danger' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={fmt}
            onChange={setFmt}
            options={[
              { value: 'bf16', label: '推理端 BF16' },
              { value: 'fp16', label: '推理端 FP16' },
            ]}
          />
          <Slider className="w-40" label="logit 偏移" value={scale} min={2} max={60} step={1} onChange={setScale} />
          <Slider className="w-32" label="TIS 上限 C" value={C} min={1.01} max={2} step={0.01} onChange={setC} format={(v) => v.toFixed(2)} />
          <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
            重新采样
          </Button>
        </>
      }
    >
      <div className="grid items-center gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <Svg>
          <Axes x={x} y={y} />
          <line x1={sx(1e-4)} y1={sy(1e-4)} x2={sx(1)} y2={sy(1)} stroke="var(--border-strong)" strokeDasharray="4 3" />
          {pts.map((p, i) => (
            <circle key={i} cx={sx(cl(p.pi))} cy={sy(cl(p.pt))} r={3.6} fill={toneColor(p.pt / p.pi > C ? 'danger' : 'accent')} opacity={0.8} />
          ))}
        </Svg>
        <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
          <Stat label="最大 |log 比值|" value={maxAbs.toFixed(3)} tone="danger" />
          <Stat label="平均 |log 比值|" value={(lr.reduce((a, b) => a + Math.abs(b), 0) / lr.length).toFixed(4)} />
          <Stat label="TIS 截断数" value={`${nTrunc} / ${POS}`} tone="warning" />
        </div>
      </div>
    </VizFrame>
  )
}
