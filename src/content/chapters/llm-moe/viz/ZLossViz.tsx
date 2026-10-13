/**
 * Router z-loss 的动机：bf16 只有 8 位有效精度，logits 绝对值越大，舍入步长越大，exp 之后误差被放大。
 * softmax 对整体平移不变，但 bf16 舍入误差不是——把 logits 整体加 100，概率就明显变了。
 * z-loss = (log Σ e^{x_j})² 把 logits 的整体水平拉回 0 附近。
 */
import { useMemo, useState } from 'react'
import { Legend, ProbBars, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { softmax } from './moeMath'
import { Stat } from './plot'

const D = [1.15, 0.9, 0.3, -0.4] // 各专家 logit 的相对差
const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)
/** 按 round-to-nearest-even 转为 bfloat16 再转回 */
function bf16(x: number) {
  f32[0] = x
  const b = u32[0]
  const r = (b + 0x7fff + ((b >>> 16) & 1)) & 0xffff0000
  u32[0] = r >>> 0
  return f32[0]
}
/** bf16 在 |x| 附近的相邻可表示数间距：2^(⌊log2|x|⌋ − 7) */
const ulp = (x: number) => (x === 0 ? 0 : 2 ** (Math.floor(Math.log2(Math.abs(x))) - 7))

export function ZLossViz() {
  const [offset, setOffset] = useState(80)
  const [amp, setAmp] = useState(1)
  const { exact, rounded, lse, maxErr, logits, q } = useMemo(() => {
    const logits = D.map((d) => offset + amp * d)
    const q = logits.map(bf16)
    const exact = softmax(logits)
    const rounded = softmax(q)
    const m = Math.max(...logits)
    const lse = m + Math.log(logits.reduce((a, v) => a + Math.exp(v - m), 0))
    const maxErr = Math.max(...exact.map((p, i) => Math.abs(p - rounded[i])))
    return { exact, rounded, lse, maxErr, logits, q }
  }, [offset, amp])

  return (
    <VizFrame
      title="z-loss：把 router logits 拉回 0 附近"
      caption={
        <>
          logits ≈ {offset.toFixed(0)} 附近时，bf16 的相邻可表示数间距为 <b className="font-mono">{ulp(Math.max(...logits.map(Math.abs))).toString()}</b>，
          舍入后 logits = [{q.map((v) => v.toString()).join(', ')}]，路由概率最大误差 <b className="font-mono">{maxErr.toFixed(3)}</b>。
          {Math.abs(offset) < 4 ? ' 整体水平在 0 附近时舍入误差可以忽略。' : ' 平移本不该改变 softmax，但舍入误差随 |logit| 增大——这就是 z-loss 要惩罚 log Z 的原因。'}
          此时 z-loss 项 (log Z)² = <b className="font-mono">{(lse * lse).toFixed(1)}</b>。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'fp32 精确概率', tone: 'accent' },
            { label: 'logits 先舍入到 bf16 再 softmax', tone: 'warning' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[200px] flex-1" label="logits 整体偏移（softmax 理论上不变）" value={offset} min={-100} max={100} step={1} onChange={setOffset} />
          <Slider className="w-44" label="logits 幅度" value={amp} min={0.25} max={4} step={0.05} onChange={setAmp} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-2">
          <div className="text-xs font-medium text-muted">fp32 计算（精确）</div>
          <ProbBars items={exact.map((p, i) => ({ label: `E${i}`, p, tone: 'accent' as const }))} max={1} />
          <div className="text-xs font-medium text-muted">bf16 logits 计算</div>
          <ProbBars items={rounded.map((p, i) => ({ label: `E${i}`, p, tone: 'warning' as const, note: `${p - exact[i] >= 0 ? '+' : ''}${(p - exact[i]).toFixed(3)}` }))} max={1} />
        </div>
        <div className="grid min-w-0 grid-cols-2 gap-2">
          <Stat label="log Z（logsumexp）" value={lse.toFixed(2)} />
          <Stat label="z-loss 项 (log Z)²" value={(lse * lse).toFixed(1)} tone={Math.abs(lse) > 10 ? 'danger' : 'success'} />
          <Stat label="bf16 间距（ulp）" value={ulp(Math.max(...logits.map(Math.abs))).toString()} />
          <Stat label="概率最大误差" value={maxErr.toFixed(4)} tone={maxErr > 0.01 ? 'danger' : 'success'} />
          <div className="col-span-2 text-xs leading-relaxed text-muted">
            bf16 有 8 位有效精度（7 位显式尾数），[64, 128) 内相邻两数相差 0.5，[0.5, 1) 内相差 1/256。ST-MoE 的 z-loss 系数 c_z = 0.001；Switch 还让 router 内部在 fp32 下计算（selective precision）。
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
