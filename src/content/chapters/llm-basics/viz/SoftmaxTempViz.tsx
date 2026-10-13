/**
 * softmax 的两个基本性质：
 *  1) 平移不变：所有 logit 加同一个常数 c，概率不变（这正是“减最大值”合法的原因）
 *  2) 温度：T→0 趋向 one-hot（argmax），T→∞ 趋向均匀分布；熵随 T 单调增
 */
import { useMemo, useState } from 'react'
import { ArrayView, Legend, ProbBars, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { Stat } from './plot'

const TOK = ['the', 'a', 'cat', 'dog', 'sat', '<eos>']
const Z = [3.2, 2.5, 1.8, 0.4, -0.6, -1.5]

function softmax(z: number[]) {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export function SoftmaxTempViz() {
  const [logT, setLogT] = useState(0) // T = 10^logT
  const [c, setC] = useState(0)
  const T = 10 ** logT
  const shifted = useMemo(() => Z.map((v) => v + c), [c])
  const p = useMemo(() => softmax(shifted.map((v) => v / T)), [shifted, T])
  const H = -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0)
  const Hmax = Math.log(Z.length)
  const top = p.indexOf(Math.max(...p))

  return (
    <VizFrame
      title="softmax：平移不变，温度控制“尖锐程度”"
      caption={
        <>
          T = <b className="font-mono">{T.toFixed(2)}</b>，所有 logit 再加 c = <b className="font-mono">{c}</b>。
          {c !== 0 && ' 注意概率条一点没变——softmax(z + c) = softmax(z)。'}
          {T < 0.3 && ' 温度很低：几乎所有概率都压到最大 logit 上，接近 greedy（argmax）。'}
          {T > 2.5 && ' 温度很高：分布接近均匀，熵逼近上限 ln 6 ≈ 1.792。'}
          {T >= 0.3 && T <= 2.5 && c === 0 && ' 拖动温度看分布变尖或变平；拖动 c 验证平移不变性。'}
        </>
      }
      legend={<Legend items={[{ label: '概率最大的 token', tone: 'success', variant: 'soft' }, { label: '其它 token', tone: 'accent' }]} />}
      controls={
        <>
          <Slider className="min-w-[180px] flex-1" label="温度 T（对数刻度）" value={logT} min={-1.3} max={0.85} step={0.01} onChange={setLogT} format={() => T.toFixed(2)} />
          <Slider className="min-w-[180px] flex-1" label="平移常数 c" value={c} min={-50} max={50} step={1} onChange={setC} />
        </>
      }
    >
      <div className="grid gap-5 @2xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">① logits z + c（除以 T 之前）</div>
          <ArrayView values={shifted} cellSize={52} format={(v) => Number(v).toFixed(1)} highlights={{ [top]: 'success' }} showIndex={false} />
          <div className="grid grid-cols-3 gap-2">
            <Stat label="熵 H(p)" value={H.toFixed(3)} sub={`上限 ln6=${Hmax.toFixed(3)}`} />
            <Stat label="max p" value={p[top].toFixed(3)} tone="success" />
            <Stat label="argmax" value={TOK[top]} />
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">② p = softmax((z + c) / T)</div>
          <ProbBars items={TOK.map((t, i) => ({ label: t, p: p[i], tone: i === top ? 'success' : 'accent' }))} max={1} />
        </div>
      </div>
    </VizFrame>
  )
}
