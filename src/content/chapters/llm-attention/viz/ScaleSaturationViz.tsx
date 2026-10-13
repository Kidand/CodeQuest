/**
 * 为什么除以 √d_k：q、k 各分量 iid N(0,1) 时 Var(q·k) = d_k。
 * 拖动 d_k、开关缩放，观察 logits 标准差、softmax 是否饱和（接近 one-hot）以及梯度 p(1-p) 是否消失。
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { ProbBars, VizFrame, Legend } from '@/viz'
import { Button, Slider, Switch } from '@/components/ui'
import { cn } from '@/lib/utils'
import { entropy, gauss, softmax } from './math'

const N_KEYS = 8
const TRIALS = 60

function simulate(dk: number, scaled: boolean, seed: number) {
  const g = gauss(seed)
  const c = scaled ? 1 / Math.sqrt(dk) : 1
  let sumMax = 0
  let sumGrad = 0
  let sumSq = 0
  let cnt = 0
  let first: number[] = []
  let firstLogits: number[] = []
  for (let t = 0; t < TRIALS; t++) {
    const q = Array.from({ length: dk }, g)
    const logits: number[] = []
    for (let j = 0; j < N_KEYS; j++) {
      let s = 0
      for (let d = 0; d < dk; d++) s += q[d] * g()
      logits.push(s * c)
      sumSq += (s * c) ** 2
      cnt++
    }
    const p = softmax(logits)
    sumMax += Math.max(...p)
    // softmax 雅可比对角元 p(1-p) 的最大值：反映能回传多少梯度
    sumGrad += Math.max(...p.map((x) => x * (1 - x)))
    if (t === 0) {
      first = p
      firstLogits = logits
    }
  }
  return { std: Math.sqrt(sumSq / cnt), avgMax: sumMax / TRIALS, avgGrad: sumGrad / TRIALS, p: first, logits: firstLogits }
}

function Stat({ label, value, warn, note }: { label: string; value: string; warn?: boolean; note?: string }) {
  return (
    <div className={cn('rounded-xl border px-3 py-2', warn ? 'border-danger bg-danger-soft' : 'border-border bg-surface')}>
      <div className="text-[11px] text-muted">{label}</div>
      <div className={cn('font-mono text-[15px] font-semibold tabular-nums', warn ? 'text-danger' : 'text-fg')}>{value}</div>
      {note && <div className="text-[11px] text-subtle">{note}</div>}
    </div>
  )
}

export function ScaleSaturationViz() {
  const [exp, setExp] = useState(9) // d_k = 2^exp
  const [scaled, setScaled] = useState(false)
  const [seed, setSeed] = useState(3)
  const dk = 2 ** exp
  const sim = useMemo(() => simulate(dk, scaled, seed), [dk, scaled, seed])
  const saturated = sim.avgMax > 0.9
  const H = entropy(sim.p) / Math.log(N_KEYS)

  return (
    <VizFrame
      title="d_k 越大，不缩放的 softmax 越饱和"
      caption={
        scaled ? (
          <>
            除以 <span className="font-mono">√{dk} = {Math.sqrt(dk).toFixed(1)}</span> 后，logits 标准差稳定在 ≈1，与 d_k 无关；softmax 保持「软」的分布，梯度
            p(1−p) 不会消失。
          </>
        ) : (
          <>
            未缩放：logits 标准差 ≈ √d_k = <b className="font-mono">{Math.sqrt(dk).toFixed(1)}</b>。
            {saturated
              ? ' 最大权重平均已超过 0.9，softmax 接近 one-hot，几乎所有位置的梯度 p(1−p) 都趋近 0——训练初期就学不动了。'
              : ' d_k 还小，分布尚未饱和；继续把 d_k 拉大看看。'}
          </>
        )
      }
      legend={<Legend items={[{ label: '第 1 次抽样的注意力分布（8 个 key）', tone: 'accent' }, { label: '最大权重', tone: 'danger' }]} />}
      controls={
        <>
          <Slider className="min-w-[200px] flex-1" label="d_k（key 维度）" value={exp} min={2} max={12} onChange={setExp} format={(v) => `${2 ** v}`} />
          <Switch checked={scaled} onChange={setScaled} label="除以 √d_k" />
          <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
            重新抽样
          </Button>
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-[1.2fr_1fr]">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">
            q、k 各分量 iid N(0,1)，{scaled ? 'softmax(q·k/√d_k)' : 'softmax(q·k)'}
          </div>
          <ProbBars
            items={sim.p.map((p, j) => ({ label: `k${j}`, p, tone: p === Math.max(...sim.p) ? 'danger' : 'accent' }))}
            max={1}
            format={(p) => (p < 0.001 ? '≈0' : p.toFixed(3))}
          />
          <div className="mt-2 font-mono text-[11px] leading-relaxed text-subtle">
            logits = [{sim.logits.map((x) => x.toFixed(1)).join(', ')}]
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="logits 标准差（实测）" value={sim.std.toFixed(2)} note={scaled ? '理论值 1' : `理论值 √d_k = ${Math.sqrt(dk).toFixed(2)}`} warn={!scaled && sim.std > 6} />
          <Stat label="最大权重（60 次平均）" value={sim.avgMax.toFixed(3)} note="均匀分布为 0.125" warn={saturated} />
          <Stat label="max p(1−p)（60 次平均）" value={sim.avgGrad.toFixed(3)} note="越接近 0 梯度越小" warn={sim.avgGrad < 0.05} />
          <Stat label="归一化熵（第 1 次）" value={H.toFixed(2)} note="1 = 均匀，0 = one-hot" warn={H < 0.2} />
        </div>
      </div>
    </VizFrame>
  )
}
