/**
 * 大模型场景：按概率采样下一个 token = 在前缀和（CDF）上二分。
 * 样板要点：参数驱动型可视化（无 stepper），拖动滑块实时重算；把经典算法和大模型岗位场景连起来。
 */
import { useMemo, useState } from 'react'
import { Dices } from 'lucide-react'
import { ArrayView, Legend, ProbBars, VizFrame, type ArrayPointer } from '@/viz'
import { Button, Slider } from '@/components/ui'

const TOKENS = ['the', 'a', 'cat', 'dog', 'sat', 'ran', '<eos>']
const LOGITS = [2.1, 1.6, 0.9, 0.7, 0.2, -0.4, -1.2]

function softmax(xs: number[], t: number) {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp((x - m) / t))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export function CdfSamplingViz() {
  const [u, setU] = useState(0.62)
  const [temp, setTemp] = useState(1)
  const probs = useMemo(() => softmax(LOGITS, temp), [temp])
  const cdf = useMemo(() => probs.reduce<number[]>((acc, p) => [...acc, (acc.at(-1) ?? 0) + p], []), [probs])

  // 二分：第一个 cdf[i] > u
  const trace: { lo: number; hi: number; mid: number; ok: boolean }[] = []
  let lo = 0
  let hi = cdf.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    const ok = cdf[mid] > u
    trace.push({ lo, hi, mid, ok })
    if (ok) hi = mid
    else lo = mid + 1
  }
  const pick = lo
  const pointers: ArrayPointer[] = [{ index: pick, label: `采样 → ${TOKENS[pick]}`, tone: 'success' }]

  return (
    <VizFrame
      title="逆 CDF 采样：把 u ~ U[0,1) 映射到 token"
      caption={
        <>
          u = <b className="font-mono">{u.toFixed(3)}</b>，第一个满足 <span className="font-mono">cdf[i] &gt; u</span> 的下标是{' '}
          <b className="font-mono">{pick}</b>（<span className="font-mono">{TOKENS[pick]}</span>）。二分探查了 {trace.length} 次：
          <span className="font-mono text-muted">
            {' '}
            {trace.map((t) => `mid=${t.mid}(${t.ok ? '>' : '≤'})`).join(' → ') || '—'}
          </span>
        </>
      }
      legend={<Legend items={[{ label: '被采样的 token', tone: 'success', variant: 'soft' }, { label: '其它 token', tone: 'accent' }]} />}
      controls={
        <>
          <Slider className="min-w-[180px] flex-1" label="随机数 u" value={u} min={0} max={0.999} step={0.001} onChange={setU} format={(v) => v.toFixed(3)} />
          <Slider className="min-w-[160px] flex-1" label="温度 T" value={temp} min={0.2} max={2.5} step={0.05} onChange={setTemp} format={(v) => v.toFixed(2)} />
          <Button size="sm" variant="secondary" icon={<Dices size={14} />} onClick={() => setU(Math.floor(Math.random() * 1000) / 1000)}>
            掷一次
          </Button>
        </>
      }
    >
      <div className="grid gap-6 @2xl:grid-cols-2">
        <div>
          <div className="mb-2 text-xs font-medium text-muted">① softmax(logits / T) 得到概率</div>
          <ProbBars items={TOKENS.map((t, i) => ({ label: t, p: probs[i], tone: i === pick ? 'success' : 'accent' }))} max={1} />
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">② 前缀和 cdf（单调递增 ⇒ 可二分）</div>
          <div className="overflow-x-auto py-1">
            <ArrayView values={cdf} cellSize={46} format={(v) => Number(v).toFixed(2)} highlights={{ [pick]: 'success' }} pointers={pointers} />
          </div>
          {/* u 在 [0,1) 数轴上的位置 */}
          <div className="relative mt-4 h-8">
            <div className="absolute inset-x-0 top-3 flex h-2 overflow-hidden rounded-full">
              {probs.map((p, i) => (
                <div key={i} style={{ width: `${p * 100}%`, background: i === pick ? 'var(--success)' : i % 2 ? 'var(--surface-3)' : 'var(--border-strong)' }} />
              ))}
            </div>
            <div className="absolute top-0 h-8 w-0.5 bg-fg" style={{ left: `${u * 100}%` }} />
            <div className="absolute -bottom-3 -translate-x-1/2 font-mono text-[10px] text-muted" style={{ left: `${u * 100}%` }}>
              u
            </div>
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
