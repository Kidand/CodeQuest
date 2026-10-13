/**
 * softmax + 交叉熵对 logits 的梯度 = p − q（q 为目标分布）。
 * label smoothing：q = (1−ε)·onehot + ε/C。ε>0 时最优 p_y = 1−ε+ε/C < 1，
 * 所以当 z_y 很大时梯度会把它“推回来”——这就是 label smoothing 抑制 logit 无限增长的机制。
 */
import { useMemo, useState } from 'react'
import { Legend, ProbBars, VizFrame } from '@/viz'
import { Slider } from '@/components/ui'
import { DivBars, Stat } from './plot'

const TOK = ['cat', 'dog', 'car', 'sky', 'run']
const BASE = [1.2, 0.6, -0.3, 0.1, -0.8]

export function CEGradViz() {
  const [y, setY] = useState(0)
  const [zy, setZy] = useState(1.2)
  const [eps, setEps] = useState(0)
  const C = TOK.length
  const z = useMemo(() => BASE.map((v, i) => (i === y ? zy : v)), [y, zy])
  const { p, q, loss, g } = useMemo(() => {
    const m = Math.max(...z)
    const e = z.map((v) => Math.exp(v - m))
    const s = e.reduce((a, b) => a + b, 0)
    const p = e.map((v) => v / s)
    const logp = z.map((v) => v - m - Math.log(s))
    const q = z.map((_, i) => (i === y ? 1 - eps : 0) + eps / C)
    const loss = -q.reduce((acc, qi, i) => acc + qi * logp[i], 0)
    const g = p.map((pi, i) => pi - q[i])
    return { p, q, loss, g }
  }, [z, y, eps, C])
  const best = 1 - eps + eps / C
  const push = g[y] > 1e-4 ? '梯度 g_y > 0：梯度下降会把 z_y 往下拉（已经“过度自信”了）' : g[y] < -1e-4 ? '梯度 g_y < 0：梯度下降会继续增大 z_y' : '梯度 ≈ 0：p 已等于目标分布'

  return (
    <VizFrame
      title="交叉熵对 logits 的梯度就是 p − q"
      caption={
        <>
          目标类 <b className="font-mono">{TOK[y]}</b>，z_y = <b className="font-mono">{zy.toFixed(1)}</b>，ε = <b className="font-mono">{eps.toFixed(2)}</b>：
          loss = −Σ q·log p = <b className="font-mono">{loss.toFixed(3)}</b>。{push}。
          {eps > 0 && ` 有 label smoothing 时最优的 p_y = 1−ε+ε/C = ${best.toFixed(3)}，而不是 1。`}
          {' '}点击左侧 token 可更换目标类。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '目标类', tone: 'success' },
            { label: '梯度 > 0（logit 将减小）', tone: 'danger' },
            { label: '梯度 < 0（logit 将增大）', tone: 'info' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="min-w-[170px] flex-1" label="目标类 logit z_y" value={zy} min={-4} max={10} step={0.1} onChange={setZy} format={(v) => v.toFixed(1)} />
          <Slider className="min-w-[170px] flex-1" label="label smoothing ε" value={eps} min={0} max={0.3} step={0.01} onChange={setEps} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid gap-5 @3xl:grid-cols-3">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">① 预测 p = softmax(z)（点击选目标类）</div>
          <div className="space-y-1.5">
            {TOK.map((t, i) => (
              <button
                key={t}
                type="button"
                onClick={() => {
                  setY(i)
                  setZy(BASE[i])
                }}
                className="block w-full text-left"
                aria-label={`选择 ${t} 为目标类`}
              >
                <ProbBars items={[{ label: t, p: p[i], tone: i === y ? 'success' : 'accent' }]} max={1} />
              </button>
            ))}
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">② 目标 q = (1−ε)·onehot + ε/C</div>
          <ProbBars items={TOK.map((t, i) => ({ label: t, p: q[i], tone: i === y ? 'success' : 'muted' }))} max={1} />
        </div>
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">③ ∂loss/∂z = p − q（各分量之和恒为 0）</div>
          <DivBars items={TOK.map((t, i) => ({ label: t, v: g[i] }))} max={1} />
          <div className="grid grid-cols-2 gap-2">
            <Stat label="loss" value={loss.toFixed(3)} />
            <Stat label="Σ(p − q)" value={Math.abs(g.reduce((a, b) => a + b, 0)).toFixed(3)} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
