/**
 * 门控权重的三种做法：softmax→top-k 不归一 / softmax→top-k 重归一化 / sigmoid→top-k 归一化×缩放。
 * 洞见：不归一化时，权重和 = top-k 概率覆盖率，会随专家数 E 增大而变小；
 *       重归一化让权重和恒为 1，但 k=1 时权重恒等于 1，router 从主损失拿不到梯度。
 */
import { useMemo, useState } from 'react'
import { Legend, ProbBars, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { normals, sigmoid, softmax, topk } from './moeMath'
import { Stat } from './plot'

type Mode = 'raw' | 'norm' | 'sig'
const MODES: { value: Mode; label: string }[] = [
  { value: 'raw', label: 'softmax 不归一' },
  { value: 'norm', label: 'softmax 归一' },
  { value: 'sig', label: 'sigmoid 归一×2.5' },
]
const SHOW = 10

export function RouterGateViz() {
  const [E, setE] = useState(8)
  const [k, setK] = useState(2)
  const [mode, setMode] = useState<Mode>('norm')
  const logits = useMemo(() => normals(100 + E, E).map((v) => v * 1.2), [E])
  const scores = useMemo(() => (mode === 'sig' ? logits.map(sigmoid) : softmax(logits)), [logits, mode])
  const kk = Math.min(k, E)
  const idx = topk(scores, kk)
  const raw = idx.map((i) => scores[i])
  const rawSum = raw.reduce((a, b) => a + b, 0)
  const w = mode === 'raw' ? raw : raw.map((v) => (v / rawSum) * (mode === 'sig' ? 2.5 : 1))
  const wSum = w.reduce((a, b) => a + b, 0)

  const order = topk(scores, E)
  const shown = order.slice(0, Math.min(SHOW, E))
  const rest = order.slice(shown.length)
  const restSum = rest.reduce((a, i) => a + scores[i], 0)
  const deadGrad = mode !== 'raw' && kk === 1

  return (
    <VizFrame
      title="门控权重：归一化与否，差别在哪"
      caption={
        mode === 'raw' ? (
          <>
            不归一化：权重就是 top-{kk} 的 softmax 概率，和为 <b className="font-mono">{wSum.toFixed(3)}</b>。把 E 调到 256 看看——概率被摊薄，
            输出幅度随之缩小（DeepSeek-V2 用 routed_scaling_factor=16 放大，可理解为对此的补偿）。
          </>
        ) : mode === 'norm' ? (
          <>
            重归一化（Mixtral / Qwen3 的做法）：只在选中的 {kk} 个专家里再做一次归一化，权重和恒为 <b className="font-mono">1</b>，与 E 无关。
            {deadGrad && <b className="text-danger"> 注意 k=1：唯一的权重恒为 1，router 从主损失得不到任何梯度！</b>}
          </>
        ) : (
          <>
            DeepSeek-V3：sigmoid 给每个专家一个<b>独立</b>的亲和度（不相互竞争，和不为 1），选 top-{kk} 后归一化，再乘 routed_scaling_factor = 2.5，权重和恒为{' '}
            <b className="font-mono">{wSum.toFixed(2)}</b>。
            {deadGrad && <b className="text-danger"> k=1 时同样退化为常数权重。</b>}
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: '被选中的专家', tone: 'accent' },
            { label: '未选中', tone: 'muted' },
          ]}
        />
      }
      controls={
        <>
          <Segmented size="sm" value={mode} onChange={setMode} options={MODES} />
          <Segmented
            size="sm"
            value={E}
            onChange={setE}
            options={[8, 64, 256].map((v) => ({ value: v, label: `E=${v}` }))}
          />
          <Slider className="w-40" label="top-k" value={kk} min={1} max={8} onChange={setK} />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-1.5">
          <div className="text-xs font-medium text-muted">
            ① 路由分数 {mode === 'sig' ? 'sigmoid(logit)（各自独立，和不为 1）' : 'softmax(logits)'}，按大小排序
          </div>
          <ProbBars
            items={[
              ...shown.map((i) => ({ label: `E${i}`, p: scores[i], tone: idx.includes(i) ? ('accent' as const) : ('muted' as const), dim: !idx.includes(i) })),
              ...(rest.length ? [{ label: `其余${rest.length}个`, p: restSum, tone: 'muted' as const, dim: true, note: '合计' }] : []),
            ]}
            max={1}
          />
        </div>
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">② 最终门控权重 g（乘到专家输出上）</div>
          <ProbBars items={idx.map((i, j) => ({ label: `E${i}`, p: w[j], tone: 'accent' as const }))} max={mode === 'sig' ? 2.5 : 1} />
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Σ 门控权重" value={wSum.toFixed(3)} tone={mode === 'raw' && wSum < 0.5 ? 'danger' : undefined} />
            <Stat label={mode === 'sig' ? 'top-k 分数之和（归一化前）' : 'top-k 概率覆盖率'} value={rawSum.toFixed(3)} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
