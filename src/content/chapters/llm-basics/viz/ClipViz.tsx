/**
 * 全局范数梯度裁剪：total = √(Σ‖g_i‖²)，coef = min(1, max_norm / (total + 1e-6))，所有张量乘同一个系数。
 * 对比“逐张量裁剪”：每个张量各自裁到 max_norm，会改变整体梯度方向（余弦相似度 < 1）。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { DivBars, Stat } from './plot'

const GROUPS = [
  { name: 'emb', tone: 1 as const, g: [0.12, -0.08, 0.05, 0.1] },
  { name: 'attn', tone: 2 as const, g: [0.3, -0.22, 0.15, -0.1] },
  { name: 'ffn', tone: 3 as const, g: [0.25, 0.4, -0.35, 0.2] },
]
type Mode = 'none' | 'global' | 'tensor'
const norm = (v: number[]) => Math.sqrt(v.reduce((a, b) => a + b * b, 0))

export function ClipViz() {
  const [spike, setSpike] = useState(8)
  const [maxNorm, setMaxNorm] = useState(1)
  const [mode, setMode] = useState<Mode>('global')
  const res = useMemo(() => {
    const raw = GROUPS.map((G) => (G.name === 'ffn' ? G.g.map((x) => x * spike) : G.g.slice()))
    const norms = raw.map(norm)
    const total = Math.sqrt(norms.reduce((a, b) => a + b * b, 0))
    const coef = Math.min(1, maxNorm / (total + 1e-6))
    const clipped =
      mode === 'none'
        ? raw
        : mode === 'global'
          ? raw.map((v) => v.map((x) => x * coef))
          : raw.map((v, i) => v.map((x) => x * Math.min(1, maxNorm / (norms[i] + 1e-6))))
    const a = raw.flat()
    const b = clipped.flat()
    const cos = a.reduce((s, x, i) => s + x * b[i], 0) / (norm(a) * norm(b))
    const after = Math.sqrt(clipped.map(norm).reduce((s, x) => s + x * x, 0))
    return { raw, norms, total, coef, clipped, cos, after }
  }, [spike, maxNorm, mode])
  const maxAbs = Math.max(...res.raw.flat().map(Math.abs))

  return (
    <VizFrame
      title="全局范数裁剪：所有梯度乘同一个系数，方向不变"
      caption={
        <>
          三个张量的范数分别为 {res.norms.map((n) => n.toFixed(2)).join('、')}，全局范数 total = <b className="font-mono">{res.total.toFixed(2)}</b>。
          {mode === 'none' && ' 不裁剪：ffn 的梯度尖刺会让这一步的更新量失控。'}
          {mode === 'global' &&
            (res.coef < 1
              ? ` 全局裁剪系数 = ${maxNorm}/${res.total.toFixed(2)} = ${res.coef.toFixed(3)}，裁剪后总范数 = ${res.after.toFixed(2)}，方向余弦 = ${res.cos.toFixed(3)}（完全不变）。`
              : ' total ≤ max_norm，不触发裁剪。')}
          {mode === 'tensor' && ` 逐张量裁剪后，emb、attn 的相对比重被放大，方向余弦降到 ${res.cos.toFixed(3)}，总范数仍有 ${res.after.toFixed(2)}——这已经不是原来的下降方向了。`}
        </>
      }
      legend={<Legend items={GROUPS.map((G) => ({ label: G.name, tone: G.tone }))} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'none', label: '不裁剪' },
              { value: 'global', label: '全局范数裁剪' },
              { value: 'tensor', label: '逐张量裁剪' },
            ]}
          />
          <Slider className="min-w-[150px] flex-1" label="ffn 梯度尖刺倍数" value={spike} min={1} max={20} step={1} onChange={setSpike} format={(v) => `×${v}`} />
          <Slider className="min-w-[150px] flex-1" label="max_norm" value={maxNorm} min={0.25} max={5} step={0.25} onChange={setMaxNorm} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid gap-5 @3xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">① 原始梯度</div>
          <DivBars
            max={maxAbs}
            items={GROUPS.flatMap((G, gi) => G.g.map((_, k) => ({ label: `${G.name}${k}`, v: res.raw[gi][k], tone: G.tone })))}
          />
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">② {mode === 'none' ? '不裁剪' : mode === 'global' ? '全局裁剪后（同一坐标尺度）' : '逐张量裁剪后（同一坐标尺度）'}</div>
          <DivBars
            max={maxAbs}
            items={GROUPS.flatMap((G, gi) => G.g.map((_, k) => ({ label: `${G.name}${k}`, v: res.clipped[gi][k], tone: G.tone })))}
          />
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 @xl:grid-cols-4">
        <Stat label="全局范数 total" value={res.total.toFixed(3)} />
        <Stat label="裁剪系数" value={mode === 'global' ? res.coef.toFixed(3) : mode === 'tensor' ? '各张量不同' : '1'} tone="accent" />
        <Stat label="裁剪后总范数" value={res.after.toFixed(3)} />
        <Stat label="与原方向余弦" value={res.cos.toFixed(3)} tone={res.cos > 0.9999 ? 'success' : 'danger'} />
      </div>
    </VizFrame>
  )
}
