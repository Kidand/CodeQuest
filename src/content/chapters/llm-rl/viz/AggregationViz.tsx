/**
 * 同一组回答在不同 loss 聚合方式下，每个 token 的梯度权重（on-policy、β=0 时 −∂loss/∂log π）。
 * 拖动「错误长回答」的长度：GRPO 原式下它每个 token 的惩罚随长度 1/|o| 变小。
 */
import { useMemo, useState } from 'react'
import { Legend, ProbBars, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'

type Mode = 'grpo' | 'dapo' | 'drgrpo'
const MAX_TOKENS = 4096

const MODE_INFO: Record<Mode, { name: string; formula: string }> = {
  grpo: { name: 'seq-mean-token-mean（GRPO 原式）', formula: 'Â_i / (G·|o_i|)，Â 除以 std' },
  dapo: { name: 'token-mean（DAPO）', formula: 'Â_i / Σ_j|o_j|，Â 除以 std' },
  drgrpo: { name: 'seq-mean-token-sum-norm（Dr.GRPO）', formula: '(r_i − mean) / (G·MAX_TOKENS)' },
}

export function AggregationViz() {
  const [mode, setMode] = useState<Mode>('grpo')
  const [longLen, setLongLen] = useState(3200)
  const resp = useMemo(
    () => [
      { name: 'o1 对·短', ok: true, len: 400 },
      { name: 'o2 对·长', ok: true, len: 1600 },
      { name: 'o3 错·短', ok: false, len: 500 },
      { name: 'o4 错·长', ok: false, len: longLen },
    ],
    [longLen],
  )
  const G = resp.length
  const r: number[] = resp.map((x) => (x.ok ? 1 : 0))
  const mean = r.reduce((a, b) => a + b, 0) / G
  const std = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / (G - 1))
  const nTok = resp.reduce((a, b) => a + b.len, 0)
  const perTok = resp.map((x, i) => {
    const dev = r[i] - mean
    if (mode === 'grpo') return dev / std / (G * x.len)
    if (mode === 'dapo') return dev / std / nTok
    return dev / (G * MAX_TOKENS)
  })
  const total = perTok.map((w, i) => w * resp[i].len)
  const maxTok = Math.max(...perTok.map(Math.abs))
  const maxTot = Math.max(...total.map(Math.abs))
  const ratio = Math.abs(perTok[3] / perTok[2])

  return (
    <VizFrame
      title="loss 聚合方式 = 给每个 token 分配梯度权重"
      caption={
        <>
          {MODE_INFO[mode].name}：每 token 权重 = {MODE_INFO[mode].formula}。
          「错·长」与「错·短」的<b>每 token 惩罚之比</b> = <b className="font-mono">{ratio.toFixed(2)}</b>
          {mode === 'grpo'
            ? `（= 500/${longLen}）：越长越罚不动，而每条回答的总推力相同——错误回答“变长”就能稀释惩罚。`
            : '：同样的 token 受同样的惩罚，长度不再改变每个 token 的待遇。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '正权重：推高概率', tone: 'success' },
            { label: '负权重：压低概率', tone: 'danger' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'grpo', label: 'GRPO' },
              { value: 'dapo', label: 'DAPO token-mean' },
              { value: 'drgrpo', label: 'Dr.GRPO' },
            ]}
          />
          <Slider className="min-w-[200px] flex-1" label="o4（错误回答）长度" value={longLen} min={500} max={4000} step={100} onChange={setLongLen} />
        </>
      }
    >
      <div className="grid gap-5 @3xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">① 每个 token 的梯度权重（×10⁴）</div>
          <ProbBars
            items={resp.map((x, i) => ({ label: x.name, p: perTok[i] * 1e4, tone: perTok[i] > 0 ? 'success' : 'danger', note: `${x.len} tok` }))}
            max={maxTok * 1e4}
            format={(v) => (v > 0 ? '+' : '') + v.toFixed(3)}
          />
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">② 整条回答的总推力 = 权重 × 长度</div>
          <ProbBars
            items={resp.map((x, i) => ({ label: x.name, p: total[i], tone: total[i] > 0 ? 'success' : 'danger' }))}
            max={maxTot}
            format={(v) => (v > 0 ? '+' : '') + v.toFixed(3)}
          />
        </div>
      </div>
    </VizFrame>
  )
}
