/**
 * GRPO 的 token 级 ratio vs GSPO 的序列级 ratio s = exp(mean_t log r_t)。
 * 打开「路由抖动」给某个 token 注入大偏移，看两者的裁剪行为差异。
 */
import { useMemo, useState } from 'react'
import { ArrayView, Legend, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { Stat } from './plot'

// 均值为 0：正常更新下各 token 的 log ratio 有正有负，几何平均几乎为 1
const BASE = [0.03, -0.05, 0.08, -0.02, 0.12, -0.09, 0.04, 0.01, -0.06, 0.05].map((v) => v - 0.011)
const TOK = ['Let', 'x', '=', '3', ',', 'then', 'Wait', ',', 'so', '9']

export function SeqRatioViz() {
  const [drift, setDrift] = useState(0)
  const [spike, setSpike] = useState<'off' | 'on'>('off')
  const [sign, setSign] = useState<'pos' | 'neg'>('pos')
  const logr = useMemo(() => BASE.map((v, i) => v + drift + (spike === 'on' && i === 6 ? 0.9 : 0)), [drift, spike])
  const r = logr.map(Math.exp)
  const s = Math.exp(logr.reduce((a, b) => a + b, 0) / logr.length)
  const A = sign === 'pos' ? 1 : -1
  // GRPO 基线（GSPO 论文中调过的）：ε_low=0.2, ε_high=0.27；GSPO：3e-4 / 4e-4
  const tokClipped = r.map((x) => (A > 0 ? x > 1.27 : x < 0.8))
  const seqClipped = A > 0 ? s > 1 + 4e-4 : s < 1 - 3e-4
  const nClip = tokClipped.filter(Boolean).length

  return (
    <VizFrame
      title="序列级 ratio：同一条回答的所有 token 共用一个权重 s"
      caption={
        <>
          GRPO：{nClip} / 10 个 token 被裁剪（梯度为 0），其余 token 各自带着不同的权重 r_t。
          GSPO：s = exp(mean log r) = <b className="font-mono">{s.toFixed(4)}</b>，
          {seqClipped ? <b>整条回答被裁剪</b> : <>整条回答<b>未被裁剪</b>，所有 token 共用同一个权重 s</>}。
          {spike === 'on' &&
            ` 路由抖动让 “Wait” 的 r 跳到 ${r[6].toFixed(2)}：GRPO 中 A>0 时它被裁掉，A<0 时它带着 ${r[6].toFixed(2)} 倍的权重（无上界）；GSPO 中它只把 s 推到约 exp(0.09)，所有 token 权重相同（A>0 时整条回答超出可信域、被排除）。`}
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'GRPO 中被裁剪的 token', tone: 'danger', variant: 'soft' },
            { label: '可信域内', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={sign}
            onChange={setSign}
            options={[
              { value: 'pos', label: 'A > 0' },
              { value: 'neg', label: 'A < 0' },
            ]}
          />
          <Segmented
            size="sm"
            value={spike}
            onChange={setSpike}
            options={[
              { value: 'off', label: '无抖动' },
              { value: 'on', label: '“Wait” 处路由抖动' },
            ]}
          />
          <Slider className="min-w-[180px] flex-1" label="整体漂移 Δlog r" value={drift} min={-0.001} max={0.001} step={0.00005} onChange={setDrift} format={(v) => v.toFixed(5)} />
        </>
      }
    >
      <div className="grid gap-4 @3xl:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">token 级 ratio r_t（GRPO 可信域 [0.8, 1.27]）</div>
          <ArrayView
            values={r}
            format={(v) => Number(v).toFixed(2)}
            cellSize={50}
            highlights={Object.fromEntries(tokClipped.map((c, i) => [i, c ? 'danger' : 'success'] as const))}
            showIndex={false}
          />
          <div className="flex flex-wrap justify-center gap-1 font-mono text-[11px] text-muted">
            {TOK.map((t, i) => (
              <span key={i} className="w-[46px] truncate text-center">
                {t}
              </span>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2 @3xl:grid-cols-1">
          <Stat label="GSPO 序列 ratio s" value={s.toFixed(4)} tone={seqClipped ? 'danger' : 'success'} />
          <Stat label="GSPO 可信域" value="[0.9997, 1.0004]" tone="warning" />
          <Stat label="GRPO 裁剪 token 数" value={`${nClip} / 10`} tone={nClip ? 'danger' : 'success'} />
        </div>
      </div>
    </VizFrame>
  )
}
