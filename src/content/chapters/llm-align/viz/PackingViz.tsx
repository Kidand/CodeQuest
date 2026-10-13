/**
 * Packing：把 3 条样本拼进一行。对比三种做法：
 *   naive  —— 位置连续 + 普通因果掩码：跨样本注意力 + 跨样本预测
 *   pos    —— 只重置 position_ids：RoPE 位置对了，但注意力仍然串门
 *   varlen —— 重置 position_ids + 块对角因果掩码（FlashAttention varlen / cu_seqlens）
 */
import { useState } from 'react'
import { Legend, MatrixView, TokenRow, VizFrame, toneColor, toneSoft, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './plot'

const SAMPLES = [
  ['a1', 'a2', 'a3'],
  ['b1', 'b2', 'b3'],
  ['c1', 'c2'],
]
const TOKS = SAMPLES.flat()
const SEG = SAMPLES.flatMap((s, k) => s.map(() => k))
const START = new Set(SAMPLES.reduce<number[]>((acc, s, k) => [...acc, k === 0 ? 0 : acc[k - 1] + SAMPLES[k - 1].length], []))
const CU = SAMPLES.reduce<number[]>((acc, s) => [...acc, acc[acc.length - 1] + s.length], [0])
const TONE: VizTone[] = [1, 2, 3]
type Mode = 'naive' | 'pos' | 'varlen'

export function PackingViz() {
  const [mode, setMode] = useState<Mode>('naive')
  const [q, setQ] = useState(4)
  const pos = mode === 'naive' ? TOKS.map((_, i) => i) : SAMPLES.flatMap((s) => s.map((_, i) => i))
  const allowed = (r: number, c: number) => c <= r && (mode !== 'varlen' || SEG[r] === SEG[c])
  const leaks = TOKS.reduce((n, _, r) => n + TOKS.filter((__, c) => allowed(r, c) && SEG[r] !== SEG[c]).length, 0)
  const sep = mode !== 'naive'
  const sees = TOKS.filter((_, c) => allowed(q, c))
  const bad = TOKS.filter((_, c) => allowed(q, c) && SEG[c] !== SEG[q])

  return (
    <VizFrame
      title="Packing：位置要重置，注意力要隔离"
      badge={<span className="shrink-0 font-mono text-xs" style={{ color: leaks ? toneColor('danger') : toneColor('success') }}>泄漏 {leaks} 处</span>}
      caption={
        <>
          点中的 query <b className="font-mono">{TOKS[q]}</b>（position_id = {pos[q]}）能看到：<span className="font-mono">{sees.join(' ')}</span>。
          {bad.length > 0 ? (
            <> 其中 <span className="font-mono" style={{ color: toneColor('danger') }}>{bad.join(' ')}</span> 属于别的样本——训练时它“偷看”了无关上下文，推理时却看不到，形成训推不一致。</>
          ) : (
            <> 只看得到本样本的前文，与单独训练这条样本完全等价。</>
          )}
          {mode === 'pos' && ' 只改 position_ids 只影响 RoPE，不会改变注意力掩码。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '样本 A / B / C', tone: 1, variant: 'soft' },
            { label: '跨样本（泄漏）', tone: 'danger', variant: 'soft' },
            { label: '被掩码', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <Segmented
          size="sm"
          value={mode}
          onChange={setMode}
          options={[
            { value: 'naive', label: '朴素拼接' },
            { value: 'pos', label: '只重置 position_ids' },
            { value: 'varlen', label: '重置 + 块对角掩码' },
          ]}
        />
      }
    >
      <div className="grid items-start gap-5 @2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="flex min-w-0 flex-col gap-3">
          <div>
            <div className="mb-1 text-xs font-medium text-muted">input_ids（小字 = position_ids）</div>
            <TokenRow tokens={TOKS.map((t, i) => ({ text: t, tone: TONE[SEG[i]], sub: String(pos[i]) }))} onClick={setQ} />
          </div>
          <div>
            <div className="mb-1 text-xs font-medium text-muted">labels（模型内部再 shift）</div>
            <div className="flex flex-wrap gap-1.5">
              {TOKS.map((t, i) => {
                const isSep = sep && START.has(i)
                const cross = !sep && START.has(i) && i > 0
                return (
                  <span
                    key={i}
                    className="rounded-lg border px-2 py-1 font-mono text-[13px] leading-tight"
                    style={{
                      borderColor: cross ? toneColor('danger') : isSep ? 'var(--border-strong)' : toneColor(TONE[SEG[i]]),
                      background: cross ? toneSoft('danger', 22) : isSep ? 'var(--surface-2)' : toneSoft(TONE[SEG[i]], 18),
                      color: isSep ? 'var(--subtle)' : 'var(--fg)',
                    }}
                  >
                    {isSep ? '-100' : t}
                  </span>
                )
              })}
            </div>
            <div className="mt-1.5 text-[12px] leading-relaxed text-muted">
              {sep ? (
                <>每条样本的第一个 label 置为 -100（HF <C>DataCollatorWithFlattening</C> 的 separator），上一条的最后一个 token 不再去预测下一条的开头。</>
              ) : (
                <>红框：b1 由 a3 的位置预测、c1 由 b3 预测——跨样本的“下一个 token”被当成监督信号。</>
              )}
            </div>
          </div>
          {mode === 'varlen' && (
            <div className="rounded-xl border border-border bg-surface px-3 py-2 font-mono text-[12.5px] text-fg">
              cu_seqlens = [{CU.join(', ')}] · max_seqlen = {Math.max(...SAMPLES.map((s) => s.length))}
            </div>
          )}
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium text-muted">注意力掩码（行 = query，列 = key；点击某行）</div>
          <MatrixView
            data={TOKS.map((_, r) => TOKS.map((__, c) => (allowed(r, c) ? 1 : null)))}
            rowLabels={TOKS.map((t, i) => (i === q ? <b key={i} style={{ color: toneColor('accent') }}>▶ {t}</b> : t))}
            colLabels={TOKS}
            scale="none"
            showValues={false}
            cellSize={30}
            masked={(r, c) => !allowed(r, c)}
            highlight={(r, c) => (allowed(r, c) ? (SEG[r] === SEG[c] ? TONE[SEG[r]] : 'danger') : null)}
            onCellClick={(r) => setQ(r)}
          />
        </div>
      </div>
    </VizFrame>
  )
}
