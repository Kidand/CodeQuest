/**
 * 增量解码：有 KV cache 时每步只为新 token 算一行 K/V，并用 1 个 query 去查全部缓存；
 * 无 cache 时每步重算整个前缀的 K/V 和 t×t 注意力，却只用最后一行。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, MatrixView, StepControls, TokenRow, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { dot, randMatrix, softmax } from './math'

const PROMPT = ['我', '爱', '北', '京']
const GEN = ['天', '安', '门', '。']
const ALL = [...PROMPT, ...GEN]
const D = 4
const X = randMatrix(ALL.length, D, 21)
const WQ = randMatrix(D, D, 22, 0.7)
const WK = randMatrix(D, D, 23, 0.7)
const WV = randMatrix(D, D, 24, 0.7)
const mv = (x: number[], W: number[][]) => W[0].map((_, c) => x.reduce((s, xi, r) => s + xi * W[r][c], 0))
const Kall = X.map((x) => mv(x, WK))
const Vall = X.map((x) => mv(x, WV))
const Qall = X.map((x) => mv(x, WQ))

type Mode = 'cache' | 'nocache'
interface Frame {
  t: number // 当前序列长度（含新 token）
  prefill: boolean
  line: number[]
  kvOps: number
  attnOps: number
}

const CODE: Record<Mode, string> = {
  cache: `# prefill：一次性处理整个 prompt
K, V = X @ Wk, X @ Wv               # 缓存 (T0, d)
for step in range(max_new):
    x_t = embed(next_tok)           # (1, d) 只有新 token
    q = x_t @ Wq
    K = concat([K, x_t @ Wk])       # 追加一行
    V = concat([V, x_t @ Wv])
    p = softmax(q @ K.T / sqrt(d))  # (1, t)
    out = p @ V                     # (1, d)`,
  nocache: `# 无缓存：每步把整个前缀重新跑一遍
for step in range(max_new):
    X = embed(all_tokens)           # (t, d) 整个前缀
    Q, K, V = X @ Wq, X @ Wk, X @ Wv  # 重算全部 K/V
    S = causal(Q @ K.T / sqrt(d))   # (t, t)
    out = (softmax(S) @ V)[-1]      # 只用最后一行！`,
}

function buildFrames(mode: Mode): Frame[] {
  const fr: Frame[] = []
  let kvOps = 0
  let attnOps = 0
  // prefill 两种模式一样
  kvOps += PROMPT.length
  attnOps += (PROMPT.length * (PROMPT.length + 1)) / 2
  fr.push({ t: PROMPT.length, prefill: true, line: mode === 'cache' ? [2] : [3, 4], kvOps, attnOps })
  for (let s = 1; s <= GEN.length; s++) {
    const t = PROMPT.length + s
    if (mode === 'cache') {
      kvOps += 1
      attnOps += t
      fr.push({ t, prefill: false, line: [4, 5, 6, 7, 8], kvOps, attnOps })
    } else {
      kvOps += t
      attnOps += (t * (t + 1)) / 2
      fr.push({ t, prefill: false, line: [3, 4, 5, 6], kvOps, attnOps })
    }
  }
  return fr
}

export function KVCacheDecodeViz() {
  const [mode, setMode] = useState<Mode>('cache')
  const frames = useMemo(() => buildFrames(mode), [mode])
  const st = useStepper(frames, { intervalMs: 1400 })
  const f = st.frame
  const t = f.t
  const qi = t - 1
  const p = softmax(Kall.slice(0, t).map((k) => dot(Qall[qi], k) / Math.sqrt(D)))
  const isNew = (r: number) => (f.prefill ? true : mode === 'cache' ? r === t - 1 : true)
  const tone = (r: number) => (isNew(r) ? (mode === 'nocache' && !f.prefill && r < t - 1 ? 'danger' : 'success') : null)
  const rows = ALL.slice(0, t)

  return (
    <VizFrame
      title="KV cache：每步只算新 token 的一行 K/V"
      caption={
        f.prefill ? (
          <>
            <b>Prefill</b>：prompt 的 {PROMPT.length} 个 token 一次性并行计算，K/V 全部写入缓存（计算密集，像训练前向）。之后进入逐 token 的 <b>decode</b>。
          </>
        ) : mode === 'cache' ? (
          <>
            Decode 第 {t - PROMPT.length} 步：只为新 token「{ALL[t - 1]}」计算 q、k、v（绿色新行），用 <b>1 个</b> query 与缓存中的 {t} 个 key 打分。本步 K/V 投影 1 次、点积 {t} 次。
          </>
        ) : (
          <>
            无缓存第 {t - PROMPT.length} 步：前 {t - 1} 行 K/V 与上一步<b className="text-danger">完全相同却被重算</b>（红色），还要算 {t}×{t} 的因果注意力（{(t * (t + 1)) / 2} 次点积），最终只取最后一行。
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: '本步新算', tone: 'success', variant: 'soft' },
            { label: '重复计算（浪费）', tone: 'danger', variant: 'soft' },
            { label: '直接读缓存', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'cache', label: '有 KV cache' },
              { value: 'nocache', label: '无 cache（重算前缀）' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <TokenRow
          tokens={ALL.map((w, i) => ({
            text: w,
            sub: i < PROMPT.length ? 'prompt' : 'gen',
            tone: i === t - 1 ? 'accent' : i < t ? 'info' : undefined,
            dim: i >= t,
            dashed: i >= t,
          }))}
        />
        <div className="grid items-start gap-4 @3xl:grid-cols-[1fr_1.1fr]">
          <div className="grid grid-cols-2 gap-3">
            {[
              ['K cache', Kall],
              ['V cache', Vall],
            ].map(([name, M]) => (
              <div key={name as string} className="min-w-0">
                <div className="mb-1 text-xs font-medium text-muted">
                  {name as string}（{t}×{D}）
                </div>
                <MatrixView
                  data={(M as number[][]).slice(0, t)}
                  rowLabels={rows}
                  scale="none"
                  cellSize={30}
                  format={(v) => v.toFixed(1)}
                  highlight={(r) => tone(r)}
                />
              </div>
            ))}
          </div>
          <div className="min-w-0 space-y-3">
            <div>
              <div className="mb-1 text-xs font-medium text-muted">新 query「{ALL[qi]}」对缓存的注意力（1×{t}）</div>
              <MatrixView data={[p]} colLabels={rows} domain={[0, 1]} cellSize={36} format={(v) => v.toFixed(2).replace(/^0/, '')} />
            </div>
            <div className="grid grid-cols-2 gap-2 font-mono text-[12.5px]">
              <div className="rounded-xl border border-border bg-surface px-3 py-2">
                <div className="text-[11px] text-muted">累计 K/V 投影（行）</div>
                <div className="text-[15px] font-semibold">{f.kvOps}</div>
              </div>
              <div className="rounded-xl border border-border bg-surface px-3 py-2">
                <div className="text-[11px] text-muted">累计 q·k 点积</div>
                <div className="text-[15px] font-semibold">{f.attnOps}</div>
              </div>
            </div>
            <CodeTrace code={CODE[mode]} active={f.line} maxHeight={240} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
