/**
 * labels 与 shift：mask 定义在“被预测的 token”上。
 * 逐位置步进：位置 t 的 logits 预测 labels[t+1]；label 为 -100 的预测不计 loss。
 * 关键反直觉点：assistant 模板头最后一个 token（⏎）自己的 label 是 -100，但它的 logits 要负责预测回复的第一个 token。
 */
import { useMemo } from 'react'
import { CodeTrace, StepControls, VizFrame, useStepper, toneColor, toneSoft } from '@/viz'
import { cn } from '@/lib/utils'
import { C } from './plot'

const TOKS = ['<|im_start|>', 'user', '\n', 'hi', '<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', 'Hello', '!', '<|im_end|>']
const FIRST_RESP = 9
const LABELS = TOKS.map((t, i) => (i >= FIRST_RESP ? t : null))
const show = (t: string) => t.replace(/\n/g, '⏎')

const CODE = `shift_logits = logits[:, :-1, :]   # 丢掉最后一个位置
shift_labels = labels[:, 1:]       # 丢掉第一个 token
loss = F.cross_entropy(
    shift_logits.reshape(-1, V), shift_labels.reshape(-1),
    ignore_index=-100)             # 只对 label != -100 的位置取平均`

interface Frame {
  t: number
  counted: boolean
  last: boolean
  sofar: number
}

export function ShiftViz() {
  const frames = useMemo<Frame[]>(() => {
    let sofar = 0
    return TOKS.map((_, t) => {
      const last = t === TOKS.length - 1
      const counted = !last && LABELS[t + 1] != null
      if (counted) sofar++
      return { t, counted, last, sofar }
    })
  }, [])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame
  const target = f.t + 1

  const caption = f.last ? (
    <>位置 {f.t} 是最后一个 token，它的 logits 预测的是序列之外的东西，被 <C>logits[:, :-1]</C> 丢掉。整条样本共 <b>{f.sofar}</b> 项计入 loss。</>
  ) : f.counted ? (
    <>
      位置 {f.t}（输入 <b className="font-mono">{show(TOKS[f.t])}</b>）的 logits 预测 labels[{target}] = <b className="font-mono">{show(TOKS[target])}</b>，<b>计入 loss</b>。
      {f.t === FIRST_RESP - 1 && <> 注意：这个 ⏎ 自己的 label 是 -100，但它的输出负责预测回复的第一个 token。</>}
    </>
  ) : (
    <>
      位置 {f.t}（输入 <b className="font-mono">{show(TOKS[f.t])}</b>）的 logits 预测 labels[{target}] = <span className="font-mono">-100</span>：这是 prompt / 模板 token，<b>不计 loss</b>（但前向照算，它是后面 token 的上下文）。
    </>
  )

  return (
    <VizFrame title="Shift：位置 t 预测第 t+1 个 token" caption={caption} controls={<StepControls stepper={st} />}>
      <div className="grid items-start gap-4 @3xl:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <div className="min-w-0 overflow-hidden rounded-xl border border-border bg-surface text-[12.5px]">
          <div className="grid grid-cols-[2rem_minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1.2fr)] border-b border-border bg-surface-2 px-2 py-1.5 text-[11.5px] text-muted">
            <span>t</span>
            <span>input_ids[t]</span>
            <span>labels[t]</span>
            <span>logits[t] 预测</span>
          </div>
          {TOKS.map((tok, i) => {
            const isCur = i === f.t
            const isTarget = i === target && !f.last
            const lab = LABELS[i]
            const pred = i === TOKS.length - 1 ? '丢弃' : LABELS[i + 1] != null ? `→ ${show(TOKS[i + 1])}` : '→ -100'
            return (
              <div
                key={i}
                className={cn('grid grid-cols-[2rem_minmax(0,1.3fr)_minmax(0,1fr)_minmax(0,1.2fr)] items-center px-2 py-[3px] font-mono transition-colors')}
                style={{ background: isCur ? toneSoft('accent', 18) : undefined }}
              >
                <span className="text-subtle">{i}</span>
                <span className="truncate text-fg">{show(tok)}</span>
                <span
                  className="truncate rounded px-1"
                  style={{
                    color: lab == null ? 'var(--subtle)' : 'var(--fg)',
                    background: isTarget ? toneSoft(f.counted ? 'success' : 'muted', 35) : undefined,
                    boxShadow: isTarget ? `inset 0 0 0 1.5px ${toneColor(f.counted ? 'success' : 'muted')}` : undefined,
                  }}
                >
                  {lab == null ? '-100' : show(lab)}
                </span>
                <span className="truncate" style={{ color: i === TOKS.length - 1 ? 'var(--subtle)' : LABELS[i + 1] != null ? toneColor('success') : 'var(--subtle)' }}>
                  {pred}
                </span>
              </div>
            )
          })}
        </div>
        <div className="min-w-0">
          <CodeTrace code={CODE} active={f.last ? 1 : f.counted ? [3, 4, 5] : 2} vars={{ t: f.t, 已计入项数: f.sofar }} />
        </div>
      </div>
    </VizFrame>
  )
}
