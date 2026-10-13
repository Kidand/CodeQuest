import { useRef } from 'react'
import { Calculator } from 'lucide-react'
import type { FillQ, NumericQ } from '@/content/types'
import { approxEqual, cn, parseNumberExpr } from '@/lib/utils'
import { blankCorrect, formatParsed } from './grade'
import { MarkIcon, type BodyProps } from './marks'

/** 填空题干：把 ____(n)____ 渲染成内联“空位”（显示已填内容），便于对照 */
export function fillPrompt(prompt: string, values: string[]) {
  return prompt.replace(/_{2,}\s*\((\d+)\)\s*_{2,}/g, (_m, n: string) => {
    const v = (values[Number(n) - 1] ?? '').replace(/`/g, '').trim()
    return v ? `\`(${n}) ${v}\`` : `\`(${n}) ＿＿＿\``
  })
}

/** 数值解析预览 */
function NumPreview({ input, className }: { input: string; className?: string }) {
  if (!input.trim()) return null
  const x = parseNumberExpr(input)
  const ok = Number.isFinite(x)
  return (
    <div className={cn('flex min-h-5 items-center gap-1.5 text-xs', ok ? 'text-muted' : 'text-warning', className)} aria-live="polite">
      <Calculator size={12} className="shrink-0" />
      {ok ? (
        <span>
          解析为 <span className="font-mono font-semibold text-fg tabular-nums">{formatParsed(x)}</span>
        </span>
      ) : (
        <span>无法解析：支持 1e9、2^30、3*4、7Gi、(1+2)*3 等写法</span>
      )}
    </div>
  )
}

const inputCls = (state: 'idle' | 'correct' | 'wrong') =>
  cn(
    'h-11 w-full min-w-0 rounded-xl border bg-surface px-3 font-mono text-[15px] text-fg outline-none transition-[border-color,box-shadow] placeholder:text-subtle',
    state === 'idle' && 'border-border-strong focus:border-accent focus:shadow-[0_0_0_3px_var(--ring)]',
    state === 'correct' && 'border-success bg-success-soft',
    state === 'wrong' && 'border-danger bg-danger-soft',
  )

export function FillBody({ q, value, onChange, submitted, onSubmitRequest }: BodyProps<FillQ>) {
  const refs = useRef<(HTMLInputElement | null)[]>([])
  const vals = value as string[]
  return (
    <div className="flex flex-col gap-3">
      {q.blanks.map((b, i) => {
        const v = vals[i] ?? ''
        const ok = submitted ? blankCorrect(b, v) : null
        return (
          <div key={i} className="flex flex-col gap-1">
            <label className="flex items-center gap-2">
              <span className="w-12 shrink-0 text-center font-mono text-sm font-semibold text-muted">{b.label ?? `(${i + 1})`}</span>
              <input
                ref={(el) => {
                  refs.current[i] = el
                }}
                value={v}
                disabled={submitted}
                inputMode="text"
                autoComplete="off"
                autoCapitalize="off"
                autoCorrect="off"
                spellCheck={false}
                enterKeyHint={i === q.blanks.length - 1 ? 'done' : 'next'}
                placeholder={b.placeholder ?? (b.numeric ? '数值，可写表达式' : '填写答案')}
                aria-label={`空 ${b.label ?? i + 1}`}
                onChange={(e) => {
                  const next = vals.slice()
                  next[i] = e.target.value
                  onChange(next)
                }}
                onKeyDown={(e) => {
                  if (e.key !== 'Enter' || e.nativeEvent.isComposing) return
                  e.preventDefault()
                  e.stopPropagation()
                  const nextEmpty = vals.findIndex((s, k) => k !== i && !s.trim())
                  if (nextEmpty >= 0 && v.trim()) refs.current[nextEmpty]?.focus()
                  else onSubmitRequest?.()
                }}
                className={inputCls(ok == null ? 'idle' : ok ? 'correct' : 'wrong')}
              />
              {ok != null && <MarkIcon m={ok ? 'correct' : 'wrong'} />}
            </label>
            {b.numeric && !submitted && <NumPreview input={v} className="pl-14" />}
            {submitted && !ok && (
              <div className="pl-14 text-[13px] text-muted">
                正确答案：<span className="font-mono font-semibold text-success">{b.answers[0]}</span>
                {b.answers.length > 1 && !b.numeric && (
                  <span className="text-subtle">（也可：{b.answers.slice(1).join(' / ')}）</span>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function NumericBody({ q, value, onChange, submitted, onSubmitRequest }: BodyProps<NumericQ>) {
  const v = value as string
  const x = parseNumberExpr(v)
  const ok = submitted ? approxEqual(x, q.answer, q.tolerance ?? 0.01) : null
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2">
        <input
          value={v}
          disabled={submitted}
          inputMode="text"
          autoComplete="off"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          enterKeyHint="done"
          placeholder="输入数值或表达式，如 2^30、1.5e9、7*1024"
          aria-label="数值答案"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
              e.preventDefault()
              e.stopPropagation()
              onSubmitRequest?.()
            }
          }}
          className={cn(inputCls(ok == null ? 'idle' : ok ? 'correct' : 'wrong'), 'max-w-md text-base')}
        />
        {q.unit && <span className="shrink-0 text-sm font-medium text-muted">{q.unit}</span>}
        {ok != null && <MarkIcon m={ok ? 'correct' : 'wrong'} />}
      </div>
      {!submitted && <NumPreview input={v} />}
      {submitted && (
        <div className="text-[13px] text-muted">
          {Number.isFinite(x) && (
            <>
              你的答案 <span className="font-mono text-fg tabular-nums">{formatParsed(x)}</span>
              {' · '}
            </>
          )}
          参考答案 <span className="font-mono font-semibold text-success tabular-nums">{formatParsed(q.answer)}</span>
          {q.unit ? ` ${q.unit}` : ''}
          <span className="text-subtle">
            {' '}
            （允许相对误差 {+(((q.tolerance ?? 0.01) * 100).toFixed(2))}%）
          </span>
        </div>
      )}
    </div>
  )
}
