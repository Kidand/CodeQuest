import { Check, X } from 'lucide-react'
import type { JudgeQ, MultiQ, SingleQ } from '@/content/types'
import { InlineMarkdown } from '@/components/Markdown'
import { cn } from '@/lib/utils'
import { Kbd, MarkIcon, markCls, type BodyProps, type Mark } from './marks'

export const LETTERS = 'ABCDEFGHIJ'

/** 单选 / 多选：选项卡片保持作者给定顺序 */
export function ChoiceBody({ q, value, onChange, submitted }: BodyProps<SingleQ | MultiQ>) {
  const multi = q.type === 'multi'
  const chosen = multi ? (value as number[]) : value == null ? [] : [value as number]
  const correct = multi ? (q as MultiQ).answer : [(q as SingleQ).answer]

  const toggle = (i: number) => {
    if (submitted) return
    if (multi) {
      const cur = value as number[]
      ;(onChange as (v: number[]) => void)(cur.includes(i) ? cur.filter((x) => x !== i) : [...cur, i].sort((a, b) => a - b))
    } else (onChange as (v: number | null) => void)(i)
  }

  return (
    <div className="flex flex-col gap-2" role={multi ? 'group' : 'radiogroup'}>
      {q.options.map((opt, i) => {
        const sel = chosen.includes(i)
        const ok = correct.includes(i)
        let m: Mark = sel ? 'selected' : 'idle'
        if (submitted) m = sel ? (ok ? 'correct' : 'wrong') : ok ? (multi ? 'missed' : 'correct') : 'dim'
        return (
          <button
            key={i}
            type="button"
            role={multi ? 'checkbox' : 'radio'}
            aria-checked={sel}
            disabled={submitted}
            data-option={i}
            onClick={() => toggle(i)}
            className={cn(
              'group flex min-h-12 w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-[15px] leading-relaxed text-fg disabled:cursor-default',
              markCls(m, !submitted),
            )}
          >
            <span
              className={cn(
                'inline-flex h-7 w-7 shrink-0 items-center justify-center font-mono text-[13px] font-semibold transition-colors',
                multi ? 'rounded-md' : 'rounded-full',
                m === 'selected'
                  ? 'bg-accent text-accent-fg'
                  : m === 'correct'
                    ? 'bg-success text-surface'
                    : m === 'wrong'
                      ? 'bg-danger text-surface'
                      : 'border border-border-strong bg-surface-2 text-muted',
              )}
            >
              {multi && (m === 'selected' || m === 'correct') ? <Check size={15} strokeWidth={3} /> : LETTERS[i]}
            </span>
            <span className="min-w-0 flex-1">
              <InlineMarkdown>{opt}</InlineMarkdown>
            </span>
            {submitted && <MarkIcon m={m} />}
          </button>
        )
      })}
    </div>
  )
}

/** 判断：两个大按钮 */
export function JudgeBody({ q, value, onChange, submitted }: BodyProps<JudgeQ>) {
  const opts: { v: boolean; label: string; icon: typeof Check; key: string }[] = [
    { v: true, label: '正确', icon: Check, key: '1' },
    { v: false, label: '错误', icon: X, key: '2' },
  ]
  return (
    <div className="grid grid-cols-2 gap-3" role="radiogroup">
      {opts.map((o) => {
        const sel = value === o.v
        const ok = q.answer === o.v
        let m: Mark = sel ? 'selected' : 'idle'
        if (submitted) m = sel ? (ok ? 'correct' : 'wrong') : ok ? 'correct' : 'dim'
        const Icon = o.icon
        return (
          <button
            key={String(o.v)}
            type="button"
            role="radio"
            aria-checked={sel}
            disabled={submitted}
            onClick={() => !submitted && onChange(o.v)}
            className={cn(
              'relative flex h-20 flex-col items-center justify-center gap-1 rounded-2xl text-base font-semibold text-fg disabled:cursor-default sm:h-24',
              markCls(m, !submitted),
            )}
          >
            <Icon
              size={26}
              strokeWidth={2.5}
              className={cn(
                m === 'correct' ? 'text-success' : m === 'wrong' ? 'text-danger' : sel ? 'text-accent' : o.v ? 'text-success/70' : 'text-danger/70',
              )}
            />
            {o.label}
            {!submitted && <Kbd className="absolute right-2 top-2 max-md:hidden">{o.key}</Kbd>}
            {submitted && <MarkIcon m={m} className="absolute right-2 top-2" />}
          </button>
        )
      })}
    </div>
  )
}
