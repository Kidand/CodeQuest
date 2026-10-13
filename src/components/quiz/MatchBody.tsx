import { useState } from 'react'
import type { MatchQ } from '@/content/types'
import { InlineMarkdown } from '@/components/Markdown'
import { cn } from '@/lib/utils'
import { MarkIcon, type BodyProps } from './marks'

/** 配对色：用可视化分类色区分每一对 */
const pairColor = (k: number) => `var(--viz-${(k % 8) + 1})`

/** 配对：先点左列再点右列（或反之）；配好的一对用同色标记；再点同一对可取消 */
export function MatchBody({ q, value, onChange, submitted, layout }: BodyProps<MatchQ>) {
  const v = value as (number | null)[]
  const right = layout.right ?? q.pairs.map((_, i) => i)
  const [selL, setSelL] = useState<number | null>(null)
  const [selR, setSelR] = useState<number | null>(null)

  /** 右列 j 当前配到的左项 */
  const leftOf = (j: number) => v.findIndex((x) => x === j)

  const assign = (i: number, j: number) => {
    const next = v.map((x) => (x === j ? null : x))
    next[i] = v[i] === j ? null : j
    onChange(next)
    setSelL(null)
    setSelR(null)
  }

  const tapLeft = (i: number) => {
    if (submitted) return
    if (selR != null) return assign(i, selR)
    setSelL(selL === i ? null : i)
  }
  const tapRight = (j: number) => {
    if (submitted) return
    if (selL != null) return assign(selL, j)
    setSelR(selR === j ? null : j)
  }

  const itemCls = (active: boolean, color: string | null, state: 'ok' | 'bad' | null) =>
    cn(
      'relative flex min-h-12 w-full items-center gap-2 rounded-xl border px-2.5 py-2 text-left text-[14px] leading-snug text-fg transition-[background,border-color,box-shadow] disabled:cursor-default',
      state === 'ok' && 'border-success bg-success-soft',
      state === 'bad' && 'border-danger bg-danger-soft',
      !state && active && 'border-accent bg-accent-soft shadow-[0_0_0_2px_var(--ring)]',
      !state && !active && !color && 'border-border bg-surface hover:border-border-strong hover:bg-surface-2',
      !state && !active && color && 'bg-surface',
    )

  const dot = (k: number | null, label?: string) =>
    k == null ? (
      <span className="h-5 w-5 shrink-0 rounded-full border-2 border-dashed border-border-strong" />
    ) : (
      <span
        className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full font-mono text-[11px] font-bold text-surface"
        style={{ background: pairColor(k) }}
      >
        {label}
      </span>
    )

  const pairedCount = v.filter((x) => x != null).length

  return (
    <div>
      {!submitted && (
        <p className="mb-2 text-xs text-subtle">
          先点左侧一项，再点右侧与之对应的一项；再次点同一对可取消。已配对 {pairedCount}/{q.pairs.length}
        </p>
      )}
      <div className="grid grid-cols-2 gap-x-2.5 gap-y-2 sm:gap-x-4">
        <div className="flex flex-col gap-2">
          {q.pairs.map(([l], i) => {
            const j = v[i]
            const paired = j != null
            const state = submitted ? (j === i ? 'ok' : 'bad') : null
            return (
              <div key={i} className="flex flex-col gap-1">
                <button
                  type="button"
                  disabled={submitted}
                  aria-pressed={selL === i}
                  onClick={() => tapLeft(i)}
                  className={itemCls(selL === i, paired ? pairColor(i) : null, state)}
                  style={!state && paired && selL !== i ? { borderColor: pairColor(i), boxShadow: `inset 3px 0 0 ${pairColor(i)}` } : undefined}
                >
                  {dot(paired ? i : null, String(i + 1))}
                  <span className="min-w-0 flex-1">
                    <InlineMarkdown>{l}</InlineMarkdown>
                  </span>
                  {state && <MarkIcon m={state === 'ok' ? 'correct' : 'wrong'} />}
                </button>
                {state === 'bad' && (
                  <div className="pl-2 text-xs leading-snug text-muted">
                    应配：<InlineMarkdown className="text-success">{q.pairs[i][1]}</InlineMarkdown>
                  </div>
                )}
              </div>
            )
          })}
        </div>
        <div className="flex flex-col gap-2">
          {right.map((j) => {
            const li = leftOf(j)
            const paired = li >= 0
            const state = submitted ? (paired && li === j ? 'ok' : 'bad') : null
            return (
              <button
                key={j}
                type="button"
                disabled={submitted}
                aria-pressed={selR === j}
                onClick={() => tapRight(j)}
                className={itemCls(selR === j, paired ? pairColor(li) : null, state)}
                style={!state && paired && selR !== j ? { borderColor: pairColor(li), boxShadow: `inset 3px 0 0 ${pairColor(li)}` } : undefined}
              >
                {dot(paired ? li : null, paired ? String(li + 1) : undefined)}
                <span className="min-w-0 flex-1">
                  <InlineMarkdown>{q.pairs[j][1]}</InlineMarkdown>
                </span>
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
