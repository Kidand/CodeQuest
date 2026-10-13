import { Check, X } from 'lucide-react'
import type { Question } from '@/content/types'
import type { AnswerOf, QLayout } from './grade'
import { cn } from '@/lib/utils'

/** 选项 / 行 / 格子的判定态 */
export type Mark = 'idle' | 'selected' | 'correct' | 'wrong' | 'missed' | 'dim'

/** 卡片式可点击元素的通用样式 */
export function markCls(m: Mark, interactive = true) {
  return cn(
    'border transition-[background,border-color,box-shadow,opacity] duration-150',
    m === 'idle' && 'border-border bg-surface',
    m === 'idle' && interactive && 'hover:border-border-strong hover:bg-surface-2',
    m === 'selected' && 'border-accent bg-accent-soft shadow-[0_0_0_1px_var(--accent)]',
    m === 'correct' && 'border-success bg-success-soft shadow-[0_0_0_1px_var(--success)]',
    m === 'wrong' && 'border-danger bg-danger-soft shadow-[0_0_0_1px_var(--danger)]',
    m === 'missed' && 'border-dashed border-success bg-surface shadow-[0_0_0_1px_var(--success)]',
    m === 'dim' && 'border-border bg-surface opacity-55',
  )
}

/** 判定小图标 */
export function MarkIcon({ m, className }: { m: Mark; className?: string }) {
  if (m === 'correct')
    return (
      <span className={cn('inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-success text-surface', className)}>
        <Check size={13} strokeWidth={3} />
      </span>
    )
  if (m === 'wrong')
    return (
      <span className={cn('inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-danger text-surface', className)}>
        <X size={13} strokeWidth={3} />
      </span>
    )
  if (m === 'missed')
    return <span className={cn('shrink-0 rounded-full bg-success-soft px-1.5 py-0.5 text-[11px] font-semibold text-success', className)}>漏选</span>
  return null
}

/** 题型渲染组件的统一 props */
export interface BodyProps<Q extends Question> {
  q: Q
  value: AnswerOf<Q>
  onChange: (v: AnswerOf<Q>) => void
  /** 已提交：锁定并显示逐项对错 */
  submitted: boolean
  /** 只读展示参考答案 */
  reveal: boolean
  layout: QLayout
  /** 输入框中按 Enter 请求提交 */
  onSubmitRequest?: () => void
}

/** 键位提示 */
export function Kbd({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        'inline-flex h-5 min-w-5 items-center justify-center rounded-md border border-border-strong bg-surface px-1 font-mono text-[11px] font-semibold text-muted shadow-[0_1px_0_var(--border-strong)]',
        className,
      )}
    >
      {children}
    </kbd>
  )
}
