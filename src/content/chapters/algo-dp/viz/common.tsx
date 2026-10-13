/** 本章可视化共用的小组件 */
import type { ReactNode } from 'react'

/** 说明文字里的行内代码 */
export const C = ({ children }: { children: ReactNode }) => (
  <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12.5px]">{children}</code>
)

/** 小标题（图中分区说明） */
export const Sub = ({ children }: { children: ReactNode }) => <div className="mb-1.5 text-xs font-medium text-muted">{children}</div>

/** 数字统计胶囊 */
export function Stat({ label, value, tone = 'fg' }: { label: ReactNode; value: ReactNode; tone?: 'fg' | 'success' | 'danger' | 'accent' | 'warning' }) {
  const color = tone === 'fg' ? 'var(--fg)' : `var(--${tone})`
  return (
    <span className="inline-flex items-baseline gap-1.5 rounded-lg border border-border bg-surface px-2 py-1 text-xs">
      <span className="text-muted">{label}</span>
      <b className="font-mono tabular-nums" style={{ color }}>
        {value}
      </b>
    </span>
  )
}
