import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/**
 * 可视化外框：标题 + 舞台 + 控件 + 说明。所有 viz 组件都应包在它里面，保证全站风格统一。
 *  - 舞台区域自适应宽度；SVG 请使用 viewBox + width="100%"
 *  - controls 放 StepControls / Slider / Segmented 等
 *  - caption 放当前步骤的文字说明（会随步骤变化）
 */
export function VizFrame({
  title,
  badge,
  children,
  controls,
  caption,
  legend,
  className,
  stageClassName,
}: {
  title?: ReactNode
  badge?: ReactNode
  children: ReactNode
  controls?: ReactNode
  caption?: ReactNode
  legend?: ReactNode
  className?: string
  stageClassName?: string
}) {
  return (
    <figure className={cn('not-prose my-1 overflow-hidden rounded-2xl border border-border bg-surface shadow-sm', className)}>
      {(title || badge) && (
        <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-2.5">
          <div className="min-w-0 text-[13px] font-semibold leading-snug text-fg">{title}</div>
          {badge}
        </div>
      )}
      <div className={cn('@container thin-scroll overflow-x-auto bg-[color-mix(in_oklab,var(--surface-2)_45%,var(--surface))] px-3 py-4 sm:px-5', stageClassName)}>
        {children}
      </div>
      {legend && <div className="flex flex-wrap gap-x-4 gap-y-1.5 border-t border-border px-4 py-2 text-xs text-muted">{legend}</div>}
      {caption && (
        <figcaption className="min-h-[2.75rem] border-t border-border px-4 py-2.5 text-[13.5px] leading-relaxed text-fg">{caption}</figcaption>
      )}
      {controls && <div className="flex flex-wrap items-center gap-x-4 gap-y-3 border-t border-border bg-surface px-4 py-3">{controls}</div>}
    </figure>
  )
}
