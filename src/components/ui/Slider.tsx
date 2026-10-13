import { useId } from 'react'
import { cn } from '@/lib/utils'

/** 带标签与数值显示的滑块（可视化参数调节用） */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v) => String(v),
  className,
}: {
  label?: React.ReactNode
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => React.ReactNode
  className?: string
}) {
  const id = useId()
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className={cn('flex min-w-0 flex-col gap-1.5', className)}>
      {label != null && (
        <div className="flex items-baseline justify-between gap-3 text-[13px]">
          <label htmlFor={id} className="truncate text-muted">
            {label}
          </label>
          <span className="font-mono text-fg tabular-nums">{format(value)}</span>
        </div>
      )}
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="cq-range h-6 w-full cursor-pointer appearance-none bg-transparent"
        style={{ ['--pct' as string]: `${pct}%` }}
      />
    </div>
  )
}
