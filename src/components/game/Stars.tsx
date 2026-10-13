import { Star } from 'lucide-react'
import { cn } from '@/lib/utils'

/** 星级展示（0–3）。已获得为金色实心，未获得为描边灰色 */
export function Stars({ value, size = 14, className, gap = 2 }: { value: number; size?: number; className?: string; gap?: number }) {
  return (
    <span className={cn('inline-flex items-center', className)} style={{ gap }} aria-label={`${value} 星`} role="img">
      {[0, 1, 2].map((i) => (
        <Star
          key={i}
          size={size}
          strokeWidth={2}
          className={i < value ? 'fill-gold text-gold' : 'fill-transparent text-border-strong'}
          aria-hidden
        />
      ))}
    </span>
  )
}
