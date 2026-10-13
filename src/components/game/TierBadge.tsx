/** 分层 / 频率 / 路线徽章：明暗主题下都保证可读（文字色向前景色混合，见 app.css .ink-*） */
import { Crown, Sparkles } from 'lucide-react'
import { FREQ_LABEL, TIER_LABEL, type Tier, type Track } from '@/content/types'
import { cn } from '@/lib/utils'

type ToneKey = 'danger' | 'warning' | 'info' | 'success' | 'accent' | 'llm' | 'algo' | 'gold' | 'neutral'

const SOFT: Record<ToneKey, string> = {
  danger: 'bg-danger-soft ink-danger',
  warning: 'bg-warning-soft ink-warning',
  info: 'bg-info-soft ink-info',
  success: 'bg-success-soft ink-success',
  accent: 'bg-accent-soft ink-accent',
  llm: 'bg-llm-soft ink-llm',
  algo: 'bg-algo-soft ink-algo',
  gold: 'bg-gold-soft ink-gold',
  neutral: 'bg-surface-2 text-muted',
}
const DOT: Record<ToneKey, string> = {
  danger: 'bg-danger',
  warning: 'bg-warning',
  info: 'bg-info',
  success: 'bg-success',
  accent: 'bg-accent',
  llm: 'bg-llm',
  algo: 'bg-algo',
  gold: 'bg-gold',
  neutral: 'bg-subtle',
}

export const TIER_TONE: Record<Tier, ToneKey> = {
  must: 'danger',
  high: 'warning',
  advanced: 'info',
  foundation: 'success',
  modern: 'accent',
  frontier: 'llm',
}
export const tierDotClass = (t: Tier) => DOT[TIER_TONE[t]]
export const tierSoftClass = (t: Tier) => SOFT[TIER_TONE[t]]

const base = 'inline-flex shrink-0 items-center gap-1 whitespace-nowrap rounded-full font-medium leading-none'
const sizes = { xs: 'h-5 px-1.5 text-[11px]', sm: 'h-6 px-2 text-xs' } as const

export function TierBadge({ tier, size = 'sm', className }: { tier: Tier; size?: keyof typeof sizes; className?: string }) {
  return (
    <span className={cn(base, sizes[size], SOFT[TIER_TONE[tier]], className)}>
      {tier === 'frontier' ? <Sparkles size={size === 'xs' ? 10 : 12} aria-hidden /> : <i className={cn('h-1.5 w-1.5 rounded-full', DOT[TIER_TONE[tier]])} aria-hidden />}
      {TIER_LABEL[tier]}
    </span>
  )
}

/** 面试频率：用信号格表示 */
export function FreqBadge({ freq, size = 'sm', className }: { freq: 'high' | 'medium' | 'low'; size?: keyof typeof sizes; className?: string }) {
  const n = freq === 'high' ? 3 : freq === 'medium' ? 2 : 1
  return (
    <span className={cn(base, sizes[size], 'border border-border bg-surface text-muted', className)} title={`面试频率：${FREQ_LABEL[freq]}`}>
      <span className="inline-flex items-end gap-[1.5px]" aria-hidden>
        {[1, 2, 3].map((i) => (
          <i key={i} className={cn('w-[3px] rounded-[1px]', i <= n ? 'bg-fg/70' : 'bg-border-strong')} style={{ height: 3 + i * 2.5 }} />
        ))}
      </span>
      {FREQ_LABEL[freq]}
    </span>
  )
}

export function TrackBadge({ track, size = 'sm', className }: { track: Track; size?: keyof typeof sizes; className?: string }) {
  return (
    <span className={cn(base, sizes[size], track === 'algo' ? SOFT.algo : SOFT.llm, className)}>
      <i className={cn('h-1.5 w-1.5 rounded-full', track === 'algo' ? 'bg-algo' : 'bg-llm')} aria-hidden />
      {track === 'algo' ? '经典算法' : '大模型'}
    </span>
  )
}

export function BossBadge({ size = 'sm', className }: { size?: keyof typeof sizes; className?: string }) {
  return (
    <span className={cn(base, sizes[size], SOFT.gold, 'font-semibold', className)}>
      <Crown size={size === 'xs' ? 10 : 12} aria-hidden />
      BOSS
    </span>
  )
}

export function ToneChip({ tone = 'neutral', className, children }: { tone?: ToneKey; className?: string; children: React.ReactNode }) {
  return <span className={cn(base, sizes.sm, SOFT[tone], className)}>{children}</span>
}
