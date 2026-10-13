/** 关卡相关的小组件：阶段进度点、难度点、元信息行 */
import { Clock } from 'lucide-react'
import type { LevelProgress } from '@/lib/game'
import { DIFFICULTY_LABEL, STAGE_LABEL, stageProgress } from '@/lib/selectors'
import { cn } from '@/lib/utils'

/** 五阶段完成度：完成 = 实心，部分 = 半色，未开始 = 灰 */
export function StagePips({ progress, showLabels, className }: { progress: LevelProgress | undefined; showLabels?: boolean; className?: string }) {
  const sp = stageProgress(progress)
  return (
    <div className={cn('flex items-end gap-1', className)} aria-label="阶段进度">
      {sp.map((s) => {
        const title = `${STAGE_LABEL[s.stage]}：${s.complete ? '已完成' : s.total != null ? `${s.stage === 'quiz' ? Math.round(s.ratio * 100) + '%' : `${s.done}/${s.total}`}` : s.done > 0 ? '进行中' : '未开始'}`
        return (
          <div key={s.stage} className="flex min-w-0 flex-1 flex-col items-center gap-1" title={title}>
            <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
              <i
                className={cn('absolute inset-y-0 left-0 rounded-full', s.complete ? 'bg-success' : 'bg-accent/70')}
                style={{ width: `${(s.complete ? 1 : s.ratio) * 100}%` }}
              />
            </div>
            {showLabels && <span className={cn('text-[10.5px] leading-none', s.complete ? 'ink-success font-medium' : 'text-subtle')}>{STAGE_LABEL[s.stage]}</span>}
          </div>
        )
      })}
    </div>
  )
}

export function DifficultyDots({ value, className }: { value: number; className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-1 text-xs text-muted', className)} title={`难度：${DIFFICULTY_LABEL[value] ?? value}`}>
      <span className="inline-flex gap-[3px]" aria-hidden>
        {[1, 2, 3, 4, 5].map((i) => (
          <i key={i} className={cn('h-1.5 w-1.5 rounded-full', i <= value ? 'bg-fg/60' : 'bg-border-strong')} />
        ))}
      </span>
      {DIFFICULTY_LABEL[value]}
    </span>
  )
}

export function Minutes({ value, className }: { value: number; className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-1 text-xs text-muted tabular-nums', className)}>
      <Clock size={12} aria-hidden />
      约 {value} 分钟
    </span>
  )
}
