import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Stepper } from './useStepper'

const SPEEDS = [0.5, 1, 2, 4]

/** 播放控制条：重置 / 上一步 / 播放暂停 / 下一步 / 进度条 / 倍速。支持键盘 ← → 空格 */
export function StepControls<T>({ stepper, className, showSpeed = true }: { stepper: Stepper<T>; className?: string; showSpeed?: boolean }) {
  const s = stepper
  const btn =
    'inline-flex h-9 w-9 items-center justify-center rounded-xl border border-border bg-surface text-fg transition-colors hover:bg-surface-2 disabled:opacity-40 active:scale-95'
  return (
    <div
      className={cn('flex w-full flex-wrap items-center gap-2', className)}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'ArrowRight') { e.preventDefault(); s.next() }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); s.prev() }
        else if (e.key === ' ') { e.preventDefault(); s.toggle() }
      }}
    >
      <div className="flex items-center gap-1.5">
        <button className={btn} onClick={s.reset} title="重置" aria-label="重置">
          <RotateCcw size={16} />
        </button>
        <button className={btn} onClick={s.prev} disabled={s.atStart} title="上一步 (←)" aria-label="上一步">
          <ChevronLeft size={18} />
        </button>
        <button
          className={cn(btn, 'w-11 border-accent bg-accent text-accent-fg hover:bg-accent-hover')}
          onClick={s.toggle}
          title="播放/暂停 (空格)"
          aria-label={s.playing ? '暂停' : '播放'}
        >
          {s.playing ? <Pause size={17} /> : <Play size={17} className="translate-x-px" />}
        </button>
        <button className={btn} onClick={s.next} disabled={s.atEnd} title="下一步 (→)" aria-label="下一步">
          <ChevronRight size={18} />
        </button>
        {showSpeed && (
          <button
            className="h-9 shrink-0 rounded-lg border border-border px-2 font-mono text-xs text-muted hover:text-fg"
          onClick={() => s.setSpeed(SPEEDS[(SPEEDS.indexOf(s.speed) + 1) % SPEEDS.length])}
          title="播放速度"
        >
          {s.speed}×
        </button>
        )}
      </div>
      <div className="flex min-w-[140px] flex-1 items-center gap-2">
        <input
          type="range"
          min={0}
          max={Math.max(0, s.total - 1)}
          value={s.i}
          onChange={(e) => s.setI(Number(e.target.value))}
          className="cq-range h-6 min-w-0 flex-1 cursor-pointer appearance-none bg-transparent"
          style={{ ['--pct' as string]: `${s.total > 1 ? (s.i / (s.total - 1)) * 100 : 0}%` }}
          aria-label="进度"
        />
        <span className="shrink-0 font-mono text-xs text-muted tabular-nums">
          {s.i + 1}/{s.total}
        </span>
      </div>
    </div>
  )
}
