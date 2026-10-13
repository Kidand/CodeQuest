/** 通关结算弹窗 + 本关星级 / XP 监听 */
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'
import { ArrowRight, Check, Flame, Lock, NotebookText, Star, X } from 'lucide-react'
import { STAGES, type LevelContent } from '@/content/types'
import type { LevelRef } from '@/content/catalog'
import { useProgress } from '@/lib/store'
import { starsOf } from '@/lib/game'
import { Button } from '@/components/ui'
import { cn } from '@/lib/utils'
import { Modal } from './Overlay'
import { STAGE_META, stageState } from './status'

/* ------------------------------------------------------------------ */
/* 星级 / XP 监听                                                       */
/* ------------------------------------------------------------------ */
/**
 * 直接监听进度 store（不消费 useEvents 队列——XP / 升级 / 成就浮层由壳层的 GameEventLayer 全局负责）。
 *  - 本关星级提升 → onStars(before, after)
 *  - sessionXp：进入本关后累计获得的 XP
 */
export function useLevelWatch(levelId: string, onStars: (before: number, after: number) => void) {
  const [sessionXp, setSessionXp] = useState(0)
  const onStarsRef = useRef(onStars)
  onStarsRef.current = onStars
  useEffect(() => {
    let xp0 = useProgress.getState().xp
    let stars0 = starsOf(useProgress.getState().levels[levelId])
    setSessionXp(0)
    return useProgress.subscribe((s) => {
      const stars = starsOf(s.levels[levelId])
      if (stars > stars0) onStarsRef.current(stars0, stars)
      stars0 = stars
      if (s.xp < xp0) xp0 = s.xp // 重置 / 导入
      setSessionXp(s.xp - xp0)
    })
  }, [levelId])
  return { sessionXp }
}

/* ------------------------------------------------------------------ */
/* 通关结算                                                             */
/* ------------------------------------------------------------------ */
export function LevelSummary({
  open,
  onClose,
  levelId,
  levelRef,
  L,
  sessionXp,
  starsBefore,
  onCheatsheet,
  nextLevel,
}: {
  open: boolean
  onClose: () => void
  levelId: string
  levelRef: LevelRef
  L: LevelContent
  sessionXp: number
  /** 打开结算时用于判断哪几颗星是新点亮的 */
  starsBefore: number
  onCheatsheet: () => void
  nextLevel: { id: string; title: string; locked: boolean } | null
}) {
  const p = useProgress((s) => s.levels[levelId])
  const stars = starsOf(p)
  const cleared = stars >= 1
  return (
    <Modal open={open} onClose={onClose} label="关卡结算" className="sm:max-w-lg">
      <div className="relative p-6 text-center sm:p-8" data-summary>
        <button type="button" onClick={onClose} className="absolute right-3 top-3 rounded-lg p-1.5 text-muted hover:bg-surface-2" aria-label="关闭">
          <X size={18} />
        </button>
        <div className="text-xs font-semibold tracking-wider text-accent">{cleared ? '关卡通关' : '关卡进度'}</div>
        <h2 className="mt-1 text-xl font-bold text-fg">{levelRef.meta.title}</h2>
        <div className="mt-5 flex items-end justify-center gap-3">
          {[1, 2, 3].map((k) => {
            const on = k <= stars
            const fresh = on && k > starsBefore
            return (
              <Star
                key={k}
                size={k === 2 ? 56 : 44}
                strokeWidth={1.6}
                className={cn(on ? 'text-gold drop-shadow-sm' : 'text-border-strong', fresh && 'cq-star-in', k === 2 && '-translate-y-2')}
                style={fresh ? { animationDelay: `${0.2 + (k - starsBefore - 1) * 0.3}s` } : undefined}
                fill={on ? 'currentColor' : 'none'}
              />
            )
          })}
        </div>
        <p className="mt-2 text-sm text-muted">
          {stars === 3
            ? '三星满分！实战全部独立通过。'
            : stars === 2
              ? '★★★ 还差：实战全部通过（且通过前没看题解）'
              : stars === 1
                ? '★★ 还差：随堂 ≥ 80% 且跟练全部完成'
                : '★ 需要：图解完成且随堂 ≥ 60%'}
        </p>
        {sessionXp > 0 && (
          <div className="mt-3 inline-flex items-center gap-1 rounded-full bg-gold-soft px-3 py-1 text-sm font-semibold text-gold">
            <Flame size={14} /> 本次获得 +{sessionXp} XP
          </div>
        )}
        <ul className="mt-5 grid grid-cols-5 gap-1.5 text-[11px]">
          {STAGES.map((s) => {
            const st = stageState(s, L, p)
            const M = STAGE_META[s]
            return (
              <li
                key={s}
                className={cn(
                  'flex flex-col items-center gap-1 rounded-xl border px-1 py-2',
                  st.empty ? 'border-border opacity-50' : st.done ? 'border-success/40 bg-success-soft/50' : 'border-border',
                )}
              >
                <span className={cn('inline-flex h-6 w-6 items-center justify-center rounded-full', st.done && !st.empty ? 'bg-success text-surface' : 'bg-surface-3 text-muted')}>
                  {st.done && !st.empty ? <Check size={13} strokeWidth={3} /> : <M.icon size={12} />}
                </span>
                <span className="font-medium text-fg">{M.short}</span>
                <span className="text-subtle">{st.empty ? '—' : st.text}</span>
              </li>
            )
          })}
        </ul>
        <div className="mt-6 flex flex-col gap-2">
          {nextLevel &&
            (nextLevel.locked ? (
              <div className="flex items-center justify-center gap-1.5 rounded-xl bg-surface-2 px-4 py-3 text-sm text-muted">
                <Lock size={15} /> 下一关「{nextLevel.title}」：本关拿到 ★ 后解锁
              </div>
            ) : (
              <Link
                to={`/level/${nextLevel.id}`}
                onClick={onClose}
                className="inline-flex h-12 items-center justify-center gap-2 rounded-xl bg-accent px-6 text-base font-semibold text-accent-fg shadow-sm hover:bg-accent-hover"
                data-action="next-level"
              >
                下一关：{nextLevel.title}
                <ArrowRight size={18} />
              </Link>
            ))}
          <div className="flex gap-2">
            <Button variant="secondary" className="flex-1" icon={<NotebookText size={16} />} onClick={onCheatsheet}>
              速记卡
            </Button>
            <Button variant="ghost" className="flex-1" onClick={onClose}>
              留在本关
            </Button>
          </div>
        </div>
      </div>
    </Modal>
  )
}
