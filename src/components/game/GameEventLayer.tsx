/**
 * 游戏事件浮层：消费 useEvents 队列。
 *  - XP：右上角浮层，短时间内多次加分合并显示
 *  - 升级：弹窗 + 彩带
 *  - 成就：toast（同时最多 3 条，其余折叠）
 *  - 星级：三星彩带庆祝，1/2 星轻提示
 *  - 连续打卡：轻提示
 * 挂载在 Root（App.tsx），关卡页（不套 AppLayout）同样可见。
 */
import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Flame, Sparkles, Star, X } from 'lucide-react'
import { useEvents, useProgress, type GameEvent } from '@/lib/store'
import { titleForLevel, xpToReach } from '@/lib/game'
import { ACHIEVEMENT_BY_ID, type AchievementTone } from '@/lib/achievements'
import { loadCatalog } from '@/content/catalog'
import { Button } from '@/components/ui'
import { Modal } from './Modal'
import { Stars } from './Stars'
import { burst } from './confetti'
import { cn } from '@/lib/utils'

type Toast =
  | { id: number; kind: 'achievement'; achievementId: string }
  | { id: number; kind: 'stars'; levelId: string; stars: number; title?: string }
  | { id: number; kind: 'streak'; days: number }

const TOAST_MS = 4600
const XP_MERGE_MS = 1600

export const TONE_BG: Record<AchievementTone, string> = {
  gold: 'bg-gold-soft ink-gold',
  accent: 'bg-accent-soft ink-accent',
  algo: 'bg-algo-soft ink-algo',
  llm: 'bg-llm-soft ink-llm',
  success: 'bg-success-soft ink-success',
  warning: 'bg-warning-soft ink-warning',
  danger: 'bg-danger-soft ink-danger',
  info: 'bg-info-soft ink-info',
}

export function GameEventLayer() {
  const [xp, setXp] = useState<{ amount: number; reason: string; key: number } | null>(null)
  const xpTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const lastXpAt = useRef(0)
  const [levelUp, setLevelUp] = useState<number | null>(null)
  const [toasts, setToasts] = useState<Toast[]>([])

  useEffect(() => {
    const dismissLater = (id: number) => setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== id)), TOAST_MS)

    const handle = (e: GameEvent) => {
      switch (e.type) {
        case 'xp': {
          const now = Date.now()
          const merge = now - lastXpAt.current < XP_MERGE_MS
          lastXpAt.current = now
          setXp((cur) =>
            merge && cur ? { amount: cur.amount + e.amount, reason: e.reason, key: cur.key } : { amount: e.amount, reason: e.reason, key: e.id },
          )
          clearTimeout(xpTimer.current)
          xpTimer.current = setTimeout(() => setXp(null), 2200)
          break
        }
        case 'levelup':
          setLevelUp((cur) => Math.max(cur ?? 0, e.level))
          burst('big')
          break
        case 'achievement':
          setToasts((ts) => [...ts, { id: e.id, kind: 'achievement', achievementId: e.achievementId }])
          dismissLater(e.id)
          break
        case 'stars': {
          const id = e.id
          setToasts((ts) => [...ts, { id, kind: 'stars', levelId: e.levelId, stars: e.stars }])
          loadCatalog().then((cat) => {
            const title = cat.levelById.get(e.levelId)?.meta.title
            if (title) setToasts((ts) => ts.map((t) => (t.id === id && t.kind === 'stars' ? { ...t, title } : t)))
          })
          if (e.stars === 3) burst('small')
          dismissLater(id)
          break
        }
        case 'streak':
          setToasts((ts) => [...ts, { id: e.id, kind: 'streak', days: e.days }])
          dismissLater(e.id)
          break
      }
    }

    const drain = () => {
      const q = useEvents.getState().queue
      if (!q.length) return
      const ids = new Set(q.map((e) => e.id))
      q.forEach(handle)
      useEvents.setState((s) => ({ queue: s.queue.filter((e) => !ids.has(e.id)) }))
    }
    drain()
    const unsub = useEvents.subscribe(drain)
    return () => {
      unsub()
      clearTimeout(xpTimer.current)
    }
  }, [])

  const visible = toasts.slice(-3)
  const hidden = toasts.length - visible.length

  return (
    <>
      {/* XP 浮层 */}
      <div className="pointer-events-none fixed right-3 bottom-[calc(env(safe-area-inset-bottom)+100px)] z-[90] md:bottom-auto md:right-6 md:top-[72px]" aria-live="polite">
        <AnimatePresence>
          {xp && (
            <motion.div
              key={xp.key}
              initial={{ opacity: 0, y: -10, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ type: 'spring', stiffness: 500, damping: 30 }}
              className="flex items-center gap-2 rounded-full border border-gold/40 bg-surface py-1.5 pl-1.5 pr-3.5 shadow-md"
            >
              <span className="grid h-7 w-7 place-items-center rounded-full bg-gold-soft ink-gold">
                <Sparkles size={15} />
              </span>
              <motion.span key={xp.amount} initial={{ scale: 1.35 }} animate={{ scale: 1 }} className="text-[15px] font-bold tabular-nums ink-gold">
                +{xp.amount} XP
              </motion.span>
              <span className="hidden max-w-[12rem] truncate text-xs text-muted sm:inline">{xp.reason}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Toast 区：手机顶部，桌面右下 */}
      <div
        className="pointer-events-none fixed inset-x-3 top-[calc(env(safe-area-inset-top)+64px)] z-[95] flex flex-col items-center gap-2 md:inset-x-auto md:bottom-6 md:right-6 md:top-auto md:w-[360px] md:items-stretch"
        aria-live="polite"
      >
        <AnimatePresence initial={false}>
          {visible.map((t) => (
            <motion.div
              key={t.id}
              layout
              initial={{ opacity: 0, y: -16, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96, transition: { duration: 0.18 } }}
              transition={{ type: 'spring', stiffness: 460, damping: 32 }}
              className="pointer-events-auto w-full max-w-[400px]"
            >
              <ToastCard t={t} onClose={() => setToasts((ts) => ts.filter((x) => x.id !== t.id))} />
            </motion.div>
          ))}
        </AnimatePresence>
        {hidden > 0 && (
          <div className="pointer-events-auto rounded-full border border-border bg-surface px-3 py-1 text-xs text-muted shadow-sm">另有 {hidden} 条通知</div>
        )}
      </div>

      {/* 升级弹窗 */}
      <Modal open={levelUp != null} onClose={() => setLevelUp(null)} size="sm" hideClose>
        {levelUp != null && <LevelUpBody level={levelUp} onClose={() => setLevelUp(null)} />}
      </Modal>
    </>
  )
}

function LevelUpBody({ level, onClose }: { level: number; onClose: () => void }) {
  const xp = useProgress((s) => s.xp)
  const title = titleForLevel(level)
  const newTitle = titleForLevel(level - 1) !== title
  return (
    <div className="flex flex-col items-center pt-4 text-center">
      <motion.div
        initial={{ scale: 0.4, rotate: -12, opacity: 0 }}
        animate={{ scale: 1, rotate: 0, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 260, damping: 14, delay: 0.05 }}
        className="relative grid h-28 w-28 place-items-center rounded-[32px] bg-gradient-to-br from-accent to-llm text-accent-fg shadow-lg"
      >
        <span className="text-[13px] font-semibold opacity-80">LEVEL</span>
        <span className="-mt-3 text-5xl font-black tabular-nums">{level}</span>
      </motion.div>
      <div className="mt-5 text-xl font-bold">升级了！</div>
      <div className="mt-1 text-sm text-muted">
        {newTitle ? (
          <>
            解锁新称号 <span className="font-semibold text-fg">「{title}」</span>
          </>
        ) : (
          <>当前称号「{title}」 · 距下一级还需 {Math.max(0, xpToReach(level + 1) - xp)} XP</>
        )}
      </div>
      <Button variant="primary" size="lg" className="mt-6 w-full" onClick={onClose} data-autofocus>
        继续闯关
      </Button>
    </div>
  )
}

function ToastCard({ t, onClose }: { t: Toast; onClose: () => void }) {
  let icon: React.ReactNode
  let label: string
  let title: string
  let desc: string | undefined
  let iconCls: string
  if (t.kind === 'achievement') {
    const a = ACHIEVEMENT_BY_ID.get(t.achievementId)
    const Icon = a?.icon ?? Sparkles
    icon = <Icon size={20} />
    iconCls = TONE_BG[a?.tone ?? 'gold']
    label = '解锁成就'
    title = a?.title ?? t.achievementId
    desc = a?.desc
  } else if (t.kind === 'stars') {
    icon = <Star size={20} className="fill-current" />
    iconCls = TONE_BG.gold
    label = t.stars === 3 ? '三星通关！' : t.stars === 1 ? '关卡通关' : '星级提升'
    title = t.title ?? '关卡'
    desc = undefined
  } else {
    icon = <Flame size={20} />
    iconCls = TONE_BG.warning
    label = '连续学习'
    title = `已连续 ${t.days} 天`
    desc = '保持节奏，明天继续'
  }
  return (
    <div className={cn('flex items-center gap-3 rounded-2xl border border-border bg-surface p-3 pr-2 shadow-lg', t.kind === 'stars' && t.stars === 3 && 'border-gold/50')}>
      <div className={cn('grid h-11 w-11 shrink-0 place-items-center rounded-xl', iconCls)}>{icon}</div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wide text-subtle">
          {label}
          {t.kind === 'stars' && <Stars value={t.stars} size={11} />}
        </div>
        <div className="truncate text-sm font-semibold">{title}</div>
        {desc && <div className="truncate text-xs text-muted">{desc}</div>}
      </div>
      <button type="button" onClick={onClose} aria-label="关闭通知" className="grid h-8 w-8 shrink-0 place-items-center rounded-lg text-subtle hover:bg-surface-2 hover:text-fg">
        <X size={15} />
      </button>
    </div>
  )
}
