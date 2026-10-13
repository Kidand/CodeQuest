/**
 * 成就定义 + 检测 hook。
 * 每个成就用 progress(ctx) → [当前, 目标] 描述，当前 ≥ 目标（且目标 > 0）即解锁。
 * useAchievementWatcher 在 Root 挂载一次：监听进度 / 模拟面试 store，满足条件时调用 unlockAchievement。
 */
import { useEffect } from 'react'
import { accountSwitchInProgress } from './account-storage'
import {
  Award,
  BookCheck,
  CalendarCheck,
  CheckCheck,
  Crown,
  Flag,
  Flame,
  GitMerge,
  Hammer,
  Repeat,
  ShieldCheck,
  Sparkles,
  SquareTerminal,
  Star,
  Swords,
  Target,
  Timer,
  Trophy,
  Zap,
  Gauge,
  type LucideIcon,
} from 'lucide-react'
import { loadCatalog, useCatalog, type Catalog } from '@/content/catalog'
import type { Tier } from '@/content/types'
import { levelFromXp, starsOf } from './game'
import { useProgress, type ProgressState } from './store'
import { useMock, type MockRecord } from './mock'

export interface AchievementCtx {
  state: Pick<ProgressState, 'xp' | 'levels' | 'streak' | 'daily' | 'review' | 'settings'>
  cat: Catalog | null
  mock: MockRecord[]
}

export type AchievementTone = 'gold' | 'accent' | 'algo' | 'llm' | 'success' | 'warning' | 'danger' | 'info'

export interface AchievementDef {
  id: string
  title: string
  desc: string
  icon: LucideIcon
  tone: AchievementTone
  progress: (ctx: AchievementCtx) => [number, number]
}

/* ---------------- 工具 ---------------- */
const levelsOf = (c: AchievementCtx) => Object.values(c.state.levels)
const clearedCount = (c: AchievementCtx) => levelsOf(c).filter((p) => starsOf(p) >= 1).length
const passedChallenges = (c: AchievementCtx) =>
  levelsOf(c).reduce((n, p) => n + Object.values(p?.challenges ?? {}).filter((x) => x.passed).length, 0)
const capped = (cur: number, target: number): [number, number] => [Math.min(cur, target), target]

function tierAll(c: AchievementCtx, tier: Tier): [number, number] {
  if (!c.cat) return [0, 0]
  const metas = c.cat.chapters.flatMap((ch) => ch.levels).filter((l) => l.tier === tier)
  return [metas.filter((m) => starsOf(c.state.levels[m.id]) >= 1).length, metas.length]
}

/* ---------------- 定义（≥ 15） ---------------- */
export const ACHIEVEMENTS: AchievementDef[] = [
  {
    id: 'first-clear',
    title: '初次通关',
    desc: '通关任意一关（拿到第一颗星）',
    icon: Flag,
    tone: 'success',
    progress: (c) => capped(clearedCount(c), 1),
  },
  {
    id: 'first-three-star',
    title: '完美通关',
    desc: '任意一关拿到三星',
    icon: Star,
    tone: 'gold',
    progress: (c) => capped(levelsOf(c).filter((p) => starsOf(p) === 3).length, 1),
  },
  {
    id: 'quiz-perfect',
    title: '满分答卷',
    desc: '任意一关的随堂练习全部答对',
    icon: CheckCheck,
    tone: 'success',
    progress: (c) => capped(levelsOf(c).filter((p) => p && p.quizAttempts > 0 && p.quizBest >= 1).length, 1),
  },
  {
    id: 'first-challenge',
    title: '首次手撕',
    desc: '通过第一道实战题',
    icon: SquareTerminal,
    tone: 'accent',
    progress: (c) => capped(passedChallenges(c), 1),
  },
  {
    id: 'challenges-10',
    title: '手撕熟练工',
    desc: '累计通过 10 道实战题',
    icon: Hammer,
    tone: 'accent',
    progress: (c) => capped(passedChallenges(c), 10),
  },
  {
    id: 'challenges-50',
    title: '人形编译器',
    desc: '累计通过 50 道实战题',
    icon: Swords,
    tone: 'llm',
    progress: (c) => capped(passedChallenges(c), 50),
  },
  {
    id: 'hard-clean',
    title: '硬核手撕',
    desc: '不看提示、不看题解，通过一道 hard 实战',
    icon: Zap,
    tone: 'danger',
    progress: (c) =>
      capped(
        levelsOf(c).filter((p) =>
          Object.values(p?.challenges ?? {}).some((x) => x.passed && x.difficulty === 'hard' && x.hintsUsed === 0 && !x.usedSolution),
        ).length,
        1,
      ),
  },
  {
    id: 'speedrun',
    title: '快准狠',
    desc: '5 分钟内通过一道 medium 及以上的实战',
    icon: Gauge,
    tone: 'warning',
    progress: (c) =>
      capped(
        levelsOf(c).filter((p) =>
          Object.values(p?.challenges ?? {}).some(
            (x) => x.passed && (x.difficulty === 'medium' || x.difficulty === 'hard') && x.bestTimeSec != null && x.bestTimeSec <= 300,
          ),
        ).length,
        1,
      ),
  },
  {
    id: 'streak-3',
    title: '渐入佳境',
    desc: '连续学习 3 天',
    icon: Flame,
    tone: 'warning',
    progress: (c) => capped(c.state.streak.best, 3),
  },
  {
    id: 'streak-7',
    title: '一周不断',
    desc: '连续学习 7 天',
    icon: Flame,
    tone: 'warning',
    progress: (c) => capped(c.state.streak.best, 7),
  },
  {
    id: 'streak-30',
    title: '月度坚持',
    desc: '连续学习 30 天',
    icon: CalendarCheck,
    tone: 'danger',
    progress: (c) => capped(c.state.streak.best, 30),
  },
  {
    id: 'daily-goal',
    title: '今日达标',
    desc: '单日 XP 达到每日目标',
    icon: Target,
    tone: 'accent',
    progress: (c) => {
      const goal = c.state.settings.dailyGoal
      const best = Math.max(0, ...Object.values(c.state.daily).map((d) => d.xp))
      return capped(best, goal)
    },
  },
  {
    id: 'chapter-clear',
    title: '一章通关',
    desc: '通关某一章的全部关卡（含 BOSS）',
    icon: BookCheck,
    tone: 'success',
    progress: (c) => {
      if (!c.cat) return [0, 0]
      const any = c.cat.chapters.some((ch) => ch.levels.length > 0 && ch.levels.every((l) => starsOf(c.state.levels[l.id]) >= 1))
      return [any ? 1 : 0, 1]
    },
  },
  {
    id: 'boss-clear',
    title: 'BOSS 终结者',
    desc: '击败任意一个章节 BOSS',
    icon: Crown,
    tone: 'gold',
    progress: (c) => {
      if (!c.cat) return [0, 0]
      const n = c.cat.chapters.flatMap((ch) => ch.levels).filter((l) => l.kind === 'boss' && starsOf(c.state.levels[l.id]) >= 1).length
      return capped(n, 1)
    },
  },
  {
    id: 'both-tracks',
    title: '双线并进',
    desc: '经典算法与大模型两条路线各通关至少 1 关',
    icon: GitMerge,
    tone: 'info',
    progress: (c) => {
      if (!c.cat) return [0, 0]
      const has = (t: 'algo' | 'llm') => c.cat!.byTrack[t].some((ch) => ch.levels.some((l) => starsOf(c.state.levels[l.id]) >= 1))
      return [(has('algo') ? 1 : 0) + (has('llm') ? 1 : 0), 2]
    },
  },
  {
    id: 'all-must',
    title: '必考全通',
    desc: '通关全部「必考」关卡',
    icon: ShieldCheck,
    tone: 'algo',
    progress: (c) => tierAll(c, 'must'),
  },
  {
    id: 'all-modern',
    title: '新必会全通',
    desc: '通关全部「新必会」关卡',
    icon: ShieldCheck,
    tone: 'llm',
    progress: (c) => tierAll(c, 'modern'),
  },
  {
    id: 'all-frontier',
    title: '前沿探索者',
    desc: '通关全部「前沿」关卡',
    icon: Sparkles,
    tone: 'llm',
    progress: (c) => tierAll(c, 'frontier'),
  },
  {
    id: 'stars-30',
    title: '星光熠熠',
    desc: '累计获得 30 颗星',
    icon: Star,
    tone: 'gold',
    progress: (c) => capped(levelsOf(c).reduce((n, p) => n + starsOf(p), 0), 30),
  },
  {
    id: 'review-10',
    title: '温故知新',
    desc: '在复习池中掌握 10 道错题 / 闪卡',
    icon: Repeat,
    tone: 'info',
    progress: (c) => capped(Object.values(c.state.review).filter((r) => r.mastered).length, 10),
  },
  {
    id: 'first-mock',
    title: '初上考场',
    desc: '完成一次模拟面试',
    icon: Timer,
    tone: 'accent',
    progress: (c) => capped(c.mock.length, 1),
  },
  {
    id: 'mock-80',
    title: '面试稳了',
    desc: '模拟面试得分达到 80',
    icon: Trophy,
    tone: 'gold',
    progress: (c) => capped(Math.max(0, ...c.mock.map((m) => m.score)), 80),
  },
  {
    id: 'level-10',
    title: '资深玩家',
    desc: '玩家等级达到 10 级',
    icon: Award,
    tone: 'gold',
    progress: (c) => capped(levelFromXp(c.state.xp).level, 10),
  },
]

export const ACHIEVEMENT_BY_ID = new Map(ACHIEVEMENTS.map((a) => [a.id, a]))

/** 成就进度（成就墙展示用） */
export function achievementProgress(a: AchievementDef, ctx: AchievementCtx) {
  const [cur, target] = a.progress(ctx)
  return { cur, target, ratio: target > 0 ? Math.min(1, cur / target) : 0 }
}

function makeCtx(cat: Catalog | null): AchievementCtx {
  const s = useProgress.getState()
  return {
    state: { xp: s.xp, levels: s.levels, streak: s.streak, daily: s.daily, review: s.review, settings: s.settings },
    cat,
    mock: useMock.getState().history,
  }
}

export function useAchievementCtx(): AchievementCtx {
  const cat = useCatalog()
  const xp = useProgress((s) => s.xp)
  const levels = useProgress((s) => s.levels)
  const streak = useProgress((s) => s.streak)
  const daily = useProgress((s) => s.daily)
  const review = useProgress((s) => s.review)
  const settings = useProgress((s) => s.settings)
  const mock = useMock((s) => s.history)
  return { state: { xp, levels, streak, daily, review, settings }, cat, mock }
}

/** 在 Root 挂载一次：检测并解锁成就 */
export function useAchievementWatcher() {
  useEffect(() => {
    let cat: Catalog | null = null
    let running = false
    const check = () => {
      if (running || accountSwitchInProgress) return
      running = true
      try {
        const ctx = makeCtx(cat)
        const st = useProgress.getState()
        for (const a of ACHIEVEMENTS) {
          if (st.achievements[a.id]) continue
          const [cur, target] = a.progress(ctx)
          if (target > 0 && cur >= target) st.unlockAchievement(a.id)
        }
      } finally {
        running = false
      }
    }
    let alive = true
    loadCatalog().then((c) => {
      if (!alive) return
      cat = c
      check()
    })
    check()
    const u1 = useProgress.subscribe(check)
    const u2 = useMock.subscribe(check)
    return () => {
      alive = false
      u1()
      u2()
    }
  }, [])
}
