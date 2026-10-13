/**
 * 全局进度 Store（zustand + localStorage 持久化）。
 *
 * 约定：所有“发放 XP”的动作都通过本文件的 action 完成，action 内部负责
 *  - 防重复发放（首次 / 增量）
 *  - 更新连续打卡、每日活动
 *  - 推送 GameEvent（XP 浮层、升级、成就由平台层消费 useEvents）
 */
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { accountKey } from './account-storage'
import {
  XP,
  challengeXp,
  emptyLevelProgress,
  levelFromXp,
  starsOf,
  today,
  daysBetween,
  addDays,
  type LevelProgress,
  type LevelShape,
} from './game'

export type ThemePref = 'light' | 'dark' | 'system'

export interface Settings {
  theme: ThemePref
  /** 自由模式：解锁全部关卡（考前突击按需跳转） */
  freeMode: boolean
  editorFontSize: number
  reduceMotion: boolean
  /** 每日 XP 目标 */
  dailyGoal: number
  /** 音效（预留） */
  sound: boolean
}

/** 间隔复习条目（错题 / 面试闪卡） */
export interface ReviewItem {
  kind: 'question' | 'qa'
  levelId: string
  /** 题目 id 或 闪卡 id */
  itemId: string
  /** 来源阶段，如 'quiz' | 'checkpoint' | 'qa' */
  source: string
  wrongCount: number
  /** 连续答对次数，达到 3 视为掌握 */
  streak: number
  /** 当前间隔（天） */
  interval: number
  /** 下次复习日期 YYYY-MM-DD */
  due: string
  lastAt: string
  mastered: boolean
}

export interface DailyStat {
  xp: number
  answered: number
  correct: number
  solved: number
}

export interface ProgressState {
  xp: number
  levels: Record<string, LevelProgress>
  streak: { current: number; best: number; lastDate?: string }
  daily: Record<string, DailyStat>
  achievements: Record<string, string>
  review: Record<string, ReviewItem>
  settings: Settings
  lastVisited?: { levelId: string; stage: string }
  createdAt: string

  /* ---------- actions ---------- */
  registerLevelShape: (levelId: string, shape: LevelShape) => void
  visitLevel: (levelId: string, stage: string) => void
  markSectionRead: (levelId: string, sectionId: string) => void
  answerCheckpoint: (levelId: string, sectionId: string, correct: boolean, questionId?: string) => void
  completeLearn: (levelId: string) => void
  /** 记录单题作答（随堂/复习/模拟面试共用）：更新每日统计 + 错题本 */
  recordAnswer: (levelId: string, questionId: string, correct: boolean, source: string) => void
  completeQuiz: (levelId: string, correct: number, total: number) => number
  completeGuided: (levelId: string, guidedId: string) => number
  takeChallengeHint: (levelId: string, challengeId: string) => void
  viewChallengeSolution: (levelId: string, challengeId: string) => void
  attemptChallenge: (levelId: string, challengeId: string) => void
  passChallenge: (
    levelId: string,
    challengeId: string,
    difficulty: 'easy' | 'medium' | 'hard',
    timeSec?: number,
  ) => number
  rateQA: (levelId: string, qaId: string, rating: 'know' | 'fuzzy' | 'dont') => void
  /** 复习结果：答对拉长间隔，答错重置 */
  reviewResult: (key: string, correct: boolean) => void
  /** 从复习池移除（题目已下线 / 用户手动移出） */
  dismissReview: (key: string) => void
  awardXp: (amount: number, reason: string) => void
  unlockAchievement: (id: string) => boolean
  updateSettings: (patch: Partial<Settings>) => void
  importState: (json: string) => void
  resetAll: () => void
}

/* ------------------------------------------------------------------ */
/* 事件队列（非持久化）：平台层订阅并渲染 XP 浮层 / 升级 / 成就弹窗           */
/* ------------------------------------------------------------------ */
export type GameEvent =
  | { id: number; type: 'xp'; amount: number; reason: string }
  | { id: number; type: 'levelup'; level: number }
  | { id: number; type: 'stars'; levelId: string; stars: number }
  | { id: number; type: 'achievement'; achievementId: string }
  | { id: number; type: 'streak'; days: number }

type NewEvent = GameEvent extends infer E ? (E extends GameEvent ? Omit<E, 'id'> : never) : never

let eventSeq = 1
export const useEvents = create<{ queue: GameEvent[]; push: (e: NewEvent) => void; shift: () => void; clear: () => void }>(
  (set) => ({
    queue: [],
    push: (e) => set((s) => ({ queue: [...s.queue, { ...e, id: eventSeq++ } as GameEvent] })),
    shift: () => set((s) => ({ queue: s.queue.slice(1) })),
    clear: () => set({ queue: [] }),
  }),
)
const emit = (e: NewEvent) => useEvents.getState().push(e)

/* ------------------------------------------------------------------ */

export const reviewKey = (levelId: string, itemId: string, kind: ReviewItem['kind'] = 'question') =>
  `${kind}::${levelId}::${itemId}`

const defaultSettings: Settings = {
  theme: 'system',
  freeMode: false,
  editorFontSize: 14,
  reduceMotion: false,
  dailyGoal: 100,
  sound: false,
}

export const initialData = () => ({
  xp: 0,
  levels: {} as Record<string, LevelProgress>,
  streak: { current: 0, best: 0 } as ProgressState['streak'],
  daily: {} as Record<string, DailyStat>,
  achievements: {} as Record<string, string>,
  review: {} as Record<string, ReviewItem>,
  settings: defaultSettings,
  lastVisited: undefined as ProgressState['lastVisited'],
  createdAt: new Date().toISOString(),
})

const emptyDaily = (): DailyStat => ({ xp: 0, answered: 0, correct: 0, solved: 0 })

export const useProgress = create<ProgressState>()(
  persist(
    (set, get) => {
      /** 读取（不存在则创建）关卡进度并以不可变方式更新 */
      const patchLevel = (levelId: string, fn: (p: LevelProgress) => LevelProgress) => {
        const before = get().levels[levelId]
        const cur = before ?? emptyLevelProgress()
        const next = fn({ ...cur })
        const starsBefore = starsOf(before)
        const starsAfter = starsOf(next)
        if (starsAfter >= 1 && !next.firstCompletedAt) next.firstCompletedAt = new Date().toISOString()
        set((s) => ({ levels: { ...s.levels, [levelId]: next } }))
        if (starsAfter > starsBefore) emit({ type: 'stars', levelId, stars: starsAfter })
      }

      const touchDaily = (fn: (d: DailyStat) => DailyStat) => {
        const d = today()
        set((s) => ({ daily: { ...s.daily, [d]: fn({ ...(s.daily[d] ?? emptyDaily()) }) } }))
      }

      const bumpStreak = () => {
        const d = today()
        const { streak } = get()
        if (streak.lastDate === d) return
        const gap = streak.lastDate ? daysBetween(streak.lastDate, d) : Infinity
        const current = gap === 1 ? streak.current + 1 : 1
        set({ streak: { current, best: Math.max(streak.best, current), lastDate: d } })
        if (current > 1) emit({ type: 'streak', days: current })
      }

      const awardXp = (amount: number, reason: string) => {
        if (amount <= 0) return
        const before = levelFromXp(get().xp).level
        set((s) => ({ xp: s.xp + amount }))
        touchDaily((dd) => ({ ...dd, xp: dd.xp + amount }))
        bumpStreak()
        emit({ type: 'xp', amount, reason })
        const after = levelFromXp(get().xp).level
        if (after > before) emit({ type: 'levelup', level: after })
      }

      const upsertReview = (key: string, base: Omit<ReviewItem, 'wrongCount' | 'streak' | 'interval' | 'due' | 'lastAt' | 'mastered'>, correct: boolean) => {
        const cur = get().review[key]
        const d = today()
        if (correct) {
          if (!cur) return // 首次就答对：不进错题本
          const streak = cur.streak + 1
          const interval = Math.min(60, Math.max(1, Math.round(cur.interval * 2.2)))
          set((s) => ({
            review: { ...s.review, [key]: { ...cur, streak, interval, due: addDays(d, interval), lastAt: d, mastered: streak >= 3 } },
          }))
        } else {
          const item: ReviewItem = {
            ...base,
            wrongCount: (cur?.wrongCount ?? 0) + 1,
            streak: 0,
            interval: 1,
            due: d, // 当天即可复习
            lastAt: d,
            mastered: false,
          }
          set((s) => ({ review: { ...s.review, [key]: item } }))
        }
      }

      return {
        ...initialData(),

        registerLevelShape: (levelId, shape) =>
          patchLevel(levelId, (p) => ({ ...p, shape })),

        visitLevel: (levelId, stage) => {
          set({ lastVisited: { levelId, stage } })
          patchLevel(levelId, (p) => ({ ...p, lastVisitedAt: new Date().toISOString() }))
        },

        markSectionRead: (levelId, sectionId) => {
          const p = get().levels[levelId]
          if (p?.sectionsRead.includes(sectionId)) return
          patchLevel(levelId, (q) => ({ ...q, sectionsRead: [...q.sectionsRead, sectionId] }))
          awardXp(XP.learnSection, '读完一节图解')
        },

        answerCheckpoint: (levelId, sectionId, correct, questionId) => {
          touchDaily((d) => ({ ...d, answered: d.answered + 1, correct: d.correct + (correct ? 1 : 0) }))
          if (questionId) upsertReview(reviewKey(levelId, questionId), { kind: 'question', levelId, itemId: questionId, source: 'checkpoint' }, correct)
          if (!correct) return
          const p = get().levels[levelId]
          if (p?.checkpointsCorrect.includes(sectionId)) return
          patchLevel(levelId, (q) => ({ ...q, checkpointsCorrect: [...q.checkpointsCorrect, sectionId] }))
          awardXp(XP.checkpoint, '节末小测答对')
        },

        completeLearn: (levelId) => {
          if (get().levels[levelId]?.learnDone) return
          patchLevel(levelId, (p) => ({ ...p, learnDone: true }))
          awardXp(XP.learnDone, '完成图解')
        },

        recordAnswer: (levelId, questionId, correct, source) => {
          touchDaily((d) => ({ ...d, answered: d.answered + 1, correct: d.correct + (correct ? 1 : 0) }))
          upsertReview(reviewKey(levelId, questionId), { kind: 'question', levelId, itemId: questionId, source }, correct)
        },

        completeQuiz: (levelId, correct, total) => {
          const p = get().levels[levelId] ?? emptyLevelProgress()
          const gainCount = Math.max(0, correct - p.quizBestCorrect)
          const ratio = total > 0 ? correct / total : 1
          const perfectFirst = ratio === 1 && p.quizBest < 1
          patchLevel(levelId, (q) => ({
            ...q,
            quizAttempts: q.quizAttempts + 1,
            quizBestCorrect: Math.max(q.quizBestCorrect, correct),
            quizBest: Math.max(q.quizBest, ratio),
          }))
          const gained = gainCount * XP.quizCorrect + (perfectFirst ? XP.quizPerfect : 0)
          awardXp(gained, perfectFirst ? '随堂练习全对！' : '完成随堂练习')
          return gained
        },

        completeGuided: (levelId, guidedId) => {
          if (get().levels[levelId]?.guidedDone.includes(guidedId)) return 0
          patchLevel(levelId, (p) => ({ ...p, guidedDone: [...p.guidedDone, guidedId] }))
          awardXp(XP.guided, '完成跟练')
          return XP.guided
        },

        takeChallengeHint: (levelId, challengeId) =>
          patchLevel(levelId, (p) => {
            const c = p.challenges[challengeId] ?? { passed: false, usedSolution: false, hintsUsed: 0, attempts: 0 }
            if (c.passed) return p
            return { ...p, challenges: { ...p.challenges, [challengeId]: { ...c, hintsUsed: c.hintsUsed + 1 } } }
          }),

        viewChallengeSolution: (levelId, challengeId) =>
          patchLevel(levelId, (p) => {
            const c = p.challenges[challengeId] ?? { passed: false, usedSolution: false, hintsUsed: 0, attempts: 0 }
            if (c.passed) return p
            return { ...p, challenges: { ...p.challenges, [challengeId]: { ...c, usedSolution: true } } }
          }),

        attemptChallenge: (levelId, challengeId) =>
          patchLevel(levelId, (p) => {
            const c = p.challenges[challengeId] ?? { passed: false, usedSolution: false, hintsUsed: 0, attempts: 0 }
            return { ...p, challenges: { ...p.challenges, [challengeId]: { ...c, attempts: c.attempts + 1 } } }
          }),

        passChallenge: (levelId, challengeId, difficulty, timeSec) => {
          const p = get().levels[levelId]
          const c = p?.challenges[challengeId] ?? { passed: false, usedSolution: false, hintsUsed: 0, attempts: 0 }
          const first = !c.passed
          patchLevel(levelId, (q) => ({
            ...q,
            challenges: {
              ...q.challenges,
              [challengeId]: {
                ...c,
                passed: true,
                passedAt: c.passedAt ?? new Date().toISOString(),
                difficulty,
                bestTimeSec: timeSec != null ? Math.min(c.bestTimeSec ?? Infinity, timeSec) : c.bestTimeSec,
              },
            },
          }))
          if (!first) return 0
          touchDaily((d) => ({ ...d, solved: d.solved + 1 }))
          const gained = challengeXp(difficulty, c.hintsUsed, c.usedSolution)
          awardXp(gained, '实战通过')
          return gained
        },

        rateQA: (levelId, qaId, rating) => {
          const first = !get().levels[levelId]?.qaRated.includes(qaId)
          if (first) {
            patchLevel(levelId, (p) => ({ ...p, qaRated: [...p.qaRated, qaId] }))
            awardXp(XP.qaCard, '面试追问自评')
          }
          upsertReview(
            reviewKey(levelId, qaId, 'qa'),
            { kind: 'qa', levelId, itemId: qaId, source: 'qa' },
            rating === 'know',
          )
        },

        reviewResult: (key, correct) => {
          const cur = get().review[key]
          if (!cur) return
          touchDaily((d) => ({ ...d, answered: d.answered + 1, correct: d.correct + (correct ? 1 : 0) }))
          const { kind, levelId, itemId, source } = cur
          upsertReview(key, { kind, levelId, itemId, source }, correct)
          if (correct && (cur.streak === 0 || cur.lastAt !== today())) awardXp(XP.review, '复习答对')
        },

        dismissReview: (key) =>
          set((s) => {
            if (!s.review[key]) return s
            const review = { ...s.review }
            delete review[key]
            return { review }
          }),

        awardXp,

        unlockAchievement: (id) => {
          if (get().achievements[id]) return false
          set((s) => ({ achievements: { ...s.achievements, [id]: new Date().toISOString() } }))
          emit({ type: 'achievement', achievementId: id })
          return true
        },

        updateSettings: (patch) => set((s) => ({ settings: { ...s.settings, ...patch } })),

        importState: (json) => {
          const data = JSON.parse(json)
          const src = data?.state ?? data
          if (typeof src !== 'object' || typeof src.xp !== 'number' || typeof src.levels !== 'object')
            throw new Error('不是有效的 CodeQuest 进度文件')
          const defaults = initialData()
          const fields = Object.fromEntries(Object.keys(defaults).filter((key) => key in src).map((key) => [key, src[key]]))
          set({ ...defaults, ...fields, settings: { ...defaultSettings, ...(src.settings ?? {}) } })
        },

        resetAll: () => set({ ...initialData(), settings: get().settings }),
      }
    },
    {
      name: accountKey('cq-progress'),
      version: 1,
      storage: createJSONStorage(() => localStorage),
      // settings 深合并：旧版本 / 外部写入的 settings 缺字段时用默认值补齐
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<ProgressState>
        return { ...current, ...p, settings: { ...current.settings, ...(p.settings ?? {}) } }
      },
      partialize: (s) => ({
        xp: s.xp,
        levels: s.levels,
        streak: s.streak,
        daily: s.daily,
        achievements: s.achievements,
        review: s.review,
        settings: s.settings,
        lastVisited: s.lastVisited,
        createdAt: s.createdAt,
      }),
    },
  ),
)

/** 导出进度 JSON（备份 / 跨设备迁移） */
export function exportProgress(): string {
  const s = useProgress.getState()
  const { xp, levels, streak, daily, achievements, review, settings, lastVisited, createdAt } = s
  return JSON.stringify({ app: 'codequest', version: 1, state: { xp, levels, streak, daily, achievements, review, settings, lastVisited, createdAt } }, null, 2)
}

/* ------------------------------------------------------------------ */
/* 代码草稿（单独持久化，避免主 store 过大）                              */
/* ------------------------------------------------------------------ */
export const useDrafts = create<{
  drafts: Record<string, string>
  setDraft: (key: string, code: string) => void
  clearDraft: (key: string) => void
}>()(
  persist(
    (set) => ({
      drafts: {},
      setDraft: (key, code) => set((s) => ({ drafts: { ...s.drafts, [key]: code } })),
      clearDraft: (key) =>
        set((s) => {
          const d = { ...s.drafts }
          delete d[key]
          return { drafts: d }
        }),
    }),
    { name: accountKey('cq-drafts'), storage: createJSONStorage(() => localStorage) },
  ),
)
export const draftKey = (levelId: string, kind: 'challenge' | 'guided', id: string) => `${kind}::${levelId}::${id}`
