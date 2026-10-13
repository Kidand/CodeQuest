/**
 * 游戏化规则：XP、玩家等级、星级。纯函数，无副作用。
 * 成就（achievements）定义由平台层在 src/lib/achievements.ts 中扩展。
 */

export const XP = {
  learnSection: 4, // 每读完一节（首次）
  learnDone: 20, // 图解阶段全部完成（首次）
  checkpoint: 3, // 节末小测答对（首次）
  quizCorrect: 5, // 随堂练习：每“新增”一道答对（按历史最佳的增量发放，防刷）
  quizPerfect: 20, // 首次全对奖励
  guided: 30, // 每个跟练首次完成
  challenge: { easy: 40, medium: 70, hard: 110 } as const,
  qaCard: 2, // 面试追问首次自评
  review: 3, // 复习答对
  mockFinish: 50, // 完成一次模拟面试
} as const

/** 挑战奖励：看过题解 ×0.5；每用一条提示 -10%（最低 50%） */
export function challengeXp(difficulty: keyof typeof XP.challenge, hintsUsed: number, usedSolution: boolean) {
  const base = XP.challenge[difficulty]
  if (usedSolution) return Math.round(base * 0.5)
  return Math.round(base * Math.max(0.5, 1 - 0.1 * hintsUsed))
}

/** 玩家等级：升到 L+1 级累计需要 50·L·(L+1) XP（L1→2: 100, L2→3: 300, …） */
export function xpToReach(level: number) {
  return 50 * (level - 1) * level
}
export function levelFromXp(xp: number) {
  let lv = 1
  while (xpToReach(lv + 1) <= xp) lv++
  const cur = xpToReach(lv)
  const next = xpToReach(lv + 1)
  return { level: lv, intoLevel: xp - cur, levelSpan: next - cur, progress: (xp - cur) / (next - cur) }
}

const TITLES: [number, string][] = [
  [1, '算法实习生'],
  [3, '初级算法工程师'],
  [6, '算法工程师'],
  [9, '高级算法工程师'],
  [12, '资深算法专家'],
  [16, '大模型架构师'],
  [20, '首席科学家'],
]
export function titleForLevel(level: number) {
  let t = TITLES[0][1]
  for (const [lv, name] of TITLES) if (level >= lv) t = name
  return t
}

/** 关卡结构摘要（关卡正文加载后登记，用于计算星级） */
export interface LevelShape {
  sections: number
  quiz: number
  guided: string[]
  challenges: string[]
  /** 面试追问闪卡数量（可选；登记后章节页可显示「追问」阶段进度） */
  qa?: number
}

export interface ChallengeRecord {
  passed: boolean
  /** 通过前是否看过题解 */
  usedSolution: boolean
  hintsUsed: number
  attempts: number
  passedAt?: string
  /** 最快通过用时（秒） */
  bestTimeSec?: number
  /** 题目难度（passChallenge 时记录，供成就判定） */
  difficulty?: 'easy' | 'medium' | 'hard'
}

export interface LevelProgress {
  shape?: LevelShape
  sectionsRead: string[]
  checkpointsCorrect: string[]
  learnDone: boolean
  /** 历史最佳答对数与比例 */
  quizBestCorrect: number
  quizBest: number
  quizAttempts: number
  guidedDone: string[]
  challenges: Record<string, ChallengeRecord>
  qaRated: string[]
  firstCompletedAt?: string
  lastVisitedAt?: string
}

export const emptyLevelProgress = (): LevelProgress => ({
  sectionsRead: [],
  checkpointsCorrect: [],
  learnDone: false,
  quizBestCorrect: 0,
  quizBest: 0,
  quizAttempts: 0,
  guidedDone: [],
  challenges: {},
  qaRated: [],
})

/**
 * 星级：
 *  ★   图解完成 且 随堂练习 ≥ 60%（= 通关，解锁下一关）
 *  ★★  随堂练习 ≥ 80% 且 全部跟练完成
 *  ★★★ 全部实战通过，且通过前未看题解
 * 题目数为 0 的阶段视为已满足。
 */
export function starsOf(p: LevelProgress | undefined): 0 | 1 | 2 | 3 {
  if (!p) return 0
  const shape = p.shape
  const quizOk = (min: number) => !shape || shape.quiz === 0 || p.quizBest >= min
  if (!(p.learnDone && quizOk(0.6) && p.quizAttempts + (shape?.quiz === 0 ? 1 : 0) > 0)) return 0
  const guidedAll = !shape || shape.guided.every((g) => p.guidedDone.includes(g))
  if (!(quizOk(0.8) && guidedAll)) return 1
  const chAll =
    !shape ||
    shape.challenges.every((c) => p.challenges[c]?.passed && !p.challenges[c]?.usedSolution)
  return chAll ? 3 : 2
}

export const isLevelCleared = (p: LevelProgress | undefined) => starsOf(p) >= 1

/** 本地日期 YYYY-MM-DD */
export function today(d = new Date()) {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}
export function daysBetween(a: string, b: string) {
  const da = new Date(a + 'T00:00:00')
  const db = new Date(b + 'T00:00:00')
  return Math.round((db.getTime() - da.getTime()) / 86400000)
}
export function addDays(date: string, n: number) {
  const d = new Date(date + 'T00:00:00')
  d.setDate(d.getDate() + n)
  return today(d)
}
