import { BookOpen, Code2, ListChecks, MessagesSquare, PenLine, type LucideIcon } from 'lucide-react'
import { STAGES, type LevelContent, type Stage } from '@/content/types'
import type { LevelProgress } from '@/lib/game'

export const STAGE_META: Record<Stage, { label: string; short: string; icon: LucideIcon; desc: string }> = {
  learn: { label: '图解', short: '图解', icon: BookOpen, desc: '建立直觉' },
  quiz: { label: '随堂练习', short: '练习', icon: ListChecks, desc: '检验理解' },
  guided: { label: '跟练', short: '跟练', icon: PenLine, desc: '逐空写出' },
  challenge: { label: '实战手撕', short: '实战', icon: Code2, desc: '从零手撕' },
  qa: { label: '面试追问', short: '追问', icon: MessagesSquare, desc: '讲清原理' },
}

export const isStage = (s: string | undefined): s is Stage => !!s && (STAGES as string[]).includes(s)

/** 阶段是否有内容 */
export function stageHasContent(stage: Stage, L: LevelContent) {
  switch (stage) {
    case 'learn':
      return L.learn.length > 0
    case 'quiz':
      return L.quiz.length > 0
    case 'guided':
      return L.guided.length > 0
    case 'challenge':
      return L.challenges.length > 0
    case 'qa':
      return L.interview.length > 0
  }
}

export interface StageState {
  stage: Stage
  empty: boolean
  done: boolean
  /** 0..1 */
  ratio: number
  /** 简短进度文案 */
  text: string
}

export function stageState(stage: Stage, L: LevelContent, p: LevelProgress | undefined): StageState {
  const empty = !stageHasContent(stage, L)
  const base = { stage, empty }
  if (empty) return { ...base, done: true, ratio: 1, text: '本关无此阶段' }
  switch (stage) {
    case 'learn': {
      const read = L.learn.filter((s) => p?.sectionsRead.includes(s.id)).length
      return { ...base, done: !!p?.learnDone, ratio: p?.learnDone ? 1 : read / L.learn.length, text: `${read}/${L.learn.length} 节` }
    }
    case 'quiz': {
      const best = p?.quizBest ?? 0
      const tried = (p?.quizAttempts ?? 0) > 0
      return { ...base, done: tried && best >= 0.6, ratio: best, text: tried ? `最佳 ${Math.round(best * 100)}%` : `${L.quiz.length} 题` }
    }
    case 'guided': {
      const n = L.guided.filter((g) => p?.guidedDone.includes(g.id)).length
      return { ...base, done: n === L.guided.length, ratio: n / L.guided.length, text: `${n}/${L.guided.length}` }
    }
    case 'challenge': {
      const n = L.challenges.filter((c) => p?.challenges[c.id]?.passed).length
      return { ...base, done: n === L.challenges.length, ratio: n / L.challenges.length, text: `${n}/${L.challenges.length} 通过` }
    }
    case 'qa': {
      const n = L.interview.filter((q) => p?.qaRated.includes(q.id)).length
      return { ...base, done: n === L.interview.length, ratio: n / L.interview.length, text: `${n}/${L.interview.length}` }
    }
  }
}

/** 推荐阶段：第一个未完成且有内容的阶段；全部完成则回到图解 */
export function recommendedStage(L: LevelContent, p: LevelProgress | undefined): Stage {
  for (const s of STAGES) {
    const st = stageState(s, L, p)
    if (!st.empty && !st.done) return s
  }
  return STAGES.find((s) => stageHasContent(s, L)) ?? 'learn'
}

/** 下一个有内容的阶段 */
export function nextStage(stage: Stage, L: LevelContent): Stage | null {
  const i = STAGES.indexOf(stage)
  for (let k = i + 1; k < STAGES.length; k++) if (stageHasContent(STAGES[k], L)) return STAGES[k]
  return null
}
