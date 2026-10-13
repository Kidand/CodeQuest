/**
 * 派生数据（纯函数 + 少量 hook）：路线 / 章节统计、分层覆盖、阶段进度、复习队列、推荐、热力图。
 * 页面只读这里的结果，不在组件里重复写统计逻辑。
 */
import { useEffect, useMemo, useState } from 'react'
import { loadLevel, type Catalog } from '@/content/catalog'
import type { Challenge, ChapterDef, InterviewQA, LevelContent, LevelMeta, Question, Stage, Tier, Track } from '@/content/types'
import { addDays, starsOf, today, type LevelProgress } from './game'
import { useProgress, type DailyStat, type ReviewItem } from './store'
import { chapterStatuses, type LevelStatus, type LevelsProgress } from './unlock'

/* ------------------------------------------------------------------ */
/* 元信息                                                              */
/* ------------------------------------------------------------------ */

export const TRACKS: Track[] = ['algo', 'llm']

export const TRACK_META: Record<Track, { title: string; short: string; desc: string; tiers: Tier[] }> = {
  algo: {
    title: '经典算法',
    short: '算法',
    desc: '二分、双指针、排序、链表、树与图、DP、回溯——大模型岗手撕环节的基本盘',
    tiers: ['must', 'high', 'advanced'],
  },
  llm: {
    title: '大模型',
    short: '大模型',
    desc: '从 Softmax 到 MLA、MoE、GRPO 与投机解码——能手撕、能推导、能讲清',
    tiers: ['foundation', 'modern', 'frontier'],
  },
}

export const TIER_DESC: Record<Tier, string> = {
  must: '手撕出现率最高，写不出来基本挂',
  high: '经常出现，或作为必考题的追问 / 变体',
  advanced: '难度高或偏冷门，用来拉开差距',
  foundation: 'Transformer 时代的基本功，人人都会被考',
  modern: '2023–2025 年成为主流标配，面试默认你懂',
  frontier: '2025–2026 新进展，用来区分候选人',
}

export const STAGE_LABEL: Record<Stage, string> = {
  learn: '图解',
  quiz: '随堂',
  guided: '跟练',
  challenge: '实战',
  qa: '追问',
}

export const DIFFICULTY_LABEL = ['', '入门', '简单', '中等', '较难', '困难'] as const

/** 推荐排序用的分层优先级：必考 / 新必会 最优先 */
const TIER_PRIORITY: Record<Tier, number> = { must: 0, modern: 0, foundation: 1, high: 1, advanced: 2, frontier: 2 }
const FREQ_PRIORITY = { high: 0, medium: 1, low: 2 } as const

/* ------------------------------------------------------------------ */
/* 阶段进度                                                            */
/* ------------------------------------------------------------------ */

export interface StageProgress {
  stage: Stage
  done: number
  /** 未登记关卡结构时为 null（不知道总数） */
  total: number | null
  ratio: number
  complete: boolean
}

export function stageProgress(p: LevelProgress | undefined): StageProgress[] {
  const s = p?.shape
  const mk = (stage: Stage, done: number, total: number | null, complete?: boolean): StageProgress => {
    const ratio = total == null ? (done > 0 ? 1 : 0) : total === 0 ? (complete ? 1 : 0) : Math.min(1, done / total)
    return { stage, done, total, ratio, complete: complete ?? (total != null && total > 0 && done >= total) }
  }
  if (!p) return (['learn', 'quiz', 'guided', 'challenge', 'qa'] as Stage[]).map((st) => mk(st, 0, null, false))
  const passed = Object.values(p.challenges).filter((c) => c.passed).length
  return [
    mk('learn', p.learnDone ? (s?.sections ?? 1) : p.sectionsRead.length, s ? s.sections : null, p.learnDone),
    // 随堂：用历史最佳正确率表示，≥ 60% 视为完成
    {
      stage: 'quiz',
      done: p.quizBestCorrect,
      total: s ? s.quiz : null,
      ratio: p.quizAttempts > 0 ? p.quizBest : 0,
      complete: p.quizAttempts > 0 && (s?.quiz === 0 || p.quizBest >= 0.6),
    },
    mk('guided', s ? s.guided.filter((g) => p.guidedDone.includes(g)).length : p.guidedDone.length, s ? s.guided.length : null),
    mk('challenge', s ? s.challenges.filter((c) => p.challenges[c]?.passed).length : passed, s ? s.challenges.length : null),
    mk('qa', p.qaRated.length, s?.qa ?? null),
  ]
}

/* ------------------------------------------------------------------ */
/* 统计                                                                */
/* ------------------------------------------------------------------ */

export interface Coverage {
  total: number
  cleared: number
}

export interface LevelsStats {
  levels: number
  cleared: number
  stars: number
  maxStars: number
  threeStars: number
  tiers: Partial<Record<Tier, Coverage>>
}

export function statsOf(metas: LevelMeta[], levels: LevelsProgress): LevelsStats {
  const out: LevelsStats = { levels: metas.length, cleared: 0, stars: 0, maxStars: metas.length * 3, threeStars: 0, tiers: {} }
  for (const m of metas) {
    const st = starsOf(levels[m.id])
    out.stars += st
    if (st >= 1) out.cleared++
    if (st === 3) out.threeStars++
    const t = (out.tiers[m.tier] ??= { total: 0, cleared: 0 })
    t.total++
    if (st >= 1) t.cleared++
  }
  return out
}

export const chapterStats = (c: ChapterDef, levels: LevelsProgress) => statsOf(c.levels, levels)

export function trackStats(cat: Catalog, track: Track, levels: LevelsProgress) {
  const chapters = cat.byTrack[track]
  return { chapters: chapters.length, ...statsOf(chapters.flatMap((c) => c.levels), levels) }
}

export function allLevels(cat: Catalog): LevelMeta[] {
  return cat.chapters.flatMap((c) => c.levels)
}

export function challengesPassed(levels: LevelsProgress) {
  let n = 0
  for (const p of Object.values(levels)) if (p) for (const c of Object.values(p.challenges)) if (c.passed) n++
  return n
}

export function totalStars(levels: LevelsProgress) {
  let n = 0
  for (const p of Object.values(levels)) n += starsOf(p)
  return n
}

export function answerAccuracy(daily: Record<string, DailyStat>) {
  let answered = 0
  let correct = 0
  for (const d of Object.values(daily)) {
    answered += d.answered
    correct += d.correct
  }
  return { answered, correct, ratio: answered ? correct / answered : 0 }
}

export function trackOfLevel(cat: Catalog | null, levelId: string | undefined): Track | undefined {
  if (!cat || !levelId) return undefined
  return cat.levelById.get(levelId)?.chapter.track
}

/* ------------------------------------------------------------------ */
/* 复习                                                                */
/* ------------------------------------------------------------------ */

export type ReviewEntry = ReviewItem & { key: string }

export function reviewBuckets(review: Record<string, ReviewItem>, d = today()) {
  const due: ReviewEntry[] = []
  const pending: ReviewEntry[] = []
  const mastered: ReviewEntry[] = []
  for (const [key, it] of Object.entries(review)) {
    const e = { ...it, key }
    if (it.mastered) mastered.push(e)
    else {
      pending.push(e)
      if (it.due <= d) due.push(e)
    }
  }
  const byDue = (a: ReviewEntry, b: ReviewEntry) => (a.due < b.due ? -1 : a.due > b.due ? 1 : b.wrongCount - a.wrongCount)
  due.sort(byDue)
  pending.sort(byDue)
  mastered.sort((a, b) => (a.lastAt < b.lastAt ? 1 : -1))
  return { due, pending, mastered }
}

export function useDueCount() {
  const review = useProgress((s) => s.review)
  return useMemo(() => {
    const d = today()
    let n = 0
    for (const it of Object.values(review)) if (!it.mastered && it.due <= d) n++
    return n
  }, [review])
}

/* ------------------------------------------------------------------ */
/* 推荐关卡                                                            */
/* ------------------------------------------------------------------ */

export interface Recommendation {
  chapter: ChapterDef
  status: LevelStatus
  inProgress: boolean
}

export function recommendLevels(
  cat: Catalog,
  levels: LevelsProgress,
  freeMode: boolean,
  n = 4,
  exclude?: string,
): Recommendation[] {
  const pool: (Recommendation & { score: number })[] = []
  for (const chapter of cat.chapters) {
    const sts = chapterStatuses(chapter, levels, freeMode)
    for (const st of sts) {
      if (!st.unlocked || st.cleared || st.meta.id === exclude) continue
      const p = levels[st.meta.id]
      const inProgress = !!p && (p.sectionsRead.length > 0 || p.quizAttempts > 0 || p.learnDone)
      const fq = st.meta.frequency ? FREQ_PRIORITY[st.meta.frequency] : 1
      const score =
        TIER_PRIORITY[st.meta.tier] * 10 + fq * 3 - (inProgress ? 6 : 0) - (st.current ? 2 : 0) + chapter.order * 0.1 + st.index * 0.01
      pool.push({ chapter, status: st, inProgress, score })
    }
  }
  pool.sort((a, b) => a.score - b.score)
  // 两条路线交替，避免推荐全挤在一条线上
  const byTrack: Record<Track, typeof pool> = { algo: [], llm: [] }
  for (const r of pool) byTrack[r.chapter.track].push(r)
  const out: Recommendation[] = []
  let i = 0
  while (out.length < n && (byTrack.algo.length || byTrack.llm.length)) {
    const t: Track = i++ % 2 === 0 ? 'algo' : 'llm'
    const pick = byTrack[t].shift() ?? byTrack[t === 'algo' ? 'llm' : 'algo'].shift()
    if (pick) out.push({ chapter: pick.chapter, status: pick.status, inProgress: pick.inProgress })
  }
  return out
}

/** 某章「下一步」：当前关 → 第一个已解锁未通关 → 第一关 */
export function chapterNext(chapter: ChapterDef, levels: LevelsProgress, freeMode: boolean) {
  const sts = chapterStatuses(chapter, levels, freeMode)
  return sts.find((s) => s.current) ?? sts.find((s) => s.unlocked && !s.cleared) ?? sts[0]
}

/* ------------------------------------------------------------------ */
/* 热力图                                                              */
/* ------------------------------------------------------------------ */

export interface HeatCell {
  date: string
  xp: number
  /** 0..4；未来日期为 -1 */
  level: number
}

/** 近 weeks 周的活动（列 = 周，从周一开始；最后一列包含今天） */
export function heatmap(daily: Record<string, DailyStat>, goal: number, weeks = 20, end = today()): HeatCell[][] {
  const endDate = new Date(end + 'T00:00:00')
  const dow = (endDate.getDay() + 6) % 7 // 周一 = 0
  const start = addDays(end, -dow - (weeks - 1) * 7)
  const g = Math.max(1, goal)
  const cols: HeatCell[][] = []
  for (let w = 0; w < weeks; w++) {
    const col: HeatCell[] = []
    for (let d = 0; d < 7; d++) {
      const date = addDays(start, w * 7 + d)
      const xp = daily[date]?.xp ?? 0
      const level = date > end ? -1 : xp <= 0 ? 0 : xp < g * 0.3 ? 1 : xp < g * 0.65 ? 2 : xp < g ? 3 : 4
      col.push({ date, xp, level })
    }
    cols.push(col)
  }
  return cols
}

/* ------------------------------------------------------------------ */
/* 其它                                                                */
/* ------------------------------------------------------------------ */

const LAST_TRACK_KEY = 'cq-last-track'
export function rememberTrack(t: Track) {
  try {
    localStorage.setItem(LAST_TRACK_KEY, t)
  } catch {}
}
export function recallTrack(): Track {
  try {
    const v = localStorage.getItem(LAST_TRACK_KEY)
    if (v === 'algo' || v === 'llm') return v
  } catch {}
  return 'algo'
}

/** 早上好 / 下午好 … */
export function greeting(d = new Date()) {
  const h = d.getHours()
  if (h < 5) return '夜深了'
  if (h < 11) return '早上好'
  if (h < 13) return '中午好'
  if (h < 18) return '下午好'
  return '晚上好'
}

/** 关卡路由 */
export const levelHref = (levelId: string, stage?: string) => `/level/${levelId}${stage ? `/${stage}` : ''}`

/* ------------------------------------------------------------------ */
/* 关卡正文查询                                                        */
/* ------------------------------------------------------------------ */

/** 随堂题或节末小测 */
export function findQuestion(content: LevelContent, id: string): Question | undefined {
  return content.quiz.find((q) => q.id === id) ?? content.learn.find((s) => s.checkpoint?.id === id)?.checkpoint
}
export function findChallenge(content: LevelContent, id: string): Challenge | undefined {
  return content.challenges.find((c) => c.id === id)
}
/** 面试追问闪卡（含实战的追问） */
export function findQA(content: LevelContent, id: string): InterviewQA | undefined {
  return content.interview.find((q) => q.id === id) ?? content.challenges.flatMap((c) => c.followUps ?? []).find((q) => q.id === id)
}

/** markdown → 单行纯文本预览 */
export function plainPreview(md: string, max = 80) {
  const s = md
    .replace(/```[\s\S]*?```/g, ' [代码] ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' [公式] ')
    .replace(/\$([^$\n]+)\$/g, '$1')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/[*_>#|]/g, '')
    .replace(/\[!\w+\]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
  return s.length > max ? s.slice(0, max) + '…' : s
}

/** 按需加载多个关卡正文；值为 null 表示加载失败，未出现表示加载中 */
export function useLevelContents(ids: string[]) {
  const key = Array.from(new Set(ids)).sort().join('|')
  const [map, setMap] = useState<Map<string, LevelContent | null>>(new Map())
  useEffect(() => {
    let alive = true
    const list = key ? key.split('|') : []
    list.forEach((id) => {
      loadLevel(id)
        .then((c) => alive && setMap((m) => (m.get(id) === c ? m : new Map(m).set(id, c))))
        .catch(() => alive && setMap((m) => new Map(m).set(id, null)))
    })
    return () => {
      alive = false
    }
  }, [key])
  return map
}
