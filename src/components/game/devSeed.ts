/**
 * 开发用：生成演示进度（仅 import.meta.env.DEV 下注册，生产构建不会暴露入口）。
 *   window.__cqSeed()     写入演示进度（会先备份当前进度到 cq-progress-backup）
 *   window.__cqRestore()  恢复备份
 *   #/xxx?__seed=1        打开页面时自动生成（截图脚本用）
 */
import { loadCatalog, loadLevel } from '@/content/catalog'
import { addDays, emptyLevelProgress, today, type LevelProgress } from '@/lib/game'
import { reviewKey, useProgress, type DailyStat, type ReviewItem } from '@/lib/store'
import { useMock, type MockRecord } from '@/lib/mock'
import { rng } from '@/lib/utils'

const BACKUP = 'cq-progress-backup'

export async function seedDemo() {
  try {
    const cur = localStorage.getItem('cq-progress')
    if (cur && !localStorage.getItem(BACKUP)) localStorage.setItem(BACKUP, cur)
  } catch {}
  const r = rng(42)
  const cat = await loadCatalog()
  const d0 = today()
  const levels: Record<string, LevelProgress> = {}
  const review: Record<string, ReviewItem> = {}
  let lastVisited: { levelId: string; stage: string } | undefined

  for (const ch of cat.chapters) {
    const n = ch.levels.length
    const clearUpTo = Math.max(1, Math.floor(n * (0.35 + r() * 0.4)))
    for (let i = 0; i < n && i <= clearUpTo; i++) {
      const meta = ch.levels[i]
      let content
      try {
        content = await loadLevel(meta.id)
      } catch {
        continue
      }
      const shape = {
        sections: content.learn.length,
        quiz: content.quiz.length,
        guided: content.guided.map((g) => g.id),
        challenges: content.challenges.map((c) => c.id),
        qa: content.interview.length,
      }
      const p: LevelProgress = { ...emptyLevelProgress(), shape, lastVisitedAt: new Date().toISOString() }
      if (i < clearUpTo) {
        const stars = 1 + Math.floor(r() * 3)
        p.sectionsRead = content.learn.map((s) => s.id)
        p.learnDone = true
        p.quizAttempts = 1 + Math.floor(r() * 2)
        p.quizBest = stars >= 2 ? (r() > 0.5 ? 1 : 0.85) : 0.7
        p.quizBestCorrect = Math.round(p.quizBest * shape.quiz)
        if (stars >= 2) p.guidedDone = [...shape.guided]
        if (stars >= 3)
          for (const c of content.challenges)
            p.challenges[c.id] = { passed: true, usedSolution: false, hintsUsed: 0, attempts: 2, passedAt: new Date().toISOString(), bestTimeSec: 600, difficulty: c.difficulty }
        p.qaRated = content.interview.slice(0, 3).map((q) => q.id)
        p.firstCompletedAt = new Date().toISOString()
        // 错题与闪卡
        content.quiz.slice(0, 2).forEach((q, k) => {
          review[reviewKey(meta.id, q.id)] = {
            kind: 'question', levelId: meta.id, itemId: q.id, source: 'quiz', wrongCount: 1 + k, streak: k, interval: 1 + k,
            due: addDays(d0, -k), lastAt: addDays(d0, -2), mastered: false,
          }
        })
        const q3 = content.quiz[2]
        if (q3)
          review[reviewKey(meta.id, q3.id)] = {
            kind: 'question', levelId: meta.id, itemId: q3.id, source: 'quiz', wrongCount: 1, streak: 3, interval: 11,
            due: addDays(d0, 9), lastAt: addDays(d0, -1), mastered: true,
          }
        const qa = content.interview[0]
        if (qa)
          review[reviewKey(meta.id, qa.id, 'qa')] = {
            kind: 'qa', levelId: meta.id, itemId: qa.id, source: 'qa', wrongCount: 1, streak: 0, interval: 1,
            due: d0, lastAt: addDays(d0, -1), mastered: false,
          }
      } else {
        p.sectionsRead = content.learn.slice(0, Math.ceil(content.learn.length / 2)).map((s) => s.id)
        lastVisited ??= { levelId: meta.id, stage: 'learn' }
      }
      levels[meta.id] = p
    }
  }

  const daily: Record<string, DailyStat> = {}
  let xp = 0
  for (let k = 0; k < 140; k++) {
    if (r() < 0.38 && k > 6) continue
    const v = Math.round(r() * r() * 220) + (k < 6 ? 60 : 5)
    const answered = Math.round(v / 8)
    daily[addDays(d0, -k)] = { xp: v, answered, correct: Math.round(answered * (0.6 + r() * 0.35)), solved: r() > 0.7 ? 1 : 0 }
    xp += v
  }

  useProgress.setState({
    xp,
    levels,
    review,
    daily,
    streak: { current: 6, best: 12, lastDate: d0 },
    lastVisited,
  })

  const items = Object.values(review).slice(0, 4)
  const rec = (k: number, score: number): MockRecord => ({
    id: `demo-${k}`,
    startedAt: new Date(Date.now() - k * 86400_000 * 3).toISOString(),
    finishedAt: new Date(Date.now() - k * 86400_000 * 3 + 40 * 60_000).toISOString(),
    durationMin: 45,
    usedSec: 40 * 60,
    score,
    timeUp: false,
    xpAwarded: 50,
    items: items.map((it, i) => ({
      kind: 'question' as const, levelId: it.levelId, itemId: it.itemId, track: 'algo' as const,
      title: cat.levelById.get(it.levelId)?.meta.title ?? it.levelId, status: i % 2 ? 'wrong' as const : 'correct' as const, timeSec: 90 + i * 30,
    })),
  })
  useMock.setState({ history: [rec(1, 72), rec(2, 58)] })
  return { levels: Object.keys(levels).length, review: Object.keys(review).length, xp }
}

export function restoreBackup() {
  try {
    const b = localStorage.getItem(BACKUP)
    if (!b) return false
    localStorage.setItem('cq-progress', b)
    localStorage.removeItem(BACKUP)
    location.reload()
    return true
  } catch {
    return false
  }
}

export function registerDevSeed() {
  const w = window as unknown as Record<string, unknown>
  w.__cqSeed = seedDemo
  w.__cqRestore = restoreBackup
  w.__cqStore = useProgress
  // 截图自动化：URL 带 ?__seed=1（hash 内查询串）时自动生成演示进度
  if (/[?&]__seed=1/.test(location.hash)) void seedDemo()
}
