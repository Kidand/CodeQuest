/**
 * 关卡解锁规则（DESIGN §3）：
 *  - 各章第一关默认开放（面试准备要能按需跳转）；
 *  - 章内顺序解锁：上一关 starsOf ≥ 1 才解锁下一关；
 *  - BOSS 关：本章其余关卡全部通关后解锁；
 *  - settings.freeMode：解锁全部；
 *  - 章节 prerequisites 只作「建议先学」提示，不参与锁定。
 */
import type { Catalog } from '@/content/catalog'
import type { ChapterDef, LevelMeta } from '@/content/types'
import { starsOf, type LevelProgress } from './game'

export type LevelsProgress = Record<string, LevelProgress | undefined>

export type NodeState = 'locked' | 'available' | 'current' | 'cleared'

export interface LevelStatus {
  meta: LevelMeta
  index: number
  unlocked: boolean
  cleared: boolean
  stars: 0 | 1 | 2 | 3
  /** 本章中「下一个该打的关」：第一个已解锁且未通关的关卡 */
  current: boolean
  state: NodeState
  /** 锁定原因（未锁定时为空） */
  lockReason?: string
  isBoss: boolean
}

export const isBoss = (m: LevelMeta) => m.kind === 'boss'

/** 单关是否解锁 */
export function isLevelUnlocked(chapter: ChapterDef, index: number, levels: LevelsProgress, freeMode: boolean): boolean {
  if (freeMode) return true
  const meta = chapter.levels[index]
  if (!meta) return false
  if (isBoss(meta)) {
    const others = chapter.levels.filter((l) => l.id !== meta.id)
    // 只有 BOSS 一关的章节：直接开放
    return others.every((l) => starsOf(levels[l.id]) >= 1)
  }
  if (index === 0) return true
  const prev = chapter.levels[index - 1]
  return starsOf(levels[prev.id]) >= 1
}

/** 人类可读的锁定原因 */
export function lockReasonOf(chapter: ChapterDef, index: number, levels: LevelsProgress): string {
  const meta = chapter.levels[index]
  if (isBoss(meta)) {
    const left = chapter.levels.filter((l) => l.id !== meta.id && starsOf(levels[l.id]) < 1).length
    return `通关本章其余关卡后解锁 BOSS（还差 ${left} 关）`
  }
  const prev = chapter.levels[index - 1]
  return prev ? `通关上一关「${prev.title}」后解锁` : '暂未解锁'
}

/** 计算整章每一关的状态 */
export function chapterStatuses(chapter: ChapterDef, levels: LevelsProgress, freeMode: boolean): LevelStatus[] {
  let currentAssigned = false
  return chapter.levels.map((meta, index) => {
    const stars = starsOf(levels[meta.id])
    const cleared = stars >= 1
    const unlocked = isLevelUnlocked(chapter, index, levels, freeMode)
    let current = false
    if (!currentAssigned && unlocked && !cleared) {
      // 自由模式下全部解锁：「当前」仍按顺序取第一个未通关的
      current = true
      currentAssigned = true
    }
    const state: NodeState = cleared ? 'cleared' : !unlocked ? 'locked' : current ? 'current' : 'available'
    return {
      meta,
      index,
      unlocked,
      cleared,
      stars,
      current,
      state,
      lockReason: unlocked ? undefined : lockReasonOf(chapter, index, levels),
      isBoss: isBoss(meta),
    }
  })
}

/** 按 levelId 查询状态（未知关卡返回 null） */
export function levelStatusById(cat: Catalog, levelId: string, levels: LevelsProgress, freeMode: boolean): LevelStatus | null {
  const ref = cat.levelById.get(levelId)
  if (!ref) return null
  return chapterStatuses(ref.chapter, levels, freeMode)[ref.index] ?? null
}

/** 章节的「建议先学」：prerequisites 中尚未全部通关的章节 */
export function pendingPrerequisites(cat: Catalog, chapter: ChapterDef, levels: LevelsProgress): ChapterDef[] {
  return (chapter.prerequisites ?? [])
    .map((id) => cat.chapterById.get(id))
    .filter((c): c is ChapterDef => !!c)
    .filter((c) => c.levels.some((l) => starsOf(levels[l.id]) < 1))
}
