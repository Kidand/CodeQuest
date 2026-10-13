/**
 * 模拟面试：配置 → 组卷 → 进行中（可刷新恢复）→ 报告。
 * 进行中的会话与历史记录存入独立的 persist store（cq-mock），不污染主进度。
 */
import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { accountKey } from './account-storage'
import { loadLevel, type Catalog } from '@/content/catalog'
import type { Challenge, LevelContent, LevelMeta, Question, Tier, Track } from '@/content/types'
import { shuffle } from './utils'
import { chapterStatuses, type LevelsProgress } from './unlock'

export type MockDuration = 30 | 45 | 60

export interface MockConfig {
  durationMin: MockDuration
  algoChallenges: number
  llmChallenges: number
  concepts: number
  scope: 'unlocked' | 'all'
  /** 为空表示不限 */
  tiers: Tier[]
}

export const DEFAULT_MOCK_CONFIG: MockConfig = {
  durationMin: 45,
  algoChallenges: 1,
  llmChallenges: 1,
  concepts: 6,
  scope: 'unlocked',
  tiers: [],
}

export interface MockItem {
  kind: 'challenge' | 'question'
  levelId: string
  itemId: string
  track: Track
  /** 组卷时记录的标题，便于在内容下线后仍能展示报告 */
  title: string
  difficulty?: Challenge['difficulty']
}

export type MockItemStatus = 'pending' | 'correct' | 'wrong' | 'passed' | 'skipped'

export interface MockItemResult {
  status: MockItemStatus
  /** 停留时长（秒） */
  timeSec: number
}

export interface MockSession {
  id: string
  config: MockConfig
  items: MockItem[]
  results: MockItemResult[]
  startedAt: number
  deadline: number
  active: number
  activeSince: number
}

export interface MockRecord {
  id: string
  startedAt: string
  finishedAt: string
  durationMin: number
  usedSec: number
  score: number
  timeUp: boolean
  xpAwarded: number
  items: (MockItem & MockItemResult)[]
}

/* ------------------------------------------------------------------ */
/* 计分                                                                */
/* ------------------------------------------------------------------ */

/** 实战权重 3、概念题权重 1 */
export const itemWeight = (it: Pick<MockItem, 'kind'>) => (it.kind === 'challenge' ? 3 : 1)
export const isItemGood = (s: MockItemStatus) => s === 'correct' || s === 'passed'
export const isItemDone = (s: MockItemStatus) => s !== 'pending'

export function scoreOf(items: MockItem[], results: MockItemResult[]) {
  let total = 0
  let got = 0
  items.forEach((it, i) => {
    const w = itemWeight(it)
    total += w
    if (isItemGood(results[i]?.status ?? 'pending')) got += w
  })
  return total ? Math.round((100 * got) / total) : 0
}

/* ------------------------------------------------------------------ */
/* Store                                                               */
/* ------------------------------------------------------------------ */

interface MockState {
  lastConfig: MockConfig
  active: MockSession | null
  history: MockRecord[]
  setConfig: (c: MockConfig) => void
  start: (s: MockSession) => void
  /** 切换题目（结算当前题的停留时长） */
  goTo: (index: number) => void
  setResult: (index: number, status: MockItemStatus) => void
  /** 结算并写入历史；返回记录 */
  finish: (opts: { timeUp: boolean; xpAwarded: number }) => MockRecord | null
  discard: () => void
  clearHistory: () => void
}

const uid = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`

export const useMock = create<MockState>()(
  persist(
    (set, get) => ({
      lastConfig: DEFAULT_MOCK_CONFIG,
      active: null,
      history: [],
      setConfig: (c) => set({ lastConfig: c }),
      start: (s) => set({ active: s, lastConfig: s.config }),
      goTo: (index) =>
        set((st) => {
          const s = st.active
          if (!s || index === s.active || index < 0 || index >= s.items.length) return st
          const now = Math.min(Date.now(), s.deadline)
          const results = s.results.slice()
          const cur = results[s.active]
          results[s.active] = { ...cur, timeSec: cur.timeSec + Math.max(0, (now - s.activeSince) / 1000) }
          return { active: { ...s, results, active: index, activeSince: now } }
        }),
      setResult: (index, status) =>
        set((st) => {
          const s = st.active
          if (!s) return st
          const results = s.results.slice()
          // 已通过 / 已作答的不再被覆盖为更差的状态
          if (isItemGood(results[index].status) && !isItemGood(status)) return st
          results[index] = { ...results[index], status }
          return { active: { ...s, results } }
        }),
      finish: ({ timeUp, xpAwarded }) => {
        const s = get().active
        if (!s) return null
        const now = Math.min(Date.now(), s.deadline)
        const results = s.results.slice()
        const cur = results[s.active]
        results[s.active] = { ...cur, timeSec: cur.timeSec + Math.max(0, (now - s.activeSince) / 1000) }
        const end = Math.min(now, s.deadline)
        const rec: MockRecord = {
          id: s.id,
          startedAt: new Date(s.startedAt).toISOString(),
          finishedAt: new Date(end).toISOString(),
          durationMin: s.config.durationMin,
          usedSec: Math.round((end - s.startedAt) / 1000),
          score: scoreOf(s.items, results),
          timeUp,
          xpAwarded,
          items: s.items.map((it, i) => ({ ...it, ...results[i], timeSec: Math.round(results[i].timeSec) })),
        }
        set((st) => ({ active: null, history: [rec, ...st.history].slice(0, 50) }))
        return rec
      },
      discard: () => set({ active: null }),
      clearHistory: () => set({ history: [] }),
    }),
    { name: accountKey('cq-mock'), version: 1, storage: createJSONStorage(() => localStorage) },
  ),
)

/* ------------------------------------------------------------------ */
/* 组卷                                                                */
/* ------------------------------------------------------------------ */

export interface Candidate {
  meta: LevelMeta
  track: Track
}

/** 符合范围与分层筛选的候选关卡 */
export function mockCandidates(cat: Catalog, levels: LevelsProgress, freeMode: boolean, config: Pick<MockConfig, 'scope' | 'tiers'>): Candidate[] {
  const out: Candidate[] = []
  for (const ch of cat.chapters) {
    const sts = chapterStatuses(ch, levels, freeMode)
    for (const st of sts) {
      if (config.scope === 'unlocked' && !st.unlocked) continue
      if (config.tiers.length && !config.tiers.includes(st.meta.tier)) continue
      out.push({ meta: st.meta, track: ch.track })
    }
  }
  return out
}

/** 快速作答的题型优先（模拟面试里的「八股」） */
const QUICK_TYPES = new Set<Question['type']>(['single', 'multi', 'judge', 'fill', 'numeric', 'pick', 'bug'])

export interface BuiltSession {
  items: MockItem[]
  /** 实际数量少于配置时的说明 */
  notes: string[]
}

export async function buildMockSession(cands: Candidate[], config: MockConfig, seed = Date.now()): Promise<BuiltSession> {
  const notes: string[] = []
  const cache = new Map<string, LevelContent | null>()
  const get = async (id: string) => {
    if (cache.has(id)) return cache.get(id) ?? null
    try {
      const c = await loadLevel(id)
      cache.set(id, c)
      return c
    } catch {
      cache.set(id, null)
      return null
    }
  }
  const items: MockItem[] = []
  const usedLevels = new Set<string>()

  const pickChallenges = async (track: Track, need: number) => {
    if (need <= 0) return
    const pool = shuffle(cands.filter((c) => c.track === track), seed + (track === 'algo' ? 1 : 2))
    let got = 0
    // 最多尝试 16 个关卡，避免一次加载过多正文
    for (const c of pool.slice(0, 16)) {
      if (got >= need) break
      const content = await get(c.meta.id)
      if (!content?.challenges.length) continue
      // 偏好 medium，其次随机
      const chs = shuffle(content.challenges, seed + got)
      const ch = chs.find((x) => x.difficulty === 'medium') ?? chs[0]
      items.push({ kind: 'challenge', levelId: c.meta.id, itemId: ch.id, track, title: ch.title, difficulty: ch.difficulty })
      usedLevels.add(c.meta.id)
      got++
    }
    if (got < need) notes.push(`${track === 'algo' ? '算法' : '大模型'}实战只找到 ${got} / ${need} 道符合条件的题`)
  }
  await pickChallenges('algo', config.algoChallenges)
  await pickChallenges('llm', config.llmChallenges)

  if (config.concepts > 0) {
    const pool = shuffle(cands, seed + 3)
    // 优先选择不同关卡，每关 1 题；关卡不够时同一关可出多题
    const picked: MockItem[] = []
    const perLevel = new Map<string, Question[]>()
    for (const c of pool.slice(0, Math.max(config.concepts * 2, 8))) {
      if (picked.length >= config.concepts) break
      const content = await get(c.meta.id)
      if (!content?.quiz.length) continue
      const qs = shuffle(content.quiz, seed + picked.length)
      const sorted = [...qs.filter((q) => QUICK_TYPES.has(q.type)), ...qs.filter((q) => !QUICK_TYPES.has(q.type))]
      perLevel.set(c.meta.id, sorted.slice(1))
      const q = sorted[0]
      picked.push({ kind: 'question', levelId: c.meta.id, itemId: q.id, track: c.track, title: c.meta.title })
    }
    // 补足
    for (const [levelId, rest] of perLevel) {
      for (const q of rest) {
        if (picked.length >= config.concepts) break
        const c = cands.find((x) => x.meta.id === levelId)!
        picked.push({ kind: 'question', levelId, itemId: q.id, track: c.track, title: c.meta.title })
      }
      if (picked.length >= config.concepts) break
    }
    if (picked.length < config.concepts) notes.push(`概念题只找到 ${picked.length} / ${config.concepts} 道`)
    items.push(...picked)
  }
  return { items, notes }
}

export function newSession(config: MockConfig, items: MockItem[]): MockSession {
  const now = Date.now()
  return {
    id: uid(),
    config,
    items,
    results: items.map(() => ({ status: 'pending', timeSec: 0 })),
    startedAt: now,
    deadline: now + config.durationMin * 60_000,
    active: 0,
    activeSince: now,
  }
}
