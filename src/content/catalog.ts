/**
 * 内容目录：自动发现 src/content/chapters/<id>/index.ts（无需手动注册）。
 * 单个章节加载失败不会拖垮全站（记录到 errors）。
 */
import { useEffect, useState } from 'react'
import type { ChapterDef, LevelContent, LevelMeta, Track } from './types'

const chapterModules = import.meta.glob<{ default: ChapterDef }>('./chapters/*/index.ts')

export interface LevelRef {
  chapter: ChapterDef
  meta: LevelMeta
  /** 在章节内的下标 */
  index: number
  /** 同章节下一关 / 上一关 */
  next?: LevelMeta
  prev?: LevelMeta
}

export interface Catalog {
  chapters: ChapterDef[]
  byTrack: Record<Track, ChapterDef[]>
  chapterById: Map<string, ChapterDef>
  levelById: Map<string, LevelRef>
  /** 加载失败的章节模块 */
  errors: { path: string; error: string }[]
}

let catalogPromise: Promise<Catalog> | null = null
let catalogValue: Catalog | null = null

export function loadCatalog(): Promise<Catalog> {
  if (catalogPromise) return catalogPromise
  catalogPromise = (async () => {
    const entries = Object.entries(chapterModules)
    const settled = await Promise.allSettled(entries.map(([, load]) => load()))
    const chapters: ChapterDef[] = []
    const errors: Catalog['errors'] = []
    settled.forEach((r, i) => {
      if (r.status === 'fulfilled' && r.value?.default) chapters.push(r.value.default)
      else {
        const error = r.status === 'rejected' ? String(r.reason?.message ?? r.reason) : 'missing default export'
        errors.push({ path: entries[i][0], error })
        console.error('[catalog] 章节加载失败', entries[i][0], error)
      }
    })
    chapters.sort((a, b) => (a.track === b.track ? a.order - b.order : a.track === 'algo' ? -1 : 1))
    const byTrack: Catalog['byTrack'] = { algo: [], llm: [] }
    const chapterById = new Map<string, ChapterDef>()
    const levelById = new Map<string, LevelRef>()
    for (const c of chapters) {
      byTrack[c.track].push(c)
      chapterById.set(c.id, c)
      c.levels.forEach((meta, index) => {
        levelById.set(meta.id, { chapter: c, meta, index, next: c.levels[index + 1], prev: c.levels[index - 1] })
      })
    }
    catalogValue = { chapters, byTrack, chapterById, levelById, errors }
    return catalogValue
  })()
  return catalogPromise
}

/** 读取目录；首次加载期间返回 null */
export function useCatalog(): Catalog | null {
  const [cat, setCat] = useState<Catalog | null>(catalogValue)
  useEffect(() => {
    if (!cat) loadCatalog().then(setCat)
  }, [cat])
  return cat
}

const levelCache = new Map<string, Promise<LevelContent>>()

export async function loadLevel(levelId: string): Promise<LevelContent> {
  const cached = levelCache.get(levelId)
  if (cached) return cached
  const p = (async () => {
    const cat = await loadCatalog()
    const ref = cat.levelById.get(levelId)
    if (!ref) throw new Error(`关卡不存在：${levelId}`)
    const mod = await ref.meta.load()
    return mod.default
  })()
  levelCache.set(levelId, p)
  p.catch(() => levelCache.delete(levelId))
  return p
}

export function useLevelContent(levelId: string | undefined) {
  const [state, setState] = useState<{ content?: LevelContent; error?: string; loading: boolean }>({ loading: true })
  useEffect(() => {
    if (!levelId) return
    let alive = true
    setState({ loading: true })
    loadLevel(levelId)
      .then((content) => alive && setState({ content, loading: false }))
      .catch((e) => alive && setState({ error: String(e?.message ?? e), loading: false }))
    return () => {
      alive = false
    }
  }, [levelId])
  return state
}
