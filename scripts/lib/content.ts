/**
 * Node 侧加载内容（tsx 运行，支持 @/ 别名与 .tsx）。
 * 用法：脚本参数里的章节 id（空格或逗号分隔）作为过滤；--level <id> 只看单关。
 */
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import type { ChapterDef, LevelContent, LevelMeta } from '../../src/content/types'

export const CHAPTER_DIR = path.resolve(import.meta.dirname, '../../src/content/chapters')

export function parseArgs(argv = process.argv.slice(2)) {
  const chapters: string[] = []
  let level: string | undefined
  const flags = new Set<string>()
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--level') level = argv[++i]
    else if (a === '--chapter') chapters.push(...argv[++i].split(','))
    else if (a.startsWith('--')) flags.add(a.slice(2))
    else chapters.push(...a.split(',').filter(Boolean))
  }
  return { chapters, level, flags }
}

export interface LoadedChapter {
  dirId: string
  chapter?: ChapterDef
  error?: string
}

export async function loadChapters(filter: string[] = []): Promise<LoadedChapter[]> {
  const ids = fs
    .readdirSync(CHAPTER_DIR)
    .filter((d) => fs.existsSync(path.join(CHAPTER_DIR, d, 'index.ts')))
    .sort()
  const out: LoadedChapter[] = []
  for (const id of ids) {
    if (filter.length && !filter.includes(id)) continue
    try {
      const mod = await import(pathToFileURL(path.join(CHAPTER_DIR, id, 'index.ts')).href)
      out.push({ dirId: id, chapter: mod.default as ChapterDef })
    } catch (e) {
      out.push({ dirId: id, error: String((e as Error)?.stack ?? e) })
    }
  }
  return out
}

export async function loadLevelContent(meta: LevelMeta): Promise<LevelContent> {
  const mod = await meta.load()
  return mod.default
}

export const c = {
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
}
