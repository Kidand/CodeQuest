import { createContext, useContext } from 'react'
import type { LevelRef } from '@/content/catalog'
import type { LevelContent, Stage } from '@/content/types'

export interface LevelCtx {
  levelId: string
  L: LevelContent
  ref: LevelRef
  stage: Stage
  goStage: (s: Stage) => void
  /** 下一个有内容的阶段（没有则为 null） */
  next: Stage | null
  /** 打开通关结算 */
  openSummary: () => void
}

export const LevelContext = createContext<LevelCtx | null>(null)
export function useLevel() {
  const v = useContext(LevelContext)
  if (!v) throw new Error('useLevel 必须在 LevelPage 内使用')
  return v
}
