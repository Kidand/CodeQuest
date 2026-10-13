/**
 * 【P2 平台-关卡 负责】关卡页（沉浸式，不套 AppLayout）：图解 → 随堂练习 → 跟练 → 实战 → 面试追问。
 *   /level/:levelId/:stage?   stage 缺省时进入推荐阶段（第一个未完成的阶段）
 */
import { lazy, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'
import { MotionConfig } from 'motion/react'
import { NotebookText } from 'lucide-react'
import { useCatalog, useLevelContent } from '@/content/catalog'
import type { Stage } from '@/content/types'
import { useProgress } from '@/lib/store'
import { starsOf } from '@/lib/game'
import { Markdown } from '@/components/Markdown'
import { PageSpinner } from '@/components/PageSpinner'
import { LevelContext, type LevelCtx } from '@/components/level/context'
import { LevelTopBar } from '@/components/level/LevelTopBar'
import { Drawer, ErrorBoundary } from '@/components/level/Overlay'
import { LevelSummary, useLevelWatch } from '@/components/level/LevelSummary'
import { isStage, nextStage, recommendedStage, STAGE_META } from '@/components/level/status'
import { LearnStage } from '@/components/level/LearnStage'
import { QuizStage } from '@/components/level/QuizStage'
import { GuidedStage } from '@/components/level/GuidedStage'
import { QAStage } from '@/components/level/QAStage'
import '@/styles/level.css'

// 实战阶段（含 CodeMirror）单独分包
const ChallengeStage = lazy(() => import('@/components/level/ChallengeStage').then((m) => ({ default: m.ChallengeStage })))

export default function LevelPage() {
  const { levelId = '', stage: stageParam } = useParams()
  const navigate = useNavigate()
  const cat = useCatalog()
  const { content: L, error, loading } = useLevelContent(levelId)
  const levelRef = cat?.levelById.get(levelId)
  const p = useProgress((s) => s.levels[levelId])
  const freeMode = useProgress((s) => s.settings.freeMode)
  const reduceMotion = useProgress((s) => s.settings.reduceMotion)
  const registerLevelShape = useProgress((s) => s.registerLevelShape)
  const visitLevel = useProgress((s) => s.visitLevel)

  const [cheatOpen, setCheatOpen] = useState(false)
  const [summary, setSummary] = useState<{ open: boolean; starsBefore: number }>({ open: false, starsBefore: 0 })
  const headerRef = useRef<HTMLElement>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  // 登记关卡结构（用于星级计算）
  useEffect(() => {
    if (!L) return
    registerLevelShape(levelId, {
      sections: L.learn.length,
      quiz: L.quiz.length,
      guided: L.guided.map((g) => g.id),
      challenges: L.challenges.map((c) => c.id),
      qa: L.interview.length,
    })
  }, [L, levelId, registerLevelShape])

  // 阶段：URL 优先，缺省为推荐阶段
  const stage: Stage | null = useMemo(() => {
    if (!L) return null
    if (isStage(stageParam)) return stageParam
    return recommendedStage(L, useProgress.getState().levels[levelId])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [L, stageParam, levelId])

  useEffect(() => {
    if (L && stage && stageParam !== stage) navigate(`/level/${levelId}/${stage}`, { replace: true })
  }, [L, stage, stageParam, levelId, navigate])

  useEffect(() => {
    if (stage) visitLevel(levelId, stage)
  }, [stage, levelId, visitLevel])

  useEffect(() => {
    if (levelRef) document.title = `${levelRef.meta.title} · CodeQuest`
  }, [levelRef])

  // 顶栏高度 → CSS 变量（供各阶段 sticky / 满高布局使用）
  useLayoutEffect(() => {
    const el = headerRef.current
    const root = rootRef.current
    if (!el || !root) return
    const apply = () => root.style.setProperty('--lv-top', `${el.offsetHeight}px`)
    apply()
    const ro = new ResizeObserver(apply)
    ro.observe(el)
    return () => ro.disconnect()
  })

  const goStage = useCallback(
    (s: Stage) => {
      navigate(`/level/${levelId}/${s}`, { replace: true })
      window.scrollTo({ top: 0 })
    },
    [navigate, levelId],
  )

  const starsNow = starsOf(p)
  const openSummary = useCallback(() => setSummary({ open: true, starsBefore: starsOf(useProgress.getState().levels[levelId]) }), [levelId])

  // 星级提升 → 稍后弹出结算（先让用户看到当前阶段的即时反馈；彩带由全局 GameEventLayer 负责）
  const starsTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const { sessionXp } = useLevelWatch(levelId, (before) => {
    if (starsTimer.current) clearTimeout(starsTimer.current)
    starsTimer.current = setTimeout(() => setSummary({ open: true, starsBefore: before }), 1800)
  })
  useEffect(() => () => void (starsTimer.current && clearTimeout(starsTimer.current)), [])

  // 下一关：同章下一关，否则同路线下一章第一关
  const nextLevel = useMemo(() => {
    if (!cat || !levelRef) return null
    let meta = levelRef.next
    if (!meta) {
      const list = cat.byTrack[levelRef.chapter.track]
      const ci = list.findIndex((c) => c.id === levelRef.chapter.id)
      meta = list[ci + 1]?.levels[0]
      if (meta) return { id: meta.id, title: meta.title, locked: false }
      return null
    }
    return { id: meta.id, title: meta.title, locked: !freeMode && starsNow < 1 }
  }, [cat, levelRef, freeMode, starsNow])

  const ctx: LevelCtx | null = useMemo(
    () =>
      L && levelRef && stage
        ? { levelId, L, ref: levelRef, stage, goStage, next: nextStage(stage, L), openSummary }
        : null,
    [L, levelRef, stage, levelId, goStage, openSummary],
  )

  if (loading || !cat) return <PageSpinner label="加载关卡…" />
  if (error || !L || !levelRef || !ctx || !stage) {
    return (
      <div className="mx-auto max-w-md px-4 py-24 text-center">
        <h1 className="text-lg font-semibold text-fg">关卡加载失败</h1>
        <p className="mt-2 break-all text-sm text-muted">{error ?? (levelRef ? '内容为空' : `找不到关卡：${levelId}`)}</p>
        <div className="mt-6 flex justify-center gap-3 text-sm">
          <Link to="/" className="rounded-xl bg-accent px-4 py-2 font-medium text-accent-fg">
            回到首页
          </Link>
          {levelRef && (
            <Link to={`/chapter/${levelRef.chapter.id}`} className="rounded-xl border border-border px-4 py-2 text-fg">
              返回章节
            </Link>
          )}
        </div>
      </div>
    )
  }

  return (
    <MotionConfig reducedMotion={reduceMotion ? 'always' : 'user'}>
      <LevelContext.Provider value={ctx}>
        <div ref={rootRef} className="min-h-dvh bg-bg" data-level={levelId} data-current-stage={stage}>
          <LevelTopBar
            ref={headerRef}
            levelRef={levelRef}
            L={L}
            p={p}
            stage={stage}
            onGo={goStage}
            onCheatsheet={() => setCheatOpen(true)}
          />
          <main key={stage}>
            <ErrorBoundary name={STAGE_META[stage].label}>
              {stage === 'learn' && <LearnStage />}
              {stage === 'quiz' && <QuizStage />}
              {stage === 'guided' && <GuidedStage />}
              {stage === 'challenge' && (
                <Suspense fallback={<PageSpinner label="加载实战工作台…" />}>
                  <ChallengeStage />
                </Suspense>
              )}
              {stage === 'qa' && <QAStage />}
            </ErrorBoundary>
          </main>

          <Drawer
            open={cheatOpen}
            onClose={() => setCheatOpen(false)}
            title={`速记卡 · ${levelRef.meta.title}`}
            icon={<NotebookText size={18} className="text-accent" />}
          >
            {L.cheatsheet ? <Markdown>{L.cheatsheet}</Markdown> : <p className="text-sm text-muted">本关暂无速记卡。</p>}
          </Drawer>

          <LevelSummary
            open={summary.open}
            onClose={() => setSummary((s) => ({ ...s, open: false }))}
            levelId={levelId}
            levelRef={levelRef}
            L={L}
            sessionXp={sessionXp}
            starsBefore={summary.starsBefore}
            onCheatsheet={() => {
              setSummary((s) => ({ ...s, open: false }))
              setCheatOpen(true)
            }}
            nextLevel={nextLevel}
          />
        </div>
      </LevelContext.Provider>
    </MotionConfig>
  )
}
