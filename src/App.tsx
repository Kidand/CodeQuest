/**
 * 路由表（HashRouter，便于静态托管）。
 * 页面归属：
 *   平台-壳层（P1）：AppLayout、Home、MapPage、ChapterPage、Review、Mock、Profile、Cheatsheet、NotFound
 *   平台-关卡（P2）：LevelPage（沉浸式，不套 AppLayout）
 *   开发工具：DevIndex、DevLevel（内容作者自检用）
 */
import { lazy, Suspense, useEffect } from 'react'
import { createHashRouter, Link, Navigate, Outlet, RouterProvider, ScrollRestoration } from 'react-router'
import { MotionConfig } from 'motion/react'
import { useThemeSync } from '@/lib/theme'
import { useProgress } from '@/lib/store'
import { useAccount } from '@/lib/accounts'
import { initializeCloud, useCloud, cloudStatusText } from '@/lib/cloud'
import { useAchievementWatcher } from '@/lib/achievements'
import AppLayout from '@/components/layout/AppLayout'
import { CommandPalette } from '@/components/layout/CommandPalette'
import { GameEventLayer } from '@/components/game/GameEventLayer'
import { PageSpinner } from '@/components/PageSpinner'

// 开发环境：注册 window.__cqSeed() / __cqRestore()（生成 / 恢复演示进度），生产构建会被摇树移除
if (import.meta.env.DEV) import('@/components/game/devSeed').then((m) => m.registerDevSeed())

const Home = lazy(() => import('@/pages/Home'))
const MapPage = lazy(() => import('@/pages/MapPage'))
const ChapterPage = lazy(() => import('@/pages/ChapterPage'))
const LevelPage = lazy(() => import('@/pages/LevelPage'))
const Review = lazy(() => import('@/pages/Review'))
const Mock = lazy(() => import('@/pages/Mock'))
const Profile = lazy(() => import('@/pages/Profile'))
const Account = lazy(() => import('@/pages/Account'))
const Cheatsheet = lazy(() => import('@/pages/Cheatsheet'))
const NotFound = lazy(() => import('@/pages/NotFound'))
const DevIndex = lazy(() => import('@/pages/dev/DevIndex'))
const DevLevel = lazy(() => import('@/pages/dev/DevLevel'))

function Root() {
  const ready = useCloud((s) => s.ready)
  useEffect(() => { void initializeCloud() }, [])
  const revision = useAccount((s) => s.revision)
  const switching = useAccount((s) => s.switching)
  if (!ready) return <PageSpinner label="正在连接学习存档…" />
  if (switching) return <PageSpinner label="正在切换学习存档…" />
  return <RootSession key={revision} />
}

function RootSession() {
  useThemeSync()
  useAchievementWatcher()
  const reduceMotion = useProgress((s) => s.settings.reduceMotion)
  return (
    <MotionConfig reducedMotion={reduceMotion ? 'always' : 'user'}>
      <Suspense fallback={<PageSpinner />}>
        <Outlet />
      </Suspense>
      {/* 全局：游戏事件浮层（XP / 升级 / 成就 / 星级）与 ⌘K 命令面板，关卡页同样可用 */}
      <GameEventLayer />
      <CommandPalette />
      <CloudNotice />
      <ScrollRestoration />
    </MotionConfig>
  )
}

function CloudNotice() {
  const status = useCloud((s) => s.status)
  const user = useCloud((s) => s.user)
  if (!user || !['offline', 'conflict', 'expired', 'error'].includes(status)) return null
  return <Link to="/account" role="status" className="fixed bottom-20 right-3 z-40 max-w-[calc(100vw-24px)] rounded-xl border border-border bg-surface px-4 py-2.5 text-xs font-medium shadow-lg md:bottom-4">{cloudStatusText[status]} · 查看存档</Link>
}

const router = createHashRouter([
  {
    element: <Root />,
    children: [
      {
        element: <AppLayout />,
        children: [
          { index: true, element: <Home /> },
          { path: 'track', element: <Navigate to="/track/algo" replace /> },
          { path: 'track/:track', element: <MapPage /> },
          { path: 'chapter/:chapterId', element: <ChapterPage /> },
          { path: 'review', element: <Review /> },
          { path: 'mock', element: <Mock /> },
          { path: 'profile', element: <Profile /> },
          { path: 'account', element: <Account /> },
          { path: 'cheatsheet', element: <Cheatsheet /> },
          { path: 'dev', element: <DevIndex /> },
          { path: '*', element: <NotFound /> },
        ],
      },
      // 关卡页沉浸式全屏，不套 AppLayout
      { path: 'level/:levelId/:stage?', element: <LevelPage /> },
      { path: 'dev/level/:levelId', element: <DevLevel /> },
    ],
  },
])

export default function App() {
  return <RouterProvider router={router} />
}
