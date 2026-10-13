/** 我的 /profile：玩家卡、统计、热力图、章节进度、分层覆盖、成就墙、设置、数据导入导出 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router'
import { Check, Download, FlaskConical, Flame, Lock, RotateCcw, Trash2, TriangleAlert, Upload } from 'lucide-react'
import { useCatalog, type Catalog } from '@/content/catalog'
import { TIER_LABEL } from '@/content/types'
import { useDrafts, useEvents, useProgress, type ThemePref } from '@/lib/store'
import { exportBackup, importBackup } from '@/lib/backup'
import { levelFromXp, starsOf, titleForLevel, today, xpToReach } from '@/lib/game'
import { TRACK_META, TRACKS, answerAccuracy, challengesPassed, chapterStats, heatmap, totalStars, trackStats } from '@/lib/selectors'
import { ACHIEVEMENTS, achievementProgress, useAchievementCtx } from '@/lib/achievements'
import { useMock } from '@/lib/mock'
import { switchAccount, useAccount } from '@/lib/accounts'
import { cloudStatusText, useCloud, migrateCloudSnapshot, parseCloudBackup, markCloudDirty } from '@/lib/cloud'
import { cn } from '@/lib/utils'
import { setAccountSwitching } from '@/lib/account-storage'
import { Button, Card, Segmented, Slider, Switch } from '@/components/ui'
import { PageSpinner } from '@/components/PageSpinner'
import { ChapterIcon } from '@/components/ChapterIcon'
import { Bar } from '@/components/game/ProgressRing'
import { ConfirmDialog } from '@/components/game/Modal'
import { tierDotClass } from '@/components/game/TierBadge'
import { TONE_BG } from '@/components/game/GameEventLayer'

export default function Profile() {
  const cat = useCatalog()
  const { hash } = useLocation()
  useEffect(() => {
    if (!cat || !hash) return
    const t = setTimeout(() => document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80)
    return () => clearTimeout(t)
  }, [cat, hash])
  if (!cat) return <PageSpinner />
  return (
    <div className="space-y-6">
      <h1 className="sr-only">我的</h1>
      <PlayerCard />
      <AccountCard />
      <StatsGrid />
      <HeatmapCard />
      <div className="grid gap-4 lg:grid-cols-2">
        <ChapterProgress cat={cat} />
        <TierCoverage cat={cat} />
      </div>
      <AchievementWall />
      <SettingsCard />
      <DataCard />
      {import.meta.env.DEV && <DevCard />}
    </div>
  )
}

function AccountCard() {
  const account = useAccount((s) => s.account)
  const cloud = useCloud()
  return <Card className="flex flex-wrap items-center gap-4 p-5">
    <div className="min-w-0 flex-1">
      <h2 className="font-semibold">{cloud.user ? `你好，${cloud.user.username || cloud.user.email || '学习者'}` : account ? `本机账号：${account.username}` : '正在使用游客存档'}</h2>
      <p className="mt-1 text-xs leading-6 text-muted">{cloud.user ? `${cloudStatusText[cloud.status]} · 同一账号可在不同设备继续学习` : '本机进度已保留 · 登录云端后可迁移并自动同步'}</p>
    </div>
    <Link to="/account" className="rounded-xl bg-accent-soft px-4 py-2.5 text-sm font-semibold text-accent">{cloud.user ? '云端账号与存档' : '登录云端 / 迁移存档'}</Link>
    {!cloud.user && account && <Button variant="ghost" onClick={() => void switchAccount(null)}>退出本机账号</Button>}
  </Card>
}

function SectionTitle({ children, hint, id }: { children: React.ReactNode; hint?: React.ReactNode; id?: string }) {
  return (
    <div id={id} className="mb-3 flex scroll-mt-24 items-baseline justify-between gap-2">
      <h2 className="text-[15px] font-semibold">{children}</h2>
      {hint && <span className="text-xs text-subtle">{hint}</span>}
    </div>
  )
}

function PlayerCard() {
  const xp = useProgress((s) => s.xp)
  const streak = useProgress((s) => s.streak)
  const createdAt = useProgress((s) => s.createdAt)
  const lv = levelFromXp(xp)
  return (
    <Card className="relative overflow-hidden p-5 md:p-7">
      <div className="pointer-events-none absolute -right-20 -top-24 h-64 w-64 rounded-full bg-llm-soft opacity-70" aria-hidden />
      <div className="pointer-events-none absolute -bottom-24 right-24 h-52 w-52 rounded-full bg-algo-soft opacity-60" aria-hidden />
      <div className="relative flex flex-col gap-5 sm:flex-row sm:items-center">
        <div className="grid h-20 w-20 shrink-0 place-items-center rounded-[26px] bg-gradient-to-br from-accent to-llm text-accent-fg shadow-md">
          <span className="text-[11px] font-semibold opacity-80">LV</span>
          <span className="-mt-2 text-3xl font-black tabular-nums">{lv.level}</span>
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold text-subtle">玩家等级</div>
          <div className="mt-0.5 text-2xl font-bold tracking-tight">{titleForLevel(lv.level)}</div>
          <div className="mt-3 flex items-center gap-3">
            <Bar value={lv.progress} color="var(--gold)" className="flex-1" height={10} />
            <span className="shrink-0 text-xs text-muted tabular-nums">
              {lv.intoLevel} / {lv.levelSpan}
            </span>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-muted">
            <span>
              累计 <b className="ink-gold">{xp}</b> XP
            </span>
            <span>距 Lv.{lv.level + 1} 还需 {xpToReach(lv.level + 1) - xp} XP</span>
            <span className="inline-flex items-center gap-1">
              <Flame size={14} className={streak.current ? 'fill-current ink-warning' : ''} /> 连续 {streak.current} 天 · 最长 {streak.best} 天
            </span>
            <span>加入于 {new Date(createdAt).toLocaleDateString('zh-CN')}</span>
          </div>
        </div>
      </div>
    </Card>
  )
}

function StatsGrid() {
  const levels = useProgress((s) => s.levels)
  const daily = useProgress((s) => s.daily)
  const review = useProgress((s) => s.review)
  const mocks = useMock((s) => s.history)
  const cleared = Object.values(levels).filter((p) => starsOf(p) >= 1).length
  const acc = answerAccuracy(daily)
  const mastered = Object.values(review).filter((r) => r.mastered).length
  const items = [
    { label: '通关关卡', value: cleared },
    { label: '获得星数', value: totalStars(levels) },
    { label: '实战通过', value: challengesPassed(levels) },
    { label: '答题正确率', value: acc.answered ? `${Math.round(acc.ratio * 100)}%` : '—', sub: acc.answered ? `${acc.correct}/${acc.answered}` : undefined },
    { label: '复习已掌握', value: mastered },
    { label: '模拟面试', value: mocks.length, sub: mocks.length ? `最高 ${Math.max(...mocks.map((m) => m.score))} 分` : undefined },
  ]
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      {items.map((it) => (
        <Card key={it.label} className="p-4">
          <div className="text-xs text-muted">{it.label}</div>
          <div className="mt-1 text-2xl font-bold tabular-nums">{it.value}</div>
          {it.sub && <div className="text-[11px] text-subtle tabular-nums">{it.sub}</div>}
        </Card>
      ))}
    </div>
  )
}

function HeatmapCard() {
  const daily = useProgress((s) => s.daily)
  const goal = useProgress((s) => s.settings.dailyGoal)
  const cols = useMemo(() => heatmap(daily, goal, 20), [daily, goal])
  const d0 = today()
  const activeDays = cols.flat().filter((c) => c.xp > 0).length
  const sumXp = cols.flat().reduce((n, c) => n + c.xp, 0)
  return (
    <section>
      <SectionTitle hint={`近 20 周 · 活跃 ${activeDays} 天 · ${sumXp} XP`}>学习热力图</SectionTitle>
      <Card className="p-4 md:p-5">
        <div className="grid gap-[3px]" style={{ gridTemplateColumns: `auto repeat(${cols.length}, minmax(0, 1fr))` }}>
          <span />
          {cols.map((c, i) => {
            const m = Number(c[0].date.slice(5, 7))
            const prev = i > 0 ? Number(cols[i - 1][0].date.slice(5, 7)) : -1
            return (
              <span key={i} className="h-4 overflow-visible whitespace-nowrap text-[10px] text-subtle">
                {m !== prev && i < cols.length - 1 ? `${m}月` : ''}
              </span>
            )
          })}
          {[0, 1, 2, 3, 4, 5, 6].map((d) => (
            <HeatRow key={d} d={d} cols={cols} today={d0} />
          ))}
        </div>
        <div className="mt-3 flex items-center justify-end gap-1.5 text-[11px] text-subtle">
          少
          {[0, 1, 2, 3, 4].map((l) => (
            <i key={l} className="heat-cell inline-block h-3 w-3" data-level={l} />
          ))}
          多（≥ 每日目标）
        </div>
      </Card>
    </section>
  )
}

function HeatRow({ d, cols, today: d0 }: { d: number; cols: ReturnType<typeof heatmap>; today: string }) {
  return (
    <>
      <span className="pr-1.5 text-right text-[10px] leading-none text-subtle" style={{ alignSelf: 'center' }}>
        {d % 2 === 0 ? '一二三四五六日'[d] : ''}
      </span>
      {cols.map((c, i) => {
        const cell = c[d]
        return (
          <i
            key={i}
            className="heat-cell aspect-square w-full max-w-[22px] justify-self-center"
            data-level={cell.level}
            data-today={cell.date === d0}
            title={cell.level < 0 ? '' : `${cell.date}：${cell.xp} XP`}
          />
        )
      })}
    </>
  )
}

function ChapterProgress({ cat }: { cat: Catalog }) {
  const levels = useProgress((s) => s.levels)
  return (
    <section>
      <SectionTitle>各章进度</SectionTitle>
      <Card className="divide-y divide-border">
        {cat.chapters.length === 0 && <div className="p-5 text-sm text-muted">章节内容正在制作中。</div>}
        {TRACKS.map((t) =>
          cat.byTrack[t].map((c) => {
            const st = chapterStats(c, levels)
            return (
              <Link key={c.id} to={`/chapter/${c.id}`} className={cn('flex items-center gap-3 px-4 py-3 transition-colors hover:bg-surface-2/60', `track-${t}`)}>
                <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-track-soft text-track">
                  <ChapterIcon name={c.icon} size={16} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="truncate text-sm font-medium">{c.title}</span>
                    <span className="shrink-0 text-xs text-muted tabular-nums">
                      {st.cleared}/{st.levels}
                    </span>
                  </div>
                  <Bar value={st.levels ? st.cleared / st.levels : 0} color="var(--track)" height={5} className="mt-1.5" />
                </div>
              </Link>
            )
          }),
        )}
      </Card>
    </section>
  )
}

function TierCoverage({ cat }: { cat: Catalog }) {
  const levels = useProgress((s) => s.levels)
  return (
    <section>
      <SectionTitle>分层覆盖</SectionTitle>
      <Card className="space-y-5 p-5">
        {TRACKS.map((t) => {
          const st = trackStats(cat, t, levels)
          return (
            <div key={t} className={`track-${t}`}>
              <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                <i className="h-2 w-2 rounded-full bg-track" />
                {TRACK_META[t].title}
                <span className="ml-auto text-xs font-normal text-muted tabular-nums">
                  {st.cleared}/{st.levels}
                </span>
              </div>
              <div className="space-y-2">
                {TRACK_META[t].tiers.map((tier) => {
                  const c = st.tiers[tier]
                  return (
                    <div key={tier} className="flex items-center gap-3 text-[13px]">
                      <span className="flex w-20 shrink-0 items-center gap-1.5 text-muted">
                        <i className={cn('h-1.5 w-1.5 rounded-full', tierDotClass(tier))} />
                        {TIER_LABEL[tier]}
                      </span>
                      <Bar value={c ? c.cleared / c.total : 0} color="var(--track)" height={6} className="flex-1" />
                      <span className="w-12 shrink-0 text-right font-medium tabular-nums">{c ? `${c.cleared}/${c.total}` : '—'}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </Card>
    </section>
  )
}

function AchievementWall() {
  const ctx = useAchievementCtx()
  const unlocked = useProgress((s) => s.achievements)
  const n = ACHIEVEMENTS.filter((a) => unlocked[a.id]).length
  const sorted = [...ACHIEVEMENTS].sort((a, b) => Number(!!unlocked[b.id]) - Number(!!unlocked[a.id]))
  return (
    <section>
      <SectionTitle hint={`${n} / ${ACHIEVEMENTS.length}`}>成就</SectionTitle>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {sorted.map((a) => {
          const at = unlocked[a.id]
          const pr = achievementProgress(a, ctx)
          const Icon = a.icon
          return (
            <Card key={a.id} className={cn('flex flex-col p-4', !at && 'bg-surface-2/40 shadow-none')}>
              <div className="flex items-start justify-between">
                <span className={cn('grid h-10 w-10 place-items-center rounded-xl', at ? TONE_BG[a.tone] : 'bg-surface-3 text-subtle')}>
                  {at ? <Icon size={19} /> : <Lock size={16} />}
                </span>
                {at && <Check size={15} className="ink-success" />}
              </div>
              <div className={cn('mt-3 text-sm font-semibold', !at && 'text-muted')}>{a.title}</div>
              <div className="mt-0.5 flex-1 text-xs leading-relaxed text-muted">{a.desc}</div>
              {at ? (
                <div className="mt-2 text-[11px] text-subtle">{new Date(at).toLocaleDateString('zh-CN')} 解锁</div>
              ) : pr.target > 1 ? (
                <div className="mt-2 flex items-center gap-2">
                  <Bar value={pr.ratio} height={4} className="flex-1" />
                  <span className="text-[11px] text-subtle tabular-nums">
                    {pr.cur}/{pr.target}
                  </span>
                </div>
              ) : null}
            </Card>
          )
        })}
      </div>
    </section>
  )
}

function Row({ title, desc, children }: { title: string; desc?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="min-w-0">
        <div className="text-sm font-medium">{title}</div>
        {desc && <div className="mt-0.5 text-xs leading-relaxed text-muted">{desc}</div>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  )
}

function SettingsCard() {
  const s = useProgress((st) => st.settings)
  const update = useProgress((st) => st.updateSettings)
  return (
    <section>
      <SectionTitle id="settings">设置</SectionTitle>
      <Card className="divide-y divide-border">
        <Row title="主题">
          <Segmented<ThemePref>
            value={s.theme}
            onChange={(v) => update({ theme: v })}
            options={[
              { value: 'light', label: '浅色' },
              { value: 'dark', label: '深色' },
              { value: 'system', label: '跟随系统' },
            ]}
          />
        </Row>
        <Row title="自由模式" desc="解锁全部关卡，考前突击可按需跳转。星级与奖励规则不变。">
          <Switch checked={s.freeMode} onChange={(v) => update({ freeMode: v })} />
        </Row>
        <Row title="每日 XP 目标" desc="影响首页目标环与热力图深浅。">
          <Segmented
            value={s.dailyGoal}
            onChange={(v) => update({ dailyGoal: v })}
            options={[50, 100, 150, 200, 300].map((v) => ({ value: v, label: String(v) }))}
          />
        </Row>
        <Row title="编辑器字号" desc="实战与跟练中代码编辑器的字号。">
          <div className="flex w-56 items-center gap-3">
            <Slider value={s.editorFontSize} min={12} max={20} onChange={(v) => update({ editorFontSize: v })} className="flex-1" />
            <span className="w-10 text-right font-mono text-sm tabular-nums">{s.editorFontSize}px</span>
          </div>
        </Row>
        <Row title="减少动效" desc="关闭彩带、弹跳等动画（系统开启「减少动态效果」时自动生效）。">
          <Switch checked={s.reduceMotion} onChange={(v) => update({ reduceMotion: v })} />
        </Row>
      </Card>
    </section>
  )
}

function DataCard() {
  const cloudUser = useCloud((s) => s.user)
  const resetAll = useProgress((s) => s.resetAll)
  const clearMock = useMock((s) => s.clearHistory)
  const fileRef = useRef<HTMLInputElement>(null)
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const [confirm, setConfirm] = useState<null | 'reset' | { json: string; name: string }>(null)

  const doExport = () => {
    const blob = new Blob([exportBackup()], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `codequest-progress-${today()}.json`
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
    setMsg({ ok: true, text: '已导出进度、代码草稿与模拟面试记录（不包含账号密码）。' })
  }
  const onFile = async (f: File | undefined) => {
    if (!f) return
    try {
      const text = await f.text()
      JSON.parse(text)
      setConfirm({ json: text, name: f.name })
    } catch {
      setMsg({ ok: false, text: '文件不是有效的 JSON。' })
    } finally {
      if (fileRef.current) fileRef.current.value = ''
    }
  }
  return (
    <section>
      <SectionTitle>数据</SectionTitle>
      <Card className="p-5">
        <p className="text-[13px] leading-relaxed text-muted">{cloudUser ? '登录后的进度自动同步到云端，本机保留离线副本。也可以导出完整备份；导入或重置会同步到当前云端账号。' : '当前使用本机存档。到「云端账号与存档」登录并迁移后，可在不同设备继续学习。清理浏览器数据前请先导出备份。'}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Button variant="secondary" icon={<Download size={15} />} onClick={doExport}>
            导出进度
          </Button>
          <Button variant="secondary" icon={<Upload size={15} />} onClick={() => fileRef.current?.click()}>
            导入进度
          </Button>
          <input ref={fileRef} type="file" accept="application/json,.json" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />
          <Button variant="ghost" className="ink-danger hover:bg-danger-soft" icon={<Trash2 size={15} />} onClick={() => setConfirm('reset')}>
            重置全部进度
          </Button>
        </div>
        {msg && (
          <div className={cn('mt-3 flex items-center gap-1.5 text-[13px]', msg.ok ? 'ink-success' : 'ink-danger')} role="status">
            {msg.ok ? <Check size={14} /> : <TriangleAlert size={14} />}
            {msg.text}
          </div>
        )}
      </Card>
      <ConfirmDialog
        open={confirm === 'reset'}
        title="重置全部进度？"
        danger
        confirmLabel="确认重置"
        onCancel={() => setConfirm(null)}
        onConfirm={() => {
          setAccountSwitching(true)
          try {
            resetAll()
            clearMock()
            useMock.getState().discard()
            useDrafts.setState({ drafts: {} })
            useEvents.getState().clear()
          } finally { setAccountSwitching(false); markCloudDirty() }
          setConfirm(null)
          setMsg({ ok: true, text: '进度已重置（设置保留）。' })
        }}
      >
        当前账号的 XP、星级、错题本、成就、代码草稿和模拟面试记录都会被清空，且无法撤销。建议先导出备份。
      </ConfirmDialog>
      <ConfirmDialog
        open={typeof confirm === 'object' && confirm !== null}
        title="导入并覆盖当前进度？"
        confirmLabel="导入"
        onCancel={() => setConfirm(null)}
        onConfirm={async () => {
          if (typeof confirm !== 'object' || !confirm) return
          try {
            if (cloudUser) await migrateCloudSnapshot(parseCloudBackup(confirm.json))
            else importBackup(confirm.json)
            setMsg({ ok: true, text: `已从 ${confirm.name} 导入进度。` })
          } catch (e) {
            setMsg({ ok: false, text: String((e as Error)?.message ?? e) })
          }
          setConfirm(null)
        }}
      >
        将用「{typeof confirm === 'object' && confirm ? confirm.name : ''}」中的进度覆盖当前数据。{cloudUser && '这也会替换当前云端账号的存档，其他设备下一次同步后会更新。'}
      </ConfirmDialog>
    </section>
  )
}

function DevCard() {
  const [busy, setBusy] = useState(false)
  const [ok, setOk] = useState<string | null>(null)
  const [confirm, setConfirm] = useState(false)
  return (
    <section>
      <SectionTitle hint="仅开发环境可见">开发者工具</SectionTitle>
      <Card className="flex flex-wrap items-center gap-3 border-dashed p-5">
        <FlaskConical size={18} className="text-muted" />
        <span className="min-w-0 flex-1 basis-[calc(100%-40px)] text-[13px] text-muted sm:basis-0">生成演示进度（星级、热力图、错题、模拟面试记录），用于检查页面视觉。当前进度会先备份。</span>
        <Button variant="secondary" size="sm" disabled={busy} onClick={() => setConfirm(true)}>
          生成演示进度
        </Button>
        <Button
          variant="ghost"
          size="sm"
          icon={<RotateCcw size={14} />}
          onClick={async () => {
            const m = await import('@/components/game/devSeed')
            if (!m.restoreBackup()) setOk('没有可恢复的备份')
          }}
        >
          恢复备份
        </Button>
        {ok && <span className="w-full text-xs ink-success">{ok}</span>}
      </Card>
      <ConfirmDialog
        open={confirm}
        title="生成演示进度？"
        confirmLabel="生成"
        onCancel={() => setConfirm(false)}
        onConfirm={async () => {
          setConfirm(false)
          setBusy(true)
          const m = await import('@/components/game/devSeed')
          const r = await m.seedDemo()
          setBusy(false)
          setOk(`已生成：${r.levels} 关进度、${r.review} 个复习项、${r.xp} XP`)
        }}
      >
        会覆盖当前进度（首次生成前自动备份到 cq-progress-backup，可用「恢复备份」还原）。
      </ConfirmDialog>
    </section>
  )
}
