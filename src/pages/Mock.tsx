/**
 * 模拟面试 /mock：配置 → 进行中（倒计时、题目导航，可刷新恢复）→ 报告（得分、用时、薄弱点、建议复习）。
 * 实战使用 ChallengeWorkspace mode="interview"；概念题使用 QuestionView。
 */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router'
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  Clock,
  Code,
  Flag,
  History,
  ListChecks,
  Loader2,
  Minus,
  Plus,
  RotateCcw,
  SkipForward,
  Timer,
  TriangleAlert,
  Trophy,
  X,
} from 'lucide-react'
import { useCatalog, type Catalog } from '@/content/catalog'
import { TIER_LABEL, type Tier } from '@/content/types'
import { useProgress } from '@/lib/store'
import { XP } from '@/lib/game'
import {
  DEFAULT_MOCK_CONFIG,
  buildMockSession,
  isItemDone,
  isItemGood,
  mockCandidates,
  newSession,
  useMock,
  type MockConfig,
  type MockItemStatus,
  type MockRecord,
  type MockSession,
} from '@/lib/mock'
import { TRACK_META, findChallenge, findQuestion, levelHref, useLevelContents } from '@/lib/selectors'
import { cn, fmtDuration } from '@/lib/utils'
import { Button, Card, Segmented } from '@/components/ui'
import { PageSpinner } from '@/components/PageSpinner'
import { QuestionView } from '@/components/quiz/QuestionView'
import { ChallengeWorkspace } from '@/components/code/ChallengeWorkspace'
import { ConfirmDialog } from '@/components/game/Modal'
import { ProgressRing } from '@/components/game/ProgressRing'
import { TierBadge, TrackBadge } from '@/components/game/TierBadge'
import { EmptyState } from '@/components/game/EmptyState'

export default function Mock() {
  const cat = useCatalog()
  const active = useMock((s) => s.active)
  const [report, setReport] = useState<MockRecord | null>(null)
  if (!cat) return <PageSpinner />
  if (active) return <Runner cat={cat} session={active} onFinished={setReport} />
  if (report) return <Report cat={cat} rec={report} onBack={() => setReport(null)} />
  return <Setup cat={cat} onOpen={setReport} />
}

/* ------------------------------------------------------------------ */
/* 配置                                                                */
/* ------------------------------------------------------------------ */

function Stepper({ label, value, min, max, onChange, hint }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void; hint?: string }) {
  return (
    <div className="flex items-center justify-between gap-3 py-2.5">
      <div className="min-w-0">
        <div className="text-sm font-medium">{label}</div>
        {hint && <div className="text-xs text-subtle">{hint}</div>}
      </div>
      <div className="flex items-center gap-1">
        <Button variant="secondary" size="icon-sm" aria-label={`减少${label}`} disabled={value <= min} onClick={() => onChange(value - 1)}>
          <Minus size={15} />
        </Button>
        <span className="w-8 text-center text-[15px] font-semibold tabular-nums">{value}</span>
        <Button variant="secondary" size="icon-sm" aria-label={`增加${label}`} disabled={value >= max} onClick={() => onChange(value + 1)}>
          <Plus size={15} />
        </Button>
      </div>
    </div>
  )
}

function Setup({ cat, onOpen }: { cat: Catalog; onOpen: (r: MockRecord) => void }) {
  const levels = useProgress((s) => s.levels)
  const freeMode = useProgress((s) => s.settings.freeMode)
  const lastConfig = useMock((s) => s.lastConfig)
  const history = useMock((s) => s.history)
  const start = useMock((s) => s.start)
  const setConfig = useMock((s) => s.setConfig)
  const [cfg, setCfg] = useState<MockConfig>({ ...DEFAULT_MOCK_CONFIG, ...lastConfig })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string[] | null>(null)
  const patch = (p: Partial<MockConfig>) => {
    setError(null)
    setCfg((c) => ({ ...c, ...p }))
  }
  const cands = useMemo(() => mockCandidates(cat, levels, freeMode, cfg), [cat, levels, freeMode, cfg])
  const nAlgo = cands.filter((c) => c.track === 'algo').length
  const nLlm = cands.length - nAlgo
  const total = cfg.algoChallenges + cfg.llmChallenges + cfg.concepts

  const go = async () => {
    setBusy(true)
    setError(null)
    try {
      const built = await buildMockSession(cands, cfg)
      if (!built.items.length) {
        setError(['没有找到符合条件的题目。试试把范围改为「全部关卡」，或放宽分层筛选。', ...built.notes])
        return
      }
      setConfig(cfg)
      start(newSession(cfg, built.items))
    } finally {
      setBusy(false)
    }
  }

  const toggleTier = (t: Tier) => patch({ tiers: cfg.tiers.includes(t) ? cfg.tiers.filter((x) => x !== t) : [...cfg.tiers, t] })

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight md:text-[28px]">模拟面试</h1>
        <p className="mt-1 text-sm text-muted">限时完成「手撕 + 八股」，结束后给出得分、各题用时和建议复习的关卡。实战题隐藏提示与题解。</p>
      </header>

      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <Card className="divide-y divide-border">
          <div className="p-5">
            <div className="mb-2 text-sm font-semibold">时长</div>
            <Segmented
              value={cfg.durationMin}
              onChange={(v) => patch({ durationMin: v })}
              options={[30, 45, 60].map((v) => ({ value: v as MockConfig['durationMin'], label: `${v} 分钟` }))}
            />
          </div>
          <div className="px-5 py-3">
            <div className="pt-2 text-sm font-semibold">题目组成</div>
            <Stepper label="算法实战" hint="经典算法手撕" value={cfg.algoChallenges} min={0} max={3} onChange={(v) => patch({ algoChallenges: v })} />
            <Stepper label="大模型实战" hint="NumPy 手撕模块" value={cfg.llmChallenges} min={0} max={3} onChange={(v) => patch({ llmChallenges: v })} />
            <Stepper label="概念题" hint="八股 / 计算 / 找 Bug" value={cfg.concepts} min={0} max={12} onChange={(v) => patch({ concepts: v })} />
          </div>
          <div className="p-5">
            <div className="mb-2 text-sm font-semibold">出题范围</div>
            <Segmented
              value={cfg.scope}
              onChange={(v) => patch({ scope: v })}
              options={[
                { value: 'unlocked', label: '已解锁的关卡' },
                { value: 'all', label: '全部关卡' },
              ]}
            />
            <div className="mb-2 mt-5 flex items-baseline justify-between text-sm font-semibold">
              分层筛选
              <button type="button" className="text-xs font-normal text-muted hover:text-fg" onClick={() => patch({ tiers: [] })}>
                {cfg.tiers.length ? '清除' : '不限'}
              </button>
            </div>
            <div className="space-y-2">
              {(['algo', 'llm'] as const).map((t) => (
                <div key={t} className="flex flex-wrap items-center gap-1.5">
                  <span className="w-16 shrink-0 text-xs text-muted">{TRACK_META[t].short}</span>
                  {TRACK_META[t].tiers.map((tier) => {
                    const on = cfg.tiers.includes(tier)
                    return (
                      <button
                        key={tier}
                        type="button"
                        aria-pressed={on}
                        onClick={() => toggleTier(tier)}
                        className={cn(
                          'rounded-full border px-2.5 py-1 text-xs font-medium transition-colors',
                          on ? 'border-accent bg-accent-soft text-fg' : 'border-border text-muted hover:border-border-strong hover:text-fg',
                        )}
                      >
                        {TIER_LABEL[tier]}
                      </button>
                    )
                  })}
                </div>
              ))}
            </div>
          </div>
        </Card>

        <div className="space-y-4">
          <Card className="p-5">
            <div className="flex items-center gap-3">
              <div className="grid h-11 w-11 place-items-center rounded-xl bg-accent-soft ink-accent">
                <Timer size={20} />
              </div>
              <div>
                <div className="font-semibold">
                  {total} 道题 · {cfg.durationMin} 分钟
                </div>
                <div className="text-xs text-muted">
                  候选关卡：算法 {nAlgo} · 大模型 {nLlm}
                </div>
              </div>
            </div>
            <ul className="mt-4 space-y-1.5 text-[13px] text-muted">
              <li>· 实战题权重 3，概念题权重 1，满分 100</li>
              <li>· 可以随时跳题、回看；时间到自动交卷</li>
              <li>· 作答过半即可获得 +{XP.mockFinish} XP；答错的题进入错题本</li>
            </ul>
            {error && (
              <div className="mt-4 space-y-1 rounded-xl bg-warning-soft px-3 py-2 text-[13px] ink-warning">
                {error.map((e) => (
                  <div key={e} className="flex gap-1.5">
                    <TriangleAlert size={14} className="mt-0.5 shrink-0" />
                    {e}
                  </div>
                ))}
              </div>
            )}
            <Button variant="primary" size="lg" className="mt-5 w-full" disabled={busy || total === 0 || cands.length === 0} onClick={go} icon={busy ? <Loader2 size={16} className="animate-spin" /> : <Flag size={16} />}>
              {busy ? '正在组卷…' : '开始模拟面试'}
            </Button>
            {cands.length === 0 && <div className="mt-2 text-center text-xs text-subtle">当前范围内没有可用关卡</div>}
          </Card>
        </div>
      </div>

      <section>
        <h2 className="mb-3 flex items-center gap-2 text-[15px] font-semibold">
          <History size={16} className="text-muted" /> 历史记录
        </h2>
        {history.length === 0 ? (
          <EmptyState icon={Trophy} title="还没有模拟面试记录">
            第一次建议选 30 分钟、1 道算法实战 + 4 道概念题，熟悉节奏。
          </EmptyState>
        ) : (
          <ul className="divide-y divide-border overflow-hidden rounded-2xl border border-border bg-surface shadow-sm">
            {history.map((h) => (
              <li key={h.id}>
                <button type="button" onClick={() => onOpen(h)} className="flex w-full items-center gap-4 px-4 py-3 text-left transition-colors hover:bg-surface-2">
                  <ScoreBadge score={h.score} />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium">{new Date(h.startedAt).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</div>
                    <div className="text-xs text-muted">
                      {h.items.length} 题 · 用时 {fmtDuration(h.usedSec)} / {h.durationMin} 分钟{h.timeUp && ' · 超时交卷'}
                    </div>
                  </div>
                  <ChevronRight size={16} className="text-subtle" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

function ScoreBadge({ score }: { score: number }) {
  const tone = score >= 80 ? 'bg-success-soft ink-success' : score >= 60 ? 'bg-warning-soft ink-warning' : 'bg-danger-soft ink-danger'
  return <span className={cn('grid h-11 w-11 shrink-0 place-items-center rounded-xl text-[15px] font-bold tabular-nums', tone)}>{score}</span>
}

/* ------------------------------------------------------------------ */
/* 进行中                                                              */
/* ------------------------------------------------------------------ */

const STATUS_CLS: Record<MockItemStatus, string> = {
  pending: 'border-border bg-surface text-muted',
  correct: 'border-transparent bg-success-soft ink-success',
  passed: 'border-transparent bg-success-soft ink-success',
  wrong: 'border-transparent bg-danger-soft ink-danger',
  skipped: 'border-transparent bg-surface-3 text-subtle',
}

function useNow(ms = 1000) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(t)
  }, [ms])
  return now
}

function Runner({ cat, session, onFinished }: { cat: Catalog; session: MockSession; onFinished: (r: MockRecord) => void }) {
  const goTo = useMock((s) => s.goTo)
  const setResult = useMock((s) => s.setResult)
  const finish = useMock((s) => s.finish)
  const discard = useMock((s) => s.discard)
  const awardXp = useProgress((s) => s.awardXp)
  const recordAnswer = useProgress((s) => s.recordAnswer)
  const contents = useLevelContents(session.items.map((i) => i.levelId))
  const now = useNow()
  const left = Math.max(0, Math.round((session.deadline - now) / 1000))
  const [confirm, setConfirm] = useState<null | 'submit' | 'quit'>(null)
  const i = session.active
  const item = session.items[i]
  const res = session.results[i]
  const answered = session.results.filter((r) => isItemDone(r.status) && r.status !== 'skipped').length

  const submit = (timeUp: boolean) => {
    const s = useMock.getState().active
    if (!s) return
    const done = s.results.filter((r) => isItemDone(r.status) && r.status !== 'skipped').length
    const xp = done >= Math.ceil(s.items.length / 2) ? XP.mockFinish : 0
    if (xp) awardXp(xp, '完成模拟面试')
    const rec = finish({ timeUp, xpAwarded: xp })
    if (rec) onFinished(rec)
  }

  useEffect(() => {
    if (left === 0) submit(true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [left === 0])

  const content = contents.get(item.levelId)
  const ref = cat.levelById.get(item.levelId)
  const next = () => i < session.items.length - 1 && goTo(i + 1)

  let body: React.ReactNode
  if (content === undefined) body = <PageSpinner label="正在加载题目…" />
  else if (item.kind === 'challenge') {
    const ch = content ? findChallenge(content, item.itemId) : undefined
    body = ch ? (
      <div className="h-[max(560px,calc(100dvh-260px))]" data-mock-workspace>
        <ChallengeWorkspace key={`${session.id}-${i}`} levelId={item.levelId} challenge={ch} mode="interview" interviewId={session.id} onPassed={() => setResult(i, 'passed')} />
      </div>
    ) : (
      <Missing onSkip={() => { setResult(i, 'skipped'); next() }} />
    )
  } else {
    const q = content ? findQuestion(content, item.itemId) : undefined
    body = q ? (
      <div>
        {isItemDone(res.status) && res.status !== 'skipped' ? (
          <>
            <div className={cn('mb-3 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium', STATUS_CLS[res.status])}>
              {isItemGood(res.status) ? <Check size={13} /> : <X size={13} />}
              {isItemGood(res.status) ? '已答对' : '已答错'}
            </div>
            <QuestionView key={`r-${session.id}-${i}`} question={q} reveal seed={i + 1} />
          </>
        ) : (
          <QuestionView
            key={`${session.id}-${i}`}
            question={q}
            seed={i + 1}
            onAnswered={(ok) => {
              setResult(i, ok ? 'correct' : 'wrong')
              recordAnswer(item.levelId, item.itemId, ok, 'mock')
            }}
            onNext={next}
            nextLabel={i < session.items.length - 1 ? '下一题' : '完成'}
          />
        )}
      </div>
    ) : (
      <Missing onSkip={() => { setResult(i, 'skipped'); next() }} />
    )
  }

  return (
    <div className="space-y-4">
      <div className="sticky top-[calc(env(safe-area-inset-top)+56px)] z-20 -mx-4 border-b border-border bg-bg/90 px-4 py-3 backdrop-blur-md md:top-0 md:-mx-8 md:px-8">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              'flex items-center gap-1.5 rounded-xl px-3 py-1.5 font-mono text-lg font-bold tabular-nums',
              left < 300 ? 'bg-danger-soft ink-danger' : 'bg-surface-2 text-fg',
            )}
            role="timer"
            aria-label={`剩余 ${fmtDuration(left)}`}
          >
            <Clock size={17} />
            {fmtDuration(left)}
          </div>
          <div className="hidden text-sm text-muted sm:block">
            已作答 {answered} / {session.items.length}
          </div>
          <div className="ml-auto flex gap-2">
            <Button variant="ghost" size="sm" onClick={() => setConfirm('quit')}>
              放弃
            </Button>
            <Button variant="primary" size="sm" icon={<Flag size={14} />} onClick={() => setConfirm('submit')}>
              交卷
            </Button>
          </div>
        </div>
        <div className="-mx-1 mt-2 flex gap-1.5 overflow-x-auto p-1 no-scrollbar" role="tablist" aria-label="题目导航">
          {session.items.map((it, k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={k === i}
              onClick={() => goTo(k)}
              title={`${k + 1}. ${it.kind === 'challenge' ? '实战' : '概念题'} · ${it.title}`}
              className={cn(
                'flex h-8 shrink-0 items-center gap-1 rounded-lg border px-2 text-xs font-semibold tabular-nums transition-colors',
                STATUS_CLS[session.results[k].status],
                k === i && 'ring-2 ring-accent ring-offset-1 ring-offset-bg',
              )}
            >
              {it.kind === 'challenge' ? <Code size={12} /> : <ListChecks size={12} />}
              {k + 1}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
        <span className="font-semibold text-fg">
          第 {i + 1} 题 · {item.kind === 'challenge' ? `实战${item.difficulty ? `（${item.difficulty}）` : ''}` : '概念题'}
        </span>
        <TrackBadge track={item.track} size="xs" />
        {ref && <TierBadge tier={ref.meta.tier} size="xs" />}
        {item.kind === 'challenge' && <span>建议用时 15–25 分钟</span>}
      </div>

      {body}

      <div className="flex items-center justify-between gap-2 pb-2">
        <Button variant="secondary" icon={<ArrowLeft size={15} />} disabled={i === 0} onClick={() => goTo(i - 1)}>
          上一题
        </Button>
        {res.status === 'pending' && (
          <Button
            variant="ghost"
            icon={<SkipForward size={15} />}
            onClick={() => {
              setResult(i, 'skipped')
              next()
            }}
          >
            跳过
          </Button>
        )}
        <Button variant="secondary" disabled={i === session.items.length - 1} onClick={() => goTo(i + 1)}>
          下一题 <ArrowRight size={15} />
        </Button>
      </div>

      <ConfirmDialog
        open={confirm === 'submit'}
        title="确认交卷？"
        confirmLabel="交卷"
        onCancel={() => setConfirm(null)}
        onConfirm={() => {
          setConfirm(null)
          submit(false)
        }}
      >
        已作答 {answered} / {session.items.length} 题，剩余 {fmtDuration(left)}。交卷后将生成报告{answered >= Math.ceil(session.items.length / 2) ? `并获得 ${XP.mockFinish} XP` : '（作答未过半，不发放 XP）'}。
      </ConfirmDialog>
      <ConfirmDialog
        open={confirm === 'quit'}
        title="放弃本场模拟面试？"
        confirmLabel="放弃"
        danger
        onCancel={() => setConfirm(null)}
        onConfirm={() => {
          setConfirm(null)
          discard()
        }}
      >
        本场不会生成报告，也不会记入历史。已作答的错题仍会保留在错题本里。
      </ConfirmDialog>
    </div>
  )
}

function Missing({ onSkip }: { onSkip: () => void }) {
  return (
    <Card className="p-6 text-center text-sm text-muted">
      这道题的内容加载失败或已下线。
      <div className="mt-3">
        <Button variant="secondary" onClick={onSkip}>
          跳过此题
        </Button>
      </div>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/* 报告                                                                */
/* ------------------------------------------------------------------ */

function Report({ cat, rec, onBack }: { cat: Catalog; rec: MockRecord; onBack: () => void }) {
  const score = rec.score
  const verdict = score >= 85 ? '稳了：这个水平可以放心去面' : score >= 70 ? '不错：查漏补缺后更稳' : score >= 50 ? '及格边缘：薄弱点需要专项练习' : '需要加强：先回到关卡把基础打牢'
  const bad = rec.items.filter((it) => !isItemGood(it.status))
  const weakChapters = new Map<string, { title: string; n: number; track: 'algo' | 'llm' }>()
  const weakLevels = new Map<string, string>()
  for (const it of bad) {
    const ref = cat.levelById.get(it.levelId)
    if (!ref) continue
    const w = weakChapters.get(ref.chapter.id) ?? { title: ref.chapter.title, n: 0, track: ref.chapter.track }
    w.n++
    weakChapters.set(ref.chapter.id, w)
    weakLevels.set(it.levelId, ref.meta.title)
  }
  const statusLabel: Record<MockItemStatus, string> = { pending: '未完成', correct: '答对', passed: '通过', wrong: '答错', skipped: '跳过' }

  return (
    <div className="space-y-5">
      <button type="button" onClick={onBack} className="inline-flex items-center gap-1 text-sm text-muted hover:text-fg">
        <ArrowLeft size={15} /> 模拟面试
      </button>
      <Card className="flex flex-col items-center gap-5 p-6 text-center sm:flex-row sm:text-left md:p-8">
        <ProgressRing value={score / 100} size={120} stroke={11} color={score >= 80 ? 'var(--success)' : score >= 60 ? 'var(--warning)' : 'var(--danger)'} label={`得分 ${score}`}>
          <div className="leading-none">
            <div className="text-4xl font-black tabular-nums">{score}</div>
            <div className="mt-1 text-xs text-subtle">/ 100</div>
          </div>
        </ProgressRing>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold text-subtle">{new Date(rec.startedAt).toLocaleString('zh-CN')}</div>
          <h1 className="mt-1 text-xl font-bold md:text-2xl">{verdict}</h1>
          <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1 text-sm text-muted sm:justify-start">
            <span>用时 {fmtDuration(rec.usedSec)} / {rec.durationMin} 分钟</span>
            <span>
              正确 {rec.items.filter((x) => isItemGood(x.status)).length} / {rec.items.length}
            </span>
            {rec.timeUp && <span className="ink-warning">时间到自动交卷</span>}
            {rec.xpAwarded > 0 && <span className="ink-gold">+{rec.xpAwarded} XP</span>}
          </div>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
        <Card className="overflow-hidden">
          <div className="border-b border-border px-5 py-3 text-sm font-semibold">各题表现</div>
          <ul className="divide-y divide-border">
            {rec.items.map((it, k) => (
              <li key={k} className="flex items-center gap-3 px-5 py-3">
                <span className="w-5 shrink-0 text-xs font-semibold text-subtle tabular-nums">{k + 1}</span>
                <span className={cn('grid h-7 w-7 shrink-0 place-items-center rounded-lg', it.kind === 'challenge' ? 'bg-accent-soft ink-accent' : 'bg-surface-2 text-muted')}>
                  {it.kind === 'challenge' ? <Code size={14} /> : <ListChecks size={14} />}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm">{it.title}</div>
                  <div className="text-[11.5px] text-subtle">{it.kind === 'challenge' ? '实战' : '概念题'} · 用时 {fmtDuration(it.timeSec)}</div>
                </div>
                <span className={cn('shrink-0 rounded-full px-2 py-0.5 text-xs font-medium', STATUS_CLS[it.status])}>{statusLabel[it.status]}</span>
              </li>
            ))}
          </ul>
        </Card>
        <div className="space-y-4">
          <Card className="p-5">
            <div className="mb-3 text-sm font-semibold">薄弱点</div>
            {weakChapters.size === 0 ? (
              <div className="text-sm text-muted">全部答对，没有明显短板。</div>
            ) : (
              <ul className="space-y-2">
                {[...weakChapters.entries()]
                  .sort((a, b) => b[1].n - a[1].n)
                  .map(([id, w]) => (
                    <li key={id} className="flex items-center justify-between gap-2 text-sm">
                      <Link to={`/chapter/${id}`} className="truncate hover:underline">
                        {w.title}
                      </Link>
                      <span className="shrink-0 text-xs ink-danger">失分 {w.n} 题</span>
                    </li>
                  ))}
              </ul>
            )}
          </Card>
          {weakLevels.size > 0 && (
            <Card className="p-5">
              <div className="mb-3 text-sm font-semibold">建议复习的关卡</div>
              <ul className="space-y-1.5">
                {[...weakLevels.entries()].map(([id, title]) => (
                  <li key={id}>
                    <Link to={levelHref(id)} className="flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors hover:bg-surface-2">
                      <span className="truncate">{title}</span>
                      <ArrowRight size={14} className="shrink-0 text-subtle" />
                    </Link>
                  </li>
                ))}
              </ul>
            </Card>
          )}
          <div className="flex gap-2">
            <Button variant="primary" className="flex-1" icon={<RotateCcw size={15} />} onClick={onBack}>
              再来一场
            </Button>
            <Link to="/review" className="inline-flex h-10 flex-1 items-center justify-center rounded-xl border border-border text-sm font-medium hover:bg-surface-2">
              去复习错题
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
