/**
 * 开发工具：静态展开整关内容，供内容作者截图自检。
 *   /dev/level/:id                     全部
 *   /dev/level/:id?part=learn          只看图解（part: learn|quiz|guided|challenge|qa|cheat）
 *   /dev/level/:id?section=<sectionId> 只看某一节（截图某个可视化时用）
 */
import { Link, useParams, useSearchParams } from 'react-router'
import { useCatalog, useLevelContent } from '@/content/catalog'
import { QUESTION_TYPE_LABEL, TIER_LABEL, type Question } from '@/content/types'
import { InlineMarkdown, Markdown } from '@/components/Markdown'
import { CodeBlock } from '@/components/CodeBlock'
import { PageSpinner } from '@/components/PageSpinner'
import { ThemeToggle } from '@/components/ui'
import { Component, type ReactNode } from 'react'

class Boundary extends Component<{ children: ReactNode; name: string }, { err?: Error }> {
  state: { err?: Error } = {}
  static getDerivedStateFromError(err: Error) {
    return { err }
  }
  render() {
    if (this.state.err)
      return (
        <pre className="whitespace-pre-wrap rounded-xl border border-danger bg-danger-soft p-3 text-xs text-danger">
          [{this.props.name}] 渲染异常：{String(this.state.err.stack ?? this.state.err)}
        </pre>
      )
    return this.props.children
  }
}

function QStatic({ q, idx }: { q: Question; idx?: number }) {
  return (
    <div className="space-y-2 rounded-xl border border-border bg-surface p-4">
      <div className="text-xs text-subtle">
        {idx != null ? `#${idx + 1} ` : ''}[{QUESTION_TYPE_LABEL[q.type]}] {q.id} {q.difficulty ? `· 难度 ${q.difficulty}` : ''}
      </div>
      <Markdown>{q.prompt}</Markdown>
      {(q.type === 'single' || q.type === 'multi') && (
        <ol className="list-[upper-alpha] space-y-1 pl-6 text-sm">
          {q.options.map((o, i) => {
            const ok = q.type === 'single' ? q.answer === i : q.answer.includes(i)
            return (
              <li key={i} className={ok ? 'font-semibold text-success' : ''}>
                <Markdown className="!text-sm">{o}</Markdown>
              </li>
            )
          })}
        </ol>
      )}
      {q.type === 'judge' && <div className="text-sm text-success">答案：{q.answer ? '正确' : '错误'}</div>}
      {q.type === 'fill' && <div className="text-sm text-success">答案：{q.blanks.map((b) => b.answers.join(' / ')).join(' ； ')}</div>}
      {q.type === 'numeric' && (
        <div className="text-sm text-success">
          答案：{q.answer} {q.unit} （容差 {q.tolerance ?? 0.01}）
        </div>
      )}
      {q.type === 'order' && (
        <ol className="list-decimal pl-6 text-sm text-success">
          {q.items.map((s, i) => (
            <li key={i}>
              <Markdown className="!text-sm">{s}</Markdown>
            </li>
          ))}
        </ol>
      )}
      {q.type === 'match' && (
        <ul className="text-sm text-success">
          {q.pairs.map(([a, b], i) => (
            <li key={i}>
              <InlineMarkdown>{a}</InlineMarkdown> ⟷ <InlineMarkdown>{b}</InlineMarkdown>
            </li>
          ))}
        </ul>
      )}
      {q.type === 'parsons' && <CodeBlock code={q.lines.join('\n') + (q.distractors?.length ? `\n# --- 干扰行 ---\n${q.distractors.join('\n')}` : '')} />}
      {q.type === 'bug' && <CodeBlock code={q.code} lineNumbers highlightLines={q.bugLines} />}
      {q.type === 'pick' && (
        <div className="grid gap-1 text-xs" style={{ gridTemplateColumns: `repeat(${q.cols ?? Math.min(8, q.items.length)}, minmax(0, 1fr))` }}>
          {q.items.map((it, i) => (
            <span key={i} className={`rounded border px-1.5 py-1 text-center font-mono ${q.answer.includes(i) ? 'border-success bg-success-soft' : 'border-border'}`}>
              {it}
            </span>
          ))}
        </div>
      )}
      <div className="rounded-lg bg-surface-2 p-3">
        <div className="mb-1 text-xs font-semibold text-muted">解析</div>
        <Markdown className="!text-sm">{q.explanation}</Markdown>
      </div>
    </div>
  )
}

export default function DevLevel() {
  const { levelId } = useParams()
  const [sp] = useSearchParams()
  const part = sp.get('part')
  const sectionId = sp.get('section')
  const cat = useCatalog()
  const { content: L, error, loading } = useLevelContent(levelId)
  const ref = cat?.levelById.get(levelId ?? '')
  const show = (p: string) => !part || part === p
  if (loading) return <PageSpinner />
  if (error || !L) return <div className="p-6 text-danger">加载失败：{error}</div>
  const sections = sectionId ? L.learn.filter((s) => s.id === sectionId) : L.learn
  return (
    <div className="mx-auto max-w-3xl space-y-8 px-4 py-6">
      <header className="flex items-start justify-between gap-3">
        <div>
          <Link to="/dev" className="text-xs text-muted underline">
            ← 全部
          </Link>
          <h1 className="text-2xl font-bold">{ref?.meta.title}</h1>
          <div className="text-sm text-muted">
            {ref?.chapter.title} · {ref ? TIER_LABEL[ref.meta.tier] : ''} · 难度 {ref?.meta.difficulty} · {ref?.meta.estMinutes} 分钟 ·{' '}
            {ref?.meta.problems?.join('、')}
          </div>
          {ref?.meta.llmLink && <div className="text-sm text-llm">大模型场景：{ref.meta.llmLink}</div>}
        </div>
        <ThemeToggle />
      </header>

      {show('learn') &&
        sections.map((s) => (
          <section key={s.id} id={`sec-${s.id}`} className="space-y-4" data-section={s.id}>
            <h2 className="text-lg font-bold">
              {s.title} <span className="font-mono text-xs text-subtle">#{s.id}</span>
            </h2>
            <Markdown>{s.body}</Markdown>
            {s.viz && (
              <Boundary name={s.id}>
                <div data-viz={s.id}>
                  <s.viz />
                </div>
              </Boundary>
            )}
            {s.after && <Markdown>{s.after}</Markdown>}
            {s.checkpoint && <QStatic q={s.checkpoint} />}
          </section>
        ))}

      {show('quiz') && !sectionId && (
        <section className="space-y-3">
          <h2 className="text-lg font-bold">随堂练习（{L.quiz.length} 题）</h2>
          {L.quiz.map((q, i) => (
            <QStatic key={q.id} q={q} idx={i} />
          ))}
        </section>
      )}

      {show('guided') &&
        !sectionId &&
        L.guided.map((g) => (
          <section key={g.id} className="space-y-3">
            <h2 className="text-lg font-bold">跟练：{g.title}</h2>
            <Markdown>{g.intro}</Markdown>
            <CodeBlock code={g.template} lineNumbers />
            {g.steps.map((st) => (
              <div key={st.blank} className="rounded-xl border border-border bg-surface p-3">
                <div className="text-xs font-semibold text-muted">空位 {`{{${st.blank}}}`}</div>
                <Markdown>{st.instruction}</Markdown>
                <CodeBlock code={st.answer} />
                {st.explain && <Markdown className="mt-2 !text-sm text-muted">{st.explain}</Markdown>}
              </div>
            ))}
            {g.outro && <Markdown>{g.outro}</Markdown>}
          </section>
        ))}

      {show('challenge') &&
        !sectionId &&
        L.challenges.map((c) => (
          <section key={c.id} className="space-y-3">
            <h2 className="text-lg font-bold">
              实战：{c.title} <span className="text-sm font-normal text-muted">[{c.difficulty}] {c.source}</span>
            </h2>
            <Markdown>{c.description}</Markdown>
            <div className="text-xs font-semibold text-muted">starter</div>
            <CodeBlock code={c.starter} />
            <div className="text-xs font-semibold text-muted">solution</div>
            <CodeBlock code={c.solution} />
            {c.solutionAlt?.map((a) => (
              <div key={a.label}>
                <div className="text-xs font-semibold text-muted">{a.label}</div>
                <CodeBlock code={a.code} />
              </div>
            ))}
            <Markdown>{c.explanation}</Markdown>
            <div className="text-xs font-semibold text-muted">hints</div>
            <ol className="list-decimal space-y-1 pl-6">
              {c.hints.map((h, i) => (
                <li key={i}>
                  <Markdown>{h}</Markdown>
                </li>
              ))}
            </ol>
            <div className="text-xs font-semibold text-muted">tests（{c.tests.length}）</div>
            {c.tests.map((t) => (
              <div key={t.name}>
                <div className="text-xs text-subtle">
                  {t.name} {t.hidden ? '（隐藏）' : ''}
                </div>
                <CodeBlock code={t.code} />
              </div>
            ))}
          </section>
        ))}

      {show('qa') && !sectionId && (
        <section className="space-y-3">
          <h2 className="text-lg font-bold">面试追问</h2>
          {L.interview.map((q) => (
            <div key={q.id} className="rounded-xl border border-border bg-surface p-4">
              <Markdown className="font-semibold">{q.q}</Markdown>
              <Markdown className="mt-2 text-muted">{q.a}</Markdown>
            </div>
          ))}
        </section>
      )}

      {show('cheat') && !sectionId && L.cheatsheet && (
        <section className="space-y-3">
          <h2 className="text-lg font-bold">速记卡</h2>
          <div className="rounded-xl border border-border bg-surface p-4">
            <Markdown>{L.cheatsheet}</Markdown>
          </div>
        </section>
      )}
    </div>
  )
}
