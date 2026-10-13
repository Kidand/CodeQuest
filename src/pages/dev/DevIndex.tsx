/** 开发工具：列出全部章节/关卡，链接到预览页。 */
import { Link } from 'react-router'
import { useCatalog } from '@/content/catalog'
import { TIER_LABEL } from '@/content/types'
import { PageSpinner } from '@/components/PageSpinner'

export default function DevIndex() {
  const cat = useCatalog()
  if (!cat) return <PageSpinner />
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold">内容开发预览</h1>
      {cat.errors.length > 0 && (
        <div className="rounded-xl border border-danger/40 bg-danger-soft p-3 text-sm text-danger">
          {cat.errors.map((e) => (
            <div key={e.path}>
              {e.path}: {e.error}
            </div>
          ))}
        </div>
      )}
      {cat.chapters.map((c) => (
        <section key={c.id}>
          <h2 className="mb-2 font-semibold">
            [{c.track}] {c.order}. {c.title} <span className="font-mono text-xs text-subtle">{c.id}</span>
          </h2>
          <ul className="space-y-1 text-sm">
            {c.levels.map((l) => (
              <li key={l.id} className="flex flex-wrap items-center gap-2">
                <Link className="text-accent underline" to={`/dev/level/${l.id}`}>
                  {l.title}
                </Link>
                <span className="font-mono text-xs text-subtle">{l.id}</span>
                <span className="text-xs text-muted">
                  {TIER_LABEL[l.tier]} {l.kind === 'boss' ? '· BOSS' : ''}
                </span>
                <Link className="text-xs text-muted underline" to={`/level/${l.id}`}>
                  正式页
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}
