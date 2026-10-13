import { toneColor, toneSoft, type VizTone } from './tones'

export function Legend({ items }: { items: { label: React.ReactNode; tone: VizTone; variant?: 'solid' | 'soft' | 'line' }[] }) {
  return (
    <>
      {items.map((it, i) => (
        <span key={i} className="inline-flex items-center gap-1.5">
          {it.variant === 'line' ? (
            <span className="inline-block h-0.5 w-4 rounded" style={{ background: toneColor(it.tone) }} />
          ) : (
            <span
              className="inline-block h-3 w-3 rounded-[4px] border"
              style={{
                background: it.variant === 'soft' ? toneSoft(it.tone, 30) : toneColor(it.tone),
                borderColor: toneColor(it.tone),
              }}
            />
          )}
          {it.label}
        </span>
      ))}
    </>
  )
}
