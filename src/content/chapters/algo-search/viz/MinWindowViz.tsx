/**
 * LC76 最小覆盖子串：need 计数 + missing 计数器，O(1) 判断“窗口是否覆盖 t”。
 */
import { useMemo } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type ArrayRange, type VizTone } from '@/viz'
import { cn } from '@/lib/utils'
import { C } from './common'

const S = 'ADOBECBANC'
const T = 'ABC'

const CODE = `def min_window(s, t):
    need = Counter(t)                 # 还需要的各字符数量（可为负 = 多余）
    missing = len(t)                  # 还缺多少个字符
    l, best = 0, (0, float('inf'))
    for r, c in enumerate(s):
        if need[c] > 0:
            missing -= 1              # c 是一个“有用”的字符
        need[c] -= 1
        while missing == 0:           # 已覆盖 t → 左缩找更短
            if r - l < best[1] - best[0]:
                best = (l, r)
            need[s[l]] += 1
            if need[s[l]] > 0:
                missing += 1          # 缩掉了一个必需字符
            l += 1
    return '' if best[1] == float('inf') else s[best[0]:best[1] + 1]`

interface Frame {
  l: number
  r: number
  need: Record<string, number>
  missing: number
  best?: [number, number]
  line: number
  caption: React.ReactNode
}

function build(): Frame[] {
  const need: Record<string, number> = {}
  for (const ch of T) need[ch] = (need[ch] ?? 0) + 1
  let missing = T.length
  let l = 0
  let best: [number, number] | undefined
  const fr: Frame[] = []
  fr.push({ l: 0, r: -1, need: { ...need }, missing, line: 3, caption: <><C>need = {JSON.stringify(need)}</C>，<C>missing = {missing}</C>：窗口需要包含 1 个 A、1 个 B、1 个 C。</> })
  for (let r = 0; r < S.length; r++) {
    const c = S[r]
    const useful = (need[c] ?? 0) > 0
    if (useful) missing--
    need[c] = (need[c] ?? 0) - 1
    fr.push({
      l, r, need: { ...need }, missing, best, line: useful ? 7 : 8,
      caption: useful ? <>纳入 <C>'{c}'</C>：它是还缺的字符，<C>missing → {missing}</C>。</> : <>纳入 <C>'{c}'</C>：不需要（或已多余），<C>need['{c}'] = {need[c]}</C>，missing 不变。</>,
    })
    while (missing === 0) {
      if (!best || r - l < best[1] - best[0]) best = [l, r]
      fr.push({ l, r, need: { ...need }, missing, best, line: 11, caption: <><C>missing == 0</C>：<C>"{S.slice(l, r + 1)}"</C> 覆盖了 t，记录长度 {r - l + 1}，尝试左缩。</> })
      const d = S[l]
      need[d] = (need[d] ?? 0) + 1
      const lost = need[d] > 0
      if (lost) missing++
      l++
      fr.push({
        l, r, need: { ...need }, missing, best, line: lost ? 14 : 15,
        caption: lost ? <>移出 <C>'{d}'</C> 后 <C>need['{d}'] = 1 &gt; 0</C>：缺了一个必需字符，<C>missing = {missing}</C>，回到右扩。</> : <>移出 <C>'{d}'</C>：它是多余的（need 回到 {need[d]} ≤ 0），窗口仍覆盖 t，继续缩。</>,
      })
    }
  }
  fr.push({ l, r: S.length - 1, need: { ...need }, missing, best, line: 16, caption: <>结束：最小覆盖子串 <C>"{best ? S.slice(best[0], best[1] + 1) : ''}"</C>。missing 计数让“是否覆盖”的判断变成 <C>O(1)</C>，总复杂度 <C>O(|s| + |t|)</C>。</> })
  return fr
}

export function MinWindowViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 950 })
  const f = st.frame
  const ranges: ArrayRange[] = []
  if (f.r >= f.l) ranges.push({ from: f.l, to: f.r, tone: f.missing === 0 ? 'success' : 'accent', label: f.missing === 0 ? '覆盖 t' : '窗口' })
  if (f.best) ranges.push({ from: f.best[0], to: f.best[1], tone: 'gold', label: 'best' })
  const highlights: Record<number, VizTone> = {}
  S.split('').forEach((ch, i) => {
    if (T.includes(ch)) highlights[i] = 'info'
  })
  const pointers: ArrayPointer[] = [{ index: f.l, label: 'l', tone: 'info' }]
  if (f.r >= 0) pointers.push({ index: f.r, label: 'r', tone: 'warning' })

  return (
    <VizFrame
      title="最小覆盖子串：missing 计数器 O(1) 判断覆盖"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: 't 中的字符', tone: 'info', variant: 'soft' },
            { label: '当前窗口', tone: 'accent', variant: 'soft' },
            { label: '目前最短', tone: 'gold', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-4 py-2">
          <div className="text-xs text-muted">
            s = <span className="font-mono text-fg">"{S}"</span>，t = <span className="font-mono text-fg">"{T}"</span>
          </div>
          <ArrayView values={S.split('')} cellSize={38} ranges={ranges} highlights={highlights} pointers={pointers} dim={(i) => i < f.l || i > f.r} />
          <div className="flex flex-wrap items-center gap-1.5 font-mono text-[12px]">
            <span className="text-xs text-muted">need：</span>
            {Object.entries(f.need)
              .filter(([k]) => T.includes(k))
              .map(([k, v]) => (
                <span key={k} className={cn('rounded-md border px-1.5 py-0.5', v > 0 ? 'border-danger/60 bg-danger-soft' : 'border-success/60 bg-success-soft')}>
                  {k}: {v}
                </span>
              ))}
            <span className={cn('ml-2 rounded-md border px-1.5 py-0.5', f.missing === 0 ? 'border-success bg-success-soft' : 'border-border')}>missing = {f.missing}</span>
          </div>
        </div>
        <CodeTrace code={CODE} active={f.line} maxHeight={330} vars={{ l: f.l, r: f.r, missing: f.missing }} />
      </div>
    </VizFrame>
  )
}
