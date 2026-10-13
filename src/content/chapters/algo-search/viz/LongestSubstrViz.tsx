/**
 * LC3 无重复字符的最长子串：两种写法对照。
 *  - 计数 + while 收缩：窗口内每个字符计数 ≤ 1
 *  - 下标跳跃：记录字符最近出现的位置，l 直接跳到重复位置之后（注意 last[c] >= l 的判断）
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type ArrayRange, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { cn } from '@/lib/utils'
import { C } from './common'

type Mode = 'count' | 'jump'
const S = 'abcbdefbgh'

const CODE: Record<Mode, string> = {
  count: `def length_of_longest(s):
    cnt = {}
    best = l = 0
    for r, c in enumerate(s):
        cnt[c] = cnt.get(c, 0) + 1
        while cnt[c] > 1:            # 有重复 → 左缩直到无重复
            cnt[s[l]] -= 1
            l += 1
        best = max(best, r - l + 1)  # 求最长：收缩之后更新
    return best`,
  jump: `def length_of_longest(s):
    last = {}                        # 字符 → 最近一次出现的下标
    best = l = 0
    for r, c in enumerate(s):
        if c in last and last[c] >= l:
            l = last[c] + 1          # 直接跳到重复字符之后
        last[c] = r
        best = max(best, r - l + 1)
    return best`,
}

interface Frame {
  l: number
  r: number
  line: number
  table: Record<string, number>
  best: number
  bestRange?: [number, number]
  dup?: number
  caption: React.ReactNode
}

function buildCount(): Frame[] {
  const fr: Frame[] = []
  const cnt: Record<string, number> = {}
  let best = 0
  let bestRange: [number, number] | undefined
  let l = 0
  for (let r = 0; r < S.length; r++) {
    const c = S[r]
    cnt[c] = (cnt[c] ?? 0) + 1
    if (cnt[c] > 1) {
      fr.push({ l, r, line: 6, table: { ...cnt }, best, bestRange, dup: r, caption: <>右扩纳入 <C>'{c}'</C>，计数变成 {cnt[c]}：窗口里有重复了，开始左缩。</> })
      while (cnt[c] > 1) {
        cnt[S[l]] -= 1
        l++
        fr.push({ l, r, line: 8, table: { ...cnt }, best, bestRange, caption: <>移出 <C>'{S[l - 1]}'</C>，<C>l = {l}</C>{cnt[c] > 1 ? '，仍有重复。' : `，'${c}' 的计数回到 1，窗口重新合法。`}</> })
      }
    }
    if (r - l + 1 > best) {
      best = r - l + 1
      bestRange = [l, r]
    }
    fr.push({ l, r, line: 9, table: { ...cnt }, best, bestRange, caption: <>窗口 <C>"{S.slice(l, r + 1)}"</C> 无重复，长度 {r - l + 1}，best = {best}。</> })
  }
  fr.push({ l, r: S.length - 1, line: 10, table: { ...cnt }, best, bestRange, caption: <>结束：最长无重复子串 <C>"{bestRange ? S.slice(bestRange[0], bestRange[1] + 1) : ''}"</C>，长度 {best}。l 和 r 都只前进，<C>O(n)</C>。</> })
  return fr
}

function buildJump(): Frame[] {
  const fr: Frame[] = []
  const last: Record<string, number> = {}
  let best = 0
  let bestRange: [number, number] | undefined
  let l = 0
  for (let r = 0; r < S.length; r++) {
    const c = S[r]
    if (c in last && last[c] >= l) {
      const old = l
      l = last[c] + 1
      fr.push({ l, r, line: 6, table: { ...last }, best, bestRange, dup: last[c], caption: <><C>'{c}'</C> 上次出现在下标 {last[c]}（≥ l = {old}，在窗口内）：l 直接跳到 <C>{l}</C>，一步完成收缩。</> })
    } else if (c in last) {
      fr.push({ l, r, line: 5, table: { ...last }, best, bestRange, caption: <><C>'{c}'</C> 上次出现在下标 {last[c]}，但它 &lt; l = {l}，已经不在窗口里——<b>不能</b>让 l 往回跳。</> })
    }
    last[c] = r
    if (r - l + 1 > best) {
      best = r - l + 1
      bestRange = [l, r]
    }
    fr.push({ l, r, line: 8, table: { ...last }, best, bestRange, caption: <>记录 <C>last['{c}'] = {r}</C>；窗口 <C>"{S.slice(l, r + 1)}"</C>，长度 {r - l + 1}，best = {best}。</> })
  }
  fr.push({ l, r: S.length - 1, line: 9, table: { ...last }, best, bestRange, caption: <>结束：长度 {best}。跳跃写法每个 r 只做 O(1) 工作，但必须检查 <C>last[c] &gt;= l</C>。</> })
  return fr
}

export function LongestSubstrViz() {
  const [mode, setMode] = useState<Mode>('count')
  const frames = useMemo(() => (mode === 'count' ? buildCount() : buildJump()), [mode])
  const st = useStepper(frames, { intervalMs: 1000 })
  const f = st.frame
  const ranges: ArrayRange[] = [{ from: f.l, to: f.r, tone: 'accent', label: '窗口' }]
  if (f.bestRange) ranges.push({ from: f.bestRange[0], to: f.bestRange[1], tone: 'success', label: `best=${f.best}` })
  const highlights: Record<number, VizTone> = {}
  if (f.dup != null) highlights[f.dup] = 'danger'
  highlights[f.r] = 'warning'
  const pointers: ArrayPointer[] = [
    { index: f.l, label: 'l', tone: 'info' },
    { index: f.r, label: 'r', tone: 'warning' },
  ]

  return (
    <VizFrame
      title="无重复最长子串：窗口里每个字符最多一次"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '窗口', tone: 'accent', variant: 'soft' },
            { label: '目前最长', tone: 'success', variant: 'soft' },
            { label: '重复字符', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'count', label: '计数 + while 收缩' },
              { value: 'jump', label: '下标跳跃' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-4 py-2">
          <ArrayView values={S.split('')} cellSize={38} ranges={ranges} highlights={highlights} pointers={pointers} dim={(i) => i < f.l || i > f.r} />
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">{mode === 'count' ? 'cnt：窗口内各字符的计数' : 'last：字符最近一次出现的下标'}</div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(f.table).map(([k, v]) => {
                const inWin = mode === 'count' ? v > 0 : v >= f.l
                return (
                  <span
                    key={k}
                    className={cn(
                      'rounded-md border px-1.5 py-0.5 font-mono text-[12px]',
                      inWin ? 'border-accent/60 bg-accent-soft text-fg' : 'border-border text-subtle',
                      mode === 'count' && v > 1 && 'border-danger bg-danger-soft',
                    )}
                  >
                    {k}: {v}
                  </span>
                )
              })}
            </div>
          </div>
        </div>
        <CodeTrace code={CODE[mode]} active={f.line} vars={{ l: f.l, r: f.r, best: f.best }} />
      </div>
    </VizFrame>
  )
}
