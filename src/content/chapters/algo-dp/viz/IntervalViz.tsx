/**
 * 区间 DP：
 *  - PalinTableViz：回文判定（5/647）与最长回文子序列（516）的上三角表，对比三种遍历顺序
 *  - CenterExpandViz：中心扩展，2n-1 个中心
 *  - BalloonViz：戳气球（312）——枚举区间内「最后一个被戳」的 k
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer } from '@/viz'
import { Segmented } from '@/components/ui'
import { DpGrid, type Cell, type DpArrow } from './DpGrid'
import { C, Stat, Sub } from './common'

type P = [number, number]

/* ------------------------------------------------------------------ */
/* 回文表                                                               */
/* ------------------------------------------------------------------ */

const S = 'abcbaad'
type Mode = 'pal' | 'lps'
type Order = 'len' | 'idesc' | 'iasc'

function code(mode: Mode, order: Order) {
  const head =
    mode === 'pal'
      ? `def palindrome_table(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]    # dp[i][j]: s[i..j] 是否回文`
      : `def longest_palindrome_subseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]        # dp[i][j]: s[i..j] 的最长回文子序列`
  const loops =
    order === 'len'
      ? `    for length in range(1, n + 1):          # 按区间长度递增
        for i in range(n - length + 1):
            j = i + length - 1`
      : order === 'idesc'
        ? `    for i in range(n - 1, -1, -1):          # i 倒序
        for j in range(i, n):               # j 正序
            # 此时第 i+1 行已经全部算好`
        : `    for i in range(n):                      # ✗ i 正序
        for j in range(i, n):
            # 此时第 i+1 行还没算！`
  const body =
    mode === 'pal'
      ? `            dp[i][j] = s[i] == s[j] and (j - i < 2 or dp[i+1][j-1])`
      : `            if i == j:
                dp[i][j] = 1
            elif s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2         # 两端都要
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])   # 丢一端`
  const tail = mode === 'pal' ? `    return dp` : `    return dp[0][n-1]`
  return `${head}\n${loops}\n${body}\n${tail}`
}

interface PFrame {
  val: (string | number | null)[][]
  cur?: P
  arrows: DpArrow[]
  bad: boolean[][]
  caption: React.ReactNode
  bodyLine: number
}

function buildPal(mode: Mode, order: Order): PFrame[] {
  const n = S.length
  const val: (string | number | null)[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (j < i ? '' : null)))
  const bad = Array.from({ length: n }, () => Array(n).fill(false))
  const done = Array.from({ length: n }, () => Array(n).fill(false))
  const cells: P[] = []
  if (order === 'len') for (let L = 1; L <= n; L++) for (let i = 0; i + L - 1 < n; i++) cells.push([i, i + L - 1])
  else if (order === 'idesc') for (let i = n - 1; i >= 0; i--) for (let j = i; j < n; j++) cells.push([i, j])
  else for (let i = 0; i < n; i++) for (let j = i; j < n; j++) cells.push([i, j])
  const get = (i: number, j: number): number | boolean => {
    if (j < i) return mode === 'pal' ? true : 0
    if (!done[i][j]) return mode === 'pal' ? false : 0 // 未计算：读到初始值
    return mode === 'pal' ? val[i][j] === 'T' : (val[i][j] as number)
  }
  const fr: PFrame[] = []
  const baseLine = 7 // 第一行 body 的行号
  for (const [i, j] of cells) {
    const deps: P[] = []
    let caption: React.ReactNode
    let arrows: DpArrow[] = []
    let line = baseLine
    if (mode === 'pal') {
      const eq = S[i] === S[j]
      let v: boolean
      if (j - i < 2) {
        v = eq
        caption = j === i ? <>单个字符 <C>s[{i}]</C> 一定是回文。</> : <>长度 2：<C>s[{i}]{eq ? ' == ' : ' ≠ '}s[{j}]</C> → {eq ? 'T' : 'F'}。</>
      } else {
        deps.push([i + 1, j - 1])
        const inner = get(i + 1, j - 1) as boolean
        v = eq && inner
        caption = (
          <>
            <C>s[{i}..{j}]</C>：两端 <C>'{S[i]}'</C>{eq ? '==' : '≠'}<C>'{S[j]}'</C>
            {eq ? <>，还要看内部 <C>dp[{i + 1}][{j - 1}] = {inner ? 'T' : 'F'}</C></> : '，直接 F'} → <b>{v ? 'T' : 'F'}</b>。
          </>
        )
      }
      val[i][j] = v ? 'T' : 'F'
      arrows = deps.map((d) => ({ from: d, to: [i, j] as P, tone: done[d[0]][d[1]] ? 'success' : 'danger' }))
    } else {
      let v: number
      if (i === j) {
        v = 1
        line = baseLine + 1
        caption = <>单个字符：<C>dp[{i}][{i}] = 1</C>。</>
      } else if (S[i] === S[j]) {
        deps.push([i + 1, j - 1])
        v = (get(i + 1, j - 1) as number) + 2
        line = baseLine + 3
        caption = <>两端都是 <C>'{S[i]}'</C>：都放进回文，<C>dp[{i + 1}][{j - 1}] + 2 = {v}</C>。</>
      } else {
        deps.push([i + 1, j], [i, j - 1])
        const a = get(i + 1, j) as number
        const b = get(i, j - 1) as number
        v = Math.max(a, b)
        line = baseLine + 5
        caption = <><C>'{S[i]}' ≠ '{S[j]}'</C>：至少丢一端，<C>max(下 {a}, 左 {b}) = {v}</C>。</>
      }
      val[i][j] = v
      arrows = deps.filter((d) => d[1] >= d[0]).map((d) => ({ from: d, to: [i, j] as P, tone: done[d[0]][d[1]] ? 'success' : 'danger' }))
    }
    const isBad = deps.some((d) => d[1] >= d[0] && (!done[d[0]][d[1]] || bad[d[0]][d[1]]))
    bad[i][j] = isBad
    if (deps.some((d) => d[1] >= d[0] && !done[d[0]][d[1]]))
      caption = (
        <>
          {caption} <b className="text-danger">✗ 依赖的 dp[{i + 1}][…] 还没算（i 正序时下一行在后面），结果不可信！</b>
        </>
      )
    done[i][j] = true
    fr.push({ val: val.map((r) => [...r]), cur: [i, j], arrows, bad: bad.map((r) => [...r]), caption, bodyLine: line })
  }
  const summary =
    mode === 'pal' ? (
      <>
        共 <C>{val.flat().filter((x) => x === 'T').length}</C> 个回文子串（LC 647 的答案），最长回文子串是右上方向最长的 T。
      </>
    ) : (
      <>
        答案 <C>dp[0][{n - 1}] = {val[0][n - 1]}</C>。
      </>
    )
  fr.push({
    val: val.map((r) => [...r]),
    arrows: [],
    bad: bad.map((r) => [...r]),
    caption:
      order === 'iasc' ? (
        <>i 正序读到了还没算的下一行，红色格子都是错的。区间 DP 的依赖指向<b>更短的区间</b>（左下方），必须先算它们。</>
      ) : (
        summary
      ),
    bodyLine: mode === 'pal' ? 8 : 13,
  })
  return fr
}

export function PalinTableViz() {
  const [mode, setMode] = useState<Mode>('pal')
  const [order, setOrder] = useState<Order>('len')
  const frames = useMemo(() => buildPal(mode, order), [mode, order])
  const st = useStepper(frames, { intervalMs: 700 })
  const f = st.frame
  const isCur = (r: number, c: number) => !!f.cur && f.cur[0] === r && f.cur[1] === c
  return (
    <VizFrame
      title="区间 DP：依赖指向更短的区间，按长度（或 i 倒序）填上三角"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前区间 [i, j]', tone: 'accent', variant: 'soft' },
            { label: '依赖（已算好）', tone: 'success', variant: 'line' },
            { label: '依赖未计算 / 结果被污染', tone: 'danger', variant: 'soft' },
            ...(mode === 'pal' ? [{ label: '回文 T', tone: 'success' as const, variant: 'soft' as const }] : []),
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
              { value: 'pal', label: '回文判定（5 / 647）' },
              { value: 'lps', label: '回文子序列（516）' },
            ]}
          />
          <Segmented
            size="sm"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'len', label: '按长度 ✓' },
              { value: 'idesc', label: 'i 倒序 ✓' },
              { value: 'iasc', label: 'i 正序 ✗' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <div className="text-[13px] text-muted">
          s = <C>"{S}"</C>，行是左端点 i，列是右端点 j；左下三角（i &gt; j）无意义。
        </div>
        <DpGrid
          data={f.val as Cell[][]}
          rowLabels={S.split('').map((ch, i) => `${i} ${ch}`)}
          colLabels={S.split('').map((ch, j) => `${j} ${ch}`)}
          cellSize={44}
          arrows={f.arrows}
          dim={(r, c) => c < r}
          ring={(r, c) => (isCur(r, c) ? 'accent' : undefined)}
          fill={(r, c) =>
            f.bad[r][c] ? 'danger' : isCur(r, c) ? 'accent' : mode === 'pal' && f.val[r][c] === 'T' ? 'success' : undefined
          }
        />
        <CodeTrace code={code(mode, order)} active={f.bodyLine} vars={f.cur ? { i: f.cur[0], j: f.cur[1], 长度: f.cur[1] - f.cur[0] + 1 } : undefined} />
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 中心扩展                                                             */
/* ------------------------------------------------------------------ */

const CS = 'cbabcad'

interface CFrame {
  c: number // 中心编号 0..2n-2
  l: number
  r: number
  ok: boolean
  best: [number, number]
  caption: React.ReactNode
}

function buildCenter(): CFrame[] {
  const n = CS.length
  const fr: CFrame[] = []
  let best: [number, number] = [0, 0]
  for (let c = 0; c < 2 * n - 1; c++) {
    let l = Math.floor(c / 2)
    let r = l + (c % 2)
    const kind = c % 2 === 0 ? `字符 s[${l}]` : `s[${l}] 与 s[${r}] 之间`
    while (l >= 0 && r < n && CS[l] === CS[r]) {
      if (r - l > best[1] - best[0]) best = [l, r]
      fr.push({ c, l, r, ok: true, best: [...best] as [number, number], caption: <>中心 #{c}（{kind}）：<C>s[{l}..{r}] = "{CS.slice(l, r + 1)}"</C> 是回文，继续向两侧扩展。</> })
      l--
      r++
    }
    fr.push({
      c,
      l,
      r,
      ok: false,
      best: [...best] as [number, number],
      caption:
        l < 0 || r >= n ? (
          <>中心 #{c}（{kind}）：碰到边界，停止。以它为中心的最长回文是 <C>"{CS.slice(l + 1, r)}"</C>。</>
        ) : (
          <>
            中心 #{c}（{kind}）：<C>s[{l}]='{CS[l]}' ≠ s[{r}]='{CS[r]}'</C>，停止。最长为 <C>"{CS.slice(l + 1, r) || '∅'}"</C>。
          </>
        ),
    })
  }
  return fr
}

export function CenterExpandViz() {
  const frames = useMemo(buildCenter, [])
  const st = useStepper(frames, { intervalMs: 700 })
  const f = st.frame
  const n = CS.length
  const inPal = (i: number) => (f.ok ? i >= f.l && i <= f.r : i > f.l && i < f.r)
  const pointers: ArrayPointer[] = []
  if (f.l >= 0) pointers.push({ index: f.l, label: 'L', tone: f.ok ? 'info' : 'danger' })
  if (f.r < n) pointers.push({ index: f.r, label: 'R', tone: f.ok ? 'warning' : 'danger' })
  const hl: Record<number, 'success' | 'accent' | 'danger'> = {}
  for (let i = 0; i < n; i++) if (inPal(i)) hl[i] = 'success'
  if (!f.ok) {
    if (f.l >= 0) hl[f.l] = 'danger'
    if (f.r < n) hl[f.r] = 'danger'
  }
  return (
    <VizFrame
      title="中心扩展：2n−1 个中心，各自向两边扩，O(n²) 时间 O(1) 空间"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前回文', tone: 'success', variant: 'soft' },
            { label: '扩展失败的位置', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Stat label="中心" value={`${f.c + 1} / ${2 * n - 1}`} />
          <Stat label="最长回文" value={`"${CS.slice(f.best[0], f.best[1] + 1)}"`} tone="success" />
        </div>
        <ArrayView values={CS.split('')} cellSize={46} highlights={hl} pointers={pointers} />
        <div className="text-[13px] text-muted">
          偶数编号的中心是字符本身（奇数长度回文），奇数编号的中心是两个字符之间（偶数长度回文）。漏掉后者是最常见的错误。
        </div>
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 戳气球                                                               */
/* ------------------------------------------------------------------ */

const BAL = [3, 1, 5, 8]
const PB = [1, ...BAL, 1]

interface BFrame {
  dp: (number | null)[][]
  cur?: P
  bestK?: number
  cands: { k: number; v: number }[]
  caption: React.ReactNode
}

function buildBalloon(): BFrame[] {
  const m = PB.length
  const dp: (number | null)[][] = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => (j - i < 2 ? (j > i ? 0 : null) : null)))
  const fr: BFrame[] = [
    { dp: dp.map((r) => [...r]), cands: [], caption: <>两端补上虚拟气球 1：<C>p = [{PB.join(', ')}]</C>。<C>dp[i][j]</C> = 戳光<b>开区间</b> (i, j) 内所有气球的最大得分；相邻（j = i+1）时区间为空，得分 0。</> },
  ]
  for (let len = 2; len < m; len++)
    for (let i = 0; i + len < m; i++) {
      const j = i + len
      const cands = []
      for (let k = i + 1; k < j; k++) cands.push({ k, v: dp[i][k]! + dp[k][j]! + PB[i] * PB[k] * PB[j] })
      const best = cands.reduce((a, b) => (b.v > a.v ? b : a))
      dp[i][j] = best.v
      fr.push({
        dp: dp.map((r) => [...r]),
        cur: [i, j],
        bestK: best.k,
        cands,
        caption: (
          <>
            区间 ({i}, {j})：枚举<b>最后一个</b>被戳的 k。最后戳 k 时它的邻居恰好是 p[{i}] 和 p[{j}]：
            <C>dp[{i}][{best.k}] + dp[{best.k}][{j}] + {PB[i]}×{PB[best.k]}×{PB[j]} = {best.v}</C>（k = {best.k} 最优）。
          </>
        ),
      })
    }
  fr.push({ dp: dp.map((r) => [...r]), cands: [], caption: <>答案 <C>dp[0][{m - 1}] = {dp[0][m - 1]}</C>。</> })
  return fr
}

export function BalloonViz() {
  const frames = useMemo(buildBalloon, [])
  const st = useStepper(frames, { intervalMs: 1500 })
  const f = st.frame
  const arrows: DpArrow[] =
    f.cur && f.bestK != null
      ? [
          { from: [f.cur[0], f.bestK], to: f.cur, tone: 'success' },
          { from: [f.bestK, f.cur[1]], to: f.cur, tone: 'info' },
        ]
      : []
  const pointers: ArrayPointer[] = f.cur
    ? [
        { index: f.cur[0], label: 'i', tone: 'info' },
        { index: f.cur[1], label: 'j', tone: 'info' },
        ...(f.bestK != null ? [{ index: f.bestK, label: 'k 最后戳', tone: 'accent' as const, side: 'above' as const }] : []),
      ]
    : []
  return (
    <VizFrame
      title="戳气球：不想“先戳谁”，而想“最后戳谁”——子问题才互相独立"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前区间 (i, j)', tone: 'accent', variant: 'soft' },
            { label: 'dp[i][k]（左半）', tone: 'success', variant: 'line' },
            { label: 'dp[k][j]（右半）', tone: 'info', variant: 'line' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="space-y-4">
        <div>
          <Sub>补了虚拟边界的气球 p（i、j 本身不戳）</Sub>
          <ArrayView
            values={PB}
            cellSize={44}
            pointers={pointers}
            highlights={f.cur ? Object.fromEntries(PB.map((_, t) => [t, t === f.bestK ? 'accent' : t > f.cur![0] && t < f.cur![1] ? 'warning' : 'muted'] as const).filter(([t]) => t >= f.cur![0] && t <= f.cur![1])) : {}}
            dim={(t) => !!f.cur && (t < f.cur[0] || t > f.cur[1])}
          />
        </div>
        {f.cands.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {f.cands.map((c) => (
              <Stat key={c.k} label={`k=${c.k}`} value={c.v} tone={c.k === f.bestK ? 'success' : 'fg'} />
            ))}
          </div>
        )}
        <DpGrid
          data={f.dp as Cell[][]}
          rowLabels={PB.map((_, i) => `i=${i}`)}
          colLabels={PB.map((_, j) => `j=${j}`)}
          cellSize={46}
          arrows={arrows}
          dim={(r, c) => c <= r}
          ring={(r, c) => (f.cur && f.cur[0] === r && f.cur[1] === c ? 'accent' : undefined)}
          fill={(r, c) => (f.cur && f.cur[0] === r && f.cur[1] === c ? 'accent' : undefined)}
        />
        <CodeTrace
          code={`def max_coins(nums):
    p = [1] + nums + [1]
    m = len(p)
    dp = [[0] * m for _ in range(m)]       # dp[i][j]: 开区间 (i, j)
    for length in range(2, m):             # 按区间长度递增
        for i in range(m - length):
            j = i + length
            dp[i][j] = max(dp[i][k] + dp[k][j] + p[i] * p[k] * p[j]
                           for k in range(i + 1, j))   # k = 最后戳的
    return dp[0][m - 1]`}
          active={f.cur ? [8, 9] : st.atEnd ? 10 : 2}
          vars={f.cur ? { i: f.cur[0], j: f.cur[1], 最优k: f.bestK ?? '—' } : undefined}
        />
      </div>
    </VizFrame>
  )
}
