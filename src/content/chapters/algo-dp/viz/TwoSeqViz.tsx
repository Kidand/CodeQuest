/**
 * 双序列 DP：LCS 填表 + 回溯、编辑距离填表 + 操作回放、WER/CER 词级对齐。
 * 洞见：dp[i][j] 的三个方向分别对应「配对 / 丢 a 的末尾 / 丢 b 的末尾」，回溯路径就是对齐方式。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, TokenRow, VizFrame, toneColor, toneSoft, useStepper, type TokenItem, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { DpGrid, type Cell, type DpArrow } from './DpGrid'
import { C, Stat, Sub } from './common'

type P = [number, number]

/* ------------------------------------------------------------------ */
/* 通用：编辑距离表 + 回溯出操作序列                                    */
/* ------------------------------------------------------------------ */

export type Op = { op: 'keep' | 'sub' | 'del' | 'ins'; a: string; b: string; cell: P; from: P }

export function editTable<T>(a: T[], b: T[]) {
  const m = a.length
  const n = b.length
  const dp = Array.from({ length: m + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)))
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] : 1 + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
  return dp
}

export function backtrackOps(a: string[], b: string[], dp: number[][]): Op[] {
  const ops: Op[] = []
  let i = a.length
  let j = b.length
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1] && dp[i][j] === dp[i - 1][j - 1]) {
      ops.push({ op: 'keep', a: a[i - 1], b: b[j - 1], cell: [i, j], from: [i - 1, j - 1] })
      i--
      j--
    } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
      ops.push({ op: 'sub', a: a[i - 1], b: b[j - 1], cell: [i, j], from: [i - 1, j - 1] })
      i--
      j--
    } else if (i > 0 && dp[i][j] === dp[i - 1][j] + 1) {
      ops.push({ op: 'del', a: a[i - 1], b: '', cell: [i, j], from: [i - 1, j] })
      i--
    } else {
      ops.push({ op: 'ins', a: '', b: b[j - 1], cell: [i, j], from: [i, j - 1] })
      j--
    }
  }
  return ops.reverse()
}

const OP_TONE: Record<Op['op'], VizTone> = { keep: 'muted', sub: 'warning', del: 'danger', ins: 'success' }
const OP_NAME: Record<Op['op'], string> = { keep: '保留', sub: '替换', del: '删除', ins: '插入' }

/* ------------------------------------------------------------------ */
/* LCS                                                                  */
/* ------------------------------------------------------------------ */

const LCS_PRESETS: [string, string][] = [
  ['ABCBDAB', 'BDCABA'],
  ['AGGTAB', 'GXTXAYB'],
]
const LCS_CODE = `def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]   # 第 0 行/列 = 空串
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1      # 两个末尾配对
            else:
                dp[i][j] = max(dp[i-1][j],       # 丢掉 a 的末尾
                               dp[i][j-1])       # 丢掉 b 的末尾
    return dp[m][n]`

interface LFrame {
  dp: (number | null)[][]
  cur?: P
  arrows: DpArrow[]
  line: number | number[]
  caption: React.ReactNode
  path?: Set<string>
  matched?: Set<string>
}

function buildLcs(a: string, b: string): LFrame[] {
  const m = a.length
  const n = b.length
  const dp: (number | null)[][] = Array.from({ length: m + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (i === 0 || j === 0 ? 0 : null)))
  const fr: LFrame[] = [{ dp: dp.map((r) => [...r]), arrows: [], line: 3, caption: <>第 0 行、第 0 列对应空串：与空串的 LCS 为 0。</> }]
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++) {
      let arrows: DpArrow[]
      let caption: React.ReactNode
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1]! + 1
        arrows = [{ from: [i - 1, j - 1], to: [i, j], tone: 'success', label: '+1' }]
        caption = <><C>a[{i - 1}] = b[{j - 1}] = '{a[i - 1]}'</C>：末尾配对，<C>dp[{i}][{j}] = dp[{i - 1}][{j - 1}] + 1 = {dp[i][j]}</C>。</>
      } else {
        const up = dp[i - 1][j]!
        const left = dp[i][j - 1]!
        dp[i][j] = Math.max(up, left)
        arrows = [
          { from: [i - 1, j], to: [i, j], tone: up >= left ? 'success' : 'muted', dashed: up < left },
          { from: [i, j - 1], to: [i, j], tone: up >= left ? 'muted' : 'success', dashed: up >= left },
        ]
        caption = <><C>'{a[i - 1]}' ≠ '{b[j - 1]}'</C>：两者至少有一个不在 LCS 末尾，<C>max(上 {up}, 左 {left}) = {dp[i][j]}</C>。</>
      }
      fr.push({ dp: dp.map((r) => [...r]), cur: [i, j], arrows, line: a[i - 1] === b[j - 1] ? 7 : [9, 10], caption })
    }
  const path = new Set<string>()
  const matched = new Set<string>()
  let i = m
  let j = n
  const chars: string[] = []
  while (i > 0 && j > 0) {
    path.add(`${i},${j}`)
    if (a[i - 1] === b[j - 1]) {
      matched.add(`${i},${j}`)
      chars.push(a[i - 1])
      i--
      j--
    } else if (dp[i - 1][j]! >= dp[i][j - 1]!) i--
    else j--
  }
  path.add(`${i},${j}`)
  fr.push({
    dp: dp.map((r) => [...r]),
    arrows: [],
    line: 11,
    path,
    matched,
    caption: (
      <>
        LCS 长度 <C>{dp[m][n]}</C>。从右下角回溯：字符相等走对角线（绿色实心格 = 配对字符），否则走向较大的一侧。得到 <C>"{chars.reverse().join('')}"</C>。
      </>
    ),
  })
  return fr
}

export function LcsViz() {
  const [k, setK] = useState(0)
  const [a, b] = LCS_PRESETS[k]
  const frames = useMemo(() => buildLcs(a, b), [a, b])
  const st = useStepper(frames, { intervalMs: 600 })
  const f = st.frame
  const isCur = (r: number, c: number) => !!f.cur && f.cur[0] === r && f.cur[1] === c
  return (
    <VizFrame
      title="LCS：字符相等走对角线 +1，不等取上、左的较大值"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前格', tone: 'accent', variant: 'soft' },
            { label: '取用的依赖', tone: 'success', variant: 'line' },
            { label: '回溯路径 / 配对字符', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented size="sm" value={k} onChange={setK} options={LCS_PRESETS.map((p, i) => ({ value: i, label: `${p[0]} / ${p[1]}` }))} />
        </>
      }
    >
      <div className="space-y-4">
        <DpGrid
          data={f.dp as Cell[][]}
          rowLabels={['∅', ...a.split('')]}
          colLabels={['∅', ...b.split('')]}
          cellSize={42}
          arrows={f.arrows}
          ring={(r, c) => (isCur(r, c) ? 'accent' : f.path?.has(`${r},${c}`) ? 'success' : undefined)}
          fill={(r, c) => (isCur(r, c) ? 'accent' : f.matched?.has(`${r},${c}`) ? 'success' : undefined)}
        />
        <CodeTrace code={LCS_CODE} active={f.line} vars={f.cur ? { i: f.cur[0], j: f.cur[1], [`dp[${f.cur[0]}][${f.cur[1]}]`]: f.dp[f.cur[0]][f.cur[1]] ?? '—' } : undefined} />
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 编辑距离：填表 + 操作回放                                            */
/* ------------------------------------------------------------------ */

const ED_PRESETS: [string, string][] = [
  ['horse', 'ros'],
  ['intention', 'execution'],
  ['kitten', 'sitting'],
]
const ED_CODE = `def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i      # 删光 word1 的前 i 个
    for j in range(n + 1): dp[0][j] = j      # 插入 word2 的前 j 个
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]          # 末尾相同：不操作
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1], # 替换
                                   dp[i-1][j],   # 删除 word1[i-1]
                                   dp[i][j-1])   # 插入 word2[j-1]
    return dp[m][n]`

interface EFrame {
  dp: (number | null)[][]
  cur?: P
  arrows: DpArrow[]
  line: number | number[]
  caption: React.ReactNode
}

function buildEdFill(a: string, b: string): EFrame[] {
  const m = a.length
  const n = b.length
  const dp: (number | null)[][] = Array.from({ length: m + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : null)))
  const fr: EFrame[] = [
    { dp: dp.map((r) => [...r]), arrows: [], line: [4, 5], caption: <>边界：<C>dp[i][0] = i</C>（把 word1 前 i 个字符全删掉），<C>dp[0][j] = j</C>（从空串插入 j 个字符）。</> },
  ]
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1]
        fr.push({
          dp: dp.map((r) => [...r]),
          cur: [i, j],
          arrows: [{ from: [i - 1, j - 1], to: [i, j], tone: 'success', label: '=' }],
          line: 9,
          caption: <><C>'{a[i - 1]}' == '{b[j - 1]}'</C>：末尾相同不用操作，<C>dp[{i}][{j}] = dp[{i - 1}][{j - 1}] = {dp[i][j]}</C>。</>,
        })
      } else {
        const rep = dp[i - 1][j - 1]!
        const del = dp[i - 1][j]!
        const ins = dp[i][j - 1]!
        const mn = Math.min(rep, del, ins)
        dp[i][j] = 1 + mn
        const win = rep === mn ? 'rep' : del === mn ? 'del' : 'ins'
        fr.push({
          dp: dp.map((r) => [...r]),
          cur: [i, j],
          arrows: [
            { from: [i - 1, j - 1], to: [i, j], tone: win === 'rep' ? 'warning' : 'muted', dashed: win !== 'rep', label: win === 'rep' ? '替' : undefined },
            { from: [i - 1, j], to: [i, j], tone: win === 'del' ? 'danger' : 'muted', dashed: win !== 'del', label: win === 'del' ? '删' : undefined },
            { from: [i, j - 1], to: [i, j], tone: win === 'ins' ? 'success' : 'muted', dashed: win !== 'ins', label: win === 'ins' ? '插' : undefined },
          ],
          line: [11, 12, 13],
          caption: (
            <>
              <C>'{a[i - 1]}' ≠ '{b[j - 1]}'</C>：1 + min(替换 {rep}, 删除 {del}, 插入 {ins}) = <C>{dp[i][j]}</C>。
            </>
          ),
        })
      }
    }
  fr.push({ dp: dp.map((r) => [...r]), arrows: [], line: 14, caption: <>编辑距离 = <C>dp[{m}][{n}] = {dp[m][n]}</C>。切换到「操作回放」看它对应哪些操作。</> })
  return fr
}

interface RFrame {
  k: number // 已执行的操作数
  tokens: TokenItem[]
  caption: React.ReactNode
}

function buildReplay(a: string, ops: Op[]): RFrame[] {
  const fr: RFrame[] = []
  for (let k = 0; k <= ops.length; k++) {
    const toks: TokenItem[] = []
    let consumed = 0
    ops.slice(0, k).forEach((o, idx) => {
      const isLast = idx === k - 1
      if (o.op !== 'ins') consumed++
      if (o.op === 'keep') toks.push({ text: o.b, tone: isLast ? 'accent' : undefined, sub: '=' })
      else if (o.op === 'sub') toks.push({ text: o.b, tone: 'warning', sub: `${o.a}→${o.b}` })
      else if (o.op === 'del') toks.push({ text: o.a, tone: 'danger', dashed: true, dim: !isLast, sub: '删' })
      else toks.push({ text: o.b, tone: 'success', sub: '插' })
    })
    a.slice(consumed)
      .split('')
      .forEach((ch) => toks.push({ text: ch, sub: '待处理' }))
    const o = ops[k - 1]
    const edits = ops.slice(0, k).filter((x) => x.op !== 'keep').length
    fr.push({
      k,
      tokens: toks,
      caption:
        k === 0 ? (
          <>从 <C>"{a}"</C> 出发，沿回溯路径从左上到右下依次执行操作。</>
        ) : (
          <>
            第 {k} 步（格子 ({o.cell[0]},{o.cell[1]})）：<b>{OP_NAME[o.op]}</b>{' '}
            {o.op === 'keep' ? <C>'{o.a}'</C> : o.op === 'sub' ? <C>'{o.a}' → '{o.b}'</C> : o.op === 'del' ? <C>'{o.a}'</C> : <C>'{o.b}'</C>}
            。已用 {edits} 次编辑。{k === ops.length ? <b> 完成！总编辑次数 = 编辑距离 = {edits}。</b> : null}
          </>
        ),
    })
  }
  return fr
}

export function EditDistanceViz() {
  const [k, setK] = useState(0)
  const [mode, setMode] = useState<'fill' | 'replay'>('fill')
  const [a, b] = ED_PRESETS[k]
  const full = useMemo(() => editTable(a.split(''), b.split('')), [a, b])
  const ops = useMemo(() => backtrackOps(a.split(''), b.split(''), full), [a, b, full])
  const fillFrames = useMemo(() => buildEdFill(a, b), [a, b])
  const replayFrames = useMemo(() => buildReplay(a, ops), [a, ops])
  const fs = useStepper(fillFrames, { intervalMs: 450 })
  const rs = useStepper(replayFrames, { intervalMs: 1100 })
  const pathCells = useMemo(() => new Set(['0,0', ...ops.map((o) => `${o.cell[0]},${o.cell[1]}`)]), [ops])

  let grid: Cell[][]
  let arrows: DpArrow[] = []
  let ring: (r: number, c: number) => VizTone | undefined
  let fill: (r: number, c: number) => VizTone | undefined
  let caption: React.ReactNode
  if (mode === 'fill') {
    const f = fs.frame
    grid = f.dp
    arrows = f.arrows
    const isCur = (r: number, c: number) => !!f.cur && f.cur[0] === r && f.cur[1] === c
    ring = (r, c) => (isCur(r, c) ? 'accent' : undefined)
    fill = (r, c) => (isCur(r, c) ? 'accent' : undefined)
    caption = f.caption
  } else {
    const f = rs.frame
    grid = full
    const o = f.k > 0 ? ops[f.k - 1] : undefined
    if (o) arrows = [{ from: o.from, to: o.cell, tone: o.op === 'keep' ? 'accent' : OP_TONE[o.op], label: OP_NAME[o.op] }]
    const doneCells = new Set(['0,0', ...ops.slice(0, f.k).map((x) => `${x.cell[0]},${x.cell[1]}`)])
    ring = (r, c) => (pathCells.has(`${r},${c}`) ? (doneCells.has(`${r},${c}`) ? 'success' : 'muted') : undefined)
    fill = (r, c) => (o && o.cell[0] === r && o.cell[1] === c ? (o.op === 'keep' ? 'accent' : OP_TONE[o.op]) : undefined)
    caption = f.caption
  }

  return (
    <VizFrame
      title="编辑距离：对角 = 替换/保留，向下 = 删除，向右 = 插入"
      caption={caption}
      legend={
        <Legend
          items={[
            { label: '替换', tone: 'warning', variant: 'line' },
            { label: '删除', tone: 'danger', variant: 'line' },
            { label: '插入', tone: 'success', variant: 'line' },
            { label: '落选的来源', tone: 'muted', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          {mode === 'fill' ? <StepControls stepper={fs} /> : <StepControls stepper={rs} />}
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'fill', label: '填表' },
              { value: 'replay', label: '操作回放' },
            ]}
          />
          <Segmented size="sm" value={k} onChange={setK} options={ED_PRESETS.map((p, i) => ({ value: i, label: `${p[0]}→${p[1]}` }))} />
        </>
      }
    >
      <div className="space-y-4">
        {mode === 'replay' && (
          <div className="rounded-xl border border-border bg-surface p-3">
            <Sub>
              word1 = <C>{a}</C> → word2 = <C>{b}</C>（共 {ops.filter((o) => o.op !== 'keep').length} 次编辑）
            </Sub>
            <TokenRow tokens={rs.frame.tokens} />
          </div>
        )}
        <DpGrid data={grid} rowLabels={['∅', ...a.split('')]} colLabels={['∅', ...b.split('')]} cellSize={40} arrows={arrows} ring={ring} fill={fill} />
        {mode === 'fill' && (
          <CodeTrace
            code={ED_CODE}
            active={fs.frame.line}
            vars={fs.frame.cur ? { i: fs.frame.cur[0], j: fs.frame.cur[1], [`dp[${fs.frame.cur[0]}][${fs.frame.cur[1]}]`]: fs.frame.dp[fs.frame.cur[0]][fs.frame.cur[1]] ?? '—' } : undefined}
          />
        )}
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* WER / CER                                                            */
/* ------------------------------------------------------------------ */

const WER_PRESETS = [
  { label: '英文 WER', ref: 'the cat sat on the mat', hyp: 'the cat sit on mat', unit: 'word' as const },
  { label: '中文 CER', ref: '今天天气很好', hyp: '今天天汽真很好', unit: 'char' as const },
  { label: 'WER > 100%', ref: 'hello world', hyp: 'oh hello there big world', unit: 'word' as const },
]

export function WerViz() {
  const [k, setK] = useState(0)
  const [showTable, setShowTable] = useState(false)
  const p = WER_PRESETS[k]
  const split = (s: string) => (p.unit === 'word' ? s.split(' ') : s.split(''))
  const ref = split(p.ref)
  const hyp = split(p.hyp)
  const dp = editTable(ref, hyp)
  const ops = backtrackOps(ref, hyp, dp)
  const cnt = { S: 0, D: 0, I: 0, C: 0 }
  ops.forEach((o) => (o.op === 'sub' ? cnt.S++ : o.op === 'del' ? cnt.D++ : o.op === 'ins' ? cnt.I++ : cnt.C++))
  const N = ref.length
  const wer = (cnt.S + cnt.D + cnt.I) / N
  const tag: Record<Op['op'], string> = { keep: 'C', sub: 'S', del: 'D', ins: 'I' }
  const path = new Set(['0,0', ...ops.map((o) => `${o.cell[0]},${o.cell[1]}`)])
  return (
    <VizFrame
      title="WER = (S + D + I) / N：编辑距离在语音 / OCR 评测里的直接应用"
      caption={
        <>
          {p.unit === 'word' ? 'WER' : 'CER'} = (S {cnt.S} + D {cnt.D} + I {cnt.I}) / N {N} = <b className="font-mono">{(wer * 100).toFixed(1)}%</b>
          {wer > 1 ? <>（插入不受 N 限制，所以可以超过 100%）</> : null}。分母是<b>参考文本</b>的长度，不是识别结果的长度。
        </>
      }
      legend={
        <Legend
          items={[
            { label: 'C 正确', tone: 'muted', variant: 'soft' },
            { label: 'S 替换', tone: 'warning', variant: 'soft' },
            { label: 'D 删除（漏识别）', tone: 'danger', variant: 'soft' },
            { label: 'I 插入（多识别）', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Segmented size="sm" value={k} onChange={setK} options={WER_PRESETS.map((x, i) => ({ value: i, label: x.label }))} />
          <Segmented
            size="sm"
            value={showTable ? 'on' : 'off'}
            onChange={(v) => setShowTable(v === 'on')}
            options={[
              { value: 'off', label: '只看对齐' },
              { value: 'on', label: '显示 DP 表' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          <Stat label="N（参考长度）" value={N} />
          <Stat label="S" value={cnt.S} tone="warning" />
          <Stat label="D" value={cnt.D} tone="danger" />
          <Stat label="I" value={cnt.I} tone="success" />
          <Stat label={p.unit === 'word' ? 'WER' : 'CER'} value={`${(wer * 100).toFixed(1)}%`} tone="accent" />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {ops.map((o, i) => {
            const tone = OP_TONE[o.op]
            return (
              <div
                key={i}
                className="flex min-w-[44px] flex-col items-center rounded-lg border px-1.5 py-1 font-mono text-[12.5px]"
                style={{ borderColor: o.op === 'keep' ? 'var(--border)' : toneColor(tone), background: o.op === 'keep' ? 'var(--surface)' : toneSoft(tone, 14) }}
              >
                <span className="text-fg">{o.a || '—'}</span>
                <span className="text-muted">{o.b || '—'}</span>
                <span className="mt-0.5 text-[10.5px] font-semibold" style={{ color: toneColor(tone) }}>
                  {tag[o.op]}
                </span>
              </div>
            )
          })}
        </div>
        <div className="text-xs text-muted">每列上方为参考（ref），下方为识别结果（hyp）。</div>
        {showTable && (
          <DpGrid
            data={dp}
            rowLabels={['∅', ...ref]}
            colLabels={['∅', ...hyp]}
            corner="ref\hyp"
            cellSize={40}
            ring={(r, c) => (path.has(`${r},${c}`) ? 'success' : undefined)}
          />
        )}
      </div>
    </VizFrame>
  )
}
