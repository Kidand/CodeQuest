/**
 * 背包问题可视化：
 *  - Knapsack2DViz：0/1 背包二维表，dp[i][j] 依赖上一行的 j 与 j-w
 *  - RollingViz：一维滚动数组，倒序（0/1）vs 正序（完全背包）同步对比
 *  - ComboPermViz：零钱兑换 II（组合）vs 组合总和 IV（排列）——循环顺序决定计数对象
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { Segmented } from '@/components/ui'
import { DpGrid, type Cell, type DpArrow } from './DpGrid'
import { C, Stat, Sub } from './common'

/* ------------------------------------------------------------------ */
/* 0/1 背包二维表                                                       */
/* ------------------------------------------------------------------ */

const W = [1, 3, 4, 5]
const V = [1, 4, 5, 7]
const CAP = 7
const CODE_2D = `def knapsack(w, v, C):
    n = len(w)
    dp = [[0] * (C + 1) for _ in range(n + 1)]   # dp[i][j]: 前 i 个物品、容量 j
    for i in range(1, n + 1):
        for j in range(C + 1):
            dp[i][j] = dp[i-1][j]                 # 不选第 i 个
            if j >= w[i-1]:
                dp[i][j] = max(dp[i][j],
                               dp[i-1][j - w[i-1]] + v[i-1])   # 选第 i 个
    return dp[n][C]`

interface KFrame {
  dp: (number | null)[][]
  cur?: [number, number]
  arrows: DpArrow[]
  line: number | number[]
  caption: React.ReactNode
  chosen?: Set<number>
  path?: Set<string>
}

function build2D(): KFrame[] {
  const n = W.length
  const dp: (number | null)[][] = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: CAP + 1 }, () => (i === 0 ? 0 : null)))
  const fr: KFrame[] = [{ dp: dp.map((r) => [...r]), arrows: [], line: 3, caption: <>第 0 行：一个物品都不选，价值全为 0。</> }]
  for (let i = 1; i <= n; i++)
    for (let j = 0; j <= CAP; j++) {
      const w = W[i - 1]
      const v = V[i - 1]
      const skip = dp[i - 1][j]!
      if (j < w) {
        dp[i][j] = skip
        fr.push({
          dp: dp.map((r) => [...r]),
          cur: [i, j],
          arrows: [{ from: [i - 1, j], to: [i, j], tone: 'success' }],
          line: 6,
          caption: <>容量 {j} 装不下物品 {i}（w={w}）：只能不选，<C>dp[{i}][{j}] = dp[{i - 1}][{j}] = {skip}</C>。</>,
        })
      } else {
        const take = dp[i - 1][j - w]! + v
        dp[i][j] = Math.max(skip, take)
        const t = take > skip
        fr.push({
          dp: dp.map((r) => [...r]),
          cur: [i, j],
          arrows: [
            { from: [i - 1, j], to: [i, j], tone: t ? 'muted' : 'success', dashed: t },
            { from: [i - 1, j - w], to: [i, j], tone: t ? 'success' : 'muted', dashed: !t, label: `+${v}` },
          ],
          line: [8, 9],
          caption: (
            <>
              物品 {i}（w={w}, v={v}）：不选 <C>{skip}</C>；选 <C>dp[{i - 1}][{j - w}] + {v} = {take}</C> → <C>{dp[i][j]}</C>。
              注意“选”的来源在<b>上一行</b>：每个物品只能用一次。
            </>
          ),
        })
      }
    }
  const chosen = new Set<number>()
  const path = new Set<string>()
  let j = CAP
  for (let i = n; i >= 1; i--) {
    path.add(`${i},${j}`)
    if (dp[i][j] !== dp[i - 1][j]) {
      chosen.add(i)
      j -= W[i - 1]
    }
  }
  path.add(`0,${j}`)
  fr.push({
    dp: dp.map((r) => [...r]),
    arrows: [],
    line: 10,
    chosen,
    path,
    caption: (
      <>
        最大价值 <C>dp[{n}][{CAP}] = {dp[n][CAP]}</C>。回溯：<C>dp[i][j] ≠ dp[i-1][j]</C> 说明选了物品 i，容量减去 w。选中物品{' '}
        {[...chosen].sort().join('、')}（重量 {[...chosen].map((i) => W[i - 1]).join('+')}，价值 {[...chosen].map((i) => V[i - 1]).join('+')}）。
      </>
    ),
  })
  return fr
}

export function Knapsack2DViz() {
  const frames = useMemo(build2D, [])
  const st = useStepper(frames, { intervalMs: 700 })
  const f = st.frame
  const isCur = (r: number, c: number) => !!f.cur && f.cur[0] === r && f.cur[1] === c
  return (
    <VizFrame
      title="0/1 背包：dp[i][j] 只从上一行转移，所以每个物品最多用一次"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前格', tone: 'accent', variant: 'soft' },
            { label: '取用的来源', tone: 'success', variant: 'line' },
            { label: '回溯路径', tone: 'success', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {W.map((w, i) => (
            <Stat key={i} label={`物品 ${i + 1}`} value={`w=${w}, v=${V[i]}`} tone={f.chosen?.has(i + 1) ? 'success' : f.cur?.[0] === i + 1 ? 'accent' : 'fg'} />
          ))}
          <Stat label="容量 C" value={CAP} />
        </div>
        <DpGrid
          data={f.dp as Cell[][]}
          rowLabels={['i=0', ...W.map((_, i) => `i=${i + 1}`)]}
          colLabels={Array.from({ length: CAP + 1 }, (_, j) => `j=${j}`)}
          cellSize={46}
          arrows={f.arrows}
          ring={(r, c) => (isCur(r, c) ? 'accent' : f.path?.has(`${r},${c}`) ? 'success' : undefined)}
          fill={(r, c) => (isCur(r, c) ? 'accent' : f.path?.has(`${r},${c}`) && f.chosen?.has(r) ? 'success' : undefined)}
        />
        <CodeTrace code={CODE_2D} active={f.line} vars={f.cur ? { i: f.cur[0], j: f.cur[1], [`dp[${f.cur[0]}][${f.cur[1]}]`]: f.dp[f.cur[0]][f.cur[1]] ?? '—' } : undefined} />
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 一维滚动：倒序 vs 正序                                               */
/* ------------------------------------------------------------------ */

const RW = [2, 3]
const RV = [3, 4]
const RC = 7

interface RowState {
  dp: number[]
  j: number
  src: number
  srcFresh: boolean // 读到的 dp[j-w] 是否本轮已被更新
  updated: Set<number>
}
interface RFrame {
  item: number // -1 = 初始, RW.length = 结束
  back?: RowState
  fwd?: RowState
  caption: React.ReactNode
}

function buildRolling(): RFrame[] {
  const back = Array(RC + 1).fill(0)
  const fwd = Array(RC + 1).fill(0)
  const fr: RFrame[] = [{ item: -1, caption: <>两行 dp 都从全 0 开始，依次处理物品 1（w=2, v=3）、物品 2（w=3, v=4），容量 7。上面倒序、下面正序，<b>同步</b>走。</> }]
  for (let k = 0; k < RW.length; k++) {
    const w = RW[k]
    const v = RV[k]
    const ub = new Set<number>()
    const uf = new Set<number>()
    for (let s = 0; s <= RC - w; s++) {
      const jb = RC - s // 倒序：C → w
      const jf = w + s // 正序：w → C
      const bFresh = ub.has(jb - w)
      const fFresh = uf.has(jf - w)
      back[jb] = Math.max(back[jb], back[jb - w] + v)
      fwd[jf] = Math.max(fwd[jf], fwd[jf - w] + v)
      ub.add(jb)
      uf.add(jf)
      fr.push({
        item: k,
        back: { dp: [...back], j: jb, src: jb - w, srcFresh: bFresh, updated: new Set(ub) },
        fwd: { dp: [...fwd], j: jf, src: jf - w, srcFresh: fFresh, updated: new Set(uf) },
        caption: fFresh ? (
          <>
            正序读到的 <C>dp[{jf - w}]</C> <b className="text-danger">已经在本轮被物品 {k + 1} 更新过</b>——相当于物品 {k + 1} 被再放一次！倒序读到的{' '}
            <C>dp[{jb - w}]</C> 仍是上一轮的旧值。
          </>
        ) : (
          <>
            物品 {k + 1}：倒序更新 <C>dp[{jb}]</C>（读 <C>dp[{jb - w}]</C>），正序更新 <C>dp[{jf}]</C>（读 <C>dp[{jf - w}]</C>）。两边读到的都是上一轮的旧值。
          </>
        ),
      })
    }
  }
  fr.push({
    item: RW.length,
    back: { dp: [...back], j: -1, src: -1, srcFresh: false, updated: new Set() },
    fwd: { dp: [...fwd], j: -1, src: -1, srcFresh: false, updated: new Set() },
    caption: (
      <>
        结果：倒序 <C>dp[7] = {back[RC]}</C>（两件各用一次：3+4=7）；正序 <C>dp[7] = {fwd[RC]}</C>（物品 1 用 2 次 + 物品 2 用 1 次：3+3+4=10）——正序算的是<b>完全背包</b>。
      </>
    ),
  })
  return fr
}

function RollRow({ title, s, item }: { title: React.ReactNode; s?: RowState; item: number }) {
  const dp: Cell[] = s ? s.dp : Array(RC + 1).fill(0)
  const arrows: DpArrow[] =
    s && s.j >= 0 ? [{ from: [0, s.src], to: [0, s.j], tone: s.srcFresh ? 'danger' : 'success', label: `+${RV[item]}`, arc: true }] : []
  return (
    <div>
      <Sub>{title}</Sub>
      <DpGrid
        data={[dp]}
        colLabels={Array.from({ length: RC + 1 }, (_, j) => `${j}`)}
        cellSize={46}
        arrows={arrows}
        ring={(_r, c) => (s && c === s.j ? 'accent' : s && c === s.src && s.j >= 0 ? (s.srcFresh ? 'danger' : 'success') : undefined)}
        fill={(_r, c) => (s && c === s.src && s.srcFresh ? 'danger' : s?.updated.has(c) ? 'warning' : undefined)}
      />
    </div>
  )
}

export function RollingViz() {
  const frames = useMemo(buildRolling, [])
  const st = useStepper(frames, { intervalMs: 1300 })
  const f = st.frame
  const it = Math.max(0, Math.min(RW.length - 1, f.item))
  return (
    <VizFrame
      title="一维滚动：倒序读旧值 = 0/1 背包；正序读新值 = 完全背包"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '本轮已更新的格子', tone: 'warning', variant: 'soft' },
            { label: '读到旧值（正确）', tone: 'success', variant: 'line' },
            { label: '读到本轮新值（物品被重复使用）', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="space-y-5">
        <div className="flex flex-wrap gap-2">
          {RW.map((w, k) => (
            <Stat key={k} label={`物品 ${k + 1}`} value={`w=${w}, v=${RV[k]}`} tone={f.item === k ? 'accent' : 'fg'} />
          ))}
        </div>
        <RollRow title={<>倒序 <C>for j in range(C, w - 1, -1)</C> → 0/1 背包</>} s={f.back} item={it} />
        <RollRow title={<>正序 <C>for j in range(w, C + 1)</C> → 完全背包</>} s={f.fwd} item={it} />
        <CodeTrace
          code={`for w, v in zip(weights, values):
    for j in range(C, w - 1, -1):    # 0/1：倒序，dp[j-w] 还是上一轮的
        dp[j] = max(dp[j], dp[j - w] + v)
# 完全背包：for j in range(w, C + 1)，dp[j-w] 可能已含本物品`}
          active={f.item >= 0 && f.item < RW.length ? [2, 3] : undefined}
        />
      </div>
    </VizFrame>
  )
}

/* ------------------------------------------------------------------ */
/* 组合数 vs 排列数                                                     */
/* ------------------------------------------------------------------ */

const COINS = [1, 2, 3]
const AMT = 4
type Order = 'combo' | 'perm'

interface CFrame {
  dp: number[]
  seqs: string[][]
  j: number
  c: number
  caption: React.ReactNode
  line: number | number[]
}

function buildCP(order: Order): CFrame[] {
  const dp = Array(AMT + 1).fill(0)
  dp[0] = 1
  const seqs: string[][] = Array.from({ length: AMT + 1 }, (_, j) => (j === 0 ? [''] : []))
  const fr: CFrame[] = [{ dp: [...dp], seqs: seqs.map((s) => [...s]), j: -1, c: -1, line: 2, caption: <><C>dp[0] = 1</C>：凑出 0 只有“什么都不选”一种方式。</> }]
  const step = (j: number, c: number) => {
    dp[j] += dp[j - c]
    seqs[j] = [...seqs[j], ...seqs[j - c].map((s) => (s ? `${s}+${c}` : `${c}`))]
    fr.push({
      dp: [...dp],
      seqs: seqs.map((s) => [...s]),
      j,
      c,
      line: order === 'combo' ? 5 : 5,
      caption: (
        <>
          <C>dp[{j}] += dp[{j - c}]</C>：在凑出 {j - c} 的每种方式末尾放一枚 <C>{c}</C>，dp[{j}] 变为 {dp[j]}。
        </>
      ),
    })
  }
  if (order === 'combo') for (const c of COINS) for (let j = c; j <= AMT; j++) step(j, c)
  else for (let j = 1; j <= AMT; j++) for (const c of COINS) if (c <= j) step(j, c)
  fr.push({
    dp: [...dp],
    seqs: seqs.map((s) => [...s]),
    j: AMT,
    c: -1,
    line: 6,
    caption:
      order === 'combo' ? (
        <>
          共 <C>{dp[AMT]}</C> 种：每个方案里硬币都是按 1 → 2 → 3 的顺序放入的（非降），所以 <C>1+3</C> 与 <C>3+1</C> 只算一次——这是<b>组合数</b>（LC 518）。
        </>
      ) : (
        <>
          共 <C>{dp[AMT]}</C> 种：对每个金额都枚举“最后一枚是谁”，<C>1+3</C> 和 <C>3+1</C> 是不同的序列——这是<b>排列数</b>（LC 377）。
        </>
      ),
  })
  return fr
}

export function ComboPermViz() {
  const [order, setOrder] = useState<Order>('combo')
  const frames = useMemo(() => buildCP(order), [order])
  const st = useStepper(frames, { intervalMs: 1200 })
  const f = st.frame
  const code =
    order === 'combo'
      ? `dp = [0] * (amount + 1)
dp[0] = 1
for c in coins:                      # 外层：物品
    for j in range(c, amount + 1):   # 内层：容量（正序，可重复用）
        dp[j] += dp[j - c]
return dp[amount]                    # 组合数（LC 518）`
      : `dp = [0] * (amount + 1)
dp[0] = 1
for j in range(1, amount + 1):       # 外层：容量
    for c in coins:                  # 内层：物品
        if c <= j: dp[j] += dp[j - c]
return dp[amount]                    # 排列数（LC 377）`
  const arrows: DpArrow[] = f.c > 0 ? [{ from: [0, f.j - f.c], to: [0, f.j], tone: 'success', label: `+${f.c}`, arc: true }] : []
  return (
    <VizFrame
      title="循环顺序决定计数对象：外物品 = 组合，外容量 = 排列"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前更新 dp[j]', tone: 'accent', variant: 'soft' },
            { label: '来源 dp[j-c]', tone: 'success', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={order}
            onChange={setOrder}
            options={[
              { value: 'combo', label: '外物品内容量（518 组合）' },
              { value: 'perm', label: '外容量内物品（377 排列）' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <div className="text-[13px] text-muted">
          硬币 <C>[1, 2, 3]</C>（每种无限枚），凑出金额 <C>{AMT}</C>。
        </div>
        <DpGrid
          data={[f.dp]}
          colLabels={Array.from({ length: AMT + 1 }, (_, j) => `j=${j}`)}
          cellSize={52}
          arrows={arrows}
          ring={(_r, c) => (c === f.j && f.c > 0 ? 'accent' : undefined)}
          fill={(_r, c) => (c === f.j && f.c > 0 ? 'accent' : undefined)}
        />
        <div className="grid gap-2 @lg:grid-cols-5">
          {f.seqs.map((ss, j) => (
            <div key={j} className={`rounded-lg border p-2 ${j === f.j && f.c > 0 ? 'border-accent' : 'border-border'} bg-surface`}>
              <div className="mb-1 font-mono text-[11px] text-muted">
                j={j}（{ss.length} 种）
              </div>
              <div className="flex flex-wrap gap-1">
                {ss.map((s, k) => (
                  <span key={k} className="rounded bg-surface-2 px-1 font-mono text-[11.5px] text-fg">
                    {s || '∅'}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
        <CodeTrace code={code} active={f.line} />
      </div>
    </VizFrame>
  )
}
