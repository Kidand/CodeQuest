/**
 * online softmax 逐块演示：维护 running max m、running sum ℓ 与未归一化输出 acc；
 * 新块的最大值更大时，用 α = exp(m_old − m_new) 把旧统计量整体缩放。最后 acc / ℓ 与一次性 softmax 完全相同。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, ProbBars, StepControls, VizFrame, useStepper, type ArrayRange } from '@/viz'
import { Segmented } from '@/components/ui'

const X = [1.2, 0.3, 2.0, -0.5, 3.1, 0.8, 1.7, 4.0, -1.0, 2.6]
const V = [0.5, -1.0, 2.0, 1.5, -0.5, 1.0, 0.0, 3.0, -2.0, 1.0]
const N = X.length

interface Frame {
  done: number // 已处理元素数
  blk?: [number, number]
  m: number
  l: number
  acc: number
  alpha?: number
  mBlk?: number
  line: number[]
  text: React.ReactNode
  final?: boolean
}

const CODE = `m, l, acc = -inf, 0.0, 0.0
for blk in blocks(x, v):                 # 每次只看一块
    m_new = max(m, blk.x.max())
    alpha = exp(m - m_new)               # 旧统计量的缩放因子
    p = exp(blk.x - m_new)
    l = alpha * l + p.sum()
    acc = alpha * acc + p @ blk.v        # 未归一化的输出
    m = m_new
out = acc / l                            # 最后统一归一化`

const f3 = (x: number) => (Number.isFinite(x) ? x.toFixed(3) : '−∞')

function build(bs: number): Frame[] {
  const fr: Frame[] = [{ done: 0, m: -Infinity, l: 0, acc: 0, line: [1], text: <>初始化：m = −∞，ℓ = 0，acc = 0。块大小 {bs}，共 {Math.ceil(N / bs)} 块。</> }]
  let m = -Infinity
  let l = 0
  let acc = 0
  for (let s = 0; s < N; s += bs) {
    const e = Math.min(N, s + bs)
    const xs = X.slice(s, e)
    const mBlk = Math.max(...xs)
    const mNew = Math.max(m, mBlk)
    const alpha = Math.exp(m - mNew)
    const p = xs.map((x) => Math.exp(x - mNew))
    const lNew = alpha * l + p.reduce((a, b) => a + b, 0)
    const accNew = alpha * acc + p.reduce((a, pi, k) => a + pi * V[s + k], 0)
    fr.push({
      done: s,
      blk: [s, e - 1],
      m,
      l,
      acc,
      mBlk,
      alpha,
      line: [3, 4],
      text:
        mNew > m && Number.isFinite(m) ? (
          <>
            块 [{s}, {e - 1}] 的最大值 {mBlk.toFixed(2)} 超过了旧的 m = {m.toFixed(2)}：m 更新为 {mNew.toFixed(2)}，旧的 ℓ 与 acc 都要乘以 α = e^({m.toFixed(2)}−{mNew.toFixed(2)}) ={' '}
            <b className="font-mono">{alpha.toFixed(4)}</b>——它们是以旧 m 为基准算的指数，要换成新基准。
          </>
        ) : (
          <>
            块 [{s}, {e - 1}] 的最大值 {mBlk.toFixed(2)}，m_new = {mNew.toFixed(2)}，α = <b className="font-mono">{alpha.toFixed(4)}</b>
            {Number.isFinite(m) ? '（最大值没变，α = 1，无需缩放）' : '（第一块：旧统计量为 0，α 乘 0 无影响）'}。
          </>
        ),
    })
    m = mNew
    l = lNew
    acc = accNew
    fr.push({ done: e, blk: [s, e - 1], m, l, acc, alpha, line: [5, 6, 7, 8], text: <>ℓ ← α·ℓ + Σe^(x−m) = {l.toFixed(4)}；acc ← α·acc + Σe^(x−m)·v = {acc.toFixed(4)}。这一块读完就可以丢掉，SRAM 里只保留 (m, ℓ, acc)。</> })
  }
  // 一次性 softmax 对照
  const mx = Math.max(...X)
  const ex = X.map((x) => Math.exp(x - mx))
  const Z = ex.reduce((a, b) => a + b, 0)
  const exact = ex.reduce((a, e, j) => a + (e / Z) * V[j], 0)
  fr.push({
    done: N,
    m,
    l,
    acc,
    line: [9],
    final: true,
    text: (
      <>
        out = acc / ℓ = <b className="font-mono">{(acc / l).toFixed(6)}</b>；一次性 softmax 的结果 = <b className="font-mono">{exact.toFixed(6)}</b>，差值{' '}
        {Math.abs(acc / l - exact).toExponential(1)}。只扫描一遍、不保存整行分数，结果完全精确。
      </>
    ),
  })
  return fr
}

export function OnlineSoftmaxViz() {
  const [bs, setBs] = useState(3)
  const frames = useMemo(() => build(bs), [bs])
  const st = useStepper(frames, { intervalMs: 1500 })
  const f = st.frame
  const ranges: ArrayRange[] = []
  if (f.done > 0) ranges.push({ from: 0, to: f.done - 1, tone: 'success', label: '已处理' })
  if (f.blk && f.done <= f.blk[0]) ranges.push({ from: f.blk[0], to: f.blk[1], tone: 'accent', label: '当前块' })
  // 以当前 m、ℓ 为基准的「部分 softmax」：已处理元素的相对权重
  const partial = X.map((x, j) => (j < f.done && f.l > 0 ? Math.exp(x - f.m) / f.l : 0))

  return (
    <VizFrame
      title="online softmax：一遍扫描、逐块修正"
      caption={f.text}
      legend={<Legend items={[{ label: '已处理（可丢弃）', tone: 'success', variant: 'soft' }, { label: '当前块', tone: 'accent', variant: 'soft' }, { label: '尚未读取', tone: 'muted', variant: 'soft' }]} />}
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented size="sm" value={bs} onChange={setBs} options={[2, 3, 4, 5].map((b) => ({ value: b, label: `块大小 ${b}` }))} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[1.25fr_1fr]">
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">分数 x（一行 q·k）</div>
          <ArrayView values={X} cellSize={40} format={(v) => Number(v).toFixed(1)} ranges={ranges} dim={(i) => i >= f.done && !(f.blk && i >= f.blk[0] && i <= f.blk[1])} highlights={f.blk ? Object.fromEntries(Array.from({ length: f.blk[1] - f.blk[0] + 1 }, (_, k) => [f.blk![0] + k, 'accent' as const])) : {}} />
          <div className="text-xs font-medium text-muted">value v（标量便于观察）</div>
          <ArrayView values={V} cellSize={40} format={(v) => Number(v).toFixed(1)} showIndex={false} dim={(i) => i >= f.done} />
          <div className="grid grid-cols-4 gap-2 font-mono text-[12px]">
            {[
              ['m', f3(f.m)],
              ['ℓ', f.l.toFixed(3)],
              ['acc', f.acc.toFixed(3)],
              ['α', f.alpha != null ? f.alpha.toFixed(3) : '—'],
            ].map(([k, v]) => (
              <div key={k} className="rounded-lg border border-border bg-surface px-2 py-1.5">
                <div className="text-[11px] text-muted">{k}</div>
                <div className="font-semibold text-fg">{v}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="min-w-0 space-y-3">
          <CodeTrace code={CODE} active={f.line} maxHeight={260} />
          <div>
            <div className="mb-1 text-xs font-medium text-muted">已处理部分的权重 e^(x−m)/ℓ（新块到来后旧权重会被整体缩小）</div>
            <ProbBars items={X.map((_, j) => ({ label: `x${j}`, p: partial[j], dim: j >= f.done, tone: 'accent' }))} max={1} format={(p) => (p ? p.toFixed(3) : '—')} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
