/**
 * 投机采样的单步数学：
 *  - AcceptRuleViz：接受概率 min(1, p/q) 与「被接受的质量」min(p, q)
 *  - ProofViz：输出分布 = min(p,q)（接受部分）+ (1-β)·r（拒绝后重采样部分）= p；
 *    对比两种错误做法，并用固定种子的蒙特卡洛模拟验证。
 */
import { useMemo, useState } from 'react'
import { Dices } from 'lucide-react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Button, Segmented, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { C, PQBars, Stat, pct, sampleIdx } from './common'

const LABELS = ['苹果', '香蕉', '面包', '米饭', '面条', '。']
const P = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
const BAD = [0.05, 0.1, 0.4, 0.05, 0.3, 0.1]
const qOf = (lam: number) => P.map((p, i) => (1 - lam) * p + lam * BAD[i])
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0)

export function AcceptRuleViz() {
  const [lam, setLam] = useState(0.6)
  const [sel, setSel] = useState(2)
  const q = useMemo(() => qOf(lam), [lam])
  const acc = P.map((p, i) => Math.min(1, p / q[i]))
  const mins = P.map((p, i) => Math.min(p, q[i]))
  const beta = sum(mins)
  const over = q[sel] > P[sel]

  return (
    <VizFrame
      title="接受概率 min(1, p/q)"
      caption={
        <>
          草稿以概率 q({LABELS[sel]}) = {q[sel].toFixed(3)} 提出「{LABELS[sel]}」，被接受的概率为 <C>min(1, p/q) = {acc[sel].toFixed(3)}</C>，
          所以「提出且被接受」的概率质量是 <C>q·min(1,p/q) = min(p,q) = {mins[sel].toFixed(3)}</C>。
          {over ? ' 草稿高估了它（q > p），多出来的那部分要按比例砍掉。' : ' 草稿没有高估它（q ≤ p），提出就一定接受；但还差 p − q 的质量，要靠拒绝后的重采样补上。'}
        </>
      }
      legend={<Legend items={[{ label: 'target p', tone: 'accent' }, { label: '草稿 q', tone: 'info' }, { label: '点击某一行查看该 token', tone: 'gold', variant: 'line' }]} />}
      controls={
        <Slider
          className="min-w-[220px] flex-1"
          label="草稿偏离程度 λ（q = (1−λ)·p + λ·b）"
          value={lam}
          min={0}
          max={1}
          step={0.05}
          onChange={setLam}
          format={(v) => v.toFixed(2)}
        />
      }
    >
      <div className="grid items-start gap-4 @2xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <PQBars labels={LABELS} p={P} q={q} highlight={sel} onRowClick={setSel} extra={(i) => `接受 ${pct(acc[i], 0)}`} />
        <div className="grid grid-cols-2 gap-2">
          <Stat label="接受率 β" value={beta.toFixed(3)} tone="success" sub="Σ min(p,q)" />
          <Stat label="拒绝率 1 − β" value={(1 - beta).toFixed(3)} tone="danger" sub="= TV(p, q)" />
          <Stat label={`q(${LABELS[sel]})`} value={q[sel].toFixed(3)} tone="info" />
          <Stat label={`p(${LABELS[sel]})`} value={P[sel].toFixed(3)} tone="accent" />
          <div className="col-span-2 text-[12px] leading-relaxed text-muted">
            β 只取决于两个分布的<b>重叠面积</b>：λ = 0 时 q = p，β = 1，永远接受；λ 越大重叠越少，拒绝越多——但无论 λ 多大，输出分布都仍然是 p（下一节证明）。
          </div>
        </div>
      </div>
    </VizFrame>
  )
}

type Method = 'residual' | 'p' | 'q'
const N_SIM = 20000

function theory(q: number[], method: Method) {
  const mins = P.map((p, i) => Math.min(p, q[i]))
  const beta = sum(mins)
  if (method === 'q') return { a: mins, b: q.map((v, i) => v - mins[i]) }
  if (method === 'p') return { a: mins, b: P.map((p) => (1 - beta) * p) }
  return { a: mins, b: P.map((p, i) => Math.max(0, p - q[i])) }
}

function simulate(q: number[], method: Method, seed: number) {
  const r = rng(seed)
  const res = P.map((p, i) => Math.max(0, p - q[i]))
  const cnt = Array(P.length).fill(0)
  for (let k = 0; k < N_SIM; k++) {
    const x = sampleIdx(q, r())
    if (method === 'q') {
      cnt[x]++
      continue
    }
    if (r() < Math.min(1, P[x] / q[x])) cnt[x]++
    else cnt[sampleIdx(method === 'p' ? P : res, r())]++
  }
  return cnt.map((c) => c / N_SIM)
}

export function ProofViz() {
  const [lam, setLam] = useState(0.6)
  const [method, setMethod] = useState<Method>('residual')
  const [seed, setSeed] = useState(1)
  const q = useMemo(() => qOf(lam), [lam])
  const th = useMemo(() => theory(q, method), [q, method])
  const emp = useMemo(() => simulate(q, method, seed), [q, method, seed])
  const out = th.a.map((a, i) => a + th.b[i])
  const m = Math.max(...out, ...P) * 1.12
  const tvTheory = sum(out.map((v, i) => Math.abs(v - P[i]))) / 2
  const tvEmp = sum(emp.map((v, i) => Math.abs(v - P[i]))) / 2
  const beta = sum(th.a)
  const bTone = method === 'residual' ? 'warning' : 'danger'

  return (
    <VizFrame
      title="输出分布 = 接受部分 + 重采样部分"
      caption={
        method === 'residual' ? (
          <>
            绿色 = 草稿提出且被接受的质量 <C>min(p,q)</C>；橙色 = 拒绝（总概率 1−β = {(1 - beta).toFixed(3)}）后从残差分布补上的 <C>(1−β)·r(x) = max(0, p−q)</C>。
            两段之和<b>逐个 token 恰好等于 p</b>（竖线）。{N_SIM.toLocaleString()} 次模拟的经验分布（◆）与 p 的总变差距离只有 {tvEmp.toFixed(4)}。
          </>
        ) : method === 'p' ? (
          <>
            错误做法：拒绝后直接从 p 重采样。接受部分 min(p,q) 已经偏向「q 也喜欢的 token」，再按 p 补 (1−β)·p 会<b>重复计入</b>这些 token、
            低估 p &gt; q 的 token。理论偏差（TV）= {tvTheory.toFixed(3)}，模拟 = {tvEmp.toFixed(3)}。
          </>
        ) : (
          <>
            不做验证、直接用草稿：输出就是 q，与 target 的偏差 TV = 1 − β = {tvTheory.toFixed(3)}。红色是 q 超出 p 的那部分「未经检验」的质量——
            拒绝采样做的事情正是把它砍掉，再把缺口按 max(0, p−q) 补回来。
          </>
        )
      }
      legend={
        <Legend
          items={[
            { label: '接受部分 min(p,q)', tone: 'success', variant: 'soft' },
            { label: method === 'residual' ? '重采样部分 max(0,p−q)' : method === 'p' ? '重采样部分 (1−β)·p' : '多余质量 q−min(p,q)', tone: bTone, variant: 'soft' },
            { label: 'target p(x)', tone: 'muted', variant: 'line' },
            { label: '◆ 模拟频率', tone: 'accent', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={method}
            onChange={setMethod}
            options={[
              { value: 'residual', label: '残差重采样 ✓' },
              { value: 'p', label: '从 p 重采样 ✗' },
              { value: 'q', label: '不验证 ✗' },
            ]}
          />
          <Slider className="min-w-[180px] flex-1" label="草稿偏离程度 λ" value={lam} min={0} max={1} step={0.05} onChange={setLam} format={(v) => v.toFixed(2)} />
          <Button size="sm" variant="secondary" icon={<Dices size={14} />} onClick={() => setSeed((s) => s + 1)}>
            再模拟 2 万次
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-2">
        {LABELS.map((lab, i) => (
          <div key={i} className="grid items-center gap-2 text-[12px]" style={{ gridTemplateColumns: '3.2rem minmax(0,1fr) 7.5rem' }}>
            <span className="text-right font-mono text-fg">{lab}</span>
            <div className="relative h-6 min-w-0 rounded-md bg-surface-2">
              <div
                className="absolute inset-y-0 left-0 rounded-l-md transition-[width] duration-300"
                style={{ width: `${(th.a[i] / m) * 100}%`, background: toneSoft('success', 70), borderRight: `1px solid ${toneColor('success')}` }}
              />
              <div
                className="absolute inset-y-0 transition-[left,width] duration-300"
                style={{ left: `${(th.a[i] / m) * 100}%`, width: `${(th.b[i] / m) * 100}%`, background: toneSoft(bTone, 70) }}
              />
              <div className="absolute -inset-y-1 w-0.5 bg-fg" style={{ left: `${(P[i] / m) * 100}%` }} title={`p = ${P[i]}`} />
              <div
                className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rotate-45 border border-surface transition-[left] duration-300"
                style={{ left: `${(emp[i] / m) * 100}%`, background: toneColor('accent') }}
              />
            </div>
            <span className="font-mono text-[11px] text-muted tabular-nums">
              {out[i].toFixed(3)} <span className="text-subtle">/ p {P[i].toFixed(2)}</span>
            </span>
          </div>
        ))}
        <div className="mt-1 grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="β = Σ min(p,q)" value={beta.toFixed(3)} tone="success" />
          <Stat label="理论输出 vs p（TV）" value={tvTheory.toFixed(4)} tone={tvTheory < 1e-9 ? 'success' : 'danger'} />
          <Stat label="模拟输出 vs p（TV）" value={tvEmp.toFixed(4)} tone={tvEmp < 0.01 ? 'success' : 'danger'} />
          <Stat label="模拟次数" value={N_SIM.toLocaleString()} />
        </div>
      </div>
    </VizFrame>
  )
}
