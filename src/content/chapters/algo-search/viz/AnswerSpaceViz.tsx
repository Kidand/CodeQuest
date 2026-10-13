/**
 * 二分答案（LC875 爱吃香蕉的珂珂）：答案空间上的 check(k) 呈 F…F T…T。
 * 横轴是候选速度 k，柱高是 hours(k)，红线是时限 h；柱子低于红线 ⇔ check(k)=True。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, useStepper, type VizTone } from '@/viz'
import { Slider } from '@/components/ui'
import { C, ThresholdBars, type BarPointer } from './common'

const PILES = [3, 6, 7, 11]
const MAXK = Math.max(...PILES)
const hours = (k: number) => PILES.reduce((s, p) => s + Math.ceil(p / k), 0)

const CODE = `def min_eating_speed(piles, h):
    def check(k):                      # 速度 k 能否在 h 小时内吃完
        return sum((p + k - 1) // k for p in piles) <= h
    lo, hi = 1, max(piles)             # 答案一定在 [1, max(piles)]
    while lo < hi:
        mid = (lo + hi) // 2
        if not check(mid):
            lo = mid + 1               # 太慢，答案在右边
        else:
            hi = mid                   # 够快，mid 可能就是最小值
    return lo`

interface Frame {
  lo: number
  hi: number
  mid?: number
  line: number
  caption: React.ReactNode
}

function build(h: number): Frame[] {
  const fr: Frame[] = []
  let lo = 1
  let hi = MAXK
  fr.push({ lo, hi, line: 4, caption: <>答案区间 <C>[1, {MAXK}]</C>：速度再快也不会超过最大一堆（每堆至少 1 小时），<C>k = max(piles)</C> 一定可行。</> })
  let round = 0
  while (lo < hi) {
    round++
    const mid = (lo + hi) >> 1
    const hm = hours(mid)
    if (hm > h) {
      fr.push({ lo, hi, mid, line: 8, caption: <>第 {round} 轮：<C>hours({mid}) = {hm} &gt; h = {h}</C>，check 为 False——速度 {mid} 太慢，比它更慢的也都不行：<C>lo = {mid + 1}</C>。</> })
      lo = mid + 1
    } else {
      fr.push({ lo, hi, mid, line: 10, caption: <>第 {round} 轮：<C>hours({mid}) = {hm} ≤ h = {h}</C>，check 为 True——{mid} 可行，但可能还有更小的：<C>hi = {mid}</C>。</> })
      hi = mid
    }
  }
  fr.push({ lo, hi, line: 11, caption: <>收敛：最小可行速度 <C>k = {lo}</C>（<C>hours = {hours(lo)}</C>）。只调用了 {round} 次 check，而不是逐个试 {MAXK} 个速度。</> })
  return fr
}

export function AnswerSpaceViz() {
  const [h, setH] = useState(8)
  const frames = useMemo(() => build(h), [h])
  const st = useStepper(frames, { intervalMs: 1200 })
  const f = st.frame
  const ks = Array.from({ length: MAXK }, (_, i) => i + 1)
  const hs = ks.map(hours)
  const tones: VizTone[] = hs.map((v) => (v <= h ? 'success' : 'muted'))
  if (f.mid != null) tones[f.mid - 1] = 'accent'
  const pointers: BarPointer[] = [
    { index: f.lo - 1, label: 'lo', tone: 'info' },
    { index: f.hi - 1, label: 'hi', tone: 'warning' },
  ]
  if (f.mid != null) pointers.push({ index: f.mid - 1, label: 'mid', tone: 'accent' })

  return (
    <VizFrame
      title="二分答案：check(k) 在答案空间上单调"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: 'check(k)=True：hours(k) ≤ h', tone: 'success' },
            { label: 'check(k)=False', tone: 'muted' },
            { label: '本轮试探的 mid', tone: 'accent' },
            { label: '时限 h', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[180px] flex-1" label="时限 h（小时）" value={h} min={PILES.length} max={hours(1)} onChange={setH} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-1 text-xs text-muted">
            piles = <span className="font-mono">[{PILES.join(', ')}]</span>，柱高 = 速度 k 下需要的小时数
          </div>
          <ThresholdBars
            values={hs}
            labels={ks.map(String)}
            threshold={h}
            thresholdLabel={`h=${h}`}
            tones={tones}
            pointers={pointers}
            dim={(i) => i + 1 < f.lo || i + 1 > f.hi}
            xTitle="速度 k（根/小时）"
          />
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ lo: f.lo, hi: f.hi, mid: f.mid ?? '—', 'hours(mid)': f.mid != null ? hours(f.mid) : '—', h }} />
      </div>
    </VizFrame>
  )
}
