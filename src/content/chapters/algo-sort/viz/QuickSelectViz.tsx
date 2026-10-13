/**
 * 快速选择（LC215 第 k 大）：划分后只进入目标所在的一侧。
 * 每轮区间大小以几何级数缩小 ⇒ 期望 O(n)；有序输入 + 固定 pivot ⇒ 每轮只少一个 ⇒ O(n²)。
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type VizTone } from '@/viz'
import { Button, Segmented, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { C } from './common'

type Input = 'random' | 'sorted'
type Pivot = 'random' | 'last'
const INPUTS: Record<Input, number[]> = {
  random: [7, 2, 9, 4, 11, 1, 6, 10, 3, 8, 5],
  sorted: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

const CODE = `def kth_largest(nums, k):
    target = len(nums) - k            # 第 k 大 = 升序下的下标 n-k
    lo, hi = 0, len(nums) - 1
    while True:
        p = partition(nums, lo, hi)   # pivot 归位到 p（Lomuto）
        if p == target:
            return nums[p]
        elif p < target:
            lo = p + 1                # 只进右侧
        else:
            hi = p - 1                # 只进左侧`

interface Frame {
  vals: number[]
  ids: number[]
  lo: number
  hi: number
  p?: number
  pivotId?: number
  line: number
  sizes: number[]
  cmps: number
  caption: React.ReactNode
}

function build(input: number[], k: number, pivot: Pivot, seed: number): Frame[] {
  const vals = [...input]
  const ids = input.map((_, i) => i)
  const n = vals.length
  const target = n - k
  const r = rng(seed)
  const fr: Frame[] = []
  const sizes: number[] = []
  let cmps = 0
  let lo = 0
  let hi = n - 1
  fr.push({ vals: [...vals], ids: [...ids], lo, hi, line: 2, sizes: [], cmps, caption: <>找第 <C>k = {k}</C> 大，即升序后下标 <C>n − k = {target}</C> 的元素。</> })
  for (;;) {
    sizes.push(hi - lo + 1)
    const idx = pivot === 'last' ? hi : lo + Math.floor(r() * (hi - lo + 1))
    ;[vals[idx], vals[hi]] = [vals[hi], vals[idx]]
    ;[ids[idx], ids[hi]] = [ids[hi], ids[idx]]
    const pv = vals[hi]
    const pid = ids[hi]
    let i = lo
    for (let j = lo; j < hi; j++) {
      cmps++
      if (vals[j] < pv) {
        ;[vals[i], vals[j]] = [vals[j], vals[i]]
        ;[ids[i], ids[j]] = [ids[j], ids[i]]
        i++
      }
    }
    ;[vals[i], vals[hi]] = [vals[hi], vals[i]]
    ;[ids[i], ids[hi]] = [ids[hi], ids[i]]
    const p = i
    const base = { vals: [...vals], ids: [...ids], lo, hi, p, pivotId: pid, sizes: [...sizes], cmps }
    if (p === target) {
      fr.push({ ...base, line: 7, caption: <>pivot <C>{pv}</C> 归位到 <C>{p}</C> = 目标下标，返回 <C>{pv}</C>。共 {sizes.length} 轮、{cmps} 次比较。</> })
      return fr
    }
    if (p < target) {
      fr.push({ ...base, line: 9, caption: <>在 <C>[{lo}, {hi}]</C> 上划分（pivot = <C>{pv}</C>），pivot 落在 <C>{p}</C> &lt; {target}：目标在右侧，<b>左侧 {p - lo + 1} 个元素整体丢弃</b>。</> })
      lo = p + 1
    } else {
      fr.push({ ...base, line: 11, caption: <>在 <C>[{lo}, {hi}]</C> 上划分（pivot = <C>{pv}</C>），pivot 落在 <C>{p}</C> &gt; {target}：目标在左侧，<b>右侧 {hi - p + 1} 个元素整体丢弃</b>。</> })
      hi = p - 1
    }
    if (lo === hi) {
      sizes.push(1)
      fr.push({ vals: [...vals], ids: [...ids], lo, hi, p: lo, line: 7, sizes: [...sizes], cmps, caption: <>区间只剩 <C>[{lo}]</C>，就是答案 <C>{vals[lo]}</C>。共 {cmps} 次比较。</> })
      return fr
    }
  }
}

export function QuickSelectViz() {
  const [input, setInput] = useState<Input>('random')
  const [pivot, setPivot] = useState<Pivot>('random')
  const [k, setK] = useState(3)
  const [seed, setSeed] = useState(5)
  const frames = useMemo(() => build(INPUTS[input], k, pivot, seed), [input, k, pivot, seed])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame
  const n = f.vals.length
  const target = n - k
  const highlights: Record<number, VizTone> = {}
  f.ids.forEach((id, i) => {
    if (id === f.pivotId) highlights[i] = 'gold'
  })
  if (st.atEnd && f.p != null) highlights[f.p] = 'success'
  const pointers: ArrayPointer[] = [{ index: target, label: `目标 n−k=${target}`, tone: 'success' }]

  return (
    <VizFrame
      title="快速选择：划分后只进一侧，期望 O(n)"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '本轮 pivot（已归位）', tone: 'gold' },
            { label: '答案', tone: 'success' },
            { label: '已丢弃的一侧（变暗）', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="min-w-[140px] flex-1" label="k（第 k 大）" value={k} min={1} max={11} onChange={setK} />
          <Segmented
            size="sm"
            value={input}
            onChange={setInput}
            options={[
              { value: 'random', label: '随机输入' },
              { value: 'sorted', label: '有序输入' },
            ]}
          />
          <Segmented
            size="sm"
            value={pivot}
            onChange={setPivot}
            options={[
              { value: 'random', label: '随机 pivot' },
              { value: 'last', label: '固定末尾 pivot' },
            ]}
          />
          {pivot === 'random' && (
            <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
              换随机种子
            </Button>
          )}
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-4 py-1">
          <ArrayView values={f.vals} ids={f.ids} mode="bars" barHeight={130} cellSize={36} highlights={highlights} pointers={pointers} dim={(i) => i < f.lo || i > f.hi} />
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">每轮区间大小（总工作量 = 之和）</div>
            <div className="flex flex-wrap items-end gap-1">
              {f.sizes.map((s, i) => (
                <div key={i} className="flex flex-col items-center">
                  <div className="w-6 rounded-t bg-accent/60" style={{ height: 4 + s * 5 }} />
                  <span className="font-mono text-[10px] text-muted">{s}</span>
                </div>
              ))}
            </div>
            <div className="mt-1 font-mono text-xs text-muted">
              比较 {f.cmps} 次 · 参考 2n = {2 * n}，n(n−1)/2 = {(n * (n - 1)) / 2}
            </div>
          </div>
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ lo: f.lo, hi: f.hi, p: f.p ?? '—', target }} />
      </div>
    </VizFrame>
  )
}
