/**
 * LC4 两个正序数组的中位数：在较短数组上二分“分割线”。
 * A 切在 i 前、B 切在 j = half - i 前；合法当且仅当 A[i-1] ≤ B[j] 且 B[j-1] ≤ A[i]。
 */
import { useMemo, useState } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './common'

const EXAMPLES = [
  { A: [1, 3, 8, 9, 15], B: [7, 11, 18, 19, 21, 25] },
  { A: [1, 3, 5, 7], B: [2, 4, 6, 8, 10] },
  { A: [10, 20, 30], B: [1, 2, 3, 4, 5] },
]

const CODE = `def find_median(A, B):
    if len(A) > len(B):
        A, B = B, A                    # 在较短的数组上二分
    m, n = len(A), len(B)
    half = (m + n + 1) // 2            # 左半部分的元素个数
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2             # A 贡献 i 个给左半
        j = half - i                   # B 贡献 j 个
        Al = A[i - 1] if i > 0 else -inf
        Ar = A[i] if i < m else inf
        Bl = B[j - 1] if j > 0 else -inf
        Br = B[j] if j < n else inf
        if Al > Br:
            hi = i - 1                 # A 给多了
        elif Bl > Ar:
            lo = i + 1                 # A 给少了
        else:
            left = max(Al, Bl)
            if (m + n) % 2:
                return left
            return (left + min(Ar, Br)) / 2`

const fmt = (x: number) => (x === Infinity ? '+∞' : x === -Infinity ? '−∞' : String(x))

interface Frame {
  lo: number
  hi: number
  i?: number
  j?: number
  line: number
  ok?: boolean
  caption: React.ReactNode
}

function build(A: number[], B: number[]): Frame[] {
  const m = A.length
  const n = B.length
  const half = (m + n + 1) >> 1
  const fr: Frame[] = []
  let lo = 0
  let hi = m
  fr.push({ lo, hi, line: 6, caption: <>m = {m}、n = {n}，左半应有 <C>half = ({m}+{n}+1)//2 = {half}</C> 个元素。在 A 上二分“A 贡献几个”：<C>i ∈ [0, {m}]</C>。</> })
  while (lo <= hi) {
    const i = (lo + hi) >> 1
    const j = half - i
    const Al = i > 0 ? A[i - 1] : -Infinity
    const Ar = i < m ? A[i] : Infinity
    const Bl = j > 0 ? B[j - 1] : -Infinity
    const Br = j < n ? B[j] : Infinity
    if (Al > Br) {
      fr.push({ lo, hi, i, j, line: 15, caption: <><C>i = {i}, j = {j}</C>：<C>A[i-1] = {fmt(Al)} &gt; B[j] = {fmt(Br)}</C>，A 左边有元素比 B 右边还大 ⇒ A 给多了，<C>hi = {i - 1}</C>。</> })
      hi = i - 1
    } else if (Bl > Ar) {
      fr.push({ lo, hi, i, j, line: 17, caption: <><C>i = {i}, j = {j}</C>：<C>B[j-1] = {fmt(Bl)} &gt; A[i] = {fmt(Ar)}</C>，B 左边有元素比 A 右边还大 ⇒ A 给少了，<C>lo = {i + 1}</C>。</> })
      lo = i + 1
    } else {
      const left = Math.max(Al, Bl)
      const odd = (m + n) % 2 === 1
      const med = odd ? left : (left + Math.min(Ar, Br)) / 2
      fr.push({
        lo, hi, i, j, line: odd ? 21 : 22, ok: true,
        caption: (
          <>
            ✅ <C>{fmt(Al)} ≤ {fmt(Br)}</C> 且 <C>{fmt(Bl)} ≤ {fmt(Ar)}</C>：左半 {half} 个元素全部 ≤ 右半。
            {odd ? <>总数为奇数，中位数 = 左半最大 <C>max({fmt(Al)}, {fmt(Bl)}) = {med}</C>。</> : <>总数为偶数，中位数 = <C>(max({fmt(Al)}, {fmt(Bl)}) + min({fmt(Ar)}, {fmt(Br)})) / 2 = {med}</C>。</>}
          </>
        ),
      })
      return fr
    }
  }
  return fr
}

function side(arr: number[], cut: number | undefined, name: string): { pointers: ArrayPointer[]; highlights: Record<number, VizTone> } {
  if (cut == null) return { pointers: [], highlights: {} }
  const highlights: Record<number, VizTone> = {}
  arr.forEach((_, k) => (highlights[k] = k < cut ? 'info' : 'warning'))
  // 分割线在下标 cut 之前：指针指向右半的第一个位置（cut == len 时画在数组外侧）
  return { pointers: [{ index: cut, label: `${name}=${cut}`, tone: 'accent' }], highlights }
}

export function PartitionViz() {
  const [ex, setEx] = useState(0)
  const { A, B } = EXAMPLES[ex]
  const frames = useMemo(() => build(A, B), [A, B])
  const st = useStepper(frames, { intervalMs: 1400 })
  const f = st.frame
  const a = side(A, f.i, 'i')
  const b = side(B, f.j, 'j')

  return (
    <VizFrame
      title="两个有序数组的中位数：二分的是“分割线”"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '左半（共 half 个）', tone: 'info', variant: 'soft' },
            { label: '右半', tone: 'warning', variant: 'soft' },
            { label: '▲ 分割线右侧第一个', tone: 'accent' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={ex}
            onChange={setEx}
            options={[
              { value: 0, label: '例 1（奇数）' },
              { value: 1, label: '例 2' },
              { value: 2, label: '例 3（偶数 + 边界）' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <div>
            <div className="mb-1 text-xs text-muted">
              A（较短）{f.i != null && <span className="font-mono">：i = {f.i}</span>}
            </div>
            <ArrayView values={A} cellSize={40} pointers={a.pointers} highlights={a.highlights} />
          </div>
          <div>
            <div className="mb-1 text-xs text-muted">
              B{f.j != null && <span className="font-mono">：j = half − i = {f.j}</span>}
            </div>
            <ArrayView values={B} cellSize={40} pointers={b.pointers} highlights={b.highlights} />
          </div>
          <div className="font-mono text-xs text-muted">
            i 的搜索区间 [lo, hi] = [{f.lo}, {f.hi}]
          </div>
        </div>
        <CodeTrace code={CODE} active={f.line} maxHeight={340} vars={{ lo: f.lo, hi: f.hi, i: f.i ?? '—', j: f.j ?? '—' }} />
      </div>
    </VizFrame>
  )
}
