/**
 * 合并两个有序数组 + 顺带统计逆序对：
 * 取右边的 R[j] 时，左边剩下的 L[i:] 全都 > R[j]，一次性贡献 len(L) - i 个逆序对。
 */
import { useMemo } from 'react'
import { ArrayView, CodeTrace, Legend, StepControls, VizFrame, useStepper, type ArrayPointer, type VizTone } from '@/viz'
import { C } from './common'

const L = [3, 7, 9, 12]
const R = [1, 5, 8, 10]

const CODE = `def merge_count(L, R):
    out, inv = [], 0
    i = j = 0
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:              # 相等时先取左边 → 稳定
            out.append(L[i]); i += 1
        else:
            out.append(R[j]); j += 1
            inv += len(L) - i         # L[i:] 全部 > R[j]
    out += L[i:] + R[j:]
    return out, inv`

interface Frame {
  i: number
  j: number
  out: number[]
  inv: number
  line: number
  counted?: [number, number] // L 中被计数的区间
  caption: React.ReactNode
}

function build(): Frame[] {
  const fr: Frame[] = []
  const out: number[] = []
  let i = 0
  let j = 0
  let inv = 0
  fr.push({ i, j, out: [], inv, line: 3, caption: <>两个有序段：左 <C>[{L.join(', ')}]</C>、右 <C>[{R.join(', ')}]</C>。逆序对 = 左边元素 &gt; 右边元素的数对（跨越两段的那部分）。</> })
  while (i < L.length && j < R.length) {
    if (L[i] <= R[j]) {
      out.push(L[i])
      fr.push({ i: i + 1, j, out: [...out], inv, line: 6, caption: <><C>L[{i}] = {L[i]} ≤ R[{j}] = {R[j]}</C>：取左边。左边元素进入结果时不产生新的跨段逆序对。</> })
      i++
    } else {
      out.push(R[j])
      inv += L.length - i
      fr.push({ i, j: j + 1, out: [...out], inv, line: 9, counted: [i, L.length - 1], caption: <><C>L[{i}] = {L[i]} &gt; R[{j}] = {R[j]}</C>：取右边。左段有序，所以 <C>L[{i}:]</C> = [{L.slice(i).join(', ')}] <b>全部</b> &gt; {R[j]}，一次加上 <C>{L.length - i}</C> 个逆序对。</> })
      j++
    }
  }
  const rest = [...L.slice(i), ...R.slice(j)]
  out.push(...rest)
  fr.push({ i: L.length, j: R.length, out: [...out], inv, line: 10, caption: <>剩余 <C>[{rest.join(', ')}]</C> 直接接到末尾。跨段逆序对共 <C>{inv}</C> 个——只用了 O(n) 而不是 O(n²) 枚举。整体逆序对 = 左段内部 + 右段内部 + 跨段，递归即可。</> })
  return fr
}

export function MergeInvViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 1100 })
  const f = st.frame
  const lHi: Record<number, VizTone> = {}
  for (let k = 0; k < f.i; k++) lHi[k] = 'muted'
  if (f.counted) for (let k = f.counted[0]; k <= f.counted[1]; k++) lHi[k] = 'danger'
  const rHi: Record<number, VizTone> = {}
  for (let k = 0; k < f.j; k++) rHi[k] = 'muted'
  const lp: ArrayPointer[] = f.i < L.length ? [{ index: f.i, label: 'i', tone: 'info' }] : []
  const rp: ArrayPointer[] = f.j < R.length ? [{ index: f.j, label: 'j', tone: 'warning' }] : []

  return (
    <VizFrame
      title="合并时顺手数逆序对：取右边元素时一次算一批"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '本步计入逆序对的左段元素', tone: 'danger', variant: 'soft' },
            { label: '已输出', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="mb-1 text-xs text-muted">左段 L</div>
              <ArrayView values={L} cellSize={34} highlights={lHi} pointers={lp} />
            </div>
            <div>
              <div className="mb-1 text-xs text-muted">右段 R</div>
              <ArrayView values={R} cellSize={34} highlights={rHi} pointers={rp} />
            </div>
          </div>
          <div>
            <div className="mb-1 text-xs text-muted">
              out · 逆序对 inv = <span className="font-mono font-semibold text-danger">{f.inv}</span>
            </div>
            <div className="min-h-[44px]">
              {f.out.length > 0 && <ArrayView values={f.out} cellSize={34} highlights={{ [f.out.length - 1]: 'success' }} />}
            </div>
          </div>
        </div>
        <CodeTrace code={CODE} active={f.line} vars={{ i: f.i, j: f.j, inv: f.inv }} />
      </div>
    </VizFrame>
  )
}
