/**
 * 复盘：同一份输入，不同排序算法各要做多少次比较？
 * 切换输入分布（随机 / 有序 / 逆序 / 少量重复值 / 全相同）和规模 n，比较次数实时计算。
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Button, Segmented, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { C } from './common'

type Input = 'random' | 'sorted' | 'reversed' | 'few' | 'equal'

function makeInput(kind: Input, n: number, seed: number) {
  const r = rng(seed)
  if (kind === 'sorted') return Array.from({ length: n }, (_, i) => i)
  if (kind === 'reversed') return Array.from({ length: n }, (_, i) => n - i)
  if (kind === 'equal') return Array(n).fill(7)
  if (kind === 'few') return Array.from({ length: n }, () => Math.floor(r() * 3))
  const a = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function insertion(a0: number[]) {
  const a = [...a0]
  let c = 0
  for (let i = 1; i < a.length; i++) {
    const x = a[i]
    let j = i - 1
    while (j >= 0) {
      c++
      if (a[j] <= x) break
      a[j + 1] = a[j]
      j--
    }
    a[j + 1] = x
  }
  return c
}
function merge(a0: number[]) {
  let c = 0
  const rec = (a: number[]): number[] => {
    if (a.length <= 1) return a
    const m = a.length >> 1
    const L = rec(a.slice(0, m))
    const R = rec(a.slice(m))
    const out: number[] = []
    let i = 0
    let j = 0
    while (i < L.length && j < R.length) {
      c++
      out.push(L[i] <= R[j] ? L[i++] : R[j++])
    }
    return out.concat(L.slice(i), R.slice(j))
  }
  rec(a0)
  return c
}
function quick(a0: number[], seed: number, three: boolean, fixed: boolean) {
  const a = [...a0]
  const r = rng(seed * 13 + 7)
  let c = 0
  const stack: [number, number][] = [[0, a.length - 1]]
  while (stack.length) {
    const [lo, hi] = stack.pop()!
    if (lo >= hi) continue
    const k = fixed ? hi : lo + Math.floor(r() * (hi - lo + 1))
    ;[a[k], a[hi]] = [a[hi], a[k]]
    const p = a[hi]
    if (!three) {
      let i = lo
      for (let j = lo; j < hi; j++) {
        c++
        if (a[j] < p) {
          ;[a[i], a[j]] = [a[j], a[i]]
          i++
        }
      }
      ;[a[i], a[hi]] = [a[hi], a[i]]
      stack.push([lo, i - 1], [i + 1, hi])
    } else {
      let lt = lo
      let i = lo
      let gt = hi
      while (i <= gt) {
        c++
        if (a[i] < p) {
          ;[a[lt], a[i]] = [a[i], a[lt]]
          lt++
          i++
        } else {
          c++
          if (a[i] > p) {
            ;[a[i], a[gt]] = [a[gt], a[i]]
            gt--
          } else i++
        }
      }
      stack.push([lo, lt - 1], [gt + 1, hi])
    }
  }
  return c
}
function heap(a0: number[]) {
  const a = [...a0]
  const n = a.length
  let c = 0
  const down = (i: number, size: number) => {
    while (2 * i + 1 < size) {
      let ch = 2 * i + 1
      if (ch + 1 < size) {
        c++
        if (a[ch + 1] > a[ch]) ch++
      }
      c++
      if (a[i] >= a[ch]) break
      ;[a[i], a[ch]] = [a[ch], a[i]]
      i = ch
    }
  }
  for (let i = (n >> 1) - 1; i >= 0; i--) down(i, n)
  for (let end = n - 1; end > 0; end--) {
    ;[a[0], a[end]] = [a[end], a[0]]
    down(0, end)
  }
  return c
}

export function SortRaceViz() {
  const [input, setInput] = useState<Input>('random')
  const [n, setN] = useState(128)
  const [seed, setSeed] = useState(1)
  const rows = useMemo(() => {
    const a = makeInput(input, n, seed)
    return [
      { name: '插入排序', c: insertion(a), stable: true },
      { name: '归并排序', c: merge(a), stable: true },
      { name: '堆排序', c: heap(a), stable: false },
      { name: '快排：固定末尾 pivot', c: quick(a, seed, false, true), stable: false },
      { name: '快排：随机 + 二路', c: quick(a, seed, false, false), stable: false },
      { name: '快排：随机 + 三路', c: quick(a, seed, true, false), stable: false },
    ]
  }, [input, n, seed])
  const nlogn = Math.round(n * Math.log2(n))
  const n2 = (n * (n - 1)) / 2
  const max = Math.max(...rows.map((r) => r.c), nlogn)

  return (
    <VizFrame
      title="同一份输入，各算法要做多少次比较？"
      caption={
        <>
          n = {n}：参考 <C>n·log₂n ≈ {nlogn}</C>、<C>n(n−1)/2 = {n2}</C>。
          试试：<b>已排序</b>时插入排序只要 n−1 次、固定 pivot 快排退化为 n²/2；<b>全相同 / 少量重复</b>时二路快排退化而三路快排几乎线性；归并与堆排序对任何输入都稳定在 n log n 附近。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '≤ 1.5·n log₂n', tone: 'success' },
            { label: '介于两者之间', tone: 'warning' },
            { label: '≥ n²/4（退化）', tone: 'danger' },
            { label: 'n·log₂n 参考线', tone: 'accent', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={input}
            onChange={setInput}
            options={[
              { value: 'random', label: '随机' },
              { value: 'sorted', label: '已排序' },
              { value: 'reversed', label: '逆序' },
              { value: 'few', label: '只有 0/1/2' },
              { value: 'equal', label: '全相同' },
            ]}
          />
          <Slider className="min-w-[160px] flex-1" label="n" value={n} min={16} max={512} step={16} onChange={setN} />
          <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
            换随机种子
          </Button>
        </>
      }
    >
      <div className="space-y-2">
        {rows.map((r) => {
          const tone = r.c >= n2 / 2 ? 'danger' : r.c <= 1.5 * nlogn ? 'success' : 'warning'
          return (
            <div key={r.name} className="grid grid-cols-[7.5rem_1fr] items-center gap-2 @md:grid-cols-[10rem_1fr]">
              <div className="truncate text-xs text-fg">
                {r.name}
                {r.stable && <span className="ml-1 text-[10px] text-success">稳定</span>}
              </div>
              <div className="relative h-6 min-w-0 rounded bg-surface-2">
                <div
                  className="absolute inset-y-0 left-0 rounded transition-[width] duration-300"
                  style={{ width: `${(r.c / max) * 100}%`, background: toneColor(tone), opacity: 0.75 }}
                />
                <div className="absolute inset-y-0 w-0.5 bg-accent" style={{ left: `${(nlogn / max) * 100}%` }} />
                <span className="absolute inset-y-0 right-1 flex items-center font-mono text-[11px] text-fg">{r.c}</span>
              </div>
            </div>
          )
        })}
      </div>
    </VizFrame>
  )
}
