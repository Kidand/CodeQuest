/**
 * 递归树的形状决定快排的复杂度（冰柱图：每一行是一层递归，每个矩形是一次子调用覆盖的区间）。
 * 选择输入分布、pivot 策略、二路 / 三路划分，观察深度与比较次数。
 */
import { useMemo, useState } from 'react'
import { Shuffle } from 'lucide-react'
import { Legend, VizFrame, toneColor, toneSoft, useElementWidth } from '@/viz'
import { Button, Segmented } from '@/components/ui'
import { rng } from '@/lib/utils'
import { C } from './common'

type Input = 'random' | 'sorted' | 'equal' | 'few'
type Pivot = 'last' | 'random'
type Scheme = 'two' | 'three'
const N = 16

interface Call {
  lo: number
  hi: number
  depth: number
  eqLo: number
  eqHi: number
}

function makeInput(kind: Input, seed: number): number[] {
  const r = rng(seed)
  if (kind === 'sorted') return Array.from({ length: N }, (_, i) => i + 1)
  if (kind === 'equal') return Array(N).fill(5)
  if (kind === 'few') return Array.from({ length: N }, () => 1 + Math.floor(r() * 3))
  const a = Array.from({ length: N }, (_, i) => i + 1)
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function simulate(input: number[], pivot: Pivot, scheme: Scheme, seed: number) {
  const a = [...input]
  const r = rng(seed * 7 + 1)
  const calls: Call[] = []
  let cmp = 0
  const rec = (lo: number, hi: number, depth: number) => {
    if (lo > hi) return
    if (lo === hi) {
      calls.push({ lo, hi, depth, eqLo: lo, eqHi: hi })
      return
    }
    const k = pivot === 'last' ? hi : lo + Math.floor(r() * (hi - lo + 1))
    ;[a[k], a[hi]] = [a[hi], a[k]]
    const p = a[hi]
    if (scheme === 'two') {
      let i = lo
      for (let j = lo; j < hi; j++) {
        cmp++
        if (a[j] < p) {
          ;[a[i], a[j]] = [a[j], a[i]]
          i++
        }
      }
      ;[a[i], a[hi]] = [a[hi], a[i]]
      calls.push({ lo, hi, depth, eqLo: i, eqHi: i })
      rec(lo, i - 1, depth + 1)
      rec(i + 1, hi, depth + 1)
    } else {
      let lt = lo
      let i = lo
      let gt = hi
      while (i <= gt) {
        cmp++
        if (a[i] < p) {
          ;[a[lt], a[i]] = [a[i], a[lt]]
          lt++
          i++
        } else {
          cmp++
          if (a[i] > p) {
            ;[a[i], a[gt]] = [a[gt], a[i]]
            gt--
          } else i++
        }
      }
      calls.push({ lo, hi, depth, eqLo: lt, eqHi: gt })
      rec(lo, lt - 1, depth + 1)
      rec(gt + 1, hi, depth + 1)
    }
  }
  rec(0, N - 1, 0)
  const depth = Math.max(...calls.map((c) => c.depth)) + 1
  return { calls, cmp, depth }
}

export function RecursionShapeViz() {
  const [input, setInput] = useState<Input>('sorted')
  const [pivot, setPivot] = useState<Pivot>('last')
  const [scheme, setScheme] = useState<Scheme>('two')
  const [seed, setSeed] = useState(3)
  const arr = useMemo(() => makeInput(input, seed), [input, seed])
  const sim = useMemo(() => simulate(arr, pivot, scheme, seed), [arr, pivot, scheme, seed])
  const [ref, measured] = useElementWidth()
  const width = Math.max(300, measured || 640)
  const cell = width / N
  const rowH = 17
  const height = N * rowH + 4
  const nlogn = N * Math.log2(N)
  const worst = (N * (N - 1)) / 2
  const verdict = sim.depth <= 2 * Math.log2(N) + 1 ? '接近平衡' : sim.depth >= N - 2 ? '退化成链' : '偏斜'

  return (
    <VizFrame
      title="递归树的形状决定复杂度：平衡 O(n log n)，退化 O(n²)"
      caption={
        <>
          n = {N}：递归深度 <C>{sim.depth}</C>（{verdict}），比较次数 <C>{sim.cmp}</C>；参考 <C>n·log₂n = {nlogn}</C>、<C>n(n−1)/2 = {worst}</C>。
          {input === 'sorted' && pivot === 'last' && ' 有序输入 + 固定末尾 pivot：每次只切掉一个元素。'}
          {input === 'equal' && scheme === 'two' && ' 全相同元素：二路划分把所有元素都分到一侧，随机 pivot 也救不了。'}
          {input === 'equal' && scheme === 'three' && ' 三路划分：等于 pivot 的元素一次性归位，一层结束。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '一次递归调用覆盖的区间', tone: 'accent', variant: 'soft' },
            { label: '本次归位的 pivot（三路时为 == pivot 段）', tone: 'gold' },
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
              { value: 'random', label: '随机排列' },
              { value: 'sorted', label: '已排序' },
              { value: 'equal', label: '全相同' },
              { value: 'few', label: '只有 1/2/3' },
            ]}
          />
          <Segmented
            size="sm"
            value={pivot}
            onChange={setPivot}
            options={[
              { value: 'last', label: 'pivot = 末尾' },
              { value: 'random', label: 'pivot = 随机' },
            ]}
          />
          <Segmented
            size="sm"
            value={scheme}
            onChange={setScheme}
            options={[
              { value: 'two', label: '二路（Lomuto）' },
              { value: 'three', label: '三路' },
            ]}
          />
          <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
            重新随机
          </Button>
        </>
      }
    >
      <div className="space-y-2">
        <div className="font-mono text-xs text-muted">输入：[{arr.join(', ')}]</div>
        <div ref={ref} className="w-full min-w-0">
          <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="block">
            {sim.calls.map((c, k) => {
              const x = c.lo * cell + 1
              const w = (c.hi - c.lo + 1) * cell - 2
              const y = c.depth * rowH + 2
              return (
                <g key={k}>
                  <rect x={x} y={y} width={w} height={rowH - 3} rx={3} fill={toneSoft('accent', 22)} stroke={toneColor('accent')} strokeOpacity={0.5} />
                  <rect x={c.eqLo * cell + 2} y={y + 2} width={(c.eqHi - c.eqLo + 1) * cell - 4} height={rowH - 7} rx={2} fill={toneColor('gold')} fillOpacity={0.85} />
                </g>
              )
            })}
          </svg>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
          <span>
            行数 = 递归深度 = <b className="font-mono text-fg">{sim.depth}</b>
          </span>
          <span>
            比较次数 = <b className="font-mono text-fg">{sim.cmp}</b>
          </span>
        </div>
      </div>
    </VizFrame>
  )
}
