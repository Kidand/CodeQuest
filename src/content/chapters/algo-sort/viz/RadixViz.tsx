/**
 * LSD 基数排序（基数 10）：从个位开始，每一位做一次**稳定**的桶分配 + 按桶顺序收集。
 */
import { useMemo } from 'react'
import { Legend, StepControls, VizFrame, useStepper } from '@/viz'
import { cn } from '@/lib/utils'
import { C } from './common'

const DATA = [170, 45, 75, 90, 802, 24, 2, 66]
const PASSES = 3
const PLACE = ['个位', '十位', '百位']

interface Frame {
  arr: number[]
  buckets: number[][]
  pass: number
  cur?: number // 正在分配的元素在 arr 中的下标
  collected?: boolean
  caption: React.ReactNode
}

const digit = (x: number, pass: number) => Math.floor(x / 10 ** pass) % 10

function build(): Frame[] {
  let arr = [...DATA]
  const fr: Frame[] = [{ arr: [...arr], buckets: Array.from({ length: 10 }, () => []), pass: 0, caption: <>最大值 802 有 3 位，做 3 轮。每轮按某一位把元素<b>依次</b>放进 10 个桶，再按桶号顺序收集。</> }]
  for (let pass = 0; pass < PASSES; pass++) {
    const buckets: number[][] = Array.from({ length: 10 }, () => [])
    arr.forEach((x, i) => {
      buckets[digit(x, pass)].push(x)
      fr.push({ arr: [...arr], buckets: buckets.map((b) => [...b]), pass, cur: i, caption: <>第 {pass + 1} 轮（{PLACE[pass]}）：<C>{x}</C> 的{PLACE[pass]}是 <C>{digit(x, pass)}</C>，追加到 {digit(x, pass)} 号桶<b>末尾</b>。</> })
    })
    arr = buckets.flat()
    fr.push({
      arr: [...arr], buckets: buckets.map((b) => [...b]), pass, collected: true,
      caption:
        pass < PASSES - 1 ? (
          <>按 0→9 号桶顺序收集：<C>[{arr.join(', ')}]</C>。现在序列按{PLACE[pass]}有序；{pass > 0 ? `由于每轮都稳定，${PLACE[pass]}相同的元素仍保持上一轮按低位排好的顺序。` : '下一轮按十位分配时，桶内顺序会保留这一轮的结果。'}</>
        ) : (
          <>3 轮后完全有序：<C>[{arr.join(', ')}]</C>。时间 <C>O(d·(n + b))</C>，d = 位数，b = 基数；没有一次元素间比较。</>
        ),
    })
  }
  return fr
}

export function RadixViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 800 })
  const f = st.frame

  return (
    <VizFrame
      title="LSD 基数排序：逐位稳定分桶，不做比较"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '当前处理的位', tone: 'accent', variant: 'soft' },
            { label: '正在分配的元素', tone: 'warning', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="space-y-4">
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">{f.collected ? '收集后的序列' : '本轮输入序列'}</div>
          <div className="flex flex-wrap gap-1.5">
            {f.arr.map((x, i) => {
              const s = String(x).padStart(3, '0')
              const pos = 2 - f.pass
              return (
                <span key={i} className={cn('rounded-md border px-1.5 py-0.5 font-mono text-[13px]', i === f.cur ? 'border-warning bg-warning-soft' : 'border-border bg-surface')}>
                  {s.split('').map((ch, j) => (
                    <span key={j} className={cn(j === pos && 'rounded bg-accent-soft font-bold text-accent', j < 3 - String(x).length && 'text-subtle')}>
                      {ch}
                    </span>
                  ))}
                </span>
              )
            })}
          </div>
        </div>
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">10 个桶（按{PLACE[f.pass]}）</div>
          <div className="grid grid-cols-5 gap-1.5 @lg:grid-cols-10">
            {f.buckets.map((b, d) => (
              <div key={d} className="min-h-[64px] rounded-lg border border-border bg-surface p-1">
                <div className="mb-1 text-center font-mono text-[11px] text-muted">{d}</div>
                <div className="flex flex-col items-center gap-0.5">
                  {b.map((x, j) => (
                    <span key={j} className="rounded bg-surface-2 px-1 font-mono text-[11.5px]">
                      {x}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
