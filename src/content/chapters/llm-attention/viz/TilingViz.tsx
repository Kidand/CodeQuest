/**
 * FlashAttention 的二维分块：N=8、块大小 2 → 4×4 个 tile。
 * FA1：外层遍历 K/V 块、内层遍历 Q 块（每个内层步都要读写 O_i、ℓ_i、m_i）；
 * FA2：外层遍历 Q 块、内层遍历 K/V 块（O_i 留在片上，最后只写一次；外层可并行到不同 thread block）。
 * 因果时整块位于上三角的 tile 直接跳过，对角 tile 需在块内加掩码。
 */
import { useMemo, useState } from 'react'
import { Legend, StepControls, VizFrame, toneColor, toneSoft, useStepper } from '@/viz'
import { Segmented, Switch } from '@/components/ui'
import { cn } from '@/lib/utils'

const NB = 4
type Order = 'fa1' | 'fa2'
interface Step {
  i: number
  j: number
  reads: number
  writes: number
  note: string
}

function build(order: Order, causal: boolean) {
  const steps: Step[] = []
  let reads = 0
  let writes = 0
  const skip = (i: number, j: number) => causal && j > i
  if (order === 'fa1') {
    for (let j = 0; j < NB; j++) {
      reads += 2 // K_j, V_j
      for (let i = 0; i < NB; i++) {
        if (skip(i, j)) continue
        reads += 2 // Q_i, O_i（含 ℓ_i、m_i）
        writes += 1 // O_i（含 ℓ_i、m_i）
        steps.push({ i, j, reads, writes, note: `外层 j=${j}（K_${j}, V_${j} 常驻 SRAM），内层 i=${i}：从 HBM 读 Q_${i} 和部分结果 O_${i}、ℓ_${i}、m_${i}，更新后写回 HBM。` })
      }
    }
  } else {
    for (let i = 0; i < NB; i++) {
      reads += 1 // Q_i
      for (let j = 0; j < NB; j++) {
        if (skip(i, j)) continue
        reads += 2 // K_j, V_j
        const last = causal ? j === i : j === NB - 1
        if (last) writes += 1
        steps.push({
          i,
          j,
          reads,
          writes,
          note: last
            ? `外层 i=${i}（Q_${i} 与累加器 O_${i}、m、ℓ 常驻片上），处理完最后一个 K/V 块后做一次 O_${i}/ℓ 并写回 O_${i} 与 logsumexp L_${i}。`
            : `外层 i=${i}（Q_${i} 常驻），内层 j=${j}：读 K_${j}, V_${j}，online softmax 更新片上累加器，不写 HBM。`,
        })
      }
    }
  }
  return steps
}

export function TilingViz() {
  const [order, setOrder] = useState<Order>('fa2')
  const [causal, setCausal] = useState(true)
  const steps = useMemo(() => build(order, causal), [order, causal])
  const st = useStepper(steps, { intervalMs: 900 })
  const s = st.frame
  const doneSet = new Set(steps.slice(0, st.i).map((x) => `${x.i},${x.j}`))
  const total = steps.length

  return (
    <VizFrame
      title="分块：一个 tile 一个 tile 地算，N×N 矩阵从不落到 HBM"
      caption={
        <>
          {s.note}{' '}
          <span className="text-muted">
            （累计读 {s.reads} 块、写 {s.writes} 块；共计算 {total} / {NB * NB} 个 tile{causal ? `，跳过 ${NB * NB - total} 个全被遮的 tile` : ''}）
          </span>
        </>
      }
      legend={
        <Legend
          items={[
            { label: '当前 tile', tone: 'accent', variant: 'soft' },
            { label: '已完成', tone: 'success', variant: 'soft' },
            { label: '对角 tile：块内需掩码', tone: 'warning', variant: 'line' },
            { label: '整块被遮：跳过', tone: 'muted', variant: 'soft' },
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
              { value: 'fa1', label: 'FA1：外层 K/V' },
              { value: 'fa2', label: 'FA2：外层 Q' },
            ]}
          />
          <Switch checked={causal} onChange={setCausal} label="因果" />
        </>
      }
    >
      <div className="grid items-center gap-5 @2xl:grid-cols-[auto_1fr]">
        <div className="mx-auto">
          <div className="mb-1 grid grid-cols-[28px_repeat(4,52px)] gap-1 text-center font-mono text-[11px] text-muted">
            <span />
            {Array.from({ length: NB }, (_, j) => (
              <span key={j}>K/V{j}</span>
            ))}
          </div>
          {Array.from({ length: NB }, (_, i) => (
            <div key={i} className="mb-1 grid grid-cols-[28px_repeat(4,52px)] gap-1">
              <span className="flex items-center justify-end pr-1 font-mono text-[11px] text-muted">Q{i}</span>
              {Array.from({ length: NB }, (_, j) => {
                const skipped = causal && j > i
                const cur = s.i === i && s.j === j
                const done = doneSet.has(`${i},${j}`)
                const diag = causal && i === j
                return (
                  <div
                    key={j}
                    className={cn('flex h-[52px] items-center justify-center rounded-lg font-mono text-[11px] transition-colors', skipped && 'text-subtle')}
                    style={{
                      background: skipped
                        ? 'repeating-linear-gradient(135deg, transparent 0 5px, color-mix(in oklab, var(--subtle) 30%, transparent) 5px 6px), var(--surface)'
                        : cur
                          ? toneSoft('accent', 45)
                          : done
                            ? toneSoft('success', 28)
                            : 'var(--surface)',
                      boxShadow: `inset 0 0 0 ${diag ? 2 : 1}px ${diag ? toneColor('warning') : cur ? toneColor('accent') : 'var(--border)'}`,
                    }}
                  >
                    {skipped ? '跳过' : `S${i}${j}`}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
        <div className="min-w-0 space-y-2 text-[12.5px]">
          <div className="rounded-xl border border-border bg-surface p-3">
            <div className="mb-1 text-xs font-medium text-muted">SRAM 中此刻的内容</div>
            <div className="font-mono leading-relaxed">
              Q<sub>{s.i}</sub>, K<sub>{s.j}</sub>, V<sub>{s.j}</sub>, S<sub>{s.i}{s.j}</sub> = Q<sub>{s.i}</sub>K<sub>{s.j}</sub>ᵀ（B<sub>r</sub>×B<sub>c</sub>），O<sub>{s.i}</sub> 累加器, m<sub>{s.i}</sub>, ℓ<sub>{s.i}</sub>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-surface p-3 text-muted">
            {order === 'fa1'
              ? 'FA1：O_i 的部分结果每个内层步都要在 HBM 与 SRAM 之间往返；外层循环的各次迭代都要写同一组 O_i，不便在序列维上并行。'
              : 'FA2：每个 Q 块的工作完全独立，可以分给不同的 thread block 并行（序列维并行）；O_i 只在最后写一次，且推迟到最后才除以 ℓ，减少非矩阵乘运算。'}
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
