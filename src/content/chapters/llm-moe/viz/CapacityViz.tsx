/**
 * Capacity factor 与 token 丢弃：每个专家只有 C = ⌈k·T·CF / E⌉ 个槽位。
 * 先处理所有 token 的第 1 选择，再处理第 2 选择（GShard 的做法）；同一轮内按位置或按 router 分数（BPR）排队。
 * 洞见：CF 小 → 丢 token；CF 大 → 空槽（padding）浪费算力与显存。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor, toneSoft } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { softmax, topk } from './moeMath'
import { Stat } from './plot'

const T = 12
const E = 4
// 确定性的 router logits：E0 明显偏热
const LOGITS = [
  [2.1, 0.3, 0.9, -0.2],
  [1.2, 1.5, 0.1, 0.0],
  [1.9, 0.2, 0.4, 1.1],
  [0.4, 0.1, 1.6, 0.3],
  [2.4, 1.0, 0.2, 0.1],
  [1.1, 0.2, 0.3, 1.3],
  [1.7, 1.4, 0.0, 0.2],
  [0.2, 1.8, 0.6, 0.1],
  [2.0, 0.1, 1.2, 0.3],
  [1.5, 0.3, 0.2, 0.4],
  [0.3, 0.2, 0.4, 1.9],
  [1.8, 0.9, 1.0, 0.2],
]

type Prio = 'pos' | 'score'

export function CapacityViz() {
  const [cf, setCf] = useState(1.0)
  const [k, setK] = useState<1 | 2>(1)
  const [prio, setPrio] = useState<Prio>('pos')
  const C = Math.ceil((k * T * cf) / E)

  const { slots, status, used } = useMemo(() => {
    const route = LOGITS.map((l) => {
      const p = softmax(l)
      return { p, idx: topk(p, k) }
    })
    const order = Array.from({ length: T }, (_, t) => t)
    if (prio === 'score') order.sort((a, b) => Math.max(...route[b].p) - Math.max(...route[a].p) || a - b)
    const slots: { t: number; r: number }[][] = Array.from({ length: E }, () => [])
    const kept = route.map(() => new Array(k).fill(false))
    for (let r = 0; r < k; r++) {
      for (const t of order) {
        const e = route[t].idx[r]
        if (slots[e].length < C) {
          slots[e].push({ t, r })
          kept[t][r] = true
        }
      }
    }
    const status = kept.map((ks) => (ks.every(Boolean) ? 'full' : ks.some(Boolean) ? 'part' : 'none'))
    const used = slots.reduce((a, s) => a + s.length, 0)
    return { slots, status, used }
  }, [cf, k, prio, C])

  const dropped = T * k - used
  const nNone = status.filter((s) => s === 'none').length

  return (
    <VizFrame
      title="容量因子：丢 token 还是浪费槽位"
      caption={
        <>
          C = ⌈k·T·CF / E⌉ = ⌈{k}×{T}×{cf} / {E}⌉ = <b>{C}</b> 个槽位/专家。{T * k} 次派发中丢弃 <b className="font-mono">{dropped}</b> 次
          {nNone > 0 && <>，有 <b className="text-danger">{nNone}</b> 个 token 一个专家都没进（只能经残差直通）</>}；
          槽位利用率 <b className="font-mono">{((used / (E * C)) * 100).toFixed(0)}%</b>（空槽 = 白算的 padding）。
          {prio === 'score' && ' 按 router 分数排队（BPR）时，最自信的 token 优先占槽。'}
        </>
      }
      legend={
        <Legend
          items={[
            { label: '第 1 选择', tone: 'accent', variant: 'soft' },
            { label: '第 2 选择', tone: 'info', variant: 'soft' },
            { label: '空槽（padding）', tone: 'muted', variant: 'line' },
            { label: 'token 部分派发被丢', tone: 'warning', variant: 'soft' },
            { label: 'token 全部被丢（残差直通）', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Slider className="w-48" label="capacity factor" value={cf} min={0.5} max={2} step={0.25} onChange={setCf} format={(v) => v.toFixed(2)} />
          <Segmented size="sm" value={k} onChange={setK} options={[{ value: 1, label: 'top-1' }, { value: 2, label: 'top-2' }]} />
          <Segmented
            size="sm"
            value={prio}
            onChange={setPrio}
            options={[
              { value: 'pos', label: '按位置排队' },
              { value: 'score', label: '按分数排队（BPR）' },
            ]}
          />
        </>
      }
    >
      <div className="space-y-4">
        <div className="grid grid-cols-4 gap-2">
          {slots.map((s, e) => (
            <div key={e} className="min-w-0 rounded-xl border border-border bg-surface p-1.5">
              <div className="mb-1 text-center text-xs font-semibold text-fg">
                E{e} <span className="font-mono font-normal text-muted">{s.length}/{C}</span>
              </div>
              <div className="flex flex-col gap-1">
                {Array.from({ length: C }, (_, j) => {
                  const it = s[j]
                  return (
                    <div
                      key={j}
                      className="flex h-6 items-center justify-center rounded-md font-mono text-[11px]"
                      style={
                        it
                          ? { background: toneSoft(it.r === 0 ? 'accent' : 'info', 30), border: `1px solid ${toneColor(it.r === 0 ? 'accent' : 'info')}`, color: 'var(--fg)' }
                          : { border: '1px dashed var(--border-strong)', color: 'var(--subtle)' }
                      }
                    >
                      {it ? `t${it.t}` : '空'}
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">各 token 的去向</div>
          <div className="grid grid-cols-6 gap-1 @xl:grid-cols-12">
            {status.map((st, t) => (
              <div
                key={t}
                className="rounded-md py-1 text-center font-mono text-[11px]"
                style={{
                  background: st === 'full' ? toneSoft('success', 22) : st === 'part' ? toneSoft('warning', 28) : toneSoft('danger', 30),
                  border: `1px solid ${toneColor(st === 'full' ? 'success' : st === 'part' ? 'warning' : 'danger')}`,
                }}
                title={st === 'full' ? '全部派发成功' : st === 'part' ? '部分被丢' : '全部被丢，走残差'}
              >
                t{t}
                <div className="text-[10px] text-muted">{st === 'full' ? '完整' : st === 'part' ? '缺 1' : '直通'}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <Stat label="丢弃率" value={`${((dropped / (T * k)) * 100).toFixed(0)}%`} tone={dropped ? 'danger' : 'success'} />
          <Stat label="槽位利用率" value={`${((used / (E * C)) * 100).toFixed(0)}%`} />
          <Stat label="总槽位 E·C" value={E * C} />
        </div>
      </div>
    </VizFrame>
  )
}
