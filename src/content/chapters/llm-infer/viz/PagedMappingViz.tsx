/**
 * 块表（block table）：逻辑块 → 物理块的映射动画。
 * 两个请求交替解码，最后一块写满才从空闲链表取新物理块；请求结束后物理块归还，被新请求复用。
 * 每帧高亮刚写入的槽位，并给出 slot = block_table[pos // B] * B + pos % B 的计算过程。
 */
import { useMemo } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, catColor, toneColor, useStepper } from '@/viz'
import { C } from './common'

const B = 4
const NB = 8
const SEQS = [
  { id: 'A', toks: ['今天', '天气', '很', '好', '，', '我们', '去', '公园', '散步', '吧'], prompt: 3 },
  { id: 'B', toks: ['推理', '系统', '的', '核心', '是', '显存', '管理'], prompt: 4 },
  { id: 'C', toks: ['好', '的', '！', '出发'], prompt: 2 },
]

const CODE = `def append_tokens(seq, tokens):
    for tok in tokens:
        pos = seq.len
        if pos % B == 0:                         # 最后一块已满（或还没有块）
            seq.block_table.append(free_list.pop(0))
        phys = seq.block_table[pos // B]         # 逻辑块 → 物理块
        slot = phys * B + pos % B                # slot_mapping
        kv_pool[slot] = kv(tok)                  # 写入 K/V
        seq.len += 1

def free(seq):                                   # 请求结束：物理块归还
    free_list[:0] = seq.block_table`

interface SeqState {
  len: number
  table: number[]
  active: boolean
}
interface Frame {
  seqs: SeqState[]
  pool: ({ s: number; pos: number } | null)[] // 每个物理槽位存的是哪个序列的第几个 token
  free: number[]
  hl: { s: number; pos: number; slot: number; phys: number }[]
  newBlock?: number[]
  freed?: number[]
  line: number[]
  caption: React.ReactNode
}

function build(): Frame[] {
  const seqs: SeqState[] = SEQS.map(() => ({ len: 0, table: [], active: false }))
  const pool: Frame['pool'] = Array(NB * B).fill(null)
  let free = [5, 1, 6, 2, 7, 0, 3, 4]
  const frames: Frame[] = []
  const snap = (rest: Omit<Frame, 'seqs' | 'pool' | 'free'>): Frame => ({
    seqs: seqs.map((x) => ({ ...x, table: [...x.table] })),
    pool: [...pool],
    free: [...free],
    ...rest,
  })
  const append = (s: number, n: number) => {
    const hl: Frame['hl'] = []
    const newBlock: number[] = []
    for (let k = 0; k < n; k++) {
      const st = seqs[s]
      const pos = st.len
      if (pos % B === 0) {
        const b = free.shift()!
        st.table.push(b)
        newBlock.push(b)
      }
      const phys = st.table[Math.floor(pos / B)]
      const slot = phys * B + (pos % B)
      pool[slot] = { s, pos }
      st.len++
      hl.push({ s, pos, slot, phys })
    }
    return { hl, newBlock }
  }
  const release = (s: number) => {
    const st = seqs[s]
    st.table.forEach((b) => {
      for (let o = 0; o < B; o++) pool[b * B + o] = null
    })
    const freed = [...st.table]
    free = [...freed, ...free]
    st.table = []
    st.active = false
    return freed
  }
  const slotText = (h: Frame['hl'][number]) => (
    <C>
      pos {h.pos} → 逻辑块 {Math.floor(h.pos / B)} → 物理块 #{h.phys} → slot {h.phys}×{B}+{h.pos % B} = {h.slot}
    </C>
  )

  frames.push(snap({ hl: [], line: [], caption: <>显存里有 {NB} 个物理块，每块存 {B} 个 token 的 K/V。空闲链表的顺序是任意的——物理块不需要连续。</> }))

  seqs[0].active = true
  let r = append(0, SEQS[0].prompt)
  frames.push(
    snap({
      ...r,
      line: [4, 5, 6, 7],
      caption: (
        <>
          请求 A 的 prompt（3 个 token）做 prefill：需要 ⌈3/4⌉ = 1 块，从空闲链表取出 #{r.newBlock[0]}，块表 A = [{seqs[0].table.join(', ')}]。最后一个 token：{slotText(r.hl[2])}
        </>
      ),
    }),
  )
  seqs[1].active = true
  r = append(1, SEQS[1].prompt)
  frames.push(snap({ ...r, line: [4, 5, 6, 7], caption: <>请求 B 的 prompt 恰好 4 个 token，占满一整块 #{r.newBlock[0]}。块表 B = [{seqs[1].table.join(', ')}]。</> }))

  // 交替解码
  const decodeStep = (ids: number[], extra?: (hl: Frame['hl']) => React.ReactNode) => {
    const hl: Frame['hl'] = []
    const nb: number[] = []
    ids.forEach((s) => {
      const x = append(s, 1)
      hl.push(...x.hl)
      nb.push(...x.newBlock)
    })
    return { hl, newBlock: nb, extra: extra?.(hl) }
  }
  let d = decodeStep([0, 1])
  frames.push(
    snap({
      hl: d.hl,
      newBlock: d.newBlock,
      line: [3, 4, 5, 6, 7],
      caption: (
        <>
          解码一步：A 写在第 0 块的第 4 个槽位（块内还有空位，无需分配）；B 的第 0 块已满（pos 4 % 4 == 0），取新块 #{d.newBlock[0]}。{slotText(d.hl[1])}
        </>
      ),
    }),
  )
  d = decodeStep([0, 1])
  frames.push(
    snap({
      hl: d.hl,
      newBlock: d.newBlock,
      line: [4, 5],
      caption: <>A 的第 0 块写满，pos 4 触发分配新块 #{d.newBlock[0]}。注意 A 的两个逻辑块对应的物理块 #{seqs[0].table.join('、#')} 并不相邻。</>,
    }),
  )
  d = decodeStep([0, 1])
  frames.push(snap({ hl: d.hl, line: [6, 7, 8], caption: <>A、B 各写一个 token，都落在已有块里。{slotText(d.hl[0])}</> }))
  const freedB = release(1)
  frames.push(
    snap({
      hl: [],
      freed: freedB,
      line: [11, 12],
      caption: <>B 生成结束，物理块 #{freedB.join('、#')} 整块归还到空闲链表头部。没有任何「洞」需要整理：块是统一大小的，谁都能用。</>,
    }),
  )
  for (let k = 0; k < 2; k++) {
    d = decodeStep([0])
    frames.push(snap({ hl: d.hl, newBlock: d.newBlock, line: [6, 7, 8], caption: <>A 继续解码。{slotText(d.hl[0])}</> }))
  }
  d = decodeStep([0])
  frames.push(
    snap({
      hl: d.hl,
      newBlock: d.newBlock,
      line: [4, 5],
      caption: <>A 的 pos 8 需要第 3 个逻辑块，拿到的正是 B 刚释放的 #{d.newBlock[0]}——物理块被立即复用。</>,
    }),
  )
  d = decodeStep([0])
  frames.push(snap({ hl: d.hl, line: [6, 7, 8], caption: <>A 生成完毕：10 个 token 占 3 块（12 个槽位），只浪费最后一块里的 2 个槽位。</> }))
  const freedA = release(0)
  seqs[2].active = true
  r = append(2, SEQS[2].prompt)
  frames.push(
    snap({
      ...r,
      freed: freedA,
      line: [4, 5, 11, 12],
      caption: (
        <>
          A 释放 #{freedA.join('、#')}；新请求 C 到达，prefill 2 个 token 拿到 #{r.newBlock[0]}。整个过程中逻辑上连续的序列，在物理上可以分散在任意块里。
        </>
      ),
    }),
  )
  return frames
}

function Slot({ v, hl, tone }: { v: string | null; hl?: boolean; tone?: number }) {
  return (
    <div
      className="flex h-7 min-w-0 items-center justify-center truncate rounded border px-0.5 text-[11px] transition-colors duration-300"
      style={{
        background: v ? `color-mix(in oklab, ${catColor(tone ?? 0)} ${hl ? 55 : 22}%, var(--surface))` : 'var(--surface-2)',
        borderColor: hl ? toneColor('gold') : v ? catColor(tone ?? 0) : 'var(--border)',
        borderWidth: hl ? 2 : 1,
        color: 'var(--fg)',
      }}
    >
      {v ?? ''}
    </div>
  )
}

export function PagedMappingViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 1500 })
  const f = st.frame
  const hlSlots = new Set(f.hl.map((h) => h.slot))

  return (
    <VizFrame
      title="块表：逻辑块 → 物理块"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '请求 A', tone: 1, variant: 'soft' },
            { label: '请求 B', tone: 2, variant: 'soft' },
            { label: '请求 C', tone: 3, variant: 'soft' },
            { label: '本步写入', tone: 'gold', variant: 'line' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-3">
          <div className="text-xs font-medium text-muted">逻辑视图（每个请求看到的连续序列）+ 块表</div>
          {SEQS.map((sq, s) => {
            const x = f.seqs[s]
            if (!x.active && x.len === 0) return null
            const nLogical = Math.max(1, Math.ceil(x.len / B))
            return (
              <div key={sq.id} className="rounded-lg border border-border bg-surface p-2" style={{ opacity: x.active ? 1 : 0.45 }}>
                <div className="mb-1 flex items-center justify-between font-mono text-[11.5px]">
                  <span style={{ color: catColor(s) }}>
                    请求 {sq.id} · len = {x.len}
                    {!x.active && '（已结束）'}
                  </span>
                  <span className="text-muted">block_table = [{x.table.join(', ')}]</span>
                </div>
                <div className="space-y-1">
                  {Array.from({ length: nLogical }, (_, lb) => (
                    <div key={lb} className="grid items-center gap-1" style={{ gridTemplateColumns: `3.4rem repeat(${B}, minmax(0,1fr)) 2.6rem` }}>
                      <span className="font-mono text-[10.5px] text-subtle">逻辑块 {lb}</span>
                      {Array.from({ length: B }, (_, o) => {
                        const pos = lb * B + o
                        const has = pos < x.len
                        return <Slot key={o} v={has ? sq.toks[pos] : null} tone={s} hl={f.hl.some((h) => h.s === s && h.pos === pos)} />
                      })}
                      <span className="text-right font-mono text-[11px]" style={{ color: catColor(s) }}>
                        → #{x.table[lb] ?? '—'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
          <CodeTrace code={CODE} active={f.line} maxHeight={250} />
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium text-muted">物理 KV 池（{NB} 块 × {B} 槽位）</div>
          <div className="grid grid-cols-2 gap-2">
            {Array.from({ length: NB }, (_, b) => {
              const isNew = f.newBlock?.includes(b)
              const isFreed = f.freed?.includes(b)
              return (
                <div
                  key={b}
                  className="rounded-lg border p-1.5 transition-colors duration-300"
                  style={{ borderColor: isNew ? toneColor('success') : isFreed ? toneColor('danger') : 'var(--border)', borderStyle: isFreed ? 'dashed' : 'solid' }}
                >
                  <div className="mb-1 flex justify-between font-mono text-[10.5px] text-muted">
                    <span>#{b}</span>
                    <span>{isNew ? '新分配' : isFreed ? '已释放' : f.free.includes(b) ? '空闲' : ''}</span>
                  </div>
                  <div className="grid grid-cols-4 gap-0.5">
                    {Array.from({ length: B }, (_, o) => {
                      const c = f.pool[b * B + o]
                      return <Slot key={o} v={c ? SEQS[c.s].toks[c.pos] : null} tone={c?.s} hl={hlSlots.has(b * B + o)} />
                    })}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="mt-2 font-mono text-[11.5px] text-muted">
            free_list = [{f.free.map((b) => `#${b}`).join(', ')}]
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
