/**
 * 大模型场景：vLLM 风格的前缀缓存块管理（哈希表 + 引用计数 + 双向链表空闲队列，LRU 淘汰）。
 * 洞见：ref_count = 0 的块进入空闲队列但“保留内容”，仍可被命中复用；只有从队头被分配走时才真正淘汰其哈希。
 * 参考：vLLM 文档 Automatic Prefix Caching 设计说明（free_block_queue 为双向链表，请求结束时块按逆序加入队尾）。
 */
import { useMemo } from 'react'
import { Legend, StepControls, VizFrame, toneColor, toneSoft, useStepper, type VizTone } from '@/viz'
import { C, Sub } from './common'

interface Block {
  hash: string | null
  ref: number
}
interface Frame {
  blocks: Block[]
  queue: number[]
  reqs: { name: string; blocks: number[]; done?: boolean }[]
  marks: Record<number, VizTone>
  caption: React.ReactNode
}

function build(): Frame[] {
  const N = 8
  const blocks: Block[] = Array.from({ length: N }, () => ({ hash: null, ref: 0 }))
  let queue = Array.from({ length: N }, (_, i) => i)
  const cached = new Map<string, number>()
  const reqs: Frame['reqs'] = []
  const frames: Frame[] = []
  const push = (caption: React.ReactNode, marks: Record<number, VizTone> = {}) =>
    frames.push({ blocks: blocks.map((b) => ({ ...b })), queue: [...queue], reqs: reqs.map((r) => ({ ...r, blocks: [...r.blocks] })), marks, caption })

  /** 为一个请求分配 prompt 块：命中则复用，否则从空闲队列头取块（必要时淘汰其旧哈希） */
  const alloc = (name: string, hashes: string[]) => {
    const marks: Record<number, VizTone> = {}
    const got: number[] = []
    const notes: string[] = []
    let prefixHit = true
    for (const h of hashes) {
      const hit = prefixHit ? cached.get(h) : undefined
      if (hit != null) {
        if (blocks[hit].ref === 0) {
          queue = queue.filter((x) => x !== hit)
          notes.push(`${h} 命中空闲队列里的 #${hit}，O(1) 从链表中间摘下`)
        } else notes.push(`${h} 命中 #${hit}（共享，ref+1）`)
        blocks[hit].ref++
        marks[hit] = 'accent'
        got.push(hit)
        continue
      }
      prefixHit = false
      const b = queue.shift()!
      if (blocks[b].hash) {
        cached.delete(blocks[b].hash!)
        notes.push(`队头 #${b} 存着 ${blocks[b].hash}，淘汰其哈希后分配给 ${h}`)
        marks[b] = 'danger'
      } else {
        marks[b] = 'success'
      }
      blocks[b] = { hash: h, ref: 1 }
      cached.set(h, b)
      got.push(b)
    }
    reqs.push({ name, blocks: got })
    return { marks, notes }
  }
  const finish = (name: string) => {
    const r = reqs.find((x) => x.name === name)!
    r.done = true
    const marks: Record<number, VizTone> = {}
    for (const b of [...r.blocks].reverse()) {
      blocks[b].ref--
      if (blocks[b].ref === 0) {
        queue.push(b)
        marks[b] = 'warning'
      }
    }
    return marks
  }

  push(<>初始：8 个物理块全部在<b>空闲队列</b>（双向链表）里，哈希表为空。块大小按 vLLM 默认可理解为 16 个 token。</>)
  let r = alloc('A', ['sys', 'A1', 'A2'])
  push(<>请求 A 的 prompt 有 3 个满块（系统提示 sys + A1 + A2），全部未命中：从队头依次取 #0 #1 #2，登记 <C>hash → 块号</C>，ref = 1。</>, r.marks)
  r = alloc('B', ['sys', 'B1'])
  push(<>请求 B 与 A 共享系统提示：<b>{r.notes[0]}</b>；B1 未命中，取队头 #3。共享前缀的 KV 只算一次、只存一份。</>, r.marks)
  let m = finish('A')
  push(<>A 结束：sys 的 ref 2→1（B 还在用，不进队列）；A1、A2 的 ref 归零，<b>按逆序</b>（先 A2 后 A1）加到队尾。它们的 KV 内容和哈希都还在——随时可被命中。</>, m)
  r = alloc('C', ['sys', 'A1', 'C1'])
  push(<>请求 C 的前缀是 sys + A1：{r.notes.join('；')}；C1 取队头 #4。这一步就是“哈希表 O(1) 定位 + 双向链表 O(1) 删除”的组合。</>, r.marks)
  r = alloc('D', ['D1', 'D2', 'D3', 'D4'])
  push(<>请求 D 需要 4 个新块：先取走未缓存的 #5 #6 #7，第 4 块只能取队头的 #2——{r.notes[r.notes.length - 1]}。队头就是<b>最久未被使用</b>的块：LRU。</>, r.marks)
  m = finish('B')
  push(<>B 结束：sys 仍被 C 使用；B1（#3）ref 归零进入队尾。逆序入队的原因：越靠后的块对应越长的前缀，被别的请求复用的概率越低，应该先淘汰。</>, m)
  return frames
}

const toneOf = (b: Block, mark?: VizTone): VizTone | undefined => mark ?? (b.ref > 0 ? 'info' : b.hash ? 'gold' : undefined)

export function PrefixCacheViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 2200 })
  const f = st.frame
  return (
    <VizFrame
      title="前缀缓存：引用计数 + LRU 空闲队列"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '使用中 ref>0', tone: 'info', variant: 'soft' },
            { label: '空闲但仍缓存', tone: 'gold', variant: 'soft' },
            { label: '命中复用', tone: 'accent', variant: 'soft' },
            { label: '新分配', tone: 'success', variant: 'soft' },
            { label: '淘汰旧内容', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="flex min-w-0 flex-col gap-4">
        <div>
          <Sub>物理 KV 块（块号 · 内容哈希 · ref_count）</Sub>
          <div className="grid grid-cols-4 gap-1.5 @2xl:grid-cols-8">
            {f.blocks.map((b, i) => {
              const tone = toneOf(b, f.marks[i])
              return (
                <div
                  key={i}
                  className="flex flex-col items-center rounded-lg border-2 px-1 py-1.5 font-mono leading-tight transition-colors duration-300"
                  style={{ borderColor: tone ? toneColor(tone) : 'var(--border)', background: tone ? toneSoft(tone, 22) : 'var(--surface)' }}
                >
                  <span className="text-[10.5px] text-muted">#{i}</span>
                  <span className="text-[13px] font-semibold text-fg">{b.hash ?? '—'}</span>
                  <span className="text-[10.5px] text-muted">ref={b.ref}</span>
                </div>
              )
            })}
          </div>
        </div>
        <div className="grid gap-4 @xl:grid-cols-2">
          <div className="min-w-0">
            <Sub>空闲队列（双向链表，队头先被淘汰）</Sub>
            <div className="flex flex-wrap items-center gap-1 font-mono text-[11.5px]">
              <span className="text-subtle">head</span>
              {f.queue.map((b) => (
                <span key={b} className="flex items-center gap-1">
                  <span className="text-subtle">⇄</span>
                  <span
                    className="rounded-md border px-1.5 py-0.5"
                    style={{ borderColor: f.blocks[b].hash ? toneColor('gold') : 'var(--border)', background: f.blocks[b].hash ? toneSoft('gold', 18) : 'var(--surface)' }}
                  >
                    #{b}
                    {f.blocks[b].hash ? `(${f.blocks[b].hash})` : ''}
                  </span>
                </span>
              ))}
              <span className="text-subtle">⇄ tail</span>
            </div>
          </div>
          <div className="min-w-0">
            <Sub>请求 → 块表（block table）</Sub>
            <div className="flex flex-col gap-1 font-mono text-[11.5px]">
              {f.reqs.length === 0 && <span className="text-subtle">（暂无请求）</span>}
              {f.reqs.map((r) => (
                <span key={r.name} className={r.done ? 'text-subtle line-through' : 'text-fg'}>
                  {r.name}: [{r.blocks.map((b) => `#${b}`).join(', ')}]{r.done ? ' 已结束' : ''}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
