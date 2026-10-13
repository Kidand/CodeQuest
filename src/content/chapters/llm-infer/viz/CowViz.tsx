/**
 * 写时复制（copy-on-write）：parallel sampling 共享 prompt 的物理块，引用计数 > 1 的块被写入时才复制。
 * 分配器语义与本关实战题一致：总是取编号最小的空闲块；满块共享永不复制；释放时引用计数归零才回收。
 */
import { useMemo } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, catColor, toneColor, useStepper } from '@/viz'
import { C } from './common'

const B = 4
const NB = 8
const PROMPT = ['用', '一', '句', '话', '描', '述', '你', '眼', '中', '的']
const CONT = [
  ['春', '天', '很', '暖'],
  ['秋', '天', '很', '凉'],
]

const CODE = `def append_token(seq):
    pos = length[seq]
    if pos % B == 0:                     # 最后一块已满 → 直接分配新块
        table[seq].append(alloc())
    else:
        last = table[seq][-1]
        if ref[last] > 1:                # 共享块被写 → 写时复制
            new = alloc(); copy(last, new)
            ref[last] -= 1
            table[seq][-1] = new
    length[seq] += 1

def fork(parent, child):                 # 共享全部块，只加引用计数
    table[child] = list(table[parent])
    for b in table[child]: ref[b] += 1`

interface Frame {
  tables: number[][]
  lens: number[]
  ref: number[]
  content: (string | null)[] // 物理槽位内容
  alive: boolean[]
  copy?: [number, number]
  touched?: number[]
  line: number[]
  caption: React.ReactNode
}

function build(): Frame[] {
  const tables: number[][] = [[], []]
  const lens = [0, 0]
  const ref = Array(NB).fill(0)
  const content: (string | null)[] = Array(NB * B).fill(null)
  const alive = [true, false]
  const toks: string[][] = [[...PROMPT], [...PROMPT]]
  const frames: Frame[] = []
  const alloc = () => {
    const b = ref.findIndex((r) => r === 0)
    ref[b] = 1
    return b
  }
  const snap = (rest: Partial<Frame> & { line: number[]; caption: React.ReactNode }): Frame => ({
    tables: tables.map((t) => [...t]),
    lens: [...lens],
    ref: [...ref],
    content: [...content],
    alive: [...alive],
    ...rest,
  })
  const append = (s: number, tok: string): { copy?: [number, number]; touched: number } => {
    const pos = lens[s]
    let copy: [number, number] | undefined
    if (pos % B === 0) tables[s].push(alloc())
    else {
      const last = tables[s][tables[s].length - 1]
      if (ref[last] > 1) {
        const nw = alloc()
        for (let o = 0; o < B; o++) content[nw * B + o] = content[last * B + o]
        ref[last] -= 1
        tables[s][tables[s].length - 1] = nw
        copy = [last, nw]
      }
    }
    toks[s][pos] = tok
    const phys = tables[s][Math.floor(pos / B)]
    content[phys * B + (pos % B)] = tok
    lens[s]++
    return { copy, touched: phys }
  }

  PROMPT.forEach((t) => append(0, t))
  frames.push(
    snap({
      touched: [0, 1, 2],
      line: [],
      caption: <>prompt 共 10 个 token，prefill 后占物理块 #0、#1（满）和 #2（2/4）。要并行采样 n = 2 个回答。</>,
    }),
  )
  tables[1] = [...tables[0]]
  lens[1] = lens[0]
  tables[1].forEach((b) => ref[b]++)
  alive[1] = true
  frames.push(
    snap({
      line: [13, 14, 15],
      caption: <>fork：样本 2 复制的只是<b>块表</b>（几个整数），三个块的引用计数都变为 2。prompt 的 KV 没有被复制，显存一字节都没多用。</>,
    }),
  )
  let r = append(0, CONT[0][0])
  frames.push(
    snap({
      copy: r.copy,
      touched: [r.touched],
      line: [5, 6, 7, 8, 9, 10],
      caption: (
        <>
          样本 1 要写 pos 10，落在共享的未满块 #{r.copy![0]}（ref = 2）里。直接写会污染样本 2 → <b>写时复制</b>：分配 #{r.copy![1]}，拷贝块里已有的 2 个 token，再写入「{CONT[0][0]}」。#{r.copy![0]} 的
          ref 降为 1。
        </>
      ),
    }),
  )
  r = append(1, CONT[1][0])
  frames.push(
    snap({
      touched: [r.touched],
      line: [6, 7, 11],
      caption: <>样本 2 写 pos 10：块 #{r.touched} 的 ref 已经是 1，独占，原地写入，<b>不需要再复制</b>。整个过程只复制了 1 个块。</>,
    }),
  )
  const r1 = append(0, CONT[0][1])
  const r2 = append(1, CONT[1][1])
  frames.push(snap({ touched: [r1.touched, r2.touched], line: [6, 7, 11], caption: <>两个样本各自在独占块里原地写入，最后一块都写满了。</> }))
  const r3 = append(0, CONT[0][2])
  const r4 = append(1, CONT[1][2])
  frames.push(
    snap({
      touched: [r3.touched, r4.touched],
      line: [3, 4],
      caption: (
        <>
          pos 12 是块边界：各自分配新块 #{r3.touched}、#{r4.touched}。注意 #0、#1 一直是两个样本共享的满块，<C>ref = 2</C>——<b>满块永远只读，不会触发复制</b>。
        </>
      ),
    }),
  )
  append(0, CONT[0][3])
  append(1, CONT[1][3])
  const used = ref.filter((x) => x > 0).length
  const noShare = Math.ceil(lens[0] / B) + Math.ceil(lens[1] / B)
  frames.push(
    snap({
      line: [6, 7, 11],
      caption: (
        <>
          共享前缀后两个样本只占 {used} 个物理块；如果各自存一份完整 KV 需要 {noShare} 块。prompt 越长、样本越多（beam search、n 很大的采样），省得越多。
        </>
      ),
    }),
  )
  // 释放样本 2（例如 beam 剪枝，或用户只要最好的一个）
  const freed: number[] = []
  tables[1].forEach((b) => {
    ref[b]--
    if (ref[b] === 0) {
      freed.push(b)
      for (let o = 0; o < B; o++) content[b * B + o] = null
    }
  })
  tables[1] = []
  alive[1] = false
  frames.push(
    snap({
      touched: freed,
      line: [],
      caption: (
        <>
          样本 2 被丢弃（如 beam search 剪枝）：它的每个块 ref − 1。#{freed.join('、#')} 归零被回收；#0、#1 仍被样本 1 引用（ref = 1），保留。这就是引用计数的意义。
        </>
      ),
    }),
  )
  void toks
  return frames
}

export function CowViz() {
  const frames = useMemo(build, [])
  const st = useStepper(frames, { intervalMs: 1800 })
  const f = st.frame
  return (
    <VizFrame
      title="写时复制：共享 prompt 的 KV"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '样本 1', tone: 1, variant: 'soft' },
            { label: '样本 2', tone: 2, variant: 'soft' },
            { label: '共享块（ref > 1）', tone: 'gold', variant: 'soft' },
            { label: '刚复制出的块', tone: 'success', variant: 'line' },
          ]}
        />
      }
      controls={<StepControls stepper={st} />}
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-2">
          {[0, 1].map((s) => (
            <div key={s} className="rounded-lg border border-border bg-surface p-2" style={{ opacity: f.alive[s] ? 1 : 0.4 }}>
              <div className="mb-1 flex justify-between font-mono text-[11.5px]">
                <span style={{ color: catColor(s) }}>
                  样本 {s + 1} · len = {f.alive[s] ? f.lens[s] : 0}
                  {!f.alive[s] && (s === 1 && f.lens[0] > 10 ? '（已释放）' : '（未创建）')}
                </span>
                <span className="text-muted">table = [{f.tables[s].join(', ')}]</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {f.tables[s].map((b, lb) => (
                  <span
                    key={lb}
                    className="rounded border px-1.5 py-0.5 font-mono text-[11px]"
                    style={{
                      borderColor: f.ref[b] > 1 ? toneColor('gold') : catColor(s),
                      background: f.ref[b] > 1 ? `color-mix(in oklab, ${toneColor('gold')} 18%, var(--surface))` : 'var(--surface)',
                    }}
                  >
                    逻辑 {lb} → #{b}
                  </span>
                ))}
              </div>
            </div>
          ))}
          <CodeTrace code={CODE} active={f.line} maxHeight={260} />
        </div>
        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium text-muted">物理块（右上角为引用计数）</div>
          <div className="grid grid-cols-2 gap-2">
            {Array.from({ length: NB }, (_, b) => {
              const r = f.ref[b]
              const owners = [0, 1].filter((s) => f.tables[s].includes(b))
              const isCopy = f.copy?.[1] === b
              const isSrc = f.copy?.[0] === b
              const tone = r > 1 ? toneColor('gold') : owners.length ? catColor(owners[0]) : 'var(--border)'
              return (
                <div
                  key={b}
                  className="rounded-lg border p-1.5 transition-colors duration-300"
                  style={{
                    borderColor: isCopy ? toneColor('success') : tone,
                    borderWidth: isCopy || f.touched?.includes(b) ? 2 : 1,
                    opacity: r === 0 ? 0.55 : 1,
                  }}
                >
                  <div className="mb-1 flex justify-between font-mono text-[10.5px]">
                    <span className="text-muted">
                      #{b}
                      {isCopy && <span className="text-success"> ← 复制自 #{f.copy![0]}</span>}
                      {isSrc && <span className="text-warning"> 源</span>}
                    </span>
                    <span style={{ color: r > 1 ? toneColor('gold') : 'var(--muted)' }}>{r ? `ref=${r}` : '空闲'}</span>
                  </div>
                  <div className="grid grid-cols-4 gap-0.5">
                    {Array.from({ length: B }, (_, o) => {
                      const v = f.content[b * B + o]
                      return (
                        <div
                          key={o}
                          className="flex h-7 items-center justify-center rounded border text-[12px]"
                          style={{
                            background: v ? `color-mix(in oklab, ${tone} 20%, var(--surface))` : 'var(--surface-2)',
                            borderColor: 'var(--border)',
                          }}
                        >
                          {v ?? ''}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
