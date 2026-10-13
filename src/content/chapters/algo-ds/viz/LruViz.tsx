/**
 * LRU 缓存：哈希表 + 双向链表 双视图逐帧演示（LC146）。
 * 洞见：哈希表负责 O(1) “找到节点”，双向链表负责 O(1) “调整顺序 / 删除表尾”；节点里必须存 key，淘汰时才能反查删除哈希项。
 */
import { useMemo, useState } from 'react'
import { motion } from 'motion/react'
import { Shuffle } from 'lucide-react'
import { CodeTrace, Legend, StepControls, VizFrame, catColor, toneColor, toneSoft, useStepper, type VizTone } from '@/viz'
import { Button, Slider } from '@/components/ui'
import { rng } from '@/lib/utils'
import { C, Sub } from './common'

const CODE = `class LRUCache:
    def get(self, key):
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)          # 从原位置摘下
        self._add_front(node)       # 插到表头（最近使用）
        return node.val

    def put(self, key, value):
        if key in self.map:
            node = self.map[key]
            node.val = value
            self._remove(node); self._add_front(node)
            return
        if len(self.map) == self.cap:
            lru = self.tail.prev    # 表尾 = 最久未使用
            self._remove(lru)
            del self.map[lru.key]   # 节点里存 key 的原因
        node = Node(key, value)
        self._add_front(node)
        self.map[key] = node`

type Op = { kind: 'get'; key: number } | { kind: 'put'; key: number; val: number }

interface Frame {
  op: number
  order: number[] // MRU → LRU
  vals: Record<number, number>
  hl?: { key: number; tone: VizTone }
  lines: number[]
  results: (number | null)[]
  caption: React.ReactNode
}

const PRESET: Op[] = [
  { kind: 'put', key: 1, val: 1 },
  { kind: 'put', key: 2, val: 2 },
  { kind: 'get', key: 1 },
  { kind: 'put', key: 3, val: 3 },
  { kind: 'get', key: 2 },
  { kind: 'put', key: 4, val: 4 },
  { kind: 'get', key: 1 },
  { kind: 'get', key: 3 },
  { kind: 'get', key: 4 },
]

const opText = (o: Op) => (o.kind === 'get' ? `get(${o.key})` : `put(${o.key},${o.val})`)

function build(ops: Op[], cap: number): Frame[] {
  let order: number[] = []
  const vals: Record<number, number> = {}
  const results: (number | null)[] = ops.map(() => null)
  const frames: Frame[] = []
  const push = (op: number, lines: number[], caption: React.ReactNode, hl?: Frame['hl']) =>
    frames.push({ op, order: [...order], vals: { ...vals }, hl, lines, results: [...results], caption })
  push(-1, [], <>容量 <C>capacity = {cap}</C>。链表靠近 head 的一端是<b>最近使用</b>，靠近 tail 的一端是<b>最久未使用</b>。</>)
  ops.forEach((o, i) => {
    const has = order.includes(o.key)
    if (o.kind === 'get') {
      if (!has) {
        results[i] = -1
        push(i, [3, 4], <><C>get({o.key})</C>：哈希表里没有 key {o.key}，O(1) 判定未命中，返回 <b>-1</b>。链表不变。</>)
        return
      }
      push(i, [5], <><C>get({o.key})</C>：哈希表 O(1) 直接拿到节点（不用遍历链表），值为 {vals[o.key]}。</>, { key: o.key, tone: 'accent' })
      order = [o.key, ...order.filter((k) => k !== o.key)]
      results[i] = vals[o.key]
      push(i, [6, 7, 8], <>命中后把节点摘下、插到表头：它成了<b>最近使用</b>的。返回 {vals[o.key]}。双向链表让“摘下”只需改 4 个指针。</>, { key: o.key, tone: 'success' })
      return
    }
    if (has) {
      vals[o.key] = o.val
      push(i, [11, 12, 13], <><C>put({o.key},{o.val})</C>：key 已存在，原地更新值。</>, { key: o.key, tone: 'accent' })
      order = [o.key, ...order.filter((k) => k !== o.key)]
      push(i, [14, 15], <>更新也算一次“使用”：移到表头。</>, { key: o.key, tone: 'success' })
      return
    }
    if (order.length === cap) {
      const lru = order[order.length - 1]
      push(i, [16, 17], <><C>put({o.key},{o.val})</C>：缓存已满（{cap} 个）。表尾 <C>tail.prev</C> 是最久未使用的 key {lru}，准备淘汰。</>, { key: lru, tone: 'danger' })
      order = order.slice(0, -1)
      delete vals[lru]
      push(i, [18, 19], <>从链表摘下 key {lru}，再用<b>节点里存的 key</b> 删除哈希表项 <C>del map[{lru}]</C>——节点不存 key 就无从删起。</>)
    }
    vals[o.key] = o.val
    order = [o.key, ...order]
    push(i, [20, 21, 22], <>新建节点 ({o.key}, {o.val})，插到表头并登记到哈希表。</>, { key: o.key, tone: 'success' })
  })
  return frames
}

function randomOps(seed: number): Op[] {
  const r = rng(seed)
  return Array.from({ length: 10 }, () => {
    const key = 1 + Math.floor(r() * 5)
    return r() < 0.5 ? { kind: 'get' as const, key } : { kind: 'put' as const, key, val: 10 + Math.floor(r() * 90) }
  })
}

const Box = ({ k, v, tone, sentinel }: { k?: number; v?: number; tone?: VizTone; sentinel?: string }) => (
  <motion.div
    layout
    transition={{ type: 'spring', stiffness: 320, damping: 30 }}
    className="flex h-10 w-11 shrink-0 flex-col items-center justify-center rounded-lg border-2 font-mono leading-none @md:w-12"
    style={{
      borderStyle: sentinel ? 'dashed' : 'solid',
      borderColor: tone ? toneColor(tone) : sentinel ? 'var(--border-strong)' : catColor((k ?? 1) - 1),
      background: tone ? toneSoft(tone, 26) : 'var(--surface)',
    }}
  >
    {sentinel ? (
      <span className="text-[10.5px] text-muted">{sentinel}</span>
    ) : (
      <>
        <span className="text-[12.5px] font-semibold text-fg">{k}</span>
        <span className="mt-0.5 text-[10px] text-muted">={v}</span>
      </>
    )}
  </motion.div>
)

export function LruViz() {
  const [seed, setSeed] = useState(0)
  const [cap, setCap] = useState(2)
  const ops = useMemo(() => (seed === 0 ? PRESET : randomOps(seed)), [seed])
  const frames = useMemo(() => build(ops, cap), [ops, cap])
  const st = useStepper(frames, { intervalMs: 1300 })
  const f = st.frame
  const keys = [...f.order].sort((a, b) => a - b)
  const toneOf = (k: number) => (f.hl?.key === k ? f.hl.tone : undefined)

  return (
    <VizFrame
      title="LRU：哈希表定位 + 双向链表排序"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '正在访问', tone: 'accent', variant: 'soft' },
            { label: '移到表头 / 新插入', tone: 'success', variant: 'soft' },
            { label: '将被淘汰', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Slider className="w-40" label="容量" value={cap} min={2} max={4} onChange={setCap} />
          <Button size="sm" variant="ghost" icon={<Shuffle size={14} />} onClick={() => setSeed((s) => s + 1)}>
            随机操作序列
          </Button>
          {seed !== 0 && (
            <Button size="sm" variant="ghost" onClick={() => setSeed(0)}>
              恢复 LC146 示例
            </Button>
          )}
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-4">
          <div>
            <Sub>操作序列</Sub>
            <div className="flex flex-wrap gap-1.5">
              {ops.map((o, i) => (
                <span
                  key={i}
                  className="rounded-md border px-1.5 py-0.5 font-mono text-[11.5px]"
                  style={{
                    borderColor: i === f.op ? 'var(--accent)' : 'var(--border)',
                    background: i === f.op ? toneSoft('accent', 18) : 'var(--surface)',
                    opacity: i > f.op ? 0.55 : 1,
                  }}
                >
                  {opText(o)}
                  {f.results[i] != null && <b className={f.results[i] === -1 ? 'text-danger' : 'text-success'}> →{f.results[i]}</b>}
                </span>
              ))}
            </div>
          </div>
          <div className="grid gap-4 @md:grid-cols-[auto_1fr]">
            <div>
              <Sub>哈希表 key → 节点</Sub>
              <div className="flex min-h-[44px] flex-wrap gap-1.5 @md:flex-col">
                {keys.length === 0 && <span className="text-xs text-subtle">（空）</span>}
                {keys.map((k) => (
                  <div
                    key={k}
                    className="flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[12px]"
                    style={{ borderColor: toneOf(k) ? toneColor(toneOf(k)!) : 'var(--border)', background: toneOf(k) ? toneSoft(toneOf(k)!, 18) : 'var(--surface)' }}
                  >
                    <b className="text-fg">{k}</b>
                    <span className="text-subtle">→</span>
                    <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: catColor(k - 1) }} />
                  </div>
                ))}
              </div>
            </div>
            <div className="min-w-0">
              <Sub>双向链表（head 侧 = 最近使用）</Sub>
              <div className="flex items-center gap-0.5 overflow-x-auto py-1">
                <Box sentinel="head" />
                {f.order.map((k) => (
                  <motion.div key={k} layout className="flex items-center gap-0.5" transition={{ type: 'spring', stiffness: 320, damping: 30 }}>
                    <span className="w-3.5 text-center text-[11px] text-subtle">⇄</span>
                    <Box k={k} v={f.vals[k]} tone={toneOf(k)} />
                  </motion.div>
                ))}
                <span className="w-3.5 text-center text-[11px] text-subtle">⇄</span>
                <Box sentinel="tail" />
              </div>
              <div className="mt-1 text-[11px] text-subtle">
                {f.order.length}/{cap} 已用 · 淘汰时取 <C>tail.prev</C>
              </div>
            </div>
          </div>
        </div>
        <CodeTrace code={CODE} active={f.lines} maxHeight={300} />
      </div>
    </VizFrame>
  )
}
