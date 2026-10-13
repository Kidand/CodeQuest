/**
 * Hash 路由：专家只由 token id 查表决定（DeepSeek-V4 前 3 个 MoE 层：indices = tid2eid[input_ids]）。
 * 洞见：路由与学习无关 → 不会坍缩、计算前就已知；但 token 频率服从 Zipf，表怎么建决定了负载是否均衡。
 * V4 论文只说「预定义的、关于 token id 的哈希函数」，表的构造方法未详述——这里对比两种可能的构造。
 */
import { useMemo, useState } from 'react'
import { Legend, TokenRow, VizFrame, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { maxVio } from './moeMath'
import { LoadBars, Stat } from './plot'

const V = 32
const E = 8
const WORDS = ['the', ',', '.', 'of', 'and', 'to', 'a', 'in', 'is', 'that', 'for', 'it', 'with', 'as', 'was', 'on', 'be', 'by', 'this', 'are', 'at', 'from', 'or', 'an', 'model', 'expert', 'token', 'layer', 'route', 'sparse', 'hash', 'MoE']
const SENT = [0, 26, 8, 31, 4, 0, 25, 1, 28, 7, 0, 27, 2]

type Mode = 'hash' | 'greedy'

/** 固定的整数哈希（乘法 + 移位混合，取高位） */
const hashId = (id: number, salt: number) => {
  let h = Math.imul(id + 1, 0x9e3779b1) ^ Math.imul(salt + 1, 0x85ebca6b)
  h ^= h >>> 15
  h = Math.imul(h, 0xc2b2ae35)
  h ^= h >>> 13
  return Math.floor(((h >>> 0) / 2 ** 32) * E)
}

function buildTable(mode: Mode, freq: number[], k: number) {
  if (mode === 'hash') {
    return freq.map((_, id) => {
      const out: number[] = []
      let salt = 0
      while (out.length < k) {
        const e = hashId(id, salt++)
        if (!out.includes(e)) out.push(e)
      }
      return out
    })
  }
  // 按词频从高到低，每次把 token 的 k 个副本放到当前最空的 k 个专家（LPT 贪心）
  const load = new Array(E).fill(0)
  const table: number[][] = new Array(V)
  const order = freq.map((f, id) => [f, id] as const).sort((a, b) => b[0] - a[0])
  for (const [f, id] of order) {
    const pick = load.map((l, e) => [l, e] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, k).map(([, e]) => e)
    pick.forEach((e) => (load[e] += f))
    table[id] = pick
  }
  return table
}

export function HashRoutingViz() {
  const [mode, setMode] = useState<Mode>('hash')
  const [s, setS] = useState(1.0)
  const [k, setK] = useState(2)
  const freq = useMemo(() => {
    const raw = Array.from({ length: V }, (_, r) => 1 / (r + 1) ** s)
    const z = raw.reduce((a, b) => a + b, 0)
    return raw.map((v) => v / z)
  }, [s])
  const table = useMemo(() => buildTable(mode, freq, k), [mode, freq, k])
  const load = useMemo(() => {
    const l = new Array(E).fill(0)
    table.forEach((es, id) => es.forEach((e) => (l[e] += freq[id] * 100)))
    return l
  }, [table, freq])
  const mean = (100 * k) / E
  const vio = maxVio(load)

  return (
    <VizFrame
      title="Hash 路由：查表决定专家，负载看表怎么建"
      caption={
        <>
          每个 token 固定送往 tid2eid[id] 中的 {k} 个专家，router 不参与选择（V4 中门控权重仍由 router 分数计算）。{mode === 'hash' ? '按 id 随机哈希' : '按词频贪心装箱'}时
          最忙的专家负载是均值的 <b className="font-mono">{(1 + vio).toFixed(2)}×</b>。
          {mode === 'hash'
            ? ' 词表上均匀≠流量上均匀：高频词（the、逗号…）落在哪个专家，哪个专家就被挤爆。'
            : ` 利用词频建表后接近均衡；但单个 token 的流量无法再拆分——最高频的 “the” 占 ${(freq[0] * 100).toFixed(1)}% 的 token，它的每个专家至少承担这么多。`}
        </>
      }
      legend={<Legend items={[{ label: '红虚线 = 均值负载', tone: 'danger', variant: 'line' }]} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'hash', label: '按 id 随机哈希' },
              { value: 'greedy', label: '按词频贪心建表' },
            ]}
          />
          <div className="flex items-center gap-2 text-[13px] text-muted">
            每 token 专家数
            <Segmented size="sm" value={k} onChange={setK} options={[1, 2, 3].map((v) => ({ value: v, label: `k=${v}` }))} />
          </div>
          <Slider className="w-44" label="Zipf 指数（越大越偏）" value={s} min={0.5} max={1.4} step={0.05} onChange={setS} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="space-y-4">
        <div>
          <div className="mb-1.5 text-xs font-medium text-muted">一句话里每个 token 被查表送往的专家（下标）</div>
          <TokenRow tokens={SENT.map((id) => ({ text: WORDS[id], sub: table[id].map((e) => `E${e}`).join(' '), tone: ((table[id][0] % 8) + 1) as VizTone }))} />
        </div>
        <div className="grid items-start gap-4 @3xl:grid-cols-[3fr_2fr]">
          <div className="min-w-0">
            <div className="mb-1 text-xs font-medium text-muted">各专家承担的 token 流量（% ，按 Zipf 词频加权）</div>
            <LoadBars values={load} labels={load.map((_, e) => `E${e}`)} tones={load.map((v) => (v > mean * 1.3 ? 'danger' : v < mean * 0.7 ? 'info' : 'success'))} refValue={mean} refLabel={`均值 ${mean.toFixed(1)}%`} fmt={(v) => v.toFixed(1)} />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="MaxVio" value={vio.toFixed(2)} tone={vio > 0.3 ? 'danger' : 'success'} />
            <Stat label="最高频 token 占比" value={`${(freq[0] * 100).toFixed(1)}%`} />
            <Stat label="选专家用到的可学习参数" value="0" tone="success" />
            <Stat label="何时知道路由结果" value="拿到 token id 时" />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
