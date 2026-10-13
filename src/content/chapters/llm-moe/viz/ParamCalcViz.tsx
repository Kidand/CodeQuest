/**
 * 总参数 vs 激活参数：按公开配置逐项累加（SwiGLU 专家 = 3·d·d_e）。
 * 洞见：总参数几乎全在路由专家里，激活参数只随 k 增长——这就是 MoE「参数与计算解耦」。
 */
import { useState } from 'react'
import { Legend, VizFrame, catColor, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { fmtB } from './moeMath'
import { Stat } from './plot'

interface Cfg {
  name: string
  vocab: number
  d: number
  L: number
  dense: number
  dffDense: number
  attn: number
  E: number
  k: number
  shared: number
  de: number
  tied: boolean
  official: string
}

// 注意力参数/层：Mixtral 为 GQA(32 Q 头、8 KV 头、d_h=128)；DeepSeek-V3 为 MLA（q_lora 1536、kv_lora 512、128 头、d_nope 128、d_rope 64、d_v 128）
const MLA = 7168 * 1536 + 1536 * 128 * 192 + 7168 * (512 + 64) + 512 * 128 * 256 + 128 * 128 * 7168
const PRESETS: Record<'mixtral' | 'dsv3', Cfg> = {
  mixtral: { name: 'Mixtral 8×7B', vocab: 32000, d: 4096, L: 32, dense: 0, dffDense: 0, attn: 4096 * 4096 * 2 + 4096 * 1024 * 2, E: 8, k: 2, shared: 0, de: 14336, tied: false, official: '官方 46.7B / 12.9B' },
  dsv3: { name: 'DeepSeek-V3', vocab: 129280, d: 7168, L: 61, dense: 3, dffDense: 18432, attn: MLA, E: 256, k: 8, shared: 1, de: 2048, tied: false, official: '官方 671B / 37B' },
}

function count(c: Cfg, E: number, k: number) {
  const emb = c.vocab * c.d * (c.tied ? 1 : 2)
  const attn = c.L * c.attn
  const dense = c.dense * 3 * c.d * c.dffDense
  const moeL = c.L - c.dense
  const expert = 3 * c.d * c.de
  const router = moeL * c.d * E
  const parts = {
    emb,
    attn,
    dense,
    routedTotal: moeL * E * expert,
    routedActive: moeL * Math.min(k, E) * expert,
    shared: moeL * c.shared * expert,
    router,
  }
  const total = emb + attn + dense + parts.routedTotal + parts.shared + router
  const active = emb + attn + dense + parts.routedActive + parts.shared + router
  return { parts, total, active }
}

const SEG = [
  { key: 'emb', label: '嵌入 + LM head' },
  { key: 'attn', label: '注意力' },
  { key: 'dense', label: '稠密 FFN 层' },
  { key: 'shared', label: '共享专家' },
  { key: 'routed', label: '路由专家' },
  { key: 'router', label: 'router' },
] as const

export function ParamCalcViz() {
  const [preset, setPreset] = useState<'mixtral' | 'dsv3'>('mixtral')
  const c = PRESETS[preset]
  const [Es, setEs] = useState<Record<string, number>>({ mixtral: 8, dsv3: 256 })
  const [Ks, setKs] = useState<Record<string, number>>({ mixtral: 2, dsv3: 8 })
  const E = Es[preset]
  const k = Math.min(Ks[preset], E)
  const r = count(c, E, k)
  const atDefault = E === c.E && k === c.k

  const segVals = (active: boolean) => [r.parts.emb, r.parts.attn, r.parts.dense, r.parts.shared, active ? r.parts.routedActive : r.parts.routedTotal, r.parts.router]
  const bar = (active: boolean) => {
    const vals = segVals(active)
    const denom = active ? r.active : r.total
    return (
      <div className="flex h-6 w-full overflow-hidden rounded-md bg-surface-2">
        {vals.map((v, i) => (
          <div key={i} title={`${SEG[i].label}: ${fmtB(v)}`} style={{ width: `${(v / denom) * 100}%`, background: catColor(i), transition: 'width 300ms' }} />
        ))}
      </div>
    )
  }

  return (
    <VizFrame
      title="总参数与激活参数：钱花在哪"
      caption={
        <>
          {c.name}（E={E}，k={k}）：总参数 <b className="font-mono">{fmtB(r.total)}</b>，每 token 激活 <b className="font-mono">{fmtB(r.active)}</b>，
          激活比例 <b className="font-mono">{((r.active / r.total) * 100).toFixed(1)}%</b>。
          {atDefault ? <> 与{c.official}吻合（激活数这里计入了两份 embedding，各家口径略有差异）。</> : <> 拖动滑块：E 只改变总参数，k 才改变激活参数（≈ 每 token 计算量）。</>}
        </>
      }
      legend={<Legend items={SEG.map((s, i) => ({ label: s.label, tone: ((i % 8) + 1) as VizTone }))} />}
      controls={
        <>
          <Segmented
            size="sm"
            value={preset}
            onChange={setPreset}
            options={[
              { value: 'mixtral', label: 'Mixtral 8×7B' },
              { value: 'dsv3', label: 'DeepSeek-V3' },
            ]}
          />
          <Slider
            className="w-48"
            label="路由专家数 E"
            value={E}
            min={preset === 'mixtral' ? 1 : 8}
            max={preset === 'mixtral' ? 64 : 512}
            step={preset === 'mixtral' ? 1 : 8}
            onChange={(v) => setEs({ ...Es, [preset]: v })}
          />
          <Slider className="w-40" label="top-k" value={k} min={1} max={preset === 'mixtral' ? 8 : 16} onChange={(v) => setKs({ ...Ks, [preset]: v })} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-muted">
            <span>总参数（显存里要放下的）</span>
            <span className="font-mono text-fg">{fmtB(r.total)}</span>
          </div>
          {bar(false)}
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-muted">
            <span>激活参数（每个 token 实际参与计算的，按总参数等比例画）</span>
            <span className="font-mono text-fg">{fmtB(r.active)}</span>
          </div>
          <div style={{ width: `${(r.active / r.total) * 100}%`, minWidth: 6, transition: 'width 300ms' }}>{bar(true)}</div>
        </div>
        <div className="grid grid-cols-2 gap-2 @xl:grid-cols-4">
          <Stat label="单个专家 3·d·d_e" value={fmtB(3 * c.d * c.de)} />
          <Stat label="路由专家占总参数" value={`${((r.parts.routedTotal / r.total) * 100).toFixed(1)}%`} />
          <Stat label="激活比例" value={`${((r.active / r.total) * 100).toFixed(1)}%`} tone="accent" />
          <Stat label="前向 FLOPs/token ≈ 2×激活" value={`${((2 * r.active) / 1e9).toFixed(0)} GFLOPs`} />
        </div>
      </div>
    </VizFrame>
  )
}
