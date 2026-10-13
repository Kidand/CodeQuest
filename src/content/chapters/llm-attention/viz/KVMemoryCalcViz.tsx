/**
 * KV cache 显存计算器：bytes = 2 · L · B · S · n_kv · d_h · bytes_per_elem（MLA：L · B · S · (d_c + d_h^R) · bytes）。
 * 预设为公开配置：Llama-2-7B / Llama-3-8B / Llama-3-70B / Qwen2.5-7B / DeepSeek-V3。
 */
import { useState } from 'react'
import { ProbBars, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { cn } from '@/lib/utils'
import { fmtGiB } from './math'

interface Cfg {
  L: number
  nq: number
  nkv: number
  dh: number
  mla?: number // MLA 每层每 token 缓存的元素数（d_c + d_h^R）
}
const PRESETS: Record<string, Cfg & { label: string }> = {
  l2: { label: 'Llama-2-7B', L: 32, nq: 32, nkv: 32, dh: 128 },
  l3: { label: 'Llama-3-8B', L: 32, nq: 32, nkv: 8, dh: 128 },
  l70: { label: 'Llama-3-70B', L: 80, nq: 64, nkv: 8, dh: 128 },
  qw: { label: 'Qwen2.5-7B', L: 28, nq: 28, nkv: 4, dh: 128 },
  ds: { label: 'DeepSeek-V3', L: 61, nq: 128, nkv: 128, dh: 128, mla: 576 },
}
const DTYPES = [
  { value: 4, label: 'FP32' },
  { value: 2, label: 'BF16' },
  { value: 1, label: 'FP8' },
  { value: 0.5, label: 'INT4' },
]

export function KVMemoryCalcViz() {
  const [preset, setPreset] = useState('l3')
  const [cfg, setCfg] = useState<Cfg>(PRESETS.l3)
  const [sExp, setSExp] = useState(13) // 8192
  const [batch, setBatch] = useState(1)
  const [bytes, setBytes] = useState(2)
  const S = 2 ** sExp
  const perLayerTok = cfg.mla ?? 2 * cfg.nkv * cfg.dh // 每层每 token 元素数
  const perTok = cfg.L * perLayerTok * bytes
  const total = perTok * S * batch
  const mhaPerLayer = 2 * cfg.nq * cfg.dh
  const set = (k: keyof Cfg, v: number) => {
    setPreset('custom')
    setCfg((c) => {
      const n = { ...c, [k]: v, mla: undefined }
      if (k === 'nq' && n.nkv > v) n.nkv = v
      if (k === 'nkv' && v > n.nq) n.nq = v
      return n
    })
  }
  const variants = [
    { label: `MHA`, el: mhaPerLayer, tone: 'danger' as const },
    { label: 'GQA-8', el: 2 * Math.min(8, cfg.nq) * cfg.dh, tone: 'warning' as const },
    { label: 'MQA', el: 2 * cfg.dh, tone: 'success' as const },
    { label: 'MLA', el: 576, tone: 'info' as const },
  ]

  return (
    <VizFrame
      title="KV cache 显存计算器"
      caption={
        <>
          每 token 缓存 <b className="font-mono">{fmtGiB(perTok)}</b>（{cfg.mla ? `${cfg.L} 层 × ${cfg.mla} 元素` : `2 × ${cfg.L} 层 × ${cfg.nkv} 头 × ${cfg.dh}`} ×{' '}
          {bytes} B）；batch {batch} × 长度 {S.toLocaleString()} 共 <b className="font-mono text-accent">{fmtGiB(total)}</b>。
          {!cfg.mla && cfg.nkv < cfg.nq && <> 相比同配置 MHA 节省 {((1 - cfg.nkv / cfg.nq) * 100).toFixed(1)}%。</>}
          {cfg.mla && <> 若用同头数的 MHA（每层 {mhaPerLayer.toLocaleString()} 元素），将是 {(mhaPerLayer / cfg.mla).toFixed(1)} 倍。</>}
        </>
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={preset}
            onChange={(v) => {
              setPreset(v)
              if (v !== 'custom') setCfg(PRESETS[v])
            }}
            options={[...Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.label })), { value: 'custom', label: '自定义' }]}
          />
          <Slider className="min-w-[140px] flex-1" label="层数 L" value={cfg.L} min={1} max={128} onChange={(v) => set('L', v)} />
          <Slider className="min-w-[140px] flex-1" label="Q 头数" value={cfg.nq} min={1} max={128} onChange={(v) => set('nq', v)} />
          <Slider className="min-w-[140px] flex-1" label={cfg.mla ? 'KV 头数（MLA 不适用）' : 'KV 头数 n_kv'} value={cfg.nkv} min={1} max={128} onChange={(v) => set('nkv', v)} />
          <Slider className="min-w-[140px] flex-1" label="head_dim" value={cfg.dh} min={32} max={256} step={32} onChange={(v) => set('dh', v)} />
          <Slider className="min-w-[140px] flex-1" label="序列长度 S" value={sExp} min={10} max={20} onChange={setSExp} format={(v) => `${2 ** (v - 10)}K`} />
          <Slider className="min-w-[140px] flex-1" label="batch" value={batch} min={1} max={128} onChange={setBatch} />
          <Segmented size="sm" value={bytes} onChange={setBytes} options={DTYPES} />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-[1fr_1.2fr]">
        <div className="space-y-2">
          <div className={cn('rounded-2xl border border-border bg-surface p-4 text-center')}>
            <div className="text-xs text-muted">KV cache 总量</div>
            <div className="font-mono text-3xl font-bold text-accent tabular-nums">{fmtGiB(total)}</div>
            <div className="mt-1 font-mono text-xs text-subtle">{total.toExponential(3)} 字节（1 GiB = 2³⁰）</div>
          </div>
          <div className="rounded-xl border border-border bg-surface p-3 font-mono text-[12px] leading-relaxed text-muted">
            {cfg.mla ? (
              <>bytes = L · B · S · (d_c + d_h^R) · bytes<br />= {cfg.L} · {batch} · {S} · 576 · {bytes}</>
            ) : (
              <>bytes = 2 · L · B · S · n_kv · d_h · bytes<br />= 2 · {cfg.L} · {batch} · {S} · {cfg.nkv} · {cfg.dh} · {bytes}</>
            )}
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">同样 {cfg.L} 层、head_dim={cfg.dh}：每层每 token 缓存元素数</div>
          <ProbBars items={variants.map((v) => ({ label: v.label, p: v.el, tone: v.tone, note: v.el.toLocaleString() }))} format={(x) => `${((x / mhaPerLayer) * 100).toFixed(1)}%`} />
          <div className="mt-2 text-[11px] text-subtle">百分比相对 MHA；MLA 一栏按 DeepSeek-V2/V3 的 d_c + d_h^R = 512 + 64 计（与 head_dim 无关）。</div>
        </div>
      </div>
    </VizFrame>
  )
}
