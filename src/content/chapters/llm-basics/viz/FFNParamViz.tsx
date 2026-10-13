/**
 * SwiGLU 隐藏维 8/3·d 的由来：三个矩阵 3·d·h = 8d²（与标准 FFN 的 2·d·4d 相同）⇒ h = 8d/3，
 * 再按 LLaMA 代码向上取整到 multiple_of 的倍数（可选先乘 ffn_dim_multiplier）。
 * 预设均为官方 params.json / config 中的真实取值。
 */
import { useMemo, useState } from 'react'
import { ProbBars, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { C, Stat } from './plot'

type Preset = 'llama7b' | 'llama3_8b' | 'llama2_70b' | 'custom'
const PRESETS: Record<Exclude<Preset, 'custom'>, { d: number; mult: number | null; mo: number; name: string; real: number }> = {
  llama7b: { d: 4096, mult: null, mo: 256, name: 'LLaMA-7B / Llama 2-7B', real: 11008 },
  llama3_8b: { d: 4096, mult: 1.3, mo: 1024, name: 'Llama 3-8B', real: 14336 },
  llama2_70b: { d: 8192, mult: 1.3, mo: 4096, name: 'Llama 2-70B / Llama 3-70B', real: 28672 },
}
const MOS = [64, 128, 256, 1024] as const

function hiddenDim(d: number, mo: number, mult: number | null) {
  let h = 4 * d
  const s1 = Math.trunc((2 * h) / 3)
  h = s1
  const s2 = mult != null ? Math.trunc(mult * h) : h
  h = s2
  const s3 = mo * Math.ceil(h / mo)
  return { s1, s2, s3 }
}
const fmtM = (n: number) => `${(n / 1e6).toFixed(2)}M`

export function FFNParamViz() {
  const [preset, setPreset] = useState<Preset>('llama7b')
  const [d, setD] = useState(2048)
  const [mo, setMo] = useState<(typeof MOS)[number]>(256)
  const [mult, setMult] = useState(1)
  const cfg = preset === 'custom' ? { d, mult: mult === 1 ? null : mult, mo, name: '自定义', real: 0 } : PRESETS[preset]
  const r = useMemo(() => hiddenDim(cfg.d, cfg.mo, cfg.mult), [cfg.d, cfg.mo, cfg.mult])
  const std = 2 * cfg.d * 4 * cfg.d
  const swi = 3 * cfg.d * r.s3
  const exact = (8 * cfg.d) / 3

  return (
    <VizFrame
      title="SwiGLU 的隐藏维为什么是 8/3·d"
      caption={
        <>
          {cfg.name}：d = {cfg.d}。标准 FFN 两个矩阵共 2·d·4d = 8d² = {fmtM(std)}；SwiGLU 有三个矩阵，令 3·d·h = 8d² 得 h = 8d/3 = {exact.toFixed(1)}，
          按 LLaMA 代码取整后 h = <b className="font-mono">{r.s3}</b>
          {cfg.real ? (r.s3 === cfg.real ? '（与官方配置一致 ✓）' : `（官方 ${cfg.real}）`) : ''}，参数为标准 FFN 的 {((swi / std) * 100).toFixed(1)}%。
          {cfg.mult != null && ' Llama 3 等模型又乘了 ffn_dim_multiplier = 1.3，FFN 比“等参数”的设定更宽。'}
        </>
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={preset}
            onChange={setPreset}
            options={[
              { value: 'llama7b', label: 'LLaMA-7B' },
              { value: 'llama3_8b', label: 'Llama 3-8B' },
              { value: 'llama2_70b', label: 'Llama 2-70B' },
              { value: 'custom', label: '自定义' },
            ]}
          />
          {preset === 'custom' && (
            <>
              <Slider className="min-w-[150px] flex-1" label="d_model" value={d} min={512} max={8192} step={128} onChange={setD} />
              <Slider className="min-w-[150px] flex-1" label="ffn_dim_multiplier" value={mult} min={1} max={1.5} step={0.05} onChange={setMult} format={(v) => (v === 1 ? '无' : v.toFixed(2))} />
              <div className="flex flex-col gap-1.5">
                <span className="text-[13px] text-muted">multiple_of</span>
                <Segmented size="sm" value={mo} onChange={setMo} options={MOS.map((v) => ({ value: v, label: String(v) }))} />
              </div>
            </>
          )}
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <div className="min-w-0 space-y-2 text-[13px]">
          <div className="text-xs font-medium text-muted">LLaMA 源码中的计算（hidden_dim 初值 = 4d）</div>
          <ol className="space-y-1.5">
            <li>
              ① <C>int(2 * {4 * cfg.d} / 3)</C> = <b className="font-mono">{r.s1}</b>
            </li>
            <li>
              ② {cfg.mult != null ? <C>int({cfg.mult} * {r.s1})</C> : <span className="text-muted">无 ffn_dim_multiplier，跳过</span>}
              {cfg.mult != null && (
                <>
                  {' '}= <b className="font-mono">{r.s2}</b>
                </>
              )}
            </li>
            <li>
              ③ 向上取整到 <C>multiple_of = {cfg.mo}</C> 的倍数 → <b className="font-mono">{r.s3}</b>（= {cfg.mo} × {r.s3 / cfg.mo}）
            </li>
          </ol>
          <div className="grid grid-cols-2 gap-2 pt-1">
            <Stat label="h / d" value={(r.s3 / cfg.d).toFixed(3)} sub="8/3 ≈ 2.667" />
            <Stat label="SwiGLU 参数" value={fmtM(swi)} tone="accent" sub="3·d·h" />
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">每层 FFN 参数量（无 bias）</div>
          <ProbBars
            max={Math.max(std, swi) * 1.05}
            format={(v) => fmtM(v)}
            items={[
              { label: '标准', p: std, tone: 'muted', note: '2·d·4d' },
              { label: 'SwiGLU', p: swi, tone: 'accent', note: '3·d·h' },
              { label: '注意力', p: 4 * cfg.d * cfg.d, tone: 'info', note: 'MHA 4d²' },
            ]}
          />
          <div className="mt-2 text-[12px] leading-relaxed text-muted">
            标准 MHA 的注意力约 4d²，FFN 约 8d²：FFN 占每层参数的约 2/3。GQA 会进一步缩小注意力的 K、V 投影，FFN 的占比更高。
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
