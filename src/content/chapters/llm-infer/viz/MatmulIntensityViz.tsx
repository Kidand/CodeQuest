/**
 * 同一个线性层 Y = X·W（X: T×d，W: d×d）在不同 token 数 T 下的算术强度：
 *   FLOPs = 2·T·d²，Bytes = b·(d² + 2·T·d)，AI = T·d / (d + 2T) · (2/b)
 * T 小（decode）时 W 每个元素读进来只做 T 次乘加；T 大（prefill）时强度饱和到 ≈ d/2·(2/b)。
 */
import { useState } from 'react'
import { VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { C, Stat } from './common'
import { GPUS, fmtNum } from './hw'

export function MatmulIntensityViz() {
  const [te, setTe] = useState(0)
  const [d, setD] = useState(4096)
  const [gid, setGid] = useState('h100')
  const g = GPUS.find((x) => x.id === gid)!
  const T = 2 ** te
  const b = 2
  const flops = 2 * T * d * d
  const wBytes = b * d * d
  const aBytes = b * 2 * T * d
  const ai = flops / (wBytes + aBytes)
  const ridge = g.tflops / g.tbs
  const tStar = d > 2 * ridge ? Math.ceil((ridge * d) / (d - 2 * ridge)) : Infinity
  const logMax = Math.log10(5000)
  const pos = (v: number) => `${Math.max(1.5, (Math.log10(Math.max(1, v)) / logMax) * 100)}%`
  // 矩阵示意尺寸（像素）：W 为 S×S 方块，X/Y 宽 S（= d）、高按 T/d 等比例（最小 3px，T > d 时封顶）
  const S = 90
  const h = Math.max(3, Math.min(S, (T / d) * S))
  return (
    <VizFrame
      title="同一个线性层：token 越多，强度越高"
      caption={
        <>
          T = {T} 个 token 过 <C>{d}×{d}</C> 的权重：计算 {fmtNum(flops, 1)}FLOPs，读写 {fmtNum(wBytes + aBytes, 1)}B（权重占 {((wBytes / (wBytes + aBytes)) * 100).toFixed(0)}%），
          强度 <b>{ai.toFixed(1)}</b> FLOP/B。{g.name} 的拐点约 {ridge.toFixed(0)}，所以 T ≥ <b>{Number.isFinite(tStar) ? tStar : '∞'}</b> 时这个 GEMM 才会受算力限制。
          {T === 1 ? ' decode（batch=1）就是 T = 1 的 GEMV：每个权重读进来只做一次乘加。' : ''}
        </>
      }
      controls={
        <>
          <Slider className="min-w-[200px] flex-1" label="token 数 T（decode 时 = batch）" value={te} min={0} max={13} onChange={setTe} format={(v) => String(2 ** v)} />
          <Segmented size="sm" value={d} onChange={setD} options={[{ value: 4096, label: 'd=4096' }, { value: 8192, label: 'd=8192' }]} />
          <Segmented size="sm" value={gid} onChange={setGid} options={GPUS.map((x) => ({ value: x.id, label: x.name.split(' ')[0] }))} />
        </>
      }
    >
      <div className="grid items-center gap-4 @2xl:grid-cols-[auto_minmax(0,1fr)]">
        <svg width={3 * S + 48} height={S + 34} className="mx-auto block overflow-visible">
          <g fontSize={11} fill="var(--muted)">
            <rect x={0} y={(S - h) / 2 + 18} width={S} height={h} rx={2} fill={toneColor('info')} opacity={0.8} style={{ transition: 'all 0.3s' }} />
            <text x={S / 2} y={10} textAnchor="middle">X（T×d）</text>
            <text x={S + 12} y={S / 2 + 22} textAnchor="middle" fill="var(--fg)">×</text>
            <rect x={S + 24} y={18} width={S} height={S} rx={3} fill={toneColor('accent')} opacity={0.55} />
            <text x={1.5 * S + 24} y={10} textAnchor="middle">W（d×d）</text>
            <text x={2 * S + 36} y={S / 2 + 22} textAnchor="middle" fill="var(--fg)">=</text>
            <rect x={2 * S + 48} y={(S - h) / 2 + 18} width={S} height={h} rx={2} fill={toneColor('success')} opacity={0.8} style={{ transition: 'all 0.3s' }} />
            <text x={2.5 * S + 48} y={10} textAnchor="middle">Y（T×d）</text>
          </g>
        </svg>
        <div className="min-w-0 space-y-3">
          <div className="relative h-7 rounded-md bg-surface-2">
            <div className="absolute inset-y-0 left-0 rounded-md transition-[width] duration-300" style={{ width: pos(ai), background: toneColor(ai >= ridge ? 'success' : 'danger') }} />
            <div className="absolute -inset-y-1 w-0.5 bg-fg" style={{ left: pos(ridge) }} />
            <span className="absolute inset-y-0 right-2 flex items-center font-mono text-[11.5px] text-fg">{ai.toFixed(1)} FLOP/B</span>
          </div>
          <div className="flex justify-between font-mono text-[10.5px] text-subtle">
            <span>1</span>
            <span>竖线 = 拐点 {ridge.toFixed(0)}（对数轴，上限 5000）</span>
            <span>5000</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Stat label="FLOPs" value={fmtNum(flops, 1)} />
            <Stat label="权重字节" value={`${fmtNum(wBytes, 1)}B`} />
            <Stat label="激活字节" value={`${fmtNum(aBytes, 1)}B`} />
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
