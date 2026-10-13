/**
 * 标准注意力（FlashAttention 论文 Algorithm 0）与 FlashAttention（Algorithm 1）的前向 HBM 读写量与额外显存（单个头，FP16）。
 * 计数按算法逐行累加：
 *   标准：读 Q,K(2Nd) + 写 S(N²) + 读 S(N²) + 写 P(N²) + 读 P,V(N²+Nd) + 写 O(Nd) = 4Nd + 4N²
 *   Flash：K,V 各读一次(2Nd) + T_c 轮 ×（读 Q_i,O_i,ℓ,m + 写 O_i,ℓ,m）= 2Nd + T_c(3Nd + 4N)，B_c = ⌈M/4d⌉，B_r = min(B_c, d)
 */
import { useState } from 'react'
import { ProbBars, VizFrame } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { fmtGiB } from './math'

export function IOCompareViz() {
  const [nExp, setNExp] = useState(12)
  const [d, setD] = useState(64)
  const [mKB, setMKB] = useState(192)
  const N = 2 ** nExp
  const M = (mKB * 1024) / 2 // SRAM 可容纳的 FP16 元素数
  const Bc = Math.ceil(M / (4 * d))
  const Br = Math.min(Bc, d)
  const Tc = Math.ceil(N / Bc)
  const std = 4 * N * d + 4 * N * N
  const flash = 2 * N * d + Tc * (3 * N * d + 4 * N)
  const bytes = (el: number) => el * 2
  const extraStd = 2 * N * N // S 与 P
  const extraFlash = 2 * N // m 与 ℓ

  return (
    <VizFrame
      title="FlashAttention 省的是 HBM 读写与 N² 显存，不是 FLOPs"
      caption={
        <>
          N = {N.toLocaleString()}、d = {d}、M = {mKB} KB：块大小 B_c = ⌈M/4d⌉ = {Bc}，B_r = {Br}，外层 {Tc} 轮。前向 HBM 读写 标准 <b className="font-mono">{fmtGiB(bytes(std))}</b> vs Flash{' '}
          <b className="font-mono">{fmtGiB(bytes(flash))}</b>（{(std / flash).toFixed(1)}×）；额外显存 {fmtGiB(bytes(extraStd))} vs {fmtGiB(bytes(extraFlash))}。d 越小、SRAM 越大，HBM 读写节省越多：
          Θ(N²d²/M) vs Θ(Nd + N²)。
        </>
      }
      controls={
        <>
          <Slider className="min-w-[180px] flex-1" label="序列长度 N" value={nExp} min={9} max={17} onChange={setNExp} format={(v) => (2 ** v).toLocaleString()} />
          <Slider className="min-w-[160px] flex-1" label="SRAM 大小 M" value={mKB} min={64} max={256} step={16} onChange={setMKB} format={(v) => `${v} KB`} />
          <Segmented size="sm" value={d} onChange={setD} options={[64, 128].map((v) => ({ value: v, label: `d = ${v}` }))} />
        </>
      }
    >
      <div className="grid gap-5 @2xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">前向 HBM 读写量（单个头，FP16）</div>
          <ProbBars
            items={[
              { label: '标准', p: bytes(std), tone: 'danger', note: '4Nd+4N²' },
              { label: 'Flash', p: bytes(flash), tone: 'success', note: '≈12N²d²/M' },
            ]}
            format={(b) => fmtGiB(b)}
          />
        </div>
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">为反向保存的额外显存（单个头）</div>
          <ProbBars
            items={[
              { label: '标准', p: bytes(extraStd), tone: 'danger', note: 'S、P：2N²' },
              { label: 'Flash', p: bytes(extraFlash), tone: 'success', note: 'm、ℓ：2N' },
            ]}
            format={(b) => fmtGiB(b)}
          />
        </div>
      </div>
      <div className="mt-3 text-[11px] text-subtle">
        计数按论文 Algorithm 0 / 1 逐行累加，常数因子与实现相关；FA2 起反向只保存 logsumexp（N 个数）。A100：HBM 40–80 GB、1.5–2.0 TB/s；每个 SM 192 KB 片上 SRAM（共 108 个）、约 19 TB/s。
      </div>
    </VizFrame>
  )
}
