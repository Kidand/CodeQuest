/**
 * MHA 参数量与 FLOPs：头数 h 不改变参数量与 FLOPs；序列长度 T 超过 2D 后，注意力的两次矩阵乘开始超过四个投影。
 */
import { useState } from 'react'
import { ProbBars, VizFrame } from '@/viz'
import { Segmented, Slider, Switch } from '@/components/ui'
import { fmtCount } from '@/lib/utils'

export function ParamFlopsViz() {
  const [D, setD] = useState(4096)
  const [h, setH] = useState(32)
  const [tExp, setTExp] = useState(12)
  const [bias, setBias] = useState(false)
  const T = 2 ** tExp
  const params = 4 * D * D + (bias ? 4 * D : 0)
  const fQKV = 6 * T * D * D
  const fScore = 2 * T * T * D
  const fPV = 2 * T * T * D
  const fO = 2 * T * D * D
  const total = fQKV + fScore + fPV + fO
  const attnShare = (fScore + fPV) / total

  return (
    <VizFrame
      title="参数量 4D² 与头数无关；T > 2D 后注意力 FLOPs 占大头"
      caption={
        <>
          d_h = D / h = <b className="font-mono">{D / h}</b>；参数量 <b className="font-mono">{fmtCount(params)}</b>（{bias ? '4D² + 4D' : '4D²'}），改变 h 不会变。
          单序列前向共 <b className="font-mono">{fmtCount(total)}</b> FLOPs，其中 QKᵀ 与 PV 占 <b className="font-mono">{(attnShare * 100).toFixed(1)}%</b>
          {T > 2 * D ? '——T 已超过 2D，注意力矩阵乘成为主要开销。' : `——T 还小于 2D = ${2 * D}，投影矩阵乘占主导。`}
        </>
      }
      controls={
        <>
          <Slider className="min-w-[160px] flex-1" label="D（d_model）" value={D} min={512} max={8192} step={512} onChange={setD} />
          <Slider className="min-w-[160px] flex-1" label="T（序列长度）" value={tExp} min={9} max={17} onChange={setTExp} format={(v) => (2 ** v).toLocaleString()} />
          <Segmented size="sm" value={h} onChange={setH} options={[8, 16, 32, 64].map((v) => ({ value: v, label: `h=${v}` }))} />
          <Switch checked={bias} onChange={setBias} label="带 bias" />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-muted">前向 FLOPs 分解（B=1，按 2·MAC 计）</div>
          <ProbBars
            items={[
              { label: 'Q/K/V 投影', p: fQKV / total, tone: 1, note: '6TD²' },
              { label: 'QKᵀ', p: fScore / total, tone: 'warning', note: '2T²D' },
              { label: 'P·V', p: fPV / total, tone: 'warning', note: '2T²D' },
              { label: 'W_O 投影', p: fO / total, tone: 2, note: '2TD²' },
            ]}
            max={1}
            format={(p) => `${(p * 100).toFixed(1)}%`}
          />
        </div>
        <div className="space-y-2 text-[13px]">
          <div className="rounded-xl border border-border bg-surface p-3 font-mono text-[12.5px] leading-relaxed">
            <div>W_Q, W_K, W_V, W_O：各 D×D = {fmtCount(D * D)}</div>
            <div>拆头只是把 D 列分成 h 组，每组 d_h = {D / h} 列</div>
            <div className="mt-1 text-muted">
              交叉点：4T²D = 8TD² ⇔ T = 2D = {(2 * D).toLocaleString()}
            </div>
          </div>
          <div className="text-xs text-muted">因果注意力若跳过上三角（FlashAttention 会跳过整块被遮的 tile），QKᵀ 与 PV 的实际计算量约减半。</div>
        </div>
      </div>
    </VizFrame>
  )
}
