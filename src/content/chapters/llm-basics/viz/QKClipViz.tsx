/**
 * QK-Clip（Kimi K2 的 MuonClip）：每步更新后，对最大 attention logit 超过 τ 的头，按 γ = τ / S_max 缩放该头的 W_q、W_k。
 * MLA 中：q^C、k^C（头独有）各乘 √γ；q^R（头独有的 RoPE 部分）乘 γ；k^R 跨头共享，不动——
 * 两部分的点积都恰好缩小 γ 倍，新的最大 logit = τ。
 * 各头的 S_max 为示意数值（K2 报告：不加约束时中等规模实验的最大 logit 会超过 1000；K2 用 τ = 100）。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { C, Stat } from './plot'

const SMAX = [35, 62, 180, 48, 410, 96]
type Arch = 'mha' | 'mla'

export function QKClipViz() {
  const [tau, setTau] = useState(100)
  const [sel, setSel] = useState(4)
  const [arch, setArch] = useState<Arch>('mla')
  const rows = useMemo(
    () =>
      SMAX.map((s) => {
        const g = s > tau ? tau / s : 1
        return { s, g, after: s * g }
      }),
    [tau],
  )
  const top = Math.max(...SMAX) * 1.05
  const r = rows[sel]
  const sg = Math.sqrt(r.g)

  return (
    <VizFrame
      title="QK-Clip：只缩放超标的头，把最大 logit 拉回 τ"
      caption={
        <>
          head {sel}：S_max = {r.s}，{r.g < 1 ? <>γ = τ/S_max = {r.g.toFixed(3)}。</> : <>未超过 τ，γ = 1，不做任何修改。</>}
          {r.g < 1 &&
            (arch === 'mha' ? (
              <> W_q、W_k 各乘 √γ = {sg.toFixed(3)}，logit = q·k 整体缩小 γ 倍 → 新的最大 logit = {r.after.toFixed(0)}。</>
            ) : (
              <> MLA：q<sup>C</sup>、k<sup>C</sup> 各乘 √γ = {sg.toFixed(3)}，q<sup>R</sup> 乘 γ = {r.g.toFixed(3)}，<b>共享的 k<sup>R</sup> 不动</b>（动它会影响所有头）。内容部分 √γ·√γ = γ，RoPE 部分 γ·1 = γ，最大 logit 同样变为 {r.after.toFixed(0)}。</>
            ))}
          {' '}裁剪发生在优化器更新之后，不改变当前步的前向与反向。点击条形选择不同的头。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '裁剪前 S_max', tone: 'muted', variant: 'soft' },
            { label: '裁剪后', tone: 'accent' },
            { label: 'τ', tone: 'danger', variant: 'line' },
          ]}
        />
      }
      controls={
        <>
          <Segmented
            size="sm"
            value={arch}
            onChange={setArch}
            options={[
              { value: 'mha', label: 'MHA / GQA' },
              { value: 'mla', label: 'MLA（Kimi K2）' },
            ]}
          />
          <Slider className="min-w-[180px] flex-1" label="阈值 τ" value={tau} min={30} max={300} step={10} onChange={setTau} />
        </>
      }
    >
      <div className="grid items-start gap-5 @3xl:grid-cols-[1fr_230px]">
        <div className="min-w-0 space-y-2">
          <div className="text-xs font-medium text-muted">各头的最大 logit（示意数值）</div>
          {rows.map((row, i) => (
            <button key={i} type="button" onClick={() => setSel(i)} className="flex w-full items-center gap-2 text-left" aria-label={`选择 head ${i}`}>
              <span className={`w-14 shrink-0 font-mono text-[12.5px] ${i === sel ? 'font-bold text-fg' : 'text-muted'}`}>head {i}</span>
              <span className="relative h-6 min-w-0 flex-1 rounded-md bg-surface-2">
                <span className="absolute inset-y-0 left-0 rounded-md" style={{ width: `${(row.s / top) * 100}%`, background: toneColor('muted'), opacity: 0.35 }} />
                <span
                  className="absolute inset-y-1 left-0 rounded-sm transition-all duration-300"
                  style={{ width: `${(row.after / top) * 100}%`, background: toneColor(row.g < 1 ? 'accent' : 'success') }}
                />
                <span className="absolute inset-y-[-3px] w-0.5" style={{ left: `${(tau / top) * 100}%`, background: toneColor('danger') }} />
              </span>
              <span className="w-24 shrink-0 text-right font-mono text-[12px] tabular-nums text-muted">
                {row.s} → {row.after.toFixed(0)}
              </span>
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 gap-2 @3xl:grid-cols-1">
          <Stat label={`head ${sel} 的 γ`} value={r.g.toFixed(3)} tone={r.g < 1 ? 'accent' : 'success'} />
          {arch === 'mha' ? (
            <>
              <Stat label="W_q 缩放" value={`×${sg.toFixed(3)}`} sub="√γ" />
              <Stat label="W_k 缩放" value={`×${sg.toFixed(3)}`} sub="√γ" />
            </>
          ) : (
            <>
              <Stat label={<>q<sup>C</sup> / k<sup>C</sup> 缩放</>} value={`×${sg.toFixed(3)}`} sub="头独有，各 √γ" />
              <Stat label={<>q<sup>R</sup> 缩放</>} value={`×${r.g.toFixed(3)}`} sub="头独有，γ" />
              <Stat label={<>k<sup>R</sup> 缩放</>} value="×1" sub="所有头共享，不动" />
            </>
          )}
        </div>
      </div>
      <div className="mt-3 text-[12px] leading-relaxed text-muted">
        logit<sub>h</sub>(i, j) = (q<sub>i</sub><sup>C</sup>·k<sub>j</sub><sup>C</sup> + q<sub>i</sub><sup>R</sup>·k<sub>j</sub><sup>R</sup>) / √d。逐头裁剪而不是全局裁剪：K2 报告指出只有少数头的 logit 会爆炸，逐头处理对其余头零干扰。
      </div>
    </VizFrame>
  )
}
