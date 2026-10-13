/**
 * 交互式 Roofline：选择模型 / GPU（或拖动算力、带宽滑块）、batch、上下文长度、权重精度，
 * 看 decode 整步、decode 的线性层与注意力、prefill 分别落在 roofline 的哪个区域。
 * 规格与公式见 ./hw.ts。单卡等效：显存放不下时按最小 TP 度均摊（忽略通信）。
 */
import { useMemo, useState } from 'react'
import { VizFrame, toneColor, type VizTone } from '@/viz'
import { Segmented, Slider } from '@/components/ui'
import { PlotFrame, Stat, sampleAxis, pathOf, type Axis } from './common'
import { GPUS, MODELS, decodeCost, fmtMs, fmtNum, kvPerToken, minTP, prefillCost } from './hw'

const W_OPTS = [
  { value: 2, label: 'BF16' },
  { value: 1, label: 'FP8' },
  { value: 0.5, label: 'INT4' },
]

export function RooflineViz() {
  const [mid, setMid] = useState('8b')
  const [gid, setGid] = useState('h100')
  const [peak, setPeak] = useState(989)
  const [bw, setBw] = useState(3.35)
  const [be, setBe] = useState(0) // batch = 2^be
  const [ce, setCe] = useState(12) // ctx = 2^ce
  const [wb, setWb] = useState(2)
  const m = MODELS.find((x) => x.id === mid)!
  const g = GPUS.find((x) => x.id === gid)!
  const B = 2 ** be
  const S = 2 ** ce
  const custom = peak !== g.tflops || bw !== g.tbs

  const r = useMemo(() => {
    const k = minTP(m, g, wb)
    const kvSplit = Math.min(k, m.kvHeads)
    const d = decodeCost(m, B, S, wb)
    const flops = d.flops / k
    const bytes = d.wB / k + d.kvB / kvSplit
    const tC = flops / (peak * 1e12)
    const tM = bytes / (bw * 1e12)
    const pf = prefillCost(m, S, wb)
    const memNeed = d.wB + d.kvB * (k / kvSplit)
    return {
      k,
      ai: flops / bytes,
      aiLin: d.linF / d.wB,
      aiAtt: d.attF / d.kvB,
      aiPre: pf.flops / pf.bytes,
      tC,
      tM,
      step: Math.max(tC, tM),
      memNeed,
      memHave: k * g.gb * 1e9,
      kvShare: d.kvB / kvSplit / bytes,
    }
  }, [m, g, B, S, wb, peak, bw])

  const ridge = (peak * 1e12) / (bw * 1e12)
  const yMax = 10 ** Math.ceil(Math.log10(peak * 1.5))
  const x: Axis = { min: 0.5, max: 2e5, log: true, ticks: [1, 10, 100, 1e3, 1e4, 1e5], fmt: (v) => (v >= 1e3 ? `${v / 1e3}K` : String(v)), label: '算术强度（FLOP / Byte，对数）' }
  const y: Axis = {
    min: 0.5,
    max: yMax,
    log: true,
    ticks: [1, 10, 100, 1000, 10000].filter((t) => t <= yMax),
    fmt: (v) => (v >= 1000 ? `${v / 1000}P` : `${v}T`),
    label: '可达性能（FLOP/s，对数）',
  }
  const roof = (ai: number) => Math.min(peak, bw * ai)
  const pts: { ai: number; label: string; tone: VizTone }[] = [
    { ai: r.aiLin, label: 'decode 线性层', tone: 'info' },
    { ai: r.aiAtt, label: 'decode 注意力', tone: 'warning' },
    { ai: r.ai, label: `decode 整步（B=${B}）`, tone: 'accent' },
    { ai: r.aiPre, label: `prefill（P=${fmtNum(S, 0)}）`, tone: 'success' },
  ]
  const memBound = r.tM >= r.tC

  return (
    <VizFrame
      title="Roofline：decode 与 prefill 在哪"
      legend={
        <>
          {pts.map((p) => (
            <span key={p.label} className="inline-flex items-center gap-1.5">
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={p.tone === 'accent' ? { border: `2.5px solid ${toneColor(p.tone)}` } : { background: toneColor(p.tone) }}
              />
              {p.label}：<b className="font-mono text-fg">{p.ai < 10 ? p.ai.toFixed(1) : fmtNum(p.ai, p.ai < 1000 ? 0 : 1)}</b>
            </span>
          ))}
        </>
      }
      caption={
        <>
          {m.name} · {custom ? '自定义 GPU' : g.name}
          {r.k > 1 && ` × ${r.k}（TP${r.k}）`} · batch {B} · 上下文 {fmtNum(S, 0)}：decode 每卡算术强度 <b>{r.ai.toFixed(1)}</b> FLOP/B，拐点 {ridge.toFixed(0)} →{' '}
          <b className={memBound ? 'text-danger' : 'text-success'}>{memBound ? '访存瓶颈' : '算力瓶颈'}</b>。单步至少 {fmtMs(r.step)}（TPOT 下界），吞吐上界约{' '}
          <b>{fmtNum(B / r.step, 1)}</b> token/s；其中 KV cache 占每步读取字节的 {(r.kvShare * 100).toFixed(0)}%。注意：注意力的强度 ≈ n_h / n_kv = {r.aiAtt.toFixed(0)}，与 batch 无关。
        </>
      }
      controls={
        <>
          <Segmented size="sm" value={mid} onChange={setMid} options={MODELS.map((x) => ({ value: x.id, label: x.name.replace('Llama-3.1-', '') }))} />
          <Segmented
            size="sm"
            value={gid}
            onChange={(v) => {
              setGid(v)
              const gg = GPUS.find((x) => x.id === v)!
              setPeak(gg.tflops)
              setBw(gg.tbs)
            }}
            options={GPUS.map((x) => ({ value: x.id, label: x.name.split(' ')[0] }))}
          />
          <Segmented size="sm" value={wb} onChange={setWb} options={W_OPTS} />
          <Slider className="min-w-[140px] flex-1" label="batch" value={be} min={0} max={10} onChange={setBe} format={(v) => String(2 ** v)} />
          <Slider className="min-w-[140px] flex-1" label="上下文 S" value={ce} min={9} max={17} onChange={setCe} format={(v) => fmtNum(2 ** v, 0)} />
          <Slider className="min-w-[140px] flex-1" label="算力 TFLOPS" value={peak} min={100} max={2500} step={1} onChange={setPeak} />
          <Slider className="min-w-[140px] flex-1" label="带宽 TB/s" value={bw} min={0.5} max={10} step={0.01} onChange={setBw} format={(v) => v.toFixed(2)} />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <PlotFrame x={x} y={y} height={270} pad={{ l: 46, r: 12, t: 22, b: 38 }}>
          {(s) => (
            <g>
              <rect x={s.l} y={s.t} width={Math.max(0, s.sx(ridge) - s.l)} height={s.H - s.b - s.t} fill={toneColor('danger')} opacity={0.06} />
              <path d={pathOf(sampleAxis(x, 200).map((v) => [v, roof(v)]), s, y)} fill="none" stroke="var(--fg)" strokeWidth={2.2} />
              <line x1={s.sx(ridge)} x2={s.sx(ridge)} y1={s.t} y2={s.H - s.b} stroke="var(--muted)" strokeDasharray="4 4" />
              <text x={s.sx(ridge) + 4} y={s.H - s.b - 6} fontSize={11} fill="var(--muted)">
                拐点 {ridge.toFixed(0)}
              </text>
              <text x={s.l + 6} y={s.t + 14} fontSize={11} fill={toneColor('danger')}>
                访存瓶颈区
              </text>
              <text x={s.W - s.r - 6} y={s.sy(peak) - 6} fontSize={11} fill="var(--muted)" textAnchor="end">
                峰值 {peak} TFLOPS
              </text>
              {pts.map((p) => {
                const px = s.sx(Math.max(x.min, Math.min(x.max, p.ai)))
                const py = s.sy(roof(p.ai))
                const ring = p.tone === 'accent'
                return ring ? (
                  <circle key={p.label} cx={px} cy={py} r={9} fill="none" stroke={toneColor(p.tone)} strokeWidth={2.6} />
                ) : (
                  <circle key={p.label} cx={px} cy={py} r={5.5} fill={toneColor(p.tone)} stroke="var(--surface)" strokeWidth={1.5} />
                )
              })}
            </g>
          )}
        </PlotFrame>
        <div className="grid grid-cols-2 gap-2">
          <Stat label="decode 每卡强度" value={`${r.ai.toFixed(1)}`} tone={memBound ? 'danger' : 'success'} sub={memBound ? '< 拐点：访存瓶颈' : '≥ 拐点：算力瓶颈'} />
          <Stat label="单步时间（TPOT 下界）" value={fmtMs(r.step)} sub={`算 ${fmtMs(r.tC)} / 读 ${fmtMs(r.tM)}`} />
          <Stat label="吞吐上界" value={`${fmtNum(B / r.step, 1)} tok/s`} sub={`每序列 ${fmtNum(1 / r.step, 1)} tok/s`} />
          <Stat
            label={`显存需求 / ${r.k} 卡`}
            value={`${fmtNum(r.memNeed, 0)}B`}
            tone={r.memNeed > r.memHave ? 'danger' : undefined}
            sub={r.memNeed > r.memHave ? `超出 ${fmtNum(r.memHave, 0)}B，batch 放不下` : `共 ${fmtNum(r.memHave, 0)}B`}
          />
          <div className="col-span-2 text-[11.5px] leading-relaxed text-subtle">
            线性层强度 ≈ 2B / 每参数字节数（BF16 下 ≈ batch）；注意力强度 = n_h/n_kv（GQA 组大小），只有 KV 精度能改变它。每 token KV = {(kvPerToken(m) / 1024).toFixed(0)} KiB。
            GPU 数字为 NVIDIA 官网 BF16 稠密规格；上界忽略了通信、kernel 效率与调度开销。
          </div>
        </div>
      </div>
    </VizFrame>
  )
}
