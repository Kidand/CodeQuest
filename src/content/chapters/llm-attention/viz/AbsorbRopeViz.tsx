/**
 * 为什么 MLA 必须 decouple RoPE：用小维度实算。
 *  - decouple：内容部分 s = c_q (W_uq W_ukᵀ) c_jᵀ 可用一个固定矩阵吸收；位置部分 q^R·k^R 单独算 → 吸收与展开逐位相等。
 *  - 直接对 k^C 加 RoPE：s_j = c_q W_uq R_{j−t} W_ukᵀ c_jᵀ，需要的矩阵随 j 变化 → 固定矩阵吸收只在 j = t 时正确。
 */
import { useMemo, useState } from 'react'
import { Legend, MatrixView, VizFrame } from '@/viz'
import { Slider, Switch } from '@/components/ui'
import { dot, randMatrix } from './math'

const DCQ = 5
const DC = 6
const DH = 4
const S = 8
const CQ = randMatrix(1, DCQ, 31)[0]
const WUQ = randMatrix(DCQ, DH, 32, 0.6)
const WUK = randMatrix(DC, DH, 33, 0.6)
const CKV = randMatrix(S, DC, 34)
const QR = randMatrix(1, 2, 35)[0] // decouple 时的 q^R（未旋转）
const KR = randMatrix(S, 2, 36) // 每个位置的 k^R（未旋转）
const FREQ = [1, 0.1] // rotate_half 两组频率（base=100, d=4）

const vm = (x: number[], W: number[][]) => W[0].map((_, c) => x.reduce((s, xi, r) => s + xi * W[r][c], 0))
const T = (W: number[][]) => W[0].map((_, c) => W.map((r) => r[c]))
const mm = (A: number[][], B: number[][]) => A.map((row) => vm(row, B))

/** rotate_half 风格 RoPE（d=4：维度 (0,2) 与 (1,3) 成对） */
function rope4(x: number[], pos: number) {
  const out = x.slice()
  for (let k = 0; k < 2; k++) {
    const a = pos * FREQ[k]
    out[k] = x[k] * Math.cos(a) - x[k + 2] * Math.sin(a)
    out[k + 2] = x[k + 2] * Math.cos(a) + x[k] * Math.sin(a)
  }
  return out
}
function rope2(x: number[], pos: number) {
  const a = pos * FREQ[0]
  return [x[0] * Math.cos(a) - x[1] * Math.sin(a), x[1] * Math.cos(a) + x[0] * Math.sin(a)]
}

const W_ABS = mm(WUQ, T(WUK)) // (DCQ, DC) = W_uq W_ukᵀ
const QC = vm(CQ, WUQ)
const QLAT = vm(CQ, W_ABS)

export function AbsorbRopeViz() {
  const [t, setT] = useState(5)
  const [decouple, setDecouple] = useState(true)
  const rows = useMemo(() => {
    const truth: number[] = []
    const absorbed: number[] = []
    for (let j = 0; j < S; j++) {
      const kc = vm(CKV[j], WUK)
      if (decouple) {
        const posPart = dot(rope2(QR, t), rope2(KR[j], j))
        truth.push(dot(QC, kc) + posPart) // 展开：q^C·k^C + q^R·k^R
        absorbed.push(dot(QLAT, CKV[j]) + posPart) // 吸收：(c_q W_abs)·c_j + q^R·k^R
      } else {
        truth.push(dot(rope4(QC, t), rope4(kc, j))) // RoPE 直接作用在 q^C、k^C 上
        absorbed.push(dot(QLAT, CKV[j])) // 固定矩阵 W_uq W_ukᵀ（= j−t=0 时的正确矩阵）
      }
    }
    return [truth, absorbed, truth.map((v, j) => v - absorbed[j])]
  }, [t, decouple])
  const maxErr = Math.max(...rows[2].slice(0, t + 1).map(Math.abs))

  return (
    <VizFrame
      title="RoPE 夹在 W_UQ 与 W_UK 之间，就没法预先吸收"
      caption={
        decouple ? (
          <>
            <b>decouple RoPE</b>：内容部分 <span className="font-mono">q^C·k^C_j = c_q (W_UQ W_UKᵀ) c_jᵀ</span>，中间是与位置无关的固定矩阵，可以预先算好；位置部分由缓存的 k^R 单独计算。
            吸收结果与展开结果的最大误差 <b className="font-mono">{maxErr.toExponential(1)}</b>（浮点舍入）。
          </>
        ) : (
          <>
            <b>直接对 k^C 施加 RoPE</b>：<span className="font-mono">s_j = c_q W_UQ R_t ᵀR_j W_UKᵀ c_jᵀ = c_q W_UQ R_(j−t) W_UKᵀ c_jᵀ</span>，中间矩阵依赖 j−t。
            用任何一个固定矩阵吸收，都只能在 <b>j = t</b> 一处恰好正确（差值为 0），其余位置全错——要么为每个历史位置重新展开 k^C，失去 MLA 的意义。
          </>
        )
      }
      legend={<Legend items={[{ label: '正值', tone: 'accent', variant: 'soft' }, { label: '负值', tone: 'danger', variant: 'soft' }, { label: '因果：j > t 不参与', tone: 'muted', variant: 'soft' }]} />}
      controls={
        <>
          <Slider className="min-w-[200px] flex-1" label="query 位置 t" value={t} min={0} max={S - 1} onChange={setT} />
          <Switch checked={decouple} onChange={setDecouple} label="decouple RoPE（MLA 的做法）" />
        </>
      }
    >
      <MatrixView
        data={rows}
        rowLabels={['展开（真值）', '固定矩阵吸收', '差值']}
        colLabels={Array.from({ length: S }, (_, j) => `j=${j}`)}
        scale="div"
        cellSize={48}
        format={(v) => (Math.abs(v) < 5e-13 ? '0' : v.toFixed(2))}
        masked={(_, c) => c > t}
        highlight={(r, c) => (r === 2 && c <= t ? (Math.abs(rows[2][c]) < 1e-9 ? 'success' : 'danger') : null)}
      />
    </VizFrame>
  )
}
