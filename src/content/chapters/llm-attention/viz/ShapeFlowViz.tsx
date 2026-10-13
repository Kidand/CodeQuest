/**
 * MHA 拆头 / 合头的张量形状逐步变换：每个元素标成「token 字母 + 维度下标」，看 reshape 与 transpose 把它们搬到哪里。
 * 切换「错误写法」：直接 reshape(B, h, T, d_h)，头里混进了别的 token。
 */
import { useMemo, useState } from 'react'
import { CodeTrace, Legend, StepControls, VizFrame, toneColor, toneSoft, useStepper, type VizTone } from '@/viz'
import { Segmented } from '@/components/ui'

const TOK = ['A', 'B', 'C', 'D']
const T = 4
const D = 8
const H = 2
const DH = 4

interface Block {
  title: string
  rowLabels: string[]
  cells: { text: string; tone: VizTone | null; bad?: boolean }[][]
}
interface Frame {
  shape: string
  line: number
  blocks: Block[]
  caption: React.ReactNode
}

const headTone = (c: number): VizTone => (c < DH ? 1 : 2)
const Code = ({ children }: { children: React.ReactNode }) => <code className="rounded bg-surface-2 px-1 font-mono text-[12.5px]">{children}</code>

// q 元素 (t, c) 的标签：token 字母 + 列号；输出用小写
const lab = (t: number, c: number, out = false) => `${out ? TOK[t].toLowerCase() : TOK[t]}${c}`

function flat(out: boolean, mixed = false) {
  return {
    title: '(T, D)',
    rowLabels: TOK.map((t) => (out ? t.toLowerCase() : t)),
    cells: TOK.map((_, t) => Array.from({ length: D }, (_, c) => ({ text: mixed ? `y${t}${c}` : lab(t, c, out), tone: mixed ? ('muted' as VizTone) : headTone(c) }))),
  }
}
function byToken(out: boolean): Block[] {
  return TOK.map((_, t) => ({
    title: `token ${out ? TOK[t].toLowerCase() : TOK[t]}：(h, d_h)`,
    rowLabels: ['h0', 'h1'],
    cells: [0, 1].map((h) => Array.from({ length: DH }, (_, d) => ({ text: lab(t, h * DH + d, out), tone: headTone(h * DH + d) }))),
  }))
}
function byHead(out: boolean): Block[] {
  return [0, 1].map((h) => ({
    title: `头 ${h}：(T, d_h)`,
    rowLabels: TOK.map((t) => (out ? t.toLowerCase() : t)),
    cells: TOK.map((_, t) => Array.from({ length: DH }, (_, d) => ({ text: lab(t, h * DH + d, out), tone: headTone(h * DH + d) }))),
  }))
}
function scores(): Block[] {
  return [0, 1].map((h) => ({
    title: `头 ${h}：(T, T)`,
    rowLabels: TOK,
    cells: TOK.map((qt) => TOK.map((kt) => ({ text: `${qt}·${kt}`, tone: (h ? 2 : 1) as VizTone }))),
  }))
}
function wrongHeads(): Block[] {
  // 直接 reshape(B, h, T, d_h)：按内存顺序每 d_h 个元素切一行
  const rows: { t: number; c0: number }[] = []
  for (let t = 0; t < T; t++) for (let k = 0; k < D / DH; k++) rows.push({ t, c0: k * DH })
  return [0, 1].map((h) => ({
    title: `「头 ${h}」：(T, d_h)`,
    rowLabels: rows.slice(h * T, h * T + T).map((_, i) => `行${i}`),
    cells: rows.slice(h * T, h * T + T).map(({ t, c0 }) =>
      Array.from({ length: DH }, (_, d) => ({ text: lab(t, c0 + d), tone: headTone(c0 + d), bad: true })),
    ),
  }))
}

function buildFrames(wrong: boolean): Frame[] {
  const f0: Frame = {
    shape: '(B, T, D) = (1, 4, 8)',
    line: 4,
    blocks: [flat(false)],
    caption: <>线性投影 <Code>q = x @ Wq</Code>：每行是一个 token 的 D=8 维 query。颜色表示这一列<b>将来</b>属于哪个头：前 4 列给头 0，后 4 列给头 1。</>,
  }
  if (wrong) {
    return [
      f0,
      {
        shape: '(B, h, T, d_h) = (1, 2, 4, 4) ❌',
        line: 5,
        blocks: wrongHeads(),
        caption: (
          <>
            直接 <Code>reshape(B, h, T, d_h)</Code>：reshape 只按内存顺序「每 4 个切一行」。结果「头 0」的 4 行是 A 的前半、A 的后半、B 的前半、B 的后半——
            <b className="text-danger">它其实只包含 token A、B，头 1 只包含 C、D</b>。形状完全正确，代码不会报错，但注意力在错误的「token」之间计算。
          </>
        ),
      },
      {
        shape: '(B, h, T, T) ❌',
        line: 8,
        blocks: wrongHeads().map((b) => ({ ...b, title: b.title.replace('(T, d_h)', '打分对象') })),
        caption: <>后续所有计算都建立在错误的切分上：例如「头 0」第 0 行与第 1 行的点积，其实是 token A 的头 0 维度与 A 的头 1 维度相乘。正确做法：先 <Code>reshape(B, T, h, d_h)</Code> 再 <Code>transpose(0, 2, 1, 3)</Code>。</>,
      },
    ]
  }
  return [
    f0,
    {
      shape: '(B, T, h, d_h) = (1, 4, 2, 4)',
      line: 5,
      blocks: byToken(false),
      caption: <><Code>reshape(B, T, h, d_h)</Code>：把每个 token 的 8 维切成 2 段，每段 4 维。只改「看法」，内存里的元素顺序没变（O(1) 的 view）。</>,
    },
    {
      shape: '(B, h, T, d_h) = (1, 2, 4, 4)',
      line: 6,
      blocks: byHead(false),
      caption: <><Code>transpose(0, 2, 1, 3)</Code>：把头维提到 T 前面。现在每个头是一张独立的 (T, d_h) 矩阵，包含<b>全部 4 个 token</b>的对应 4 维——可以把 (B, h) 当作批维做矩阵乘。</>,
    },
    {
      shape: '(B, h, T, T) = (1, 2, 4, 4)',
      line: 8,
      blocks: scores(),
      caption: <>每个头独立算 <Code>q @ kᵀ / √d_h</Code>：缩放用的是 √d_h = 2 而不是 √D。两个头的打分矩阵互不相干，各自 softmax。</>,
    },
    {
      shape: '(B, h, T, d_h) = (1, 2, 4, 4)',
      line: 9,
      blocks: byHead(true),
      caption: <><Code>softmax(s) @ v</Code>：每个头输出 (T, d_h)。小写字母表示输出，例如 a5 = token A 在头 1 中输出的第 2 维（全局第 5 维）。</>,
    },
    {
      shape: '(B, T, h, d_h) = (1, 4, 2, 4)',
      line: 10,
      blocks: byToken(true),
      caption: <>合头第一步：<Code>transpose(0, 2, 1, 3)</Code> 把 T 换回前面。在 PyTorch 中此时张量<b>不连续</b>，<Code>.view()</Code> 会报错，要用 <Code>.contiguous().view()</Code> 或 <Code>.reshape()</Code>。</>,
    },
    {
      shape: '(B, T, D) = (1, 4, 8)',
      line: 11,
      blocks: [flat(true)],
      caption: <><Code>reshape(B, T, D)</Code>：每个 token 的两个头输出首尾拼接（concat）。注意此时两个头的信息只是「并排放着」，还没有交互。</>,
    },
    {
      shape: '(B, T, D) = (1, 4, 8)',
      line: 12,
      blocks: [flat(true, true)],
      caption: <><Code>@ Wo</Code>：W_O 的每一列都读取全部 8 维，把不同头的输出线性混合。等价写法 <Code>Σ_i o_i @ Wo[i·d_h:(i+1)·d_h]</Code>——MLA 的「W_UV 吸收进 W_O」正是利用了这个分块求和。</>,
    },
  ]
}

const CODE = `def mha(x, Wq, Wk, Wv, Wo, h):
    B, T, D = x.shape
    d_h = D // h
    q = x @ Wq                           # (B, T, D)
    q = q.reshape(B, T, h, d_h)          # (B, T, h, d_h)
    q = q.transpose(0, 2, 1, 3)          # (B, h, T, d_h)
    # k、v 同样拆头
    s = q @ k.transpose(0, 1, 3, 2) / d_h ** 0.5
    o = softmax(s) @ v                   # (B, h, T, d_h)
    o = o.transpose(0, 2, 1, 3)          # (B, T, h, d_h)
    o = o.reshape(B, T, D)               # (B, T, D)
    return o @ Wo`

const WRONG_CODE = CODE.replace('q = q.reshape(B, T, h, d_h)          # (B, T, h, d_h)', 'q = q.reshape(B, h, T, d_h)          # ❌ 直接拆').replace(
  'q = q.transpose(0, 2, 1, 3)          # (B, h, T, d_h)',
  '# （漏了 transpose）',
)

function Grid({ b }: { b: Block }) {
  const cols = b.cells[0].length
  return (
    <div className="min-w-0">
      <div className="mb-1 font-mono text-[11px] text-muted">{b.title}</div>
      <div className="grid gap-[3px]" style={{ gridTemplateColumns: `auto repeat(${cols}, 28px)` }}>
        {b.cells.map((row, r) => (
          <div key={r} className="contents">
            <div className="flex items-center justify-end pr-1 font-mono text-[10px] text-subtle">{b.rowLabels[r]}</div>
            {row.map((c, k) => (
              <div
                key={k}
                className="flex h-7 items-center justify-center rounded-md font-mono text-[10.5px] text-fg"
                style={{
                  background: c.tone ? toneSoft(c.tone, 30) : 'var(--surface)',
                  boxShadow: `inset 0 0 0 ${c.bad ? 2 : 1}px ${c.bad ? toneColor('danger') : c.tone ? toneColor(c.tone) : 'var(--border)'}`,
                }}
              >
                {c.text}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

export function ShapeFlowViz() {
  const [mode, setMode] = useState<'right' | 'wrong'>('right')
  const frames = useMemo(() => buildFrames(mode === 'wrong'), [mode])
  const st = useStepper(frames, { intervalMs: 1800 })
  const f = st.frame
  return (
    <VizFrame
      title="拆头 = reshape + transpose，两步缺一不可"
      caption={f.caption}
      legend={
        <Legend
          items={[
            { label: '头 0 的维度', tone: 1, variant: 'soft' },
            { label: '头 1 的维度', tone: 2, variant: 'soft' },
            { label: '错误切分', tone: 'danger', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <StepControls stepper={st} />
          <Segmented
            size="sm"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'right', label: '正确：reshape → transpose' },
              { value: 'wrong', label: '错误：直接 reshape' },
            ]}
          />
        </>
      }
    >
      <div className="grid items-start gap-4 @3xl:grid-cols-[1.25fr_1fr]">
        <div className="flex min-h-[170px] flex-wrap content-start gap-x-5 gap-y-3">
          <div className="w-full font-mono text-[12.5px] font-semibold text-accent">{f.shape}</div>
          {f.blocks.map((b, i) => (
            <Grid key={i} b={b} />
          ))}
        </div>
        <CodeTrace code={mode === 'wrong' ? WRONG_CODE : CODE} active={f.line} vars={{ B: 1, T, D, h: H, d_h: DH }} />
      </div>
    </VizFrame>
  )
}
