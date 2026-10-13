/**
 * vLLM Automatic Prefix Caching：每个满块的哈希 = hash(父块哈希, 本块 token, 额外键)。
 * 哈希链把「整个前缀」编码进每个块的键里：任何一处不同，之后所有块的哈希都不同。
 * 这里用 FNV-1a 32 位做演示（vLLM 0.11 起默认 sha256）。
 */
import { useMemo, useState } from 'react'
import { Legend, VizFrame, toneColor } from '@/viz'
import { Segmented } from '@/components/ui'
import { C } from './common'

function fnv(s: string) {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h >>> 0
}

const SYS = '你是一个乐于助人的助手。'.split('')

const SCENES = {
  share: {
    label: '共享 system prompt',
    a: { toks: [...SYS, ...'讲个笑话吧'.split('')], extra: '' },
    b: { toks: [...SYS, ...'帮我写首诗'.split('')], extra: '' },
    note: '两个请求共用 12 个 token 的 system prompt：完全落在它里面的满块哈希相同、直接命中；从第一个含有不同 token 的块开始，哈希全都不同（块越大，命中的粒度越粗）。',
  },
  edit: {
    label: '开头改了一个字',
    a: { toks: [...SYS, ...'讲个笑话吧'.split('')], extra: '' },
    b: { toks: ['您', ...SYS.slice(1), ...'讲个笑话吧'.split('')], extra: '' },
    note: '只改了第 1 个 token，后面 16 个 token 一模一样——但每个块的哈希都包含父块哈希，第 1 块变了，后面全部跟着变，一个块也命中不了。KV 依赖整个前缀，这正是我们想要的语义。',
  },
  lora: {
    label: '不同 LoRA',
    a: { toks: [...SYS, ...'讲个笑话吧'.split('')], extra: 'lora=A' },
    b: { toks: [...SYS, ...'讲个笑话吧'.split('')], extra: 'lora=B' },
    note: 'token 完全相同，但挂的 LoRA 适配器不同，同一前缀算出的 K/V 也不同。LoRA ID 作为额外键参与每个块的哈希，所以不会错误命中。多模态输入的图像哈希、cache salt 也是这样处理的。',
  },
} as const
type SceneKey = keyof typeof SCENES

function chain(toks: string[], B: number, extra: string) {
  const out: { toks: string[]; hash: number | null }[] = []
  let parent = 0
  for (let s = 0; s < toks.length; s += B) {
    const blk = toks.slice(s, s + B)
    if (blk.length < B) {
      out.push({ toks: blk, hash: null })
      break
    }
    const h = fnv(`${parent}|${blk.join(',')}|${extra}`)
    out.push({ toks: blk, hash: h })
    parent = h
  }
  return out
}

const hex = (h: number) => h.toString(16).padStart(8, '0').slice(0, 6)

export function BlockHashViz() {
  const [scene, setScene] = useState<SceneKey>('share')
  const [B, setB] = useState(4)
  const sc = SCENES[scene]
  const A = useMemo(() => chain([...sc.a.toks], B, sc.a.extra), [sc, B])
  const Bc = useMemo(() => chain([...sc.b.toks], B, sc.b.extra), [sc, B])
  const cacheA = new Set(A.filter((x) => x.hash != null).map((x) => x.hash))
  let hit = 0
  for (const blk of Bc) {
    if (blk.hash != null && cacheA.has(blk.hash)) hit++
    else break
  }

  const Row = ({ name, blocks, extra, isB }: { name: string; blocks: typeof A; extra: string; isB?: boolean }) => (
    <div className="min-w-0">
      <div className="mb-1 font-mono text-[11.5px] text-muted">
        {name}
        {extra && <span className="ml-2 text-llm">{extra}</span>}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {blocks.map((b, i) => {
          const ok = isB ? b.hash != null && i < hit : b.hash != null && Bc[i]?.hash === b.hash && i < hit
          const tone = b.hash == null ? 'muted' : ok ? 'success' : isB ? 'warning' : 'accent'
          return (
            <div
              key={i}
              className="rounded-lg border px-1.5 py-1"
              style={{
                borderColor: toneColor(tone),
                borderStyle: b.hash == null ? 'dashed' : 'solid',
                background: `color-mix(in oklab, ${toneColor(tone)} ${b.hash == null ? 4 : 14}%, var(--surface))`,
              }}
            >
              <div className="font-mono text-[12.5px] tracking-wide text-fg">{b.toks.join('')}</div>
              <div className="font-mono text-[10.5px] text-muted">{b.hash == null ? '未满·不缓存' : `#${hex(b.hash)}`}</div>
            </div>
          )
        })}
      </div>
    </div>
  )

  return (
    <VizFrame
      title="链式块哈希：前缀变了，后面全变"
      caption={
        <>
          {sc.note} 请求 2 命中 <b>{hit}</b> 个块 = <b>{hit * B}</b> 个 token。哈希 <C>h_i = hash(h_(i-1), 块内 token, 额外键)</C>，只为<b>满块</b>计算。
        </>
      }
      legend={
        <Legend
          items={[
            { label: '命中（哈希在缓存中）', tone: 'success', variant: 'soft' },
            { label: '未命中，需要 prefill', tone: 'warning', variant: 'soft' },
            { label: '最后不满的块不参与缓存', tone: 'muted', variant: 'soft' },
          ]}
        />
      }
      controls={
        <>
          <Segmented size="sm" value={scene} onChange={setScene} options={(Object.keys(SCENES) as SceneKey[]).map((k) => ({ value: k, label: SCENES[k].label }))} />
          <div className="flex items-center gap-2 text-[13px] text-muted">
            block_size
            <Segmented size="sm" value={B} onChange={setB} options={[2, 4, 8].map((v) => ({ value: v, label: String(v) }))} />
          </div>
        </>
      }
    >
      <div className="space-y-3">
        <Row name="请求 1（已缓存）" blocks={A} extra={sc.a.extra} />
        <Row name="请求 2（查缓存）" blocks={Bc} extra={sc.b.extra} isB />
      </div>
    </VizFrame>
  )
}
