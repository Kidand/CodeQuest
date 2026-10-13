import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { toneColor, toneSoft, type VizTone } from './tones'

export interface TokenItem {
  text: string
  tone?: VizTone
  /** token 下方的小字（id、概率、位置…） */
  sub?: string
  dim?: boolean
  /** 虚线框：草稿/待验证 token */
  dashed?: boolean
}

/** 一行 token 胶囊（分词、解码序列、投机解码草稿…）。空格显示为 ␣ */
export function TokenRow({ tokens, className, onClick }: { tokens: TokenItem[]; className?: string; onClick?: (i: number) => void }) {
  return (
    <div className={cn('flex flex-wrap items-start gap-1.5', className)}>
      {tokens.map((t, i) => (
        <motion.div
          key={i}
          layout
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: t.dim ? 0.38 : 1, scale: 1 }}
          transition={{ duration: 0.2 }}
          onClick={onClick ? () => onClick(i) : undefined}
          className={cn('flex flex-col items-center', onClick && 'cursor-pointer')}
        >
          <span
            className={cn('whitespace-pre rounded-lg border px-2 py-1 font-mono text-[13px] leading-tight', t.dashed && 'border-dashed')}
            style={{
              background: t.tone ? toneSoft(t.tone, 24) : 'var(--surface)',
              borderColor: t.tone ? toneColor(t.tone) : 'var(--border-strong)',
              color: 'var(--fg)',
            }}
          >
            {t.text.replace(/ /g, '␣').replace(/\n/g, '⏎') || '∅'}
          </span>
          {t.sub != null && <span className="mt-0.5 font-mono text-[10px] text-subtle">{t.sub}</span>}
        </motion.div>
      ))}
    </div>
  )
}
