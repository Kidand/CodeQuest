/**
 * 章节图标：显式映射 ChapterDef.icon（lucide 名称）→ 组件。
 * 只引入这里列出的图标，避免把整个 lucide 打进包里；未知名称回退 BookOpen。
 */
import {
  ArrowUpDown,
  BookOpen,
  Boxes,
  Brain,
  BrainCircuit,
  Compass,
  Cpu,
  Gauge,
  Grid3x3,
  HeartHandshake,
  Layers,
  Network,
  Rocket,
  ScanEye,
  Search,
  Server,
  Shrink,
  Sigma,
  Type,
  Undo2,
  Zap,
  type LucideIcon,
  type LucideProps,
} from 'lucide-react'

const ICONS: Record<string, LucideIcon> = {
  // 经典算法
  Search,
  ArrowUpDown,
  Layers,
  Network,
  Grid3x3,
  Grid3X3: Grid3x3,
  Undo2,
  // 大模型
  Sigma,
  Type,
  ScanEye,
  Gauge,
  Compass,
  Boxes,
  Shrink,
  HeartHandshake,
  BrainCircuit,
  Rocket,
  Server,
  // 常见备选
  Brain,
  Cpu,
  Zap,
  BookOpen,
}

export function chapterIconOf(name: string | undefined): LucideIcon {
  return (name && ICONS[name]) || BookOpen
}

export function ChapterIcon({ name, ...rest }: { name?: string } & LucideProps) {
  const Icon = chapterIconOf(name)
  return <Icon {...rest} />
}

export default ChapterIcon
