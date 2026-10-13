/**
 * CodeQuest 内容 Schema —— 全站内容的唯一契约。
 *
 * 结构：Track(路线) → Chapter(章节/世界) → Level(关卡) → Stage(阶段)
 *   每个关卡依次包含 5 个阶段：
 *     learn     图解：分节讲解 + 可交互可视化 + 节内小测
 *     quiz      随堂练习：多题型即时反馈
 *     guided    跟练：代码骨架逐空填写（逐步讲解）
 *     challenge 实战：从零手撕，浏览器内 Pyodide 跑测试
 *     qa        面试追问：闪卡自评
 *
 * 写作规范见 docs/AUTHORING.md。修改本文件需同步更新渲染端与校验脚本。
 */
import type { ComponentType } from 'react'

export type Track = 'algo' | 'llm'
export type Tier = 'must' | 'high' | 'advanced' | 'foundation' | 'modern' | 'frontier'
export const TIER_LABEL: Record<Tier, string> = {
  must: '必考',
  high: '高频',
  advanced: '进阶',
  foundation: '基础必会',
  modern: '新必会',
  frontier: '前沿',
}
export const FREQ_LABEL = { high: '高频', medium: '中频', low: '低频' } as const
export type Stage = 'learn' | 'quiz' | 'guided' | 'challenge' | 'qa'
export const STAGES: Stage[] = ['learn', 'quiz', 'guided', 'challenge', 'qa']

/* ------------------------------------------------------------------ */
/* 章节 / 关卡元数据（轻量，启动时全部加载）                              */
/* ------------------------------------------------------------------ */

export interface ChapterDef {
  /** 全局唯一，kebab-case，与目录名一致，如 'llm-attention' */
  id: string
  track: Track
  /** 路线内排序（1 开始） */
  order: number
  title: string
  subtitle: string
  /** lucide-react 图标名（PascalCase），如 'Search' | 'Brain' | 'Zap' */
  icon: string
  /** 解锁前置章节 id（自由模式下忽略） */
  prerequisites?: string[]
  /** 章节导语（markdown）：本章学什么、面试里怎么考 */
  intro?: string
  levels: LevelMeta[]
}

export interface LevelMeta {
  /** 全局唯一，建议 `${chapterId}-${slug}`，如 'llm-attention-mha' */
  id: string
  title: string
  /** 一句话简介（地图节点 hover / 卡片展示） */
  summary: string
  /** boss = 章节综合关（放在章节最后） */
  kind?: 'normal' | 'boss'
  difficulty: 1 | 2 | 3 | 4 | 5
  /** 预计完成时间（分钟） */
  estMinutes: number
  tags: string[]
  /**
   * 分层（必填）：
   *   经典算法：must 必考 / high 高频 / advanced 进阶
   *   大模型：  foundation 基础必会 / modern 新必会（2023-2025 成为标配）/ frontier 前沿（近两年重点、2026 新进展）
   */
  tier: Tier
  /** 面试出现频率（大模型关卡必填；经典算法由 tier 体现） */
  frequency?: 'high' | 'medium' | 'low'
  /** 代表题，如 ['LC 146 LRU 缓存', 'LC 460 LFU 缓存'] 或 ['手撕 MHA'] */
  problems?: string[]
  /** 经典算法关卡：在大模型岗位/系统中的出现场景（一句话） */
  llmLink?: string
  /** 懒加载关卡正文：() => import('./levels/xxx') */
  load: () => Promise<{ default: LevelContent }>
}

/* ------------------------------------------------------------------ */
/* 关卡正文（懒加载）                                                    */
/* ------------------------------------------------------------------ */

export interface LevelContent {
  learn: LearnSection[]
  quiz: Question[]
  guided: GuidedExercise[]
  challenges: Challenge[]
  interview: InterviewQA[]
  /** 速记卡（markdown）：公式、模板代码、易错点。会汇总进「速查手册」 */
  cheatsheet?: string
}

/** 图解阶段的一节 */
export interface LearnSection {
  id: string
  title: string
  /** 正文（markdown，支持 KaTeX 与 > [!TIP] 等提示块） */
  body: string
  /** 可交互可视化组件（无 props，自带状态） */
  viz?: ComponentType
  /** 可视化下方的说明 / 结论（markdown） */
  after?: string
  /** 节末小测（单题，答对才“点亮”本节，答错不扣分） */
  checkpoint?: Question
}

/* ------------------------------------------------------------------ */
/* 题目                                                                 */
/* ------------------------------------------------------------------ */

interface QBase {
  /** 关卡内唯一 */
  id: string
  /** 题干（markdown） */
  prompt: string
  /** 解析（markdown）：必须说明正确答案为什么对、干扰项为什么错 */
  explanation: string
  hint?: string
  difficulty?: 1 | 2 | 3
}

/** 单选 */
export interface SingleQ extends QBase { type: 'single'; options: string[]; answer: number }
/** 多选（answer 为正确选项下标，至少 2 个） */
export interface MultiQ extends QBase { type: 'multi'; options: string[]; answer: number[] }
/** 判断 */
export interface JudgeQ extends QBase { type: 'judge'; answer: boolean }
/**
 * 填空：题干中用 ①②③ 或 `____(1)____` 标出位置，blanks 依次对应。
 * 文本比较时会做归一化（去空白、统一引号、忽略大小写）。
 */
export interface FillQ extends QBase { type: 'fill'; blanks: FillBlank[] }
export interface FillBlank {
  /** 输入框前的标签，如 '(1)' */
  label?: string
  /** 可接受答案（任一匹配即正确）。numeric=true 时第一个元素为数值字符串 */
  answers: string[]
  /** 按数值比较（支持 1e9 / 2^30 / 3*4 等表达式输入） */
  numeric?: boolean
  /** numeric 时的相对误差容忍度，默认 0.01 */
  tolerance?: number
  placeholder?: string
}
/** 计算题：输入一个数值（支持简单表达式），按相对误差判定 */
export interface NumericQ extends QBase {
  type: 'numeric'
  answer: number
  /** 相对误差容忍度，默认 0.01（1%） */
  tolerance?: number
  /** 单位提示，如 'GB'、'M 参数' */
  unit?: string
}
/** 排序：items 按“正确顺序”书写，界面会打乱 */
export interface OrderQ extends QBase { type: 'order'; items: string[] }
/** 连线配对：pairs 为 [左, 右]，界面打乱右列 */
export interface MatchQ extends QBase { type: 'match'; pairs: [string, string][] }
/**
 * 代码拼图（Parsons）：lines 按正确顺序书写且自带缩进；
 * distractors 为干扰行（不应被选中）。
 */
export interface ParsonsQ extends QBase { type: 'parsons'; lines: string[]; distractors?: string[] }
/** 找 Bug：点击有问题的行（1 起始行号），可多行 */
export interface BugQ extends QBase { type: 'bug'; code: string; bugLines: number[] }
/**
 * 点选：在一组元素中点出满足条件的全部元素。
 * cols 指定网格列数（如 4x4 因果掩码 → items 长度 16, cols=4），
 * rowLabels/colLabels 可选（网格模式下显示行列标签）。
 */
export interface PickQ extends QBase {
  type: 'pick'
  items: string[]
  answer: number[]
  cols?: number
  rowLabels?: string[]
  colLabels?: string[]
}

export type Question = SingleQ | MultiQ | JudgeQ | FillQ | NumericQ | OrderQ | MatchQ | ParsonsQ | BugQ | PickQ
export type QuestionType = Question['type']
export const QUESTION_TYPE_LABEL: Record<QuestionType, string> = {
  single: '单选',
  multi: '多选',
  judge: '判断',
  fill: '填空',
  numeric: '计算',
  order: '排序',
  match: '配对',
  parsons: '代码拼图',
  bug: '找 Bug',
  pick: '点选',
}

/* ------------------------------------------------------------------ */
/* 跟练：代码骨架逐空填写                                                */
/* ------------------------------------------------------------------ */

export interface GuidedExercise {
  id: string
  title: string
  /** 开场说明（markdown）：要实现什么、接口约定 */
  intro: string
  /**
   * Python 代码骨架。空位写作 {{1}}、{{2}}…（与 steps[].blank 对应）。
   * 一个空位通常是一行或一个表达式；多行答案会按占位符所在行的缩进对齐。
   */
  template: string
  /** 按顺序逐个填写的步骤，每步对应一个空位 */
  steps: GuidedStep[]
  /** 隐藏的辅助代码（在用户代码之后执行），参考实现请命名为 _ref_xxx */
  setup?: string
  /** 用完整拼装后的代码运行的测试；也用于“等价写法”判定 */
  tests: TestCase[]
  /** 完成后的总结（markdown） */
  outro?: string
}

export interface GuidedStep {
  /** 对应 template 中的 {{n}} */
  blank: number
  /** 这一步写什么、为什么（markdown） */
  instruction: string
  /** 参考答案（用于拼装、对比、揭晓） */
  answer: string
  /** 其它可接受写法（归一化后比较）；字符串不匹配时还会尝试“跑测试判等价” */
  accept?: string[]
  hint?: string
  /** 答对后展示的讲解（markdown） */
  explain?: string
}

/* ------------------------------------------------------------------ */
/* 实战：从零手撕                                                        */
/* ------------------------------------------------------------------ */

export interface Challenge {
  id: string
  title: string
  difficulty: 'easy' | 'medium' | 'hard'
  /** 题面（markdown）：描述、函数签名、输入输出约定、示例、约束 */
  description: string
  /** 初始代码（含函数签名与 TODO），不得通过测试 */
  starter: string
  /** 参考解（必须通过全部测试，由 npm run verify:py 校验） */
  solution: string
  /** 其它语言/框架的参考写法，仅展示（如 PyTorch 版） */
  solutionAlt?: { label: string; code: string }[]
  /** 题解（markdown）：思路、关键点、复杂度、易错点 */
  explanation: string
  complexity?: { time: string; space: string }
  /** 渐进式提示（markdown），越往后越具体；每看一条扣少量奖励 */
  hints: string[]
  /** 隐藏辅助代码（在用户代码之后执行），参考实现请命名为 _ref_xxx */
  setup?: string
  tests: TestCase[]
  /** 面试官追问 */
  followUps?: InterviewQA[]
  /** 建议用时（分钟），模拟面试计时用 */
  timeLimitMin?: number
  /** 出处，如 'LeetCode 704' / '经典手撕' */
  source?: string
}

/**
 * 测试用例：一段 Python 代码，在“用户代码 + setup”的命名空间副本中执行，
 * 抛出 AssertionError 即失败（assert 的消息会展示给用户，建议写清期望与实际）。
 */
export interface TestCase {
  name: string
  code: string
  /** 隐藏用例：只显示通过与否，不显示代码 */
  hidden?: boolean
}

/* ------------------------------------------------------------------ */
/* 面试追问闪卡                                                          */
/* ------------------------------------------------------------------ */

export interface InterviewQA {
  id: string
  /** 问题（markdown） */
  q: string
  /** 参考回答（markdown），按“结论 → 原因 → 延伸”组织 */
  a: string
  /** 出现频率 1-3 */
  freq?: 1 | 2 | 3
}
