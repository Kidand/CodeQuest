import { defineChapter, md } from '@/content/helpers'

export default defineChapter({
  id: 'llm-moe',
  track: 'llm',
  order: 6,
  title: '混合专家 MoE',
  subtitle: '路由与手撕 · 负载均衡 · DeepSeekMoE · 2026 前沿',
  icon: 'Boxes',
  prerequisites: ['llm-arch'],
  intro: md`
    2024 年之后几乎所有一线开源大模型都是 MoE：DeepSeek-V3/V4、Qwen3/3.5、Kimi K2/K3、gpt-oss……
    面试里 MoE 是「手撕 + 原理 + 计算」三合一的高频考点：**手写 MoE 层前向**、**讲清负载均衡（aux loss / z-loss / 容量）**、
    **DeepSeekMoE 为什么能去掉辅助损失**，以及**总参数与激活参数怎么算**。

    本章从 router、top-k、dispatch / combine 写起，再到 Switch / ST-MoE 的均衡与稳定性技巧、DeepSeekMoE 的细粒度 + 共享专家与无辅助损失偏置均衡，
    最后是 2026 年的前沿：DeepSeek-V4 的 Sqrt(Softplus) 亲和度与 Hash 路由、Kimi K3 的 896 选 16 与 Quantile Balancing。
    所有模型配置均按论文与官方 config 核对。
  `,
  levels: [
    {
      id: 'llm-moe-basics',
      title: 'MoE 基础与手撕',
      summary: 'router → top-k → 是否重归一化 → dispatch → 专家 FFN → 加权 combine；总参数 vs 激活参数',
      difficulty: 3,
      estMinutes: 50,
      tier: 'modern',
      frequency: 'high',
      tags: ['MoE', 'router', 'top-k', 'dispatch', '参数量'],
      problems: ['手撕 MoE 层前向', 'MoE 总参数 / 激活参数计算'],
      load: () => import('./levels/basics'),
    },
    {
      id: 'llm-moe-balance',
      title: '负载均衡与容量',
      summary: '专家坍缩、Switch aux loss 为何是 f·P、router z-loss 与 bf16 舍入、capacity factor 与 token 丢弃',
      difficulty: 4,
      estMinutes: 55,
      tier: 'modern',
      frequency: 'high',
      tags: ['负载均衡', 'aux loss', 'z-loss', 'capacity factor'],
      problems: ['aux loss + z-loss 前向与梯度', '按容量分发（GShard 式）'],
      load: () => import('./levels/balance'),
    },
    {
      id: 'llm-moe-deepseek',
      title: 'DeepSeekMoE 与无辅助损失均衡',
      summary: '细粒度 + 共享专家、sigmoid 门控、bias 只管选择不管权重、节点受限路由、序列级均衡损失',
      difficulty: 4,
      estMinutes: 55,
      tier: 'modern',
      frequency: 'high',
      tags: ['DeepSeekMoE', '共享专家', '无辅助损失', '节点受限路由'],
      problems: ['手撕 DeepSeek-V3 门控（分组 top-k）', '多步 bias 更新模拟'],
      load: () => import('./levels/deepseek'),
    },
    {
      id: 'llm-moe-frontier',
      title: '2026 MoE 前沿',
      summary: 'DeepSeek-V4 的 Sqrt(Softplus) 与 Hash 路由、Kimi K3 的 LatentMoE 与 Quantile Balancing、激活比例的真实趋势',
      difficulty: 4,
      estMinutes: 50,
      tier: 'frontier',
      frequency: 'medium',
      tags: ['DeepSeek-V4', 'Hash 路由', 'Kimi K3', 'Quantile Balancing', '激活比例'],
      problems: ['手撕 V4 门控（含 Hash 层）', 'Quantile Balancing 更新'],
      load: () => import('./levels/frontier'),
    },
    {
      id: 'llm-moe-boss',
      title: 'BOSS：MoE 综合',
      summary: '全章混合题 + 两道压轴手撕：完整 DeepSeekMoE 层（含序列级均衡损失）、排序式 permute / unpermute 分发',
      kind: 'boss',
      difficulty: 5,
      estMinutes: 60,
      tier: 'modern',
      frequency: 'high',
      tags: ['MoE', 'DeepSeekMoE', 'permute', 'grouped GEMM'],
      problems: ['手撕完整 DeepSeekMoE 层', '排序式分发（MegaBlocks 风格）'],
      load: () => import('./levels/boss'),
    },
  ],
})
