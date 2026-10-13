import { defineChapter, md } from '@/content/helpers'

export default defineChapter({
id: "llm-dist",
track: "llm",
order: 11,
title: "分布式训练",
subtitle: "显存账本 · ZeRO · TP / SP · PP · EP / CP",
icon: "Network",
intro: md`通过可操作的小规模模拟理解系统设计：每关从计算与存储语义出发，完成随堂练习、代码跟练、NumPy 手撕与面试追问。参考解另附 PyTorch 展示写法。`,
levels: [
{
id: "llm-dist-memory",
title: "显存账本与混合精度",
summary: "拆分模型状态、激活、临时缓冲与数值范围",
difficulty: 3,
estMinutes: 45,
tier: "foundation",
frequency: "high",
tags: [
"显存账本与混合精度"
],
load: () => import('./levels/memory')
},
{
id: "llm-dist-dp",
title: "数据并行、All-Reduce 与 ZeRO",
summary: "环通信的块所有权与逐阶段状态分片",
difficulty: 3,
estMinutes: 45,
tier: "foundation",
frequency: "high",
tags: [
"数据并行、All-Reduce 与 ZeRO"
],
load: () => import('./levels/dp')
},
{
id: "llm-dist-tp",
title: "张量并行与序列并行",
summary: "从轴切分推导通信位置与词表归一化",
difficulty: 3,
estMinutes: 45,
tier: "modern",
frequency: "medium",
tags: [
"张量并行与序列并行"
],
load: () => import('./levels/tp')
},
{
id: "llm-dist-pp",
title: "流水线并行",
summary: "从气泡公式到依赖正确的 1F1B 事件序列",
difficulty: 3,
estMinutes: 45,
tier: "modern",
frequency: "medium",
tags: [
"流水线并行"
],
load: () => import('./levels/pp')
},
{
id: "llm-dist-ep-cp",
title: "专家并行、上下文并行与并行策略",
summary: "按数据所有权选择 all-to-all、环传与精确归约",
difficulty: 3,
estMinutes: 45,
tier: "modern",
frequency: "medium",
tags: [
"专家并行、上下文并行与并行策略"
],
load: () => import('./levels/ep-cp')
},
{
id: "llm-dist-boss",
title: "BOSS：分布式综合",
summary: "从显存预算筛选可执行配置，再判断通信代价",
difficulty: 5,
estMinutes: 60,
tier: "modern",
frequency: "high",
tags: [
"BOSS：分布式综合"
],
load: () => import('./levels/boss'),
kind: "boss"
}
]
})
