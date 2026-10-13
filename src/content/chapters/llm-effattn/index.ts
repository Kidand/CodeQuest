import { defineChapter, md } from '@/content/helpers'

export default defineChapter({
id: "llm-effattn",
track: "llm",
order: 4,
title: "高效注意力与长上下文",
subtitle: "SWA · NSA / DSA · CSA / HCA · Gated DeltaNet",
icon: "Zap",
intro: md`通过可操作的小规模模拟理解系统设计：每关从计算与存储语义出发，完成随堂练习、代码跟练、NumPy 手撕与面试追问。参考解另附 PyTorch 展示写法。`,
levels: [
{
id: "llm-effattn-swa",
title: "滑动窗口、Attention Sink 与门控注意力",
summary: "局部感受野、环形缓存与零值 sink 的归一化",
difficulty: 3,
estMinutes: 45,
tier: "modern",
frequency: "medium",
tags: [
"滑动窗口、Attention Sink 与门控注意力"
],
load: () => import('./levels/swa')
},
{
id: "llm-effattn-sparse",
title: "稀疏注意力：NSA 与 DSA",
summary: "从硬件友好分块到轻量 indexer 与 top-k",
difficulty: 3,
estMinutes: 45,
tier: "modern",
frequency: "medium",
tags: [
"稀疏注意力：NSA 与 DSA"
],
load: () => import('./levels/sparse')
},
{
id: "llm-effattn-v4",
title: "DeepSeek-V4：CSA 与 HCA",
summary: "双路通道压缩、动态选择与真实 KV 显存账本",
difficulty: 3,
estMinutes: 45,
tier: "frontier",
frequency: "medium",
tags: [
"DeepSeek-V4：CSA 与 HCA"
],
load: () => import('./levels/v4')
},
{
id: "llm-effattn-linear",
title: "线性注意力与 Gated DeltaNet",
summary: "固定状态、残差写入、通道遗忘与混合模型",
difficulty: 3,
estMinutes: 45,
tier: "frontier",
frequency: "medium",
tags: [
"线性注意力与 Gated DeltaNet"
],
load: () => import('./levels/linear')
},
{
id: "llm-effattn-boss",
title: "BOSS：高效注意力综合",
summary: "在一个正确的因果集合上融合远程、局部与 sink",
difficulty: 5,
estMinutes: 60,
tier: "frontier",
frequency: "medium",
tags: [
"BOSS：高效注意力综合"
],
load: () => import('./levels/boss'),
kind: "boss"
}
]
})
