import { defineChapter, md } from '@/content/helpers'
export default defineChapter({id:'llm-arch',track:'llm',order:5,title:'位置编码与模型架构',subtitle:'RoPE · 长上下文 · Decoder · 计算账本 · 残差演进',icon:'Layers',intro:md`从二维旋转推导相对位置，把 GQA、SwiGLU 和残差组装成可运行语言模型；以算子账本估算成本，并辨析前沿残差连接。`,levels:[
{id:'llm-arch-rope',title:'RoPE 旋转位置编码',summary:'相对位置推导、两种布局与 partial RoPE',difficulty:3,estMinutes:45,tier:'foundation',frequency:'high',tags:['RoPE','位置编码'],load:()=>import('./levels/rope')},
{id:'llm-arch-longctx',title:'长上下文外推',summary:'PI、NTK-aware、YaRN、ABF 与评测',difficulty:3,estMinutes:45,tier:'modern',frequency:'high',tags:['YaRN','长上下文'],load:()=>import('./levels/longctx')},
{id:'llm-arch-block',title:'Decoder Block 组装',summary:'完整 GQA+RoPE+SwiGLU 前向与因果验证',difficulty:4,estMinutes:60,tier:'foundation',frequency:'high',tags:['GQA','SwiGLU','Pre-Norm'],load:()=>import('./levels/block')},
{id:'llm-arch-compute',title:'参数量、FLOPs 与显存估算',summary:'逐算子参数账本、6ND、MFU 与 KV 缓存',difficulty:3,estMinutes:40,tier:'foundation',frequency:'high',tags:['FLOPs','显存','Chinchilla'],load:()=>import('./levels/compute')},
{id:'llm-arch-residual',title:'残差流演进',summary:'HC、mHC 双随机约束与 Attention Residuals',difficulty:4,estMinutes:50,tier:'frontier',frequency:'medium',tags:['mHC','Sinkhorn','AttnRes'],load:()=>import('./levels/residual')},
{id:'llm-arch-boss',title:'BOSS：架构综合',summary:'从 token ID 到词表 logits 的迷你 GPT',kind:'boss',difficulty:5,estMinutes:65,tier:'foundation',frequency:'high',tags:['综合实战'],load:()=>import('./levels/boss')}
]})
