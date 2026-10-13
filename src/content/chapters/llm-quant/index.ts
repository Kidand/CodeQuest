import { defineChapter, md } from '@/content/helpers'
export default defineChapter({id:'llm-quant',track:'llm',order:7,title:'微调与量化',subtitle:'LoRA · 分组量化 · SmoothQuant · 低精度格式',icon:'SlidersHorizontal',intro:md`从低秩更新到离散数值网格，区分存储、计算和训练精度。用有限差分、量化误差界与打包反解验证工程实现。`,levels:[
{id:'llm-quant-lora',title:'LoRA 家族',summary:'低秩前向、梯度、merge、QLoRA 与 DoRA',difficulty:3,estMinutes:45,tier:'foundation',frequency:'high',tags:['LoRA','QLoRA','DoRA'],load:()=>import('./levels/lora')},
{id:'llm-quant-basics',title:'量化基础',summary:'对称/非对称、分组粒度、误差与存储',difficulty:3,estMinutes:45,tier:'foundation',frequency:'high',tags:['INT4','INT8','分组量化'],load:()=>import('./levels/basics')},
{id:'llm-quant-methods',title:'LLM 量化方法',summary:'离群值、SmoothQuant、GPTQ、AWQ 与 KV 量化',difficulty:3,estMinutes:45,tier:'modern',frequency:'medium',tags:['SmoothQuant','GPTQ','AWQ'],load:()=>import('./levels/methods')},
{id:'llm-quant-lowprec',title:'低精度格式：FP8 / MXFP4 / NVFP4',summary:'浮点编码、块尺度、FP8训练与MXFP4量化',difficulty:4,estMinutes:60,tier:'modern',frequency:'medium',tags:['FP8','MXFP4','NVFP4'],load:()=>import('./levels/lowprec')},
{id:'llm-quant-boss',title:'BOSS：量化综合',summary:'打包 INT4 基座与高精度 LoRA 前向',kind:'boss',difficulty:4,estMinutes:55,tier:'modern',frequency:'medium',tags:['综合实战','nibble'],load:()=>import('./levels/boss')}
]})
