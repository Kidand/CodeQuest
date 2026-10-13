import { defineLevel, md, py } from '@/content/helpers'
import { BlockViz } from '../viz/ArchViz'
import { blockPython,blockTorch,weightSetup } from '../kernels'
import { single,judge,numeric,multi,order,fill,bug } from '../shared'
export default defineLevel({learn:[{id:'model',title:'从 token ID 到词表 logits 的完整路径',body:md`本关构建一个无 batch 轴的迷你因果语言模型：embedding lookup → 多层 GQA/RoPE/SwiGLU block → final RMSNorm → tied LM head。所有权重均显式输入，不使用训练好的大模型，因而可在浏览器中快速检查 shape、参数共享与因果性。重点不是背一个框架 API，而是能解释每个维度、残差连接和 mask 的意义。模型返回 logits，采样策略属于解码层；交叉熵训练时还要做 next-token shift，本关不在前向内部隐藏这一步。`,viz:BlockViz}],quiz:[single('lookup',1,md`E=[V,D]，tokens=[T]，E[tokens] 的形状？`,['[T,D]','[V,T]','[T,V]','[D,V]'],0,md`每个 ID 选择一行长度 D 的 embedding。`),judge('finalnorm',1,md`Pre-Norm 架构的 final norm 可以因为各层有 norm 就自动省略。`,false,md`各层 norm 在分支内部，残差流末端的尺度还需要按模型定义归一化。`),numeric('head',2,md`tied E=[100,16]，LM head 新增多少独立参数？`,0,md`复用 E 转置，不新增独立权重，但仍执行输出投影。`),multi('causal',2,md`可以发现未来泄露的检查包括？`,['对比完整序列前缀 logits 与截断前缀 logits','改变后半输入并比较前半输出','仅检查 logits 的最后维等于 V','检查 mask 是否包括未来列'],[0,1,3],md`shape 无法证明因果性；数值不变量与 mask 审核可以。`),order('forward',md`完整前向顺序：`,['embedding lookup','逐层 decoder block','final RMSNorm','乘 E 的转置得到 logits'],md`输出不是概率，解码 softmax 在模型外。`),fill('rope',md`m、n 同加常数 c，固定 q,k 的 RoPE 内积取决于 ____(1)____。`,['n-m','m-n'],md`依赖相对位置差；符号随内积书写约定，关键是共同平移不变。`),single('gqa',2,md`Q=8头、KV=2头，连续分组的第 5 个 Q 头（下标4）用哪个 KV？`,['0','1','2','4'],1,md`每组4头，下标4属于 KV1。`),judge('cost',3,md`更大上下文只增加 KV，不增加注意力 FLOPs。`,false,md`全量注意力 QK 与 PV 随 T² 增长，decode 单步注意力读写也随历史长度增长。`),numeric('projection',2,md`V=100、D=16、T=2，输出 hEᵀ 按乘加2 FLOPs 计需要多少 FLOPs？`,6400,md`2×T×D×V=6400。共享参数不消除计算。`),bug('transpose',py`import numpy as np
def head(h, embedding, norm, eps):
    variance = (h*h).mean(-1,keepdims=True)
    h = h / np.sqrt(variance+eps)
    h = h * norm
    logits = h @ embedding
    assert logits.ndim == 2
    return logits`,[6],md`embedding 为 [V,D]，应 h@embedding.T 得 [T,V]；恰好 V=D 时该错误形状也能通过。`) ],guided:[],challenges:[{id:'gpt',title:'迷你 GPT：完整 tied 前向',difficulty:'hard',description:md`实现 \`mini_gpt(tokens: np.ndarray, embedding: np.ndarray, layers: list[dict], final_norm: np.ndarray, heads: int, kv_heads: int, eps: float=1e-5) -> np.ndarray\`。tokens 为 [T] int，1≤T≤12；embedding [V,D]，V,D≤32，final_norm [D]；layers 包含 0–3 个上一关规格的 block 参数。请在答案内实现完整 GQA、交错 RoPE、因果注意力与 SwiGLU，不能依赖其它题的函数；可复用上一关自己的实现。输出 [T,V] logits，LM head 与 embedding 共享，禁止 softmax 和 shift。`,starter:py`import numpy as np
def mini_gpt(tokens, embedding, layers, final_norm, heads, kv_heads, eps=1e-5):
    raise NotImplementedError`,solution:py`${blockPython}

def mini_gpt(tokens, embedding, layers, final_norm, heads, kv_heads, eps=1e-5):
    h = embedding[tokens].copy()
    for w in layers:
        h = decoder_block(h, w, heads, kv_heads, eps)
    h = _norm(h, final_norm, eps)
    return h @ embedding.T`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`${blockTorch}

def mini_gpt(tokens,embedding,layers,final_norm,heads,kv_heads,eps=1e-5):
    h=embedding[tokens]
    for w in layers:h=decoder_block(h,w,heads,kv_heads,eps)
    h=h*torch.rsqrt(h.square().mean(-1,keepdim=True)+eps)*final_norm
    return h@embedding.T`}],explanation:md`embedding 查表与 tied 输出投影使用同一 E；每个 block 沿完整序列执行因果计算。最后 RMSNorm 的缩放参数独立于各层 norm，之后得到未归一化 logits。复杂度 $O(L(TD^2+TDF+T^2D)+TVD)$，输出 O(TV)，教学注意力工作区 O(HT²)。`,hints:[md`把 block 函数带入当前答案，避免依赖其它编辑器的状态。`,md`零层模型仍有 embedding、final norm、输出投影。`,md`用前缀不变量检查完整模型的 mask 与位置处理。`],setup:py`${weightSetup}
_rng=np.random.default_rng(44)
_E=_rng.normal(size=(11,8))*.2
_N=_rng.uniform(.8,1.2,8)`,tests:[{name:'零层完整 head',code:py`ids=np.array([1,3]);h=_E[ids];exp=(h/np.sqrt((h*h).mean(-1,keepdims=True)+1e-5)*_N)@_E.T
a=mini_gpt(ids,_E,[],_N,4,2)
assert a.shape==(2,11) and np.allclose(a,exp), f"期望 {exp}，得到 {a}"`},{name:'一层 oracle',code:py`ids=np.array([1,2,4]);w=_weights();h=_oracle(_E[ids],w,4,2);exp=(h/np.sqrt((h*h).mean(-1,keepdims=True)+1e-5)*_N)@_E.T
a=mini_gpt(ids,_E,[w],_N,4,2)
assert np.allclose(a,exp), f"最大误差 {np.max(abs(a-exp))}"`},{name:'完整与截断前缀',hidden:true,code:py`ids=np.array([2,3,5,1,7]);ws=[_weights(seed=1),_weights(seed=2)]
a=mini_gpt(ids,_E,ws,_N,4,2);b=mini_gpt(ids[:3],_E,ws,_N,4,2)
assert np.allclose(a[:3],b), f"前缀误差 {np.max(abs(a[:3]-b))}"`},{name:'两层 MQA 与输入不可变',hidden:true,code:py`ids=np.array([3]);ws=[_weights(kv=1,seed=9),_weights(kv=1,seed=8)];old=_E.copy();h=_E[ids]
for w in ws:h=_oracle(h,w,4,1)
exp=(h/np.sqrt((h*h).mean(-1,keepdims=True)+1e-5)*_N)@_E.T
a=mini_gpt(ids,_E,ws,_N,4,1)
assert np.allclose(a,exp), f"最大误差 {np.max(abs(a-exp))}"
assert np.array_equal(_E,old), "共享 embedding 被修改"`}],followUps:[{id:'cache',q:md`下一步如何改为增量推理？`,a:md`为每层保存历史 K/V，新 token 的 RoPE position 使用缓存长度偏移；查询只算新位置，与历史缓存拼接后注意力。需要保证全量前向与逐 token 前向的 logits 数值一致，再考虑页表、批处理和采样。`}]}],interview:[],cheatsheet:md`E[token] → L×Block → Final RMSNorm → hEᵀ。模型返回 logits；softmax、温度、top-p 属于解码。验证：shape、独立 oracle、前缀一致、共享权重不变。`})
