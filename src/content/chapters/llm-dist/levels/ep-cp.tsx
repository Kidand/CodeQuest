import { defineLevel, md, py } from '@/content/helpers'
import { RingViz } from '../viz/DistributedViz'

export default defineLevel({
learn: [
{
id: "expert",
title: "专家参数切开，token 按路由移动",
body: md`EP把专家分布在不同设备；router先为每个token选择若干专家，dispatch把输入送过去，专家计算后combine按路由权重还原输出。top-k意味着一个token可能产生多个派发项，不是每token只传一次。热点专家、跨节点路由与容量限制都会影响负载，drop/padding策略还可能改变训练语义。[DeepEP](https://github.com/deepseek-ai/DeepEP) 是面向这种专家通信的实现，优化通信不等于自动解决路由不均。应分别观察专家token数、跨节点流量和计算通信重叠程度。`,
viz: RingViz,
checkpoint: {
type: "numeric",
id: "dispatch",
prompt: md`100个token，每个top-2路由，忽略本地命中和容量限制，一共形成多少个token-专家派发项？`,
explanation: md`每token两条路由，共100×2=200；不意味着每项都一定跨网络。`,
difficulty: 2,
answer: 200,
tolerance: 0.001
}
},
{
id: "context",
title: "轮转 KV，保持每个 query 的全局分母",
body: md`[Ring Attention](https://arxiv.org/abs/2310.01889) 沿序列分片，各卡保留自己的Q块，让KV沿环依次到达。每卡维护每query的m、l、o统计，使用与FlashAttention相同的稳定合并：$m'=\max(m,m_b)$，旧l和o乘$e^{m-m'}$，加新块贡献。最后统一o/l。分别对块softmax再平均是错误的，因为块间概率质量不同。CP降低单卡序列存储和计算负担，但总数学FLOPs并未自动从T²变成T；通信开销与能否被计算覆盖决定伸缩性。`,
checkpoint: {
type: "numeric",
id: "softmax",
prompt: md`两个块各一个value，分别为0、10；对应logit为0、ln3。全局注意力输出为多少？`,
explanation: md`指数质量为1和3，所以输出(0×1+10×3)/4=7.5；平均块输出误得5。`,
difficulty: 3,
answer: 7.5,
tolerance: 0.001
}
},
{
id: "strategy",
title: "并行组要符合硬件与整除约束",
body: md`[Ulysses](https://arxiv.org/abs/2309.14509) 用all-to-all在序列切分与头切分布局间转换，注意头数等可整除约束；GQA/MQA的KV头更少时尤其需要核对支持方式。多维并行先画设备网格：TP常放高带宽组，PP跨阶段，DP处理独立样本，CP分担长序列，EP负责专家。EP不一定是额外独立维度，某些框架与已有组共享设备集合。最后检查全局batch=DP×microbatch×累积步数，以及rank group是否匹配；只列一串degree而不说明映射，无法判断可运行性。`
}
],
quiz: [
{
type: "single",
id: "ep",
prompt: md`专家并行最核心的跨设备通信模式是什么？`,
explanation: md`token要送到持有目标专家的设备，专家结果再返回；实际可融合或用专用通信库。`,
difficulty: 1,
options: [
md`按路由dispatch与combine的all-to-all`,
md`永远只广播一个标量`,
md`只在最后层all-reduce`,
md`完全不需要通信`
],
answer: 0
},
{
type: "judge",
id: "ringexact",
prompt: md`Ring Attention分块后只要合并online softmax统计量，就可以保持稠密注意力语义。`,
explanation: md`前提是每块恰好访问一次、掩码与缩放一致、分子分母正确重标定；浮点舍入可能不同。`,
difficulty: 1,
answer: true
},
{
type: "numeric",
id: "dispatch",
prompt: md`100个token，每个top-2路由，忽略本地命中和容量限制，一共形成多少个token-专家派发项？`,
explanation: md`每token两条路由，共100×2=200；不意味着每项都一定跨网络。`,
difficulty: 2,
answer: 200,
tolerance: 0.001
},
{
type: "multi",
id: "parallel",
prompt: md`哪些描述正确？`,
explanation: md`EP组可能与数据/张量并行组的布局有重叠或特定组合，不能盲目再乘一个维度。`,
difficulty: 2,
options: [
md`CP可沿序列切分注意力工作`,
md`Ulysses用all-to-all转换序列与头的布局`,
md`EP degree在所有框架中都必须额外乘入DP×TP×PP×CP`,
md`MoE热点专家会引起负载与通信不均`
],
answer: [
0,
1,
3
]
},
{
type: "match",
id: "map",
prompt: md`匹配通信与工作布局`,
explanation: md`三者的通信数据与group语义不同，需要按模型与网络设计。`,
difficulty: 2,
pairs: [
[
md`EP`,
md`把token发给专家再回收`
],
[
md`Ring Attention`,
md`固定query、轮转KV`
],
[
md`Ulysses`,
md`通过all-to-all交换序列与头切分布局`
]
]
},
{
type: "order",
id: "online",
prompt: md`合并下一个KV块的步骤`,
explanation: md`分别算softmax再平均缺少每块的真实质量，不能得到全局softmax。`,
difficulty: 2,
items: [
md`过滤全局因果不可见位置`,
md`求当前块分数和新全局最大值`,
md`重标定旧分子与分母`,
md`累加新块指数质量与加权value`,
md`所有块结束后统一相除`
]
},
{
type: "bug",
id: "merge",
prompt: md`这一处为什么丢失块的归一化质量？`,
explanation: md`应累加mass*output（weights已经是全局归一化块质量）。各块候选数和分数不同，不能平均输出。`,
difficulty: 3,
code: py`def merge(outputs, weights):
    # outputs 是各块局部 softmax 的输出
    # weights 是各块对应的全局概率质量
    total = outputs[0]*0
    for output, mass in zip(outputs, weights):
        total += output/len(outputs)
    result = total
    return result`,
bugLines: [
6
]
},
{
type: "numeric",
id: "softmax",
prompt: md`两个块各一个value，分别为0、10；对应logit为0、ln3。全局注意力输出为多少？`,
explanation: md`指数质量为1和3，所以输出(0×1+10×3)/4=7.5；平均块输出误得5。`,
difficulty: 3,
answer: 7.5,
tolerance: 0.001
}
],
guided: [
{
id: "g-ring-attn",
title: "逐步实现：用 online softmax 合并环上的 KV 块",
intro: md`实现 \`ring_attention(q,k,v,devices)\`。q/k:(T,D)，v:(T,Dv)，有限float64，devices≥1，T≥0。用np.array_split把绝对token索引切为devices块。rank r保留本地query，依次访问来源(r-step) mod devices的KV块，用全局绝对位置做因果mask，并以online softmax合并。返回(T,Dv)，结果应与稠密因果注意力一致；T不必整除devices，允许空块。不能构造完整T×T矩阵，不能平均各块独立softmax输出。`,
template: py`import numpy as np

def ring_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, devices: int) -> np.ndarray:
    n, d = q.shape
    blocks = np.array_split(np.arange(n), devices)
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for rank, queries in enumerate(blocks):
        for t in queries:
            maximum, denominator = -np.inf, 0.0
            numerator = np.zeros(v.shape[1], dtype=np.float64)
            for step in range(devices):
                ids = blocks[(rank - step) % devices]
                ids = {{1}}
                if len(ids) == 0:
                    continue
                logits = q[t] @ k[ids].T / np.sqrt(d)
                new_max = max(maximum, float(logits.max()))
                correction = {{2}}
                weights = np.exp(logits - new_max)
                numerator = correction * numerator + weights @ v[ids]
                denominator = {{3}}
                maximum = new_max
            out[t] = numerator / denominator
    return out`,
steps: [
{
blank: 1,
instruction: md`全局因果mask按绝对token位置判断，整块未来时得到空集合。`,
answer: "ids[ids <= t]",
explain: md`全局因果mask按绝对token位置判断，整块未来时得到空集合。`
},
{
blank: 2,
instruction: md`最大值改变后，将旧分子分母重标定到同一指数尺度。`,
answer: "np.exp(maximum - new_max)",
explain: md`最大值改变后，将旧分子分母重标定到同一指数尺度。`
},
{
blank: 3,
instruction: md`把块的未归一化质量合并，而不是平均块输出。`,
answer: "correction * denominator + weights.sum()",
explain: md`把块的未归一化质量合并，而不是平均块输出。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_dense(q,k,v):
    n=len(q)
    if n==0:return np.empty((0,v.shape[1]))
    s=q@k.T/np.sqrt(q.shape[1]);s=np.where(np.tri(n,dtype=bool),s,-np.inf)
    p=np.exp(s-s.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
    return p@v`,
tests: [
{
name: "三位置均匀因果注意力",
code: py`q=np.zeros((3,1));v=np.array([[1.],[3.],[8.]])
_check(ring_attention(q,q,v,2),[[1.],[2.],[4.]])`,
hidden: false
},
{
name: "极大logit跨块重标定",
code: py`q=np.array([[1000.],[1000.]]);k=np.array([[1.],[2.]]);v=np.array([[1.,2.],[3.,4.]])
_check(ring_attention(q,k,v,2),[[1.,2.],[3.,4.]])`,
hidden: false
},
{
name: "改变未来不会影响过去",
code: py`r=np.random.default_rng(51);q=r.normal(size=(8,3));k=r.normal(size=(8,3));v=r.normal(size=(8,2))
a=ring_attention(q,k,v,3);k[4:]*=100;v[4:]+=100
_check(ring_attention(q,k,v,3)[:4],a[:4])`,
hidden: true
}
]
}
],
challenges: [
{
id: "ring-attn",
title: "用 online softmax 合并环上的 KV 块",
difficulty: "hard",
timeLimitMin: 25,
description: md`实现 \`ring_attention(q,k,v,devices)\`。q/k:(T,D)，v:(T,Dv)，有限float64，devices≥1，T≥0。用np.array_split把绝对token索引切为devices块。rank r保留本地query，依次访问来源(r-step) mod devices的KV块，用全局绝对位置做因果mask，并以online softmax合并。返回(T,Dv)，结果应与稠密因果注意力一致；T不必整除devices，允许空块。不能构造完整T×T矩阵，不能平均各块独立softmax输出。`,
starter: py`import numpy as np

def ring_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, devices: int) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def ring_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, devices: int) -> np.ndarray:
    n, d = q.shape
    blocks = np.array_split(np.arange(n), devices)
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for rank, queries in enumerate(blocks):
        for t in queries:
            maximum, denominator = -np.inf, 0.0
            numerator = np.zeros(v.shape[1], dtype=np.float64)
            for step in range(devices):
                ids = blocks[(rank - step) % devices]
                ids = ids[ids <= t]
                if len(ids) == 0:
                    continue
                logits = q[t] @ k[ids].T / np.sqrt(d)
                new_max = max(maximum, float(logits.max()))
                correction = np.exp(maximum - new_max)
                weights = np.exp(logits - new_max)
                numerator = correction * numerator + weights @ v[ids]
                denominator = correction * denominator + weights.sum()
                maximum = new_max
            out[t] = numerator / denominator
    return out`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def ring_attention(q,k,v,devices):
    blocks=torch.tensor_split(torch.arange(len(q),device=q.device),devices)
    result=q.new_zeros((len(q),v.shape[1]))
    for rank,queries in enumerate(blocks):
        for t in queries:
            maximum=q.new_tensor(-torch.inf);denominator=q.new_tensor(0.)
            numerator=q.new_zeros(v.shape[1])
            for step in range(devices):
                ids=blocks[(rank-step)%devices];ids=ids[ids<=t]
                if ids.numel()==0:continue
                logits=q[t]@k[ids].T/q.shape[1]**0.5
                new_max=torch.maximum(maximum,logits.max())
                correction=torch.exp(maximum-new_max);weights=torch.exp(logits-new_max)
                numerator=correction*numerator+weights@v[ids]
                denominator=correction*denominator+weights.sum();maximum=new_max
            result[t]=numerator/denominator
    return result`
}
],
explanation: md`每个query维护全局最大值m、平移后指数和l、未归一化加权值o。新块使最大值变大时，旧l/o要同时乘exp(m−m_new)。最后输出o/l。全遮块直接跳过，避免−∞减−∞；绝对因果位置不能用本地块下标替代。教学实现总FLOPs仍O(T²(D+Dv))，单query工作空间O(T/devices+Dv)；真实CP让设备并行持有自己的Q块并重叠KV传输。`,
hints: [
md`全局因果mask按绝对token位置判断，整块未来时得到空集合。`,
md`最大值改变后，将旧分子分母重标定到同一指数尺度。`,
md`把块的未归一化质量合并，而不是平均块输出。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_dense(q,k,v):
    n=len(q)
    if n==0:return np.empty((0,v.shape[1]))
    s=q@k.T/np.sqrt(q.shape[1]);s=np.where(np.tri(n,dtype=bool),s,-np.inf)
    p=np.exp(s-s.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
    return p@v`,
tests: [
{
name: "三位置均匀因果注意力",
code: py`q=np.zeros((3,1));v=np.array([[1.],[3.],[8.]])
_check(ring_attention(q,q,v,2),[[1.],[2.],[4.]])`,
hidden: false
},
{
name: "极大logit跨块重标定",
code: py`q=np.array([[1000.],[1000.]]);k=np.array([[1.],[2.]]);v=np.array([[1.,2.],[3.,4.]])
_check(ring_attention(q,k,v,2),[[1.,2.],[3.,4.]])`,
hidden: false
},
{
name: "空块与单卡",
code: py`r=np.random.default_rng(35)
for n in [0,1,2,9]:
 q=r.normal(size=(n,3));k=r.normal(size=(n,3));v=r.normal(size=(n,2))
 for devices in [1,2,4,12]:
  _check(ring_attention(q,k,v,devices),_ref_dense(q,k,v))`,
hidden: true
},
{
name: "改变未来不会影响过去",
code: py`r=np.random.default_rng(51);q=r.normal(size=(8,3));k=r.normal(size=(8,3));v=r.normal(size=(8,2))
a=ring_attention(q,k,v,3);k[4:]*=100;v[4:]+=100
_check(ring_attention(q,k,v,3)[:4],a[:4])`,
hidden: true
}
],
followUps: [
{
id: "follow",
q: md`怎样检验实现没有悄悄改变语义？`,
a: md`先用小矩阵与独立的直接定义对拍，再分别验证边界、形状与数值稳定性。性能优化允许浮点舍入差异，但不能改变掩码、归一化的集合或通信求和的语义。加入会使错误版本明显失败的反例，比只看正常随机输入更可靠。`,
freq: 2
}
]
}
],
interview: [
{
id: "merge",
q: md`为什么块局部softmax的输出无法直接相加平均？`,
a: md`每块内部归一化丢失了该块相对于其它块的指数质量。两个单元素块无论logit多大，局部权重都是1，但它们在全局softmax中的质量可能相差上百倍。需要保留logsumexp或m/l统计，先把块质量换算到同一尺度，再加权合并输出。`,
freq: 3
},
{
id: "mask",
q: md`序列切分后因果mask最容易错在哪里？`,
a: md`将每个块的本地下标直接代入下三角mask。query块和key块的绝对起点不同，有的整块位于未来，有的整块都可见，边界块才需要逐元素判断。应使用全局position构造条件j≤i；完全不可见块直接跳过，以免online softmax出现−∞减−∞造成NaN。`,
freq: 3
},
{
id: "ep",
q: md`EP为什么可能成为跨机瓶颈？`,
a: md`token按数据相关路由动态分布，热门专家会形成热点，且top-k扩展派发量。小消息、网络拓扑和跨节点带宽会限制吞吐；容量padding又可能增加无效传输。改善需要结合负载均衡、专家放置、通信融合与重叠，不能只换一个all-to-all实现就假设所有不均衡消失。`,
freq: 2
}
],
cheatsheet: md`EP：dispatch → expert → combine；派发量约token数×top-k。

CP环：Q不动，KV轮转；每块访问一次；因果用绝对位置。

online：新最大值m′；旧l/o同时乘exp(m−m′)；最后o/l。

网格先说明group映射；EP不盲目额外乘入world size。`
})
