import { defineLevel, md, py } from '@/content/helpers'
import { DeltaViz } from '../viz/EfficientViz'

export default defineLevel({
learn: [
{
id: "models",
title: "同为 3:1，具体层数仍要查配置",
body: md`Qwen3-Next-80B-A3B的[官方配置](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)写明full_attention_interval=4；Qwen3.5-397B-A17B的[配置](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)明确重复三个linear_attention加一个full_attention。[Kimi K3报告](https://arxiv.org/html/2607.24653v1#S2.SS1) 则在3个KDA加1个Gated MLA的模式外，末尾额外放一个Gated MLA，因此不能把全模型层数机械按3:1整分。KDA沿key通道施加Diag(α)后再做delta更新；与本关标量α递推的区别，是可以选择性遗忘不同状态方向。面试回答应给出具体型号、层表与状态维度，避免从系列名推断所有变体相同。`
},
{
id: "kernel",
title: "把历史累积进矩阵状态",
body: md`若相似度可分解为 $\phi(q)^\top\phi(k)$，则 $S_t=S_{t-1}+\phi(k_t)v_t^\top$、$z_t=z_{t-1}+\phi(k_t)$，输出为 $\phi(q_t)^\top S_t/(\phi(q_t)^\top z_t)$。这利用矩阵乘法结合律，使解码状态大小与T无关；代价是固定容量和核函数选择。分母为0时要定义安全行为。它通常不是 softmax 的精确重排。加入衰减后，旧信息权重按门控乘积逐渐降低，RetNet/Mamba2 等结构用不同参数化组织这种记忆。理解状态形状比笼统背「线性复杂度」更有用。`,
viz: DeltaViz,
checkpoint: {
type: "numeric",
id: "state",
prompt: md`每头状态 Dk=64、Dv=128，共8头，FP32每数4字节，仅状态共多少字节？`,
explanation: md`8×64×128×4=262144，不随已处理序列长度增长。`,
difficulty: 2,
answer: 262144,
tolerance: 0.001
}
},
{
id: "delta",
title: "只写入预测误差，避免相同 key 无限制累加",
body: md`本题以 $S\in\mathbb{R}^{d_k\times d_v}$ 表示状态：先 U=αS，再 r=v−kᵀU，最后 $S'=U+\beta kr^\top$。与反复累加kvᵀ相比，delta rule 会纠正同一key关联的旧值；单位key、β=1时，该方向更新为新值。先衰减再算残差是 [Gated DeltaNet](https://arxiv.org/abs/2412.06464) 的核心组合。KDA 进一步让遗忘更细粒度地作用于通道，不能简单等同于每头一个标量门。稳定训练还涉及Q/K归一化、参数化与输出门；练习只实现可明确验证的递推核心。`,
checkpoint: {
type: "numeric",
id: "overwrite",
prompt: md`一维 S_old=4，k=q=1，v=10，alpha=0.5、beta=0.25，更新后输出为多少？`,
explanation: md`先衰减到2，残差为8，写回0.25×8=2，得到4。若用旧状态算残差会得到3.5。`,
difficulty: 3,
answer: 4,
tolerance: 0.001
}
},
{
id: "parallel",
title: "递推适合解码，分块适合训练",
body: md`把更新展开为 $S_t=A_tS_{t-1}+B_t$ 后，相邻更新可通过 $(A_2A_1,A_2B_1+B_2)$ 组合。这种结合性提供并行 scan 或 chunkwise 计算的基础，但朴素构造完整A矩阵会很贵；高效内核利用其对角、低秩等结构。训练时可块内并行、块间传状态，解码时逐 token 更新。固定状态容易在精确检索上遇到容量瓶颈，因此混合全局注意力仍有价值。[Kimi Linear](https://arxiv.org/abs/2510.26692) 研究3:1的KDA与MLA混合；这是该模型的设计选择，不是所有混合模型普适最优比例。`
}
],
quiz: [
{
type: "single",
id: "memory",
prompt: md`线性注意力递推通常用什么替代随长度增长的全部 KV？`,
explanation: md`状态把历史聚合进有限维矩阵；它不是保留原始历史的等价存储。`,
difficulty: 1,
options: [
md`固定形状的矩阵状态`,
md`长度固定为1的原始token`,
md`完整T×T概率矩阵`,
md`动态词表`
],
answer: 0
},
{
type: "numeric",
id: "state",
prompt: md`每头状态 Dk=64、Dv=128，共8头，FP32每数4字节，仅状态共多少字节？`,
explanation: md`8×64×128×4=262144，不随已处理序列长度增长。`,
difficulty: 2,
answer: 262144,
tolerance: 0.001
},
{
type: "judge",
id: "exact",
prompt: md`把 softmax 注意力写成任何有限维 φ(q)φ(k)ᵀ，就总能保持完全等价。`,
explanation: md`一般是替换或近似核；不能在没有条件时把 softmax 的非线性两两作用当作有限维精确分解。`,
difficulty: 1,
answer: false
},
{
type: "multi",
id: "gates",
prompt: md`关于本题 gated delta rule，哪些正确？`,
explanation: md`beta=0 只是不写入，旧状态仍按 alpha 衰减；只有 alpha=0 等条件才会清空。`,
difficulty: 2,
options: [
md`alpha 决定整体遗忘`,
md`beta 决定残差写入量`,
md`beta=0 时状态一定归零`,
md`残差应读取衰减后的状态`
],
answer: [
0,
1,
3
]
},
{
type: "match",
id: "mapping",
prompt: md`匹配技术与关键想法`,
explanation: md`它们不是只换名称；遗忘和写入规则决定状态表达能力与并行实现。`,
difficulty: 2,
pairs: [
[
md`累加线性注意力`,
md`外积不断累积`
],
[
md`Delta rule`,
md`按预测残差纠正记忆`
],
[
md`KDA`,
md`更细粒度通道遗忘门`
]
]
},
{
type: "order",
id: "delta-order",
prompt: md`本题一个 token 的更新顺序`,
explanation: md`把残差放到衰减之前改变了数学定义；输出前后读取也需服从题设。`,
difficulty: 2,
items: [
md`衰减旧状态`,
md`用 key 读取预测`,
md`计算真实值减预测的残差`,
md`沿 key 外积写入残差`,
md`用 query 读取更新后的状态`
]
},
{
type: "bug",
id: "residual",
prompt: md`点出应使用衰减状态却用了旧状态的一行。`,
explanation: md`prediction 应为 k@decayed；否则alpha改变了保留量，却未一致改变需覆盖的预测。`,
difficulty: 3,
code: py`import numpy as np
def step(s,k,v,alpha,beta):
    old = s.copy()
    decayed = alpha * old
    prediction = k @ old
    residual = v - prediction
    new = decayed + beta*np.outer(k,residual)
    return new`,
bugLines: [
5
]
},
{
type: "numeric",
id: "overwrite",
prompt: md`一维 S_old=4，k=q=1，v=10，alpha=0.5、beta=0.25，更新后输出为多少？`,
explanation: md`先衰减到2，残差为8，写回0.25×8=2，得到4。若用旧状态算残差会得到3.5。`,
difficulty: 3,
answer: 4,
tolerance: 0.001
}
],
guided: [
{
id: "g-delta",
title: "逐步实现：手写 gated delta rule",
intro: md`实现 \`gated_delta(q,k,v,alpha,beta)\`。q/k:(T,Dk)，v:(T,Dv)，alpha/beta:(T,)，有限 float64；门在[0,1]，k 已逐行 L2 归一化，T 可为0。状态 S:(Dk,Dv) 初始0。按顺序计算 U=alpha[t]S、r=v[t]−k[t]U、S=U+beta[t]outer(k[t],r)，输出 q[t]S。返回(T,Dv)，不再对结果做 softmax 或分母归一化。本题是核心递推，省略实际模型的卷积、输出门等。时间 O(T Dk Dv)，额外状态 O(Dk Dv)。`,
template: py`import numpy as np

def gated_delta(q: np.ndarray, k: np.ndarray, v: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n, dk = q.shape
    state = np.zeros((dk, v.shape[1]), dtype=np.float64)
    out = np.empty((n, v.shape[1]), dtype=np.float64)
    for t in range(n):
        state = {{1}}
        residual = {{2}}
        state = state + {{3}}
        out[t] = q[t] @ state
    return out`,
steps: [
{
blank: 1,
instruction: md`先遗忘旧状态，残差必须读取衰减后的版本。`,
answer: "alpha[t] * state",
explain: md`先遗忘旧状态，残差必须读取衰减后的版本。`
},
{
blank: 2,
instruction: md`预测误差是新 value 减去当前记忆的预测。`,
answer: "v[t] - k[t] @ state",
explain: md`预测误差是新 value 减去当前记忆的预测。`
},
{
blank: 3,
instruction: md`外积沿 key 方向写回残差，beta 控制更新幅度。`,
answer: "beta[t] * np.outer(k[t], residual)",
explain: md`外积沿 key 方向写回残差，beta 控制更新幅度。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_delta(q,k,v,alpha,beta):
    state=np.zeros((q.shape[1],v.shape[1]));out=[]
    for t in range(len(q)):
        transition=alpha[t]*(np.eye(q.shape[1])-beta[t]*np.outer(k[t],k[t]))
        state=transition@state+beta[t]*np.outer(k[t],v[t])
        out.append(q[t]@state)
    return np.asarray(out).reshape(len(q),v.shape[1])`,
tests: [
{
name: "单位 key 完整覆盖",
code: py`q=np.ones((3,1));v=np.array([[1.],[3.],[-2.]])
_check(gated_delta(q,q,v,np.ones(3),np.ones(3)),v)`,
hidden: false
},
{
name: "只遗忘不更新",
code: py`q=np.ones((3,1));v=np.ones((3,1));a=np.array([1.,.5,.5]);b=np.array([1.,0.,0.])
_check(gated_delta(q,q,v,a,b),[[1.],[.5],[.25]])`,
hidden: false
},
{
name: "矩阵状态转移对拍",
code: py`r=np.random.default_rng(44)
for n in [1,7,20]:
 q=r.normal(size=(n,3));k=r.normal(size=(n,3));k/=np.linalg.norm(k,axis=1,keepdims=True);v=r.normal(size=(n,2));a=r.uniform(size=n);b=r.uniform(size=n)
 _check(gated_delta(q,k,v,a,b),_ref_delta(q,k,v,a,b))`,
hidden: true
}
]
}
],
challenges: [
{
id: "kernel-recurrence",
title: "证明递推与显式因果核注意力一致",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`kernel_attention(q,k,v,decay)\`。q/k:(T,D)是已经过特征映射φ的非负有限float64向量，v:(T,Dv)可正可负；T≥0，D/Dv≥1，decay∈[0,1]。权重的直接定义是a[t,s]=(q[t]·k[s])·decay^(t−s)，s≤t，然后按行归一化；0⁰约定1，整行质量为0则输出零。返回(T,Dv)，必须用固定矩阵状态递推，不构造T×T矩阵。测试用独立显式矩阵定义验证等价；这不是softmax注意力。`,
starter: py`import numpy as np

def kernel_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, decay: float) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def kernel_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, decay: float) -> np.ndarray:
    n, d = q.shape
    state = np.zeros((d, v.shape[1]), dtype=np.float64)
    normalizer = np.zeros(d, dtype=np.float64)
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for t in range(n):
        state = decay * state + np.outer(k[t], v[t])
        normalizer = decay * normalizer + k[t]
        denominator = q[t] @ normalizer
        if denominator > 0:
            out[t] = q[t] @ state / denominator
    return out`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def kernel_attention(q,k,v,decay):
    state=q.new_zeros((q.shape[1],v.shape[1]));normalizer=q.new_zeros(q.shape[1]);result=[]
    for t in range(len(q)):
        state=decay*state+torch.outer(k[t],v[t]);normalizer=decay*normalizer+k[t]
        den=q[t]@normalizer
        result.append(q[t]@state/den if den>0 else v.new_zeros(v.shape[1]))
    return torch.stack(result) if result else v.new_empty((0,v.shape[1]))`
}
],
explanation: md`把S与z递推展开，历史第s项被乘decay^(t−s)，恰好对应显式核权重。矩阵乘法结合律将逐历史读取变成固定状态读取。S和z必须使用同一个衰减，否则分子分母不一致。时间O(TDDv)，状态O(DDv+D)，相对显式T²矩阵节省序列相关工作空间。`,
hints: [
md`衰减历史外积和后写入当前项。`,
md`分母状态与分子保持相同衰减。`,
md`用同一query读取分子并除以实际质量。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_kernel(q,k,v,decay):
    n=len(q);i=np.arange(n);lag=i[:,None]-i[None,:]
    weights=(q@k.T)*np.where(lag>=0,decay**np.maximum(lag,0),0)
    den=weights.sum(1,keepdims=True)
    p=np.divide(weights,den,out=np.zeros_like(weights),where=den>0)
    return p@v`,
tests: [
{
name: "无衰减的加权平均",
code: py`q=np.ones((3,1));v=np.array([[1.],[3.],[8.]])
_check(kernel_attention(q,q,v,1.),[[1.],[2.],[4.]])`,
hidden: false
},
{
name: "衰减零等于只读当前",
code: py`q=np.ones((3,1));v=np.array([[1.],[3.],[8.]])
_check(kernel_attention(q,q,v,0.),v)`,
hidden: false
},
{
name: "分母零与空序列",
code: py`q=np.zeros((3,2));v=np.ones((3,4));_check(kernel_attention(q,q,v,.9),np.zeros((3,4)))
_check(kernel_attention(np.empty((0,2)),np.empty((0,2)),np.empty((0,3)),.9),np.empty((0,3)))`,
hidden: true
},
{
name: "递推与并行显式矩阵一致",
code: py`r=np.random.default_rng(13)
for n in [1,5,13]:
 q=r.uniform(size=(n,3));k=r.uniform(size=(n,3));v=r.normal(size=(n,2))
 for decay in [0.,.4,1.]:
  _check(kernel_attention(q,k,v,decay),_ref_kernel(q,k,v,decay))`,
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
},
{
id: "delta",
title: "手写 gated delta rule",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`gated_delta(q,k,v,alpha,beta)\`。q/k:(T,Dk)，v:(T,Dv)，alpha/beta:(T,)，有限 float64；门在[0,1]，k 已逐行 L2 归一化，T 可为0。状态 S:(Dk,Dv) 初始0。按顺序计算 U=alpha[t]S、r=v[t]−k[t]U、S=U+beta[t]outer(k[t],r)，输出 q[t]S。返回(T,Dv)，不再对结果做 softmax 或分母归一化。本题是核心递推，省略实际模型的卷积、输出门等。时间 O(T Dk Dv)，额外状态 O(Dk Dv)。`,
starter: py`import numpy as np

def gated_delta(q: np.ndarray, k: np.ndarray, v: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def gated_delta(q: np.ndarray, k: np.ndarray, v: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n, dk = q.shape
    state = np.zeros((dk, v.shape[1]), dtype=np.float64)
    out = np.empty((n, v.shape[1]), dtype=np.float64)
    for t in range(n):
        state = alpha[t] * state
        residual = v[t] - k[t] @ state
        state = state + beta[t] * np.outer(k[t], residual)
        out[t] = q[t] @ state
    return out`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def gated_delta(q, k, v, alpha, beta):
    state=q.new_zeros((q.shape[1],v.shape[1]));result=[]
    for t in range(len(q)):
        decayed=alpha[t]*state
        residual=v[t]-k[t]@decayed
        state=decayed+beta[t]*torch.outer(k[t],residual)
        result.append(q[t]@state)
    return torch.stack(result) if result else v.new_empty((0,v.shape[1]))`
}
],
explanation: md`先衰减旧状态，再用衰减后的状态预测当前 value，最后只补预测残差。将式子展开得到 S=α(I−βkkᵀ)S_old+βkvᵀ，可用独立矩阵转移验证。β=1 且 key 单位范数时，同一个 key 的值可被精确覆盖；β=0 只有遗忘。与累加式线性注意力不同，这里没有额外 softmax 分母。`,
hints: [
md`先遗忘旧状态，残差必须读取衰减后的版本。`,
md`预测误差是新 value 减去当前记忆的预测。`,
md`外积沿 key 方向写回残差，beta 控制更新幅度。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_delta(q,k,v,alpha,beta):
    state=np.zeros((q.shape[1],v.shape[1]));out=[]
    for t in range(len(q)):
        transition=alpha[t]*(np.eye(q.shape[1])-beta[t]*np.outer(k[t],k[t]))
        state=transition@state+beta[t]*np.outer(k[t],v[t])
        out.append(q[t]@state)
    return np.asarray(out).reshape(len(q),v.shape[1])`,
tests: [
{
name: "单位 key 完整覆盖",
code: py`q=np.ones((3,1));v=np.array([[1.],[3.],[-2.]])
_check(gated_delta(q,q,v,np.ones(3),np.ones(3)),v)`,
hidden: false
},
{
name: "只遗忘不更新",
code: py`q=np.ones((3,1));v=np.ones((3,1));a=np.array([1.,.5,.5]);b=np.array([1.,0.,0.])
_check(gated_delta(q,q,v,a,b),[[1.],[.5],[.25]])`,
hidden: false
},
{
name: "空序列与清空状态",
code: py`_check(gated_delta(np.empty((0,2)),np.empty((0,2)),np.empty((0,3)),np.empty(0),np.empty(0)),np.empty((0,3)))
q=np.ones((2,1));_check(gated_delta(q,q,np.array([[8.],[3.]]),np.array([1.,0.]),np.array([1.,.5])),[[8.],[1.5]])`,
hidden: true
},
{
name: "矩阵状态转移对拍",
code: py`r=np.random.default_rng(44)
for n in [1,7,20]:
 q=r.normal(size=(n,3));k=r.normal(size=(n,3));k/=np.linalg.norm(k,axis=1,keepdims=True);v=r.normal(size=(n,2));a=r.uniform(size=n);b=r.uniform(size=n)
 _check(gated_delta(q,k,v,a,b),_ref_delta(q,k,v,a,b))`,
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
id: "normalization",
q: md`核化线性注意力和 delta rule 都要除以累计分母吗？`,
a: md`不一定。归一化核注意力通过z状态构造分母，使输出具有归一化加权平均的解释；本题delta递推定义的是关联记忆更新，并没有同一个分母。把一种公式中的归一化直接搬到另一种，会改变模型。面试时先约定状态方向、读写顺序和输出定义，再讨论数值稳定。`,
freq: 3
},
{
id: "chunk",
q: md`为什么递推公式还能在训练时并行？`,
a: md`每一步对旧状态是一个仿射变换，变换的复合满足结合律。因此可以把块内的多步作用合成为一个变换，再做块间scan。但数学可并行不等于GPU高效；完整转移矩阵会扩大计算量，实际方法利用低秩更新和专门的分块表示保持矩阵乘友好。`,
freq: 3
},
{
id: "hybrid",
q: md`固定状态这么省，为什么还保留全局注意力？`,
a: md`固定状态必须把越来越多历史压进有限容量，精确检索、复制和多关联存储可能互相干扰。全局注意力保留历史KV，给需要细节的层提供直接访问路径。混合层用一部分缓存与计算换取能力，比例应在目标任务和硬件上验证；3:1只是已公开架构中的具体取舍，不能当作定理。`,
freq: 2
}
],
cheatsheet: md`- 归一化核注意力：$S+=\phi(k)v^\top$，$z+=\phi(k)$。
- Gated delta：U=αS；r=v−kᵀU；S=U+βkrᵀ。
- 状态：每头 Dk×Dv；β=0只遗忘；α=0先清空。
- 仿射组合：$(A_2A_1,A_2B_1+B_2)$。
- 固定状态节省缓存，但检索容量有限。`
})
