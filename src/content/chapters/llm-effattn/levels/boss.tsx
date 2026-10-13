import { defineLevel, md, py } from '@/content/helpers'
import { SparseViz } from '../viz/EfficientViz'

export default defineLevel({
learn: [
{
id: "review",
title: "先说明改变了什么，再说明快在哪里",
body: md`这一章把效率分成三类：减少可见边（窗口/稀疏），减少历史条目（压缩），把历史折叠成固定状态（线性递推）。它们都可能改变模型能力；FlashAttention则在既定注意力上优化执行。综合题要求把「候选是谁、何时可用、如何归一化」写成可测试的约定。本关使用远程top-k与滑窗的集合并集，再加入零值sink；它不是分别计算两条分支后门控求和。面试中先说清这点，才能避免把多种论文结构拼成一个未经定义的公式。`,
viz: SparseViz
}
],
quiz: [
{
type: "single",
id: "choose",
prompt: md`完全保留稠密注意力数学定义，同时优化 IO，应优先考虑？`,
explanation: md`FlashAttention是精确分块执行；其他选项改变连接或历史表示。`,
difficulty: 1,
options: [
md`FlashAttention`,
md`任意截断top-k`,
md`把每128个token求均值`,
md`固定状态delta递推`
],
answer: 0
},
{
type: "judge",
id: "window",
prompt: md`窗口与top-k列表有交集时，直接拼接等价于集合并集。`,
explanation: md`重复项在softmax中多占一份指数质量；必须明确采用集合并集还是不同分支，各自语义不同。`,
difficulty: 1,
answer: false
},
{
type: "numeric",
id: "union",
prompt: md`远程集合{0,2,4}，窗口{3,4,5}，并集有多少个真实位置？`,
explanation: md`交集只有4，3+3−1=5；sink是额外虚拟槽，不属于真实位置。`,
difficulty: 2,
answer: 5,
tolerance: 0.001
},
{
type: "multi",
id: "select",
prompt: md`严格因果的压缩/稀疏系统应满足？`,
explanation: md`先选未来会改变过去候选，事后删除仍不安全；另外三项是重要不变量。`,
difficulty: 2,
options: [
md`未来token扰动不改变过去输出`,
md`摘要只在对应信息可用时发布`,
md`先对全序列top-k再删掉未来项总安全`,
md`位置元数据与环形槽分开`
],
answer: [
0,
1,
3
]
},
{
type: "match",
id: "scope",
prompt: md`匹配节省的主要对象`,
explanation: md`它们解决不同瓶颈，因此可以组合，但每种都要检查质量代价。`,
difficulty: 2,
pairs: [
[
md`SWA`,
md`长期局部缓存与候选数量`
],
[
md`序列压缩`,
md`长期历史条目数`
],
[
md`动态选择`,
md`当前读取工作集`
],
[
md`线性递推`,
md`用固定状态替代逐token缓存`
]
]
},
{
type: "order",
id: "review",
prompt: md`实现前的合理审查顺序`,
explanation: md`语义没定清之前谈kernel加速，可能只是在更快计算错误结果。`,
difficulty: 2,
items: [
md`定义因果与位置语义`,
md`定义候选集合与重复处理`,
md`实现稳定归一化`,
md`对拍边界和随机用例`,
md`评测内存与端到端时延`
]
},
{
type: "bug",
id: "dup",
prompt: md`以下融合代码哪一行违背集合并集约定？`,
explanation: md`应使用 np.union1d 或等价去重方法；仅重排索引不能消除重复概率质量。`,
difficulty: 3,
code: py`import numpy as np
def combine(selected,local,q,k,v):
    ids = np.concatenate([selected,local])
    scores = q @ k[ids].T
    scores -= scores.max()
    p = np.exp(scores)
    p /= p.sum()
    return p @ v[ids]`,
bugLines: [
3
]
},
{
type: "numeric",
id: "state",
prompt: md`32层中8层全局注意力，每层每token KV为1024字节，T=1000，忽略其它层固定状态与额外开销。全局KV多少字节？`,
explanation: md`8×1000×1024=8,192,000。不能把所有32层都当全局，也不能省略剩余24层的实际固定状态。`,
difficulty: 3,
answer: 8192000,
tolerance: 0.001
},
{
type: "judge",
id: "ratio",
prompt: md`Kimi Linear 的3:1配置说明所有任务用75%线性层都一定最优。`,
explanation: md`它是在特定训练设置下的架构选择；任务、模型规模、硬件都会改变折中。`,
difficulty: 2,
answer: false
},
{
type: "numeric",
id: "read",
prompt: md`T=8192，m=4，CSA长期摘要2048项；top-k=128。忽略局部窗口，当前最多读多少项摘要？`,
explanation: md`缓存2048项，而当前查询最多读取128项；存储与读取是两张不同账。`,
difficulty: 2,
answer: 128,
tolerance: 0.001
}
],
guided: [
{
id: "g-hybrid",
title: "逐步实现：融合远程候选、局部窗口与 sink",
intro: md`实现 \`hybrid_attention(q,k,v,scores,count,window,sink)\`。q/k:(T,D)，v:(T,Dv)，scores:(T,T)，均有限 float64；count≥0，window≥1。每个t从历史scores[t,:t+1]选count个最高分（同分较小索引优先），与窗口取并集去重，做带零value sink的注意力；返回(T,Dv)。T可为0。此题是明确约定的稀疏集合融合，不代表 NSA 的分支门控公式或 V4 完整实现。额外工作空间 O(T)，不计输入与输出。`,
template: py`import numpy as np

def hybrid_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scores: np.ndarray, count: int, window: int, sink: float) -> np.ndarray:
    n, d = q.shape
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for t in range(n):
        selected = np.argsort(-{{1}}, kind="stable")[:count]
        local = np.arange(max(0,t-window+1), t+1)
        ids = {{2}}
        logits = q[t] @ k[ids].T / np.sqrt(d)
        m = max(float(logits.max()), float(sink))
        p = np.exp(logits - m)
        out[t] = p @ v[ids] / {{3}}
    return out`,
steps: [
{
blank: 1,
instruction: md`先限制因果候选，未来分数不应影响输出。`,
answer: "scores[t, :t+1]",
explain: md`先限制因果候选，未来分数不应影响输出。`
},
{
blank: 2,
instruction: md`局部与远程重叠的 token 只出现一次。`,
answer: "np.union1d(selected, local)",
explain: md`局部与远程重叠的 token 只出现一次。`
},
{
blank: 3,
instruction: md`真实集合与零值 sink 一起归一化。`,
answer: "(p.sum() + np.exp(sink-m))",
explain: md`真实集合与零值 sink 一起归一化。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_hybrid(q,k,v,scores,count,window,sink):
    result=[]
    for t in range(len(q)):
        selected=sorted(range(t+1),key=lambda j:(-scores[t,j],j))[:count]
        ids=sorted(set(selected)|set(range(max(0,t-window+1),t+1)))
        s=[float(q[t]@k[j])/np.sqrt(q.shape[1]) for j in ids]+[sink];m=max(s)
        p=np.exp(np.array(s)-m);p/=p.sum()
        result.append(sum(p[a]*v[j] for a,j in enumerate(ids)))
    return np.asarray(result).reshape(len(q),v.shape[1])`,
tests: [
{
name: "窗口与top-k重叠只计一次",
code: py`q=np.zeros((2,1));v=np.array([[2.],[6.]]);s=np.ones((2,2))
_check(hybrid_attention(q,q,v,s,2,2,0.),[[1.],[8/3]])`,
hidden: false
},
{
name: "零个远程候选也能使用窗口",
code: py`q=np.zeros((3,1));v=np.arange(2.,8.,2.)[:,None];s=np.zeros((3,3))
_check(hybrid_attention(q,q,v,s,0,1,0.),v/2)`,
hidden: false
},
{
name: "未来扰动不改变过去",
code: py`r=np.random.default_rng(7);n=8;q=r.normal(size=(n,2));k=r.normal(size=(n,2));v=r.normal(size=(n,3));s=r.normal(size=(n,n))
a=hybrid_attention(q,k,v,s,2,2,0.)
k[4:]+=100;v[4:]-=100;s[:,4:]+=999
b=hybrid_attention(q,k,v,s,2,2,0.)
_check(a[:4],b[:4])`,
hidden: true
}
]
}
],
challenges: [
{
id: "hybrid",
title: "融合远程候选、局部窗口与 sink",
difficulty: "hard",
timeLimitMin: 25,
description: md`实现 \`hybrid_attention(q,k,v,scores,count,window,sink)\`。q/k:(T,D)，v:(T,Dv)，scores:(T,T)，均有限 float64；count≥0，window≥1。每个t从历史scores[t,:t+1]选count个最高分（同分较小索引优先），与窗口取并集去重，做带零value sink的注意力；返回(T,Dv)。T可为0。此题是明确约定的稀疏集合融合，不代表 NSA 的分支门控公式或 V4 完整实现。额外工作空间 O(T)，不计输入与输出。`,
starter: py`import numpy as np

def hybrid_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scores: np.ndarray, count: int, window: int, sink: float) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def hybrid_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scores: np.ndarray, count: int, window: int, sink: float) -> np.ndarray:
    n, d = q.shape
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for t in range(n):
        selected = np.argsort(-scores[t, :t+1], kind="stable")[:count]
        local = np.arange(max(0,t-window+1), t+1)
        ids = np.union1d(selected, local)
        logits = q[t] @ k[ids].T / np.sqrt(d)
        m = max(float(logits.max()), float(sink))
        p = np.exp(logits - m)
        out[t] = p @ v[ids] / (p.sum() + np.exp(sink-m))
    return out`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def hybrid_attention(q,k,v,scores,count,window,sink):
    result=[]
    for t in range(len(q)):
        selected=torch.argsort(scores[t,:t+1],descending=True,stable=True)[:count]
        local=torch.arange(max(0,t-window+1),t+1,device=q.device)
        ids=torch.unique(torch.cat([selected,local]))
        s=q[t]@k[ids].T/q.shape[1]**0.5
        p=torch.softmax(torch.cat([s,s.new_tensor([sink])]),dim=0)[:-1]
        result.append(p@v[ids])
    return torch.stack(result) if result else v.new_empty((0,v.shape[1]))`
}
],
explanation: md`窗口和远程集合必须去重，否则重叠 token 在 softmax 分母中出现两次，相当于人为增加其 logit。先限制因果前缀，再选择；之后合并集合，一次归一化。此教学方案和先分别归一化两个分支再门控相加具有不同语义。稳定全排序总时间含 O(T² log T)，实际可用选择算法降低排序成本。`,
hints: [
md`先限制因果候选，未来分数不应影响输出。`,
md`局部与远程重叠的 token 只出现一次。`,
md`真实集合与零值 sink 一起归一化。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_hybrid(q,k,v,scores,count,window,sink):
    result=[]
    for t in range(len(q)):
        selected=sorted(range(t+1),key=lambda j:(-scores[t,j],j))[:count]
        ids=sorted(set(selected)|set(range(max(0,t-window+1),t+1)))
        s=[float(q[t]@k[j])/np.sqrt(q.shape[1]) for j in ids]+[sink];m=max(s)
        p=np.exp(np.array(s)-m);p/=p.sum()
        result.append(sum(p[a]*v[j] for a,j in enumerate(ids)))
    return np.asarray(result).reshape(len(q),v.shape[1])`,
tests: [
{
name: "窗口与top-k重叠只计一次",
code: py`q=np.zeros((2,1));v=np.array([[2.],[6.]]);s=np.ones((2,2))
_check(hybrid_attention(q,q,v,s,2,2,0.),[[1.],[8/3]])`,
hidden: false
},
{
name: "零个远程候选也能使用窗口",
code: py`q=np.zeros((3,1));v=np.arange(2.,8.,2.)[:,None];s=np.zeros((3,3))
_check(hybrid_attention(q,q,v,s,0,1,0.),v/2)`,
hidden: false
},
{
name: "空与随机独立对拍",
code: py`r=np.random.default_rng(11)
for n in [0,1,8]:
 q=r.normal(size=(n,3));k=r.normal(size=(n,3));v=r.normal(size=(n,2));s=r.normal(size=(n,n))
 for count in [0,2,20]:
  _check(hybrid_attention(q,k,v,s,count,3,1.),_ref_hybrid(q,k,v,s,count,3,1.))`,
hidden: true
},
{
name: "未来扰动不改变过去",
code: py`r=np.random.default_rng(7);n=8;q=r.normal(size=(n,2));k=r.normal(size=(n,2));v=r.normal(size=(n,3));s=r.normal(size=(n,n))
a=hybrid_attention(q,k,v,s,2,2,0.)
k[4:]+=100;v[4:]-=100;s[:,4:]+=999
b=hybrid_attention(q,k,v,s,2,2,0.)
_check(a[:4],b[:4])`,
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
id: "design",
q: md`长文检索、显存紧张，应该如何选择方案？`,
a: md`先确认瓶颈是持久KV、每步带宽还是prefill计算。稀疏读取主要减少工作集，压缩或混合线性层才改变长期历史表示；全局层保留精细检索能力。选择后用远距离检索和真实任务验证质量，再测端到端延迟及峰值显存。不能凭一个渐近复杂度就判断最终方案。`,
freq: 3
},
{
id: "invariant",
q: md`最值得写进隐藏测试的因果不变量是什么？`,
a: md`固定前缀，任意改变后缀的q/k/v及选择分数，前缀输出应保持不变。这个测试能同时抓住错误掩码、全序列top-k后过滤和提前读取完成块摘要等问题。浮点允许合理容差，但不能接受未来内容改变过去输出。再加集合去重与空输入测试，能覆盖许多正常随机对拍容易遗漏的错误。`,
freq: 3
},
{
id: "benchmark",
q: md`一个可信的加速结论需要哪些条件？`,
a: md`先说明baseline、硬件、精度、batch、上下文和mask，再保证任务质量或数值语义可比。计时应包含indexer、选择、数据搬运和同步，峰值显存应含临时缓冲。稀疏率、压缩率和速度比并不是同一个指标，不能用理论读取量替代实测服务时延。`,
freq: 2
}
],
cheatsheet: md`复盘顺序：语义 → 因果 → 集合 → 归一化 → 对拍 → 性能。

最常见四坑：未来参与top-k；窗口与远程重复；sink未计分母；缓存长度用k代替T。

分别报告：持久状态、当前工作集、总FLOPs、实测时延、任务质量。`
})
