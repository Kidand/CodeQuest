import { defineLevel, md, py } from '@/content/helpers'
import { SparseViz } from '../viz/EfficientViz'

export default defineLevel({
learn: [
{
id: "nsa",
title: "稀疏模式既要选得准，也要读得快",
body: md`在 T×T 矩阵上把大部分项置零，不会自动省掉计算；必须让 kernel 跳过被删的块或条目。[NSA](https://arxiv.org/abs/2502.11089) 将粗粒度压缩、重要块选择和局部滑窗结合，再学习门控组合。压缩摘要负责全局覆盖，选择恢复重要区域的细节，窗口维持局部连续依赖。块选择比随机零散 token 更容易形成连续访存与矩阵乘。代价是块内也可能有无用 token，压缩又会损失细节；因此稀疏注意力通常需要训练适配，不能把任意稠密模型直接截断后假设质量不变。`,
viz: SparseViz,
checkpoint: {
type: "single",
id: "branches",
prompt: md`NSA 的三个分支是什么？`,
explanation: md`NSA 按不同粒度覆盖全局摘要、重要块和局部细节；Q/K/V 是注意力的通用输入。`,
difficulty: 1,
options: [
md`压缩、选择、滑动窗口`,
md`Q、K、V`,
md`预训练、SFT、RL`,
md`FP32、BF16、FP8`
],
answer: 0
}
},
{
id: "dsa",
title: "Indexing 与真正注意力是两个问题",
body: md`[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) 的 indexer 先估计相关性：$I_{t,s}=\sum_h w_{t,h}\operatorname{ReLU}(q^I_{t,h}\cdot k^I_s)$，然后选择位置交给主 MLA。这里 ReLU 在逐头点积之后，求和之前；索引权重与最终注意力权重不是同一个对象。只选历史位置十分关键：先对未来也打分、选完再过滤，会使未来影响可见集合。并列分数在数学上有多个合法 top-k；测试和部署复现需要明确 tie-break。教学代码采用稳定排序，生产通常使用专门选择 kernel。`,
checkpoint: {
type: "numeric",
id: "score",
prompt: md`两个 indexer 头点积是 −2、3，gate 都为 1。先逐头 ReLU 再加权求和，分数是多少？`,
explanation: md`ReLU(−2)+ReLU(3)=0+3=3；先求和后 ReLU 会误得1。`,
difficulty: 2,
answer: 3,
tolerance: 0.001
}
},
{
id: "cost",
title: "不要把主分支的复杂度当总复杂度",
body: md`主注意力读 k 个候选，可把昂贵的高维计算从 O(T²d) 降到 O(Tkd)。但一个逐 query 扫描历史的 indexer 仍做 O(T²Hᵢdᵢ) 评分；低维、少头和低精度只改善常数。解码时稀疏读取还可能破坏连续访存，要把索引生成、top-k、gather 和主 kernel 都纳入测速。稀疏选择也不等于缓存只存 k 项，因为下一个 query 可能选中不同的历史。正确的报告应分开列计算量、缓存大小和端到端时延，并用检索召回率与下游任务评估选择误差。`
}
],
quiz: [
{
type: "single",
id: "branches",
prompt: md`NSA 的三个分支是什么？`,
explanation: md`NSA 按不同粒度覆盖全局摘要、重要块和局部细节；Q/K/V 是注意力的通用输入。`,
difficulty: 1,
options: [
md`压缩、选择、滑动窗口`,
md`Q、K、V`,
md`预训练、SFT、RL`,
md`FP32、BF16、FP8`
],
answer: 0
},
{
type: "judge",
id: "linear",
prompt: md`DSA 只选 k 个 KV，所以包含 indexer 在内的整体 prefill 一定严格 O(Tk)。`,
explanation: md`主注意力可缩减到 O(Tk)，但扫描全前缀的 indexer 仍有 T² 评分；低维和低精度降低常数而非次数。`,
difficulty: 1,
answer: false
},
{
type: "numeric",
id: "score",
prompt: md`两个 indexer 头点积是 −2、3，gate 都为 1。先逐头 ReLU 再加权求和，分数是多少？`,
explanation: md`ReLU(−2)+ReLU(3)=0+3=3；先求和后 ReLU 会误得1。`,
difficulty: 2,
answer: 3,
tolerance: 0.001
},
{
type: "multi",
id: "correct",
prompt: md`哪些实现决定会影响正确性？`,
explanation: md`因果性、归一化集合与并列约定都影响输出。indexer 分数是筛选信号，不是最终主注意力概率。`,
difficulty: 2,
options: [
md`top-k 前排除未来位置`,
md`主 softmax 只在选中项归一化`,
md`把 indexer 分数直接当最终概率`,
md`固定相同分数的 tie-break`
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
prompt: md`匹配机制`,
explanation: md`NSA 的多分支设计与 DSA 的索引选择不是同一实现，不要把分支名称机械套用。`,
difficulty: 2,
pairs: [
[
md`NSA 压缩分支`,
md`提供粗粒度全局信息`
],
[
md`NSA 选择分支`,
md`访问重要的细粒度块`
],
[
md`DSA indexer`,
md`低成本估计候选相关性`
]
]
},
{
type: "order",
id: "steps",
prompt: md`按顺序组成一个 DSA 教学前向`,
explanation: md`筛选分数和主分数的投影可以不同，读取顺序也可以重新排序以改善访存。`,
difficulty: 2,
items: [
md`计算因果候选的 indexer 分数`,
md`选择 top-k 位置`,
md`读取对应主 KV`,
md`主注意力归一化并聚合`
]
},
{
type: "bug",
id: "future",
prompt: md`这一行为什么会泄露未来？点出最早出错行。`,
explanation: md`需先将评分候选限制在 k[:t+1]。全序列先 top-k 再删除未来，会让未来内容影响留下的历史数量及集合。`,
difficulty: 3,
code: py`import numpy as np
def select(q, k, t, count):
    # t 是当前 token 位置
    score = q[t] @ k.T
    ids = np.argsort(-score)[:count]
    valid = ids[ids <= t]
    chosen = valid.tolist()
    return chosen`,
bugLines: [
4
]
},
{
type: "numeric",
id: "cache",
prompt: md`T=32768、k=1024。忽略 indexer、滑窗与头维，主注意力候选数减少为原来的多少分之一？填写分母。`,
explanation: md`32768/1024=32。这只描述每 query 的主注意力读取，不能直接推出总时延加速32倍。`,
difficulty: 3,
answer: 32,
tolerance: 0.001
}
],
guided: [
{
id: "g-indexer",
title: "逐步实现：Indexer 选择与稀疏注意力",
intro: md`实现 \`sparse_attention(qi, ki, gate, q, k, v, topk)\`。qi:(T,H,Di)，ki:(T,Di)，gate:(T,H)，q/k:(T,D)，v:(T,Dv)，全部有限 float64，topk≥1；T 可为0。索引分数为各头 gate·ReLU(qi·ki) 的和，只允许 s≤t。同分按较小索引优先；选择 min(topk,t+1) 项后，使用独立 q/k 做标准注意力，返回 (T,Dv)。教学实现允许稳定全排序；应说明生产 top-k 的复杂度优势。`,
template: py`import numpy as np

def sparse_attention(qi: np.ndarray, ki: np.ndarray, gate: np.ndarray, q: np.ndarray, k: np.ndarray, v: np.ndarray, topk: int) -> np.ndarray:
    n = q.shape[0]
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for t in range(n):
        score = np.sum(gate[t, :, None] * {{1}}, axis=0)
        ids = {{2}}
        logits = {{3}}
        p = np.exp(logits - logits.max())
        out[t] = (p @ v[ids]) / p.sum()
    return out`,
steps: [
{
blank: 1,
instruction: md`逐头 ReLU，且 indexer 只能看到因果前缀。`,
answer: "np.maximum(qi[t] @ ki[:t+1].T, 0)",
explain: md`逐头 ReLU，且 indexer 只能看到因果前缀。`
},
{
blank: 2,
instruction: md`降序稳定排序，兼顾并列分数和历史长度不足。`,
answer: "np.argsort(-score, kind=\"stable\")[:min(topk, t+1)]",
explain: md`降序稳定排序，兼顾并列分数和历史长度不足。`
},
{
blank: 3,
instruction: md`选完位置再用主注意力的 q/k 重新计算分数。`,
answer: "q[t] @ k[ids].T / np.sqrt(q.shape[1])",
explain: md`选完位置再用主注意力的 q/k 重新计算分数。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_sparse(qi, ki, gate, q, k, v, topk):
    out=[]
    for t in range(len(q)):
        scores=[sum(gate[t,h]*max(float(qi[t,h]@ki[s]),0.) for h in range(qi.shape[1])) for s in range(t+1)]
        ids=sorted(range(t+1),key=lambda j:(-scores[j],j))[:topk]
        s=np.array([q[t]@k[j]/np.sqrt(q.shape[1]) for j in ids]); p=np.exp(s-s.max());p/=p.sum()
        out.append(sum(p[a]*v[j] for a,j in enumerate(ids)))
    return np.asarray(out).reshape(len(q),v.shape[1])`,
tests: [
{
name: "全相同 indexer 分数按较小索引优先",
code: py`qi=np.zeros((3,1,1));ki=np.zeros((3,1));g=np.ones((3,1));q=ki.copy();v=np.arange(1,4.)[:,None]
_check(sparse_attention(qi,ki,g,q,q,v,1),[[1.],[1.],[1.]])`,
hidden: false
},
{
name: "top-k 大于因果历史",
code: py`qi=np.zeros((3,1,1));ki=np.zeros((3,1));g=np.ones((3,1));v=np.array([[2.],[4.],[6.]])
_check(sparse_attention(qi,ki,g,ki,ki,v,20),[[2.],[3.],[4.]])`,
hidden: false
},
{
name: "随机与负 gate",
code: py`r=np.random.default_rng(14)
for n in [1,4,9]:
 qi=r.normal(size=(n,2,3));ki=r.normal(size=(n,3));gate=r.normal(size=(n,2));q=r.normal(size=(n,4));k=r.normal(size=(n,4));v=r.normal(size=(n,2))
 for topk in [1,3,12]:
  _check(sparse_attention(qi,ki,gate,q,k,v,topk),_ref_sparse(qi,ki,gate,q,k,v,topk))`,
hidden: true
}
]
}
],
challenges: [
{
id: "indexer",
title: "Indexer 选择与稀疏注意力",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`sparse_attention(qi, ki, gate, q, k, v, topk)\`。qi:(T,H,Di)，ki:(T,Di)，gate:(T,H)，q/k:(T,D)，v:(T,Dv)，全部有限 float64，topk≥1；T 可为0。索引分数为各头 gate·ReLU(qi·ki) 的和，只允许 s≤t。同分按较小索引优先；选择 min(topk,t+1) 项后，使用独立 q/k 做标准注意力，返回 (T,Dv)。教学实现允许稳定全排序；应说明生产 top-k 的复杂度优势。`,
starter: py`import numpy as np

def sparse_attention(qi: np.ndarray, ki: np.ndarray, gate: np.ndarray, q: np.ndarray, k: np.ndarray, v: np.ndarray, topk: int) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def sparse_attention(qi: np.ndarray, ki: np.ndarray, gate: np.ndarray, q: np.ndarray, k: np.ndarray, v: np.ndarray, topk: int) -> np.ndarray:
    n = q.shape[0]
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for t in range(n):
        score = np.sum(gate[t, :, None] * np.maximum(qi[t] @ ki[:t+1].T, 0), axis=0)
        ids = np.argsort(-score, kind="stable")[:min(topk, t+1)]
        logits = q[t] @ k[ids].T / np.sqrt(q.shape[1])
        p = np.exp(logits - logits.max())
        out[t] = (p @ v[ids]) / p.sum()
    return out`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def sparse_attention(qi, ki, gate, q, k, v, topk):
    result = []
    for t in range(q.shape[0]):
        s = (gate[t, :, None] * torch.relu(qi[t] @ ki[:t+1].T)).sum(dim=0)
        ids = torch.argsort(s, descending=True, stable=True)[:topk]
        p = torch.softmax(q[t] @ k[ids].T / q.shape[1]**0.5, dim=0)
        result.append(p @ v[ids])
    return torch.stack(result) if result else v.new_zeros((0, v.shape[1]))`
}
],
explanation: md`索引器与主注意力是两套投影。先对每个头的点积 ReLU，再乘 gate 求和；交换求和和 ReLU 一般不等价。排序只在因果前缀中进行，避免未来 token 挤掉历史。主 softmax 只归一化选中项。教学稳定排序总时间包含 O(T² log T)，生产可用 top-k kernel；索引器本身依然扫描前缀，不能把整体复杂度宣称成严格 O(Tk)。`,
hints: [
md`逐头 ReLU，且 indexer 只能看到因果前缀。`,
md`降序稳定排序，兼顾并列分数和历史长度不足。`,
md`选完位置再用主注意力的 q/k 重新计算分数。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_sparse(qi, ki, gate, q, k, v, topk):
    out=[]
    for t in range(len(q)):
        scores=[sum(gate[t,h]*max(float(qi[t,h]@ki[s]),0.) for h in range(qi.shape[1])) for s in range(t+1)]
        ids=sorted(range(t+1),key=lambda j:(-scores[j],j))[:topk]
        s=np.array([q[t]@k[j]/np.sqrt(q.shape[1]) for j in ids]); p=np.exp(s-s.max());p/=p.sum()
        out.append(sum(p[a]*v[j] for a,j in enumerate(ids)))
    return np.asarray(out).reshape(len(q),v.shape[1])`,
tests: [
{
name: "全相同 indexer 分数按较小索引优先",
code: py`qi=np.zeros((3,1,1));ki=np.zeros((3,1));g=np.ones((3,1));q=ki.copy();v=np.arange(1,4.)[:,None]
_check(sparse_attention(qi,ki,g,q,q,v,1),[[1.],[1.],[1.]])`,
hidden: false
},
{
name: "top-k 大于因果历史",
code: py`qi=np.zeros((3,1,1));ki=np.zeros((3,1));g=np.ones((3,1));v=np.array([[2.],[4.],[6.]])
_check(sparse_attention(qi,ki,g,ki,ki,v,20),[[2.],[3.],[4.]])`,
hidden: false
},
{
name: "空序列",
code: py`_check(sparse_attention(np.empty((0,2,3)),np.empty((0,3)),np.empty((0,2)),np.empty((0,4)),np.empty((0,4)),np.empty((0,5)),2),np.empty((0,5)))`,
hidden: true
},
{
name: "随机与负 gate",
code: py`r=np.random.default_rng(14)
for n in [1,4,9]:
 qi=r.normal(size=(n,2,3));ki=r.normal(size=(n,3));gate=r.normal(size=(n,2));q=r.normal(size=(n,4));k=r.normal(size=(n,4));v=r.normal(size=(n,2))
 for topk in [1,3,12]:
  _check(sparse_attention(qi,ki,gate,q,k,v,topk),_ref_sparse(qi,ki,gate,q,k,v,topk))`,
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
id: "training",
q: md`为什么不直接用主注意力分数做精确 top-k？`,
a: md`精确计算所有主分数就已经支付了昂贵的全量点积成本；即使之后省下 value 聚合，收益也有限。低成本 indexer 用代理分数寻找候选，把算力留给少量主注意力。代价是选错位置的风险，所以需要训练或蒸馏，让代理评分接近实际相关性，并同时评估质量与 kernel 开销。`,
freq: 3
},
{
id: "cache",
q: md`DSA 的缓存能只保留当前 top-k 吗？`,
a: md`一般不能。top-k 是 query 相关的动态集合，今天未选中的 token 明天可能成为关键证据。除非模型另有压缩或淘汰机制，历史 KV 仍要保存；省的是每步读入与参与计算的项。应把持久 KV 显存和工作集显存分开，避免用 k 替换 T 导致容量估算失真。`,
freq: 3
},
{
id: "hardware",
q: md`为什么理论稀疏率高，实测却不快？`,
a: md`top-k 生成、分散 gather、负载不均和 kernel 启动都有成本。小序列时稠密 FlashAttention 利用率可能更高；长序列时 indexer 又可能成为主要瓶颈。块对齐、索引排序和融合能改善访存，但不保证线性加速。测速必须固定精度、batch、上下文和硬件，并检查所用稀疏 kernel 的实际覆盖。`,
freq: 2
}
],
cheatsheet: md`- NSA：压缩 + 选择 + 窗口，按门控混合。
- DSA：indexer → top-k → 主 MLA；两套分数别混用。
- 因果 mask 在 top-k 前；softmax 在选中集上。
- 主分支 O(Tk)，扫描式 indexer 仍有 O(T²)。
- 读取 k 个 ≠ 持久缓存只存 k 个。`
})
