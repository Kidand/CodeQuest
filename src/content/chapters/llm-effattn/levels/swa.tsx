import { defineLevel, md, py } from '@/content/helpers'
import { WindowViz, SinkViz } from '../viz/EfficientViz'

export default defineLevel({
learn: [
{
id: "window",
title: "窗口限制单层连接，深度扩展感受野",
body: md`标准注意力训练读 T² 个位置对；窗口宽度 W 将可见集合改成 $0\le i-j<W$，主注意力成为 O(TW)。W 包含当前 token；W=1 是只看自己，不是完全不注意任何位置。经过 L 个纯局部层，理论感受野上界是 $1+L(W-1)$。信息仍需穿过多次混合，不能把「有路径」当成可靠逐字检索。插入全局层可直接连接任意历史，但该层仍需长期 KV。Gemma 2 等混合局部/全局架构体现了这种取舍；选择窗口大小应结合检索任务、prefill 长度和内存预算，而不是只比较算力。`,
viz: WindowViz,
checkpoint: {
type: "numeric",
id: "reach",
prompt: md`因果窗口 W=4（包含自己），连续 L=3 层，没有全局层。在远离序列起点处，理论可达多少个输入位置？`,
explanation: md`最远每层向左走 W−1 步，因此 1+L(W−1)=10。`,
difficulty: 2,
answer: 10,
tolerance: 0.001
}
},
{
id: "sink",
title: "为什么流式推理要保留前缀锚点",
body: md`softmax 强迫所有候选分走总概率 1。有的注意力头把无须输出的信息投向前几个 token；直接驱逐它们会改变分布。StreamingLLM 保留少量初始 token 和最近窗口，以保持流式解码稳定；它没有恢复被删掉的历史细节。另一种设计显式加入可学习 logit a、零 value 槽：$o=\sum_j e^{s_j}v_j/(e^a+\sum_j e^{s_j})$。这里真实概率和可以小于 1。稳定实现须以 max(a,max s) 一起平移，不能在真实 softmax 后再随便附加 sink。参见 [StreamingLLM](https://arxiv.org/abs/2309.17453)。`,
viz: SinkViz,
checkpoint: {
type: "numeric",
id: "prob",
prompt: md`两个真实 token 和一个零值 sink 的 logit 都为 0，真实 value 均为 6，输出标量是多少？`,
explanation: md`三项各分到 1/3，输出是 6/3+6/3+0/3=4。排除 sink 后重新归一化会错误得到 6。`,
difficulty: 3,
answer: 4,
tolerance: 0.001
}
},
{
id: "gate",
title: "缓存位置、门控与实际模型",
body: md`环形缓存以 t mod W 写入，物理槽和逻辑 token 位置必须分开：覆盖一个槽不会把新 token 的位置重置成旧位置。保留前缀 sink 时，通常为它另留固定区，再维护近期环形区。输出门 $o'=\sigma(g)\odot o$ 在注意力后增加输入相关的非线性与稀疏性，并不等于把 QK logits 乘门，也不会自动缩小 KV。Qwen 团队的 [Gated Attention](https://arxiv.org/abs/2505.06708) 研究了这类输出门。面试中应区分局部 mask、概率 sink、输出 gate 三个作用位置；全局层、前缀缓存和滚动位置策略还需按具体模型配置验证。`
}
],
quiz: [
{
type: "single",
id: "goal",
prompt: md`SWA 最直接减少哪一项？`,
explanation: md`窗口把每行从 T 个候选限制到 W 个；参数量与词表不随 mask 改变，多层感受野会扩展。`,
difficulty: 1,
options: [
md`单层每个 query 读取的历史 KV 数`,
md`所有参数的数量`,
md`词表大小`,
md`所有层的感受野必为固定 W`
],
answer: 0
},
{
type: "numeric",
id: "reach",
prompt: md`因果窗口 W=4（包含自己），连续 L=3 层，没有全局层。在远离序列起点处，理论可达多少个输入位置？`,
explanation: md`最远每层向左走 W−1 步，因此 1+L(W−1)=10。`,
difficulty: 2,
answer: 10,
tolerance: 0.001
},
{
type: "judge",
id: "history",
prompt: md`保留前四个 attention sink token，就能保证完整回忆已经从窗口驱逐的所有历史。`,
explanation: md`sink 帮助维持注意力分布稳定，却不存储任意旧 token 的独立 KV；无限流式输入不等于无限检索。`,
difficulty: 1,
answer: false
},
{
type: "multi",
id: "sink",
prompt: md`关于 sink 与 gate，哪些正确？`,
explanation: md`sink 提供吸收概率的槽；gate 在输出后作用。二者都不自动决定缓存淘汰，真实前缀 token 与零值虚拟槽也不同。`,
difficulty: 2,
options: [
md`零值 sink 仍参与 softmax 分母`,
md`输出 sigmoid gate 可缩放已有注意力输出`,
md`gate 必然删除 KV cache`,
md`首 token sink 与可学习虚拟 sink 完全相同`
],
answer: [
0,
1
]
},
{
type: "match",
id: "concept",
prompt: md`匹配机制与直接作用`,
explanation: md`三者分别改变连接结构、恢复全局连接和调制输出幅度，可以组合。`,
difficulty: 2,
pairs: [
[
md`SWA`,
md`限制可见历史位置`
],
[
md`全局层`,
md`直接连接远距离 token`
],
[
md`输出门`,
md`控制注意力输出分量`
]
]
},
{
type: "order",
id: "cache",
prompt: md`环形 KV 写入后按逻辑时间读取，应遵循怎样的步骤？`,
explanation: md`物理槽会复用，但 RoPE 的位置语义及缓存有效性需要独立元数据。`,
difficulty: 2,
items: [
md`记录新 token 的绝对位置 t`,
md`写入物理槽 t mod W`,
md`计算仍有效的逻辑位置区间`,
md`按逻辑位置取槽并应用因果规则`
]
},
{
type: "bug",
id: "offby",
prompt: md`以下代码希望窗口含自己共 W 个位置。点出边界错误。`,
explanation: md`应为 rows-cols < w。≤w 会保留自己及前 w 个位置，共 w+1 个。`,
difficulty: 3,
code: py`import numpy as np
def mask(n, w):
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    causal = cols <= rows
    near = rows - cols <= w
    keep = causal & near
    return keep`,
bugLines: [
6
]
},
{
type: "numeric",
id: "prob",
prompt: md`两个真实 token 和一个零值 sink 的 logit 都为 0，真实 value 均为 6，输出标量是多少？`,
explanation: md`三项各分到 1/3，输出是 6/3+6/3+0/3=4。排除 sink 后重新归一化会错误得到 6。`,
difficulty: 3,
answer: 4,
tolerance: 0.001
}
],
guided: [
{
id: "g-window-sink",
title: "逐步实现：带可学习 sink 的滑窗注意力",
intro: md`实现 \`window_attention(q, k, v, window, sink) -> np.ndarray\`。q、k 为有限 float64 数组 (T,D)，v 为 (T,Dv)，允许 T=0，D、Dv≥1；window≥1。位置 i 仅看 max(0,i-window+1)…i。加入一个 logit=sink、value=0 的虚拟槽，同真实位置一起归一化。返回 (T,Dv)，不能构造 T×T 矩阵。例：q=k=0，v=[[2],[4]]，window=2、sink=0，输出 [[1],[2]]。时间 O(T·window·(D+Dv))，额外空间 O(window+Dv)，不计输出。`,
template: py`import numpy as np

def window_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, window: int, sink: float) -> np.ndarray:
    n, d = q.shape
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for i in range(n):
        start = {{1}}
        scores = q[i] @ k[start:i+1].T / np.sqrt(d)
        maximum = {{2}}
        weights = np.exp(scores - maximum)
        denom = {{3}}
        out[i] = weights @ v[start:i+1] / denom
    return out`,
steps: [
{
blank: 1,
instruction: md`窗口宽度包含当前位置，因此左端点减 window 后要加一。`,
answer: "max(0, i - window + 1)",
explain: md`窗口宽度包含当前位置，因此左端点减 window 后要加一。`
},
{
blank: 2,
instruction: md`数值稳定平移必须同时覆盖真实分数和 sink。`,
answer: "max(float(np.max(scores)), float(sink))",
explain: md`数值稳定平移必须同时覆盖真实分数和 sink。`
},
{
blank: 3,
instruction: md`把 sink 的指数项也计入分母。`,
answer: "weights.sum() + np.exp(sink - maximum)",
explain: md`把 sink 的指数项也计入分母。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "均匀两位置与零值 sink",
code: py`q=np.zeros((2,1)); k=q.copy(); v=np.array([[2.],[4.]])
_check(window_attention(q,k,v,2,0.), [[1.],[2.]])`,
hidden: false
},
{
name: "窗口为一",
code: py`q=np.zeros((3,1)); v=np.array([[2.],[4.],[6.]])
_check(window_attention(q,q,v,1,0.),v/2)`,
hidden: false
},
{
name: "独立稠密掩码随机对拍",
code: py`r=np.random.default_rng(21)
for n in [1,2,7]:
 for w in [1,3,10]:
  q=r.normal(size=(n,3));k=r.normal(size=(n,3));v=r.normal(size=(n,2));sink=1.2
  s=q@k.T/np.sqrt(3);i=np.arange(n)
  s=np.where((i[:,None]>=i)&(i[:,None]-i<w),s,-np.inf)
  s=np.column_stack([s,np.full(n,sink)]);p=np.exp(s-s.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
  _check(window_attention(q,k,v,w,sink),p[:,:n]@v)`,
hidden: true
}
]
}
],
challenges: [
{
id: "ring-cache",
title: "环形 KV 缓冲区的逻辑顺序",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`ring_cache(tokens,window,start) -> (positions, values)\`。tokens:(T,D)表示连续到达的KV特征，有限float64；T≥0、D≥1、window≥1、start≥0。初始缓存为空，第i个token绝对位置start+i，必须写物理槽position mod window。处理完返回最后min(T,window)个逻辑位置int64数组，以及按时间升序排列的KV矩阵。例：连续位置7、8、9，window=2，返回位置[8,9]，不是物理槽[0,1]。实现需使用固定window容量缓冲，不是把全部tokens存成增长列表。`,
starter: py`import numpy as np

def ring_cache(tokens: np.ndarray, window: int, start: int) -> tuple[np.ndarray, np.ndarray]:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def ring_cache(tokens: np.ndarray, window: int, start: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = tokens.shape
    buffer = np.empty((window, d), dtype=tokens.dtype)
    for i in range(n):
        position = start + i
        buffer[position % window] = tokens[i]
    positions = np.arange(start + max(0, n-window), start+n, dtype=np.int64)
    return positions, buffer[positions % window].copy()`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def ring_cache(tokens,window,start):
    n,d=tokens.shape
    buffer=tokens.new_empty((window,d))
    for i in range(n):buffer[(start+i)%window]=tokens[i]
    positions=torch.arange(start+max(0,n-window),start+n,device=tokens.device)
    return positions,buffer[positions%window].clone()`
}
],
explanation: md`缓存写入按物理槽复用，读取按仍然有效的绝对位置排序。未写入的槽不能返回；T=0返回(0,)和(0,D)。总时间O(TD+min(T,W)D)，额外空间O(WD)。实现可用同一缓冲保存拼接KV，但真实多层多头系统还需记录请求与位置元数据。`,
hints: [
md`取模只决定写入物理槽。`,
md`最老的有效逻辑位置由总长度与窗口共同决定。`,
md`按逻辑顺序收集物理槽，并返回独立结果。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "绕回后按时间读取",
code: py`positions,values=ring_cache(np.arange(10.).reshape(5,2),3,0)
_check(positions,[2,3,4]);_check(values,[[4.,5.],[6.,7.],[8.,9.]])`,
hidden: false
},
{
name: "非零位置不能当物理槽",
code: py`p,v=ring_cache(np.array([[1.],[2.],[3.]]),2,7)
_check(p,[8,9]);_check(v,[[2.],[3.]])`,
hidden: false
},
{
name: "空输入与窗口大于输入",
code: py`x=np.empty((0,3));p,v=ring_cache(x,4,10);_check(p,[]);_check(v,x)
x=np.arange(6.).reshape(2,3);p,v=ring_cache(x,9,100);_check(p,[100,101]);_check(v,x)`,
hidden: true
},
{
name: "固定种子对拍与输入不被改写",
code: py`r=np.random.default_rng(40)
for n in [1,2,13]:
 for w in [1,4,20]:
  x=r.normal(size=(n,3));saved=x.copy();p,v=ring_cache(x,w,23)
  _check(p,np.arange(23+max(0,n-w),23+n));_check(v,x[-w:]);_check(x,saved)`,
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
id: "window-sink",
title: "带可学习 sink 的滑窗注意力",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`window_attention(q, k, v, window, sink) -> np.ndarray\`。q、k 为有限 float64 数组 (T,D)，v 为 (T,Dv)，允许 T=0，D、Dv≥1；window≥1。位置 i 仅看 max(0,i-window+1)…i。加入一个 logit=sink、value=0 的虚拟槽，同真实位置一起归一化。返回 (T,Dv)，不能构造 T×T 矩阵。例：q=k=0，v=[[2],[4]]，window=2、sink=0，输出 [[1],[2]]。时间 O(T·window·(D+Dv))，额外空间 O(window+Dv)，不计输出。`,
starter: py`import numpy as np

def window_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, window: int, sink: float) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def window_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, window: int, sink: float) -> np.ndarray:
    n, d = q.shape
    out = np.zeros((n, v.shape[1]), dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        scores = q[i] @ k[start:i+1].T / np.sqrt(d)
        maximum = max(float(np.max(scores)), float(sink))
        weights = np.exp(scores - maximum)
        denom = weights.sum() + np.exp(sink - maximum)
        out[i] = weights @ v[start:i+1] / denom
    return out`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def window_attention(q, k, v, window, sink):
    n, d = q.shape
    if n == 0:
        return v.new_zeros((0, v.shape[1]))
    i = torch.arange(n, device=q.device)
    keep = (i[:, None] >= i[None, :]) & (i[:, None]-i[None, :] < window)
    s = (q @ k.T / d**0.5).masked_fill(~keep, -torch.inf)
    s = torch.cat([s, s.new_full((n, 1), sink)], dim=1)
    return torch.softmax(s, dim=1)[:, :n] @ v`
}
],
explanation: md`逐 query 切出因果窗口，sink 只进入稳定 softmax 的最大值和分母。真实权重之和可以小于 1，因为剩余概率分给零值槽。窗口上界包含当前 token；T=0 时直接返回预分配空输出。与先 softmax 再乘掩码不同，这里在正确候选集合上重新归一化。`,
hints: [
md`窗口宽度包含当前位置，因此左端点减 window 后要加一。`,
md`数值稳定平移必须同时覆盖真实分数和 sink。`,
md`把 sink 的指数项也计入分母。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "均匀两位置与零值 sink",
code: py`q=np.zeros((2,1)); k=q.copy(); v=np.array([[2.],[4.]])
_check(window_attention(q,k,v,2,0.), [[1.],[2.]])`,
hidden: false
},
{
name: "窗口为一",
code: py`q=np.zeros((3,1)); v=np.array([[2.],[4.],[6.]])
_check(window_attention(q,q,v,1,0.),v/2)`,
hidden: false
},
{
name: "空序列",
code: py`_check(window_attention(np.empty((0,2)),np.empty((0,2)),np.empty((0,3)),2,0.),np.empty((0,3)))`,
hidden: true
},
{
name: "大 logit 与非方形 value",
code: py`q=np.array([[1000.],[1000.] ]); k=q.copy(); v=np.array([[1.,2.],[3.,4.]])
_check(window_attention(q,k,v,9,0.),[[1.,2.],[2.,3.]])`,
hidden: true
},
{
name: "独立稠密掩码随机对拍",
code: py`r=np.random.default_rng(21)
for n in [1,2,7]:
 for w in [1,3,10]:
  q=r.normal(size=(n,3));k=r.normal(size=(n,3));v=r.normal(size=(n,2));sink=1.2
  s=q@k.T/np.sqrt(3);i=np.arange(n)
  s=np.where((i[:,None]>=i)&(i[:,None]-i<w),s,-np.inf)
  s=np.column_stack([s,np.full(n,sink)]);p=np.exp(s-s.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
  _check(window_attention(q,k,v,w,sink),p[:,:n]@v)`,
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
id: "trade",
q: md`为什么 SWA 的可达长度够长，仍可能检索不好？`,
a: md`理论感受野只说明计算图中有路径。远处的精确信息要经过多层有限维隐藏状态，多次混合会产生干扰；窗口也可能永久丢弃其原始 KV。全局层让 query 直接访问历史，提高检索能力，但牺牲该层的缓存与计算。应使用跨段检索、复制和真实长文任务评价，不能仅用可达长度证明模型有效。`,
freq: 3
},
{
id: "position",
q: md`环形缓存能不能直接用槽位当 RoPE position？`,
a: md`通常不能。槽位会每 W 步循环，而位置编码描述逻辑序列关系。直接复用槽号会把相距 W 的 token 变成同一位置。应记录位置元数据，明确采用原始绝对位置还是经模型验证的流式重定位策略。已有 RoPE 的 key 也不能随意挪位置而不重新旋转。`,
freq: 3
},
{
id: "difference",
q: md`真实前缀 sink、可学习 sink 和输出门有什么区别？`,
a: md`真实前缀 sink 是输入 token，有自己的 key/value；可学习虚拟 sink 通常只提供一项分母及零 value；输出门在聚合之后逐头或逐通道调节结果。三者可影响注意力的无效信息流，却有不同训练方式和缓存成本。不能因为名字都与 sink 有关，就把它们的公式互换。`,
freq: 2
}
],
cheatsheet: md`- SWA：$0\le i-j<W$；L 层可达 $1+L(W-1)$。
- 环形物理槽：t mod W；逻辑 position 单独记录。
- 零值 sink：分母加 $e^a$，分子不加。
- gate 在输出后调制；稀疏 mask 在 softmax 前。
- 流式稳定 ≠ 任意远距离精确检索。`
})
