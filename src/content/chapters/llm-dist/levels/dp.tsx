import { defineLevel, md, py } from '@/content/helpers'
import { ZeroViz } from '../viz/DistributedViz'

export default defineLevel({
learn: [
{
id: "reduction",
title: "同一步更新，先对齐梯度语义",
body: md`同步DP让每张卡用同一参数处理不同数据，再把梯度组合后更新。若各卡loss是相同样本数的均值，平均本地梯度对应全局均值；样本数或有效token数不同则需按权重归约。All-Reduce SUM本身只求和，除以world size可能由框架负责，不能重复除。梯度累积可以减少同步频率，但需按全局有效样本数保持loss尺度。不同rank必须以兼容的顺序调用collective，否则可能死锁，而不是仅仅算出一个错误数。`,
viz: ZeroViz,
checkpoint: {
type: "judge",
id: "average",
prompt: md`All-Reduce SUM天然等于全局平均梯度，无需任何额外归一化。`,
explanation: md`SUM仅求和；是否除P要结合框架与loss归一化。不同样本或token数还可能需要加权。`,
difficulty: 1,
answer: false
}
},
{
id: "ring",
title: "两阶段环通信如何覆盖所有设备",
body: md`把M字节向量分成P块。reduce-scatter让部分和沿环传递P−1轮，每卡最终持有一块完整归约结果；all-gather再传播P−1轮，让所有rank拿齐。大消息等长分块理想模型下，每卡发送 $2(P-1)M/P$ 字节，接收相同。延迟还包含2(P−1)次阶段启动，实际库会按拓扑与消息大小选择树或其他算法。教学模拟必须使用每轮开始的快照，否则Python逐rank原地更新会让部分和在同一轮走过多跳，破坏真实通信语义。`,
checkpoint: {
type: "numeric",
id: "traffic",
prompt: md`P=4，梯度向量大小M=120MB，理想Ring All-Reduce每卡发送多少MB？仅计发送，不把接收再加一次。`,
explanation: md`2(P−1)M/P=2×3/4×120=180MB。若计双向总流量需明确另一套口径。`,
difficulty: 2,
answer: 180,
tolerance: 0.001
}
},
{
id: "zero",
title: "只切冗余状态，算子执行时再取回",
body: md`[ZeRO](https://arxiv.org/abs/1910.02054) 按阶段切分优化器、梯度、参数。以权重2P、梯度2P、优化器含主权重12P为例，N卡单卡持久量分别为：普通DP16P；ZeRO-1为4P+12P/N；ZeRO-2为2P+14P/N；ZeRO-3为16P/N。参数分片不意味着算子只需碎片就能执行，通常要按层all-gather，并在反向reduce-scatter梯度。FSDP也采用全分片思想，但API、重分片、预取与混合精度策略需按框架配置理解。临时聚合和通信重叠可能抬高峰值，应另计。`
}
],
quiz: [
{
type: "single",
id: "dp",
prompt: md`普通同步DP每张卡通常持有什么？`,
explanation: md`DP复制模型，各卡对不同样本计算梯度，再归约形成同一步更新。`,
difficulty: 1,
options: [
md`完整模型与不同数据分片`,
md`不同层且相同数据`,
md`只有一份随机参数子集`,
md`独立训练完全不同模型`
],
answer: 0
},
{
type: "numeric",
id: "traffic",
prompt: md`P=4，梯度向量大小M=120MB，理想Ring All-Reduce每卡发送多少MB？仅计发送，不把接收再加一次。`,
explanation: md`2(P−1)M/P=2×3/4×120=180MB。若计双向总流量需明确另一套口径。`,
difficulty: 2,
answer: 180,
tolerance: 0.001
},
{
type: "judge",
id: "average",
prompt: md`All-Reduce SUM天然等于全局平均梯度，无需任何额外归一化。`,
explanation: md`SUM仅求和；是否除P要结合框架与loss归一化。不同样本或token数还可能需要加权。`,
difficulty: 1,
answer: false
},
{
type: "multi",
id: "zero",
prompt: md`按常见ZeRO定义，哪些正确？`,
explanation: md`ZeRO-3执行某层前通常需按需聚合该层参数，峰值不只是持久分片大小。`,
difficulty: 2,
options: [
md`ZeRO-1分片优化器状态`,
md`ZeRO-2进一步分片梯度`,
md`ZeRO-3进一步分片参数`,
md`ZeRO-3完全不需要临时完整参数`
],
answer: [
0,
1,
2
]
},
{
type: "match",
id: "collective",
prompt: md`匹配collective语义`,
explanation: md`all-gather只是拼接，不能替代梯度求和；两阶段组合能实现all-reduce。`,
difficulty: 2,
pairs: [
[
md`All-Reduce`,
md`每个rank得到归约后的完整向量`
],
[
md`Reduce-Scatter`,
md`每个rank得到归约结果的一片`
],
[
md`All-Gather`,
md`把各rank片段拼成完整向量`
]
]
},
{
type: "order",
id: "ring",
prompt: md`Ring All-Reduce的逻辑过程`,
explanation: md`归约与聚合阶段作用不同；错误的块编号会造成重复加或缺rank。`,
difficulty: 2,
items: [
md`一致地分块`,
md`多轮传递并累加到负责rank`,
md`持有各自完整归约块`,
md`多轮广播归约块`
]
},
{
type: "bug",
id: "snapshot",
prompt: md`下面每轮的模拟在哪一行丢失同步快照？`,
explanation: md`old必须深拷贝本轮读取的数据。别名会让较后rank读取已经被更新的值，偏离同时通信语义。`,
difficulty: 3,
code: py`def one_round(buffers):
    p = len(buffers)
    old = buffers
    for r in range(p):
        source = (r-1)%p
        incoming = old[source]
        buffers[r] = old[r] + incoming
    return buffers`,
bugLines: [
3
]
},
{
type: "numeric",
id: "zero3",
prompt: md`按16B/参数假设，P=8卡的ZeRO-3，忽略临时聚合，每参数每卡持久状态多少字节？`,
explanation: md`16/8=2B；这只是分片后持久状态，不是总峰值。`,
difficulty: 3,
answer: 2,
tolerance: 0.001
}
],
guided: [
{
id: "g-ring-sum",
title: "逐步实现：用虚拟设备模拟 Ring All-Reduce",
intro: md`实现 \`ring_sum(values: np.ndarray) -> np.ndarray\`，values:(P,N)，P≥1、N≥0，有限float64；每行是一个设备的本地向量。先用np.array_split按同样边界切P块，执行P−1轮reduce-scatter，再P−1轮all-gather。每轮只能读取该轮开始的缓冲快照；不能直接对设备轴求和实现。输出(P,N)，每行是所有输入行的逐元素和，不改变输入。教学模拟可复制缓冲；此复制不代表实际设备的内存开销。`,
template: py`import numpy as np

def ring_sum(values: np.ndarray) -> np.ndarray:
    p, n = values.shape
    buffers = [[c.copy() for c in np.array_split(values[r].astype(np.float64), p)] for r in range(p)]
    for step in range(p - 1):
        old = [[c.copy() for c in row] for row in buffers]
        for r in range(p):
            source = (r - 1) % p
            chunk = {{1}}
            buffers[r][chunk] = {{2}}
    for step in range(p - 1):
        old = [[c.copy() for c in row] for row in buffers]
        for r in range(p):
            source = (r - 1) % p
            chunk = {{3}}
            buffers[r][chunk] = old[source][chunk].copy()
    return np.stack([np.concatenate(row) for row in buffers])`,
steps: [
{
blank: 1,
instruction: md`reduce-scatter中rank r接收上一rank正在传递的块编号。`,
answer: "(r - step - 1) % p",
explain: md`reduce-scatter中rank r接收上一rank正在传递的块编号。`
},
{
blank: 2,
instruction: md`归约阶段把接收的部分和与本地对应部分相加。`,
answer: "old[r][chunk] + old[source][chunk]",
explain: md`归约阶段把接收的部分和与本地对应部分相加。`
},
{
blank: 3,
instruction: md`all-gather从reduce-scatter结束的块所有权继续转发。`,
answer: "(r - step) % p",
explain: md`all-gather从reduce-scatter结束的块所有权继续转发。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "两卡求和",
code: py`_check(ring_sum(np.array([[1.,2.],[3.,4.]])),[[4.,6.],[4.,6.]])`,
hidden: false
},
{
name: "单卡不改变输入",
code: py`x=np.array([[3.,-2.,8.] ]);_check(ring_sum(x),x)`,
hidden: false
},
{
name: "同步轮次随机对拍",
code: py`r=np.random.default_rng(29)
for p in [2,3,4,7]:
 x=r.normal(size=(p,13));_check(ring_sum(x),np.tile(x.sum(axis=0),(p,1)))`,
hidden: true
}
]
}
],
challenges: [
{
id: "ring-sum",
title: "用虚拟设备模拟 Ring All-Reduce",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`ring_sum(values: np.ndarray) -> np.ndarray\`，values:(P,N)，P≥1、N≥0，有限float64；每行是一个设备的本地向量。先用np.array_split按同样边界切P块，执行P−1轮reduce-scatter，再P−1轮all-gather。每轮只能读取该轮开始的缓冲快照；不能直接对设备轴求和实现。输出(P,N)，每行是所有输入行的逐元素和，不改变输入。教学模拟可复制缓冲；此复制不代表实际设备的内存开销。`,
starter: py`import numpy as np

def ring_sum(values: np.ndarray) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def ring_sum(values: np.ndarray) -> np.ndarray:
    p, n = values.shape
    buffers = [[c.copy() for c in np.array_split(values[r].astype(np.float64), p)] for r in range(p)]
    for step in range(p - 1):
        old = [[c.copy() for c in row] for row in buffers]
        for r in range(p):
            source = (r - 1) % p
            chunk = (r - step - 1) % p
            buffers[r][chunk] = old[r][chunk] + old[source][chunk]
    for step in range(p - 1):
        old = [[c.copy() for c in row] for row in buffers]
        for r in range(p):
            source = (r - 1) % p
            chunk = (r - step) % p
            buffers[r][chunk] = old[source][chunk].copy()
    return np.stack([np.concatenate(row) for row in buffers])`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch
import torch.distributed as dist

def ring_sum_distributed(local):
    # torch.distributed 选择实际collective算法，此处展示同样的求和语义
    result=local.clone()
    dist.all_reduce(result,op=dist.ReduceOp.SUM)
    return result

def ring_sum_reference(values):
    return values.sum(dim=0,keepdim=True).expand_as(values).clone()`
}
],
explanation: md`每轮快照模拟同步收发，避免在同一Python循环里读取已被本轮覆盖的数据。reduce-scatter后rank r持有完整和的块(r+1) mod P；all-gather从此所有权继续传播。每块恰好合并每个rank一次。实际等长大消息环上，每卡单向发送量为2(P−1)N/P个元素；接收量相同。`,
hints: [
md`reduce-scatter中rank r接收上一rank正在传递的块编号。`,
md`归约阶段把接收的部分和与本地对应部分相加。`,
md`all-gather从reduce-scatter结束的块所有权继续转发。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"

import inspect`,
tests: [
{
name: "两卡求和",
code: py`_check(ring_sum(np.array([[1.,2.],[3.,4.]])),[[4.,6.],[4.,6.]])`,
hidden: false
},
{
name: "单卡不改变输入",
code: py`x=np.array([[3.,-2.,8.] ]);_check(ring_sum(x),x)`,
hidden: false
},
{
name: "不整除分块与空向量",
code: py`for p,n in [(4,0),(4,2),(3,7),(5,11)]:
 x=np.arange(p*n,dtype=float).reshape(p,n);saved=x.copy()
 _check(ring_sum(x),np.broadcast_to(x.sum(0),(p,n)))
 _check(x,saved)`,
hidden: true
},
{
name: "同步轮次随机对拍",
code: py`r=np.random.default_rng(29)
for p in [2,3,4,7]:
 x=r.normal(size=(p,13));_check(ring_sum(x),np.tile(x.sum(axis=0),(p,1)))`,
hidden: true
},
{
name: "不绕过环通信模拟",
code: py`src=inspect.getsource(ring_sum)
assert ".sum(" not in src and "np.sum(" not in src, "请实现两阶段逐轮通信，不能直接沿设备轴求和"`,
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
id: "sum",
q: md`不等长样本时怎样得到全局token平均梯度？`,
a: md`先明确每rank损失是token和还是token均值。若是局部均值，乘本rank有效token数恢复为局部和，对梯度和token计数分别归约，再除全局token总数。直接平均各rank均值会让短序列rank权重过大。梯度累积跨多个microbatch时也应使用同一统计口径。`,
freq: 3
},
{
id: "zero",
q: md`ZeRO-3能不能把显存近似当作16P/N？`,
a: md`只能把它当作特定dtype与优化器下的持久状态下界。执行层时的完整参数聚合、预取下一层、通信缓冲、激活和碎片都额外占空间。大层或过大的预取窗口可能成为峰值瓶颈。因此要根据all-gather粒度和时间线估算，再用实际峰值指标验证。`,
freq: 3
},
{
id: "latency",
q: md`Ring流量最优就意味着所有场景最快吗？`,
a: md`不意味着。大消息时带宽主导，环的均匀分块很有优势；小消息或卡数多时，轮次数带来的启动延迟可能主导。硬件拓扑、跨机网络、负载和通信计算重叠也会影响结果。应测真实消息分布，不能用一个流量公式给所有collective选型。`,
freq: 2
}
],
cheatsheet: md`环：reduce-scatter + all-gather；每卡发送 $2(N-1)M/N$。

16B假设下每卡状态：DP16P；Z1=4P+12P/N；Z2=2P+14P/N；Z3=16P/N。

SUM≠平均；不同token数要加权；临时聚合≠持久分片。`
})
