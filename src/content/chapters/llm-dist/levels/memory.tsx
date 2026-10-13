import { defineLevel, md, py } from '@/content/helpers'
import { MemoryViz } from '../viz/DistributedViz'

export default defineLevel({
learn: [
{
id: "ledger",
title: "先列对象，别只用参数乘精度",
body: md`一份P参数BF16权重仅2P字节，不能代表训练显存。常见混合精度Adam配置还保留2P低精度梯度、4P主权重和8P两个FP32矩，总计16P。这个数字是约定而非自然常数：FP32梯度、没有主权重、低位优化器都会改变它。大模型训练还需激活、通信bucket、临时all-gather、算子工作区和分配器碎片。先按对象列账，再沿时间线找同时存活的峰值；简单把所有临时量相加可能过高，只数静态参数又必然偏低。`,
viz: MemoryViz,
checkpoint: {
type: "numeric",
id: "7b",
prompt: md`按每参数16字节，7×10⁹参数的持久状态为多少GB？1GB=10⁹字节。`,
explanation: md`7×10⁹×16/10⁹=112GB，未含激活、通信缓冲和碎片。`,
difficulty: 2,
answer: 112,
tolerance: 0.001
}
},
{
id: "activation",
title: "重计算用 FLOPs 换存活时间",
body: md`一个激活张量的容量等于各轴乘积乘dtype字节数；但整层会保存多个中间量。普通注意力可能保存T²分数，FlashAttention通过重算改变保存结构，所以不能套同一激活公式。重计算保留层边界或选择的中间张量，反向时重新执行前向。选择性重计算优先丢弃占显存大、重算代价低的张量。代价是额外FLOPs和更复杂的随机状态处理；dropout等随机算子需复现前向随机结果。microbatch减小也会省激活，但可能降低GPU利用率。`,
checkpoint: {
type: "match",
id: "storage",
prompt: md`匹配存储对象`,
explanation: md`主权重和两个矩职责不同，不能用一份FP32状态概括全部。`,
difficulty: 2,
pairs: [
[
md`主权重`,
md`保留更高精度参数更新`
],
[
md`Adam一阶矩`,
md`梯度指数滑动平均`
],
[
md`Adam二阶矩`,
md`梯度平方指数滑动平均`
]
]
}
},
{
id: "precision",
title: "精度影响范围、误差和通信",
body: md`FP16尾数精度与BF16不同，FP16指数范围较小，容易在反向出现下溢或溢出。loss scaling将loss乘s，反向后再把梯度除s；若出现Inf/NaN应跳过更新并调整scale。裁剪必须作用于unscale后的真实梯度，否则阈值被误缩小。BF16拥有接近FP32的指数范围，但尾数较短，归约和优化器状态仍常使用更高精度。混合精度不仅是把参数cast成半精度，还包括哪些算子累加用FP32、通信传什么dtype、优化器如何维护主权重。参考 [ZeRO 论文中的状态拆解](https://arxiv.org/abs/1910.02054)。`
}
],
quiz: [
{
type: "single",
id: "sixteen",
prompt: md`16字节/参数的常见混合精度Adam账本包含什么？`,
explanation: md`这是明确的存储假设；不同梯度dtype、优化器和主权重策略会改变账本。`,
difficulty: 1,
options: [
md`2权重+2梯度+4主权重+8两个矩`,
md`只有FP16参数`,
md`权重与KVcache合计`,
md`任何训练框架都固定16字节`
],
answer: 0
},
{
type: "numeric",
id: "7b",
prompt: md`按每参数16字节，7×10⁹参数的持久状态为多少GB？1GB=10⁹字节。`,
explanation: md`7×10⁹×16/10⁹=112GB，未含激活、通信缓冲和碎片。`,
difficulty: 2,
answer: 112,
tolerance: 0.001
},
{
type: "judge",
id: "bf",
prompt: md`BF16指数范围较FP16更大，因此任何BF16训练都不需要监控溢出与数值问题。`,
explanation: md`范围更大通常降低loss scaling的必要性，但不保证softmax、归约或优化器绝对稳定。`,
difficulty: 1,
answer: false
},
{
type: "multi",
id: "activation",
prompt: md`哪些通常会影响训练峰值激活？`,
explanation: md`激活来自实际计算图和保存策略，参数总量不能单独决定峰值。`,
difficulty: 2,
options: [
md`序列长度`,
md`microbatch大小`,
md`保存与重计算策略`,
md`只有参数总量`
],
answer: [
0,
1,
2
]
},
{
type: "match",
id: "storage",
prompt: md`匹配存储对象`,
explanation: md`主权重和两个矩职责不同，不能用一份FP32状态概括全部。`,
difficulty: 2,
pairs: [
[
md`主权重`,
md`保留更高精度参数更新`
],
[
md`Adam一阶矩`,
md`梯度指数滑动平均`
],
[
md`Adam二阶矩`,
md`梯度平方指数滑动平均`
]
]
},
{
type: "order",
id: "scale",
prompt: md`FP16动态loss scaling时，正确处理顺序？`,
explanation: md`裁剪应在unscale后；若梯度非有限，应跳过本次更新并调整scale。`,
difficulty: 2,
items: [
md`放大loss并反向`,
md`把梯度除以scale`,
md`检查非有限值并决定是否跳过更新`,
md`对有限的真实梯度裁剪并更新`
]
},
{
type: "bug",
id: "clip",
prompt: md`哪一行错误地先裁剪了放大梯度？`,
explanation: md`第4行开始在放大尺度上计算范数；应先将grads除以scale，再检测非有限值并对有限的真实梯度计算范数和裁剪。否则有效裁剪阈值缩小了scale倍。`,
difficulty: 3,
code: py`import numpy as np
def update(grads, scale, clip):
    # grads 来自 scaled loss
    norm = sum((g*g).sum() for g in grads)**0.5
    factor = min(1.0, clip/max(norm,1e-12))
    grads = [g*factor for g in grads]
    grads = [g/scale for g in grads]
    finite = all(np.isfinite(g).all() for g in grads)
    return grads, finite`,
bugLines: [
4
]
},
{
type: "numeric",
id: "act",
prompt: md`一个(B=2,T=4096,D=1024)激活，BF16每元素2字节，共多少MiB？1MiB=2²⁰字节。`,
explanation: md`2×4096×1024×2/2²⁰=16MiB；仅是一份激活，不是整层总量。`,
difficulty: 2,
answer: 16,
tolerance: 0.001
}
],
guided: [
{
id: "g-ledger",
title: "逐步实现：建立可解释的训练显存账本",
intro: md`实现 \`memory_ledger(params, shapes, activation_bytes, keep, reserve) -> np.ndarray\`。模型假设：低精度权重2B、梯度2B、FP32主权重4B、Adam一二阶矩共8B，每参数16B。shapes是显式保留激活张量的形状列表（非负整数维度），activation_bytes为每元素字节数，keep∈[0,1]是抽象保留比例；reserve为通信/碎片等预留字节。返回float64数组[权重,梯度,主权重,两个矩,激活,预留]，长度6，单位字节。params和reserve≥0；空形状列表表示无激活。该估算不声明复现某个框架峰值。`,
template: py`import numpy as np

def memory_ledger(params: int, shapes: list[tuple[int, ...]], activation_bytes: int, keep: float, reserve: int) -> np.ndarray:
    components = {{1}}
    elements = {{2}}
    activations = {{3}}
    return np.concatenate([components, [activations, float(reserve)]])`,
steps: [
{
blank: 1,
instruction: md`先按四类持久对象计算，不能把Adam两个矩只计一次。`,
answer: "np.array([2, 2, 4, 8], dtype=np.float64) * params",
explain: md`先按四类持久对象计算，不能把Adam两个矩只计一次。`
},
{
blank: 2,
instruction: md`激活元素数是每个张量各轴乘积，再求总和。`,
answer: "sum(int(np.prod(shape, dtype=np.int64)) for shape in shapes)",
explain: md`激活元素数是每个张量各轴乘积，再求总和。`
},
{
blank: 3,
instruction: md`重计算仅作用于本题激活保留量，不缩减参数和优化器状态。`,
answer: "elements * activation_bytes * keep",
explain: md`重计算仅作用于本题激活保留量，不缩减参数和优化器状态。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "十参数与两份激活",
code: py`_check(memory_ledger(10,[(2,3),(4,)],2,1.,7),[20,20,40,80,20,7])`,
hidden: false
},
{
name: "保留一半激活",
code: py`_check(memory_ledger(1,[(2,4,8)],2,.5,0),[2,2,4,8,64,0])`,
hidden: false
},
{
name: "大参数避免32位溢出",
code: py`got=memory_ledger(7_000_000_000,[(1,4096,4096)],2,.25,2**30)
_check(got,[14e9,14e9,28e9,56e9,2**23,2**30])
assert got.sum()>112e9, f"期望总量大于112GB，得到{got.sum()}"`,
hidden: true
}
]
}
],
challenges: [
{
id: "ledger",
title: "建立可解释的训练显存账本",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`memory_ledger(params, shapes, activation_bytes, keep, reserve) -> np.ndarray\`。模型假设：低精度权重2B、梯度2B、FP32主权重4B、Adam一二阶矩共8B，每参数16B。shapes是显式保留激活张量的形状列表（非负整数维度），activation_bytes为每元素字节数，keep∈[0,1]是抽象保留比例；reserve为通信/碎片等预留字节。返回float64数组[权重,梯度,主权重,两个矩,激活,预留]，长度6，单位字节。params和reserve≥0；空形状列表表示无激活。该估算不声明复现某个框架峰值。`,
starter: py`import numpy as np

def memory_ledger(params: int, shapes: list[tuple[int, ...]], activation_bytes: int, keep: float, reserve: int) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def memory_ledger(params: int, shapes: list[tuple[int, ...]], activation_bytes: int, keep: float, reserve: int) -> np.ndarray:
    components = np.array([2, 2, 4, 8], dtype=np.float64) * params
    elements = sum(int(np.prod(shape, dtype=np.int64)) for shape in shapes)
    activations = elements * activation_bytes * keep
    return np.concatenate([components, [activations, float(reserve)]])`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import math
import torch

def memory_ledger(params,shapes,activation_bytes,keep,reserve):
    components=torch.tensor([2,2,4,8],dtype=torch.float64)*params
    activations=sum(math.prod(s) for s in shapes)*activation_bytes*keep
    return torch.cat([components,torch.tensor([activations,reserve],dtype=torch.float64)])`
}
],
explanation: md`持久状态按对象分别计算，再对显式激活形状求元素乘积。float64避免把巨大字节数装进int32；每个维度乘积先用int64。keep是题设抽象系数，实际重计算策略的峰值需要时间线验证。复杂度为所有形状维数之和，返回常数长度账本。`,
hints: [
md`先按四类持久对象计算，不能把Adam两个矩只计一次。`,
md`激活元素数是每个张量各轴乘积，再求总和。`,
md`重计算仅作用于本题激活保留量，不缩减参数和优化器状态。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "十参数与两份激活",
code: py`_check(memory_ledger(10,[(2,3),(4,)],2,1.,7),[20,20,40,80,20,7])`,
hidden: false
},
{
name: "保留一半激活",
code: py`_check(memory_ledger(1,[(2,4,8)],2,.5,0),[2,2,4,8,64,0])`,
hidden: false
},
{
name: "空激活与零维度",
code: py`_check(memory_ledger(0,[],2,1,0),np.zeros(6))
_check(memory_ledger(2,[(0,2,3)],4,.25,10),[4,4,8,16,0,10])`,
hidden: true
},
{
name: "大参数避免32位溢出",
code: py`got=memory_ledger(7_000_000_000,[(1,4096,4096)],2,.25,2**30)
_check(got,[14e9,14e9,28e9,56e9,2**23,2**30])
assert got.sum()>112e9, f"期望总量大于112GB，得到{got.sum()}"`,
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
id: "16",
q: md`为什么不同人对每参数字节数给出不同答案？`,
a: md`因为存储假设不同。16B对应低精度权重和梯度各2B、主权重4B以及两个FP32矩8B；若梯度为FP32，就变成18B，若框架直接维护FP32参数并按需cast，也需重新计算临时权重。应先问优化器、dtype和分片策略，再计算激活和工作区。只背16B却不说明对象，容易在峰值估算中漏项。`,
freq: 3
},
{
id: "recompute",
q: md`重计算为什么可能明显省显存但训练只慢一点？`,
a: md`很多被保存的大中间张量生成成本较低，而GPU高吞吐矩阵乘可在反向时重建它们；同时较低显存允许提高microbatch或减少通信压力。但收益依赖算子和硬件，不能承诺固定比例。实际选择应在时间线中定位峰值存活张量，比较每个张量的保存大小与重算耗时。`,
freq: 3
},
{
id: "scale",
q: md`梯度累积配loss scaling需要注意什么？`,
a: md`同一更新内的缩放和归一化应保持一致，完成累积后再统一unscale、检测非有限值和裁剪。还要确认loss按microbatch均值还是全局token数归一化，尤其是变长序列。跳过更新时不能只跳参数而仍更新优化器矩或步数；否则状态和权重会失配。`,
freq: 2
}
],
cheatsheet: md`常见假设：2P权重 + 2P梯度 + 4P主权重 + 8P矩 = 16P字节。

总峰值 = 同时存活状态 + 激活 + 临时通信/算子缓冲 + 碎片。

FP16：scale loss → backward → unscale → finite check → clip → step。

GB=10⁹，GiB=2³⁰；计算题先写单位。`
})
