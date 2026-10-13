import { defineLevel, md, py } from '@/content/helpers'
import { TensorViz } from '../viz/DistributedViz'

export default defineLevel({
learn: [
{
id: "mlp",
title: "列切到行切，非线性不用等待归约",
body: md`设MLP为 $Y=f(XW_1+b_1)W_2+b_2$，W₁形状D×H。沿H切列后，每张卡得到一组完整隐藏分量，可本地做逐元素非线性。W₂沿相同H边界切行，每片产生B×O的部分输出，最后SUM归约并加一次b₂。若反过来先切W₁输入维，非线性之前就得先汇总内积，否则f(a+b)不等于f(a)+f(b)。[Megatron-LM](https://arxiv.org/abs/1909.08053) 的核心价值是把数学分解与通信位置一起设计。框架权重常用[out,in]存储，面试时要先声明轴约定。`,
viz: TensorViz,
checkpoint: {
type: "judge",
id: "relu",
prompt: md`把输入内积拆成部分和后，可以先对各部分ReLU再求和而保持等价。`,
explanation: md`ReLU(a+b)通常不等于ReLU(a)+ReLU(b)。列并行输出每个分量已完成完整内积，才可本地做非线性。`,
difficulty: 1,
answer: false
}
},
{
id: "attention",
title: "头并行与词表并行都要守住全局语义",
body: md`注意力可沿独立头切分，输出投影再按输入行切并汇总；GQA的KV头较少时要处理复制或分组可整除约束，不能盲目按Q头数规划。词表并行CE无需收集全部logits：先跨卡MAX得到全局m，再跨卡SUM得到 $Z=\sum_v e^{z_v-m}$，定位目标token所在分片取其logit，loss=m+log Z−z_y。各片独立softmax再拼接是错的，因为分母不一致。反向仍需正确归约跨片依赖，不能只看前向输出形状。`,
checkpoint: {
type: "numeric",
id: "ce",
prompt: md`词表分成两片，局部exp和分别2和6（已减同一个全局max），正确全局softmax分母是多少？`,
explanation: md`SUM得到8。不能分别除以2与6后拼接，那会令全词表概率总和为2。`,
difficulty: 3,
answer: 8,
tolerance: 0.001
}
},
{
id: "sp",
title: "序列并行不是局部注意力",
body: md`TP某些边界会让激活在所有TP卡上重复存储。序列并行让LayerNorm、dropout等可逐token执行的部分按序列轴分片，适时使用all-gather或reduce-scatter转换布局，从而减少冗余激活。在本章语境中，这与把完整长序列注意力分到多卡的CP区分开。SP不会允许query忽略其它分片的key；全局注意力仍要通过适当通信保证同一数学结果。收益取决于microbatch、序列长度、通信带宽及融合内核，通常优先在高带宽设备组内使用TP。`
}
],
quiz: [
{
type: "single",
id: "column",
prompt: md`X:[B,D]，W:[D,H]按输出列切成P片，每片线性输出形状是什么？`,
explanation: md`输出列对应隐藏特征维；batch与输入维没有在此操作中切分。`,
difficulty: 1,
options: [
md`[B,H/P]`,
md`[B/P,H]`,
md`[D/P,H]`,
md`[B,D]`
],
answer: 0
},
{
type: "judge",
id: "relu",
prompt: md`把输入内积拆成部分和后，可以先对各部分ReLU再求和而保持等价。`,
explanation: md`ReLU(a+b)通常不等于ReLU(a)+ReLU(b)。列并行输出每个分量已完成完整内积，才可本地做非线性。`,
difficulty: 1,
answer: false
},
{
type: "numeric",
id: "bias",
prompt: md`P=4张卡，每卡部分输出0，输出偏置b=3。若每卡先加b再all-reduce SUM，会错误得到多少？`,
explanation: md`四份偏置求和得到4×3=12，正确应为3。`,
difficulty: 2,
answer: 12,
tolerance: 0.001
},
{
type: "multi",
id: "sp",
prompt: md`Megatron风格TP/SP中，哪些正确？`,
explanation: md`SP并不自动把全局注意力变成局部注意力；需要在合适边界collective恢复算子所需布局。`,
difficulty: 2,
options: [
md`TP线性层可能需要all-reduce或等价组合`,
md`SP可让部分逐token操作的激活按序列分片`,
md`SP等于所有注意力都只看本地token`,
md`Norm/dropout等操作可利用序列分片减少复制激活`
],
answer: [
0,
1,
3
]
},
{
type: "match",
id: "axes",
prompt: md`匹配并行维度`,
explanation: md`先确定数学轴，再对齐具体框架中权重可能转置存储的约定。`,
difficulty: 2,
pairs: [
[
md`列并行线性层`,
md`切输出特征`
],
[
md`行并行线性层`,
md`切输入特征并归约部分和`
],
[
md`词表并行`,
md`分片词表logits与相关归约`
]
]
},
{
type: "order",
id: "mlp",
prompt: md`本关MLP的计算顺序`,
explanation: md`第一层和第二层的切片边界必须一致；中间隐藏无需完整all-gather。`,
difficulty: 2,
items: [
md`复制输入给各TP分片`,
md`各片算第一层完整列及ReLU`,
md`用第二层对应行形成部分输出`,
md`归约部分输出并加一次b2`
]
},
{
type: "bug",
id: "bias-bug",
prompt: md`定位导致输出偏置重复的代码。`,
explanation: md`应将bias移到sum(outputs)之后添加；当前每个分片都加一次，会多出(P−1)bias。`,
difficulty: 3,
code: py`def combine(parts, bias):
    outputs = []
    for hidden, weight in parts:
        partial = hidden @ weight
        partial = partial + bias
        outputs.append(partial)
    merged = sum(outputs)
    return merged`,
bugLines: [
5
]
},
{
type: "numeric",
id: "ce",
prompt: md`词表分成两片，局部exp和分别2和6（已减同一个全局max），正确全局softmax分母是多少？`,
explanation: md`SUM得到8。不能分别除以2与6后拼接，那会令全词表概率总和为2。`,
difficulty: 3,
answer: 8,
tolerance: 0.001
}
],
guided: [
{
id: "g-tp-mlp",
title: "逐步实现：列并行接行并行 MLP",
intro: md`实现 \`tp_mlp(x,w1,b1,w2,b2,parts)\`，x:(B,D)，w1:(D,H)，b1:(H,)，w2:(H,O)，b2:(O,)，有限float64。H能被parts整除，parts≥1、B≥0、D/H/O≥1。模拟parts张设备：第一层沿输出列切，第二层沿输入行按相同边界切；激活ReLU。返回(B,O)，等价于ReLU(x@w1+b1)@w2+b2。不能先算完整隐藏激活；各设备部分输出求和，b2仅加一次。`,
template: py`import numpy as np

def tp_mlp(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, parts: int) -> np.ndarray:
    width = w1.shape[1] // parts
    out = np.zeros((x.shape[0], w2.shape[1]), dtype=np.float64)
    for rank in range(parts):
        sl = slice(rank * width, (rank + 1) * width)
        hidden = np.maximum(x @ {{1}} + b1[sl], 0)
        out += hidden @ {{2}}
    return {{3}}`,
steps: [
{
blank: 1,
instruction: md`第一层按输出列切，保留完整输入维。`,
answer: "w1[:, sl]",
explain: md`第一层按输出列切，保留完整输入维。`
},
{
blank: 2,
instruction: md`第二层按对应隐藏输入行切，形成输出部分和。`,
answer: "w2[sl, :]",
explain: md`第二层按对应隐藏输入行切，形成输出部分和。`
},
{
blank: 3,
instruction: md`输出偏置在部分和合并之后只加入一次。`,
answer: "out + b2",
explain: md`输出偏置在部分和合并之后只加入一次。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "两卡、输出偏置只加一次",
code: py`x=np.array([[1.,2.]]);w1=np.eye(2);w2=np.eye(2);b1=np.zeros(2);b2=np.array([3.,4.])
_check(tp_mlp(x,w1,b1,w2,b2,2),[[4.,6.]])`,
hidden: false
},
{
name: "非线性发生在完整列输出后",
code: py`x=np.array([[-1.,2.]]);w1=np.array([[1.,-1.],[1.,1.]]);w2=np.ones((2,1))
_check(tp_mlp(x,w1,np.zeros(2),w2,np.zeros(1),2),[[4.]])`,
hidden: false
},
{
name: "全负激活只剩偏置",
code: py`x=-np.ones((2,3));w1=np.ones((3,4));w2=np.ones((4,2));b2=np.array([5.,-3.])
_check(tp_mlp(x,w1,np.zeros(4),w2,b2,2),np.tile(b2,(2,1)))`,
hidden: true
}
]
}
],
challenges: [
{
id: "tp-mlp",
title: "列并行接行并行 MLP",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`tp_mlp(x,w1,b1,w2,b2,parts)\`，x:(B,D)，w1:(D,H)，b1:(H,)，w2:(H,O)，b2:(O,)，有限float64。H能被parts整除，parts≥1、B≥0、D/H/O≥1。模拟parts张设备：第一层沿输出列切，第二层沿输入行按相同边界切；激活ReLU。返回(B,O)，等价于ReLU(x@w1+b1)@w2+b2。不能先算完整隐藏激活；各设备部分输出求和，b2仅加一次。`,
starter: py`import numpy as np

def tp_mlp(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, parts: int) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def tp_mlp(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray, parts: int) -> np.ndarray:
    width = w1.shape[1] // parts
    out = np.zeros((x.shape[0], w2.shape[1]), dtype=np.float64)
    for rank in range(parts):
        sl = slice(rank * width, (rank + 1) * width)
        hidden = np.maximum(x @ w1[:, sl] + b1[sl], 0)
        out += hidden @ w2[sl, :]
    return out + b2`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def tp_mlp(x,w1,b1,w2,b2,parts):
    width=w1.shape[1]//parts
    partial=[]
    for rank in range(parts):
        sl=slice(rank*width,(rank+1)*width)
        hidden=torch.relu(x@w1[:,sl]+b1[sl])
        partial.append(hidden@w2[sl,:])
    # 分布式版本在这里对各rank partial执行SUM all-reduce
    return torch.stack(partial).sum(dim=0)+b2`
}
],
explanation: md`列切后每个隐藏分量已获得完整输入内积，因此可以本地做ReLU。对应行切第二层得到完整输出的部分和，只需一次求和归约；输出偏置在归约后加入，避免乘上parts。总教学算术量O(BDH+BHO)，单片隐藏工作空间O(BH/parts)，外加输出。`,
hints: [
md`第一层按输出列切，保留完整输入维。`,
md`第二层按对应隐藏输入行切，形成输出部分和。`,
md`输出偏置在部分和合并之后只加入一次。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "两卡、输出偏置只加一次",
code: py`x=np.array([[1.,2.]]);w1=np.eye(2);w2=np.eye(2);b1=np.zeros(2);b2=np.array([3.,4.])
_check(tp_mlp(x,w1,b1,w2,b2,2),[[4.,6.]])`,
hidden: false
},
{
name: "非线性发生在完整列输出后",
code: py`x=np.array([[-1.,2.]]);w1=np.array([[1.,-1.],[1.,1.]]);w2=np.ones((2,1))
_check(tp_mlp(x,w1,np.zeros(2),w2,np.zeros(1),2),[[4.]])`,
hidden: false
},
{
name: "随机与单卡公式一致",
code: py`r=np.random.default_rng(28)
for batch in [0,1,5]:
 x=r.normal(size=(batch,3));w1=r.normal(size=(3,12));b1=r.normal(size=12);w2=r.normal(size=(12,4));b2=r.normal(size=4)
 for parts in [1,2,3,4,6,12]:
  _check(tp_mlp(x,w1,b1,w2,b2,parts),np.maximum(x@w1+b1,0)@w2+b2)`,
hidden: true
},
{
name: "全负激活只剩偏置",
code: py`x=-np.ones((2,3));w1=np.ones((3,4));w2=np.ones((4,2));b2=np.array([5.,-3.])
_check(tp_mlp(x,w1,np.zeros(4),w2,b2,2),np.tile(b2,(2,1)))`,
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
id: "nonlinear",
q: md`为什么不能对行切第一层的部分输出分别做ReLU？`,
a: md`因为行切产生的是同一隐藏分量的部分内积，需要先求和才得到非线性输入。例如a=−1、b=2，ReLU(a+b)=1，而ReLU(a)+ReLU(b)=2。列切则分配不同输出分量，每个分量已包含全部输入维度，所以可直接本地激活。这说明通信位置由算子代数性质决定，而不只是由张量形状决定。`,
freq: 3
},
{
id: "ce",
q: md`词表并行为什么至少需要全局最大值与全局指数和？`,
a: md`全局最大值让所有词表分片共享数值稳定平移，指数和提供统一归一化分母。若各卡减不同最大值后直接加局部指数和，尺度不一致；若分别softmax，概率和也不为1。目标logit只由持有该词表项的rank提供，再按约定归约，避免对全部词表all-gather。`,
freq: 3
},
{
id: "spcp",
q: md`SP与CP该如何区分？`,
a: md`在Megatron常用术语中，SP主要减少TP组内部分激活复制，让逐token操作处理序列分片；CP把长序列及注意力计算分到不同设备，通过KV交换等保持全局连接。不同框架命名可能交叉，因此面试最好直接描述切哪个轴、哪些状态被分片、注意力需要什么通信，避免仅背缩写。`,
freq: 2
}
],
cheatsheet: md`W₁:[D,H]列切 → 本地f → W₂:[H,O]行切 → SUM → b₂一次。

ReLU(a+b)≠ReLU(a)+ReLU(b)。

词表CE：全局m=MAX(logits)；Z=SUM(exp(logits−m))；loss=m+logZ−目标logit。

SP分片激活，CP分担长序列注意力；明确具体框架语义。`
})
