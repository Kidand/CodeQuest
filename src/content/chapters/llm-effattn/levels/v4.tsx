import { defineLevel, md, py } from '@/content/helpers'
import { CompressionViz } from '../viz/EfficientViz'

export default defineLevel({
learn: [
{
id: "architecture",
title: "先减少条目，再决定读哪些条目",
body: md`[DeepSeek-V4 官方报告](https://arxiv.org/html/2606.19348v1) 公开 CSA/HCA 混合注意力。CSA 轻压缩后按 indexer 选择摘要；HCA 重压缩后对摘要稠密注意力；局部未压缩窗口补足细节。§4.2.1 给出 m=4、m′=128、窗口128；Flash top-k=512，Pro=1024。以下推导与练习聚焦这些算子的教学抽象，省略完整投影与归一化，不等于复刻模型。一个重要反例是把每块求均值当作精确注意力：均值抹掉块内差异，随后 query 已无法分别选择原始 token，所以它依赖训练适配。`,
viz: CompressionViz,
checkpoint: {
type: "single",
id: "difference",
prompt: md`CSA 与 HCA 的关键差异是什么？`,
explanation: md`二者都压缩序列维。HCA 的摘要更少，因此直接读取所有已完成摘要；局部窗口补充细节。`,
difficulty: 1,
options: [
md`CSA 压缩后再选择，HCA 更重压缩后稠密读取摘要`,
md`CSA 不压缩，HCA 不注意任何全局信息`,
md`HCA 是另一个模型的 MLP`,
md`二者只改变精度`
],
answer: 0
}
},
{
id: "compression",
title: "重叠不翻倍，通道各自挑选",
body: md`教学压缩器组合当前块 a 路和前一块 b 路，两个长度 m 的序列合成一个候选集合。对每个通道 d，令 $p_{j,d}=\exp(z_{j,d})/\sum_u\exp(z_{u,d})$，摘要为 $c_d=\sum_j p_{j,d}x_{j,d}$。这种逐通道权重比单个 token 标量更灵活。首块的缺失 b 路不能用零 logit 填充，否则分母平白多出概率质量；可省略它或填 −∞。尽管看到了2m个候选，每推进m个 token只生成一个摘要，长期条目仍约T/m。块末才完成的摘要不能暴露给块内更早的 query。`,
checkpoint: {
type: "bug",
id: "axis",
prompt: md`找出归一化轴错误。`,
explanation: md`应按 token 轴 axis=0 归一化，让每个通道的权重和为1；axis=1 是在通道间竞争。`,
difficulty: 3,
code: py`import numpy as np
def pool(c, z):
    # c,z: [token, channel]
    shifted = z-z.max(axis=0,keepdims=True)
    p = np.exp(shifted)
    p = p/p.sum(axis=1,keepdims=True)
    out = (p*c).sum(axis=0)
    return out`,
bugLines: [
6
]
}
},
{
id: "memory",
title: "按实际存储对象估算 KV",
body: md`先明确是 K、V 分离，还是共享压缩条目；再区分长期摘要、indexer key、滑窗、未完成尾块与临时缓冲。教学共享摘要的字节数可写为 $B L (\lfloor T/m\rfloor+W)c b$，但混合模型需逐层按 m 或 m′ 相加，indexer 还要另计。top-k k 决定本次读取量，不决定长期缓存长度。实际部署还有量化 scale、块对齐、并发和重复前缀等影响。比较时必须固定型号、精度与上下文，不能把某个摘要维度压缩率直接当作总服务显存下降比例。`
}
],
quiz: [
{
type: "single",
id: "difference",
prompt: md`CSA 与 HCA 的关键差异是什么？`,
explanation: md`二者都压缩序列维。HCA 的摘要更少，因此直接读取所有已完成摘要；局部窗口补充细节。`,
difficulty: 1,
options: [
md`CSA 压缩后再选择，HCA 更重压缩后稠密读取摘要`,
md`CSA 不压缩，HCA 不注意任何全局信息`,
md`HCA 是另一个模型的 MLP`,
md`二者只改变精度`
],
answer: 0
},
{
type: "numeric",
id: "count",
prompt: md`忽略尾块和额外缓存，T=4096、m=4，CSA 的长期压缩条目数为多少？`,
explanation: md`每 m 个新 token 产生一个摘要，4096/4=1024；两路重叠不会变成2048。`,
difficulty: 2,
answer: 1024,
tolerance: 0.001
},
{
type: "judge",
id: "equal",
prompt: md`加权压缩是对任意 Q 都与未压缩 softmax 注意力完全等价的精确重排。`,
explanation: md`压缩会丢失细节，注意力又是 query 相关的非线性操作；它是需要训练的结构，不是 FlashAttention 那类精确执行优化。`,
difficulty: 1,
answer: false
},
{
type: "multi",
id: "facts",
prompt: md`按官方报告 §4.2.1，哪些配置正确？`,
explanation: md`报告给出 Flash top-k=512、Pro=1024；不能把一个型号的选择数套给另一个。`,
difficulty: 2,
options: [
md`CSA m=4`,
md`HCA m′=128`,
md`Flash 与 Pro 的 top-k 完全相同`,
md`附加窗口 n_win=128`
],
answer: [
0,
1,
3
]
},
{
type: "match",
id: "roles",
prompt: md`匹配三种处理`,
explanation: md`三者优化的对象不同，通常需要联合设计；窗口不能恢复任意远处细节。`,
difficulty: 2,
pairs: [
[
md`压缩`,
md`减少长期条目数量`
],
[
md`选择`,
md`减少当前主注意力读取`
],
[
md`局部窗口`,
md`保留近期未压缩细节`
]
]
},
{
type: "order",
id: "causal",
prompt: md`一个安全的流式压缩流程是？`,
explanation: md`不完整块含未来 token 的离线摘要不能提前暴露给块内 query。`,
difficulty: 2,
items: [
md`写入当前未压缩 token`,
md`判断是否完成压缩块`,
md`完成时计算并发布摘要`,
md`query 只读取已发布且满足因果条件的摘要`
]
},
{
type: "bug",
id: "axis",
prompt: md`找出归一化轴错误。`,
explanation: md`应按 token 轴 axis=0 归一化，让每个通道的权重和为1；axis=1 是在通道间竞争。`,
difficulty: 3,
code: py`import numpy as np
def pool(c, z):
    # c,z: [token, channel]
    shifted = z-z.max(axis=0,keepdims=True)
    p = np.exp(shifted)
    p = p/p.sum(axis=1,keepdims=True)
    out = (p*c).sum(axis=0)
    return out`,
bugLines: [
6
]
},
{
type: "numeric",
id: "bytes",
prompt: md`教学单层共享 KV 摘要，每项512维，每数2字节；T=4096、m=4，外加128项未压缩窗口，忽略 indexer、尾块、对齐和额外元数据。共多少字节？`,
explanation: md`(4096/4+128)×512×2=1,179,648。题设 K/V 共享条目，不应机械乘2。`,
difficulty: 3,
answer: 1179648,
tolerance: 0.001
}
],
guided: [
{
id: "g-csa-compress",
title: "逐步实现：双路重叠、逐通道加权压缩",
intro: md`实现 \`compress(ca, cb, za, zb, ba, bb, block)\`。前四个数组形状 (T,D)，位置偏置 ba/bb 为 (block,D)，均为有限 float64；block≥1，T≥0。第 i 个已完成块组合 ca 的当前块与 cb 的前一块；对应 logits 为 za+ba、zb+bb。每个通道分别沿 token 轴 softmax，加权得到 (floor(T/block),D)。首块只有 a 路；未完成尾块暂不发布。此题实现 CSA 的压缩核心，省略投影、RMSNorm、RoPE 与 indexer。例：a=[1,3,5,7]，b=10a、所有 logits=0、block=2，输出 [2,13]。`,
template: py`import numpy as np

def compress(ca: np.ndarray, cb: np.ndarray, za: np.ndarray, zb: np.ndarray, ba: np.ndarray, bb: np.ndarray, block: int) -> np.ndarray:
    n, d = ca.shape
    result = np.empty(({{1}}, d), dtype=np.float64)
    for i in range(n // block):
        start = i * block
        values = ca[start:start+block]
        logits = za[start:start+block] + ba
        if i > 0:
            values = np.concatenate([values, cb[start-block:start]], axis=0)
            logits = np.concatenate([logits, zb[start-block:start] + bb], axis=0)
        p = np.exp(logits - {{2}})
        p /= p.sum(axis=0, keepdims=True)
        result[i] = {{3}}
    return result`,
steps: [
{
blank: 1,
instruction: md`只发布完成的整块；尾部细节仍由缓存另行管理。`,
answer: "n // block",
explain: md`只发布完成的整块；尾部细节仍由缓存另行管理。`
},
{
blank: 2,
instruction: md`每个通道沿 token 轴求最大值。`,
answer: "logits.max(axis=0, keepdims=True)",
explain: md`每个通道沿 token 轴求最大值。`
},
{
blank: 3,
instruction: md`同一通道的 token 加权和形成一个摘要分量。`,
answer: "np.sum(p * values, axis=0)",
explain: md`同一通道的 token 加权和形成一个摘要分量。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_compress(ca,cb,za,zb,ba,bb,block):
    out=np.zeros((len(ca)//block,ca.shape[1]))
    for i in range(len(out)):
        for d in range(ca.shape[1]):
            entries=[(ca[i*block+j,d],za[i*block+j,d]+ba[j,d]) for j in range(block)]
            if i:
                entries += [(cb[(i-1)*block+j,d],zb[(i-1)*block+j,d]+bb[j,d]) for j in range(block)]
            m=max(z for _,z in entries);den=sum(np.exp(z-m) for _,z in entries)
            out[i,d]=sum(x*np.exp(z-m) for x,z in entries)/den
    return out`,
tests: [
{
name: "均匀权重、前块重叠",
code: py`a=np.array([[1.],[3.],[5.],[7.]]);b=a*10;z=np.zeros_like(a);bias=np.zeros((2,1))
_check(compress(a,b,z,z,bias,bias,2),[[2.],[13.]])`,
hidden: false
},
{
name: "尾部不足整块暂不发布",
code: py`a=np.arange(5.)[:,None];z=np.zeros_like(a);bias=np.zeros((2,1))
_check(compress(a,a,z,z,bias,bias,2),[[0.5],[1.5]])`,
hidden: false
},
{
name: "按通道归一化与极端 logits",
code: py`r=np.random.default_rng(32)
for n,b in [(0,2),(4,1),(7,3),(12,4)]:
 arrays=[r.normal(size=(n,3)) for _ in range(4)];arrays[2]*=1000;arrays[3]*=1000
 ba=r.normal(size=(b,3));bb=r.normal(size=(b,3))
 _check(compress(*arrays,ba,bb,b),_ref_compress(*arrays,ba,bb,b))`,
hidden: true
}
]
}
],
challenges: [
{
id: "csa-compress",
title: "双路重叠、逐通道加权压缩",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`compress(ca, cb, za, zb, ba, bb, block)\`。前四个数组形状 (T,D)，位置偏置 ba/bb 为 (block,D)，均为有限 float64；block≥1，T≥0。第 i 个已完成块组合 ca 的当前块与 cb 的前一块；对应 logits 为 za+ba、zb+bb。每个通道分别沿 token 轴 softmax，加权得到 (floor(T/block),D)。首块只有 a 路；未完成尾块暂不发布。此题实现 CSA 的压缩核心，省略投影、RMSNorm、RoPE 与 indexer。例：a=[1,3,5,7]，b=10a、所有 logits=0、block=2，输出 [2,13]。`,
starter: py`import numpy as np

def compress(ca: np.ndarray, cb: np.ndarray, za: np.ndarray, zb: np.ndarray, ba: np.ndarray, bb: np.ndarray, block: int) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def compress(ca: np.ndarray, cb: np.ndarray, za: np.ndarray, zb: np.ndarray, ba: np.ndarray, bb: np.ndarray, block: int) -> np.ndarray:
    n, d = ca.shape
    result = np.empty((n // block, d), dtype=np.float64)
    for i in range(n // block):
        start = i * block
        values = ca[start:start+block]
        logits = za[start:start+block] + ba
        if i > 0:
            values = np.concatenate([values, cb[start-block:start]], axis=0)
            logits = np.concatenate([logits, zb[start-block:start] + bb], axis=0)
        p = np.exp(logits - logits.max(axis=0, keepdims=True))
        p /= p.sum(axis=0, keepdims=True)
        result[i] = np.sum(p * values, axis=0)
    return result`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def compress(ca, cb, za, zb, ba, bb, block):
    result = []
    for i in range(ca.shape[0] // block):
        start = i * block
        values, logits = ca[start:start+block], za[start:start+block] + ba
        if i > 0:
            values = torch.cat([values, cb[start-block:start]], dim=0)
            logits = torch.cat([logits, zb[start-block:start] + bb], dim=0)
        result.append((torch.softmax(logits, dim=0) * values).sum(dim=0))
    return torch.stack(result) if result else ca.new_empty((0, ca.shape[1]))`
}
],
explanation: md`两路重叠不意味着输出翻倍：每前进 block 个 token，只生成一个摘要。首块缺少前块，应当对缺失位置加 −∞ 或等价地完全忽略，而不是把零值与零 logit 加入分母。按通道、沿 token 轴做 softmax，允许各通道选择不同来源。时间 O(TD)，工作空间 O(block·D)，另有输出 O(TD/block)。`,
hints: [
md`只发布完成的整块；尾部细节仍由缓存另行管理。`,
md`每个通道沿 token 轴求最大值。`,
md`同一通道的 token 加权和形成一个摘要分量。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"


def _ref_compress(ca,cb,za,zb,ba,bb,block):
    out=np.zeros((len(ca)//block,ca.shape[1]))
    for i in range(len(out)):
        for d in range(ca.shape[1]):
            entries=[(ca[i*block+j,d],za[i*block+j,d]+ba[j,d]) for j in range(block)]
            if i:
                entries += [(cb[(i-1)*block+j,d],zb[(i-1)*block+j,d]+bb[j,d]) for j in range(block)]
            m=max(z for _,z in entries);den=sum(np.exp(z-m) for _,z in entries)
            out[i,d]=sum(x*np.exp(z-m) for x,z in entries)/den
    return out`,
tests: [
{
name: "均匀权重、前块重叠",
code: py`a=np.array([[1.],[3.],[5.],[7.]]);b=a*10;z=np.zeros_like(a);bias=np.zeros((2,1))
_check(compress(a,b,z,z,bias,bias,2),[[2.],[13.]])`,
hidden: false
},
{
name: "尾部不足整块暂不发布",
code: py`a=np.arange(5.)[:,None];z=np.zeros_like(a);bias=np.zeros((2,1))
_check(compress(a,a,z,z,bias,bias,2),[[0.5],[1.5]])`,
hidden: false
},
{
name: "没有完成的块",
code: py`a=np.ones((1,3));bias=np.zeros((2,3))
_check(compress(a,a,a,a,bias,bias,2),np.empty((0,3)))`,
hidden: true
},
{
name: "按通道归一化与极端 logits",
code: py`r=np.random.default_rng(32)
for n,b in [(0,2),(4,1),(7,3),(12,4)]:
 arrays=[r.normal(size=(n,3)) for _ in range(4)];arrays[2]*=1000;arrays[3]*=1000
 ba=r.normal(size=(b,3));bb=r.normal(size=(b,3))
 _check(compress(*arrays,ba,bb,b),_ref_compress(*arrays,ba,bb,b))`,
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
id: "two",
q: md`两路重叠为什么可能改善压缩？`,
a: md`单块边界是人为的，重要信息可能横跨边界。将当前与前块的不同投影合并，允许摘要保留跨边界关系，并利用逐通道权重分配有限容量。代价是需要维护上一块状态和更复杂的压缩 kernel；它仍无法保证保留所有 token 细节，不能据此宣称无损压缩。`,
freq: 3
},
{
id: "causal",
q: md`离线压缩最容易出现什么因果错误？`,
a: md`先用整个块生成摘要，再让块内每个 query 读取它，会把块后半段的信息泄露给块前半段。必须明确摘要的可用时刻，通常仅在块完成后发布，同时用未压缩窗口覆盖近期位置。测试可修改未来 token，确认过去输出保持不变；这种不变性测试比正常前向对拍更容易发现泄露。`,
freq: 3
},
{
id: "comparison",
q: md`为什么不只使用极高压缩率的 HCA？`,
a: md`更高压缩率减少摘要数量，却把更多细节压进同一向量，容易形成检索瓶颈。轻压缩加选择保留较细粒度的长期候选，重压缩提供较便宜的全局覆盖，局部窗口保留近期精细信息。混合比例要由质量与成本的实验证据确定，而不是因为单层 HCA 最省就全部替换。`,
freq: 2
}
],
cheatsheet: md`- CSA：压缩 → indexer → top-k；HCA：重压缩 → 摘要稠密注意力。
- 官方配置：m=4、m′=128、窗口128；k：Flash512 / Pro1024。
- 两路2m候选 → 每m个新token产出1摘要。
- softmax 沿 token 轴、逐通道；首块缺失分支不能稀释分母。
- 显存计摘要 + indexer + 窗口 + 尾块，不用k替代T/m。`
})
