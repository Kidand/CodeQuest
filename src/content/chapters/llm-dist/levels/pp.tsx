import { defineLevel, md, py } from '@/content/helpers'
import { PipelineViz } from '../viz/DistributedViz'

export default defineLevel({
learn: [
{
id: "bubbles",
title: "微批越多，越能摊薄首尾空闲",
body: md`把模型切成p级，每个batch再切成m个microbatch。第一份数据到达后级之前，它们只能等待；最后一份离开前级后，前级也会空闲。理想同速、无通信的GPipe气泡率为 $(p-1)/(m+p-1)$，因此增加m能摊薄首尾开销。但microbatch变小会降低GEMM效率，通信次数也增加；m不能无限增大。真实训练的最慢阶段决定稳态吞吐，切层时要考虑注意力、FFN、embedding、loss等不同代价，而非仅让层数相等。`,
viz: PipelineViz,
checkpoint: {
type: "numeric",
id: "bubble",
prompt: md`理想GPipe，p=4级、m=12微批，各级同速且忽略通信，气泡率(p−1)/(m+p−1)是多少？填小数。`,
explanation: md`3/(12+3)=0.2。该公式不是任意1F1B或不均衡调度的通用精确时延。`,
difficulty: 2,
answer: 0.2,
tolerance: 0.001
}
},
{
id: "onef",
title: "1F1B先解决在飞激活的压力",
body: md`GPipe可以先完成所有微批前向再反向，因此需要保留很多激活。非交错1F1B在warmup后交替执行一次前向和一次反向，及时释放旧微批激活。rank r的典型warmup是min(p−r−1,m)，末级无需等待更多前向。每卡依旧受邻卡的前向数据、下游梯度、发送接收完成等约束；局部事件表只是偏序，不能把第i个事件都排到同一个全局时隙。优化器一般在完整batch梯度聚合后更新，以免微批看到不同权重版本。`,
checkpoint: {
type: "numeric",
id: "warmnum",
prompt: md`p=6，rank=2（从0计），m=2，本关warmup数量是多少？`,
explanation: md`min(6−2−1,2)=2。不能无条件暖机3个，因为只有2个微批。`,
difficulty: 3,
answer: 2,
tolerance: 0.001
}
},
{
id: "advanced",
title: "减少气泡需要付出新的代价",
body: md`Interleaved调度给一张卡放多个虚拟阶段，缩短阶段粒度但增加跨阶段传输。Zero-bubble利用反向中输入梯度与权重梯度的不同依赖，把部分工作移入空隙；额外内存、实现约束与精确负载决定能否接近零气泡。[DualPipe 官方实现](https://github.com/deepseek-ai/DualPipe) 展示双向流水与通信计算重叠，它不是简单把GPipe气泡公式中的p除以二。面试应说清依赖、激活峰值与网络条件，再比较调度；单报一个理论利用率无法解释实际吞吐。`
}
],
quiz: [
{
type: "single",
id: "pp",
prompt: md`PP直接切分哪一维工作？`,
explanation: md`PP把连续或虚拟模型阶段放到不同设备，再让microbatch流过；TP切的是单个算子。`,
difficulty: 1,
options: [
md`模型层或阶段`,
md`同一矩阵的列`,
md`词表项`,
md`只有batch`
],
answer: 0
},
{
type: "numeric",
id: "bubble",
prompt: md`理想GPipe，p=4级、m=12微批，各级同速且忽略通信，气泡率(p−1)/(m+p−1)是多少？填小数。`,
explanation: md`3/(12+3)=0.2。该公式不是任意1F1B或不均衡调度的通用精确时延。`,
difficulty: 2,
answer: 0.2,
tolerance: 0.001
},
{
type: "judge",
id: "zero",
prompt: md`1F1B一定消除所有流水线气泡。`,
explanation: md`1F1B主要限制在飞激活；仍有暖机和排空，且可能有通信与阶段不平衡。`,
difficulty: 1,
answer: false
},
{
type: "multi",
id: "one",
prompt: md`1F1B通常需要考虑哪些约束？`,
explanation: md`本地列表仅给执行顺序，不是全局同步时隙；依赖决定能否实际启动。`,
difficulty: 2,
options: [
md`同一微批B不能早于其F`,
md`反向还要等待下游梯度`,
md`各卡本地事件表无需通信依赖就能同时按索引执行`,
md`微批增多可能增加调度与通信开销`
],
answer: [
0,
1,
3
]
},
{
type: "match",
id: "versions",
prompt: md`匹配调度思路`,
explanation: md`更少气泡通常伴随不同的内存与通信折中；名字不等于任何配置都零气泡。`,
difficulty: 2,
pairs: [
[
md`GPipe`,
md`前向批次后统一反向`
],
[
md`1F1B`,
md`稳态交替前后向`
],
[
md`Interleaved`,
md`每卡多个虚拟阶段`
],
[
md`Zero-bubble`,
md`拆分反向工作以填空闲`
]
]
},
{
type: "order",
id: "sequence",
prompt: md`生成本地1F1B事件的顺序`,
explanation: md`末级warmup为0；微批不足时warmup必须截断。`,
difficulty: 2,
items: [
md`确定warmup数量`,
md`先生成warmup前向`,
md`交替剩余前向和最早未完成反向`,
md`排空余下反向`
]
},
{
type: "bug",
id: "warm",
prompt: md`这个warmup计算漏了什么边界？`,
explanation: md`应为min(p-rank-1,m)。当m较小时会生成不存在的microbatch并使remaining为负。`,
difficulty: 3,
code: py`def warmup_events(p,m,rank):
    warm = p-rank-1
    events = []
    for i in range(warm):
        events.append((1,i))
    remaining = m-warm
    assert remaining >= 0
    return events`,
bugLines: [
2
]
},
{
type: "numeric",
id: "warmnum",
prompt: md`p=6，rank=2（从0计），m=2，本关warmup数量是多少？`,
explanation: md`min(6−2−1,2)=2。不能无条件暖机3个，因为只有2个微批。`,
difficulty: 3,
answer: 2,
tolerance: 0.001
}
],
guided: [
{
id: "g-schedule",
title: "逐步实现：生成非交错 1F1B 本地事件表",
intro: md`实现 \`schedule_1f1b(stages, microbatches, rank) -> np.ndarray\`。stages≥1，microbatches≥0，0≤rank<stages。返回int64数组(2m,2)，每行[op,microbatch_id]，op=1前向、−1反向。warmup=min(stages−rank−1,m)，先发这些F；稳态每次发下一个F再发最早尚未反向的B；最后排空剩余B。每卡事件顺序不等于全局时隙：实际执行仍需等待邻卡依赖和通信。m=0返回(0,2)。`,
template: py`import numpy as np

def schedule_1f1b(stages: int, microbatches: int, rank: int) -> np.ndarray:
    warm = {{1}}
    events = [(1, i) for i in range(warm)]
    remaining = microbatches - warm
    for i in range(remaining):
        events.append({{2}})
        events.append((-1, i))
    events.extend((-1, i) for i in {{3}})
    return np.asarray(events, dtype=np.int64).reshape(-1, 2)`,
steps: [
{
blank: 1,
instruction: md`距末级越远，必须先暖机更多微批；但不能超过微批总数。`,
answer: "min(stages - rank - 1, microbatches)",
explain: md`距末级越远，必须先暖机更多微批；但不能超过微批总数。`
},
{
blank: 2,
instruction: md`稳态发尚未执行的下一个前向。`,
answer: "(1, warm + i)",
explain: md`稳态发尚未执行的下一个前向。`
},
{
blank: 3,
instruction: md`稳态已反向remaining个，排空从这个编号开始。`,
answer: "range(remaining, microbatches)",
explain: md`稳态已反向remaining个，排空从这个编号开始。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "三级第一卡三微批",
code: py`_check(schedule_1f1b(3,3,0),[[1,0],[1,1],[1,2],[-1,0],[-1,1],[-1,2]])`,
hidden: false
},
{
name: "末级直接交替",
code: py`_check(schedule_1f1b(3,2,2),[[1,0],[-1,0],[1,1],[-1,1]])`,
hidden: false
},
{
name: "每个前后向一次且前向先于反向",
code: py`for p in [1,2,4,8]:
 for m in [1,2,7]:
  for rank in range(p):
   s=schedule_1f1b(p,m,rank);assert s.shape==(2*m,2),f"期望{(2*m,2)}，得到{s.shape}"
   forwards=[int(i) for op,i in s if op==1];backwards=[int(i) for op,i in s if op==-1]
   assert forwards==list(range(m)) and backwards==list(range(m)),f"遗漏或重复事件{s}"
   live=0;peak=0;seen=set()
   for op,i in s:
    if op==1:seen.add(int(i));live+=1
    else:assert int(i) in seen,f"反向早于前向{i}";live-=1
    peak=max(peak,live)
   assert peak<=min(p-rank,m),f"峰值{peak}超过1F1B边界"`,
hidden: true
}
]
}
],
challenges: [
{
id: "schedule",
title: "生成非交错 1F1B 本地事件表",
difficulty: "medium",
timeLimitMin: 25,
description: md`实现 \`schedule_1f1b(stages, microbatches, rank) -> np.ndarray\`。stages≥1，microbatches≥0，0≤rank<stages。返回int64数组(2m,2)，每行[op,microbatch_id]，op=1前向、−1反向。warmup=min(stages−rank−1,m)，先发这些F；稳态每次发下一个F再发最早尚未反向的B；最后排空剩余B。每卡事件顺序不等于全局时隙：实际执行仍需等待邻卡依赖和通信。m=0返回(0,2)。`,
starter: py`import numpy as np

def schedule_1f1b(stages: int, microbatches: int, rank: int) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def schedule_1f1b(stages: int, microbatches: int, rank: int) -> np.ndarray:
    warm = min(stages - rank - 1, microbatches)
    events = [(1, i) for i in range(warm)]
    remaining = microbatches - warm
    for i in range(remaining):
        events.append((1, warm + i))
        events.append((-1, i))
    events.extend((-1, i) for i in range(remaining, microbatches))
    return np.asarray(events, dtype=np.int64).reshape(-1, 2)`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def schedule_1f1b(stages,microbatches,rank):
    warm=min(stages-rank-1,microbatches)
    events=[(1,i) for i in range(warm)]
    for i in range(microbatches-warm):
        events.extend([(1,warm+i),(-1,i)])
    events.extend((-1,i) for i in range(microbatches-warm,microbatches))
    return torch.tensor(events,dtype=torch.int64).reshape(-1,2)`
}
],
explanation: md`warmup由到最后一级的距离决定，并受微批总数限制。稳态前向序号领先反向warm个，因此最多保留约warm+1份在飞激活；排空只发剩余反向。返回本地偏序而不编造全局并行时间，通信或负载不均会插入等待。时间与空间O(m)。`,
hints: [
md`距末级越远，必须先暖机更多微批；但不能超过微批总数。`,
md`稳态发尚未执行的下一个前向。`,
md`稳态已反向remaining个，排空从这个编号开始。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "三级第一卡三微批",
code: py`_check(schedule_1f1b(3,3,0),[[1,0],[1,1],[1,2],[-1,0],[-1,1],[-1,2]])`,
hidden: false
},
{
name: "末级直接交替",
code: py`_check(schedule_1f1b(3,2,2),[[1,0],[-1,0],[1,1],[-1,1]])`,
hidden: false
},
{
name: "空批次与微批小于warmup",
code: py`_check(schedule_1f1b(8,0,0),np.empty((0,2)))
_check(schedule_1f1b(8,1,0),[[1,0],[-1,0]])`,
hidden: true
},
{
name: "每个前后向一次且前向先于反向",
code: py`for p in [1,2,4,8]:
 for m in [1,2,7]:
  for rank in range(p):
   s=schedule_1f1b(p,m,rank);assert s.shape==(2*m,2),f"期望{(2*m,2)}，得到{s.shape}"
   forwards=[int(i) for op,i in s if op==1];backwards=[int(i) for op,i in s if op==-1]
   assert forwards==list(range(m)) and backwards==list(range(m)),f"遗漏或重复事件{s}"
   live=0;peak=0;seen=set()
   for op,i in s:
    if op==1:seen.add(int(i));live+=1
    else:assert int(i) in seen,f"反向早于前向{i}";live-=1
    peak=max(peak,live)
   assert peak<=min(p-rank,m),f"峰值{peak}超过1F1B边界"`,
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
id: "local",
q: md`为什么本地1F1B列表不能直接当全局时间表？`,
a: md`每个事件是否能启动取决于输入激活或下游梯度是否到达。不同阶段前反向耗时不一定相等，通信也会等待，因此各卡第i个事件没有共同时间含义。正确的全局模拟需要依赖图、设备资源和传输时延；本题先验证局部顺序、每微批前后向各一次及激活上界，这些是调度正确性的必要条件。`,
freq: 3
},
{
id: "m",
q: md`怎样选microbatch数量？`,
a: md`增加数量可以摊薄暖机排空，但固定全局batch下会缩小单个microbatch，使矩阵乘利用率下降并增加通信启动成本。还需保持与DP数量、梯度累积和样本长度一致。合理做法是先满足显存与整除条件，再在几个候选配置上测吞吐、气泡和峰值激活，而不是让m尽可能大。`,
freq: 3
},
{
id: "imbalance",
q: md`层数均分为什么仍可能失衡？`,
a: md`层的成本取决于实际序列长度、FFN/MoE负载、注意力类型以及embedding和输出词表。最慢一级会让其他卡在稳态也等待。应按profile的前后向耗时切分，必要时使用虚拟阶段改善均衡，同时检查跨设备传输成本，避免切得过细导致通信反而主导。`,
freq: 2
}
],
cheatsheet: md`GPipe理想气泡率 $(p-1)/(m+p-1)$。

本地1F1B：warm=min(p−rank−1,m)；warmup F → 稳态 F/B → cooldown B。

本地顺序≠全局时隙；1F1B省在飞激活，不保证零气泡。

更多microbatch需权衡GEMM效率、通信次数与全局batch。`
})
