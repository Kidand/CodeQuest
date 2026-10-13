import { defineLevel, md, py } from '@/content/helpers'
import { ZeroViz } from '../viz/DistributedViz'

export default defineLevel({
learn: [
{
id: "review",
title: "并行策略是带约束的设计问题",
body: md`先把每个维度的职责说清：DP增加独立数据副本；TP切算子；PP切模型阶段；CP切同一样本的上下文；EP按专家放置路由，组关系由框架定义。然后明确哪些状态被切分、哪些临时恢复完整、同一时刻有多少在飞激活。本关将这些条件写成一个可执行筛选器，返回满足世界大小、全局batch、整除与显存预算的候选。它只保证题设模型下的可行性，不保证吞吐最优；实际选型仍需profile网络、kernel效率与阶段不平衡。`,
viz: ZeroViz
}
],
quiz: [
{
type: "single",
id: "bottleneck",
prompt: md`7B模型权重能装下，但Adam训练OOM，首先应检查？`,
explanation: md`训练状态远多于推理权重；还要计峰值激活和临时量。`,
difficulty: 1,
options: [
md`持久优化器/梯度状态和激活峰值`,
md`只缩小词表显示字体`,
md`把每参数字节固定当2`,
md`只看磁盘checkpoint大小`
],
answer: 0
},
{
type: "numeric",
id: "batch",
prompt: md`DP=4，microbatch=2，累积8步，全局batch是多少？`,
explanation: md`4×2×8=64。TP/PP/CP处理同一样本的不同计算，不应再乘入样本数。`,
difficulty: 2,
answer: 64,
tolerance: 0.001
},
{
type: "judge",
id: "cp-memory",
prompt: md`把序列长度分成CP=4份，就可以无条件将所有模型参数与Adam状态也除以4。`,
explanation: md`CP切序列，通常不直接切参数；状态分片由TP/PP/ZeRO等机制决定。`,
difficulty: 1,
answer: false
},
{
type: "multi",
id: "correct",
prompt: md`以下哪些发布前验证有意义？`,
explanation: md`shape正确仍可能有偏置重复、归一化错误或因果泄露。正确性和实际性能都需验证。`,
difficulty: 2,
options: [
md`单卡与分布式小规模数值对拍`,
md`不同全局token数的loss归一化检查`,
md`只检查最终张量shape`,
md`记录真实峰值与通信耗时`
],
answer: [
0,
1,
3
]
},
{
type: "match",
id: "choose",
prompt: md`匹配典型瓶颈与可考虑手段`,
explanation: md`方案可以组合，但要支付对应通信和调度成本。`,
difficulty: 2,
pairs: [
[
md`冗余优化器状态太大`,
md`ZeRO分片`
],
[
md`单层矩阵计算太大`,
md`张量并行`
],
[
md`模型层太多`,
md`流水线并行`
],
[
md`单样本上下文太长`,
md`上下文并行`
]
]
},
{
type: "order",
id: "plan",
prompt: md`制定并行策略的合理顺序`,
explanation: md`先排除不能运行或改变语义的配置，再用真实性能数据选择。`,
difficulty: 2,
items: [
md`列清模型与激活账本`,
md`确定设备拓扑和目标batch`,
md`构造满足整除与group约束的配置`,
md`做小规模语义验证`,
md`profile吞吐与显存后选择`
]
},
{
type: "bug",
id: "batch-bug",
prompt: md`全局batch公式哪里错？`,
explanation: md`replicas应为dp。TP和PP并行同一模型处理同一批样本，不增加独立数据副本。`,
difficulty: 3,
code: py`def global_batch(dp,tp,pp,micro,accum):
    # micro是每个DP副本的微批样本数
    samples_per_replica = micro*accum
    replicas = dp*tp*pp
    total = replicas*samples_per_replica
    assert total > 0
    result = total
    return result`,
bugLines: [
4
]
},
{
type: "numeric",
id: "z2",
prompt: md`P=1e9参数、DP=4，仅ZeRO-2持久状态按16B约定，每卡多少GB？1GB=1e9字节。`,
explanation: md`权重2P仍复制，梯度与优化器14P分4份：2+14/4=5.5GB。`,
difficulty: 3,
answer: 5.5,
tolerance: 0.001
},
{
type: "judge",
id: "speed",
prompt: md`显存最低的可行配置一定有最高训练吞吐。`,
explanation: md`更激进的分片可能增加通信或减小有效GEMM规模；容量和吞吐目标不同，需测量。`,
difficulty: 2,
answer: false
},
{
type: "numeric",
id: "tp-output",
prompt: md`TP=2，两卡输出部分和分别[1,2]和[3,4]，输出偏置[1,1]。最终结果第二个分量是多少？`,
explanation: md`2+4+1=7，输出偏置只加一次。`,
difficulty: 2,
answer: 7,
tolerance: 0.001
}
],
guided: [
{
id: "g-planner",
title: "逐步实现：筛选满足显存与并行约束的配置",
intro: md`实现 \`feasible_plans(params,activation,layer,configs,budget,world,batch,heads,length,zero)\`。configs为(C,6)整数数组，每行[DP,TP,PP,CP,microbatch,accum]，可能含非正无效项；zero∈{0,1,2,3}。需满足设备数乘积=world、DP·micro·accum=batch、heads可被TP整除、length可被CP整除。模型状态按16B基准及ZeRO阶段分片；每卡参数量=params/(TP·PP)。activation是单样本未分片激活字节，本题假设均匀分到TP·PP·CP并保留min(PP,accum)个微批。ZeRO-3另加2·layer/TP字节临时聚合，其它阶段为0。峰值不超过budget时返回原行索引，按峰值升序、同分索引升序，dtype int64。params/activation/layer≥0，其余标量为正；C可为0。这是明确假设的规划练习，不是生产峰值预测器，EP不纳入该设备网格。`,
template: py`import numpy as np

def feasible_plans(params: int, activation: int, layer: int, configs: np.ndarray, budget: float, world: int, batch: int, heads: int, length: int, zero: int) -> np.ndarray:
    feasible = []
    for index, row in enumerate(configs):
        dp, tp, pp, cp, micro, accum = map(int, row)
        if min(dp,tp,pp,cp,micro,accum) < 1:
            continue
        if dp*tp*pp*cp != world or dp*micro*accum != batch:
            continue
        if heads % tp or length % cp:
            continue
        coefficient = {{1}}
        persistent = params / (tp*pp) * coefficient
        live = {{2}}
        saved = activation * micro * live / (tp*pp*cp)
        gather = 2*layer/tp if zero == 3 else 0.0
        peak = {{3}}
        if peak <= budget:
            feasible.append((peak, index))
    feasible.sort()
    return np.asarray([index for peak,index in feasible], dtype=np.int64)`,
steps: [
{
blank: 1,
instruction: md`按ZeRO阶段分别分片优化器、梯度与参数，不要总是直接除DP。`,
answer: "[16, 4+12/dp, 2+14/dp, 16/dp][zero]",
explain: md`按ZeRO阶段分别分片优化器、梯度与参数，不要总是直接除DP。`
},
{
blank: 2,
instruction: md`本题采用首级1F1B在飞微批数上界，最多PP份且不超过总微批数。`,
answer: "min(pp, accum)",
explain: md`本题采用首级1F1B在飞微批数上界，最多PP份且不超过总微批数。`
},
{
blank: 3,
instruction: md`峰值账本包含持久状态、在飞激活与临时聚合。`,
answer: "persistent + saved + gather",
explain: md`峰值账本包含持久状态、在飞激活与临时聚合。`
}
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "同显存按原索引稳定排序",
code: py`c=np.array([[2,2,1,1,1,2],[2,2,1,1,2,1],[2,2,1,1,1,2]])
_check(feasible_plans(100,40,20,c,900,4,4,4,8,0),[0,2,1])`,
hidden: false
},
{
name: "边界预算与ZeRO-3聚合",
code: py`c=np.array([[4,1,1,1,1,1]])
_check(feasible_plans(100,40,20,c,480,4,4,4,8,3),[0])
_check(feasible_plans(100,40,20,c,479,4,4,4,8,3),[])`,
hidden: false
},
{
name: "流水线在飞激活与独立账本",
code: py`c=np.array([[1,1,4,1,1,8],[1,1,4,1,4,2],[2,1,2,1,1,4]])
# Z2: 前两项状态400，第三项450；激活分别40、80、40
_check(feasible_plans(100,40,20,c,490,4,8,8,8,2),[0,1,2])`,
hidden: true
}
]
}
],
challenges: [
{
id: "planner",
title: "筛选满足显存与并行约束的配置",
difficulty: "hard",
timeLimitMin: 25,
description: md`实现 \`feasible_plans(params,activation,layer,configs,budget,world,batch,heads,length,zero)\`。configs为(C,6)整数数组，每行[DP,TP,PP,CP,microbatch,accum]，可能含非正无效项；zero∈{0,1,2,3}。需满足设备数乘积=world、DP·micro·accum=batch、heads可被TP整除、length可被CP整除。模型状态按16B基准及ZeRO阶段分片；每卡参数量=params/(TP·PP)。activation是单样本未分片激活字节，本题假设均匀分到TP·PP·CP并保留min(PP,accum)个微批。ZeRO-3另加2·layer/TP字节临时聚合，其它阶段为0。峰值不超过budget时返回原行索引，按峰值升序、同分索引升序，dtype int64。params/activation/layer≥0，其余标量为正；C可为0。这是明确假设的规划练习，不是生产峰值预测器，EP不纳入该设备网格。`,
starter: py`import numpy as np

def feasible_plans(params: int, activation: int, layer: int, configs: np.ndarray, budget: float, world: int, batch: int, heads: int, length: int, zero: int) -> np.ndarray:
    # TODO: 按题面约定实现
    raise NotImplementedError`,
solution: py`import numpy as np

def feasible_plans(params: int, activation: int, layer: int, configs: np.ndarray, budget: float, world: int, batch: int, heads: int, length: int, zero: int) -> np.ndarray:
    feasible = []
    for index, row in enumerate(configs):
        dp, tp, pp, cp, micro, accum = map(int, row)
        if min(dp,tp,pp,cp,micro,accum) < 1:
            continue
        if dp*tp*pp*cp != world or dp*micro*accum != batch:
            continue
        if heads % tp or length % cp:
            continue
        coefficient = [16, 4+12/dp, 2+14/dp, 16/dp][zero]
        persistent = params / (tp*pp) * coefficient
        live = min(pp, accum)
        saved = activation * micro * live / (tp*pp*cp)
        gather = 2*layer/tp if zero == 3 else 0.0
        peak = persistent + saved + gather
        if peak <= budget:
            feasible.append((peak, index))
    feasible.sort()
    return np.asarray([index for peak,index in feasible], dtype=np.int64)`,
solutionAlt: [
{
label: "PyTorch 版（仅展示）",
code: py`import torch

def feasible_plans(params,activation,layer,configs,budget,world,batch,heads,length,zero):
    result=[]
    for index,row in enumerate(configs.tolist()):
        dp,tp,pp,cp,micro,accum=map(int,row)
        if min(row)<1:continue
        if dp*tp*pp*cp!=world or dp*micro*accum!=batch:continue
        if heads%tp or length%cp:continue
        coefficient=[16,4+12/dp,2+14/dp,16/dp][zero]
        peak=params/(tp*pp)*coefficient+activation*micro*min(pp,accum)/(tp*pp*cp)
        if zero==3:peak+=2*layer/tp
        if peak<=budget:result.append((peak,index))
    result.sort()
    return torch.tensor([index for peak,index in result],dtype=torch.int64,device=configs.device)`
}
],
explanation: md`先验证配置的整数可行性，再计算状态、在飞激活和临时聚合。TP/PP分片模型，DP影响ZeRO分片；CP只改变本题激活项，不能把参数也除CP。按(peak,index)排序保证并列稳定。总复杂度O(C log C)，空间O(C)。真实系统还需通信、拓扑、非均匀层、框架SP实现和碎片profile，此题只验证明示模型。`,
hints: [
md`按ZeRO阶段分别分片优化器、梯度与参数，不要总是直接除DP。`,
md`本题采用首级1F1B在飞微批数上界，最多PP份且不超过总微批数。`,
md`峰值账本包含持久状态、在飞激活与临时聚合。`
],
setup: py`import numpy as np

def _check(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    assert got.shape == expected.shape, f"期望形状 {expected.shape}，得到 {got.shape}"
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-7), f"期望 {expected}，得到 {got}"`,
tests: [
{
name: "同显存按原索引稳定排序",
code: py`c=np.array([[2,2,1,1,1,2],[2,2,1,1,2,1],[2,2,1,1,1,2]])
_check(feasible_plans(100,40,20,c,900,4,4,4,8,0),[0,2,1])`,
hidden: false
},
{
name: "边界预算与ZeRO-3聚合",
code: py`c=np.array([[4,1,1,1,1,1]])
_check(feasible_plans(100,40,20,c,480,4,4,4,8,3),[0])
_check(feasible_plans(100,40,20,c,479,4,4,4,8,3),[])`,
hidden: false
},
{
name: "过滤无效世界大小、batch和整除",
code: py`c=np.array([[2,2,1,1,1,2],[2,2,1,1,1,3],[1,3,1,1,1,4],[1,1,1,4,1,4],[0,1,1,1,1,4]])
_check(feasible_plans(100,40,20,c,1e9,4,4,3,6,1),[])`,
hidden: true
},
{
name: "零行输入",
code: py`_check(feasible_plans(100,40,20,np.empty((0,6),dtype=int),1e9,4,4,8,8,2),[])`,
hidden: true
},
{
name: "流水线在飞激活与独立账本",
code: py`c=np.array([[1,1,4,1,1,8],[1,1,4,1,4,2],[2,1,2,1,1,4]])
# Z2: 前两项状态400，第三项450；激活分别40、80、40
_check(feasible_plans(100,40,20,c,490,4,8,8,8,2),[0,1,2])`,
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
id: "strategy",
q: md`8卡训练先开TP=8，还是DP=8加ZeRO？`,
a: md`要从单层计算、模型状态、激活和网络条件判断。若模型在单卡或ZeRO分片后能容纳且batch充足，DP通常更直接；若单层或计算规模需要拆分，则考虑高带宽组内TP。PP/CP解决不同瓶颈，不能只按卡数机械设degree。先验证可行性，再测目标序列与batch下的吞吐。`,
freq: 3
},
{
id: "debug",
q: md`多卡loss与单卡不一致，优先排查什么？`,
a: md`固定权重和输入，关闭随机性，先比较小规模前向和逐层输出。检查TP偏置是否重复、词表CE是否全局归一化、DP是否重复除world size、变长token是否加权、CP是否用绝对因果位置。随后再看随机数、浮点归约次序与混合精度容差。不要一开始就把明显偏差归因于数值误差。`,
freq: 3
},
{
id: "feasible",
q: md`为什么规划器选出可行配置仍可能OOM？`,
a: md`估算器依赖均匀切层、激活保留比例和预取窗口等假设。实际大层临时聚合、通信bucket重叠、碎片以及不同框架存储方式会造成更高峰值。应保留合理余量，并记录峰值时间线找出同时存活的对象，更新估算模型；可行性预测应允许被实测修正。`,
freq: 2
}
],
cheatsheet: md`全局batch=DP×microbatch×accum。

独立网格设备数=DP×TP×PP×CP；EP需另讲group映射。

分片谁、临时聚合谁、同时存活谁——三个问题先回答。

单卡数值对拍 → collectives/因果检查 → 峰值profile → 吞吐比较。`
})
