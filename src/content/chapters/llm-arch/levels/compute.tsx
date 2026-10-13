import { defineLevel, md, py } from '@/content/helpers'
import type { Challenge } from '@/content/types'
import { ComputeViz } from '../viz/ArchViz'
import { single,judge,numeric,multi,order,fill,bug,practice } from '../shared'
const c:Challenge={id:'estimate',title:'GQA 参数账本与训练时间',difficulty:'medium',description:md`实现 \`estimate(config: dict, tokens: int, gpus: int, peak_tflops: float, mfu: float) -> dict\`。config 包含 vocab、dim、layers、heads、kv_heads、ffn、tied、batch、seq、bytes，均为正整数（tied 为 bool），dim 能被 heads 整除，heads 能被 kv_heads 整除；忽略全部 bias。每层为 GQA+SwiGLU+两个带 dim 个缩放参数的 RMSNorm；模型另有 final RMSNorm 和输入 embedding，untied 时加独立 LM head。返回整数 params、kv_bytes，浮点 train_flops=6×params×tokens 与 seconds=train_flops/(gpus×peak_tflops×10^12×mfu)。KV 缓存只含 K/V。tokens≥0、0<mfu≤1。这是主项估计，未计注意力二次项、重计算或通信。`,starter:py`import numpy as np
def estimate(config: dict, tokens: int, gpus: int, peak_tflops: float, mfu: float) -> dict:
    raise NotImplementedError`,solution:py`import numpy as np
def estimate(config: dict, tokens: int, gpus: int, peak_tflops: float, mfu: float) -> dict:
    c=config;d=c['dim'];dh=d//c['heads'];dkv=c['kv_heads']*dh
    attention = 2*d*d + 2*d*dkv
    ffn = 3*d*c['ffn']
    per_layer = attention + ffn + 2*d
    embeddings = c['vocab']*d*(1 if c['tied'] else 2)
    params = embeddings + c['layers']*per_layer + d
    kv_bytes = 2*c['layers']*c['batch']*c['seq']*dkv*c['bytes']
    train_flops = 6.0*params*tokens
    seconds = train_flops/(gpus*peak_tflops*1e12*mfu)
    return dict(params=int(params),kv_bytes=int(kv_bytes),train_flops=train_flops,seconds=seconds)`,solutionAlt:[{label:'PyTorch 版（仅展示）',code:py`# 配置估算无需 GPU 张量；实际模块可独立核对共享参数。
def count_actual(model):
    # model.parameters() 默认去重共享的 Parameter。
    return sum(p.numel() for p in model.parameters())
def estimate_torch(c,tokens,gpus,peak_tflops,mfu):
    d=c['dim'];k=c['kv_heads']*(d//c['heads'])
    n=c['vocab']*d*(1 if c['tied'] else 2)+c['layers']*(2*d*d+2*d*k+3*d*c['ffn']+2*d)+d
    kv=2*c['layers']*c['batch']*c['seq']*k*c['bytes']
    return {'params':n,'kv_bytes':kv,'train_flops':6*n*tokens,'seconds':6*n*tokens/(gpus*peak_tflops*1e12*mfu)}`}],explanation:md`先按算子计数再汇总，避免背诵 12Ld² 套到所有架构。Q/O 为 2d²，K/V 为 2d×Hkv×dh，SwiGLU 为 3dF。时间是工作量/有效吞吐，TFLOP/s 的 T=10^12，与 TiB 等二进制单位无关。估算为 O(1)；KV 和参数分开计算，因为缓存随 batch、seq 增长而参数不变。`,hints:[md`先计算 dkv=kv_heads×head_dim。`,md`tied embedding 只计一个 VD；final norm 在 L 层之外。`,md`MFU 乘在分母的峰值吞吐上。`],setup:py`_C=dict(vocab=10,dim=8,layers=2,heads=4,kv_heads=2,ffn=12,tied=True,batch=3,seq=5,bytes=2)`,tests:[{name:'参数和 KV 账本',code:py`a=estimate(_C,100,2,1,.5)
assert a['params']==1080, f"参数期望 1080，得到 {a}"
assert a['kv_bytes']==480, f"KV 期望 480，得到 {a}"
assert np.isclose(a['train_flops'],648000) and np.isclose(a['seconds'],6.48e-7), f"估算错误 {a}"`},{name:'untied 增量',code:py`a=estimate({**_C,'tied':False},100,2,1,.5)
assert a['params']==1160, f"多一份 80 参数，得到 {a}"`},{name:'KV 缩放与零训练 token',hidden:true,code:py`a=estimate({**_C,'seq':10,'batch':6},0,1,100,.2)
assert a['kv_bytes']==1920 and a['params']==1080 and a['seconds']==0, f"期望 KV1920、参数1080、0秒，得到 {a}"`},{name:'MHA 参数修正',hidden:true,code:py`a=estimate({**_C,'kv_heads':4},100,4,1,.25)
assert a['params']==1208 and a['kv_bytes']==960, f"期望参数1208、KV960，得到 {a}"
b=estimate(_C,100,2,1,.25);c=estimate(_C,100,2,1,.5)
assert np.isclose(b['seconds'],2*c['seconds']), f"MFU 减半时间翻倍，得到 {b},{c}"`}],followUps:[{id:'dense',q:md`6ND 可以直接用于 MoE 总参数吗？`,a:md`不能。主矩阵乘主要取决于每 token 激活的参数及路由结构，embedding/head、共享专家和注意力还需单独核对。总参数更适合估存储，激活参数更接近计算主项，但通信和稀疏批处理效率仍会改变实际时长。`}]}
export default defineLevel({learn:[{id:'params',title:'12Ld² 是特定结构的近似',body:md`传统 MHA 的 Q/K/V/O 四个 d×d 矩阵贡献 4d²；普通 FFN 若中间宽度 4d，两矩阵共 8d²，因此得到每层 12d²。GQA 时 Q/O 仍为 2d²，K/V 变为 2dHkv d_h；SwiGLU 有三矩阵，为 3dF。不能一边把 FFN 换成 SwiGLU、一边仍假定 F=4d 却保留原公式。完整模型还需 embedding/LM head、norm 和 bias（若存在）；tied 权重只计一次。`,viz:ComputeViz},{id:'flops',title:'6ND 与 2N 都是主项估算',body:md`一个权重参与一次乘加按 2 FLOPs 计，密集网络前向每 token 的主项约 2N，反向约是前向的两倍，因此训练主项约 6ND_train。注意力 QKᵀ 与 PV 还有随序列长度增长的额外项，长上下文时不能忽略；训练重计算也增加执行 FLOPs。MFU 是模型工作量相对理论峰值吞吐的利用率，估时用 C/(卡数×单卡峰值×MFU)，不要把 MFU 当成加速倍数乘到时间上。精度和稀疏峰值口径必须一致。`},{id:'memory',title:'参数、状态、激活与 KV 分账',body:md`推理显存至少分权重、KV、工作区；KV 字节数为 2LBTHkv d_h b，不能把 Q 头数代入 GQA 缓存。训练还需梯度、优化器状态、可能的 FP32 主权重、保存激活与通信缓冲，具体 bytes/parameter 取决于优化器和分片策略。Chinchilla 研究的是给定计算预算下模型与数据规模的共同配置，不能把某个 tokens/parameter 比例当成所有训练目标的固定法律，尤其面向推理成本优化时目标不同。参考：[Chinchilla](https://arxiv.org/abs/2203.15556)。`}],quiz:[single('twelve',1,md`12d² 中的 8d² 来自？`,['Attention','宽度 4d 的双矩阵 FFN','embedding','KV cache'],1,md`d×4d 加 4d×d=8d²；前提是普通双矩阵 FFN。`),judge('gqa',1,md`GQA 将 Q、K、V、O 四种投影按同一比例减少。`,false,md`通常仅 K/V 输出宽度减少，Q/O 不变。`),numeric('swiglu',2,md`d=100、F=200，无 bias 的 SwiGLU 参数量？`,60000,md`3×100×200=60000。`),multi('memory',2,md`训练显存可能包含哪些？`,['优化器状态','保存的激活','梯度','只包含推理 KV'],[0,1,2],md`标准全量训练一般不使用自回归推理 KV 缓存；其内存账本不能直接照搬推理。`),order('time',md`训练估时步骤：`,['确定模型 FLOPs 口径','根据 token 数求工作量','计算同精度峰值乘 MFU','用工作量除有效吞吐'],md`工作量与吞吐单位需匹配，MFU 在吞吐端乘上去。`),fill('kv',md`GQA KV 显存中的头数应为 ____(1)____，填 Hq 或 Hkv。`,['Hkv'],md`缓存的是 K/V，Q 头数不决定 KV 存储宽度。`),single('tied',2,md`V=1000,d=64，tied 改 untied 增加多少参数？`,['64000','128000','64','0'],0,md`新增一份 V×d 的 LM head，原 embedding 已存在。`),judge('moe',3,md`MoE 每 token 主计算量可简单按所有未激活专家总参数乘 2。`,false,md`未激活专家通常不执行，该公式会高估主计算；路由与通信另计。`),bug('mfu',py`def seconds(params, tokens, gpus, peak_tflops, mfu):
    forward = 2 * params * tokens
    backward = 2 * forward
    work = forward + backward
    peak = gpus * peak_tflops * 1e12
    ideal = work / peak
    actual = ideal * mfu
    return actual`,[7],md`实际时间应 ideal/mfu；利用率越低应越慢，而不是更快。`) ],guided:[practice(c,[{find:'2*d*d + 2*d*dkv',instruction:md`按四个注意力投影分别计参数。`,explain:md`Q/O 是 d×d，K/V 是 d×dkv。`},{find:'3*d*c[\'ffn\']',instruction:md`统计 SwiGLU 参数。`,explain:md`gate、up、down 共三矩阵。`},{find:'2*c[\'layers\']*c[\'batch\']*c[\'seq\']*dkv*c[\'bytes\']',instruction:md`写 KV 缓存字节公式。`,explain:md`2 表示 K 与 V；只看 KV 宽度。`},{find:'train_flops/(gpus*peak_tflops*1e12*mfu)',instruction:md`从总工作量求秒数。`,explain:md`有效吞吐=理论峰值×利用率。`}])],challenges:[c],interview:[{id:'2n',freq:3,q:md`推理每 token 2N 是否足够估延迟？`,a:md`不够。它只估主矩阵乘 FLOPs，decode 小 batch 往往受权重和 KV 读写带宽限制，长上下文还有注意力额外算量。要结合 batch、KV 长度、权重精度和硬件 roofline，将计算下界与访存下界取较大者，再考虑通信和调度。`},{id:'mfu',freq:3,q:md`MFU 和硬件利用率有什么区别？`,a:md`MFU 的分子是约定的模型有效 FLOPs，不一定包含激活重计算等额外工作；硬件 FLOPs 利用率可以把这些实际执行工作也计入。两者都依赖峰值精度口径，通信等待、算子效率和并行策略会影响结果。报告数字时必须说明计算公式而不是只给百分比。`},{id:'chinchilla',freq:2,q:md`Chinchilla 的结论如何用于工程决策？`,a:md`它提醒我们在固定训练计算预算下同时配置模型参数和数据量，单纯增大模型可能欠训练。但部署目标还包含推理成本、可用数据质量和延迟约束，常会选择更小但训练更久的模型。经验比例来自特定实验范围，不应不加验证外推到 MoE、代码或高质量重复数据。`}],cheatsheet:md`参数：L(2d²+2dHkv d_h+3dF+2d)+Vd×(tied?1:2)+d。训练≈6ND；前向主项≈2N/token。时间=C/(GPU×峰值×MFU)。KV=2LBTHkv d_h bytes。`})
