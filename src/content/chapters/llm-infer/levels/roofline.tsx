/**
 * L10-1 Prefill / Decode 与 Roofline：算术强度推导、decode 访存瓶颈、prefill 算力瓶颈、batch 的作用与上限、
 * TTFT / TPOT / 吞吐、吞吐上界估算。
 * GPU 规格：NVIDIA 官网（A100 / H100 / H200 / DGX B200 规格页，Tensor Core 稀疏值 ÷ 2 得稠密值），详见 ../viz/hw.ts
 */
import { defineLevel, md, py } from '@/content/helpers'
import { MatmulIntensityViz } from '../viz/MatmulIntensityViz'
import { RooflineViz } from '../viz/RooflineViz'
import { BatchCurveViz } from '../viz/BatchCurveViz'

export default defineLevel({
  learn: [
    {
      id: 'phases',
      title: '两个阶段、四个指标',
      body: md`
        一次生成请求分两个阶段：

        | | Prefill（预填充） | Decode（解码） |
        |---|---|---|
        | 输入 | 整个 prompt（$P$ 个 token）一次性并行处理 | 每步只输入上一步生成的 1 个 token |
        | 做什么 | 算出所有 prompt token 的 KV、产出第 1 个 token | 读全部历史 KV，产出下一个 token |
        | 矩阵形状 | GEMM：$[P \times d] \cdot [d \times d']$ | GEMV（batch 内拼起来才是 $[B \times d]$ 的瘦 GEMM） |
        | 瓶颈 | 通常是**算力** | 通常是**显存带宽** |
        | 决定的指标 | TTFT | TPOT |

        面试和压测里最常用的指标：

        - **TTFT**（time to first token）：从请求到达到第 1 个 token 返回，包含排队 + prefill；
        - **TPOT**（time per output token）/ **ITL**（inter-token latency）：之后相邻两个 token 的间隔，由 decode 单步时间决定；
        - 端到端延迟 $\approx \text{TTFT} + (n_\text{out} - 1)\cdot \text{TPOT}$；
        - **吞吐**（tokens/s 或 requests/s）：整台机器单位时间产出多少；**goodput**（DistServe 提出）：在同时满足 TTFT 与 TPOT 的 SLO 前提下能承受的最大请求速率。

        吞吐与延迟往往此消彼长：batch 越大吞吐越高，但每一步更慢、新请求的 prefill 还会插队拖慢正在 decode 的请求。本关用 **Roofline** 把这些现象量化。
      `,
      checkpoint: {
        id: 'cp-phases',
        type: 'numeric',
        prompt: md`某请求 TTFT = 200 ms，TPOT = 20 ms，共输出 101 个 token。端到端延迟约为多少毫秒？`,
        answer: 2200,
        tolerance: 0,
        unit: 'ms',
        explanation: md`第 1 个 token 在 TTFT 时返回，之后还有 100 个 token，每个间隔 TPOT：$200 + 100 \times 20 = 2200$ ms。常见错误是写成 $200 + 101 \times 20$。`,
      },
    },
    {
      id: 'intensity',
      title: '算术强度：每读 1 字节能做多少次运算',
      body: md`
        **算术强度**（arithmetic intensity）$I = \dfrac{\text{FLOPs}}{\text{HBM 读写字节数}}$。

        以线性层 $Y = XW$ 为例，$X \in \mathbb{R}^{T \times d}$，$W \in \mathbb{R}^{d \times d}$，每个元素 $b$ 字节（BF16 时 $b = 2$）：

        $$
        \text{FLOPs} = 2Td^2,\qquad \text{Bytes} = b\,(d^2 + 2Td),\qquad I = \frac{2Td^2}{b(d^2 + 2Td)} \overset{b=2}{=} \frac{T d}{d + 2T}
        $$

        - **decode**，$T = B$（batch）很小：$I \approx T$。batch = 1 时 $I \approx 1$——每个权重从显存读进来只做一次乘加；
        - **prefill**，$T = P$ 成百上千：$I$ 随 $T$ 增长，最终饱和到 $d/2$（$d = 4096$ 时约 2048）。

        **注意力**要单独算。decode 时每个新 token 要和 $S$ 个历史位置做 $QK^\top$ 和 $AV$：每层 FLOPs $= 4\,n_h d_h S$，读 KV 的字节 $= 2\,n_{kv} d_h S\cdot b$，于是

        $$
        I_\text{attn} = \frac{4 n_h d_h S B}{2 n_{kv} d_h S B \cdot b} \overset{b=2}{=} \frac{n_h}{n_{kv}}
        $$

        **与 batch 无关**：每个序列的 KV 只服务它自己的 query，加 batch 只是同比例增加 FLOPs 和字节。MHA 下强度为 1，GQA-8 下为 8；只有 GQA / MLA（更多 query 头共享一份 KV）或更低精度的 KV 才能提高它。
      `,
      viz: MatmulIntensityViz,
      after: md`
        > [!NOTE] 为什么 prefill 的注意力是算力瓶颈
        > prefill 时 $P$ 个 query 共享同一份 $P$ 个位置的 K/V，FLOPs $\propto P^2$、字节 $\propto P$，强度随 $P$ 线性增长。FlashAttention 把它做成分块 GEMM，进一步避免了 $P \times P$ 矩阵的读写（见「注意力机制」章）。
      `,
      checkpoint: {
        id: 'cp-intensity',
        type: 'judge',
        prompt: md`decode 阶段把 batch 从 1 增大到 64，注意力部分（读 KV cache 的 $QK^\top$、$AV$）的算术强度会随之增大约 64 倍。`,
        answer: false,
        explanation: md`错误。batch 提升的是**线性层**的强度（同一份权重服务 $B$ 个 token）。注意力里每个序列读的是**自己**的 KV，FLOPs 和字节同比例增长，强度恒为 $n_h / n_{kv}$。所以长上下文、大 batch 时 KV 读取会成为 decode 的主导开销。`,
      },
    },
    {
      id: 'roofline',
      title: 'Roofline 模型：带宽斜坡与算力平顶',
      body: md`
        一个算子在某 GPU 上能达到的性能上界：

        $$
        \text{Perf}(I) = \min\big(\pi,\ \beta \cdot I\big)
        $$

        $\pi$ 是峰值算力（FLOP/s），$\beta$ 是显存带宽（Byte/s）。两条线的交点 $I^* = \pi / \beta$ 叫**拐点**（ridge point）：
        $I < I^*$ 时受带宽限制（斜坡区，Perf 随 $I$ 线性增长），$I > I^*$ 时受算力限制（平顶区）。等价地，算子耗时下界为

        $$
        t \ge \max\!\left(\frac{\text{FLOPs}}{\pi},\ \frac{\text{Bytes}}{\beta}\right)
        $$

        | GPU（NVIDIA 官网规格） | BF16 稠密算力 | HBM 带宽 | 显存 | 拐点 $\pi/\beta$ |
        |---|---|---|---|---|
        | A100 80GB SXM | 312 TFLOPS | 2.039 TB/s | 80 GB | ≈ 153 |
        | H100 SXM | 989 TFLOPS | 3.35 TB/s | 80 GB | ≈ 295 |
        | H200 SXM | 989 TFLOPS | 4.8 TB/s | 141 GB | ≈ 206 |
        | B200（DGX B200 每卡） | 2.25 PFLOPS | 8 TB/s | 180 GB | ≈ 281 |

        官网 Tensor Core 数字标注的是「with sparsity」（2:4 结构化稀疏），稠密值是它的一半；B200 由 HGX / DGX B200 的 8 卡总量折算。
        H100 上 batch = 1 的 decode 强度约为 1，只能用到峰值算力的约 $3.35 / 989 \approx 0.3\%$——GPU 几乎一直在等数据。
      `,
      viz: RooflineViz,
      after: md`
        > [!TIP] 读图
        > - 拖 batch：蓝点（线性层）沿斜坡右移，batch 到几百才碰到平顶；橙点（注意力）一动不动；
        > - 拖上下文：KV 字节变多，整步（紫色圆环）被拉向橙点——长上下文 decode 越来越像「纯读 KV」；
        > - 切 INT4：权重字节减为 1/4，线性层强度 ×4，batch = 1 时单步时间几乎减为 1/4（weight-only 量化对 decode 有效的原因），但 prefill 不受益；
        > - 选 H200：算力与 H100 相同、带宽更高，拐点更低，同样的 decode 更快。
      `,
      checkpoint: {
        id: 'cp-roofline',
        type: 'numeric',
        prompt: md`按上表，H100 SXM 的拐点约为多少 FLOP/Byte？（取整）`,
        answer: 295,
        tolerance: 0.01,
        unit: 'FLOP/B',
        explanation: md`$989 \times 10^{12} / (3.35 \times 10^{12}) \approx 295$。注意用的是**稠密** BF16 算力；若误用官网的稀疏值 1979 TFLOPS，会得到 591，高估一倍。`,
      },
    },
    {
      id: 'batch',
      title: 'batch 的作用与上限',
      body: md`
        把一步 decode 的 FLOPs 和字节写全（$N$ 为参数量，$w$ 为每参数字节，$\kappa$ 为每 token 的 KV 字节，$S$ 为上下文长度）：

        $$
        \text{FLOPs}(B) = \big(2N + 4 L n_h d_h S\big)\,B,\qquad \text{Bytes}(B) = N w + B S \kappa
        $$

        - batch 小时，字节由**权重**主导，单步时间 $\approx Nw/\beta$ 几乎不随 $B$ 变化，吞吐 $\approx B\beta/(Nw)$ **线性增长**——这就是推理服务拼命做 batching 的原因；
        - 只看权重，算力拐点约在 $B^* \approx I^* \cdot w / 2$（BF16、H100 上约 300）；
        - 但 KV 字节也随 $B$ 线性增长，强度的上限是 $\lim_{B\to\infty} I = \dfrac{2N + 4Ln_hd_hS}{S\kappa}$。上下文长时这个上限**低于拐点**：Llama-3.1-8B 在 $S = 1024$ 时约为 124，$S = 8192$ 时约为 19，而 $S \gtrsim 420$ 就已低于 H100 的 295——decode 永远到不了算力瓶颈；
        - batch 还受**显存容量**限制：$B_{\max} = \lfloor(\text{显存} - Nw - \text{其它}) / (S\kappa)\rfloor$。

        所以 decode 优化可以归结为「每读 1 字节多干点活」或「少读字节」：增大 batch（连续批处理 + PagedAttention 腾显存）、权重量化、GQA / MLA / KV 量化、投机解码（一次前向验证多个 token）。
      `,
      viz: BatchCurveViz,
      after: md`
        > [!INTERVIEW] 面试常问：「batch 越大越好吗？」
        > 吞吐上去了，TPOT 也变大，超过拐点后单步时间随 batch 线性增长、吞吐不再增加；还要受 KV 显存和 TPOT SLO 约束。
        > 线上一般以「满足 TPOT / TTFT SLO 的最大吞吐」（goodput）为目标选 batch 上限，而不是一味加大。
      `,
      checkpoint: {
        id: 'cp-batch',
        type: 'single',
        prompt: md`Llama-3.1-8B 在 H100 上做 decode，上下文 32K。持续增大 batch（显存足够），最可能出现的情况是：`,
        options: [
          'batch 到 300 左右就进入算力瓶颈，之后吞吐不再增加',
          '强度的上限低于拐点，始终受带宽限制；吞吐增长逐渐放缓，单步时间主要花在读 KV 上',
          '单步时间与 batch 无关，吞吐无限线性增长',
          '强度随 batch 无限增大',
        ],
        answer: 1,
        explanation: md`上下文很长时每个序列的 KV 字节远大于均摊到它头上的权重字节，强度上限 $\approx (2N + 4Ln_hd_hS)/(S\kappa)$ 只有几，远低于 295。A 只在短上下文、只考虑权重时成立；C、D 忽略了 KV 字节随 batch 增长。`,
      },
    },
    {
      id: 'estimate',
      title: '估算题：吞吐与延迟上界',
      body: md`
        面试的估算题几乎都是「谁是瓶颈 → 用它除」。以下都忽略通信、kernel 效率与调度开销，得到的是**上界**（实测通常明显更低）。

        **例 1：Llama-3.1-8B（$N = 8.03 \times 10^9$）BF16，H100，batch = 1，短上下文。**
        每步读权重 $16.06$ GB，$t \ge 16.06 / 3350\ \text{s} \approx 4.8$ ms，单序列最多约 **208 token/s**。

        **例 2：Llama-3.1-70B BF16，4 × H100 张量并行。**
        权重 141 GB 分到 4 卡，每卡读 35.3 GB，$t \ge 35.3 / 3350 \approx 10.5$ ms（总带宽 13.4 TB/s），单序列约 95 token/s 上界；TP 每层还有 all-reduce，实际更慢。

        **例 3：同一个 8B 模型，2048 token 的 prompt 做 prefill。**
        FLOPs $\approx 2NP + 2Ln_hd_hP^2 = 32.9 + 1.1 = 34.0$ TFLOP，$t \ge 34.0 / 989 \approx 34$ ms（100% 算力利用率），这是 TTFT 中 prefill 部分的下界。

        **例 4：MoE 的 decode。** 计算量看**激活**参数，访存要看本步**被选中的专家数**。DeepSeek-V3 每层 256 个路由专家选 8 个：batch = 1 时每层只读 8 个专家；batch = 64 时期望被选中的专家数为
        $256 \cdot \big(1 - (1 - 8/256)^{64}\big) \approx 222$（假设路由均匀独立），几乎要读全部专家，而每个专家平均只分到 $64 \times 8 / 222 \approx 2.3$ 个 token——MoE 的 decode 比同激活规模的稠密模型更难摆脱访存瓶颈，这也是大规模专家并行（EP）把专家分散到很多卡、让每卡的专家攒够 token 的动机。
      `,
      after: md`
        > [!WARN] 估算时的三个常见错误
        > 1. 用了官网的**稀疏**算力（翻倍）；
        > 2. 权重字节乘了 batch——权重每步只读一遍，被 batch 内所有 token 共享；
        > 3. 把「计算时间」和「访存时间」相加，或取较小者——它们可以重叠，下界取 **max**。
      `,
      checkpoint: {
        id: 'cp-estimate',
        type: 'numeric',
        prompt: md`Llama-3.1-8B（16.06 GB BF16 权重）在 H200（4.8 TB/s）上 batch = 1 decode，忽略 KV，单序列吞吐上界约为多少 token/s？`,
        answer: 298.9,
        tolerance: 0.02,
        unit: 'token/s',
        explanation: md`$4.8 \times 10^{12} / (16.06 \times 10^9) \approx 299$ token/s。算力与 H100 相同，但带宽从 3.35 提到 4.8 TB/s，batch = 1 的 decode 就快了约 43%——访存瓶颈下只有带宽说了算。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-why-mem',
      type: 'single',
      difficulty: 1,
      prompt: md`小 batch 的 decode 受显存带宽限制，最根本的原因是：`,
      options: [
        'decode 的 FLOPs 比 prefill 多',
        '每步要把全部权重（和 KV）从 HBM 读一遍，但每个读进来的权重只和极少的 token 做乘加，算术强度远低于 GPU 拐点',
        'decode 必须在 CPU 上采样',
        'KV cache 存在 CPU 内存里',
      ],
      answer: 1,
      explanation: md`batch = 1 时线性层强度约 1 FLOP/B，而 H100 的拐点约 295。A 恰好相反（单步 FLOPs 很少）；C、D 不是常规实现。`,
    },
    {
      id: 'q-attn-batch',
      type: 'judge',
      difficulty: 2,
      prompt: md`GQA（每 8 个 query 头共享一组 KV）把 decode 中注意力部分的算术强度提高到 MHA 的约 8 倍。`,
      answer: true,
      explanation: md`$I_\text{attn} = n_h / n_{kv}$（BF16 KV）：MHA 为 1，GQA-8 为 8。读一份 KV 服务 8 个 query 头，FLOPs 不变、字节变为 1/8。`,
    },
    {
      id: 'q-tok-s',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Llama-3.1-8B BF16 权重共 16.06 GB（GB = $10^9$ 字节），H100 SXM 带宽 3.35 TB/s。忽略 KV 与其它开销，batch = 1 时单序列 decode 速度上界约为多少 token/s？`,
      answer: 208.6,
      tolerance: 0.02,
      unit: 'token/s',
      explanation: md`$3.35 \times 10^{12} / 16.06 \times 10^9 \approx 208.6$。每生成一个 token 都要把全部权重读一遍。`,
    },
    {
      id: 'q-70b',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Llama-3.1-70B（$7.06 \times 10^{10}$ 参数，BF16）用 4 张 H100 SXM（每张 3.35 TB/s）做张量并行。忽略通信与 KV，decode 的 TPOT 下界约为多少毫秒？`,
      answer: 10.54,
      tolerance: 0.02,
      unit: 'ms',
      explanation: md`权重 $1.412 \times 10^{11}$ 字节，4 卡并行读，等效带宽 13.4 TB/s：$1.412 \times 10^{11} / 1.34 \times 10^{13} \approx 10.5$ ms。实际还要加上每层两次 all-reduce 的延迟。`,
    },
    {
      id: 'q-kvcap',
      type: 'numeric',
      difficulty: 3,
      prompt: md`Llama-3.1-8B 每 token 的 KV 为 131072 字节（BF16）。H100 80 GB，扣除权重 16.06 GB 后其余全部给 KV（GB = $10^9$，忽略激活与碎片）。上下文 8192 时最多能同时服务多少个序列？`,
      answer: 59,
      tolerance: 0,
      unit: '个',
      explanation: md`每序列 KV $= 8192 \times 131072 \approx 1.074 \times 10^9$ 字节；$(80 - 16.06) \times 10^9 / 1.074 \times 10^9 \approx 59.5$，向下取整为 59。这远低于算力拐点（约 300），说明长上下文 decode 的 batch 先被**显存容量**卡住。`,
    },
    {
      id: 'q-levers',
      type: 'multi',
      difficulty: 2,
      prompt: md`对于**小 batch、访存瓶颈**的 decode，以下哪些手段能缩短单步时间或提高每次权重读取产出的 token 数？（多选）`,
      options: [
        'INT4 weight-only 量化',
        '投机解码（一次前向验证多个草稿 token）',
        '换一张带宽相同、BF16 算力翻倍的 GPU',
        '用 GQA / MLA 替代 MHA',
        '连续批处理，把更多请求放进同一步',
      ],
      answer: [0, 1, 3, 4],
      explanation: md`A 减少权重字节；B 让一次权重读取服务多个位置；D 减少 KV 字节；E 增大 batch 提高线性层强度。C 无效：访存瓶颈下算力不是限制因素。`,
    },
    {
      id: 'q-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`BF16 线性层 $[T \times 4096] \cdot [4096 \times 4096]$，按 $I = \dfrac{Td}{d + 2T}$：$T = 1$ 时 $I \approx$ ____(1)____；$T = 4096$ 时 $I \approx$ ____(2)____（保留 1 位小数）。`,
      blanks: [
        { label: '(1)', answers: ['1'], numeric: true, tolerance: 0.01 },
        { label: '(2)', answers: ['1365.3'], numeric: true, tolerance: 0.01 },
      ],
      explanation: md`$T=1$：$4096/4098 \approx 1$。$T = 4096$：$4096 \times 4096 / (4096 + 8192) = 1365.3$。prefill 一个几千 token 的 prompt，线性层强度远超任何 GPU 的拐点。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`按 BF16 稠密规格的**拐点**（算力 / 带宽）从低到高排列：`,
      items: ['A100 80GB SXM（312 TFLOPS / 2.039 TB/s）', 'H200 SXM（989 TFLOPS / 4.8 TB/s）', 'B200（2.25 PFLOPS / 8 TB/s）', 'H100 SXM（989 TFLOPS / 3.35 TB/s）'],
      explanation: md`约 153、206、281、295。H200 的拐点比 H100 **低**：算力相同、带宽更高，访存瓶颈的 decode 在 H200 上更快，也更早进入算力瓶颈。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把指标与定义配对：`,
      pairs: [
        ['TTFT', '请求到达 → 第一个 token 返回（含排队与 prefill）'],
        ['TPOT / ITL', '相邻两个输出 token 的时间间隔'],
        ['吞吐', '整个系统单位时间产出的 token 或请求数'],
        ['goodput', '同时满足 TTFT 与 TPOT SLO 时能承受的最大请求速率'],
      ],
      explanation: md`TTFT 由排队与 prefill 决定，TPOT 由 decode 单步时间决定；goodput 由 DistServe 提出，强调「满足 SLO 的吞吐」才有意义。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面估算 decode 吞吐上界的函数有 **2 处** 错误，点出来：`,
      code: py`
        def decode_tokens_per_s(n_params, batch, ctx, kv_per_token, peak_flops, bw, w_bytes=2):
            flops = 2 * n_params * batch
            bytes_ = n_params * w_bytes * batch + batch * ctx * kv_per_token
            t = min(flops / peak_flops, bytes_ / bw)
            return batch / t
      `,
      bugLines: [3, 4],
      explanation: md`第 3 行：权重每步只读一遍，被 batch 共享，应为 \`n_params * w_bytes + batch * ctx * kv_per_token\`。第 4 行：耗时下界是两者的 **max**（谁慢谁是瓶颈），取 min 会严重高估吞吐。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 3,
      prompt: md`Llama-3.1-70B（$d = 8192$、64 头、8 个 KV 头）在 H100 上 decode，**batch = 512**、上下文 4096。下列算子中哪些仍处于**访存瓶颈**？`,
      items: ['QKV 投影', '注意力 QKᵀ / AV', 'FFN', 'RMSNorm', '输出 softmax 与采样', 'prefill 时的 FFN（P = 4096）'],
      answer: [1, 3, 4],
      explanation: md`batch = 512 时线性层强度 $\approx 512 \times 8192 / (8192 + 1024) \approx 455 > 295$，进入算力瓶颈（QKV、FFN，以及 prefill 的 FFN）。注意力强度恒为 $64/8 = 8$；RMSNorm、softmax 这类逐元素算子每字节只做几次运算，始终受带宽限制。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 1,
      prompt: md`拼出 roofline 判定函数（返回强度、瓶颈类型和耗时下界）。有 2 行干扰项：`,
      lines: [
        'def roofline(flops, bytes_, peak, bw):',
        '    intensity = flops / bytes_',
        '    ridge = peak / bw',
        '    bound = "compute" if intensity >= ridge else "memory"',
        '    t = max(flops / peak, bytes_ / bw)',
        '    return intensity, bound, t',
      ],
      distractors: ['    t = flops / peak + bytes_ / bw', '    ridge = bw / peak'],
      explanation: md`强度 ≥ 拐点即算力瓶颈；耗时下界取 max（计算与访存可以重叠）。两个干扰项分别是把两段时间相加、把拐点写倒。`,
    },
  ],

  guided: [
    {
      id: 'g-decode-step',
      title: '估算一步 decode 的耗时与强度',
      intro: md`
        写一个函数，给定模型结构、batch、上下文长度和 GPU 规格，返回一步 decode 的耗时下界与算术强度。约定：
        - FLOPs $= 2NB + 4 L n_h d_h S B$（线性层 + 注意力的 $QK^\top$、$AV$）；
        - 字节 $= N w + B S \kappa$，$\kappa = 2 L n_{kv} d_h \cdot$ \`kv_bytes\`（读一遍权重 + 读全部 KV）。
      `,
      template: py`
        def decode_step(n_params, n_layers, n_heads, n_kv_heads, head_dim, batch, ctx,
                        peak_flops, bw, w_bytes=2, kv_bytes=2):
            kv_per_token = {{1}}
            flops = 2 * n_params * batch + {{2}}
            bytes_ = {{3}}
            t = {{4}}
            return t, flops / bytes_
      `,
      steps: [
        {
          blank: 1,
          instruction: md`每个 token 的 KV 字节数：K、V 两份 × 层数 × KV 头数 × 头维 × 字节。`,
          answer: '2 * n_layers * n_kv_heads * head_dim * kv_bytes',
          accept: ['n_layers * n_kv_heads * head_dim * kv_bytes * 2', '2 * n_layers * n_kv_heads * head_dim * kv_bytes'],
          hint: '用 KV 头数 n_kv_heads，不是 query 头数。',
          explain: md`Llama-3.1-8B：$2 \times 32 \times 8 \times 128 \times 2 = 131072$ 字节 = 128 KiB。用 \`n_heads\` 会把 GQA 的 KV 高估 4 倍。`,
        },
        {
          blank: 2,
          instruction: md`注意力的 FLOPs：每层每个 query 头对 $S$ 个位置做 $QK^\top$ 与 $AV$，各 $2 d_h S$。`,
          answer: '4 * n_layers * n_heads * head_dim * ctx * batch',
          explain: md`这里用 **query 头数**：GQA 并不减少注意力的计算量，只减少要读的 KV。`,
        },
        {
          blank: 3,
          instruction: md`本步要读的总字节：权重一遍 + 全部序列的 KV。`,
          answer: 'n_params * w_bytes + batch * ctx * kv_per_token',
          accept: ['n_params * w_bytes + kv_per_token * ctx * batch'],
          explain: md`权重不乘 batch——batch 内所有 token 共享同一次权重读取，这正是 batching 提高强度的原因。`,
        },
        {
          blank: 4,
          instruction: md`耗时下界。`,
          answer: 'max(flops / peak_flops, bytes_ / bw)',
          accept: ['max(bytes_ / bw, flops / peak_flops)'],
          explain: md`计算与访存可以重叠，瓶颈决定耗时。返回的强度与拐点 \`peak_flops / bw\` 比较即可判断瓶颈。`,
        },
      ],
      setup: py`
        import math
        _H100 = dict(peak_flops=989e12, bw=3.35e12)
        _L8 = dict(n_params=8.03e9, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128)
      `,
      tests: [
        {
          name: 'batch = 1：访存瓶颈',
          code: py`
            t, ai = decode_step(**_L8, batch=1, ctx=1024, **_H100)
            exp_t = (8.03e9 * 2 + 1024 * 131072) / 3.35e12
            assert math.isclose(t, exp_t, rel_tol=1e-9), f"期望 {exp_t * 1e3:.4f} ms，得到 {t * 1e3:.4f} ms"
            assert ai < 2, f"batch=1 的强度应约为 1，得到 {ai:.2f}"
          `,
        },
        {
          name: '大 batch 短上下文：算力瓶颈',
          code: py`
            t, ai = decode_step(**_L8, batch=1024, ctx=128, **_H100)
            flops = 2 * 8.03e9 * 1024 + 4 * 32 * 32 * 128 * 128 * 1024
            assert math.isclose(t, flops / 989e12, rel_tol=1e-9), f"应为算力瓶颈，期望 {flops / 989e12 * 1e3:.3f} ms，得到 {t * 1e3:.3f} ms"
            assert ai > 295, f"强度应超过拐点 295，得到 {ai:.1f}"
          `,
        },
        {
          name: 'KV 用 FP8、GQA 头数',
          code: py`
            t, ai = decode_step(n_params=7e10, n_layers=80, n_heads=64, n_kv_heads=8, head_dim=128,
                                batch=32, ctx=8192, peak_flops=989e12, bw=3.35e12, w_bytes=1, kv_bytes=1)
            kv = 2 * 80 * 8 * 128 * 1
            exp = (7e10 * 1 + 32 * 8192 * kv) / 3.35e12
            assert math.isclose(t, exp, rel_tol=1e-9), f"期望 {exp * 1e3:.3f} ms，得到 {t * 1e3:.3f} ms"
          `,
        },
      ],
      outro: md`
        这就是 RooflineViz 背后的全部公式。注意它给的是**上界**：真实系统还有 kernel 启动、通信、采样、调度开销，且很难跑满带宽。
        面试时先给出这个量级，再说明哪些因素会让实测更慢，比报一个「精确数字」更有说服力。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-decode-bound',
      title: 'decode 吞吐上界、拐点 batch 与 KV 容量',
      difficulty: 'medium',
      source: '大模型岗估算题 · Roofline',
      timeLimitMin: 20,
      description: md`
        实现

        \`\`\`python
        def decode_bound(cfg: dict, gpu: dict, batch: int, ctx: int) -> dict: ...
        \`\`\`

        - \`cfg\`：\`n_params, n_layers, n_heads, n_kv_heads, head_dim, w_bytes, kv_bytes\`
        - \`gpu\`：\`flops\`（FLOP/s）、\`bw\`（Byte/s）、\`mem\`（Byte）

        记 $\kappa = 2 L n_{kv} d_h \cdot$ \`kv_bytes\`，对 batch $B$、上下文 $S$：
        $\text{FLOPs}(B) = 2NB + 4Ln_hd_hSB$，$\text{Bytes}(B) = N w + BS\kappa$，$t_c = \text{FLOPs}/\text{flops}$，$t_m = \text{Bytes}/\text{bw}$。

        返回字典：
        - \`step_time\` $= \max(t_c, t_m)$；\`tokens_per_s\` $= B / $\`step_time\`；
        - \`bound\`：$t_c \ge t_m$ 为 \`"compute"\`，否则 \`"memory"\`；
        - \`max_batch\`：显存能容纳的最大 batch $= \lfloor(\text{mem} - Nw) / (S\kappa)\rfloor$，不足 0 时为 0；
        - \`knee_batch\`：使 $t_c(B) \ge t_m(B)$ 的**最小**正整数 $B$；若任何 $B$ 都达不到，返回 $-1$。

        **示例**：Llama-3.1-8B（\`n_params=8.03e9, L=32, n_h=32, n_kv=8, d_h=128\`，BF16），H100（\`989e12, 3.35e12, 80e9\`）
        - \`batch=1, ctx=128\` → \`bound="memory"\`、\`tokens_per_s≈208.4\`、\`max_batch=3811\`、\`knee_batch=425\`
        - \`batch=1, ctx=1024\` → \`tokens_per_s≈206.9\`、\`max_batch=476\`、\`knee_batch=-1\`（上下文才 1K，就再也到不了算力瓶颈）
      `,
      starter: py`
        import math

        def decode_bound(cfg, gpu, batch, ctx):
            # 1) FLOPs 与 Bytes 都是 B 的一次函数：flops = a*B，bytes = c + e*B
            # 2) knee：解 a*B/F >= (c + e*B)/BW
            raise NotImplementedError
      `,
      solution: py`
        import math

        def decode_bound(cfg, gpu, batch, ctx):
            N, L = cfg["n_params"], cfg["n_layers"]
            kv = 2 * L * cfg["n_kv_heads"] * cfg["head_dim"] * cfg["kv_bytes"]   # 每 token KV 字节
            a = 2 * N + 4 * L * cfg["n_heads"] * cfg["head_dim"] * ctx            # 每个序列每步的 FLOPs
            c = N * cfg["w_bytes"]                                                # 权重：每步读一遍，batch 共享
            e = ctx * kv                                                          # 每个序列每步读的 KV
            F, BW = gpu["flops"], gpu["bw"]

            tc, tm = a * batch / F, (c + e * batch) / BW
            step = max(tc, tm)
            max_batch = max(0, math.floor((gpu["mem"] - c) / e))

            # t_c(B) >= t_m(B)  <=>  B * (a/F - e/BW) >= c/BW
            slope = a / F - e / BW
            if slope <= 0:
                knee = -1                     # 强度上限 a/e 不超过拐点：永远是访存瓶颈
            else:
                knee = max(1, math.ceil(c / BW / slope))
                while knee > 1 and a * (knee - 1) / F >= (c + e * (knee - 1)) / BW:   # 浮点修正
                    knee -= 1
                while a * knee / F < (c + e * knee) / BW:
                    knee += 1
            return {
                "step_time": step,
                "tokens_per_s": batch / step,
                "bound": "compute" if tc >= tm else "memory",
                "max_batch": max_batch,
                "knee_batch": knee,
            }
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：用实测代替估算）',
          code: py`
            import torch, time

            @torch.inference_mode()
            def measure_decode(model, batch, ctx, n_iter=20):
                """用 CUDA Graph / 真实 KV cache 测一步 decode，与 roofline 上界对比得到「带宽利用率」"""
                ids = torch.randint(0, model.config.vocab_size, (batch, 1), device="cuda")
                cache = model.prepare_cache(batch, ctx)          # 伪代码：预填 ctx 长度的 KV
                for _ in range(3):
                    model(ids, past_key_values=cache)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(n_iter):
                    model(ids, past_key_values=cache)
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n_iter      # 实测 / 上界 = 达到的带宽比例
          `,
        },
      ],
      explanation: md`
        **线性化**：FLOPs 与字节都是 $B$ 的一次函数，$\text{FLOPs} = aB$，$\text{Bytes} = c + eB$。

        **拐点 batch**：$\frac{aB}{F} \ge \frac{c + eB}{BW} \iff B\left(\frac{a}{F} - \frac{e}{BW}\right) \ge \frac{c}{BW}$。
        斜率 $\le 0$ 等价于 $a/e \le F/BW$，即「每个序列自身的强度上限不超过拐点」——长上下文时常见，此时加多少 batch 都摆脱不了访存瓶颈。

        **数值**：Llama-3.1-8B、H100，$e = S\kappa$ 中 $\kappa = 131072$。$S = 128$ 时 $a/e \approx 961 > 295$，拐点 $= \frac{c/BW}{a/F - e/BW} \approx 425$；
        $S = 1024$ 时 $a/e \approx 124 < 295$——**上下文只要超过约 420 个 token，这个模型的 decode 在 H100 上就永远是访存瓶颈**，batch 再大也只是摊薄权重读取，KV 读取越来越主导。

        **易错点**：权重字节不乘 batch；注意力 FLOPs 用 query 头数、KV 字节用 KV 头数；拐点要用整数并处理浮点边界。
      `,
      complexity: { time: 'O(1)', space: 'O(1)' },
      hints: [
        md`令 \`a = 2N + 4·L·n_h·d_h·S\`、\`c = N·w\`、\`e = S·κ\`，则 \`flops = a·B\`，\`bytes = c + e·B\`。`,
        md`\`knee\`：斜率 \`a/F - e/BW\` ≤ 0 时返回 -1；否则 \`ceil((c/BW) / 斜率)\`，再用原不等式检查一下边界。`,
        md`\`max_batch = floor((mem - c) / e)\`，别忘了与 0 取 max。`,
      ],
      setup: py`
        import math, random

        _L8 = dict(n_params=8.03e9, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, w_bytes=2, kv_bytes=2)
        _H100 = dict(flops=989e12, bw=3.35e12, mem=80e9)

        def _ref(cfg, gpu, B, S):
            kv = 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * cfg["kv_bytes"]
            a = 2 * cfg["n_params"] + 4 * cfg["n_layers"] * cfg["n_heads"] * cfg["head_dim"] * S
            c = cfg["n_params"] * cfg["w_bytes"]; e = S * kv
            tc, tm = a * B / gpu["flops"], (c + e * B) / gpu["bw"]
            return a, c, e, tc, tm

        def _ok_knee(cfg, gpu, S, k):
            a, c, e, _, _ = _ref(cfg, gpu, 1, S)
            f = lambda B: a * B / gpu["flops"] >= (c + e * B) / gpu["bw"]
            if a / gpu["flops"] - e / gpu["bw"] <= 0:
                return k == -1
            return k >= 1 and f(k) and (k == 1 or not f(k - 1))

        def _check(got, cfg, gpu, B, S, tag):
            for key in ("step_time", "tokens_per_s", "bound", "max_batch", "knee_batch"):
                assert key in got, f"{tag}: 缺少键 {key}"
            a, c, e, tc, tm = _ref(cfg, gpu, B, S)
            st = max(tc, tm)
            assert math.isclose(got["step_time"], st, rel_tol=1e-9), f"{tag}: step_time 期望 {st:.6e}，得到 {got['step_time']}"
            assert math.isclose(got["tokens_per_s"], B / st, rel_tol=1e-9), f"{tag}: tokens_per_s 期望 {B / st:.4f}，得到 {got['tokens_per_s']}"
            exp_b = "compute" if tc >= tm else "memory"
            assert got["bound"] == exp_b, f"{tag}: bound 期望 {exp_b}，得到 {got['bound']}"
            mb = max(0, math.floor((gpu["mem"] - c) / e))
            assert got["max_batch"] == mb, f"{tag}: max_batch 期望 {mb}，得到 {got['max_batch']}"
            assert _ok_knee(cfg, gpu, S, got["knee_batch"]), f"{tag}: knee_batch={got['knee_batch']} 不满足「最小的使 t_c ≥ t_m 的 B」"
      `,
      tests: [
        {
          name: '示例：Llama-3.1-8B / H100',
          code: py`
            got = decode_bound(_L8, _H100, 1, 128)
            assert got["bound"] == "memory", f"batch=1 应为访存瓶颈，得到 {got['bound']}"
            assert abs(got["tokens_per_s"] - 208.4) < 0.1, f"tokens_per_s 期望约 208.4，得到 {got['tokens_per_s']:.2f}"
            assert got["max_batch"] == 3811, f"max_batch 期望 3811，得到 {got['max_batch']}"
            assert got["knee_batch"] == 425, f"knee_batch 期望 425，得到 {got['knee_batch']}"
          `,
        },
        {
          name: 'ctx = 1024：永远到不了算力瓶颈',
          code: py`
            got = decode_bound(_L8, _H100, 16, 1024)
            assert got["knee_batch"] == -1, f"ctx=1024 时每序列强度上限约 124 < 295，应返回 -1，得到 {got['knee_batch']}"
            assert got["bound"] == "memory"
            assert got["max_batch"] == 476, f"max_batch 期望 476，得到 {got['max_batch']}"
          `,
        },
        {
          name: '权重放不下：max_batch = 0',
          code: py`
            cfg = dict(n_params=7.06e10, n_layers=80, n_heads=64, n_kv_heads=8, head_dim=128, w_bytes=2, kv_bytes=2)
            got = decode_bound(cfg, _H100, 1, 4096)
            assert got["max_batch"] == 0, f"70B BF16 权重 141 GB > 80 GB，max_batch 应为 0，得到 {got['max_batch']}"
          `,
        },
        {
          name: '完整字段校验（多组配置）',
          hidden: true,
          code: py`
            cases = [(_L8, _H100, 1, 1024), (_L8, _H100, 512, 512), (_L8, dict(flops=312e12, bw=2.039e12, mem=80e9), 64, 2048),
                     (dict(_L8, w_bytes=0.5), _H100, 8, 4096), (dict(_L8, kv_bytes=1), dict(flops=2250e12, bw=8e12, mem=180e9), 128, 1024)]
            for i, (cfg, gpu, B, S) in enumerate(cases):
                _check(decode_bound(cfg, gpu, B, S), cfg, gpu, B, S, f"case {i}")
          `,
        },
        {
          name: '随机配置对拍',
          hidden: true,
          code: py`
            rnd = random.Random(9)
            for i in range(300):
                L = rnd.choice([16, 32, 48, 80]); nkv = rnd.choice([1, 2, 4, 8]); nh = nkv * rnd.choice([1, 2, 4, 8])
                cfg = dict(n_params=rnd.uniform(5e8, 8e10), n_layers=L, n_heads=nh, n_kv_heads=nkv, head_dim=rnd.choice([64, 128]),
                           w_bytes=rnd.choice([0.5, 1, 2]), kv_bytes=rnd.choice([1, 2]))
                gpu = dict(flops=rnd.uniform(1e14, 3e15), bw=rnd.uniform(1e12, 1e13), mem=rnd.uniform(2e10, 2e11))
                B, S = rnd.randint(1, 2048), rnd.choice([128, 512, 2048, 8192, 32768])
                _check(decode_bound(cfg, gpu, B, S), cfg, gpu, B, S, f"random {i}")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-tp',
          q: md`换成 8 卡张量并行，这些结论怎么变？`,
          a: md`每卡的 FLOPs 和字节都约为 1/8，**强度不变**、拐点 batch 不变，单步时间约为 1/8，但每层多了两次 all-reduce（与 batch 成正比的小消息，常受延迟限制）。当 TP 度超过 KV 头数时，KV 头要在卡间复制，KV 字节不再按 TP 度缩小。`,
        },
      ],
    },
    {
      id: 'c-layer-roofline',
      title: '逐算子 Roofline 判定：一个 decoder 层',
      difficulty: 'medium',
      source: '大模型岗手撕 · 性能分析',
      timeLimitMin: 20,
      description: md`
        对一个 Llama 式 decoder 层的 decode 步（batch $B$，上下文 $S$），逐算子计算算术强度、瓶颈和耗时下界：

        \`\`\`python
        def layer_roofline(cfg: dict, batch: int, ctx: int, peak: float, bw: float) -> dict[str, tuple[float, str, float]]: ...
        \`\`\`

        \`cfg\`：\`d\`（hidden）、\`n_heads\`、\`n_kv_heads\`、\`head_dim\`、\`ffn\`（SwiGLU 中间维）、\`b\`（每元素字节）。

        GEMM $[M \times K] \cdot [K \times N]$ 的 FLOPs $= 2MKN$，字节 $= b(MK + KN + MN)$。算子定义：

        | 名字 | 计算 |
        |---|---|
        | \`"qkv"\` | GEMM，$M = B$，$K = d$，$N = (n_h + 2n_{kv})\,d_h$ |
        | \`"attn"\` | FLOPs $= 4 B n_h d_h S$，字节 $= 2 B S n_{kv} d_h \cdot b$ |
        | \`"o"\` | GEMM，$M = B$，$K = n_h d_h$，$N = d$ |
        | \`"ffn_up"\` | GEMM，$M = B$，$K = d$，$N = 2\cdot$\`ffn\`（gate 与 up 合并） |
        | \`"ffn_down"\` | GEMM，$M = B$，$K =$ \`ffn\`，$N = d$ |

        返回 \`{名字: (强度, "compute" 或 "memory", 耗时下界)}\`，强度 ≥ \`peak / bw\` 为 \`"compute"\`；耗时下界 $= \max(\text{FLOPs}/\text{peak}, \text{字节}/\text{bw})$。

        **示例**：Llama-3.1-8B 一层（\`d=4096, n_heads=32, n_kv_heads=8, head_dim=128, ffn=14336, b=2\`），H100，\`batch=1, ctx=2048\` → 全部为 \`"memory"\`，\`"attn"\` 的强度为 4.0。
      `,
      starter: py`
        def layer_roofline(cfg, batch, ctx, peak, bw):
            # 先写一个 gemm(M, K, N) -> (flops, bytes)，再逐个算子套用
            raise NotImplementedError
      `,
      solution: py`
        def layer_roofline(cfg, batch, ctx, peak, bw):
            d, nh, nkv, dh, f, b = cfg["d"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["ffn"], cfg["b"]
            B, S = batch, ctx

            def gemm(M, K, N):
                return 2 * M * K * N, b * (M * K + K * N + M * N)

            ops = {
                "qkv": gemm(B, d, (nh + 2 * nkv) * dh),
                "attn": (4 * B * nh * dh * S, 2 * B * S * nkv * dh * b),   # 只算读 KV
                "o": gemm(B, nh * dh, d),
                "ffn_up": gemm(B, d, 2 * f),
                "ffn_down": gemm(B, f, d),
            }
            ridge = peak / bw
            out = {}
            for name, (fl, by) in ops.items():
                ai = fl / by
                out[name] = (ai, "compute" if ai >= ridge else "memory", max(fl / peak, by / bw))
            return out
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：用 profiler 拿到真实的 FLOPs / 耗时）',
          code: py`
            import torch
            from torch.profiler import profile, ProfilerActivity

            with profile(activities=[ProfilerActivity.CUDA], with_flops=True) as prof:
                with torch.inference_mode():
                    layer(hidden_states, past_key_values=cache)     # 一步 decode
            for evt in prof.key_averages():
                if evt.flops:
                    tflops = evt.flops / (evt.device_time_total * 1e-6) / 1e12
                    print(f"{evt.key:40s} {tflops:8.1f} TFLOP/s")   # 与 roofline 上界比较
          `,
        },
      ],
      explanation: md`
        **思路**：把每个算子化成（FLOPs, 字节），与拐点比较即可。GEMM 的强度 $\frac{2MKN}{b(MK + KN + MN)}$ 在 $M$ 小时 $\approx 2M/b$。

        **结论（H100、Llama-3.1-8B）**：
        - batch = 1：所有算子都是访存瓶颈，时间几乎全花在读 FFN 和 QKV / O 的权重上；
        - batch = 512、短上下文：四个 GEMM 进入算力瓶颈，注意力仍是访存瓶颈（强度恒为 $n_h/n_{kv} = 4$），上下文一长它就成为一层中最慢的算子。

        **易错点**：注意力的 FLOPs 用 query 头数、字节用 KV 头数；QKV 投影的输出维是 $(n_h + 2n_{kv})d_h$ 而不是 $3d$（GQA）。
      `,
      complexity: { time: 'O(1)', space: 'O(1)' },
      hints: [
        md`辅助函数 \`gemm(M, K, N)\` 返回 \`(2*M*K*N, b*(M*K + K*N + M*N))\`。`,
        md`\`attn\` 直接按表中公式：FLOPs \`4*B*nh*dh*S\`，字节 \`2*B*S*nkv*dh*b\`。`,
        md`强度 ≥ \`peak / bw\` 为 \`"compute"\`；耗时 \`max(flops/peak, bytes/bw)\`。`,
      ],
      setup: py`
        import math, random
        _C8 = dict(d=4096, n_heads=32, n_kv_heads=8, head_dim=128, ffn=14336, b=2)
        _NAMES = ("qkv", "attn", "o", "ffn_up", "ffn_down")

        def _ref_layer(cfg, B, S, peak, bw):
            d, nh, nkv, dh, f, b = cfg["d"], cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"], cfg["ffn"], cfg["b"]
            g = lambda M, K, N: (2 * M * K * N, b * (M * K + K * N + M * N))
            ops = {"qkv": g(B, d, (nh + 2 * nkv) * dh), "attn": (4 * B * nh * dh * S, 2 * B * S * nkv * dh * b),
                   "o": g(B, nh * dh, d), "ffn_up": g(B, d, 2 * f), "ffn_down": g(B, f, d)}
            return {k: (fl / by, "compute" if fl / by >= peak / bw else "memory", max(fl / peak, by / bw)) for k, (fl, by) in ops.items()}

        def _cmp(got, exp, tag):
            assert set(got) == set(_NAMES), f"{tag}: 键应为 {_NAMES}，得到 {sorted(got)}"
            for k in _NAMES:
                ga, gb, gt = got[k]; ea, eb, et = exp[k]
                assert math.isclose(ga, ea, rel_tol=1e-9), f"{tag}/{k}: 强度期望 {ea:.4f}，得到 {ga}"
                assert gb == eb, f"{tag}/{k}: 瓶颈期望 {eb}，得到 {gb}"
                assert math.isclose(gt, et, rel_tol=1e-9), f"{tag}/{k}: 耗时期望 {et:.4e}，得到 {gt}"
      `,
      tests: [
        {
          name: '示例：batch = 1 全是访存瓶颈',
          code: py`
            got = layer_roofline(_C8, 1, 2048, 989e12, 3.35e12)
            assert all(v[1] == "memory" for v in got.values()), f"batch=1 应全部为 memory：{ {k: v[1] for k, v in got.items()} }"
            assert math.isclose(got["attn"][0], 4.0), f"attn 强度应为 n_h/n_kv = 4，得到 {got['attn'][0]}"
          `,
        },
        {
          name: 'batch = 512：GEMM 进入算力瓶颈，注意力不变',
          code: py`
            got = layer_roofline(_C8, 512, 1024, 989e12, 3.35e12)
            for k in ("qkv", "o", "ffn_up", "ffn_down"):
                assert got[k][1] == "compute", f"{k} 强度 {got[k][0]:.1f} 应超过拐点 295"
            assert got["attn"][1] == "memory" and math.isclose(got["attn"][0], 4.0)
          `,
        },
        {
          name: '数值与示例一致',
          code: py`
            _cmp(layer_roofline(_C8, 1, 2048, 989e12, 3.35e12), _ref_layer(_C8, 1, 2048, 989e12, 3.35e12), "8B B=1")
          `,
        },
        {
          name: '随机配置对拍',
          hidden: true,
          code: py`
            rnd = random.Random(4)
            for i in range(300):
                nkv = rnd.choice([1, 2, 4, 8]); nh = nkv * rnd.choice([1, 2, 4, 8, 16]); dh = rnd.choice([64, 128])
                cfg = dict(d=rnd.choice([1024, 2048, 4096, 8192]), n_heads=nh, n_kv_heads=nkv, head_dim=dh,
                           ffn=rnd.choice([2816, 5632, 11008, 14336, 28672]), b=rnd.choice([0.5, 1, 2]))
                B, S = rnd.randint(1, 4096), rnd.choice([1, 128, 4096, 32768])
                peak, bw = rnd.uniform(1e14, 3e15), rnd.uniform(1e12, 1e13)
                _cmp(layer_roofline(cfg, B, S, peak, bw), _ref_layer(cfg, B, S, peak, bw), f"random {i}")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-fuse',
          q: md`RMSNorm、SiLU、残差加这些逐元素算子没出现在表里，它们重要吗？`,
          a: md`它们的 FLOPs 可以忽略，但每个都要把 $[B \times d]$ 的激活读写一遍，强度只有个位数，是纯访存算子。decode 时激活很小，开销主要是 **kernel 启动**；prefill 时激活很大，读写开销显著。所以推理框架会做算子融合（如 RMSNorm 融进 GEMM 前后、SiLU·乘法融合）并用 CUDA Graph 消除 launch 开销。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-mem-compute',
      freq: 3,
      q: md`为什么 decode 是访存瓶颈、prefill 是算力瓶颈？`,
      a: md`
        **结论**：看算术强度（FLOPs / 字节）与 GPU 拐点（峰值算力 / 带宽，H100 BF16 约 295）的关系。

        **原因**：线性层的强度约为「同一份权重服务的 token 数」。prefill 一次处理 $P$ 个 token（数百到数千），强度远超拐点；decode 每个序列每步只有 1 个 token，batch 为 $B$ 时强度约为 $B$，batch 小时远低于拐点，每步时间 ≈ 权重字节 / 带宽。注意力部分：decode 时每个序列读自己的 KV，强度恒为 $n_h/n_{kv}$；prefill 时 $P$ 个 query 共享 KV，强度随 $P$ 增长。

        **延伸**：所以 decode 优化是「少读字节或每字节多干活」：batching、权重量化、GQA/MLA/KV 量化、投机解码；prefill 优化是提高算力利用率：FlashAttention、FP8 GEMM、chunked prefill 与 decode 搭配。
      `,
    },
    {
      id: 'qa-estimate',
      freq: 3,
      q: md`估算一下 70B 模型在 8×H100 上的 decode 速度上限。`,
      a: md`
        **结论**：BF16 下约 5 ms / token（单序列约 190 token/s），这是上界。

        **过程**：权重 $70 \times 10^9 \times 2 = 140$ GB；TP8 时每卡读 17.5 GB，H100 带宽 3.35 TB/s，$17.5 / 3350 \approx 5.2$ ms。batch 增大时只要仍在访存瓶颈，单步时间基本不变，吞吐 ≈ $B / 5.2\text{ ms}$；KV 读取会随 $B \cdot S$ 增加单步时间。

        **延伸**：实际还要加上每层 2 次 all-reduce（80 层共 160 次小消息通信，受延迟限制，累积起来不可忽略）、kernel 效率（一般达不到峰值带宽）、采样与调度开销。FP8 权重减半，单步上界约 2.6 ms。
      `,
    },
    {
      id: 'qa-ttft-tpot',
      freq: 2,
      q: md`TTFT 和 TPOT 分别由什么决定？为什么它们会互相干扰？`,
      a: md`
        **结论**：TTFT ≈ 排队时间 + prefill 时间（算力瓶颈，∝ prompt 长度）；TPOT ≈ decode 单步时间（访存瓶颈，∝ 权重 + KV 字节）。

        **干扰**：连续批处理把新请求的 prefill 和老请求的 decode 放进同一次迭代，一个长 prompt 会让这次迭代变慢几十倍，正在 decode 的请求都出现 TPOT 尖峰；反过来，若优先 decode、推迟 prefill，TTFT 变差。

        **解决**：chunked prefill（把长 prefill 切块，与 decode 混合，控制每次迭代的 token 预算）、PD 分离（prefill 与 decode 放到不同 GPU 池，各自按 SLO 调并行策略），见下一关。
      `,
    },
    {
      id: 'qa-moe-decode',
      freq: 2,
      q: md`MoE 模型的 decode 是否也是访存瓶颈？和稠密模型有什么不同？`,
      a: md`
        **结论**：通常更严重。计算量按激活参数算，但访存要读本步被选中的所有专家。

        **原因**：batch 小时只读少量专家，访存量接近激活参数；batch 增大后，被选中的专家迅速接近全部（DeepSeek-V3 每层 256 选 8，batch = 64 时期望约 222 个专家被选中），而每个专家只分到几个 token——总读取量接近全部参数、强度却很低。

        **延伸**：缓解办法是大规模专家并行（每卡放少量专家、让每个专家攒更多 token）、低比特专家权重、把 attention 与 MoE 分开部署；这也是 DeepSeek 推理部署使用大 EP 的原因。
      `,
    },
  ],

  cheatsheet: md`
    **Roofline**：$\text{Perf} = \min(\pi, \beta I)$，拐点 $I^* = \pi/\beta$；耗时 $t \ge \max(\text{FLOPs}/\pi, \text{Bytes}/\beta)$

    | GPU（BF16 稠密） | TFLOPS | TB/s | GB | 拐点 |
    |---|---|---|---|---|
    | A100 80GB SXM | 312 | 2.039 | 80 | 153 |
    | H100 SXM | 989 | 3.35 | 80 | 295 |
    | H200 SXM | 989 | 4.8 | 141 | 206 |
    | B200 | 2250 | 8 | 180 | 281 |

    - 线性层强度 $\frac{Td}{d+2T}$（BF16）≈ token 数；decode ≈ batch，prefill ≈ prompt 长度
    - decode 注意力强度 $= n_h/n_{kv}$，与 batch 无关
    - decode 一步：FLOPs $= (2N + 4Ln_hd_hS)B$，Bytes $= Nw + BS\kappa$，$\kappa = 2Ln_{kv}d_h b$
    - batch=1 上界：tokens/s ≈ 带宽 / 权重字节（8B BF16 @ H100 ≈ 208）
    - prefill：FLOPs ≈ $2NP + 2Ln_hd_hP^2$
    - 端到端 ≈ TTFT + (n−1)·TPOT；goodput = 满足 SLO 的最大请求率
    - 易错：官网算力是稀疏值（÷2）；权重字节不乘 batch；取 max 不取和
  `,
})
