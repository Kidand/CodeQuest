/**
 * L3-4 MLA 多头潜在注意力（DeepSeek-V2/V3）：低秩 KV 联合压缩、decoupled RoPE、推理时的矩阵吸收。
 * 参数核对来源：DeepSeek-V2（arXiv 2405.04434）§2.1、附录 D；DeepSeek-V3（arXiv 2412.19437）§2.1.1、§4.2。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { MLAFlowViz } from '../viz/MLAFlowViz'
import { AbsorbRopeViz } from '../viz/AbsorbRopeViz'
import { MLAIntensityViz } from '../viz/MLAIntensityViz'

const ROPE = py`
  def rope(x, pos, base=10000.0):
      """rotate_half 风格 RoPE。x: (..., T, d)，d 为偶数；pos: 长度 T 的绝对位置"""
      d = x.shape[-1]
      inv = base ** (-np.arange(0, d, 2) / d)                    # (d/2,)
      ang = np.asarray(pos, dtype=float)[:, None] * inv[None, :]  # (T, d/2)
      cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)
      sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)
      x1, x2 = x[..., : d // 2], x[..., d // 2:]
      return x * cos + np.concatenate([-x2, x1], -1) * sin
`

const SETUP = py`
  import numpy as np

  def _rope(x, pos, base=10000.0):
      d = x.shape[-1]
      inv = base ** (-np.arange(0, d, 2) / d)
      ang = np.asarray(pos, dtype=float)[:, None] * inv[None, :]
      cos = np.concatenate([np.cos(ang)] * 2, -1); sin = np.concatenate([np.sin(ang)] * 2, -1)
      return x * cos + np.concatenate([-x[..., d // 2:], x[..., : d // 2]], -1) * sin

  def _params(seed, D, H, dh, dhr, dc, dcq):
      r = np.random.default_rng(seed)
      g = lambda a, b: r.normal(size=(a, b)) / np.sqrt(a)
      return {"W_dq": g(D, dcq), "W_uq": g(dcq, H * dh), "W_qr": g(dcq, H * dhr),
              "W_dkv": g(D, dc), "W_uk": g(dc, H * dh), "W_uv": g(dc, H * dh),
              "W_kr": g(D, dhr), "W_o": g(H * dh, D)}

  def _ref_mla(h, P, H, dh, dhr):
      """逐头、逐位置的展开形式参考实现（因果）"""
      B, T, D = h.shape
      out = np.zeros((B, T, D))
      ckv = h @ P["W_dkv"]
      kr = _rope(h @ P["W_kr"], np.arange(T))                        # (B, T, dhr)
      for b in range(B):
          cq = h[b] @ P["W_dq"]
          heads = []
          for i in range(H):
              qc = cq @ P["W_uq"][:, i * dh:(i + 1) * dh]
              qr = _rope(cq @ P["W_qr"][:, i * dhr:(i + 1) * dhr], np.arange(T))
              kc = ckv[b] @ P["W_uk"][:, i * dh:(i + 1) * dh]
              v = ckv[b] @ P["W_uv"][:, i * dh:(i + 1) * dh]
              s = (qc @ kc.T + qr @ kr[b].T) / np.sqrt(dh + dhr)
              s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
              p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
              heads.append(p @ v)
          out[b] = np.concatenate(heads, -1) @ P["W_o"]
      return out, ckv, kr
`

export default defineLevel({
  learn: [
    {
      id: 'compress',
      title: '低秩 KV 联合压缩：只缓存一个 512 维潜向量',
      body: md`
        GQA/MQA 靠**减少 KV 头**省显存，代价是表达能力。DeepSeek-V2 提出的 **MLA**（Multi-head Latent Attention）换了个思路：
        **不减头数，而是把 K 和 V 联合压缩成一个低维潜向量**，只缓存它（论文 Eq. 9–11，下文用行向量约定 $xW$）：

        $$
        c^{KV}_t = h_t W^{DKV},\qquad k^C_t = c^{KV}_t W^{UK},\qquad v^C_t = c^{KV}_t W^{UV}
        $$

        $W^{DKV}$ 是下投影（down），$W^{UK}, W^{UV}$ 是上投影（up）。query 侧也做了低秩压缩 $c^Q_t = h_t W^{DQ}$、$q^C_t = c^Q_t W^{UQ}$，
        但那只是为了减少**训练时的激活显存**，与 KV cache 无关。

        | 超参数 | DeepSeek-V2 | DeepSeek-V3 |
        |---|---|---|
        | 层数 / 隐藏维 $d$ | 60 / 5120 | 61 / 7168 |
        | 头数 $n_h$ / 每头维 $d_h$ | 128 / 128 | 128 / 128 |
        | KV 压缩维 $d_c$ | 512 | 512 |
        | Query 压缩维 $d_c'$ | 1536 | 1536 |
        | 解耦 RoPE 维 $d_h^R$ | 64 | 64 |
        | 总参数 / 激活参数 | 236B / 21B | 671B / 37B |

        V3 还在两个压缩潜向量之后加了 RMSNorm。下图切换「展开 / 吸收」两种计算形式，点击节点查看说明。
      `,
      viz: MLAFlowViz,
      after: md`
        > [!NOTE] 只是低秩分解吗？
        > 从训练角度看，$W^{DKV} W^{UK}$ 就是对 $W_K$ 的低秩分解（秩 ≤ 512，而 MHA 中 $W_K$ 是 $7168 \times 16384$）。关键在于**推理时可以永远不展开**：
        > 缓存 $c^{KV}$，把 $W^{UK}$、$W^{UV}$ 分别「吸收」到 query 侧和输出侧（第 3 节）。V2 论文附录 D.2 的对比实验中，MLA 在多数基准上优于 MHA，
        > 而 KV cache 只有后者的 14%（小 MoE）和 4%（大 MoE）。
      `,
      checkpoint: {
        id: 'cp-compress',
        type: 'multi',
        prompt: md`MLA 在 decode 时，每层需要为每个 token 缓存哪些量？（多选）`,
        options: ['$c^{KV}_t$（512 维）', '$k^R_t$（64 维，所有头共享）', '每个头的 $k^C_{t,i}$', '每个头的 $v^C_{t,i}$', '$c^Q_t$（1536 维）'],
        answer: [0, 1],
        explanation: md`V3 论文原文：只有 $c^{KV}_t$ 和 $k^R_t$ 需要在生成时缓存。$k^C, v^C$ 可以由 $c^{KV}$ 重新展开（或被吸收、根本不展开）；$c^Q$ 只用于当前 token 的 query，与 MHA 不缓存 Q 同理。`,
      },
    },
    {
      id: 'rope',
      title: '为什么必须 decouple RoPE',
      body: md`
        「吸收」的前提是内容打分里只有位置无关的矩阵：

        $$
        q^C_{t,i}\cdot k^C_{j,i} = c^Q_t\, W^{UQ}_i\, {W^{UK}_i}^{\top}\, {c^{KV}_j}^{\top}
        $$

        中间的 $W^{UQ}_i {W^{UK}_i}^\top$ 是固定矩阵，可以预先合并（或先算 $q^C$ 再乘 $\,{W^{UK}_i}^\top$），于是 query 直接与缓存的 $c^{KV}_j$ 点积。

        如果像普通模型那样对 $k^C$ 施加 RoPE（旋转矩阵 $R_j$）：

        $$
        (q^C_{t,i} R_t)\cdot(k^C_{j,i} R_j) = c^Q_t\, W^{UQ}_i\, \underbrace{R_t R_j^{\top}}_{\text{依赖 } t-j}\, {W^{UK}_i}^{\top}\, {c^{KV}_j}^{\top}
        $$

        位置相关的旋转**夹在** $W^{UQ}$ 与 $W^{UK}$ 之间，矩阵乘法不可交换，$W^{UK}$ 无法再被吸收——decode 时只能为所有历史 token 重新展开 $k^C$，压缩就白做了。

        **解法（decoupled RoPE）**：额外用一小部分维度专门承载位置信息——每个头一个 $q^R_{t,i} \in \mathbb{R}^{64}$，所有头**共享**一个 $k^R_t = \operatorname{RoPE}(h_t W^{KR}) \in \mathbb{R}^{64}$：

        $$
        q_{t,i} = [q^C_{t,i};\, q^R_{t,i}],\quad k_{j,i} = [k^C_{j,i};\, k^R_j],\quad
        s_{t,j,i} = \frac{q^C_{t,i}\cdot k^C_{j,i} + q^R_{t,i}\cdot k^R_j}{\sqrt{d_h + d_h^R}}
        $$

        内容部分照常吸收，位置部分用缓存的 $k^R$ 直接算。代价：每 token 多缓存 64 维，且缩放因子变为 $1/\sqrt{128+64} = 1/\sqrt{192}$。
      `,
      viz: AbsorbRopeViz,
      after: md`
        > [!INTERVIEW] 一句话回答
        > 「RoPE 的旋转矩阵依赖位置，会夹在 $W^{UQ}$ 和 $W^{UK}$ 之间，导致 $W^{UK}$ 无法吸收进 query 侧，推理时必须恢复全部历史 key；
        > 所以 MLA 把位置信息拆到额外的 64 维 $q^R / k^R$ 上，$k^R$ 所有头共享并缓存，内容部分不加 RoPE 从而可以吸收。」

        > [!WARN] $k^R$ 来自 $h_t$，不是来自 $c^{KV}$
        > 论文中 $k^R_t = \operatorname{RoPE}(W^{KR} h_t)$，$q^R_t = \operatorname{RoPE}(W^{QR} c^Q_t)$。实现时常把 $W^{DKV}$ 与 $W^{KR}$ 合并成一个 $d \times 576$ 的矩阵一次算出（HF 实现中的 \`kv_a_proj_with_mqa\`）。
      `,
      checkpoint: {
        id: 'cp-rope',
        type: 'single',
        prompt: md`DeepSeek-V3 的 MLA 中，softmax 之前的缩放因子是？`,
        options: ['$1/\\sqrt{128}$', '$1/\\sqrt{192}$', '$1/\\sqrt{512}$', '$1/\\sqrt{576}$'],
        answer: 1,
        explanation: md`q·k 的点积维度是 $d_h + d_h^R = 128 + 64 = 192$（V3 论文 Eq. 10）。吸收后虽然内容部分的点积在 512 维潜空间里做，但数值与原形式相同，缩放必须仍用 $\sqrt{192}$——用 $\sqrt{512}$ 或 $\sqrt{576}$ 是典型 bug。`,
      },
    },
    {
      id: 'absorb',
      title: '矩阵吸收：decode 时 MLA 就是一个「宽 MQA」',
      body: md`
        **吸收 $W^{UK}$**（进 query 侧）：令 $\tilde q_{t,i} = q^C_{t,i}\,{W^{UK}_i}^{\top} \in \mathbb{R}^{512}$，则 $q^C_{t,i}\cdot k^C_{j,i} = \tilde q_{t,i}\cdot c^{KV}_j$。

        **吸收 $W^{UV}$**（进输出侧）：用 MHA 一关的分块求和视角，

        $$
        u_t = \sum_{i} o_{t,i} W^O_i = \sum_i \Big(\sum_j p_{t,j,i}\, c^{KV}_j\Big)\, W^{UV}_i\, W^O_i
        $$

        先在 512 维潜空间里加权求和，再经 $W^{UV}_i W^O_i$ 映射到输出。

        于是 decode 时每个头的 key 是 $[c^{KV}_j; k^R_j]$（576 维）、value 是 $c^{KV}_j$（512 维），**128 个 query 头共享这同一份 KV**——
        这就是一个 head_dim 为 576/512 的 MQA。缓存每 token 每层 576 个元素，相当于 V2 论文说的「只有 2.25 组的 GQA」（$2 \times 2.25 \times 128 = 576$）。

        **但吸收并不总是更快**：每个 (query, key) 对，展开形式每头只需 $192 + 128$ 维的乘加，吸收形式要 $576 + 512$ 维。
        prefill 时 query 与 key 数量相当、计算密集，展开更省；decode 时每步只有 1 个 query、主要开销是读缓存，吸收更优。
        vLLM 的 MLA 实现正是按这个比例选择两条路径（prefill 走 compute-friendly 的展开形式，decode 走 data-movement-friendly 的吸收形式）。
      `,
      viz: MLAIntensityViz,
      after: md`
        > [!TIP] 两种「吸收」写法
        > 1. **预乘**：$W^{UQ}_i {W^{UK}_i}^\top$（$1536\times512$）与 $W^{UV}_i W^O_i$（$512\times7168$）——128 个头合计参数会膨胀到原来的数倍，一般不这么做；
        > 2. **改变结合顺序**：先算 $q^C$，再乘 $\,{W^{UK}_i}^\top$；注意力后先乘 $W^{UV}_i$ 再乘 $W^O$。不增加参数，是工程实现的常用做法。
        > 两者数值完全等价（矩阵乘法结合律），实战题会让你验证这一点。

        > [!NEW] MLA 的扩散
        > Kimi K2 / K2.5、GLM-5 等模型采用 MLA；Kimi K3 的每个 block 由 3 层 KDA 与 1 层 Gated MLA 组成；DeepSeek-V3.2 的 DSA 稀疏注意力也构建在 MLA 之上（以 MQA 模式运行）。
        > DeepSeek 开源的 FlashMLA 是面向 decode 的 MLA kernel。部署上，V2 论文还把 KV cache 量化到平均约 6 bit。
      `,
      checkpoint: {
        id: 'cp-absorb',
        type: 'numeric',
        prompt: md`DeepSeek-V3 全部 61 层，BF16 存储 MLA 缓存，**每个 token** 的 KV cache 是多少字节？`,
        answer: 70272,
        tolerance: 0,
        unit: '字节',
        explanation: md`$61 \times (512 + 64) \times 2 = 70{,}272$ 字节 ≈ 68.6 KiB。同头数（128 × 128）的 MHA 为 $61 \times 2 \times 128 \times 128 \times 2 = 3{,}997{,}696$ 字节 ≈ 3.8 MiB，约 56.9 倍。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-576',
      type: 'numeric',
      difficulty: 1,
      prompt: md`DeepSeek-V2/V3 的 MLA 中，每层每个 token 需要缓存多少个元素？`,
      answer: 576,
      tolerance: 0,
      explanation: md`$d_c + d_h^R = 512 + 64 = 576$。不需要乘 2（K、V 共用一个潜向量），也不需要乘头数（$k^R$ 所有头共享）。`,
    },
    {
      id: 'q-ratio',
      type: 'numeric',
      difficulty: 2,
      prompt: md`与相同头配置的 MHA（$n_h = 128$，$d_h = 128$）相比，MLA 每层每 token 的 KV cache 缩小了多少倍？（保留 1 位小数）`,
      answer: 56.9,
      tolerance: 0.01,
      unit: '倍',
      explanation: md`MHA：$2 \times 128 \times 128 = 32{,}768$；MLA：576。$32768 / 576 \approx 56.9$。`,
    },
    {
      id: 'q-groups',
      type: 'fill',
      difficulty: 2,
      prompt: md`V2 论文把 MLA 的 KV cache 换算成 GQA：$d_c = 4d_h$、$d_h^R = d_h/2$ 时每层每 token 缓存 $\tfrac{9}{2}d_h$ 个元素，相当于只有 ____(1)____ 组的 GQA。`,
      blanks: [{ label: '(1)', answers: ['2.25'], numeric: true, tolerance: 0.001 }],
      explanation: md`GQA 每组缓存 $2d_h$，$\tfrac{9}{2}d_h / 2d_h = 2.25$。论文据此声称 MLA 的缓存量与 2.25 组的 GQA 相当，性能却强于 MHA。`,
    },
    {
      id: 'q-kr-shared',
      type: 'judge',
      difficulty: 1,
      prompt: md`MLA 中解耦的 RoPE key $k^R_t$ 与 query 一样，每个头各有一份，因此缓存量是 $n_h \times 64$。`,
      answer: false,
      explanation: md`$q^R$ 每头一份（$n_h \times 64$），但 $k^R_t \in \mathbb{R}^{64}$ 被**所有头共享**（论文 Eq. 15 中 $W^{KR} \in \mathbb{R}^{d_h^R \times d}$），这也是缓存只多 64 维的原因。`,
    },
    {
      id: 'q-why-decouple',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 MLA 的 decoupled RoPE，下列说法正确的是：（多选）`,
      options: [
        '若直接对 $k^C$ 施加 RoPE，位置相关的旋转矩阵会夹在 $W^{UQ}$ 与 $W^{UK}$ 之间，$W^{UK}$ 无法被吸收',
        '内容部分 $q^C, k^C$ 不施加 RoPE',
        '解耦后需要为每个历史 token 重新计算 $k^R$',
        '缩放因子变为 $1/\\sqrt{d_h + d_h^R}$',
        '解耦 RoPE 的目的是支持更长的上下文外推',
      ],
      answer: [0, 1, 3],
      explanation: md`C 错：$k^R$ 计算一次后与 $c^{KV}$ 一起缓存。E 错：解耦的动机是**让矩阵吸收成立**，与外推无关（V2 的长上下文扩展用的是 YaRN，作用在 RoPE 部分）。`,
    },
    {
      id: 'q-cached',
      type: 'single',
      difficulty: 1,
      prompt: md`MLA 中 $W^{DQ}$ / $W^{UQ}$（query 的低秩压缩）主要带来什么好处？`,
      options: ['减少 KV cache', '减少训练时的激活显存', '让 RoPE 可以吸收', '提高注意力的秩'],
      answer: 1,
      explanation: md`V2 论文明确说 query 压缩是为了减少训练中的激活显存，它不影响 KV cache（Q 本来就不缓存）。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`DeepSeek-V3（行向量约定 $xW$，$d=7168$）：把矩阵与形状配对：`,
      pairs: [
        ['$W^{DKV}$', '$7168 \\times 512$'],
        ['$W^{UK}$', '$512 \\times 16384$'],
        ['$W^{KR}$', '$7168 \\times 64$'],
        ['$W^{QR}$', '$1536 \\times 8192$'],
        ['$W^{DQ}$', '$7168 \\times 1536$'],
      ],
      explanation: md`$n_h d_h = 128 \times 128 = 16384$；$n_h d_h^R = 128 \times 64 = 8192$；$k^R$ 共享，所以 $W^{KR}$ 只有 64 列；$q^R$ 由 $c^Q$（1536 维）生成。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`decode 阶段使用吸收形式时，哪些上投影矩阵被「吸收」进别的矩阵，从而**不再作用于缓存的历史 token**？`,
      items: ['W^DQ', 'W^UQ', 'W^QR', 'W^DKV', 'W^UK', 'W^UV', 'W^KR', 'W^O'],
      answer: [4, 5],
      explanation: md`$W^{UK}$ 吸收进 query 侧（$\tilde q = q^C {W^{UK}}^\top$），$W^{UV}$ 吸收进输出侧（与 $W^O$ 结合）。其余矩阵都只作用于**当前 token**：$W^{DKV}, W^{KR}$ 用来产生新 token 的缓存，$W^{DQ}, W^{UQ}, W^{QR}$ 产生 query，$W^O$ 做输出投影。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把 MLA（吸收形式）单 token decode 的一层计算排成正确顺序：`,
      items: [
        '$c^Q = h_t W^{DQ}$，并算出 $q^C$ 与 $q^R = \\mathrm{RoPE}(c^Q W^{QR})$',
        '$\\tilde q_i = q^C_i {W^{UK}_i}^\\top$，映射到 512 维潜空间',
        '计算新 token 的 $c^{KV}_t$ 与 $k^R_t$，追加到缓存',
        '$s_{j,i} = (\\tilde q_i\\cdot c^{KV}_j + q^R_i\\cdot k^R_j)/\\sqrt{192}$，softmax',
        '在潜空间加权求和 $\\sum_j p_{j,i} c^{KV}_j$，再乘 $W^{UV}_i$ 与 $W^O$',
      ],
      explanation: md`前两步与第三步互不依赖，但新 token 的 KV 必须在打分**之前**写入缓存（它要看到自己）。缩放用 $\sqrt{192}$，输出侧先在 512 维加权、最后才上投影。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面是吸收形式的 MLA 单头打分（行向量约定，\`W_uk_i\`: \`(d_c, d_h)\`，\`c_kv\`: \`(S, d_c)\`，\`q_c\`: \`(d_h,)\`，\`q_r\`: \`(d_hr,)\`，\`k_r\`: \`(S, d_hr)\`）。点出**所有**有 bug 的行（不一定有多处）：`,
      code: py`
        def absorbed_scores(q_c, q_r, c_kv, k_r, W_uk_i):
            d_c = c_kv.shape[1]
            d_h, d_hr = q_c.shape[0], q_r.shape[0]
            q_lat = W_uk_i @ q_c                       # (d_c,)
            s_content = c_kv @ q_lat                   # (S,)
            s_pos = k_r @ q_r                          # (S,)
            s = (s_content + s_pos) / np.sqrt(d_c + d_hr)
            s = s - s.max()
            p = np.exp(s)
            return p / p.sum()
      `,
      bugLines: [7],
      explanation: md`只有第 7 行有 bug。第 4 行 \`W_uk_i @ q_c\`：\`(d_c, d_h) @ (d_h,) = (d_c,)\`，恰好等于 $q^C {W^{UK}_i}^\top$（行向量约定），是对的。第 7 行缩放应为 \`np.sqrt(d_h + d_hr)\`（=√192）：吸收只改变计算顺序，不改变点积的数值，所以不能按潜空间维度 512 缩放。`,
    },
    {
      id: 'q-prefill',
      type: 'single',
      difficulty: 3,
      prompt: md`为什么主流推理框架在 **prefill** 阶段通常不用 MLA 的吸收形式？`,
      options: [
        '吸收形式在数值上与原形式不等价',
        '吸收后每个 (query, key) 对的计算维度从 192+128 增大到 576+512；prefill 计算密集，展开形式 FLOPs 更少',
        'prefill 阶段没有 KV cache，无法吸收',
        '吸收形式不支持因果掩码',
      ],
      answer: 1,
      explanation: md`两种形式数学等价（A 错）。prefill 时 query 数 ≈ key 数，瓶颈是算力：展开一次 $k^C, v^C$ 的代价被大量 query 摊薄，之后每对只需 320 维乘加；吸收形式每对 1088 维。decode 时只有 1 个 query，瓶颈是读缓存，吸收形式只读 576 维的潜向量，优势明显。C、D 都不成立。`,
    },
    {
      id: 'q-v3-bytes',
      type: 'numeric',
      difficulty: 2,
      prompt: md`DeepSeek-V3（61 层）用 BF16 缓存，单条 128K（$131072$）token 上下文的 MLA KV cache 是多少 GiB？`,
      answer: 8.578,
      tolerance: 0.01,
      unit: 'GiB',
      explanation: md`$61 \times 576 \times 2 \times 131072 = 9{,}210,691{,}584$ 字节 $\approx 8.58$ GiB。若是 128 头 MHA 将超过 480 GiB——这就是 MLA 让 671B 模型能服务长上下文的关键。`,
    },
  ],

  guided: [
    {
      id: 'g-absorb',
      title: '一步步写出吸收形式的 MLA 解码（内容部分）',
      intro: md`
        只看内容部分（不含 RoPE 分支）验证矩阵吸收。新 token 的 query 潜向量 \`c_q\`（\`(d_cq,)\`），缓存 \`c_kv\`（\`(S, d_c)\`），
        权重 \`W_uq\`: \`(d_cq, H*dh)\`、\`W_uk, W_uv\`: \`(d_c, H*dh)\`、\`W_o\`: \`(H*dh, D)\`。

        要求：**不得**计算 \`c_kv @ W_uk\` 或 \`c_kv @ W_uv\`（不展开历史 K/V），全程在 \`d_c\` 维潜空间里做注意力。缩放用 $\sqrt{d_h}$（本题没有 RoPE 分支）。
      `,
      template: py`
        import numpy as np

        def mla_decode_absorbed(c_q, c_kv, W_uq, W_uk, W_uv, W_o, n_heads):
            d_c = c_kv.shape[1]
            dh = W_uq.shape[1] // n_heads
            q = (c_q @ W_uq).reshape(n_heads, dh)          # (H, dh)
            W_uk_h = W_uk.reshape(d_c, n_heads, dh)        # (d_c, H, dh)
            q_lat = {{1}}
            s = {{2}}
            p = np.exp(s - s.max(axis=-1, keepdims=True))
            p = p / p.sum(axis=-1, keepdims=True)
            ctx = {{3}}
            W_uv_h = W_uv.reshape(d_c, n_heads, dh)
            o = {{4}}
            return {{5}}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`吸收 $W^{UK}$：把每个头的 query 映射到潜空间，$\tilde q_i = q_i {W^{UK}_i}^\top$，结果形状 \`(H, d_c)\`。`,
          answer: "np.einsum('hd,chd->hc', q, W_uk_h)",
          accept: ["np.einsum('hd,chd->hc', q, W_uk_h, optimize=True)", "(W_uk_h.transpose(1, 0, 2) @ q[:, :, None])[..., 0]"],
          hint: 'W_uk_h[:, i, :] 是第 i 个头的 (d_c, dh) 矩阵。',
          explain: md`对每个头 $i$：$\tilde q_i[c] = \sum_d q_i[d]\, W^{UK}[c, i, d]$。这一步每个头只做 $d_c \times d_h$ 次乘加，与缓存长度无关。`,
        },
        {
          blank: 2,
          instruction: md`潜空间打分：每个头的 $\tilde q_i$ 与全部缓存潜向量点积并缩放，形状 \`(H, S)\`。`,
          answer: 'q_lat @ c_kv.T / np.sqrt(dh)',
          accept: ['q_lat @ c_kv.T / dh ** 0.5', "np.einsum('hc,sc->hs', q_lat, c_kv) / np.sqrt(dh)"],
          explain: md`所有头共享同一份 \`c_kv\`——这就是「吸收后 MLA = MQA」。缩放仍是原始点积维度 $\sqrt{d_h}$，不是 $\sqrt{d_c}$。`,
        },
        {
          blank: 3,
          instruction: md`在潜空间里加权求和：每个头得到一个 \`d_c\` 维上下文向量，形状 \`(H, d_c)\`。`,
          answer: 'p @ c_kv',
          accept: ["np.einsum('hs,sc->hc', p, c_kv)"],
          explain: md`$\sum_j p_{j,i}\, c^{KV}_j$——还没有乘 $W^{UV}$，所以 value 也不用展开。`,
        },
        {
          blank: 4,
          instruction: md`吸收 $W^{UV}$ 的第一步：每个头的潜空间上下文乘上自己的 $W^{UV}_i$，得到 \`(H, dh)\`。`,
          answer: "np.einsum('hc,chd->hd', ctx, W_uv_h)",
          accept: ["np.einsum('hc,chd->hd', ctx, W_uv_h, optimize=True)", "(ctx[:, None, :] @ W_uv_h.transpose(1, 0, 2))[:, 0]"],
          explain: md`$(\sum_j p\, c_j) W^{UV}_i = \sum_j p\, (c_j W^{UV}_i) = \sum_j p\, v_{j,i}$——结合律保证与展开形式相等，但只做了一次上投影而不是 $S$ 次。`,
        },
        {
          blank: 5,
          instruction: md`合头并做输出投影，返回 \`(D,)\`。`,
          answer: 'o.reshape(-1) @ W_o',
          accept: ['o.reshape(n_heads * dh) @ W_o', 'o.flatten() @ W_o', 'o.ravel() @ W_o'],
          explain: md`这里把 $W^{UV}_i$ 与 $W^O_i$ 分两步乘（不预乘），参数不膨胀。`,
        },
      ],
      setup: py`
        import numpy as np, inspect

        def _ref_expanded(c_q, c_kv, W_uq, W_uk, W_uv, W_o, H):
            dh = W_uq.shape[1] // H
            q = (c_q @ W_uq).reshape(H, dh)
            k = (c_kv @ W_uk).reshape(-1, H, dh)       # 展开历史 K
            v = (c_kv @ W_uv).reshape(-1, H, dh)
            outs = []
            for i in range(H):
                s = k[:, i] @ q[i] / np.sqrt(dh)
                p = np.exp(s - s.max()); p /= p.sum()
                outs.append(p @ v[:, i])
            return np.concatenate(outs) @ W_o

        def _case(seed, H=4, dh=8, dc=12, dcq=10, D=16, S=7):
            r = np.random.default_rng(seed)
            return (r.normal(size=dcq), r.normal(size=(S, dc)), r.normal(size=(dcq, H * dh)) / 3,
                    r.normal(size=(dc, H * dh)) / 3, r.normal(size=(dc, H * dh)) / 3, r.normal(size=(H * dh, D)) / 5, H)
      `,
      tests: [
        {
          name: '吸收形式 = 展开形式',
          code: py`
            args = _case(0)
            got = mla_decode_absorbed(*args)
            exp = _ref_expanded(*args)
            assert got.shape == exp.shape, f"形状应为 {exp.shape}，得到 {got.shape}"
            assert np.allclose(got, exp, atol=1e-10), f"最大误差 {np.abs(got - exp).max():.2e}"
          `,
        },
        {
          name: '不同配置',
          code: py`
            args = _case(1, H=2, dh=4, dc=6, dcq=5, D=8, S=11)
            assert np.allclose(mla_decode_absorbed(*args), _ref_expanded(*args), atol=1e-10)
          `,
        },
        {
          name: '没有展开历史 K/V',
          hidden: true,
          code: py`
            src = inspect.getsource(mla_decode_absorbed).replace(" ", "")
            for bad in ("c_kv@W_uk", "c_kv@W_uv", "c_kv.dot(W_uk", "c_kv.dot(W_uv"):
                assert bad not in src, f"发现 {bad}：吸收形式不应展开历史 K/V"
          `,
        },
      ],
      outro: md`
        你刚刚证明了：**缓存 512 维潜向量、在潜空间里做注意力**与「展开成 128 个头的 K/V 再做 MHA」完全等价。完整的 MLA 再加上 RoPE 分支：
        打分多一项 $q^R_i \cdot k^R_j$，缩放变成 $\sqrt{d_h + d_h^R}$——见实战第 2 题。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-mla-forward',
      title: '手撕 MLA 前向（展开形式，含 decoupled RoPE）',
      difficulty: 'hard',
      source: '大模型岗手撕 · MLA（DeepSeek-V2/V3）',
      timeLimitMin: 35,
      description: md`
        实现 MLA 的训练 / prefill 形式（因果自注意力，位置为 $0 \dots T-1$）：

        \`\`\`python
        def mla_forward(h, P, n_heads, d_h, d_hr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            # 返回 (out (B, T, D), c_kv (B, T, d_c), k_r (B, T, d_hr))  —— 后两者即需要缓存的量
        \`\`\`

        \`P\` 中的权重（行向量约定 \`x @ W\`，记 \`H = n_heads\`）：

        | 名称 | 形状 | 公式 |
        |---|---|---|
        | \`W_dq\` | \`(D, d_cq)\` | $c^Q = h W^{DQ}$ |
        | \`W_uq\` | \`(d_cq, H*d_h)\` | $q^C = c^Q W^{UQ}$ |
        | \`W_qr\` | \`(d_cq, H*d_hr)\` | $q^R = \mathrm{RoPE}(c^Q W^{QR})$，每头 \`d_hr\` 维 |
        | \`W_dkv\` | \`(D, d_c)\` | $c^{KV} = h W^{DKV}$ |
        | \`W_uk\`, \`W_uv\` | \`(d_c, H*d_h)\` | $k^C = c^{KV} W^{UK}$，$v = c^{KV} W^{UV}$ |
        | \`W_kr\` | \`(D, d_hr)\` | $k^R = \mathrm{RoPE}(h W^{KR})$，**所有头共享** |
        | \`W_o\` | \`(H*d_h, D)\` | 输出投影 |

        - 打分 $s = (q^C_i\cdot k^C_i + q^R_i\cdot k^R)/\sqrt{d_h + d_{hr}}$，因果掩码，softmax，乘 $v_i$，合头后乘 \`W_o\`。
        - 第 $i$ 个头使用各矩阵输出维上的第 $i$ 个列块。
        - 已给出 \`rope(x, pos)\`（rotate_half 风格，作用于最后一维）。为简化，省略 V3 中潜向量后的 RMSNorm。
      `,
      starter: 'import numpy as np\n\n' + ROPE + '\n\n' + py`
        def mla_forward(h, P, n_heads, d_h, d_hr):
            B, T, D = h.shape
            # TODO 1: query 路径：c_q → q_c (B,H,T,d_h)、q_r (B,H,T,d_hr，施加 RoPE)
            # TODO 2: KV 路径：c_kv → k_c、v；k_r = rope(h @ W_kr)（共享）
            # TODO 3: 拼接 [q_c; q_r]、[k_c; k_r]，缩放 sqrt(d_h + d_hr)，因果 softmax
            # TODO 4: 合头 @ W_o；返回 (out, c_kv, k_r)
            raise NotImplementedError
      `,
      solution: 'import numpy as np\n\n' + ROPE + '\n\n' + py`
        def mla_forward(h, P, n_heads, d_h, d_hr):
            B, T, D = h.shape
            H, pos = n_heads, np.arange(T)
            heads = lambda t, d: t.reshape(B, T, H, d).transpose(0, 2, 1, 3)   # (B, H, T, d)

            # query 路径
            c_q = h @ P["W_dq"]                                   # (B, T, d_cq)
            q_c = heads(c_q @ P["W_uq"], d_h)                     # (B, H, T, d_h)
            q_r = rope(heads(c_q @ P["W_qr"], d_hr), pos)         # (B, H, T, d_hr)

            # KV 路径（需要缓存的只有 c_kv 与 k_r）
            c_kv = h @ P["W_dkv"]                                 # (B, T, d_c)
            k_c = heads(c_kv @ P["W_uk"], d_h)                    # (B, H, T, d_h)
            v = heads(c_kv @ P["W_uv"], d_h)
            k_r = rope(h @ P["W_kr"], pos)                        # (B, T, d_hr)，所有头共享

            # 打分 = 内容项 + 位置项（k_r 广播到所有头）
            s = q_c @ k_c.transpose(0, 1, 3, 2) + q_r @ k_r[:, None].transpose(0, 1, 3, 2)
            s = s / np.sqrt(d_h + d_hr)
            s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            o = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, H * d_h)
            return o @ P["W_o"], c_kv, k_r
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，结构与 HF DeepSeek-V3 实现一致，省略 RMSNorm）',
          code: py`
            import math
            import torch
            import torch.nn as nn

            def rope(x, pos, base=10000.0):
                d = x.size(-1)
                inv = base ** (-torch.arange(0, d, 2, device=x.device) / d)
                ang = pos[:, None].float() * inv[None]
                cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
                x1, x2 = x[..., : d // 2], x[..., d // 2:]
                return x * cos + torch.cat([-x2, x1], -1) * sin

            class MLA(nn.Module):
                def __init__(self, D=7168, H=128, d_h=128, d_hr=64, d_c=512, d_cq=1536):
                    super().__init__()
                    self.H, self.d_h, self.d_hr, self.d_c = H, d_h, d_hr, d_c
                    self.q_a = nn.Linear(D, d_cq, bias=False)                       # W^DQ
                    self.q_b = nn.Linear(d_cq, H * (d_h + d_hr), bias=False)        # [W^UQ; W^QR]
                    self.kv_a = nn.Linear(D, d_c + d_hr, bias=False)                # [W^DKV; W^KR] 合并
                    self.kv_b = nn.Linear(d_c, H * 2 * d_h, bias=False)             # [W^UK; W^UV]
                    self.o = nn.Linear(H * d_h, D, bias=False)

                def forward(self, h):
                    B, T, _ = h.shape
                    pos = torch.arange(T, device=h.device)
                    q = self.q_b(self.q_a(h)).view(B, T, self.H, -1).transpose(1, 2)
                    q_c, q_r = q.split([self.d_h, self.d_hr], -1)
                    c_kv, k_r = self.kv_a(h).split([self.d_c, self.d_hr], -1)
                    k_r = rope(k_r, pos)[:, None]                                    # (B, 1, T, d_hr) 共享
                    kv = self.kv_b(c_kv).view(B, T, self.H, 2 * self.d_h).transpose(1, 2)
                    k_c, v = kv.split([self.d_h, self.d_h], -1)
                    q = torch.cat([q_c, rope(q_r, pos)], -1)
                    k = torch.cat([k_c, k_r.expand(-1, self.H, -1, -1)], -1)
                    s = q @ k.transpose(-2, -1) / math.sqrt(self.d_h + self.d_hr)
                    s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=h.device).tril(), float("-inf"))
                    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, -1)
                    return self.o(o), c_kv, k_r.squeeze(1)
          `,
        },
      ],
      explanation: md`
        **结构**：两条路径 + 一个共享 RoPE key。
        - query：$h \to c^Q \to (q^C, q^R)$，$q^R$ 每头 \`d_hr\` 维并施加 RoPE；
        - KV：$h \to c^{KV} \to (k^C, v)$；$k^R = \mathrm{RoPE}(h W^{KR})$ 只有一份，打分时广播到所有头；
        - 打分 = 内容项 + 位置项，再除以 $\sqrt{d_h + d_{hr}}$。等价写法：把 $[q^C; q^R]$ 与 $[k^C; k^R]$ 拼接后做一次点积（PyTorch 版）。

        **复杂度**：与 $d_h + d_{hr}$ 维的 MHA 相同，额外的是低秩投影；缓存每 token 每层 $d_c + d_{hr}$ 个元素。

        **易错点**：$k^R$ 每头一份（错，形状应为 \`(B, T, d_hr)\`）；RoPE 加到 $q^C/k^C$ 上；缩放用 $\sqrt{d_h}$；$k^R$ 从 $c^{KV}$ 而不是 $h$ 产生。
      `,
      complexity: { time: 'O(B·T·D·(d_c+d_cq) + B·H·T²·(d_h+d_hr))', space: 'O(B·H·T²)' },
      hints: [
        md`写一个 \`heads(t, d) = t.reshape(B, T, H, d).transpose(0, 2, 1, 3)\`，q_c、q_r、k_c、v 都用它拆头。`,
        md`\`k_r = rope(h @ P["W_kr"], np.arange(T))\` 形状 \`(B, T, d_hr)\`；打分时用 \`k_r[:, None]\` 广播到头维。`,
        md`\`s = (q_c @ k_c^T + q_r @ k_r^T) / np.sqrt(d_h + d_hr)\`，之后与普通因果 MHA 相同。`,
      ],
      setup: SETUP + '\n' + py`
        def _cmp_fwd(seed, B, T, D, H, dh, dhr, dc, dcq):
            P = _params(seed, D, H, dh, dhr, dc, dcq)
            h = np.random.default_rng(seed + 7).normal(size=(B, T, D))
            out, ckv, kr = mla_forward(h, P, H, dh, dhr)
            eo, eckv, ekr = _ref_mla(h, P, H, dh, dhr)
            assert out.shape == (B, T, D), f"out 形状应为 {(B, T, D)}，得到 {out.shape}"
            assert ckv.shape == (B, T, dc), f"c_kv 形状应为 {(B, T, dc)}，得到 {ckv.shape}"
            assert kr.shape == (B, T, dhr), f"k_r 形状应为 {(B, T, dhr)}（所有头共享），得到 {kr.shape}"
            assert np.allclose(ckv, eckv, atol=1e-10), "c_kv 不一致"
            assert np.allclose(kr, ekr, atol=1e-10), "k_r 不一致：是否由 h（而非 c_kv）经 W_kr 后施加 RoPE？"
            err = np.abs(out - eo).max()
            assert err < 1e-9, f"输出最大误差 {err:.2e}"
      `,
      tests: [
        { name: '小配置 H=2', code: py`_cmp_fwd(0, 1, 5, 16, 2, 4, 2, 6, 8)` },
        { name: 'batch=2, H=4', code: py`_cmp_fwd(1, 2, 6, 24, 4, 8, 4, 10, 12)` },
        {
          name: '缓存量：c_kv 与 k_r 每 token 共 d_c + d_hr 个元素',
          code: py`
            P = _params(3, 16, 4, 4, 2, 6, 8)
            _, ckv, kr = mla_forward(np.ones((1, 3, 16)), P, 4, 4, 2)
            assert ckv.shape[-1] + kr.shape[-1] == 8, f"每 token 缓存应为 6 + 2 = 8，得到 {ckv.shape[-1] + kr.shape[-1]}"
          `,
        },
        { name: '随机对拍（更长序列）', hidden: true, code: py`_cmp_fwd(4, 2, 12, 32, 4, 8, 4, 12, 16)` },
        {
          name: '随机对拍（d_hr = d_h）',
          hidden: true,
          code: py`
            P = _params(5, 16, 2, 4, 4, 6, 8)
            h = np.random.default_rng(1).normal(size=(1, 4, 16))
            a, _, _ = mla_forward(h, P, 2, 4, 4)
            eo, _, _ = _ref_mla(h, P, 2, 4, 4)
            assert np.allclose(a, eo, atol=1e-9), "与参考实现不一致（检查 q_r 是否施加了 RoPE）"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-merge',
          q: md`HF 的 DeepSeek-V3 实现里为什么把 $W^{DKV}$ 和 $W^{KR}$ 合成一个 \`kv_a_proj_with_mqa\`？`,
          a: md`二者输入都是 $h_t$，合并成 $d \times (512+64)$ 的一个矩阵可以一次矩阵乘得到需要缓存的全部 576 维，再 split 成 $c^{KV}$ 与 $k^R$（名字里的 mqa 指 $k^R$ 像 MQA 一样所有头共享）。同理 $W^{UK}, W^{UV}$ 合并成 \`kv_b_proj\`，$W^{UQ}, W^{QR}$ 合并成 \`q_b_proj\`。`,
        },
      ],
    },
    {
      id: 'c-mla-absorb',
      title: 'MLA 矩阵吸收：只用潜向量缓存完成 decode，并验证与展开形式等价',
      difficulty: 'hard',
      source: '大模型岗手撕 · MLA 推理',
      timeLimitMin: 35,
      description: md`
        实现两个函数（权重定义同上一题）：

        \`\`\`python
        def absorb_weights(P, n_heads, d_h) -> tuple[np.ndarray, np.ndarray]:
            # W_qk: (H, d_cq, d_c)，W_qk[i] = W_uq_i @ W_uk_i.T   （吸收 W_uk 进 query 侧）
            # W_ov: (H, d_c, D)，  W_ov[i] = W_uv_i @ W_o_i        （吸收 W_uv 进 W_o；W_o_i 是 W_o 的第 i 个行块）

        def mla_decode_absorbed(h_t, pos, c_kv, k_r, A, n_heads, d_h, d_hr):
            # 返回 (out (B, D), c_kv_new (B, S+1, d_c), k_r_new (B, S+1, d_hr))
        \`\`\`

        - \`h_t\`: \`(B, D)\` 新 token 的隐状态，其绝对位置为 \`pos\`（= 已缓存长度 S）。
        - \`c_kv\`: \`(B, S, d_c)\`，\`k_r\`: \`(B, S, d_hr)\` 为已有缓存（$S \ge 0$，可以为空）。
        - \`A\` 只包含 \`W_dq, W_qr, W_dkv, W_kr, W_qk, W_ov\`——**没有** \`W_uq / W_uk / W_uv / W_o\`，因此你无法、也不应展开任何 K/V。
        - 步骤：算新 token 的 $c^{KV}_t$、$k^R_t$（RoPE 位置 \`pos\`）并追加到缓存；$\tilde q_i = c^Q W_{qk}[i]$；
          $s_{i,j} = (\tilde q_i \cdot c^{KV}_j + q^R_i \cdot k^R_j)/\sqrt{d_h + d_{hr}}$；softmax；$\text{out} = \sum_i (\sum_j p_{i,j} c^{KV}_j) W_{ov}[i]$。
        - 已给出 \`rope(x, pos)\`。测试会把逐 token 吸收解码的结果与上一题的展开形式全量前向逐位比较。
      `,
      starter: 'import numpy as np\n\n' + ROPE + '\n\n' + py`
        def absorb_weights(P, n_heads, d_h):
            # TODO: 按头拆分 W_uq / W_uk / W_uv / W_o，再两两相乘
            raise NotImplementedError

        def mla_decode_absorbed(h_t, pos, c_kv, k_r, A, n_heads, d_h, d_hr):
            # TODO: 新 token 的 c_kv / k_r 追加到缓存 → 潜空间打分（+ RoPE 项）→ 潜空间加权 → W_ov
            raise NotImplementedError
      `,
      solution: 'import numpy as np\n\n' + ROPE + '\n\n' + py`
        def absorb_weights(P, n_heads, d_h):
            H = n_heads
            d_cq, d_c, D = P["W_uq"].shape[0], P["W_uk"].shape[0], P["W_o"].shape[1]
            W_uq = P["W_uq"].reshape(d_cq, H, d_h)
            W_uk = P["W_uk"].reshape(d_c, H, d_h)
            W_uv = P["W_uv"].reshape(d_c, H, d_h)
            W_o = P["W_o"].reshape(H, d_h, D)                        # 第 i 个行块
            W_qk = np.einsum("qhd,chd->hqc", W_uq, W_uk)             # W_uq_i @ W_uk_i.T
            W_ov = np.einsum("chd,hdo->hco", W_uv, W_o)              # W_uv_i @ W_o_i
            return W_qk, W_ov

        def mla_decode_absorbed(h_t, pos, c_kv, k_r, A, n_heads, d_h, d_hr):
            B, D = h_t.shape
            H = n_heads
            # 1) 新 token 写入缓存（它也要看到自己）
            c_new = h_t @ A["W_dkv"]                                 # (B, d_c)
            kr_new = rope((h_t @ A["W_kr"])[:, None, :], [pos])[:, 0]   # (B, d_hr)
            c_kv = np.concatenate([c_kv, c_new[:, None]], axis=1)    # (B, S+1, d_c)
            k_r = np.concatenate([k_r, kr_new[:, None]], axis=1)     # (B, S+1, d_hr)
            # 2) query：潜空间部分 + RoPE 部分
            c_q = h_t @ A["W_dq"]                                    # (B, d_cq)
            q_lat = np.einsum("bq,hqc->bhc", c_q, A["W_qk"])         # (B, H, d_c)
            q_r = rope((c_q @ A["W_qr"]).reshape(B, H, 1, d_hr), [pos])[:, :, 0]   # (B, H, d_hr)
            # 3) 打分：所有头共享同一份 (c_kv, k_r) —— 宽 MQA
            s = np.einsum("bhc,bsc->bhs", q_lat, c_kv) + np.einsum("bhr,bsr->bhs", q_r, k_r)
            s = s / np.sqrt(d_h + d_hr)                              # 仍是原始点积维度
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            # 4) 潜空间加权求和，再经吸收后的 W_ov
            ctx = np.einsum("bhs,bsc->bhc", p, c_kv)                 # (B, H, d_c)
            out = np.einsum("bhc,hco->bo", ctx, A["W_ov"])           # (B, D)
            return out, c_kv, k_r
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，不预乘权重、改变结合顺序的工程写法）',
          code: py`
            import math
            import torch

            @torch.no_grad()
            def mla_decode_absorbed(q_c, q_r, c_kv, k_r, W_uk, W_uv, W_o, d_h, d_hr):
                """q_c: (B, H, d_h)  q_r: (B, H, d_hr)（已施加 RoPE）
                c_kv: (B, S, d_c)  k_r: (B, S, d_hr)（缓存，已含新 token）
                W_uk, W_uv: (H, d_c, d_h)  W_o: (H*d_h, D)"""
                q_lat = torch.einsum("bhd,hcd->bhc", q_c, W_uk)          # 吸收 W_UK：q^C W_UK_i^T
                s = torch.einsum("bhc,bsc->bhs", q_lat, c_kv) + torch.einsum("bhr,bsr->bhs", q_r, k_r)
                p = (s / math.sqrt(d_h + d_hr)).softmax(-1)
                ctx = torch.einsum("bhs,bsc->bhc", p, c_kv)               # 潜空间加权
                o = torch.einsum("bhc,hcd->bhd", ctx, W_uv)               # 再乘 W_UV
                return o.flatten(1) @ W_o                                 # 最后 W_O
          `,
        },
      ],
      explanation: md`
        **吸收的代数**：
        - 内容打分 $q^C_i \cdot k^C_{j,i} = (c^Q W^{UQ}_i)({c_j} W^{UK}_i)^\top = c^Q\,(W^{UQ}_i {W^{UK}_i}^\top)\, c_j^\top = (c^Q W_{qk}[i]) \cdot c_j$；
        - 输出 $\sum_i o_i W^O_i = \sum_i (\sum_j p_{ij} c_j W^{UV}_i) W^O_i = \sum_i (\sum_j p_{ij} c_j)(W^{UV}_i W^O_i)$。

        两处都只用了矩阵乘法结合律，RoPE 项 $q^R_i \cdot k^R_j$ 不涉及 $W^{UK}$，单独相加——这正是 decouple 的意义。

        **复杂度（每步）**：读缓存 $S(d_c + d_{hr})$ 个元素；计算 $O(H S (2d_c + d_{hr}))$。不随 $H d_h$ 展开。

        **工程注意**：预乘的 \`W_qk\`（$H \times d_{cq} \times d_c$）与 \`W_ov\`（$H \times d_c \times D$）在 V3 尺寸下分别约 1 亿与 4.7 亿参数/层，远大于原矩阵，
        所以真实实现通常**不预乘**，而是改变结合顺序（见 PyTorch 版）。本题预乘只是为了从接口上保证你没有展开 K/V。
      `,
      complexity: { time: '每步 O(H·S·(2·d_c + d_hr) + D·(d_c + d_cq))', space: 'O(S·(d_c + d_hr))' },
      hints: [
        md`\`absorb_weights\`：把 \`W_uq\` reshape 成 \`(d_cq, H, d_h)\`，\`W_uk\` 成 \`(d_c, H, d_h)\`，\`np.einsum("qhd,chd->hqc", ...)\`；\`W_o\` reshape 成 \`(H, d_h, D)\`。`,
        md`新 token 的 RoPE：\`rope((h_t @ W_kr)[:, None, :], [pos])[:, 0]\`；q_r 先 reshape 成 \`(B, H, 1, d_hr)\` 再 rope。`,
        md`打分用 einsum：\`"bhc,bsc->bhs"\`（潜空间）+ \`"bhr,bsr->bhs"\`（RoPE），除以 \`sqrt(d_h + d_hr)\`。`,
      ],
      setup: SETUP + '\n' + py`
        def _A(P, H, dh):
            W_qk, W_ov = absorb_weights(P, H, dh)
            return {"W_dq": P["W_dq"], "W_qr": P["W_qr"], "W_dkv": P["W_dkv"], "W_kr": P["W_kr"], "W_qk": W_qk, "W_ov": W_ov}

        def _decode_all(seed, B, T, D, H, dh, dhr, dc, dcq, prefill=0):
            P = _params(seed, D, H, dh, dhr, dc, dcq)
            h = np.random.default_rng(seed + 7).normal(size=(B, T, D))
            exp, eckv, ekr = _ref_mla(h, P, H, dh, dhr)
            A = _A(P, H, dh)
            ckv, kr = eckv[:, :prefill], ekr[:, :prefill]      # 用展开形式的前缀缓存起步
            for t in range(prefill, T):
                out, ckv, kr = mla_decode_absorbed(h[:, t], t, ckv, kr, A, H, dh, dhr)
                assert out.shape == (B, D), f"out 形状应为 {(B, D)}，得到 {out.shape}"
                err = np.abs(out - exp[:, t]).max()
                assert err < 1e-9, f"位置 {t}：吸收解码与展开前向不一致，最大误差 {err:.2e}"
            assert ckv.shape == (B, T, dc) and kr.shape == (B, T, dhr), f"缓存形状错误：{ckv.shape}, {kr.shape}"
            assert np.allclose(ckv, eckv, atol=1e-10) and np.allclose(kr, ekr, atol=1e-10), "缓存内容与展开形式不一致"
      `,
      tests: [
        {
          name: 'absorb_weights 形状与数值',
          code: py`
            P = _params(0, 16, 2, 4, 2, 6, 8)
            W_qk, W_ov = absorb_weights(P, 2, 4)
            assert W_qk.shape == (2, 8, 6), f"W_qk 形状应为 (H, d_cq, d_c) = (2, 8, 6)，得到 {W_qk.shape}"
            assert W_ov.shape == (2, 6, 16), f"W_ov 形状应为 (H, d_c, D) = (2, 6, 16)，得到 {W_ov.shape}"
            assert np.allclose(W_qk[1], P["W_uq"][:, 4:8] @ P["W_uk"][:, 4:8].T), "W_qk[1] 应为 W_uq_1 @ W_uk_1.T"
            assert np.allclose(W_ov[0], P["W_uv"][:, 0:4] @ P["W_o"][0:4]), "W_ov[0] 应为 W_uv_0 @ W_o 的第 0 个行块"
          `,
        },
        { name: '从空缓存逐 token 解码 = 展开前向', code: py`_decode_all(1, 1, 6, 16, 2, 4, 2, 6, 8)` },
        { name: 'prefill 4 个 token 后继续解码（batch=2, H=4）', code: py`_decode_all(2, 2, 8, 24, 4, 8, 4, 10, 12, prefill=4)` },
        { name: '更大配置随机对拍', hidden: true, code: py`_decode_all(3, 2, 10, 32, 8, 8, 4, 12, 16, prefill=3)` },
        {
          name: '缩放因子是 sqrt(d_h + d_hr)',
          hidden: true,
          code: py`
            # d_c 与 d_h + d_hr 差别很大时，用错缩放会明显偏离
            _decode_all(4, 1, 5, 16, 2, 4, 2, 24, 8, prefill=1)
          `,
        },
      ],
      followUps: [
        {
          id: 'f-tp',
          q: md`多卡张量并行部署 MLA 时，KV cache 会怎样？`,
          a: md`MHA/GQA 可以按 KV 头切分缓存；而 MLA 的 $c^{KV}$ 与 $k^R$ 被所有头共享、不可按头切分，按头做 TP 时每张卡都要保存**完整**的 576 维缓存（缓存被复制）。因此 DeepSeek-V3 论文的部署方案中，注意力部分只用小规模张量并行（TP4）、再叠加大规模数据并行（不同卡负责不同请求），MoE 部分用专家并行。`,
        },
        {
          id: 'f-quality',
          q: md`MLA 只缓存 576 维，为什么效果不比 MHA 差？`,
          a: md`实验上 V2 论文附录 D.2 报告 MLA 在多数基准上优于 MHA。直觉解释：MHA 每头的 K、V 本来就高度冗余（K/V 投影往往具有明显的低秩结构），MLA 用训练直接学一个 512 维的共享潜空间，再用每头独立的上投影恢复多样性；同时由于缓存小，可以负担 128 个头、每头 192 维的 q·k，而同等缓存预算下的 GQA 只能用很少的 KV 头。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-mla',
      freq: 3,
      q: md`讲一下 DeepSeek 的 MLA。它和 GQA 有什么区别？`,
      a: md`
        **结论**：MLA 把 K、V 联合低秩压缩成一个潜向量 $c^{KV}$（512 维）缓存，推理时通过矩阵吸收直接在潜空间里算注意力；GQA 是减少 KV 头数。

        **细节**：$c^{KV} = hW^{DKV}$，$k^C = c^{KV}W^{UK}$，$v^C = c^{KV}W^{UV}$；位置信息由解耦的 64 维 $q^R / k^R$ 承载，$k^R$ 所有头共享。每 token 每层缓存 $512 + 64 = 576$ 个元素，相当于 2.25 组的 GQA，但保留了 128 个头、每头 192 维的打分；V2 论文报告其效果优于 MHA。

        **延伸**：decode 时吸收后的 MLA 等价于 128 个 query 头共享一个 576/512 维 KV 头的 MQA，算术强度高（适合 FlashMLA 这类 kernel）；代价是缓存不能按头切分，TP 时要复制，所以 DeepSeek-V3 的部署中注意力只用小规模 TP（TP4）并叠加数据并行。
      `,
    },
    {
      id: 'qa-rope',
      freq: 3,
      q: md`MLA 为什么要 decoupled RoPE？`,
      a: md`
        **结论**：为了让 $W^{UK}$ 能被吸收进 query 侧。

        **原因**：内容打分 $q^C\cdot k^C = c^Q W^{UQ} {W^{UK}}^\top {c^{KV}}^\top$，中间是固定矩阵，可以合并/改变结合顺序，于是 query 直接和缓存的 $c^{KV}$ 点积。若对 $k^C$ 施加 RoPE，旋转矩阵 $R_t R_j^\top$ 依赖相对位置，夹在 $W^{UQ}$ 与 $W^{UK}$ 之间，矩阵乘不可交换，$W^{UK}$ 无法吸收，推理时要为所有历史 token 重新展开 key。

        **延伸**：解法是新增每头 64 维的 $q^R$ 与共享的 64 维 $k^R$ 专门做 RoPE，打分为两项之和，缩放 $1/\sqrt{128+64}$。代价是缓存多 64 维；V2 的 128K 长上下文扩展用 YaRN 作用在这部分 RoPE 上。
      `,
    },
    {
      id: 'qa-absorb',
      freq: 3,
      q: md`MLA 的矩阵吸收具体怎么做？prefill 和 decode 都用吸收吗？`,
      a: md`
        **结论**：$W^{UK}$ 吸收进 query 侧，$W^{UV}$ 吸收进输出侧；decode 用吸收形式，prefill 通常用展开形式。

        **做法**：$\tilde q_i = q^C_i {W^{UK}_i}^\top$（512 维），打分 $\tilde q_i\cdot c^{KV}_j + q^R_i\cdot k^R_j$；输出 $\sum_i (\sum_j p_{ij} c^{KV}_j) W^{UV}_i W^O_i$。可以预乘矩阵，也可以只改变结合顺序（不增加参数，工程常用）。缩放仍为 $\sqrt{192}$。

        **取舍**：吸收后每对 (q, k) 要做 576 + 512 维乘加，展开形式只需 192 + 128 维。prefill 计算密集，展开形式更省；decode 每步 1 个 query、受带宽限制，吸收形式只读 576 维潜向量更优。vLLM 等框架按此选择两条路径。
      `,
    },
    {
      id: 'qa-numbers',
      freq: 2,
      q: md`算一下 DeepSeek-V3 的 KV cache：每 token 多大？128K 上下文多大？和 MHA 比呢？`,
      a: md`
        **结论**：每 token $61 \times 576 \times 2 = 70{,}272$ 字节 ≈ 68.6 KiB；128K 约 8.58 GiB；同头数 MHA 约 56.9 倍。

        **推导**：V3 有 61 层，每层缓存 $d_c + d_h^R = 512 + 64 = 576$ 个元素，BF16 每个 2 字节。$70272 \times 131072 \approx 9.21 \times 10^9$ 字节 ≈ 8.58 GiB。MHA（128 头 × 128 维）每层 $2 \times 128 \times 128 = 32768$ 个元素，是 576 的约 56.9 倍，128K 上下文需要约 488 GiB。

        **延伸**：再配合 FP8 缓存可再减半；V2 论文的部署中把 KV cache 量化到平均约 6 bit。DeepSeek-V4 进一步用压缩稀疏注意力（CSA/HCA）把 1M 上下文的 KV cache 压到 V3.2 的约 10%。
      `,
    },
  ],

  cheatsheet: md`
    | | 公式 / 数值（V3） |
    |---|---|
    | 压缩 | $c^{KV}=hW^{DKV}$（512），$c^Q=hW^{DQ}$（1536） |
    | 展开 | $k^C=c^{KV}W^{UK}$，$v^C=c^{KV}W^{UV}$（128 头 × 128） |
    | 解耦 RoPE | $q^R=\mathrm{RoPE}(c^QW^{QR})$（每头 64），$k^R=\mathrm{RoPE}(hW^{KR})$（**共享** 64） |
    | 打分 | $(q^C\cdot k^C + q^R\cdot k^R)/\sqrt{192}$ |
    | 缓存 | $c^{KV}$ + $k^R$ = **576**/层/token；V3 全模型 BF16 70,272 B/token |
    | 吸收 | $\tilde q_i = q^C_i {W^{UK}_i}^\top$；输出 $\sum_i(\sum_j p\,c_j)W^{UV}_iW^O_i$ |

    - RoPE 若加在 $k^C$ 上，$R_tR_j^\top$ 夹在 $W^{UQ}$、$W^{UK}$ 间 → 无法吸收
    - decode：吸收 = 128 头共享 576/512 维 KV 的 MQA；prefill：展开更省 FLOPs
    - ≈ 2.25 组 GQA 的缓存；V2 附录 D.2：效果优于 MHA
    - 缓存不可按头切分 → TP 时复制；V3 部署注意力用 TP4 + DP
  `,
})
