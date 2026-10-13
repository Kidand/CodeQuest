/**
 * L3-BOSS 注意力综合：复盘 + 混合题 + 两道综合手撕（批量 GQA 解码器、split-KV 解码）。
 */
import { defineLevel, md, py } from '@/content/helpers'

const ROPE = py`
  def rope(x, pos, base=10000.0):
      """rotate_half 风格 RoPE。x: (..., T, d)；pos: 可广播到 x.shape[:-1] 的位置（可每个样本不同）"""
      d = x.shape[-1]
      inv = base ** (-np.arange(0, d, 2) / d)
      ang = np.asarray(pos, dtype=float)[..., None] * inv          # (..., T, d/2)
      cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)
      sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)
      x1, x2 = x[..., : d // 2], x[..., d // 2:]
      return x * cos + np.concatenate([-x2, x1], -1) * sin
`

export default defineLevel({
  learn: [
    {
      id: 'recap',
      title: '复盘：从 SDPA 到 FlashAttention 的一条主线',
      body: md`
        本章所有技术都围绕两个瓶颈展开：**KV cache 显存 / 带宽**（推理）与 **$N^2$ 中间矩阵的读写**（训练与 prefill）。

        | 技术 | 解决什么 | 代价 | 关键数字 / 公式 |
        |---|---|---|---|
        | SDPA | 可微的软寻址 | $O(N^2d)$ 计算 | $\operatorname{Var}(q\cdot k)=d_k$ → 除以 $\sqrt{d_k}$ |
        | MHA | 多个子空间、多种关系 | $d_h$ 变小 | 参数 $4D^2$，拆头 reshape→transpose |
        | KV cache | decode 不重算前缀 | 显存随长度线性增长 | $2LBSn_{kv}d_h\cdot$bytes |
        | MQA / GQA | 缓存与带宽 ÷ $h/n_{kv}$ | 表达能力下降（GQA 轻微） | q 头 $i$ → KV 头 $\lfloor i/n_{\text{rep}}\rfloor$ |
        | MLA | 不减头数也能小缓存 | 实现复杂、TP 时缓存要复制 | 576 维/层/token，$\sqrt{192}$ 缩放，吸收 $W^{UK}, W^{UV}$ |
        | FlashAttention | HBM 读写与 $O(N^2)$ 显存 | FLOPs 略增（重算） | $\Theta(N^2d^2/M)$，online softmax |
        | split-KV 解码 | decode 时 SM 吃不满 | 额外的合并步骤 | $o=\sum_s e^{\text{lse}_s-\text{lse}}o_s$ |

        **贯穿全章的三个「易错不变量」**
        1. **掩码对齐**：query 是序列的最后 $L$ 个位置时，因果掩码用 \`tril(ones((L, S)), k=S-L)\`；左 padding 还要配合 \`cumsum(mask) - 1\` 的位置编号。
        2. **按什么维度缩放**：永远是 q·k 点积的真实维度（MHA 为 $d_h$，MLA 为 $d_h + d_h^R$），矩阵吸收不改变它。
        3. **换基准**：softmax 的分子分母可以同乘 $e^{c}$——数值稳定（减最大值）、online softmax、FlashAttention、split-KV 合并都是这一个事实。

        BOSS 实战：① 带左 padding、RoPE、KV cache 的批量 GQA 解码器，要求与逐条单独计算完全一致；② Flash-Decoding 风格的 split-KV 解码与 logsumexp 合并。
      `,
    },
  ],

  quiz: [
    {
      id: 'b-scale',
      type: 'single',
      difficulty: 1,
      prompt: md`Llama-3-8B（$D=4096$、32 个 query 头、8 个 KV 头）注意力的缩放因子是：`,
      options: ['$1/\\sqrt{4096}$', '$1/\\sqrt{128}$', '$1/\\sqrt{512}$', '$1/\\sqrt{32}$'],
      answer: 1,
      explanation: md`$d_h = 4096/32 = 128$，缩放 $1/\sqrt{d_h}$。GQA 只改变 KV 头数，不改变头维；$512 = 4096/8$ 是把 $D$ 除以 KV 头数的常见误算。`,
    },
    {
      id: 'b-qwen',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Qwen2.5-7B：28 层、28 个 query 头、4 个 KV 头、$d_h = 128$。BF16、batch 1、32768 token 的 KV cache 是多少 GiB？`,
      answer: 1.75,
      tolerance: 0.01,
      unit: 'GiB',
      explanation: md`$2 \times 28 \times 32768 \times 4 \times 128 \times 2 = 1{,}879{,}048{,}192$ 字节 $= 1.75$ GiB。用 query 头数 28 会算成 12.25 GiB。`,
    },
    {
      id: 'b-merge',
      type: 'numeric',
      difficulty: 2,
      prompt: md`split-KV 解码把一个头的 KV 分成两段：第 1 段 $\text{lse}_1 = 0$、部分输出 $o_1 = 1.0$；第 2 段 $\text{lse}_2 = \ln 3$、$o_2 = 5.0$（标量 value）。合并后的输出是多少？`,
      answer: 4,
      tolerance: 0.001,
      explanation: md`权重 $\propto e^{\text{lse}_s}$：$e^0 : e^{\ln 3} = 1 : 3$，所以 $o = \tfrac14 \times 1 + \tfrac34 \times 5 = 4$。$\text{lse}_s$ 就是该段指数和的对数，比例即两段各自的 softmax 质量。`,
    },
    {
      id: 'b-align',
      type: 'fill',
      difficulty: 2,
      prompt: md`chunked prefill：已缓存 10 个 token，本次送入 4 个新 token（$L=4$、$S=14$）。右下角对齐的因果掩码 \`np.tril(np.ones((4, 14), bool), k=____(1)____)\`；本块第 0 个 query 能看到 ____(2)____ 个 key。`,
      blanks: [
        { label: '(1)', answers: ['10'], numeric: true, tolerance: 0 },
        { label: '(2)', answers: ['11'], numeric: true, tolerance: 0 },
      ],
      explanation: md`$k = S - L = 10$；第 0 个 query 的绝对位置是 10，可看 $j \le 10$ 共 11 个 key（含自己）。`,
    },
    {
      id: 'b-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`GQA：16 个 query 头、4 个 KV 头。点出使用 **KV 头 2** 的 query 头：`,
      items: Array.from({ length: 16 }, (_, i) => `Q${i}`),
      cols: 8,
      answer: [8, 9, 10, 11],
      explanation: md`$n_{\text{rep}} = 4$，KV 头 2 服务 $\lfloor i/4 \rfloor = 2$ 的 Q8–Q11。若选成 Q2、Q6、Q10、Q14 就是 \`np.tile\` 的交错约定。`,
    },
    {
      id: 'b-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把技术与它省下的东西配对：`,
      pairs: [
        ['KV cache', 'decode 时重复计算历史 K/V'],
        ['GQA', 'KV 头数（缓存与带宽）'],
        ['MLA', '每个 KV 头的维度（低秩潜向量）'],
        ['FlashAttention', 'N×N 中间矩阵的 HBM 读写与显存'],
      ],
      explanation: md`KV cache 用显存换计算；GQA 和 MLA 分别从「头数」和「维度」两个方向压缩缓存；FlashAttention 不省 FLOPs，省的是 IO。`,
    },
    {
      id: 'b-order',
      type: 'order',
      difficulty: 1,
      prompt: md`按提出时间先后排序：`,
      items: ['MHA（Transformer, 2017）', 'MQA（Shazeer, 2019）', 'FlashAttention（2022）', 'GQA（Ainslie et al., 2023）', 'MLA（DeepSeek-V2, 2024）'],
      explanation: md`2017 Transformer → 2019 MQA → 2022 FlashAttention → 2023 GQA（同年 FA2）→ 2024 MLA（DeepSeek-V2，5 月）。FA3 为 2024 年 7 月，FA4 为 2026 年 3 月。`,
    },
    {
      id: 'b-absorb',
      type: 'judge',
      difficulty: 2,
      prompt: md`MLA 推理时把 $W^{UK}$ 吸收进 query 侧后，点积在 512 维潜空间里进行，因此缩放因子应改为 $1/\sqrt{512}$。`,
      answer: false,
      explanation: md`吸收只是矩阵乘法换了结合顺序，点积的**数值**与原形式 $q^C\cdot k^C$ 完全相同，缩放仍是 $1/\sqrt{d_h + d_h^R} = 1/\sqrt{192}$。`,
    },
    {
      id: 'b-multi',
      type: 'multi',
      difficulty: 3,
      prompt: md`下列说法正确的是：（多选）`,
      options: [
        'FlashAttention 的 HBM 访问量是 $\\Theta(N^2 d^2/M)$，仍是 $N$ 的二次方',
        'MLA 的缓存可以像 GQA 一样按 KV 头切分到不同 GPU',
        'decode 阶段，GQA 提高了注意力的算术强度',
        'online softmax 中若当前块全为 $-\\infty$ 且之前也没有有效元素，直接套公式会得到 NaN',
        'KV cache 让 prefill 的 FLOPs 降低为原来的 $1/T$',
      ],
      answer: [0, 2, 3],
      explanation: md`B 错：$c^{KV}$、$k^R$ 被所有头共享，不能按头切分，TP 时要复制。E 错：KV cache 省的是 decode 的重复计算，prefill 本来就只算一次。A、C、D 都是本章强调过的结论。`,
    },
    {
      id: 'b-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出单头因果注意力（\`q, k, v\` 形状均为 \`(T, d)\`；有 2 行干扰项）：`,
      lines: [
        'def attn(q, k, v):',
        '    T, d = q.shape',
        '    s = q @ k.T / np.sqrt(d)',
        '    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)',
        '    s = s - s.max(-1, keepdims=True)',
        '    p = np.exp(s)',
        '    p = p / p.sum(-1, keepdims=True)',
        '    return p @ v',
      ],
      distractors: ['    s = q @ k.T / d', '    p = p / p.sum(0, keepdims=True)'],
      explanation: md`缩放用 $\sqrt{d}$；归一化沿 key 维（最后一维）。两个干扰项分别是「过度缩放」与「沿错轴归一化」。`,
    },
    {
      id: 'b-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的 GQA 解码一步（缓存 \`k_cache\` 形状 \`(B, n_kv, t, d)\`）加载 Llama 权重后输出错误。点出所有有 bug 的行：`,
      code: py`
        def gqa_decode(q, k_new, v_new, k_cache, v_cache, n_heads):
            # q: (B, n_heads, 1, d)；k_new, v_new: (B, n_kv, 1, d)
            k_cache = np.concatenate([k_cache, k_new], axis=2)
            v_cache = np.concatenate([v_cache, v_new], axis=2)
            n_rep = n_heads // k_cache.shape[1]
            K = np.tile(k_cache, (1, n_rep, 1, 1))
            V = np.repeat(v_cache, n_rep, axis=1)
            d = q.shape[-1]
            s = q @ K.transpose(0, 1, 3, 2) / np.sqrt(d)
            p = np.exp(s - s.max(-1, keepdims=True))
            p = p / p.sum(-1, keepdims=True)
            return p @ V, k_cache, v_cache
      `,
      bugLines: [6],
      explanation: md`第 6 行用 \`np.tile\` 展开 K，得到的 KV 头顺序是 \`[0,1,…,0,1,…]\`，与 V（第 7 行 \`np.repeat\`）和权重训练时的约定 \`[0,0,…,1,1,…]\` 都不一致——每个 query 头的 key 与 value 甚至来自不同的组。改为 \`np.repeat(k_cache, n_rep, axis=1)\`。其余行正确：单 token 解码不需要因果掩码；缓存沿 \`axis=2\` 拼接。`,
    },
    {
      id: 'b-why',
      type: 'single',
      difficulty: 3,
      prompt: md`面试官问：「既然 FlashAttention 已经把注意力的显存降到 $O(N)$，为什么长上下文推理还需要 GQA / MLA？」最好的回答是：`,
      options: [
        'FlashAttention 只能用于训练，推理不能用',
        'FlashAttention 省的是 $N\\times N$ 中间矩阵；KV cache 是必须长期保存的状态，大小 $\\propto L\\cdot S\\cdot n_{kv} d_h$，FlashAttention 无法减少它，decode 每步还要完整读一遍',
        'GQA / MLA 能减少注意力的 FLOPs，而 FlashAttention 不能',
        '二者是同一种优化，只是名字不同',
      ],
      answer: 1,
      explanation: md`FlashAttention 推理时同样可用（A 错）；GQA 不减少每个 query 头的 FLOPs（C 错）。中间矩阵是「临时量」，KV cache 是「状态」——前者靠计算顺序消除，后者只能靠减少头数（GQA）、压缩维度（MLA）、降精度或缩短长度来减少。`,
    },
  ],

  guided: [],

  challenges: [
    {
      id: 'c-batched-decoder',
      title: '综合：带左 padding、RoPE 与 KV cache 的批量 GQA 解码器',
      difficulty: 'hard',
      source: '大模型岗手撕 · 推理综合',
      timeLimitMin: 45,
      description: md`
        实现一个批量推理用的 GQA 注意力层（单层、无 bias），要求**批量结果与逐条单独计算完全一致**：

        \`\`\`python
        class BatchedGQADecoder:
            def __init__(self, Wq, Wk, Wv, Wo, n_heads, n_kv_heads): ...
            def prefill(self, x, attn_mask) -> np.ndarray:   # x (B, T, D)，attn_mask (B, T)，返回 (B, T, D)
            def step(self, x_t) -> np.ndarray:               # x_t (B, D)，返回 (B, D)
        \`\`\`

        - 权重形状：\`Wq, Wo\`: \`(D, D)\`；\`Wk, Wv\`: \`(D, n_kv_heads * d_h)\`，\`d_h = D // n_heads\`；q 头 $i$ 使用 KV 头 $\lfloor i / n_{\text{rep}} \rfloor$。
        - **左 padding**：\`attn_mask[b, t] = 1\` 表示真实 token、0 表示 pad，pad 全在左侧。
        - **RoPE**：对 q 与 k 施加已给出的 \`rope\`；每条序列的第一个真实 token 位置为 0（\`cumsum(mask) - 1\`）。\`step\` 中新 token 的位置 = 该序列已有的真实 token 数。
        - **KV cache**：缓存旋转后的 k 与 v，形状 \`(B, n_kv_heads, t, d_h)\`（\`t\` 含 pad 槽位），存在 \`self.k_cache\`、\`self.v_cache\`。
        - 掩码：因果 + key padding；pad query 行会整行被遮，必须**不产生 NaN**，\`prefill\` 中 pad 位置的输出置 0。
        - 正确性：对每条序列 $b$，把它的真实 token 与之后 \`step\` 喂入的 token 拼起来，单独做一次（无 padding 的）因果 GQA + RoPE 前向，结果应与批量版本逐位一致。
      `,
      starter: 'import numpy as np\n\n' + ROPE + '\n\n' + py`
        class BatchedGQADecoder:
            def __init__(self, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
                self.Wq, self.Wk, self.Wv, self.Wo = Wq, Wk, Wv, Wo
                self.h, self.hkv = n_heads, n_kv_heads
                self.d_h = Wq.shape[1] // n_heads
                self.k_cache = self.v_cache = None

            def prefill(self, x, attn_mask):
                # TODO: 位置 = cumsum(mask) - 1；q/k 施加 RoPE；写缓存；因果 + key padding 掩码；pad 行输出 0
                raise NotImplementedError

            def step(self, x_t):
                # TODO: 新 token 的位置 = 已有真实 token 数；追加缓存；它能看到全部真实 key
                raise NotImplementedError
      `,
      solution: 'import numpy as np\n\n' + ROPE + '\n\n' + py`
        class BatchedGQADecoder:
            def __init__(self, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
                self.Wq, self.Wk, self.Wv, self.Wo = Wq, Wk, Wv, Wo
                self.h, self.hkv = n_heads, n_kv_heads
                self.d_h = Wq.shape[1] // n_heads
                self.k_cache = self.v_cache = None
                self.valid = None          # (B, t) 哪些缓存槽位是真实 token
                self.next_pos = None       # (B,) 下一个 token 的位置

            def _split(self, t, n):
                B, L, _ = t.shape
                return t.reshape(B, L, n, self.d_h).transpose(0, 2, 1, 3)

            def _attend(self, x, pos, valid_new):
                """x: (B, L, D) 新 token；pos: (B, L)；valid_new: (B, L) bool"""
                B, L, D = x.shape
                q = rope(self._split(x @ self.Wq, self.h), pos[:, None, :])
                k = rope(self._split(x @ self.Wk, self.hkv), pos[:, None, :])
                v = self._split(x @ self.Wv, self.hkv)
                if self.k_cache is None:
                    self.k_cache, self.v_cache, self.valid = k, v, valid_new
                else:
                    self.k_cache = np.concatenate([self.k_cache, k], axis=2)
                    self.v_cache = np.concatenate([self.v_cache, v], axis=2)
                    self.valid = np.concatenate([self.valid, valid_new], axis=1)
                S = self.k_cache.shape[2]
                rep = self.h // self.hkv
                K = np.repeat(self.k_cache, rep, axis=1)
                V = np.repeat(self.v_cache, rep, axis=1)
                s = q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_h)              # (B, h, L, S)
                allowed = np.tril(np.ones((L, S), dtype=bool), k=S - L)[None, None] & self.valid[:, None, None, :]
                s = np.where(allowed, s, -np.inf)
                m = s.max(axis=-1, keepdims=True)
                m = np.where(np.isfinite(m), m, 0.0)                             # 全遮行（pad query）
                p = np.exp(s - m)
                den = p.sum(axis=-1, keepdims=True)
                p = p / np.where(den > 0, den, 1.0)
                o = (p @ V).transpose(0, 2, 1, 3).reshape(B, L, D)
                return o @ self.Wo

            def prefill(self, x, attn_mask):
                mask = np.asarray(attn_mask).astype(bool)
                self.k_cache = self.v_cache = self.valid = None
                pos = np.where(mask, np.cumsum(mask, axis=-1) - 1, 0)          # 左 padding 的位置编号
                out = self._attend(x, pos, mask)
                self.next_pos = mask.sum(axis=-1)
                return np.where(mask[..., None], out, 0.0)

            def step(self, x_t):
                B = x_t.shape[0]
                out = self._attend(x_t[:, None, :], self.next_pos[:, None], np.ones((B, 1), dtype=bool))
                self.next_pos = self.next_pos + 1
                return out[:, 0]
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，核心的掩码与位置构造）',
          code: py`
            import torch
            import torch.nn.functional as F

            def build_inputs(attn_mask: torch.Tensor, L: int, S: int):
                """attn_mask: (B, S) 当前全部缓存槽位是否为真实 token；query 为最后 L 个位置"""
                causal = torch.ones(L, S, dtype=torch.bool, device=attn_mask.device).tril(S - L)
                allowed = causal[None, None] & attn_mask[:, None, None, :].bool()     # (B, 1, L, S)
                # 全遮行打开对角线（或改用 finfo.min 加性掩码），避免 SDPA 产生 NaN
                empty = ~allowed.any(-1, keepdim=True)
                diag = torch.zeros(L, S, dtype=torch.bool, device=attn_mask.device)
                diag[torch.arange(L), torch.arange(L) + S - L] = True                 # 每行「自己」所在的列
                return allowed | (empty & diag[None, None])

            # prefill：position_ids = (attn_mask.cumsum(-1) - 1).clamp(min=0)
            # o = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=allowed, enable_gqa=True)
          `,
        },
      ],
      explanation: md`
        **四个必须同时做对的点**
        1. **位置**：左 padding 时 \`pos = cumsum(mask) - 1\`；\`step\` 时各序列的新位置是各自已有的真实 token 数（不同样本不同），而不是统一的 \`T + step\`。
        2. **掩码**：因果部分右下角对齐（prefill 时 $L=S$，step 时 $L=1$），再与缓存槽位的 \`valid\` 做与；step 的新 token 也要标为 valid。
        3. **全遮行**：左 padding 的 pad query 一个可见 key 都没有，用「非有限最大值替换为 0 + 分母为 0 时除以 1」得到全 0，避免 NaN 污染。
        4. **GQA 缓存**：存 \`n_kv\` 个头、存**旋转后**的 k，计算时 \`np.repeat\` 展开。

        **为什么与单条计算一致**：RoPE 的点积只依赖相对位置，而每条序列内的相对位置与 pad 数无关；pad 槽位的权重恰为 0，不影响输出。
        若位置直接用 \`arange(T)\`，相对位置虽然不变，但 step 时新 token 的位置会与单条推理不同步——本题的隐藏测试正是在不同 pad 数下检查这一点。
      `,
      complexity: { time: '每步 O(B·h·S·d_h + B·D²)', space: 'O(B·n_kv·S·d_h)' },
      hints: [
        md`写一个 \`_attend(x, pos, valid_new)\` 同时服务 prefill 与 step：投影 → 拆头 → RoPE（\`pos[:, None, :]\` 广播到头维）→ 追加缓存 → 掩码注意力。`,
        md`掩码 = \`tril(ones((L, S)), k=S-L)[None, None] & valid[:, None, None, :]\`。`,
        md`\`step\` 中位置用 \`self.next_pos[:, None]\`（prefill 结束时为 \`mask.sum(-1)\`），每步加 1。`,
      ],
      setup: py`
        import numpy as np

        def _ref_seq(x, Wq, Wk, Wv, Wo, h, hkv):
            """单条序列（无 padding）的因果 GQA + RoPE 前向，x: (n, D)"""
            n, D = x.shape
            dh = D // h
            rep = h // hkv
            pos = np.arange(n)
            outs = []
            for i in range(h):
                j = i // rep
                q = rope((x @ Wq[:, i * dh:(i + 1) * dh])[None], pos[None])[0]
                k = rope((x @ Wk[:, j * dh:(j + 1) * dh])[None], pos[None])[0]
                v = x @ Wv[:, j * dh:(j + 1) * dh]
                s = q @ k.T / np.sqrt(dh)
                s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
                p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
                outs.append(p @ v)
            return np.concatenate(outs, -1) @ Wo

        def _run(seed, lengths, T, n_steps, D, h, hkv):
            r = np.random.default_rng(seed)
            dh = D // h
            Wq, Wo = r.normal(size=(D, D)) / np.sqrt(D), r.normal(size=(D, D)) / np.sqrt(D)
            Wk, Wv = r.normal(size=(D, hkv * dh)) / np.sqrt(D), r.normal(size=(D, hkv * dh)) / np.sqrt(D)
            B = len(lengths)
            x = r.normal(size=(B, T, D))
            mask = np.zeros((B, T), dtype=int)
            for b, n in enumerate(lengths):
                mask[b, T - n:] = 1
                x[b, : T - n] = r.normal(size=(T - n, D)) * 50     # pad 位置放垃圾值
            gen = r.normal(size=(n_steps, B, D))
            dec = BatchedGQADecoder(Wq, Wk, Wv, Wo, h, hkv)
            out = dec.prefill(x, mask)
            assert out.shape == (B, T, D), f"prefill 输出形状应为 {(B, T, D)}，得到 {out.shape}"
            assert np.all(np.isfinite(out)), "prefill 输出含 NaN/inf（pad query 整行被遮？）"
            steps = []
            for s in range(n_steps):
                o = dec.step(gen[s])
                assert o.shape == (B, D), f"step 输出形状应为 {(B, D)}，得到 {o.shape}"
                assert np.all(np.isfinite(o)), "step 输出含 NaN/inf"
                steps.append(o)
            for b, n in enumerate(lengths):
                seq = np.concatenate([x[b, T - n:], gen[:, b]], axis=0)
                ref = _ref_seq(seq, Wq, Wk, Wv, Wo, h, hkv)
                err = np.abs(out[b, T - n:] - ref[:n]).max()
                assert err < 1e-9, f"样本 {b}（真实长度 {n}）prefill 与单条计算不一致，最大误差 {err:.2e}"
                assert np.allclose(out[b, : T - n], 0), f"样本 {b} 的 pad 位置输出应为 0"
                for s in range(n_steps):
                    err = np.abs(steps[s][b] - ref[n + s]).max()
                    assert err < 1e-9, f"样本 {b} 第 {s} 步解码与单条计算不一致，最大误差 {err:.2e}（检查位置编号）"
            return dec
      `,
      tests: [
        { name: '无 padding：prefill + 2 步', code: py`_run(0, [5], 5, 2, 16, 4, 2)` },
        { name: '左 padding 的 batch', code: py`_run(1, [3, 6, 4], 6, 3, 16, 4, 2)` },
        {
          name: '缓存形状：未展开的 KV 头',
          code: py`
            dec = _run(2, [2, 4], 4, 2, 24, 6, 3)
            assert dec.k_cache.shape == (2, 3, 6, 4), f"k_cache 形状应为 (B, n_kv, T + steps, d_h) = (2, 3, 6, 4)，得到 {dec.k_cache.shape}"
          `,
        },
        { name: 'MQA + 长度差异大', hidden: true, code: py`_run(3, [1, 7, 3, 5], 7, 4, 16, 4, 1)` },
        { name: '随机配置', hidden: true, code: py`_run(4, [4, 2, 8], 8, 3, 32, 8, 4)` },
      ],
      followUps: [
        {
          id: 'f-right',
          q: md`如果改成「连续批处理」（continuous batching），不同请求随时加入、随时结束，左 padding 还合适吗？`,
          a: md`不合适：左 padding 要求同批序列一起开始、对齐到同一长度，新请求加入时得重新 pad 整个 batch。vLLM 等系统把所有 token 拼成一维（不 pad），用 \`cu_seqlens\` / 块表描述每个请求的边界，注意力 kernel 按请求读取各自的 KV 块（PagedAttention），位置编号也按请求独立维护。`,
        },
      ],
    },
    {
      id: 'c-split-kv',
      title: '综合：Flash-Decoding 风格的 split-KV 解码与 logsumexp 合并',
      difficulty: 'hard',
      source: '大模型岗手撕 · 推理 kernel 思想',
      timeLimitMin: 30,
      description: md`
        decode 时每个头只有 1 个 query，但 KV 很长。把 KV 沿序列切成若干段并行计算，再合并。实现三个函数：

        \`\`\`python
        def attention_with_lse(q, k, v) -> tuple[np.ndarray, np.ndarray]:
            # q: (H, d)；k, v: (H, S, d) → o: (H, d) 已归一化的输出，lse: (H,) 缩放后分数的 logsumexp
            # S == 0 时返回 o = 0、lse = -inf

        def merge_states(outs, lses) -> tuple[np.ndarray, np.ndarray]:
            # outs: 若干个 (H, d)，lses: 若干个 (H,) → 合并后的 (o, lse)；lse 为 -inf 的段视为空段

        def split_kv_decode(q, k, v, n_splits) -> np.ndarray:
            # 用 np.linspace(0, S, n_splits + 1).astype(int) 切段（可能出现空段），逐段 attention_with_lse，再 merge_states
        \`\`\`

        - 缩放 $1/\sqrt{d}$；不得产生 NaN（包括 \`n_splits > S\` 时的空段、所有段都为空的情况）。
        - 合并公式：$\text{lse} = \log\sum_s e^{\text{lse}_s}$，$o = \sum_s e^{\text{lse}_s - \text{lse}}\, o_s$（需数值稳定）。
      `,
      starter: py`
        import numpy as np

        def attention_with_lse(q, k, v):
            raise NotImplementedError

        def merge_states(outs, lses):
            raise NotImplementedError

        def split_kv_decode(q, k, v, n_splits):
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def attention_with_lse(q, k, v):
            H, d = q.shape
            if k.shape[1] == 0:
                return np.zeros((H, v.shape[-1])), np.full(H, -np.inf)
            s = np.einsum("hd,hsd->hs", q, k) / np.sqrt(d)           # (H, S)
            m = s.max(axis=-1)
            p = np.exp(s - m[:, None])
            l = p.sum(axis=-1)
            o = np.einsum("hs,hsd->hd", p, v) / l[:, None]
            return o, m + np.log(l)

        def merge_states(outs, lses):
            O = np.stack(outs)                                         # (n, H, d)
            Ls = np.stack(lses)                                        # (n, H)
            m = Ls.max(axis=0)                                         # (H,)
            m_safe = np.where(np.isfinite(m), m, 0.0)                  # 全空时避免 -inf - (-inf)
            w = np.exp(Ls - m_safe)                                    # 空段权重 exp(-inf) = 0
            tot = w.sum(axis=0)
            safe_tot = np.where(tot > 0, tot, 1.0)
            o = (w[..., None] * O).sum(axis=0) / safe_tot[:, None]
            lse = np.where(tot > 0, m_safe + np.log(safe_tot), -np.inf)
            return o, lse

        def split_kv_decode(q, k, v, n_splits):
            S = k.shape[1]
            bounds = np.linspace(0, S, n_splits + 1).astype(int)
            outs, lses = [], []
            for a, b in zip(bounds[:-1], bounds[1:]):                  # 各段在 GPU 上由不同 thread block 并行
                o, l = attention_with_lse(q, k[:, a:b], v[:, a:b])
                outs.append(o)
                lses.append(l)
            return merge_states(outs, lses)[0]
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import math
            import torch

            def attention_with_lse(q, k, v):
                s = torch.einsum("hd,hsd->hs", q, k) / math.sqrt(q.size(-1))
                lse = torch.logsumexp(s, dim=-1)                       # 空段时为 -inf
                o = torch.einsum("hs,hsd->hd", torch.exp(s - lse[:, None]), v)
                return torch.nan_to_num(o), lse

            def merge_states(outs, lses):
                O, L = torch.stack(outs), torch.stack(lses)            # (n, H, d), (n, H)
                lse = torch.logsumexp(L, dim=0)
                w = torch.nan_to_num(torch.exp(L - lse))               # 全空时 -inf - (-inf) → NaN → 0
                return (w[..., None] * O).sum(0), lse

            def split_kv_decode(q, k, v, n_splits):
                parts = [attention_with_lse(q, kc, vc) for kc, vc in zip(k.tensor_split(n_splits, 1), v.tensor_split(n_splits, 1))]
                return merge_states([p[0] for p in parts], [p[1] for p in parts])[0]
          `,
        },
      ],
      explanation: md`
        **为什么能合并**：段 $s$ 的部分结果满足 $o_s = \sum_{j\in s} e^{x_j - \text{lse}_s} v_j$。整体 $o = \sum_j e^{x_j - \text{lse}} v_j = \sum_s e^{\text{lse}_s - \text{lse}} \sum_{j\in s} e^{x_j - \text{lse}_s} v_j = \sum_s e^{\text{lse}_s - \text{lse}} o_s$。
        这和 online softmax 的「换基准」是同一件事，只是把顺序累加换成了并行归约。

        **数值稳定**：合并时先减去最大的 $\text{lse}_s$；空段 $\text{lse} = -\infty$ 权重为 0；全空时用安全基准 0，返回 $o = 0$、$\text{lse} = -\infty$。

        **工程意义**：decode 时并行度从 batch × head 扩展到 batch × head × 段数，长上下文、小 batch 时才能占满 SM；同一套合并逻辑也用于 Ring Attention（跨 GPU 的 KV 分段）与前缀缓存的级联注意力（共享前缀一段 + 各请求私有后缀一段）。
      `,
      complexity: { time: 'O(H·S·d)', space: 'O(n_splits·H·d)' },
      hints: [
        md`\`attention_with_lse\`：\`s = einsum("hd,hsd->hs") / sqrt(d)\`，\`m = s.max(-1)\`，\`l = exp(s - m).sum(-1)\`，\`lse = m + log(l)\`。`,
        md`\`merge_states\`：\`m = max(lses)\`，权重 \`w = exp(lses - m)\`，\`o = Σ w·o_s / Σ w\`。`,
        md`全部为 \`-inf\` 时 \`m\` 也是 \`-inf\`：先 \`m_safe = where(isfinite(m), m, 0)\`，分母为 0 时除以 1。`,
      ],
      setup: py`
        import numpy as np, inspect

        def _naive(q, k, v):
            s = np.einsum("hd,hsd->hs", q, k) / np.sqrt(q.shape[1])
            m = s.max(-1, keepdims=True)
            p = np.exp(s - m)
            lse = (m[:, 0] + np.log(p.sum(-1)))
            return np.einsum("hs,hsd->hd", p / p.sum(-1, keepdims=True), v), lse

        def _qkv(seed, H, S, d, scale=1.0):
            r = np.random.default_rng(seed)
            return r.normal(size=(H, d)) * scale, r.normal(size=(H, S, d)) * scale, r.normal(size=(H, S, d))
      `,
      tests: [
        {
          name: 'attention_with_lse',
          code: py`
            q, k, v = _qkv(0, 3, 9, 4)
            o, l = attention_with_lse(q, k, v)
            eo, el = _naive(q, k, v)
            assert o.shape == (3, 4) and l.shape == (3,), f"形状错误：o {o.shape}，lse {l.shape}"
            assert np.allclose(o, eo, atol=1e-10) and np.allclose(l, el, atol=1e-10), "o 或 lse 与朴素实现不一致"
          `,
        },
        {
          name: '两段合并 = 整体',
          code: py`
            q, k, v = _qkv(1, 2, 10, 4)
            a = attention_with_lse(q, k[:, :4], v[:, :4])
            b = attention_with_lse(q, k[:, 4:], v[:, 4:])
            o, l = merge_states([a[0], b[0]], [a[1], b[1]])
            eo, el = _naive(q, k, v)
            assert np.allclose(o, eo, atol=1e-10), f"合并输出最大误差 {np.abs(o - eo).max():.2e}"
            assert np.allclose(l, el, atol=1e-10), "合并后的 lse 不正确"
          `,
        },
        {
          name: 'split_kv_decode 多种段数',
          code: py`
            q, k, v = _qkv(2, 4, 37, 8)
            eo, _ = _naive(q, k, v)
            for n in (1, 2, 5, 8):
                got = split_kv_decode(q, k, v, n)
                assert np.allclose(got, eo, atol=1e-10), f"n_splits={n}：最大误差 {np.abs(got - eo).max():.2e}"
          `,
        },
        {
          name: '空段与大数值',
          hidden: true,
          code: py`
            q, k, v = _qkv(3, 2, 3, 4, scale=15.0)
            got = split_kv_decode(q, k, v, 7)              # 7 段只有 3 个 key，必有空段
            assert np.all(np.isfinite(got)), "出现 NaN/inf：空段没有处理好"
            assert np.allclose(got, _naive(q, k, v)[0], atol=1e-9)
          `,
        },
        {
          name: '全空合并不产生 NaN',
          hidden: true,
          code: py`
            e = attention_with_lse(np.ones((2, 3)), np.zeros((2, 0, 3)), np.zeros((2, 0, 3)))
            assert np.all(e[1] == -np.inf) and np.allclose(e[0], 0), "S=0 时应返回 o=0、lse=-inf"
            o, l = merge_states([e[0], e[0]], [e[1], e[1]])
            assert np.all(np.isfinite(o)) and np.allclose(o, 0), "全空合并的输出应为 0"
            assert np.all(l == -np.inf), "全空合并的 lse 应为 -inf"
          `,
        },
        {
          name: '确实分段计算',
          hidden: true,
          code: py`
            src = inspect.getsource(split_kv_decode)
            assert "merge_states" in src and "attention_with_lse" in src, "split_kv_decode 应逐段调用 attention_with_lse 并用 merge_states 合并"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-cascade',
          q: md`100 个请求共享同一段 4K token 的系统提示词，各自有不同的后缀。怎么用合并公式省带宽？`,
          a: md`级联（cascade）注意力：把共享前缀的 KV 只存一份，对所有请求的 query 一起算「前缀段」注意力（多个 query 共享同一份 KV，算术强度高），再对各自后缀单独算，最后用 logsumexp 合并两段。共享前缀只读一次而不是 100 次。前缀缓存（RadixAttention 等）解决「存一份」，级联注意力解决「读一次」。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-design',
      freq: 3,
      q: md`如果让你为一个要服务 128K 上下文的新模型选注意力方案，你会怎么考虑？`,
      a: md`
        **结论**：先算 KV cache 预算，再在 GQA / MLA / 稀疏或混合注意力之间权衡，kernel 层面默认 FlashAttention 系列 + split-KV decode。

        **推导**：以 60 层、BF16 为例，MHA（64 头 × 128）每 token $2\times60\times8192\times2 \approx 1.97$ MB，128K 约 258 GB（240 GiB）；GQA-8 降到 1/8；MLA（576 维）约 69 KB/token、128K 约 8.4 GiB。还要考虑：GQA 实现简单、TP 友好；MLA 缓存最小但 TP 时缓存要复制、prefill/decode 两套计算路径；超长上下文可再叠加滑窗或稀疏注意力（DSA、NSA）、线性注意力混合层（Qwen3-Next、Kimi Linear 等），见下一章。

        **延伸**：FP8 KV cache 再省一半；PagedAttention + 前缀缓存解决碎片与共享；PD 分离把 prefill（计算密集）和 decode（访存密集）放到不同机器上。
      `,
    },
    {
      id: 'qa-chain',
      freq: 2,
      q: md`把「为什么除以 √d」「online softmax」「split-KV 合并」串成一个统一的数学事实。`,
      a: md`
        **结论**：softmax 对 logits 的平移不变，且指数和可以换基准：$\sum_j e^{x_j - a} = e^{b - a}\sum_j e^{x_j - b}$。

        **展开**：减最大值保证数值稳定（平移不变）；online softmax 在最大值变化时乘 $e^{m_{\text{old}} - m_{\text{new}}}$（换基准）；split-KV / Ring Attention / 级联注意力用各段 logsumexp 加权合并（多基准统一到全局基准）；FlashAttention 反向只存 logsumexp，$P = e^{S - L}$ 直接得到归一化概率。√d 缩放则是另一回事：控制 logits 的**尺度**（方差），而不是平移。
      `,
    },
  ],

  cheatsheet: md`
    | 场景 | 必背 |
    |---|---|
    | 缩放 | $1/\sqrt{d_h}$；MLA 为 $1/\sqrt{192}$ |
    | 掩码对齐 | \`tril(ones((L,S)), k=S-L)\`；左 padding 位置 \`cumsum(mask)-1\` |
    | KV 显存 | $2LBSn_{kv}d_h\cdot$bytes；MLA $L\cdot576\cdot$bytes/token |
    | GQA | \`np.repeat(k, n_rep, axis=1)\`，缓存存 $n_{kv}$ 个头 |
    | online | $\alpha=e^{m-m'}$，$\ell'=\alpha\ell+\sum e^{x-m'}$ |
    | 合并 | $\text{lse}=\log\sum e^{\text{lse}_s}$，$o=\sum e^{\text{lse}_s-\text{lse}}o_s$ |
    | 全遮行 | 最大值为 $-\infty$ 时换成 0、分母为 0 时除以 1 |
  `,
})
