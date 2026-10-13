/**
 * L3-3 KV Cache 与 MQA/GQA：增量解码、显存公式、repeat_kv、decode 为何 memory-bound。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { KVCacheDecodeViz } from '../viz/KVCacheDecodeViz'
import { KVMemoryCalcViz } from '../viz/KVMemoryCalcViz'
import { GQAGroupingViz } from '../viz/GQAGroupingViz'

export default defineLevel({
  learn: [
    {
      id: 'incremental',
      title: '增量解码：为什么只缓存 K 和 V',
      body: md`
        自回归生成第 $t$ 个 token 时，因果注意力只需要**最后一个位置**的输出：

        $$
        o_t = \sum_{j \le t} \operatorname{softmax}_j\!\Big(\frac{q_t \cdot k_j}{\sqrt{d_h}}\Big)\, v_j
        $$

        这里用到当前的 $q_t$，以及**所有历史位置**的 $k_j, v_j$。而因果掩码保证历史位置的 $k_j, v_j$ 只依赖 $j$ 之前的 token——
        生成新 token 不会改变它们。所以把每层的 $K, V$ 存下来，下一步只需为新 token 计算一行 $k_t, v_t$ 追加进去。

        - **为什么不缓存 Q**：$q_j$（$j<t$）只在算 $o_j$ 时用过，之后再也不需要。
        - **为什么不缓存注意力权重**：新 token 加入后，旧 query 的权重不会变，但我们也不再需要旧 query 的输出。

        推理因此分成两个阶段：**prefill**（一次性并行处理整个 prompt，计算密集）和 **decode**（每步 1 个 token，读全部缓存，访存密集）。
      `,
      viz: KVCacheDecodeViz,
      after: md`
        > [!NOTE] 省了多少
        > 生成 $n$ 个 token，无缓存时 K/V 投影共 $O(n^2)$ 行、注意力 $O(n^3)$ 次点积；有缓存时分别是 $O(n)$ 与 $O(n^2)$。
        > 代价是显存：缓存随序列长度线性增长，长上下文、大 batch 时往往比模型权重还大。

        > [!WARN] 两个实现细节
        > 1. decode 时 query 长度 $L=1$、key 长度 $S=t$，因果掩码要**右下角对齐**（单 token 时其实不需要掩码）；
        > 2. 位置编码（RoPE）要用新 token 的**绝对位置** $t-1$，而不是 0——忘记加 offset 是增量解码与全量前向对不上的头号原因。
      `,
      checkpoint: {
        id: 'cp-inc',
        type: 'multi',
        prompt: md`带 KV cache 的 decode 阶段，每生成一个 token，每一层需要**新计算**哪些量？（多选）`,
        options: ['新 token 的 $q_t$', '新 token 的 $k_t, v_t$', '所有历史 token 的 $k_j, v_j$', '$q_t$ 对全部 $t$ 个 key 的注意力权重', '所有历史 token 的注意力输出'],
        answer: [0, 1, 3],
        explanation: md`历史 $k_j, v_j$ 直接从缓存读取（C 错）；历史 token 的输出早已用过，不再需要（E 错）。新 token 需要自己的 q、k、v，以及一行长度为 $t$ 的注意力。`,
      },
    },
    {
      id: 'memory',
      title: 'KV 显存公式：面试必算',
      body: md`
        每层每个 token 要存 K 和 V 各 $n_{kv} \times d_h$ 个元素，所以

        $$
        \text{KV bytes} = 2 \times L \times B \times S \times n_{kv} \times d_h \times \text{bytes per elem}
        $$

        | 模型（公开配置） | 层数 $L$ | Q 头 / KV 头 | $d_h$ | 每 token（BF16） |
        |---|---|---|---|---|
        | Llama-2-7B（MHA） | 32 | 32 / 32 | 128 | 512 KiB |
        | Llama-3-8B（GQA） | 32 | 32 / 8 | 128 | 128 KiB |
        | Llama-3-70B（GQA） | 80 | 64 / 8 | 128 | 320 KiB |
        | Qwen2.5-7B（GQA） | 28 | 28 / 4 | 128 | 56 KiB |

        例：Llama-3-8B 单条 8K 上下文 $= 2 \times 32 \times 8192 \times 8 \times 128 \times 2 = 2^{30}$ 字节 $=1$ GiB；同样的设置用 MHA 的 Llama-2-7B 是 4 GiB。
        批量服务时乘以 batch：Llama-2-7B 跑 batch 32 × 4K 上下文就要 64 GiB——比 13.5 GB 左右的 FP16 权重大好几倍。

        用下面的计算器验证这些数字，并切到 DeepSeek-V3 看 MLA 的量级。
      `,
      viz: KVMemoryCalcViz,
      after: md`
        > [!INTERVIEW] 计算题答题模板
        > 先写公式，再逐项代入并说明单位（GB = $10^9$ 还是 GiB = $2^{30}$），最后给量级判断：「这已经超过单卡 80GB 显存，需要 GQA / KV 量化 / PagedAttention / 卸载」。
        > 降低 KV 显存的四条路：**减 KV 头**（MQA/GQA）、**压缩维度**（MLA）、**降精度**（FP8/INT4 KV cache）、**减长度**（滑窗、稀疏、驱逐）。
      `,
      checkpoint: {
        id: 'cp-mem',
        type: 'numeric',
        prompt: md`Llama-3-70B（80 层、8 个 KV 头、$d_h=128$）、BF16，**每个 token** 的 KV cache 是多少 KiB（$1\text{ KiB} = 1024$ 字节）？`,
        answer: 320,
        tolerance: 0,
        unit: 'KiB',
        explanation: md`$2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ 字节 $= 320$ KiB。Q 头数 64 不进入公式——这是最常见的失分点。`,
      },
    },
    {
      id: 'gqa',
      title: 'MQA 与 GQA：让多个 query 头共享 K/V',
      body: md`
        KV cache 与 **KV 头数**成正比，于是有了：

        - **MQA**（Shazeer, 2019）：所有 query 头共享**一组** K/V，$n_{kv} = 1$。缓存缩小 $h$ 倍，但质量可能下降、训练不够稳定。
        - **GQA**（Ainslie et al., 2023）：折中方案，把 $h$ 个 query 头分成 $g$ 组，每组共享一组 K/V，$n_{kv} = g$。
          MHA（$g=h$）和 MQA（$g=1$）是两个特例。论文报告 GQA 质量接近 MHA、速度接近 MQA，并给出了把 MHA 检查点「uptrain」成 GQA 的配方：
          K/V 投影按组**均值池化**，再用约原预训练 **5%** 的算力继续训练。

        实现上 \`W_K, W_V\` 的形状从 $D \times D$ 变成 $D \times (n_{kv} d_h)$。计算注意力前用 **repeat_kv** 把 KV 头展开到 $h$ 个：
        query 头 $i$ 使用 KV 头 $\lfloor i / n_{\text{rep}} \rfloor$，$n_{\text{rep}} = h / n_{kv}$。

        \`\`\`python
        def repeat_kv(x, n_rep):              # (B, n_kv, S, d) -> (B, n_kv*n_rep, S, d)
            return np.repeat(x, n_rep, axis=1)    # 等价于 torch.repeat_interleave(x, n_rep, dim=1)
        \`\`\`

        **缓存里存的是展开之前的 $n_{kv}$ 个头**——展开只在计算时进行（高效 kernel 甚至不展开，直接让多个 query 头读同一份 K/V）。
      `,
      viz: GQAGroupingViz,
      after: md`
        > [!WARN] \`np.repeat\` vs \`np.tile\`
        > \`np.repeat(x, 2, axis=1)\` 得到 KV 头顺序 \`[0, 0, 1, 1, …]\`，\`np.tile\` 得到 \`[0, 1, 0, 1, …]\`。HF 的 \`repeat_kv\` 与 PyTorch SDPA 的 \`enable_gqa\` 都是前者（相邻的 query 头为一组）。
        > 用错了不会报错，但每个 query 头配上了别组的 K/V，加载预训练权重后输出完全错误。

        > [!NOTE] GQA 为什么还能加速 decode
        > decode 时每步要把整份 KV cache 从 HBM 读一遍，而每个元素只参与很少的计算（MHA 下每个 KV 元素只服务 1 个 query 头），是典型的 **memory-bound**。
        > GQA 下读进来的一份 K/V 服务 $n_{\text{rep}}$ 个 query 头，算术强度提高 $n_{\text{rep}}$ 倍，同时要读的字节数减少同样的倍数。
      `,
      checkpoint: {
        id: 'cp-gqa',
        type: 'single',
        prompt: md`GQA：32 个 query 头、8 个 KV 头，按 repeat_interleave 约定，query 头 13 使用哪个 KV 头？`,
        options: ['1', '3', '5', '13'],
        answer: 1,
        explanation: md`$n_{\text{rep}} = 32/8 = 4$，KV 头 $= \lfloor 13/4 \rfloor = 3$。选 5 的是 \`13 % 8\`（np.tile 的约定）。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-why-no-q',
      type: 'single',
      difficulty: 1,
      prompt: md`KV cache 为什么不缓存 Q？`,
      options: [
        'Q 太大，放不下',
        '历史位置的 query 只在计算它自己那一步的输出时用到，之后再也不需要',
        'Q 每一步都会因为新 token 的加入而改变',
        'Q 可以由 K 推出来',
      ],
      answer: 1,
      explanation: md`第 $t$ 步只需要 $q_t$；历史 $q_j$ 对应的输出 $o_j$ 早已算完。C 是常见误解：因果注意力下历史位置的 q/k/v 都不会因后续 token 改变——这正是 K、V 能被缓存的前提。Q 与 K 大小相同（A 错）。`,
    },
    {
      id: 'q-l3-8b',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Llama-3-8B：32 层、32 个 query 头、8 个 KV 头、$d_h = 128$。BF16 下 batch 1、上下文 8192 的 KV cache 是多少 GiB（$2^{30}$ 字节）？`,
      answer: 1,
      tolerance: 0.01,
      unit: 'GiB',
      explanation: md`$2 \times 32 \times 1 \times 8192 \times 8 \times 128 \times 2 = 2^{1+5+13+3+7+1} = 2^{30}$ 字节 $= 1$ GiB。若误用 32 个 query 头会得到 4 GiB。`,
    },
    {
      id: 'q-l2-7b',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Llama-2-7B（32 层、32 头 MHA、$d_h=128$）FP16，batch 8、每条 4096 token，KV cache 共多少 GiB？`,
      answer: 16,
      tolerance: 0.01,
      unit: 'GiB',
      explanation: md`$2 \times 32 \times 8 \times 4096 \times 32 \times 128 \times 2 = 17{,}179{,}869{,}184$ 字节 $= 16$ GiB，比约 13.5GB 的 FP16 权重还大。这就是 Llama 2 的 70B 版本与 Llama 3 全系改用 GQA 的原因。`,
    },
    {
      id: 'q-prefill',
      type: 'judge',
      difficulty: 1,
      prompt: md`KV cache 能显著减少 prefill 阶段的计算量。`,
      answer: false,
      explanation: md`prefill 时 prompt 的所有 K/V 本来就只需算一次（一次并行前向），KV cache 只是把它们**存下来**。省计算发生在 decode 阶段：避免每步重算整个前缀。（多轮对话 / 共享前缀场景下复用**之前请求**的缓存——prefix caching——才能省 prefill。）`,
    },
    {
      id: 'q-uptrain',
      type: 'judge',
      difficulty: 2,
      prompt: md`把 MHA 检查点的 K/V 投影按组均值池化成 GQA 后，不需要任何继续训练就能达到接近原模型的质量。`,
      answer: false,
      explanation: md`GQA 论文的配方是「均值池化初始化 + uptraining」，后者用了约原预训练 5% 的算力。直接池化不训练会有明显的质量损失。`,
    },
    {
      id: 'q-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 MQA / GQA，下列说法正确的是：（多选）`,
      options: [
        'GQA 中 $W_K$ 的形状是 $D \\times (n_{kv} d_h)$',
        'KV cache 中应保存 repeat_kv 展开之后的 $h$ 个头，避免每步重复展开',
        'MHA 与 MQA 都是 GQA 的特例',
        'GQA 减少了 KV cache，但每个 query 头的注意力计算量（FLOPs）基本不变',
        'GQA 会减少 query 头的数量',
      ],
      answer: [0, 2, 3],
      explanation: md`B 错：缓存展开后的张量就失去了节省显存的意义，应该缓存 $n_{kv}$ 个头、计算时再展开（或在 kernel 内共享读取）。E 错：query 头数不变，变的是 KV 头数。D 对：每个 query 头仍要对全部 key 打分并加权求和，省的是访存与显存。`,
    },
    {
      id: 'q-repeat',
      type: 'fill',
      difficulty: 2,
      prompt: md`补全 repeat_kv（\`x\` 形状 \`(B, n_kv, S, d)\`，结果中 query 头 $i$ 对应 KV 头 $\lfloor i/n_{\text{rep}}\rfloor$）：\`return np.____(1)____(x, n_rep, axis=____(2)____)\``,
      blanks: [
        { label: '(1)', answers: ['repeat'] },
        { label: '(2)', answers: ['1'], numeric: true },
      ],
      explanation: md`\`np.repeat\` 在第 1 维（头维）上把每个 KV 头连续复制 $n_{\text{rep}}$ 次，得到 \`[0,0,…,1,1,…]\`。\`np.tile\` 得到交错顺序，与 HF / PyTorch 的约定不符。`,
    },
    {
      id: 'q-tile',
      type: 'single',
      difficulty: 3,
      prompt: md`把加载了 Llama-3 权重的 GQA 实现中的 \`np.repeat(k, 4, axis=1)\` 换成 \`np.tile(k, (1, 4, 1, 1))\`，会发生什么？`,
      options: [
        '形状不匹配，直接报错',
        '不报错，结果完全一样，因为所有 KV 头最终都会被用到',
        '不报错，但大多数 query 头配上了别组的 K/V，输出与原模型严重不一致',
        '只有第 0 个 query 头受影响',
      ],
      answer: 2,
      explanation: md`两者形状相同。tile 下 query 头 $i$ 使用 KV 头 $i \bmod n_{kv}$，而权重训练时用的是 $\lfloor i / n_{\text{rep}} \rfloor$。除了少数巧合重合的头（例如头 0 两种约定都对应 KV 头 0），其余全部错配。D 也不对：受影响的远不止一个头。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 1,
      prompt: md`把带 KV cache 的单步 decode（某一层）排成正确顺序：`,
      items: [
        '对新 token 的隐状态计算 q、k、v',
        '对 q、k 施加 RoPE（位置 = 当前绝对位置）',
        '把新的 k、v 追加到该层缓存',
        '用 q 与缓存中全部 key 打分并 softmax',
        '对缓存中的 value 加权求和，再经 W_O 输出',
      ],
      explanation: md`RoPE 必须在写入缓存**之前**作用于 k（缓存里存的是已旋转的 k），且位置是绝对位置。先追加再打分，新 token 才能看到自己。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把注意力变体与「每层每 token 的 KV cache 元素数」配对（$h$ 个头、头维 $d_h$、$g$ 组）：`,
      pairs: [
        ['MHA', '$2 h d_h$'],
        ['GQA', '$2 g d_h$'],
        ['MQA', '$2 d_h$'],
        ['MLA（DeepSeek-V2/V3）', '$d_c + d_h^R = 576$'],
      ],
      explanation: md`前三者都存 K 和 V 两份；MLA 只存一个联合压缩的潜向量 $c^{KV}$（512 维）与一个所有头共享的 RoPE key（64 维），DeepSeek-V2 论文称其相当于只有 2.25 组的 GQA。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 1,
      prompt: md`8 个 query 头、2 个 KV 头的 GQA 中，哪些 query 头共享 **KV 头 1**？`,
      items: ['Q0', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'],
      answer: [4, 5, 6, 7],
      explanation: md`$n_{\text{rep}} = 4$，KV 头 1 服务 $\lfloor i/4 \rfloor = 1$ 的头，即 Q4–Q7。若选了 Q1、Q3、Q5、Q7，就是 \`np.tile\` 的交错约定。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面是 MHA 单 token decode 的一步（缓存形状 \`(B, h, t, d_h)\`）。点出 **2** 处 bug：`,
      code: py`
        def decode_step(x_t, k_cache, v_cache, Wq, Wk, Wv, n_heads):
            # x_t: (B, 1, D)
            B, _, D = x_t.shape
            d_h = D // n_heads
            split = lambda a: a.reshape(B, 1, n_heads, d_h).transpose(0, 2, 1, 3)
            q, k, v = split(x_t @ Wq), split(x_t @ Wk), split(x_t @ Wv)
            k_cache = np.concatenate([k_cache, k], axis=1)
            v_cache = np.concatenate([v_cache, v], axis=2)
            s = q @ k_cache.transpose(0, 1, 3, 2) / np.sqrt(d_h)
            T = s.shape[-1]
            s = np.where(np.tril(np.ones((1, T), dtype=bool)), s, -np.inf)
            p = softmax(s)
            return p @ v_cache, k_cache, v_cache
      `,
      bugLines: [7, 11],
      explanation: md`第 7 行：缓存的序列维是第 2 维，应 \`axis=2\`；沿头维拼接会让「头数」增加。第 11 行：\`tril(ones((1, T)))\` 是左上角对齐的因果掩码，单个新 token 只能看到第 0 个 key；decode 时新 token 本来就能看全部历史，应去掉掩码或用 \`tril(..., k=T-1)\`。`,
    },
  ],

  guided: [
    {
      id: 'g-gqa',
      title: '一步步写出 GQA 注意力（支持 decode）',
      intro: md`
        实现 \`gqa_attention(q, k, v, causal=True)\`：
        - \`q\`: \`(B, Hq, T, d)\`；\`k, v\`: \`(B, Hkv, S, d)\`，\`Hq\` 是 \`Hkv\` 的整数倍；
        - $T \le S$：query 是序列最后 $T$ 个位置（decode 时 $T=1$），因果掩码右下角对齐；
        - 返回 \`(B, Hq, T, d)\`。
      `,
      template: py`
        import numpy as np

        def gqa_attention(q, k, v, causal=True):
            B, Hq, T, d = q.shape
            Hkv, S = k.shape[1], k.shape[2]
            n_rep = {{1}}
            k = {{2}}
            v = np.repeat(v, n_rep, axis=1)
            scores = q @ k.swapaxes(-1, -2) / np.sqrt(d)
            if causal:
                mask = {{3}}
                scores = np.where(mask, scores, -np.inf)
            w = np.exp(scores - scores.max(axis=-1, keepdims=True))
            w = {{4}}
            return w @ v
      `,
      steps: [
        {
          blank: 1,
          instruction: md`每个 KV 头要服务多少个 query 头？`,
          answer: 'Hq // Hkv',
          accept: ['q.shape[1] // k.shape[1]'],
          explain: md`$n_{\text{rep}} = H_q / H_{kv}$；MHA 时为 1，MQA 时为 $H_q$。`,
        },
        {
          blank: 2,
          instruction: md`把 KV 头沿头维展开到 \`Hq\` 个，query 头 $i$ 对应 KV 头 $\lfloor i/n_{\text{rep}} \rfloor$。`,
          answer: 'np.repeat(k, n_rep, axis=1)',
          accept: ['k.repeat(n_rep, axis=1)'],
          hint: 'repeat 与 tile 的顺序不同。',
          explain: md`\`np.repeat\` 得到 \`[0,0,1,1,…]\`，与 HF \`repeat_kv\` / \`torch.repeat_interleave\` 一致。测试会用不对称的 KV 头检查顺序。`,
        },
        {
          blank: 3,
          instruction: md`构造右下角对齐的因果掩码，形状 \`(T, S)\`。`,
          answer: 'np.tril(np.ones((T, S), dtype=bool), k=S - T)',
          accept: ['np.tril(np.ones((T, S), bool), S - T)', 'np.tril(np.ones((T, S), dtype=bool), S - T)'],
          explain: md`query $i$ 的绝对位置是 $S-T+i$，可以看 $j \le S-T+i$。$T=S$ 时退化为普通下三角，$T=1$ 时全为 True。`,
        },
        {
          blank: 4,
          instruction: md`归一化得到注意力权重。`,
          answer: 'w / w.sum(axis=-1, keepdims=True)',
          accept: ['w / np.sum(w, axis=-1, keepdims=True)', 'w / w.sum(-1, keepdims=True)'],
          explain: md`沿 key 维归一化。`,
        },
      ],
      setup: py`
        import numpy as np

        def _ref_gqa(q, k, v, causal=True):
            B, Hq, T, d = q.shape
            Hkv, S = k.shape[1], k.shape[2]
            rep = Hq // Hkv
            out = np.zeros((B, Hq, T, v.shape[-1]))
            for b in range(B):
                for i in range(Hq):
                    j = i // rep
                    s = q[b, i] @ k[b, j].T / np.sqrt(d)
                    if causal:
                        s = np.where(np.tril(np.ones((T, S), bool), S - T), s, -np.inf)
                    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
                    out[b, i] = p @ v[b, j]
            return out
      `,
      tests: [
        {
          name: 'GQA：Hq=4, Hkv=2, T=S=5',
          code: py`
            r = np.random.default_rng(0)
            q, k, v = r.normal(size=(1, 4, 5, 8)), r.normal(size=(1, 2, 5, 8)), r.normal(size=(1, 2, 5, 8))
            got = gqa_attention(q, k, v)
            assert got.shape == (1, 4, 5, 8), f"形状应为 (1, 4, 5, 8)，得到 {got.shape}"
            assert np.allclose(got, _ref_gqa(q, k, v), atol=1e-8), "结果不一致：检查 KV 头展开顺序"
          `,
        },
        {
          name: 'decode：T=1, S=7',
          code: py`
            r = np.random.default_rng(1)
            q, k, v = r.normal(size=(2, 6, 1, 4)), r.normal(size=(2, 3, 7, 4)), r.normal(size=(2, 3, 7, 4))
            assert np.allclose(gqa_attention(q, k, v), _ref_gqa(q, k, v), atol=1e-8), "decode 时新 token 应能看到全部 7 个 key"
          `,
        },
        {
          name: 'MQA + chunk（T=3, S=8）',
          hidden: true,
          code: py`
            r = np.random.default_rng(2)
            q, k, v = r.normal(size=(1, 8, 3, 4)), r.normal(size=(1, 1, 8, 4)), r.normal(size=(1, 1, 8, 4))
            assert np.allclose(gqa_attention(q, k, v), _ref_gqa(q, k, v), atol=1e-8)
          `,
        },
      ],
      outro: md`
        高效实现可以不真的复制 K/V：把 q reshape 成 \`(B, Hkv, n_rep, T, d)\`，与 \`(B, Hkv, 1, S, d)\` 的 k 广播相乘即可——省掉 \`n_rep\` 倍的临时显存。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-gqa',
      title: '手撕 GQA 前向（含 repeat_kv）',
      difficulty: 'medium',
      source: '大模型岗手撕 · GQA',
      timeLimitMin: 20,
      description: md`
        实现两个函数：

        \`\`\`python
        def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
            # (B, n_kv, S, d) -> (B, n_kv * n_rep, S, d)，结果中第 i 个头 = 输入第 i // n_rep 个头

        def gqa_forward(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads) -> np.ndarray:   # (B, T, D)
        \`\`\`

        - \`x\`: \`(B, T, D)\`；\`d_h = D // n_heads\`；\`Wq, Wo\`: \`(D, D)\`；\`Wk, Wv\`: \`(D, n_kv_heads * d_h)\`。
        - 因果自注意力，缩放 $1/\sqrt{d_h}$，无 bias。
        - 第 $i$ 个 query 头使用第 $\lfloor i / n_{\text{rep}} \rfloor$ 个 KV 头。

        **示例**：\`n_kv_heads == n_heads\` 时就是 MHA；\`n_kv_heads == 1\` 时就是 MQA。
      `,
      starter: py`
        import numpy as np

        def repeat_kv(x, n_rep):
            # TODO
            raise NotImplementedError

        def gqa_forward(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
            # TODO: q 拆成 n_heads 个头，k/v 拆成 n_kv_heads 个头，repeat_kv 后做因果注意力
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def repeat_kv(x, n_rep):
            """(B, n_kv, S, d) -> (B, n_kv*n_rep, S, d)，等价于 torch.repeat_interleave(x, n_rep, dim=1)"""
            if n_rep == 1:
                return x
            B, H, S, d = x.shape
            return np.broadcast_to(x[:, :, None], (B, H, n_rep, S, d)).reshape(B, H * n_rep, S, d)

        def gqa_forward(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
            B, T, D = x.shape
            d_h = D // n_heads
            split = lambda t, h: t.reshape(B, T, h, d_h).transpose(0, 2, 1, 3)
            q = split(x @ Wq, n_heads)                       # (B, h, T, d_h)
            k = split(x @ Wk, n_kv_heads)                    # (B, n_kv, T, d_h)
            v = split(x @ Wv, n_kv_heads)
            n_rep = n_heads // n_kv_heads
            k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)  # (B, h, T, d_h)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_h)
            s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            o = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
            return o @ Wo
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，与 HF Llama 一致）',
          code: py`
            import torch
            import torch.nn.functional as F

            def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
                B, H, S, d = x.shape
                if n_rep == 1:
                    return x
                return x[:, :, None].expand(B, H, n_rep, S, d).reshape(B, H * n_rep, S, d)

            def gqa_forward(x, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
                B, T, D = x.shape
                d_h = D // n_heads
                q = (x @ Wq).view(B, T, n_heads, d_h).transpose(1, 2)
                k = (x @ Wk).view(B, T, n_kv_heads, d_h).transpose(1, 2)
                v = (x @ Wv).view(B, T, n_kv_heads, d_h).transpose(1, 2)
                k, v = repeat_kv(k, n_heads // n_kv_heads), repeat_kv(v, n_heads // n_kv_heads)
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                # PyTorch ≥ 2.5 也可不展开：F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                return o.transpose(1, 2).reshape(B, T, D) @ Wo
          `,
        },
      ],
      explanation: md`
        **要点**：q 拆成 \`n_heads\` 个头、k/v 拆成 \`n_kv_heads\` 个头（\`Wk\` 的输出维只有 \`n_kv_heads * d_h\`）；用 repeat_kv 展开后与 MHA 完全相同。

        **repeat_kv 的两种写法**：\`np.repeat(x, n_rep, axis=1)\`，或「插入新维 → broadcast → reshape」（与 HF 的 \`expand + reshape\` 相同，broadcast 本身不拷贝）。

        **复杂度**：与 MHA 相同的 $O(BT^2D)$ 注意力计算；节省的是 K/V 投影参数（$2D \cdot n_{kv} d_h$）和缓存。

        **易错点**：用 \`np.tile\`（顺序错）；用 \`D // n_kv_heads\` 当作 KV 的头维（头维始终是 \`D // n_heads\`）。
      `,
      complexity: { time: 'O(B·T·D² + B·T²·D)', space: 'O(B·h·T²)' },
      hints: [
        md`KV 的头维和 Q 一样是 \`d_h = D // n_heads\`，只是头数少：\`(x @ Wk).reshape(B, T, n_kv_heads, d_h)\`。`,
        md`repeat_kv：\`np.repeat(x, n_rep, axis=1)\`。`,
        md`展开后就是普通的因果 MHA。`,
      ],
      setup: py`
        import numpy as np

        def _ref_gqa_fwd(x, Wq, Wk, Wv, Wo, h, hkv):
            B, T, D = x.shape
            dh = D // h
            rep = h // hkv
            out = np.zeros((B, T, D))
            for b in range(B):
                heads = []
                for i in range(h):
                    j = i // rep
                    q = x[b] @ Wq[:, i * dh:(i + 1) * dh]
                    k = x[b] @ Wk[:, j * dh:(j + 1) * dh]
                    v = x[b] @ Wv[:, j * dh:(j + 1) * dh]
                    s = q @ k.T / np.sqrt(dh)
                    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
                    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
                    heads.append(p @ v)
                out[b] = np.concatenate(heads, -1) @ Wo
            return out

        def _w(seed, D, h, hkv):
            r = np.random.default_rng(seed)
            dh = D // h
            return (r.normal(size=(D, D)) / np.sqrt(D), r.normal(size=(D, hkv * dh)) / np.sqrt(D),
                    r.normal(size=(D, hkv * dh)) / np.sqrt(D), r.normal(size=(D, D)) / np.sqrt(D))

        def _cmp(seed, B, T, D, h, hkv):
            x = np.random.default_rng(seed + 50).normal(size=(B, T, D))
            W = _w(seed, D, h, hkv)
            got = gqa_forward(x, *W, h, hkv)
            exp = _ref_gqa_fwd(x, *W, h, hkv)
            assert got.shape == (B, T, D), f"形状应为 {(B, T, D)}，得到 {got.shape}"
            err = np.abs(got - exp).max()
            assert err < 1e-8, f"n_heads={h}, n_kv={hkv}：最大误差 {err:.2e}"
      `,
      tests: [
        {
          name: 'repeat_kv 顺序',
          code: py`
            x = np.arange(2 * 3 * 1 * 1).reshape(2, 3, 1, 1).astype(float)
            got = repeat_kv(x, 2)
            assert got.shape == (2, 6, 1, 1), f"形状应为 (2, 6, 1, 1)，得到 {got.shape}"
            assert got[0, :, 0, 0].tolist() == [0, 0, 1, 1, 2, 2], f"头顺序应为 [0,0,1,1,2,2]，得到 {got[0, :, 0, 0].tolist()}"
            assert repeat_kv(x, 1).shape == x.shape
          `,
        },
        { name: 'GQA：8 头 / 2 KV 头', code: py`_cmp(0, 1, 6, 32, 8, 2)` },
        { name: 'MHA 特例（n_kv = n_heads）', code: py`_cmp(1, 2, 5, 16, 4, 4)` },
        { name: 'MQA 特例（n_kv = 1）', hidden: true, code: py`_cmp(2, 2, 7, 24, 6, 1)` },
        { name: '随机配置', hidden: true, code: py`_cmp(3, 3, 9, 48, 12, 3)` },
      ],
      followUps: [
        {
          id: 'f-noexpand',
          q: md`不想真的复制 K/V（省临时显存），怎么写？`,
          a: md`把 q reshape 成 \`(B, n_kv, n_rep, T, d_h)\`，k、v 变成 \`(B, n_kv, 1, S, d_h)\`，广播做矩阵乘：\`s = q @ k.swapaxes(-1, -2)\` 得 \`(B, n_kv, n_rep, T, S)\`，输出再 reshape 回 \`(B, h, T, d_h)\`。FlashAttention 等 kernel 则直接让同组的 query 头读取同一块 K/V。`,
        },
      ],
    },
    {
      id: 'c-kvcache',
      title: '带 KV cache 的增量解码：与全量前向逐位一致',
      difficulty: 'hard',
      source: '大模型岗手撕 · KV Cache',
      timeLimitMin: 30,
      description: md`
        实现一个带 KV cache 的 GQA 注意力层：

        \`\`\`python
        class KVCacheAttention:
            def __init__(self, Wq, Wk, Wv, Wo, n_heads, n_kv_heads): ...
            def forward(self, x: np.ndarray) -> np.ndarray: ...
        \`\`\`

        - 权重形状同上一题（\`Wk, Wv\`: \`(D, n_kv_heads * d_h)\`），因果、无 bias、无位置编码。
        - \`forward(x)\` 接收**本次新增**的 token，\`x\`: \`(B, L, D)\`，$L \ge 1$：
          第一次调用相当于 prefill，之后可以每次 1 个 token（decode），也可以一次一小块（chunked prefill）。
          返回这 $L$ 个位置的输出 \`(B, L, D)\`。
        - 维护属性 \`self.k_cache\`、\`self.v_cache\`，形状 \`(B, n_kv_heads, t, d_h)\`（$t$ 为至今的总长度）。**缓存中存未展开的 KV 头**。
        - 正确性要求：把一个序列以任意方式切块依次喂入，拼接得到的输出与对整个序列做一次因果 GQA 前向**数值一致**。
      `,
      starter: py`
        import numpy as np

        class KVCacheAttention:
            def __init__(self, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
                self.Wq, self.Wk, self.Wv, self.Wo = Wq, Wk, Wv, Wo
                self.h, self.hkv = n_heads, n_kv_heads
                self.k_cache = None
                self.v_cache = None

            def forward(self, x):
                # TODO: 计算新 token 的 q/k/v → 追加到缓存 → 右下角对齐的因果注意力 → 合头 → W_O
                raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        class KVCacheAttention:
            def __init__(self, Wq, Wk, Wv, Wo, n_heads, n_kv_heads):
                self.Wq, self.Wk, self.Wv, self.Wo = Wq, Wk, Wv, Wo
                self.h, self.hkv = n_heads, n_kv_heads
                self.d_h = Wq.shape[1] // n_heads
                self.k_cache = None          # (B, n_kv, t, d_h)
                self.v_cache = None

            def _split(self, t, n):
                B, L, _ = t.shape
                return t.reshape(B, L, n, self.d_h).transpose(0, 2, 1, 3)

            def forward(self, x):
                B, L, D = x.shape
                q = self._split(x @ self.Wq, self.h)                 # (B, h, L, d_h)
                k = self._split(x @ self.Wk, self.hkv)               # (B, n_kv, L, d_h)
                v = self._split(x @ self.Wv, self.hkv)
                if self.k_cache is None:
                    self.k_cache, self.v_cache = k, v
                else:
                    self.k_cache = np.concatenate([self.k_cache, k], axis=2)
                    self.v_cache = np.concatenate([self.v_cache, v], axis=2)
                S = self.k_cache.shape[2]
                rep = self.h // self.hkv
                K = np.repeat(self.k_cache, rep, axis=1)             # 只在计算时展开
                V = np.repeat(self.v_cache, rep, axis=1)
                s = q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_h)  # (B, h, L, S)
                s = np.where(np.tril(np.ones((L, S), dtype=bool), k=S - L), s, -np.inf)
                s = s - s.max(axis=-1, keepdims=True)
                p = np.exp(s)
                p /= p.sum(axis=-1, keepdims=True)
                o = (p @ V).transpose(0, 2, 1, 3).reshape(B, L, D)
                return o @ self.Wo
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示，预分配缓存）',
          code: py`
            import torch
            import torch.nn.functional as F

            class KVCacheAttention(torch.nn.Module):
                def __init__(self, D, n_heads, n_kv_heads, max_len, batch):
                    super().__init__()
                    self.h, self.hkv, self.d_h = n_heads, n_kv_heads, D // n_heads
                    self.q = torch.nn.Linear(D, D, bias=False)
                    self.kv = torch.nn.Linear(D, 2 * n_kv_heads * self.d_h, bias=False)
                    self.o = torch.nn.Linear(D, D, bias=False)
                    # 预分配，避免每步 torch.cat 重新分配显存
                    self.register_buffer("k_cache", torch.zeros(batch, n_kv_heads, max_len, self.d_h), persistent=False)
                    self.register_buffer("v_cache", torch.zeros(batch, n_kv_heads, max_len, self.d_h), persistent=False)
                    self.t = 0

                def forward(self, x):
                    B, L, D = x.shape
                    q = self.q(x).view(B, L, self.h, self.d_h).transpose(1, 2)
                    k, v = self.kv(x).view(B, L, 2, self.hkv, self.d_h).permute(2, 0, 3, 1, 4)
                    self.k_cache[:B, :, self.t:self.t + L] = k
                    self.v_cache[:B, :, self.t:self.t + L] = v
                    self.t += L
                    K, V = self.k_cache[:B, :, :self.t], self.v_cache[:B, :, :self.t]
                    S = self.t
                    mask = torch.ones(L, S, dtype=torch.bool, device=x.device).tril(S - L)   # 右下角对齐
                    o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=True)
                    return self.o(o.transpose(1, 2).reshape(B, L, D))
          `,
        },
      ],
      explanation: md`
        **核心**：每次只为新 token 计算 q/k/v；k/v 沿**序列维**（axis=2）追加到缓存；query 是序列的最后 $L$ 个位置，所以掩码是右下角对齐的 \`tril(ones((L, S)), k=S-L)\`。

        **为什么与全量前向一致**：因果注意力中位置 $i$ 的输出只依赖 $\le i$ 的 token 的 q/k/v，而这些 k/v 无论分几次算都相同；右下角对齐保证每个新 query 看到的 key 集合与全量时完全一致。

        **复杂度**：生成 $n$ 个 token 共 $O(n)$ 次投影、$O(n^2 D)$ 注意力，显存 $O(n \cdot n_{kv} d_h)$。

        **工程上**：\`np.concatenate\` 每步都重新分配并拷贝整个缓存，是 $O(n^2)$ 的拷贝量；真实框架预分配 \`max_len\`（或按页分配，PagedAttention）并原地写入。
      `,
      complexity: { time: '每步 O(t·D + D²)', space: 'O(t·n_kv·d_h)' },
      hints: [
        md`第一次调用时缓存为 \`None\`，直接赋值；之后 \`np.concatenate([cache, k], axis=2)\`。`,
        md`计算时再 \`np.repeat(cache, n_heads // n_kv_heads, axis=1)\`，缓存本身保持 \`n_kv\` 个头。`,
        md`掩码：\`np.tril(np.ones((L, S), dtype=bool), k=S - L)\`。`,
      ],
      setup: py`
        import numpy as np

        def _full(x, Wq, Wk, Wv, Wo, h, hkv):
            B, T, D = x.shape
            dh = D // h
            rep = h // hkv
            sp = lambda t, n: t.reshape(B, T, n, dh).transpose(0, 2, 1, 3)
            q, k, v = sp(x @ Wq, h), np.repeat(sp(x @ Wk, hkv), rep, 1), np.repeat(sp(x @ Wv, hkv), rep, 1)
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
            s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
            p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            return (p @ v).transpose(0, 2, 1, 3).reshape(B, T, D) @ Wo

        def _setup(seed, B, T, D, h, hkv):
            r = np.random.default_rng(seed)
            dh = D // h
            W = (r.normal(size=(D, D)) / np.sqrt(D), r.normal(size=(D, hkv * dh)) / np.sqrt(D),
                 r.normal(size=(D, hkv * dh)) / np.sqrt(D), r.normal(size=(D, D)) / np.sqrt(D))
            return r.normal(size=(B, T, D)), W

        def _run_chunks(seed, B, T, D, h, hkv, chunks):
            x, W = _setup(seed, B, T, D, h, hkv)
            att = KVCacheAttention(*W, h, hkv)
            outs, i = [], 0
            for c in chunks:
                o = att.forward(x[:, i:i + c])
                assert o.shape == (B, c, D), f"forward 返回形状应为 {(B, c, D)}，得到 {o.shape}"
                outs.append(o)
                i += c
            got = np.concatenate(outs, axis=1)
            exp = _full(x, *W, h, hkv)
            err = np.abs(got - exp).max()
            assert err < 1e-8, f"切块 {chunks} 的输出与全量前向不一致，最大误差 {err:.2e}"
            return att
      `,
      tests: [
        { name: '一次性 prefill = 全量前向', code: py`_run_chunks(0, 1, 6, 16, 4, 2, [6])` },
        { name: 'prefill 4 + 逐 token decode 4 步', code: py`_run_chunks(1, 2, 8, 16, 4, 2, [4, 1, 1, 1, 1])` },
        {
          name: '缓存存未展开的 KV 头',
          code: py`
            att = _run_chunks(2, 2, 5, 24, 6, 2, [3, 1, 1])
            assert att.k_cache.shape == (2, 2, 5, 4), f"k_cache 形状应为 (B, n_kv, t, d_h) = (2, 2, 5, 4)，得到 {att.k_cache.shape}"
            assert att.v_cache.shape == (2, 2, 5, 4), f"v_cache 形状应为 (2, 2, 5, 4)，得到 {att.v_cache.shape}"
          `,
        },
        { name: 'chunked prefill（3 + 2 + 1 + 3）', hidden: true, code: py`_run_chunks(3, 1, 9, 32, 8, 4, [3, 2, 1, 3])` },
        { name: 'MQA 逐 token 从头解码', hidden: true, code: py`_run_chunks(4, 2, 6, 16, 4, 1, [1] * 6)` },
      ],
      followUps: [
        {
          id: 'f-paged',
          q: md`为什么生产系统不用「每条请求一块连续的缓存」？`,
          a: md`请求长度事先未知：按 max_len 预分配会造成大量内部碎片，按需扩展又会产生外部碎片和拷贝。vLLM 的 PagedAttention 借鉴操作系统分页，把缓存切成固定大小的块（如 16 token），用块表映射逻辑位置到物理块，碎片只发生在最后一块，还能让共享前缀的请求共用物理块（copy-on-write）。详见推理系统章节。`,
        },
        {
          id: 'f-rope',
          q: md`如果加上 RoPE，增量解码要改什么？`,
          a: md`新 token 的 q、k 要用其**绝对位置**（缓存长度 + 块内偏移）旋转；缓存中存旋转后的 k。批量左 padding 时绝对位置还要减去各自的 pad 数（position_ids 来自 \`cumsum(mask) - 1\`）。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-kvcache',
      freq: 3,
      q: md`讲一下 KV cache 的原理。为什么只缓存 K、V 不缓存 Q？`,
      a: md`
        **结论**：自回归 decode 时，历史 token 的 K、V 不会改变且每一步都要用，缓存后每步只需为新 token 计算一行 q/k/v，把每步的 K/V 投影从 $O(t)$ 降到 $O(1)$。

        **原因**：因果掩码保证位置 $j$ 的 $k_j, v_j$ 只依赖 $\le j$ 的输入；第 $t$ 步的输出 $o_t$ 需要 $q_t$ 与全部 $k_{\le t}, v_{\le t}$。历史 $q_j$ 只在第 $j$ 步用过，之后不再需要，所以不缓存。

        **延伸**：推理分 prefill（计算密集）与 decode（访存密集）两阶段；KV cache 显存 $= 2 L B S n_{kv} d_h \cdot$ bytes，长上下文下成为瓶颈，催生了 GQA/MQA、MLA、KV 量化、PagedAttention、前缀缓存、PD 分离等一系列技术。
      `,
    },
    {
      id: 'qa-calc',
      freq: 3,
      q: md`估算一下 Llama-3-70B 在 BF16、batch 16、上下文 32K 时的 KV cache 显存。`,
      a: md`
        **结论**：约 160 GiB。

        **推导**：每 token $= 2 \times 80 \text{ 层} \times 8 \text{ KV 头} \times 128 \times 2 \text{ 字节} = 327{,}680$ 字节 $= 320$ KiB；乘以 $16 \times 32768 = 524{,}288$ 个 token，得 $320 \text{ KiB} \times 2^{19} = 160$ GiB。

        **延伸**：这已经超过两张 80GB 卡，再加上约 140GB 的 BF16 权重，必须多卡张量并行、或者 FP8 KV cache（减半）、或者减小 batch。答题时一定要说明用的是 KV 头数 8 而不是 Q 头数 64，以及 GiB 与 GB 的换算。
      `,
    },
    {
      id: 'qa-gqa',
      freq: 3,
      q: md`MHA、MQA、GQA 有什么区别？各自的优缺点？`,
      a: md`
        **结论**：三者的区别只在 KV 头数：MHA 为 $h$，MQA 为 1，GQA 为 $g$（$1<g<h$）。

        **对比**：MHA 质量最好但 KV cache 最大；MQA 缓存缩小 $h$ 倍、decode 最快，但表达能力受损、训练可能不稳定；GQA 折中，Ainslie et al. (2023) 报告其质量接近 MHA、速度接近 MQA，并可由 MHA 检查点经均值池化 + 约 5% 预训练算力的 uptraining 得到。Llama 2-70B、Llama 3 全系、Mistral 7B、Qwen2.5 等都使用 GQA。

        **延伸**：GQA 不减少每个 query 头的 FLOPs，加速来自 decode 时读取的 KV 字节变少、算术强度提高。更激进的方案是 MLA：不减头数，而是把 K/V 低秩压缩成一个潜向量缓存。
      `,
    },
    {
      id: 'qa-memorybound',
      freq: 2,
      q: md`为什么说 decode 阶段是 memory-bound 的？GQA 是怎么缓解的？`,
      a: md`
        **结论**：decode 每步对每个序列只处理 1 个 token，却要把全部权重和整份 KV cache 从 HBM 读一遍，算术强度（FLOPs/字节）很低。

        **原因**：注意力部分，每个 KV 元素（2 字节）只和 1 个 query 头做一次乘加（约 2 FLOPs），强度约 1 FLOP/字节，而 H100 的算力/带宽比在数百 FLOP/字节量级，GPU 大部分时间在等数据。GQA 下一份 K/V 被 $n_{\text{rep}}$ 个 query 头复用，强度提高 $n_{\text{rep}}$ 倍、读取量减少同样倍数。

        **延伸**：权重部分靠增大 batch 提高强度（同一份权重服务更多 token），但 KV cache 是每个序列独有的，batch 再大也不能复用——这正是 KV 压缩（GQA/MLA/量化）对长上下文服务如此关键的原因。
      `,
    },
  ],

  cheatsheet: md`
    $$\text{KV bytes} = 2 \cdot L \cdot B \cdot S \cdot n_{kv} \cdot d_h \cdot \text{bytes}$$

    | 模型 | 每 token（BF16） |
    |---|---|
    | Llama-2-7B（32L, MHA 32） | 512 KiB |
    | Llama-3-8B（32L, KV 8） | 128 KiB |
    | Llama-3-70B（80L, KV 8） | 320 KiB |

    \`\`\`python
    rep = n_heads // n_kv
    K = np.repeat(k_cache, rep, axis=1)          # 不是 np.tile！
    mask = np.tril(np.ones((L, S), bool), k=S - L)   # 右下角对齐
    cache = np.concatenate([cache, k_new], axis=2)   # 沿序列维
    \`\`\`

    - 缓存 K、V（未展开的 $n_{kv}$ 个头），不缓存 Q
    - q 头 $i$ → KV 头 $\lfloor i / n_{\text{rep}} \rfloor$
    - GQA：均值池化 + ~5% 预训练算力 uptrain；省显存与带宽，不省每头 FLOPs
    - decode 时 RoPE 用绝对位置
  `,
})
