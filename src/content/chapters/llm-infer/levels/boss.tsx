/**
 * L10 BOSS：推理加速与服务综合。复盘 + 全章混合题 + 两道综合实战：
 *  1) 贪心投机解码的完整生成循环（与目标模型贪心解码逐 token 一致，统计前向次数）
 *  2) 推理服务容量规划（roofline + KV 容量 + TPOT SLO + 张量并行度选择）
 */
import { defineLevel, md, py } from '@/content/helpers'

export default defineLevel({
  learn: [
    {
      id: 'review',
      title: '复盘：一张表串起整章',
      body: md`
        推理优化的出发点永远是「瓶颈在哪」：

        | 问题（根因） | 技术 | 关键数字 / 结论 |
        |---|---|---|
        | decode 慢、GPU 闲（强度 ≈ batch ≪ 拐点） | 增大 batch、权重量化、GQA / MLA、投机解码 | batch = 1 上界 ≈ 带宽 / 权重字节；注意力强度 = $n_h/n_{kv}$ |
        | batch 上不去（KV 碎片、预留） | PagedAttention：块表 + COW | 旧系统 KV 利用率 20.4%–38.2%；vLLM 吞吐 2–4× |
        | 短请求被长请求拖住（请求级批处理） | 迭代级调度（Orca） | 完成即离开、随时补位 |
        | 相同 prompt 反复 prefill | RadixAttention / 链式块哈希 APC | APC 只缓存满块；LRU 淘汰叶子 / 逆序入队 |
        | 长 prefill 造成 TPOT 尖峰 | chunked prefill；PD 分离 | token 预算；KV 传输 ∝ prompt × 每 token KV |
        | 每 token 一次串行前向 | 投机解码 | 接受 $\min(1,p/q)$，残差 $\max(0,p-q)$；$E = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$ |
        | 草稿难部署、接受率低 | Medusa / EAGLE-3 / MTP / P-EAGLE | EAGLE-3 最高 6.5×；MTP 第 2 token 接受率 85–90% |

        几条贯穿全章的结论：
        - **一切以 roofline 估算开头**：先判断是算力还是带宽，再决定优化方向；
        - **大 batch 会改变结论**：投机解码、树验证在大 batch 下不再免费；
        - **无损 vs 有损**：拒绝采样、前缀缓存、chunked prefill、PagedAttention 都不改变输出分布（浮点误差除外）；量化、typical acceptance、KV 压缩会。
      `,
      checkpoint: {
        id: 'cp-review',
        type: 'single',
        prompt: md`某客服机器人每个请求都带一段 3000 token 的固定 system prompt，用户问题通常只有几十个 token，TTFT 超标。最直接有效的手段是：`,
        options: ['开启投机解码', '开启前缀缓存', '把权重量化成 INT4', '增大 decode 的 batch'],
        answer: 1,
        explanation: md`TTFT 主要花在 3000 token 的 prefill 上，而且这部分每个请求都完全相同——前缀缓存命中后只需 prefill 几十个 token。投机解码与权重量化主要改善 decode；增大 batch 改善吞吐而非 TTFT。`,
      },
    },
  ],

  quiz: [
    {
      id: 'b-lossless',
      type: 'multi',
      difficulty: 2,
      prompt: md`下列哪些技术**不改变**模型的输出分布（不计浮点误差）？（多选）`,
      options: ['标准投机采样（min(1,p/q) + 残差重采样）', 'Medusa 的 typical acceptance', '前缀缓存', 'chunked prefill', 'INT4 权重量化', 'PagedAttention'],
      answer: [0, 2, 3, 5],
      explanation: md`前缀缓存复用的是完全相同前缀的同一份 KV；chunked prefill 只是把同样的计算拆到几次迭代里；PagedAttention 只改变 KV 的存放位置。typical acceptance 和量化都会改变输出分布。`,
    },
    {
      id: 'b-ridge',
      type: 'numeric',
      difficulty: 1,
      prompt: md`A100 80GB SXM 的 BF16 稠密算力 312 TFLOPS、带宽 2.039 TB/s，拐点约为多少 FLOP/B？（取整）`,
      answer: 153,
      tolerance: 0.01,
      explanation: md`$312 / 2.039 \approx 153$。比 H100 的 295 低：A100 上同样的 decode 更早进入算力瓶颈，但绝对速度更慢。`,
    },
    {
      id: 'b-spec',
      type: 'numeric',
      difficulty: 2,
      prompt: md`EAGLE 类草稿：每轮起草 $\gamma = 5$ 个 token，草稿每次前向耗时为目标模型的 $c = 0.02$，平均接受率 $\alpha = 0.85$（i.i.d.）。理论加速比约为多少？（保留 2 位小数）`,
      answer: 3.77,
      tolerance: 0.01,
      unit: '×',
      explanation: md`$0.85^6 \approx 0.3771$，$E = \frac{1 - 0.3771}{0.15} \approx 4.152$；一轮耗时 $(5 \times 0.02 + 1)T = 1.1T$，加速比 $\approx 4.152 / 1.1 \approx 3.77$。草稿极便宜（$c$ 很小）时，加速比主要由接受率决定，上限是 $1/(1-\alpha) \approx 6.7$。`,
    },
    {
      id: 'b-blocks',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Llama-3.1-70B 每 token KV 为 320 KiB（BF16）。8 卡 TP 时 8 个 KV 头平均分到每卡。每卡可用于 KV 的显存 50 GiB，block size = 16。每卡能分配多少个 KV 块？`,
      answer: 81920,
      tolerance: 0,
      unit: '块',
      explanation: md`每卡每 token 只存 1 个 KV 头：$320 / 8 = 40$ KiB；每块 $16 \times 40 = 640$ KiB；$50 \times 1024 \times 1024 / 640 = 81920$ 块。块数是按「每卡」计算的——所有卡上同一个块号存的是同一批 token 的不同 KV 头。`,
    },
    {
      id: 'b-judge-pd',
      type: 'judge',
      difficulty: 1,
      prompt: md`PD 分离后，decode 实例只负责生成 token，不再需要保存 KV cache。`,
      answer: false,
      explanation: md`decode 每一步都要对全部历史做注意力，KV cache 必须在 decode 实例上。PD 分离做的是把 prefill 算好的 KV **传过去**，decode 实例照样要用 PagedAttention 管理它。`,
    },
    {
      id: 'b-order',
      type: 'order',
      difficulty: 2,
      prompt: md`一个请求进入开启了前缀缓存、chunked prefill 与投机解码的 vLLM 实例，把处理过程排序：`,
      items: [
        '调度器为 prompt 的满块计算链式哈希，查找已缓存的前缀块（命中则引用计数 +1）',
        '为未命中部分分配新的 KV 块',
        '未命中部分按 token 预算分若干次迭代做 chunked prefill，产出第一个 token',
        '进入 decode：每轮草稿起草若干 token，目标模型一次验证，按规则接受并回滚多余 KV',
        '请求结束，块引用计数 −1，归零的块带着哈希进入空闲队列（逆序，等待 LRU 淘汰或被复用）',
      ],
      explanation: md`先复用（前缀缓存）、再分配（PagedAttention）、再分块计算（chunked prefill），之后在连续批处理中逐轮投机解码，最后释放块但保留哈希以便后续请求命中。`,
    },
    {
      id: 'b-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把现象与最对症的手段配对：`,
      pairs: [
        ['长上下文 decode 时每步大部分字节花在读 KV 上', 'GQA / MLA / FP8 KV'],
        ['长 prompt 进来时正在生成的请求明显卡顿', 'chunked prefill'],
        ['beam search 的多个 beam 各存一份 prompt KV', '写时复制（COW）'],
        ['单请求低并发下生成速度慢，GPU 算力闲置', '投机解码'],
        ['batch 内短请求要等最长的请求结束', '迭代级调度'],
      ],
      explanation: md`对症下药的前提是先定位瓶颈：KV 字节 → 压缩 KV；prefill 干扰 → 切块或分离；重复存储 → 共享；算力闲置 → 用算力换串行步数；批处理粒度 → 迭代级调度。`,
    },
    {
      id: 'b-fill',
      type: 'fill',
      difficulty: 2,
      prompt: md`
        Llama-3.1-8B（16.06 GB BF16 权重）在 H100（3.35 TB/s）上，batch = 1、忽略 KV：单步下界 ≈ ____(1)____ ms（保留 1 位小数）。
        换成 INT4 weight-only 量化后（只考虑权重字节变为 1/4），单步下界 ≈ ____(2)____ ms（保留 1 位小数）。
      `,
      blanks: [
        { label: '(1)', answers: ['4.8'], numeric: true, tolerance: 0.02 },
        { label: '(2)', answers: ['1.2'], numeric: true, tolerance: 0.03 },
      ],
      explanation: md`$16.06 / 3350 \approx 4.79$ ms；INT4 后 4.015 GB，约 1.20 ms。实际收益取决于反量化 kernel 的效率，通常达不到理想的 4×。`,
    },
    {
      id: 'b-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的**贪心**投机解码主循环有 **2 处** bug（\`verify(seq, drafts)\` 一次前向返回 \`len(drafts) + 1\` 个 argmax，第 $i$ 个是在 \`seq + drafts[:i]\` 之后的预测）。点出有问题的行：`,
      code: py`
        def generate(seq, verify, draft_next, gamma, max_new):
            out = []
            while len(out) < max_new:
                drafts = []
                for _ in range(gamma):
                    drafts.append(draft_next(seq + drafts))
                g = verify(seq, drafts)
                n = 0
                while n < gamma and drafts[n] == g[n + 1]:
                    n += 1
                new = drafts[:n]
                out += new
                seq = seq + new
            return out[:max_new]
      `,
      bugLines: [9, 11],
      explanation: md`
        - 第 9 行：第 $n$ 个草稿要和「在它之前位置」的 argmax \`g[n]\` 比较，写成 \`g[n + 1]\` 错位了一格。
        - 第 11 行：漏了修正 / bonus token，应为 \`new = drafts[:n] + [g[n]]\`。否则第一个草稿就被拒时本轮一个 token 都不产出，可能死循环。
      `,
    },
    {
      id: 'b-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`下列技术中，哪些能**减少 decode 阶段每产出一个 token 需要从 HBM 读取的字节数**（或等价地，让一次读取产出更多 token）？`,
      items: ['增大 batch', 'GQA 替代 MHA', 'INT4 权重量化', '投机解码', '前缀缓存', 'chunked prefill', 'FP8 KV cache', 'PD 分离'],
      answer: [0, 1, 2, 3, 6],
      explanation: md`batch 与投机解码让一次权重读取服务更多 token；GQA、FP8 KV 减少 KV 字节；INT4 减少权重字节。前缀缓存与 chunked prefill 作用在 prefill；PD 分离改变部署方式，不改变 decode 每 token 的读取量。`,
    },
    {
      id: 'b-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 vLLM APC 中一个新请求「查前缀缓存」的逻辑（\`table\` 为哈希 → 物理块）。有 2 行干扰项：`,
      lines: [
        'def lookup(tokens, B, table, ref):',
        '    blocks, parent = [], None',
        '    for s in range(0, len(tokens) - len(tokens) % B, B):',
        '        parent = hash((parent, tuple(tokens[s:s + B])))',
        '        if parent not in table:',
        '            break',
        '        blocks.append(table[parent])',
        '        ref[table[parent]] += 1',
        '    return blocks',
      ],
      distractors: ['        parent = hash(tuple(tokens[s:s + B]))', '            continue'],
      explanation: md`链式哈希必须包含父哈希；第一次未命中就 \`break\`——命中只能是从开头连续的一段。命中块的引用计数 +1，防止在请求运行期间被淘汰。`,
    },
    {
      id: 'b-batch-spec',
      type: 'single',
      difficulty: 3,
      prompt: md`一个 70B 服务在低峰期（并发 4）开启 EAGLE-3 后 TPOT 降低了 45%，但在高峰期（并发 256）吞吐反而下降。最合理的解释是：`,
      options: [
        '高峰期草稿模型的接受率一定下降了',
        '并发 256 时 decode 已接近算力瓶颈，验证多个草稿 token 的额外 FLOPs 不再免费，被拒绝的草稿浪费了算力',
        'EAGLE-3 在大 batch 下会改变输出分布',
        '高峰期 KV cache 被前缀缓存占满',
      ],
      answer: 1,
      explanation: md`投机解码用闲置算力换更少的串行前向；大 batch 时算力不再闲置，每轮要多算 $B \times$（树节点数）个位置，接受率不变也可能净亏。常见做法是按负载动态缩小草稿树 / $\gamma$ 或关闭投机。C 错：标准验证规则下输出分布不变。`,
    },
  ],

  guided: [],

  challenges: [
    {
      id: 'c-spec-generate',
      title: '贪心投机解码：完整的生成循环',
      difficulty: 'hard',
      source: '综合 · 投机解码 + KV 回滚语义',
      timeLimitMin: 30,
      description: md`
        给定两个「模型」接口，实现贪心投机解码，输出必须与**只用目标模型贪心解码**逐 token 完全相同：

        \`\`\`python
        def spec_generate(prompt: list[int], target_verify, draft_next, gamma: int, max_new: int, eos: int | None = None) -> list[int]: ...
        \`\`\`

        - \`draft_next(seq) -> int\`：草稿模型在序列 \`seq\` 之后的 argmax（一次草稿前向）；
        - \`target_verify(seq, drafts) -> list[int]\`：目标模型**一次**前向，返回长度 \`len(drafts) + 1\` 的列表，第 $i$ 个元素是在 \`seq + drafts[:i]\` 之后的 argmax。

        **每轮的规则**（测试会统计两种前向的调用次数，必须严格一致）：
        1. 记已生成 \`len(out)\` 个，剩余 \`remaining = max_new - len(out)\`；本轮起草 \`k = min(gamma, remaining - 1)\` 个 token（依次调用 \`draft_next\`，每次把已起草的接在后面）；
        2. 调用一次 \`target_verify(seq, drafts)\` 得到 \`g\`；
        3. 从左到右，\`drafts[i] == g[i]\` 就接受，第一次不等就停；设接受了 \`n\` 个，本轮产出 \`drafts[:n] + [g[n]]\`；
        4. 依次把产出的 token 追加到输出与 \`seq\`；若遇到 \`eos\`，追加它之后**立即结束**整个生成。

        返回生成的 token（不含 prompt），长度不超过 \`max_new\`。

        **为什么 \`k = remaining - 1\`**：每轮最多产出 $k+1$ 个，这样恰好不会超过 \`max_new\`，也不浪费验证算力。
      `,
      starter: py`
        def spec_generate(prompt, target_verify, draft_next, gamma, max_new, eos=None):
            seq, out = list(prompt), []
            # while len(out) < max_new: 起草 k 个 → 一次验证 → 接受前缀 + 修正/bonus → 处理 eos
            return out
      `,
      solution: py`
        def spec_generate(prompt, target_verify, draft_next, gamma, max_new, eos=None):
            seq, out = list(prompt), []
            while len(out) < max_new:
                k = min(gamma, max_new - len(out) - 1)        # 本轮最多产出 k+1 个，正好不越界
                drafts = []
                for _ in range(k):
                    drafts.append(draft_next(seq + drafts))   # 草稿自回归
                g = target_verify(seq, drafts)                # 目标模型一次前向：k+1 个 argmax
                n = 0
                while n < k and drafts[n] == g[n]:            # 第 n 个草稿与「它之前位置」的 argmax 比较
                    n += 1
                for t in drafts[:n] + [g[n]]:                 # 接受的草稿 + 修正 / bonus
                    out.append(t)
                    seq.append(t)                             # 被拒草稿从未进入 seq（相当于 KV 回滚）
                    if eos is not None and t == eos:
                        return out
            return out
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：真实模型 + KV cache 的贪心投机解码）',
          code: py`
            import torch

            @torch.inference_mode()
            def spec_generate(target, draft, input_ids, gamma, max_new):
                """target/draft: HF 风格模型，forward(ids, past_key_values) -> (logits, past)。只演示 batch=1 的主干逻辑"""
                t_logits, t_past = target(input_ids)
                d_logits, d_past = draft(input_ids)
                nxt = t_logits[:, -1].argmax(-1, keepdim=True)          # 第一个 token 由目标模型给出
                out = [nxt]
                while sum(x.shape[1] for x in out) < max_new:
                    drafts, cur = [], nxt
                    for _ in range(gamma):                               # 草稿自回归
                        d_logits, d_past = draft(cur, past_key_values=d_past)
                        cur = d_logits[:, -1].argmax(-1, keepdim=True)
                        drafts.append(cur)
                    cand = torch.cat([nxt] + drafts, dim=1)              # 上一个 token + γ 个草稿
                    t_logits, t_past = target(cand, past_key_values=t_past)
                    g = t_logits.argmax(-1)                              # (1, γ+1)
                    d = torch.cat(drafts, dim=1)
                    n = int((d == g[:, :-1]).int().cumprod(-1).sum())    # 连续命中的草稿数
                    nxt = g[:, n:n + 1]                                  # 修正 / bonus
                    out.append(torch.cat([d[:, :n], nxt], dim=1))
                    t_past.crop(t_past.get_seq_length() - (gamma - n))   # 回滚目标模型 KV
                    d_past.crop(d_past.get_seq_length() - (gamma - n))   # 回滚草稿模型 KV（实现细节因框架而异）
                return torch.cat(out, dim=1)[:, :max_new]
          `,
        },
      ],
      explanation: md`
        **正确性**：贪心下目标模型在每个位置的输出是确定的 argmax。第 $i$ 个草稿被接受当且仅当它等于目标模型在该位置的 argmax，所以接受的草稿就是目标模型自己会生成的 token；
        第一个不等的位置用 \`g[n]\` 修正（正是目标模型在那里的 argmax），全部相等时 \`g[k]\` 是 bonus。逐 token 与纯目标贪心解码一致。

        **KV 回滚**：被拒的草稿从未写进 \`seq\`；真实系统里目标模型和草稿模型为它们写过 KV，需要把序列长度截回（PyTorch 版里的 \`crop\`）。

        **调用次数**：目标前向次数 = 轮数。草稿完美时每轮产出 $\gamma+1$ 个，轮数 $= \lceil \text{max\_new}/(\gamma+1) \rceil$；草稿全错时每轮 1 个，退化成普通解码（还多了草稿的开销）。

        **常见错误**：\`drafts[n]\` 与 \`g[n+1]\` 比较（错位）；漏掉修正 / bonus token；不按 \`remaining - 1\` 截断导致超出 \`max_new\` 或多一次验证；遇到 eos 不停止。
      `,
      complexity: { time: 'O(max_new·γ) 次模型调用（目标模型 ≤ max_new 次）', space: 'O(len(prompt) + max_new)' },
      hints: [
        md`\`k = min(gamma, max_new - len(out) - 1)\`，然后 \`for _ in range(k): drafts.append(draft_next(seq + drafts))\`。`,
        md`\`g = target_verify(seq, drafts)\`；\`n\` 从 0 开始，\`while n < k and drafts[n] == g[n]: n += 1\`。`,
        md`本轮产出 \`drafts[:n] + [g[n]]\`，逐个追加，碰到 eos 立刻返回。`,
      ],
      setup: py`
        import random

        class _Models:
            """toy 目标模型与草稿模型（确定性），并统计调用次数"""
            def __init__(self, V, seed, quality):
                self.V, self.seed, self.q = V, seed, quality
                self.t_calls = 0; self.d_calls = 0; self.max_k = 0
            def target_argmax(self, seq):
                h = hash((self.seed, tuple(seq[-3:]), len(seq) % 5))
                return h % self.V
            def draft_next(self, seq):
                self.d_calls += 1
                t = self.target_argmax(seq)
                good = hash((self.seed + 1, tuple(seq[-2:]))) % 100 < self.q * 100
                return t if good else (t + 1) % self.V
            def target_verify(self, seq, drafts):
                self.t_calls += 1
                self.max_k = max(self.max_k, len(drafts))
                seq = list(seq)
                return [self.target_argmax(seq + list(drafts[:i])) for i in range(len(drafts) + 1)]

        def _greedy(M, prompt, max_new, eos=None):
            seq, out = list(prompt), []
            while len(out) < max_new:
                t = M.target_argmax(seq); out.append(t); seq.append(t)
                if eos is not None and t == eos:
                    break
            return out

        def _ref_counts(V, seed, q, prompt, gamma, max_new, eos=None):
            M = _Models(V, seed, q); seq, out = list(prompt), []
            while len(out) < max_new:
                k = min(gamma, max_new - len(out) - 1); d = []
                for _ in range(k): d.append(M.draft_next(seq + d))
                g = M.target_verify(seq, d); n = 0
                while n < k and d[n] == g[n]: n += 1
                stop = False
                for t in d[:n] + [g[n]]:
                    out.append(t); seq.append(t)
                    if eos is not None and t == eos:
                        stop = True; break
                if stop: break
            return out, M.t_calls, M.d_calls
      `,
      tests: [
        {
          name: '草稿完美：每轮产出 γ+1 个',
          code: py`
            M = _Models(50, 1, 1.0)
            out = spec_generate([1, 2, 3], M.target_verify, M.draft_next, 4, 20)
            exp = _greedy(_Models(50, 1, 1.0), [1, 2, 3], 20)
            assert list(out) == exp, f"输出必须与目标模型贪心解码一致：期望 {exp}，得到 {out}"
            assert M.t_calls == 4, f"20 个 token、每轮 5 个，应调用目标模型 4 次，实际 {M.t_calls}"
          `,
        },
        {
          name: '草稿全错：退化为逐 token',
          code: py`
            M = _Models(50, 2, 0.0)
            out = spec_generate([7], M.target_verify, M.draft_next, 3, 10)
            assert list(out) == _greedy(_Models(50, 2, 0.0), [7], 10), "输出与贪心解码不一致"
            assert M.t_calls == 10, f"每轮只产出 1 个，应调用 10 次，实际 {M.t_calls}"
          `,
        },
        {
          name: 'γ = 0 与 max_new 截断',
          code: py`
            M = _Models(30, 3, 0.7)
            assert list(spec_generate([1], M.target_verify, M.draft_next, 0, 6)) == _greedy(_Models(30, 3, 0.7), [1], 6)
            assert M.t_calls == 6 and M.d_calls == 0
            M2 = _Models(30, 3, 1.0)
            out = spec_generate([1], M2.target_verify, M2.draft_next, 5, 7)
            assert len(out) == 7, f"长度应恰为 max_new=7，得到 {len(out)}"
            assert M2.max_k <= 5, "每轮起草数不能超过 gamma"
          `,
        },
        {
          name: '遇到 eos 立即停止',
          hidden: true,
          code: py`
            for seed in range(40):
                M = _Models(6, 100 + seed, 0.8)
                exp = _greedy(_Models(6, 100 + seed, 0.8), [seed % 6], 30, eos=0)
                got = list(spec_generate([seed % 6], M.target_verify, M.draft_next, 4, 30, eos=0))
                assert got == exp, f"seed={seed}：期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '随机对拍：输出与调用次数',
          hidden: true,
          code: py`
            rnd = random.Random(5)
            for case in range(150):
                V, seed, q = rnd.randint(3, 40), rnd.randint(0, 10**6), rnd.random()
                prompt = [rnd.randrange(V) for _ in range(rnd.randint(1, 6))]
                gamma, max_new = rnd.randint(0, 6), rnd.randint(1, 40)
                eos = rnd.choice([None, None, rnd.randrange(V)])
                M = _Models(V, seed, q)
                got = list(spec_generate(list(prompt), M.target_verify, M.draft_next, gamma, max_new, eos))
                exp, tc, dc = _ref_counts(V, seed, q, prompt, gamma, max_new, eos)
                assert got == exp, f"case {case}: 输出期望 {exp}，得到 {got}"
                assert got == _greedy(_Models(V, seed, q), prompt, max_new, eos), f"case {case}: 与贪心解码不一致"
                assert (M.t_calls, M.d_calls) == (tc, dc), f"case {case}: 调用次数（目标, 草稿）期望 {(tc, dc)}，得到 {(M.t_calls, M.d_calls)}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-sampling',
          q: md`改成按温度采样，这个循环要改哪里？`,
          a: md`草稿要记录每步的分布 $q_i$；验证时目标模型返回 $\gamma+1$ 个分布 $p_i$；接受条件变为 $u_i < p_i(x_i)/q_i(x_i)$，拒绝时从 $\mathrm{norm}(\max(0, p_i - q_i))$ 采修正 token，全部接受时从 $p_{\gamma+1}$ 采 bonus。其余（\`k\` 的截断、eos、KV 回滚）不变——这就是「投机解码原理」关卡实战题与本题的组合。`,
        },
      ],
    },
    {
      id: 'c-plan',
      title: '推理服务容量规划：选张量并行度',
      difficulty: 'hard',
      source: '综合 · Roofline + KV 容量 + SLO',
      timeLimitMin: 30,
      description: md`
        有 \`n_gpus\` 张同型号 GPU（2 的幂），部署一个稠密模型做 decode，所有请求上下文长度都按 \`ctx\` 估算，要求 TPOT ≤ \`slo\`（秒）。
        可以选张量并行度 \`tp\` ∈ {1, 2, 4, …, n_gpus}，其余用数据并行：共 \`n_gpus // tp\` 个副本。实现

        \`\`\`python
        def plan_serving(cfg: dict, gpu: dict, n_gpus: int, ctx: int, slo: float, ar_latency: float) -> dict | None: ...
        \`\`\`

        \`cfg\`：\`n_params, n_layers, n_heads, n_kv_heads, head_dim, w_bytes, kv_bytes\`；\`gpu\`：\`flops, bw, mem\`。对每个 \`tp\`（以下均为**每卡**）：
        - 权重 $W = N w / tp$；每个序列的 KV $K = S \kappa / \min(tp, n_{kv})$，$\kappa = 2 L n_{kv} d_h \cdot$ \`kv_bytes\`；
        - 显存约束：$B_{mem} = \lfloor (0.9 \cdot \text{mem} - W) / K \rfloor$，若 $B_{mem} < 1$ 则该 \`tp\` 不可行；
        - 单步时间：$t(B) = \max\!\big(\frac{(2N + 4Ln_hd_hS)B}{tp \cdot \text{flops}},\ \frac{W + BK}{\text{bw}}\big) + c$，其中通信 $c = 2L \cdot$ \`ar_latency\`（\`tp > 1\` 时），否则 0；
        - SLO 约束：$B_{slo}$ = 满足 $t(B) \le$ \`slo\` 的最大正整数（若 $t(1) >$ \`slo\`，该 \`tp\` 不可行）；
        - $B = \min(B_{mem}, B_{slo})$，吞吐 $= (\text{n\_gpus} / tp) \cdot B / t(B)$。

        返回吞吐最大的方案 \`{"tp", "batch", "tokens_per_s"}\`（吞吐相同取较小的 \`tp\`）；全部不可行返回 \`None\`。

        **示例**：Llama-3.1-70B（\`7.06e10, 80, 64, 8, 128\`，BF16 权重与 KV），8×H100（\`989e12, 3.35e12, 80e9\`），\`ctx=4096, slo=0.03, ar_latency=2e-5\`
        → \`tp=1\` 放不下；\`tp=2\` 每副本 batch 2；\`tp=4\` batch 109；\`tp=8\` batch 323，吞吐约 13106 token/s，为最优。
      `,
      starter: py`
        import math

        def plan_serving(cfg, gpu, n_gpus, ctx, slo, ar_latency):
            best = None
            tp = 1
            while tp <= n_gpus:
                # TODO: 显存约束 B_mem、SLO 约束 B_slo（t(B) 关于 B 单调递增）、吞吐
                tp *= 2
            return best
      `,
      solution: py`
        import math

        def plan_serving(cfg, gpu, n_gpus, ctx, slo, ar_latency):
            N, L = cfg["n_params"], cfg["n_layers"]
            kappa = 2 * L * cfg["n_kv_heads"] * cfg["head_dim"] * cfg["kv_bytes"]
            a = 2 * N + 4 * L * cfg["n_heads"] * cfg["head_dim"] * ctx      # 每序列每步 FLOPs
            best, tp = None, 1
            while tp <= n_gpus:
                W = N * cfg["w_bytes"] / tp
                K = ctx * kappa / min(tp, cfg["n_kv_heads"])                  # KV 头多于 tp 时才能平分
                comm = 2 * L * ar_latency if tp > 1 else 0.0

                def t(B):
                    return max(a * B / (tp * gpu["flops"]), (W + B * K) / gpu["bw"]) + comm

                b_mem = math.floor((0.9 * gpu["mem"] - W) / K)
                if b_mem >= 1 and t(1) <= slo:
                    # t(B) 是两个 B 的一次函数取 max 再加常数：先用闭式解，再用 t 本身修正浮点边界
                    room = slo - comm
                    b = math.floor(min(room * tp * gpu["flops"] / a, (room * gpu["bw"] - W) / K))
                    b = max(b, 1)
                    while t(b + 1) <= slo:
                        b += 1
                    while b > 1 and t(b) > slo:
                        b -= 1
                    B = min(b_mem, b)
                    thr = (n_gpus // tp) * B / t(B)
                    if best is None or thr > best["tokens_per_s"] * (1 + 1e-12):
                        best = {"tp": tp, "batch": B, "tokens_per_s": thr}
                tp *= 2
            return best
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：把规划结果落到 vLLM 启动参数）',
          code: py`
            # 规划只是起点：用实测校准 kernel 效率与通信，再在候选附近扫参数
            plan = plan_serving(cfg, gpu, n_gpus=8, ctx=4096, slo=0.03, ar_latency=2e-5)
            cmd = (
                f"vllm serve meta-llama/Llama-3.1-70B-Instruct "
                f"--tensor-parallel-size {plan['tp']} "
                f"--max-num-seqs {plan['batch']} "            # 每个副本的并发上限
                f"--gpu-memory-utilization 0.9 --max-model-len 4096"
            )
            # 数据并行副本数 = 8 // tp；前面再挂一个负载均衡 / KV 感知路由
            print(cmd)
          `,
        },
      ],
      explanation: md`
        **两个约束**：显存决定 batch 上限 $B_{mem}$，TPOT SLO 决定 $B_{slo}$。$t(B)$ 关于 $B$ 单调递增，所以 $B_{slo}$ 可以二分，也可以由两段线性函数直接解出再修正浮点边界。

        **为什么 TP 越大吞吐可能越高**：每卡权重变小，单步读权重的时间变短、留给 KV 的显存变多；KV 头在卡间平分，每卡每序列 KV 也变小。示例中 70B 用 TP2 只剩 1.4 GB 给 KV，batch 只有 2；TP8 每卡 54 GB 给 KV，batch 323。

        **局限**：模型忽略了 all-reduce 的带宽项（随 $B \cdot d$ 增长）、kernel 效率、调度与采样开销，也没有考虑 prefill 的干扰，所以会系统性地偏向大 TP。真实规划用它圈出候选，再实测校准。

        **常见错误**：TP 超过 KV 头数时仍按 tp 平分 KV；忘了通信常数；吞吐乘副本数时用了 \`n_gpus\` 而不是 \`n_gpus // tp\`；边界上 \`t(B) == slo\` 的处理。
      `,
      complexity: { time: 'O(log n_gpus)（每个 tp 闭式求解）', space: 'O(1)' },
      hints: [
        md`每卡：\`W = N*w/tp\`，\`K = ctx*kappa/min(tp, n_kv)\`，\`b_mem = floor((0.9*mem - W)/K)\`。`,
        md`\`t(B) = max(a*B/(tp*F), (W + B*K)/BW) + comm\`，\`a = 2N + 4*L*n_h*d_h*ctx\`；它关于 B 单调，用二分或闭式求最大的 \`t(B) <= slo\` 的 B。`,
        md`吞吐 = \`(n_gpus // tp) * B / t(B)\`，严格更大才替换，自然实现「相同取较小 tp」。`,
      ],
      setup: py`
        import math, random

        _C70 = dict(n_params=7.06e10, n_layers=80, n_heads=64, n_kv_heads=8, head_dim=128, w_bytes=2, kv_bytes=2)
        _C8 = dict(n_params=8.03e9, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, w_bytes=2, kv_bytes=2)
        _H100 = dict(flops=989e12, bw=3.35e12, mem=80e9)

        def _ref_plan(cfg, gpu, n, S, slo, ar):
            N, L = cfg["n_params"], cfg["n_layers"]
            kap = 2 * L * cfg["n_kv_heads"] * cfg["head_dim"] * cfg["kv_bytes"]
            a = 2 * N + 4 * L * cfg["n_heads"] * cfg["head_dim"] * S
            best, tp = None, 1
            while tp <= n:
                W = N * cfg["w_bytes"] / tp; K = S * kap / min(tp, cfg["n_kv_heads"])
                c = 2 * L * ar if tp > 1 else 0.0
                t = lambda B: max(a * B / (tp * gpu["flops"]), (W + B * K) / gpu["bw"]) + c
                bm = math.floor((0.9 * gpu["mem"] - W) / K)
                if bm >= 1 and t(1) <= slo:
                    lo, hi = 1, 1
                    while t(hi * 2) <= slo: hi *= 2
                    hi *= 2
                    while lo < hi - 1:                       # 二分：最大的 t(B) <= slo
                        mid = (lo + hi) // 2
                        if t(mid) <= slo: lo = mid
                        else: hi = mid
                    B = min(bm, lo); thr = (n // tp) * B / t(B)
                    if best is None or thr > best["tokens_per_s"] * (1 + 1e-12):
                        best = {"tp": tp, "batch": B, "tokens_per_s": thr}
                tp *= 2
            return best

        def _cmp(got, exp, tag):
            if exp is None:
                assert got is None, f"{tag}: 全部不可行，应返回 None，得到 {got}"
                return
            assert isinstance(got, dict), f"{tag}: 应返回 dict，得到 {got}"
            assert (got["tp"], got["batch"]) == (exp["tp"], exp["batch"]), f"{tag}: (tp, batch) 期望 {(exp['tp'], exp['batch'])}，得到 {(got['tp'], got['batch'])}"
            assert math.isclose(got["tokens_per_s"], exp["tokens_per_s"], rel_tol=1e-9), f"{tag}: 吞吐期望 {exp['tokens_per_s']:.3f}，得到 {got['tokens_per_s']}"
      `,
      tests: [
        {
          name: '示例：70B 在 8×H100',
          code: py`
            got = plan_serving(_C70, _H100, 8, 4096, 0.03, 2e-5)
            assert got is not None and got["tp"] == 8 and got["batch"] == 323, f"期望 tp=8、batch=323，得到 {got}"
            assert abs(got["tokens_per_s"] - 13106.16) < 0.5, f"吞吐期望约 13106.16，得到 {got['tokens_per_s']:.2f}"
          `,
        },
        {
          name: '放不下：返回 None',
          code: py`
            got = plan_serving(_C70, _H100, 1, 4096, 0.03, 2e-5)
            assert got is None, f"单卡 80 GB 放不下 141 GB 权重，应返回 None，得到 {got}"
          `,
        },
        {
          name: 'SLO 太紧：返回 None',
          code: py`
            got = plan_serving(_C8, _H100, 1, 2048, 0.001, 2e-5)
            assert got is None, f"单步下界约 4.9 ms，1 ms 的 SLO 不可能满足，得到 {got}"
          `,
        },
        {
          name: '与参考实现一致（多组配置）',
          hidden: true,
          code: py`
            cases = [(_C70, _H100, 8, 4096, 0.03, 2e-5), (_C8, _H100, 8, 2048, 0.02, 2e-5), (_C8, _H100, 4, 8192, 0.012, 1e-5),
                     (_C70, dict(flops=989e12, bw=4.8e12, mem=141e9), 4, 8192, 0.025, 2e-5), (_C8, _H100, 2, 128, 0.006, 0.0)]
            for i, c in enumerate(cases):
                _cmp(plan_serving(*c), _ref_plan(*c), f"case {i}")
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(12)
            for i in range(300):
                nkv = rnd.choice([1, 2, 4, 8]); L = rnd.choice([16, 32, 48, 80])
                cfg = dict(n_params=rnd.uniform(1e9, 1.5e11), n_layers=L, n_heads=nkv * rnd.choice([1, 4, 8]), n_kv_heads=nkv,
                           head_dim=rnd.choice([64, 128]), w_bytes=rnd.choice([0.5, 1, 2]), kv_bytes=rnd.choice([1, 2]))
                gpu = dict(flops=rnd.uniform(2e14, 2.5e15), bw=rnd.uniform(1.5e12, 8e12), mem=rnd.choice([40e9, 80e9, 141e9, 180e9]))
                args = (cfg, gpu, rnd.choice([1, 2, 4, 8, 16]), rnd.choice([512, 2048, 8192, 32768]), rnd.uniform(0.003, 0.08), rnd.choice([0.0, 1e-5, 3e-5]))
                _cmp(plan_serving(*args), _ref_plan(*args), f"random {i}")
          `,
        },
      ],
      followUps: [
        {
          id: 'f-prefill',
          q: md`如果还要满足 TTFT SLO，规划要怎么扩展？`,
          a: md`要估算 prefill 占用：每个新请求需要 $\approx (2NP + 2Ln_hd_hP^2)$ FLOPs，按到达率折算成每秒的算力需求；混部时用 chunked prefill 的 token 预算把它摊进每次迭代，迭代时间随之增加，$B_{slo}$ 下降。更彻底的是 PD 分离：分别为 prefill 池按 TTFT、为 decode 池按 TPOT 做上面的规划，再加上 KV 传输带宽的约束（DistServe 的思路）。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'b-qa-system',
      freq: 3,
      q: md`如果让你从零优化一个 LLM 在线服务的吞吐和延迟，你会按什么顺序做？`,
      a: md`
        **结论**：先测量定位瓶颈，再按「收益 / 成本」从高到低上手段。

        **顺序**：① 用 roofline 估算上界，对比实测（带宽利用率、MFU），明确 prefill / decode 各自的瓶颈；② 基础设施：连续批处理 + PagedAttention（vLLM / SGLang 默认）、CUDA Graph、合适的 TP / DP；③ 前缀缓存 + 应用侧调整 prompt 顺序，直接降 TTFT；④ chunked prefill 稳住 TPOT，按 SLO 调 token 预算；⑤ 量化（FP8 权重 / KV）提升 batch 与 decode 速度；⑥ 低并发场景开投机解码（EAGLE-3 / MTP），高并发时自动收缩；⑦ 规模大、SLO 严时考虑 PD 分离与 KV 分级缓存。

        **延伸**：所有改动都以 goodput（满足 SLO 的吞吐）衡量，而不是单看峰值吞吐。
      `,
    },
  ],

  cheatsheet: md`
    **定位瓶颈**：强度 vs 拐点（A100 153 · H100 295 · H200 206 · B200 281）；decode 线性层 ≈ batch，注意力 = $n_h/n_{kv}$
    **显存**：PagedAttention（块表、COW、抢占 swap / recompute）→ 更大 batch；连续批处理（迭代级调度）
    **prefill**：前缀缓存（radix / 链式哈希，只缓存满块）、chunked prefill（token 预算）、PD 分离（KV 传输按层流水）
    **decode 串行**：投机解码 $\min(1,p/q)$ + $\mathrm{norm}(\max(0,p-q))$；$E=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$，加速 $E/(\gamma c+1)$
    **草稿**：Medusa（多头 + 树）→ EAGLE（特征自回归）→ EAGLE-2（动态树）→ EAGLE-3（去特征约束 + 多层融合 + TTT）；MTP 作草稿；P-EAGLE 并行起草

    \`\`\`python
    k = min(gamma, max_new - len(out) - 1)          # 贪心投机解码一轮
    g = verify(seq, drafts)                          # 一次目标前向 → k+1 个 argmax
    n = 0
    while n < k and drafts[n] == g[n]: n += 1
    out += drafts[:n] + [g[n]]                       # 接受 + 修正 / bonus
    \`\`\`
  `,
})
