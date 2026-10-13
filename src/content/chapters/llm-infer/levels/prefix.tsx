/**
 * L10-3 前缀缓存、Chunked Prefill 与 PD 分离。
 * 一手资料：SGLang（Zheng et al., arXiv 2312.07104，NeurIPS 2024）；vLLM 文档 Automatic Prefix Caching 设计、Optimization and Tuning；
 *          Sarathi-Serve（arXiv 2403.02310，OSDI 2024）；DistServe（arXiv 2401.09670，OSDI 2024）；Mooncake（arXiv 2407.00079，FAST 2025 最佳论文）
 */
import { defineLevel, md, py } from '@/content/helpers'
import { RadixTreeViz } from '../viz/RadixTreeViz'
import { BlockHashViz } from '../viz/BlockHashViz'
import { ChunkedPrefillViz } from '../viz/ChunkedPrefillViz'
import { PDTransferViz } from '../viz/PDTransferViz'

export default defineLevel({
  learn: [
    {
      id: 'why',
      title: '为什么要跨请求复用 KV',
      body: md`
        上一关的写时复制只能在**同一个请求**的多个序列之间共享 KV。真实流量里，**不同请求**之间的公共前缀多得多：

        | 场景 | 公共前缀 |
        |---|---|
        | 同一应用的所有请求 | system prompt、工具 / 函数定义、few-shot 示例 |
        | 多轮对话 | 第 $n$ 轮的输入 = 前 $n-1$ 轮的全部历史 + 新问题 |
        | Agent / 工具调用循环 | 每一步都把之前的轨迹原样带上 |
        | 长文档问答、RAG | 同一文档被多次提问 |
        | 自洽性采样、树搜索 | 同一问题采很多条推理路径 |

        前缀的 KV 只取决于前缀本身（以及模型、LoRA 等），所以算过一次就可以留着：命中部分**跳过 prefill**（TTFT 直接下降），而且多个请求**共享同一份物理 KV**（省显存）。
        两种主流实现：SGLang 的 **RadixAttention**（radix tree，token 粒度）和 vLLM 的 **Automatic Prefix Caching**（链式块哈希，块粒度）。

        > [!TIP] 应用侧也要配合
        > 缓存只认「完全相同的前缀」。把稳定的内容（system prompt、工具定义）放在最前面，把时间戳、用户 ID 这类每次都变的内容放到后面，命中率会高很多——各家 API 的 prompt caching 文档也都这么建议。
      `,
      checkpoint: {
        id: 'cp-why',
        type: 'single',
        prompt: md`前缀缓存命中后，下列哪一项**不会**因此改善？`,
        options: ['TTFT（跳过已缓存部分的 prefill）', '多个请求共享前缀 KV，节省显存', '命中前缀之后每个新 token 的 decode 计算量', 'prefill 阶段消耗的 GPU 算力'],
        answer: 2,
        explanation: md`decode 时每个新 token 仍要对全部历史（包括缓存的前缀）做注意力、跑完整个模型，计算量不变。前缀缓存省的是**重复的 prefill** 与**重复的 KV 存储**。`,
      },
    },
    {
      id: 'radix',
      title: 'SGLang RadixAttention：radix tree + LRU 淘汰叶子',
      body: md`
        SGLang（Zheng 等，NeurIPS 2024）用一棵 **radix tree**（压缩前缀树：每条边是一段 token 序列）把 token 序列映射到它们的 KV；KV 以「每页 1 个 token」的分页方式存放，所以共享粒度是 **token 级**。

        - **匹配**：新请求从根往下走，找到最长的已缓存前缀，这部分 KV 直接复用；
        - **插入**：请求结束（或 prefill 完成）后把它的 token 插入树；若在某条边的中间分叉，就把边**分裂**成两段；
        - **淘汰**：显存不够时按 **LRU 淘汰叶子**；内部节点只有在孩子都被淘汰、自己变成叶子后才会被淘汰，所以被大量请求共享的前缀活得最久；
        - **引用计数**：正在被运行中请求使用的节点 ref > 0，不可淘汰。

        论文还提出 **cache-aware scheduling**：等待队列按「匹配到的前缀长度」从长到短排序（longest-shared-prefix-first）。这等价于按 radix tree 的**深度优先**顺序处理请求；
        论文证明离线场景下 DFS 顺序（缓存容量 ≥ 最长请求时）能达到最优命中率，在线时只能近似，而且贪心排序可能让请求饿死。
        论文报告各基准上的缓存命中率在 50%–99%，Chatbot Arena 线上部署中为 52.4%（LLaVA-Next-34B）与 74.1%（Vicuna-33B）。
      `,
      viz: RadixTreeViz,
      after: md`
        > [!NOTE] 这和「算法 · Trie」是同一个数据结构
        > radix tree 就是把只有一个孩子的链压缩成一条边的 Trie。匹配是 $O(\text{前缀长度})$，插入时最多分裂一条边。实战题 2 会让你手写它。
      `,
      checkpoint: {
        id: 'cp-radix',
        type: 'judge',
        prompt: md`RadixAttention 的 LRU 淘汰可以直接淘汰一个仍有孩子的内部节点，只要它是最久未被使用的。`,
        answer: false,
        explanation: md`只淘汰**叶子**。内部节点代表多个请求共享的前缀，孩子还在时它的 KV 必须保留（孩子的 KV 依赖它）；孩子全被淘汰后它变成叶子，才进入淘汰候选。另外，正被运行中请求引用（ref > 0）的节点也不能淘汰。`,
      },
    },
    {
      id: 'apc',
      title: 'vLLM APC：链式块哈希',
      body: md`
        vLLM 的 automatic prefix caching 不建树，而是给每个 KV 块一个**哈希**，维护「哈希 → 物理块」的表：

        $$
        h_i = \mathrm{hash}\big(h_{i-1},\ \text{第 } i \text{ 块的 token},\ \text{额外键}\big)
        $$

        - 父块哈希让 $h_i$ 编码了**从开头到第 $i$ 块的整个前缀**：任何位置不同，此后所有块都不同；
        - 额外键：LoRA ID、多模态输入的哈希、cache salt（只注入第一块，用于隔离不同租户）；
        - **只缓存满块**，最后一个未满的块不参与；
        - 哈希算法：v0.11 起默认 \`sha256\`（此前版本的哈希不保证无碰撞），也可选 xxhash；
        - **命中**：新请求逐块算哈希查表，命中的块引用计数 + 1（并从空闲队列中摘出，防止被淘汰）；
        - **淘汰**：引用计数为 0 的块并不立即清空，而是进入空闲队列（双向链表）；需要新块时从队头取，若它带着哈希就顺便把哈希删掉——这就是 LRU。
          一个请求释放时，它的块按**逆序**放到队尾：越靠后的块包含的前缀越长、越不容易被复用，应该先被淘汰。

        与 radix tree 相比：块哈希天然嵌在分页分配器里，查找是 $O(1)$ 每块；代价是只能以块为粒度命中（块越大越粗）。
      `,
      viz: BlockHashViz,
      checkpoint: {
        id: 'cp-apc',
        type: 'numeric',
        prompt: md`block_size = 16，两个请求共享一个 1000 token 的 system prompt，之后内容不同。第二个请求最多能命中多少个 token？`,
        answer: 992,
        tolerance: 0,
        unit: 'token',
        explanation: md`只有完全落在公共前缀里的**满块**能命中：$\lfloor 1000/16 \rfloor = 62$ 块，$62 \times 16 = 992$ 个 token；第 63 块包含 8 个公共 token 和 8 个不同 token，哈希不同。RadixAttention 是 token 粒度，可以命中全部 1000 个。`,
      },
    },
    {
      id: 'chunked',
      title: 'Chunked prefill：把长 prefill 切块混进 decode',
      body: md`
        连续批处理把新请求的 prefill 和老请求的 decode 放进同一次迭代。问题在于两者的代价相差悬殊：一个 4000 token 的 prefill 让这次迭代变长十几倍，
        所有正在 decode 的请求都会遇到一次「卡顿」（ITL / TPOT 尖峰）。

        **Chunked prefill**（SARATHI / Sarathi-Serve，Agrawal 等，OSDI 2024）：每次迭代设一个 **token 预算**（vLLM 中的 \`max_num_batched_tokens\`），
        先放入所有 decode 请求（每个 1 个 token），剩下的预算用来装一块 prefill；长 prompt 被切成多块，分几次迭代完成。

        这样做有两个好处：

        1. **限制每次迭代的耗时**：decode 的间隔有了上界（stall-free scheduling），TPOT 尾延迟大幅下降；
        2. **搭便车**：只有 decode 的迭代是访存瓶颈，算力大量闲置；塞进一块 prefill 几乎不增加耗时（piggybacking）。

        取舍：预算小 → decode 间隔更稳定，但 prefill 被切得更碎，每块都要重新读前面各块的 KV，TTFT 变差、prefill 效率下降；预算大 → TTFT 好，但 decode 间隔变大。
        vLLM V1 默认开启 chunked prefill，调度策略是先排所有 decode，再用剩余预算排 prefill；文档建议较小的预算（如 2048）换更好的 ITL，追求吞吐时用 > 8192。
        Sarathi-Serve 报告：在尾延迟约束下，相对 vLLM 的服务容量提升 Mistral-7B（单卡 A100）2.6×、Yi-34B（2 卡）最高 3.7×、Falcon-180B（流水并行）最高 5.6×。
      `,
      viz: ChunkedPrefillViz,
      checkpoint: {
        id: 'cp-chunked',
        type: 'fill',
        prompt: md`token 预算为 2048，当前有 30 个请求在 decode。一个 8000 token 的新 prompt 到达后，每次迭代最多处理 ____(1)____ 个 prefill token，需要 ____(2)____ 次迭代才能完成它的 prefill。`,
        blanks: [
          { label: '(1)', answers: ['2018'], numeric: true, tolerance: 0 },
          { label: '(2)', answers: ['4'], numeric: true, tolerance: 0 },
        ],
        explanation: md`decode 优先：$2048 - 30 = 2018$；$\lceil 8000 / 2018 \rceil = 4$（前三次各 2018，第四次 1946）。`,
      },
    },
    {
      id: 'pd',
      title: 'PD 分离：prefill 和 decode 放到不同的 GPU',
      body: md`
        chunked prefill 是在同一批 GPU 上「调和」两种负载；**PD 分离**（prefill-decode disaggregation）则干脆把它们拆开：prefill 实例只做 prefill，算完的 KV 传给 decode 实例继续生成。

        **动机**：
        - 两种负载的瓶颈不同（算力 vs 带宽），混在一起互相干扰：prefill 拖慢 decode 的 TPOT，decode 占着 batch 位置拖慢 prefill 的 TTFT；
        - 两个阶段的最优并行策略不同（prefill 适合更大的张量并行压低 TTFT，decode 适合更大的 batch 与数据并行），资源可以按 TTFT / TPOT 两个 SLO **分别**配比。

        **DistServe**（Zhong 等，OSDI 2024）以 **goodput**（同时满足 TTFT 与 TPOT SLO 的最大请求率）为目标，分别为两阶段选择资源与并行策略，并按集群带宽放置实例以减小传输开销；
        报告相对当时的系统可多服务 7.4× 的请求，或满足 12.6× 更紧的 SLO。同期还有微软的 Splitwise（ISCA 2024）。
        **Mooncake**（月之暗面 Kimi 的服务平台，FAST 2025 最佳论文）以 KV cache 为中心：把 GPU 服务器上闲置的 CPU 内存、DRAM、SSD 组成分布式 KV 池，prefill / decode 分离，
        调度器围绕「KV 在哪」做决策，过载时基于预测提前拒绝请求；arXiv 版摘要报告模拟场景下吞吐最高 +525%，真实负载下多处理 75% 的请求。

        **代价**：KV 要跨机器传输，量级 = prompt 长度 × 每 token KV（70B GQA 模型约 320 KiB / token）。通常按层流水传输（算完一层发一层）把它和 prefill 计算重叠；
        GQA / MLA / FP8 KV 直接减少要传的字节。另外两个池子的配比要随流量调整，系统复杂度明显上升。
      `,
      viz: PDTransferViz,
      after: md`
        > [!INTERVIEW] 什么时候不值得做 PD 分离
        > 规模小（几张卡）、prompt 短、SLO 不严时，chunked prefill 在单一实例上已经足够，分离带来的 KV 传输和运维复杂度不划算。
        > 大规模部署、长 prompt（RAG、长文档、长多轮对话）、TTFT 与 TPOT SLO 都很严时，PD 分离的收益最明显；配合跨实例的前缀缓存池（如 Mooncake）还能进一步减少 prefill。
      `,
      checkpoint: {
        id: 'cp-pd',
        type: 'numeric',
        prompt: md`Llama-3.1-70B（每 token KV = 327680 字节，BF16）一个 8192 token 的 prompt 做完 prefill，KV 经 400 Gb/s 网络（按 50 GB/s，GB = $10^9$）传给 decode 节点，大约需要多少毫秒？`,
        answer: 53.7,
        tolerance: 0.02,
        unit: 'ms',
        explanation: md`$8192 \times 327680 = 2.68 \times 10^9$ 字节，$2.68 \times 10^9 / 5 \times 10^{10} \approx 53.7$ ms。同一个 prompt 在 8×H100 上 prefill 的理论下界约 157 ms，按层流水可以把传输基本藏住；但网络慢 4 倍（100 Gb/s）时就藏不住了。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-granularity',
      type: 'single',
      difficulty: 1,
      prompt: md`SGLang RadixAttention 与 vLLM APC 在**复用粒度**上的区别是：`,
      options: [
        '两者都只能复用整个请求',
        'RadixAttention 是 token 粒度；APC 以块为粒度，且只缓存满块',
        'RadixAttention 以块为粒度；APC 是 token 粒度',
        '两者都是 token 粒度',
      ],
      answer: 1,
      explanation: md`SGLang 的 KV 以每页 1 个 token 存放，radix tree 可以在任意 token 处分叉；vLLM APC 给每个满块算链式哈希，最后不满的块不缓存，命中长度是 block_size 的整数倍。`,
    },
    {
      id: 'q-middle',
      type: 'judge',
      difficulty: 1,
      prompt: md`vLLM APC 中，两个请求的第 1 个块不同，但第 3 个块的 token 完全相同，则第二个请求的第 3 个块可以命中缓存。`,
      answer: false,
      explanation: md`$h_3$ 包含 $h_2$、$h_2$ 又包含 $h_1$，第 1 块不同导致之后所有块的哈希都不同。这是正确的语义：第 3 块的 KV 取决于它之前的全部 token，内容相同但前缀不同时 KV 并不相同。`,
    },
    {
      id: 'q-hashkey',
      type: 'multi',
      difficulty: 2,
      prompt: md`vLLM APC 计算块哈希时会纳入哪些信息？（多选）`,
      options: ['父块的哈希', '本块的 token', 'LoRA ID', '多模态输入（如图像）的哈希', '请求的采样温度', '请求 ID'],
      answer: [0, 1, 2, 3],
      explanation: md`采样温度不影响 KV（KV 只由输入 token 与模型权重决定），请求 ID 加进去就无法跨请求复用了。cache salt 也是可选的额外键（注入第一块），用于在多租户之间隔离缓存。`,
    },
    {
      id: 'q-hit',
      type: 'numeric',
      difficulty: 2,
      prompt: md`block_size = 32。请求 A 有 1500 个 token，请求 B 与 A 的前 1300 个 token 相同、之后不同。A 处理完后 B 到达，B 最多命中多少个 token？`,
      answer: 1280,
      tolerance: 0,
      unit: 'token',
      explanation: md`公共前缀 1300 个 token，其中满块 $\lfloor 1300/32 \rfloor = 40$ 块 = 1280 个 token。第 41 块（1280–1311）包含不同的 token，哈希不同。`,
    },
    {
      id: 'q-pd-bytes',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Llama-3.1-8B 每 token KV 为 131072 字节（BF16）。PD 分离下，一个 32768 token 的 prompt 需要传输多少 GiB 的 KV（GiB = $2^{30}$ 字节）？`,
      answer: 4,
      tolerance: 0.001,
      unit: 'GiB',
      explanation: md`$32768 \times 131072 = 2^{15} \times 2^{17} = 2^{32}$ 字节 = 4 GiB。用 FP8 KV 可以减半；MLA 这类压缩 KV 的结构会让这个数小一个量级。`,
    },
    {
      id: 'q-fill-chunk',
      type: 'fill',
      difficulty: 2,
      prompt: md`token 预算 1024，正在 decode 的请求有 24 个。一个 5000 token 的 prompt 到达，每次迭代可装 ____(1)____ 个 prefill token，共需 ____(2)____ 次迭代完成 prefill，最后一次装 ____(3)____ 个。`,
      blanks: [
        { label: '(1)', answers: ['1000'], numeric: true, tolerance: 0 },
        { label: '(2)', answers: ['5'], numeric: true, tolerance: 0 },
        { label: '(3)', answers: ['1000'], numeric: true, tolerance: 0 },
      ],
      explanation: md`$1024 - 24 = 1000$，$5000 / 1000 = 5$ 次，恰好整除，最后一次也是 1000。（假设这期间 decode 数量不变。）`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把一个请求在 SGLang（RadixAttention）中的生命周期排序：`,
      items: [
        '在 radix tree 中做最长前缀匹配',
        '给命中路径上的节点加引用计数（防止被淘汰）',
        '只对未命中的部分做 prefill，然后 decode',
        '请求结束，把完整序列插入 radix tree（必要时分裂边）',
        '释放引用计数；显存不足时按 LRU 淘汰 ref = 0 的叶子',
      ],
      explanation: md`匹配 → 锁定 → 计算 → 插入 → 解锁 / 淘汰。命中的节点在请求运行期间必须锁住，否则可能在使用中被淘汰。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把技术与它解决的问题配对：`,
      pairs: [
        ['RadixAttention / APC', '跨请求重复计算相同前缀的 prefill'],
        ['chunked prefill', '长 prefill 让同批 decode 出现 TPOT 尖峰'],
        ['PD 分离', '两阶段互相干扰、最优并行策略不同'],
        ['cache-aware scheduling', '请求处理顺序导致缓存被反复淘汰、命中率低'],
        ['写时复制（COW）', '同一请求的多个采样序列共享 prompt 的 KV'],
      ],
      explanation: md`前缀缓存解决跨请求复用；COW（上一关）解决同一请求内的共享；chunked prefill 与 PD 分离分别从「同机调和」和「分机部署」两个方向解决 prefill / decode 干扰；cache-aware scheduling 按前缀长度排序，近似 DFS 顺序以提高命中率。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面计算前缀块哈希并统计命中的代码有 **2 处** bug，点出来：`,
      code: py`
        def block_hashes(tokens, B):
            hashes, parent = [], None
            for s in range(0, len(tokens), B):
                block = tuple(tokens[s:s + B])
                h = hash(block)
                hashes.append(h)
                parent = h
            return hashes

        def cached_tokens(tokens, B, cache):
            n = 0
            for h in block_hashes(tokens, B):
                if h not in cache:
                    break
                n += B
            return n
      `,
      bugLines: [3, 5],
      explanation: md`
        - 第 3 行：\`range(0, len(tokens), B)\` 会把最后不满的块也算进去，应改为 \`range(0, len(tokens) - len(tokens) % B, B)\`——只缓存满块（否则 \`n += B\` 还会把不满的块按整块计数）。
        - 第 5 行：哈希没有包含父块，应为 \`hash((parent, block))\`。否则内容相同但前缀不同的块会错误命中，复用了错误的 KV。
      `,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`
        block_size = 4。请求 1 的 token 为 \`abcd efgh ijkl mn\`，已全部写入缓存。请求 2 为 \`abcd efgh ijXl mnop qrst\`。请求 2 的哪些块会命中？
      `,
      items: ['块0 abcd', '块1 efgh', '块2 ijXl', '块3 mnop', '块4 qrst'],
      answer: [0, 1],
      explanation: md`块 0、1 内容和前缀都相同，命中；块 2 内容不同；块 3 虽然以 \`mn\` 开头，但请求 1 的 \`mn\` 是不满的块、从未被缓存，而且它的前缀（块 2）也已不同。命中 8 个 token。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 radix tree 的最长前缀匹配（\`node.children\` 以首 token 为键，\`child.key\` 是边上的 token 元组）。有 2 行干扰项：`,
      lines: [
        'def match_prefix(root, tokens):',
        '    node, i = root, 0',
        '    while i < len(tokens):',
        '        child = node.children.get(tokens[i])',
        '        if child is None:',
        '            return i',
        '        k = 0',
        '        while k < len(child.key) and i + k < len(tokens) and child.key[k] == tokens[i + k]:',
        '            k += 1',
        '        i += k',
        '        if k < len(child.key):',
        '            break',
        '        node = child',
        '    return i',
      ],
      distractors: ['        i += len(child.key)', '            return 0'],
      explanation: md`在边中间失配时，已匹配的 \`k\` 个 token 仍然算命中（token 粒度），然后停止。干扰项一个是不比较就整条边计入，一个是失配就清零。`,
    },
    {
      id: 'q-pd-when',
      type: 'single',
      difficulty: 3,
      prompt: md`以下哪种场景**最不适合**引入 PD 分离？`,
      options: [
        '大规模集群，长文档问答，TTFT 与 TPOT 的 SLO 都很严',
        '单机 2 卡，短 prompt 聊天，并发不高，节点间只有 25 Gb/s 网络',
        '多轮长对话、KV 可以放进跨机的 KV 池复用',
        'prefill 与 decode 的最优并行策略差异很大的超大模型',
      ],
      answer: 1,
      explanation: md`规模小、prompt 短时 prefill 干扰本来就小，chunked prefill 足够；跨机 KV 传输在慢网络上反而成为瓶颈，复杂度也不划算。A、C、D 都是 PD 分离（以及 Mooncake 式 KV 池）收益明显的场景。`,
    },
  ],

  guided: [
    {
      id: 'g-chain-hash',
      title: '链式块哈希与命中计算',
      intro: md`
        实现 vLLM APC 的两个核心函数：
        - \`block_hashes(tokens, block_size, extra=None)\`：为每个**满块**计算 $h_i = \mathrm{hash}((h_{i-1}, \text{块}, \text{extra}))$，$h_{-1} = $ \`None\`；
        - \`cached_tokens(tokens, block_size, cache, extra=None)\`：从第一块开始逐块查 \`cache\`（一个哈希集合），返回连续命中的 token 数。
      `,
      template: py`
        def block_hashes(tokens, block_size, extra=None):
            hashes, parent = [], None
            for start in range(0, {{1}}, block_size):
                block = tuple(tokens[start:start + block_size])
                h = hash({{2}})
                hashes.append(h)
                parent = {{3}}
            return hashes

        def cached_tokens(tokens, block_size, cache, extra=None):
            n = 0
            for h in block_hashes(tokens, block_size, extra):
                if {{4}}:
                    break
                n += block_size
            return n
      `,
      steps: [
        {
          blank: 1,
          instruction: md`循环上界：只遍历**满块**。`,
          answer: 'len(tokens) - len(tokens) % block_size',
          accept: ['len(tokens) // block_size * block_size', '(len(tokens) // block_size) * block_size'],
          hint: '把不满一块的尾巴去掉。',
          explain: md`最后不满的块还会继续被写入，内容未定，不能作为缓存键。`,
        },
        {
          blank: 2,
          instruction: md`哈希的内容：父哈希、本块 token、额外键（LoRA ID 等）。`,
          answer: '(parent, block, extra)',
          accept: ['(parent, block, extra,)'],
          explain: md`父哈希把整个前缀「串」进来；extra 让不同 LoRA / 多模态输入的同一段 token 互不命中。`,
        },
        {
          blank: 3,
          instruction: md`更新父哈希。`,
          answer: 'h',
          explain: md`链式结构：$h_i$ 依赖 $h_{i-1}$。`,
        },
        {
          blank: 4,
          instruction: md`第一次未命中就停止。`,
          answer: 'h not in cache',
          accept: ['not h in cache', 'h not in set(cache)'],
          explain: md`命中只能是从开头连续的一段：中间断开之后，即使后面某块恰好在缓存里（不可能，因为哈希包含前缀）也无法使用。`,
        },
      ],
      tests: [
        {
          name: '共享前缀命中',
          code: py`
            sys_p = list(range(1, 13))
            cache = set(block_hashes(sys_p + [50, 51, 52, 53, 54], 4))
            got = cached_tokens(sys_p + [60, 61, 62, 63], 4, cache)
            assert got == 12, f"system prompt 的 3 个满块应命中 12 个 token，得到 {got}"
          `,
        },
        {
          name: '前缀不同则全部不命中',
          code: py`
            cache = set(block_hashes([1, 2, 3, 4, 5, 6, 7, 8], 4))
            got = cached_tokens([9, 9, 9, 9, 5, 6, 7, 8], 4, cache)
            assert got == 0, f"第 1 块不同，第 2 块内容相同也不能命中，得到 {got}"
          `,
        },
        {
          name: '只缓存满块',
          code: py`
            hs = block_hashes([1, 2, 3, 4, 5, 6, 7], 4)
            assert len(hs) == 1, f"7 个 token、块大小 4，只有 1 个满块，得到 {len(hs)} 个哈希"
            assert cached_tokens([1, 2, 3, 4, 5, 6, 7], 4, set(hs)) == 4
          `,
        },
        {
          name: '额外键隔离（LoRA）',
          code: py`
            t = list(range(16))
            cache = set(block_hashes(t, 4, extra="lora-A"))
            assert cached_tokens(t, 4, cache, extra="lora-A") == 16
            assert cached_tokens(t, 4, cache, extra="lora-B") == 0, "不同 LoRA 不应命中"
          `,
        },
      ],
      outro: md`
        实战题 1 会在此基础上加上**容量**与**淘汰**：缓存满了要按 LRU 淘汰，而且同一时刻使用的块里，越靠后的块越先淘汰（vLLM 把释放的块逆序放进空闲队列）。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-prefix-hits',
      title: '链式哈希前缀缓存：带 LRU 的命中计算',
      difficulty: 'medium',
      source: 'vLLM Automatic Prefix Caching',
      timeLimitMin: 25,
      description: md`
        依次处理一串请求（每个请求是一个 token 列表），模拟一个容量为 \`capacity\` 个块的前缀缓存，返回每个请求命中的 token 数：

        \`\`\`python
        def prefix_cache_hits(requests: list[list[int]], block_size: int, capacity: int) -> list[int]: ...
        \`\`\`

        对第 $r$ 个请求（$r$ 从 0 起）：
        1. 只对**满块**计算链式哈希：$h_0 = H(\text{None}, \text{块}_0)$，$h_i = H(h_{i-1}, \text{块}_i)$（用 Python 的 \`hash(tuple)\` 即可）；
        2. **命中数** = 从第 0 块开始**连续**在缓存中的块数 × \`block_size\`；
        3. 本请求的所有满块（命中的与新插入的）都记为「在时刻 $r$ 被使用」；
        4. 插入未命中的块时，若缓存已有 \`capacity\` 个块，先淘汰一个：选**最近使用时刻最早**的块；时刻相同则淘汰**在序列中更靠后**（块下标更大）的块；
        5. 保证 \`capacity\` ≥ 任一请求的满块数（本请求的块不会被自己挤掉）。

        **示例**（\`block_size = 4\`，\`capacity = 16\`）：\`S = [1..8]\`
        \`requests = [S + [20,21,22,23,24], S + [30,31,32,33], [99] + S[1:] + [40], S + [20,21,22,23,24]]\` → \`[0, 8, 0, 12]\`

        **淘汰顺序示例**（\`block_size = 2\`，\`capacity = 4\`）：\`[[1,2,3,4], [5,6,7,8], [9,9], [1,2,3,4]]\` → \`[0, 0, 0, 2]\`：
        第 3 个请求插入时缓存已满，最旧的是请求 0 的两个块，先淘汰更靠后的 \`(3,4)\`，于是请求 3 只命中 \`(1,2)\`。
      `,
      starter: py`
        from collections import OrderedDict

        def prefix_cache_hits(requests, block_size, capacity):
            # 提示：OrderedDict 的队头 = 最先淘汰；本请求的块按「从深到浅」移到队尾
            raise NotImplementedError
      `,
      solution: py`
        from collections import OrderedDict

        def prefix_cache_hits(requests, block_size, capacity):
            cache = OrderedDict()                     # 块哈希 -> None；越靠队头越先被淘汰
            hits = []
            for toks in requests:
                n_full = len(toks) // block_size
                hashes, parent = [], None
                for b in range(n_full):               # 链式哈希：每块都编码了整个前缀
                    parent = hash((parent, tuple(toks[b * block_size:(b + 1) * block_size])))
                    hashes.append(parent)
                k = 0
                while k < n_full and hashes[k] in cache:
                    k += 1
                hits.append(k * block_size)
                for h in reversed(hashes[:k]):        # 先「锁住」命中的块：挪到队尾，避免下面插入时被淘汰
                    cache.move_to_end(h)
                for h in hashes[k:]:
                    if len(cache) >= capacity:
                        cache.popitem(last=False)     # 淘汰队头：最久未用、同时刻中最靠后的块
                    cache[h] = None
                for h in reversed(hashes):            # 本请求的块从深到浅依次放到队尾 → 深的先淘汰
                    cache.move_to_end(h)
            return hits
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：命中后把物理块拼成块表）',
          code: py`
            import torch

            def build_block_table(hashes, hash_to_block, allocator):
                """hashes: 本请求所有满块的链式哈希；hash_to_block: 哈希 -> 物理块号（GPU 上的 KV 已在这些块里）
                返回 (block_table, num_cached_tokens)，未命中部分分配新块等待 prefill"""
                table = []
                for h in hashes:
                    b = hash_to_block.get(h)
                    if b is None:
                        break
                    allocator.incref(b)                 # 命中：引用计数 +1，并从可淘汰队列摘出
                    table.append(b)
                n_cached = len(table)
                table += [allocator.alloc() for _ in range(len(hashes) - n_cached)]
                return torch.tensor(table, dtype=torch.int32), n_cached
          `,
        },
      ],
      explanation: md`
        **链式哈希**保证「第 $i$ 块命中 ⇒ 前 $i$ 块都相同」，所以命中只看从开头连续的部分。

        **淘汰顺序**用一个 \`OrderedDict\` 维护：队头最先淘汰。每处理完一个请求，把它的所有块**从深到浅**依次 \`move_to_end\`，于是队列整体按（使用时刻升序，块下标降序）排列——
        这正是 vLLM 把释放的块**逆序**放入空闲队列的原因：越靠后的块包含的前缀越长、越少被别的请求共享，应该先被淘汰；而父块总是比子块晚淘汰，缓存里不会出现「有子无父」的孤块。

        **一个坑**：插入新块前要先把命中的块挪到队尾，否则它们可能刚被命中就被淘汰。

        **复杂度**：每个块 $O(1)$（哈希 + OrderedDict 操作），总 $O(\sum |\text{请求}|)$。
      `,
      complexity: { time: 'O(Σ 请求长度)', space: 'O(capacity)' },
      hints: [
        md`先写链式哈希：\`parent = hash((parent, tuple(block)))\`，只遍历 \`len(toks) // block_size\` 个满块。`,
        md`命中：\`while k < n and hashes[k] in cache: k += 1\`。`,
        md`用 \`OrderedDict\`：插入前满了就 \`popitem(last=False)\`；最后 \`for h in reversed(hashes): cache.move_to_end(h)\`。`,
        md`别忘了在插入之前先把命中的块挪到队尾。`,
      ],
      setup: py`
        import random

        def _ref_hits(requests, B, cap):
            cache = {}                                # 哈希 -> (最近使用时刻, 块下标)
            out = []
            for r, toks in enumerate(requests):
                hs, p = [], None
                for b in range(len(toks) // B):
                    p = hash((p, tuple(toks[b * B:(b + 1) * B]))); hs.append(p)
                k = 0
                while k < len(hs) and hs[k] in cache:
                    k += 1
                out.append(k * B)
                for i in range(k):
                    cache[hs[i]] = (r, i)
                for i in range(k, len(hs)):
                    if len(cache) >= cap:
                        victim = min(cache, key=lambda h: (cache[h][0], -cache[h][1]))
                        del cache[victim]
                    cache[hs[i]] = (r, i)
            return out

        _S = list(range(1, 9))
      `,
      tests: [
        {
          name: '示例',
          code: py`
            req = [_S + [20, 21, 22, 23, 24], _S + [30, 31, 32, 33], [99] + _S[1:] + [40], _S + [20, 21, 22, 23, 24]]
            got = prefix_cache_hits(req, 4, 16)
            assert list(got) == [0, 8, 0, 12], f"期望 [0, 8, 0, 12]，得到 {got}"
          `,
        },
        {
          name: '淘汰顺序：同时刻先淘汰更靠后的块',
          code: py`
            got = prefix_cache_hits([[1, 2, 3, 4], [5, 6, 7, 8], [9, 9], [1, 2, 3, 4]], 2, 4)
            assert list(got) == [0, 0, 0, 2], f"期望 [0, 0, 0, 2]，得到 {got}"
          `,
        },
        {
          name: '只缓存满块 / 前缀不同不命中',
          code: py`
            assert list(prefix_cache_hits([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]], 4, 8)) == [0, 4]
            assert list(prefix_cache_hits([[1, 2, 3, 4, 5, 6, 7, 8], [9, 9, 9, 9, 5, 6, 7, 8]], 4, 8)) == [0, 0]
          `,
        },
        {
          name: '命中的块不会在本请求插入时被淘汰',
          hidden: true,
          code: py`
            # 容量 3：请求 1 命中 1 块后还要插入 2 块，必须淘汰别人的块而不是刚命中的块
            got = prefix_cache_hits([[1, 1], [2, 2, 3, 3], [1, 1, 4, 4, 5, 5], [1, 1, 4, 4, 5, 5]], 2, 3)
            exp = _ref_hits([[1, 1], [2, 2, 3, 3], [1, 1, 4, 4, 5, 5], [1, 1, 4, 4, 5, 5]], 2, 3)
            assert list(got) == exp, f"期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(7)
            for case in range(150):
                B = rnd.choice([1, 2, 4]); maxb = rnd.randint(1, 6); cap = rnd.randint(maxb, maxb + 8)
                prefixes = [[rnd.randint(0, 3) for _ in range(rnd.randint(0, maxb * B))] for _ in range(3)]
                reqs = []
                for _ in range(rnd.randint(1, 20)):
                    p = list(rnd.choice(prefixes))
                    t = (p + [rnd.randint(0, 3) for _ in range(rnd.randint(0, maxb * B))])[: maxb * B + rnd.randint(0, B - 1)]
                    reqs.append(t)
                exp = _ref_hits(reqs, B, cap)
                got = list(prefix_cache_hits(reqs, B, cap))
                assert got == exp, f"case {case}: B={B}, cap={cap}, reqs={reqs}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：600 个多轮对话请求',
          hidden: true,
          code: py`
            rnd = random.Random(1)
            sys_p = [rnd.randint(0, 50000) for _ in range(256)]
            convs = [[] for _ in range(20)]
            reqs = []
            for _ in range(600):
                c = rnd.randrange(20)
                convs[c] = convs[c] + [rnd.randint(0, 50000) for _ in range(rnd.randint(8, 40))]
                if len(convs[c]) > 1500:
                    convs[c] = []
                reqs.append(sys_p + convs[c])
            got = list(prefix_cache_hits(reqs, 16, 400))
            exp = _ref_hits(reqs, 16, 400)
            assert got == exp, f"命中总数期望 {sum(exp)}，得到 {sum(got)}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-radix-vs-hash',
          q: md`为什么 vLLM 选择块哈希而不是 radix tree？`,
          a: md`块哈希天然和分页分配器融为一体：缓存的单位就是物理块，命中只需逐块查哈希表、增加引用计数，空闲队列兼作 LRU，实现简单、开销是每块 $O(1)$。radix tree 能做到 token 级复用并支持树形的 cache-aware 调度，但要维护一棵树并与分配器协调。两者的命中语义几乎相同，差别主要在粒度（满块 vs token）与工程取舍。`,
        },
      ],
    },
    {
      id: 'c-radix',
      title: '手写 Radix Tree：最长前缀匹配与插入（含边分裂）',
      difficulty: 'hard',
      source: 'SGLang RadixAttention',
      timeLimitMin: 30,
      description: md`
        实现一个只管 token 序列（不存 KV）的 radix tree：

        \`\`\`python
        class RadixCache:
            def __init__(self): ...
            def match_prefix(self, tokens: list[int]) -> int: ...   # 已缓存的最长前缀长度（token 粒度）
            def insert(self, tokens: list[int]) -> int: ...         # 插入；返回插入前已缓存的前缀长度
            def num_nodes(self) -> int: ...                         # 节点数（不含根）
            def total_tokens(self) -> int: ...                      # 所有边上的 token 总数
        \`\`\`

        **要求**：必须是**压缩**的前缀树——只有一个孩子且不是某次插入终点的链要合并成一条边。插入时若在某条边的中间分叉或结束，要把这条边**分裂**成两段。

        **示例**
        - \`insert([1,2,3])\` → 0；\`insert([1,2,4])\` → 2；此时节点为 \`[1,2]\`、\`[3]\`、\`[4]\`，\`num_nodes() == 3\`，\`total_tokens() == 4\`
        - \`match_prefix([1,2,9])\` → 2；\`match_prefix([1,2,3,7])\` → 3；\`match_prefix([5])\` → 0
        - 再 \`insert([1])\` → 1（已被缓存），把边 \`[1,2]\` 分裂为 \`[1]\` 和 \`[2]\`，\`num_nodes() == 4\`，\`total_tokens()\` 仍为 4

        **隐藏测试**：插入一条 1000 个 token 的序列后 \`num_nodes()\` 必须为 1（未压缩的 Trie 会得到 1000）。
      `,
      starter: py`
        class RadixCache:
            def __init__(self):
                # 节点：边上的 token 元组 + 孩子字典（以首 token 为键）
                pass

            def match_prefix(self, tokens):
                raise NotImplementedError

            def insert(self, tokens):
                raise NotImplementedError

            def num_nodes(self):
                raise NotImplementedError

            def total_tokens(self):
                raise NotImplementedError
      `,
      solution: py`
        class _Node:
            __slots__ = ("key", "children")

            def __init__(self, key):
                self.key = key          # 进入该节点的边上的 token（tuple）
                self.children = {}      # 首 token -> 子节点

        class RadixCache:
            def __init__(self):
                self.root = _Node(())
                self._nodes = 0
                self._tokens = 0

            @staticmethod
            def _common(key, tokens, i):
                k = 0
                while k < len(key) and i + k < len(tokens) and key[k] == tokens[i + k]:
                    k += 1
                return k

            def match_prefix(self, tokens):
                node, i = self.root, 0
                while i < len(tokens):
                    child = node.children.get(tokens[i])
                    if child is None:
                        break
                    k = self._common(child.key, tokens, i)
                    i += k
                    if k < len(child.key):          # 在边中间失配：已匹配的 k 个也算命中
                        break
                    node = child
                return i

            def insert(self, tokens):
                tokens = tuple(tokens)
                node, i, n = self.root, 0, len(tokens)
                while i < n:
                    child = node.children.get(tokens[i])
                    if child is None:                # 没有可走的边：挂新叶子
                        node.children[tokens[i]] = _Node(tokens[i:])
                        self._nodes += 1
                        self._tokens += n - i
                        return i
                    k = self._common(child.key, tokens, i)
                    if k < len(child.key):           # 在边中间分叉或结束：分裂成 key[:k] 与 key[k:]
                        mid = _Node(child.key[:k])
                        child.key = child.key[k:]
                        mid.children[child.key[0]] = child
                        node.children[tokens[i]] = mid
                        self._nodes += 1
                        child = mid
                    i += k
                    node = child
                return n                              # 整条序列早已在树中

            def num_nodes(self):
                return self._nodes

            def total_tokens(self):
                return self._tokens
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：SGLang 中节点同时保存 KV 的槽位下标）',
          code: py`
            import torch

            class TreeNode:
                def __init__(self, key: tuple, value: torch.Tensor):
                    self.key = key                  # 边上的 token
                    self.value = value              # 这些 token 的 KV 在显存池中的下标（每页 1 个 token）
                    self.children = {}
                    self.lock_ref = 0               # 运行中请求的引用计数，> 0 时不可淘汰
                    self.last_access_time = 0.0

            def split(node: TreeNode, k: int) -> TreeNode:
                """在第 k 个 token 处分裂：KV 下标张量随之切成两段，显存中的 KV 不需要移动"""
                mid = TreeNode(node.key[:k], node.value[:k])
                node.key, node.value = node.key[k:], node.value[k:]
                mid.children[node.key[0]] = node
                mid.lock_ref, mid.last_access_time = node.lock_ref, node.last_access_time
                return mid
          `,
        },
      ],
      explanation: md`
        **结构**：每个节点保存「进入它的那条边」上的 token 元组，孩子用首 token 做键（同一节点的孩子首 token 互不相同，这是 radix tree 的不变量）。

        **匹配**：沿首 token 找孩子，逐 token 比较；整条边匹配就下降，中途失配就停（已匹配的部分也算，token 粒度）。

        **插入的两种情况**：① 没有对应首 token 的孩子 → 挂一个新叶子，边为剩余全部 token；② 在某条边的第 $k$ 个 token 处失配或序列结束 → 把该边分裂为 \`key[:k]\`（新的中间节点）与 \`key[k:]\`（原节点），再继续。

        **计数**：分裂只增加节点不增加 token；挂叶子同时增加 1 个节点和若干 token。

        **复杂度**：匹配与插入都是 $O(\text{序列长度})$。真实系统还要在节点上维护 KV 下标、引用计数和最近访问时间以支持 LRU 淘汰叶子（见 PyTorch 版）。
      `,
      complexity: { time: 'O(L) 每次匹配 / 插入', space: 'O(缓存的 token 数)' },
      hints: [
        md`节点类：\`key\`（tuple）+ \`children\`（dict，首 token → 子节点）。根的 key 为空。`,
        md`写一个 \`_common(key, tokens, i)\` 返回从 \`tokens[i]\` 开始与 \`key\` 的公共前缀长度。`,
        md`插入时若 \`k < len(child.key)\`：新建 \`mid = Node(child.key[:k])\`，\`child.key = child.key[k:]\`，\`mid.children[child.key[0]] = child\`，再用 \`mid\` 替换父节点里的 \`child\`。`,
        md`维护两个计数器：分裂时 \`nodes += 1\`；挂叶子时 \`nodes += 1, tokens += 剩余长度\`。`,
      ],
      setup: py`
        import random

        class _Trie:
            """未压缩 Trie 作为参考：radix 节点数 = 「孩子数 ≠ 1 或是某次插入终点」的 Trie 节点数"""
            def __init__(self):
                self.ch = [{}]; self.end = [False]
            def insert(self, t):
                v, matched = 0, 0
                for x in t:
                    if x in self.ch[v]:
                        v = self.ch[v][x]; matched += 1
                    else:
                        self.ch.append({}); self.end.append(False)
                        self.ch[v][x] = len(self.ch) - 1; v = len(self.ch) - 1
                self.end[v] = True
                return matched
            def match(self, t):
                v, m = 0, 0
                for x in t:
                    if x not in self.ch[v]: break
                    v = self.ch[v][x]; m += 1
                return m
            def radix_nodes(self):
                return sum(1 for v in range(1, len(self.ch)) if len(self.ch[v]) != 1 or self.end[v])
            def tokens(self):
                return len(self.ch) - 1
      `,
      tests: [
        {
          name: '示例：分裂与计数',
          code: py`
            R = RadixCache()
            assert R.insert([1, 2, 3]) == 0
            assert R.insert([1, 2, 4]) == 2, "第二次插入前已缓存 [1, 2]"
            assert R.num_nodes() == 3 and R.total_tokens() == 4, f"应有 3 个节点、4 个 token，得到 {R.num_nodes()}、{R.total_tokens()}"
          `,
        },
        {
          name: '匹配：边中间失配也算已匹配部分',
          code: py`
            R = RadixCache(); R.insert([1, 2, 3]); R.insert([1, 2, 4])
            assert R.match_prefix([1, 2, 9]) == 2
            assert R.match_prefix([1, 2, 3, 7]) == 3
            assert R.match_prefix([5]) == 0 and R.match_prefix([]) == 0
            R2 = RadixCache(); R2.insert([7, 8, 9, 10])
            assert R2.match_prefix([7, 8, 0]) == 2, "在边 [7,8,9,10] 中间失配，应返回 2"
          `,
        },
        {
          name: '插入已缓存的更短序列：只分裂不加 token',
          code: py`
            R = RadixCache(); R.insert([1, 2, 3]); R.insert([1, 2, 4])
            assert R.insert([1]) == 1
            assert R.num_nodes() == 4 and R.total_tokens() == 4, f"应有 4 个节点、4 个 token，得到 {R.num_nodes()}、{R.total_tokens()}"
            assert R.insert([1, 2, 3]) == 3 and R.num_nodes() == 4, "重复插入不应改变结构"
          `,
        },
        {
          name: '压缩：一条长序列只有 1 个节点',
          hidden: true,
          code: py`
            R = RadixCache(); R.insert(list(range(1000)))
            assert R.num_nodes() == 1, f"未压缩？1000 个 token 的单链应只有 1 个节点，得到 {R.num_nodes()}"
            assert R.total_tokens() == 1000 and R.match_prefix(list(range(1000)) + [5]) == 1000
          `,
        },
        {
          name: '随机对拍（与未压缩 Trie 比较）',
          hidden: true,
          code: py`
            rnd = random.Random(3)
            for case in range(150):
                R, T = RadixCache(), _Trie()
                pool = [[rnd.randint(0, 3) for _ in range(rnd.randint(0, 8))] for _ in range(4)]
                for step in range(25):
                    t = list(rnd.choice(pool)) + [rnd.randint(0, 3) for _ in range(rnd.randint(0, 5))]
                    t = t[: rnd.randint(0, len(t))]
                    if rnd.random() < 0.5:
                        e, g = T.match(t), R.match_prefix(t)
                        assert g == e, f"case {case}: match_prefix({t}) 期望 {e}，得到 {g}"
                    else:
                        e, g = T.insert(t), R.insert(t)
                        assert g == e, f"case {case}: insert({t}) 期望 {e}，得到 {g}"
                    assert R.num_nodes() == T.radix_nodes(), f"case {case}: 节点数期望 {T.radix_nodes()}，得到 {R.num_nodes()}"
                    assert R.total_tokens() == T.tokens(), f"case {case}: token 数期望 {T.tokens()}，得到 {R.total_tokens()}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-evict',
          q: md`在这个结构上怎么实现「LRU 淘汰叶子」？`,
          a: md`每个节点记录最近访问时间与引用计数；维护一个按访问时间排序的「可淘汰叶子」集合（最小堆，惰性删除过期条目）。需要释放 $x$ 个 token 时，反复弹出时间最早、ref = 0 的叶子，删除它（释放其 KV 槽位），若父节点因此变成叶子且 ref = 0，就把父节点加入堆。插入或命中时沿路径更新访问时间，运行中的请求对其路径加锁（ref + 1），结束时解锁。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-radix-vs-apc',
      freq: 3,
      q: md`SGLang 的 RadixAttention 和 vLLM 的 prefix caching 有什么异同？`,
      a: md`
        **相同**：都是跨请求复用相同前缀的 KV，命中部分跳过 prefill、共享显存；都按 LRU 淘汰，并保证被共享的前缀晚于其后缀被淘汰。

        **不同**：RadixAttention 用 radix tree，KV 每页 1 个 token，可以在任意 token 处分叉复用，淘汰的是叶子；配合 cache-aware 调度（最长共享前缀优先 ≈ DFS 顺序）提高命中率。
        vLLM APC 给每个满块算链式哈希（父哈希 + 块内 token + LoRA / 多模态 / salt 等额外键），用哈希表查找，只缓存满块；ref = 0 的块进空闲队列兼作 LRU，请求释放时逆序入队让后缀先淘汰。

        **取舍**：radix tree 粒度更细、可做树形调度；块哈希与分页分配器天然统一、实现简单、每块 $O(1)$。实际命中率差别主要来自块大小。
      `,
    },
    {
      id: 'qa-chunked',
      freq: 3,
      q: md`chunked prefill 是怎么回事？chunk 大小怎么取？`,
      a: md`
        **结论**：每次迭代设 token 预算，先放全部 decode token，剩余预算装一块 prefill；长 prompt 分多次迭代完成。

        **原因**：不切块时一个长 prefill 让整次迭代变长，同批 decode 出现 TPOT 尖峰；而纯 decode 迭代是访存瓶颈、算力闲置，塞进一块 prefill 几乎免费（piggybacking）。Sarathi-Serve 称之为 stall-free scheduling。

        **取舍**：预算小 → TPOT 稳但 prefill 被切碎，每块都要重读之前块的 KV，TTFT 变差；预算大 → TTFT 好但 TPOT 波动大。vLLM V1 默认开启，文档建议约 2048 偏 ITL、> 8192 偏吞吐。合适的值大致在「迭代刚好从访存瓶颈转为算力瓶颈」附近，再按 SLO 微调。
      `,
    },
    {
      id: 'qa-pd',
      freq: 3,
      q: md`为什么要做 PD 分离？代价和难点是什么？`,
      a: md`
        **结论**：prefill 算力瓶颈、decode 访存瓶颈，混部会相互干扰且无法分别优化；分离后可以按 TTFT 与 TPOT 两个 SLO 分别配资源、选并行策略，提升 goodput。

        **依据**：DistServe 以 goodput 为目标，报告可多服务 7.4× 请求或满足 12.6× 更紧的 SLO；Mooncake 在 Kimi 线上以 KV 为中心做分离与分级缓存。

        **代价**：① KV 跨机传输（∝ prompt 长度 × 每 token KV），需要高速网络并按层流水重叠；② prefill / decode 池的配比要随流量动态调整；③ 系统复杂度、故障处理、调度都更难。短 prompt、小规模场景用 chunked prefill 往往更划算。
      `,
    },
    {
      id: 'qa-hitrate',
      freq: 2,
      q: md`线上前缀缓存命中率低，可以从哪些方向优化？`,
      a: md`
        **应用侧**：把稳定内容（system prompt、工具定义、few-shot）放最前面；时间戳、随机 ID 等易变内容后移；多轮对话保持历史原样拼接。

        **调度侧**：单实例内按共享前缀长度排序（cache-aware scheduling）；多实例时做 KV 感知的路由，把同前缀请求打到同一实例（如 SGLang Router 的 cache-aware 负载均衡），兼顾负载均衡。

        **容量侧**：增大可用于缓存的显存（KV 量化、更小的块）、把冷 KV 下沉到 CPU 内存 / SSD 的分级缓存（Mooncake 的思路），以及选择更合适的块大小（块越大，短前缀越难命中）。
      `,
    },
  ],

  cheatsheet: md`
    **RadixAttention（SGLang）**：radix tree、每页 1 token；最长前缀匹配 → 分裂插入 → LRU 淘汰叶子（ref = 0）；cache-aware 调度 = 最长共享前缀优先 ≈ DFS
    **APC（vLLM）**：$h_i = \mathrm{hash}(h_{i-1}, \text{块 token}, \text{LoRA / 多模态 / salt})$，只缓存满块；v0.11 起默认 sha256；ref = 0 的块进空闲队列（LRU），释放时逆序入队

    \`\`\`python
    parent = None
    for b in range(len(toks) // B):                     # 只算满块
        parent = hash((parent, tuple(toks[b*B:(b+1)*B])))
    命中 = 从第 0 块起连续在缓存中的块数 × B
    \`\`\`

    **chunked prefill**：预算 = decode token + 一块 prefill；decode 优先；小预算稳 TPOT、大预算好 TTFT；vLLM V1 默认开
    **PD 分离**：prefill（算力）/ decode（带宽）分池；goodput；KV 传输 = P × 每 token KV（70B ≈ 320 KiB/token），按层流水重叠
    - DistServe：7.4× 请求 / 12.6× 更紧 SLO；Sarathi-Serve：2.6–5.6× 服务容量；Mooncake：FAST'25 最佳论文
    - SGLang 命中率 50%–99%（基准），线上 52.4% / 74.1%
  `,
})
