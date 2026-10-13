/**
 * L10-2 PagedAttention 与连续批处理：KV 显存碎片、block table、写时复制、迭代级调度、抢占。
 * 一手资料：Kwon et al. Efficient Memory Management for LLM Serving with PagedAttention. SOSP 2023（arXiv 2309.06180）
 *          vLLM 博客（2023-06-20）；Yu et al. Orca. OSDI 2022；vLLM 文档 Optimization and Tuning（V1 抢占默认 RECOMPUTE）
 */
import { defineLevel, md, py } from '@/content/helpers'
import { FragViz } from '../viz/FragViz'
import { PagedMappingViz } from '../viz/PagedMappingViz'
import { CowViz } from '../viz/CowViz'
import { BatchingViz } from '../viz/BatchingViz'

export default defineLevel({
  learn: [
    {
      id: 'frag',
      title: 'KV cache 为什么难管：动态增长 + 长度未知',
      body: md`
        推理服务的显存里有两大块：**权重**（静态）和 **KV cache**（随请求动态增长）。vLLM 论文给的例子：13B 模型跑在 A100-40GB 上，权重约占 65%，KV cache 约占 30%；
        OPT-13B 每个 token 的 KV 约为 $2 \times 5120 \times 40 \times 2\ \text{B} \approx 800\ \text{KB}$，一个 2048 token 的请求就要约 1.6 GB。
        能同时服务多少请求（batch 多大），几乎完全取决于 KV cache 管得好不好。

        难点在于：一个请求最终会生成多少 token，开始时**不知道**。早期系统（FasterTransformer、Orca）为每个请求按最大长度预留一段**连续**显存，于是产生三种浪费：

        | 浪费 | 含义 |
        |---|---|
        | 预留（reserved） | 为将来的 token 提前占着，当前没用 |
        | 内部碎片（internal） | 按 max_len 预留，但请求提前结束，这部分**永远**用不到 |
        | 外部碎片（external） | 不同大小的连续区间分配 / 释放后留下的「洞」，总量够但不连续，放不下新请求 |

        论文测得：现有系统的 KV cache 显存中只有 **20.4%–38.2%** 真正存了 token 状态（vLLM 博客的说法是浪费 60%–80%）。
      `,
      viz: FragViz,
      after: md`
        > [!TIP] 读图
        > 左边每个请求进来就占一整段 prompt + max_new_tokens；R1、R3 提前结束，斜线部分从未被使用；第 2 步起 R4 在排队，而空着的 8 个槽位因为不够连续，只能干看着（红色斜线）。
        > 右边分页后同一时刻能跑 5 个请求，全部请求在约 2/3 的迭代数内完成。切换 block_size 看看块内空位怎么变化。

        这和操作系统当年遇到的问题一模一样：进程需要的内存大小会变、会提前退出。解决办法也一样——**分页**。
      `,
      checkpoint: {
        id: 'cp-frag',
        type: 'single',
        prompt: md`按 max_len 为每个请求预留一段连续 KV 空间，请求在远未达到 max_len 时就输出了 EOS，被浪费掉的那部分属于：`,
        options: ['外部碎片', '内部碎片', '预留（将来会用到）', '不算浪费，释放后可以复用'],
        answer: 1,
        explanation: md`分配给这个请求、却直到它结束都没用上的空间是**内部碎片**。外部碎片是分配单元**之间**的空洞；「预留」指将来确实会用到的部分。请求结束后空间确实能回收，但在它运行期间这块显存不能给别的请求用，直接压低了并发。`,
      },
    },
    {
      id: 'blocktable',
      title: 'PagedAttention：块表把逻辑连续映射到物理分散',
      body: md`
        vLLM 把 KV cache 切成固定大小的**块**（block，每块存 $B$ 个 token 的 K/V，论文默认 $B = 16$），并为每个请求维护一张**块表**（block table）：

        $$
        \text{逻辑位置 } i \;\Rightarrow\; \text{物理块 } \texttt{table}[\lfloor i / B \rfloor],\ \text{块内偏移 } i \bmod B
        $$

        - 只有最后一块写满时才申请新块，**按需分配**，没有预留；
        - 所有块大小相同，任何空闲块都能给任何请求用，**没有外部碎片**；
        - 每个请求最多浪费最后一块里的 $B - 1$ 个槽位，内部碎片有界；
        - 注意力 kernel 不再假设 K/V 连续，而是按块表逐块读取（gather），这就是 **PagedAttention**。

        写入 KV 时，框架会先把每个新 token 的位置换算成一维的 **slot**：$\text{slot} = \texttt{table}[\lfloor i/B \rfloor] \cdot B + (i \bmod B)$（vLLM 里叫 \`slot_mapping\`），再一次性把 K/V 散射写进物理池。
      `,
      viz: PagedMappingViz,
      after: md`
        > [!WARN] 代价：间接寻址
        > 论文的 kernel 微基准里，PagedAttention 的注意力 kernel 延迟比高度优化的 FasterTransformer 高 **20–26%**（多了读块表、分支、按块访问）。
        > 但它换来了更大的 batch，端到端吞吐反而提升 **2–4×**（相同延迟下对比 FasterTransformer 与 Orca）。今天 FlashAttention、FlashInfer 等主流 kernel 都直接支持分页 KV。

        **block size 怎么选**：太小 → 每次只读很少的数据、GPU 并行度差，块表变长；太大 → 最后一块浪费多、前缀共享的粒度变粗。
        论文在 ShareGPT 上 16–128 效果最好，在输入输出都较短的 Alpaca 上 16、32 较好，因此默认 16。
      `,
      checkpoint: {
        id: 'cp-bt',
        type: 'fill',
        prompt: md`$B = 16$，某请求的块表为 \`[7, 2, 9]\`。位置 37 的 token：逻辑块号 = ____(1)____，块内偏移 = ____(2)____，slot = ____(3)____。`,
        blanks: [
          { label: '(1)', answers: ['2'], numeric: true, tolerance: 0 },
          { label: '(2)', answers: ['5'], numeric: true, tolerance: 0 },
          { label: '(3)', answers: ['149'], numeric: true, tolerance: 0 },
        ],
        explanation: md`$\lfloor 37/16 \rfloor = 2$，$37 \bmod 16 = 5$；物理块 \`table[2] = 9\`，slot $= 9 \times 16 + 5 = 149$。`,
      },
    },
    {
      id: 'cow',
      title: '写时复制：并行采样与 beam search 共享 KV',
      body: md`
        分页带来的第二个好处是**共享**。parallel sampling（同一个 prompt 采 $n$ 个回答）、beam search 中，多个序列有大段相同的前缀。
        有了块表，共享只需让多张块表指向同一批物理块，并给每个物理块维护**引用计数**：

        - **fork**：复制块表（几个整数），每个块 ref + 1，KV 本身不复制；
        - **写入**：要写的最后一块 ref > 1（被共享且未满）时，先**复制**到新块、旧块 ref − 1，再写——写时复制（copy-on-write）；
          最后一块已满时直接分配新块，共享的满块永远只读、不必复制；
        - **释放**：每个块 ref − 1，归零才真正回收。

        论文报告的显存节省：parallel sampling 在 Alpaca / ShareGPT 上分别为 6.1%–9.8% / 16.2%–30.5%；beam search 更多，分别为 37.6%–55.2% / 44.3%–66.3%（beam 之间共享的部分更长、更动态）。
      `,
      viz: CowViz,
      after: md`
        > [!NOTE] 同一请求内 vs 跨请求
        > COW 解决的是**同一请求的多个序列**共享前缀。**不同请求**之间共享 system prompt、多轮对话历史，要靠「前缀缓存」（vLLM 的 automatic prefix caching、SGLang 的 RadixAttention），见下一关。
      `,
      checkpoint: {
        id: 'cp-cow',
        type: 'judge',
        prompt: md`fork 之后两个序列共享一个**已写满**的物理块，此后任何一个序列追加 token，都会触发对这个满块的写时复制。`,
        answer: false,
        explanation: md`追加的 token 落在**新**的逻辑块里（位置是 $B$ 的整数倍），直接分配新块即可；满块里的 KV 不会再被修改，可以一直共享。只有共享的**未满**最后一块被写入时才需要复制。`,
      },
    },
    {
      id: 'batching',
      title: '迭代级调度：连续批处理（continuous batching）',
      body: md`
        显存省出来了，还得调度得好才能变成吞吐。传统的**请求级**（静态）批处理：凑够一个 batch 一起跑，直到 batch 里**最长**的请求生成完才整体返回、换下一批。
        早结束的请求占着槽位空转，新请求要等整批结束才能进来。

        Orca（Yu 等，OSDI 2022）提出**迭代级调度**（iteration-level scheduling）：调度粒度从「一个请求」变成「一次前向迭代」。每次迭代结束都检查：
        有请求生成完就立即返回、释放资源，排队的请求在下一次迭代立刻补进来。这就是今天说的 **continuous batching**。

        难点是 batch 里各请求的长度、阶段（prefill / decode）都不同，没法拼成规整的张量。Orca 的 **selective batching**：
        QKV 投影、FFN 这些逐 token 的算子把所有请求的 token 拼成一个 $[\sum_i n_i,\ d]$ 的矩阵一起算；只有注意力按请求分别计算。
        今天的推理引擎都是这个思路：不 padding、把 token 展平成一维，注意力 kernel 按 \`cu_seqlens\` / 块表区分请求。

        Orca 论文在 GPT-3 175B 上报告：相同延迟下吞吐是 FasterTransformer 的 **36.9×**。vLLM 就是「PagedAttention + 迭代级调度」。
      `,
      viz: BatchingViz,
      after: md`
        > [!INTERVIEW] 连续批处理带来的新问题
        > prefill 和 decode 混在同一次迭代里：一个很长的 prompt 进来，这次迭代就会被 prefill 拖慢，所有正在 decode 的请求都会感到一次「卡顿」（TPOT 尖峰）。
        > 这正是下一关 **chunked prefill** 要解决的问题。
      `,
      checkpoint: {
        id: 'cp-batch',
        type: 'numeric',
        prompt: md`4 个请求同时到达，输出长度依次为 $[7, 2, 3, 4]$，batch 最多 2 个槽位，每次迭代每个运行中的请求生成 1 个 token，FIFO。用迭代级调度时，总共需要多少次迭代？`,
        answer: 9,
        tolerance: 0,
        unit: '次',
        explanation: md`R1 占槽 0 的第 0–6 次迭代；R2 占槽 1 的第 0–1 次；R3 在第 2 次补进槽 1（2–4）；R4 在第 5 次补进（5–8），共 9 次。静态批处理是 $\max(7,2) + \max(3,4) = 11$ 次。`,
      },
    },
    {
      id: 'preempt',
      title: '显存不够时：抢占、换出与重算',
      body: md`
        按需分配意味着可能「超卖」：所有请求都在增长，空闲块用完了，必须**抢占**（preempt）一部分请求。vLLM 论文的策略：

        - **先来先服务（FCFS）**：被抢占的是最晚到达的请求；
        - **全有或全无**：一个序列的块要么全部驱逐、要么全部保留（注意力需要全部历史 KV）；同一请求的多个序列（beam / 并行采样）作为一组同时调度；
        - 被驱逐的 KV 有两种处理方式：

        | | Swap（换出） | Recompute（重算） |
        |---|---|---|
        | 做法 | 把块拷到 CPU 内存，恢复时拷回 | 直接丢弃，恢复时把 prompt + 已生成 token 当作新 prompt 重新 prefill |
        | 代价 | PCIe 带宽；小块时大量小拷贝很慢 | GPU 计算；prefill 是计算密集型，比逐 token 生成快得多 |
        | 显存外开销 | CPU swap 空间（上界为 GPU 上 KV 的大小） | 无 |

        论文的结论：小块时重算更快，大块时换出更快，block size 在 16–64 时两者相当。vLLM V1 的文档写明默认抢占模式是 **RECOMPUTE**，理由是在 V1 架构下重算开销更低；
        配合前缀缓存，被抢占请求的前缀块若还在缓存里，重算时还能直接命中。
      `,
      after: md`
        > [!WARN] 频繁抢占是配置问题
        > 抢占会让被抢占请求的延迟显著变差，还浪费算力。日志里频繁出现抢占时，应该增大 KV 显存（提高 \`gpu_memory_utilization\`、加卡做 TP / PP、KV 量化）或减小并发上限（\`max_num_seqs\`、\`max_num_batched_tokens\`）。
      `,
      checkpoint: {
        id: 'cp-preempt',
        type: 'single',
        prompt: md`为什么「重算」恢复一个被抢占的请求，往往比它当初逐 token 生成那部分要快得多？`,
        options: [
          '重算时跳过了注意力层',
          '已生成的 token 和 prompt 拼在一起做一次 prefill，并行处理，是计算密集型而非访存密集型',
          '重算只需要恢复最后一个 token 的 KV',
          '重算使用的是 CPU 上缓存的激活',
        ],
        answer: 1,
        explanation: md`decode 时这些 token 是一个一个生成的（每步受访存限制）；恢复时它们已经确定，可以和 prompt 一起作为一个长 prompt 并行 prefill，一次前向就能算完全部 KV。A、C 都不对：后续解码需要全部历史 KV；D 是 swap 的思路，与重算无关。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-goal',
      type: 'single',
      difficulty: 1,
      prompt: md`PagedAttention 最主要解决的问题是：`,
      options: ['注意力计算的 FLOPs 太高', 'KV cache 显存的碎片与无法共享，限制了 batch 大小', '长序列下 softmax 数值不稳定', '多卡之间 KV cache 的通信开销'],
      answer: 1,
      explanation: md`PagedAttention 是**显存管理**技术：消除外部碎片、把内部碎片限制在最后一块、支持块级共享，从而容纳更大的 batch。它不减少 FLOPs（A），甚至让注意力 kernel 本身略慢；C 是 online softmax 的话题；D 与分页无关。`,
    },
    {
      id: 'q-flops',
      type: 'judge',
      difficulty: 1,
      prompt: md`PagedAttention 通过分块减少了注意力的计算量，所以单个注意力 kernel 比连续存储版本更快。`,
      answer: false,
      explanation: md`计算量不变，kernel 还要多读块表、按块间接寻址。论文实测其注意力 kernel 延迟比 FasterTransformer 高 20–26%；收益来自 batch 变大带来的吞吐提升（2–4×）。`,
    },
    {
      id: 'q-blocks',
      type: 'numeric',
      difficulty: 2,
      prompt: md`Llama-3.1-8B（32 层、8 个 KV 头、head_dim 128，BF16）每个 token 的 KV 为 128 KiB。显存中可用于 KV cache 的空间为 40 GiB，block size = 16。最多能分配多少个 KV 块？`,
      answer: 20480,
      tolerance: 0,
      unit: '块',
      explanation: md`每块 $16 \times 128\ \text{KiB} = 2\ \text{MiB}$；$40\ \text{GiB} / 2\ \text{MiB} = 40 \times 1024 / 2 = 20480$ 块，即最多同时缓存 $20480 \times 16 = 327{,}680$ 个 token。（每 token KV：$2 \times 32 \times 8 \times 128 \times 2\ \text{B} = 131072\ \text{B}$。）`,
    },
    {
      id: 'q-waste',
      type: 'numeric',
      difficulty: 2,
      prompt: md`block size = 16。一个请求 prompt 100 个 token、生成了 50 个 token 后结束。它的最后一个块里浪费了多少个槽位？`,
      answer: 10,
      tolerance: 0,
      unit: '个',
      explanation: md`共 150 个 token，需要 $\lceil 150/16 \rceil = 10$ 块 = 160 个槽位，浪费 $160 - 150 = 10$ 个，即 6.25%。若按 max_len = 2048 预留连续空间，则浪费 1898 个。`,
    },
    {
      id: 'q-cow-when',
      type: 'multi',
      difficulty: 2,
      prompt: md`下列哪些操作会触发**块的复制**（copy-on-write）？（多选）`,
      options: [
        'parallel sampling 中，某个样本向与其它样本共享的、未写满的最后一块追加 token',
        '某序列与其它序列共享的最后一块已写满，它继续追加 token',
        'fork 出一个新序列',
        'beam search 中一个 beam 派生出两个子 beam，其中一个子 beam 向共享的未满块写入下一个 token',
        '释放一个与别人共享块的序列',
      ],
      answer: [0, 3],
      explanation: md`只有「向 ref > 1 的**未满**块写入」才需要复制（A、D）。B 会直接分配新块；C 只复制块表并增加引用计数；E 只减少引用计数，归零才回收。`,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把带写时复制的 \`append_token(seq)\` 的步骤排成正确顺序：`,
      items: [
        md`取当前长度 \`pos = len[seq]\``,
        md`若 \`pos % B == 0\`：分配一个新块追加到块表末尾`,
        md`否则若最后一块 \`ref > 1\`：分配新块、复制旧块内容、旧块 \`ref -= 1\`、块表末项指向新块`,
        md`计算 \`slot = table[pos // B] * B + pos % B\` 并写入 K/V`,
        md`\`len[seq] += 1\``,
      ],
      explanation: md`先判断要不要新块或复制，再算 slot 写入，最后更新长度。若先写入再复制，就已经污染了共享块。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 2,
      prompt: md`把概念与描述配对：`,
      pairs: [
        ['内部碎片', '分配给请求、但直到请求结束都没用上的空间'],
        ['外部碎片', '空闲空间总量够，但不连续，放不下新的分配'],
        ['引用计数', '决定一个物理块何时可以真正回收'],
        ['块表（block table）', '把序列的逻辑块号映射到物理块号'],
        ['selective batching', '逐 token 算子拼在一起算，注意力按请求分别算'],
      ],
      explanation: md`分页消除了外部碎片、把内部碎片限制在最后一块；引用计数 + 块表实现共享与写时复制；selective batching 是 Orca 让不同长度请求一起批处理的关键。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 3,
      prompt: md`下面的分配器在 fork + 写时复制场景下有 **2 处** bug，点出有问题的行：`,
      code: py`
        class Allocator:
            def __init__(self, n, B):
                self.B, self.free, self.ref = B, list(range(n)), [0] * n
                self.table, self.len = {}, {}

            def append_token(self, s):
                pos, t = self.len[s], self.table[s]
                if pos % self.B == 0:
                    b = self.free.pop(0); self.ref[b] = 1; t.append(b)
                elif self.ref[t[-1]] > 1:
                    b = self.free.pop(0); self.ref[b] = 1
                    t[-1] = b
                self.len[s] = pos + 1

            def free_seq(self, s):
                for b in self.table.pop(s):
                    self.ref[b] -= 1
                    self.free.append(b)
      `,
      bugLines: [12, 18],
      explanation: md`
        - 第 12 行：写时复制后，旧块少了一个引用者，必须先 \`self.ref[t[-1]] -= 1\` 再让 \`t[-1] = b\`（同时要复制 KV 内容）。漏掉会导致旧块永远无法回收（显存泄漏）。
        - 第 18 行：只有 \`ref[b] == 0\` 时才能放回空闲池；现在共享块被一个序列释放后就进了空闲池，可能被别的请求覆盖，而另一个序列还在读它。
      `,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`
        当前状态：序列 1 的块表 \`[0, 3, 5]\`，序列 2 的块表 \`[0, 3, 6]\`，序列 3 的块表 \`[1, 2]\`；引用计数 \`ref[0]=2, ref[3]=2\`，其余已用块 ref = 1。
        执行 \`free(序列 2)\` 后，哪些物理块回到空闲池？
      `,
      items: ['#0', '#1', '#2', '#3', '#4', '#5', '#6', '#7'],
      answer: [6],
      explanation: md`序列 2 的块 #0、#3 的 ref 从 2 降到 1（序列 1 仍在用），不回收；只有 #6 的 ref 归零被回收。#4、#7 原本就是空闲的，不是这次操作回收的。`,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出 \`fork\` 与 \`free\`（\`heapq\` 维护空闲块最小堆）。有 2 行干扰项：`,
      lines: [
        'def fork(self, parent, child):',
        '    self.table[child] = list(self.table[parent])',
        '    self.len[child] = self.len[parent]',
        '    for b in self.table[child]:',
        '        self.ref[b] += 1',
        'def free(self, seq):',
        '    for b in self.table.pop(seq):',
        '        self.ref[b] -= 1',
        '        if self.ref[b] == 0:',
        '            heapq.heappush(self.free_blocks, b)',
      ],
      distractors: ['    self.table[child] = self.table[parent]', '            self.ref[b] = 0'],
      explanation: md`fork 必须**拷贝**块表（\`list(...)\`）：直接赋值会让两个序列共用同一个 list 对象，子序列 COW 改 \`t[-1]\` 时会把父序列的块表也改掉。释放时只在引用计数归零时回收。`,
    },
    {
      id: 'q-cont-iter',
      type: 'numeric',
      difficulty: 2,
      prompt: md`8 个请求同时到达，输出长度 $[5, 12, 3, 8, 10, 4, 7, 6]$，batch 最多 4 个槽位，FIFO。**静态批处理**（前 4 个一批，跑完再下一批）总共需要多少次迭代？`,
      answer: 22,
      tolerance: 0,
      unit: '次',
      explanation: md`第一批 $\max(5,12,3,8) = 12$，第二批 $\max(10,4,7,6) = 10$，共 22 次；槽位利用率 $55 / (4 \times 22) = 62.5\%$。迭代级调度只需 15 次（利用率约 92%），见图解中的甘特图。`,
    },
    {
      id: 'q-preempt',
      type: 'single',
      difficulty: 3,
      prompt: md`关于 vLLM 的抢占，下列说法**正确**的是：`,
      options: [
        '被抢占的是最早到达的请求，以便尽快腾出最多的显存',
        '被抢占序列的 KV 块可以只驱逐一部分，保留最近的若干块',
        'block size 很小时，swap 需要大量小粒度 PCIe 拷贝，重算往往更划算',
        'swap 不需要额外资源，所以总是优于重算',
      ],
      answer: 2,
      explanation: md`论文：小块时 swap 的小拷贝效率低，重算更快；大块时 swap 更快。A 反了：FCFS 下抢占**最晚**到达的请求；B 错：采用 all-or-nothing，注意力需要完整的历史 KV；D 错：swap 需要 CPU 内存和 PCIe 带宽。`,
    },
  ],

  guided: [
    {
      id: 'g-paged-attn',
      title: '按块表读取 KV 的解码注意力',
      intro: md`
        实现单个请求、单个 query 头的分页解码注意力：K/V 存在物理池 \`k_cache / v_cache\`（形状 \`(num_blocks, block_size, d)\`）里，
        序列的第 $i$ 个 token 位于物理块 \`block_table[i // block_size]\` 的第 \`i % block_size\` 个槽位。返回 $\mathrm{softmax}(Kq/\sqrt{d})\,V$，形状 \`(d,)\`。
      `,
      template: py`
        import numpy as np

        def paged_decode_attention(q, k_cache, v_cache, block_table, seq_len, block_size):
            pos = np.arange(seq_len)
            phys = {{1}}
            off = {{2}}
            K = {{3}}
            V = v_cache[phys, off]
            scores = {{4}}
            w = np.exp(scores - scores.max())
            w = w / w.sum()
            return w @ V
      `,
      steps: [
        {
          blank: 1,
          instruction: md`每个位置所在的**物理块号**（向量化：\`block_table\` 可能是 Python list，先转成数组）。`,
          answer: 'np.asarray(block_table)[pos // block_size]',
          accept: ['np.array(block_table)[pos // block_size]'],
          hint: '逻辑块号 = pos // block_size，再查块表。',
          explain: md`这就是块表查找。对 Python list 不能直接用数组下标，所以要 \`np.asarray\`。`,
        },
        {
          blank: 2,
          instruction: md`块内偏移。`,
          answer: 'pos % block_size',
          explain: md`和 \`phys\` 一起构成二维下标；展平后就是 \`slot = phys * block_size + off\`。`,
        },
        {
          blank: 3,
          instruction: md`用高级索引一次 gather 出 \`(seq_len, d)\` 的 K。`,
          answer: 'k_cache[phys, off]',
          accept: ['k_cache[phys, off, :]'],
          explain: md`两个长度为 \`seq_len\` 的整数数组做下标，逐元素配对取出 \`k_cache[phys[i], off[i]]\`。真实 kernel 不会先 gather 成连续矩阵，而是在 SRAM 里按块边读边算，但语义相同。`,
        },
        {
          blank: 4,
          instruction: md`缩放点积分数，形状 \`(seq_len,)\`。`,
          answer: 'K @ q / np.sqrt(q.shape[-1])',
          accept: ['(K @ q) / np.sqrt(q.shape[-1])', 'K @ q / np.sqrt(q.shape[0])', 'K @ q / np.sqrt(len(q))', 'K.dot(q) / np.sqrt(q.size)'],
          explain: md`之后就是普通的稳定 softmax 与加权求和。块之外（以及最后一块里 \`seq_len\` 之后）的槽位里是垃圾数据，我们只按 \`pos < seq_len\` 读取，所以不受影响。`,
        },
      ],
      setup: py`
        import numpy as np

        def _ref_attn(q, K, V):
            s = K @ q / np.sqrt(q.shape[-1])
            w = np.exp(s - s.max()); w /= w.sum()
            return w @ V

        def _make(seed, seq_len, B, num_blocks=12, d=8):
            rng = np.random.default_rng(seed)
            K = rng.standard_normal((seq_len, d)); V = rng.standard_normal((seq_len, d)); q = rng.standard_normal(d)
            nb = -(-seq_len // B)
            table = [int(x) for x in rng.permutation(num_blocks)[:nb]]
            kc = rng.standard_normal((num_blocks, B, d)) * 100   # 垃圾数据
            vc = rng.standard_normal((num_blocks, B, d)) * 100
            for i in range(seq_len):
                kc[table[i // B], i % B] = K[i]; vc[table[i // B], i % B] = V[i]
            return q, kc, vc, table, K, V
      `,
      tests: [
        {
          name: 'seq_len 是 block_size 的整数倍',
          code: py`
            q, kc, vc, t, K, V = _make(0, 8, 4)
            out = paged_decode_attention(q, kc, vc, t, 8, 4)
            assert out.shape == (8,), f"形状应为 (8,)，得到 {out.shape}"
            assert np.allclose(out, _ref_attn(q, K, V), atol=1e-8), f"最大误差 {np.abs(out - _ref_attn(q, K, V)).max():.2e}"
          `,
        },
        {
          name: '最后一块未写满（不能读到垃圾槽位）',
          code: py`
            q, kc, vc, t, K, V = _make(1, 11, 4)
            out = paged_decode_attention(q, kc, vc, t, 11, 4)
            assert np.allclose(out, _ref_attn(q, K, V), atol=1e-8), f"最大误差 {np.abs(out - _ref_attn(q, K, V)).max():.2e}"
          `,
        },
        {
          name: '多种块大小与长度',
          code: py`
            for seed, L, B in [(2, 1, 4), (3, 17, 16), (4, 30, 3), (5, 5, 8)]:
                q, kc, vc, t, K, V = _make(seed, L, B)
                out = paged_decode_attention(q, kc, vc, t, L, B)
                assert np.allclose(out, _ref_attn(q, K, V), atol=1e-8), f"seq_len={L}, B={B}: 最大误差 {np.abs(out - _ref_attn(q, K, V)).max():.2e}"
          `,
        },
      ],
      outro: md`
        真实的 PagedAttention kernel 中每个线程块负责一个（请求，头），按块表逐块把 K/V 从 HBM 读进共享内存，配合 online softmax 边读边算，不会物化 \`K\`。
        GQA 下同一组的多个 query 头共享读进来的 K/V 块，这也是 decode kernel 优化的重点（见「注意力机制」章 Flash-Decoding）。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-allocator',
      title: '带引用计数与写时复制的 KV 块分配器',
      difficulty: 'hard',
      source: '大模型岗手撕 · vLLM BlockManager 简化版',
      timeLimitMin: 30,
      description: md`
        实现 vLLM 风格的 KV 块分配器（只管块号与引用计数，不存真实 KV）：

        \`\`\`python
        class BlockAllocator:
            def __init__(self, num_blocks: int, block_size: int): ...
            def allocate(self, seq_id: int, num_tokens: int) -> None: ...
            def append_token(self, seq_id: int) -> tuple[int, int] | None: ...
            def fork(self, parent_id: int, child_id: int) -> None: ...
            def free(self, seq_id: int) -> None: ...
            def block_table(self, seq_id: int) -> list[int]: ...
            def num_free_blocks(self) -> int: ...
            def ref_count(self, block: int) -> int: ...
        \`\`\`

        **语义**
        - 物理块编号 \`0 .. num_blocks-1\`。**每次分配都取编号最小的空闲块**（简化约定，便于测试；vLLM 实际使用空闲队列）。
        - \`allocate(seq_id, n)\`：为新序列的 prompt 分配 $\lceil n / B \rceil$ 个块（$n$ 可以为 0），长度记为 $n$。
        - \`append_token(seq_id)\`：追加一个 token，设当前长度为 \`pos\`：
          - \`pos % B == 0\`：分配一个新块追加到块表末尾，返回 \`None\`；
          - 否则若最后一块的引用计数 > 1：**写时复制**——分配新块 \`dst\`，原块 \`src\` 引用计数 − 1，块表末项换成 \`dst\`，返回 \`(src, dst)\`（调用方据此拷贝 KV）；
          - 否则原地写，返回 \`None\`。长度 + 1。
        - \`fork(parent, child)\`：child 拷贝 parent 的块表与长度，所有块引用计数 + 1。
        - \`free(seq_id)\`：块表中每个块引用计数 − 1，归零的块回到空闲池；删除该序列。
        - **空闲块不够时抛 \`MemoryError\`，并且状态保持不变**（不能分配一半、不能已经改了引用计数）。

        **示例**（\`B = 4\`）：\`allocate(0, 6)\` → 块表 \`[0, 1]\`；\`fork(0, 1)\`；\`append_token(1)\` → \`(1, 2)\`，序列 1 的块表 \`[0, 2]\`，\`ref_count(1) == 1\`；\`append_token(0)\` → \`None\`（块 1 已独占）。
      `,
      starter: py`
        import heapq

        class BlockAllocator:
            def __init__(self, num_blocks: int, block_size: int):
                self.block_size = block_size
                # TODO: 空闲块（取最小编号）、引用计数、每个序列的块表与长度

            def allocate(self, seq_id, num_tokens):
                raise NotImplementedError

            def append_token(self, seq_id):
                raise NotImplementedError

            def fork(self, parent_id, child_id):
                raise NotImplementedError

            def free(self, seq_id):
                raise NotImplementedError

            def block_table(self, seq_id):
                raise NotImplementedError

            def num_free_blocks(self):
                raise NotImplementedError

            def ref_count(self, block):
                raise NotImplementedError
      `,
      solution: py`
        import heapq

        class BlockAllocator:
            def __init__(self, num_blocks: int, block_size: int):
                self.block_size = block_size
                self.free_blocks = list(range(num_blocks))   # 最小堆：总取编号最小的空闲块
                heapq.heapify(self.free_blocks)
                self.ref = [0] * num_blocks
                self.tables = {}                             # seq_id -> 物理块列表
                self.lens = {}                               # seq_id -> 已写入的 token 数

            def _alloc(self):
                if not self.free_blocks:
                    raise MemoryError("没有空闲的 KV 块")
                b = heapq.heappop(self.free_blocks)
                self.ref[b] = 1
                return b

            def allocate(self, seq_id, num_tokens):
                need = -(-num_tokens // self.block_size)      # ceil
                if need > len(self.free_blocks):              # 先检查，保证失败时状态不变
                    raise MemoryError(f"需要 {need} 块，只剩 {len(self.free_blocks)} 块")
                self.tables[seq_id] = [self._alloc() for _ in range(need)]
                self.lens[seq_id] = num_tokens

            def append_token(self, seq_id):
                table, pos = self.tables[seq_id], self.lens[seq_id]
                copy = None
                if pos % self.block_size == 0:                # 最后一块已满（或还没有块）
                    table.append(self._alloc())
                else:
                    last = table[-1]
                    if self.ref[last] > 1:                    # 共享的未满块被写 → 写时复制
                        new = self._alloc()                   # 可能抛 MemoryError，此时尚未改任何状态
                        self.ref[last] -= 1
                        table[-1] = new
                        copy = (last, new)
                self.lens[seq_id] = pos + 1
                return copy

            def fork(self, parent_id, child_id):
                self.tables[child_id] = list(self.tables[parent_id])   # 拷贝块表，不拷贝 KV
                self.lens[child_id] = self.lens[parent_id]
                for b in self.tables[child_id]:
                    self.ref[b] += 1

            def free(self, seq_id):
                for b in self.tables.pop(seq_id):
                    self.ref[b] -= 1
                    if self.ref[b] == 0:                      # 没人引用了才真正回收
                        heapq.heappush(self.free_blocks, b)
                del self.lens[seq_id]

            def block_table(self, seq_id):
                return list(self.tables[seq_id])

            def num_free_blocks(self):
                return len(self.free_blocks)

            def ref_count(self, block):
                return self.ref[block]
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：COW 时真正拷贝物理 KV）',
          code: py`
            import torch

            # kv_cache: (2, num_blocks, block_size, n_kv_heads, head_dim)，K 与 V 叠在第 0 维
            def apply_copy_on_write(kv_cache: torch.Tensor, copies: list[tuple[int, int]]):
                """调度器收集本轮所有 append_token 返回的 (src, dst)，在 GPU 上一次性完成块拷贝"""
                if not copies:
                    return
                src = torch.tensor([s for s, _ in copies], device=kv_cache.device)
                dst = torch.tensor([d for _, d in copies], device=kv_cache.device)
                kv_cache[:, dst] = kv_cache[:, src]

            def write_kv(kv_cache, k, v, slot_mapping, block_size):
                """k, v: (num_tokens, n_kv_heads, head_dim)；slot = block * block_size + offset"""
                blk, off = slot_mapping // block_size, slot_mapping % block_size
                kv_cache[0, blk, off] = k
                kv_cache[1, blk, off] = v
          `,
        },
      ],
      explanation: md`
        **数据结构**：空闲块用最小堆（\`heapq\`），分配 / 回收 $O(\log N)$；每块一个引用计数；每个序列一张块表和一个长度。

        **append_token 的三种情况**：
        1. \`pos % B == 0\`：新 token 落在新的逻辑块，直接分配；即使之前的满块被共享也不用复制。
        2. 最后一块被共享（ref > 1）且未满：写时复制。先 \`_alloc()\`（可能抛异常，此时还没改状态），再把旧块 ref − 1、块表末项换成新块。
        3. 独占的未满块：原地写。

        **易错点**
        - fork 时写成 \`self.tables[child] = self.tables[parent]\`：两个序列共用同一个 list，COW 改 \`table[-1]\` 会把父序列也改掉；
        - COW 后忘记把旧块 ref − 1：旧块永远回收不了；
        - free 时不看引用计数就回收：共享块被别人覆盖；
        - 空间不够时先分配了一部分再抛异常：分配器状态被破坏（真实系统里这会导致显存泄漏或重复分配）。

        **与 vLLM 的差别**：vLLM 用双向链表实现空闲队列（结合前缀缓存时按 LRU 驱逐），COW 的块拷贝在每轮调度结束后批量下发到 GPU 执行（见 PyTorch 版）。
      `,
      complexity: { time: 'allocate O(k log N)，append/fork/free O(log N) 或 O(len(table))', space: 'O(N + Σ 块表长度)' },
      hints: [
        md`用 \`heapq\` 维护空闲块，\`self.ref = [0] * num_blocks\`，\`self.tables\` / \`self.lens\` 用字典。`,
        md`\`allocate\` 先算 \`need = ceil(n / B)\` 并检查是否够，再逐个分配，保证失败时不改状态。`,
        md`\`append_token\`：\`pos % B == 0\` → 新块；否则看 \`ref[table[-1]] > 1\` → 先分配新块，再 \`ref[old] -= 1\`、\`table[-1] = new\`，返回 \`(old, new)\`。`,
        md`\`fork\` 一定要 \`list(...)\` 拷贝块表；\`free\` 只在 \`ref == 0\` 时 \`heappush\`。`,
      ],
      setup: py`
        import heapq, random

        class _RefAlloc:
            def __init__(self, n, B):
                self.B = B; self.fb = list(range(n)); heapq.heapify(self.fb)
                self.ref = [0] * n; self.t = {}; self.L = {}
            def _a(self):
                if not self.fb: raise MemoryError
                b = heapq.heappop(self.fb); self.ref[b] = 1; return b
            def allocate(self, s, n):
                need = -(-n // self.B)
                if need > len(self.fb): raise MemoryError
                self.t[s] = [self._a() for _ in range(need)]; self.L[s] = n
            def append_token(self, s):
                t, p = self.t[s], self.L[s]; c = None
                if p % self.B == 0: t.append(self._a())
                elif self.ref[t[-1]] > 1:
                    nb = self._a(); self.ref[t[-1]] -= 1; c = (t[-1], nb); t[-1] = nb
                self.L[s] = p + 1; return c
            def fork(self, a, b):
                self.t[b] = list(self.t[a]); self.L[b] = self.L[a]
                for x in self.t[b]: self.ref[x] += 1
            def free(self, s):
                for x in self.t.pop(s):
                    self.ref[x] -= 1
                    if self.ref[x] == 0: heapq.heappush(self.fb, x)
                del self.L[s]

        def _norm(c):
            return None if c is None else tuple(int(x) for x in c)
      `,
      tests: [
        {
          name: '基本分配与追加',
          code: py`
            A = BlockAllocator(8, 4)
            A.allocate(0, 10)
            assert A.block_table(0) == [0, 1, 2], f"10 个 token 需 3 块且取最小编号，得到 {A.block_table(0)}"
            assert A.num_free_blocks() == 5
            assert _norm(A.append_token(0)) is None and _norm(A.append_token(0)) is None
            assert A.block_table(0) == [0, 1, 2], "第 11、12 个 token 仍在第 3 块里"
            A.append_token(0)
            assert A.block_table(0) == [0, 1, 2, 3], f"第 13 个 token 需要新块，得到 {A.block_table(0)}"
            assert A.num_free_blocks() == 4
          `,
        },
        {
          name: '释放后复用最小编号',
          code: py`
            A = BlockAllocator(8, 4)
            A.allocate(0, 5); A.allocate(1, 3)
            assert A.block_table(0) == [0, 1] and A.block_table(1) == [2]
            A.free(0)
            assert A.num_free_blocks() == 7
            A.allocate(2, 9)
            assert A.block_table(2) == [0, 1, 3], f"应复用 0、1 再取 3，得到 {A.block_table(2)}"
          `,
        },
        {
          name: 'fork + 写时复制（示例）',
          code: py`
            A = BlockAllocator(8, 4)
            A.allocate(0, 6); A.fork(0, 1)
            assert A.ref_count(0) == 2 and A.ref_count(1) == 2, "fork 后两个块都应被共享"
            got = _norm(A.append_token(1))
            assert got == (1, 2), f"写入共享的未满块应 COW 并返回 (1, 2)，得到 {got}"
            assert A.block_table(1) == [0, 2] and A.block_table(0) == [0, 1], f"块表：序列0 {A.block_table(0)}，序列1 {A.block_table(1)}"
            assert A.ref_count(1) == 1 and A.ref_count(2) == 1 and A.ref_count(0) == 2
            got = _norm(A.append_token(0))
            assert got is None, f"块 1 已被序列 0 独占，应原地写入，得到 {got}"
          `,
        },
        {
          name: '块边界处不复制',
          code: py`
            A = BlockAllocator(8, 4)
            A.allocate(0, 8); A.fork(0, 1)
            got = _norm(A.append_token(1))
            assert got is None, f"最后一块已满，应直接分配新块而不是复制，得到 {got}"
            assert A.block_table(1) == [0, 1, 2] and A.ref_count(1) == 2
          `,
        },
        {
          name: 'MemoryError 且状态不变',
          hidden: true,
          code: py`
            A = BlockAllocator(3, 4)
            A.allocate(0, 4)
            try:
                A.allocate(1, 12); ok = False
            except MemoryError:
                ok = True
            assert ok, "需要 3 块只剩 2 块，应抛 MemoryError"
            assert A.num_free_blocks() == 2, f"分配失败后不应占用任何块，剩余 {A.num_free_blocks()}"
            A.allocate(2, 5)                     # [1, 2]
            A.fork(2, 3)
            try:
                A.append_token(3); ok = False
            except MemoryError:
                ok = True
            assert ok, "COW 需要新块但没有空闲块，应抛 MemoryError"
            assert A.block_table(3) == [1, 2] and A.ref_count(2) == 2, f"COW 失败后状态应不变：块表 {A.block_table(3)}，ref(2)={A.ref_count(2)}"
          `,
        },
        {
          name: '随机操作对拍',
          hidden: true,
          code: py`
            rnd = random.Random(2026)
            for case in range(60):
                n, B = rnd.randint(4, 24), rnd.choice([1, 2, 4, 8])
                A, R = BlockAllocator(n, B), _RefAlloc(n, B)
                live, nid = [], 0
                for step in range(120):
                    op = rnd.random()
                    if op < 0.2 or not live:
                        k = rnd.randint(0, 3 * B)
                        e1 = e2 = None
                        try: A.allocate(nid, k)
                        except MemoryError: e1 = 1
                        try: R.allocate(nid, k)
                        except MemoryError: e2 = 1
                        assert e1 == e2, f"case {case} step {step}: allocate({nid},{k}) 异常行为不一致"
                        if e2 is None: live.append(nid)
                        nid += 1
                    elif op < 0.65:
                        s = rnd.choice(live)
                        g1 = g2 = "err"
                        try: g1 = _norm(A.append_token(s))
                        except MemoryError: pass
                        try: g2 = _norm(R.append_token(s))
                        except MemoryError: pass
                        assert g1 == g2, f"case {case} step {step}: append_token({s}) 期望 {g2}，得到 {g1}"
                    elif op < 0.8:
                        s = rnd.choice(live); A.fork(s, nid); R.fork(s, nid); live.append(nid); nid += 1
                    else:
                        s = live.pop(rnd.randrange(len(live))); A.free(s); R.free(s)
                    assert A.num_free_blocks() == len(R.fb), f"case {case} step {step}: 空闲块数期望 {len(R.fb)}，得到 {A.num_free_blocks()}"
                for s in live:
                    assert A.block_table(s) == R.t[s], f"case {case}: 序列 {s} 块表期望 {R.t[s]}，得到 {A.block_table(s)}"
                assert [A.ref_count(b) for b in range(n)] == R.ref, f"case {case}: 引用计数不一致"
          `,
        },
        {
          name: 'beam 式反复 fork / free 无泄漏',
          hidden: true,
          code: py`
            A = BlockAllocator(64, 4)
            A.allocate(0, 9)
            beams, nid = [0], 1
            rnd = random.Random(5)
            for step in range(40):
                new = []
                for b in beams:
                    for _ in range(2):
                        A.fork(b, nid); A.append_token(nid); new.append(nid); nid += 1
                    A.free(b)
                rnd.shuffle(new)
                for b in new[3:]:
                    A.free(b)
                beams = new[:3]
            for b in beams:
                A.free(b)
            assert A.num_free_blocks() == 64, f"全部释放后应有 64 个空闲块，实际 {A.num_free_blocks()}（引用计数泄漏？）"
            assert all(A.ref_count(i) == 0 for i in range(64)), "全部释放后引用计数应全为 0"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-prefix',
          q: md`如果还要支持「跨请求的前缀缓存」，这个分配器要怎么改？`,
          a: md`给每个**满块**算一个链式哈希（父块哈希 + 本块 token），维护「哈希 → 物理块」的表。分配新请求时先按哈希逐块查找命中的前缀块，命中就 ref + 1 直接复用；释放时 ref 归零的块**不立即清空**，而是放进按 LRU 排序的可驱逐队列，等真的没有空闲块时再驱逐并删除其哈希。这就是 vLLM 的 automatic prefix caching（下一关的实战题）。`,
        },
        {
          id: 'f-swap',
          q: md`抢占时选 swap，分配器还需要什么？`,
          a: md`一个 CPU 侧的块池和 GPU 块号 ↔ CPU 块号的映射：换出时为序列的每个块在 CPU 侧分配一个块、记录映射并释放 GPU 块；换入时反向操作。被共享的块（ref > 1）只需换出一次，映射也要带引用计数。由于全有或全无，需要先确认 CPU 侧空间足够。`,
        },
      ],
    },
    {
      id: 'c-batching',
      title: '模拟静态批处理与迭代级调度',
      difficulty: 'medium',
      source: 'Orca (OSDI 2022) · 迭代级调度',
      timeLimitMin: 20,
      description: md`
        $n$ 个请求在第 0 次迭代前全部到达，按下标顺序排队（FIFO）；第 $i$ 个请求需要 \`lens[i]\` 次迭代（每次迭代生成 1 个 token）。batch 最多 \`max_batch\` 个请求。实现

        \`\`\`python
        def simulate_batching(lens: list[int], max_batch: int, policy: str) -> tuple[int, list[int]]: ...
        \`\`\`

        返回 \`(总迭代数, finish)\`，\`finish[i]\` 为请求 $i$ 生成最后一个 token 的那次迭代的编号（**从 1 开始计**）。
        - \`policy == "static"\`：每次取队头最多 \`max_batch\` 个请求组成一批，这批里**最长**的请求结束后，下一批才开始；
        - \`policy == "continuous"\`：每次迭代开始前，用排队的请求（FIFO）填满所有空槽位；在第 $t$ 次迭代结束的请求，其槽位在第 $t+1$ 次迭代即可被新请求使用。

        **示例**：\`lens = [5, 12, 3, 8, 10, 4, 7, 6]\`，\`max_batch = 4\`
        - static → \`(22, [5, 12, 3, 8, 22, 16, 19, 18])\`
        - continuous → \`(15, [5, 12, 3, 8, 13, 9, 15, 15])\`

        **约束**：$0 \le n \le 3000$，$1 \le$ \`lens[i]\` $\le 100$。空输入返回 \`(0, [])\`。
      `,
      starter: py`
        import heapq

        def simulate_batching(lens, max_batch, policy):
            # static：按批，批耗时 = 批内最长
            # continuous：最早空出的槽位接纳下一个请求（最小堆）
            raise NotImplementedError
      `,
      solution: py`
        import heapq

        def simulate_batching(lens, max_batch, policy):
            n = len(lens)
            finish = [0] * n
            if n == 0:
                return 0, []
            if policy == "static":
                t = 0
                for s in range(0, n, max_batch):
                    group = range(s, min(n, s + max_batch))
                    for i in group:
                        finish[i] = t + lens[i]
                    t += max(lens[i] for i in group)     # 整批等最长的那个
                return t, finish
            # continuous：堆里放每个槽位「下一次空闲的迭代起点」
            slots = [0] * min(max_batch, n)
            heapq.heapify(slots)
            for i, L in enumerate(lens):                  # FIFO：第 i 个请求拿最早空出的槽位
                start = heapq.heappop(slots)
                finish[i] = start + L
                heapq.heappush(slots, start + L)
            return max(finish), finish
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示：迭代级调度器主循环的骨架）',
          code: py`
            import torch

            def serve_loop(model, waiting, max_batch):
                """waiting: deque[Request]；每次迭代：补满 batch → 一次前向 → 采样 → 移除已完成请求"""
                running = []
                while waiting or running:
                    while waiting and len(running) < max_batch:      # 迭代级：随时补位
                        running.append(waiting.popleft())
                    input_ids = torch.cat([r.next_input() for r in running])        # 展平，不 padding
                    logits = model(input_ids, seq_lens=[r.num_new_tokens() for r in running])
                    for r, tok in zip(running, logits.argmax(-1).tolist()):
                        r.append(tok)
                    running = [r for r in running if not r.finished()]          # 完成的立刻离开
          `,
        },
      ],
      explanation: md`
        **static**：直接按批累加，每批耗时是批内最大长度，批内请求的完成时刻 = 批开始时刻 + 自身长度。

        **continuous**：所有请求同时到达且 FIFO，所以第 $i$ 个请求一定被分到「最早空出来的槽位」。用最小堆维护各槽位的空闲时刻，每个请求 $O(\log B)$，总 $O(n \log B)$，不需要逐迭代模拟。
        若改成逐迭代模拟也正确，但复杂度为 $O(\text{总迭代数} \times B)$。

        **结论**：同样的工作量（$\sum$ lens），迭代级调度把总迭代数从 22 降到 15、平均完成时刻也明显提前——短请求不再被长请求拖住。
      `,
      complexity: { time: 'O(n log B)', space: 'O(n)' },
      hints: [
        md`static：\`for s in range(0, n, max_batch)\`，批内每个请求 \`finish = t + lens[i]\`，然后 \`t += max(批内长度)\`。`,
        md`continuous：最小堆 \`slots\` 初始为 \`[0] * min(max_batch, n)\`，每个请求 \`start = heappop(slots)\`，\`finish = start + L\`，再把 \`finish\` 推回堆。`,
        md`总迭代数 = \`max(finish)\`；空输入直接返回 \`(0, [])\`。`,
      ],
      setup: py`
        import random

        def _ref_sim(lens, B, policy):
            n = len(lens)
            if n == 0:
                return 0, []
            fin = [0] * n
            if policy == "static":
                t = 0
                for s in range(0, n, B):
                    g = list(range(s, min(n, s + B)))
                    for i in g: fin[i] = t + lens[i]
                    t += max(lens[i] for i in g)
                return t, fin
            # 逐迭代模拟（与堆解法独立实现，用于对拍）
            queue = list(range(n)); running = {}; t = 0; qi = 0
            while qi < n or running:
                while qi < n and len(running) < B:
                    running[queue[qi]] = lens[queue[qi]]; qi += 1
                t += 1
                for i in list(running):
                    running[i] -= 1
                    if running[i] == 0:
                        fin[i] = t; del running[i]
            return t, fin

        def _norm(r):
            return int(r[0]), [int(x) for x in r[1]]
      `,
      tests: [
        { name: '示例：static', code: py`assert _norm(simulate_batching([5, 12, 3, 8, 10, 4, 7, 6], 4, "static")) == (22, [5, 12, 3, 8, 22, 16, 19, 18])` },
        { name: '示例：continuous', code: py`assert _norm(simulate_batching([5, 12, 3, 8, 10, 4, 7, 6], 4, "continuous")) == (15, [5, 12, 3, 8, 13, 9, 15, 15])` },
        {
          name: '边界：空输入、batch 足够大、batch = 1',
          code: py`
            assert _norm(simulate_batching([], 4, "continuous")) == (0, [])
            assert _norm(simulate_batching([3, 1, 2], 8, "static")) == (3, [3, 1, 2])
            assert _norm(simulate_batching([3, 1, 2], 8, "continuous")) == (3, [3, 1, 2])
            assert _norm(simulate_batching([3, 1, 2], 1, "continuous")) == (6, [3, 4, 6])
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            rnd = random.Random(10)
            for _ in range(200):
                lens = [rnd.randint(1, 20) for _ in range(rnd.randint(0, 25))]
                B = rnd.randint(1, 6)
                for pol in ("static", "continuous"):
                    exp = _ref_sim(lens, B, pol)
                    got = _norm(simulate_batching(list(lens), B, pol))
                    assert got == exp, f"lens={lens}, B={B}, {pol}: 期望 {exp}，得到 {got}"
          `,
        },
        {
          name: '规模：3000 个请求',
          hidden: true,
          code: py`
            rnd = random.Random(3)
            lens = [rnd.randint(1, 100) for _ in range(3000)]
            got = _norm(simulate_batching(lens, 16, "continuous"))
            exp = _ref_sim(lens, 16, "continuous")
            assert got == exp, f"总迭代数期望 {exp[0]}，得到 {got[0]}"
            s_tot = _norm(simulate_batching(lens, 16, "static"))[0]
            assert got[0] <= s_tot, "迭代级调度的总迭代数不应多于静态批处理"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-sjf',
          q: md`如果请求的输出长度可以预测，按「最短作业优先」排队会怎样？`,
          a: md`平均完成时间（平均延迟）会下降——这是经典的 SJF 结论；但长请求可能饿死，且真实输出长度很难准确预测。工程上常见折中：按预测长度分桶、加老化（aging）防饿死，或者按 SLO 优先级调度。`,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-paged',
      freq: 3,
      q: md`PagedAttention 解决了什么问题？代价是什么？`,
      a: md`
        **结论**：它把 KV cache 按固定大小的块管理，用块表做逻辑 → 物理映射，消除了外部碎片、把内部碎片限制在最后一块，并支持块级共享（COW），从而显著增大可服务的 batch。

        **原因**：KV cache 随生成动态增长、最终长度未知；按最大长度预留连续空间会产生预留、内部碎片、外部碎片三类浪费，论文测得现有系统只有 20.4%–38.2% 的 KV 显存真正存了 token。分页后浪费不到 4%（vLLM 博客），相同延迟下吞吐比 FasterTransformer / Orca 高 2–4×。

        **代价**：注意力 kernel 需要按块表间接寻址，论文测得 kernel 延迟高 20–26%；块表维护、COW 决策都在 CPU 侧的调度器里完成，带来一定的调度开销。block size 也要权衡（默认 16）。
      `,
    },
    {
      id: 'qa-cb',
      freq: 3,
      q: md`什么是 continuous batching？和静态批处理有什么区别？`,
      a: md`
        **结论**：continuous batching 即 Orca 提出的**迭代级调度**——每一次前向迭代都重新决定 batch 成员，完成的请求立即离开，新请求立即加入。

        **原因**：LLM 生成长度差异很大，静态批处理要等批内最长的请求结束，短请求的槽位空转，新请求排队等待，吞吐和延迟都差。迭代级调度让槽位一直有活干。为了把不同长度、不同阶段（prefill / decode）的请求放进一次前向，Orca 用 selective batching：线性层拼接所有 token 一起算，注意力按请求分开算。

        **延伸**：混合 prefill 与 decode 会让长 prompt 拖慢整个迭代，引出 chunked prefill；配合 PagedAttention 才能在显存上真正支撑大 batch。Orca 报告 GPT-3 175B 上相同延迟下吞吐为 FasterTransformer 的 36.9×。
      `,
    },
    {
      id: 'qa-blocksize',
      freq: 2,
      q: md`block size 设大或设小分别有什么影响？`,
      a: md`
        **结论**：小块 → 碎片少、共享粒度细，但 kernel 访存效率低、块表长、swap 时小拷贝多；大块 → kernel 效率高，但最后一块浪费多、前缀缓存只能以更粗的粒度命中。

        **依据**：vLLM 论文在 ShareGPT 上 16–128 表现最好，Alpaca（短序列）上 16、32 较好，默认取 16。前缀缓存只缓存**满块**，块越大，短前缀越难命中。

        **延伸**：某些专用 kernel 对块大小有硬性要求（例如部分 MLA decode kernel 要求更大的块），部署时要和 kernel 对齐。
      `,
    },
    {
      id: 'qa-preempt',
      freq: 2,
      q: md`KV 块不够用时 vLLM 怎么办？swap 和 recompute 怎么选？`,
      a: md`
        **结论**：抢占最晚到达的请求（FCFS），一个序列组的块全部驱逐（all-or-nothing），之后用 swap 或 recompute 恢复。V1 默认 recompute。

        **对比**：swap 把块拷到 CPU 内存，受 PCIe 带宽限制，块小时大量小拷贝效率低；recompute 直接丢弃，恢复时把 prompt + 已生成 token 当新 prompt 一次 prefill，计算密集、速度快，配合前缀缓存还可能直接命中。论文结论：小块重算更快、大块换出更快，16–64 时相当。

        **延伸**：频繁抢占说明并发配置过高或 KV 空间不足，应调 \`gpu_memory_utilization\`、\`max_num_seqs\`、\`max_num_batched_tokens\`，或用 KV 量化、更多的卡。
      `,
    },
    {
      id: 'qa-capacity',
      freq: 2,
      q: md`vLLM 启动时怎么决定 KV cache 能分多少块？`,
      a: md`
        **结论**：先 profile，再做除法。

        **过程**：加载权重后，用最大的 batch token 数跑一次假数据前向，测出峰值激活显存；可用于 KV 的显存 = 总显存 × \`gpu_memory_utilization\`（默认 0.9）− 权重 − 峰值激活 − 其它开销；块数 = 可用显存 ÷ 每块字节数（$B \times 2 \times L \times n_{kv} \times d_h \times$ 字节数）。

        **延伸**：多卡 TP 时每卡只存自己那部分 KV 头，按每卡计算；用 FP8 KV cache 块数翻倍。这也解释了为什么 GQA / MLA 能直接提升并发：每 token 的 KV 变小，块数变多。
      `,
    },
  ],

  cheatsheet: md`
    **三类浪费**：预留 / 内部碎片（分配了但永不用）/ 外部碎片（总量够但不连续）。旧系统 KV 利用率 20.4%–38.2%；vLLM 浪费 < 4%。

    **块表**：\`slot = table[i // B] * B + i % B\`；默认 $B = 16$；每序列最多浪费 $B-1$ 个槽位。

    \`\`\`python
    def append_token(s):
        pos = L[s]
        if pos % B == 0: T[s].append(alloc())          # 满块 → 新块
        elif ref[T[s][-1]] > 1:                        # 共享未满块 → COW
            new = alloc(); ref[T[s][-1]] -= 1; T[s][-1] = new
        L[s] += 1
    fork: T[c] = list(T[p]); ref[b] += 1      free: ref[b] -= 1，归零才回收
    \`\`\`

    - Orca：迭代级调度 + selective batching；GPT-3 175B 上吞吐 36.9× FasterTransformer
    - vLLM：吞吐 2–4×（vs FT / Orca），kernel 自身慢 20–26%
    - 共享节省：beam search 37.6%–66.3%，parallel sampling 6.1%–30.5%
    - 抢占：FCFS 抢最晚的、全有或全无；swap（PCIe）vs recompute（重新 prefill）；V1 默认 recompute
    - 容量：(显存 × 0.9 − 权重 − 峰值激活) ÷ 每块字节
  `,
})
