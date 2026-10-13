/**
 * SFT：chat template、只在 assistant 回复上算 loss（labels=-100）、多轮掩码与 label shift、
 * packing（position_ids 重置、cu_seqlens、块对角注意力）、loss 按 token / 按样本平均。
 * 事实来源：Llama 2（arXiv 2307.09288 §3.1）、Llama 3（2407.21783 §4.1）、HF transformers DataCollatorWithFlattening 源码、
 * HF 博客 packing-with-FA2（2024-08）与 gradient_accumulation（2024-10-16）、TRL SFTTrainer 文档、LIMA（2305.11206）。
 */
import { defineLevel, md, py } from '@/content/helpers'
import { SftMaskViz } from '../viz/SftMaskViz'
import { ShiftViz } from '../viz/ShiftViz'
import { PackingViz } from '../viz/PackingViz'
import { LossAvgViz } from '../viz/LossAvgViz'
import { PY_REF_LOGPS } from '../shared'

const PY_CHATML = py`
  IM_START, IM_END, NL = 1, 2, 3
  ROLE = {"system": 10, "user": 11, "assistant": 12}

  def _ref_build(messages, mode="all"):
      ids, labels = [], []
      last = max([i for i, m in enumerate(messages) if m["role"] == "assistant"], default=-1)
      for i, m in enumerate(messages):
          head = [IM_START, ROLE[m["role"]], NL]
          body = list(m["content"]) + [IM_END]
          train = m["role"] == "assistant" and (mode == "all" or i == last)
          ids += head + body + [NL]
          labels += [-100] * len(head) + (body if train else [-100] * len(body)) + [-100]
      return ids, labels

  def _rand_conv(rng, n_turns=None, with_system=None):
      if with_system is None:
          with_system = rng.random() < 0.5
      if n_turns is None:
          n_turns = rng.randint(1, 4)
      msgs = []
      if with_system:
          msgs.append({"role": "system", "content": [rng.randint(100, 999) for _ in range(rng.randint(1, 4))]})
      for _ in range(n_turns):
          msgs.append({"role": "user", "content": [rng.randint(100, 999) for _ in range(rng.randint(1, 5))]})
          msgs.append({"role": "assistant", "content": [rng.randint(100, 999) for _ in range(rng.randint(0, 5))]})
      if rng.random() < 0.2:
          msgs.append({"role": "user", "content": [rng.randint(100, 999)]})   # 以 user 结尾（最后一轮没有回复）
      return msgs
`

export default defineLevel({
  learn: [
    {
      id: 'template',
      title: 'Chat template：对话怎么变成一条 token 序列',
      body: md`
        Transformer 只认一维 token 序列，多轮对话要先用 **chat template** 展开，每轮套上角色头和结束符：

        | 模板 | 一轮的样子 | 代表模型 |
        |---|---|---|
        | ChatML | \`<\|im_start\|>{role}⏎{content}<\|im_end\|>⏎\` | Qwen 系列 |
        | Llama 3 | \`<\|start_header_id\|>{role}<\|end_header_id\|>⏎⏎{content}<\|eot_id\|>\` | Llama 3.x |
        | Llama 2 | \`<s>[INST] {user} [/INST] {assistant} </s>\` | Llama 2-Chat、Mistral Instruct |

        SFT 的目标是让模型学会**以 assistant 身份回答**，所以通常只在 assistant 回复上计算 loss：其余位置的 label 置为 **-100**——
        PyTorch \`CrossEntropyLoss\` 的 \`ignore_index\` 默认值就是 -100，这些位置既不贡献 loss，也不参与求平均的分母。

        切换下面的策略和模板，数一数有多少 token 真正在被学习。
      `,
      viz: SftMaskViz,
      after: md`
        > [!WARN] 三个最常见的坑
        > 1. **结束符必须计入 loss**（\`<|im_end|>\` / \`<|eot_id|>\` / \`</s>\`）：否则模型学不会停，推理时一路生成到 max_tokens，或者自问自答下一轮 user。
        > 2. **训练与推理的模板必须逐字一致**：多一个空格、少一个换行，token 序列就变了。用 \`tokenizer.apply_chat_template\` 而不是手拼字符串。
        > 3. **base 模型没有这些特殊 token 的语义**：新增特殊 token 后要 resize embedding，并且这些 embedding 是从头学的；TRL 文档特别提醒，对 Qwen 这类自带模板的 base 模型要把 \`eos_token\` 对齐成 \`<|im_end|>\`。

        要不要“只训回复”并非铁律：Llama 2 与 Llama 3 的报告都写明 SFT 时 mask 掉 prompt 的 loss；
        但对 prompt 很短、回复很长的数据，差别不大；对 prompt 很长的数据（如长文档问答），全序列训练会让模型把大部分容量花在拟合用户输入上。
      `,
      checkpoint: {
        id: 'cp-template',
        type: 'single',
        prompt: md`在 HF / PyTorch 生态里，SFT 把不计 loss 的位置 label 设为 -100，原因是：`,
        options: [
          '`nn.CrossEntropyLoss` 的 `ignore_index` 默认值是 -100',
          '-100 是所有 tokenizer 的 pad_token_id',
          '-100 对应的 logit 会被自动置为 -inf',
          '负的 label 会让 loss 取相反数，从而抵消',
        ],
        answer: 0,
        explanation: md`\`ignore_index=-100\` 是 PyTorch 的默认约定，被忽略的位置不计入分子也不计入分母（\`reduction="mean"\` 时只对有效位置平均）。pad_token_id 因模型而异（常见 0、2、151643 等），-100 不是任何 token id。`,
      },
    },
    {
      id: 'shift',
      title: '多轮掩码与 label shift：mask 定义在“被预测的 token”上',
      body: md`
        HF 的约定是 **labels 与 input_ids 对齐、不需要你提前移位**，模型内部做 shift：\`logits[:, :-1]\` 对齐 \`labels[:, 1:]\`。所以：

        - “某 token 计入 loss”说的是**它作为被预测目标**时计入；
        - assistant 头的最后一个 token（ChatML 的 \`⏎\`）自己的 label 是 -100，但**它的 logits 负责预测回复的第一个 token**——前向必须包含它；
        - 一条 N 轮对话一次前向，就能同时训练所有 assistant 轮：因果注意力保证第 $k$ 轮回复只看到它之前的内容，与“截到第 $k$ 轮单独成样本”的 loss **逐 token 相同**。

        多轮数据的两种组织方式：
        | 方式 | 计算量 | 说明 |
        |---|---|---|
        | 一条样本，所有 assistant 轮都计 loss | 1 次前向 | 主流（TRL \`assistant_only_loss=True\`） |
        | 拆成 N 条，每条只训最后一轮 | 约 N 次前向，前缀重复计算 | 等价但浪费；有时用于只想训练最终回复的数据 |

        逐位置步进看看哪些预测进了 loss。
      `,
      viz: ShiftViz,
      after: md`
        > [!NOTE] 工具调用与思维链
        > Agent 数据里的 **tool 输出**（工具返回的结果）和 user 一样是环境给的，label 置 -100；模型自己写的 tool call 要训练。
        > 推理模型的 \`<think>…</think>\` 属于 assistant 内容，通常计入 loss；但多轮推理数据常把**历史轮**的思考过程删掉（推理时模板也会删），这时历史轮与当前轮的模板不一致，只能训最后一轮或拆样本。

        TRL 的 \`assistant_only_loss=True\` 依赖模板里的 \`{% generation %}…{% endgeneration %}\` 标记来生成 assistant mask（\`apply_chat_template(..., return_assistant_tokens_mask=True)\`）；模板里没有这对标记时，需要自己按角色边界构造 labels——这正是本关的实战题。
      `,
      checkpoint: {
        id: 'cp-shift',
        type: 'judge',
        prompt: md`单轮 ChatML 样本中，\`<|im_start|>assistant⏎\` 这三个 token 的 label 都是 -100，因此它们位置上的 logits 对 loss 没有任何贡献。`,
        answer: false,
        explanation: md`错误。label 为 -100 只表示“这些 token 作为目标时不计 loss”。由于 shift，最后一个 \`⏎\` 位置的 logits 预测的是 labels 中下一个位置——回复的第一个 token，**正是 loss 的第一项**。前两个位置的 logits 预测的目标是 -100，确实不贡献。`,
      },
    },
    {
      id: 'packing',
      title: 'Packing：拼接短样本，但要隔离',
      body: md`
        SFT 样本长短差异大，按最长的 padding 会浪费大量算力。**Packing** 把多条样本首尾拼进同一行（或拼成一条无 padding 的长序列），需要处理三件事：

        1. **position_ids 重置**：每条样本从 0 开始编号，否则 RoPE 看到的是“第 3000 个位置开始的对话”，与推理不一致；
        2. **注意力隔离**：只用下三角因果掩码时，后一条样本会看到前一条的内容。要么构造**块对角因果掩码**，要么用 FlashAttention 的 varlen 接口，传 \`cu_seqlens\`（各样本起点的前缀和，如 \`[0, 3, 6, 8]\`）；
        3. **边界 label**：上一条最后一个 token 不应去预测下一条的第一个 token。HF \`DataCollatorWithFlattening\` 的做法是把每条样本的第一个 label 换成 \`separator_id=-100\`。

        三种做法对比：
      `,
      viz: PackingViz,
      after: md`
        > [!TIP] 工程实现
        > - HF \`DataCollatorWithFlattening\`（2024 年加入）：把 batch 拼成一条无 padding 的序列，返回 \`input_ids\`、\`labels\`（每条首位为 -100）、\`position_ids\`；FlashAttention 2 据 position_ids 的重置点推出 \`cu_seqlens\`，调用 \`flash_attn_varlen_func\`。HF 博客报告在 FLAN 上吞吐最高约 2 倍、OrcaMath 上约 1.4 倍，loss 与 padding 训练一致。
        > - TRL \`packing=True\` 默认 \`packing_strategy="bfd"\`（best-fit decreasing 装箱），此时自动启用 padding-free；\`"wrapped"\` 则会把样本从中间截断硬拼。
        > - 新版 transformers 在训练时（未显式传 attention_mask、没有 KV cache）也会从 position_ids 的重置点推出“样本编号”，为 sdpa/eager 构造块对角掩码——**是框架根据 position_ids 构造了掩码**，不是 position_ids 本身隔离了注意力。

        不隔离会怎样？Llama 2 的 SFT 就是“把所有 prompt 和回答拼起来填满 4096 的序列、用特殊 token 分隔”。是否做了注意力隔离，报告没有写明；早期很多开源实现确实不隔离，实践中影响通常不大，但它是**训推不一致**，在短样本占比高时尤其明显。
      `,
      checkpoint: {
        id: 'cp-packing',
        type: 'fill',
        prompt: md`三条样本长度依次为 3、2、4，packing 成一条序列。正确的 position_ids 是 ____(1)____，FlashAttention varlen 需要的 cu_seqlens 是 ____(2)____。（用英文逗号分隔）`,
        blanks: [
          { label: '(1)', answers: ['0,1,2,0,1,0,1,2,3', '[0,1,2,0,1,0,1,2,3]'] },
          { label: '(2)', answers: ['0,3,5,9', '[0,3,5,9]'] },
        ],
        explanation: md`position_ids 在每条样本开头归零；cu_seqlens 是长度的前缀和并以 0 开头，长度为样本数 + 1，第 $i$ 条样本占据 \`[cu[i], cu[i+1])\`。`,
      },
    },
    {
      id: 'reduce',
      title: 'Loss 怎么平均：按 token 还是按样本',
      body: md`
        设 batch 中第 $i$ 条样本有 $n_i$ 个计入 loss 的 token，token loss 为 $\ell_{i,t}$：

        $$
        \mathcal L_{\text{token}}=\frac{\sum_i\sum_t\ell_{i,t}}{\sum_i n_i},\qquad
        \mathcal L_{\text{sample}}=\frac1B\sum_i\frac{1}{n_i}\sum_t\ell_{i,t}
        $$

        - **按 token 平均**（HF / TRL 默认）：每个 token 等权，长回复在梯度中占比与长度成正比；
        - **按样本平均**：每条样本等权，短回复的每个 token 权重更大——会**偏好短回复**，长回复里每个 token 学得更少。

        二者没有绝对对错，但**必须在全局一致**。坑在梯度累积和数据并行：如果每个 micro-batch 先在内部按 token 平均、再把各 micro-batch 的 loss 平均，结果既不是 ① 也不是 ②，而且依赖于 batch 怎么切。HF 在 2024-10-16 的博客中承认并修复了这个问题：正确做法是先对 loss **求和**，再除以整个累积步（以及所有卡）上的有效 token 总数（\`num_items_in_batch\`）。
      `,
      viz: LossAvgViz,
      after: md`
        > [!INTERVIEW] 面试官想听到的
        > “我用全局 token 平均：每张卡、每个 micro-batch 只算 loss 的**和**与有效 token 数，all-reduce 后再相除。这样梯度累积 4 步、8 卡训练与单卡大 batch 在数学上等价。若数据长度差异极大、又不想让长样本主导，可以显式做按样本平均或对长度加权，但要写进实验配置里。”

        同样的问题在 RL 里叫 **loss 聚合方式**（seq-mean-token-mean / token-mean / seq-mean-token-sum），是 Dr.GRPO、DAPO 讨论的核心之一，详见「推理强化学习」章节。
      `,
      checkpoint: {
        id: 'cp-reduce',
        type: 'numeric',
        prompt: md`batch 中两条样本：A 有 2 个回复 token，平均 token loss 为 3.0；B 有 8 个回复 token，平均 token loss 为 0.5。按**全局 token 平均**，batch loss 是多少？`,
        answer: 1.0,
        tolerance: 0.001,
        explanation: md`$(2\times3.0+8\times0.5)/(2+8)=10/10=1.0$。若按样本平均则为 $(3.0+0.5)/2=1.75$——短样本 A 的影响被放大了。`,
      },
    },
    {
      id: 'practice',
      title: 'SFT 在后训练中的位置与真实配方',
      body: md`
        | 来源 | SFT 配置 |
        |---|---|
        | Llama 2 | 27,540 条高质量标注（“Quality Is All You Need”）；cosine 学习率 2e-5、weight decay 0.1、batch 64、序列长 4096；拼接所有 prompt 与回答、特殊 token 分隔；**zero-out prompt 的 loss**；2 个 epoch |
        | Llama 3 | 用 RM 做拒绝采样得到的数据 + 合成数据；标准交叉熵、**mask 掉 prompt token**；最大模型 lr 1e-5、8.5K–9K 步 |
        | LIMA | 仅 1,000 条精选样本微调 LLaMA 65B，提出“表面对齐假设”：能力主要来自预训练，SFT 主要教格式与风格 |

        **SFT 解决了什么、代价是什么**：它用最低的成本教会模型对话格式、指令遵循和回复风格，是后续 RM / DPO / RL 的起点（π_ref）。
        但它只做**模仿**：只见过正例，不知道什么是“差回答”；训练时总以标准答案为前缀（teacher forcing），推理时却以自己的输出为前缀，错误会累积（exposure bias）。这正是偏好对齐与 RL 要补的课。
      `,
      after: md`
        > [!NEW] 2025–2026 的 SFT
        > 推理模型（DeepSeek-R1 等）的流水线里，SFT 常作为 RL 前的“冷启动”：用少量长思维链数据教会格式与反思模式，再交给 RLVR；蒸馏小模型则直接对教师的推理轨迹做 SFT。on-policy 蒸馏（学生自己采样、教师逐 token 打分）进一步缓解了 exposure bias，见「推理强化学习」章节。
      `,
      checkpoint: {
        id: 'cp-practice',
        type: 'multi',
        prompt: md`以下哪些是 Llama 2 技术报告中写明的 SFT 做法？`,
        options: ['只在回答 token 上反向传播（zero-out prompt loss）', '把 prompt 与回答拼接以填满 4096 长度的序列', '训练 2 个 epoch', '使用 DPO 作为 SFT 的损失', '只用了 1,000 条样本'],
        answer: [0, 1, 2],
        explanation: md`Llama 2 §3.1：cosine lr 2e-5、wd 0.1、batch 64、长度 4096，拼接 prompt 与回答并用特殊 token 分隔，zero-out prompt loss，训练 2 个 epoch，标注数据 27,540 条。1,000 条是 LIMA；DPO 是 Llama 3 偏好对齐阶段的算法。`,
      },
    },
  ],

  quiz: [
    {
      id: 'q-eos',
      type: 'judge',
      difficulty: 1,
      prompt: md`SFT 时，assistant 回复末尾的结束符（如 \`<|im_end|>\`）属于模板 token，应该和角色头一样置为 -100。`,
      answer: false,
      explanation: md`结束符必须计入 loss，这是模型学会“何时停止”的唯一信号。角色头（\`<|im_start|>assistant⏎\`）推理时由模板直接给出，可以不学；但结束符是模型自己要生成的。`,
    },
    {
      id: 'q-pick',
      type: 'pick',
      difficulty: 2,
      prompt: md`ChatML 格式的两轮对话（\`⏎\` 表示换行 token）。采用“所有 assistant 轮都计 loss、含结束符”的策略，点出 label **不是** -100 的 token：`,
      items: ['<|im_start|>', 'user', '⏎', '你好', '<|im_end|>', '⏎', '<|im_start|>', 'assistant', '⏎', '你好！', '<|im_end|>', '⏎', '<|im_start|>', 'user', '⏎', '再见', '<|im_end|>', '⏎', '<|im_start|>', 'assistant', '⏎', '拜拜', '<|im_end|>', '⏎'],
      cols: 6,
      answer: [9, 10, 21, 22],
      explanation: md`只有两轮 assistant 的内容（“你好！”“拜拜”）及其后的 \`<|im_end|>\` 计入 loss。user 轮全部、assistant 头、\`<|im_end|>\` 之后的换行都是 -100（推理时这些由模板或用户提供）。`,
    },
    {
      id: 'q-avg',
      type: 'fill',
      difficulty: 2,
      prompt: md`batch 中样本 A 有 4 个回复 token、每个 loss 都是 2.0；样本 B 有 12 个回复 token、每个 loss 都是 1.0。按全局 token 平均的 loss = ____(1)____，按样本平均的 loss = ____(2)____。`,
      blanks: [
        { label: '(1)', answers: ['1.25'], numeric: true, tolerance: 0.001 },
        { label: '(2)', answers: ['1.5'], numeric: true, tolerance: 0.001 },
      ],
      explanation: md`token 平均：$(4\times2+12\times1)/16=20/16=1.25$；样本平均：$(2.0+1.0)/2=1.5$。样本平均放大了短样本 A 的权重。`,
    },
    {
      id: 'q-pack-multi',
      type: 'multi',
      difficulty: 2,
      prompt: md`关于 SFT 的 packing，下列说法正确的有：`,
      options: [
        '拼接后每条样本的 position_ids 应从 0 重新开始',
        '只要重置了 position_ids，即使注意力仍用普通下三角因果掩码，也不会发生跨样本注意力',
        'FlashAttention 的 varlen 接口用 cu_seqlens 表示各样本边界',
        '应避免上一条样本的最后一个 token 去预测下一条样本的第一个 token',
        'packing 会改变每个 token 的 loss，因此与 padding 训练的结果在数学上不可能一致',
      ],
      answer: [0, 2, 3],
      explanation: md`
        B 错：position_ids 只影响位置编码（RoPE），注意力范围由掩码决定；需要块对角掩码或 varlen kernel（有些框架会根据 position_ids 的重置点**替你构造**掩码，但那是框架行为）。
        E 错：position 重置 + 注意力隔离 + 边界 label 置 -100 时，每个 token 的条件分布与单独训练完全相同，loss 一致（HF 博客的实验也显示验证 loss 一致），差别只在 batch 内样本数变化对按样本平均的影响。
      `,
    },
    {
      id: 'q-order',
      type: 'order',
      difficulty: 2,
      prompt: md`把一条多轮对话变成 SFT 训练信号的步骤按顺序排列：`,
      items: [
        '用 chat template 把 messages 渲染成文本（或直接渲染成 token）',
        'tokenize 得到 input_ids，并记录每个 token 属于哪个角色',
        '构造 labels：非 assistant 回复位置置为 -100（结束符保留）',
        'packing：拼接样本、重置 position_ids、记录 cu_seqlens、边界 label 置 -100',
        '前向：logits[:, :-1] 对齐 labels[:, 1:]，按有效 token 计算交叉熵',
      ],
      explanation: md`模板 → token → labels → 组 batch（packing / padding）→ shift 后的交叉熵。先 packing 再造 labels 会丢失角色边界信息。`,
    },
    {
      id: 'q-match',
      type: 'match',
      difficulty: 1,
      prompt: md`把模板特征与模型家族配对：`,
      pairs: [
        ['`<|im_start|>` / `<|im_end|>`', 'ChatML（Qwen 系列）'],
        ['`<|start_header_id|>` … `<|eot_id|>`', 'Llama 3.x'],
        ['`[INST] … [/INST]`', 'Llama 2-Chat / Mistral Instruct'],
        ['`ignore_index=-100`', 'PyTorch 交叉熵忽略标签的默认值'],
      ],
      explanation: md`三种主流模板 + 一个通用约定。面试中常被要求手写某个模板展开后的 token 序列并标出哪些位置算 loss。`,
    },
    {
      id: 'q-bug',
      type: 'bug',
      difficulty: 2,
      prompt: md`下面的 SFT loss 有 **2 处** bug（\`labels\` 与 input_ids 对齐、未移位，-100 表示不计 loss）。点出有问题的行。`,
      code: py`
        def sft_loss(logits, labels):
            # logits: (B, T, V)，labels: (B, T)
            logits = logits[:, :-1]
            labels = labels[:, :-1]
            m = logits.max(-1, keepdims=True)
            logp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
            mask = labels != -100
            safe = np.where(mask, labels, 0)
            nll = -np.take_along_axis(logp, safe[..., None], -1)[..., 0]
            return (nll * mask).sum() / labels.size
      `,
      bugLines: [4, 10],
      explanation: md`
        第 4 行：应为 \`labels[:, 1:]\`（位置 $t$ 的 logits 预测第 $t+1$ 个 token）。
        第 10 行：分母应为有效 token 数 \`mask.sum()\`；用 \`labels.size\` 会把 -100 的位置也算进分母，prompt 越长 loss 越被稀释，学习率等效变小。
      `,
    },
    {
      id: 'q-parsons',
      type: 'parsons',
      difficulty: 2,
      prompt: md`拼出“给一轮对话追加 token 并构造 labels”的代码（ChatML：头 = \`[IM_START, role_id, NL]\`，尾 = \`[IM_END, NL]\`；只训练 assistant 的内容与 IM_END）。有 2 行干扰项。`,
      lines: [
        'def append_turn(ids, labels, role, content):',
        '    head = [IM_START, ROLE[role], NL]',
        '    body = content + [IM_END]',
        '    ids += head + body + [NL]',
        '    labels += [-100] * len(head)',
        '    labels += body if role == "assistant" else [-100] * len(body)',
        '    labels += [-100]',
      ],
      distractors: ['    labels += head', '    body = content'],
      explanation: md`头和末尾换行都不训练；body 包含 IM_END，assistant 轮原样作为 label。干扰项：把头当 label 会让模型学习生成角色头；body 不含 IM_END 则学不会停。`,
    },
    {
      id: 'q-pad-waste',
      type: 'numeric',
      difficulty: 2,
      unit: '%',
      prompt: md`一个 batch 有 4 条样本，长度分别为 100、300、1000、600 token，按最长样本右侧 padding。padding token 占 batch 全部位置的百分之多少？`,
      answer: 50,
      tolerance: 0.001,
      explanation: md`总位置 $4\times1000=4000$，真实 token $100+300+1000+600=2000$，padding 占 $2000/4000=50\%$。这些位置照样要做注意力和 FFN 计算（除非用 varlen），这就是 packing / padding-free 的动机。`,
    },
    {
      id: 'q-ga',
      type: 'single',
      difficulty: 3,
      prompt: md`梯度累积 2 步：micro-batch 1 有 10 个有效 token、总 loss（求和）为 30；micro-batch 2 有 990 个有效 token、总 loss 为 990。若每个 micro-batch 先按 token 平均、再对两步取平均，得到的 loss 与“全局按 token 平均”相比：`,
      options: [
        '朴素做法 2.0，全局做法 1.02：micro-batch 1 的 10 个 token 被放大了约 50 倍权重',
        '两者相等，都是 1.02',
        '朴素做法 1.02，全局做法 2.0',
        '两者相等，都是 2.0',
      ],
      answer: 0,
      explanation: md`
        朴素：$(30/10+990/990)/2=(3+1)/2=2.0$；全局：$(30+990)/(10+990)=1.02$。
        朴素做法中 micro-batch 1 的每个 token 权重为 $\tfrac12\cdot\tfrac1{10}$，micro-batch 2 为 $\tfrac12\cdot\tfrac1{990}$，相差 99 倍；相对全局 $1/1000$ 的权重放大了 50 倍。这正是 HF 2024 年 10 月修复的梯度累积 bug：改为先求和、再除以累积步内的有效 token 总数。
      `,
    },
    {
      id: 'q-multiturn',
      type: 'judge',
      difficulty: 2,
      prompt: md`一条 3 轮对话“作为一条样本、所有 assistant 轮都计 loss”，与“拆成 3 条样本、第 k 条只在第 k 轮回复上计 loss”相比，每个回复 token 的 loss 值完全相同，区别主要在计算量。`,
      answer: true,
      explanation: md`因果注意力下，第 $k$ 轮回复的每个 token 只依赖它之前的内容，两种组织方式中它的条件分布完全一样，逐 token loss 相同（是否等价于同一个 batch loss，还取决于归一化方式与 batch 组成）。拆分做法的前缀被重复前向，计算量约为 $O(N)$ 倍。`,
    },
    {
      id: 'q-why-mask',
      type: 'single',
      difficulty: 1,
      prompt: md`SFT 只在 assistant 回复上计算 loss，最主要的目的是：`,
      options: [
        '让模型学习“以 assistant 身份回答”，而不是去拟合用户输入与 system prompt 的分布',
        '减少前向计算量',
        '避免显存溢出',
        '让 position_ids 正确',
      ],
      answer: 0,
      explanation: md`被 mask 的 token 仍然参与前向（作为上下文），计算量与显存几乎不变；mask 只改变**监督信号**。全序列训练会让模型学着生成用户的话、复述 system prompt。`,
    },
  ],

  guided: [
    {
      id: 'g-pack',
      title: '一步步写出 padding-free packing',
      intro: md`
        实现 \`pack(samples)\`：把若干条已经构造好 labels 的样本拼成**一条**无 padding 的序列（与 HF \`DataCollatorWithFlattening\` 相同的约定）。

        - 输入：\`samples\` 是 \`[(input_ids, labels), ...]\`，均为 Python list；
        - 输出 dict：\`input_ids\`、\`labels\`（每条样本的**第一个** label 换成 -100）、\`position_ids\`（每条从 0 开始）、\`cu_seqlens\`（以 0 开头的长度前缀和）。
      `,
      template: py`
        def pack(samples):
            input_ids, labels, position_ids = [], [], []
            cu_seqlens = {{1}}
            for ids, lab in samples:
                input_ids += ids
                labels += {{2}}
                position_ids += {{3}}
                {{4}}
            return {"input_ids": input_ids, "labels": labels,
                    "position_ids": position_ids, "cu_seqlens": cu_seqlens}
      `,
      steps: [
        {
          blank: 1,
          instruction: md`cu_seqlens 以 0 开头。`,
          answer: '[0]',
          explain: md`长度为“样本数 + 1”，第 $i$ 条样本占据 \`[cu[i], cu[i+1])\`。`,
        },
        {
          blank: 2,
          instruction: md`拼接 labels，但把这条样本的第一个 label 换成 -100。`,
          answer: '[-100] + lab[1:]',
          accept: ['[-100] + list(lab[1:])'],
          explain: md`模型内部 shift 后，上一条样本最后一个位置的 logits 会去预测这一位置——置为 -100 就切断了跨样本预测。对 SFT 数据它通常本来就是 -100（第一个 token 是模板头），但对预训练式数据（labels = input_ids）至关重要。`,
        },
        {
          blank: 3,
          instruction: md`每条样本的位置从 0 开始编号。`,
          answer: 'list(range(len(ids)))',
          accept: ['[i for i in range(len(ids))]'],
          explain: md`RoPE 只看 position_ids；不重置的话，第二条样本会被当作“从第几千个位置开始”的上下文。`,
        },
        {
          blank: 4,
          instruction: md`更新 cu_seqlens。`,
          answer: 'cu_seqlens.append(cu_seqlens[-1] + len(ids))',
          accept: ['cu_seqlens += [cu_seqlens[-1] + len(ids)]'],
          explain: md`FlashAttention varlen 用它切出每条样本，只在块内做因果注意力；\`max_seqlen\` 则取 \`max(len(ids))\`。`,
        },
      ],
      tests: [
        {
          name: '示例：长度 3、2、4',
          code: py`
            s = [([5, 6, 7], [-100, 6, 7]), ([8, 9], [8, 9]), ([1, 2, 3, 4], [-100, -100, 3, 4])]
            out = pack(s)
            assert out["input_ids"] == [5, 6, 7, 8, 9, 1, 2, 3, 4], f"input_ids 错：{out['input_ids']}"
            assert out["position_ids"] == [0, 1, 2, 0, 1, 0, 1, 2, 3], f"position_ids 应为 [0,1,2,0,1,0,1,2,3]，得到 {out['position_ids']}"
            assert out["cu_seqlens"] == [0, 3, 5, 9], f"cu_seqlens 应为 [0,3,5,9]，得到 {out['cu_seqlens']}"
            assert out["labels"] == [-100, 6, 7, -100, 9, -100, -100, 3, 4], f"labels 错：{out['labels']}"
          `,
        },
        {
          name: '单条样本与空列表',
          code: py`
            out = pack([([1, 2], [1, 2])])
            assert out["labels"] == [-100, 2] and out["position_ids"] == [0, 1] and out["cu_seqlens"] == [0, 2], f"得到 {out}"
            out = pack([])
            assert out["cu_seqlens"] == [0] and out["input_ids"] == [], f"空输入：得到 {out}"
          `,
        },
        {
          name: '随机对拍',
          hidden: true,
          code: py`
            import random
            rnd = random.Random(3)
            for _ in range(50):
                samples = []
                for _ in range(rnd.randint(1, 6)):
                    n = rnd.randint(1, 7)
                    ids = [rnd.randint(1, 50) for _ in range(n)]
                    lab = [x if rnd.random() < 0.6 else -100 for x in ids]
                    samples.append((ids, lab))
                out = pack(samples)
                pos, cu, labs = [], [0], []
                for ids, lab in samples:
                    pos += list(range(len(ids))); cu.append(cu[-1] + len(ids)); labs += [-100] + lab[1:]
                assert out["position_ids"] == pos and out["cu_seqlens"] == cu and out["labels"] == labs, f"samples={samples}"
          `,
        },
      ],
      outro: md`
        有了 \`cu_seqlens\`，FlashAttention 的 \`flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=m, max_seqlen_k=m, causal=True)\` 就会在每个块内独立做因果注意力；
        不用 varlen 时，等价的做法是构造块对角因果掩码：\`mask[i, j] = (j <= i) and (seg[i] == seg[j])\`。
      `,
    },
  ],

  challenges: [
    {
      id: 'c-build-labels',
      title: '多轮对话的 SFT 样本构造（ChatML + labels 掩码）',
      difficulty: 'medium',
      source: '大模型岗手撕 · SFT 数据',
      timeLimitMin: 20,
      description: md`
        用简化的 ChatML 模板（全是整数 token id）把多轮对话展开，并构造 labels：

        \`\`\`python
        IM_START, IM_END, NL = 1, 2, 3
        ROLE = {"system": 10, "user": 11, "assistant": 12}
        def build_sft_example(messages: list[dict], mode: str = "all") -> tuple[list[int], list[int]]: ...
        \`\`\`

        - 每条消息 \`{"role": ..., "content": list[int]}\` 展开为 \`[IM_START, ROLE[role], NL] + content + [IM_END, NL]\`，依次拼接；
        - labels 与 input_ids 等长、**不移位**；默认全是 \`-100\`，只有“被训练的 assistant 轮”的 **content 与其后的 IM_END** 取对应的 token id（角色头和 IM_END 后的 NL 不训练）；
        - \`mode="all"\`：训练所有 assistant 轮；\`mode="last"\`：只训练**最后一个** assistant 轮；没有 assistant 轮时 labels 全为 -100；
        - assistant 的 content 可能为空列表（此时只训练 IM_END）。

        **示例**：\`[{"role":"user","content":[100,101]}, {"role":"assistant","content":[200]}]\`
        - input_ids = \`[1,11,3,100,101,2,3, 1,12,3,200,2,3]\`
        - labels = \`[-100]*7 + [-100,-100,-100,200,2,-100]\`

        常量 \`IM_START\`、\`IM_END\`、\`NL\`、\`ROLE\` 已在初始代码中定义。
      `,
      starter: py`
        IM_START, IM_END, NL = 1, 2, 3
        ROLE = {"system": 10, "user": 11, "assistant": 12}

        def build_sft_example(messages, mode="all"):
            """返回 (input_ids, labels)，两者等长；不训练的位置为 -100"""
            input_ids, labels = [], []
            # TODO: 找出要训练的 assistant 轮；逐轮拼接 head + content + [IM_END] + [NL]
            return input_ids, labels
      `,
      solution: py`
        IM_START, IM_END, NL = 1, 2, 3
        ROLE = {"system": 10, "user": 11, "assistant": 12}

        def build_sft_example(messages, mode="all"):
            assistant_turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            if mode == "last":
                train = set(assistant_turns[-1:])
            else:
                train = set(assistant_turns)

            input_ids, labels = [], []
            for i, m in enumerate(messages):
                head = [IM_START, ROLE[m["role"]], NL]       # 模板头：推理时由模板给出，不训练
                body = list(m["content"]) + [IM_END]        # 内容 + 结束符：assistant 轮要训练
                input_ids += head + body + [NL]
                labels += [-100] * len(head)
                labels += body if i in train else [-100] * len(body)
                labels += [-100]                            # IM_END 后的换行属于下一轮的前缀
            return input_ids, labels
      `,
      solutionAlt: [
        {
          label: 'PyTorch / HF 版（仅展示：真实 tokenizer 下按角色构造 labels）',
          code: py`
            def build_sft_example_hf(tokenizer, messages, mode="all"):
                # 逐轮渲染、分段 tokenize，保证能拿到每段的边界（模板需与 apply_chat_template 完全一致）
                input_ids, labels = [], []
                asst = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
                train = set(asst[-1:] if mode == "last" else asst)
                for i, m in enumerate(messages):
                    head = tokenizer.encode(f"<|im_start|>{m['role']}\n", add_special_tokens=False)
                    body = tokenizer.encode(m["content"] + "<|im_end|>", add_special_tokens=False)
                    tail = tokenizer.encode("\n", add_special_tokens=False)
                    input_ids += head + body + tail
                    labels += [-100] * len(head) + (body if i in train else [-100] * len(body)) + [-100] * len(tail)
                return torch.tensor(input_ids), torch.tensor(labels)

            # 模板带 {% generation %} 标记时可直接：
            # out = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
            #                                     return_assistant_tokens_mask=True)
            # labels = [t if m else -100 for t, m in zip(out["input_ids"], out["assistant_masks"])]
          `,
        },
      ],
      explanation: md`
        **思路**：先确定哪些 assistant 轮要训练，再逐轮拼接 head / body / tail 三段，labels 对应地填 -100 或原 token。

        **要点**：
        - body 包含 IM_END——学会停止；
        - head 与 IM_END 之后的 NL 不训练：推理时模板会自动添加 \`<|im_start|>assistant⏎\`，而 NL 是下一轮的前缀；
        - 分段 tokenize 时要小心 BPE 跨段合并：真实 tokenizer 中“内容最后一个字符 + 结束符”可能被合并成不同 token，所以工程上要么用模板的 generation 标记，要么保证特殊 token 两侧不会发生合并，最后用 \`apply_chat_template\` 的结果核对 input_ids 完全一致。

        **复杂度**：$O(L)$，$L$ 为总 token 数。
      `,
      complexity: { time: 'O(L)', space: 'O(L)' },
      hints: [
        md`先求 \`assistant_turns\`；\`mode == "last"\` 时只保留最后一个。`,
        md`每轮三段：\`head = [IM_START, ROLE[role], NL]\`、\`body = content + [IM_END]\`、\`tail = [NL]\`。`,
        md`labels：head → -100，body → 训练轮原样 / 否则 -100，tail → -100。`,
      ],
      setup: PY_CHATML,
      tests: [
        {
          name: '示例：单轮',
          code: py`
            msgs = [{"role": "user", "content": [100, 101]}, {"role": "assistant", "content": [200]}]
            ids, lab = build_sft_example(msgs)
            assert ids == [1, 11, 3, 100, 101, 2, 3, 1, 12, 3, 200, 2, 3], f"input_ids 错：{ids}"
            exp = [-100] * 7 + [-100, -100, -100, 200, 2, -100]
            assert lab == exp, f"labels 期望 {exp}，得到 {lab}"
          `,
        },
        {
          name: '多轮 + system：mode="all" 与 "last"',
          code: py`
            msgs = [{"role": "system", "content": [50]},
                    {"role": "user", "content": [100]}, {"role": "assistant", "content": [200, 201]},
                    {"role": "user", "content": [102]}, {"role": "assistant", "content": [202]}]
            for mode in ("all", "last"):
                got = build_sft_example(msgs, mode)
                exp = _ref_build(msgs, mode)
                assert got == exp, f"mode={mode}：期望 labels {exp[1]}，得到 {got[1]}"
          `,
        },
        {
          name: '没有 assistant 轮 → 全为 -100',
          code: py`
            ids, lab = build_sft_example([{"role": "system", "content": [7]}, {"role": "user", "content": [8, 9]}])
            assert len(ids) == len(lab) == 13, f"长度应为 13，得到 {len(ids)}, {len(lab)}"
            assert all(x == -100 for x in lab), f"labels 应全为 -100，得到 {lab}"
          `,
        },
        {
          name: '空回复只训练 IM_END',
          hidden: true,
          code: py`
            msgs = [{"role": "user", "content": [100]}, {"role": "assistant", "content": []}]
            ids, lab = build_sft_example(msgs)
            assert [x for x in lab if x != -100] == [2], f"只应训练 IM_END，得到 {[x for x in lab if x != -100]}"
          `,
        },
        {
          name: '随机对拍（含以 user 结尾的对话）',
          hidden: true,
          code: py`
            import random
            rnd = random.Random(2026)
            for _ in range(300):
                msgs = _rand_conv(rnd)
                mode = rnd.choice(["all", "last"])
                got = build_sft_example(msgs, mode)
                exp = _ref_build(msgs, mode)
                assert got == exp, f"mode={mode}, messages={msgs}\n期望 {exp[1]}\n得到 {got[1]}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-truncate',
          q: md`对话超过 max_length 需要截断时，应该怎么截？`,
          a: md`
            不要简单地截掉尾部：可能把最后一轮回复截掉一半（模型学到“回答到一半就结束”，如果结束符也被截掉则更糟），或把 assistant 头留在末尾。
            常见做法：① 按轮丢弃**最早**的历史轮（保留 system 与最后几轮）；② 若单轮回复本身超长，整条样本丢弃或单独处理；③ 截断后重新检查 labels 中是否至少有一个有效 token，全为 -100 的样本要过滤（否则按 token 平均时分母为 0）。
          `,
        },
        {
          id: 'f-sysprompt',
          q: md`system prompt 要不要训练？如果数据里很多样本共用同一个超长 system prompt，有什么优化？`,
          a: md`
            不训练（label 为 -100），它是条件而不是输出。共用前缀时：packing 后同一行里会重复出现很多次；可以用前缀共享 / prefix caching 风格的训练（如把多条回复挂在同一前缀的树状注意力上），或把 system prompt 截短、改成可学习的 soft prompt。这属于进阶优化，面试中提到“前缀重复计算是浪费”即可。
          `,
        },
      ],
    },
    {
      id: 'c-sft-loss',
      title: '带 ignore_index 的 SFT loss：按 token 与按样本平均',
      difficulty: 'medium',
      source: '大模型岗手撕 · SFT loss',
      timeLimitMin: 20,
      description: md`
        \`\`\`python
        def sft_loss(logits: np.ndarray, labels: np.ndarray, reduction: str = "token") -> float: ...
        \`\`\`

        - \`logits\`：\`(B, T, V)\` float64；\`labels\`：\`(B, T)\` int，与 input_ids 对齐（**未移位**），\`-100\` 不计 loss；
        - 先 shift：\`logits[:, :-1]\` 预测 \`labels[:, 1:]\`；
        - \`reduction="token"\`：所有有效 token 的 NLL 之和 ÷ 有效 token 总数；
        - \`reduction="sample"\`：每条样本先对自己的有效 token 求平均，再对“至少有一个有效 token 的样本”求平均；
        - 没有任何有效 token 时返回 \`0.0\`（不能返回 nan）；
        - 数值稳定：logits 可能达到 $\pm10^4$。
      `,
      starter: py`
        import numpy as np

        def sft_loss(logits, labels, reduction="token"):
            # TODO: shift → 稳定 log_softmax → gather → mask → 两种归一化
            raise NotImplementedError
      `,
      solution: py`
        import numpy as np

        def sft_loss(logits, labels, reduction="token"):
            lg, lab = logits[:, :-1, :], labels[:, 1:]               # shift
            mask = lab != -100
            safe = np.where(mask, lab, 0)
            m = lg.max(axis=-1, keepdims=True)
            logp = lg - m - np.log(np.exp(lg - m).sum(axis=-1, keepdims=True))
            nll = -np.take_along_axis(logp, safe[..., None], axis=-1)[..., 0]
            nll = np.where(mask, nll, 0.0)                           # (B, T-1)

            if reduction == "token":
                n = mask.sum()
                return float(nll.sum() / n) if n > 0 else 0.0
            # sample：先样本内平均，再在“有有效 token 的样本”间平均
            cnt = mask.sum(axis=-1)
            valid = cnt > 0
            if not valid.any():
                return 0.0
            per_sample = nll.sum(axis=-1)[valid] / cnt[valid]
            return float(per_sample.mean())
      `,
      solutionAlt: [
        {
          label: 'PyTorch 版（仅展示）',
          code: py`
            import torch
            import torch.nn.functional as F

            def sft_loss(logits, labels, reduction="token"):
                lg, lab = logits[:, :-1].float(), labels[:, 1:]
                nll = F.cross_entropy(lg.transpose(1, 2), lab, ignore_index=-100, reduction="none")  # (B, T-1)
                mask = (lab != -100).float()
                if reduction == "token":
                    return nll.sum() / mask.sum().clamp(min=1)
                cnt = mask.sum(-1)
                valid = cnt > 0
                return (nll.sum(-1)[valid] / cnt[valid]).mean() if valid.any() else nll.sum() * 0

            # 梯度累积 / 多卡下的“全局 token 平均”：每步返回 nll.sum()，
            # 再除以 all_reduce 后的整个累积步的有效 token 总数（HF 的 num_items_in_batch）
          `,
        },
      ],
      explanation: md`
        **要点**：
        1. shift 后再做 mask：mask 定义在“被预测的 token”上；
        2. -100 不能直接当下标：NumPy 中 \`a[..., -100]\` 会静默取倒数第 100 个元素（V ≥ 100 时不报错！），先用 \`np.where\` 换成 0 再 gather，最后乘 mask；
        3. token 平均与样本平均的分母不同；全为 -100 的样本在样本平均中必须被排除，否则 0/0 = nan；
        4. \`F.cross_entropy(..., reduction="mean")\` 已经是 token 平均，且自动忽略 -100。

        **复杂度**：$O(B\cdot T\cdot V)$。
      `,
      complexity: { time: 'O(B·T·V)', space: 'O(B·T·V)' },
      hints: [
        md`和 DPO 关卡的 \`sequence_logps\` 一样先 shift、做稳定 log_softmax、gather。`,
        md`把 \`nll\` 在无效位置清零后，token 平均 = \`nll.sum() / mask.sum()\`。`,
        md`样本平均：\`cnt = mask.sum(-1)\`，只对 \`cnt > 0\` 的样本求 \`nll.sum(-1) / cnt\` 再平均。`,
      ],
      setup: PY_REF_LOGPS + '\n' + py`
        def _ref_sft(logits, labels, reduction="token"):
            tok, mask = _ref_token_logps(logits, labels)
            nll = -tok
            if reduction == "token":
                n = mask.sum()
                return float(nll.sum() / n) if n else 0.0
            cnt = mask.sum(-1)
            v = cnt > 0
            return float((nll.sum(-1)[v] / cnt[v]).mean()) if v.any() else 0.0

        def _make_sft(seed, B=4, T=10, V=8, scale=2.0):
            rng = np.random.default_rng(seed)
            logits = rng.normal(0, scale, size=(B, T, V))
            labels = np.full((B, T), -100, dtype=np.int64)
            for b in range(B):
                p = int(rng.integers(1, T - 1))
                L = int(rng.integers(0, T - p + 1))
                labels[b, p:p + L] = rng.integers(0, V, size=L)
            return logits, labels
      `,
      tests: [
        {
          name: '手算示例',
          code: py`
            lg = np.log(np.array([[[0.5, 0.5], [0.25, 0.75], [0.9, 0.1]]]))
            lab = np.array([[-100, 1, 1]])
            got = sft_loss(lg, lab)
            exp = -(np.log(0.5) + np.log(0.75)) / 2
            assert abs(got - exp) < 1e-9, f"期望 {exp:.6f}，得到 {got}"
          `,
        },
        {
          name: 'token 平均与样本平均不同',
          code: py`
            logits, labels = _make_sft(1)
            t, s = sft_loss(logits, labels, "token"), sft_loss(logits, labels, "sample")
            et, es = _ref_sft(logits, labels, "token"), _ref_sft(logits, labels, "sample")
            assert abs(t - et) < 1e-8, f"token：期望 {et:.6f}，得到 {t:.6f}"
            assert abs(s - es) < 1e-8, f"sample：期望 {es:.6f}，得到 {s:.6f}"
          `,
        },
        {
          name: '全部为 -100 → 0.0',
          code: py`
            logits = np.zeros((2, 5, 4))
            labels = np.full((2, 5), -100)
            for r in ("token", "sample"):
                got = sft_loss(logits, labels, r)
                assert got == 0.0, f"reduction={r}：没有有效 token 时应返回 0.0，得到 {got}"
          `,
        },
        {
          name: '随机对拍（含整条无效的样本）',
          hidden: true,
          code: py`
            for s in range(2, 12):
                logits, labels = _make_sft(s, B=5, T=9, V=120)
                labels[0] = -100                                    # 一条全无效样本
                for r in ("token", "sample"):
                    got, exp = sft_loss(logits, labels, r), _ref_sft(logits, labels, r)
                    assert abs(got - exp) < 1e-8, f"seed={s} reduction={r}：期望 {exp:.6f}，得到 {got:.6f}"
          `,
        },
        {
          name: '数值稳定：logits ±1e4',
          hidden: true,
          code: py`
            logits, labels = _make_sft(42, scale=4000.0)
            for r in ("token", "sample"):
                got = sft_loss(logits, labels, r)
                assert np.isfinite(got), f"reduction={r} 得到 {got}：log_softmax 溢出？"
                exp = _ref_sft(logits, labels, r)
                assert abs(got - exp) < 1e-6 * max(1, abs(exp)), f"reduction={r}：期望 {exp:.4f}，得到 {got:.4f}"
          `,
        },
      ],
      followUps: [
        {
          id: 'f-ce-chunk',
          q: md`词表 15 万、序列 32K 时，logits 张量放不下，交叉熵怎么算？`,
          a: md`
            $32768\times151936\times4\ \text{B}\approx19.9\ \text{GB}$（单条、fp32）。做法：① 沿序列维分块，逐块算 lm_head 投影 + log-softmax + gather，再丢掉 logits（必要时对反向做重计算）；② 融合 kernel（如 Liger Kernel 的 fused linear cross entropy）在 SRAM 里完成投影与交叉熵，不物化完整 logits；③ 只对 label 不为 -100 的位置算 lm_head（SFT 中 prompt 很长时收益明显）。
          `,
        },
      ],
    },
  ],

  interview: [
    {
      id: 'qa-mask',
      freq: 3,
      q: md`SFT 时为什么只在 assistant 回复上计算 loss？多轮对话怎么处理？`,
      a: md`
        **结论**：把 system / user / 模板头的 label 置为 -100，只训练 assistant 内容与结束符；多轮对话作为一条样本，所有 assistant 轮都计 loss。

        **原因**：目标是学“给定上下文如何回答”。用户输入不是模型要生成的分布，拟合它会浪费容量，还会让模型学会冒充用户、复述 system prompt。被 mask 的 token 仍作为上下文参与前向，只是不提供监督。由于因果注意力，第 $k$ 轮回复的 loss 与“截断到第 $k$ 轮单独训练”完全相同，所以一次前向就能训练所有轮。

        **延伸**：结束符必须计入；tool 输出置 -100；HF 由模型内部 shift，labels 不要提前移位；TRL 用 \`assistant_only_loss=True\`（依赖模板中的 \`{% generation %}\` 标记）。
      `,
    },
    {
      id: 'qa-packing',
      freq: 3,
      q: md`SFT 的 packing 是什么？要注意哪些问题？`,
      a: md`
        **结论**：把多条短样本拼成一条长序列以消除 padding 浪费，必须做到位置重置、注意力隔离和边界 label 屏蔽。

        **原因**：① position_ids 每条从 0 开始，否则 RoPE 位置与推理不一致；② 普通因果掩码会让后一条样本看到前一条，需要块对角因果掩码或 FlashAttention varlen（传 \`cu_seqlens\`）；③ 每条样本的第一个 label 置 -100，避免上一条的末尾去预测下一条的开头。三者都做到时，每个 token 的条件分布与单独训练完全一致。

        **延伸**：装箱用 First-Fit-Decreasing / Best-Fit-Decreasing（TRL 默认 \`bfd\`）；HF \`DataCollatorWithFlattening\` + FA2 报告吞吐最高约 2 倍；packing 会改变每个 batch 的样本数，因此按样本平均的 loss 会受影响，按 token 平均不受影响。
      `,
    },
    {
      id: 'qa-reduce',
      freq: 2,
      q: md`SFT loss 是按 token 平均还是按样本平均？梯度累积时要注意什么？`,
      a: md`
        **结论**：主流是全局按 token 平均；关键是在梯度累积和多卡下保持“全局”一致。

        **原因**：按 token 平均时每个 token 等权、长回复贡献更多梯度；按样本平均时每条样本等权，短回复的 token 被放大、模型偏好短回答。若每个 micro-batch 先内部平均再平均，等价于给不同 micro-batch 的 token 不同权重，结果依赖 batch 切分——HF 在 2024 年 10 月修复了 Trainer 中的这个 bug：改为对 loss 求和，再除以累积步内（并 all-reduce 到所有卡）的有效 token 总数。

        **延伸**：RL 中同样的问题叫 loss 聚合（seq-mean-token-mean vs token-mean），DAPO 改用 token 级平均以避免长回复中的 token 被稀释。
      `,
    },
    {
      id: 'qa-sft-vs-rl',
      freq: 2,
      q: md`有了 SFT，为什么还需要 RLHF / DPO？`,
      a: md`
        **结论**：SFT 只做模仿，学不到“什么是差回答”，也有 exposure bias；偏好对齐和 RL 提供了相对比较信号和 on-policy 的训练分布。

        **原因**：① SFT 只见过正例，无法表达“A 比 B 好”这类细粒度偏好；② teacher forcing 下前缀永远是标准答案，推理时却是自己的输出，错误会累积；③ 高质量示范比偏好比较贵得多，人更擅长判断而非写作；④ SFT 的上限受示范数据质量约束，RL 可以超越示范。

        **延伸**：LIMA 的“表面对齐假设”认为能力主要来自预训练、SFT 主要学格式；推理模型中 SFT 常作为 RL 的冷启动；on-policy 蒸馏是介于两者之间的方案。
      `,
    },
  ],

  cheatsheet: md`
    **模板**：ChatML \`<|im_start|>role⏎content<|im_end|>⏎\`（Qwen）· Llama 3 \`<|start_header_id|>role<|end_header_id|>⏎⏎content<|eot_id|>\`

    **labels**：非 assistant 回复处为 **-100**（\`ignore_index\` 默认值），**结束符要训练**；labels 不移位，模型内部 \`logits[:, :-1]\` ↔ \`labels[:, 1:]\`

    \`\`\`python
    head = [IM_START, ROLE[r], NL]; body = content + [IM_END]
    ids += head + body + [NL]
    labels += [-100]*len(head) + (body if train else [-100]*len(body)) + [-100]
    # packing（padding-free）
    labels += [-100] + lab[1:]            # 每条首个 label 屏蔽
    position_ids += list(range(len(ids)))  # 每条从 0 开始
    cu_seqlens.append(cu_seqlens[-1] + len(ids))
    \`\`\`

    - packing 三件事：position 重置、块对角掩码 / varlen、边界 label -100；只改 position_ids **不能**隔离注意力
    - token 平均（默认）vs 样本平均（偏好短回复）；梯度累积要 sum 后除以全局有效 token 数（HF 2024-10 修复）
    - Llama 2 SFT：27,540 条、lr 2e-5、batch 64、4096、2 epoch、zero-out prompt loss
    - NumPy 里 \`-100\` 当下标不会报错（取倒数第 100 个）——先 \`np.where(mask, lab, 0)\`
  `,
})
