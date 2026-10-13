# CodeQuest 课程大纲（2026 版）

> 面向：大模型算法工程师面试（手撕 + 八股 + 计算题）。
> 结构：路线 → 章节 → 关卡；每关 = 图解 → 随堂练习 → 跟练 → 实战 → 面试追问。
> 分层标注：经典算法按**面试频率**分 `必考 / 高频 / 进阶`；大模型按**知识代际**分 `基础必会 / 新必会 / 前沿`，并另标面试频率。
> 时间基准：2026-10。标「⚠️待核实」的内容在写作时必须查一手资料，查不到就在正文里注明不确定，**不得编造**。

---

## 一、分层总览

### 1.1 经典算法：按面试频率分级

频率依据：CodeTop 汇总的牛客面经统计（按公司/岗位筛选）＋近两年大模型岗面经的共识。具体排名每年会有波动，下面只分档，不排名次。

| 档位 | 定义 | 代表题 |
|---|---|---|
| **必考** | 大模型岗手撕环节出现率最高；写不出来基本挂 | LRU 缓存(146)、反转链表(206)/K 个一组翻转(25)、手撕快排(912)、第 K 大(215)、二分模板(704/34/35)、搜索旋转数组(33)、无重复字符最长子串(3)、三数之和(15)、接雨水(42)、最大子数组和(53)、LIS(300)、LCS(1143)/编辑距离(72)、层序遍历(102)、岛屿数量(200)、全排列/子集(46/78)、合并区间(56) |
| **高频** | 经常出现，或作为必考题的追问/变体 | 合并 K 个有序链表(23)、环形链表 II(142)、滑动窗口最大值(239)、最小覆盖子串(76)、单调栈(739/84)、和为 K 的子数组(560)、最近公共祖先(236)、二叉树最大路径和(124)、课程表/拓扑排序(207/210)、并查集(547/684)、Dijkstra(743)、Trie(208)、背包(416/322/518)、最长回文子串(5)、股票系列(121/122/188/309)、括号生成(22)、N 皇后(51)、跳跃游戏(55/45)、数据流中位数(295)、归并排序/逆序对 |
| **进阶** | 难度高或偏冷门，用于拉开差距，或与大模型岗位强相关 | 两个正序数组的中位数(4)、LFU(460)、戳气球(312)、正则匹配(10)、单词搜索 II(212)、**Viterbi 解码**、**BPE/WordPiece 分词**、**序列打包（装箱）** |

#### 经典算法在大模型岗位中的出现场景

这些场景会写进对应关卡的「大模型岗场景」一节，部分做成实战题。

| 算法 | 大模型场景 | 落地关卡 |
|---|---|---|
| 二分 / upper_bound | 逆 CDF 采样 `searchsorted`；显存预算下二分最大 batch / 上下文长度 | algo-search-binary（已完成）、algo-search-rotated |
| 堆 / Top-K / 快速选择 | top-k 采样、beam search 维护候选、MoE top-k 路由；`argpartition` 比全排序快 | algo-sort-heap、algo-sort-select |
| 哈希 + 双向链表（LRU） | KV cache / 前缀缓存的块淘汰 | algo-ds-lru |
| 前缀和 | 变长序列打包后的 `cu_seqlens`、token → 样本定位 | algo-ds-hash |
| 滑动窗口 | 滑动窗口注意力的感受野、n-gram 重复检测 | algo-search-window |
| Trie / Radix Tree | WordPiece 最长匹配分词、RadixAttention 前缀缓存、约束解码 | algo-graph-trie |
| 拓扑排序 | autograd 反向传播顺序、计算图调度 | algo-graph-bfs-dfs |
| 并查集 | 预训练数据近重复去重（MinHash 后合并连通分量） | algo-graph-uf-shortest |
| 编辑距离 DP | WER/CER 评测、文本对齐 | algo-dp-twoseq |
| Viterbi DP | Unigram 分词、HMM/CRF 序列标注；与 beam search 对比 | algo-dp-viterbi |
| 贪心装箱 | SFT 序列打包（First-Fit-Decreasing）减少 padding | algo-backtrack-greedy |

### 1.2 大模型算法：三层知识体系

| 层 | 定义 | 面试形态 | 内容 |
|---|---|---|---|
| **基础必会** | Transformer 时代的基本功，2023 年前已定型，人人都会考 | 手撕为主（NumPy/PyTorch），附推导 | softmax / 交叉熵 / 梯度、LayerNorm / RMSNorm、GELU / SwiGLU、反向传播、Adam / AdamW / 学习率调度、SDPA / MHA / 因果掩码、KV cache、RoPE、Decoder Block、参数量 / FLOPs / 显存估算、BPE、采样（温度 / top-k / top-p）、beam search、LoRA、量化基础、SFT loss mask、奖励模型、PPO / GAE、DP / ZeRO / 显存账本 |
| **新必会** | 2023–2025 年成为主流模型标配，现在面试默认你懂 | 手撕 + 原理追问 + 计算题 | GQA / MQA、**MLA**（含矩阵吸收）、**FlashAttention**（online softmax）、长上下文外推（PI / NTK / YaRN）、**MoE**（细粒度 + 共享专家、aux loss、无辅助损失偏置均衡）、**DPO** 及变体（IPO / SimPO / KTO / ORPO）、**RLVR + GRPO**、Dr.GRPO、**DAPO**、**GSPO**、CISPO、熵坍缩 / reward hacking / pass@k、**投机解码**（拒绝采样正确性）、Medusa / **EAGLE-3** / **MTP**、PagedAttention / 连续批处理 / 前缀缓存 / chunked prefill / PD 分离、FP8 训练、GPTQ / AWQ / SmoothQuant、TP / SP / PP / EP、**Muon**、滑动窗口 + attention sink、约束解码 |
| **前沿** | 近两年重点考察，2026 年的新进展；面试中用来区分候选人 | 原理 + 设计取舍 + 少量手撕 | **DeepSeek-V4**：CSA + HCA 压缩稀疏注意力、mHC 流形约束超连接、hash 路由、sqrt(softplus) 门控、FP4 QAT、GRPO + on-policy 蒸馏合并专家；**DSA**（lightning indexer + top-k）；**线性注意力 → DeltaNet → Gated DeltaNet / KDA** 与 3:1 混合架构（Qwen3-Next / Qwen3.5、Kimi Linear、Kimi K3）；**Attention Residuals**（Kimi K3 Block AttnRes）；超稀疏 MoE（Kimi K3：896 选 16，Quantile Balancing）；MXFP4 / NVFP4；**on-policy distillation**；agentic RL；P-EAGLE 并行草稿 |

#### 已核实的 2026 关键事实（供内容作者使用，引用时注明来源）

- **DeepSeek-V4**（arXiv 2606.19348，2026-06）：V4-Pro 共 1.6T 参数、激活 49B；V4-Flash 共 284B、激活 13B；上下文 1M。
  - 注意力：CSA（每 m 个 token 压缩为 1 个 KV，两路重叠序列加可学习 softmax 权重 → lightning indexer 打分 → top-k → 在压缩条目上做 MQA）＋ HCA（压缩率 m′ ≫ m、不重叠、在全部压缩条目上做稠密注意力），二者交错；两路都附带 n_win 个未压缩的近邻 token；Q 和压缩 KV 都做 RMSNorm；只对最后 64 维施加 partial RoPE；可学习的 attention sink logits。已按[官方报告 §4.2.1](https://arxiv.org/html/2606.19348v1#S4.SS2.SSS1)核实：m=4、m′=128、n_win=128；Flash 的 top-k=512、Pro=1024。CSA 权重按通道在双路重叠的 2m 个 token 上归一化。
  - mHC：残差流扩为 n_hc 路，混合矩阵约束为双随机矩阵（Birkhoff 多面体，谱范数 ≤ 1），用 Sinkhorn-Knopp（exp 后交替行列归一化，t_max=20）投影。
  - MoE：在 DeepSeekMoE 基础上把亲和度激活由 Sigmoid 改为 Sqrt(Softplus(·))；保留无辅助损失均衡，另加很小的序列级均衡损失；去掉路由目标节点数限制；前几层用 Hash 路由（按 token id 的固定哈希选专家）替代稠密 FFN。
  - 精度：KV cache 中 RoPE 维用 BF16、其余用 FP8；indexer 用 FP4；路由专家权重 FP4，并做 FP4 QAT。
  - 优化器：Muon（Nesterov、权重衰减、update RMS rescale；混合 Newton-Schulz 共 10 步，前 8 步系数 (3.4445, −4.7750, 2.0315)、后 2 步 (2, −1.5, 0.5)）；embedding / head / RMSNorm 等仍用 AdamW；**未使用 QK-Clip**。
  - 后训练：先分领域训练专家模型（SFT + 带领域奖励的 GRPO），再用 on-policy distillation（reverse KL）合并成统一模型。
  - 效率：1M 上下文时，V4-Pro 单 token FLOPs 为 V3.2 的 27%，KV cache 为 10%。
- **Qwen3.5**（2026-02）：397B-A17B，Gated DeltaNet 与 gated attention 约 3:1 混合（延续 Qwen3-Next）；输出门控有助于消除 attention sink 与 massive activation。
- **Kimi K3**（arXiv 2607.24653，2026-07）：2.8T 参数、激活 104B，上下文 1M。每个 block 为 3 层 KDA + 1 层 Gated MLA；Block Attention Residuals（在深度方向上对前序 block 输出做注意力，K3 用 8 个 block）；Stable LatentMoE（896 个路由专家、每 token 激活 16 个，配合 Quantile Balancing）。官方称整体 scaling 效率约为 K2 的 2.5 倍。AttnRes 本身来自另一篇论文，K3 的贡献是 block 变体与工程化。Quantile Balancing 的细节 ⚠️待核实。
- **GLM-5**：MLA + DSA；**Kimi K2.5**：MLA；**MiniMax-M2.5**：标准 MHA（全注意力）。
- **DeepSeek-V3.2**：DSA 稀疏注意力（lightning indexer 选 top-k token，k=2048 ⚠️待核实）。
- **RL**：GRPO（DeepSeekMath / R1）→ DAPO（clip-higher ε_low=0.2 / ε_high=0.28、动态采样、token 级 loss、超长奖励整形）、Dr.GRPO（去掉长度归一化与 std 归一化偏差）、GSPO（序列级重要性比与裁剪，对 MoE 更稳定）、CISPO（MiniMax-M1：裁剪 IS 权重并 stop-grad，保留所有 token 的梯度）、REINFORCE++（全局批归一化优势）。2026 年还出现 DHPO、EP-GRPO 等变体（只作了解）。
- **投机解码**：EAGLE-3 是常用基线；vLLM 的 P-EAGLE（2026-03）把 EAGLE 草稿改为并行生成；MTP 头可以直接作为草稿源，但第 2、3 个草稿位置的接受率明显下降。

---

## 二、章节与关卡明细

> 列说明：**tier** 填入 `LevelMeta.tier`；**freq** 填入 `LevelMeta.frequency`（大模型关卡必填）；**实战** 为最少必做的实战题（可多做）。
> 每章最后一关为 BOSS 综合关（`kind: 'boss'`）：一节简短复盘、≥10 道混合题、1–2 道较难的综合实战。
> **写作顺序**：先写 tier/频率高的关卡，**每写完一关、校验通过后再登记到 index.ts**，保证中途中断也有可用的成果。

### 经典算法路线（track: algo）

#### A1 `algo-search` 查找：二分与双指针（order 1）
| id | 标题 | tier | 代表题 | 要点 / 可视化 / 实战 |
|---|---|---|---|---|
| algo-search-binary | 二分查找模板 | 必考 | 704 / 34 / 35 | ✅ **已完成（样板关卡）** |
| algo-search-rotated | 旋转数组与二分答案 | 必考 | 33 / 153 / 162 / 69 / 875 / 1011 | 「哪半边有序」判定；把答案空间二分抽象为 check(x) 单调。可视化：旋转数组高亮有序半段；check(x) 的 F→T 曲线。实战：33、875（或 1011）。大模型场景：二分可放下的最大 batch |
| algo-search-twopointers | 双指针 | 必考 | 167 / 15 / 11 / 42 / 283 | 对撞指针的正确性（为何可以丢弃一侧）；三数之和去重；接雨水的双指针解法。可视化：对撞过程、接雨水逐列注水。实战：15、42 |
| algo-search-window | 滑动窗口 | 必考 | 3 / 76 / 438 / 209 | 「右扩左缩」模板与窗口内计数；何时可以用滑窗（单调性）。可视化：窗口伸缩与哈希计数。实战：3、76。大模型场景：滑窗注意力感受野、n-gram 重复检测 |
| algo-search-boss | BOSS：查找综合 | 进阶 | 4 两个正序数组的中位数 | 混合题 + 实战 LC4（分割线二分） |

#### A2 `algo-sort` 排序与选择（order 2）
| id | 标题 | tier | 代表题 | 要点 / 可视化 / 实战 |
|---|---|---|---|---|
| algo-sort-quick | 快速排序 | 必考 | 912 / 75 | Lomuto / Hoare / 三路划分；随机 pivot；最坏 O(n²) 的成因；稳定性。可视化：分区过程柱状图（ids 交换动画）。实战：原地快排（大量重复元素的隐藏测试逼出三路划分或随机化）、75 颜色分类 |
| algo-sort-merge | 归并排序 | 高频 | 912 / 148 / 剑指 51 | 自顶向下 / 自底向上；链表归并；逆序对计数。可视化：递归树 + 合并。实战：逆序对、排序链表 |
| algo-sort-heap | 堆与 Top-K | 必考 | 215 / 347 / 23 / 295 | sift up / down、O(n) 建堆、heapq 用法。可视化：数组 ↔ 树双视图。实战：手写 MinHeap 类、295 数据流中位数或 23 合并 K 个有序链表。大模型场景：top-k 采样 / beam / MoE 路由 |
| algo-sort-select | 快速选择与线性排序 | 必考 | 215 / 164 | quickselect 期望 O(n)；计数 / 基数 / 桶排序；排序算法复杂度与稳定性总表。实战：quickselect（隐藏测试为有序大数组，卡死固定 pivot）。大模型场景：`argpartition` |
| algo-sort-boss | BOSS：排序综合 | 高频 | — | 混合题 + 综合实战 |

#### A3 `algo-ds` 链表、栈与哈希（order 3）
| id | 标题 | tier | 代表题 | 要点 / 可视化 / 实战 |
|---|---|---|---|---|
| algo-ds-linkedlist | 链表基本功 | 必考 | 206 / 92 / 25 / 21 / 142 / 19 | 哑节点、三指针反转、快慢指针（环入口证明）。可视化：指针逐步改向（GraphView）。实战：25 K 个一组翻转、142（setup 提供 ListNode 与构造/转换辅助函数） |
| algo-ds-monostack | 单调栈与单调队列 | 高频 | 739 / 496 / 84 / 42 / 239 | 「下一个更大元素」模板；单调队列求窗口最值。可视化：栈的进出。实战：84、239 |
| algo-ds-hash | 哈希与前缀和 | 高频 | 1 / 49 / 128 / 560 | 前缀和 + 哈希计数；差分数组。实战：560、128；大模型实战：由序列长度构造 `cu_seqlens`，并把全局 token 下标映射到 (样本, 偏移) |
| algo-ds-lru | LRU 与 LFU 缓存 | 必考 | 146 / 460 | 哈希 + 双向链表实现 O(1)；OrderedDict；LFU 的频次桶。可视化：链表 + 哈希双视图。实战：LRUCache（禁止用 OrderedDict，检查源码）、LFU（hard）。大模型场景：KV / 前缀缓存块淘汰 |
| algo-ds-boss | BOSS：数据结构综合 | 高频 | — | 混合题 + 综合实战 |

#### A4 `algo-graph` 树与图（order 4）
| id | 标题 | tier | 代表题 | 要点 / 可视化 / 实战 |
|---|---|---|---|---|
| algo-graph-tree | 二叉树遍历 | 必考 | 94 / 144 / 102 / 103 / 104 / 226 | 递归与迭代（显式栈）统一写法、层序。可视化：TreeView + 访问顺序。实战：迭代中序、103 锯齿层序（setup 提供 TreeNode / 由列表建树） |
| algo-graph-tree-adv | 二叉树进阶 | 高频 | 236 / 124 / 105 / 297 / 98 / 543 | 后序「向上返回信息」框架。实战：124、297 |
| algo-graph-search | BFS/DFS 与拓扑排序 | 必考 | 200 / 994 / 207 / 210 | 网格搜索、多源 BFS、Kahn 算法与环检测。可视化：洪水填充、入度队列。实战：200、210；大模型实战：标量 mini-autograd（按反向拓扑序传播梯度） |
| algo-graph-uf | 并查集与最短路 | 高频 | 547 / 684 / 743 | 路径压缩 + 按秩合并；Dijkstra（堆）；Bellman-Ford / Floyd 简介。实战：UnionFind 类、743；大模型实战：近重复文档分组 |
| algo-graph-trie | Trie 与前缀匹配 | 高频 | 208 / 211 / 212 | 插入 / 查询 / 前缀统计。可视化：Trie 构建、最长匹配分词。实战：Trie 类；大模型实战：WordPiece 贪心最长匹配分词（`##` 续接、`[UNK]`）。大模型场景：RadixAttention |
| algo-graph-boss | BOSS：树与图综合 | 高频 | — | 混合题 + 综合实战 |

#### A5 `algo-dp` 动态规划（order 5）
| id | 标题 | tier | 代表题 | 要点 / 可视化 / 实战 |
|---|---|---|---|---|
| algo-dp-intro | DP 四步法 | 必考 | 70 / 198 / 213 / 62 / 64 / 91 | 状态 → 转移 → 初始化 → 顺序；记忆化 vs 递推；滚动数组。可视化：填表 + 依赖箭头。实战：213、91 |
| algo-dp-knapsack | 背包问题 | 高频 | 416 / 322 / 518 / 377 | 0/1 背包倒序、完全背包正序的原因；组合数与排列数的遍历顺序。可视化：二维表 → 一维滚动。实战：416、322 |
| algo-dp-lis | LIS 与子数组 | 必考 | 300 / 53 / 152 / 354 | O(n²) → O(n log n) 的 tails 数组（贪心 + 二分）；Kadane。实战：LIS O(n log n)（隐藏测试 n=10^5 限时）、152 |
| algo-dp-twoseq | LCS 与编辑距离 | 必考 | 1143 / 72 / 583 | 二维 DP 与路径回溯。可视化：编辑距离表 + 操作回放。实战：72（返回操作序列）；大模型实战：WER 计算 |
| algo-dp-interval | 区间与回文 DP | 高频 | 5 / 516 / 647 / 312 | 按区间长度递推；中心扩展。实战：5、312 |
| algo-dp-stock | 状态机 DP：股票 | 高频 | 121 / 122 / 188 / 309 / 714 | 状态机建模。可视化：状态转移图 + 表。实战：188、309 |
| algo-dp-viterbi | 解码 DP：Viterbi | 进阶 | HMM 解码 / Unigram 分词 | 网格图（trellis）上的最优路径；log 域；与 beam search（剪枝近似）对比。实战：NumPy Viterbi、Unigram 最大概率切分 |
| algo-dp-boss | BOSS：DP 综合 | 必考 | — | 混合题 + 综合实战 |

#### A6 `algo-backtrack` 回溯与贪心（order 6）
| id | 标题 | tier | 代表题 | 要点 / 可视化 / 实战 |
|---|---|---|---|---|
| algo-backtrack-template | 回溯模板 | 必考 | 78 / 90 / 46 / 47 / 39 / 40 | 「选 / 不选」与「枚举选哪个」；排序 + 同层去重。可视化：递归树。实战：47、40 |
| algo-backtrack-pruning | 剪枝与约束搜索 | 高频 | 22 / 51 / 79 / 131 | 可行性剪枝、标记数组。可视化：N 皇后棋盘。实战：51、79 |
| algo-backtrack-greedy | 贪心 | 高频 | 55 / 45 / 56 / 435 / 135 / 763 | 交换论证证明贪心正确性。实战：56；大模型实战：SFT 序列打包（FFD 装箱，测试校验合法性与箱数上界） |
| algo-backtrack-boss | BOSS：搜索与贪心综合 | 高频 | — | 混合题 + 综合实战 |

### 大模型路线（track: llm）

#### L1 `llm-basics` 张量与基础算子（order 1）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-basics-softmax | Softmax 与交叉熵 | 基础 | high | 减最大值、log-sum-exp、log_softmax；CE 的 ignore_index 与 label smoothing；梯度 p−y；SFT 的 loss mask。实战：稳定 softmax / log_softmax（任意 axis）、带 ignore_index 与 label smoothing 的 CE |
| llm-basics-norm | LayerNorm 与 RMSNorm | 基础 | high | 公式与可学习参数；Pre-LN vs Post-LN；RMSNorm 去掉均值；QK-Norm。实战：LN / RMSNorm 前向、LN 反向（hard） |
| llm-basics-ffn | 激活函数与 FFN | 基础 | high | GELU（tanh 近似）、SiLU、GLU 族、SwiGLU 与隐藏维取 8/3·d（向 64 / 128 对齐）。实战：SwiGLU FFN、FFN 参数量函数 |
| llm-basics-backprop | 手写反向传播 | 基础 | medium | 计算图、链式法则、Linear / MLP / softmax-CE 的反向、数值梯度检查。实战：两层 MLP 前向 + 反向 + 梯度检查 |
| llm-basics-optim | 优化器与学习率 | 基础 | high | SGD 动量、Adam 偏差修正、AdamW 解耦权重衰减、全局范数梯度裁剪、warmup + cosine、优化器状态占用显存。实战：AdamW step、clip_grad_norm、学习率调度 |
| llm-basics-muon | Muon 优化器 | 新必会 | medium | 对动量矩阵做 Newton-Schulz 正交化（五次多项式系数 3.4445 / −4.7750 / 2.0315）、只用于二维隐藏层矩阵、与 AdamW 的 update RMS 对齐、MuonClip / QK-Clip（Kimi K2）、DeepSeek-V4 的混合 NS（且不用 QK-Clip 的理由）。实战：newton_schulz 正交化、Muon 更新 |
| llm-basics-boss | BOSS：基础算子综合 | 基础 | high | 综合实战：手写小 MLP 训练若干步，loss 下降 |

#### L2 `llm-decode` 分词与解码（order 2）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-decode-bpe | BPE 分词 | 基础 | high | 训练（统计 pair → 合并）与编码（按 merge rank）；byte-level BPE；BPE / WordPiece / Unigram / SentencePiece 对比；词表大小的取舍。实战：BPE 训练、BPE 编码 |
| llm-decode-sampling | 采样策略 | 基础 | high | greedy / 温度 / top-k / top-p / min-p，以及它们的组合顺序。实战：top_k_top_p_filtering（NumPy）、完整的采样函数 |
| llm-decode-beam | Beam Search | 基础 | high | 算法流程、长度惩罚、EOS 处理、为何对话场景少用。实战：在给定转移矩阵上做 beam search |
| llm-decode-constrained | 重复惩罚与约束解码 | 新必会 | medium | repetition penalty（CTRL：正 logit 除以 θ、负 logit 乘以 θ）、presence / frequency penalty、no_repeat_ngram、logit bias、基于 FSM / 文法的结构化输出（JSON mode）。实战：各种惩罚、FSM 约束下的 token mask |
| llm-decode-boss | BOSS：解码综合 | 基础 | high | — |

#### L3 `llm-attention` 注意力机制（order 3）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-attention-sdpa | 缩放点积注意力 | 基础 | high | 为什么除以 √d（方差推导）、因果掩码与 padding 掩码。实战：带掩码的 SDPA |
| llm-attention-mha | 多头注意力 | 基础 | high | 拆头时 reshape / transpose 的维度推导、W_O、参数量。实战：MHA 前向（PyTorch 版作为 alt） |
| llm-attention-kvcache | KV Cache 与 MQA/GQA | 基础 | high | 增量解码；KV 显存公式 2·L·B·S·n_kv·d_h·bytes；repeat_kv。实战：GQA 前向、增量解码结果与全量前向一致 |
| llm-attention-mla | MLA 多头潜在注意力 | 新必会 | high | 低秩 KV 压缩（DeepSeek-V2/V3：d_c=512，decoupled RoPE d_h^R=64）、推理时的矩阵吸收、每 token 缓存 576 维 vs MHA。实战：MLA 前向，并验证吸收形式与原形式等价 |
| llm-attention-flash | FlashAttention | 新必会 | high | HBM vs SRAM、online softmax、分块（tiling）、IO 复杂度、反向重计算、FA2 / FA3 的改进点。实战：online softmax、分块注意力与朴素实现结果一致 |
| llm-attention-boss | BOSS：注意力综合 | 新必会 | high | 综合实战：带 KV cache 的 GQA 解码 |

#### L4 `llm-effattn` 高效注意力与长上下文（order 4）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-effattn-swa | 滑动窗口、Attention Sink 与门控注意力 | 新必会 | medium | SWA 与全局层交替（Gemma、gpt-oss 等）及跨层感受野；StreamingLLM 的 attention sink；可学习 sink logit；gated attention（Qwen）。实战：滑窗因果掩码 + 环形 KV 缓冲区、带 sink 的 softmax |
| llm-effattn-sparse | 稀疏注意力：NSA 与 DSA | 新必会 | medium | NSA 的三个分支（压缩 / 选择 / 滑窗）与门控；DSA 的 lightning indexer（ReLU 打分）+ top-k 选择。实战：indexer 打分 + top-k + 稀疏注意力 |
| llm-effattn-v4 | DeepSeek-V4：CSA 与 HCA | 前沿 | medium | 见第一节的已核实事实；压缩、选择、滑窗三者的分工；KV 大小估算。实战：带可学习权重的分块压缩、KV 显存对比计算 |
| llm-effattn-linear | 线性注意力与 Gated DeltaNet | 前沿 | medium | 核化线性注意力的递推形式；衰减门（RetNet / Mamba2）；delta rule；Gated DeltaNet；KDA（逐通道门控）；chunkwise 并行；3:1 混合设计（Qwen3-Next / 3.5、Kimi Linear、K3）。实战：递推形式 == 并行形式、gated delta rule 递推 |
| llm-effattn-boss | BOSS：高效注意力综合 | 前沿 | medium | — |

#### L5 `llm-arch` 位置编码与模型架构（order 5）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-arch-rope | RoPE 旋转位置编码 | 基础 | high | 推导与相对位置性质；rotate_half 与交错两种实现；base θ；partial RoPE。实战：apply_rope，并验证内积只依赖 m−n |
| llm-arch-longctx | 长上下文外推 | 新必会 | high | PI、NTK-aware、YaRN（按波长分段 + 温度）、ABF。实战：计算三种缩放下的频率 |
| llm-arch-block | Decoder Block 组装 | 基础 | high | LLaMA 式 Pre-Norm block、残差、tied embedding、final norm。实战：完整 block 前向（GQA + RoPE + SwiGLU） |
| llm-arch-compute | 参数量、FLOPs 与显存估算 | 基础 | high | 12·L·d² 的来源与 GQA / SwiGLU 下的修正；训练 6ND、推理每 token 2N；Chinchilla；MFU。实战：count_params(config)、训练时长估算 |
| llm-arch-residual | 残差流演进：mHC 与 Attention Residuals | 前沿 | medium | Hyper-Connections → mHC（双随机约束 + Sinkhorn）→ Attention Residuals（在深度方向做注意力，Block 变体）。实战：Sinkhorn-Knopp 投影、深度注意力聚合 |
| llm-arch-boss | BOSS：架构综合 | 基础 | high | 综合实战：迷你 GPT 前向 |

#### L6 `llm-moe` 混合专家（order 6）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-moe-basics | MoE 基础与手撕 | 新必会 | high | router → softmax → top-k →（归一化）→ 分发 → 加权合并；总参数 vs 激活参数。实战：MoE 层前向（NumPy） |
| llm-moe-balance | 负载均衡与容量 | 新必会 | high | Switch aux loss N·Σf·P、z-loss、capacity factor、token 丢弃、专家坍缩。实战：aux / z loss、按容量分发 |
| llm-moe-deepseek | DeepSeekMoE 与无辅助损失均衡 | 新必会 | high | 细粒度 + 共享专家；sigmoid 亲和度 + top-k 后归一化；偏置 b 只影响选择、不影响权重，更新为 b ← b + γ·sign(平均负载 − 负载)；节点受限路由；V3 配置为 256 个路由专家 + 1 个共享专家、top-8。实战：多步偏置更新后负载趋于均衡 |
| llm-moe-frontier | 2026 MoE 前沿 | 前沿 | medium | V4：sqrt(softplus)、hash 路由、序列级均衡损失、去掉节点限制；K3：896 选 16、Quantile Balancing（⚠️细节待核实）；激活比例趋势。实战：hash 路由层、激活比例计算 |
| llm-moe-boss | BOSS：MoE 综合 | 新必会 | high | — |

#### L7 `llm-quant` 微调与量化（order 7）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-quant-lora | LoRA 家族 | 基础 | high | (α/r)·BA；B 初始化为 0；merge；QLoRA（NF4、双重量化）；DoRA。实战：LoRA 前向与 merge 后等价 |
| llm-quant-basics | 量化基础 | 基础 | high | 对称 / 非对称；per-tensor / per-channel / per-group；量化误差。实战：分组 INT4 / INT8 量化与反量化 |
| llm-quant-methods | LLM 量化方法 | 新必会 | medium | 激活离群值、LLM.int8、SmoothQuant（迁移因子 α）、GPTQ（逐列误差补偿）、AWQ、KV 量化。实战：SmoothQuant 等价变换 |
| llm-quant-lowprec | 低精度格式：FP8 / MXFP4 / NVFP4 | 新必会 | medium | 浮点编码；BF16 vs FP16；E4M3 / E5M2；DeepSeek-V3 细粒度 FP8（1×128 / 128×128）；MXFP4（32 元素块，E8M0 尺度）；NVFP4（16 元素块，E4M3 尺度 + FP32 张量尺度）；FP4 QAT。实战：MXFP4 块量化 |
| llm-quant-boss | BOSS：量化综合 | 新必会 | medium | — |

#### L8 `llm-align` 后训练与偏好对齐（order 8）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-align-sft | SFT | 基础 | high | chat template、只在回复部分算 loss、多轮掩码、packing 的掩码与 position id。实战：构造 labels（-100）与打包后的 position_ids |
| llm-align-rm | 奖励模型 | 基础 | medium | Bradley-Terry 损失、PRM vs ORM、生成式 RM。实战：BT loss 与梯度 |
| llm-align-ppo | PPO 与 RLHF | 基础 | high | 四个模型、逐 token KL 惩罚、GAE、clip surrogate、value 裁剪。实战：GAE、PPO loss |
| llm-align-dpo | DPO | 新必会 | high | 从 KL 约束 RL 的闭式解推导到 DPO loss；用 mask 求序列 logp。实战：由 logits 计算 dpo_loss（附隐式奖励） |
| llm-align-variants | DPO 变体 | 新必会 | medium | IPO、SimPO（长度归一化 + margin，无 ref）、KTO、ORPO、online / iterative DPO。实战：SimPO / IPO / ORPO loss |
| llm-align-boss | BOSS：对齐综合 | 新必会 | high | — |

#### L9 `llm-rl` 推理强化学习（order 9）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-rl-grpo | RLVR 与 GRPO | 新必会 | high | 可验证奖励；组内优势 (r−mean)/std；token 级 clip 目标；k3 KL 估计；无 critic。实战：grpo_loss（含 mask） |
| llm-rl-drgrpo | GRPO 的偏差与 Dr.GRPO | 新必会 | high | 长度归一化偏差、std 偏差；几种 loss 聚合方式（seq-mean-token-mean / token-mean / seq-mean-token-sum）。实战：实现各种聚合方式并比较 |
| llm-rl-dapo | DAPO | 新必会 | high | clip-higher、动态采样、token 级 loss、超长惩罚、去掉 KL。实战：DAPO loss + 过滤 + 超长惩罚 |
| llm-rl-gspo | GSPO、CISPO 与 IS 修正 | 新必会 | medium | 序列级 ratio (π/π_old)^{1/|y|}；CISPO 裁剪 IS 权重；训推不一致与 TIS；REINFORCE++。实战：gspo_loss、cispo_loss |
| llm-rl-dynamics | RL 训练动力学 | 新必会 | medium | 熵坍缩、reward hacking、格式奖励、pass@k vs pass@1、长度增长、难度筛选。实战：pass@k 无偏估计（数值稳定）、token 熵 |
| llm-rl-opd | On-policy 蒸馏与 Agentic RL | 前沿 | medium | off-policy vs on-policy 蒸馏（reverse KL，DeepSeek-V4 合并专家）；多轮工具调用 RL、工具输出不算 loss、轨迹级奖励。实战：逐 token reverse KL、工具 token 掩码 |
| llm-rl-boss | BOSS：RL 综合 | 新必会 | high | — |

#### L10 `llm-infer` 推理加速与服务（order 10）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-infer-roofline | Prefill / Decode 与 Roofline | 基础 | high | 算术强度、decode 访存瓶颈、batch 的作用、TTFT / TPOT。实战：decode 吞吐估算、roofline 判定 |
| llm-infer-paged | PagedAttention 与连续批处理 | 新必会 | high | 显存碎片、block table、写时复制、迭代级调度、抢占。实战：块分配器（alloc / append / free / fork COW） |
| llm-infer-prefix | 前缀缓存、Chunked Prefill 与 PD 分离 | 新必会 | medium | RadixAttention（LRU 淘汰叶子）、vLLM 链式 block hash、chunked prefill、PD 分离与 KV 传输。实战：链式哈希前缀块缓存命中 |
| llm-infer-specdec | 投机解码原理 | 新必会 | high | 接受规则 min(1, p/q)、残差分布 norm(max(0, p−q))、正确性证明、期望接受长度 (1−α^{γ+1})/(1−α)。实战：验证步骤（给定 p、q、草稿和随机数） |
| llm-infer-eagle | Medusa、EAGLE-3 与 MTP | 新必会 | medium | Medusa 多头 + 树注意力；EAGLE 系列（特征层草稿、动态树、EAGLE-3 多层特征融合）；DeepSeek MTP 用作草稿；P-EAGLE。实战：由草稿树构造树注意力掩码、贪心树验证 |
| llm-infer-boss | BOSS：推理综合 | 新必会 | high | — |

#### L11 `llm-dist` 分布式训练（order 11）
| id | 标题 | tier | freq | 要点 / 实战 |
|---|---|---|---|---|
| llm-dist-memory | 显存账本与混合精度 | 基础 | high | 混合精度 Adam 每参数 16 字节；激活估算；选择性重计算；FP16 的 loss scaling。实战：显存估算器 |
| llm-dist-dp | 数据并行、All-Reduce 与 ZeRO | 基础 | high | ring all-reduce = reduce-scatter + all-gather，通信量 2(N−1)/N；ZeRO-1/2/3、FSDP。实战：用 NumPy 模拟 ring all-reduce、ZeRO 显存计算 |
| llm-dist-tp | 张量并行与序列并行 | 新必会 | medium | Megatron 列切 → 行切；注意力按头切；词表并行 CE；SP。实战：在虚拟设备上模拟 TP MLP，结果与单卡一致 |
| llm-dist-pp | 流水线并行 | 新必会 | medium | GPipe 气泡率 (p−1)/(m+p−1)、1F1B、interleaved、zero-bubble、DualPipe。实战：气泡率计算、生成 1F1B 调度 |
| llm-dist-ep-cp | 专家并行、上下文并行与并行策略 | 新必会 | medium | EP all-to-all、DeepEP；Ring Attention（KV 轮转 + online softmax）；Ulysses；并行配置取舍。实战：Ring Attention 模拟 |
| llm-dist-boss | BOSS：分布式综合 | 新必会 | high | — |

**合计**：经典算法 6 章 33 关，大模型 11 章 64 关，共 97 关。

---

## 三、实现分工（多 Agent 并行）

| 角色 | 负责 | 波次 |
|---|---|---|
| P1 平台-壳层 | 布局与导航、首页、闯关地图、章节页、复习（错题本 + 闪卡）、模拟面试、个人中心（统计、成就、设置、导入导出）、速查手册、游戏化反馈 | 第 1 波 |
| P2 平台-关卡 | 关卡页五阶段、10 种题型渲染器、跟练填空器、实战工作台（CodeMirror + Pyodide）、结算页 | 第 1 波 |
| C1 | algo-search（第 2–5 关）+ algo-sort | 第 1 波 |
| C2 | algo-dp | 第 1 波 |
| C3 | llm-basics | 第 1 波 |
| C4 | llm-attention | 第 1 波 |
| C5 | llm-moe | 第 1 波 |
| C6 | llm-align | 第 1 波 |
| C7 | llm-rl | 第 1 波 |
| C8 | llm-infer | 第 1 波 |
| C9 | algo-ds + algo-backtrack | 第 2 波 |
| C10 | algo-graph | 第 2 波 |
| C11 | llm-decode | 第 2 波 |
| C12 | llm-effattn | 第 2 波 |
| C13 | llm-arch | 第 2 波 |
| C14 | llm-quant | 第 2 波 |
| C15 | llm-dist | 第 2 波 |

进度记录在 `docs/PROGRESS.md`。
