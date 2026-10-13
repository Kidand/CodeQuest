# 内容写作规范（AUTHORING）

> 读者：负责编写章节内容的 Agent 与贡献者。
> **动手前先通读本文件，并完整阅读样板关卡** `src/content/chapters/algo-search/levels/binary.tsx` 及其 `viz/` 目录。你写出的每一关，在结构、密度和质量上都应达到样板水准。

---

## 0. 铁律

1. **只改自己负责的目录** `src/content/chapters/<你的章节>/`。不要修改 `src/viz`、`src/components`、`src/lib`、`src/content/types.ts`、`scripts/` 等共享文件（其它 Agent 在并行开发）。如果需要通用能力，先在本章的 `viz/` 里自行实现；如果发现共享代码有 bug，在最终汇报中写明。
2. **不得安装 npm 依赖**，不得改 `package.json`。
3. **不得 git commit / push**。
4. **事实准确**：数字、公式、论文名、模型配置必须正确。拿不准就用 WebSearch / WebFetch 查一手资料（arXiv、官方技术报告、官方博客）；查不到就在正文里写明「细节未公开 / 不确定」。**严禁编造**参数、实验数字或引用。2026 年的已核实事实见 `docs/SYLLABUS.md` 第一节。
5. **增量登记**：每写完一关，先跑校验，通过后再把它加入 `index.ts` 的 `levels`。不要先登记还没写完的关卡，这样中途中断也不会让整章加载失败。
6. 正文用**简体中文**；技术术语保留英文原文（如 KV cache、top-p、online softmax）；代码与注释用中文注释 + 英文标识符。

---

## 1. 目录与登记

```
src/content/chapters/<chapter-id>/
  index.ts            # defineChapter({...})，levels 中每关 load: () => import('./levels/<slug>')
  levels/<slug>.tsx   # export default defineLevel({...})
  viz/<Name>.tsx      # 本章可视化组件
  shared.ts           # （可选）本章公共 Python 片段、辅助函数
```

- 章节 id 与目录名一致；关卡 id 用 `docs/SYLLABUS.md` 中给定的 id（`<chapter-id>-<slug>`）。
- `LevelMeta` 必填 `tier`（经典算法填 must / high / advanced；大模型填 foundation / modern / frontier），大模型关卡必填 `frequency`；经典算法关卡填写 `problems`（代表题）与 `llmLink`（大模型岗场景，一句话，没有就省略）。
- 目录**自动发现**，无需改任何注册表。

## 2. 每关的最低要求（校验脚本会强制检查）

| 部分 | 普通关 | BOSS 关 |
|---|---|---|
| `learn` 图解 | ≥ 3 节、≥ 1 个可交互可视化（建议 2–4 个）、建议每节带 `checkpoint` | 1 节复盘即可 |
| `quiz` 随堂练习 | ≥ 8 题（建议 10–12），≥ 5 种题型，难度 1/2/3 都要有 | ≥ 10 题，覆盖全章 |
| `guided` 跟练 | ≥ 1 个，3–6 个空，≥ 2 个测试 | 可选 |
| `challenges` 实战 | ≥ 1 道（建议 2 道：一道核心手撕，一道变体或综合），≥ 4 个测试且含隐藏测试 | 1–2 道较难的综合题 |
| `interview` 面试追问 | ≥ 3 条 | 可选 |
| `cheatsheet` 速记卡 | 必须有 | 建议有 |

**深度要求**：图解要讲清**为什么**（推导、直觉、反例），不能只给结论；练习要考到**易错点**和**变体**，不能只有概念复述；大模型关卡要说明**这个技术解决了什么问题、代价是什么、在哪些真实模型里用了**。

## 3. 文本：`md` 与 `py` 模板标签（必须使用）

```ts
import { defineLevel, md, py } from '@/content/helpers'
body: md`公式 $\frac{QK^\top}{\sqrt{d_k}}$，代码 \`x @ W\``
solution: py`print("a\nb")`
```

- 普通 JS 模板字符串会吞掉反斜杠（`\frac` 中的 `\f` 变成换页符，`\times` 中的 `\t` 变成 TAB，`\nabla` 中的 `\n` 变成换行）。`md` / `py` 标签保留原样，并自动去掉公共缩进。
- 标签内只有两个转义：`\`` 表示反引号，`\${` 表示字面量 `${`。**不要在 md 中写出 `${`**（会被当成插值），例如 `${}^{2}` 要改写。
- 含反斜杠或数学公式的短字符串（选项、配对项）也用 md 标签：`options: [md`$\sqrt{d_k}$`, ...]`。
- 校验脚本会检测控制字符，并逐个用 KaTeX 解析公式。

### Markdown 能力
- GFM 表格、列表、代码块（```python）、行内代码。
- KaTeX：行内 `$...$`，块级 `$$...$$`。
- 提示块（GitHub Alerts 风格）：
  ```
  > [!TIP] 可选标题      技巧
  > [!NOTE]              说明
  > [!WARN]              易错点
  > [!INTERVIEW]         面试官视角（高频追问、加分回答）
  > [!NEW]               前沿进展（2025–2026 新技术）
  ```
- **表格里的竖线**：表格单元格中的行内代码或公式如果含 `|`（如 `<|im_start|>`、`|y|`），会被当成列分隔符。代码中写 `\|`，公式中用 `\lvert … \rvert` 或 `\mid`（validate 会报错）。
- **中文加粗**：`**…）**处` 这种「闭合 ** 前是全角标点、后面紧跟文字」的写法，按 CommonMark 规则不会加粗，页面会显示星号。可以把标点移到 ** 外面，或在 ** 后加空格（validate 会警告）。
- 开发预览页 `/#/dev/level/<id>` 的舞台较窄（约 736px），到不了 `@3xl`；调试双栏布局时请以正式页 `/#/level/<id>` 为准。
- 篇幅：每节 body 约 150–500 字，配公式或表格；长推导可分多节。

## 4. 可视化（viz）

### 4.1 原则
- **一图一个核心洞见**：每个可视化回答一个问题（例如「为什么 online softmax 可以分块」「GQA 省在哪里」）。标题写成结论句。
- **必须可交互**：至少满足其一——逐帧步进（算法过程）、参数滑块（温度 / 维度 / 窗口大小…）、点击探索（点 token 看注意力行）。纯静态图不算。
- 「图、码、字」三者同步：逐帧类建议右侧放 `CodeTrace` 高亮当前行，`caption` 写当前步骤说明。
- 数据规模要小而真实：数组 8–12 个元素，矩阵 ≤ 8×8，token ≤ 10 个；数值经过计算，不能乱填。

### 4.2 原语库 `@/viz`（优先组合使用）
| 组件 | 用途 | 关键 props |
|---|---|---|
| `VizFrame` | 外框（**必须使用**） | title, caption, controls, legend |
| `useStepper(frames, {intervalMs})` + `StepControls` | 逐帧播放 | frames 由纯函数预计算 |
| `CodeTrace` | 代码高亮当前行 | code, active, vars |
| `ArrayView` | 数组：格子或柱状 | values, ids（交换动画）, highlights, pointers, ranges, dim, mode |
| `MatrixView` | 矩阵 / 热力图 / DP 表 / 掩码 | data, rowLabels, colLabels, scale('seq'/'div'/'none'), highlight, masked, onCellClick |
| `ProbBars` | 概率分布 / 路由权重 | items: {label, p, tone, dim, note} |
| `TokenRow` | token 序列 | tokens: {text, tone, sub, dim, dashed} |
| `TreeView` | 树（自动布局） | root: {id, label, children, tone, sub, edgeLabel} |
| `GraphView` | 图（指定坐标） | nodes {id,label,x,y}, edges {from,to,label,directed,curve} |
| `Legend`, `Tex` | 图例、公式 | — |
| `toneColor / toneSoft / heatColor / divergingColor / catColor` | 取色 | VizTone: 'accent' 'success' 'warning' 'danger' 'info' 'muted' 'gold' 'algo' 'llm' 或 1–8 |
| `useElementWidth()` / `fitCell()` | 自绘 SVG 时按容器宽度布局 | — |

UI 控件来自 `@/components/ui`：`Button`、`Segmented`、`Slider`、`Switch`、`Badge`。

### 4.3 硬性规则
- **颜色只用 token**：Tailwind 语义类（`bg-surface`、`text-muted`、`border-border`、`text-accent`…）或 `var(--xxx)` / `toneColor()`。**禁止硬编码十六进制颜色**，否则暗色模式会坏。
- **响应式用容器查询**：VizFrame 的舞台区是 `@container`，内部布局用 `@md:` / `@xl:` / `@3xl:` 等容器断点（例如 `grid @3xl:grid-cols-2`），**不要**用 `md:` / `lg:` 这类视口断点。在 390px 宽的手机上不应出现必须横向滚动才能看懂的内容（代码块除外）。
- SVG 用 `viewBox` + `width="100%"`；文字 ≥ 10px。
- 横向溢出的容器不要用 `justify-center`（会裁掉左侧），改用子元素 `mx-auto`。
- 把 `ArrayView` / `MatrixView` 放进 flex 或 grid 子项时，包裹容器要加 `min-w-0`，否则自动缩放格子不生效，手机上会被裁掉。
- 尺寸约定：VizFrame 标题 ≤ 20 个汉字（会换行，但越短越好）；`ArrayView` 在手机上能容纳约 12 个元素（格子自动缩到最小 18px），更多元素请分段展示。
- `useStepper(frames)` 的 frames **必须 `useMemo` 且依赖稳定**：每次渲染都新建数组会不停重置到第 0 帧。
- 自绘 SVG 用固定 viewBox 缩放时，手机上的文字会被缩小到 10px 以下；请用 `useElementWidth()` 按实测宽度布局，或反向补偿字号（`GraphView` 已内置补偿）。
- 组件无 props（`viz: ComponentType`）；同一组件要复用不同配置时，导出包装：`export const FooHalf = () => <Foo mode="half" />`。
- 不得 import CSS；不得访问 `localStorage`；不得发网络请求；`Math.random` 只用于用户主动点击「随机」，初始状态必须确定。
- 计算量要小（每帧 < 5ms）；动画用 `motion/react`（已安装）或 CSS transition。

## 5. 随堂练习（quiz）

题型（`QUESTION_TYPE_LABEL`）：`single` 单选、`multi` 多选、`judge` 判断、`fill` 填空、`numeric` 计算、`order` 排序、`match` 配对、`parsons` 代码拼图、`bug` 找 Bug、`pick` 点选。字段说明见 `src/content/types.ts`。

- **每题必须有解析**：说清正确答案为什么对，**以及主要干扰项为什么错**。
- 干扰项要来自真实误区（例如把 GQA 的 KV 头数当成 Q 头数、忘了 0/1 背包要倒序），不能是一眼就能排除的凑数项。
- 计算题（`numeric`）：题干给齐全部条件与单位约定（GB 是 1e9 还是 2^30 要写明），`tolerance` 视舍入情况取 0–0.02。用户可以输入 `2^30`、`1.5e9`、`7*1024` 这类表达式。
- `fill`：每个空列出全部可接受写法（比较时会去空白、统一引号、忽略大小写）；数值空设 `numeric: true`。
- `parsons`：行自带缩进；干扰行 1–3 条，必须是「看起来合理」的错误写法；不能有完全相同的两行。
- `bug`：代码 8–20 行，bug 1–2 处，要是真实会犯的错误；解析中给出修正。
- `pick`：适合「哪些位置会被 mask」「top-p 保留哪些 token」「哪些专家被选中」之类的题；网格题用 `cols` 和 `rowLabels` / `colLabels`。
- 难度分布：约 30% 为 1、50% 为 2、20% 为 3（3 = 面试压轴难度）。
- 大模型关卡至少 1 道**计算题**（参数量、显存、FLOPs、接受率…）与 1 道**代码阅读或找 bug** 题。

## 6. 跟练（guided）

- 「代码骨架 + 逐空讲解」：把核心实现拆成 3–6 个关键空，每空一行或一个表达式，按实现顺序排列。
- `instruction` 说明这一步要写什么、为什么；`explain` 在答对后补充原理或易错点。
- `answer` 是参考答案；`accept` 列出常见的等价写法。字符串不匹配时，平台会把用户答案代入并**跑测试判断是否等价**，所以测试必须能区分对错。
- 模板中空位写作 `{{1}}`、`{{2}}`…，编号与 steps 对应、不可重复。同一行尽量只放一个空。
- 校验脚本会检查：参考答案拼装后能通过测试，每个 accept 变体都能通过，空位全填 `pass` 时不能通过。

## 7. 实战（challenges）

### 7.1 题面
- 写清函数签名（带类型标注）、输入输出约定（形状、dtype、轴）、示例、约束、复杂度要求。
- 大模型题统一用 **NumPy** 实现（浏览器里没有 PyTorch）。在 `solutionAlt` 中**必须**附 PyTorch 参考写法（面试时多用 PyTorch），标注为「PyTorch 版（仅展示）」。
- 运行环境：Pyodide（Python 3.14）+ numpy 2.x；标准库可用（math、collections、heapq、itertools、functools、bisect、random、typing、inspect、dataclasses）。

### 7.2 starter / solution
- `starter` 只给签名、docstring 和 TODO，**不能通过测试**，也**不能有死循环**（建议 `raise NotImplementedError` 或返回占位值）。
- `solution` 必须通过全部测试，代码风格是面试中值得写出来的样子：清晰、带关键注释、复杂度最优或接近最优。

### 7.3 测试
- ≥ 4 个测试，**可见测试与隐藏测试都要有**（建议隐藏 ≥ 40%）：示例、边界（空、单元素、全相同、极值）、随机对拍（固定种子，与 setup 里的 `_ref_xxx` 参考实现比较）、规模或复杂度检查。
- 断言消息要给出**期望值与实际值**：`assert got == exp, f"输入 {x}：期望 {exp}，得到 {got}"`；浮点用 `np.allclose(..., atol=1e-6)`，失败时打印最大误差。
- 形状检查放在数值检查之前：`assert out.shape == (B, T, D), f"形状应为 {(B,T,D)}，得到 {out.shape}"`。
- 每个测试在浏览器中应 < 1s；所有测试合计 < 3s（校验脚本超过 4s 会报警告）。复杂度检查可以用样板里的「访问计数 `_Probe`」或者「规模 + 时间」的写法。
- 需要禁止使用某个库函数时，用 `inspect.getsource(fn)` 检查源码（见样板）。
- **让超复杂度的写法快速失败，而不是拖到 10 秒超时**（超时会丢掉整次运行的所有结果）。可复用的写法（参考 `algo-sort` 各关 setup）：
  - 计数比较次数的包装元素（重载 `__lt__`，超出预算立即抛 AssertionError）
  - 计数访问次数的只读序列（样板中的 `_Probe`，超出预算立即失败）
  - 用 `tracemalloc` 检查额外内存
- **⚠️ Pyodide 链表上限**：释放一条超过约 2000 个普通对象节点的链表时，递归析构会撑爆 JS 栈，整个 Python 运行时直接崩溃（`RangeError: Maximum call stack size exceeded`）。链表、树这类自定义节点结构，测试规模一律 **≤ 1500 个节点**，并在测试的 `finally` 中逐个执行 `node.next = None` 断开链接。
- 辅助代码放在 `setup` 里（在用户代码之后执行），名称一律加 `_ref_` 或 `_` 前缀，避免与用户函数重名。
- 随机数一律 `np.random.default_rng(seed)` 或 `random.Random(seed)`。

### 7.4 题解与追问
- `explanation`：思路、关键推导、复杂度、常见错误。`hints` 写 2–4 条，从方向性提示逐步过渡到具体做法。
- `followUps`：面试官会接着问的 1–3 个问题，附参考答案。

## 8. 面试追问（interview）与速记卡（cheatsheet）

- `interview`：高频口头题，答案按「**结论 → 原因 → 延伸**」组织，150–400 字；`freq` 表示出现频率（1–3）。
- `cheatsheet`：一屏能看完。内容是核心公式、模板代码（≤ 15 行）、关键数字（如 DeepSeek-V3 的 MLA 维度）、易错点列表。会汇总进全站的「速查手册」。

## 9. 自检流程（每关都要做）

```bash
# 1) 类型检查（只看自己目录的报错）
npx tsc --noEmit 2>&1 | grep "chapters/<你的章节>"

# 2) 结构校验（schema、题型、KaTeX、反斜杠事故）
npm run validate -- <你的章节>

# 3) Python 正确性（参考解通过、starter 不通过、跟练拼装通过）
npm run verify:py -- <你的章节>

# 4) 截图自检可视化（各 Agent 使用自己分配的端口与缓存目录，避免冲突）
VITE_CACHE_DIR=node_modules/.vite-<你的章节> npx vite --port <你的端口> --strictPort   # 后台运行
node scripts/shot.mjs --url "http://localhost:<端口>/#/dev/level/<关卡id>?part=learn" --out .shots/<章节>/<关卡>.png --full
node scripts/shot.mjs --url "http://localhost:<端口>/#/dev/level/<关卡id>?section=<节id>" --out .shots/<章节>/m.png --mobile --dark --selector figure
#   --click 'button[aria-label="下一步"]' 可以点击步进；输出里的 [console.error] / [pageerror] 都要修掉
```

- 各 Agent 共用同一个 scratchpad：日志与临时文件名必须带上你的 Agent 前缀（如 `vite-c11.log`），临时脚本放在 scratchpad 中，不要放进 node_modules 或 scripts。
- 用 Read 工具**亲眼查看截图**：桌面浅色 + 手机暗色至少各看一次。检查文字是否溢出或被裁切、颜色在暗色下是否可读、图是否真的说明了问题。
- 预览页 `/#/dev/level/<id>` 会静态展开整关（含答案），可以用来通读检查。
- 收尾时跑一次全章 `validate` + `verify:py`，两者都必须 0 错误。
