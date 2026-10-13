# CodeQuest 产品与技术设计

> 一句话：把「大模型算法工程师面试」做成一款闯关游戏——**看懂（可视化图解）→ 练熟（多题型随堂 + 代码跟练）→ 手撕（浏览器内跑测试的实战）→ 讲清（面试追问闪卡）**，并用间隔复习与模拟面试巩固。

---

## 1. 用户与目标

- **用户**：准备大模型算法岗（训练 / 后训练 / 推理 / 架构方向）面试的工程师。时间紧、目标明确，经常在手机上碎片化复习，也会在电脑上集中手撕。
- **核心目标**：
  1. 经典算法高频题能在 15–25 分钟内无 bug 写出；
  2. 大模型核心模块（Attention 家族、RoPE、MoE、DPO/GRPO、投机解码……）能手撕 NumPy/PyTorch 实现并讲清原理；
  3. 跟上 2025–2026 年主流技术（见 `SYLLABUS.md`），面试追问不掉链子。

## 2. 信息架构

```
首页 ─┬─ 继续上次 / 今日目标 / 连续打卡 / 今日复习 / 推荐关卡
      ├─ 经典算法地图（6 章 33 关）──┐
      ├─ 大模型地图（11 章 64 关）──┴─ 章节页 ─ 关卡页（沉浸式 5 阶段）
      ├─ 复习（错题本 + 面试闪卡，间隔重复）
      ├─ 模拟面试（限时：算法手撕 + 大模型手撕 + 八股）
      ├─ 速查手册（全站 cheatsheet 汇总，可搜索、可打印）
      └─ 我的（等级 / 统计 / 热力图 / 成就 / 设置 / 数据导入导出）
```

### 关卡的五个阶段（循序渐进）
| 阶段 | 目的 | 形式 |
|---|---|---|
| ① 图解 | 建立直觉，理解「为什么」 | 分节讲解 + 可交互可视化（逐帧 / 滑块 / 点击探索）+ 节末小测 |
| ② 随堂练习 | 检验理解，暴露误区 | 10 种题型：单选、多选、判断、填空、计算、排序、配对、代码拼图、找 Bug、点选 |
| ③ 跟练 | 从「看懂」到「会写」 | 代码骨架逐空填写，每空讲解；可跑测试判定等价写法 |
| ④ 实战 | 面试手撕 | CodeMirror 编辑器 + Pyodide 跑可见 / 隐藏测试；渐进提示；题解（含 PyTorch 版） |
| ⑤ 面试追问 | 能讲清楚 | 闪卡自评（会 / 模糊 / 不会），自动进入间隔复习 |

## 3. 游戏化

- **XP**（见 `src/lib/game.ts`）：读完一节 +4、图解完成 +20、节末小测 +3、随堂按「新增答对数」×5（首次全对另加 +20）、跟练 +30、实战 easy 40 / medium 70 / hard 110（每用一条提示 −10%，最低 50%；看过题解 ×0.5）、闪卡 +2、复习答对 +3、模拟面试完成 +50。**所有奖励只发首次或增量，防刷**。
- **玩家等级**：累计升级所需 XP 为 50·L·(L+1)；称号依次为 算法实习生 → 初级算法工程师 → 算法工程师 → 高级算法工程师 → 资深算法专家 → 大模型架构师 → 首席科学家。
- **星级**：★ 图解完成且随堂 ≥ 60%（通关，解锁下一关）；★★ 随堂 ≥ 80% 且跟练全部完成；★★★ 实战全部通过且通过前没看题解。
- **解锁**：各章第一关默认开放（面试准备要能按需跳转）；章内顺序解锁；BOSS 关在本章其它关全部通关后解锁。`prerequisites` 只作「建议先学」提示。**自由模式**（设置中开启）解锁全部关卡。
- **连续打卡、每日 XP 目标**（默认 100）、**成就徽章**、升级 / 三星 / 成就的庆祝动效（遵循「减少动效」设置）。
- **间隔复习**：答错的题与自评「模糊 / 不会」的闪卡进入复习池，按 1 → 2 → 5 → 11 → … 天的间隔复习，连续答对 3 次视为掌握。

## 4. 视觉与交互

- **设计 token**：`src/styles/index.css` 定义明暗两套语义色（bg / surface / fg / muted / accent / success / warning / danger / algo / llm / gold / viz-1..8 / heat）。组件只用语义类，**不硬编码颜色**。主题有浅色 / 深色 / 跟随系统三种，首屏前由内联脚本应用，避免闪烁。
- **两条路线的品牌色**：经典算法为青绿（`--algo`），大模型为紫（`--llm`）。
- **布局**：
  - 手机（<768）：顶部精简栏（logo、连续天数、XP）+ 底部 Tab（首页 / 地图 / 复习 / 我的），适配安全区；关卡页全屏沉浸，阶段导航放在顶部，主操作按钮固定在底部。
  - 平板 / 桌面：左侧导航栏（≥1024 时展开，平板只显示图标）；内容区最大宽度约 1100px；实战页左右分栏（题面 | 编辑器 + 结果）。
- **可视化内部用容器查询**（`@container`），不依赖视口断点，保证同一组件在不同栏宽下都好看。
- **可访问性**：键盘可操作（题目数字键选择、Enter 提交、← → 步进、空格播放）、焦点可见、`prefers-reduced-motion`、对比度达到 WCAG AA。
- **中文排版**：正文 15px、行高 1.75；代码用等宽字体；数字使用 tabular-nums。

## 5. 技术架构

| 层 | 选型 | 说明 |
|---|---|---|
| 构建 | Vite 8 + React 19 + TypeScript 7 | 静态站点，`base: './'` + HashRouter，可部署到任意静态托管 |
| 样式 | Tailwind CSS 4（`@theme inline` 映射 CSS 变量） | 明暗主题只改变量 |
| 路由 | react-router 8（`createHashRouter`） | 页面全部懒加载 |
| 状态 | zustand + persist（localStorage） | 游客使用 `cq-progress` / `cq-drafts` / `cq-mock`；账号使用 `::<id>` 后缀隔离，完整存档支持导出 / 导入 JSON |
| 内容 | TS 模块，`import.meta.glob` 自动发现章节，关卡正文懒加载 | 章节加载失败会被隔离，不影响其它章节 |
| 公式 / 文档 | react-markdown + remark-gfm + remark-math + rehype-katex | 自定义提示块 `[!TIP]` 等 |
| 代码高亮 | highlight.js（只读）/ CodeMirror 6（编辑） | 颜色走语义 token |
| Python 运行 | Pyodide 314（Python 3.14 + numpy）放在 Web Worker 中 | 先从站内 `public/pyodide/` 加载，失败后回退 jsDelivr / fastly；超时会终止 Worker 并重建，防止死循环卡死 |
| 动效 | motion | 支持减少动效 |

### 关键设计决策
1. **大模型手撕用 NumPy 而非 PyTorch**：浏览器中无法运行 PyTorch。NumPy 实现考察的是同样的维度推导与数值稳定性；每道题都在 `solutionAlt` 里附 PyTorch 写法，并在追问中对照 API。
2. **测试 harness 唯一**（`src/runner/harness.ts`）：浏览器 Worker 与 Node 校验脚本共用同一段 Python，保证「参考解在校验脚本里通过」等价于「同样代码在浏览器里通过」。
3. **内容即代码**：每关是一个 TS 模块，由类型系统 + `npm run validate` + `npm run verify:py` 三重把关（schema、KaTeX、反斜杠事故、参考解正确性、starter 不应通过、跟练等价写法）。
4. **跟练的等价判定**：先做字符串归一化比较，不匹配时把用户答案代入、后续空位填参考答案，跑测试判定。这样既不误杀等价写法，又能识别错误写法。
5. **离线友好**：Pyodide 自托管（约 16MB），首次进入实战阶段时后台预热。
6. **本机账号**：`cq-accounts` 保存用户名、随机盐和 600,000 轮 PBKDF2-SHA256 校验值，`sessionStorage.cq-session` 保存当前标签页账号。切换账号时暂停持久化、清空内存并重新加载目标命名空间，首个账号可认领游客存档。账号不提供云同步或密码找回，界面明确说明本机存储边界。
7. **异常恢复**：Python 初始化 60 秒上限，运行 10 秒上限；加载阶段可取消。逐测试回传结果，超时保留已完成的用例，致命错误自动重建 Worker。版本 2 备份包含草稿与面试记录，导入先完整校验，存储写入失败回滚。

## 6. 平台实现规格

### P1 平台-壳层
负责文件：`src/components/layout/*`、`src/components/game/*`、`src/pages/{Home,MapPage,ChapterPage,Review,Mock,Profile,Cheatsheet,NotFound}.tsx`、`src/lib/{achievements,unlock,mock,selectors}.ts`（新建）、`src/styles/app.css`（新建，在 AppLayout 中 import）。可以修改 `src/lib/store.ts` 与 `src/lib/game.ts`（P2 不改这两个文件），但**不得改动已有 action 的签名**。

- **AppLayout**：手机为顶栏 + 底部 Tab；平板为图标侧栏；桌面为展开侧栏（logo、导航、复习待办角标、底部迷你等级卡与主题切换）。挂载游戏事件浮层（消费 `useEvents`）：XP 浮层（合并短时间内的多次加分）、升级弹窗（confetti）、成就 toast、三星庆祝。全局 ⌘K / Ctrl+K 命令面板，可按标题、标签、代表题搜索关卡。
- **Home**：问候 + 继续学习卡（`lastVisited`）、今日 XP 目标环、连续打卡、今日待复习数、两条路线卡（进度、星数、分层覆盖率）、推荐关卡（优先未完成的必考 / 新必会关卡）、分层说明（必考 / 高频 / 进阶，基础 / 新必会 / 前沿）。
- **MapPage `/track/:track`**：Duolingo 式蜿蜒路径。每章一个区块（图标、标题、进度、分层分布）；关卡节点显示状态（锁定 / 可玩 / 当前 / 已完成 + 星级），BOSS 节点更大并带皇冠。点击节点弹出卡片：简介、tier / 频率徽章、代表题、`llmLink`、预计时长，以及「开始 / 继续 / 复习」按钮。顶部有按 tier 筛选的高亮 chips。
- **ChapterPage `/chapter/:id`**：章节导语，关卡列表（含每个阶段的完成度、星级、代表题），章节速记卡入口。
- **Review `/review`**：今日待复习（混合题目与闪卡，逐个作答：题目用 `QuestionView`，闪卡翻面后自评）、错题本（按路线 / 章节筛选，可单题重做）、已掌握。调用 `reviewResult(key, correct)`。
- **Mock `/mock`**：配置时长（30 / 45 / 60 分钟）、组成（默认 1 道算法实战 + 1 道大模型实战 + 6 道概念题）、范围（已解锁 / 全部）、tier 筛选；进行中有倒计时，实战使用 `ChallengeWorkspace mode="interview"`；结束后出报告（得分、各题用时、薄弱点、建议复习的关卡），发放 `XP.mockFinish`，历史记录存入独立的 persist store（`src/lib/mock.ts`）。
- **Profile `/profile`**：玩家卡（等级、称号、XP 进度）、统计（通关数、星数、实战数、正确率）、近 20 周活动热力图、各章进度、分层覆盖（例如「必考 12/18」）、成就墙、设置（主题、自由模式、每日目标、编辑器字号、减少动效）、数据（导出 / 导入 JSON；重置时用应用内确认框，**不要用 window.confirm / alert**）。
- **Cheatsheet `/cheatsheet`**：按路线 / 章节分组，展开章节时懒加载该章各关的 cheatsheet；支持搜索；有打印样式。
- **成就**（`src/lib/achievements.ts`）：初次通关、首个三星、首次实战、10 / 50 道实战、随堂全对、连续 3 / 7 / 30 天、完成一章、全部必考关通关、全部前沿关通关、首次模拟面试、零提示通过 hard 题等，至少 15 个。用一个监听 store 变化的 hook 检测解锁条件并调用 `unlockAchievement`。

### P2 平台-关卡
负责文件：`src/pages/LevelPage.tsx`、`src/components/level/*`、`src/components/quiz/*`（实现已冻结接口的 `QuestionView`）、`src/components/code/*`（实现已冻结接口的 `ChallengeWorkspace`，以及编辑器、结果面板）、`src/runner/*`（可修改 client / worker，但 harness 的 Python 语义不得改变）、`src/styles/level.css`（新建，在 LevelPage 中 import）。

- **LevelPage `/level/:levelId/:stage?`**：沉浸式。顶栏包含返回章节、关卡标题、tier 徽章、五阶段步进器（显示完成状态，可任意跳转）、速记卡抽屉按钮、主题切换。加载后调用 `registerLevelShape` 与 `visitLevel`。底部有「下一阶段」主按钮。
- **图解**：各节纵向排列成故事流（桌面左侧目录 + 阅读进度，手机顶部进度条）；可视化可以比正文更宽（约 1000px）；节末小测内联渲染 `QuestionView`；某节进入视口并读到末尾（或答完小测）后调用 `markSectionRead`，全部读完后调用 `completeLearn`。
- **随堂练习**：一次一题，有进度条；答对弹跳动效、答错抖动并显示解析；每题调用 `recordAnswer`；结束页显示得分、星级变化、XP、错题列表，可以重做（调用 `completeQuiz`）。支持键盘：数字键选择、Enter 提交 / 继续。
- **QuestionView**（10 种题型，**全部支持触屏点击操作**，拖拽只是增强）：
  - 单选 / 多选：选项卡片，带 A/B/C 键位提示；
  - 判断：两个大按钮；
  - 填空 / 计算：输入框，数值输入实时预览 `parseNumberExpr` 的解析结果，判定用 `approxEqual`；
  - 排序：上下移动按钮 + 拖拽（`motion` 的 Reorder）；
  - 配对：先点左列再点右列，配好的一对用同色标记；
  - 代码拼图：候选区 ↔ 答案区点选移动，答案区可以调整顺序，保留缩进；
  - 找 Bug：带行号的代码，点击行切换选中；
  - 点选：chips / 网格，带行列标签。
  - 提交后逐项标出对错，显示 Markdown 解析和「继续」按钮。order / match / parsons 按 seed 洗牌，单选 / 多选保持作者给定的顺序。
- **跟练**：代码骨架中的空位渲染为内联输入框（等宽字体），当前步骤高亮，旁边是步骤说明面板。判定流程：归一化后与 answer / accept 比较 → 不匹配就跑测试（第 k 步及之前用用户答案，之后用参考答案）→ 通过则提示「等价写法 ✓」，否则展示失败用例和提示。可以「揭晓答案」。全部完成后跑完整测试，调用 `completeGuided`，展示 outro。
- **实战 ChallengeWorkspace**：桌面左右分栏（左侧为题面 / 提示 / 题解标签页，右侧为编辑器 + 结果）；手机用「题目 / 代码 / 结果」三个标签。
  - 编辑器：CodeMirror 6 + Python 语言包，主题跟随 token，4 空格缩进，括号匹配，Ctrl / ⌘+Enter 运行，字号取自设置，草稿自动保存到 `useDrafts`；手机端编辑器上方加符号辅助栏（Tab ( ) [ ] : = _ , . ' "）。
  - 操作：运行（只跑可见测试）、提交（全部测试）、重置、提示（逐条揭示，调用 `takeChallengeHint`）、题解（应用内二次确认，调用 `viewChallengeSolution`；展示讲解、参考解、`solutionAlt` 标签页、复杂度、追问）。
  - 结果面板：逐个用例显示通过 / 失败、错误信息、stdout、耗时；隐藏用例显示为「隐藏用例 #n」；在编辑器中高亮 `errorLine`；显示 Pyodide 加载状态与进度；运行超时给出友好提示。
  - 计时器；通过时庆祝并调用 `passChallenge`；`mode="interview"` 时隐藏提示与题解。
- **面试追问**：闪卡翻面，自评后调用 `rateQA`。
- **通关结算**：星级动画、XP 汇总、下一关按钮、速记卡。
- **性能**：进入跟练 / 实战阶段（或图解阶段空闲时）预热 Pyodide；CodeMirror 懒加载。

## 7. 质量保障

- `npm run typecheck`：TypeScript 严格模式。
- `npm run validate`：内容 schema、题量与题型下限、KaTeX 可解析、反斜杠事故。
- `npm run verify:py`：参考解在 Pyodide 中通过全部测试；starter 不通过；跟练参考拼装通过、各 accept 变体通过、空模板不通过。
- `npm run shot`：无头 Chrome 截图（桌面 / 手机 × 浅色 / 深色），并收集 console 错误。
- 集成验收：在真实 Chrome 中走通完整闯关流程（双端、双主题）。

## 8. 部署

`npm run fetch:pyodide && npm run build` 后，`dist/` 是纯静态站点，可以直接放到 GitHub Pages / Vercel / Nginx。`public/pyodide/` 已在 .gitignore 中，构建前由脚本从 `node_modules/pyodide` 复制并下载 numpy wheel。
