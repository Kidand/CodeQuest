# CodeQuest · 大模型算法面试闯关

像玩游戏一样准备大模型算法工程师面试：**交互图解 → 随堂练习 → 代码跟练 → Python 实战 → 面试追问**。

公开站点：[CodeQuest](https://codequest-algorithm-academy.kadins.chatgpt.site)

## 学习内容

完整课程共 **17 章、97 关**，包含 **1,041 道练习、82 个跟练、168 道实战和 217 处交互可视化**。

- 经典算法：查找与双指针、排序与选择、链表栈与哈希、树与图、动态规划、回溯与贪心。
- 大模型：基础算子、分词与解码、Attention、高效注意力、RoPE 与架构、MoE、量化与 LoRA、偏好对齐、强化学习、推理优化、分布式训练。
- 每关配有练习解析、交互图解、跟练、可见与隐藏测试、面试闪卡及速记。大模型手撕使用 NumPy，附 PyTorch 参考实现。

## 平台功能

- 双路线地图、顺序解锁与自由模式、XP 等级、星级、连续打卡、成就。
- 十种练习题型、错题本与间隔复习、全站搜索、可打印速查手册。
- 30 / 45 / 60 分钟模拟面试、独立代码草稿、刷新恢复、评分报告。
- 浏览器 Web Worker 内运行 Python + NumPy：10 秒执行超时、保留已完成测试、自动恢复、加载阶段取消和超时保护。
- 手机 / 桌面、深色 / 浅色 / 系统主题、减少动效、键盘操作。
- ChatGPT 云端登录、跨设备自动同步、断网续学、冲突处理、游客与旧本机存档迁移、完整 JSON 导入导出。

登录同一个 ChatGPT 账号后，学习进度、草稿、复习记录、设置和模拟面试保存在云端。游客与旧本机账号仍可离线学习；到「我的 → 云端账号与存档」选择旧存档或 JSON 备份，即可迁移。迁移会先确认替换范围，并保留本机原件与恢复版本。

云端写入采用版本校验；两台设备同时修改时会显示两份存档，供下载和选择，不会静默覆盖。网络中断或登录过期时，未同步修改保存在当前设备，恢复连接后继续同步。学习记录按登录身份隔离，每份云存档上限 1.5 MB（UTF-8）；超限时会保留本地数据并提示精简草稿。

## 本地运行

建议使用 Node.js 22 或更新版本。

```bash
npm ci
npm run fetch:pyodide
npm run build
# 首次本地开发：按 docs/CLOUD.md 应用本地数据库迁移
npm run dev
```

Python 运行时优先从本站加载；自托管资源不可用时回退 CDN。`fetch:pyodide` 准备约 16 MB 的运行时与 NumPy，构建部署时必须先运行。

## 验证与构建

```bash
npm run typecheck
npm run validate
npm run verify:py
npm run build
```

真实 Chrome 回归（先启动 `npm run dev -- --host 127.0.0.1 --port 5173`）：

```bash
npm run test:cloud
npm run test:cloud:e2e
npm run test:platform
npm run test:level -- --base http://127.0.0.1:5173
```

这些回归覆盖云端账号隔离、并发与断网恢复、本地迁移、模拟面试和运行器异常恢复，以及完整闯关流程与真实判题。浏览器测试使用本机 Chrome，不会下载浏览器。

## 部署与维护

`dist/server` 是 Cloudflare Worker，`dist/client` 是课程与 Python 静态资源。云端版本需要 Sites 身份注入与 D1 数据库，不能只上传静态目录。Sites 项目标识与 DB 绑定保存在 `.openai/hosting.json`，再次发布应更新同一站点。原有 HashRouter 关卡链接保持有效。

- [产品与技术设计](docs/DESIGN.md)
- [完整课程大纲](docs/SYLLABUS.md)
- [内容写作规范](docs/AUTHORING.md)
- [实现与验收进度](docs/PROGRESS.md)
- [云端存档设计与开发](docs/CLOUD.md)

`node_modules`、`public/pyodide`、`dist` 和测试截图均不提交。Python 参考解与跟练由浏览器和校验脚本共用的 harness 执行；新增课程必须通过内容校验和 Python 测试。
