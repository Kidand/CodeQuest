# 云端账号与学习存档

## 使用与迁移

1. 打开「我的 → 云端账号与存档」，点击「使用 ChatGPT 登录云端」。在不同设备上使用同一个 ChatGPT 账号。
2. 第一次在原设备登录后，选择已有游客或本机账号存档，或选择导出的 JSON 备份。
3. 确认数据摘要和替换范围，再迁移；等待显示「已保存到云端」。本机原件不会被删除。
4. 在其他设备登录后即可继续。同步内容包括关卡阶段、XP、成就、复习池、草稿、设置、面试历史和进行中的面试。

旧本机账号密码仅用于打开本机存档；云端账号使用 Sites 的 ChatGPT 登录。站点保持公开，游客无需登录也能学习。

断网或会话过期时，当前设备保留待同步记录。不要在同步完成前清除浏览器数据。同步冲突会展示两个版本并提供下载和确认选择；迁移或冲突处理前的版本可以在「恢复之前保留的版本」下载。恢复记录只存在于当前设备。

## 身份与接口

生产认证由 Sites dispatch 处理，应用不实现独立 OAuth 或密码会话。服务端只使用可信的 `oai-authenticated-user-id` 作为数据所有者，姓名和 email 仅用于展示。不要将该 Worker 单独部署到允许任意调用者伪造这些身份头的入口。

| 接口 | 用途 |
|---|---|
| `GET /api/cloud/session` | 查询当前身份与平台登录/退出链接 |
| `GET /api/cloud/state` | 读取当前账号的存档、revision 与更新时间 |
| `PUT /api/cloud/state` | 提交 `{baseRevision, mutationId, snapshot}` |

`state` 接口未登录返回 401。`X-CodeQuest-User` 仅用于检测客户端预期身份与实际身份是否变化，不用于授予访问权。PUT 要求同源 Origin 和 JSON 类型；服务端拒绝跨源、非法结构、超量以及过时版本。所有身份和存档响应都禁止缓存，不写入包含身份或草稿的日志。

存档采用 v2 JSON 契约，定义于 `shared/cloud.ts`。UTF-8 编码后的上限为 1,500,000 bytes，给 D1 的单行容量留余量。服务端只接受明确允许的数据字段，不接受 store actions、原型键或客户端指定的所有者。

## 一致性与恢复

- D1 事务内以 revision 比较并更新。两个写入基于相同版本时只有一个成功，另一个返回 409 和新云版本。
- mutation ID、原始 baseRevision 和规范化内容 hash 共同识别重复提交。响应丢失时复用同一个请求，不重复写入；旧成功请求在更晚版本出现后仍返回冲突。
- 每用户只保存最近 128 条轻量幂等记录，不保存每次自动同步的整份历史副本。
- 客户端缓存以账号隔离；每个文档有独立 writer ID，复制标签页不会共用待同步日志。刷新使用恢复指针找回该标签的未同步版本。
- 自动保存串行执行；上传期间继续编辑会安排后续保存。旧 GET、旧账号的 PUT、旧本机密码回调均不能回退或污染当前存档。
- 应用云快照前先完成本地原子写入；存储额度失败会回滚。未被同一内容确认的继承日志不会被回收。

## 数据库与迁移

`.openai/hosting.json` 使用原项目 ID，声明 `d1: "DB"`。Sites 负责创建或重新绑定托管数据库，并在发布前应用 schema migrations。

`drizzle/0000_certain_thena.sql` 及原始 metadata 从旧站历史逐字保留。`0001_wealthy_stephen_strange.sql` 只新增 `cq_cloud_states` 和 `cq_cloud_mutations`，不删除旧表或记录。已应用的迁移不可重写；后续修改用 `npm run db:generate` 生成增量。

本地数据库与线上独立。第一次开发先 `npm ci && npm run fetch:pyodide && npm run build`，然后依次应用尚未应用的迁移（不要重复执行已有建表文件）：

```bash
node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js d1 execute DB --local --config dist/server/wrangler.json --persist-to .wrangler/state --file drizzle/0000_certain_thena.sql
node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js d1 execute DB --local --config dist/server/wrangler.json --persist-to .wrangler/state --file drizzle/0001_wealthy_stephen_strange.sql
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

本地使用官方 `sites()` 插件的 loopback 登录模拟；它会移除传入伪造身份头，通过本地 HttpOnly cookie 注入 `local_seedy`。模拟登录不会进入生产构建。`.wrangler`、浏览器测试数据和恢复备份不提交。

## 验证

- `npm run test:cloud`：真实 Miniflare/D1 的身份隔离、CAS、幂等、验证、Origin、UTF-8 大存档测试。
- `npm run test:cloud:e2e`：本地双浏览器恢复、迁移、离线、冲突、登录过期、延迟响应与克隆标签竞态。
- `npm run test:platform`：原有本地存档、模拟面试、Python 运行与错误恢复回归。
- 生产发布后用匿名请求检查拒绝伪造身份，用正常平台登录入口确认实际登录流程。
