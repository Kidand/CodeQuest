# 云端账号与学习存档

## 使用与迁移

1. 打开「我的 → 云端账号与存档」，选择「注册网站账号」，设置用户名和密码。在不同设备上使用同一个网站账号登录。
2. 第一次在原设备登录后，选择已有游客或本机账号存档，或选择导出的 JSON 备份。
3. 确认数据摘要和替换范围，再迁移；等待显示「已保存到云端」。本机原件不会被删除。
4. 在其他设备登录后即可继续。同步内容包括关卡阶段、XP、成就、复习池、草稿、设置、面试历史和进行中的面试。

旧本机账号密码仅用于打开本机存档；需要单独注册网站账号并迁移。站点保持公开，游客无需登录也能学习。以前第三方账号的云存档可从此设备保存的副本或 JSON 迁移；不能凭同名新账号领取旧身份的服务端记录。所有旧表和旧记录保留。

断网或会话过期时，当前设备保留待同步记录。不要在同步完成前清除浏览器数据。同步冲突会展示两个版本并提供下载和确认选择；迁移或冲突处理前的版本可以在「恢复之前保留的版本」下载。恢复记录只存在于当前设备。

## 身份与接口

认证由网站自行实现，不依赖 ChatGPT 登录或平台身份头。用户名经过 NFKC 规范化且英文字母不区分大小写，支持 2–24 个汉字、英文字母、数字、下划线和连字符。密码为 8–128 个字符，保留首尾空格；当前不提供密码找回。

密码用运行时原生 scrypt 加盐派生（N=16384、r=8、p=5、随机 16-byte 盐、32-byte 结果），只保存哈希和算法参数，使用常量时间比较。未知用户名也执行 KDF。每个 isolate 最多同时执行两次 KDF，D1 对来源 IP 和规范化用户名分别执行原子频率限制，并有界清理过期记录。

会话令牌为随机 32 bytes，D1 只保存 SHA-256 摘要，固定 30 天有效期。生产 Cookie 使用 `__Host-codequest_session`、Secure、HttpOnly、SameSite=Lax 和 Path=/，不设置 Domain；本地 loopback HTTP 使用独立 dev cookie。每次登录发新令牌，最多保留 20 个设备会话。退出立即吊销当前会话，不影响其他设备。

| 接口 | 用途 |
|---|---|
| `GET /api/cloud/session` | 返回 `{user, auth:'password'}` |
| `POST /api/auth/register` | 用 `{username,password}` 注册并建立会话 |
| `POST /api/auth/login` | 校验用户名密码并建立会话 |
| `POST /api/auth/logout` | 吊销当前会话并清除 Cookie |
| `GET /api/cloud/state` | 读取当前账号的存档、revision 与更新时间 |
| `PUT /api/cloud/state` | 提交 `{baseRevision, mutationId, snapshot}` |

未登录的 `state` 请求返回 401。`X-CodeQuest-User` 只检测客户端预期身份，不授予权限；也防止旧标签页注销另一标签页刚登录的新账号。所有写入要求同源 Origin 和 JSON 类型；拒绝跨源、非法结构和超量内容。身份和存档响应禁止缓存；密码、令牌和学习数据不进入日志或客户端持久化认证配置。

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

`drizzle/0000_certain_thena.sql` 及原始 metadata 从旧站历史逐字保留。`0001_wealthy_stephen_strange.sql` 新增云存档与幂等表；`0002_chemical_dracula.sql` 新增独立用户名账号、会话和频率限制表。两次增量都不删除旧表或记录。已应用的迁移不可重写；后续修改用 `npm run db:generate` 生成增量。

本地数据库与线上独立。第一次开发先 `npm ci && npm run fetch:pyodide && npm run build`，然后依次应用尚未应用的迁移（不要重复执行已有建表文件）：

```bash
node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js d1 execute DB --local --config dist/server/wrangler.json --persist-to .wrangler/state --file drizzle/0000_certain_thena.sql
node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js d1 execute DB --local --config dist/server/wrangler.json --persist-to .wrangler/state --file drizzle/0001_wealthy_stephen_strange.sql
node --import ./scripts/sites-env.mjs ./node_modules/wrangler/bin/wrangler.js d1 execute DB --local --config dist/server/wrangler.json --persist-to .wrangler/state --file drizzle/0002_chemical_dracula.sql
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

本地和生产使用相同的注册登录代码；`sites({mockAuth:false})` 只保留构建集成，关闭平台登录模拟。浏览器测试通过真实注册和 Cookie 建立身份。`.wrangler`、浏览器测试数据和恢复备份不提交。

## 验证

- `npm run test:cloud`：真实 Miniflare/D1 的身份隔离、CAS、幂等、验证、Origin、UTF-8 大存档测试。
- `npm run test:cloud:e2e`：本地双浏览器恢复、迁移、离线、冲突、登录过期、延迟响应与克隆标签竞态。
- `npm run test:platform`：原有本地存档、模拟面试、Python 运行与错误恢复回归。
- 生产发布后检查匿名请求和伪造身份头被拒。`CQ_PUBLIC_BASE=https://your-site npm run test:auth:live` 会创建一个随机 QA 账号，实测注册、登录、跨设备存档和注销；随后清空该账号的样例存档并吊销测试会话，不接触现有用户数据。
