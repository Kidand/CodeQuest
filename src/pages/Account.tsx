import { useRef, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router'
import { ArrowLeft, ArrowRight, Cloud, Download, RefreshCw, Upload } from 'lucide-react'
import { Button, Card, Segmented } from '@/components/ui'
import { useCloud, cloudStatusText, refreshCloud, flushCloud, captureCloudSnapshot, downloadCloudBackup, summarizeCloudSnapshot, resolveCloudConflict, localMigrationSources, parseCloudBackup, migrateCloudSnapshot, cloudRecoveryBackups, authenticateCloud, logoutCloud, CloudUnsavedChangesError } from '@/lib/cloud'
import type { CloudSnapshot } from '../../shared/cloud'
import { loginAccount, switchAccount, useAccount } from '@/lib/accounts'

const inputClass = 'mt-2 w-full rounded-xl border border-border bg-bg px-4 py-3 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20'

function CloudAccountForm() {
  const user = useCloud((s) => s.user)
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState(user?.username ?? '')
  const [password, setPassword] = useState('')
  const [repeat, setRepeat] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [unsaved, setUnsaved] = useState(false)
  async function submit(e?: FormEvent, allowUnsaved = false) {
    e?.preventDefault()
    if (busy) return
    setError('')
    if (mode === 'register' && password !== repeat) { setError('两次输入的密码不一致。'); return }
    setBusy(true)
    try {
      await authenticateCloud(mode, username, password, allowUnsaved)
      setPassword(''); setRepeat(''); setUnsaved(false)
    } catch (e) {
      if (e instanceof CloudUnsavedChangesError) setUnsaved(true)
      else setError((e as Error).message)
    } finally { setBusy(false) }
  }
  return <div className="max-w-lg border-t border-border pt-5">
    <Segmented value={mode} onChange={(next) => { setMode(next); setError(''); setUnsaved(false); setPassword(''); setRepeat('') }} options={[{ value: 'login', label: '登录网站账号' }, { value: 'register', label: '注册网站账号' }]} />
    <form data-testid="cloud-auth-form" onSubmit={(e) => void submit(e)} className="mt-5 space-y-4">
      <label className="block text-sm font-medium" htmlFor="cloud-username">用户名
        <input id="cloud-username" name="username" autoComplete="username" value={username} onChange={(e) => { setUsername(e.target.value); setUnsaved(false) }} required placeholder="2–24 个汉字、英文字母、数字、_ 或 -" className={inputClass} disabled={busy} aria-describedby="cloud-username-hint" />
      </label>
      <p id="cloud-username-hint" className="text-xs leading-6 text-muted">网站用户名不区分英文字母大小写。旧本机账号需要先注册网站账号，再迁移存档。</p>
      <label className="block text-sm font-medium" htmlFor="cloud-password">密码
        <input id="cloud-password" name="password" type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} value={password} onChange={(e) => { setPassword(e.target.value); setUnsaved(false) }} required placeholder="8–128 个字符" className={inputClass} disabled={busy} />
      </label>
      {mode === 'register' && <label className="block text-sm font-medium" htmlFor="cloud-repeat-password">确认密码
        <input id="cloud-repeat-password" name="password-confirmation" type="password" autoComplete="new-password" value={repeat} onChange={(e) => setRepeat(e.target.value)} required className={inputClass} disabled={busy} />
      </label>}
      {error && <p role="alert" className="rounded-xl bg-danger-soft p-3 text-sm leading-6 text-danger">{error}</p>}
      <Button type="submit" variant="primary" disabled={busy || unsaved} icon={<ArrowRight size={16} />} className="w-full">{busy ? '正在连接账号…' : mode === 'register' ? '注册并开启云存档' : '登录并继续学习'}</Button>
      {unsaved && <div className="space-y-3 rounded-xl border border-warning p-4" data-testid="auth-unsaved-confirmation">
        <p className="text-sm leading-6">当前账号还有未同步的进度，原件会保留在此设备。切换后，其他设备暂时看不到这些更改；请下载备份，之后登录原账号可继续同步。</p>
        <div className="flex flex-wrap gap-2"><Button type="button" size="sm" variant="secondary" onClick={() => downloadCloudBackup(captureCloudSnapshot(), 'before-login')}>先下载当前备份</Button><Button type="button" size="sm" disabled={busy} onClick={() => void submit(undefined, true)}>保留设备副本并继续登录</Button><Button type="button" size="sm" variant="ghost" disabled={busy} onClick={() => setUnsaved(false)}>取消</Button></div>
      </div>}
      <p className="text-xs leading-6 text-muted">账号与学习进度保存在云端。请妥善保存密码；当前暂不提供密码找回。</p>
    </form>
  </div>
}

function LocalAccountForm() {
  const navigate = useNavigate()
  const account = useAccount((s) => s.account)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  async function submit(e: FormEvent) {
    e.preventDefault()
    if (busy) return
    setBusy(true); setError('')
    try { await loginAccount(username, password); navigate('/profile', { replace: true }) }
    catch (e) { setError((e as Error).message); setBusy(false) }
  }
  return <div className="max-w-lg">
    {account && <p className="mb-4 text-sm text-muted">当前本机账号：{account.username}</p>}
    <form data-testid="local-auth-form" onSubmit={submit} className="space-y-4">
      <label className="block text-sm font-medium" htmlFor="local-username">旧本机用户名
        <input id="local-username" name="local-username" autoComplete="off" value={username} onChange={(e) => setUsername(e.target.value)} required className={inputClass} disabled={busy} />
      </label>
      <label className="block text-sm font-medium" htmlFor="local-password">旧本机密码
        <input id="local-password" name="local-password" type="password" autoComplete="off" value={password} onChange={(e) => setPassword(e.target.value)} required className={inputClass} disabled={busy} />
      </label>
      {error && <p role="alert" className="rounded-xl bg-danger-soft p-3 text-sm text-danger">{error}</p>}
      <div className="flex flex-wrap gap-2"><Button type="submit" variant="secondary" disabled={busy}>{busy ? '正在打开…' : '打开旧本机账号'}</Button><Button type="button" variant="ghost" disabled={busy} onClick={async () => { setBusy(true); try { await switchAccount(null); navigate('/', { replace: true }) } catch (e) { setError((e as Error).message); setBusy(false) } }}>以游客身份学习</Button></div>
    </form>
    <p className="mt-4 text-xs leading-6 text-muted">此入口只打开当前浏览器里的旧存档，不会登录网站账号。可以到「我的」导出备份，或返回此页注册网站账号后迁移。清理浏览器数据前请先备份。</p>
  </div>
}

export default function Account() {
  const cloud = useCloud()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [showAuth, setShowAuth] = useState(false)
  const [pending, setPending] = useState<{ label: string; snapshot: CloudSnapshot } | null>(null)
  const [resolve, setResolve] = useState<'local' | 'remote' | null>(null)
  const [logout, setLogout] = useState<'saved' | 'unsaved' | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const sources = cloud.user ? localMigrationSources() : []
  const recoveries = cloudRecoveryBackups()
  async function act(task: () => Promise<unknown>) {
    setBusy(true); setError(''); setNotice('')
    try { await task() } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  async function readFile(file?: File) {
    if (!file) return
    try {
      if (file.size > 20_000_000) throw new Error('备份文件过大，请选择小于 20 MB 的 JSON。')
      setPending({ label: file.name, snapshot: parseCloudBackup(await file.text()) }); setError('')
    } catch (e) { setError((e as Error).message) }
    if (fileRef.current) fileRef.current.value = ''
  }
  return <div className="mx-auto max-w-4xl space-y-6 py-3 md:py-8">
    <Link to="/profile" className="inline-flex items-center gap-2 text-sm text-muted hover:text-fg"><ArrowLeft size={16} />返回我的</Link>
    <div><span className="inline-flex items-center gap-2 text-xs font-semibold text-accent"><Cloud size={16} />云端学习存档</span><h1 className="mt-3 text-3xl font-bold tracking-tight">换一台设备，接着闯关。</h1><p className="mt-3 text-sm leading-7 text-muted">使用 CodeQuest 用户名和密码登录，学习进度、代码草稿、复习记录和模拟面试会自动同步。原来的本机存档和以前的云存档设备副本可以迁移，原件会保留。</p></div>
    <Card className="space-y-4 p-5 md:p-7">
      <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="font-semibold">{cloud.user ? `你好，${cloud.user.username || cloud.user.email || '学习者'}` : 'CodeQuest 网站账号'}</h2><p className="mt-1 text-sm text-muted" role="status">{cloudStatusText[cloud.status]}{cloud.updatedAt && cloud.status === 'saved' ? ` · ${new Date(cloud.updatedAt).toLocaleString('zh-CN')}` : ''}</p>{cloud.user?.email && <p className="mt-1 break-all text-xs text-subtle">{cloud.user.email}</p>}</div><Cloud className="text-accent" size={26} /></div>
      {cloud.error && <p role="alert" className="rounded-xl bg-warning-soft p-3 text-sm leading-6">{cloud.error}</p>}
      <div className="flex flex-wrap gap-2">
        {cloud.user && <><Button variant="secondary" disabled={busy} icon={<RefreshCw size={15} />} onClick={() => void act(async () => { await refreshCloud(); await flushCloud() })}>立即同步</Button><Button variant="ghost" disabled={busy} onClick={() => void act(async () => { setLogout(await flushCloud() ? 'saved' : 'unsaved') })}>退出云端账号</Button><Button variant="ghost" disabled={busy} onClick={() => setShowAuth((value) => !value)}>切换网站账号</Button><Button variant="ghost" icon={<Download size={15} />} onClick={() => downloadCloudBackup(captureCloudSnapshot(), 'device')}>下载此设备备份</Button></>}
      </div>
      {(!cloud.user || cloud.status === 'expired' || showAuth) && <CloudAccountForm />}
      {logout && <div className="space-y-3 rounded-xl border border-border p-4"><p className="text-sm leading-6">{logout === 'saved' ? '进度已经保存到云端，可以退出。' : '当前进度尚未同步成功。退出后仍会保留在此设备的账号存档中，其他设备暂时看不到这些更改。建议先下载备份；下次请登录同一账号继续同步。'}</p><div className="flex flex-wrap gap-2"><Button disabled={busy} onClick={() => void act(async () => { try { await logoutCloud(logout === 'unsaved'); setLogout(null) } catch (e) { if (e instanceof CloudUnsavedChangesError) setLogout('unsaved'); else throw e } })}>{logout === 'saved' ? '确认退出云端账号' : '保留本机存档并退出'}</Button><Button variant="ghost" onClick={() => setLogout(null)}>取消</Button></div></div>}
    </Card>
    {cloud.conflict && <Card className="space-y-4 border-warning p-5" data-testid="cloud-conflict">
      <h2 className="font-semibold">两台设备的存档发生冲突</h2><p className="text-sm leading-6 text-muted">你离线或学习期间，云端存档也发生了变化。请核对两个版本后选择要继续使用的版本；可分别下载备份，两个版本也会保留在此设备的恢复记录中。</p>
      <div className="grid gap-3 sm:grid-cols-2"><div className="rounded-xl bg-surface-2 p-4"><h3 className="text-sm font-semibold">此设备版本</h3><p className="my-2 text-xs leading-6 text-muted">{summarizeCloudSnapshot(captureCloudSnapshot())}</p><Button size="sm" variant="secondary" onClick={() => downloadCloudBackup(captureCloudSnapshot(), 'conflict-device')}>下载此设备版本</Button></div><div className="rounded-xl bg-surface-2 p-4"><h3 className="text-sm font-semibold">云端版本 · v{cloud.conflict.revision}</h3><p className="my-2 text-xs leading-6 text-muted">{summarizeCloudSnapshot(cloud.conflict.snapshot)}</p>{cloud.conflict.updatedAt && <p className="mb-2 text-xs text-subtle">{new Date(cloud.conflict.updatedAt).toLocaleString('zh-CN')}</p>}{cloud.conflict.snapshot && <Button size="sm" variant="secondary" onClick={() => downloadCloudBackup(cloud.conflict!.snapshot!, 'conflict-cloud')}>下载云端版本</Button>}</div></div>
      <div className="flex flex-wrap gap-2"><Button disabled={busy} variant="secondary" onClick={() => setResolve('remote')}>使用云端版本</Button><Button disabled={busy} variant="secondary" onClick={() => setResolve('local')}>使用此设备版本</Button></div>
      {resolve && <div className="space-y-3 rounded-xl border border-border p-4"><p className="text-sm">{resolve === 'remote' ? '确认用云端版本替换此设备的当前进度？' : '确认用此设备版本替换云端进度？其他设备下一次同步时会看到这个版本。'}</p><Button disabled={busy} onClick={() => void act(async () => { await resolveCloudConflict(resolve); setResolve(null) })}>确认选用{resolve === 'remote' ? '云端' : '此设备'}版本</Button><Button variant="ghost" onClick={() => setResolve(null)}>取消</Button></div>}
    </Card>}
    {cloud.user && <Card className="space-y-4 p-5 md:p-7"><div><h2 className="font-semibold">迁移本机存档</h2><p className="mt-2 text-sm leading-6 text-muted">选择游客、旧本机账号或以前的云存档设备副本，也可以导入 JSON 备份。迁移会整体替换当前云端存档，不会叠加经验值；本机原件保持不变。</p></div>
      <p className="text-xs leading-6 text-muted">以前的云存档只能从保存过副本的原设备或导出的 JSON 迁移。同名新账号不会自动取得旧账号的数据；此设备没有副本时，请回到原设备导出后再导入。</p>
      {sources.length > 0 ? <div className="space-y-3">{sources.map((source) => <div key={source.id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border p-4"><div><h3 className="text-sm font-semibold">{source.label}</h3><p className="mt-1 text-xs leading-6 text-muted">{source.error || summarizeCloudSnapshot(source.snapshot)}</p></div><Button size="sm" variant="secondary" disabled={busy || !source.snapshot || !!cloud.conflict} onClick={() => setPending({ label: source.label, snapshot: source.snapshot! })}>迁移此存档</Button></div>)}</div> : <p className="text-sm text-subtle">此浏览器没有可迁移的旧存档。请在原设备用网站账号登录后迁移，或在此导入已导出的 JSON。</p>}
      <Button icon={<Upload size={15} />} disabled={busy || !!cloud.conflict} variant="secondary" onClick={() => fileRef.current?.click()}>选择 JSON 备份</Button><input type="file" ref={fileRef} accept="application/json,.json" className="hidden" onChange={(e) => void readFile(e.target.files?.[0])} />
      {pending && <div className="space-y-3 rounded-xl bg-accent-soft p-4"><h3 className="text-sm font-semibold">确认迁移「{pending.label}」？</h3><p className="text-xs leading-6">要迁移的存档：{summarizeCloudSnapshot(pending.snapshot)}<br />当前云端账号的设备存档：{summarizeCloudSnapshot(captureCloudSnapshot())}</p><p className="text-sm leading-6">{cloud.revision > 0 ? '已有的云端进度将被此存档替换。' : '将把这个存档上传到你的云端账号。'}替换前的版本会保留为本机恢复备份。</p><div className="flex flex-wrap gap-2"><Button disabled={busy} onClick={() => void act(async () => { await migrateCloudSnapshot(pending.snapshot); setPending(null); setNotice(useCloud.getState().status === 'saved' ? '迁移完成，已保存到云端。可以在其他设备登录继续学习。' : '存档已载入此设备；请根据上方同步状态完成保存。') })}>{cloud.revision > 0 ? '确认替换云端存档' : '确认迁移到云端'}</Button><Button variant="ghost" onClick={() => setPending(null)}>取消</Button></div></div>}
    </Card>}
    {recoveries.length > 0 && <details className="rounded-2xl border border-border bg-surface p-5"><summary className="cursor-pointer text-sm font-semibold">恢复之前保留的版本</summary><p className="my-3 text-xs leading-6 text-muted">冲突处理和迁移前保留的备份，只存于此设备。下载后可通过「选择 JSON 备份」重新导入。</p><div className="space-y-3">{recoveries.map((item) => <div key={item.id} className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3"><div><p className="text-xs font-medium">{item.label}</p><p className="mt-1 text-xs text-muted">{summarizeCloudSnapshot(item.snapshot)}</p></div><Button size="sm" variant="secondary" onClick={() => downloadCloudBackup(item.snapshot, 'recovery')}>下载恢复备份</Button></div>)}</div></details>}
    {error && <p role="alert" className="rounded-xl bg-danger-soft p-4 text-sm text-danger">{error}</p>}{notice && <p role="status" className="rounded-xl bg-success-soft p-4 text-sm text-success">{notice}</p>}
    {!cloud.user && <details className="rounded-2xl border border-border bg-surface p-5"><summary className="cursor-pointer text-sm font-semibold">打开旧本机账号 / 使用游客存档</summary><p className="my-4 text-xs leading-6 text-muted">这里的账号只在此浏览器有效。云端登录后，可以从上面的迁移入口选择这份存档。</p><LocalAccountForm /></details>}
  </div>
}
