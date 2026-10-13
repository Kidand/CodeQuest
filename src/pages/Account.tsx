import { useRef, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router'
import { ArrowLeft, ArrowRight, BookOpen, Code2, HardDrive, UserRound, Cloud, Download, RefreshCw, Upload } from 'lucide-react'
import { Button, Card, Segmented } from '@/components/ui'
import { useCloud, cloudStatusText, refreshCloud, flushCloud, captureCloudSnapshot, downloadCloudBackup, summarizeCloudSnapshot, resolveCloudConflict, localMigrationSources, parseCloudBackup, migrateCloudSnapshot, announceCloudAuthChange, cloudRecoveryBackups } from '@/lib/cloud'
import type { CloudSnapshot } from '../../shared/cloud'
import { canClaimGuest, loginAccount, registerAccount, switchAccount, useAccount } from '@/lib/accounts'

function LocalAccountForm() {
  const navigate = useNavigate()
  const account = useAccount((s) => s.account)
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [repeat, setRepeat] = useState('')
  const [claim, setClaim] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  let eligible = false
  try { eligible = canClaimGuest() } catch { /* Display actionable errors on submit; guest access stays available. */ }
  async function submit(e: FormEvent) {
    e.preventDefault()
    if (busy) return
    setError('')
    if (mode === 'register' && password !== repeat) { setError('两次输入的密码不一致。'); return }
    setBusy(true)
    try {
      if (mode === 'register') await registerAccount(username, password, claim && eligible)
      else await loginAccount(username, password)
      navigate('/profile', { replace: true })
    } catch (e) { setError((e as Error).message); setBusy(false) }
  }
  async function guest() {
    setBusy(true)
    try { await switchAccount(null); navigate('/', { replace: true }) }
    catch (e) { setError((e as Error).message); setBusy(false) }
  }
  const inputClass = 'mt-2 w-full rounded-xl border border-border bg-bg px-4 py-3 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20'
  return (
    <div className="mx-auto max-w-4xl py-3">
      <Link to="/profile" className="mb-8 inline-flex items-center gap-2 text-sm text-muted hover:text-fg"><ArrowLeft size={16} />返回我的</Link>
      <div className="grid items-start gap-8 lg:grid-cols-2 lg:gap-12">
        <div className="pt-4">
          <span className="inline-flex items-center gap-2 rounded-full bg-accent-soft px-3 py-1.5 text-xs font-semibold text-accent"><UserRound size={14} />你的专属学习存档</span>
          <h1 className="mt-5 text-3xl font-bold leading-tight tracking-tight md:text-4xl">每一步进步，<br />都有迹可循。</h1>
          <p className="mt-4 text-sm leading-7 text-muted">从第一道二分，到手写 Attention。建立本机账号，保存你的星级、错题和实战代码，下次回来接着闯关。</p>
          <div className="mt-8 space-y-4 text-sm">
            <p className="flex items-center gap-3"><BookOpen size={18} className="text-algo" />独立进度、成就与间隔复习</p>
            <p className="flex items-center gap-3"><Code2 size={18} className="text-llm" />自动保存代码草稿与面试记录</p>
            <p className="flex items-center gap-3"><HardDrive size={18} className="text-accent" />导出备份，随时迁移学习存档</p>
          </div>
        </div>
        <Card className="p-5 md:p-7">
          {account && <p className="mb-4 rounded-xl bg-success-soft p-3 text-sm text-success">当前账号：{account.username}。登录后将切换学习存档。</p>}
          <Segmented value={mode} onChange={(m) => { setMode(m); setError('') }} options={[{ value: 'login', label: '登录' }, { value: 'register', label: '注册本机账号' }]} />
          <form onSubmit={submit} className="mt-6 space-y-4">
            <label className="block text-sm font-medium" htmlFor="username">用户名
              <input id="username" name="username" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required minLength={2} maxLength={24} placeholder="给你的冒险取个名字" className={inputClass} disabled={busy} />
            </label>
            <label className="block text-sm font-medium" htmlFor="password">密码
              <input id="password" name="password" type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} maxLength={128} placeholder="至少 8 个字符" className={inputClass} disabled={busy} />
            </label>
            {mode === 'register' && <>
              <label className="block text-sm font-medium" htmlFor="repeat-password">确认密码
                <input id="repeat-password" type="password" autoComplete="new-password" value={repeat} onChange={(e) => setRepeat(e.target.value)} required minLength={8} maxLength={128} className={inputClass} disabled={busy} />
              </label>
              {eligible && <label className="flex items-start gap-2 text-sm leading-6"><input type="checkbox" checked={claim} onChange={(e) => setClaim(e.target.checked)} className="mt-1.5 accent-accent" />将游客进度复制到此本机账号（保留原存档）</label>}
            </>}
            {error && <p role="alert" className="rounded-xl bg-danger-soft p-3 text-sm text-danger">{error}</p>}
            <Button type="submit" variant="primary" className="w-full" disabled={busy} icon={<ArrowRight size={16} />}>{busy ? '正在保存…' : mode === 'login' ? '登录并继续学习' : '创建账号，开始闯关'}</Button>
            <Button type="button" variant="ghost" className="w-full" disabled={busy} onClick={guest}>以游客身份学习</Button>
          </form>
          <p className="mt-5 border-t border-border pt-4 text-xs leading-6 text-muted">这是本机账号，数据仅保存在当前浏览器，不会自动同步到云端。清理浏览器数据会删除账号与存档，换设备前请到「我的」导出备份。本地密码用于区分存档，不加密学习数据，也不提供找回服务。</p>
        </Card>
      </div>
    </div>
  )
}

export default function Account() {
  const cloud = useCloud()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
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
  const linkClass = 'inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-accent-fg hover:opacity-90'
  return <div className="mx-auto max-w-4xl space-y-6 py-3 md:py-8">
    <Link to="/profile" className="inline-flex items-center gap-2 text-sm text-muted hover:text-fg"><ArrowLeft size={16} />返回我的</Link>
    <div><span className="inline-flex items-center gap-2 text-xs font-semibold text-accent"><Cloud size={16} />云端学习存档</span><h1 className="mt-3 text-3xl font-bold tracking-tight">换一台设备，接着闯关。</h1><p className="mt-3 text-sm leading-7 text-muted">使用同一个 ChatGPT 账号登录，学习进度、代码草稿、复习记录和模拟面试会自动同步。原来的本机存档可以迁移，原件会保留。</p></div>
    <Card className="space-y-4 p-5 md:p-7">
      <div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="font-semibold">{cloud.user ? `你好，${cloud.user.username || cloud.user.email || '学习者'}` : '登录云端账号'}</h2><p className="mt-1 text-sm text-muted" role="status">{cloudStatusText[cloud.status]}{cloud.updatedAt && cloud.status === 'saved' ? ` · ${new Date(cloud.updatedAt).toLocaleString('zh-CN')}` : ''}</p>{cloud.user?.email && <p className="mt-1 break-all text-xs text-subtle">{cloud.user.email}</p>}</div><Cloud className="text-accent" size={26} /></div>
      {cloud.error && <p role="alert" className="rounded-xl bg-warning-soft p-3 text-sm leading-6">{cloud.error}</p>}
      <div className="flex flex-wrap gap-2">
        {(!cloud.user || cloud.status === 'expired') && <a href={cloud.signInUrl} target="_top" className={linkClass} onClick={announceCloudAuthChange}>{cloud.status === 'expired' ? '重新登录云端账号' : '使用 ChatGPT 登录云端'}</a>}
        {cloud.user && <><Button variant="secondary" disabled={busy} icon={<RefreshCw size={15} />} onClick={() => void act(async () => { await refreshCloud(); await flushCloud() })}>立即同步</Button><Button variant="ghost" disabled={busy} onClick={() => void act(async () => { setLogout(await flushCloud() ? 'saved' : 'unsaved') })}>退出云端账号</Button><Button variant="ghost" icon={<Download size={15} />} onClick={() => downloadCloudBackup(captureCloudSnapshot(), 'device')}>下载此设备备份</Button></>}
      </div>
      {!cloud.user && <p className="text-xs leading-6 text-muted">云端账号使用 ChatGPT 登录身份。本机用户名与密码仍可用于打开旧存档；在原设备登录云端后，再选择需要迁移的存档。</p>}
      {logout && <div className="space-y-3 rounded-xl border border-border p-4"><p className="text-sm leading-6">{logout === 'saved' ? '进度已经保存到云端，可以退出。' : '当前进度尚未同步成功。退出后仍会保留在此设备的账号存档中，其他设备暂时看不到这些更改。建议先下载备份；下次请登录同一账号继续同步。'}</p><div className="flex flex-wrap gap-2"><a href={cloud.signOutUrl} target="_top" className={linkClass} onClick={() => { try { localStorage.removeItem('cq-cloud-last-user') } catch {} announceCloudAuthChange() }}>{logout === 'saved' ? '确认退出云端账号' : '保留本机存档并退出'}</a><Button variant="ghost" onClick={() => setLogout(null)}>取消</Button></div></div>}
    </Card>
    {cloud.conflict && <Card className="space-y-4 border-warning p-5" data-testid="cloud-conflict">
      <h2 className="font-semibold">两台设备的存档发生冲突</h2><p className="text-sm leading-6 text-muted">你离线或学习期间，云端存档也发生了变化。请核对两个版本后选择要继续使用的版本；可分别下载备份，两个版本也会保留在此设备的恢复记录中。</p>
      <div className="grid gap-3 sm:grid-cols-2"><div className="rounded-xl bg-surface-2 p-4"><h3 className="text-sm font-semibold">此设备版本</h3><p className="my-2 text-xs leading-6 text-muted">{summarizeCloudSnapshot(captureCloudSnapshot())}</p><Button size="sm" variant="secondary" onClick={() => downloadCloudBackup(captureCloudSnapshot(), 'conflict-device')}>下载此设备版本</Button></div><div className="rounded-xl bg-surface-2 p-4"><h3 className="text-sm font-semibold">云端版本 · v{cloud.conflict.revision}</h3><p className="my-2 text-xs leading-6 text-muted">{summarizeCloudSnapshot(cloud.conflict.snapshot)}</p>{cloud.conflict.updatedAt && <p className="mb-2 text-xs text-subtle">{new Date(cloud.conflict.updatedAt).toLocaleString('zh-CN')}</p>}{cloud.conflict.snapshot && <Button size="sm" variant="secondary" onClick={() => downloadCloudBackup(cloud.conflict!.snapshot!, 'conflict-cloud')}>下载云端版本</Button>}</div></div>
      <div className="flex flex-wrap gap-2"><Button disabled={busy} variant="secondary" onClick={() => setResolve('remote')}>使用云端版本</Button><Button disabled={busy} variant="secondary" onClick={() => setResolve('local')}>使用此设备版本</Button></div>
      {resolve && <div className="space-y-3 rounded-xl border border-border p-4"><p className="text-sm">{resolve === 'remote' ? '确认用云端版本替换此设备的当前进度？' : '确认用此设备版本替换云端进度？其他设备下一次同步时会看到这个版本。'}</p><Button disabled={busy} onClick={() => void act(async () => { await resolveCloudConflict(resolve); setResolve(null) })}>确认选用{resolve === 'remote' ? '云端' : '此设备'}版本</Button><Button variant="ghost" onClick={() => setResolve(null)}>取消</Button></div>}
    </Card>}
    {cloud.user && <Card className="space-y-4 p-5 md:p-7"><div><h2 className="font-semibold">迁移本机存档</h2><p className="mt-2 text-sm leading-6 text-muted">选择游客或旧本机账号存档，也可以导入 JSON 备份。迁移会整体替换当前云端存档，不会叠加经验值；本机原件保持不变。</p></div>
      {sources.length > 0 ? <div className="space-y-3">{sources.map((source) => <div key={source.id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border p-4"><div><h3 className="text-sm font-semibold">{source.label}</h3><p className="mt-1 text-xs leading-6 text-muted">{source.error || summarizeCloudSnapshot(source.snapshot)}</p></div><Button size="sm" variant="secondary" disabled={busy || !source.snapshot || !!cloud.conflict} onClick={() => setPending({ label: source.label, snapshot: source.snapshot! })}>迁移此存档</Button></div>)}</div> : <p className="text-sm text-subtle">此浏览器没有旧本机存档。请在原设备登录后迁移，或在此导入已导出的 JSON。</p>}
      <Button icon={<Upload size={15} />} disabled={busy || !!cloud.conflict} variant="secondary" onClick={() => fileRef.current?.click()}>选择 JSON 备份</Button><input type="file" ref={fileRef} accept="application/json,.json" className="hidden" onChange={(e) => void readFile(e.target.files?.[0])} />
      {pending && <div className="space-y-3 rounded-xl bg-accent-soft p-4"><h3 className="text-sm font-semibold">确认迁移「{pending.label}」？</h3><p className="text-xs leading-6">要迁移的存档：{summarizeCloudSnapshot(pending.snapshot)}<br />当前云端账号的设备存档：{summarizeCloudSnapshot(captureCloudSnapshot())}</p><p className="text-sm leading-6">{cloud.revision > 0 ? '已有的云端进度将被此存档替换。' : '将把这个存档上传到你的云端账号。'}替换前的版本会保留为本机恢复备份。</p><div className="flex flex-wrap gap-2"><Button disabled={busy} onClick={() => void act(async () => { await migrateCloudSnapshot(pending.snapshot); setPending(null); setNotice(useCloud.getState().status === 'saved' ? '迁移完成，已保存到云端。可以在其他设备登录继续学习。' : '存档已载入此设备；请根据上方同步状态完成保存。') })}>{cloud.revision > 0 ? '确认替换云端存档' : '确认迁移到云端'}</Button><Button variant="ghost" onClick={() => setPending(null)}>取消</Button></div></div>}
    </Card>}
    {recoveries.length > 0 && <details className="rounded-2xl border border-border bg-surface p-5"><summary className="cursor-pointer text-sm font-semibold">恢复之前保留的版本</summary><p className="my-3 text-xs leading-6 text-muted">冲突处理和迁移前保留的备份，只存于此设备。下载后可通过「选择 JSON 备份」重新导入。</p><div className="space-y-3">{recoveries.map((item) => <div key={item.id} className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3"><div><p className="text-xs font-medium">{item.label}</p><p className="mt-1 text-xs text-muted">{summarizeCloudSnapshot(item.snapshot)}</p></div><Button size="sm" variant="secondary" onClick={() => downloadCloudBackup(item.snapshot, 'recovery')}>下载恢复备份</Button></div>)}</div></details>}
    {error && <p role="alert" className="rounded-xl bg-danger-soft p-4 text-sm text-danger">{error}</p>}{notice && <p role="status" className="rounded-xl bg-success-soft p-4 text-sm text-success">{notice}</p>}
    {!cloud.user && <details className="rounded-2xl border border-border bg-surface p-5"><summary className="cursor-pointer text-sm font-semibold">打开旧本机账号 / 使用游客存档</summary><p className="my-4 text-xs leading-6 text-muted">这里的账号只在此浏览器有效。云端登录后，可以从上面的迁移入口选择这份存档。</p><LocalAccountForm /></details>}
  </div>
}
