import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router'
import { ArrowLeft, ArrowRight, BookOpen, Code2, HardDrive, UserRound } from 'lucide-react'
import { Button, Card, Segmented } from '@/components/ui'
import { canClaimGuest, loginAccount, registerAccount, switchAccount, useAccount } from '@/lib/accounts'

export default function Account() {
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
    <div className="mx-auto max-w-4xl py-3 md:py-12">
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
              {eligible && <label className="flex items-start gap-2 text-sm leading-6"><input type="checkbox" checked={claim} onChange={(e) => setClaim(e.target.checked)} className="mt-1.5 accent-accent" />将当前游客的进度、草稿和面试记录移入此账号</label>}
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
