import { scrypt, timingSafeEqual } from 'node:crypto'
import type { CloudUser } from '../shared/cloud'
import type { WorkerEnv } from './index'
import { json, writeAllowed } from './http'

const SESSION_SECONDS = 30 * 24 * 60 * 60
const AUTH_BODY_BYTES = 4_096
const PASSWORD_SCHEME = 'scrypt-v1:N16384:r8:p5'
// OWASP's 16 MiB / p=5 scrypt profile. Use the runtime implementation, not
// custom password hashing or Workers' historically capped WebCrypto PBKDF2.
const SCRYPT_OPTIONS = { N: 16_384, r: 8, p: 5, maxmem: 32 * 1024 * 1024 }
const DUMMY_SALT = new Uint8Array([12, 71, 108, 49, 52, 251, 61, 158, 97, 211, 16, 183, 44, 38, 207, 80])
const DUMMY_HASH = '0'.repeat(64)
let activeDerivations = 0

interface PasswordRow { id: string; username: string; password_salt: string; password_hash: string; password_scheme: string }
interface AttemptRow { attempts: number; expires_at: number }
class AuthError extends Error {
  constructor(readonly code: string, message: string, readonly status = 400, readonly retryAfter?: number) { super(message) }
}
const hex = (bytes: Uint8Array) => Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('')
function unhex(value: string): Uint8Array {
  if (!/^(?:[a-f0-9]{2})+$/.test(value)) throw new Error('Invalid stored authentication data')
  return Uint8Array.from(value.match(/../g)!, pair => Number.parseInt(pair, 16))
}
async function hash(value: string): Promise<string> { return hex(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)))) }
function randomToken(): string { return btoa(String.fromCharCode(...crypto.getRandomValues(new Uint8Array(32)))).replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '') }

function cookieSettings(request: Request): { name: string; secure: boolean } | null {
  const url = new URL(request.url)
  if (url.protocol === 'https:') return { name: '__Host-codequest_session', secure: true }
  if (url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) return { name: 'cq_dev_session', secure: false }
  return null
}
function sessionCookie(request: Request, token: string, clear = false): string {
  const settings = cookieSettings(request)
  if (!settings) throw new AuthError('https_required', '请通过 HTTPS 安全连接登录。')
  return `${settings.name}=${clear ? '' : token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${clear ? 0 : SESSION_SECONDS}${settings.secure ? '; Secure' : ''}`
}
async function requestTokenHash(request: Request): Promise<string | null> {
  const settings = cookieSettings(request)
  if (!settings) return null
  // Reject ambiguous duplicate cookies rather than selecting an attacker value.
  const matches = (request.headers.get('cookie') ?? '').split(';').map(value => value.trim()).filter(value => value.startsWith(`${settings.name}=`))
  if (matches.length !== 1) return null
  const token = matches[0].slice(settings.name.length + 1)
  return /^[A-Za-z0-9_-]{43}$/.test(token) ? hash(token) : null
}

/** Identity is obtained exclusively from our opaque, server-side session. */
export async function requestUser(request: Request, env: WorkerEnv): Promise<CloudUser | null> {
  const tokenHash = await requestTokenHash(request)
  if (!tokenHash) return null
  const row = await env.DB.prepare(`SELECT u.id, u.username FROM cq_password_sessions s
    INNER JOIN cq_password_users u ON u.id = s.user_id
    WHERE s.token_hash = ? AND s.expires_at > ?`).bind(tokenHash, Date.now()).first<CloudUser>()
  return row ? { id: row.id, username: row.username } : null
}

async function readAuthJSON(request: Request): Promise<Record<string, unknown>> {
  const tooLarge = () => new AuthError('payload_too_large', '请求内容过大，请检查用户名和密码长度。', 413)
  const invalid = () => new AuthError('invalid_request', '请求格式不正确，请重新提交。')
  const length = request.headers.get('content-length')
  if (length && (!/^\d+$/.test(length) || Number(length) > AUTH_BODY_BYTES)) throw tooLarge()
  if (!request.body) throw invalid()
  const reader = request.body.getReader()
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      size += value.byteLength
      if (size > AUTH_BODY_BYTES) { await reader.cancel(); throw tooLarge() }
      chunks.push(value)
    }
    const bytes = new Uint8Array(size)
    let offset = 0
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength }
    const body: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))
    if (!body || typeof body !== 'object' || Array.isArray(body)) throw invalid()
    return body as Record<string, unknown>
  } catch (error) {
    if (error instanceof AuthError) throw error
    throw invalid()
  } finally { reader.releaseLock() }
}

function credentials(body: Record<string, unknown>): { username: string; key: string; password: string } {
  if (Object.keys(body).some(key => key !== 'username' && key !== 'password')) throw new AuthError('invalid_request', '请求只能包含用户名和密码。')
  const username = typeof body.username === 'string' && body.username.length <= 128 ? body.username.normalize('NFKC').trim() : ''
  // Restrict case-bearing characters to ASCII: lowercasing is a complete case
  // fold for this alphabet, including full-width input normalized by NFKC.
  if (!/^[\p{Script=Han}A-Za-z0-9_-]{2,24}$/u.test(username)) throw new AuthError('invalid_username', '用户名需为 2–24 个中文、英文字母、数字、下划线或连字符。')
  const password = body.password
  if (typeof password !== 'string' || password.length > 256 || Array.from(password).length < 8 || Array.from(password).length > 128 || new TextDecoder().decode(new TextEncoder().encode(password)) !== password) {
    throw new AuthError('invalid_password', '密码需为 8–128 个字符，首尾空格也计入密码。')
  }
  return { username, key: username.toLowerCase(), password }
}

async function rateLimit(request: Request, env: WorkerEnv, kind: 'register' | 'login', username: string): Promise<void> {
  const suppliedIP = request.headers.get('cf-connecting-ip') ?? ''
  // Cloudflare supplies CF-Connecting-IP. Do not trust X-Forwarded-For. If the
  // edge omits the address, use a shared conservative bucket, never fail open.
  const ip = /^[\da-f:.]{3,64}$/i.test(suppliedIP) ? suppliedIP : 'unknown'
  const now = Date.now()
  const window = kind === 'register' ? 60 * 60_000 : 15 * 60_000
  const dimensions = [
    { key: await hash(`${kind}:ip:${ip}`), limit: kind === 'register' ? 10 : 60 },
    { key: await hash(`${kind}:username:${username}`), limit: kind === 'register' ? 5 : 10 },
  ]
  // Update the IP bucket first, then admit a username bucket only if that IP is
  // still within its budget. Rotating names cannot grow storage after an IP ban.
  const statements = dimensions.flatMap(({ key, limit }, index) => [
    env.DB.prepare(`INSERT INTO cq_password_attempts (attempt_key, attempts, expires_at)
      SELECT ?, 1, ? WHERE ? = 0 OR EXISTS (
        SELECT 1 FROM cq_password_attempts WHERE attempt_key = ? AND attempts <= ?
      )
      ON CONFLICT(attempt_key) DO UPDATE SET
      attempts = CASE WHEN expires_at <= ? THEN 1 ELSE MIN(attempts + 1, ?) END,
      expires_at = CASE WHEN expires_at <= ? THEN excluded.expires_at ELSE expires_at END`).bind(key, now + window, index, dimensions[0].key, dimensions[0].limit, now, limit + 1, now),
    env.DB.prepare('SELECT attempts, expires_at FROM cq_password_attempts WHERE attempt_key = ?').bind(key),
  ])
  // Bounded cleanup uses expiry indexes and keeps attacker-driven work fixed.
  statements.push(env.DB.prepare('DELETE FROM cq_password_attempts WHERE attempt_key IN (SELECT attempt_key FROM cq_password_attempts WHERE expires_at <= ? ORDER BY expires_at LIMIT 100)').bind(now))
  statements.push(env.DB.prepare('DELETE FROM cq_password_sessions WHERE token_hash IN (SELECT token_hash FROM cq_password_sessions WHERE expires_at <= ? ORDER BY expires_at LIMIT 100)').bind(now))
  const results = await env.DB.batch(statements)
  if (results.some(result => !result.success)) throw new Error('Authentication storage unavailable')
  let retryAfter = 0
  for (const [i, dimension] of dimensions.entries()) {
    const row = results[i * 2 + 1].results[0] as unknown as AttemptRow | undefined
    if (row && row.attempts > dimension.limit) retryAfter = Math.max(retryAfter, Math.ceil((row.expires_at - now) / 1000))
  }
  if (retryAfter) throw new AuthError('rate_limited', '尝试次数过多，请稍后再试。', 429, retryAfter)
}

async function passwordHash(password: string, salt: Uint8Array): Promise<Uint8Array> {
  // Two concurrent native allocations use ~32 MiB of the 128 MiB isolate.
  // Reject excess work immediately; an unbounded queue could itself exhaust RAM.
  if (activeDerivations >= 2) throw new AuthError('rate_limited', '登录服务繁忙，请稍后重试。', 429, 2)
  activeDerivations++
  try {
    return await new Promise<Uint8Array>((resolve, reject) => {
      scrypt(password, salt, 32, SCRYPT_OPTIONS, (error, key) => error ? reject(error) : resolve(key))
    })
  } finally { activeDerivations-- }
}

async function authenticate(request: Request, env: WorkerEnv, kind: 'register' | 'login', body: Record<string, unknown>): Promise<Response> {
  const { username, key, password } = credentials(body)
  await rateLimit(request, env, kind, key)
  const now = Date.now()
  let user: CloudUser
  const token = randomToken()
  const tokenHash = await hash(token)
  const oldTokenHash = await requestTokenHash(request)
  const statements: D1PreparedStatement[] = []
  if (kind === 'register') {
    user = { id: `cqp_${crypto.randomUUID()}`, username }
    const salt = crypto.getRandomValues(new Uint8Array(16))
    const derived = await passwordHash(password, salt)
    statements.push(env.DB.prepare(`INSERT INTO cq_password_users
      (id, username, username_key, password_salt, password_hash, password_scheme, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(username_key) DO NOTHING RETURNING id`)
      .bind(user.id, username, key, hex(salt), hex(derived), PASSWORD_SCHEME, now))
  } else {
    const record = await env.DB.prepare('SELECT id, username, password_salt, password_hash, password_scheme FROM cq_password_users WHERE username_key = ?').bind(key).first<PasswordRow>()
    if (record && (record.password_scheme !== PASSWORD_SCHEME || !/^[a-f0-9]{32}$/.test(record.password_salt) || !/^[a-f0-9]{64}$/.test(record.password_hash))) throw new Error('Invalid stored authentication data')
    // Unknown usernames perform exactly the same expensive KDF and comparison.
    const derived = await passwordHash(password, record ? unhex(record.password_salt) : DUMMY_SALT)
    const matches = timingSafeEqual(derived, unhex(record?.password_hash ?? DUMMY_HASH))
    if (!record || !matches) throw new AuthError('invalid_credentials', '用户名或密码不正确。', 401)
    user = { id: record.id, username: record.username }
  }
  // Registration and session creation are one transaction. A losing duplicate
  // registration can neither create a session nor revoke an existing session.
  statements.push(env.DB.prepare(`INSERT INTO cq_password_sessions (token_hash, user_id, created_at, expires_at)
    SELECT ?, id, ?, ? FROM cq_password_users WHERE id = ?`).bind(tokenHash, now, now + SESSION_SECONDS * 1000, user.id))
  if (oldTokenHash) statements.push(env.DB.prepare(`DELETE FROM cq_password_sessions WHERE token_hash = ?
    AND EXISTS (SELECT 1 FROM cq_password_sessions WHERE token_hash = ?)`).bind(oldTokenHash, tokenHash))
  // Keep a bounded number of devices without extending fixed expiry on reads.
  statements.push(env.DB.prepare(`DELETE FROM cq_password_sessions WHERE token_hash IN (
    SELECT token_hash FROM cq_password_sessions WHERE user_id = ? AND token_hash <> ?
    ORDER BY created_at DESC, token_hash DESC LIMIT -1 OFFSET 19
  )`).bind(user.id, tokenHash))
  const results = await env.DB.batch(statements)
  if (results.some(result => !result.success)) throw new Error('Authentication storage unavailable')
  if (kind === 'register' && !results[0].results.length) throw new AuthError('username_taken', '这个用户名已被使用，请换一个用户名。', 409)
  return json({ user, auth: 'password' }, kind === 'register' ? 201 : 200, { 'Set-Cookie': sessionCookie(request, token) })
}

export async function handleAuth(request: Request, env: WorkerEnv): Promise<Response> {
  try {
    const action = new URL(request.url).pathname.slice('/api/auth/'.length)
    if (!['register', 'login', 'logout'].includes(action)) return json({ error: 'not_found' }, 404)
    if (request.method !== 'POST') return json({ error: 'method_not_allowed' }, 405)
    if (!cookieSettings(request)) throw new AuthError('https_required', '请通过 HTTPS 安全连接登录。')
    if (!writeAllowed(request)) throw new AuthError('invalid_origin', '请从本站页面提交登录请求。', 403)
    if (request.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/json') throw new AuthError('unsupported_media_type', '请使用正确的登录表单提交请求。', 415)
    const body = await readAuthJSON(request)
    if (action === 'logout') {
      if (Object.keys(body).length) throw new AuthError('invalid_request', '退出请求格式不正确。')
      const expectedUser = request.headers.get('x-codequest-user')
      if (expectedUser !== null) {
        const user = await requestUser(request, env)
        if (user && user.id !== expectedUser) return json({ error: 'identity_changed', message: '登录账号已在其他标签页切换，请刷新后再操作。', user }, 409)
      }
      const tokenHash = await requestTokenHash(request)
      if (tokenHash) await env.DB.prepare('DELETE FROM cq_password_sessions WHERE token_hash = ?').bind(tokenHash).run()
      return json({ ok: true }, 200, { 'Set-Cookie': sessionCookie(request, '', true) })
    }
    return await authenticate(request, env, action as 'register' | 'login', body)
  } catch (error) {
    if (error instanceof AuthError) return json({ error: error.code, message: error.message }, error.status, error.retryAfter ? { 'Retry-After': String(error.retryAfter) } : undefined)
    // Do not log request bodies, credentials, session tokens, identity, or SQL.
    return json({ error: 'storage_unavailable', message: '账号服务暂时不可用，请稍后重试。' }, 503)
  }
}
