import {
  CloudValidationError,
  MAX_CLOUD_BYTES,
  normalizeCloudSnapshot,
  type CloudSnapshot,
  type CloudStateResponse,
  type CloudUser,
} from '../shared/cloud'

export interface WorkerEnv {
  DB: D1Database
  ASSETS: { fetch(request: Request): Promise<Response> }
}
interface StateRow { payload: string | null; revision: number; updated_at: string | null }
interface MutationRow { base_revision: number; payload_hash: string; revision: number }
const MAX_REQUEST_BYTES = MAX_CLOUD_BYTES + 8_192
const RETAINED_MUTATIONS = 128

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'private, no-store, max-age=0',
      'Pragma': 'no-cache',
      'Vary': 'Cookie, oai-authenticated-user-id',
      'X-Content-Type-Options': 'nosniff',
    },
  })
}

/** These headers are injected by Sites dispatch; do not expose this Worker elsewhere. */
function requestUser(request: Request): CloudUser | null {
  const id = request.headers.get('oai-authenticated-user-id')
  const email = request.headers.get('oai-authenticated-user-email')
  if (!id || id.length > 512 || !email || email.length > 512) return null
  let fullName = request.headers.get('oai-authenticated-user-full-name') ?? ''
  if (request.headers.get('oai-authenticated-user-full-name-encoding') === 'percent-encoded-utf-8') {
    try { fullName = decodeURIComponent(fullName) } catch { fullName = '' }
  }
  return { id, email, username: fullName.trim().slice(0, 256) || email.split('@')[0] || 'CodeQuest 玩家' }
}

function returnPath(url: URL): string {
  const path = url.searchParams.get('returnTo') ?? '/#/account'
  if (!path.startsWith('/') || path.startsWith('//') || /[\\\u0000-\u001f\u007f]/.test(path) || path.length > 2_000) return '/#/account'
  try {
    const parsed = new URL(path, url.origin)
    if (['/signin-with-chatgpt', '/signout-with-chatgpt', '/callback'].includes(parsed.pathname)) return '/#/account'
    return parsed.origin === url.origin ? `${parsed.pathname}${parsed.search}${parsed.hash}` : '/#/account'
  } catch { return '/#/account' }
}

function stateResponse(user: CloudUser, row?: StateRow | null): CloudStateResponse {
  return {
    user,
    revision: row?.revision ?? 0,
    updatedAt: row?.updated_at ?? null,
    snapshot: row?.payload ? normalizeCloudSnapshot(JSON.parse(row.payload)) : null,
  }
}

function writeAllowed(request: Request): boolean {
  // Require Origin even when Fetch Metadata is unavailable (older browsers).
  const origin = request.headers.get('origin')
  const fetchSite = request.headers.get('sec-fetch-site')
  return origin === new URL(request.url).origin && (fetchSite === null || fetchSite === 'same-origin')
}

async function readBoundedJSON(request: Request): Promise<unknown> {
  const length = request.headers.get('content-length')
  if (length && (!/^\d+$/.test(length) || Number(length) > MAX_REQUEST_BYTES)) throw new CloudValidationError('payload_too_large')
  if (!request.body) throw new CloudValidationError()
  const reader = request.body.getReader()
  const chunks: Uint8Array[] = []
  let size = 0
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      size += value.byteLength
      if (size > MAX_REQUEST_BYTES) {
        await reader.cancel()
        throw new CloudValidationError('payload_too_large')
      }
      chunks.push(value)
    }
    const buffer = new Uint8Array(size)
    let offset = 0
    for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.byteLength }
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(buffer))
  } catch (error) {
    if (error instanceof CloudValidationError) throw error
    throw new CloudValidationError()
  } finally { reader.releaseLock() }
}

async function writeState(request: Request, env: WorkerEnv, user: CloudUser): Promise<Response> {
  if (!writeAllowed(request)) return json({ error: 'invalid_origin' }, 403)
  if (request.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/json') return json({ error: 'unsupported_media_type' }, 415)
  const input = await readBoundedJSON(request)
  if (!input || typeof input !== 'object' || Array.isArray(input)) return json({ error: 'invalid_request' }, 400)
  const body = input as Record<string, unknown>
  if (Object.keys(body).some(key => !['baseRevision', 'snapshot', 'mutationId'].includes(key))) return json({ error: 'invalid_request' }, 400)
  if (!Number.isSafeInteger(body.baseRevision) || (body.baseRevision as number) < 0 || (body.baseRevision as number) >= Number.MAX_SAFE_INTEGER) return json({ error: 'invalid_revision' }, 400)
  if (typeof body.mutationId !== 'string' || !/^[a-zA-Z0-9_-]{8,128}$/.test(body.mutationId)) return json({ error: 'invalid_mutation_id' }, 400)
  const base = body.baseRevision as number
  const mutationId = body.mutationId
  const snapshot: CloudSnapshot = normalizeCloudSnapshot(body.snapshot)
  const payload = JSON.stringify(snapshot)
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(payload))
  const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')
  const now = new Date().toISOString()

  // D1 batch is transactional and sequential. Only a matching revision can claim
  // a mutation; the exact same mutation/hash/base tuple is needed for the update.
  // Read both resulting records inside that transaction, so a concurrent save
  // cannot turn a successful response into a response for somebody else's write.
  const result = await env.DB.batch([
    env.DB.prepare('INSERT OR IGNORE INTO cq_cloud_states (user_id, payload, revision, updated_at) VALUES (?, NULL, 0, NULL)').bind(user.id),
    env.DB.prepare(`INSERT OR IGNORE INTO cq_cloud_mutations (user_id, mutation_id, base_revision, payload_hash, revision, created_at)
      SELECT ?, ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM cq_cloud_states WHERE user_id = ? AND revision = ?)`)
      .bind(user.id, mutationId, base, hash, base + 1, now, user.id, base),
    env.DB.prepare(`UPDATE cq_cloud_states SET payload = ?, revision = revision + 1, updated_at = ?
      WHERE user_id = ? AND revision = ? AND EXISTS (
        SELECT 1 FROM cq_cloud_mutations WHERE user_id = ? AND mutation_id = ? AND base_revision = ? AND payload_hash = ? AND revision = ?
      )`).bind(payload, now, user.id, base, user.id, mutationId, base, hash, base + 1),
    env.DB.prepare('SELECT base_revision, payload_hash, revision FROM cq_cloud_mutations WHERE user_id = ? AND mutation_id = ?').bind(user.id, mutationId),
    env.DB.prepare('SELECT payload, revision, updated_at FROM cq_cloud_states WHERE user_id = ?').bind(user.id),
    env.DB.prepare(`DELETE FROM cq_cloud_mutations WHERE user_id = ? AND revision <=
      (SELECT revision - ? FROM cq_cloud_states WHERE user_id = ?)`)
      .bind(user.id, RETAINED_MUTATIONS, user.id),
  ])
  if (result.some(r => !r.success)) throw new Error('Cloud transaction failed')
  const mutation = result[3].results[0] as unknown as MutationRow | undefined
  const row = result[4].results[0] as unknown as StateRow | undefined
  const state = stateResponse(user, row)
  if (mutation?.base_revision === base && mutation.payload_hash === hash && row?.revision === mutation.revision) return json(state)
  return json({ error: 'conflict', ...state }, 409)
}

export async function handleRequest(request: Request, env: WorkerEnv): Promise<Response> {
  const url = new URL(request.url)
  if (!url.pathname.startsWith('/api/')) return env.ASSETS.fetch(request)
  try {
    const user = requestUser(request)
    if (url.pathname === '/api/cloud/session') {
      if (request.method !== 'GET') return json({ error: 'method_not_allowed' }, 405)
      const path = encodeURIComponent(returnPath(url))
      return json({ user, signInUrl: `/signin-with-chatgpt?return_to=${path}`, signOutUrl: `/signout-with-chatgpt?return_to=${path}` })
    }
    if (url.pathname !== '/api/cloud/state') return json({ error: 'not_found' }, 404)
    if (request.method !== 'GET' && request.method !== 'PUT') return json({ error: 'method_not_allowed' }, 405)
    if (!user) return json({ error: 'unauthenticated' }, 401)
    const expectedUser = request.headers.get('x-codequest-user')
    if (expectedUser !== null && expectedUser !== user.id) return json({ error: 'identity_changed', user }, 409)
    if (request.method === 'PUT') return await writeState(request, env, user)
    const row = await env.DB.prepare('SELECT payload, revision, updated_at FROM cq_cloud_states WHERE user_id = ?').bind(user.id).first<StateRow>()
    return json(stateResponse(user, row))
  } catch (error) {
    if (error instanceof CloudValidationError) return json({ error: error.code, message: error.message }, error.code === 'payload_too_large' ? 413 : 400)
    // Never log request bodies, headers, stored code, identity, or D1 details.
    return json({ error: 'storage_unavailable', message: '云端存档暂时不可用，你的本地进度仍然保留。请稍后重试。' }, 503)
  }
}

export default { fetch: handleRequest }
