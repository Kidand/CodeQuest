/** Responses containing account or study data must never be cached. */
export function json(data: unknown, status = 200, headers?: HeadersInit): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'private, no-store, max-age=0',
      Pragma: 'no-cache',
      Vary: 'Cookie',
      'X-Content-Type-Options': 'nosniff',
      ...Object.fromEntries(new Headers(headers)),
    },
  })
}

export function writeAllowed(request: Request): boolean {
  // Origin is mandatory even in browsers without Fetch Metadata support.
  const origin = request.headers.get('origin')
  const fetchSite = request.headers.get('sec-fetch-site')
  return origin === new URL(request.url).origin && (fetchSite === null || fetchSite === 'same-origin')
}
