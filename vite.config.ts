import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'
import { sites } from './build/sites-vite-plugin.ts'

export default defineConfig(async () => {
  process.env.CLOUDFLARE_CF_FETCH_ENABLED ??= 'false'
  process.env.WRANGLER_SEND_METRICS ??= 'false'
  process.env.WRANGLER_WRITE_LOGS ??= 'false'
  process.env.WRANGLER_LOG_PATH ??= '.wrangler/logs'
  process.env.WRANGLER_REGISTRY_PATH ??= '.wrangler/dev-registry'
  process.env.MINIFLARE_REGISTRY_PATH ??= '.wrangler/registry'
  const { cloudflare } = await import('@cloudflare/vite-plugin')
  return {
    base: '/',
    plugins: [react(), tailwindcss(), sites({ mockAuth: false }), cloudflare({
      viteEnvironment: { name: 'server' },
      inspectorPort: false,
      persistState: { path: '.wrangler/state' },
      config: {
        name: 'codequest',
        main: './worker/index.ts',
        compatibility_date: '2026-05-15',
        compatibility_flags: ['nodejs_compat'],
        assets: { binding: 'ASSETS', not_found_handling: 'single-page-application', run_worker_first: ['/api/*'] },
        d1_databases: [{ binding: 'DB', database_name: 'codequest-d1', database_id: '00000000-0000-4000-8000-000000000000' }],
      },
    })],
    resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
    cacheDir: process.env.VITE_CACHE_DIR || 'node_modules/.vite',
    worker: { format: 'es' as const },
    server: { host: '127.0.0.1', watch: { useFsEvents: false } },
    build: { chunkSizeWarningLimit: 2000 },
  }
})
