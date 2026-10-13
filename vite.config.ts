import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  // 相对路径，方便部署到 GitHub Pages 等任意子路径（配合 HashRouter）
  base: './',
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  // 多个 dev server 并行时各自使用独立的依赖缓存目录，避免互相覆盖
  cacheDir: process.env.VITE_CACHE_DIR || 'node_modules/.vite',
  worker: { format: 'es' },
  server: { host: true },
  build: { chunkSizeWarningLimit: 2000 },
})
