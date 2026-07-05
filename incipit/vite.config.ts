import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-time only: Vite serves the UI and proxies /api to incipit.py.
// In production incipit.py serves the built dist/ itself -- same origin,
// no proxy, no CORS anywhere.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8790' },
  },
})
