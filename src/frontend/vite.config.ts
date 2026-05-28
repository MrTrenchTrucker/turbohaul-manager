import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/status': 'http://localhost:11401',
      '/health': 'http://localhost:11401',
      '/api': 'http://localhost:11401',
      '/v1': 'http://localhost:11401',
      '/ws': { target: 'ws://localhost:11401', ws: true },
    },
  },
});
