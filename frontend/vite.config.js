import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/voice-agent': 'http://127.0.0.1:8000',
      '/query': 'http://127.0.0.1:8000',
      '/guardrail': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
});
