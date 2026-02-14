// vite.config.js
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    watch: {
      usePolling: true, // Полезно для Docker и некоторых ОС
      interval: 100, // Интервал проверки изменений
    },
    hmr: {
      overlay: true, // Показывать ошибки в браузере
      host: 'localhost',
      port: 3000,
      protocol: 'ws', // WebSocket протокол
    },
    hot: true, // Включить HMR
  },
})