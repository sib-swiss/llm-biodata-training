import {defineConfig} from 'vite';

// https://vitejs.dev/config/
export default defineConfig({
  // The name of your repository if you deploy to GitHub pages:
  base: '/revealjs-template-markdown-vite/',
  server: {
    port: 3000,
  },
  // Required to automatically reload the page when a markdown file changes:
  plugins: [
    {
      name: 'reload',
      configureServer(server) {
        server.watcher.on('change', file => {
          if (file.endsWith('.md')) server.ws.send({type: 'full-reload'})
        })
      }
    }
  ],
});
