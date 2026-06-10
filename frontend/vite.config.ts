import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/app/",
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  optimizeDeps: {
    include: ["@langchain/core/messages", "@langchain/langgraph-sdk"],
  },
  server: {
    proxy: {
      // 代理所有 LangGraph API 请求到后端
      "/threads": {
        target: "http://127.0.0.1:47569",
        changeOrigin: true,
      },
      "/runs": {
        target: "http://127.0.0.1:47569",
        changeOrigin: true,
      },
      "/assistants": {
        target: "http://127.0.0.1:47569",
        changeOrigin: true,
      },
      "/api": {
        target: "http://127.0.0.1:47569",
        changeOrigin: true,
      },
    },
  },
});