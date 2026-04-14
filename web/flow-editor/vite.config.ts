import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  root: path.resolve(__dirname),
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, "../../app/static/flow-editor"),
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(__dirname, "./src/main.tsx"),
      output: {
        entryFileNames: "flow-editor.js",
        chunkFileNames: "flow-editor-[name].js",
        assetFileNames: (assetInfo) =>
          assetInfo.name && assetInfo.name.endsWith(".css")
            ? "flow-editor.css"
            : "flow-editor-[name][extname]",
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.ts"],
  },
});
