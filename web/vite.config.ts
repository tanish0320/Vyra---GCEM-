import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backend = env.VYRA_BACKEND_URL || "http://127.0.0.1:5000";
  const proxy = Object.fromEntries(
    [
      "/api",
      "/export",
      "/media",
      "/health",
      "/achilles-heel",
      "/eda",
      "/about",
    ].map((path) => [path, backend]),
  );
  // Backend figures and legacy-page assets; never intercept this SPA's /static/ui/.
  proxy["^/static/(?!ui(?:/|$))"] = backend;
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
    },
    server: { proxy },
    preview: { proxy },
  };
});
