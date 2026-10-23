import { defineConfig } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
const runtime = ".sites-runtime/browser.json";
const executablePath =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  (existsSync(runtime)
    ? JSON.parse(readFileSync(runtime, "utf8")).executablePath
    : undefined);
export default defineConfig({
  testDir: ".",
  testMatch: "capture.spec.ts",
  timeout: 1_800_000,
  workers: 1,
  reporter: "line",
  outputDir: "/tmp/vyra-eye-film-test-results",
  projects: [{ name: "preview" }, { name: "film" }, { name: "resume" }],
  use: {
    baseURL: "http://127.0.0.1:4182",
    viewport: { width: 1600, height: 1200 },
    launchOptions: { executablePath },
    deviceScaleFactor: 1,
  },
  webServer: {
    command: "npm run dev -- --port 4182 --strictPort",
    url: "http://127.0.0.1:4182",
    reuseExistingServer: true,
  },
});
