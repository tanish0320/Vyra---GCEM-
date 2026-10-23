import { defineConfig } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
const runtimeFile = ".sites-runtime/browser.json";
const executablePath =
  process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
  (existsSync(runtimeFile)
    ? JSON.parse(readFileSync(runtimeFile, "utf8")).executablePath
    : undefined);
const baseURL = process.env.VYRA_BASE_URL || "http://127.0.0.1:4181";
export default defineConfig({
  testDir: "./tests/browser",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  workers: 1,
  retries: 0,
  reporter: "list",
  outputDir: "/tmp/vyra-browser-results",
  use: {
    baseURL,
    launchOptions: { executablePath },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { viewport: { width: 1440, height: 1000 } } },
    {
      name: "mobile",
      use: {
        viewport: { width: 390, height: 844 },
        isMobile: true,
        hasTouch: true,
      },
    },
    {
      name: "tablet",
      use: { viewport: { width: 768, height: 1024 }, hasTouch: true },
    },
  ],
  webServer: process.env.VYRA_BASE_URL
    ? undefined
    : {
        command: "npm run dev -- --port 4181 --strictPort",
        url: baseURL,
        reuseExistingServer: true,
      },
});
