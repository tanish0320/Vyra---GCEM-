import { defineConfig } from "@playwright/test";
import app from "./playwright.app.config";
export default defineConfig({
  ...app,
  testMatch: "**/live-integration.spec.ts",
});
