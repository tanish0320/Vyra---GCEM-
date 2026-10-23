import { defineConfig } from "@playwright/test";
import shared from "./playwright.config";

// The backend developer starts an isolated Integration runtime first.
// No Python/model setup or patient-database mutation occurs on config import.
export default defineConfig({
  ...shared,
  use: {
    ...shared.use,
    baseURL: process.env.VYRA_BASE_URL || "http://127.0.0.1:5000",
  },
  webServer: undefined,
  outputDir: "/tmp/vyra-app-browser-results",
});
