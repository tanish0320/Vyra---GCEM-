import { defineConfig } from "@playwright/test";
import shared from "./playwright.config";

const origin = "http://127.0.0.1:4183";
export default defineConfig({
  ...shared,
  use: { ...shared.use, baseURL: origin },
  webServer: {
    command: "npm run preview:integration -- --port 4183 --strictPort",
    url: origin + "/static/ui/",
    reuseExistingServer: false,
  },
  outputDir: "/tmp/vyra-integration-ui-results",
  reporter: [
    ["list"],
    ["json", { outputFile: "/tmp/vyra-integration-ui-results.json" }],
  ],
});
