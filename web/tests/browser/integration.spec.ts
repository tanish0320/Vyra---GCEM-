import { test, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { setup, ready, scan, job, stress, photo } from "./integration-fixtures";
const evidence = "/tmp/vyra-integration-evidence";
mkdirSync(evidence, { recursive: true });

async function openResult(page) {
  await page.goto("/static/ui/#/history/42");
  await expect(
    page.getByRole("heading", { name: "Screen positive", exact: true }),
  ).toBeVisible();
}
async function noOverflow(page) {
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize().width + 1);
}

test("dual capture submits both files and requires review of both tissues", async ({
  page,
}) => {
  await setup(page);
  let uploaded = "",
    confirmed: any = null;
  await page.route("**/api/v1/jobs", async (r) => {
    uploaded = r.request().postDataBuffer()!.toString();
    await r.fulfill({ json: job, status: 202 });
  });
  await page.route("**/api/v1/jobs/ui-job/confirm", async (r) => {
    confirmed = r.request().postDataJSON();
    await r.fulfill({
      json: { ...job, status: "running", stage: "normalization" },
      status: 202,
    });
  });
  await page.goto("/static/ui/#/studio");
  await page.getByRole("radio", { name: "Eye + nail", exact: true }).click();
  await page
    .getByLabel("Choose a photo", { exact: true })
    .setInputFiles({ name: "eye.png", mimeType: "image/png", buffer: photo });
  await page
    .getByLabel("Choose a nail photo", { exact: true })
    .setInputFiles({ name: "nail.png", mimeType: "image/png", buffer: photo });
  await page
    .getByLabel("Patient name or case ID", { exact: false })
    .fill("Dual UI test");
  await page.getByLabel("Age in years", { exact: false }).fill("24");
  await page
    .getByLabel("Biological sex", { exact: false })
    .selectOption("not-stated");
  await page
    .getByRole("button", { name: "Review images", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Analyze both images", exact: true }),
  ).toBeDisabled();
  await page
    .getByLabel("I have reviewed the lower eyelid region in the first image.", {
      exact: true,
    })
    .check();
  await expect(
    page.getByRole("button", { name: "Analyze both images", exact: true }),
  ).toBeDisabled();
  await page
    .getByLabel("I have reviewed the nail bed region in the second image.", {
      exact: true,
    })
    .check();
  await page
    .getByRole("button", { name: "Analyze both images", exact: true })
    .click();
  await expect.poll(() => confirmed).toBeTruthy();
  expect(uploaded).toContain('name="secondary_image"');
  expect(confirmed).toEqual({
    roi: job.roi,
    secondary_roi: job.secondary!.roi,
  });
  await noOverflow(page);
});

test("missing secondary review blocks dual analysis", async ({ page }) => {
  await setup(page);
  await page.route("**/api/v1/jobs/ui-job", (r) =>
    r.fulfill({ json: { ...job, secondary: null } }),
  );
  await page.goto("/static/ui/#/studio/jobs/ui-job");
  await expect(
    page.getByRole("button", { name: "Analyze both images", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Reload both images", exact: true }),
  ).toBeVisible();
});

test("older services cannot request unsupported dual capture", async ({
  page,
}) => {
  await setup(page, { system: { ...ready, capabilities: {} } });
  await page.goto("/static/ui/#/studio");
  await expect(
    page.getByLabel("Choose a photo", { exact: true }),
  ).toBeAttached();
  await expect(
    page.getByRole("radio", { name: "Eye + nail", exact: true }),
  ).toHaveCount(0);
});

test("results show sourced colour data and percent fusion with working blend and PDF", async ({
  page,
}, info) => {
  await setup(page);
  await openResult(page);
  await expect(page.locator("#downloadPdfButton")).toHaveAttribute(
    "href",
    "/export/pdf/42",
  );
  await expect(page.locator("#erythemaFillBar")).toBeAttached();
  await expect(page.locator("main")).toContainText("38.2");
  await expect(page.locator("main")).toContainText("59.0%");
  await expect(page.locator("main")).toContainText("95.9%");
  await page.locator("#btnModeBlend").click();
  await page.locator("#gradcamSlider").fill("0");
  await expect(page.locator("#gradcamSlider")).toHaveValue("0");
  await page.locator("#btnModeSplit").click();
  await expect(page.locator("#gradcamSlider")).toHaveAttribute(
    "aria-label",
    "Attention comparison",
  );
  await expect(page.locator("#stressTestLink")).toHaveAttribute(
    "href",
    "/achilles-heel?scan_id=42",
  );
  await noOverflow(page);
  await page.screenshot({
    path: `${evidence}/${info.project.name}-result.png`,
    fullPage: true,
  });
  await page
    .getByRole("link", { name: "Open stress workspace", exact: true })
    .click();
  await expect(page).toHaveURL(/#\/achilles-heel\?scan_id=42$/);
  await expect(page.locator("main")).toContainText("Screening 42");
});

test("legacy and placeholder measurements stay unavailable; nail Hb is withheld", async ({
  page,
}) => {
  const result = {
    ...scan,
    modality: "fingernail" as const,
    hb_g_dl: 99.99,
    multimodal_fusion: null,
    biophysical_metrics: {
      ...scan.biophysical_metrics!,
      has_real_pixels: false,
      erythema_index: 52.4,
    },
  };
  await setup(page, { system: { ...ready, capabilities: {} }, result });
  await openResult(page);
  await expect(page.locator("main")).not.toContainText("52.4");
  await expect(page.locator("main")).not.toContainText("99.99");
  await expect(page.locator("#erythemaFillBar")).toHaveCount(0);
  await expect(page.locator("#assistantSendBtn")).toBeDisabled();
  await noOverflow(page);
});

test("assistant formats safe text, labels fallback, switches language and shows emergency", async ({
  page,
}, info) => {
  await setup(page);
  let payload: any;
  await page.route("**/api/v1/explain/42", async (r) => {
    payload = r.request().postDataJSON();
    const emergency = payload.question.includes("chest");
    await r.fulfill({
      json: emergency
        ? {
            is_emergency: true,
            response: "Seek urgent professional help.",
            model_used: "emergency_firewall",
            hotlines: ["108", "112"],
          }
        : {
            is_emergency: false,
            response:
              "**Image finding**\n- Review the tissue crop.\n\n<img src=x onerror=alert(1)>\n\n⚠️ **Clinical Disclaimer**: Educational information only.",
            model_used: "deterministic_guardrail",
          },
    });
  });
  await openResult(page);
  await page.locator("#assistantInput").fill("Explain my image");
  await page.locator("#assistantSendBtn").click();
  await expect(
    page
      .locator("#assistantChatStream strong")
      .filter({ hasText: "Image finding" }),
  ).toBeVisible();
  await expect(page.locator("#assistantChatStream")).toContainText(
    "deterministic guardrail",
  );
  await expect(page.locator("#assistantChatStream img")).toHaveCount(0);
  await expect(page.locator("#assistantChatStream")).toContainText(
    "<img src=x",
  );
  await page.locator('.lang-btn[data-lang="hi"]').click();
  await page.locator("#assistantInput").fill("chest pain");
  await page.locator("#assistantSendBtn").click();
  await expect.poll(() => payload.language).toBe("hi");
  await expect(page.locator("#assistantChatStream")).toContainText(
    "Seek urgent professional help.",
  );
  await expect(page.locator("#assistantChatStream")).toContainText("112");
  await noOverflow(page);
  await page.screenshot({
    path: `${evidence}/${info.project.name}-assistant.png`,
    fullPage: true,
  });
});

test("assistant ignores a response after its conversation is cleared", async ({
  page,
}) => {
  await setup(page);
  let release: () => void;
  await page.route("**/api/v1/explain/42", async (r) => {
    await new Promise<void>((resolve) => (release = resolve));
    await r
      .fulfill({
        json: {
          is_emergency: false,
          response: "Stale answer should disappear",
          model_used: "qwen2.5:7b",
        },
      })
      .catch(() => {});
  });
  await openResult(page);
  await page.locator("#assistantInput").fill("Hi");
  await page.locator("#assistantSendBtn").click();
  await expect.poll(() => Boolean(release)).toBe(true);
  await page
    .getByRole("button", { name: "Clear conversation", exact: true })
    .click();
  release!();
  await expect(page.locator("#assistantChatStream")).not.toContainText(
    "Stale answer",
  );
  await expect(page.locator("#assistantInput")).toHaveValue("");
});

test("stress lab renders actual response units and research pages without seeding data", async ({
  page,
}, info) => {
  await setup(page);
  let stressFile = false;
  await page.route("**/api/v1/achilles-heel", async (r) => {
    stressFile = r
      .request()
      .postDataBuffer()!
      .toString()
      .includes('name="image"');
    await r.fulfill({ json: stress });
  });
  await page.route("**/api/v1/research", (r) =>
    r.fulfill({
      json: {
        summary: "Fixture-only research catalog",
        datasets: [
          {
            name: "Permitted sample collection",
            description: "Test provenance",
            count: 2,
          },
        ],
        plots: [],
      },
    }),
  );
  await page.goto("/static/ui/#/achilles-heel?scan_id=42");
  await page
    .getByRole("button", { name: /Run.*(stress|battery|test)/i })
    .click();
  await expect(page.locator("main")).toContainText("Baseline ROI");
  await expect(page.locator("main")).toContainText("60.0%");
  expect(stressFile).toBe(true);
  await noOverflow(page);
  await page.screenshot({
    path: `${evidence}/${info.project.name}-stress.png`,
    fullPage: true,
  });
  await page.goto("/static/ui/#/research");
  await expect(page.locator("main")).toContainText(
    "Fixture-only research catalog",
  );
  await noOverflow(page);
  await page.goto("/static/ui/#/analytics");
  await expect(page.locator("main")).toContainText("1");
  await noOverflow(page);
  await page.goto("/static/ui/#/about");
  await expect(page.locator("h1")).toBeVisible();
  await noOverflow(page);
});

test("camera HUD samples real frames and releases tracks after capture", async ({
  page,
}) => {
  await setup(page);
  await page.addInitScript(() => {
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      value: async () => {
        const c = document.createElement("canvas");
        c.width = 640;
        c.height = 480;
        const x = c.getContext("2d")!;
        x.fillStyle = "#c0c0c0";
        x.fillRect(0, 0, 640, 480);
        for (let y = 0; y < 480; y += 8)
          for (let z = 0; z < 640; z += 8) {
            x.fillStyle = (y + z) % 16 ? "#666" : "#ccc";
            x.fillRect(z, y, 8, 8);
          }
        const s = c.captureStream(10);
        (window as any).__cameraTracks = s.getTracks();
        return s;
      },
    });
  });
  await page.goto("/static/ui/#/studio");
  await page.getByRole("button", { name: "Use camera", exact: true }).click();
  await expect(page.locator("#cameraVideo")).toBeVisible();
  await expect(page.locator("#targetingReticle")).toBeVisible();
  await expect(page.locator("#shutterBtn")).toBeEnabled();
  await expect(page.locator("#lightingBadge")).toHaveAttribute(
    "data-luminance",
    /^\d+(\.\d+)?$/,
  );
  await expect(page.locator("#focusBadge")).toHaveAttribute(
    "data-detail",
    /^\d+(\.\d+)?$/,
  );
  await page.locator("#shutterBtn").click();
  await expect(page.getByAltText("Selected screening photo")).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(() =>
        (window as any).__cameraTracks.every((t) => t.readyState === "ended"),
      ),
    )
    .toBe(true);
});

test("assistant service errors remain errors and never become invented fallback answers", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/v1/explain/42", (r) =>
    r.fulfill({
      status: 503,
      json: { error: "Explanation service unavailable" },
    }),
  );
  await openResult(page);
  await page.locator("#assistantInput").fill("Hi");
  await page.locator("#assistantSendBtn").click();
  await expect(page.locator("#assistantSection [role=alert]")).toContainText(
    "Explanation service unavailable",
  );
  await expect(
    page.locator("#assistantChatStream .assistant-message-assistant"),
  ).toHaveCount(0);
  await expect(page.locator("#assistantInput")).toHaveValue("Hi");
  await expect(page.locator("#assistantSendBtn")).toBeEnabled();
});

test("failed secondary results cannot display a combined score", async ({
  page,
}) => {
  await setup(page, {
    result: {
      ...scan,
      multimodal_fusion: {
        ...scan.multimodal_fusion!,
        secondary_error: "Nail image could not be analyzed",
      },
    },
  });
  await openResult(page);
  await expect(page.locator("main")).toContainText(
    "Nail image could not be analyzed",
  );
  await expect(
    page.getByRole("heading", { name: "Weighted combined score", exact: true }),
  ).toHaveCount(0);
});

test("malformed stress responses are rejected without experiment scores", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/v1/achilles-heel", (r) =>
    r.fulfill({
      json: { ...stress, conditions: stress.conditions.slice(0, 3) },
    }),
  );
  await page.goto("/static/ui/#/achilles-heel?scan_id=42");
  await page
    .getByRole("button", { name: "Run stress test", exact: true })
    .click();
  await expect(
    page.getByRole("alert").filter({ hasText: "Experiment not completed" }),
  ).toContainText("invalid stress-test report");
  await expect(
    page.getByRole("heading", {
      name: "What changed across conditions",
      exact: true,
    }),
  ).toHaveCount(0);
});

test("new workspaces explain missing backend capabilities and navigation stays usable", async ({
  page,
}) => {
  await setup(page, { system: { ...ready, capabilities: {} } });
  await page.goto("/static/ui/#/achilles-heel");
  await expect(page.locator("main")).toContainText(
    /not.*(available|enabled)|unavailable/i,
  );
  await expect(
    page.getByRole("button", { name: "Run stress test", exact: true }),
  ).toBeDisabled();
  await page.goto("/static/ui/#/research");
  await expect(page.locator("main")).toContainText(
    "Research data is not available yet",
  );
  const menu = page.getByRole("button", {
    name: "Open navigation",
    exact: true,
  });
  if (await menu.isVisible()) await menu.click();
  await page.locator(".more-navigation summary").click();
  await page.getByRole("link", { name: "Analytics", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Screening analytics", exact: true }),
  ).toBeVisible();
  await noOverflow(page);
});
