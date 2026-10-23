import { expect, test, type Page } from "@playwright/test";
import { readFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
const sample = readFileSync(resolve("tests/fixtures/test_eye_conjunctiva.png"));
const ready = {
  api_version: "1",
  patient_intake_version: 1,
  ready: true,
  model: {
    name: "Verified test service",
    sha256: "test-only",
    temperature: 1.339,
    regression_supported: true,
  },
  detector: { available: true },
  limits: { max_bytes: 10 * 1024 * 1024, min_side: 50 },
  warnings: [],
};
const patient = {
  reference: "BROWSER-CASE-001",
  age: 24,
  sex: "not-stated",
  phone: "",
  notes: "",
};
async function fillPatient(page: Page) {
  await page
    .getByLabel("Patient name or case ID *", { exact: true })
    .fill(patient.reference);
  await page
    .getByLabel("Age in years *", { exact: true })
    .fill(String(patient.age));
  await page
    .getByLabel("Biological sex *", { exact: true })
    .selectOption(patient.sex);
}
async function mockReadiness(page: Page) {
  await page.route("**/api/v1/system", (route) =>
    route.fulfill({ json: ready }),
  );
}
test("home is readable, responsive, and links to each workspace", async ({
  page,
}, info) => {
  await mockReadiness(page);
  await page.goto("/static/ui/");
  await expect(page).toHaveTitle(/Vyra/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "anemia screening",
  );
  await expect(
    page
      .locator("main")
      .getByRole("link", { name: "Start screening", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
  mkdirSync("/tmp/vyra-browser-evidence", { recursive: true });
  await page.screenshot({
    path: `/tmp/vyra-browser-evidence/${info.project.name}-home.png`,
    fullPage: true,
  });
  await page
    .locator("main")
    .getByRole("link", { name: "Start screening", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Screening studio", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Review image", exact: true }),
  ).toBeDisabled();
});
test("unavailable model prevents detection and exposes retry", async ({
  page,
}) => {
  await page.route("**/api/v1/system", (route) =>
    route.fulfill({
      json: {
        ...ready,
        ready: false,
        warnings: ["Checkpoint verification failed."],
      },
    }),
  );
  await page.goto("/static/ui/#/studio");
  await expect(page.getByRole("alert")).toContainText(
    "Checkpoint verification failed.",
  );
  await fillPatient(page);
  await page
    .getByLabel("Choose a photo", { exact: true })
    .setInputFiles({ name: "eye.png", mimeType: "image/png", buffer: sample });
  await expect(
    page.getByRole("button", { name: "Review image", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Check connection" }),
  ).toBeVisible();
});
test("invalid upload stays local and does not create a screening", async ({
  page,
}) => {
  await mockReadiness(page);
  let posts = 0;
  await page.route("**/api/v1/jobs", (route) => {
    posts++;
    return route.fulfill({
      status: 500,
      json: { error: { message: "Unexpected upload" } },
    });
  });
  await page.goto("/static/ui/#/studio");
  await page.getByLabel("Choose a photo", { exact: true }).setInputFiles({
    name: "script.svg",
    mimeType: "image/svg+xml",
    buffer: Buffer.from("<svg/>"),
  });
  await expect(page.getByRole("alert")).toContainText(
    "Choose a JPEG, PNG, or WebP image.",
  );
  await expect(
    page.getByRole("button", { name: "Review image", exact: true }),
  ).toBeDisabled();
  expect(posts).toBe(0);
});
test("failed automatic detection requires deliberate manual region review", async ({
  page,
}) => {
  await mockReadiness(page);
  const image = "/api/v1/artifacts/fixture/oriented";
  await page.route(`**${image}`, (route) =>
    route.fulfill({ contentType: "image/png", body: sample }),
  );
  const fixture = {
    id: "fixture",
    patient,
    status: "awaiting_roi",
    stage: "roi",
    modality: "conjunctiva",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    elapsed_seconds: 0.15,
    image: { url: image, width: 430, height: 225 },
    detection: {
      success: false,
      roi: null,
      type: "detection_failed",
      confidence: 0,
      message: "No reliable region found.",
      retake_guidance: "Retake a focused photo.",
    },
    roi: null,
    warnings: [],
    stages: [],
    scan_id: null,
    error: null,
  };
  await page.route("**/api/v1/jobs", (route) =>
    route.fulfill({ status: 202, json: fixture }),
  );
  let confirmations = 0;
  await page.route("**/api/v1/jobs/fixture/confirm", (route) => {
    confirmations++;
    return route.fulfill({
      status: 202,
      json: { ...fixture, status: "running", stage: "normalization" },
    });
  });
  await page.route("**/api/v1/jobs/fixture", (route) =>
    route.fulfill({
      json: { ...fixture, status: "running", stage: "normalization" },
    }),
  );
  await page.goto("/static/ui/#/studio");
  await fillPatient(page);
  await page
    .getByLabel("Choose a photo", { exact: true })
    .setInputFiles({ name: "eye.png", mimeType: "image/png", buffer: sample });
  await page.getByRole("button", { name: "Review image", exact: true }).click();
  await expect(
    page.getByText("Manual review needed", { exact: true }),
  ).toBeVisible();
  const confirm = page.getByRole("button", {
    name: "Analyze image",
    exact: true,
  });
  await expect(confirm).toBeDisabled();
  expect(confirmations).toBe(0);
  await page
    .getByLabel(
      "I have selected the exposed lower eyelid or nail bed shown in this photo.",
    )
    .check();
  await expect(confirm).toBeEnabled();
  await page.getByLabel("ROI Width", { exact: true }).fill("20");
  await expect(confirm).toBeDisabled();
  await expect(page.getByRole("alert").last()).toContainText("at least 50");
  await page.getByLabel("ROI Width", { exact: true }).fill("100");
  await expect(confirm).toBeEnabled();
  await confirm.click();
  await expect(
    page.getByRole("heading", { name: "Analyzing your image" }),
  ).toBeVisible();
  expect(confirmations).toBe(1);
});
test("empty history is truthful and filter controls work", async ({ page }) => {
  await mockReadiness(page);
  await page.route("**/api/v1/scans", (route) =>
    route.fulfill({ json: { scans: [] } }),
  );
  await page.route("**/api/v1/analytics", (route) =>
    route.fulfill({
      json: {
        total_scans: 0,
        by_modality: { conjunctiva: 0, fingernail: 0 },
        screen_positive: 0,
        screen_negative: 0,
        mean_probability: null,
        eye_hb_count: 0,
        mean_eye_hb: null,
      },
    }),
  );
  await page.goto("/static/ui/#/history");
  await expect(
    page.getByText("No screenings saved yet", { exact: true }),
  ).toBeVisible();
  await page
    .getByLabel("Search case reference or ID", { exact: true })
    .fill("missing");
  await expect(
    page.getByText("No matching screenings", { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
});
test("camera dialog handles missing permission and closes with keyboard", async ({
  page,
}) => {
  await mockReadiness(page);
  await page.addInitScript(() => {
    if (navigator.mediaDevices)
      navigator.mediaDevices.getUserMedia = () =>
        Promise.reject(new DOMException("Denied", "NotAllowedError"));
  });
  await page.goto("/static/ui/#/studio");
  await page.getByRole("button", { name: "Use camera", exact: true }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("alert")).toContainText("Camera unavailable");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Use camera", exact: true }),
  ).toBeFocused();
});

test("retrying an upload reuses its idempotency key", async ({ page }) => {
  await mockReadiness(page);
  const keys: string[] = [];
  await page.route("**/api/v1/jobs", (route) => {
    keys.push(route.request().headers()["idempotency-key"]);
    return route.fulfill({
      status: 503,
      json: {
        error: {
          code: "RETRY",
          message: "Temporary test service interruption.",
        },
      },
    });
  });
  await page.goto("/static/ui/#/studio");
  await fillPatient(page);
  await page
    .getByLabel("Choose a photo", { exact: true })
    .setInputFiles({ name: "eye.png", mimeType: "image/png", buffer: sample });
  await page.getByRole("button", { name: "Review image", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Temporary test service interruption.",
  );
  await page.getByRole("button", { name: "Review image", exact: true }).click();
  await expect.poll(() => keys.length).toBe(2);
  expect(keys[0]).toBeTruthy();
  expect(keys[1]).toBe(keys[0]);
});

test("archiving the active saved result clears it from the studio", async ({
  page,
}) => {
  await mockReadiness(page);
  const scan = {
    id: "delete-fixture",
    job_id: "delete-job",
    reference: "UI-DELETE-TEST",
    age: 24,
    sex: "not-stated",
    phone: "",
    notes: "",
    modality: "fingernail",
    created_at: "2026-10-23T12:00:00Z",
    probability_anemia: 0.56,
    predicted_label: "anemic",
    hb_g_dl: null,
    regression_status: "unavailable",
    threshold: 0.5,
    temperature: 1.339,
    model_sha256: "test-checkpoint",
    normalization_version: "test-policy",
    roi: { x0: 0, y0: 0, x1: 70, y1: 58 },
    image: { width: 70, height: 58 },
    warnings: [],
    artifacts: {
      oriented: "/api/v1/test-image",
      roi: "/api/v1/test-image",
      normalized: "/api/v1/test-image",
      gradcam: null,
    },
    explanation: { status: "unavailable", message: "Test fixture." },
    features: Array(27).fill(0.1),
    processing_seconds: 0.5,
  };
  const job = {
    id: "delete-job",
    patient: { ...patient, reference: scan.reference },
    status: "completed",
    stage: "saving",
    modality: "fingernail",
    scan_id: scan.id,
    created_at: scan.created_at,
    updated_at: scan.created_at,
    elapsed_seconds: 0.5,
    image: null,
    detection: null,
    roi: scan.roi,
    stages: [],
    warnings: [],
    error: null,
  };
  let deleted = false;
  await page.addInitScript(() =>
    sessionStorage.setItem("vyra.active-job.v1", "delete-job"),
  );
  await page.route("**/api/v1/jobs/delete-job", (route) =>
    route.fulfill({ json: job }),
  );
  await page.route("**/api/v1/scans/delete-fixture", (route) => {
    if (route.request().method() === "DELETE") {
      deleted = true;
      return route.fulfill({ status: 204 });
    }
    return route.fulfill({ json: scan });
  });
  await page.route("**/api/v1/scans", (route) =>
    route.fulfill({ json: { scans: deleted ? [] : [scan] } }),
  );
  await page.route("**/api/v1/analytics", (route) =>
    route.fulfill({
      json: {
        total_scans: deleted ? 0 : 1,
        by_modality: { conjunctiva: 0, fingernail: deleted ? 0 : 1 },
        screen_positive: deleted ? 0 : 1,
        screen_negative: 0,
        eye_hb_count: 0,
        mean_eye_hb: null,
      },
    }),
  );
  await page.route("**/api/v1/test-image", (route) =>
    route.fulfill({ contentType: "image/png", body: sample }),
  );
  await page.goto("/static/ui/#/studio");
  await expect(
    page.getByRole("heading", { name: "Screen positive", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Open screening history" }).click();
  await page
    .getByRole("button", { name: "Archive UI-DELETE-TEST", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Archive screening", exact: true })
    .click();
  await expect(
    page.getByText("No screenings saved yet", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("link", { name: "Open screening studio", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Choose your image",
      exact: true,
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => sessionStorage.getItem("vyra.active-job.v1")),
  ).toBeNull();
});

test("a delayed status retry cannot resurrect a cancelled screening", async ({
  page,
}) => {
  await mockReadiness(page);
  const fixture = {
    id: "retry-race",
    patient,
    status: "awaiting_roi",
    stage: "roi",
    modality: "conjunctiva",
    created_at: "2026-10-23T12:00:00Z",
    updated_at: "2026-10-23T12:00:00Z",
    elapsed_seconds: 0.2,
    image: { url: "/api/v1/race-image", width: 430, height: 225 },
    detection: {
      success: true,
      roi: { x0: 0, y0: 0, x1: 430, y1: 225 },
      type: "pre_cropped_macro",
      confidence: 0.9,
      message: "Review this proposed region.",
      retake_guidance: null,
    },
    roi: null,
    warnings: [],
    stages: [],
    scan_id: null,
    error: null,
  };
  await page.route("**/api/v1/race-image", (route) =>
    route.fulfill({ contentType: "image/png", body: sample }),
  );
  await page.route("**/api/v1/jobs", (route) =>
    route.fulfill({ status: 202, json: fixture }),
  );
  await page.route("**/api/v1/jobs/retry-race/confirm", (route) =>
    route.fulfill({
      status: 503,
      json: {
        error: { code: "TEST", message: "Temporary confirmation failure." },
      },
    }),
  );
  let release: (() => void) | undefined;
  let retryRequested = false;
  await page.route("**/api/v1/jobs/retry-race", async (route) => {
    retryRequested = true;
    await new Promise<void>((resolve) => {
      release = resolve;
    });
    await route.fulfill({ json: fixture });
  });
  await page.route("**/api/v1/jobs/retry-race/cancel", (route) =>
    route.fulfill({ json: { ...fixture, status: "cancelled", image: null } }),
  );
  await page.goto("/static/ui/#/studio");
  await fillPatient(page);
  await page
    .getByLabel("Choose a photo", { exact: true })
    .setInputFiles({ name: "eye.png", mimeType: "image/png", buffer: sample });
  await page.getByRole("button", { name: "Review image", exact: true }).click();
  await page
    .getByRole("button", { name: "Analyze image", exact: true })
    .click();
  await expect(page.getByRole("alert").first()).toContainText(
    "Temporary confirmation failure.",
  );
  await page.getByRole("button", { name: "Retry status", exact: true }).click();
  await expect.poll(() => retryRequested).toBe(true);
  await page
    .getByRole("button", { name: "Cancel screening", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Screening cancelled", exact: true }),
  ).toBeVisible();
  const oldResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/jobs/retry-race") &&
      response.request().method() === "GET",
  );
  release?.();
  await oldResponse;
  await page.getByRole("button", { name: "Start again", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Choose your image",
      exact: true,
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => sessionStorage.getItem("vyra.active-job.v1")),
  ).toBeNull();
});

test("an earlier running backend cannot silently discard the new patient fields", async ({
  page,
}) => {
  const { patient_intake_version, ...olderService } = ready;
  await page.route("**/api/v1/system", (route) =>
    route.fulfill({ json: olderService }),
  );
  let submissions = 0;
  await page.route("**/api/v1/jobs", (route) => {
    submissions++;
    return route.abort();
  });
  await page.goto("/static/ui/#/studio");
  await page
    .getByLabel("Patient name or case ID *", { exact: true })
    .fill("RESTART-TEST");
  await page.getByLabel("Age in years *", { exact: true }).fill("24");
  await page
    .getByLabel("Biological sex *", { exact: true })
    .selectOption("not-stated");
  await page
    .getByLabel("Choose a photo", { exact: true })
    .setInputFiles({ name: "eye.png", mimeType: "image/png", buffer: sample });
  await expect(page.getByRole("alert")).toContainText(
    "Updated backend required",
  );
  await expect(
    page.getByRole("button", { name: "Review image", exact: true }),
  ).toBeDisabled();
  expect(submissions).toBe(0);
});
