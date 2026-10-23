import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Job, PatientDetails, Scan, StageName } from "../../src/types";

const image = readFileSync(resolve("tests/fixtures/test_eye_conjunctiva.png"));
const patient: PatientDetails = {
  reference: "WORKFLOW-024",
  age: 24,
  sex: "not-stated",
  phone: "+91 90000 00000",
  notes: "<b>Plain text</b> & field notes",
};
const names: StageName[] = [
  "validation",
  "detection",
  "roi",
  "normalization",
  "features",
  "inference",
  "explanation",
  "saving",
];
const stages = names.map((name, index) => ({
  name,
  status: index < 3 ? "completed" : index === 3 ? "running" : "pending",
  elapsed_seconds: index < 3 ? 0.1 : 0,
}));
const running: Job = {
  id: "workflow-job",
  status: "running",
  stage: "normalization",
  modality: "conjunctiva",
  patient,
  created_at: "2026-10-23T12:00:00Z",
  updated_at: "2026-10-23T12:00:01Z",
  elapsed_seconds: 1,
  image: { url: "/api/v1/workflow-image", width: 430, height: 225 },
  detection: {
    success: true,
    roi: { x0: 0, y0: 0, x1: 430, y1: 225 },
    type: "pre_cropped_macro",
    confidence: 0.9,
    message: "Review this proposed region.",
    retake_guidance: null,
  },
  roi: { x0: 0, y0: 0, x1: 430, y1: 225 },
  warnings: [],
  stages,
  scan_id: null,
  error: null,
};
const scan: Scan = {
  id: "workflow-result",
  job_id: running.id,
  created_at: running.created_at,
  ...patient,
  modality: "conjunctiva",
  probability_anemia: 0.2,
  predicted_label: "non-anemic",
  hb_g_dl: 10.9,
  regression_status: "validated",
  threshold: 0.5,
  temperature: 1.339,
  model_sha256: "test-only",
  normalization_version: "test-only",
  roi: running.roi!,
  image: { width: 430, height: 225 },
  warnings: [],
  artifacts: {
    oriented: "/api/v1/workflow-image",
    roi: "/api/v1/workflow-image",
    normalized: "/api/v1/workflow-image",
    gradcam: "/api/v1/workflow-image",
  },
  explanation: {
    status: "available",
    message:
      "Grad-CAM highlights model attention; it does not diagnose disease.",
  },
  features: Array(27).fill(0.1),
  processing_seconds: 1,
  follow_up: {
    version: 1,
    title: "Recommended follow-up",
    summary:
      "This image screened negative. This result does not rule out anemia.",
    steps: [
      "Discuss symptoms or concerns with a clinician and ask about a laboratory complete blood count (CBC).",
    ],
    model_input_note:
      "Patient details are saved with this record and are not used by the current model.",
  },
};
async function setup(page: Page) {
  await page.route("**/api/v1/system", (route) =>
    route.fulfill({
      json: {
        api_version: "1",
        patient_intake_version: 1,
        ready: true,
        model: {
          name: "Test model",
          sha256: "test-only",
          temperature: 1.339,
          regression_supported: true,
        },
        detector: { available: true },
        limits: { max_bytes: 10485760, min_side: 50 },
        warnings: [],
      },
    }),
  );
  await page.route("**/api/v1/workflow-image", (route) =>
    route.fulfill({ contentType: "image/png", body: image }),
  );
}
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
  await page.getByLabel("Phone Optional", { exact: true }).fill(patient.phone);
  await page
    .getByLabel("Clinical / field notes Optional", { exact: true })
    .fill(patient.notes);
}

test("intake requires explicit details and sends optional metadata as plain text", async ({
  page,
}, info) => {
  await setup(page);
  let submitted = "";
  await page.route("**/api/v1/jobs", (route) => {
    submitted = route.request().postData() || "";
    return route.fulfill({ status: 202, json: running });
  });
  await page.route("**/api/v1/jobs/workflow-job", (route) =>
    route.fulfill({ json: running }),
  );
  await page.goto("/static/ui/#/studio");
  await page
    .getByLabel("Choose a photo", { exact: true })
    .setInputFiles({ name: "eye.png", mimeType: "image/png", buffer: image });
  const submit = page.getByRole("button", {
    name: "Review image",
    exact: true,
  });
  await expect(submit).toBeDisabled();
  await page
    .getByLabel("Patient name or case ID *", { exact: true })
    .fill("   ");
  await page.getByLabel("Age in years *", { exact: true }).fill("121");
  await expect(
    page.getByText("Use a whole number from 0 to 120."),
  ).toBeVisible();
  await expect(submit).toBeDisabled();
  await fillPatient(page);
  await page.getByLabel("Age in years *", { exact: true }).fill("24.5");
  await expect(submit).toBeDisabled();
  await page.getByLabel("Age in years *", { exact: true }).fill("0");
  await expect(submit).toBeEnabled();
  await expect(
    page.getByLabel("Patient name or case ID *", { exact: true }),
  ).toHaveAttribute("maxlength", "120");
  await expect(
    page.getByLabel("Phone Optional", { exact: true }),
  ).toHaveAttribute("maxlength", "32");
  await expect(
    page.getByLabel("Clinical / field notes Optional", { exact: true }),
  ).toHaveAttribute("maxlength", "2000");
  await page.getByLabel("Age in years *", { exact: true }).fill("24");
  mkdirSync("/tmp/vyra-workflow-qa", { recursive: true });
  await page.screenshot({
    path: `/tmp/vyra-workflow-qa/${info.project.name}-intake.png`,
    fullPage: true,
  });
  await submit.click();
  await expect(page).toHaveURL(/#\/studio\/jobs\/workflow-job$/);
  for (const [name, value] of Object.entries(patient)) {
    expect(submitted).toContain(`name="${name}"\r\n\r\n${value}`);
  }
  await expect(page.locator(".analysis-patient")).toContainText(
    patient.reference,
  );
});

test("an explicit job URL wins over another saved job and its delayed response", async ({
  page,
}) => {
  await setup(page);
  await page.addInitScript(() =>
    sessionStorage.setItem("vyra.active-job.v1", "old-job"),
  );
  let release: (() => void) | undefined;
  let oldRequested = false;
  await page.route("**/api/v1/jobs/old-job", async (route) => {
    oldRequested = true;
    await new Promise<void>((resolve) => {
      release = resolve;
    });
    await route
      .fulfill({
        json: {
          ...running,
          id: "old-job",
          patient: { ...patient, reference: "OLD-PATIENT" },
        },
      })
      .catch(() => {});
  });
  await page.route("**/api/v1/jobs/workflow-job", (route) =>
    route.fulfill({ json: running }),
  );
  await page.route("**/api/v1/scans/workflow-result", (route) =>
    route.fulfill({ json: scan }),
  );
  try {
    await page.goto("/static/ui/#/history/workflow-result");
    await expect.poll(() => oldRequested).toBe(true);
    await expect(
      page.getByRole("heading", { name: "Screen negative", exact: true }),
    ).toBeVisible();
    await page.goto("/static/ui/#/studio/jobs/workflow-job");
    await expect(
      page.getByRole("heading", { name: "Analyzing your image", exact: true }),
    ).toBeVisible();
    const oldResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/jobs/old-job") &&
        response.request().method() === "GET",
    );
    expect(release).toBeDefined();
    release!();
    await oldResponse;
    await expect(page.locator(".analysis-patient")).toContainText(
      "WORKFLOW-024",
    );
    await expect(page.locator(".analysis-patient")).not.toContainText(
      "OLD-PATIENT",
    );
    await expect
      .poll(() =>
        page.evaluate(() => sessionStorage.getItem("vyra.active-job.v1")),
      )
      .toBe("workflow-job");
    await expect(page).toHaveURL(/#\/studio\/jobs\/workflow-job$/);
  } finally {
    release?.();
  }
});

test("progress reload restores patient and completes only server-confirmed stages", async ({
  page,
}, info) => {
  await setup(page);
  let polls = 0,
    resultRequests = 0;
  await page.route("**/api/v1/jobs/workflow-job", (route) => {
    polls++;
    return route.fulfill({ json: running });
  });
  await page.route("**/api/v1/scans/**", (route) => {
    resultRequests++;
    return route.fulfill({ status: 500 });
  });
  await page.goto("/static/ui/#/studio/jobs/workflow-job");
  await expect(page.locator(".analysis-patient")).toContainText("24 years");
  await expect(
    page.locator('.analysis-stages [data-state="complete"]'),
  ).toHaveCount(1);
  await expect(
    page.locator('.analysis-stages [data-state="running"]'),
  ).toHaveCount(1);
  await expect(
    page.locator('.analysis-stages [data-state="pending"]'),
  ).toHaveCount(1);
  await expect.poll(() => polls).toBeGreaterThanOrEqual(3);
  await expect(
    page.locator('.analysis-stages [data-state="complete"]'),
  ).toHaveCount(1);
  await expect(page.getByText("100%", { exact: true })).toHaveCount(0);
  mkdirSync("/tmp/vyra-workflow-qa", { recursive: true });
  await page.screenshot({
    path: `/tmp/vyra-workflow-qa/${info.project.name}-progress.png`,
    fullPage: true,
  });
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Analyzing your image", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".analysis-patient")).toContainText(
    patient.reference,
  );
  await expect(
    page.locator('.analysis-stages [data-state="pending"]'),
  ).toHaveCount(1);
  expect(resultRequests).toBe(0);
});

test("a missing job offers a working status retry without showing unrelated intake", async ({
  page,
}) => {
  await setup(page);
  let available = false;
  await page.route("**/api/v1/jobs/workflow-job", (route) =>
    route.fulfill(
      available
        ? { json: running }
        : {
            status: 404,
            json: {
              error: {
                code: "not_found",
                message: "This screening job no longer exists.",
              },
            },
          },
    ),
  );
  await page.goto("/static/ui/#/studio/jobs/workflow-job");
  await expect(page.getByRole("alert")).toContainText(
    "This screening job no longer exists.",
  );
  await expect(
    page.getByRole("heading", { name: "Choose your image", exact: true }),
  ).toHaveCount(0);
  expect(
    await page.evaluate(() => sessionStorage.getItem("vyra.active-job.v1")),
  ).toBeNull();
  available = true;
  await page.getByRole("button", { name: "Retry status", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Analyzing your image", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".analysis-patient")).toContainText(
    patient.reference,
  );
});

for (const status of ["interrupted", "cancelled", "failed"] as const) {
  test(`${status} progress stays truthful and starts a fresh intake`, async ({
    page,
  }) => {
    await setup(page);
    await page.route("**/api/v1/jobs/workflow-job", (route) =>
      route.fulfill({
        json: {
          ...running,
          status,
          stages: stages.map((s) =>
            s.status === "running" ? { ...s, status } : s,
          ),
          error:
            status === "cancelled"
              ? null
              : {
                  code: status,
                  message: "The screening stopped before a result was saved.",
                },
        },
      }),
    );
    await page.goto("/static/ui/#/studio/jobs/workflow-job");
    const name =
      status === "failed"
        ? "Screening could not finish"
        : `Screening ${status}`;
    await expect(
      page.getByRole("heading", { name, exact: true }),
    ).toBeVisible();
    await expect(page.locator(".analysis-emblem.is-processing")).toHaveCount(0);
    await expect(
      page.locator('.analysis-stages [data-state="complete"]'),
    ).toHaveCount(1);
    await expect(page.locator(".analysis-stages")).toContainText(
      "Not completed",
    );
    await page
      .getByRole("button", { name: "Start again", exact: true })
      .click();
    await expect(page).toHaveURL(/#\/studio$/);
    await expect(
      page.getByRole("heading", { name: "Choose your image", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByLabel("Patient name or case ID *", { exact: true }),
    ).toHaveValue("");
    expect(
      await page.evaluate(() => sessionStorage.getItem("vyra.active-job.v1")),
    ).toBeNull();
  });
}

test("result keeps independent outputs, accessible attention comparison, print and fresh screening", async ({
  page,
}, info) => {
  await setup(page);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() =>
    sessionStorage.setItem("vyra.active-job.v1", "workflow-job"),
  );
  await page.route("**/api/v1/jobs/workflow-job", (route) =>
    route.fulfill({
      json: {
        ...running,
        status: "completed",
        stage: "saving",
        stages: stages.map((s) => ({ ...s, status: "completed" })),
        scan_id: scan.id,
      },
    }),
  );
  await page.route("**/api/v1/scans/workflow-result", (route) =>
    route.fulfill({ json: scan }),
  );
  await page.goto("/static/ui/#/history/workflow-result");
  await expect(
    page.getByRole("heading", { name: "Screen negative", exact: true }),
  ).toBeFocused();
  await expect(page.locator(".hb-reading")).toContainText("10.90");
  await expect(page.locator(".result-columns > section")).toHaveCount(3);
  await expect(page.locator(".patient-record")).toContainText(patient.phone);
  await expect(page.locator(".result-notes")).toContainText(patient.notes);
  await expect(page.locator(".result-notes b")).toHaveCount(0);
  await expect(page.locator(".follow-up-summary")).toHaveText(
    scan.follow_up!.summary,
  );
  await expect(page.getByText("No anemia", { exact: true })).toHaveCount(0);
  const slider = page.getByRole("slider", {
    name: "Attention comparison",
    exact: true,
  });
  await slider.focus();
  await page.keyboard.press("Home");
  await expect(slider).toHaveValue("0");
  await expect(page.locator(".comparison-line")).toHaveCSS("left", "0px");
  await page.keyboard.press("End");
  await expect(slider).toHaveValue("100");
  await expect(
    page.getByAltText("Grad-CAM attention overlay", { exact: true }),
  ).toHaveCSS("clip-path", "inset(0px 0px 0px 100%)");
  await page.evaluate(() => {
    window.print = () => {
      document.body.dataset.printRequested = "true";
    };
  });
  await page.getByRole("button", { name: "Print", exact: true }).click();
  await expect(page.locator("body")).toHaveAttribute(
    "data-print-requested",
    "true",
  );
  await page.emulateMedia({ media: "print" });
  await expect(page.locator(".result-actions")).toBeHidden();
  await expect(slider).toBeHidden();
  await expect(page.locator(".follow-up-panel")).toBeVisible();
  await page.emulateMedia({ media: "screen" });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
  mkdirSync("/tmp/vyra-workflow-qa", { recursive: true });
  await page.screenshot({
    path: `/tmp/vyra-workflow-qa/${info.project.name}-result.png`,
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "New screening", exact: true })
    .click();
  await expect(page).toHaveURL(/#\/studio$/);
  await expect(
    page.getByLabel("Patient name or case ID *", { exact: true }),
  ).toHaveValue("");
  await expect(
    page.getByRole("button", { name: "Review image", exact: true }),
  ).toBeDisabled();
  expect(
    await page.evaluate(() => sessionStorage.getItem("vyra.active-job.v1")),
  ).toBeNull();
  expect(errors).toEqual([]);
});

test("new screening escapes a pending saved-job restore and ignores its late response", async ({
  page,
}) => {
  await setup(page);
  await page.addInitScript(() =>
    sessionStorage.setItem("vyra.active-job.v1", "pending-restore"),
  );
  let release: (() => void) | undefined;
  let requested = false;
  await page.route("**/api/v1/jobs/pending-restore", async (route) => {
    requested = true;
    await new Promise<void>((resolve) => {
      release = resolve;
    });
    await route
      .fulfill({ json: { ...running, id: "pending-restore" } })
      .catch(() => {});
  });
  await page.route("**/api/v1/scans/workflow-result", (route) =>
    route.fulfill({ json: scan }),
  );
  try {
    await page.goto("/static/ui/#/history/workflow-result");
    await expect.poll(() => requested).toBe(true);
    await expect(
      page.getByRole("heading", { name: "Screen negative", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "New screening", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Choose your image", exact: true }),
    ).toBeVisible();
    await expect(page).toHaveURL(/#\/studio$/);
    const lateResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/jobs/pending-restore") &&
        response.request().method() === "GET",
    );
    release?.();
    await lateResponse;
    await expect(
      page.getByRole("heading", { name: "Choose your image", exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Analyzing your image", exact: true }),
    ).toHaveCount(0);
    expect(
      await page.evaluate(() => sessionStorage.getItem("vyra.active-job.v1")),
    ).toBeNull();
  } finally {
    release?.();
  }
});
