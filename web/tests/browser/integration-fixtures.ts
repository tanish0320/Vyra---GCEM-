/** Test-only data. Never imported by src/ or seeded into a running service. */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Page } from "@playwright/test";
import type { Job, Scan, SystemStatus } from "../../src/types";
import type { StressReport } from "../../src/integration";
export const photo = readFileSync(
  resolve("tests/fixtures/test_eye_conjunctiva.png"),
);
const suppliedReport = JSON.parse(
  readFileSync(resolve("tests/fixtures/champion_test_report.json"), "utf8"),
);
export const ready: SystemStatus = {
  api_version: "1",
  patient_intake_version: 1,
  ready: true,
  capabilities: {
    dual_scan: true,
    assistant: true,
    stress_test: true,
    colorimetry: true,
    research: true,
  },
  model: {
    name: "Fixture service only",
    sha256: "test-only",
    temperature: 1.339,
    regression_supported: true,
  },
  detector: { available: true },
  limits: { max_bytes: 10485760, min_side: 50 },
  warnings: [],
};
export const patient = {
  reference: "UI-INTEGRATION-ONLY",
  age: 24,
  sex: "not-stated" as const,
  phone: "",
  notes: "Review note",
};
const detection = {
  success: true,
  roi: { x0: 0, y0: 0, x1: 430, y1: 225 },
  type: "fixture",
  confidence: 0.9,
  message: "Review this region",
  retake_guidance: null,
};
export const job: Job = {
  id: "ui-job",
  status: "awaiting_roi",
  stage: "roi",
  modality: "conjunctiva",
  patient,
  created_at: "2026-10-24T01:00:00Z",
  updated_at: "2026-10-24T01:00:01Z",
  elapsed_seconds: 1,
  image: { url: "/api/v1/artifacts/ui-job/oriented", width: 430, height: 225 },
  detection,
  roi: detection.roi,
  warnings: [],
  stages: [],
  scan_id: null,
  error: null,
  dual_scan: true,
  secondary: {
    modality: "fingernail",
    image: {
      url: "/api/v1/artifacts/ui-job/oriented",
      width: 430,
      height: 225,
    },
    detection,
    roi: detection.roi,
    error: null,
  },
};
export const scan: Scan = {
  id: "42",
  job_id: job.id,
  created_at: job.created_at,
  ...patient,
  modality: "conjunctiva",
  probability_anemia: 0.608464,
  predicted_label: "anemic",
  hb_g_dl: 10.77,
  regression_status: "validated",
  threshold: 0.5,
  temperature: 1.339,
  model_sha256: "test-only",
  normalization_version: "test-only",
  roi: detection.roi,
  image: { width: 430, height: 225 },
  warnings: [],
  features: Array(27).fill(0.1),
  processing_seconds: 1,
  artifacts: {
    oriented: "/api/v1/artifacts/ui-job/oriented",
    roi: "/api/v1/artifacts/ui-job/oriented",
    normalized: "/api/v1/artifacts/ui-job/oriented",
    gradcam: "/api/v1/artifacts/ui-job/oriented",
    heatmap: "/api/v1/artifacts/ui-job/oriented",
  },
  explanation: { status: "available", message: "Test-only attention output" },
  pdf_url: "/export/pdf/42",
  biophysical_metrics: {
    has_real_pixels: true,
    erythema_index: 38.2,
    l_star: 58.1,
    a_star: 16.4,
    b_star: 12.3,
    source: "Confirmed tissue crop",
  },
  multimodal_fusion: {
    active: true,
    primary_modality: "Eye Conjunctiva",
    primary_prob: 60.8,
    primary_class: "anemic",
    secondary_modality: "Fingernail Bed",
    secondary_prob: 56.7,
    secondary_class: "anemic",
    joint_probability: 59.0,
    concordance_pct: 95.9,
    agreement_level: "High Cross-Site Agreement",
    method: "Weighted eye/nail score",
  },
};
export const stress: StressReport = {
  modality: "conjunctiva",
  baseline_probability: 60,
  baseline_class: "anemic",
  conditions: [
    "baseline",
    "raw_image",
    "tissue_only",
    "tissue_occluded",
    "warm",
    "cool",
    "blur",
    "border",
  ].map((id, i) => ({
    condition_id: id,
    name: [
      "Baseline ROI",
      "Full uncropped image",
      "Isolated tissue",
      "Tissue occluded",
      "Warm simulation",
      "Cool simulation",
      "Defocus blur",
      "Border crop",
    ][i],
    category: i ? "Perturbation" : "Baseline",
    probability: 0.6 - i * 0.025,
    predicted_class: i < 4 ? "anemic" : "non-anemic",
    delta_prob: i * 0.025,
    percent_shift: i * 2.5,
    thumbnail_b64: "data:image/png;base64," + photo.toString("base64"),
    description: "Controlled test fixture transformation",
  })),
  ranked_failure_mechanisms: [
    "Background appearance",
    "Context leakage",
    "Defocus",
    "Border framing",
    "Colour shift",
  ].map((name, i) => ({
    id: "mechanism-" + i,
    name,
    risk_level: "Moderate",
    sensitivity_score: 40 - i * 5,
    evidence: "Fixture probability changed under this condition",
    mechanism_detail: "Test evidence",
    clinical_implication: "Research result, not a diagnosis",
  })),
  overall_shortcut_index: 30,
  overall_status: "Moderate test sensitivity",
  overall_class: "risk-moderate",
  big_reveal_note: "Fixture research report; does not prove a causal mechanism",
};
export async function setup(
  page: Page,
  options: { system?: SystemStatus; result?: Scan } = {},
) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/system")
      return route.fulfill({ json: options.system || ready });
    if (url.pathname === "/api/v1/evidence")
      return route.fulfill({
        json: {
          ...suppliedReport,
          checkpoint_sha256: "test-only",
          normalization_version: "test-only",
          temperature: suppliedReport.temperature_scaling_T,
          provenance:
            "Test-only UI projection of the supplied champion evaluation; no new evaluation performed.",
          limitations: ["Live backend evaluation is outside this UI test."],
        },
      });
    if (url.pathname === "/api/v1/artifacts/ui-job/oriented")
      return route.fulfill({ body: photo, contentType: "image/png" });
    if (url.pathname === "/api/v1/scans/42")
      return route.fulfill({ json: options.result || scan });
    if (url.pathname === "/api/v1/scans")
      return route.fulfill({ json: { scans: [options.result || scan] } });
    if (url.pathname === "/api/v1/jobs/ui-job")
      return route.fulfill({ json: job });
    if (url.pathname === "/api/v1/analytics")
      return route.fulfill({
        json: {
          total_scans: 1,
          by_modality: { conjunctiva: 1, fingernail: 0 },
          screen_positive: 1,
          screen_negative: 0,
          mean_probability: 0.608464,
          eye_hb_count: 1,
          mean_eye_hb: 10.77,
        },
      });
    return route.fulfill({
      status: 404,
      json: {
        error: {
          code: "NOT_AVAILABLE",
          message: "This test endpoint is unavailable",
        },
      },
    });
  });
}
