import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";

const API = "/api/v1";

for (const modality of ["conjunctiva", "fingernail"] as const) {
  test(`live model: ${modality} upload, artifacts, reports and deletion`, async ({
    page,
    request,
  }, info) => {
    const ready = await request.get(`${API}/system`);
    expect(
      ready.ok(),
      "Start the local API before running the live browser suite.",
    ).toBeTruthy();
    expect((await ready.json()).ready).toBe(true);

    const errors: string[] = [];
    const external: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("request", (req) => {
      const url = new URL(req.url());
      if (
        url.protocol.startsWith("http") &&
        !["127.0.0.1", "localhost"].includes(url.hostname)
      )
        external.push(req.url());
    });
    let scanId: string | undefined;
    let jobId: string | undefined;
    try {
      await page.goto("/static/ui/#/studio");
      await expect(
        page.getByText("Local model ready", { exact: true }),
      ).toBeVisible();
      if (modality === "fingernail")
        await page
          .getByRole("radio", { name: "Fingernail", exact: true })
          .click();
      await page
        .getByRole("button", { name: "Use reference photo", exact: true })
        .click();
      const reference = `QA-${info.project.name}-${modality}`;
      await page
        .getByLabel("Patient name or case ID *", { exact: true })
        .fill(reference);
      await page.getByLabel("Age in years *", { exact: true }).fill("24");
      await page
        .getByLabel("Biological sex *", { exact: true })
        .selectOption("not-stated");
      await page
        .getByLabel("Phone Optional", { exact: true })
        .fill("+91 90000 00000");
      await page
        .getByLabel("Clinical / field notes Optional", { exact: true })
        .fill("Automated QA; synthetic case details.");
      const detect = page.getByRole("button", {
        name: "Review image",
        exact: true,
      });
      await expect(detect).toBeEnabled();
      const created = page.waitForResponse(
        (res) =>
          res.url().endsWith(`${API}/jobs`) &&
          res.request().method() === "POST",
      );
      await detect.click();
      const uploaded = await created;
      expect(uploaded.status()).toBe(202);
      const initialJob = await uploaded.json();
      jobId = initialJob.id;
      expect(initialJob.patient).toEqual({
        reference,
        age: 24,
        sex: "not-stated",
        phone: "+91 90000 00000",
        notes: "Automated QA; synthetic case details.",
      });
      await expect(page).toHaveURL(new RegExp(`/studio/jobs/${jobId}$`));
      await expect(
        page.getByRole("heading", { name: "Confirm the screening region" }),
      ).toBeVisible();
      await expect(
        page.getByRole("heading", { name: "Confirm the screening region" }),
      ).toBeFocused();
      await expect(
        page.getByText("Automatic region proposal", { exact: true }),
      ).toBeVisible();
      mkdirSync("/tmp/vyra-redesign-qa", { recursive: true });
      await page.screenshot({
        path: `/tmp/vyra-redesign-qa/${info.project.name}-${modality}-review.png`,
        fullPage: true,
      });
      await page
        .getByRole("button", { name: "Analyze image", exact: true })
        .click();
      await expect(
        page.getByRole("heading", { name: "Screen positive", exact: true }),
      ).toBeVisible({ timeout: 25_000 });
      await expect(
        page.getByRole("heading", { name: "Screen positive", exact: true }),
      ).toBeFocused();

      const job = await (await request.get(`${API}/jobs/${jobId}`)).json();
      expect(job.status).toBe("completed");
      expect(
        job.stages.every(
          (stage: { status: string }) => stage.status === "completed",
        ),
      ).toBe(true);
      scanId = job.scan_id;
      const scan = await (await request.get(`${API}/scans/${scanId}`)).json();
      await expect(page).toHaveURL(new RegExp(`/history/${scanId}$`));
      expect(scan.reference).toBe(reference);
      expect(scan.age).toBe(24);
      expect(scan.sex).toBe("not-stated");
      expect(scan.phone).toBe("+91 90000 00000");
      expect(scan.notes).toBe("Automated QA; synthetic case details.");
      expect(scan.follow_up.model_input_note).toContain(
        "not used by the current model",
      );
      await expect(page.locator(".patient-record")).toContainText(reference);
      await expect(page.locator(".follow-up-summary")).toHaveText(
        scan.follow_up.summary,
      );
      expect(scan.features).toHaveLength(27);
      expect(scan.probability_anemia).toBeCloseTo(
        modality === "conjunctiva" ? 0.608464 : 0.566868,
        3,
      );
      expect(scan.hb_g_dl).toBe(modality === "conjunctiva" ? 10.77 : null);
      expect(scan.explanation.status).toBe("available");
      expect(scan.input_fingerprint).toBe(scan.explanation.input_fingerprint);
      if (modality === "conjunctiva") {
        await expect(page.locator(".hb-reading")).toContainText("10.77");
      } else {
        await expect(page.locator(".hb-reading")).toContainText(
          "Unavailable for nails",
        );
      }
      await page
        .getByRole("button", { name: "Enlarge Normalized input", exact: true })
        .click();
      await expect(page.getByRole("dialog")).toBeVisible();
      await expect(
        page.getByAltText("Normalized input, enlarged"),
      ).toBeVisible();
      await page.keyboard.press("Escape");
      await expect(page.getByRole("dialog")).toHaveCount(0);
      await expect(
        page.getByRole("button", {
          name: "Enlarge Normalized input",
          exact: true,
        }),
      ).toBeFocused();
      for (const url of Object.values(scan.artifacts) as string[]) {
        const artifact = await request.get(url);
        expect(artifact.status()).toBe(200);
        expect(artifact.headers()["content-type"]).toContain("image/png");
      }
      await page
        .getByText("Inspect 27 color features and model details", {
          exact: true,
        })
        .click();
      await expect(page.locator(".feature-table")).toBeVisible();
      await expect(page.locator(".feature-table > div")).toHaveCount(27);
      const pdf = await request.get(`${API}/scans/${scanId}/report.pdf`);
      expect((await pdf.body()).subarray(0, 4).toString()).toBe("%PDF");
      const csv = await request.get(`${API}/scans.csv`);
      const csvText = await csv.text();
      expect(csvText).toContain(scanId);
      expect(csvText).toContain(reference);
      expect(csvText).toContain("Automated QA; synthetic case details.");
      await page.reload();
      await expect(
        page.getByRole("heading", { name: "Screen positive", exact: true }),
      ).toBeVisible();
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
      mkdirSync("/tmp/vyra-browser-evidence", { recursive: true });
      await page.screenshot({
        path: `/tmp/vyra-browser-evidence/${info.project.name}-${modality}-result.png`,
        fullPage: true,
      });
      await page.getByRole("link", { name: "Open screening history" }).click();
      await expect(
        page.getByRole("heading", { name: /screening history/i }),
      ).toBeVisible();
      expect(errors).toEqual([]);
      expect(
        external,
        "The local workflow must not fetch external resources.",
      ).toEqual([]);
    } finally {
      if (scanId) {
        const removed = await request.delete(`${API}/scans/${scanId}`);
        expect(removed.status()).toBe(204);
        expect((await request.get(`${API}/scans/${scanId}`)).status()).toBe(
          404,
        );
        expect(
          (await request.get(`${API}/artifacts/${jobId}/oriented`)).status(),
        ).toBe(404);
      } else if (jobId) {
        await request.post(`${API}/jobs/${jobId}/cancel`);
      }
    }
  });
}
