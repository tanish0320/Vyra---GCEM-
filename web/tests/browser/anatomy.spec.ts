import { expect, test, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
const qaDirectory = "/tmp/vyra-atlas-qa";
mkdirSync(qaDirectory, { recursive: true });
async function openExplorer(page: Page) {
  await page.goto("/static/ui/#/anatomy");
  await page.locator(".anatomy-render-stage").scrollIntoViewIfNeeded();
  await expect(page.getByTestId("anatomy-canvas")).toHaveAttribute(
    "data-ready",
    "true",
    { timeout: 30000 },
  );
  await page.evaluate(() => document.fonts.ready);
}
async function structure(page: Page, name: string) {
  const trigger = page.getByRole("button", { name: "Structures", exact: true });
  if (
    (await trigger.isVisible()) &&
    (await trigger.getAttribute("aria-expanded")) === "false"
  )
    await trigger.click();
  await page
    .locator(".atlas-structure-name")
    .filter({ hasText: new RegExp(`^${name}$`) })
    .click();
}

test("focused eye tissues and complete nail layers render without refetching assets", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const assetRequests: string[] = [];
  page.on("request", (request) => {
    if (/\/anatomy\/(eye|nail)\.glb$/.test(request.url()))
      assetRequests.push(request.url());
  });
  await openExplorer(page);
  const canvas = page.getByTestId("anatomy-canvas");
  const view = page.locator(".anatomy-view").first();
  const eyeRequests = assetRequests.filter((url) =>
    url.endsWith("/eye.glb"),
  ).length;
  const expectedEyeParts = [
    "conjunctiva",
    "upper-conjunctiva",
    "eyelids",
    "sclera",
    "limbus",
    "cornea",
    "iris",
    "pupil",
  ].sort();
  const actualParts = async () =>
    (await view.getAttribute("data-visible-parts"))!.split(",").sort();
  await expect(canvas).toHaveAttribute("data-view", "tissue");
  await expect(page.locator(".atlas-inspector h2")).toHaveText(
    "Lower conjunctiva",
  );
  await expect.poll(actualParts).toEqual(expectedEyeParts);
  await expect(page.locator(".atlas-structure-name")).toHaveText([
    "Lower conjunctiva",
    "Upper conjunctiva",
  ]);
  await expect(
    page.getByRole("radio", { name: "Exploded", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("radio", { name: "Cutaway", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("slider", { name: "Layer separation", exact: true }),
  ).toHaveCount(0);
  await page.locator(".atlas-experience").screenshot({
    path: `${qaDirectory}/${info.project.name}-eye-tissue.png`,
    animations: "disabled",
  });
  await structure(page, "Upper conjunctiva");
  await expect(page.locator(".atlas-inspector h2")).toHaveText(
    "Upper conjunctiva",
  );
  await expect(canvas).toHaveAttribute("data-view", "tissue");
  await view.click({
    position: {
      x: (await view.boundingBox())!.width / 2,
      y: (await view.boundingBox())!.height / 2,
    },
  });
  await expect(page.locator(".atlas-inspector h2")).toHaveText(
    "Upper conjunctiva",
  );
  await page.getByRole("button", { name: "Isolate", exact: true }).click();
  await expect(view).toHaveAttribute("data-visible-parts", "upper-conjunctiva");
  await page.getByRole("button", { name: "Show all", exact: true }).click();
  await expect.poll(actualParts).toEqual(expectedEyeParts);
  await page
    .getByRole("radio", { name: "Pallor comparison", exact: true })
    .click();
  await expect(canvas).toHaveAttribute("data-view", "pallor");
  await expect(page.locator(".atlas-inspector h2")).toHaveText(
    "Upper conjunctiva",
  );
  await expect.poll(actualParts).toEqual(expectedEyeParts);
  await page.getByRole("radio", { name: "Tissue view", exact: true }).click();
  expect(assetRequests.filter((url) => url.endsWith("/eye.glb"))).toHaveLength(
    eyeRequests,
  );

  await page.getByRole("radio", { name: "The nail", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-ready", "true", {
    timeout: 30000,
  });
  await expect(view).toHaveAttribute("data-visible-parts", /nail-bed/);
  await expect(canvas).toHaveAttribute("data-view", "exploded");
  const before = await page.locator(".anatomy-webgl").screenshot();
  await page
    .getByRole("slider", { name: "Layer separation", exact: true })
    .fill("95");
  await expect
    .poll(async () => Number(await canvas.getAttribute("data-explosion")))
    .toBeGreaterThan(0.94);
  expect(
    (await page.locator(".anatomy-webgl").screenshot()).equals(before),
  ).toBe(false);
  await structure(page, "Nail matrix");
  await expect(page.locator(".atlas-inspector h2")).toHaveText("Nail matrix");
  await page.getByRole("button", { name: "Isolate", exact: true }).click();
  await expect(view).toHaveAttribute("data-visible-parts", "matrix");
  await page.getByRole("button", { name: "Show all", exact: true }).click();
  await page.getByRole("radio", { name: "Assembled", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-view", "assembled");
  await expect
    .poll(async () => Number(await canvas.getAttribute("data-explosion")))
    .toBe(0);
  await page.getByRole("radio", { name: "Cutaway", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-view", "cutaway");
  await expect(view).toHaveAttribute("data-visible-parts", /distal-phalanx/);
  await page.locator(".atlas-experience").screenshot({
    path: `${qaDirectory}/${info.project.name}-nail-cutaway.png`,
    animations: "disabled",
  });
  await page.getByRole("radio", { name: "Exploded", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-view", "exploded");
  await page.locator(".atlas-experience").screenshot({
    path: `${qaDirectory}/${info.project.name}-nail-exploded.png`,
    animations: "disabled",
  });
  await page.getByRole("radio", { name: "The eye", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-view", "tissue");
  await expect.poll(actualParts).toEqual(expectedEyeParts);
  await expect(page.locator(".atlas-inspector h2")).toHaveText(
    "Lower conjunctiva",
  );
  expect(assetRequests.filter((url) => url.endsWith("/eye.glb"))).toHaveLength(
    eyeRequests,
  );
  expect(assetRequests.filter((url) => url.endsWith("/nail.glb"))).toHaveLength(
    1,
  );
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
  expect(errors).toEqual([]);
});

test("pallor comparison synchronizes orbit, keyboard, zoom and appearance state", async ({
  page,
}, info) => {
  await openExplorer(page);
  await page
    .getByRole("button", { name: "Open pallor comparison", exact: true })
    .click();
  const views = page.locator(".anatomy-view");
  await expect(views.nth(1)).toBeVisible();
  await page.locator(".atlas-experience").screenshot({
    path: `${qaDirectory}/${info.project.name}-eye-pallor-comparison.png`,
    animations: "disabled",
  });
  await expect(page.locator(".atlas-inspector h2")).toContainText(
    "conjunctiva",
  );
  await views.first().scrollIntoViewIfNeeded();
  await views.first().focus();
  await views.first().press("ArrowRight");
  await expect
    .poll(() => views.first().getAttribute("data-camera-yaw"))
    .not.toBe("0");
  expect(await views.first().getAttribute("data-camera-yaw")).toBe(
    await views.nth(1).getAttribute("data-camera-yaw"),
  );
  const box = (await views.first().boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.7);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, {
    steps: 6,
  });
  await page.mouse.up();
  expect(await views.first().getAttribute("data-camera-yaw")).toBe(
    await views.nth(1).getAttribute("data-camera-yaw"),
  );
  await page
    .getByRole("button", { name: "Reset anatomy view", exact: true })
    .click();
  await expect(views.first()).toHaveAttribute("data-camera-yaw", "0");
  if (info.project.name === "mobile") {
    await views.first().scrollIntoViewIfNeeded();
    const rect = (await views.first().boundingBox())!,
      cx = rect.x + rect.width / 2,
      cy = rect.y + rect.height / 2;
    const before = Number(
      await views.first().getAttribute("data-camera-distance"),
    );
    const c = await page.context().newCDPSession(page);
    await c.send("Input.dispatchTouchEvent", {
      type: "touchStart",
      touchPoints: [
        { x: cx - 20, y: cy, id: 1 },
        { x: cx + 20, y: cy, id: 2 },
      ],
    });
    await c.send("Input.dispatchTouchEvent", {
      type: "touchMove",
      touchPoints: [
        { x: cx - 45, y: cy, id: 1 },
        { x: cx + 45, y: cy, id: 2 },
      ],
    });
    await c.send("Input.dispatchTouchEvent", {
      type: "touchEnd",
      touchPoints: [],
    });
    await expect
      .poll(async () =>
        Number(await views.first().getAttribute("data-camera-distance")),
      )
      .toBeLessThan(before);
    await c.detach();
  }
  await page.getByRole("radio", { name: "Transition", exact: true }).click();
  await expect(views.nth(1)).not.toBeVisible();
  await page
    .getByRole("slider", { name: "Illustrative tissue pallor" })
    .fill("0");
  const colored = await page.locator(".anatomy-webgl").screenshot();
  await page
    .getByRole("slider", { name: "Illustrative tissue pallor" })
    .fill("100");
  await expect
    .poll(
      async () =>
        !(await page.locator(".anatomy-webgl").screenshot()).equals(colored),
    )
    .toBe(true);
  await page.getByRole("radio", { name: "Compare", exact: true }).click();
  await page.getByRole("radio", { name: "The nail", exact: true }).click();
  await expect(page.getByTestId("anatomy-canvas")).toHaveAttribute(
    "data-ready",
    "true",
    { timeout: 30000 },
  );
  await page.getByRole("button", { name: "deep skin appearance" }).click();
  await expect(
    page.getByRole("button", { name: "deep skin appearance" }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.locator(".atlas-experience").screenshot({
    path: `${qaDirectory}/${info.project.name}-pallor-comparison.png`,
    animations: "disabled",
  });
});

test("presentation view traps focus, disposes old context, and restores focus on Escape", async ({
  page,
}, info) => {
  await openExplorer(page);
  const inline = await page.locator(".anatomy-webgl").elementHandle();
  const beforeView = page.locator(".anatomy-view").first();
  await beforeView.focus();
  const originalYaw = await beforeView.getAttribute("data-camera-yaw");
  await beforeView.press("ArrowRight");
  await expect
    .poll(() => beforeView.getAttribute("data-camera-yaw"))
    .not.toBe(originalYaw);
  const savedYaw = await beforeView.getAttribute("data-camera-yaw");
  await page.getByRole("button", { name: "Expand anatomy explorer" }).click();
  const dialog = page.getByRole("dialog", {
    name: "Expanded anatomy explorer",
  });
  await expect(dialog).toBeVisible();
  const dialogBox = await dialog.boundingBox();
  expect(dialogBox!.width).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
  expect(
    await dialog.evaluate((element) => element.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
  await expect(page.locator("#root")).toHaveAttribute("inert", "");
  await expect(
    page.getByRole("button", { name: "Close expanded view" }),
  ).toBeFocused();
  await expect
    .poll(() =>
      inline!.evaluate(
        (canvas) =>
          !(canvas as HTMLCanvasElement).getContext("webgl2") ||
          (canvas as HTMLCanvasElement).getContext("webgl2")!.isContextLost(),
      ),
    )
    .toBe(true);
  await expect(page.getByTestId("anatomy-canvas")).toHaveAttribute(
    "data-ready",
    "true",
    { timeout: 30000 },
  );
  await expect(page.locator(".anatomy-view").first()).toHaveAttribute(
    "data-camera-yaw",
    savedYaw!,
  );
  await page.locator(".atlas-presentation").screenshot({
    path: `${qaDirectory}/${info.project.name}-presentation.png`,
    animations: "disabled",
  });
  await page
    .getByRole("button", { name: "Pallor comparison", exact: true })
    .focus();
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("radio", { name: "The eye", exact: true }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(page.locator("#root")).not.toHaveAttribute("inert", "");
  await expect(
    page.getByRole("button", { name: "Expand anatomy explorer" }),
  ).toBeFocused();
});

test("reduced motion disables rotation and context loss exposes a truthful static reference", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openExplorer(page);
  await expect(
    page.getByRole("button", { name: "Auto-rotate anatomy" }),
  ).toBeDisabled();
  await page
    .locator(".anatomy-webgl")
    .evaluate((canvas: HTMLCanvasElement) =>
      canvas
        .getContext("webgl2")
        ?.getExtension("WEBGL_lose_context")
        ?.loseContext(),
    );
  await expect(page.getByTestId("anatomy-fallback")).toBeVisible();
  await expect(page.getByTestId("anatomy-fallback")).toContainText(
    "static reference",
  );
  await structure(page, "Upper conjunctiva");
  await expect(page.locator(".atlas-inspector h2")).toHaveText(
    "Upper conjunctiva",
  );
});

test("failed assets show a retryable reference and preserve structure descriptions", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/anatomy/eye.glb", (route) => route.abort("failed"));
  await page.goto("/static/ui/#/anatomy");
  await expect(page.getByTestId("anatomy-fallback")).toBeVisible({
    timeout: 30000,
  });
  const reference = page.getByTestId("anatomy-fallback").locator("img");
  await expect(reference).toHaveAttribute("src", /eye-tissue-poster\.png$/);
  await expect
    .poll(() =>
      reference.evaluate(
        (image: HTMLImageElement) => image.complete && image.naturalWidth > 0,
      ),
    )
    .toBe(true);
  await structure(page, "Upper conjunctiva");
  await expect(page.locator(".atlas-inspector h2")).toHaveText(
    "Upper conjunctiva",
  );
  await expect(
    page.getByRole("button", { name: "Retry 3D viewer" }),
  ).toBeVisible();
  await page.unroute("**/anatomy/eye.glb");
  await page.getByRole("button", { name: "Retry 3D viewer" }).click();
  await expect(page.getByTestId("anatomy-canvas")).toHaveAttribute(
    "data-ready",
    "true",
    { timeout: 30000 },
  );
  await expect(page.getByTestId("anatomy-fallback")).toHaveCount(0);
  await expect(page.locator(".atlas-inspector h2")).toHaveText(
    "Upper conjunctiva",
  );
  expect(errors).toEqual([]);
});
