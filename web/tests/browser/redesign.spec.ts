import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { setup } from "./integration-fixtures";

test("the landing eye film stays local and has accessible motion controls", async ({
  page,
}, info) => {
  const errors: string[] = [];
  const external: string[] = [];
  const heavy: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (
      /\.glb(?:\?|$)|\.parts\.json|\/three(?:[._/-]|$)|AnatomyCanvas|models\.ts/.test(
        url.pathname,
      )
    )
      heavy.push(url.pathname);
    if (
      /^https?:$/.test(url.protocol) &&
      !/^(127\.0\.0\.1|localhost)$/.test(url.hostname)
    )
      external.push(url.href);
  });
  await page.goto("/static/ui/");
  const video = page.locator(".eye-film-video");
  await video.scrollIntoViewIfNeeded();
  await expect(video).toBeVisible();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.readyState))
    .toBeGreaterThanOrEqual(2);
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(false);
  await expect(video).toHaveAttribute("playsinline", "");
  expect(await video.evaluate((el: HTMLVideoElement) => el.muted)).toBe(true);
  expect(
    await video.evaluate((el: HTMLVideoElement) => el.duration),
  ).toBeCloseTo(20, 1);
  await expect(page.locator(".atlas-hero canvas")).toHaveCount(0);
  await expect(page.locator(".atlas-hero")).not.toContainText(
    /nail|drag to rotate/i,
  );
  await expect(page.locator(".atlas-hero input[type=range]")).toHaveCount(0);
  await page.evaluate(() => document.fonts.ready);
  await expect
    .poll(() =>
      page
        .locator("h1")
        .evaluate((el) =>
          [...el.querySelectorAll("*")].every(
            (child) => Number(getComputedStyle(child).opacity) > 0.99,
          ),
        ),
    )
    .toBe(true);
  await page
    .getByRole("heading", { name: "From image to result." })
    .scrollIntoViewIfNeeded();
  for (const item of await page.locator(".workflow-item").all()) {
    await item.scrollIntoViewIfNeeded();
    await expect
      .poll(() => item.evaluate((el) => Number(getComputedStyle(el).opacity)))
      .toBeGreaterThan(0.99);
  }
  await expect
    .poll(() =>
      page
        .locator(".workflow-item")
        .evaluateAll((items) =>
          items.every((item) => Number(getComputedStyle(item).opacity) > 0.99),
        ),
    )
    .toBe(true);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "instant" }));
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
  mkdirSync("/tmp/vyra-redesign-qa", { recursive: true });
  await page.screenshot({
    path: `/tmp/vyra-redesign-qa/${info.project.name}-home-ready.png`,
    fullPage: true,
  });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Explore the anatomy behind anemia screening.",
  );
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(true);
  await page
    .getByRole("button", { name: "Play eye film", exact: true })
    .click();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(false);
  await video.evaluate((el: HTMLVideoElement) => {
    el.currentTime = 10;
  });
  await page
    .getByRole("button", { name: "Replay eye film", exact: true })
    .click();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.currentTime))
    .toBeLessThan(2);
  await page
    .getByRole("button", { name: "Pause eye film", exact: true })
    .click();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(true);
  // A deliberate pause survives a viewport leave/return.
  const viewport = page.viewportSize()!;
  await page.setViewportSize({ width: viewport.width, height: 420 });
  await page.evaluate(() =>
    window.scrollTo({ top: document.body.scrollHeight, behavior: "instant" }),
  );
  await page.setViewportSize(viewport);
  await video.scrollIntoViewIfNeeded();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(true);
  await page.screenshot({
    path: `/tmp/vyra-redesign-qa/${info.project.name}-home-film-paused.png`,
    fullPage: true,
  });
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
  expect(heavy).toEqual([]);
});

test("the eye film waits for a deliberate play with reduced motion", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/static/ui/");
  const video = page.locator(".eye-film-video");
  await video.scrollIntoViewIfNeeded();
  await expect(
    page.getByRole("button", { name: "Play eye film", exact: true }),
  ).toBeVisible();
  expect(
    await video.evaluate(
      (el: HTMLVideoElement) => el.paused && el.currentTime === 0,
    ),
  ).toBe(true);
  await page
    .getByRole("button", { name: "Play eye film", exact: true })
    .click();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(false);
  const viewport = page.viewportSize()!;
  await page.setViewportSize({ width: viewport.width, height: 420 });
  await page.evaluate(() =>
    window.scrollTo({ top: document.body.scrollHeight, behavior: "instant" }),
  );
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(true);
  await page.setViewportSize(viewport);
  await video.scrollIntoViewIfNeeded();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(false);
});

test("blocked eye film autoplay leaves a usable play control", async ({
  page,
}) => {
  await page.addInitScript(() => {
    let allowed = false;
    document.addEventListener("pointerdown", (event) => {
      if (event.isTrusted) allowed = true;
    });
    const play = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = function () {
      if (!allowed)
        return Promise.reject(
          new DOMException("Autoplay denied", "NotAllowedError"),
        );
      return play.call(this);
    };
  });
  await page.goto("/static/ui/");
  const video = page.locator(".eye-film-video");
  await video.scrollIntoViewIfNeeded();
  await expect(
    page.getByRole("button", { name: "Play eye film", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Play eye film", exact: true })
    .click();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.paused))
    .toBe(false);
});

test("an unavailable WebM source falls back to playable MP4", async ({
  page,
}) => {
  await page.route(/\/anatomy\/eye-film(?:-small)?\.webm$/, (route) =>
    route.fulfill({ status: 404, body: "Unavailable format" }),
  );
  await page.goto("/static/ui/");
  const video = page.locator(".eye-film-video");
  await video.scrollIntoViewIfNeeded();
  await expect
    .poll(() => video.evaluate((el: HTMLVideoElement) => el.readyState))
    .toBeGreaterThanOrEqual(2);
  expect(await video.evaluate((el: HTMLVideoElement) => el.currentSrc)).toMatch(
    /\.mp4$/,
  );
  await expect(
    page.getByRole("button", { name: "Pause eye film", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Pause eye film", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Play eye film", exact: true }),
  ).toBeEnabled();
});

test("missing film files retain the poster and navigation", async ({
  page,
}) => {
  await page.route(/\/anatomy\/eye-film(?:-small)?\.(?:webm|mp4)$/, (route) =>
    route.fulfill({ status: 404, body: "Unavailable film" }),
  );
  await page.goto("/static/ui/");
  const video = page.locator(".eye-film-video");
  await video.scrollIntoViewIfNeeded();
  await expect(page.getByRole("status")).toHaveText("Film unavailable");
  await expect(
    page.getByRole("button", { name: "Play eye film", exact: true }),
  ).toBeDisabled();
  await expect(video).toHaveAttribute("poster", /eye-film-poster\.webp$/);
  expect(
    await video.evaluate(
      (el: HTMLVideoElement) =>
        new Promise<boolean>((resolve) => {
          const image = new window.Image();
          image.onload = () => resolve(image.naturalWidth > 0);
          image.onerror = () => resolve(false);
          image.src = el.poster;
        }),
    ),
  ).toBe(true);
  await expect(
    page.getByRole("link", { name: "Explore anatomy", exact: true }),
  ).toBeVisible();
});

test("the scientific workspace is readable and motion changes preserve the selected image", async ({
  page,
}, info) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await setup(page);
  await page.goto("/static/ui/#/studio");
  await expect(
    page.getByRole("heading", { name: "Screening studio", exact: true }),
  ).toBeVisible();
  await page
    .getByLabel("Choose a photo", { exact: true })
    .setInputFiles(resolve("tests/fixtures/test_eye_conjunctiva.png"));
  await expect(page.getByAltText("Selected screening photo")).toBeVisible();
  await page
    .getByLabel("Patient name or case ID", { exact: false })
    .fill("QA-REDESIGN");
  await page.getByLabel("Age in years", { exact: false }).fill("24");
  await page
    .getByLabel("Biological sex", { exact: false })
    .selectOption("not-stated");
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(page.getByAltText("Selected screening photo")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Review image", exact: true }),
  ).toBeEnabled();
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await expect(page.getByAltText("Selected screening photo")).toBeVisible();
  const typography = await page
    .getByRole("heading", { level: 1 })
    .evaluate((el) => ({
      font: getComputedStyle(el).fontFamily,
      size: parseFloat(getComputedStyle(el).fontSize),
    }));
  expect(typography.font).toContain("Space Grotesk");
  expect(typography.size).toBeGreaterThanOrEqual(34);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
  mkdirSync("/tmp/vyra-redesign-qa", { recursive: true });
  await page.screenshot({
    path: `/tmp/vyra-redesign-qa/${info.project.name}-studio.png`,
    fullPage: true,
  });
  for (const [route, heading] of [
    ["history", "Screening history"],
    ["evidence", "Model evidence"],
  ]) {
    await page.goto(`/static/ui/#/${route}`);
    await expect(
      page.getByRole("heading", { name: heading, exact: true }),
    ).toBeVisible();
    await expect(page.locator("vite-error-overlay")).toHaveCount(0);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
    ).toBeLessThanOrEqual(page.viewportSize()!.width + 1);
    await page.screenshot({
      path: `/tmp/vyra-redesign-qa/${info.project.name}-${route}.png`,
      fullPage: true,
    });
  }
  expect(errors).toEqual([]);
});
