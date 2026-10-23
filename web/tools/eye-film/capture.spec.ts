import { test, expect } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
const directory = process.env.VYRA_FILM_FRAMES || "/tmp/vyra-eye-film-frames";
test("render the eye film deterministically", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await mkdir(directory, { recursive: true });
  // Source formatting or unrelated app edits must not reload a long offline capture.
  await page.routeWebSocket("**", (socket) => socket.close());
  await page.goto("/tools/eye-film/index.html");
  await page.waitForFunction(() => !!(window as any).eyeFilm, {
    timeout: 60_000,
  });
  const preview = testInfo.project.name === "preview";
  const frames = preview
    ? [0, 120, 150, 195, 300, 390, 480, 599]
    : Array.from({ length: 600 }, (_, i) => i);
  for (const index of frames) {
    if (
      testInfo.project.name === "resume" &&
      existsSync(`${directory}/${String(index).padStart(4, "0")}.png`)
    )
      continue;
    const png = await page.evaluate((time) => {
      (window as any).eyeFilm.frame(time);
      return document
        .querySelector("canvas")!
        .toDataURL("image/png")
        .split(",")[1];
    }, index / 30);
    await writeFile(
      `${directory}/${String(index).padStart(4, "0")}.png`,
      Buffer.from(png, "base64"),
    );
  }
  const tissue = await page.evaluate(() => {
    (window as any).eyeFilm.tissuePoster();
    return document
      .querySelector("canvas")!
      .toDataURL("image/png")
      .split(",")[1];
  });
  await writeFile(
    "public/anatomy/eye-tissue-poster.png",
    Buffer.from(tissue, "base64"),
  );
  await writeFile(
    `${directory}/scene.json`,
    JSON.stringify(
      await page.evaluate(() => {
        const { camera, envelope, width, height } = (window as any).eyeFilm;
        return {
          camera,
          envelope,
          width,
          height,
          fps: 30,
          frames: 600,
          seconds: 20,
        };
      }),
      null,
      2,
    ),
  );
  expect(errors).toEqual([]);
});
