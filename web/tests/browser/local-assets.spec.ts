import { expect, test } from "@playwright/test";
import { setup } from "./integration-fixtures";

test("built app loads anatomy and evidence with external network blocked", async ({
  page,
}) => {
  const external: string[] = [];
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (
      url.protocol.startsWith("http") &&
      !["127.0.0.1", "localhost"].includes(url.hostname)
    ) {
      external.push(url.href);
      return route.abort();
    }
    return route.continue();
  });
  await setup(page);
  await page.goto("/static/ui/");
  await expect(page).toHaveTitle(/Vyra/i);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.goto("/static/ui/#/anatomy");
  await page.locator(".anatomy-render-stage").scrollIntoViewIfNeeded();
  await expect(page.getByTestId("anatomy-canvas")).toHaveAttribute(
    "data-ready",
    "true",
    { timeout: 20_000 },
  );
  await page.getByRole("radio", { name: "The nail", exact: true }).click();
  await expect(page.getByTestId("anatomy-canvas")).toHaveAttribute(
    "data-ready",
    "true",
  );
  await page.goto("/static/ui/#/evidence");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expect(page.locator("main")).toContainText("1,025");
  expect(external).toEqual([]);
  expect(errors).toEqual([]);
});
