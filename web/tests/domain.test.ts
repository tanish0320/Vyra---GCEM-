import test from "node:test";
import assert from "node:assert/strict";
import {
  FILE_LIMIT,
  clampBox,
  centeredBox,
  validAge,
  validateDimensions,
  validateFile,
} from "../src/lib/domain.ts";
import {
  initialAnatomy,
  switchAnatomy,
  clampUnit,
} from "../src/anatomy/domain.ts";
test("upload boundary accepts JPEG PNG WebP and rejects empty oversized unsupported files", () => {
  for (const type of ["image/jpeg", "image/png", "image/webp"])
    assert.equal(validateFile(type, FILE_LIMIT), null);
  for (const type of ["image/svg+xml", "image/heic", "application/pdf"])
    assert.ok(validateFile(type, 100));
  assert.ok(validateFile("image/png", 0));
  assert.ok(validateFile("image/png", FILE_LIMIT + 1));
});
test("small reference nail passes fifty pixel boundary without upscaling", () => {
  assert.equal(validateDimensions(70, 58), null);
  assert.equal(validateDimensions(50, 50), null);
  assert.ok(validateDimensions(49, 100));
  assert.ok(validateDimensions(NaN, 100));
  assert.ok(validateDimensions(Infinity, 100));
});
test("crop coordinates remain exclusive and within oriented image even for reversed or invalid inputs", () => {
  for (let i = -200; i < 400; i += 3) {
    const box = clampBox(
      { x0: i, y0: i + 10, x1: i - 20, y1: i + 600 },
      200,
      100,
    );
    assert.ok(box.x0 >= 0 && box.y0 >= 0 && box.x1 <= 200 && box.y1 <= 100);
    assert.ok(box.x1 > box.x0 && box.y1 > box.y0);
  }
  assert.throws(() => clampBox(centeredBox(200, 100), 0, 100));
});
test("optional age rejects fractions, negative values, whitespace and outside range", () => {
  for (const value of ["", "0", "120", "24"])
    assert.equal(validAge(value), true);
  for (const value of ["-1", "121", "23.5", "hello", " ", "Infinity"])
    assert.equal(validAge(value), false);
});
test("switching anatomy preserves comparison preferences and stops rotation", () => {
  const previous = {
    ...initialAnatomy,
    mode: "compare" as const,
    skin: "deep" as const,
    autoRotate: true,
    hotspot: 2,
  };
  const next = switchAnatomy(previous, "fingernail");
  assert.equal(next.autoRotate, false);
  assert.equal(next.hotspot, 0);
  assert.equal(next.mode, "compare");
  assert.equal(next.skin, "deep");
  assert.equal(previous.autoRotate, true);
});
test("illustrative appearance values are finite and bounded", () => {
  assert.equal(clampUnit(NaN), 0);
  assert.equal(clampUnit(Infinity), 0);
  assert.equal(clampUnit(-1), 0);
  assert.equal(clampUnit(2), 1);
  assert.equal(clampUnit(0.5), 0.5);
});
