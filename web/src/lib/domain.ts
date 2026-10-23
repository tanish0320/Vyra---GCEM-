import type { Box, Modality } from "../types.ts";
export const FILE_LIMIT = 10 * 1024 * 1024;
export const MODALITY_LABELS: Record<Modality, string> = {
  conjunctiva: "Eye",
  fingernail: "Nail",
};
export const centeredBox = (width: number, height: number): Box => ({
  x0: Math.floor(width * 0.15),
  y0: Math.floor(height * 0.15),
  x1: Math.ceil(width * 0.85),
  y1: Math.ceil(height * 0.85),
});
export function clampBox(box: Box, width: number, height: number): Box {
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width < 1 ||
    height < 1
  )
    throw new Error("Image dimensions must be positive");
  const x0 = Math.max(
    0,
    Math.min(width - 1, Math.round(Number.isFinite(box.x0) ? box.x0 : 0)),
  );
  const y0 = Math.max(
    0,
    Math.min(height - 1, Math.round(Number.isFinite(box.y0) ? box.y0 : 0)),
  );
  return {
    x0,
    y0,
    x1: Math.max(
      x0 + 1,
      Math.min(width, Math.round(Number.isFinite(box.x1) ? box.x1 : width)),
    ),
    y1: Math.max(
      y0 + 1,
      Math.min(height, Math.round(Number.isFinite(box.y1) ? box.y1 : height)),
    ),
  };
}
export function validateFile(type: string, size: number): string | null {
  if (!["image/jpeg", "image/png", "image/webp"].includes(type))
    return "Choose a JPEG, PNG, or WebP image.";
  if (size > FILE_LIMIT)
    return "This image is larger than 10 MB. Choose a smaller file.";
  if (size === 0) return "This file is empty. Choose another image.";
  return null;
}
export const validateDimensions = (
  width: number,
  height: number,
): string | null =>
  !Number.isFinite(width) ||
  !Number.isFinite(height) ||
  Math.min(width, height) < 50
    ? "Both image dimensions must be at least 50 pixels."
    : null;
export const validAge = (age: string) =>
  age === "" || (/^\d+$/.test(age) && Number(age) <= 120);
export const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
export const displayDate = (value: string) =>
  new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
