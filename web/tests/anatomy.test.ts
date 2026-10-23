import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  initialAnatomy,
  initialExplorerAnatomy,
  switchExplorerAnatomy,
  EYE_CONTEXT_IDS,
  EYE_TISSUE_IDS,
  selectablePart,
  pallorTarget,
  assetView,
  switchView,
  switchAnatomy,
  togglePartVisibility,
  validManifest,
  skinAppearanceMultiplier,
  visiblePart,
  type AnatomyManifest,
} from "../src/anatomy/domain.ts";

test("focused eye scene contains only the two tissues and their visible context", () => {
  const manifest: AnatomyManifest = JSON.parse(
    readFileSync(
      new URL("../public/anatomy/eye.parts.json", import.meta.url),
      "utf8",
    ),
  );
  const profile = "eye-screening";
  assert.deepEqual(
    manifest.parts
      .filter((part) => selectablePart(part.id, profile))
      .map((part) => part.id)
      .sort(),
    [...EYE_TISSUE_IDS].sort(),
  );
  const allowed = [...EYE_TISSUE_IDS, ...EYE_CONTEXT_IDS].sort();
  for (const view of ["tissue", "pallor"] as const) {
    const state = { ...initialExplorerAnatomy, view };
    assert.deepEqual(
      manifest.parts
        .filter((part) => visiblePart(state, part, profile))
        .map((part) => part.id)
        .sort(),
      allowed,
    );
    assert.equal(assetView(view), "pallor");
  }
  const lower = manifest.parts.find((part) => part.id === "conjunctiva")!;
  assert.equal(
    visiblePart(
      togglePartVisibility(initialExplorerAnatomy, lower.id),
      lower,
      profile,
    ),
    false,
  );
  const isolated = {
    ...initialExplorerAnatomy,
    isolatedPart: "upper-conjunctiva",
  };
  assert.deepEqual(
    manifest.parts
      .filter((part) => visiblePart(isolated, part, profile))
      .map((part) => part.id),
    ["upper-conjunctiva"],
  );
  assert.equal(selectablePart("iris", profile), false);
  assert.equal(selectablePart("nail-bed", "full"), true);
});

test("eye starts at exposed baseline and modality changes preserve full nail views", () => {
  assert.equal(initialExplorerAnatomy.view, "tissue");
  assert.equal(initialExplorerAnatomy.pallor, 0);
  assert.equal(initialExplorerAnatomy.selectedPart, "conjunctiva");
  const nail = switchExplorerAnatomy(initialExplorerAnatomy, "fingernail");
  assert.equal(nail.view, "exploded");
  assert.equal(nail.explosion, 0.65);
  assert.equal(nail.selectedPart, "nail-bed");
  const eye = switchExplorerAnatomy(switchView(nail, "cutaway"), "conjunctiva");
  assert.equal(eye.view, "tissue");
  assert.equal(eye.selectedPart, "conjunctiva");
  assert.equal(eye.mode, "single");
  assert.equal(
    switchView({ ...eye, selectedPart: "upper-conjunctiva" }, "pallor")
      .selectedPart,
    "upper-conjunctiva",
  );
});

test("pallor targets tissue IDs even when a non-target shares its material name", () => {
  const bytes = readFileSync(
    new URL("../public/anatomy/eye.glb", import.meta.url),
  );
  const json = JSON.parse(
    bytes.toString("utf8", 20, 20 + bytes.readUInt32LE(12)),
  );
  const tissueNodes = json.nodes.filter(
    (node: { extras?: { anatomyPart?: string }; mesh?: number }) =>
      node.mesh !== undefined && pallorTarget(node.extras?.anatomyPart),
  );
  assert.deepEqual(
    tissueNodes.map((node: { name: string }) => node.name).sort(),
    ["Eye_conjunctiva", "Eye_upper_conjunctiva"],
  );
  const lid = json.nodes.find(
    (node: { name: string }) => node.name === "Eye_lid_margin",
  );
  assert.equal(
    json.materials[json.meshes[lid.mesh].primitives[0].material].name,
    "Conjunctiva",
  );
  assert.equal(pallorTarget(lid.extras.anatomyPart), false);
  assert.equal(pallorTarget("nail-bed"), true);
  assert.equal(pallorTarget("nail-plate"), false);
});

for (const file of ["eye", "nail"]) {
  test(`${file}: selectable structure manifests resolve to real GLB nodes`, () => {
    const manifest: unknown = JSON.parse(
      readFileSync(
        new URL(`../public/anatomy/${file}.parts.json`, import.meta.url),
        "utf8",
      ),
    );
    assert.ok(validManifest(manifest));
    const bytes = readFileSync(
      new URL(`../public/anatomy/${file}.glb`, import.meta.url),
    );
    assert.equal(bytes.toString("ascii", 0, 4), "glTF");
    const jsonLength = bytes.readUInt32LE(12);
    const json = JSON.parse(bytes.toString("utf8", 20, 20 + jsonLength));
    if (file === "nail") {
      const bed = json.materials.find(
        (material: { name: string }) => material.name === "NailBed",
      );
      const plate = json.materials.find(
        (material: { name: string }) => material.name === "NailPlate",
      );
      const bedColor = bed?.pbrMetallicRoughness?.baseColorFactor;
      assert.ok(
        bedColor && bedColor[0] > bedColor[1] * 1.5,
        "Export must retain the authored pink nail-bed tint",
      );
      assert.equal(
        plate?.alphaMode,
        "BLEND",
        "Nail plate must remain translucent",
      );
      assert.ok(plate.pbrMetallicRoughness.baseColorFactor[3] < 0.5);
    }
    const names = new Set(
      json.nodes.map((node: { name?: string }) => node.name),
    );
    for (const part of manifest.parts) {
      assert.ok(part.nodes.length > 0, `${part.id} must own geometry`);
      for (const node of part.nodes)
        assert.ok(names.has(node), `${part.id}: missing ${node}`);
      assert.ok(part.anchor.every(Number.isFinite));
      assert.ok(part.explode.every(Number.isFinite));
    }
    const cutNodes = json.nodes.filter(
      (node: { extras?: { anatomyViews?: string[] } }) =>
        node.extras?.anatomyViews?.includes("cutaway"),
    );
    assert.ok(
      cutNodes.length > 0,
      "asset must contain actual cutaway geometry",
    );
    assert.ok(
      manifest.parts.some((part) => part.pallor),
      "asset must identify its screening tissue",
    );
    assert.ok(
      manifest.credits.length > 0,
      "asset must retain source attribution",
    );
  });
}
test("changing view restores hidden structures and comparison has a defined target", () => {
  const state = {
    ...initialAnatomy,
    hiddenParts: ["iris"],
    isolatedPart: "iris",
    autoRotate: true,
  };
  const next = switchView(state, "pallor");
  assert.equal(next.view, "pallor");
  assert.equal(next.mode, "compare");
  assert.equal(next.autoRotate, false);
  assert.deepEqual(next.hiddenParts, []);
  assert.equal(next.isolatedPart, null);
  assert.equal(switchView(next, "exploded").mode, "single");
  assert.deepEqual(state.hiddenParts, ["iris"]);
});
test("isolation, manual visibility and view-specific geometry compose predictably", () => {
  const manifest = JSON.parse(
    readFileSync(
      new URL("../public/anatomy/eye.parts.json", import.meta.url),
      "utf8",
    ),
  ) as AnatomyManifest;
  const iris = manifest.parts.find((part) => part.id === "iris")!;
  assert.equal(visiblePart(initialAnatomy, iris), true);
  const hidden = togglePartVisibility(initialAnatomy, iris.id);
  assert.equal(visiblePart(hidden, iris), false);
  assert.equal(visiblePart(togglePartVisibility(hidden, iris.id), iris), true);
  assert.equal(
    visiblePart({ ...initialAnatomy, isolatedPart: "sclera" }, iris),
    false,
  );
  const tissue = manifest.parts.find((part) => part.pallor)!;
  if (tissue.views)
    assert.equal(
      visiblePart({ ...initialAnatomy, view: "pallor" }, tissue),
      true,
    );
  const switched = switchAnatomy(hidden, "fingernail");
  assert.deepEqual(switched.hiddenParts, []);
  assert.equal(switched.selectedPart, "nail-bed");
});
test("malformed or duplicate structure definitions are rejected before model loading", () => {
  const valid = JSON.parse(
    readFileSync(
      new URL("../public/anatomy/eye.parts.json", import.meta.url),
      "utf8",
    ),
  ) as AnatomyManifest;
  assert.equal(
    validManifest({ ...valid, parts: [valid.parts[0], valid.parts[0]] }),
    false,
  );
  assert.equal(
    validManifest({
      ...valid,
      parts: [{ ...valid.parts[0], anchor: [NaN, 0, 0] }],
    }),
    false,
  );
  assert.equal(validManifest({ ...valid, parts: [] }), false);
  assert.equal(validManifest({ ...valid, parts: [null] }), false);
  assert.equal(
    validManifest({ ...valid, bounds: { min: [0, 0, 0], max: [NaN, 1, 1] } }),
    false,
  );
  assert.equal(validManifest({ ...valid, credits: [null] }), false);
  assert.equal(validManifest({ version: 1, id: "eye" }), false);
});

test("skin display presets lighten and darken authored samples in the stated order", () => {
  const luminance = (rgb: number[]) =>
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
  for (const sample of [
    [0.45, 0.26, 0.16],
    [0.17, 0.09, 0.05],
    [0.77, 0.65, 0.5],
  ]) {
    const values = (["deep", "medium", "light"] as const).map((preset) =>
      luminance(
        sample.map(
          (channel, index) => channel * skinAppearanceMultiplier(preset)[index],
        ),
      ),
    );
    assert.ok(values[0] < values[1] && values[1] < values[2]);
    assert.deepEqual(
      sample.map(
        (channel, index) => channel * skinAppearanceMultiplier("medium")[index],
      ),
      sample,
    );
  }
});
