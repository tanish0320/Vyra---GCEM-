import type { Modality } from "../types.ts";

export type CompareMode = "single" | "compare";
export type SkinPreset = "light" | "medium" | "deep";
export type CameraAction = "front" | "angle" | "zoom-in" | "zoom-out" | "reset";
export type AnatomyView =
  "assembled" | "exploded" | "cutaway" | "tissue" | "pallor";
export type AnatomyProfile = "full" | "eye-screening";
export const EYE_TISSUE_IDS = ["conjunctiva", "upper-conjunctiva"] as const;
export const EYE_CONTEXT_IDS = [
  "eyelids",
  "sclera",
  "limbus",
  "cornea",
  "iris",
  "pupil",
] as const;
const eyeTissues = new Set<string>(EYE_TISSUE_IDS);
const eyeContext = new Set<string>(EYE_CONTEXT_IDS);
export function selectablePart(id: string, profile: AnatomyProfile = "full") {
  return profile === "full" || eyeTissues.has(id);
}
export function renderedPart(id: string, profile: AnatomyProfile = "full") {
  return profile === "full" || eyeTissues.has(id) || eyeContext.has(id);
}
/** Explicit anatomical targets: similarly named lid/skin materials are not tissue. */
export function pallorTarget(id: string | undefined) {
  return !!id && (eyeTissues.has(id) || id === "nail-bed");
}
export function assetView(view: AnatomyView): AnatomyView {
  return view === "tissue" ? "pallor" : view;
}
export type Vec3 = [number, number, number];
export interface AnatomyPart {
  id: string;
  label: string;
  description: string;
  nodes: string[];
  anchor: Vec3;
  explode: Vec3;
  pallor?: boolean;
  views?: AnatomyView[];
}
export interface AnatomyCredit {
  author: string;
  title: string;
  url: string;
  license: string;
  licenseUrl: string;
  changes: string;
}
export interface AnatomyManifest {
  version: 1;
  id: "eye" | "nail";
  bounds: { min: Vec3; max: Vec3 };
  parts: AnatomyPart[];
  credits: AnatomyCredit[];
  poster?: string;
}
export interface AnatomyState {
  modality: Modality;
  view: AnatomyView;
  explosion: number;
  selectedPart: string | null;
  hiddenParts: string[];
  isolatedPart: string | null;
  mode: CompareMode;
  pallor: number;
  exposure: number;
  skin: SkinPreset;
  /** Retained for existing domain consumers; selections use anatomical IDs. */
  hotspot: number;
  labels: boolean;
  autoRotate: boolean;
}
export const initialAnatomy: AnatomyState = {
  modality: "conjunctiva",
  view: "exploded",
  explosion: 0.65,
  selectedPart: "iris",
  hiddenParts: [],
  isolatedPart: null,
  mode: "single",
  pallor: 0.7,
  exposure: 0.75,
  skin: "medium",
  hotspot: 0,
  labels: true,
  autoRotate: false,
};
export const initialExplorerAnatomy: AnatomyState = {
  ...initialAnatomy,
  view: "tissue",
  explosion: 0,
  selectedPart: "conjunctiva",
  pallor: 0,
};
export const SKIN_COLORS: Record<SkinPreset, string> = {
  light: "#e5bba3",
  medium: "#bc8f74",
  deep: "#77523f",
};
export const VIEW_LABELS: Record<AnatomyView, string> = {
  assembled: "Assembled",
  exploded: "Exploded",
  cutaway: "Cutaway",
  tissue: "Tissue view",
  pallor: "Pallor comparison",
};
const eyeParts = [
  [
    "cornea",
    "Cornea",
    "The transparent curved surface at the front of the eye helps focus incoming light.",
  ],
  [
    "iris",
    "Iris",
    "The iris controls how much light enters the eye through the pupil.",
  ],
  [
    "lens",
    "Lens",
    "The clear lens sits behind the iris and helps focus light onto the retina.",
  ],
  [
    "ciliary-body",
    "Ciliary body",
    "The ciliary body supports the lens and contributes to its focusing mechanism.",
  ],
  [
    "retina",
    "Retina",
    "The light-sensitive inner layer of the eye converts light into signals carried toward the brain.",
  ],
  [
    "choroid",
    "Choroid",
    "The vascular layer between the retina and sclera supplies the outer retina.",
  ],
  [
    "sclera",
    "Sclera",
    "The firm outer coat protects the eye and maintains its shape.",
  ],
  [
    "optic-nerve",
    "Optic nerve",
    "The optic nerve carries visual information from the retina toward the brain.",
  ],
  [
    "conjunctiva",
    "Lower conjunctiva",
    "The tissue lining the inner lower eyelid is the region used for eye screening in this project.",
  ],
  [
    "upper-conjunctiva",
    "Upper conjunctiva",
    "The conjunctiva lines the inner surface of the upper eyelid.",
  ],
];
const nailParts = [
  [
    "nail-plate",
    "Nail plate",
    "The curved translucent keratin plate protects the fingertip and lies over the nail bed.",
  ],
  [
    "nail-bed",
    "Nail bed",
    "The tissue beneath the nail plate contributes to its visible color.",
  ],
  [
    "matrix",
    "Nail matrix",
    "The matrix beneath the proximal nail fold produces the nail plate.",
  ],
  [
    "skin-folds",
    "Skin folds",
    "The surrounding skin folds border and protect the nail unit.",
  ],
  [
    "cuticle",
    "Cuticle",
    "The thin rim at the base of the plate helps seal the space beneath the proximal nail fold.",
  ],
  [
    "distal-phalanx",
    "Distal phalanx",
    "The bone of the fingertip supports the nail unit and surrounding soft tissue.",
  ],
  [
    "soft-tissue",
    "Soft tissue",
    "The fingertip pulp contains supportive tissue, vessels, and sensory structures.",
  ],
  [
    "vessels",
    "Blood vessels",
    "Small vessels supply the living tissues of the fingertip and nail unit.",
  ],
];
export const DEFAULT_PARTS: Record<Modality, AnatomyPart[]> =
  Object.fromEntries(
    [
      ["conjunctiva", eyeParts],
      ["fingernail", nailParts],
    ].map(([modality, items]) => [
      modality,
      (items as string[][]).map(([id, label, description]) => ({
        id,
        label,
        description,
        nodes: [],
        anchor: [0, 0, 0],
        explode: [0, 0, 0],
        pallor: id === "conjunctiva" || id === "nail-bed",
      })),
    ]),
  ) as Record<Modality, AnatomyPart[]>;
export const HOTSPOTS = Object.fromEntries(
  Object.entries(DEFAULT_PARTS).map(([key, parts]) => [
    key,
    parts.map((p) => ({ name: p.label, description: p.description })),
  ]),
);
export function clampUnit(value: number) {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}
export function switchAnatomy(
  state: AnatomyState,
  modality: Modality,
): AnatomyState {
  return {
    ...state,
    modality,
    hotspot: 0,
    selectedPart: modality === "conjunctiva" ? "iris" : "nail-bed",
    hiddenParts: [],
    isolatedPart: null,
    autoRotate: false,
  };
}
export function switchExplorerAnatomy(
  state: AnatomyState,
  modality: Modality,
): AnatomyState {
  const next = switchAnatomy(state, modality);
  return modality === "conjunctiva"
    ? {
        ...next,
        view: "tissue",
        mode: "single",
        selectedPart: "conjunctiva",
        pallor: 0,
        explosion: 0,
      }
    : state.view === "tissue"
      ? { ...next, view: "exploded", mode: "single", explosion: 0.65 }
      : next;
}
export function switchView(
  state: AnatomyState,
  view: AnatomyView,
): AnatomyState {
  return {
    ...state,
    view,
    hiddenParts: [],
    isolatedPart: null,
    autoRotate: false,
    mode: view === "pallor" ? "compare" : "single",
    selectedPart:
      view === "pallor" || view === "tissue"
        ? state.modality === "conjunctiva"
          ? state.selectedPart && eyeTissues.has(state.selectedPart)
            ? state.selectedPart
            : "conjunctiva"
          : "nail-bed"
        : state.selectedPart,
  };
}
export function togglePartVisibility(
  state: AnatomyState,
  id: string,
): AnatomyState {
  return {
    ...state,
    isolatedPart: null,
    hiddenParts: state.hiddenParts.includes(id)
      ? state.hiddenParts.filter((p) => p !== id)
      : [...state.hiddenParts, id],
  };
}
export function visiblePart(
  state: AnatomyState,
  part: AnatomyPart,
  profile: AnatomyProfile = "full",
) {
  return (
    renderedPart(part.id, profile) &&
    !state.hiddenParts.includes(part.id) &&
    (!state.isolatedPart || state.isolatedPart === part.id) &&
    (!part.views || part.views.includes(assetView(state.view)))
  );
}
export function anatomyFile(modality: Modality) {
  return modality === "conjunctiva" ? "eye" : "nail";
}
export function validManifest(value: unknown): value is AnatomyManifest {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<AnatomyManifest>;
  const vector = (a: unknown): a is Vec3 =>
    Array.isArray(a) &&
    a.length === 3 &&
    a.every((n) => typeof n === "number" && Number.isFinite(n));
  if (
    candidate.version !== 1 ||
    !["eye", "nail"].includes(candidate.id || "") ||
    !candidate.bounds ||
    !vector(candidate.bounds.min) ||
    !vector(candidate.bounds.max)
  )
    return false;
  if (candidate.bounds.min.some((n, i) => n > candidate.bounds!.max[i]))
    return false;
  if (
    !Array.isArray(candidate.parts) ||
    !candidate.parts.length ||
    !Array.isArray(candidate.credits)
  )
    return false;
  const validParts = candidate.parts.every(
    (p) =>
      p &&
      typeof p === "object" &&
      typeof p.id === "string" &&
      p.id.length > 0 &&
      typeof p.label === "string" &&
      typeof p.description === "string" &&
      Array.isArray(p.nodes) &&
      p.nodes.length > 0 &&
      p.nodes.every((n) => typeof n === "string") &&
      vector(p.anchor) &&
      vector(p.explode) &&
      (!p.views ||
        (Array.isArray(p.views) &&
          p.views.every((view) =>
            ["assembled", "exploded", "cutaway", "pallor"].includes(view),
          ))),
  );
  if (
    !validParts ||
    new Set(candidate.parts.map((p) => p.id)).size !== candidate.parts.length
  )
    return false;
  return candidate.credits.every(
    (c) =>
      c &&
      typeof c === "object" &&
      [c.author, c.title, c.url, c.license, c.licenseUrl, c.changes].every(
        (v) => typeof v === "string",
      ),
  );
}

/** Linear-light ratios relative to each model's authored medium appearance. */
export function skinAppearanceMultiplier(preset: SkinPreset): Vec3 {
  if (preset === "light") return [1.12, 1.08, 1.04];
  if (preset === "deep") return [0.55, 0.41, 0.32];
  return [1, 1, 1];
}
