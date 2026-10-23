import {
  Component,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import {
  Eye,
  EyeOff,
  Hand,
  Maximize2,
  Minimize2,
  RotateCcw,
  Minus,
  Plus,
  Move3D,
  Pause,
  Rotate3D,
  ArrowRight,
  ChevronDown,
  X,
  Layers,
  Columns2,
  Circle,
  Info,
  ExternalLink,
} from "lucide-react";
import { ToggleGroup, ToggleGroupItem } from "../components/ui/toggle-group";
import { Button } from "../components/ui/button";
import {
  anatomyFile,
  DEFAULT_PARTS,
  initialExplorerAnatomy,
  switchExplorerAnatomy,
  selectablePart,
  switchView,
  togglePartVisibility,
  VIEW_LABELS,
  SKIN_COLORS,
  clampUnit,
  type AnatomyManifest,
  type AnatomyProfile,
  type AnatomyState,
  type AnatomyView,
  type CameraAction,
  type Vec3,
} from "./domain";
import { useReducedMotion } from "../hooks/useReducedMotion";
import "./anatomy.css";
const AnatomyCanvas = lazy(() => import("./AnatomyCanvas"));

export class AnatomyBoundary extends Component<
  { children: ReactNode; fallback: ReactNode; onError: () => void },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onError();
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}
export function AnatomyPoster({
  modality,
  failed = false,
  retry,
  focused = false,
}: {
  modality: AnatomyState["modality"];
  failed?: boolean;
  retry?: () => void;
  focused?: boolean;
}) {
  const [imageFailed, setImageFailed] = useState(false);
  useEffect(() => setImageFailed(false), [modality, focused]);
  return (
    <div
      className={`atlas-poster ${failed ? "atlas-poster-failed" : ""}`}
      data-testid={failed ? "anatomy-fallback" : undefined}
    >
      {!imageFailed && (
        <img
          src={`${import.meta.env.BASE_URL}anatomy/${anatomyFile(modality)}${focused && modality === "conjunctiva" ? "-tissue" : ""}-poster.png`}
          alt={`Rendered ${modality === "conjunctiva" ? (focused ? "conjunctival tissue" : "eye") : "nail"} anatomy reference`}
          onError={() => setImageFailed(true)}
        />
      )}
      <div className="atlas-poster-message" role="status">
        {failed ? (
          <>
            <strong>3D is unavailable on this device.</strong>
            <span>
              This is a static reference. Explore the structure descriptions
              below, or retry the viewer.
            </span>
            {retry && (
              <Button variant="outline" onClick={retry}>
                Retry 3D viewer
              </Button>
            )}
          </>
        ) : (
          <>
            <span className="atlas-loading-ring" />
            <span>Preparing the anatomy…</span>
          </>
        )}
      </div>
    </div>
  );
}

export function AnatomyExplorer() {
  const [state, setState] = useState(initialExplorerAnatomy);
  const [manifest, setManifest] = useState<AnatomyManifest | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [structuresOpen, setStructuresOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [active, setActive] = useState(true);
  const [failed, setFailed] = useState(false);
  const [retryKey, setRetryKey] = useState(0);
  const [command, setCommand] = useState<{ action: CameraAction; id: number }>({
    action: "front",
    id: 0,
  });
  const reduced = useReducedMotion();
  const shell = useRef<HTMLDivElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const expandButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const cameraMemory = useRef<Vec3>([0, 0.04, 1]);
  const eye = state.modality === "conjunctiva";
  const profile: AnatomyProfile = eye ? "eye-screening" : "full";
  const allParts =
    manifest?.id === anatomyFile(state.modality)
      ? manifest.parts
      : DEFAULT_PARTS[state.modality];
  const parts = allParts.filter((part) => selectablePart(part.id, profile));
  const selected =
    parts.find((part) => part.id === state.selectedPart) || parts[0];
  const pallorPart =
    parts.find((part) => part.pallor) ||
    parts.find((part) => /conjunctiva|nail-bed/.test(part.id));
  const patch = (value: Partial<AnatomyState>) =>
    setState((previous) => ({ ...previous, ...value }));
  const stopRotation = useCallback(
    () =>
      setState((previous) =>
        previous.autoRotate ? { ...previous, autoRotate: false } : previous,
      ),
    [],
  );
  const camera = (action: CameraAction) => {
    stopRotation();
    setCommand((previous) => ({ action, id: previous.id + 1 }));
  };
  const selectView = (view: AnatomyView) => {
    setState((previous) => ({
      ...switchView(previous, view),
      ...(view === "pallor" && pallorPart && !eye
        ? { selectedPart: pallorPart.id }
        : {}),
    }));
    setStructuresOpen(false);
    camera(
      view === "assembled" || view === "pallor" || view === "tissue"
        ? "front"
        : "angle",
    );
  };
  const selectPart = (id: string) => {
    const part = parts.find((p) => p.id === id);
    if (!part) return;
    if (
      part?.views &&
      state.view !== "tissue" &&
      !part.views.includes(state.view) &&
      part.views.includes("pallor")
    )
      camera("front");
    setState((previous) => ({
      ...previous,
      selectedPart: id,
      isolatedPart: previous.isolatedPart ? id : null,
      hiddenParts: previous.hiddenParts.filter((p) => p !== id),
      ...(part?.views &&
      previous.view !== "tissue" &&
      !part.views.includes(previous.view) &&
      part.views.includes("pallor")
        ? { view: "pallor" as const, mode: "compare" as const }
        : {}),
    }));
    setInspectorOpen(true);
    setStructuresOpen(false);
  };
  const switchModality = (modality: AnatomyState["modality"]) => {
    const next = switchExplorerAnatomy(state, modality);
    setState(next);
    setManifest(null);
    setFailed(false);
    setStructuresOpen(false);
    setInspectorOpen(true);
    camera(
      next.view === "pallor" ||
        next.view === "tissue" ||
        next.view === "assembled"
        ? "front"
        : "angle",
    );
  };
  useEffect(() => {
    if (reduced) stopRotation();
  }, [reduced, stopRotation]);
  useEffect(() => {
    if (expanded) {
      setActive(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => setActive(entries[0].isIntersecting),
      { rootMargin: "80px" },
    );
    if (shell.current) observer.observe(shell.current);
    return () => observer.disconnect();
  }, [expanded]);
  useEffect(() => {
    if (!expanded) return;
    const root = document.getElementById("root")!;
    const wasInert = root.inert,
      overflow = document.body.style.overflow;
    root.inert = true;
    document.body.style.overflow = "hidden";
    closeButton.current?.focus();
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setExpanded(false);
        return;
      }
      if (event.key !== "Tab") return;
      const items = [
        ...dialog.current!.querySelectorAll<HTMLElement>(
          'button:not(:disabled),input:not(:disabled),a[href],[tabindex="0"],summary',
        ),
      ].filter(
        (item) =>
          item.offsetParent !== null &&
          item.tabIndex >= 0 &&
          getComputedStyle(item).visibility !== "hidden",
      );
      const first = items[0],
        last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", key);
    return () => {
      root.inert = wasInert;
      document.body.style.overflow = overflow;
      document.removeEventListener("keydown", key);
      requestAnimationFrame(() =>
        expandButton.current?.focus({ preventScroll: true }),
      );
    };
  }, [expanded]);
  const fallback = (
    <AnatomyPoster
      modality={state.modality}
      focused={eye}
      failed
      retry={() => {
        setFailed(false);
        setRetryKey((value) => value + 1);
      }}
    />
  );
  const modalitySelector = (
    <ToggleGroup
      type="single"
      value={state.modality}
      onValueChange={(value) => {
        if (value) switchModality(value as AnatomyState["modality"]);
      }}
      className="atlas-modalities"
      aria-label="Anatomy modality"
    >
      <ToggleGroupItem value="conjunctiva" aria-label="The eye">
        <Eye data-icon="inline-start" />
        Eye
      </ToggleGroupItem>
      <ToggleGroupItem value="fingernail" aria-label="The nail">
        <Hand data-icon="inline-start" />
        Nail
      </ToggleGroupItem>
    </ToggleGroup>
  );
  const viewTabs = (
    <ToggleGroup
      type="single"
      value={!eye && state.view === "pallor" ? "" : state.view}
      onValueChange={(value) => {
        if (value) selectView(value as AnatomyView);
      }}
      className="atlas-view-tabs"
      aria-label="Anatomy view"
    >
      {(eye
        ? (["tissue", "pallor"] as const)
        : (["assembled", "exploded", "cutaway"] as const)
      ).map((view) => (
        <ToggleGroupItem key={view} value={view}>
          {VIEW_LABELS[view]}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
  const inspector = (
    <aside
      className={`atlas-inspector ${inspectorOpen ? "is-open" : ""}`}
      aria-label="Structure inspector"
    >
      <div className="atlas-inspector-heading">
        <span>Selected structure</span>
        {expanded && (
          <button
            className="atlas-icon"
            aria-label="Close structure inspector"
            onClick={() => setInspectorOpen(false)}
          >
            <X size={18} />
          </button>
        )}
      </div>
      <h2>{selected.label}</h2>
      <p>{selected.description}</p>
      {eye && selected.id === "upper-conjunctiva" && (
        <p className="atlas-tissue-context">
          The upper tissue is shown for anatomical context. Vyra’s eye screening
          uses the lower conjunctiva.
        </p>
      )}
      <div className="atlas-inspector-actions">
        <Button
          disabled={failed}
          onClick={() =>
            patch({
              isolatedPart:
                state.isolatedPart === selected.id ? null : selected.id,
              hiddenParts: [],
            })
          }
        >
          {state.isolatedPart === selected.id ? "Restore view" : "Isolate"}
        </Button>
        <Button
          variant="outline"
          disabled={failed}
          onClick={() => patch({ hiddenParts: [], isolatedPart: null })}
        >
          Show all
        </Button>
      </div>
      {!expanded && state.view !== "pallor" && (
        <div className="atlas-screening-region">
          <span>Screening region</span>
          <h3>
            {state.modality === "conjunctiva"
              ? "Lower conjunctiva"
              : "Nail bed"}
          </h3>
          <Button
            variant="outline"
            onClick={() => selectView("pallor")}
            disabled={failed}
          >
            Open pallor comparison
            <ArrowRight data-icon="inline-end" />
          </Button>
          <p>
            {eye
              ? "Compare the two conjunctival regions under the same lighting."
              : "Compare illustrative tissue color under the same lighting."}
          </p>
        </div>
      )}
      {state.view === "pallor" && (
        <div className="atlas-pallor-controls">
          <ToggleGroup
            type="single"
            value={state.mode}
            onValueChange={(mode) => {
              if (mode) patch({ mode: mode as "single" | "compare" });
            }}
            aria-label="Pallor comparison layout"
            className="atlas-compare-switch"
          >
            <ToggleGroupItem value="compare">
              <Columns2 size={14} />
              Compare
            </ToggleGroupItem>
            <ToggleGroupItem value="single">
              <Circle size={14} />
              Transition
            </ToggleGroupItem>
          </ToggleGroup>
          {state.mode === "single" && (
            <label className="atlas-range-label">
              Illustrative pallor
              <input
                type="range"
                min="0"
                max="100"
                value={Math.round(state.pallor * 100)}
                aria-label="Illustrative tissue pallor"
                onChange={(event) =>
                  patch({ pallor: clampUnit(Number(event.target.value) / 100) })
                }
              />
              <span>
                Typical appearance <span>Paler tissue</span>
              </span>
            </label>
          )}
          <div className="atlas-skin-controls">
            <span>Skin appearance</span>
            <div>
              {(["light", "medium", "deep"] as const).map((skin) => (
                <button
                  key={skin}
                  aria-label={`${skin} skin appearance`}
                  aria-pressed={state.skin === skin}
                  style={{ backgroundColor: SKIN_COLORS[skin] }}
                  onClick={() => patch({ skin })}
                />
              ))}
            </div>
          </div>
          <p className="atlas-pallor-note">
            Illustrative color variation. This control does not measure
            hemoglobin or determine anemia.
          </p>
        </div>
      )}
      {!expanded && (
        <details className="atlas-credits">
          <summary>Model sources & credits</summary>
          {manifest?.credits.map((credit, index) => (
            <div key={index}>
              <strong>{credit.title}</strong>
              <span>{credit.author}</span>
              {credit.url && (
                <a href={credit.url} target="_blank" rel="noreferrer">
                  Source <ExternalLink size={12} />
                </a>
              )}
              {credit.licenseUrl ? (
                <a href={credit.licenseUrl} target="_blank" rel="noreferrer">
                  {credit.license}
                </a>
              ) : (
                <span>{credit.license}</span>
              )}
              <small>{credit.changes}</small>
            </div>
          )) || <p>Source information will appear when the model is loaded.</p>}
        </details>
      )}
    </aside>
  );
  const experience = (
    <div
      ref={dialog}
      className={`atlas-experience ${expanded ? "atlas-presentation" : ""}`}
      role={expanded ? "dialog" : undefined}
      aria-modal={expanded ? true : undefined}
      aria-label={
        expanded ? "Expanded anatomy explorer" : "Interactive anatomy explorer"
      }
    >
      {expanded && (
        <header className="atlas-presentation-header">
          <div>
            <strong>VYRA</strong>
            <span>Anatomy explorer</span>
          </div>
          {modalitySelector}
          <Button
            ref={closeButton}
            variant="outline"
            aria-label="Close expanded view"
            onClick={() => setExpanded(false)}
          >
            <Minimize2 data-icon="inline-start" />
            <span className="atlas-presentation-button-text">
              Exit presentation
            </span>
          </Button>
        </header>
      )}
      <div className="atlas-workspace">
        {!expanded && (
          <aside
            className="atlas-structure-rail"
            aria-label="Anatomical structures"
          >
            {modalitySelector}
            <button
              className="atlas-structures-trigger"
              aria-expanded={structuresOpen}
              onClick={() => setStructuresOpen((value) => !value)}
            >
              Structures
              <ChevronDown size={16} />
            </button>
            <div
              className={`atlas-structure-list ${structuresOpen ? "is-open" : ""}`}
            >
              <h2>Structures</h2>
              {parts.map((part) => (
                <div
                  key={part.id}
                  className={`atlas-structure-row ${state.selectedPart === part.id ? "is-selected" : ""}`}
                >
                  <button
                    className="atlas-visibility"
                    aria-label={`${state.hiddenParts.includes(part.id) ? "Show" : "Hide"} ${part.label}`}
                    onClick={() =>
                      setState((previous) =>
                        togglePartVisibility(previous, part.id),
                      )
                    }
                    disabled={failed}
                  >
                    {state.hiddenParts.includes(part.id) ? (
                      <EyeOff size={17} />
                    ) : (
                      <Eye size={17} />
                    )}
                  </button>
                  <button
                    className="atlas-structure-name"
                    aria-pressed={state.selectedPart === part.id}
                    onClick={() => selectPart(part.id)}
                  >
                    {part.label}
                  </button>
                </div>
              ))}
            </div>
          </aside>
        )}
        <div
          className={`atlas-stage ${state.view === "pallor" ? "atlas-stage-pallor" : ""}`}
        >
          {!expanded && <div className="atlas-stage-view-tabs">{viewTabs}</div>}
          {expanded && (
            <div className="atlas-presentation-title">
              <h1>
                {state.modality === "conjunctiva" ? "The eye" : "The nail"}
              </h1>
              <p>
                {VIEW_LABELS[state.view]}
                {state.view !== "pallor" && state.view !== "tissue"
                  ? " view"
                  : ""}
              </p>
            </div>
          )}
          <div
            className="anatomy-render-stage"
            aria-label="Educational anatomy models"
          >
            {failed ? (
              fallback
            ) : (
              <AnatomyBoundary
                key={retryKey}
                fallback={fallback}
                onError={() => setFailed(true)}
              >
                <Suspense
                  fallback={
                    <AnatomyPoster modality={state.modality} focused={eye} />
                  }
                >
                  <AnatomyCanvas
                    presentation={expanded}
                    profile={profile}
                    state={{
                      ...state,
                      autoRotate: state.autoRotate && !reduced,
                    }}
                    parts={parts}
                    command={command}
                    cameraMemory={cameraMemory}
                    active={active || expanded}
                    reduced={reduced}
                    onManual={stopRotation}
                    onError={() => setFailed(true)}
                    onSelect={selectPart}
                    onManifest={(value) => {
                      setManifest(value);
                      setState((previous) =>
                        previous.view === "pallor" &&
                        !value.parts.some(
                          (part) =>
                            part.id === previous.selectedPart &&
                            selectablePart(part.id, profile),
                        )
                          ? {
                              ...previous,
                              selectedPart:
                                value.parts.find((p) => p.pallor)?.id ||
                                previous.selectedPart,
                            }
                          : previous,
                      );
                    }}
                  />
                </Suspense>
              </AnatomyBoundary>
            )}
          </div>
          {expanded && inspectorOpen && inspector}
          <div className="atlas-dock">
            {expanded && viewTabs}
            {state.view === "exploded" && (
              <label className="atlas-separation">
                Separation
                <input
                  type="range"
                  min="0"
                  max="100"
                  aria-label="Layer separation"
                  value={Math.round(state.explosion * 100)}
                  onChange={(event) =>
                    patch({
                      explosion: clampUnit(Number(event.target.value) / 100),
                    })
                  }
                  disabled={failed}
                />
              </label>
            )}
            <button
              className="atlas-label-toggle"
              aria-pressed={state.labels}
              onClick={() => patch({ labels: !state.labels })}
              disabled={failed}
            >
              <span className="atlas-switch" />
              Labels
            </button>
            <button
              className="atlas-reset"
              aria-label="Reset anatomy view"
              onClick={() => {
                patch({
                  hiddenParts: [],
                  isolatedPart: null,
                  explosion: eye ? 0 : 0.65,
                });
                camera("reset");
              }}
              disabled={failed}
            >
              <RotateCcw size={16} />
              <span>Reset</span>
            </button>
            <div className="atlas-zoom">
              <button
                className="atlas-icon"
                aria-label="Zoom out anatomy"
                onClick={() => camera("zoom-out")}
                disabled={failed}
              >
                <Minus size={17} />
              </button>
              <button
                className="atlas-icon"
                aria-label="Zoom in anatomy"
                onClick={() => camera("zoom-in")}
                disabled={failed}
              >
                <Plus size={17} />
              </button>
            </div>
          </div>
          <div className="atlas-stage-bottom">
            <span>
              <Move3D size={16} />
              Drag to rotate · {expanded ? "scroll" : "focus and scroll"} to
              zoom
            </span>
            <div>
              <button
                className="atlas-icon"
                aria-label="Front view"
                title="Front view"
                onClick={() => camera("front")}
                disabled={failed}
              >
                <Eye size={17} />
              </button>
              <button
                className="atlas-icon"
                aria-label="Three-quarter view"
                title="Three-quarter view"
                onClick={() => camera("angle")}
                disabled={failed}
              >
                <Layers size={17} />
              </button>
              <button
                className="atlas-icon"
                aria-label={
                  state.autoRotate ? "Pause rotation" : "Auto-rotate anatomy"
                }
                aria-pressed={state.autoRotate}
                onClick={() => patch({ autoRotate: !state.autoRotate })}
                disabled={failed || reduced}
              >
                {state.autoRotate ? (
                  <Pause size={16} />
                ) : (
                  <Rotate3D size={17} />
                )}
              </button>
            </div>
          </div>
          {expanded && (
            <button
              className="atlas-presentation-pallor"
              onClick={() => {
                selectView(
                  state.view === "pallor"
                    ? eye
                      ? "tissue"
                      : "exploded"
                    : "pallor",
                );
                setInspectorOpen(true);
              }}
              disabled={failed}
            >
              <Eye size={17} />
              {state.view === "pallor"
                ? eye
                  ? "Tissue view"
                  : "Explore anatomy"
                : "Pallor comparison"}
              <ArrowRight size={16} />
            </button>
          )}
        </div>
        {!expanded && inspector}
      </div>
      {!expanded && (
        <footer className="atlas-footnote">
          <Info size={15} />
          <p>
            Educational anatomy. Appearance alone cannot confirm or exclude
            anemia. The comparison illustrates color variation; it does not
            measure hemoglobin.
          </p>
        </footer>
      )}
    </div>
  );
  return (
    <section className="atlas-shell" ref={shell}>
      <header className="atlas-page-heading">
        <div>
          <h1>Anatomy explorer</h1>
          <p>Explore screening tissues. Compare visible pallor.</p>
        </div>
        <Button
          ref={expandButton}
          variant="outline"
          aria-label="Expand anatomy explorer"
          onClick={() => setExpanded(true)}
        >
          <Maximize2 data-icon="inline-start" />
          <span className="atlas-presentation-button-text">
            Presentation view
          </span>
        </Button>
      </header>
      {expanded ? (
        <>
          <div className="atlas-presentation-placeholder">
            <Layers size={32} />
            <span>Anatomy is open in presentation view.</span>
            <Button variant="outline" onClick={() => setExpanded(false)}>
              Return to the page
            </Button>
          </div>
          {createPortal(experience, document.body)}
        </>
      ) : (
        experience
      )}
      {state.view === "pallor" && !expanded && (
        <p className="atlas-comparison-description">
          {pallorPart?.description}{" "}
          {state.mode === "compare"
            ? "Both views use"
            : "The tissue transition uses"}{" "}
          the same camera and lighting; skin presets change presentation only.
        </p>
      )}
    </section>
  );
}
