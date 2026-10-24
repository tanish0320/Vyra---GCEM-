import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  FlaskConical,
  ImageOff,
  Info,
  LoaderCircle,
  RotateCcw,
  Upload,
  X,
} from "lucide-react";
import { useSession } from "../context";
import { api, errorMessage } from "../services/api";
import { validateDimensions, validateFile } from "../lib/domain";
import type { Modality } from "../types";
import type { StressReport } from "../integration";
import { Button } from "../components/ui/button";
import {
  NativeSelect,
  NativeSelectOption,
} from "../components/ui/native-select";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { safeMediaUrl } from "./Research";
import "../research.css";

const inRange = (
  value: unknown,
  minimum: number,
  maximum: number,
): value is number =>
  typeof value === "number" &&
  Number.isFinite(value) &&
  value >= minimum &&
  value <= maximum;

function validateReport(value: StressReport): StressReport {
  if (
    !value ||
    typeof value.modality !== "string" ||
    typeof value.baseline_class !== "string" ||
    !inRange(value.baseline_probability, 0, 100) ||
    !inRange(value.overall_shortcut_index, 0, 100) ||
    typeof value.overall_status !== "string" ||
    typeof value.overall_class !== "string" ||
    typeof value.big_reveal_note !== "string" ||
    !Array.isArray(value.conditions) ||
    value.conditions.length !== 8 ||
    !value.conditions.every(
      (condition) =>
        condition &&
        typeof condition.condition_id === "string" &&
        typeof condition.name === "string" &&
        typeof condition.category === "string" &&
        typeof condition.predicted_class === "string" &&
        typeof condition.description === "string" &&
        typeof condition.thumbnail_b64 === "string" &&
        inRange(condition.probability, 0, 1) &&
        inRange(condition.delta_prob, 0, 1) &&
        inRange(condition.percent_shift, 0, 100),
    ) ||
    new Set(value.conditions.map((condition) => condition.condition_id))
      .size !== 8 ||
    !Array.isArray(value.ranked_failure_mechanisms) ||
    !value.ranked_failure_mechanisms.length ||
    !value.ranked_failure_mechanisms.every(
      (mechanism) =>
        mechanism &&
        typeof mechanism.id === "string" &&
        typeof mechanism.name === "string" &&
        typeof mechanism.risk_level === "string" &&
        typeof mechanism.evidence === "string" &&
        typeof mechanism.mechanism_detail === "string" &&
        typeof mechanism.clinical_implication === "string" &&
        inRange(mechanism.sensitivity_score, 0, 100),
    )
  ) {
    throw new Error(
      "The backend returned an incomplete or invalid stress-test report. No experiment scores have been displayed. Please retry.",
    );
  }
  return value;
}

function StressThumbnail({ source, name }: { source: string; name: string }) {
  const [failed, setFailed] = useState(false);
  const valid =
    source.length <= 1_000_000 &&
    /^data:image\/(?:jpeg|png|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(source);
  return valid && !failed ? (
    <img
      src={source}
      alt={`${name} experiment input`}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  ) : (
    <div className="research-image-missing">
      <ImageOff aria-hidden="true" />
      <span>Thumbnail unavailable</span>
    </div>
  );
}

export function AchillesHeel() {
  const { system, serviceError, refreshSystem } = useSession();
  const [params, setParams] = useSearchParams();
  const savedId = params.get("scan_id")?.trim() || "";
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [modality, setModality] = useState<Modality>("conjunctiva");
  const [report, setReport] = useState<StressReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const request = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  const supported = system?.capabilities?.stress_test === true;
  const enabled = Boolean(system?.ready && supported);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      request.current?.abort();
    };
  }, []);
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    setPreviewFailed(false);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  useEffect(() => {
    request.current?.abort();
    setReport(null);
    setError(null);
  }, [savedId]);

  const chooseFile = (selected: File | undefined) => {
    if (!selected || busy) return;
    setReport(null);
    const message = validateFile(selected.type, selected.size);
    setError(message);
    setFile(message ? null : selected);
    if (savedId) setParams({}, { replace: true });
  };
  const run = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (request.current || !enabled) return;
    if (!file && !savedId) {
      setError("Choose a photo before running the experiment.");
      return;
    }
    const controller = new AbortController();
    request.current = controller;
    setBusy(true);
    setError(null);
    setReport(null);
    let timedOut = false;
    const timer = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 120_000);
    try {
      let source = file;
      let tissue = modality;
      let scanRoi: { x0: number; y0: number; x1: number; y1: number } | null = null;
      if (savedId) {
        const scan = await api.scan(savedId, controller.signal);
        if (scan.modality !== "conjunctiva" && scan.modality !== "fingernail")
          throw new Error(
            "This saved screening has an unsupported tissue type. Choose a new photo.",
          );
        const url = safeMediaUrl(scan.artifacts?.oriented);
        if (!url)
          throw new Error(
            "The saved source image is unavailable. Choose a new photo to run an experiment.",
          );
        const response = await fetch(url, {
          signal: controller.signal,
          credentials: "same-origin",
        });
        if (!response.ok)
          throw new Error(
            "The saved source image could not be loaded. The original screening remains available in History.",
          );
        const blob = await response.blob();
        source = new File([blob], `screening-${scan.id}`, { type: blob.type });
        tissue = scan.modality;
        scanRoi = scan.roi || null;
        if (mounted.current && !controller.signal.aborted) setModality(tissue);
      }
      if (!source)
        throw new Error("Choose a photo before running the experiment.");
      const inputError = validateFile(source.type, source.size);
      if (inputError) throw new Error(inputError);
      const bitmap = await createImageBitmap(source);
      const dimensionError = validateDimensions(bitmap.width, bitmap.height);
      const tooLarge = bitmap.width * bitmap.height > 24_000_000;
      bitmap.close();
      if (dimensionError) throw new Error(dimensionError);
      if (tooLarge)
        throw new Error("Choose a photo no larger than 24 megapixels.");
      if (controller.signal.aborted)
        throw new DOMException("Aborted", "AbortError");
      const value = validateReport(
        await api.stressTest(source, tissue, controller.signal, scanRoi),
      );
      if (mounted.current && !controller.signal.aborted) setReport(value);
    } catch (caught) {
      if (mounted.current && request.current === controller)
        setError(
          timedOut
            ? "The experiment exceeded two minutes. The server may still finish its work. Please retry after checking backend availability."
            : controller.signal.aborted
              ? "This request was cancelled. The server may still finish the experiment. Your original screening is unchanged."
              : caught instanceof DOMException &&
                  caught.name === "InvalidStateError"
                ? "This image could not be decoded. Choose an intact JPEG, PNG, or WebP photo."
                : errorMessage(caught),
        );
    } finally {
      window.clearTimeout(timer);
      if (request.current === controller) {
        request.current = null;
        if (mounted.current) setBusy(false);
      }
    }
  };

  return (
    <div className="workspace-page wrap research-page stress-page">
      <div className="workspace-heading">
        <div>
          <span className="eyebrow">MODEL RESEARCH TOOL</span>
          <h1>Achilles Heel</h1>
          <p>
            See how the model responds when one image is changed in eight
            controlled conditions.
          </p>
        </div>
        <Button asChild variant="outline">
          <Link to="/evidence">
            View model evidence <ArrowRight data-icon="inline-end" />
          </Link>
        </Button>
      </div>
      <Alert>
        <Info />
        <AlertTitle>An experiment on image sensitivity</AlertTitle>
        <AlertDescription>
          The server compares a baseline with simulated lighting, framing, blur
          and tissue-masking conditions. These scores describe this experiment;
          they are not clinical accuracy or proof of a causal model shortcut.
          The original screening result is kept separately.
        </AlertDescription>
      </Alert>
      {!supported && (
        <Alert>
          <AlertTitle>Stress testing is not connected yet</AlertTitle>
          <AlertDescription>
            {serviceError ||
              "The backend has not enabled the stress-test capability. Connect the adapter before running an experiment."}
            <Button variant="outline" onClick={() => void refreshSystem()}>
              Check backend again
            </Button>
            <a className="research-inline-link" href="/achilles-heel">
              Open the legacy research tool <ArrowRight size={15} />
            </a>
          </AlertDescription>
        </Alert>
      )}
      {supported && !system?.ready && (
        <Alert>
          <AlertTitle>Model is not ready</AlertTitle>
          <AlertDescription>
            Start the configured backend and verify that its model is available
            before running this experiment.
            <Button variant="outline" onClick={() => void refreshSystem()}>
              Check model again
            </Button>
          </AlertDescription>
        </Alert>
      )}
      <form
        className="research-card stress-form"
        onSubmit={run}
        aria-busy={busy}
      >
        <div className="stress-input-copy">
          <FlaskConical size={26} aria-hidden="true" />
          <h2>Choose an experiment image</h2>
          <p>
            Use a new photo or the saved source linked from a screening. Running
            this tool does not create another patient screening.
          </p>
        </div>
        <div className="stress-input-controls">
          {savedId ? (
            <div className="stress-saved-source">
              <span className="eyebrow">SAVED SOURCE IMAGE</span>
              <strong>Screening {savedId}</strong>
              <p>
                The original image will load when you run the experiment. Its
                saved tissue type will be used.
              </p>
              <Link
                className="research-inline-link"
                to={`/history/${encodeURIComponent(savedId)}`}
              >
                <ArrowLeft size={15} /> Back to saved screening
              </Link>
              <Button
                type="button"
                variant="outline"
                disabled={busy}
                onClick={() => setParams({}, { replace: true })}
              >
                Choose a different photo
              </Button>
            </div>
          ) : (
            <>
              <label className="stress-file-label" htmlFor="stress-photo">
                <Upload size={20} /> Experiment photo
              </label>
              <input
                id="stress-photo"
                className="research-file-input"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                disabled={busy}
                onChange={(event) => chooseFile(event.currentTarget.files?.[0])}
              />
              <p className="research-help">
                JPEG, PNG or WebP · up to 10 MB · at least 50 × 50 pixels.
              </p>
              <label
                className="stress-modality-label"
                htmlFor="stress-modality"
              >
                Anatomical modality
              </label>
              <NativeSelect
                id="stress-modality"
                value={modality}
                disabled={busy}
                onChange={(event) => {
                  setModality(event.target.value as Modality);
                  setReport(null);
                }}
              >
                <NativeSelectOption value="conjunctiva">
                  Eye conjunctiva
                </NativeSelectOption>
                <NativeSelectOption value="fingernail">
                  Fingernail
                </NativeSelectOption>
              </NativeSelect>
              {preview && (
                <figure className="stress-source-preview">
                  {previewFailed ? (
                    <span>
                      Photo preview unavailable. Choose another image if this
                      file is not intact.
                    </span>
                  ) : (
                    <img
                      src={preview}
                      alt="Selected experiment photo"
                      onError={() => setPreviewFailed(true)}
                    />
                  )}
                  <figcaption>{file?.name}</figcaption>
                </figure>
              )}
            </>
          )}
          <div className="stress-run-actions">
            <Button
              type="submit"
              disabled={!enabled || busy || (!file && !savedId)}
            >
              {busy ? (
                <LoaderCircle className="spin" data-icon="inline-start" />
              ) : (
                <FlaskConical data-icon="inline-start" />
              )}
              {busy
                ? "Running experiment…"
                : error
                  ? "Retry stress test"
                  : "Run stress test"}
            </Button>
            {busy && (
              <Button
                type="button"
                variant="outline"
                onClick={() => request.current?.abort()}
              >
                <X data-icon="inline-start" /> Cancel request
              </Button>
            )}
          </div>
          {busy && (
            <p className="research-help" role="status">
              Waiting for the server’s complete experiment report. No stage
              percentages are simulated.
            </p>
          )}
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Experiment not completed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>
      </form>
      {report && (
        <section aria-label="Stress-test results" className="stress-results">
          <div className="research-section-heading">
            <div>
              <span className="eyebrow">SERVER EXPERIMENT RESULTS</span>
              <h2>What changed across conditions</h2>
            </div>
            <span>
              {report.modality === "fingernail"
                ? "Fingernail"
                : "Eye conjunctiva"}
            </span>
          </div>
          <div className="research-metrics stress-summary">
            <div>
              <span>Baseline anemia probability</span>
              <strong>{report.baseline_probability.toFixed(1)}%</strong>
              <small>Baseline class: {report.baseline_class}</small>
            </div>
            <div>
              <span>Overall shortcut index</span>
              <strong>
                {report.overall_shortcut_index.toFixed(1)}
                <em>/100</em>
              </strong>
              <small>Server-generated experiment score</small>
            </div>
            <div className="stress-interpretation">
              <span>Server interpretation</span>
              <p>{report.overall_status}</p>
              <small>
                This label is an experimental heuristic, not clinical
                validation.
              </small>
            </div>
          </div>
          <div
            className="stress-chart research-card"
            aria-labelledby="stress-chart-title"
          >
            <h3 id="stress-chart-title">Anemia probability by condition</h3>
            <p>
              Bars use a common 0–100% scale. Change is the absolute difference
              from the baseline in percentage points.
            </p>
            <div className="stress-chart-rows">
              {report.conditions.map((condition) => (
                <div className="stress-chart-row" key={condition.condition_id}>
                  <span>{condition.name}</span>
                  <div className="stress-bar-track" aria-hidden="true">
                    <span
                      style={{ width: `${condition.probability * 100}%` }}
                    />
                  </div>
                  <strong>{(condition.probability * 100).toFixed(1)}%</strong>
                  <small>Δ {condition.percent_shift.toFixed(1)} pp</small>
                </div>
              ))}
            </div>
          </div>
          <div className="stress-condition-grid">
            {report.conditions.map((condition) => (
              <article
                className="research-card stress-condition"
                key={condition.condition_id}
              >
                <div className="stress-condition-image">
                  <StressThumbnail
                    key={condition.thumbnail_b64}
                    source={condition.thumbnail_b64}
                    name={condition.name}
                  />
                </div>
                <div>
                  <span className="eyebrow">{condition.category}</span>
                  <h3>{condition.name}</h3>
                  <p>{condition.description}</p>
                  <dl>
                    <div>
                      <dt>Anemia probability</dt>
                      <dd>{(condition.probability * 100).toFixed(1)}%</dd>
                    </div>
                    <div>
                      <dt>Change from baseline</dt>
                      <dd>{condition.percent_shift.toFixed(1)} pp</dd>
                    </div>
                    <div>
                      <dt>Model class</dt>
                      <dd>{condition.predicted_class}</dd>
                    </div>
                  </dl>
                </div>
              </article>
            ))}
          </div>
          <div className="research-section-heading">
            <h2>Ranked sensitivity hypotheses</h2>
            <span>Order supplied by the server</span>
          </div>
          <ol className="stress-mechanisms">
            {report.ranked_failure_mechanisms.map((mechanism, index) => (
              <li className="research-card" key={`${mechanism.id}-${index}`}>
                <div className="stress-mechanism-heading">
                  <span className="stress-rank">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <div>
                    <h3>{mechanism.name}</h3>
                    <span className="research-help">
                      Server label: {mechanism.risk_level}
                    </span>
                  </div>
                  <strong>
                    {mechanism.sensitivity_score.toFixed(1)}
                    <small>/100</small>
                  </strong>
                </div>
                <p>{mechanism.evidence}</p>
                <details>
                  <summary>Mechanism and interpretation</summary>
                  <p>{mechanism.mechanism_detail}</p>
                  <p>{mechanism.clinical_implication}</p>
                </details>
              </li>
            ))}
          </ol>
          {report.big_reveal_note && (
            <Alert>
              <Info />
              <AlertTitle>How to interpret the experiment</AlertTitle>
              <AlertDescription>{report.big_reveal_note}</AlertDescription>
            </Alert>
          )}
          <div className="research-page-actions">
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setReport(null);
                document.getElementById("stress-photo")?.focus();
              }}
            >
              <RotateCcw data-icon="inline-start" /> Clear experiment results
            </Button>
            <Button asChild variant="outline">
              <Link to="/research">
                Explore research assets <ArrowRight data-icon="inline-end" />
              </Link>
            </Button>
          </div>
        </section>
      )}
    </div>
  );
}
