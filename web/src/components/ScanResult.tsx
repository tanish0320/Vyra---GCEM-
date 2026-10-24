import { useEffect, useRef, useState } from "react";
import {
  Download,
  ArrowUpRight,
  ImageOff,
  Plus,
  Printer,
  History,
  ShieldCheck,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import type { Scan } from "../types";
import { percent, displayDate } from "../lib/domain";
import { api } from "../services/api";
import { useSession } from "../context";
import { Button } from "./ui/button";
import { AssistantCard } from "./AssistantCard";
import FlipCard from "./FlipCard";
import { ColorimetryResults, FusionResults } from "./IntegrationResults";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "./ui/dialog";
import "../screening.css";
import "../integration-results.css";

function reportUrl(scan: Scan) {
  const fallback = /^\d+$/.test(scan.id)
    ? `/export/pdf/${encodeURIComponent(scan.id)}`
    : api.report(scan.id);
  if (!scan.pdf_url) return fallback;
  try {
    const url = new URL(scan.pdf_url, window.location.origin);
    return url.origin === window.location.origin &&
      /^https?:$/.test(url.protocol)
      ? url.pathname + url.search
      : fallback;
  } catch {
    return fallback;
  }
}
export const FEATURE_NAMES = [
  "Brightness mean",
  "Brightness standard deviation",
  "Brightness 10th percentile",
  "Brightness 25th percentile",
  "Brightness median",
  "Brightness 75th percentile",
  "Brightness 90th percentile",
  "Redness ratio mean",
  "Redness ratio standard deviation",
  "White pallor ratio",
  "Pink ratio",
  "Red mean",
  "Red standard deviation",
  "Green mean",
  "Green standard deviation",
  "Blue mean",
  "Blue standard deviation",
  "Red / green ratio",
  "Red / blue ratio",
  "Red–blue contrast",
  "Hemoglobin color proxy mean",
  "Hemoglobin color proxy deviation",
  "Top–bottom gradient",
  "Center–surround contrast",
  "Gradient magnitude mean",
  "Gradient magnitude deviation",
  "Local texture variance",
];
export function ScanResult({ scan }: { scan: Scan }) {
  const [preview, setPreview] = useState<{ name: string; url: string } | null>(
    null,
  );
  const [comparison, setComparison] = useState(50);
  const [attentionMode, setAttentionMode] = useState<"split" | "blend" | "flip">("split");
  const [attentionSite, setAttentionSite] = useState<"primary" | "secondary">("primary");
  const hasDualScan = Boolean(
    scan.multimodal_fusion?.active &&
      (scan.multimodal_fusion?.secondary_artifacts || scan.multimodal_fusion?.secondary_image_url)
  );
  const primaryModalityName = scan.multimodal_fusion?.primary_modality || (scan.modality === "conjunctiva" ? "Eye Conjunctiva" : "Fingernail Bed");
  const secondaryModalityName = scan.multimodal_fusion?.secondary_modality || (scan.modality === "conjunctiva" ? "Fingernail Bed" : "Eye Conjunctiva");
  const currentArtifacts = (hasDualScan && attentionSite === "secondary" && scan.multimodal_fusion?.secondary_artifacts)
    ? {
        oriented: scan.multimodal_fusion.secondary_artifacts.oriented || scan.artifacts.oriented,
        roi: scan.multimodal_fusion.secondary_artifacts.roi || scan.artifacts.roi,
        normalized: scan.multimodal_fusion.secondary_artifacts.normalized || scan.artifacts.normalized,
        gradcam: scan.multimodal_fusion.secondary_artifacts.gradcam || null,
        heatmap: scan.multimodal_fusion.secondary_artifacts.heatmap || null,
      }
    : scan.artifacts;
  const currentAttentionUrl = currentArtifacts.heatmap || currentArtifacts.gradcam;
  const currentNormalized = currentArtifacts.normalized;
  const previewTrigger = useRef<HTMLButtonElement | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const { reset, job, system } = useSession();
  const navigate = useNavigate();
  useEffect(() => {
    resultHeading.current?.focus();
    setComparison(50);
    setAttentionMode("split");
    setAttentionSite("primary");
    setPreview(null);
  }, [scan.id]);
  const positive = scan.probability_anemia >= scan.threshold;
  const eye = scan.modality === "conjunctiva";
  const hasEyeHb =
    eye && typeof scan.hb_g_dl === "number" && Number.isFinite(scan.hb_g_dl);
  const attentionUrl = scan.artifacts.heatmap || scan.artifacts.gradcam;
  const rawHeatmap = Boolean(scan.artifacts.heatmap);
  const activeJob =
    job &&
    !["completed", "failed", "cancelled", "interrupted"].includes(job.status);
  const guidance = scan.follow_up || {
    title: "Follow-up guidance",
    summary: positive
      ? "Arrange a clinical assessment and confirmatory blood test."
      : "A screen-negative result does not rule out anemia.",
    steps: [
      positive
        ? "Share this screening record with a qualified clinician."
        : "Seek a clinical assessment if symptoms persist or you have concerns.",
      ...(scan.warnings.length
        ? [
            "Retake the photo in even light with sharp focus to address the image-quality limitations.",
          ]
        : []),
    ],
    model_input_note:
      "Patient details accompany this record; they do not change the model prediction.",
  };
  const artifact = (name: string, url: string, main = false) => (
    <figure key={name} className={"result-image" + (main ? " main-image" : "")}>
      <button
        className="artifact-open"
        aria-label={"Enlarge " + name}
        onClick={(event) => {
          previewTrigger.current = event.currentTarget;
          setPreview({ name, url });
        }}
      >
        <img src={url} alt={name} loading="lazy" />
      </button>
      <figcaption>{name}</figcaption>
    </figure>
  );
  return (
    <div className="scan-result result-screen">
      <section className="result-overview" data-positive={positive}>
        <span className="eyebrow">
          SCREENING RESULT ·{" "}
          {eye ? "EYE / CONJUNCTIVA" : "FINGERNAIL / NAIL BED"}
        </span>
        <h1 ref={resultHeading} tabIndex={-1}>
          {positive ? "Screen positive" : "Screen negative"}
        </h1>
        <p>
          {positive
            ? "The image falls above the model’s screening threshold. A clinical assessment and blood test are needed to confirm anemia."
            : "The image falls below the model’s screening threshold. This result does not exclude anemia or replace a clinical assessment."}
        </p>
        <p className="result-case">
          <strong>{scan.reference || "Reference not supplied"}</strong>
          <span>·</span>
          <span>{displayDate(scan.created_at)}</span>
          <span>· Saved locally</span>
        </p>
      </section>
      <div className="result-columns">
        <section className="result-panel">
          <span className="eyebrow">01 / ACQUISITION</span>
          <h2>Image & quality</h2>
          {hasDualScan && (
            <div className="attention-modes" role="group" aria-label="Select tissue for acquisition" style={{ marginBottom: "14px" }}>
              <button
                type="button"
                aria-pressed={attentionSite === "primary"}
                onClick={() => setAttentionSite("primary")}
              >
                {primaryModalityName}
              </button>
              <button
                type="button"
                aria-pressed={attentionSite === "secondary"}
                onClick={() => setAttentionSite("secondary")}
              >
                {secondaryModalityName}
              </button>
            </div>
          )}
          {artifact("Oriented image", currentArtifacts.oriented, true)}
          <div className="result-crops">
            {artifact("Confirmed region", currentArtifacts.roi)}
            {artifact("Normalized input", currentArtifacts.normalized)}
          </div>
          <div className="quality-summary">
            <h3>Image quality notes</h3>
            {scan.warnings.length ? (
              <ul>
                {scan.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            ) : (
              <p>
                No image-quality warnings were recorded. Lighting and region
                selection can still affect the output.
              </p>
            )}
          </div>
        </section>
        <section className="result-panel">
          <span className="eyebrow">02 / MODEL OUTPUT</span>
          <h2>Screening prediction</h2>
          <div className="probability-reading">
            <div className="reading-label">Model probability of anemia</div>
            <p className="large-reading">{percent(scan.probability_anemia)}</p>
            <div
              className="probability-track"
              role="img"
              aria-label={
                "Model probability " +
                percent(scan.probability_anemia) +
                ", threshold " +
                percent(scan.threshold)
              }
            >
              <span style={{ width: percent(scan.probability_anemia) }} />
              <i style={{ left: percent(scan.threshold) }} />
            </div>
            <div className="reading-scale">
              <span>0%</span>
              <span>Threshold {percent(scan.threshold)}</span>
              <span>100%</span>
            </div>
          </div>
          <div className="hb-reading">
            <div className="reading-label">Estimated hemoglobin</div>
            <p className="large-reading">
              {hasEyeHb ? scan.hb_g_dl!.toFixed(2) : "—"} <small>g/dL</small>
            </p>
            <h3>
              {hasEyeHb
                ? "Image-based estimate"
                : eye
                  ? "Estimate unavailable"
                  : "Unavailable for nails"}
            </h3>
            <p>
              {hasEyeHb
                ? "An independent model estimate. A laboratory blood test is needed to measure hemoglobin."
                : eye
                  ? "No valid hemoglobin estimate was supplied for this image."
                  : "The supplied model has no validated nail hemoglobin regression. No estimate is generated."}
            </p>
          </div>
          <dl className="patient-record">
            <div>
              <dt>Patient / case</dt>
              <dd>{scan.reference || "Not supplied"}</dd>
            </div>
            <div>
              <dt>Age</dt>
              <dd>
                {scan.age === null || scan.age === undefined
                  ? "Not supplied"
                  : scan.age + " years"}
              </dd>
            </div>
            <div>
              <dt>Biological sex</dt>
              <dd>
                {scan.sex === "female"
                  ? "Female"
                  : scan.sex === "male"
                    ? "Male"
                    : "Not stated"}
              </dd>
            </div>
            {scan.phone && (
              <div>
                <dt>Phone</dt>
                <dd>{scan.phone}</dd>
              </div>
            )}
            <div>
              <dt>Analysis time</dt>
              <dd>{scan.processing_seconds.toFixed(2)} s</dd>
            </div>
          </dl>
          {scan.notes && (
            <p className="result-notes">
              <strong>Clinical / field notes</strong>
              {scan.notes}
            </p>
          )}
        </section>
        <section className="result-panel">
          <span className="eyebrow">03 / EXPLANATION</span>
          <h2>Model attention</h2>
          <p className="attention-intro">
            Explore the areas that influenced this prediction. Both views show
            the same normalized tissue crop.
          </p>
          {hasDualScan && (
            <div className="attention-modes" role="group" aria-label="Select tissue for attention" style={{ marginBottom: "14px" }}>
              <button
                type="button"
                id="btnSitePrimary"
                aria-pressed={attentionSite === "primary"}
                onClick={() => {
                  setAttentionSite("primary");
                  setComparison(50);
                }}
              >
                {primaryModalityName}
              </button>
              <button
                type="button"
                id="btnSiteSecondary"
                aria-pressed={attentionSite === "secondary"}
                onClick={() => {
                  setAttentionSite("secondary");
                  setComparison(50);
                }}
              >
                {secondaryModalityName}
              </button>
            </div>
          )}
          {currentAttentionUrl ? (
            <>
              <div
                className="attention-modes"
                role="group"
                aria-label="Attention display mode"
              >
                <button
                  type="button"
                  id="btnModeSplit"
                  className="cursor-target"
                  data-cursor-label="Split View"
                  aria-pressed={attentionMode === "split"}
                  onClick={() => {
                    setAttentionMode("split");
                    setComparison(50);
                  }}
                >
                  Split view
                </button>
                <button
                  type="button"
                  id="btnModeBlend"
                  className="cursor-target"
                  data-cursor-label="Blend View"
                  aria-pressed={attentionMode === "blend"}
                  onClick={() => {
                    setAttentionMode("blend");
                    setComparison(50);
                  }}
                >
                  Blend view
                </button>
                <button
                  type="button"
                  id="btnModeFlip"
                  className="cursor-target"
                  data-cursor-label="3D Flip Card"
                  aria-pressed={attentionMode === "flip"}
                  onClick={() => {
                    setAttentionMode("flip");
                  }}
                >
                  3D Flip card
                </button>
              </div>
              {attentionMode === "flip" ? (
                <div
                  id="explainabilityViewer"
                  className="flip-card-wrapper cursor-target"
                  data-cursor-label="Click or Drag to Flip"
                  data-mode="flip"
                >
                  <FlipCard
                    width={480}
                    height={320}
                    radius={16}
                    background="#18181b"
                    color="#f4f4f5"
                    shadowColor="#000000"
                    shadowOpacity={0.4}
                    tilt={true}
                    tiltMax={12}
                    glare={true}
                    glareOpacity={0.25}
                    hoverScale={1.02}
                    front={
                      <div className="flip-card-face-content">
                        <img
                          src={currentNormalized}
                          alt={`Normalized ${attentionSite === "primary" ? primaryModalityName : secondaryModalityName} crop`}
                        />
                        <div className="flip-face-badge">
                          <span>Normalized Input</span>
                          <span className="badge-sub">{attentionSite === "primary" ? primaryModalityName : secondaryModalityName}</span>
                        </div>
                      </div>
                    }
                    back={
                      <div className="flip-card-face-content">
                        <img
                          src={currentAttentionUrl}
                          alt={`Grad-CAM attention map for ${attentionSite === "primary" ? primaryModalityName : secondaryModalityName}`}
                        />
                        <div className="flip-face-badge badge-accent">
                          <span>Grad-CAM Attention</span>
                          <span className="badge-sub">{rawHeatmap ? "Raw Heatmap" : "Microvascular Saliency"}</span>
                        </div>
                      </div>
                    }
                  />
                </div>
              ) : (
                <div
                  id="explainabilityViewer"
                  className="attention-compare"
                  data-mode={attentionMode}
                >
                  <img
                    src={currentNormalized}
                    alt="Normalized crop for attention comparison"
                  />
                  <img
                    src={currentAttentionUrl}
                    alt={
                      rawHeatmap
                        ? "Raw Grad-CAM heatmap"
                        : "Grad-CAM attention overlay"
                    }
                    style={
                      attentionMode === "split"
                        ? { clipPath: "inset(0 0 0 " + comparison + "%)" }
                        : { opacity: comparison / 100 }
                    }
                  />
                  {attentionMode === "split" && (
                    <span
                      className="comparison-line"
                      style={{ left: comparison + "%" }}
                    />
                  )}
                </div>
              )}
              {attentionMode === "flip" ? (
                <div className="flip-card-hint">
                  <span>✨ <strong>3D Flip Card:</strong> Click to flip or drag/flick horizontally to turn. Hover leans toward cursor with dynamic glare.</span>
                </div>
              ) : (
                <label className="attention-control">
                  {attentionMode === "split"
                    ? `Compare normalized input and Grad-CAM (${attentionSite === "primary" ? primaryModalityName : secondaryModalityName})`
                    : rawHeatmap
                      ? "Heatmap opacity"
                      : "Precomposited overlay visibility"}
                  <input
                    id="gradcamSlider"
                    type="range"
                    min={0}
                    max={100}
                    value={comparison}
                    onChange={(event) =>
                      setComparison(Number(event.target.value))
                    }
                    aria-label="Attention comparison"
                  />
                </label>
              )}
              <div className="attention-labels">
                <span>Normalized input ({attentionSite === "primary" ? primaryModalityName : secondaryModalityName})</span>
                <span>{rawHeatmap ? "Raw heatmap" : "Grad-CAM overlay"}</span>
              </div>
              <p className="attention-mode-note">
                {rawHeatmap
                  ? "The raw heatmap is aligned with the normalized region. Blend adjusts its opacity over that same crop."
                  : "The supplied PNG already combines the heatmap and normalized crop. Blend cross-fades that image; it does not change the underlying attention values."}
              </p>
            </>
          ) : (
            <div className="artifact-unavailable">
              <ImageOff size={24} />
              <span>Attention map unavailable for {attentionSite === "primary" ? primaryModalityName : secondaryModalityName}</span>
            </div>
          )}
          <p className="artifact-caption">
            {scan.explanation.message ||
              "Grad-CAM highlights features that influenced the model; it does not locate a disease."}
          </p>
        </section>
      </div>
      <ColorimetryResults metrics={scan.biophysical_metrics} />
      <FusionResults fusion={scan.multimodal_fusion} />
      <section className="follow-up-panel">
        <span className="eyebrow">
          NEXT STEPS · SEPARATE FROM MODEL INFERENCE
        </span>
        <h2>{guidance.title}</h2>
        <p className="follow-up-summary">{guidance.summary}</p>
        <div className="follow-up-grid">
          <ul>
            {guidance.steps.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ul>
          <div className="clinical-note">
            <h3>
              <ShieldCheck
                size={16}
                style={{
                  display: "inline",
                  verticalAlign: "middle",
                  marginRight: 7,
                }}
              />
              Clinical confirmation
            </h3>
            <p>
              This is research screening support. A laboratory blood test and
              clinical assessment are needed to diagnose anemia.
            </p>
            <p style={{ marginTop: 10 }}>{guidance.model_input_note}</p>
            <p style={{ marginTop: 10 }}>
              <Link to="/evidence">View measured model performance</Link>
            </p>
          </div>
        </div>
      </section>
      <AssistantCard
        key={scan.id}
        scanId={scan.id}
        enabled={system ? system.capabilities?.assistant !== false : true}
      />
      <div className="result-actions">
        <Button asChild>
          <a id="downloadPdfButton" href={reportUrl(scan)} download className="cursor-target" data-cursor-label="Download PDF">
            <Download data-icon="inline-start" />
            Download PDF
          </a>
        </Button>
        <Button
          variant="outline"
          onClick={() => {
            if (!activeJob) reset();
            navigate("/studio");
          }}
        >
          <Plus data-icon="inline-start" />
          {activeJob ? "Resume active screening" : "New screening"}
        </Button>
        <Button asChild variant="outline">
          <Link to="/history">
            <History data-icon="inline-start" />
            View history
          </Link>
        </Button>
        <Button className="cursor-target" data-cursor-label="Print Report" variant="outline" onClick={() => window.print()}>
          <Printer data-icon="inline-start" />
          Print
        </Button>
        {system?.capabilities?.stress_test && (
          <Button asChild variant="outline">
            <a
              id="stressTestLink"
              href={`/achilles-heel?scan_id=${encodeURIComponent(scan.id)}`}
            >
              Test image robustness <ArrowUpRight size={16} />
            </a>
          </Button>
        )}
        {system?.capabilities?.stress_test && (
          <Button asChild variant="outline">
            <Link to={`/achilles-heel?scan_id=${encodeURIComponent(scan.id)}`}>
              Open stress workspace <ArrowUpRight size={16} />
            </Link>
          </Button>
        )}
      </div>
      <Dialog
        open={preview !== null}
        onOpenChange={(open) => {
          if (!open) setPreview(null);
        }}
      >
        <DialogContent
          className="artifact-dialog"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            previewTrigger.current?.focus();
          }}
        >
          <DialogHeader>
            <DialogTitle>{preview?.name}</DialogTitle>
            <DialogDescription>
              Actual image from this screening’s processing record.
            </DialogDescription>
          </DialogHeader>
          {preview && (
            <img src={preview.url} alt={`${preview.name}, enlarged`} />
          )}
        </DialogContent>
      </Dialog>
      <details className="feature-details">
        <summary>Inspect 27 color features and model details</summary>
        <div className="trace-metadata">
          <div>
            <span>Checkpoint</span>
            <code>{scan.model_sha256}</code>
          </div>
          <div>
            <span>Normalization</span>
            <code>{scan.normalization_version}</code>
          </div>
          <div>
            <span>Calibration temperature</span>
            <strong>{scan.temperature.toFixed(6)}</strong>
          </div>
          <div>
            <span>Processing time</span>
            <strong>{scan.processing_seconds.toFixed(2)} seconds</strong>
          </div>
          <div>
            <span>Confirmed region</span>
            <strong>
              {scan.roi.x1 - scan.roi.x0} × {scan.roi.y1 - scan.roi.y0} px
            </strong>
          </div>
        </div>
        <div className="feature-table">
          {scan.features.map((value, index) => (
            <div key={index}>
              <span>
                {String(index + 1).padStart(2, "0")} ·{" "}
                {FEATURE_NAMES[index] || `Feature ${index + 1}`}
              </span>
              <code>{value.toFixed(5)}</code>
            </div>
          ))}
        </div>
      </details>
      <div className="result-end">
        <span>
          {displayDate(scan.created_at)} · Result {scan.id.slice(0, 12)}
        </span>
        <Link className="text-link" to="/history">
          Open screening history <ArrowUpRight size={16} />
        </Link>
      </div>
    </div>
  );
}
