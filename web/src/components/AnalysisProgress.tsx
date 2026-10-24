import { useEffect, useRef } from "react";
import {
  Check,
  ScanLine,
  LoaderCircle,
  RotateCcw,
  X,
  Circle,
} from "lucide-react";
import type { Job, StageName } from "../types";
import { STAGES } from "./LivePipeline";
import { Button } from "./ui/button";
import { Alert, AlertTitle, AlertDescription } from "./ui/alert";
import RefineFrame, { type RefineFrameStatus } from "./RefineFrame";

const GROUPS: { title: string; stages: StageName[]; detail: string }[] = [
  {
    title: "Image & region review",
    stages: ["validation", "detection", "roi"],
    detail: "Check the photo and confirm the tissue to analyze.",
  },
  {
    title: "Normalization & colour features",
    stages: ["normalization", "features"],
    detail: "Prepare the confirmed crop and extract its colour features.",
  },
  {
    title: "Prediction & explanation",
    stages: ["inference", "explanation", "saving"],
    detail: "Run the model, generate its attention map, and save the result.",
  },
];
const terminal = new Set(["completed", "failed", "cancelled", "interrupted"]);
const done = (status?: string) =>
  status === "completed" || status === "complete" || status === "skipped";

const getRefineStatus = (
  status: Job["status"],
  stage: Job["stage"],
): RefineFrameStatus => {
  if (status === "failed" || status === "cancelled" || status === "interrupted")
    return "error";
  if (status === "completed") return "complete";
  if (stage === "validation" || stage === "detection" || stage === "roi")
    return "queued";
  if (stage === "normalization" || stage === "features") return "generating";
  if (stage === "inference" || stage === "explanation" || stage === "saving")
    return "refining";
  return "generating";
};
export function AnalysisProgress({
  job,
  busy,
  onCancel,
  onRestart,
  previewUrl,
  previewWidth,
  previewHeight,
}: {
  job: Job;
  busy: boolean;
  onCancel: () => void;
  onRestart: () => void;
  previewUrl?: string;
  previewWidth?: number;
  previewHeight?: number;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    heading.current?.focus();
  }, [job.id]);
  const ended = terminal.has(job.status);
  const active = STAGES.find((stage) => stage.name === job.stage);
  const title =
    job.status === "completed"
      ? "Opening your result"
      : job.status === "failed"
        ? "Screening could not finish"
        : job.status === "cancelled"
          ? "Screening cancelled"
          : job.status === "interrupted"
            ? "Screening interrupted"
            : job.status === "detecting" || job.stage === "validation"
              ? "Preparing your image"
              : "Analyzing your image";
  return (
    <section className="analysis-page" aria-label="Live screening progress">
      <div className={"analysis-emblem" + (!ended ? " is-processing" : "")}>
        <ScanLine size={36} />
      </div>
      <span className="eyebrow">
        {ended ? "SCREENING STATUS" : "LOCAL MODEL · LIVE PROCESSING"}
      </span>
      <h2 ref={heading} tabIndex={-1}>
        {title}
      </h2>
      <p className="analysis-intro">
        {job.status === "interrupted"
          ? "The service restarted before this screening finished. Start again to create a new analysis."
          : job.status === "cancelled"
            ? "This screening stopped. No prediction has been reported."
            : job.status === "failed"
              ? "The service could not complete this screening. Review the message below before trying again."
              : "Each stage updates when the service completes its work. You can return to this page while processing continues."}
      </p>
      {job.patient && (
        <div className="analysis-patient">
          <strong>{job.patient.reference || "Earlier screening"}</strong>
          <span>
            {job.patient.age === null
              ? "Age not supplied"
              : job.patient.age + " years"}{" "}
            · {job.modality === "conjunctiva" ? "Lower eyelid" : "Fingernail"}
          </span>
        </div>
      )}
      {(job.image?.url || previewUrl) && (
        <div className="analysis-refine-container">
          <RefineFrame
            status={getRefineStatus(job.status, job.stage)}
            aspectRatio={`${job.image?.width || previewWidth || 4} / ${job.image?.height || previewHeight || 3}`}
            width={340}
            radius={14}
            background="#0d1b1a"
            color="#026e6d"
            stageDuration={420}
            sweep={true}
            showStatus={true}
            hideAfter={0}
            labels={{
              queued: "Awaiting preprocessing",
              generating: "Extracting colorimetry & features",
              refining: "Synthesizing microvascular attention",
              complete: "Inference finalized",
              error: "Inference interrupted",
            }}
          >
            <img
              src={job.image?.url || previewUrl}
              alt="Tissue scan processing frame"
              crossOrigin="anonymous"
            />
          </RefineFrame>
        </div>
      )}
      <ol className="analysis-stages">
        {GROUPS.map((group, index) => {
          const states = group.stages.map((name) =>
            job.stages.find((stage) => stage.name === name),
          );
          const complete = states.every((stage) => done(stage?.status));
          const running =
            !ended && states.some((stage) => stage?.status === "running");
          const skipped = states.some((stage) => stage?.status === "skipped");
          return (
            <li
              key={group.title}
              data-state={
                complete ? "complete" : running ? "running" : "pending"
              }
            >
              <span className="analysis-stage-number">
                {complete ? (
                  <Check size={18} />
                ) : running ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  index + 1
                )}
              </span>
              <div>
                <h3>{group.title}</h3>
                <p>{running ? active?.description : group.detail}</p>
                <small>
                  {complete
                    ? skipped
                      ? "Finished with an unavailable stage"
                      : "Complete"
                    : running
                      ? active?.title
                      : ended
                        ? "Not completed"
                        : "Waiting"}
                </small>
              </div>
            </li>
          );
        })}
      </ol>
      <p className="analysis-clock" role="status" aria-live="polite">
        <Circle size={7} fill="currentColor" />
        {job.status === "queued"
          ? "Queued"
          : ended
            ? title
            : active?.title} · {job.elapsed_seconds.toFixed(1)} s elapsed
      </p>
      {job.error && (
        <Alert variant="destructive">
          <AlertTitle>Processing stopped</AlertTitle>
          <AlertDescription>{job.error.message}</AlertDescription>
        </Alert>
      )}
      <div className="analysis-actions">
        {ended && job.status !== "completed" ? (
          <Button onClick={onRestart}>
            <RotateCcw data-icon="inline-start" />
            Start again
          </Button>
        ) : !ended ? (
          <Button variant="outline" disabled={busy} onClick={onCancel}>
            <X data-icon="inline-start" />
            Cancel screening
          </Button>
        ) : (
          <LoaderCircle className="spin" aria-label="Loading saved result" />
        )}
      </div>
    </section>
  );
}
