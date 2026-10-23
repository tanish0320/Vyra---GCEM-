import { Check, Circle, LoaderCircle } from "lucide-react";
import type { Job, StageName } from "../types";
import { cn } from "../lib/utils";
export const STAGES: { name: StageName; title: string; description: string }[] =
  [
    {
      name: "validation",
      title: "Image validation",
      description:
        "Decode the image, apply orientation, and check its dimensions.",
    },
    {
      name: "detection",
      title: "Region detection",
      description: "Locate the exposed lower eyelid or nail bed.",
    },
    {
      name: "roi",
      title: "Region review",
      description: "Your confirmation determines the exact pixels analyzed.",
    },
    {
      name: "normalization",
      title: "Optical preparation",
      description:
        "Apply glare correction, white balance, and illumination normalization.",
    },
    {
      name: "features",
      title: "Color features",
      description: "Extract the 27 ordered features used by the trained model.",
    },
    {
      name: "inference",
      title: "Model inference",
      description:
        "Run the verified dual-head model and apply temperature calibration.",
    },
    {
      name: "explanation",
      title: "Attention map",
      description:
        "Compute Grad-CAM using the same image, features, and modality.",
    },
    {
      name: "saving",
      title: "Save result",
      description: "Store the result and its preparation images locally.",
    },
  ];
export function LivePipeline({ job }: { job: Job }) {
  const active = STAGES.find((s) => s.name === job.stage);
  return (
    <section className="live-pipeline" aria-label="Processing stages">
      <div className="section-line">
        <h3>Processing record</h3>
        <span>{job.elapsed_seconds.toFixed(1)} s elapsed</span>
      </div>
      <ol>
        {STAGES.map((stage, index) => {
          const state = job.stages.find((s) => s.name === stage.name);
          const done =
            state?.status === "completed" || state?.status === "complete";
          const running =
            job.stage === stage.name &&
            !["completed", "failed", "cancelled", "interrupted"].includes(
              job.status,
            );
          return (
            <li
              key={stage.name}
              className={cn(done && "stage-done", running && "stage-active")}
            >
              <span className="stage-indicator">
                {done ? (
                  <Check size={14} />
                ) : running && job.status !== "awaiting_roi" ? (
                  <LoaderCircle className="spin" size={14} />
                ) : (
                  <Circle size={10} />
                )}
              </span>
              <div>
                <strong>{stage.title}</strong>
                <p>{stage.description}</p>
              </div>
              <small>
                {done
                  ? "Complete"
                  : running
                    ? job.status === "awaiting_roi"
                      ? "Your review"
                      : "In progress"
                    : `${String(index + 1).padStart(2, "0")}`}
              </small>
            </li>
          );
        })}
      </ol>
      <p className="pipeline-status" role="status" aria-live="polite">
        {job.status === "awaiting_roi"
          ? "Waiting for you to confirm the region."
          : job.status === "completed"
            ? "All available processing stages are complete."
            : job.status === "failed"
              ? job.error?.message
              : job.status === "cancelled"
                ? "Screening cancelled."
                : job.status === "interrupted"
                  ? "Processing stopped when the service restarted."
                  : `${active?.title || "Queued"} · stages update from the local service.`}
      </p>
    </section>
  );
}
