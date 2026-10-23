import { Link } from "react-router-dom";
import {
  ArrowRight,
  BarChart3,
  Download,
  Info,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";
import { api } from "../services/api";
import { useResource } from "../hooks/useResource";
import type { Analytics as AnalyticsData } from "../types";
import { Button } from "../components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { requestWithTimeout } from "./Research";
import "../research.css";

async function fetchAnalytics(signal: AbortSignal): Promise<AnalyticsData> {
  const value = await requestWithTimeout(api.analytics, signal);
  const counts = [
    value?.total_scans,
    value?.screen_positive,
    value?.screen_negative,
    value?.eye_hb_count,
    value?.by_modality?.conjunctiva,
    value?.by_modality?.fingernail,
  ];
  if (
    !value ||
    !counts.every((n) => Number.isSafeInteger(n) && n >= 0) ||
    (value.mean_probability !== null &&
      (!Number.isFinite(value.mean_probability) ||
        value.mean_probability < 0 ||
        value.mean_probability > 1)) ||
    (value.mean_eye_hb !== null &&
      (!Number.isFinite(value.mean_eye_hb) || value.mean_eye_hb <= 0))
  ) {
    throw new Error(
      "The backend returned incomplete analytics. Please check the analytics adapter and retry.",
    );
  }
  return value;
}

export function Analytics() {
  const { value, error, loading, reload } = useResource(fetchAnalytics);
  const total = value?.total_scans || 0;
  const share = (count: number) =>
    total ? `${((count / total) * 100).toFixed(1)}%` : "No saved scans";
  return (
    <div className="workspace-page wrap research-page">
      <div className="workspace-heading">
        <div>
          <span className="eyebrow">SAVED SCREENING ACTIVITY</span>
          <h1>Screening analytics</h1>
          <p>A summary of the records stored by this backend.</p>
        </div>
        <Button asChild variant="outline">
          <a href={api.csv} download>
            <Download data-icon="inline-start" /> Export records
          </a>
        </Button>
      </div>
      {loading ? (
        <div className="loading-state" role="status">
          <LoaderCircle className="spin" /> Loading screening activity…
        </div>
      ) : error ? (
        <Alert variant="destructive">
          <AlertTitle>Analytics unavailable</AlertTitle>
          <AlertDescription>
            {error}
            <Button variant="outline" onClick={reload}>
              <RotateCcw data-icon="inline-start" /> Try again
            </Button>
          </AlertDescription>
        </Alert>
      ) : (
        value && (
          <>
            <div className="research-metrics">
              <div>
                <span>Saved screenings</span>
                <strong>{total.toLocaleString()}</strong>
                <small>Actual records in this backend</small>
              </div>
              <div>
                <span>Screen positive</span>
                <strong>{value.screen_positive.toLocaleString()}</strong>
                <small>
                  {share(value.screen_positive)}
                  {total ? " of saved screenings" : ""}
                </small>
              </div>
              <div>
                <span>Screen negative</span>
                <strong>{value.screen_negative.toLocaleString()}</strong>
                <small>
                  {share(value.screen_negative)}
                  {total ? " of saved screenings" : ""}
                </small>
              </div>
              <div>
                <span>Mean model probability</span>
                <strong>
                  {total && value.mean_probability !== null
                    ? `${(value.mean_probability * 100).toFixed(1)}%`
                    : "—"}
                </strong>
                <small>
                  {total
                    ? "Across saved model outputs"
                    : "Available after a screening is saved"}
                </small>
              </div>
            </div>
            {!total && (
              <section className="research-empty research-empty-action">
                <BarChart3 size={30} />
                <h2>No saved screenings yet</h2>
                <p>
                  Complete a screening to populate this view. The dashboard uses
                  actual records.
                </p>
                <Button asChild>
                  <Link to="/studio">
                    Start screening <ArrowRight data-icon="inline-end" />
                  </Link>
                </Button>
              </section>
            )}
            <div className="research-two-column">
              <section
                className="research-card"
                aria-labelledby="analytics-modalities"
              >
                <span className="eyebrow">CAPTURE MODALITY</span>
                <h2 id="analytics-modalities">Screenings by tissue</h2>
                <div className="research-count-list">
                  {(
                    [
                      ["Eye conjunctiva", value.by_modality.conjunctiva],
                      ["Fingernail", value.by_modality.fingernail],
                    ] as const
                  ).map(([label, count]) => (
                    <div key={label}>
                      <div>
                        <span>{label}</span>
                        <strong>{count.toLocaleString()}</strong>
                      </div>
                      <meter
                        min={0}
                        max={Math.max(total, count, 1)}
                        value={count}
                        aria-label={`${label}: ${count} saved screenings`}
                      />
                    </div>
                  ))}
                </div>
              </section>
              <section className="research-card" aria-labelledby="analytics-hb">
                <span className="eyebrow">EYE MODEL OUTPUT</span>
                <h2 id="analytics-hb">Hemoglobin estimates</h2>
                <div className="research-secondary-reading">
                  <strong>
                    {value.eye_hb_count && value.mean_eye_hb !== null
                      ? value.mean_eye_hb.toFixed(2)
                      : "—"}
                  </strong>
                  <span>g/dL mean image estimate</span>
                </div>
                <p>
                  {value.eye_hb_count
                    ? `${value.eye_hb_count.toLocaleString()} eye screenings include a hemoglobin estimate.`
                    : "No saved eye hemoglobin estimates are available."}{" "}
                  Fingernail screenings do not produce a hemoglobin estimate.
                </p>
              </section>
            </div>
            <Alert>
              <Info />
              <AlertTitle>
                Activity is separate from clinical evidence
              </AlertTitle>
              <AlertDescription>
                These counts describe saved screenings, not population anemia
                prevalence or diagnostic accuracy. Image estimates are separate
                from laboratory measurements.{" "}
                <Link className="research-inline-link" to="/evidence">
                  View the supplied model evaluation <ArrowRight size={15} />
                </Link>
              </AlertDescription>
            </Alert>
            <div className="research-page-actions">
              <Button asChild variant="outline">
                <Link to="/history">Open screening history</Link>
              </Button>
              <Button variant="outline" onClick={reload}>
                <RotateCcw data-icon="inline-start" /> Refresh activity
              </Button>
            </div>
          </>
        )
      )}
    </div>
  );
}
