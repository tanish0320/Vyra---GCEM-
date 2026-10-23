import { useCallback, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowUpRight,
  BookOpen,
  ImageOff,
  Info,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";
import { useSession } from "../context";
import { useResource } from "../hooks/useResource";
import { api } from "../services/api";
import type { ResearchOverview } from "../integration";
import { Button } from "../components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import "../research.css";

/** Media is served by this backend, never by an arbitrary response URL. */
export function safeMediaUrl(value: unknown): string | null {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const url = new URL(value, window.location.origin);
    if (
      url.origin !== window.location.origin ||
      url.username ||
      url.password ||
      !/^https?:$/.test(url.protocol)
    )
      return null;
    if (!/^\/(static|media|api\/v1\/artifacts)\//.test(url.pathname))
      return null;
    return url.href;
  } catch {
    return null;
  }
}

export async function requestWithTimeout<T>(
  work: (signal: AbortSignal) => Promise<T>,
  signal: AbortSignal,
  milliseconds = 30_000,
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort();
  if (signal.aborted) controller.abort();
  else signal.addEventListener("abort", cancel, { once: true });
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, milliseconds);
  try {
    return await work(controller.signal);
  } catch (error) {
    if (timedOut)
      throw new Error(
        "The service took too long to respond. Please try again.",
      );
    throw error;
  } finally {
    window.clearTimeout(timer);
    signal.removeEventListener("abort", cancel);
  }
}

export function ResearchImage({ url, title }: { url: string; title: string }) {
  const source = safeMediaUrl(url);
  const [failed, setFailed] = useState(false);
  return source && !failed ? (
    <a
      className="research-plot-link"
      href={source}
      target="_blank"
      rel="noreferrer"
      aria-label={`Open ${title} at full size`}
    >
      <img
        src={source}
        alt={title}
        loading="lazy"
        onError={() => setFailed(true)}
      />
    </a>
  ) : (
    <div className="research-image-missing">
      <ImageOff aria-hidden="true" />
      <span>Image unavailable from this backend.</span>
    </div>
  );
}

function validOverview(value: ResearchOverview): ResearchOverview {
  if (
    !value ||
    typeof value.summary !== "string" ||
    !Array.isArray(value.datasets) ||
    !Array.isArray(value.plots) ||
    !value.datasets.every(
      (item) =>
        item &&
        typeof item.name === "string" &&
        typeof item.description === "string" &&
        (item.count === null ||
          (Number.isSafeInteger(item.count) && item.count >= 0)),
    ) ||
    !value.plots.every(
      (item) =>
        item &&
        typeof item.title === "string" &&
        typeof item.url === "string" &&
        typeof item.description === "string",
    )
  ) {
    throw new Error(
      "The backend returned incomplete research data. Please retry after checking the research adapter.",
    );
  }
  return value;
}

export function Research() {
  const { system, serviceError, refreshSystem } = useSession();
  const supported = system?.capabilities?.research === true;
  const fetchOverview = useCallback(
    (signal: AbortSignal) =>
      supported
        ? requestWithTimeout(api.research, signal).then(validOverview)
        : Promise.reject(
            new Error("Research data is not enabled by this backend."),
          ),
    [supported],
  );
  const { value, error, loading, reload } = useResource(fetchOverview);
  return (
    <div className="workspace-page wrap research-page">
      <div className="workspace-heading">
        <div>
          <span className="eyebrow">DATA & EXPERIMENTS</span>
          <h1>Research library</h1>
          <p>
            Inspect the datasets and research figures supplied by the backend.
          </p>
        </div>
        <Button asChild variant="outline">
          <Link to="/evidence">
            <BookOpen data-icon="inline-start" /> Model evidence
          </Link>
        </Button>
      </div>
      {!supported ? (
        <Alert>
          <Info />
          <AlertTitle>Research data is not available yet</AlertTitle>
          <AlertDescription>
            {serviceError ||
              "The backend has not enabled its research-data adapter. Dataset counts and plots will appear when actual assets are connected."}
            <Button variant="outline" onClick={() => void refreshSystem()}>
              Check backend again
            </Button>
            <a className="research-inline-link" href="/eda">
              Open the legacy research page <ArrowUpRight size={15} />
            </a>
          </AlertDescription>
        </Alert>
      ) : loading ? (
        <div className="loading-state" role="status">
          <LoaderCircle className="spin" /> Loading supplied research…
        </div>
      ) : error ? (
        <Alert variant="destructive">
          <AlertTitle>Research could not load</AlertTitle>
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
            <section className="research-intro">
              <BookOpen aria-hidden="true" />
              <p>
                {value.summary ||
                  "The backend has not supplied a research summary."}
              </p>
            </section>
            <section aria-labelledby="research-datasets">
              <div className="research-section-heading">
                <h2 id="research-datasets">Dataset provenance</h2>
                <span>
                  {value.datasets.length} supplied{" "}
                  {value.datasets.length === 1 ? "dataset" : "datasets"}
                </span>
              </div>
              {value.datasets.length ? (
                <div className="research-dataset-grid">
                  {value.datasets.map((dataset, index) => (
                    <article
                      className="research-card"
                      key={`${dataset.name}-${index}`}
                    >
                      <span className="eyebrow">
                        DATASET {String(index + 1).padStart(2, "0")}
                      </span>
                      <h3>{dataset.name}</h3>
                      <p>{dataset.description}</p>
                      <div className="research-card-foot">
                        {dataset.count === null
                          ? "Sample count not supplied"
                          : `${dataset.count.toLocaleString()} samples reported by the backend`}
                      </div>
                    </article>
                  ))}
                </div>
              ) : (
                <p className="research-empty">
                  No dataset metadata was supplied. No sample totals have been
                  assumed.
                </p>
              )}
            </section>
            <section aria-labelledby="research-plots">
              <div className="research-section-heading">
                <h2 id="research-plots">Research figures</h2>
                <span>Actual backend assets</span>
              </div>
              {value.plots.length ? (
                <div className="research-plot-grid">
                  {value.plots.map((plot, index) => (
                    <figure
                      className="research-card research-plot"
                      key={`${plot.url}-${index}`}
                    >
                      <ResearchImage
                        key={plot.url}
                        url={plot.url}
                        title={plot.title}
                      />
                      <figcaption>
                        <h3>{plot.title}</h3>
                        <p>{plot.description}</p>
                      </figcaption>
                    </figure>
                  ))}
                </div>
              ) : (
                <p className="research-empty">
                  No research figures are available from this backend. View the
                  supplied evaluation metrics on the Evidence page.
                </p>
              )}
            </section>
          </>
        )
      )}
    </div>
  );
}
