import type { BiophysicalMetrics, MultimodalFusion } from "../integration";

const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const percentage = (value: unknown) =>
  finite(value) && value >= 0 && value <= 100
    ? `${value.toFixed(1)}%`
    : "Unavailable";

function screeningClass(value: unknown) {
  const normalized =
    typeof value === "string" ? value.toLowerCase().replaceAll("_", "-") : "";
  if (normalized === "anemic") return "Screen positive";
  if (normalized === "non-anemic") return "Screen negative";
  return "Class unavailable";
}

export function ColorimetryResults({
  metrics,
}: {
  metrics?: BiophysicalMetrics | null;
}) {
  const readings = [
    {
      label: "Erythema index",
      value: metrics?.erythema_index,
      description: "Image redness index",
    },
    { label: "L*", value: metrics?.l_star, description: "Lightness" },
    {
      label: "a*",
      value: metrics?.a_star,
      description: "Green–red coordinate",
    },
    {
      label: "b*",
      value: metrics?.b_star,
      description: "Blue–yellow coordinate",
    },
  ];
  const available =
    metrics?.has_real_pixels === true &&
    readings.every(({ value }) => finite(value));
  return (
    <section
      className="integration-result-section"
      aria-labelledby="colorimetryHeading"
    >
      <span className="eyebrow">IMAGE MEASUREMENTS</span>
      <h2 id="colorimetryHeading">Colorimetry</h2>
      {available ? (
        <>
          <p>
            Pixel-derived color measurements supplied by the service. These
            values do not measure blood flow or hemoglobin.
          </p>
          <dl className="colorimetry-readings">
            {readings.map(({ label, value, description }) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>
                  {finite(value) ? value.toFixed(1) : "Unavailable"}
                  <span>{description}</span>
                </dd>
              </div>
            ))}
          </dl>
          <div className="erythema-gauge">
            <label htmlFor="erythemaFillBar">
              Erythema index · display scale 0–100
            </label>
            <meter
              id="erythemaFillBar"
              min={0}
              max={100}
              value={metrics!.erythema_index!}
              aria-label="Erythema index display scale"
            />
            <span>
              Visual index scale only; no clinical cutoffs are implied.
            </span>
          </div>
          {typeof metrics?.source === "string" && metrics.source && (
            <p className="integration-detail">
              Measurement source: {metrics.source}
            </p>
          )}
        </>
      ) : (
        <p className="integration-unavailable">
          Color measurements unavailable. This record does not contain a
          complete set of measurements verified as coming from image pixels.
        </p>
      )}
    </section>
  );
}

export function FusionResults({
  fusion,
}: {
  fusion?: MultimodalFusion | null;
}) {
  if (!fusion) return null;
  return (
    <section
      className="integration-result-section"
      aria-labelledby="fusionHeading"
    >
      <span className="eyebrow">TWO IMAGE SCREENING</span>
      <h2 id="fusionHeading">Eye and nail comparison</h2>
      <p>
        Individual predictions and the weighted combined score are reported
        separately. Agreement describes the two model outputs; it is not
        diagnostic certainty.
      </p>
      {fusion.active === true &&
      !fusion.secondary_error &&
      [
        fusion.primary_prob,
        fusion.secondary_prob,
        fusion.joint_probability,
        fusion.concordance_pct,
      ].every((value) => finite(value) && value >= 0 && value <= 100) ? (
        <>
          <div className="fusion-readings">
            <article>
              <h3>{fusion.primary_modality || "Primary image"}</h3>
              <p className="integration-value">
                {percentage(fusion.primary_prob)}
              </p>
              <p>{screeningClass(fusion.primary_class)}</p>
              <span>Primary image · anemia probability</span>
            </article>
            <article>
              <h3>{fusion.secondary_modality || "Secondary image"}</h3>
              <p className="integration-value">
                {percentage(fusion.secondary_prob)}
              </p>
              <p>{screeningClass(fusion.secondary_class)}</p>
              <span>Secondary image · anemia probability</span>
            </article>
            <article className="fusion-combined">
              <h3>Weighted combined score</h3>
              <p className="integration-value">
                {percentage(fusion.joint_probability)}
              </p>
              <p>Calculated by the backend</p>
              <span>This does not replace either individual prediction.</span>
            </article>
          </div>
          <div className="fusion-agreement">
            <span>Agreement between image predictions</span>
            <strong>{percentage(fusion.concordance_pct)}</strong>
            {fusion.agreement_level && <span>{fusion.agreement_level}</span>}
          </div>
          {fusion.method && (
            <p className="integration-detail">
              Combination method: {fusion.method}
            </p>
          )}
        </>
      ) : (
        <p className="integration-unavailable">
          A combined result is unavailable for this record. The primary
          screening result remains available above.
        </p>
      )}
      {fusion.secondary_error && (
        <p className="integration-error" role="status">
          Secondary image: {fusion.secondary_error}
        </p>
      )}
    </section>
  );
}
