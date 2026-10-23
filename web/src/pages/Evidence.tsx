import { Download, Info, ShieldCheck, LoaderCircle } from "lucide-react";
import { useResource } from "../hooks/useResource";
import { api } from "../services/api";
import { percent } from "../lib/domain";
import { Button } from "../components/ui/button";
import { Alert, AlertTitle, AlertDescription } from "../components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
const fetchEvidence = (signal: AbortSignal) => api.evidence(signal);
export function Evidence() {
  const { value, error, loading, reload } = useResource(fetchEvidence);
  return (
    <div className="workspace-page evidence-live wrap">
      <div className="workspace-heading">
        <div>
          <h1>Model evidence</h1>
          <p>
            Measured performance, model identity, and the limits of this
            research prototype.
          </p>
        </div>
        <Button asChild variant="outline">
          <a href="/api/v1/evidence" download="vyra-model-evidence.json">
            <Download data-icon="inline-start" /> Evaluation JSON
          </a>
        </Button>
      </div>
      {loading ? (
        <div className="loading-state" role="status">
          <LoaderCircle className="spin" />
          Loading model evidence…
        </div>
      ) : error ? (
        <Alert variant="destructive">
          <AlertTitle>Evidence unavailable</AlertTitle>
          <AlertDescription>
            {error}
            <Button variant="outline" onClick={reload}>
              Try again
            </Button>
          </AlertDescription>
        </Alert>
      ) : (
        value && (
          <>
            <div className="evidence-live-metrics">
              <div>
                <strong>{percent(value.classification.accuracy)}</strong>
                <span>Classification accuracy</span>
                <small>
                  {value.test_dataset_size.toLocaleString()} test images
                </small>
              </div>
              <div>
                <strong>
                  {percent(value.classification.sensitivity_recall)}
                </strong>
                <span>Sensitivity / recall</span>
                <small>Share of anemic cases identified</small>
              </div>
              <div>
                <strong>{value.classification.roc_auc.toFixed(3)}</strong>
                <span>ROC AUC</span>
                <small>Discrimination across thresholds</small>
              </div>
              <div>
                <strong>
                  {value.hemoglobin_regression.mae_g_dl.toFixed(2)}{" "}
                  <em>g/dL</em>
                </strong>
                <span>Hemoglobin mean absolute error</span>
                <small>
                  {value.hemoglobin_regression.paired_samples} paired laboratory
                  values
                </small>
              </div>
            </div>
            <Alert>
              <Info />
              <AlertTitle>
                Prototype evaluation, not clinical validation
              </AlertTitle>
              <AlertDescription>
                {value.provenance} The supplied evaluation does not establish
                safe clinical deployment.
              </AlertDescription>
            </Alert>
            <div className="evidence-split">
              <section className="evidence-block">
                <div className="section-line">
                  <h2>Classification</h2>
                  <span>Held-out test set</span>
                </div>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Metric</TableHead>
                      <TableHead>Value</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[
                      ["Accuracy", percent(value.classification.accuracy)],
                      [
                        "Sensitivity / recall",
                        percent(value.classification.sensitivity_recall),
                      ],
                      [
                        "Specificity",
                        percent(value.classification.specificity),
                      ],
                      ["Precision", percent(value.classification.precision)],
                      ["F1 score", value.classification.f1_score.toFixed(3)],
                      ["ROC AUC", value.classification.roc_auc.toFixed(3)],
                      [
                        "Brier score",
                        value.classification.brier_score.toFixed(3),
                      ],
                    ].map(([name, metric]) => (
                      <TableRow key={name}>
                        <TableCell>{name}</TableCell>
                        <TableCell>{metric}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <div className="confusion-matrix">
                  <h3>Confusion matrix</h3>
                  <div>
                    <span />
                    <span>Predicted negative</span>
                    <span>Predicted positive</span>
                    <span>Actual negative</span>
                    <strong>
                      {value.classification.confusion_matrix.TN}
                      <small>True negative</small>
                    </strong>
                    <strong>
                      {value.classification.confusion_matrix.FP}
                      <small>False positive</small>
                    </strong>
                    <span>Actual positive</span>
                    <strong>
                      {value.classification.confusion_matrix.FN}
                      <small>False negative</small>
                    </strong>
                    <strong>
                      {value.classification.confusion_matrix.TP}
                      <small>True positive</small>
                    </strong>
                  </div>
                </div>
              </section>
              <section className="evidence-block">
                <div className="section-line">
                  <h2>Hemoglobin regression</h2>
                  <span>Eye modality</span>
                </div>
                <p className="block-intro">
                  Image-based estimates compared with available laboratory
                  hemoglobin labels.
                </p>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Metric</TableHead>
                      <TableHead>Value</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[
                      [
                        "Paired samples",
                        value.hemoglobin_regression.paired_samples.toLocaleString(),
                      ],
                      [
                        "Mean absolute error",
                        `${value.hemoglobin_regression.mae_g_dl.toFixed(2)} g/dL`,
                      ],
                      [
                        "Root mean square error",
                        `${value.hemoglobin_regression.rmse_g_dl.toFixed(2)} g/dL`,
                      ],
                      ["R²", value.hemoglobin_regression.r2_score.toFixed(3)],
                      [
                        "Mean bias",
                        `${value.hemoglobin_regression.bland_altman.mean_bias_g_dl.toFixed(2)} g/dL`,
                      ],
                      [
                        "95% agreement interval",
                        `${value.hemoglobin_regression.bland_altman.loa_lower_95_g_dl.toFixed(2)} to ${value.hemoglobin_regression.bland_altman.loa_upper_95_g_dl.toFixed(2)} g/dL`,
                      ],
                    ].map(([name, metric]) => (
                      <TableRow key={name}>
                        <TableCell>{name}</TableCell>
                        <TableCell>{metric}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                <Alert>
                  <AlertTitle>Nail hemoglobin is unavailable</AlertTitle>
                  <AlertDescription>
                    The supplied model does not have validated nail regression.
                    Nail screening returns classification probability only.
                  </AlertDescription>
                </Alert>
              </section>
            </div>
            <section className="model-identity">
              <div>
                <ShieldCheck size={26} />
                <h2>Model identity</h2>
                <p>
                  The local service verifies the checkpoint before inference.
                </p>
              </div>
              <dl>
                <div>
                  <dt>Model version</dt>
                  <dd>{value.model_version}</dd>
                </div>
                <div>
                  <dt>Checkpoint SHA-256</dt>
                  <dd>
                    <code>{value.checkpoint_sha256}</code>
                  </dd>
                </div>
                <div>
                  <dt>Normalization policy</dt>
                  <dd>{value.normalization_version}</dd>
                </div>
                <div>
                  <dt>Calibration temperature</dt>
                  <dd>{value.temperature.toFixed(6)}</dd>
                </div>
              </dl>
            </section>
            <section className="limitations-block">
              <h2>Limitations to present clearly</h2>
              <ol>
                {value.limitations.map((limitation, index) => (
                  <li key={index}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <p>{limitation}</p>
                  </li>
                ))}
              </ol>
            </section>
          </>
        )
      )}
    </div>
  );
}
