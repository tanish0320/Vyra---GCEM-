import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  Download,
  ArrowLeft,
  ArrowUpRight,
  Archive,
  FolderOpen,
  LoaderCircle,
  RefreshCw,
} from "lucide-react";
import { api, errorMessage } from "../services/api";
import { useResource } from "../hooks/useResource";
import { useSession } from "../context";
import { percent, displayDate } from "../lib/domain";
import type { Scan } from "../types";
import { ScanResult } from "../components/ScanResult";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Field, FieldGroup, FieldLabel } from "../components/ui/field";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import { Badge } from "../components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "../components/ui/toggle-group";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
  EmptyContent,
} from "../components/ui/empty";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "../components/ui/dialog";
const fetchHistory = async (signal: AbortSignal) => {
  const [history, analytics] = await Promise.all([
    api.scans(signal),
    api.analytics(signal),
  ]);
  return { ...history, analytics };
};
export function History() {
  const { id } = useParams();
  return id ? <HistoryDetail id={id} /> : <HistoryList />;
}
function HistoryDetail({ id }: { id: string }) {
  const fetchScan = useMemo(
    () => (signal: AbortSignal) => api.scan(id, signal),
    [id],
  );
  const { value, error, loading, reload } = useResource(fetchScan);
  return (
    <div className="workspace-page wrap">
      <Link className="text-link" to="/history">
        <ArrowLeft size={16} /> Screening history
      </Link>
      {loading ? (
        <div className="loading-state" role="status">
          <LoaderCircle className="spin" />
          Loading saved result…
        </div>
      ) : error ? (
        <Alert variant="destructive">
          <AlertTitle>Result unavailable</AlertTitle>
          <AlertDescription>
            {error}
            <Button variant="outline" onClick={reload}>
              Try again
            </Button>
          </AlertDescription>
        </Alert>
      ) : (
        value && <ScanResult scan={value} />
      )}
    </div>
  );
}
function HistoryList() {
  const { scan: activeScan, job: activeJob, reset } = useSession();
  const { value, error, loading, reload } = useResource(fetchHistory);
  const [query, setQuery] = useState(""),
    [modality, setModality] = useState("all"),
    [deleting, setDeleting] = useState<Scan | null>(null),
    [busy, setBusy] = useState(false),
    [deleteError, setDeleteError] = useState("");
  const navigate = useNavigate();
  const rows =
    value?.scans.filter(
      (scan) =>
        (modality === "all" || scan.modality === modality) &&
        `${scan.reference} ${scan.id}`
          .toLowerCase()
          .includes(query.toLowerCase()),
    ) || [];
  const remove = async () => {
    if (!deleting || busy) return;
    setBusy(true);
    setDeleteError("");
    try {
      await api.remove(deleting.id);
      if (
        activeScan?.id === deleting.id ||
        activeJob?.scan_id === deleting.id
      ) {
        reset();
      }
      setDeleting(null);
      reload();
    } catch (e) {
      setDeleteError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="workspace-page history-page wrap">
      <div className="workspace-heading">
        <div>
          <h1>Screening history</h1>
          <p>Saved results and processing artifacts on this computer.</p>
        </div>
        <div className="heading-actions">
          <Button
            variant="outline"
            onClick={reload}
            disabled={loading}
            aria-label="Refresh history"
          >
            <RefreshCw data-icon="inline-start" />
          </Button>
          <Button asChild variant="outline">
            <a href={api.csv} download>
              <Download data-icon="inline-start" /> Export CSV
            </a>
          </Button>
        </div>
      </div>
      {error && (
        <Alert variant="destructive">
          <AlertTitle>History unavailable</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {value && (
        <div className="history-insights" aria-label="Screening insights">
          <div>
            <span>Saved screenings</span>
            <strong>{value.analytics.total_scans}</strong>
            <small>
              {value.analytics.by_modality.conjunctiva} eye ·{" "}
              {value.analytics.by_modality.fingernail} nail
            </small>
          </div>
          <div>
            <span>Screen positive</span>
            <strong>{value.analytics.screen_positive}</strong>
            <small>At the model’s classification threshold</small>
          </div>
          <div>
            <span>Screen negative</span>
            <strong>{value.analytics.screen_negative}</strong>
            <small>A negative screen does not exclude anemia</small>
          </div>
          <div>
            <span>Eye Hb estimates</span>
            <strong>{value.analytics.eye_hb_count}</strong>
            <small>
              {value.analytics.mean_eye_hb === null
                ? "No estimates recorded"
                : `Mean ${value.analytics.mean_eye_hb.toFixed(2)} g/dL · image based`}
            </small>
          </div>
        </div>
      )}
      <div className="history-toolbar">
        <FieldGroup>
          <Field>
            <FieldLabel htmlFor="history-search" className="sr-only">
              Search case reference or ID
            </FieldLabel>
            <Input
              id="history-search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search reference or result ID"
            />
          </Field>
        </FieldGroup>
        <ToggleGroup
          type="single"
          variant="outline"
          value={modality}
          onValueChange={(v) => v && setModality(v)}
          aria-label="Filter history by modality"
        >
          <ToggleGroupItem value="all">All</ToggleGroupItem>
          <ToggleGroupItem value="conjunctiva">Eye</ToggleGroupItem>
          <ToggleGroupItem value="fingernail">Nail</ToggleGroupItem>
        </ToggleGroup>
      </div>
      {loading ? (
        <div className="loading-state" role="status">
          <LoaderCircle className="spin" />
          Loading history…
        </div>
      ) : rows.length ? (
        <div className="history-table">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Screening</TableHead>
                <TableHead>Modality</TableHead>
                <TableHead>Model output</TableHead>
                <TableHead>Estimated Hb</TableHead>
                <TableHead>
                  <span className="sr-only">Actions</span>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((scan) => (
                <TableRow key={scan.id}>
                  <TableCell>
                    <Link
                      className="history-reference"
                      to={`/history/${scan.id}`}
                    >
                      {scan.reference || `Screening ${scan.id.slice(0, 8)}`}
                    </Link>
                    <span className="table-subline">
                      {displayDate(scan.created_at)}
                    </span>
                  </TableCell>
                  <TableCell>
                    {scan.modality === "conjunctiva" ? "Eye" : "Nail"}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        scan.probability_anemia >= scan.threshold
                          ? "default"
                          : "secondary"
                      }
                    >
                      {percent(scan.probability_anemia)}
                    </Badge>
                    <span className="table-subline">
                      {scan.probability_anemia >= scan.threshold
                        ? "Screen positive"
                        : "Screen negative"}
                    </span>
                  </TableCell>
                  <TableCell>
                    {scan.modality === "conjunctiva" && scan.hb_g_dl !== null
                      ? `${scan.hb_g_dl.toFixed(2)} g/dL`
                      : "Unavailable"}
                  </TableCell>
                  <TableCell>
                    <div className="row-actions">
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={`Open ${scan.reference || scan.id}`}
                        onClick={() => navigate(`/history/${scan.id}`)}
                      >
                        <ArrowUpRight />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={`Archive ${scan.reference || scan.id}`}
                        onClick={() => {
                          setDeleting(scan);
                          setDeleteError("");
                        }}
                      >
                        <Archive />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      ) : (
        <Empty>
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <FolderOpen />
            </EmptyMedia>
            <EmptyTitle>
              {query || modality !== "all"
                ? "No matching screenings"
                : "No screenings saved yet"}
            </EmptyTitle>
            <EmptyDescription>
              {query || modality !== "all"
                ? "Try a different reference or modality filter."
                : "Run an image through the screening studio. Completed results will appear here."}
            </EmptyDescription>
          </EmptyHeader>
          <EmptyContent>
            <Button asChild>
              <Link to="/studio">Open screening studio</Link>
            </Button>
          </EmptyContent>
        </Empty>
      )}
      <p className="history-note">
        These counts describe this computer’s saved screenings. They are not a
        population study or clinical accuracy measure.
      </p>
      <Dialog
        open={!!deleting}
        onOpenChange={(open) => {
          if (!open && !busy) setDeleting(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Archive this screening?</DialogTitle>
            <DialogDescription>
              This removes {deleting?.reference || "the selected result"} from
              active history. The backend retains the archived record and its
              images according to its retention policy.
            </DialogDescription>
          </DialogHeader>
          {deleteError && (
            <Alert variant="destructive">
              <AlertDescription>{deleteError}</AlertDescription>
            </Alert>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => setDeleting(null)}
            >
              Keep screening
            </Button>
            <Button
              variant="destructive"
              disabled={busy}
              onClick={() => void remove()}
            >
              {busy ? (
                <LoaderCircle className="spin" data-icon="inline-start" />
              ) : (
                <Archive data-icon="inline-start" />
              )}
              Archive screening
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
