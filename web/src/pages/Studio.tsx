import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Upload,
  Eye,
  Hand,
  LoaderCircle,
  RotateCcw,
  Focus,
  ShieldCheck,
  AlertTriangle,
  X,
} from "lucide-react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useSession } from "../context";
import {
  centeredBox,
  validAge,
  validateFile,
  validateDimensions,
} from "../lib/domain";
import { api, errorMessage } from "../services/api";
import type { Box, CaseDetails, Modality, Photo } from "../types";
import { ROIEditor } from "../components/ROIEditor";
import RefineFrame, { type RefineFrameStatus } from "../components/RefineFrame";
import { AnalysisProgress } from "../components/AnalysisProgress";
import "../screening.css";
import "../capture.css";
import { CameraCapture } from "../components/CameraCapture";
import { Button } from "../components/ui/button";
import { Badge } from "../components/ui/badge";
import { Input } from "../components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "../components/ui/alert";
import {
  Field,
  FieldGroup,
  FieldLabel,
  FieldDescription,
} from "../components/ui/field";
import {
  NativeSelect,
  NativeSelectOption,
} from "../components/ui/native-select";
import { ToggleGroup, ToggleGroupItem } from "../components/ui/toggle-group";
import { cn } from "../lib/utils";

function validReviewBox(
  box: Box | null,
  image: { width: number; height: number } | null | undefined,
) {
  return Boolean(
    box &&
    image &&
    Object.values(box).every(Number.isFinite) &&
    box.x0 >= 0 &&
    box.y0 >= 0 &&
    box.x1 <= image.width &&
    box.y1 <= image.height &&
    box.x1 - box.x0 >= 50 &&
    box.y1 - box.y0 >= 50,
  );
}

export function Studio() {
  const {
    system,
    serviceError,
    refreshSystem,
    job,
    scan,
    busy,
    error,
    restoring,
    upload,
    confirm,
    cancel,
    reset,
    retry,
    openJob,
  } = useSession();
  const { id } = useParams();
  const navigate = useNavigate();
  const [modality, setModality] = useState<Modality>("conjunctiva");
  const [dual, setDual] = useState(false);
  const dualAvailable = system?.capabilities?.dual_scan === true;
  const [details, setDetails] = useState<CaseDetails>({
    reference: "",
    age: "",
    sex: "",
    phone: "",
    notes: "",
  });
  const [file, setFile] = useState<File | null>(null),
    [photo, setPhoto] = useState<Photo | null>(null),
    [localError, setLocalError] = useState(""),
    [loading, setLoading] = useState(false),
    [dragging, setDragging] = useState(false),
    [roi, setRoi] = useState<Box | null>(null),
    [manual, setManual] = useState(false);
  const [secondaryFile, setSecondaryFile] = useState<File | null>(null),
    [secondaryPhoto, setSecondaryPhoto] = useState<Photo | null>(null),
    [secondaryLoading, setSecondaryLoading] = useState(false),
    [secondaryDragging, setSecondaryDragging] = useState(false),
    [secondaryRoi, setSecondaryRoi] = useState<Box | null>(null),
    [primaryReviewed, setPrimaryReviewed] = useState(false),
    [secondaryReviewed, setSecondaryReviewed] = useState(false),
    [primaryRefineStatus, setPrimaryRefineStatus] =
      useState<RefineFrameStatus>("complete"),
    [secondaryRefineStatus, setSecondaryRefineStatus] =
      useState<RefineFrameStatus>("complete");
  const input = useRef<HTMLInputElement>(null),
    secondaryInput = useRef<HTMLInputElement>(null),
    sequence = useRef(0),
    secondarySequence = useRef(0),
    heading = useRef<HTMLHeadingElement>(null),
    focusedReview = useRef<string | null>(null);
  useEffect(
    () => () => {
      sequence.current++;
      secondarySequence.current++;
    },
    [],
  );
  useEffect(() => {
    const url = photo?.url;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [photo?.url]);
  useEffect(() => {
    const url = secondaryPhoto?.url;
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [secondaryPhoto?.url]);
  useEffect(() => {
    if (job?.status === "awaiting_roi" && job.image) {
      setRoi(
        job.detection?.roi || centeredBox(job.image.width, job.image.height),
      );
      setManual(Boolean(job.detection?.success));
      setPrimaryReviewed(false);
      setSecondaryReviewed(false);
      setSecondaryRoi(
        job.secondary?.image
          ? job.secondary.detection?.roi ||
              centeredBox(job.secondary.image.width, job.secondary.image.height)
          : null,
      );
    }
  }, [job?.id, job?.status]);
  useEffect(() => {
    if (job?.status !== "awaiting_roi") {
      focusedReview.current = null;
    } else if (roi && heading.current && focusedReview.current !== job.id) {
      focusedReview.current = job.id;
      heading.current.focus();
    }
  }, [roi, job?.id, job?.status]);
  const selectFile = async (
    selected: File,
    secondary = false,
    reservedTicket?: number,
  ) => {
    const requestSequence = secondary ? secondarySequence : sequence;
    const ticket = reservedTicket ?? ++requestSequence.current;
    if (ticket !== requestSequence.current) return;
    const updateFile = secondary ? setSecondaryFile : setFile;
    const updatePhoto = secondary ? setSecondaryPhoto : setPhoto;
    const updateLoading = secondary ? setSecondaryLoading : setLoading;
    setLocalError("");
    const validation = validateFile(selected.type, selected.size);
    if (validation) {
      updateFile(null);
      updatePhoto(null);
      updateLoading(false);
      setLocalError(validation);
      return;
    }
    updateLoading(true);
    const url = URL.createObjectURL(selected);
    try {
      const image = new Image();
      await new Promise<void>((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = reject;
        image.src = url;
      });
      if (ticket !== requestSequence.current) {
        URL.revokeObjectURL(url);
        return;
      }
      const dimensions = validateDimensions(
        image.naturalWidth,
        image.naturalHeight,
      );
      if (dimensions) throw new Error(dimensions);
      updateFile(selected);
      updatePhoto({
        url,
        name: selected.name,
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
      if (!secondary) {
        setPrimaryRefineStatus("generating");
        setTimeout(() => setPrimaryRefineStatus("refining"), 400);
        setTimeout(() => setPrimaryRefineStatus("complete"), 800);
      } else {
        setSecondaryRefineStatus("generating");
        setTimeout(() => setSecondaryRefineStatus("refining"), 400);
        setTimeout(() => setSecondaryRefineStatus("complete"), 800);
      }
    } catch (e) {
      URL.revokeObjectURL(url);
      if (ticket === requestSequence.current) {
        updateFile(null);
        updatePhoto(null);
        setLocalError(
          e instanceof Error
            ? e.message
            : "This image could not be opened. Choose a different image.",
        );
      }
    } finally {
      if (ticket === requestSequence.current) updateLoading(false);
    }
  };
  const useReference = async (secondary = false) => {
    const requestSequence = secondary ? secondarySequence : sequence;
    const ticket = ++requestSequence.current;
    const updateLoading = secondary ? setSecondaryLoading : setLoading;
    updateLoading(true);
    setLocalError("");
    try {
      const selected = await api.sample(secondary ? "fingernail" : modality);
      await selectFile(selected, secondary, ticket);
    } catch (e) {
      if (ticket === requestSequence.current) {
        setLocalError(errorMessage(e));
        updateLoading(false);
      }
    }
  };
  const changeModality = (value: string) => {
    if (
      value === "conjunctiva" ||
      value === "fingernail" ||
      (value === "dual" && dualAvailable)
    ) {
      sequence.current++;
      secondarySequence.current++;
      setDual(value === "dual");
      setModality(value === "fingernail" ? "fingernail" : "conjunctiva");
      setFile(null);
      setPhoto(null);
      setSecondaryFile(null);
      setSecondaryPhoto(null);
      setLoading(false);
      setSecondaryLoading(false);
      setDragging(false);
      setSecondaryDragging(false);
      setLocalError("");
    }
  };
  useEffect(() => {
    if (id && job?.id !== id) void openJob(id);
    // An explicit route ID owns this load; subsequent job updates are polled by the session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, openJob]);
  useEffect(() => {
    if (!id && job && !restoring)
      navigate("/studio/jobs/" + job.id, { replace: true });
    if (
      job &&
      job.id === id &&
      job.status === "completed" &&
      job.scan_id &&
      scan?.id === job.scan_id
    )
      navigate("/history/" + job.scan_id, { replace: true });
  }, [id, job?.id, job?.status, job?.scan_id, scan?.id, restoring, navigate]);
  const fresh = () => {
    reset();
    sequence.current++;
    secondarySequence.current++;
    setFile(null);
    setPhoto(null);
    setSecondaryFile(null);
    setSecondaryPhoto(null);
    setLoading(false);
    setSecondaryLoading(false);
    setDragging(false);
    setSecondaryDragging(false);
    setRoi(null);
    setSecondaryRoi(null);
    setPrimaryReviewed(false);
    setSecondaryReviewed(false);
    setManual(false);
    setLocalError("");
    if (job?.patient)
      setDetails({
        reference: job.patient.reference,
        age: job.patient.age === null ? "" : String(job.patient.age),
        sex: job.patient.sex,
        phone: job.patient.phone || "",
        notes: job.patient.notes || "",
      });
    navigate("/studio", { replace: true });
  };
  const mode = job?.modality || modality;
  const dualReview = Boolean(job?.dual_scan || job?.secondary);
  const secondaryReviewReady = Boolean(
    job?.secondary?.image &&
    !job.secondary.error &&
    job.secondary.modality !== mode &&
    (job.secondary.modality === "fingernail" ||
      job.secondary.modality === "conjunctiva"),
  );
  const canConfirm =
    validReviewBox(roi, job?.image) &&
    (dualReview
      ? primaryReviewed &&
        secondaryReviewed &&
        secondaryReviewReady &&
        validReviewBox(secondaryRoi, job?.secondary?.image)
      : manual);
  const detailsValid = Boolean(
    details.reference.trim() &&
    details.age !== "" &&
    validAge(details.age) &&
    details.sex,
  );
  const needsRestart = Boolean(
    system?.ready && system.patient_intake_version !== 1,
  );
  const canUpload = Boolean(
    file &&
    (!dual || (dualAvailable && secondaryFile)) &&
    detailsValid &&
    !busy &&
    !loading &&
    !secondaryLoading &&
    system?.ready &&
    !needsRestart,
  );
  const routeLoading = restoring || Boolean(id && job?.id !== id && !error);
  const review = job?.status === "awaiting_roi";
  const current = !job ? 0 : review ? 1 : 2;
  return (
    <div className="workspace-page wrap screening-workspace">
      <div className="workspace-heading">
        <div>
          <span className="eyebrow">IMAGE-BASED SCREENING</span>
          <h1>Screening studio</h1>
          <p>Upload a photo, add patient details, and review the screening.</p>
        </div>
        <Badge variant={system?.ready ? "secondary" : "outline"}>
          {system?.ready ? (
            <ShieldCheck data-icon="inline-start" />
          ) : (
            <LoaderCircle data-icon="inline-start" />
          )}
          {system?.ready ? "Local model ready" : "Service not ready"}
        </Badge>
      </div>
      <ol className="screening-steps" aria-label="Screening steps">
        {["Image & details", "Review region", "Analysis", "Results"].map(
          (label, index) => (
            <li
              key={label}
              className={cn(
                current === index && "current",
                current > index && "complete",
              )}
              aria-current={current === index ? "step" : undefined}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              {label}
            </li>
          ),
        )}
      </ol>
      {needsRestart && (
        <Alert>
          <RotateCcw />
          <AlertTitle>Updated backend required</AlertTitle>
          <AlertDescription>
            This service does not support the current patient form yet. Ask its
            operator to enable the Integration compatibility endpoints and
            restart the updated backend, then check the connection again.
            <Button
              variant="outline"
              size="sm"
              onClick={() => void refreshSystem()}
            >
              Check connection
            </Button>
          </AlertDescription>
        </Alert>
      )}
      {(serviceError || (system && !system.ready)) && (
        <Alert variant="destructive">
          <AlertTriangle />
          <AlertTitle>The model is not ready</AlertTitle>
          <AlertDescription>
            {serviceError ||
              system?.warnings.join(" ") ||
              "Model verification has not completed."}
            <Button
              variant="outline"
              size="sm"
              onClick={() => void refreshSystem()}
            >
              Check connection
            </Button>
          </AlertDescription>
        </Alert>
      )}
      {(error || localError) && (
        <Alert variant="destructive">
          <AlertTriangle />
          <AlertTitle>Unable to continue</AlertTitle>
          <AlertDescription>
            {localError || error}
            {error && (
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  job ? retry() : id ? void openJob(id) : retry()
                }
              >
                Retry status
              </Button>
            )}
            {id && !job && (
              <Button variant="outline" onClick={fresh}>
                New screening
              </Button>
            )}
          </AlertDescription>
        </Alert>
      )}
      {routeLoading ? (
        <div className="loading-state" role="status">
          <LoaderCircle className="spin" />
          Restoring the screening…
        </div>
      ) : id && job?.id !== id ? null : !job ? (
        <form
          className="intake-grid"
          onSubmit={(event) => {
            event.preventDefault();
            if (canUpload && file)
              void upload(
                file,
                modality,
                details,
                dual ? secondaryFile || undefined : undefined,
              );
          }}
        >
          <section className="intake-card image-intake">
            <div className="intake-card-title">
              <span className="eyebrow">01 / SCREENING IMAGE</span>
              <h2>{dual ? "Choose both images" : "Choose your image"}</h2>
              <p>Select the tissue you want to screen.</p>
            </div>
            <FieldGroup>
              <Field>
                <FieldLabel id="modality-label">Screening modality</FieldLabel>
                <ToggleGroup
                  type="single"
                  variant="outline"
                  size="lg"
                  value={dual ? "dual" : modality}
                  onValueChange={changeModality}
                  aria-labelledby="modality-label"
                  className="modality-toggle"
                  disabled={busy || loading || secondaryLoading}
                >
                  <ToggleGroupItem value="conjunctiva">
                    <Eye data-icon="inline-start" />
                    Lower eyelid
                  </ToggleGroupItem>
                  <ToggleGroupItem value="fingernail">
                    <Hand data-icon="inline-start" />
                    Fingernail
                  </ToggleGroupItem>
                  {dualAvailable && (
                    <ToggleGroupItem value="dual">
                      <Focus data-icon="inline-start" />
                      Eye + nail
                    </ToggleGroupItem>
                  )}
                </ToggleGroup>
              </Field>
            </FieldGroup>
            {dual && (
              <div className="dual-capture-intro">
                <Badge variant="secondary">Two images · one screening</Badge>
                <p>
                  Use an eye and a nail photo from the same person. You will
                  review each region separately.
                </p>
                <h3>
                  <Eye size={19} /> 1. Lower eyelid photo
                </h3>
              </div>
            )}
            <div className="capture-tip">
              <Focus size={19} />
              <p>
                {mode === "conjunctiva"
                  ? "Gently lower the eyelid to expose the pink inner surface. Use even light, sharp focus, and avoid reflections."
                  : "Photograph a clean, unpolished nail in even light. Keep the complete nail bed in focus."}
              </p>
            </div>
            <input
              className="sr-only"
              ref={input}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              aria-label="Choose a photo"
              onChange={(e) => {
                if (e.target.files?.[0]) void selectFile(e.target.files[0]);
                e.target.value = "";
              }}
              disabled={busy || loading || secondaryLoading}
            />
            <div
              id="primaryDropzone"
              className={cn(
                "live-upload-zone",
                dragging && "dragging",
                photo && "has-photo",
              )}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                if (
                  !busy &&
                  !loading &&
                  !secondaryLoading &&
                  e.dataTransfer.files[0]
                )
                  void selectFile(e.dataTransfer.files[0]);
              }}
            >
              {photo ? (
                <>
                  <RefineFrame
                    status={busy ? "generating" : primaryRefineStatus}
                    aspectRatio={`${photo.width} / ${photo.height}`}
                    width={photo.width > photo.height ? 360 : 280}
                    radius={12}
                    stageDuration={380}
                    sweep={true}
                    showStatus={true}
                    hideAfter={2500}
                    labels={{
                      queued: "Awaiting preprocessing",
                      generating: "Analyzing tissue ROI…",
                      refining: "Calibrating resolution…",
                      complete: "Oriented upload ready",
                      error: "Upload failed",
                    }}
                  >
                    <img src={photo.url} alt="Selected screening photo" />
                  </RefineFrame>
                  <div>
                    <strong>{photo.name}</strong>
                    <span>
                      {photo.width} × {photo.height} px
                    </span>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => input.current?.click()}
                      disabled={busy || loading || secondaryLoading}
                    >
                      Change photo
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="upload-ring">
                    {loading ? <LoaderCircle className="spin" /> : <Upload />}
                  </div>
                  <h3>{loading ? "Opening image…" : "Drop your photo here"}</h3>
                  <p>
                    JPEG, PNG, or WebP · up to 10 MB
                    <br />
                    At least 50 × 50 pixels
                  </p>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => input.current?.click()}
                    disabled={busy || loading || secondaryLoading}
                  >
                    Choose a photo <ArrowRight data-icon="inline-end" />
                  </Button>
                </>
              )}
            </div>
            <div className="image-source-actions">
              <CameraCapture
                onCapture={(f) => void selectFile(f)}
                modality={modality}
                disabled={busy || loading || secondaryLoading}
              />
              <Button
                type="button"
                variant="ghost"
                disabled={busy || loading || secondaryLoading}
                onClick={() => void useReference()}
              >
                Use reference photo
              </Button>
            </div>
            <p className="source-caption">
              The reference photo is a supplied test image. It runs through the
              same live model as your upload.
            </p>
            {dual && (
              <section
                className="secondary-capture"
                aria-labelledby="secondary-image-title"
              >
                <h3 id="secondary-image-title">
                  <Hand size={19} /> 2. Fingernail photo
                </h3>
                <p className="secondary-capture-tip">
                  Use an unpolished nail in even light. Keep the whole nail bed
                  sharp and avoid reflections.
                </p>
                <input
                  className="sr-only"
                  ref={secondaryInput}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  aria-label="Choose a nail photo"
                  onChange={(event) => {
                    if (event.target.files?.[0])
                      void selectFile(event.target.files[0], true);
                    event.target.value = "";
                  }}
                  disabled={busy || loading || secondaryLoading}
                />
                <div
                  id="secondaryDropzone"
                  className={cn(
                    "live-upload-zone",
                    secondaryDragging && "dragging",
                    secondaryPhoto && "has-photo",
                  )}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setSecondaryDragging(true);
                  }}
                  onDragLeave={() => setSecondaryDragging(false)}
                  onDrop={(event) => {
                    event.preventDefault();
                    setSecondaryDragging(false);
                    if (
                      !busy &&
                      !loading &&
                      !secondaryLoading &&
                      event.dataTransfer.files[0]
                    )
                      void selectFile(event.dataTransfer.files[0], true);
                  }}
                >
                  {secondaryPhoto ? (
                    <>
                      <RefineFrame
                        status={busy ? "generating" : secondaryRefineStatus}
                        aspectRatio={`${secondaryPhoto.width} / ${secondaryPhoto.height}`}
                        width={secondaryPhoto.width > secondaryPhoto.height ? 360 : 280}
                        radius={12}
                        stageDuration={380}
                        sweep={true}
                        showStatus={true}
                        hideAfter={2500}
                        labels={{
                          queued: "Awaiting preprocessing",
                          generating: "Analyzing nail bed ROI…",
                          refining: "Calibrating resolution…",
                          complete: "Nail upload ready",
                          error: "Upload failed",
                        }}
                      >
                        <img
                          src={secondaryPhoto.url}
                          alt="Selected fingernail screening photo"
                        />
                      </RefineFrame>
                      <div>
                        <strong>{secondaryPhoto.name}</strong>
                        <span>
                          {secondaryPhoto.width} × {secondaryPhoto.height} px
                        </span>
                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => secondaryInput.current?.click()}
                          disabled={busy || loading || secondaryLoading}
                        >
                          Change nail photo
                        </Button>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="upload-ring">
                        {secondaryLoading ? (
                          <LoaderCircle className="spin" />
                        ) : (
                          <Upload />
                        )}
                      </div>
                      <h3>
                        {secondaryLoading
                          ? "Opening nail image…"
                          : "Drop your nail photo here"}
                      </h3>
                      <p>
                        JPEG, PNG, or WebP · up to 10 MB
                        <br />
                        At least 50 × 50 pixels
                      </p>
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => secondaryInput.current?.click()}
                        disabled={busy || loading || secondaryLoading}
                      >
                        Choose a nail photo{" "}
                        <ArrowRight data-icon="inline-end" />
                      </Button>
                    </>
                  )}
                </div>
                <div className="image-source-actions">
                  <CameraCapture
                    modality="fingernail"
                    triggerLabel="Use camera for nail"
                    onCapture={(captured) => void selectFile(captured, true)}
                    disabled={busy || loading || secondaryLoading}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => void useReference(true)}
                    disabled={busy || loading || secondaryLoading}
                  >
                    Use nail reference photo
                  </Button>
                </div>
                {!dualAvailable && (
                  <p className="field-error" role="alert">
                    Dual screening is no longer available. Choose a single
                    modality to continue.
                  </p>
                )}
              </section>
            )}
          </section>
          <section className="intake-card patient-intake">
            <div className="intake-card-title">
              <span className="eyebrow">02 / PATIENT DETAILS</span>
              <h2>About this screening</h2>
              <p>Required fields are marked with an asterisk.</p>
            </div>
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="reference">
                  Patient name or case ID *
                </FieldLabel>
                <Input
                  id="reference"
                  required
                  maxLength={120}
                  autoComplete="off"
                  value={details.reference}
                  onChange={(e) =>
                    setDetails((d) => ({ ...d, reference: e.target.value }))
                  }
                  placeholder="e.g. CASE-001"
                />
                <FieldDescription>
                  A case ID works if you prefer not to record a name.
                </FieldDescription>
              </Field>
              <div className="patient-field-pair">
                <Field data-invalid={!validAge(details.age)}>
                  <FieldLabel htmlFor="age">Age in years *</FieldLabel>
                  <Input
                    id="age"
                    type="number"
                    required
                    min={0}
                    max={120}
                    step={1}
                    inputMode="numeric"
                    value={details.age}
                    aria-invalid={!validAge(details.age)}
                    onChange={(e) =>
                      setDetails((d) => ({ ...d, age: e.target.value }))
                    }
                    placeholder="Years"
                  />
                  {!validAge(details.age) && (
                    <FieldDescription>
                      Use a whole number from 0 to 120.
                    </FieldDescription>
                  )}
                </Field>
                <Field>
                  <FieldLabel htmlFor="sex">Biological sex *</FieldLabel>
                  <NativeSelect
                    id="sex"
                    required
                    value={details.sex}
                    onChange={(e) =>
                      setDetails((d) => ({ ...d, sex: e.target.value }))
                    }
                  >
                    <NativeSelectOption value="" disabled>
                      Select
                    </NativeSelectOption>
                    <NativeSelectOption value="female">
                      Female
                    </NativeSelectOption>
                    <NativeSelectOption value="male">Male</NativeSelectOption>
                    <NativeSelectOption value="not-stated">
                      Not stated
                    </NativeSelectOption>
                  </NativeSelect>
                </Field>
              </div>
              <Field>
                <FieldLabel htmlFor="phone">
                  Phone <span className="optional-label">Optional</span>
                </FieldLabel>
                <Input
                  id="phone"
                  type="tel"
                  maxLength={32}
                  value={details.phone}
                  onChange={(e) =>
                    setDetails((d) => ({ ...d, phone: e.target.value }))
                  }
                  placeholder="For your follow-up record"
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="notes">
                  Clinical / field notes{" "}
                  <span className="optional-label">Optional</span>
                </FieldLabel>
                <textarea
                  className="patient-notes"
                  id="notes"
                  maxLength={2000}
                  rows={4}
                  value={details.notes}
                  onChange={(e) =>
                    setDetails((d) => ({ ...d, notes: e.target.value }))
                  }
                  placeholder="Symptoms, relevant history, or capture conditions"
                />
              </Field>
            </FieldGroup>
            <p className="patient-context-note">
              Patient details accompany this record. The model analyzes the
              image, its colour features, and the selected modality.
            </p>
            <Button
              type="submit"
              size="lg"
              className="intake-submit cursor-target" data-cursor-label="Proceed to Region Review"
              disabled={!canUpload}
            >
              {busy ? (
                <LoaderCircle className="spin" data-icon="inline-start" />
              ) : (
                <Focus data-icon="inline-start" />
              )}
              {busy ? "Uploading…" : dual ? "Review images" : "Review image"}
              <ArrowRight data-icon="inline-end" />
            </Button>
            <p className="intake-footer">
              <ShieldCheck size={15} />
              Saved on this computer. Review the detected region before
              analysis.
            </p>
          </section>
        </form>
      ) : review && job.image && roi ? (
        <div
          className={cn(
            "screening-layout is-review",
            dualReview && "is-dual-review",
          )}
        >
          <section className="screening-panel">
            <div className="panel-intro">
              <span className="eyebrow">REVIEW THE MODEL INPUT</span>
              <h2 ref={heading} tabIndex={-1}>
                {dualReview
                  ? "Confirm both screening regions"
                  : "Confirm the screening region"}
              </h2>
              <p>
                {dualReview
                  ? "Review the eye and nail regions independently. Confirm both selections before starting the combined screening."
                  : job.detection?.success
                    ? "The detector proposed the region below. Check it carefully before running the model."
                    : "The detector could not identify a reliable region. Retake the photo or deliberately select the correct tissue."}
              </p>
            </div>
            <section
              className="review-image-section"
              aria-label={
                dualReview ? "Primary image region" : "Screening image region"
              }
            >
              {dualReview && (
                <h3>
                  {mode === "conjunctiva"
                    ? "1. Eye / lower eyelid"
                    : "1. Fingernail / nail bed"}
                </h3>
              )}
              <ROIEditor
                width={job.image.width}
                height={job.image.height}
                box={roi}
                onChange={(value) => {
                  setRoi(value);
                  setManual(true);
                  setPrimaryReviewed(false);
                }}
                onReset={() => {
                  setRoi(
                    job.detection?.roi ||
                      centeredBox(job.image!.width, job.image!.height),
                  );
                  setManual(Boolean(job.detection?.success));
                  setPrimaryReviewed(false);
                }}
                modality={mode}
                photo={{ ...job.image, name: "Oriented upload" }}
                glare={false}
              />
              {dualReview && (
                <label className="manual-confirm dual-region-confirm">
                  <input
                    type="checkbox"
                    checked={primaryReviewed}
                    onChange={(event) =>
                      setPrimaryReviewed(event.target.checked)
                    }
                  />
                  {mode === "conjunctiva"
                    ? "I have reviewed the lower eyelid region in the first image."
                    : "I have reviewed the nail bed region in the first image."}
                </label>
              )}
            </section>
            {dualReview &&
              (secondaryReviewReady && job.secondary?.image && secondaryRoi ? (
                <section
                  className="review-image-section secondary-region"
                  aria-label="Secondary image region"
                >
                  <h3>
                    {job.secondary.modality === "fingernail"
                      ? "2. Fingernail / nail bed"
                      : "2. Eye / lower eyelid"}
                  </h3>
                  <p>
                    {job.secondary.modality === "fingernail"
                      ? "Keep the nail bed inside the region with minimal surrounding skin."
                      : "Keep the exposed inner lower eyelid inside the region."}
                  </p>
                  <Alert
                    variant={
                      job.secondary.detection?.success
                        ? "default"
                        : "destructive"
                    }
                  >
                    <Focus />
                    <AlertTitle>
                      {job.secondary.detection?.success
                        ? "Second region proposal"
                        : "Select the second region manually"}
                    </AlertTitle>
                    <AlertDescription>
                      {job.secondary.detection?.message ||
                        "No reliable region was proposed for the second image. Deliberately select the correct tissue."}
                      {!job.secondary.detection?.success &&
                        job.secondary.detection?.retake_guidance && (
                          <p>{job.secondary.detection.retake_guidance}</p>
                        )}
                    </AlertDescription>
                  </Alert>
                  <ROIEditor
                    width={job.secondary.image.width}
                    height={job.secondary.image.height}
                    box={secondaryRoi}
                    onChange={(value) => {
                      setSecondaryRoi(value);
                      setSecondaryReviewed(false);
                    }}
                    onReset={() => {
                      const image = job.secondary!.image!;
                      setSecondaryRoi(
                        job.secondary?.detection?.roi ||
                          centeredBox(image.width, image.height),
                      );
                      setSecondaryReviewed(false);
                    }}
                    modality={job.secondary.modality}
                    photo={{
                      ...job.secondary.image,
                      name: "Second oriented upload",
                    }}
                    glare={false}
                  />
                  <label className="manual-confirm dual-region-confirm">
                    <input
                      type="checkbox"
                      checked={secondaryReviewed}
                      onChange={(event) =>
                        setSecondaryReviewed(event.target.checked)
                      }
                    />
                    {job.secondary.modality === "fingernail"
                      ? "I have reviewed the nail bed region in the second image."
                      : "I have reviewed the lower eyelid region in the second image."}
                  </label>
                  {!validReviewBox(secondaryRoi, job.secondary.image) && (
                    <p className="field-error" role="alert">
                      The second region must stay within its image and be at
                      least 50 × 50 pixels.
                    </p>
                  )}
                </section>
              ) : (
                <Alert variant="destructive" className="secondary-region">
                  <AlertTriangle />
                  <AlertTitle>
                    The second image is not ready for review
                  </AlertTitle>
                  <AlertDescription>
                    {job.secondary?.error?.message ||
                      "The service did not provide a usable second image and region. Both images must be reviewed before this screening can continue."}
                    <Button
                      variant="outline"
                      disabled={busy}
                      onClick={() => void openJob(job.id)}
                    >
                      Reload both images
                    </Button>
                  </AlertDescription>
                </Alert>
              ))}
          </section>
          <aside className="screening-guide">
            <span className="eyebrow">
              {job.patient?.reference || "SCREENING RECORD"}
            </span>
            <h3>
              {dualReview
                ? "Eye + nail screening"
                : mode === "conjunctiva"
                  ? "Eye / conjunctiva"
                  : "Fingernail / nail bed"}
            </h3>
            <p>
              {dualReview
                ? "Confirm the exposed lower eyelid and the nail bed in their separate images. Both selections are required."
                : mode === "conjunctiva"
                  ? "The selected region should contain the inner lower eyelid."
                  : "The selected region should contain the nail bed, with minimal surrounding skin."}
            </p>
            <ul aria-label="Capture checklist">
              <li>
                {mode === "conjunctiva"
                  ? "Inner lower eyelid visible"
                  : "Nail bed clearly visible"}
              </li>
              <li>Sharp focus</li>
              <li>Even lighting</li>
            </ul>
            <Alert variant={job.detection?.success ? "default" : "destructive"}>
              <Focus />
              <AlertTitle>
                {job.detection?.success
                  ? "Automatic region proposal"
                  : "Manual review needed"}
              </AlertTitle>
              <AlertDescription>
                {job.detection?.message}
                {!job.detection?.success && job.detection?.retake_guidance && (
                  <p>{job.detection.retake_guidance}</p>
                )}
              </AlertDescription>
            </Alert>
            {!dualReview && !job.detection?.success && (
              <label className="manual-confirm">
                <input
                  type="checkbox"
                  checked={manual}
                  onChange={(e) => setManual(e.target.checked)}
                />{" "}
                I have selected the exposed lower eyelid or nail bed shown in
                this photo.
              </label>
            )}
            {!validReviewBox(roi, job.image) && (
              <p className="field-error" role="alert">
                The selected region must be at least 50 × 50 pixels. Choose a
                larger region or a closer photo.
              </p>
            )}
            {job.warnings.length > 0 && (
              <p className="quality-notes">{job.warnings.join(" ")}</p>
            )}
            <div className="review-actions">
              <Button
                variant="outline"
                disabled={busy}
                onClick={() => void cancel()}
              >
                <X data-icon="inline-start" />
                Cancel screening
              </Button>
              <Button
                disabled={!canConfirm || busy || !system?.ready || needsRestart}
                onClick={() => {
                  if (canConfirm && roi)
                    void confirm(
                      roi,
                      dualReview ? secondaryRoi || undefined : undefined,
                    );
                }}
              >
                {busy ? (
                  <LoaderCircle className="spin" data-icon="inline-start" />
                ) : (
                  <ArrowRight data-icon="inline-end" />
                )}
                {dualReview ? "Analyze both images" : "Analyze image"}
              </Button>
            </div>
            <p className="guide-boundary-note">
              {dualReview
                ? "Both confirmed regions are analyzed together. The eye and nail results stay linked to this screening record."
                : "Only the confirmed region is sent through normalization and prediction."}
            </p>
            <Link className="text-link" to="/anatomy">
              Explore the anatomy <ArrowRight size={16} />
            </Link>
          </aside>
        </div>
      ) : (
        <AnalysisProgress
          job={job}
          busy={busy}
          onCancel={() => void cancel()}
          onRestart={fresh}
          previewUrl={photo?.url}
          previewWidth={photo?.width}
          previewHeight={photo?.height}
        />
      )}
    </div>
  );
}
