import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, Focus, LoaderCircle, Sun } from "lucide-react";
import type { Modality } from "../types";
import { useReducedMotion } from "../hooks/useReducedMotion";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import "../camera-capture.css";

interface FrameReading {
  luminance: number;
  detail: number;
}

/** Local image heuristics, not a tissue detector or a clinical quality score. */
function readFrame(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
): FrameReading {
  const pixels = context.getImageData(0, 0, width, height).data;
  const luma = new Float32Array(width * height);
  let total = 0;
  for (let index = 0; index < luma.length; index++) {
    const offset = index * 4;
    const value =
      pixels[offset] * 0.2126 +
      pixels[offset + 1] * 0.7152 +
      pixels[offset + 2] * 0.0722;
    luma[index] = value;
    total += value;
  }
  let sum = 0,
    squared = 0,
    count = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      const edge =
        luma[index - 1] +
        luma[index + 1] +
        luma[index - width] +
        luma[index + width] -
        4 * luma[index];
      sum += edge;
      squared += edge * edge;
      count++;
    }
  }
  return {
    luminance: total / luma.length,
    detail: Math.max(0, squared / count - (sum / count) ** 2),
  };
}

export function CameraCapture({
  onCapture,
  disabled,
  modality = "conjunctiva",
  triggerLabel = "Use camera",
}: {
  onCapture: (file: File) => void;
  disabled?: boolean;
  modality?: Modality;
  triggerLabel?: string;
}) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [reading, setReading] = useState<FrameReading | null>(null);
  const [aspectRatio, setAspectRatio] = useState("4 / 3");
  const reducedMotion = useReducedMotion();
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const telemetryTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const captureTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const generation = useRef(0);
  const capturePending = useRef(false);

  const stopCamera = useCallback(() => {
    generation.current++;
    if (telemetryTimer.current !== null) clearInterval(telemetryTimer.current);
    if (captureTimer.current !== null) clearTimeout(captureTimer.current);
    telemetryTimer.current = null;
    captureTimer.current = null;
    capturePending.current = false;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
  }, []);

  const changeOpen = (value: boolean) => {
    if (!value) stopCamera();
    else {
      // Mount a fresh video immediately, including after a permission error.
      setError("");
      setReady(false);
      setCapturing(false);
      setReading(null);
    }
    setOpen(value);
  };

  useEffect(() => {
    if (!open) return;
    stopCamera();
    const token = generation.current;
    setReady(false);
    setCapturing(false);
    setReading(null);
    setError("");
    const closeWhenHidden = () => {
      if (document.hidden) {
        stopCamera();
        setOpen(false);
      }
    };
    document.addEventListener("visibilitychange", closeWhenHidden);
    const fail = (message: string) => {
      if (token !== generation.current) return;
      stopCamera();
      setReady(false);
      setError(message);
    };
    const sample = () => {
      const source = video.current,
        target = canvas.current;
      if (
        token !== generation.current ||
        document.hidden ||
        capturePending.current ||
        !source ||
        !target ||
        source.readyState < 2 ||
        !source.videoWidth
      )
        return;
      // Sample the same central area outlined by the modality reticle.
      const cropWidth =
        source.videoWidth * (modality === "conjunctiva" ? 0.76 : 0.4);
      const cropHeight =
        source.videoHeight * (modality === "conjunctiva" ? 0.38 : 0.74);
      const sampleWidth = 160;
      const sampleHeight = Math.max(
        32,
        Math.round((sampleWidth * cropHeight) / cropWidth),
      );
      target.width = sampleWidth;
      target.height = sampleHeight;
      const context = target.getContext("2d", { willReadFrequently: true });
      if (!context) return;
      try {
        context.drawImage(
          source,
          (source.videoWidth - cropWidth) / 2,
          (source.videoHeight - cropHeight) / 2,
          cropWidth,
          cropHeight,
          0,
          0,
          sampleWidth,
          sampleHeight,
        );
        setReading(readFrame(context, sampleWidth, sampleHeight));
      } catch {
        // Camera frames can briefly disappear during device orientation changes.
        setReading(null);
      }
    };
    if (document.hidden) closeWhenHidden();
    else if (!navigator.mediaDevices?.getUserMedia) {
      fail(
        "Camera access is unavailable in this browser. Choose a photo from your device instead.",
      );
    } else {
      navigator.mediaDevices
        .getUserMedia({
          video: {
            facingMode: { ideal: "environment" },
            width: { ideal: 1280 },
          },
          audio: false,
        })
        .then(async (value) => {
          if (
            token !== generation.current ||
            document.hidden ||
            !video.current
          ) {
            value.getTracks().forEach((track) => track.stop());
            return;
          }
          stream.current = value;
          value
            .getVideoTracks()
            .forEach((track) =>
              track.addEventListener(
                "ended",
                () =>
                  fail(
                    "The camera disconnected. Close it and try again, or choose a photo.",
                  ),
                { once: true },
              ),
            );
          video.current.srcObject = value;
          try {
            await video.current.play();
            if (token !== generation.current) return;
            telemetryTimer.current = setInterval(sample, 280);
            sample();
          } catch {
            fail("Camera playback could not start. Choose a photo instead.");
          }
        })
        .catch(() =>
          fail(
            "The camera could not be opened. Check browser permissions or choose a photo from your device.",
          ),
        );
    }
    return () => {
      document.removeEventListener("visibilitychange", closeWhenHidden);
      stopCamera();
    };
  }, [open, modality, stopCamera]);

  const capture = () => {
    const source = video.current,
      target = canvas.current;
    if (
      !ready ||
      capturePending.current ||
      !source?.videoWidth ||
      !target ||
      document.hidden
    )
      return;
    const token = generation.current;
    capturePending.current = true;
    setCapturing(true);
    if (telemetryTimer.current !== null) clearInterval(telemetryTimer.current);
    telemetryTimer.current = null;
    target.width = source.videoWidth;
    target.height = source.videoHeight;
    const fail = () => {
      if (token !== generation.current) return;
      stopCamera();
      setReady(false);
      setCapturing(false);
      setError(
        "This frame could not be captured. Close the camera and try again, or choose a photo.",
      );
    };
    try {
      const context = target.getContext("2d");
      if (!context) return fail();
      context.drawImage(source, 0, 0);
      target.toBlob((blob) => {
        if (token !== generation.current || document.hidden) return;
        if (!blob) return fail();
        captureTimer.current = setTimeout(
          () => {
            if (token !== generation.current || document.hidden) return;
            const file = new File(
              [blob],
              `capture-${modality}-${Date.now()}.png`,
              { type: "image/png" },
            );
            stopCamera();
            setOpen(false);
            onCapture(file);
          },
          reducedMotion ? 0 : 340,
        );
      }, "image/png");
    } catch {
      fail();
    }
  };

  const lighting = !reading
    ? "Waiting"
    : reading.luminance < 55
      ? "Low"
      : reading.luminance > 220
        ? "High"
        : "In range";
  const detail = !reading
    ? "Waiting"
    : reading.detail < 35
      ? "Low"
      : "Edges visible";
  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogTrigger asChild>
        <Button type="button" variant="outline" disabled={disabled}>
          <Camera data-icon="inline-start" /> {triggerLabel}
        </Button>
      </DialogTrigger>
      <DialogContent className="guided-camera-dialog">
        <DialogHeader>
          <DialogTitle>Capture a screening photo</DialogTitle>
          <DialogDescription>
            {modality === "conjunctiva"
              ? "Frame the exposed inner lower eyelid in the guide."
              : "Frame one clean, unpolished nail inside the guide."}{" "}
            Use even light and hold steady. The image is uploaded only when you
            start detection.
          </DialogDescription>
        </DialogHeader>
        {error ? (
          <Alert variant="destructive">
            <AlertTitle>Camera unavailable</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : (
          <>
            <div
              className={`camera-preview guided-camera-preview${capturing ? " is-capturing" : ""}`}
              style={{ aspectRatio }}
            >
              <video
                id="cameraVideo"
                ref={video}
                playsInline
                muted
                onResize={(event) => {
                  const source = event.currentTarget;
                  if (
                    stream.current &&
                    source.videoWidth &&
                    source.videoHeight
                  ) {
                    setAspectRatio(
                      `${source.videoWidth} / ${source.videoHeight}`,
                    );
                  }
                }}
                aria-label={
                  modality === "conjunctiva"
                    ? "Live eye camera preview"
                    : "Live nail camera preview"
                }
                onLoadedData={() => {
                  if (
                    stream.current &&
                    video.current?.videoWidth &&
                    !document.hidden
                  ) {
                    setAspectRatio(
                      `${video.current.videoWidth} / ${video.current.videoHeight}`,
                    );
                    setReady(true);
                  }
                }}
              />
              <div
                id="targetingReticle"
                className={`capture-reticle capture-reticle-${modality}`}
                aria-hidden="true"
              >
                <span className="capture-reticle-cross" />
              </div>
              {capturing && (
                <span className="capture-beam" aria-hidden="true" />
              )}
              {!ready && (
                <LoaderCircle className="spin" aria-label="Starting camera" />
              )}
            </div>
            <div className="capture-telemetry" aria-label="Live camera checks">
              <output
                id="lightingBadge"
                aria-live="off"
                data-state={
                  !reading
                    ? "waiting"
                    : lighting === "In range"
                      ? "good"
                      : "adjust"
                }
                data-luminance={reading?.luminance.toFixed(1)}
              >
                <Sun size={16} /> Lighting: {lighting}
              </output>
              <output
                id="focusBadge"
                aria-live="off"
                data-state={
                  !reading
                    ? "waiting"
                    : detail === "Edges visible"
                      ? "good"
                      : "adjust"
                }
                data-detail={reading?.detail.toFixed(1)}
              >
                <Focus size={16} /> Detail: {detail}
              </output>
            </div>
            <p className="capture-check-note">
              Brightness and edge detail are estimated from the guide area.
              Check focus and reflections yourself; these checks do not detect
              tissue.
            </p>
          </>
        )}
        <canvas id="cameraCanvas" ref={canvas} hidden aria-hidden="true" />
        <Button
          id="shutterBtn"
          type="button"
          disabled={!ready || !!error || capturing}
          onClick={capture}
        >
          {capturing ? (
            <LoaderCircle className="spin" data-icon="inline-start" />
          ) : (
            <Camera data-icon="inline-start" />
          )}{" "}
          {capturing ? "Capturing…" : "Capture photo"}
        </Button>
      </DialogContent>
    </Dialog>
  );
}
